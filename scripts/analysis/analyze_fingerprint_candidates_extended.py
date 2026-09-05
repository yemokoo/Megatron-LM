#!/usr/bin/env python3
"""Extended candidate audit beyond 2-D PCA and pairwise cosine summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.utils.extmath import randomized_svd

from analyze_router_fingerprint_smoke import (
    distribution_metrics,
    domain_selectivity,
    heldout_masks,
    load_dump,
    logits_to_routing,
    orthonormalize,
    paired_basic,
    projection_routing_metrics,
    set_metrics,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ranks", default="8,16,32,64,128")
    return parser.parse_args()


def inverse_sqrt(matrix, ridge=1e-6):
    values, vectors = np.linalg.eigh(matrix)
    floor = max(float(values.max()) * ridge, 1e-12)
    return (vectors * (1.0 / np.sqrt(np.maximum(values, floor)))) @ vectors.T


def truncated_cca(a, b, rank, seed):
    ac = a.astype(np.float64) - a.mean(axis=0, keepdims=True)
    bc = b.astype(np.float64) - b.mean(axis=0, keepdims=True)
    rank = min(rank, min(ac.shape) - 1, min(bc.shape) - 1)
    _ua, _sa, va = randomized_svd(ac, n_components=rank, random_state=seed)
    _ub, _sb, vb = randomized_svd(bc, n_components=rank, random_state=seed + 1)
    xa, xb = ac @ va.T, bc @ vb.T
    scale = max(len(ac) - 1, 1)
    caa = xa.T @ xa / scale
    cbb = xb.T @ xb / scale
    cab = xa.T @ xb / scale
    wa, wb = inverse_sqrt(caa), inverse_sqrt(cbb)
    u, correlations, _vt = np.linalg.svd(wa @ cab @ wb, full_matrices=False)
    correlations = np.clip(correlations, 0.0, 1.0)
    canonical_x = xa @ wa @ u
    q, _r = np.linalg.qr(canonical_x)
    weights = np.sum(np.abs(q.T @ ac), axis=1)
    weights /= max(weights.sum(), 1e-20)
    return {
        "rank": int(rank),
        "svcca_mean": float(correlations.mean()),
        "svcca_min": float(correlations.min()),
        "pwcca": float(weights @ correlations),
        "canonical_correlations": correlations.tolist(),
        "note": "PCA-truncated CCA; PWCCA weights use projections onto the original centered activations.",
    }


def rank_order_metrics(p, q):
    rank_p = np.argsort(np.argsort(p, axis=-1), axis=-1).astype(np.float64)
    rank_q = np.argsort(np.argsort(q, axis=-1), axis=-1).astype(np.float64)
    rank_p -= rank_p.mean(axis=-1, keepdims=True)
    rank_q -= rank_q.mean(axis=-1, keepdims=True)
    corr = np.sum(rank_p * rank_q, axis=-1) / np.maximum(
        np.linalg.norm(rank_p, axis=-1) * np.linalg.norm(rank_q, axis=-1), 1e-20
    )
    return {
        "spearman_expert_rank_mean": float(corr.mean()),
        "spearman_expert_rank_p05": float(np.percentile(corr, 5)),
    }


def stable_basis_payload(wb, wa, teacher_weight, discovery, max_rank, seed):
    mean = wb[discovery].mean(axis=0, keepdims=True)
    centered = wb[discovery] - mean
    max_rank = min(max_rank, min(centered.shape) - 1)
    _u, singular, vt = randomized_svd(centered, n_components=max_rank, random_state=seed)
    pca_basis = vt.T
    signal = np.square(singular) / max(discovery.sum() - 1, 1)
    delta = wa[discovery] - wb[discovery]
    drift = np.mean(np.square(delta @ pca_basis), axis=0)
    sensitivity = np.sum(np.square(teacher_weight @ pca_basis), axis=0)
    floor = max(float(np.median(drift)) * 1e-6, 1e-12)
    order = np.argsort(signal * np.maximum(sensitivity, 1e-20) / (drift + floor))[::-1]
    return mean, pca_basis, order, signal, drift, sensitivity


def shuffled_label_baseline(wiki_projected, code_projected, discovery, evaluation, seed):
    train_x = np.concatenate([wiki_projected[discovery], code_projected[discovery]])
    train_y = np.concatenate([np.zeros(discovery.sum()), np.ones(discovery.sum())])
    eval_x = np.concatenate([wiki_projected[evaluation], code_projected[evaluation]])
    eval_y = np.concatenate([np.zeros(evaluation.sum()), np.ones(evaluation.sum())])
    rng = np.random.default_rng(seed)
    aucs = []
    for repeat in range(5):
        shuffled = rng.permutation(train_y)
        model = LogisticRegression(max_iter=300, random_state=seed + repeat).fit(train_x, shuffled)
        aucs.append(roc_auc_score(eval_y, model.predict_proba(eval_x)[:, 1]))
    return {"repeats": 5, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


def prototype_metrics(x, teacher_weight, teacher_probs, teacher_topk, discovery, evaluation):
    top1 = teacher_topk[:, 0]
    mean = x[discovery].mean(axis=0)
    prototypes = []
    counts = []
    for expert in range(teacher_weight.shape[0]):
        mask = discovery & (top1 == expert)
        counts.append(int(mask.sum()))
        prototypes.append(x[mask].mean(axis=0) if mask.any() else mean)
    prototypes = np.stack(prototypes)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    normalized = prototypes / np.maximum(norms, 1e-20)
    eval_normalized = x[evaluation] / np.maximum(
        np.linalg.norm(x[evaluation], axis=1, keepdims=True), 1e-20
    )
    nearest = np.argmax(eval_normalized @ normalized.T, axis=1)
    oracle_vectors = prototypes[top1[evaluation]]
    nearest_vectors = prototypes[nearest]
    oracle_probs, oracle_topk, _ = logits_to_routing(
        oracle_vectors @ teacher_weight.T, teacher_topk.shape[-1]
    )
    nearest_probs, nearest_topk, _ = logits_to_routing(
        nearest_vectors @ teacher_weight.T, teacher_topk.shape[-1]
    )
    return {
        "prototype_counts": counts,
        "stored_float_count": int(prototypes.size),
        "oracle_expert_id_known": {
            **set_metrics(teacher_topk[evaluation], oracle_topk),
            **distribution_metrics(teacher_probs[evaluation], oracle_probs),
        },
        "nearest_prototype_no_expert_id": {
            **set_metrics(teacher_topk[evaluation], nearest_topk),
            **distribution_metrics(teacher_probs[evaluation], nearest_probs),
            "top1_prototype_assignment_accuracy": float(np.mean(nearest == top1[evaluation])),
        },
    }


def expert_diagnostics(dump, layer_index):
    indices = dump["router_topk_indices"][layer_index]
    hidden = dump["router_input"][layer_index].astype(np.float64)
    outputs = dump["expert_outputs"][layer_index].astype(np.float64)
    rows = []
    for expert in range(dump["router_weights"].shape[1]):
        selected = indices == expert
        token_mask = selected.any(axis=1)
        slots = np.argmax(selected, axis=1)
        expert_outputs = outputs[np.arange(len(outputs))[token_mask], slots[token_mask]]
        rows.append(
            {
                "expert": expert,
                "topk_assignments": int(selected.sum()),
                "assigned_tokens": int(token_mask.sum()),
                "activation_centroid_norm": float(np.linalg.norm(hidden[token_mask].mean(0)))
                if token_mask.any()
                else None,
                "expert_output_norm_mean": float(np.linalg.norm(expert_outputs, axis=1).mean())
                if token_mask.any()
                else None,
                "expert_output_centroid_norm": float(np.linalg.norm(expert_outputs.mean(0)))
                if token_mask.any()
                else None,
            }
        )
    return rows


def paired_expert_diagnostics(before, after, layer_index):
    ib = before["router_topk_indices"][layer_index]
    ia = after["router_topk_indices"][layer_index]
    ob = before["expert_outputs"][layer_index].astype(np.float64)
    oa = after["expert_outputs"][layer_index].astype(np.float64)
    rows = []
    for expert in range(min(before["router_weights"].shape[1], after["router_weights"].shape[1])):
        mb, ma = (ib == expert).any(axis=1), (ia == expert).any(axis=1)
        common = mb & ma
        row = {
            "expert": expert,
            "assignment_token_jaccard": float(np.sum(common) / max(np.sum(mb | ma), 1)),
            "common_tokens": int(common.sum()),
        }
        if common.any():
            sb, sa = np.argmax(ib == expert, axis=1), np.argmax(ia == expert, axis=1)
            row["paired_expert_output"] = paired_basic(
                ob[np.arange(len(ob))[common], sb[common]],
                oa[np.arange(len(oa))[common], sa[common]],
            )
        rows.append(row)
    return rows


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    ranks = sorted({int(value) for value in args.ranks.split(",") if int(value) > 0})
    wiki_b = load_dump(root / "dumps/wiki/B.npz")
    wiki_c = load_dump(root / "dumps/wiki/C_vocabkl.npz")
    code_b = load_dump(root / "dumps/code/B.npz")
    if not np.array_equal(wiki_b["layer_numbers"], wiki_c["layer_numbers"]):
        raise RuntimeError("B/C layer alignment failed")
    discovery, evaluation = heldout_masks(wiki_b["sample_indices"], args.seed)
    result = {
        "status": "single_seed_smoke; D/E/I unavailable; actual-forward intervention pending",
        "layers": [],
        "cross_domain_router_assignment": [],
    }
    plot_rows = []

    for layer_index, layer_number in enumerate(wiki_b["layer_numbers"]):
        wb = wiki_b["router_input"][layer_index].astype(np.float64)
        wc = wiki_c["router_input"][layer_index].astype(np.float64)
        cb = code_b["router_input"][layer_index].astype(np.float64)
        teacher_weight = wiki_b["router_weights"][layer_index].astype(np.float64)
        teacher_probs = wiki_b["router_full_probs"][layer_index].astype(np.float64)
        teacher_topk = wiki_b["router_topk_indices"][layer_index]
        max_rank = min(max(ranks), min(wb[discovery].shape) - 1)
        mean, pca_basis, stable_order, signal, drift, sensitivity = stable_basis_payload(
            wb, wc, teacher_weight, discovery, max_rank, args.seed + layer_index
        )
        teacher_eval_logits = wb[evaluation] @ teacher_weight.T
        teacher_eval_probs, teacher_eval_topk, _ = logits_to_routing(
            teacher_eval_logits, teacher_topk.shape[-1]
        )
        channel_signal = np.var(wb[discovery], axis=0)
        channel_drift = np.mean(np.square(wc[discovery] - wb[discovery]), axis=0)
        channel_floor = max(float(np.median(channel_drift)) * 1e-6, 1e-12)
        channel_order = np.argsort(channel_signal / (channel_drift + channel_floor))[::-1]
        rng = np.random.default_rng(args.seed + 200 + layer_index)
        channel_curves = []
        for rank in ranks:
            rank = min(rank, wb.shape[1])
            basis = np.eye(wb.shape[1])[:, channel_order[:rank]]
            metrics = projection_routing_metrics(
                wb[evaluation], mean, basis, teacher_weight, teacher_eval_probs, teacher_eval_topk
            )
            random_scores = []
            for _repeat in range(5):
                random_basis = np.eye(wb.shape[1])[:, rng.choice(wb.shape[1], rank, replace=False)]
                random_scores.append(
                    projection_routing_metrics(
                        wb[evaluation],
                        mean,
                        random_basis,
                        teacher_weight,
                        teacher_eval_probs,
                        teacher_eval_topk,
                    )["fingerprint_only"]["topk_agreement"]
                )
            channel_curves.append(
                {
                    "rank": rank,
                    **metrics,
                    "random_channel_topk_mean": float(np.mean(random_scores)),
                    "random_channel_topk_std": float(np.std(random_scores)),
                }
            )

        best_rank = min(32, max_rank)
        stable_basis = orthonormalize(pca_basis[:, stable_order], best_rank)
        stable_w = (wb - mean) @ stable_basis
        stable_c = (cb - mean) @ stable_basis
        selectivity = domain_selectivity(
            wb, cb, mean, stable_basis, discovery, evaluation, args.seed + layer_index
        )
        selectivity["shuffled_label_baseline"] = shuffled_label_baseline(
            stable_w, stable_c, discovery, evaluation, args.seed + layer_index
        )
        projected_eval = mean + ((wb[evaluation] - mean) @ stable_basis) @ stable_basis.T
        projected_probs, projected_topk, _ = logits_to_routing(
            projected_eval @ teacher_weight.T, teacher_topk.shape[-1]
        )
        success = np.all(
            np.sort(projected_topk, axis=-1) == np.sort(teacher_eval_topk, axis=-1), axis=-1
        )
        projection_residual = np.linalg.norm(wb[evaluation] - projected_eval, axis=1)
        margin = np.sort(teacher_eval_probs, axis=-1)[:, -1] - np.sort(
            teacher_eval_probs, axis=-1
        )[:, -2]
        plot_rows.append(
            {
                "layer": int(layer_number),
                "wiki_projection_norm": np.linalg.norm(stable_w[evaluation], axis=1),
                "code_projection_norm": np.linalg.norm(stable_c[evaluation], axis=1),
                "wiki_orthogonal_norm": np.linalg.norm(
                    (wb[evaluation] - mean) - stable_w[evaluation] @ stable_basis.T, axis=1
                ),
                "code_orthogonal_norm": np.linalg.norm(
                    (cb[evaluation] - mean) - stable_c[evaluation] @ stable_basis.T, axis=1
                ),
                "success": success,
                "margin": margin,
                "residual": projection_residual,
            }
        )

        route_stable = np.all(
            np.sort(wiki_b["router_topk_indices"][layer_index], axis=-1)
            == np.sort(wiki_c["router_topk_indices"][layer_index], axis=-1),
            axis=-1,
        )
        entropy = wiki_b["router_entropy"][layer_index]
        original_margin = wiki_b["router_top1_margin"][layer_index]
        margin_auc = roc_auc_score(route_stable.astype(int), original_margin)
        entropy_auc = roc_auc_score(route_stable.astype(int), -entropy)
        row_u, row_s, row_vt = np.linalg.svd(teacher_weight, full_matrices=False)
        row_rank = int(np.sum(row_s > row_s.max() * 1e-8))
        row_basis = row_vt[:row_rank].T
        stable_overlap = np.linalg.svd(stable_basis.T @ row_basis, compute_uv=False)

        b_top1 = wiki_b["router_topk_indices"][layer_index, :, 0]
        code_top1 = code_b["router_topk_indices"][layer_index, :, 0]
        domain_labels = np.concatenate([np.zeros(len(b_top1)), np.ones(len(code_top1))])
        assignments = np.concatenate([b_top1, code_top1])
        result["cross_domain_router_assignment"].append(
            {
                "layer": int(layer_number),
                "normalized_mutual_information_domain_top1_expert": float(
                    normalized_mutual_info_score(domain_labels, assignments)
                ),
            }
        )
        result["layers"].append(
            {
                "layer": int(layer_number),
                "cca": truncated_cca(wb, wc, min(128, max_rank), args.seed + layer_index),
                "router_rank_order": rank_order_metrics(
                    wiki_b["router_full_probs"][layer_index],
                    wiki_c["router_full_probs"][layer_index],
                ),
                "margin_entropy_conditioning": {
                    "routing_stable_fraction": float(route_stable.mean()),
                    "margin_stable_mean": float(original_margin[route_stable].mean()),
                    "margin_changed_mean": float(original_margin[~route_stable].mean()),
                    "entropy_stable_mean": float(entropy[route_stable].mean()),
                    "entropy_changed_mean": float(entropy[~route_stable].mean()),
                    "margin_auc_for_routing_stability": float(margin_auc),
                    "negative_entropy_auc_for_routing_stability": float(entropy_auc),
                },
                "analytic_router_jacobian": {
                    "rank": row_rank,
                    "singular_values": row_s.tolist(),
                    "condition_number_nonzero": float(row_s[0] / row_s[row_rank - 1]),
                    "stable_sensitive_r32_rowspace_principal_cosines": stable_overlap.tolist(),
                    "interpretation": "For a linear router, d(logits)/d(hidden) is exactly the router weight; its row space is the complete routing-sensitive subspace and its orthogonal complement is the local null space.",
                },
                "stable_channel_subset": channel_curves,
                "stable_sensitive_r32_selectivity": selectivity,
                "expert_activation_prototypes": prototype_metrics(
                    wb,
                    teacher_weight,
                    teacher_probs,
                    teacher_topk,
                    discovery,
                    evaluation,
                ),
                "expert_diagnostics_B": expert_diagnostics(wiki_b, layer_index),
                "expert_diagnostics_C": expert_diagnostics(wiki_c, layer_index),
                "paired_expert_B_to_C": paired_expert_diagnostics(
                    wiki_b, wiki_c, layer_index
                ),
                "stable_sensitive_r32_projection": {
                    **set_metrics(teacher_eval_topk, projected_topk),
                    **distribution_metrics(teacher_eval_probs, projected_probs),
                },
                "pca_direction_summaries": {
                    "signal": signal.tolist(),
                    "drift": drift.tolist(),
                    "router_sensitivity": sensitivity.tolist(),
                },
            }
        )

    metrics_path = root / "metrics" / "fingerprint_candidates_extended.json"
    write_json(metrics_path, result)

    fig, axes = plt.subplots(len(plot_rows), 2, figsize=(12, 4.5 * len(plot_rows)))
    axes = np.asarray(axes).reshape(len(plot_rows), 2)
    for row_index, row in enumerate(plot_rows):
        ax = axes[row_index, 0]
        ax.scatter(
            row["wiki_projection_norm"], row["wiki_orthogonal_norm"],
            s=8, alpha=0.35, label="Wiki", color="#2563eb",
        )
        ax.scatter(
            row["code_projection_norm"], row["code_orthogonal_norm"],
            s=8, alpha=0.35, label="Code", color="#f97316",
        )
        ax.set_title(f"L{row['layer']} stable-sensitive r32 vs orthogonal")
        ax.set_xlabel("stable projection norm")
        ax.set_ylabel("orthogonal norm")
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)

        ax = axes[row_index, 1]
        ax.scatter(
            row["margin"][~row["success"]], row["residual"][~row["success"]],
            s=9, alpha=0.4, label="top-k failed", color="#dc2626",
        )
        ax.scatter(
            row["margin"][row["success"]], row["residual"][row["success"]],
            s=9, alpha=0.4, label="top-k restored", color="#059669",
        )
        ax.set_title(f"L{row['layer']} teacher top-k restoration")
        ax.set_xlabel("teacher top-1 probability margin")
        ax.set_ylabel("projection residual norm")
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    plot_path = root / "plots" / "stable_orthogonal_and_routing_restoration.png"
    fig.savefig(plot_path, dpi=190)
    plt.close(fig)
    print(json.dumps({"metrics": str(metrics_path), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
