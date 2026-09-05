#!/usr/bin/env python3
"""Analyze aligned representation/routing dumps for compact router fingerprints.

The script is intentionally checkpoint-agnostic: it consumes the detailed NPZ
dumps produced by ``run_fingerprint_router_smoke_mha.sh`` and writes all metrics
and figures below one analysis directory.  Candidate subspaces are learned on a
sample-disjoint discovery split and evaluated on held-out sample IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd


STAGE_ORDER = ("A", "B", "C_vocabkl", "D", "E", "F", "G", "H", "I")
STAGE_COLORS = {
    "A": "#2563eb",
    "B": "#f97316",
    "C_vocabkl": "#9333ea",
    "D": "#059669",
    "E": "#dc2626",
    "F": "#7c3aed",
    "G": "#0891b2",
    "H": "#be123c",
    "I": "#4b5563",
}
COMPONENTS = (
    "layer_input",
    "attention_output",
    "post_attention_hidden",
    "router_input",
    "ffn_output",
    "layer_output",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ranks", default="8,16,32,64,128")
    return parser.parse_args()


def load_dump(path: Path):
    with np.load(path, allow_pickle=False) as z:
        required = {
            "layer_numbers",
            "token_ids",
            "positions",
            "sample_indices",
            "router_input",
            "router_logits",
            "router_full_probs",
            "router_topk_indices",
            "router_topk_weights",
            "router_weights",
            "metadata",
        }
        missing = sorted(required - set(z.files))
        if missing:
            raise RuntimeError(f"{path}: missing detailed dump arrays: {missing}")
        return {
            "path": str(path),
            "metadata": json.loads(str(z["metadata"])),
            **{key: z[key].copy() for key in z.files if key != "metadata"},
        }


def discover_dumps(root: Path):
    result = {}
    for domain_dir in sorted((root / "dumps").glob("*")):
        if not domain_dir.is_dir():
            continue
        stages = {}
        for path in sorted(domain_dir.glob("*.npz")):
            # Ignore atomic-write temporaries and interrupted dumps.  Only
            # named experimental stages are eligible analysis inputs.
            if path.stem not in STAGE_ORDER:
                continue
            stages[path.stem] = load_dump(path)
        if stages:
            result[domain_dir.name] = stages
    return result


def identity_digest(dump):
    digest = hashlib.sha256()
    for key in ("sample_indices", "positions", "token_ids"):
        digest.update(np.ascontiguousarray(dump[key]).tobytes())
    return digest.hexdigest()


def validate_alignment(dumps):
    audit = {}
    for domain, stages in dumps.items():
        first_name = next(iter(stages))
        reference = stages[first_name]
        rows = {}
        for stage, dump in stages.items():
            exact = {
                key: bool(np.array_equal(reference[key], dump[key]))
                for key in ("layer_numbers", "sample_indices", "positions", "token_ids")
            }
            if not all(exact.values()):
                raise RuntimeError(f"{domain}: token/layer alignment failed for {stage}: {exact}")
            rows[stage] = {
                "path": dump["path"],
                "size_bytes": Path(dump["path"]).stat().st_size,
                "identity_exact": exact,
                "identity_sha256": identity_digest(dump),
                "metadata": dump["metadata"],
                "array_shapes": {
                    key: list(value.shape)
                    for key, value in dump.items()
                    if isinstance(value, np.ndarray)
                },
                "all_finite": bool(
                    all(
                        np.isfinite(value).all()
                        for key, value in dump.items()
                        if isinstance(value, np.ndarray)
                        and np.issubdtype(value.dtype, np.number)
                        and key != "router_topk_margin"
                    )
                ),
            }
        audit[domain] = rows
    return audit


def paired_basic(a, b):
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    dot = np.sum(a * b, axis=-1)
    denom = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
    cosine = np.clip(dot / denom, -1.0, 1.0)
    nl2 = 2.0 * np.linalg.norm(b - a, axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1), 1e-12
    )
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "fraction_cosine_ge_0_99": float(np.mean(cosine >= 0.99)),
        "fraction_cosine_ge_0_999": float(np.mean(cosine >= 0.999)),
        "nl2_median": float(np.median(nl2)),
        "nl2_p95": float(np.percentile(nl2, 95)),
    }


def geometry_metrics(a, b, ranks, seed):
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    ac = a - a.mean(axis=0, keepdims=True)
    bc = b - b.mean(axis=0, keepdims=True)
    scale = max(len(a) - 1, 1)
    aa = (ac.T @ ac) / scale
    bb = (bc.T @ bc) / scale
    ab = (ac.T @ bc) / scale
    cka = np.square(np.linalg.norm(ab, ord="fro")) / max(
        np.linalg.norm(aa, ord="fro") * np.linalg.norm(bb, ord="fro"), 1e-20
    )
    covariance_drift = np.linalg.norm(aa - bb, ord="fro") / max(
        0.5 * (np.linalg.norm(aa, ord="fro") + np.linalg.norm(bb, ord="fro")), 1e-20
    )
    centroid_scale = np.sqrt(
        0.5 * (np.mean(np.sum(ac * ac, axis=1)) + np.mean(np.sum(bc * bc, axis=1)))
    )
    max_rank = min(max(ranks), min(ac.shape) - 1)
    _ua, _sa, va = randomized_svd(ac, n_components=max_rank, random_state=seed)
    _ub, _sb, vb = randomized_svd(bc, n_components=max_rank, random_state=seed + 1)
    angles = {}
    projection = {}
    for rank in ranks:
        rank = min(rank, max_rank)
        singular = np.linalg.svd(va[:rank] @ vb[:rank].T, compute_uv=False)
        degree = np.degrees(np.arccos(np.clip(singular, -1.0, 1.0)))
        angles[str(rank)] = {
            "mean_degrees": float(degree.mean()),
            "max_degrees": float(degree.max()),
            "cosine_mean": float(singular.mean()),
        }
        ua = va[:rank].T
        residual = bc - (bc @ ua) @ ua.T
        projection[str(rank)] = float(
            np.mean(np.sum(residual * residual, axis=1))
            / max(np.mean(np.sum(bc * bc, axis=1)), 1e-20)
        )
    return {
        "centered_linear_cka": float(cka),
        "centroid_drift_normalized": float(
            np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)) / max(centroid_scale, 1e-20)
        ),
        "covariance_drift_relative_frobenius": float(covariance_drift),
        "principal_angles": angles,
        "after_projection_residual_on_before_subspace": projection,
    }


def set_metrics(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    intersection = (a[..., :, None] == b[..., None, :]).any(axis=-1).sum(axis=-1)
    union = a.shape[-1] + b.shape[-1] - intersection
    a_sorted = np.sort(a, axis=-1)
    b_sorted = np.sort(b, axis=-1)
    exact = np.all(a_sorted == b_sorted, axis=-1)
    return {
        "topk_agreement": float(exact.mean()),
        "expert_set_jaccard": float(np.mean(intersection / np.maximum(union, 1))),
        "top1_agreement": float(np.mean(a[..., 0] == b[..., 0])),
    }


def distribution_metrics(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-20)
    q = q / np.maximum(q.sum(axis=-1, keepdims=True), 1e-20)
    m = 0.5 * (p + q)
    kl_pq = np.sum(p * (np.log(np.maximum(p, 1e-20)) - np.log(np.maximum(q, 1e-20))), axis=-1)
    js = 0.5 * (
        np.sum(p * (np.log(np.maximum(p, 1e-20)) - np.log(np.maximum(m, 1e-20))), axis=-1)
        + np.sum(q * (np.log(np.maximum(q, 1e-20)) - np.log(np.maximum(m, 1e-20))), axis=-1)
    )
    return {"routing_kl_mean": float(kl_pq.mean()), "routing_js_mean": float(js.mean())}


def logits_to_routing(logits, topk):
    logits = logits.astype(np.float64, copy=False)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    indices = np.argsort(-probs, axis=-1)[..., :topk]
    weights = np.take_along_axis(probs, indices, axis=-1)
    return probs, indices, weights


def drift_decomposition(before, after, layer_index):
    h0 = before["router_input"][layer_index].astype(np.float64)
    h1 = after["router_input"][layer_index].astype(np.float64)
    w0 = before["router_weights"][layer_index].astype(np.float64)
    w1 = after["router_weights"][layer_index].astype(np.float64)
    common = min(len(w0), len(w1))
    topk = min(before["router_topk_indices"].shape[-1], common)
    base_logits = h0 @ w0[:common].T
    rep_logits = h1 @ w0[:common].T
    boundary_logits = h1 @ w1[:common].T
    base_probs, base_topk, _ = logits_to_routing(base_logits, topk)
    rep_probs, rep_topk, _ = logits_to_routing(rep_logits, topk)
    boundary_probs, boundary_topk, _ = logits_to_routing(boundary_logits, topk)
    current_topk = after["router_topk_indices"][layer_index]

    cosine = np.sum(h0 * h1, axis=-1) / np.maximum(
        np.linalg.norm(h0, axis=-1) * np.linalg.norm(h1, axis=-1), 1e-20
    )
    routing_stable = np.all(
        np.sort(before["router_topk_indices"][layer_index], axis=-1)
        == np.sort(current_topk, axis=-1),
        axis=-1,
    )
    representation_stable = cosine >= 0.99
    groups = {
        "representation_stable_routing_stable": representation_stable & routing_stable,
        "representation_stable_routing_changed": representation_stable & ~routing_stable,
        "representation_changed_routing_stable": ~representation_stable & routing_stable,
        "representation_changed_routing_changed": ~representation_stable & ~routing_stable,
    }
    result = {
        "common_experts": int(common),
        "representation_only_common_router": set_metrics(base_topk, rep_topk),
        "boundary_only_common_experts": set_metrics(rep_topk, boundary_topk),
        "representation_plus_common_boundary": set_metrics(base_topk, boundary_topk),
        "common_distribution_representation_only": distribution_metrics(base_probs, rep_probs),
        "common_distribution_boundary_only": distribution_metrics(rep_probs, boundary_probs),
        "full_model_total": set_metrics(before["router_topk_indices"][layer_index], current_topk),
        "four_token_groups": {
            name: {"tokens": int(mask.sum()), "fraction": float(mask.mean())}
            for name, mask in groups.items()
        },
    }
    if len(w1) > common:
        result["new_expert_competition"] = set_metrics(boundary_topk, current_topk)
        result["new_expert_topk_slot_fraction"] = float(np.mean(current_topk >= common))
    return result


def comparison_metrics(dumps, ranks, seed):
    requested = {
        "A_to_B": ("A", "B"),
        "B_to_C_vocabkl": ("B", "C_vocabkl"),
        "B_to_D": ("B", "D"),
        "A_to_E": ("A", "E"),
        "F_to_G": ("F", "G"),
        "G_to_H": ("G", "H"),
        "G_to_I": ("G", "I"),
    }
    result = {}
    for domain, stages in dumps.items():
        result[domain] = {}
        for label, (before_name, after_name) in requested.items():
            resolved_before = "C_vocabkl" if before_name == "F" and "F" not in stages else before_name
            if resolved_before not in stages or after_name not in stages:
                continue
            before, after = stages[resolved_before], stages[after_name]
            layers = []
            for layer_index, layer_number in enumerate(before["layer_numbers"]):
                basic = {
                    component: paired_basic(
                        before[component][layer_index], after[component][layer_index]
                    )
                    for component in COMPONENTS
                }
                layers.append(
                    {
                        "layer": int(layer_number),
                        "components": basic,
                        "router_input_geometry": geometry_metrics(
                            before["router_input"][layer_index],
                            after["router_input"][layer_index],
                            ranks,
                            seed + layer_index,
                        ),
                        "routing": {
                            **set_metrics(
                                before["router_topk_indices"][layer_index],
                                after["router_topk_indices"][layer_index],
                            ),
                            "top1_margin_before_mean": float(
                                before["router_top1_margin"][layer_index].mean()
                            ),
                            "top1_margin_after_mean": float(
                                after["router_top1_margin"][layer_index].mean()
                            ),
                            "entropy_before_mean": float(
                                before["router_entropy"][layer_index].mean()
                            ),
                            "entropy_after_mean": float(
                                after["router_entropy"][layer_index].mean()
                            ),
                        },
                        "drift_decomposition": drift_decomposition(
                            before, after, layer_index
                        ),
                    }
                )
                if before["router_full_probs"].shape[-1] == after["router_full_probs"].shape[-1]:
                    layers[-1]["routing"].update(
                        distribution_metrics(
                            before["router_full_probs"][layer_index],
                            after["router_full_probs"][layer_index],
                        )
                    )
                else:
                    common = before["router_full_probs"].shape[-1]
                    layers[-1]["routing"]["old_expert_conditional"] = distribution_metrics(
                        before["router_full_probs"][layer_index],
                        after["router_full_probs"][layer_index, :, :common],
                    )
            result[domain][label] = {"before": before_name, "after": after_name, "layers": layers}
    return result


def heldout_masks(sample_indices, seed):
    unique = np.unique(sample_indices)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    split = len(shuffled) // 2
    discovery_ids = set(shuffled[:split].tolist())
    discovery = np.array([value in discovery_ids for value in sample_indices])
    return discovery, ~discovery


def routing_margin_recovery(teacher_logits, candidate_logits, topk):
    teacher_sorted = np.sort(teacher_logits, axis=-1)[:, ::-1]
    candidate_sorted = np.sort(candidate_logits, axis=-1)[:, ::-1]
    teacher_top1 = teacher_sorted[:, 0] - teacher_sorted[:, 1]
    candidate_top1 = candidate_sorted[:, 0] - candidate_sorted[:, 1]
    teacher_topk = teacher_sorted[:, topk - 1] - teacher_sorted[:, topk]
    candidate_topk = candidate_sorted[:, topk - 1] - candidate_sorted[:, topk]

    def summarize(target, predicted):
        centered_target = target - target.mean()
        centered_predicted = predicted - predicted.mean()
        correlation = np.sum(centered_target * centered_predicted) / max(
            np.linalg.norm(centered_target) * np.linalg.norm(centered_predicted), 1e-20
        )
        return {
            "mae": float(np.mean(np.abs(target - predicted))),
            "normalized_mae": float(
                np.mean(np.abs(target - predicted)) / max(np.mean(np.abs(target)), 1e-20)
            ),
            "pearson": float(correlation),
        }

    return {
        "top1_margin": summarize(teacher_top1, candidate_top1),
        "topk_boundary_margin": summarize(teacher_topk, candidate_topk),
    }


def orthonormalize(vectors, rank=None):
    if rank is not None:
        vectors = vectors[:, :rank]
    q, _ = np.linalg.qr(vectors)
    return q


def projection_routing_metrics(x, mean, basis, router_weight, teacher_probs, teacher_topk):
    centered = x - mean
    projected = mean + (centered @ basis) @ basis.T
    removed = mean + centered - (centered @ basis) @ basis.T
    proj_logits = projected @ router_weight.T
    removed_logits = removed @ router_weight.T
    proj_probs, proj_topk, _ = logits_to_routing(proj_logits, teacher_topk.shape[-1])
    removed_probs, removed_topk, _ = logits_to_routing(removed_logits, teacher_topk.shape[-1])
    teacher_logits = x @ router_weight.T
    topk = teacher_topk.shape[-1]
    return {
        "fingerprint_only": {
            **set_metrics(teacher_topk, proj_topk),
            **distribution_metrics(teacher_probs, proj_probs),
            "router_logit_nmse": float(
                np.mean(np.square(teacher_logits - proj_logits))
                / max(np.mean(np.square(teacher_logits)), 1e-20)
            ),
            "margin_recovery": routing_margin_recovery(
                teacher_logits, proj_logits, topk
            ),
        },
        "fingerprint_removed": {
            **set_metrics(teacher_topk, removed_topk),
            **distribution_metrics(teacher_probs, removed_probs),
            "router_logit_nmse": float(
                np.mean(np.square(teacher_logits - removed_logits))
                / max(np.mean(np.square(teacher_logits)), 1e-20)
            ),
            "margin_recovery": routing_margin_recovery(
                teacher_logits, removed_logits, topk
            ),
        },
    }


def domain_selectivity(wiki, code, mean, basis, wiki_discovery, wiki_evaluation, seed):
    # Paired dump jobs use the same sample count and tokens-per-sample for each
    # domain.  Reusing the split preserves a sample-disjoint comparison.  If a
    # future dump differs in size, fall back to a deterministic token split.
    if len(code) == len(wiki):
        code_discovery, code_evaluation = wiki_discovery, wiki_evaluation
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(code))
        code_discovery = np.zeros(len(code), dtype=bool)
        code_discovery[order[: len(order) // 2]] = True
        code_evaluation = ~code_discovery
    w = (wiki - mean) @ basis
    c = (code - mean) @ basis
    within = 0.5 * (
        np.mean(np.sum(np.square(w[wiki_evaluation] - w[wiki_evaluation].mean(0)), axis=1))
        + np.mean(np.sum(np.square(c[code_evaluation] - c[code_evaluation].mean(0)), axis=1))
    )
    between = np.sum(np.square(w[wiki_evaluation].mean(0) - c[code_evaluation].mean(0)))
    train_x = np.concatenate([w[wiki_discovery], c[code_discovery]], axis=0)
    train_y = np.concatenate(
        [np.zeros(wiki_discovery.sum()), np.ones(code_discovery.sum())]
    )
    eval_x = np.concatenate([w[wiki_evaluation], c[code_evaluation]], axis=0)
    eval_y = np.concatenate(
        [np.zeros(wiki_evaluation.sum()), np.ones(code_evaluation.sum())]
    )
    classifier = LogisticRegression(max_iter=300, random_state=seed).fit(train_x, train_y)
    probability = classifier.predict_proba(eval_x)[:, 1]
    return {
        "between_domain_over_within_domain": float(between / max(within, 1e-20)),
        "heldout_domain_accuracy": float(np.mean((probability >= 0.5) == eval_y)),
        "heldout_domain_roc_auc": float(roc_auc_score(eval_y, probability)),
        "projected_variance": float(np.var(w[wiki_evaluation], axis=0).sum()),
    }


def sufficiency_metrics(dumps, ranks, seed):
    if not all(domain in dumps for domain in ("wiki", "code")):
        return {"status": "missing Wiki or Code dumps"}
    if not all(stage in dumps["wiki"] for stage in ("B", "C_vocabkl")):
        return {"status": "missing B/C Wiki pair"}
    if "B" not in dumps["code"]:
        return {"status": "missing B Code dump"}

    wiki_before = dumps["wiki"]["B"]
    wiki_after = dumps["wiki"]["C_vocabkl"]
    code_before = dumps["code"]["B"]
    discovery, evaluation = heldout_masks(wiki_before["sample_indices"], seed)
    result = {
        "status": "heldout_router_sufficiency_only_no_forward_lm_intervention_yet",
        "discovery_samples": np.unique(wiki_before["sample_indices"][discovery]).tolist(),
        "evaluation_samples": np.unique(wiki_before["sample_indices"][evaluation]).tolist(),
        "layers": [],
    }

    for layer_index, layer_number in enumerate(wiki_before["layer_numbers"]):
        wb = wiki_before["router_input"][layer_index].astype(np.float64)
        wa = wiki_after["router_input"][layer_index].astype(np.float64)
        cb = code_before["router_input"][layer_index].astype(np.float64)
        mean = wb[discovery].mean(axis=0, keepdims=True)
        centered = wb[discovery] - mean
        max_rank = min(max(ranks), min(centered.shape) - 1)
        _u, singular, vt = randomized_svd(
            centered, n_components=max_rank, random_state=seed + layer_index
        )
        pca_basis = vt.T
        signal = np.square(singular) / max(discovery.sum() - 1, 1)
        delta = wa[discovery] - wb[discovery]
        drift = np.mean(np.square(delta @ pca_basis), axis=0)
        router_weight = wiki_before["router_weights"][layer_index].astype(np.float64)
        sensitivity = np.sum(np.square(router_weight @ pca_basis), axis=0)
        floor = max(float(np.median(drift)) * 1e-6, 1e-12)
        stable_order = np.argsort(signal / (drift + floor))[::-1]
        stable_routing_order = np.argsort(
            signal * np.maximum(sensitivity, 1e-20) / (drift + floor)
        )[::-1]
        row_u, row_s, row_vt = np.linalg.svd(router_weight, full_matrices=False)
        row_rank = int(np.sum(row_s > row_s.max() * 1e-8))
        router_basis = row_vt[:row_rank].T
        discovery_top1 = wiki_before["router_topk_indices"][layer_index, discovery, 0]
        routing_centroids = []
        for expert in np.unique(discovery_top1):
            mask = discovery_top1 == expert
            if mask.sum() >= 2:
                routing_centroids.append(wb[discovery][mask].mean(axis=0) - mean[0])
        if routing_centroids:
            _du, _ds, routing_vt = np.linalg.svd(
                np.stack(routing_centroids), full_matrices=False
            )
            routing_discriminative_basis = routing_vt.T
        else:
            routing_discriminative_basis = router_basis[:, :1]

        eval_x = wb[evaluation]
        teacher_logits = eval_x @ router_weight.T
        teacher_probs, teacher_topk, _ = logits_to_routing(
            teacher_logits, wiki_before["router_topk_indices"].shape[-1]
        )
        layer = {
            "layer": int(layer_number),
            "router_row_rank": row_rank,
            "dead_direction_variance_floor": float(np.max(signal) * 1e-8),
            "curves": [],
        }
        rng = np.random.default_rng(seed + 100 + layer_index)
        for rank in ranks:
            rank = min(rank, max_rank)
            bases = {
                "stable": orthonormalize(pca_basis[:, stable_order], rank),
                "stable_routing_sensitive": orthonormalize(
                    pca_basis[:, stable_routing_order], rank
                ),
                "top_variance_pca": pca_basis[:, :rank],
                "router_row_space": router_basis[:, : min(rank, row_rank)],
                "routing_discriminative": routing_discriminative_basis[
                    :, : min(rank, routing_discriminative_basis.shape[1])
                ],
            }
            random_scores = []
            for repeat in range(5):
                random_basis = orthonormalize(
                    rng.standard_normal((wb.shape[-1], rank))
                )
                random_scores.append(
                    projection_routing_metrics(
                        eval_x,
                        mean,
                        random_basis,
                        router_weight,
                        teacher_probs,
                        teacher_topk,
                    )["fingerprint_only"]["topk_agreement"]
                )
            for name, basis in bases.items():
                metrics = projection_routing_metrics(
                    eval_x, mean, basis, router_weight, teacher_probs, teacher_topk
                )
                selectivity = domain_selectivity(
                    wb,
                    cb,
                    mean,
                    basis,
                    discovery,
                    evaluation,
                    seed + layer_index,
                )
                layer["curves"].append(
                    {
                        "basis": name,
                        "requested_rank": int(rank),
                        "effective_rank": int(basis.shape[1]),
                        "wiki_signal_fraction": float(
                            np.mean(np.sum(np.square((wb[evaluation] - mean) @ basis), axis=1))
                            / max(
                                np.mean(np.sum(np.square(wb[evaluation] - mean), axis=1)),
                                1e-20,
                            )
                        ),
                        "wiki_drift_fraction": float(
                            np.mean(np.sum(np.square((wa[evaluation] - wb[evaluation]) @ basis), axis=1))
                            / max(
                                np.mean(
                                    np.sum(
                                        np.square(wa[evaluation] - wb[evaluation]), axis=1
                                    )
                                ),
                                1e-20,
                            )
                        ),
                        "selectivity": selectivity,
                        **metrics,
                    }
                )
            layer["curves"].append(
                {
                    "basis": "random",
                    "requested_rank": int(rank),
                    "effective_rank": int(rank),
                    "repeats": 5,
                    "fingerprint_only_topk_agreement_mean": float(np.mean(random_scores)),
                    "fingerprint_only_topk_agreement_std": float(np.std(random_scores)),
                }
            )
        result["layers"].append(layer)
    return result


def save_intervention_candidates(dumps, ranks, seed, out_dir):
    """Persist compact B-teacher anchors for actual-forward interventions."""
    if not all(domain in dumps for domain in ("wiki", "code")):
        return []
    if not all(stage in dumps["wiki"] for stage in ("B", "C_vocabkl")):
        return []
    wiki_before = dumps["wiki"]["B"]
    wiki_after = dumps["wiki"]["C_vocabkl"]
    discovery, _evaluation = heldout_masks(wiki_before["sample_indices"], seed)
    layer_payloads = []
    for layer_index, layer_number in enumerate(wiki_before["layer_numbers"]):
        wb = wiki_before["router_input"][layer_index].astype(np.float64)
        wa = wiki_after["router_input"][layer_index].astype(np.float64)
        mean = wb[discovery].mean(axis=0)
        centered = wb[discovery] - mean
        max_rank = min(max(ranks), min(centered.shape) - 1)
        _u, singular, vt = randomized_svd(
            centered, n_components=max_rank, random_state=seed + layer_index
        )
        pca_basis = vt.T
        signal = np.square(singular) / max(discovery.sum() - 1, 1)
        delta = wa[discovery] - wb[discovery]
        drift = np.mean(np.square(delta @ pca_basis), axis=0)
        teacher_weight = wiki_before["router_weights"][layer_index].astype(np.float64)
        sensitivity = np.sum(np.square(teacher_weight @ pca_basis), axis=0)
        floor = max(float(np.median(drift)) * 1e-6, 1e-12)
        stable_order = np.argsort(signal / (drift + floor))[::-1]
        stable_routing_order = np.argsort(
            signal * np.maximum(sensitivity, 1e-20) / (drift + floor)
        )[::-1]
        _row_u, row_s, row_vt = np.linalg.svd(teacher_weight, full_matrices=False)
        row_rank = int(np.sum(row_s > row_s.max() * 1e-8))
        layer_payloads.append(
            {
                "layer": int(layer_number),
                "mean": mean,
                "teacher_weight": teacher_weight,
                "stable": pca_basis[:, stable_order],
                "stable_routing_sensitive": pca_basis[:, stable_routing_order],
                "top_variance_pca": pca_basis,
                "router_row_space": row_vt[:row_rank].T,
                "routing_discriminative": None,
            }
        )
        discovery_top1 = wiki_before["router_topk_indices"][layer_index, discovery, 0]
        routing_centroids = []
        for expert in np.unique(discovery_top1):
            mask = discovery_top1 == expert
            if mask.sum() >= 2:
                routing_centroids.append(wb[discovery][mask].mean(axis=0) - mean)
        if routing_centroids:
            _du, _ds, routing_vt = np.linalg.svd(
                np.stack(routing_centroids), full_matrices=False
            )
            layer_payloads[-1]["routing_discriminative"] = routing_vt.T
        else:
            layer_payloads[-1]["routing_discriminative"] = row_vt[:1].T

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    rng = np.random.default_rng(seed + 900)
    for requested_rank in ranks:
        for basis_name in (
            "stable",
            "stable_routing_sensitive",
            "top_variance_pca",
            "router_row_space",
            "routing_discriminative",
        ):
            effective_rank = min(
                requested_rank,
                min(payload[basis_name].shape[1] for payload in layer_payloads),
            )
            bases = np.stack(
                [
                    orthonormalize(payload[basis_name], effective_rank).astype(np.float32)
                    for payload in layer_payloads
                ]
            )
            path = out_dir / f"{basis_name}_r{requested_rank}_effective{effective_rank}.npz"
            np.savez_compressed(
                path,
                layer_numbers=np.asarray(
                    [payload["layer"] for payload in layer_payloads], dtype=np.int64
                ),
                means=np.stack([payload["mean"] for payload in layer_payloads]).astype(np.float32),
                bases=bases,
                teacher_router_weights=np.stack(
                    [payload["teacher_weight"] for payload in layer_payloads]
                ).astype(np.float32),
                metadata=json.dumps(
                    {
                        "basis": basis_name,
                        "requested_rank": requested_rank,
                        "effective_rank": effective_rank,
                        "teacher_stage": "B",
                        "oracle_discovery_domain": "wiki",
                        "oracle_stability_pair": "B_to_C_vocabkl",
                        "intervened_layers": [payload["layer"] for payload in layer_payloads],
                    },
                    sort_keys=True,
                ),
            )
            written.append(str(path))

        random_bases = np.stack(
            [
                orthonormalize(rng.standard_normal((payload["mean"].shape[0], requested_rank)))
                .astype(np.float32)
                for payload in layer_payloads
            ]
        )
        path = out_dir / f"random_r{requested_rank}_repeat0.npz"
        np.savez_compressed(
            path,
            layer_numbers=np.asarray(
                [payload["layer"] for payload in layer_payloads], dtype=np.int64
            ),
            means=np.stack([payload["mean"] for payload in layer_payloads]).astype(np.float32),
            bases=random_bases,
            teacher_router_weights=np.stack(
                [payload["teacher_weight"] for payload in layer_payloads]
            ).astype(np.float32),
            metadata=json.dumps(
                {
                    "basis": "random",
                    "requested_rank": requested_rank,
                    "effective_rank": requested_rank,
                    "repeat": 0,
                    "teacher_stage": "B",
                    "intervened_layers": [payload["layer"] for payload in layer_payloads],
                },
                sort_keys=True,
            ),
        )
        written.append(str(path))
    return written


def plot_common_pca(dumps, out_path, seed):
    domains = sorted(dumps)
    reference = next(iter(next(iter(dumps.values())).values()))
    layers = reference["layer_numbers"]
    fig, axes = plt.subplots(len(layers), len(domains), figsize=(7 * len(domains), 5 * len(layers)))
    axes = np.asarray(axes).reshape(len(layers), len(domains))
    for layer_index, layer_number in enumerate(layers):
        arrays = []
        labels = []
        for domain in domains:
            for stage in STAGE_ORDER:
                if stage in dumps[domain]:
                    arrays.append(dumps[domain][stage]["router_input"][layer_index])
                    labels.append((domain, stage))
        joined = np.concatenate(arrays, axis=0)
        pca = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit(joined)
        coords = [pca.transform(array) for array in arrays]
        all_coords = np.concatenate(coords)
        lo = np.percentile(all_coords, 0.5, axis=0)
        hi = np.percentile(all_coords, 99.5, axis=0)
        pad = 0.08 * np.maximum(hi - lo, 1e-6)
        limits = (lo - pad, hi + pad)
        by_label = dict(zip(labels, coords))
        for domain_index, domain in enumerate(domains):
            ax = axes[layer_index, domain_index]
            for stage in STAGE_ORDER:
                points = by_label.get((domain, stage))
                if points is None:
                    continue
                hist, xedge, yedge = np.histogram2d(
                    points[:, 0], points[:, 1], bins=80,
                    range=[[limits[0][0], limits[1][0]], [limits[0][1], limits[1][1]]],
                )
                hist = gaussian_filter(hist, sigma=1.25)
                positive = hist[hist > 0]
                if positive.size:
                    levels = np.quantile(positive, [0.55, 0.75, 0.9])
                    levels = np.unique(levels)
                    if len(levels) > 1:
                        xc = 0.5 * (xedge[:-1] + xedge[1:])
                        yc = 0.5 * (yedge[:-1] + yedge[1:])
                        ax.contour(xc, yc, hist.T, levels=levels, colors=STAGE_COLORS[stage], linewidths=1.4)
                ax.scatter([], [], color=STAGE_COLORS[stage], label=stage)
            ax.set_xlim(limits[0][0], limits[1][0])
            ax.set_ylim(limits[0][1], limits[1][1])
            ax.set_title(
                f"Layer {int(layer_number)} | {domain} | common PCA "
                f"({pca.explained_variance_ratio_.sum():.1%})"
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.grid(alpha=0.15)
            ax.legend(frameon=False)
    fig.suptitle("Router-input density on one joint PCA basis per layer")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_token_displacement(dumps, out_path, seed):
    panels = []
    for domain, before_name, after_name in (
        ("wiki", "A", "B"),
        ("wiki", "B", "C_vocabkl"),
        ("code", "A", "B"),
    ):
        if domain in dumps and before_name in dumps[domain] and after_name in dumps[domain]:
            panels.append((domain, before_name, after_name))
    if not panels:
        return
    reference = dumps[panels[0][0]][panels[0][1]]
    layers = reference["layer_numbers"]
    fig, axes = plt.subplots(len(layers), len(panels), figsize=(6 * len(panels), 5 * len(layers)))
    axes = np.asarray(axes).reshape(len(layers), len(panels))
    rng = np.random.default_rng(seed)
    for layer_index, layer_number in enumerate(layers):
        for panel_index, (domain, before_name, after_name) in enumerate(panels):
            before = dumps[domain][before_name]
            after = dumps[domain][after_name]
            x = before["router_input"][layer_index]
            y = after["router_input"][layer_index]
            pca = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit(
                np.concatenate([x, y])
            )
            xy0, xy1 = pca.transform(x), pca.transform(y)
            count = min(600, len(x))
            idx = rng.choice(len(x), size=count, replace=False)
            stable = np.all(
                np.sort(before["router_topk_indices"][layer_index], axis=-1)
                == np.sort(after["router_topk_indices"][layer_index], axis=-1),
                axis=-1,
            )
            ax = axes[layer_index, panel_index]
            displacement = xy1 - xy0
            selected_stable = stable[idx]
            ax.scatter(
                displacement[idx[selected_stable], 0],
                displacement[idx[selected_stable], 1],
                s=9, color="#059669", alpha=0.42, label="routing stable",
            )
            ax.scatter(
                displacement[idx[~selected_stable], 0],
                displacement[idx[~selected_stable], 1],
                s=13, color="#dc2626", alpha=0.7, label="routing changed",
            )
            ax.axhline(0, color="#6b7280", linewidth=0.7, alpha=0.5)
            ax.axvline(0, color="#6b7280", linewidth=0.7, alpha=0.5)
            ax.set_title(
                f"L{int(layer_number)} {domain}: {before_name}→{after_name}\n"
                f"green=routing stable ({stable.mean():.1%})"
            )
            ax.set_xlabel("ΔPC1")
            ax.set_ylabel("ΔPC2")
            ax.legend(frameon=False)
            ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_code_conversation_displacement(dumps, out_path, seed):
    """Plot F->G and G->H token motion for every available domain.

    F and C_vocabkl are the same physical checkpoint in this checkpoint
    family.  Most Wiki/Code dump jobs therefore store it only once under the
    C_vocabkl name; resolve that alias without duplicating a GPU dump.
    """
    panels = []
    for before_label, after_name in (("F", "G"), ("G", "H")):
        for domain in ("wiki", "code", "conversation"):
            if domain not in dumps:
                continue
            stages = dumps[domain]
            before_name = before_label
            if before_label == "F" and "F" not in stages and "C_vocabkl" in stages:
                before_name = "C_vocabkl"
            if before_name in stages and after_name in stages:
                panels.append((domain, before_label, before_name, after_name))
    if not panels:
        return

    reference = dumps[panels[0][0]][panels[0][2]]
    layers = reference["layer_numbers"]
    fig, axes = plt.subplots(
        len(layers), len(panels),
        figsize=(4.8 * len(panels), 4.5 * len(layers)),
        squeeze=False,
    )
    rng = np.random.default_rng(seed + 71)
    for layer_index, layer_number in enumerate(layers):
        for panel_index, (domain, before_label, before_name, after_name) in enumerate(panels):
            before = dumps[domain][before_name]
            after = dumps[domain][after_name]
            if not np.array_equal(before["layer_numbers"], layers):
                raise RuntimeError(f"{domain}/{before_name}: displacement layer mismatch")
            if not np.array_equal(after["layer_numbers"], layers):
                raise RuntimeError(f"{domain}/{after_name}: displacement layer mismatch")
            x = before["router_input"][layer_index]
            y = after["router_input"][layer_index]
            pca = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit(
                np.concatenate([x, y])
            )
            xy0, xy1 = pca.transform(x), pca.transform(y)
            count = min(600, len(x))
            idx = rng.choice(len(x), size=count, replace=False)
            stable = np.all(
                np.sort(before["router_topk_indices"][layer_index], axis=-1)
                == np.sort(after["router_topk_indices"][layer_index], axis=-1),
                axis=-1,
            )
            displacement = xy1 - xy0
            selected_stable = stable[idx]
            ax = axes[layer_index, panel_index]
            ax.scatter(
                displacement[idx[selected_stable], 0],
                displacement[idx[selected_stable], 1],
                s=8, color="#059669", alpha=0.4, label="routing stable",
            )
            ax.scatter(
                displacement[idx[~selected_stable], 0],
                displacement[idx[~selected_stable], 1],
                s=12, color="#dc2626", alpha=0.68, label="routing changed",
            )
            ax.axhline(0, color="#6b7280", linewidth=0.7, alpha=0.5)
            ax.axvline(0, color="#6b7280", linewidth=0.7, alpha=0.5)
            ax.set_title(
                f"L{int(layer_number)} {domain}: {before_label}→{after_name}\n"
                f"routing stable={stable.mean():.1%}"
            )
            ax.set_xlabel("ΔPC1")
            ax.set_ylabel("ΔPC2")
            ax.legend(frameon=False)
            ax.grid(alpha=0.15)
    fig.suptitle("Code→Conversation router-input displacement on pairwise common PCA bases")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_model_domain_joint_density(dumps, out_path, seed):
    if not all(domain in dumps for domain in ("wiki", "code")):
        return
    stages = [
        stage for stage in STAGE_ORDER
        if stage in dumps["wiki"] and stage in dumps["code"]
    ]
    if not stages:
        return
    reference = dumps["wiki"][stages[0]]
    layers = reference["layer_numbers"]
    fig, axes = plt.subplots(
        len(layers), len(stages), figsize=(6 * len(stages), 4.8 * len(layers))
    )
    axes = np.asarray(axes).reshape(len(layers), len(stages))
    domain_colors = {"wiki": "#2563eb", "code": "#f97316"}
    for layer_index, layer_number in enumerate(layers):
        arrays = [
            dumps[domain][stage]["router_input"][layer_index]
            for stage in stages for domain in ("wiki", "code")
        ]
        joined = np.concatenate(arrays)
        pca = PCA(n_components=2, svd_solver="randomized", random_state=seed).fit(joined)
        coords = [pca.transform(array) for array in arrays]
        all_coords = np.concatenate(coords)
        lo = np.percentile(all_coords, 0.5, axis=0)
        hi = np.percentile(all_coords, 99.5, axis=0)
        pad = 0.08 * np.maximum(hi - lo, 1e-6)
        lo, hi = lo - pad, hi + pad
        cursor = 0
        for stage_index, stage in enumerate(stages):
            ax = axes[layer_index, stage_index]
            for domain in ("wiki", "code"):
                points = coords[cursor]
                cursor += 1
                hist, xedge, yedge = np.histogram2d(
                    points[:, 0], points[:, 1], bins=80,
                    range=[[lo[0], hi[0]], [lo[1], hi[1]]],
                )
                hist = gaussian_filter(hist, sigma=1.25)
                positive = hist[hist > 0]
                if positive.size:
                    levels = np.unique(np.quantile(positive, [0.55, 0.75, 0.9]))
                    if len(levels) > 1:
                        xc = 0.5 * (xedge[:-1] + xedge[1:])
                        yc = 0.5 * (yedge[:-1] + yedge[1:])
                        ax.contour(
                            xc, yc, hist.T, levels=levels,
                            colors=domain_colors[domain], linewidths=1.4,
                        )
                ax.scatter([], [], color=domain_colors[domain], label=domain)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_title(
                f"L{int(layer_number)} {stage} Wiki–Code | common PCA "
                f"({pca.explained_variance_ratio_.sum():.1%})"
            )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.legend(frameon=False)
            ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def plot_expert_usage(dumps, out_path):
    reference = next(iter(next(iter(dumps.values())).values()))
    layers = reference["layer_numbers"]
    fig, axes = plt.subplots(len(layers), 1, figsize=(13, 4.5 * len(layers)))
    axes = np.atleast_1d(axes)
    for layer_index, layer_number in enumerate(layers):
        ax = axes[layer_index]
        offset = 0
        ticks, ticklabels = [], []
        for domain in sorted(dumps):
            for stage in STAGE_ORDER:
                if stage not in dumps[domain]:
                    continue
                indices = dumps[domain][stage]["router_topk_indices"][layer_index].reshape(-1)
                num_experts = dumps[domain][stage]["router_weights"].shape[1]
                hist = np.bincount(indices, minlength=num_experts) / len(indices)
                x = np.arange(num_experts) + offset
                ax.bar(x, hist, width=0.8, color=STAGE_COLORS[stage], alpha=0.78)
                ticks.append(offset + (num_experts - 1) / 2)
                ticklabels.append(f"{domain}\n{stage}")
                offset += num_experts + 2
        ax.set_xticks(ticks, ticklabels)
        ax.set_ylabel("top-k slot fraction")
        ax.set_title(f"Layer {int(layer_number)} expert assignment")
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.write_text(json.dumps(sanitize(value), indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    ranks = sorted({int(rank) for rank in args.ranks.split(",") if int(rank) > 0})
    dumps = discover_dumps(root)
    if not dumps:
        raise SystemExit(f"No detailed dumps found under {root / 'dumps'}")
    for subdir in ("inventory", "metrics", "plots", "report"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    audit = validate_alignment(dumps)
    comparisons = comparison_metrics(dumps, ranks, args.seed)
    sufficiency = sufficiency_metrics(dumps, ranks, args.seed)
    candidate_paths = save_intervention_candidates(
        dumps, ranks, args.seed, root / "metrics" / "fingerprints"
    )
    write_json(root / "inventory" / "dump_manifest.json", audit)
    write_json(root / "metrics" / "representation_routing_drift.json", comparisons)
    write_json(root / "metrics" / "fingerprint_sufficiency.json", sufficiency)
    plot_common_pca(dumps, root / "plots" / "common_pca_router_input_density.png", args.seed)
    plot_token_displacement(dumps, root / "plots" / "token_displacement_routing_stability.png", args.seed)
    plot_code_conversation_displacement(
        dumps,
        root / "plots" / "code_conversation_token_displacement.png",
        args.seed,
    )
    plot_model_domain_joint_density(
        dumps, root / "plots" / "model_wiki_code_joint_density.png", args.seed
    )
    plot_expert_usage(dumps, root / "plots" / "expert_assignment_by_layer.png")
    print(
        json.dumps(
            {
                "root": str(root),
                "domains": {domain: sorted(stages) for domain, stages in dumps.items()},
                "outputs": {
                    "manifest": str(root / "inventory" / "dump_manifest.json"),
                    "drift": str(root / "metrics" / "representation_routing_drift.json"),
                    "sufficiency": str(root / "metrics" / "fingerprint_sufficiency.json"),
                    "intervention_candidates": candidate_paths,
                    "plots": str(root / "plots"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
