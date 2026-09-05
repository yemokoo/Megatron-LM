#!/usr/bin/env python3
"""Diagnose compact, stable hidden-space fingerprint candidates.

The analysis learns directions only from a discovery split of matched Wiki
tokens and reports all headline metrics on a held-out split.  It does not claim
causal sufficiency for routing; that requires router-output intervention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wiki-before", required=True)
    p.add_argument("--wiki-after", required=True)
    p.add_argument("--wiki-original", default=None)
    p.add_argument("--code-before", required=True)
    p.add_argument("--code-after", required=True)
    p.add_argument("--code-original", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--discovery-fraction", type=float, default=0.5)
    p.add_argument("--ranks", default="8,16,32,64,128,256,512")
    return p.parse_args()


def load_dump(path):
    with np.load(path, allow_pickle=False) as z:
        return {
            "hidden": z["hidden_layers"].astype(np.float32),
            "layers": z["layer_numbers"].astype(np.int64),
            "token_ids": z["token_ids"].astype(np.int64),
            "positions": z["positions"].astype(np.int64),
            "sample_indices": z["sample_indices"].astype(np.int64),
        }


def validate_pair(a, b, name):
    for key in ("layers", "token_ids", "positions", "sample_indices"):
        if not np.array_equal(a[key], b[key]):
            raise SystemExit(f"{name}: unmatched {key}")
    if a["hidden"].shape != b["hidden"].shape:
        raise SystemExit(f"{name}: hidden shape mismatch")


def paired_arrays(before, after):
    dot = np.sum(before * after, axis=1)
    denom = np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-8
    )
    cosine = np.clip(dot / denom, -1.0, 1.0)
    nl2 = 2.0 * np.linalg.norm(after - before, axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) + np.linalg.norm(after, axis=1), 1e-8
    )
    return cosine, nl2


def paired_stats(before, after):
    cosine, nl2 = paired_arrays(before, after)
    return {
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "fraction_cosine_ge_0_99": float(np.mean(cosine >= 0.99)),
        "fraction_cosine_ge_0_999": float(np.mean(cosine >= 0.999)),
        "nl2_median": float(np.median(nl2)),
        "nl2_p95": float(np.percentile(nl2, 95)),
    }


def original_alignment_stats(original, before, after, evaluation):
    """Test whether features stable during T->S also remain aligned to W0."""
    ts_cos, ts_distance = paired_arrays(before[evaluation], after[evaluation])
    os_cos, os_distance = paired_arrays(original[evaluation], after[evaluation])
    result = {
        "all_tokens": {
            "original_to_after_cosine_mean": float(os_cos.mean()),
            "original_to_after_fraction_cosine_ge_0_99": float(np.mean(os_cos >= 0.99)),
            "original_to_after_nl2_median": float(np.median(os_distance)),
        },
        "pearson_distill_vs_original_distance": float(
            np.corrcoef(ts_distance, os_distance)[0, 1]
        ),
        "stable_sets": {},
    }
    for fraction in (0.10, 0.25):
        threshold = float(np.quantile(ts_distance, fraction))
        selected = ts_distance <= threshold
        result["stable_sets"][f"lowest_{int(fraction * 100)}pct_distill_distance"] = {
            "tokens": int(selected.sum()),
            "distill_distance_threshold": threshold,
            "teacher_to_student_cosine_mean": float(ts_cos[selected].mean()),
            "original_to_after_cosine_mean": float(os_cos[selected].mean()),
            "original_to_after_fraction_cosine_ge_0_99": float(
                np.mean(os_cos[selected] >= 0.99)
            ),
            "original_to_after_nl2_median": float(np.median(os_distance[selected])),
        }
    return result


def energy(x):
    return float(np.mean(np.sum(np.square(x, dtype=np.float64), axis=1)))


def learn_stable_basis(wiki_before, wiki_after, discovery):
    """Rank Wiki PCA directions by signal variance / observed Wiki drift."""
    wb = wiki_before[discovery].astype(np.float64)
    wa = wiki_after[discovery].astype(np.float64)
    mean = wb.mean(axis=0, keepdims=True)
    centered = wb - mean
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    wiki_delta = wa - wb
    drift = np.mean(np.square(wiki_delta @ vectors), axis=0)
    floor = max(float(np.median(drift)) * 1e-6, 1e-12)
    score = values / (drift + floor)
    stable_order = np.argsort(score)[::-1]
    return mean.astype(np.float32), vectors[:, stable_order].astype(np.float32), {
        "signal_variance": values[stable_order],
        "wiki_drift": drift[stable_order],
        "stability_score": score[stable_order],
    }


def eval_rank_curve(basis, wiki_before, wiki_after, code_before, code_after, evaluation, ranks):
    wb = wiki_before[evaluation]
    wa = wiki_after[evaluation]
    cb = code_before[evaluation]
    ca = code_after[evaluation]
    dw = wa - wb
    dc = ca - cb
    wb_centered = wb - wb.mean(axis=0, keepdims=True)
    total_signal = max(energy(wb_centered), 1e-12)
    total_wiki_drift = max(energy(dw), 1e-12)
    total_code_drift = max(energy(dc), 1e-12)
    rows = []
    for rank in ranks:
        rank = min(rank, basis.shape[1])
        u = basis[:, :rank]
        signal = energy(wb_centered @ u)
        wiki_drift = energy(dw @ u)
        code_drift = energy(dc @ u)
        rows.append(
            {
                "rank": int(rank),
                "wiki_signal_fraction": signal / total_signal,
                "wiki_drift_fraction": wiki_drift / total_wiki_drift,
                "code_drift_fraction": code_drift / total_code_drift,
                "wiki_relative_drift": wiki_drift / max(signal, 1e-12),
                "code_to_wiki_projected_drift": code_drift / max(wiki_drift, 1e-12),
            }
        )
    return rows


def aggregate_curves(layer_rows, ranks):
    keys = (
        "wiki_signal_fraction",
        "wiki_drift_fraction",
        "code_drift_fraction",
        "wiki_relative_drift",
        "code_to_wiki_projected_drift",
    )
    result = []
    # Layers 2..L-1 are the non-trivial intermediate MoE outputs. Layer 1 is
    # exactly unchanged in this FFN-only setup and would inflate stability;
    # the final output does not feed another Transformer layer.
    selected = layer_rows[1:-1] if len(layer_rows) > 2 else layer_rows
    for idx, rank in enumerate(ranks):
        row = {"rank": rank}
        for key in keys:
            vals = [layer["rank_curve"][idx][key] for layer in selected]
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_min"] = float(np.min(vals))
            row[f"{key}_max"] = float(np.max(vals))
        result.append(row)
    return result


def plot_token_stability(layer_rows, out_path):
    layers = [r["layer"] for r in layer_rows]
    wiki = [r["wiki_eval"]["fraction_cosine_ge_0_99"] for r in layer_rows]
    code = [r["code_eval"]["fraction_cosine_ge_0_99"] for r in layer_rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(layers, wiki, marker="o", label="Wiki", color="#2563eb")
    ax.plot(layers, code, marker="o", label="Code", color="#f97316")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Held-out tokens with cosine ≥ 0.99")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(layers)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Stable Same-Token Hidden Population")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_rank_curve(aggregate, out_path):
    ranks = [r["rank"] for r in aggregate]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for key, label, color in (
        ("wiki_signal_fraction_mean", "Wiki signal captured", "#2563eb"),
        ("wiki_drift_fraction_mean", "Wiki drift captured", "#dc2626"),
        ("code_drift_fraction_mean", "Code drift captured", "#f97316"),
    ):
        axes[0].plot(ranks, [r[key] for r in aggregate], marker="o", label=label, color=color)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xlabel("Stable-subspace dimension")
    axes[0].set_ylabel("Fraction of full-space energy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Held-Out Compactness")

    axes[1].plot(
        ranks,
        [r["code_to_wiki_projected_drift_mean"] for r in aggregate],
        marker="o",
        color="#059669",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Stable-subspace dimension")
    axes[1].set_ylabel("Projected Code drift / Wiki drift")
    axes[1].grid(alpha=0.25)
    axes[1].set_title("Domain-Selective Plasticity")
    fig.suptitle("Wiki-Stable Hidden Fingerprint Candidate (intermediate MoE layer mean)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_original_alignment(layer_rows, out_path):
    layers = [r["layer"] for r in layer_rows]
    stats = [r["wiki_distill_stability_vs_original_alignment"] for r in layer_rows]
    all_tokens = [
        s["all_tokens"]["original_to_after_fraction_cosine_ge_0_99"] for s in stats
    ]
    stable_25 = [
        s["stable_sets"]["lowest_25pct_distill_distance"][
            "original_to_after_fraction_cosine_ge_0_99"
        ]
        for s in stats
    ]
    stable_10 = [
        s["stable_sets"]["lowest_10pct_distill_distance"][
            "original_to_after_fraction_cosine_ge_0_99"
        ]
        for s in stats
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(layers, all_tokens, marker="o", label="All Wiki tokens", color="#94a3b8")
    ax.plot(layers, stable_25, marker="o", label="Lowest 25% T→S distance", color="#2563eb")
    ax.plot(layers, stable_10, marker="o", label="Lowest 10% T→S distance", color="#059669")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("W0→S tokens with cosine ≥ 0.99")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(layers)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Do Distillation-Stable Wiki Features Stay Aligned to the Original Model?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    ranks = sorted({int(v) for v in args.ranks.split(",") if int(v) > 0})
    wiki_before = load_dump(args.wiki_before)
    wiki_after = load_dump(args.wiki_after)
    wiki_original = load_dump(args.wiki_original) if args.wiki_original else None
    code_before = load_dump(args.code_before)
    code_after = load_dump(args.code_after)
    code_original = load_dump(args.code_original) if args.code_original else None
    validate_pair(wiki_before, wiki_after, "wiki")
    validate_pair(code_before, code_after, "code")
    if wiki_original is not None:
        validate_pair(wiki_original, wiki_before, "wiki original/before")
    if code_original is not None:
        validate_pair(code_original, code_before, "code original/before")
    if not np.array_equal(wiki_before["layers"], code_before["layers"]):
        raise SystemExit("Wiki/Code layer mismatch")
    n = min(wiki_before["hidden"].shape[1], code_before["hidden"].shape[1])
    if n < 32:
        raise SystemExit("Too few matched tokens")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_discovery = int(round(n * args.discovery_fraction))
    n_discovery = min(max(n_discovery, 16), n - 16)
    discovery, evaluation = perm[:n_discovery], perm[n_discovery:]

    layer_rows = []
    for layer_idx, layer_number in enumerate(wiki_before["layers"]):
        wb = wiki_before["hidden"][layer_idx, :n]
        wa = wiki_after["hidden"][layer_idx, :n]
        cb = code_before["hidden"][layer_idx, :n]
        ca = code_after["hidden"][layer_idx, :n]
        _mean, basis, component = learn_stable_basis(wb, wa, discovery)
        curves = eval_rank_curve(basis, wb, wa, cb, ca, evaluation, ranks)
        row = {
                "layer": int(layer_number),
                "wiki_eval": paired_stats(wb[evaluation], wa[evaluation]),
                "code_eval": paired_stats(cb[evaluation], ca[evaluation]),
                "rank_curve": curves,
                "top_component_stability_score": float(component["stability_score"][0]),
            }
        if wiki_original is not None:
            row["wiki_distill_stability_vs_original_alignment"] = original_alignment_stats(
                wiki_original["hidden"][layer_idx, :n], wb, wa, evaluation
            )
        if code_original is not None:
            row["code_distill_stability_vs_original_alignment"] = original_alignment_stats(
                code_original["hidden"][layer_idx, :n], cb, ca, evaluation
            )
        layer_rows.append(row)
    aggregate = aggregate_curves(layer_rows, ranks)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "candidate_only_not_causal_sufficiency",
        "tokens_used": n,
        "discovery_tokens": len(discovery),
        "evaluation_tokens": len(evaluation),
        "seed": args.seed,
        "layers": layer_rows,
        "intermediate_moe_layer_mean": aggregate,
    }
    (out / "fingerprint_candidate_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    plot_token_stability(layer_rows, out / "stable_token_fraction_by_layer.png")
    plot_rank_curve(aggregate, out / "stable_subspace_rank_curve.png")
    if wiki_original is not None:
        plot_original_alignment(layer_rows, out / "distill_stable_original_alignment.png")
    print(json.dumps({"out": str(out), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
