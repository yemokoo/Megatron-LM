#!/usr/bin/env python3
"""Analyze residual-included layer-output stability from streaming block moments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import linalg


RANKS = (8, 16, 32, 64, 128, 256)


def parse_named_path(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    return label, os.path.abspath(path)


def load_block(path: str, index: int) -> dict:
    with np.load(os.path.join(path, f"block_{index:03d}.npz")) as data:
        return {key: data[key] for key in data.files}


def combine(blocks: list[dict]) -> dict:
    keys = ("sum_x", "sum_y", "sum_delta", "xx", "yy", "xy",
            "cosine_hist", "relative_l2_hist", "scalar_sums", "stable_counts")
    result = {key: sum((block[key] for block in blocks)) for key in keys}
    for key in ("ffn_cosine_hist", "ffn_relative_l2_hist", "ffn_scalar_sums", "ffn_stable_counts"):
        if key in blocks[0]:
            result[key] = sum((block[key] for block in blocks))
    result["count"] = int(sum(int(block["count"]) for block in blocks))
    result["layer_numbers"] = blocks[0]["layer_numbers"]
    return result


def layer_moments(stats: dict, layer_index: int) -> dict:
    n = float(stats["count"])
    mx = stats["sum_x"][layer_index] / n
    my = stats["sum_y"][layer_index] / n
    md = stats["sum_delta"][layer_index] / n
    exx = stats["xx"][layer_index] / n
    eyy = stats["yy"][layer_index] / n
    exy = stats["xy"][layer_index] / n
    sigma_x = (exx - np.outer(mx, mx))
    sigma_y = (eyy - np.outer(my, my))
    cross = (exy - np.outer(mx, my))
    raw_delta = exx + eyy - exy - exy.T
    centered_delta = raw_delta - np.outer(md, md)
    return {
        "mean_x": mx,
        "mean_y": my,
        "mean_delta": md,
        "sigma_x": (sigma_x + sigma_x.T) / 2,
        "sigma_y": (sigma_y + sigma_y.T) / 2,
        "cross": cross,
        "raw_delta": (raw_delta + raw_delta.T) / 2,
        "centered_delta": (centered_delta + centered_delta.T) / 2,
    }


def orthonormal(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.qr(matrix, mode="reduced")[0]


def stable_basis(sigma: np.ndarray, delta: np.ndarray, rank: int) -> tuple[np.ndarray, float]:
    hidden = sigma.shape[0]
    scale = max(float(np.trace(delta)) / hidden, 1e-12)
    epsilon = scale * 1e-4
    metric = delta + epsilon * np.eye(hidden)
    try:
        _values, vectors = linalg.eigh(
            sigma, metric, subset_by_index=(hidden - rank, hidden - 1),
            driver="gvx", check_finite=False,
        )
    except linalg.LinAlgError:
        epsilon = scale * 1e-2
        metric = delta + epsilon * np.eye(hidden)
        _values, vectors = linalg.eigh(
            sigma, metric, subset_by_index=(hidden - rank, hidden - 1),
            driver="gvx", check_finite=False,
        )
    return orthonormal(vectors[:, ::-1]), epsilon


def pca_bases(sigma: np.ndarray, rank: int):
    hidden = sigma.shape[0]
    _values, top = linalg.eigh(
        sigma, subset_by_index=(hidden - rank, hidden - 1), driver="evr", check_finite=False
    )
    _values, bottom = linalg.eigh(
        sigma, subset_by_index=(0, rank - 1), driver="evr", check_finite=False
    )
    return top[:, ::-1], bottom


def projected_trace(matrix: np.ndarray, basis: np.ndarray) -> float:
    return float(np.einsum("ik,ij,jk->", basis, matrix, basis, optimize=True))


def basis_metrics(moments: dict, basis: np.ndarray) -> dict:
    old = projected_trace(moments["sigma_x"], basis)
    drift = projected_trace(moments["raw_delta"], basis)
    total_old = max(float(np.trace(moments["sigma_x"])), 1e-20)
    total_drift = max(float(np.trace(moments["raw_delta"])), 1e-20)
    return {
        "old_variance_fraction": old / total_old,
        "drift_energy_fraction": drift / total_drift,
        "old_variance_per_direction": old / basis.shape[1],
        "drift_per_direction": drift / basis.shape[1],
        "variance_to_drift_ratio": old / max(drift, 1e-20),
    }


def whole_alignment(moments: dict) -> dict:
    sx, sy, cross = moments["sigma_x"], moments["sigma_y"], moments["cross"]
    cka = np.square(cross).sum() / max(
        np.sqrt(np.square(sx).sum() * np.square(sy).sum()), 1e-20
    )
    procrustes = np.linalg.svd(cross, compute_uv=False).sum() / max(
        np.sqrt(np.trace(sx) * np.trace(sy)), 1e-20
    )
    return {"linear_cka": float(cka), "normalized_procrustes": float(procrustes)}


def _hist_quantile(hist: np.ndarray, low: float, high: float, quantile: float) -> float:
    cumulative = np.cumsum(hist)
    if cumulative[-1] <= 0:
        return float("nan")
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return low + (index + 0.5) * (high - low) / len(hist)


def token_drift_summary(stats: dict, layer_index: int, prefix: str = "") -> dict:
    n = float(stats["count"])
    sums = stats[f"{prefix}scalar_sums"][layer_index]
    counts = stats[f"{prefix}stable_counts"][layer_index]
    cosine_hist = stats[f"{prefix}cosine_hist"][layer_index]
    relative_hist = stats[f"{prefix}relative_l2_hist"][layer_index][1:-1]
    return {
        "mean_cosine": float(sums[0] / n),
        "mean_relative_l2": float(sums[1] / n),
        "mean_before_norm": float(sums[2] / n),
        "mean_after_norm": float(sums[3] / n),
        "mean_delta_norm": float(sums[4] / n),
        "mean_delta_squared_norm": float(sums[5] / n),
        "cosine_q10_q50_q90": [
            _hist_quantile(cosine_hist, -1, 1, q) for q in (0.1, 0.5, 0.9)
        ],
        "relative_l2_q10_q50_q90": [
            _hist_quantile(relative_hist, 0, 2, q) for q in (0.1, 0.5, 0.9)
        ],
        "fractions": {
            "cosine_ge_0.99": float(counts[0] / n),
            "cosine_ge_0.999": float(counts[1] / n),
            "relative_l2_le_0.01": float(counts[2] / n),
            "relative_l2_le_0.05": float(counts[3] / n),
            "relative_l2_le_0.10": float(counts[4] / n),
            "cosine_ge_0.99_and_relative_l2_le_0.05": float(counts[5] / n),
        },
    }


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return {
        "mean_squared_cosine": float(np.square(singular).mean()),
        "min_cosine": float(singular.min()),
        "mean_angle_degrees": float(np.degrees(np.arccos(np.clip(singular, -1, 1))).mean()),
    }


def random_summary(moments: dict, hidden: int, rank: int, repeats: int, seed: int) -> dict:
    rows = []
    rng = np.random.default_rng(seed)
    for _ in range(repeats):
        rows.append(basis_metrics(moments, orthonormal(rng.standard_normal((hidden, rank)))))
    result = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows])
        result[key] = {
            "mean": float(values.mean()),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        }
    return result


def analyze_pair(label: str, path: str, repeats: int, seed: int):
    blocks = [load_block(path, index) for index in range(5)]
    discovery = combine(blocks[:3])
    validation = combine([blocks[3]])
    test = combine([blocks[4]])
    layer_numbers = discovery["layer_numbers"].astype(int).tolist()
    pair_result = {"label": label, "path": path, "layers": {}, "split_tokens": [6000000, 2000000, 2000000]}
    bases = {}
    means = {}
    for layer_index, layer in enumerate(layer_numbers):
        train_m = layer_moments(discovery, layer_index)
        valid_m = layer_moments(validation, layer_index)
        test_m = layer_moments(test, layer_index)
        hidden = train_m["sigma_x"].shape[0]
        max_rank = min(max(RANKS), hidden)
        stable_full, epsilon = stable_basis(train_m["sigma_x"], train_m["raw_delta"], max_rank)
        pca_top_full, pca_bottom_full = pca_bases(train_m["sigma_x"], max_rank)
        bases[layer] = stable_full
        means[layer] = train_m["mean_x"]
        layer_result = {
            "epsilon": epsilon,
            "full_alignment_validation": whole_alignment(valid_m),
            "full_alignment_test": whole_alignment(test_m),
            "full_hidden": {
                "validation": basis_metrics(valid_m, np.eye(hidden)),
                "test": basis_metrics(test_m, np.eye(hidden)),
            },
            "token_drift_validation": token_drift_summary(validation, layer_index),
            "token_drift_test": token_drift_summary(test, layer_index),
            "ranks": {},
            "block_reproducibility": {},
        }
        if "ffn_scalar_sums" in validation:
            layer_result["ffn_output_diagnostic_validation"] = token_drift_summary(
                validation, layer_index, "ffn_"
            )
            layer_result["ffn_output_diagnostic_test"] = token_drift_summary(
                test, layer_index, "ffn_"
            )
        for rank in (value for value in RANKS if value <= hidden):
            candidates = {
                "stable": stable_full[:, :rank],
                "pca_top": pca_top_full[:, :rank],
                "pca_bottom": pca_bottom_full[:, :rank],
            }
            row = {}
            for name, basis in candidates.items():
                row[name] = {
                    "validation": basis_metrics(valid_m, basis),
                    "test": basis_metrics(test_m, basis),
                }
            row["random"] = {
                "validation": random_summary(valid_m, hidden, rank, repeats, seed + layer * 1000 + rank),
                "test": random_summary(test_m, hidden, rank, repeats, seed + layer * 2000 + rank),
            }
            layer_result["ranks"][str(rank)] = row
        reproducibility_rank = min(64, hidden)
        full_discovery_basis = stable_full[:, :reproducibility_rank]
        for block_index in range(3):
            block_m = layer_moments(blocks[block_index], layer_index)
            block_basis, _ = stable_basis(block_m["sigma_x"], block_m["raw_delta"], reproducibility_rank)
            layer_result["block_reproducibility"][str(block_index)] = subspace_overlap(
                full_discovery_basis, block_basis
            )
        pair_result["layers"][str(layer)] = layer_result
    return pair_result, bases, means


def code_projection(code_path: str, wiki_bases: dict[int, np.ndarray], repeats: int, seed: int) -> dict:
    with np.load(os.path.join(code_path, "delta_reservoir.npz")) as data:
        layers = data["layer_numbers"].astype(int).tolist()
        delta = data["delta"].astype(np.float32)
    result = {}
    for index, layer in enumerate(layers):
        rows = delta[index]
        denominator = np.square(rows).sum(axis=1)
        layer_rows = {}
        available_ranks = [value for value in RANKS if value <= wiki_bases[layer].shape[1]]
        random_by_rank = {rank: [] for rank in available_ranks}
        rng = np.random.default_rng(seed + layer * 1000)
        max_rank = max(available_ranks)
        for _ in range(repeats):
            random_basis = orthonormal(rng.standard_normal((rows.shape[1], max_rank)))
            cumulative = np.cumsum(np.square(rows @ random_basis), axis=1)
            for rank in available_ranks:
                random_ratio = cumulative[:, rank - 1] / np.maximum(denominator, 1e-20)
                random_by_rank[rank].append({
                    "mean_projection_energy_fraction": float(random_ratio.mean()),
                    "fraction_ge_0.10": float((random_ratio >= 0.10).mean()),
                    "fraction_ge_0.25": float((random_ratio >= 0.25).mean()),
                    "fraction_ge_0.50": float((random_ratio >= 0.50).mean()),
                })
        for rank in available_ranks:
            basis = wiki_bases[layer][:, :rank]
            ratio = np.square(rows @ basis).sum(axis=1) / np.maximum(denominator, 1e-20)
            stable = {
                "mean_projection_energy_fraction": float(ratio.mean()),
                "median": float(np.median(ratio)),
                "q90": float(np.quantile(ratio, 0.9)),
                "fraction_ge_0.10": float((ratio >= 0.10).mean()),
                "fraction_ge_0.25": float((ratio >= 0.25).mean()),
                "fraction_ge_0.50": float((ratio >= 0.50).mean()),
                "reservoir_tokens": int(ratio.size),
            }
            random_rows = random_by_rank[rank]
            random_summary_row = {}
            for key in random_rows[0]:
                values = np.asarray([row[key] for row in random_rows])
                random_summary_row[key] = {
                    "mean": float(values.mean()),
                    "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
                }
            layer_rows[str(rank)] = {"stable": stable, "random_control": random_summary_row}
        result[str(layer)] = layer_rows
    return result


def directory_bytes(path: str) -> int:
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-pair", action="append", required=True, type=parse_named_path)
    parser.add_argument("--code-pair", action="append", default=[], type=parse_named_path)
    parser.add_argument("--basis-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    report = {"method": "Sigma_old u = rho (E[delta delta^T] + epsilon I) u", "wiki_pairs": {}}
    all_bases = {}
    all_means = {}
    for label, path in args.wiki_pair:
        pair, bases, means = analyze_pair(label, path, args.random_repeats, args.seed)
        pair["actual_storage_bytes"] = directory_bytes(path)
        report["wiki_pairs"][label] = pair
        all_bases[label] = bases
        all_means[label] = means
    if args.basis_label not in all_bases:
        raise SystemExit(f"--basis-label {args.basis_label!r} is not a --wiki-pair label")
    report["code_update_projection_on_wiki_stable_subspace"] = {}
    for label, path in args.code_pair:
        report["code_update_projection_on_wiki_stable_subspace"][label] = code_projection(
            path, all_bases[args.basis_label], args.random_repeats, args.seed
        )
    hidden = next(iter(all_bases[args.basis_label].values())).shape[0]
    layers = len(all_bases[args.basis_label])
    report["fingerprint_storage_bytes"] = {
        str(rank): {
            "fp16": layers * hidden * (rank + 1) * 2,
            "fp32": layers * hidden * (rank + 1) * 4,
            "includes": "projection basis plus one centering mean per layer",
        }
        for rank in RANKS
    }
    report["fingerprint_storage_bytes"]["full"] = {
        "fp16": layers * hidden * (hidden + 1) * 2,
        "fp32": layers * hidden * (hidden + 1) * 4,
        "includes": "full projection plus one centering mean per layer",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    basis_dir = os.path.splitext(os.path.abspath(args.output))[0] + "_bases"
    os.makedirs(basis_dir, exist_ok=True)
    report["basis_files"] = {}
    for label, bases in all_bases.items():
        basis_path = os.path.join(basis_dir, f"{label}.npz")
        layer_numbers = sorted(bases)
        np.savez(
            basis_path,
            layer_numbers=np.asarray(layer_numbers, dtype=np.int16),
            stable_bases_rank256=np.stack([bases[layer] for layer in layer_numbers]).astype(np.float32),
            old_means=np.stack([all_means[label][layer] for layer in layer_numbers]).astype(np.float32),
            definition=np.asarray("Sigma_old u = rho (E[delta delta^T] + epsilon I) u"),
        )
        report["basis_files"][label] = basis_path
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(args.output)


if __name__ == "__main__":
    main()
