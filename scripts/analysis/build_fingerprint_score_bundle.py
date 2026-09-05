#!/usr/bin/env python3
"""Build stable/PCA/random rank-prefix bases for Phase-1 token scoring."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import linalg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-seed", type=int, default=20260810)
    args = parser.parse_args()

    root = Path(args.stability_root).resolve()
    stable_path = root / (
        "analysis/stable_subspace_results_bases/"
        "code_only_no_replay_no_router_ft.npz"
    )
    stats_dir = root / (
        "streaming_stats/wiki/"
        "code_only_no_replay_no_router_ft_after_expansion_kd_init"
    )
    with np.load(stable_path, allow_pickle=False) as stable_payload:
        layers = stable_payload["layer_numbers"].astype(np.int16)
        stable = stable_payload["stable_bases_rank256"][:, :, :64].astype(np.float64)
        means = stable_payload["old_means"].astype(np.float64)

    blocks = []
    for index in range(3):
        with np.load(stats_dir / f"block_{index:03d}.npz", allow_pickle=False) as payload:
            blocks.append({key: payload[key] for key in ("count", "sum_x", "xx")})
    count = sum(int(block["count"]) for block in blocks)
    sum_x = sum(block["sum_x"] for block in blocks)
    xx = sum(block["xx"] for block in blocks)
    computed_means = sum_x / count
    mean_error = float(np.max(np.abs(computed_means - means)))
    if mean_error > 2e-5:
        raise RuntimeError(f"stored mean does not match 6M discovery moments: {mean_error}")

    pca = []
    random = []
    rng = np.random.default_rng(args.random_seed)
    for layer_index in range(len(layers)):
        covariance = xx[layer_index] / count - np.outer(means[layer_index], means[layer_index])
        covariance = (covariance + covariance.T) / 2
        hidden = covariance.shape[0]
        _values, vectors = linalg.eigh(
            covariance,
            subset_by_index=(hidden - 64, hidden - 1),
            driver="evr",
            check_finite=False,
        )
        pca.append(vectors[:, ::-1])
        random.append(np.linalg.qr(rng.standard_normal((hidden, 64)), mode="reduced")[0])

    bases = np.stack((stable, np.stack(pca), np.stack(random))).astype(np.float32)
    max_gram_error = float(
        max(
            np.max(np.abs(basis.T @ basis - np.eye(64)))
            for representation in bases for basis in representation
        )
    )
    if max_gram_error > 1e-4 or not np.isfinite(bases).all():
        raise RuntimeError(f"invalid score bundle: gram_error={max_gram_error}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + ".inprogress.npz")
    metadata = {
        "definition": "centered projection-energy ratio",
        "discovery_tokens": count,
        "representations": ["stable", "pca_top", "random"],
        "ranks": [16, 32, 64],
        "random_seed": args.random_seed,
        "source_stable_basis": str(stable_path),
        "source_wiki_stats": str(stats_dir),
        "mean_max_abs_error": mean_error,
        "max_orthonormal_gram_error": max_gram_error,
    }
    np.savez(
        temporary,
        representation_names=np.asarray(["stable", "pca_top", "random"]),
        layer_numbers=layers,
        ranks=np.asarray([16, 32, 64], dtype=np.int16),
        means=means.astype(np.float32),
        bases=bases,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
