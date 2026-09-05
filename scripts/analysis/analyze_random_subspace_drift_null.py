#!/usr/bin/env python3
"""Place the fixed fingerprint random-r64 basis in a Haar-random drift null."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_delta_second(path: Path) -> tuple[dict, np.ndarray]:
    metadata = json.loads((path / "metadata.json").read_text())
    blocks = sorted(path.glob("block_*.npz"))
    if len(blocks) != 5:
        raise RuntimeError(f"expected five 2M-token blocks, found {len(blocks)} in {path}")
    count = 0
    total = None
    for block in blocks:
        with np.load(block, allow_pickle=False) as payload:
            current = (
                payload["xx"] + payload["yy"]
                - payload["xy"] - payload["xy"].swapaxes(1, 2)
            )
            total = current if total is None else total + current
            count += int(payload["count"])
    if count != 10_000_000 or count != metadata["target_tokens"]:
        raise RuntimeError(f"unexpected token count: {count}")
    return metadata, total / count


def energy(second: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("lhk,lhm,lmk->l", basis, second, basis, optimize=True)


def summarize_null(samples: np.ndarray, observed: float) -> dict:
    mean = float(samples.mean())
    std = float(samples.std(ddof=1))
    return {
        "observed": float(observed),
        "null_mean": mean,
        "null_std": std,
        "z_score": float((observed - mean) / std),
        "empirical_percentile": float((np.count_nonzero(samples <= observed) + 0.5) / (len(samples) + 1)),
        "null_q025": float(np.quantile(samples, 0.025)),
        "null_q50": float(np.quantile(samples, 0.5)),
        "null_q975": float(np.quantile(samples, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming-dir", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--draws", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    if args.draws < 100:
        raise ValueError("use at least 100 random draws")

    metadata, second = load_delta_second(Path(args.streaming_dir))
    if metadata["representation"] != "residual_included_transformer_layer_output":
        raise RuntimeError(f"wrong representation: {metadata['representation']}")
    if metadata["layers"] != list(range(2, 10)):
        raise RuntimeError(f"wrong layer set: {metadata['layers']}")

    with np.load(args.bundle, allow_pickle=False) as payload:
        names = payload["representation_names"].tolist()
        layers = payload["layer_numbers"].astype(int)
        bases = payload["bases"].astype(np.float64)
    stable_basis = bases[names.index("stable")]
    fixed_random_basis = bases[names.index("random")]
    if stable_basis.shape != (8, 1024, 64):
        raise RuntimeError(f"unexpected basis shape: {stable_basis.shape}")

    full = np.trace(second, axis1=1, axis2=2)
    stable = energy(second, stable_basis)
    fixed_random = energy(second, fixed_random_basis)
    eigenvalues = np.linalg.eigvalsh(second)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    rng = np.random.default_rng(args.seed)
    null = np.empty((args.draws, len(layers)), dtype=np.float64)
    for draw in range(args.draws):
        for layer_index in range(len(layers)):
            gaussian = rng.standard_normal((1024, 64))
            q, _ = np.linalg.qr(gaussian, mode="reduced")
            row_leverage = np.square(q).sum(axis=1)
            null[draw, layer_index] = np.dot(eigenvalues[layer_index], row_leverage)

    per_layer = {}
    for index, layer in enumerate(layers):
        per_layer[str(int(layer))] = {
            "full_mse": float(full[index]),
            "rank_over_hidden_expectation": 64 / 1024,
            "fixed_random_fraction": float(fixed_random[index] / full[index]),
            "stable_fraction": float(stable[index] / full[index]),
            "fixed_random_null": summarize_null(null[:, index], fixed_random[index]),
            "stable_in_random_null": summarize_null(null[:, index], stable[index]),
        }

    null_sum = null.sum(axis=1)
    aggregate = {
        "full_mse": float(full.sum()),
        "fixed_random_mse": float(fixed_random.sum()),
        "stable_mse": float(stable.sum()),
        "fixed_random_fraction": float(fixed_random.sum() / full.sum()),
        "stable_fraction": float(stable.sum() / full.sum()),
        "fixed_random_null": summarize_null(null_sum, fixed_random.sum()),
        "stable_in_random_null": summarize_null(null_sum, stable.sum()),
    }
    lucky_high = aggregate["fixed_random_null"]["empirical_percentile"] >= 0.975
    result = {
        "question": "Was the fixed random-r64 KD basis an unusually high-drift lucky draw before KD training?",
        "source": {
            "checkpoint_pair": "expansion KD-init complete to 200-step Code LM-only",
            "domain": "Wiki",
            "tokens": 10_000_000,
            "representation": metadata["representation"],
            "layers": metadata["layers"],
            "rank": 64,
            "hidden_size": 1024,
        },
        "null": {
            "distribution": "independent Haar-random orthonormal rank-64 basis per layer",
            "draws": args.draws,
            "seed": args.seed,
        },
        "aggregate_layers_2_to_9": aggregate,
        "per_layer": per_layer,
        "decision": {
            "fixed_random_is_97p5_percentile_high_drift_draw": lucky_high,
            "interpretation": (
                "A high percentile would support a lucky high-drift intersection explanation. "
                "A non-extreme percentile means total pre-KD drift exposure alone does not explain "
                "the random-r64 training advantage."
            ),
            "guardrail": (
                "This null tests geometric drift exposure without training. It cannot replace "
                "multi-seed random-r64 KD runs for behavioral robustness."
            ),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = Path(str(output_json) + ".inprogress")
    tmp_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_json, output_json)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Fixed random-r64 Wiki-drift null audit",
        "",
        f"The fixed training basis is compared with {args.draws} independent Haar-random rank-64 bases per layer using the existing 10M-token Wiki LM-only drift covariance.",
        "",
        "| scope | fixed random drift fraction | random percentile | stable drift fraction | stable percentile |",
        "|---|---:|---:|---:|---:|",
        f"| layers 2–9 sum | {pct(aggregate['fixed_random_fraction'])} | {pct(aggregate['fixed_random_null']['empirical_percentile'])} | {pct(aggregate['stable_fraction'])} | {pct(aggregate['stable_in_random_null']['empirical_percentile'])} |",
    ]
    for layer in layers:
        row = per_layer[str(int(layer))]
        lines.append(
            f"| layer {int(layer)} | {pct(row['fixed_random_fraction'])} | "
            f"{pct(row['fixed_random_null']['empirical_percentile'])} | "
            f"{pct(row['stable_fraction'])} | "
            f"{pct(row['stable_in_random_null']['empirical_percentile'])} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"Fixed random-r64 is {'an extreme high-drift draw' if lucky_high else 'not an extreme 97.5th-percentile high-drift draw'} in the aggregate null.",
        "",
        "This result addresses geometric luck only. Behavioral robustness still requires multiple independently trained random-r64 seeds before treating random projection as a method.",
        "",
        f"Machine-readable result: `{output_json}`",
    ])
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = Path(str(output_md) + ".inprogress")
    tmp_md.write_text("\n".join(lines) + "\n")
    os.replace(tmp_md, output_md)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
