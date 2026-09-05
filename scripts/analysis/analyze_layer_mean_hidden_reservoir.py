#!/usr/bin/env python3
"""Analyze cosine after averaging raw layer-output vectors in hidden reservoirs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(x * y, axis=-1, dtype=np.float32)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--exclude-layers", default="1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = {int(value) for value in args.exclude_layers.split(",") if value.strip()}
    files = sorted(args.root.glob("rank_*/reservoir/hidden_reservoir.npz"))
    if not files:
        raise RuntimeError(f"no hidden reservoirs below {args.root}")

    identity: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "positions": [],
        "token_ids": [],
        "worker_index": [],
    }
    references: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    selected_layers: np.ndarray | None = None
    for worker_index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            layers = data["layer_numbers"]
            selection = np.asarray([index for index, layer in enumerate(layers) if int(layer) not in excluded])
            if selection.size == 0:
                raise RuntimeError("layer exclusion removed every layer")
            current_selected = layers[selection]
            if selected_layers is None:
                selected_layers = current_selected.copy()
            elif not np.array_equal(selected_layers, current_selected):
                raise RuntimeError(f"reservoir layer mismatch in {path}")
            references.append(data["reference"][:, selection].astype(np.float32))
            currents.append(data["current"][:, selection].astype(np.float32))
            count = int(data["sample_ids"].size)
            identity["sample_ids"].append(data["sample_ids"].copy())
            identity["positions"].append(data["positions"].copy())
            identity["token_ids"].append(data["token_ids"].copy())
            identity["worker_index"].append(np.full(count, worker_index, dtype=np.uint8))

    assert selected_layers is not None
    reference = np.concatenate(references, axis=0)
    current = np.concatenate(currents, axis=0)
    reference_mean = reference.mean(axis=1, dtype=np.float32)
    current_mean = current.mean(axis=1, dtype=np.float32)
    score = _cosine(reference_mean, current_mean).astype(np.float32)
    if not np.isfinite(score).all() or float(score.min()) < -1.00001 or float(score.max()) > 1.00001:
        raise RuntimeError("invalid layer-mean hidden cosine")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "layer_mean_hidden_cosine_reservoir.npz"
    temporary = score_path.with_name(score_path.name + ".inprogress")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            selected_layers=selected_layers.astype(np.int16),
            score=score,
            **{name: np.concatenate(chunks) for name, chunks in identity.items()},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, score_path)

    percentile_rows = []
    for top_percent in range(1, 21):
        cutoff = float(np.quantile(score, 1.0 - top_percent / 100.0))
        percentile_rows.append({"top_percent": top_percent, "cosine_cutoff": cutoff})
    csv_path = output_dir / "top_1_to_20_percent_cutoffs.csv"
    temporary_csv = csv_path.with_name(csv_path.name + ".inprogress")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("top_percent", "cosine_cutoff"))
        writer.writeheader()
        for row in percentile_rows:
            writer.writerow({"top_percent": row["top_percent"], "cosine_cutoff": f"{row['cosine_cutoff']:.7f}"})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_csv, csv_path)

    summary = {
        "schema": "layer_mean_hidden_reservoir_analysis_v1",
        "source": str(args.root.resolve()),
        "source_dtype": "float16 raw-hidden reservoir; computation float32",
        "population_or_sample": "deterministic stratified reservoir sample, not the full token census",
        "reservoir_tokens": int(score.size),
        "selected_layers": [int(value) for value in selected_layers],
        "excluded_layers": sorted(excluded),
        "definition": "cos(mean_layer(reference_hidden), mean_layer(current_hidden))",
        "mean": float(score.mean()),
        "median": float(np.median(score)),
        "min": float(score.min()),
        "max": float(score.max()),
        "p05": float(np.quantile(score, 0.05)),
        "p95": float(np.quantile(score, 0.95)),
        "fraction_ge": {
            str(threshold): float((score >= threshold).mean()) for threshold in (0.9, 0.95, 0.99, 0.999)
        },
        "top_percent_cosine_cutoffs": percentile_rows,
        "artifacts": {"scores_npz": str(score_path), "cutoffs_csv": str(csv_path)},
        "gt_assigned": False,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
