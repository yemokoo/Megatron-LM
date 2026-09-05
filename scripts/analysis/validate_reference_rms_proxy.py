#!/usr/bin/env python3
"""Validate sqrt(delta_mse)/relative_l2 against raw hidden reservoirs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    valid = 0
    total = 0
    reservoirs = sorted(args.root.glob("rank_[0-9][0-9][0-9]/reservoir/hidden_reservoir.npz"))
    if not reservoirs:
        raise SystemExit("no reservoirs found")
    for path in reservoirs:
        with np.load(path, allow_pickle=False) as data:
            layers = data["layer_numbers"]
            if not np.array_equal(layers, np.arange(1, 10)):
                raise RuntimeError(f"unexpected layers in {path}")
            reference = data["reference"][:, 1:, :].astype(np.float32)
            current = data["current"][:, 1:, :].astype(np.float32)
        delta = current - reference
        reference_norm = np.linalg.norm(reference, axis=-1)
        delta_norm = np.linalg.norm(delta, axis=-1)
        relative_l2 = delta_norm / np.maximum(reference_norm, 1e-12)
        delta_mse = np.square(delta).mean(axis=-1)
        proxy = np.sqrt(delta_mse) / np.maximum(relative_l2, 1e-12)
        direct = reference_norm / np.sqrt(reference.shape[-1])
        mask = (relative_l2 > 1e-12) & (delta_mse > 1e-24)
        error = np.abs(proxy[mask] - direct[mask]) / np.maximum(direct[mask], 1e-12)
        errors.append(error)
        valid += int(mask.sum())
        total += int(mask.size)
    all_errors = np.concatenate(errors)
    report = {
        "schema": "reference_rms_proxy_validation_v1",
        "passed": bool(valid == total and np.isfinite(all_errors).all()),
        "reservoirs": len(reservoirs),
        "layers": list(range(2, 10)),
        "token_layer_entries": total,
        "valid_entries": valid,
        "formula": "sqrt(delta_mse)/relative_l2",
        "direct": "l2(reference)/sqrt(hidden_size)",
        "relative_error_median": float(np.median(all_errors)),
        "relative_error_p99": float(np.quantile(all_errors, 0.99)),
        "relative_error_max": float(all_errors.max(initial=0.0)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".inprogress")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
