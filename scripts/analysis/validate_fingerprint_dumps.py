#!/usr/bin/env python3
"""Numerically validate aligned hidden/router/expert dump semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


KNOWN_STAGES = {"A", "B", "C_vocabkl", "D", "E", "F", "G", "H", "I"}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = {"domains": {}, "all_checks_pass": True}
    for domain_dir in sorted((root / "dumps").iterdir()):
        if not domain_dir.is_dir():
            continue
        dumps = {
            path.stem: load(path)
            for path in sorted(domain_dir.glob("*.npz"))
            if path.stem in KNOWN_STAGES
        }
        if not dumps:
            continue
        reference = next(iter(dumps.values()))
        domain_result = {"stages": {}, "aligned_stage_pairs": {}}
        for stage, dump in dumps.items():
            layer_rows = []
            sample_ids, sample_counts = np.unique(dump["sample_indices"], return_counts=True)
            for layer_index, layer_number in enumerate(dump["layer_numbers"]):
                hidden = dump["router_input"][layer_index].astype(np.float64)
                weight = dump["router_weights"][layer_index].astype(np.float64)
                stored_logits = dump["router_logits"][layer_index].astype(np.float64)
                reconstructed_logits = hidden @ weight.T
                shifted = stored_logits - stored_logits.max(axis=-1, keepdims=True)
                reconstructed_probs = np.exp(shifted)
                reconstructed_probs /= reconstructed_probs.sum(axis=-1, keepdims=True)
                stored_probs = dump["router_full_probs"][layer_index].astype(np.float64)
                topk_indices = dump["router_topk_indices"][layer_index]
                exact_topk = np.argsort(-stored_probs, axis=-1)[:, : topk_indices.shape[-1]]
                gathered = np.take_along_axis(stored_probs, topk_indices, axis=-1)
                expert_outputs = dump["expert_outputs"][layer_index].astype(np.float64)
                stored_expert_norms = dump["expert_output_norms"][layer_index].astype(np.float64)
                recomposed = np.sum(
                    expert_outputs * dump["router_topk_weights"][layer_index, :, :, None], axis=1
                )
                ffn = dump["ffn_output"][layer_index].astype(np.float64)
                cosine = np.sum(recomposed * ffn, axis=-1) / np.maximum(
                    np.linalg.norm(recomposed, axis=-1) * np.linalg.norm(ffn, axis=-1), 1e-20
                )
                relative_l2 = np.linalg.norm(recomposed - ffn, axis=-1) / np.maximum(
                    np.linalg.norm(ffn, axis=-1), 1e-20
                )
                row = {
                    "layer": int(layer_number),
                    "router_logits_max_abs_error": float(
                        np.max(np.abs(stored_logits - reconstructed_logits))
                    ),
                    "router_probabilities_max_abs_error": float(
                        np.max(np.abs(stored_probs - reconstructed_probs))
                    ),
                    "topk_indices_exact": bool(np.array_equal(topk_indices, exact_topk)),
                    "topk_weights_max_abs_error": float(
                        np.max(np.abs(dump["router_topk_weights"][layer_index] - gathered))
                    ),
                    "expert_norms_max_abs_error": float(
                        np.max(
                            np.abs(
                                stored_expert_norms - np.linalg.norm(expert_outputs, axis=-1)
                            )
                        )
                    ),
                    "expert_recomposition_cosine_mean": float(cosine.mean()),
                    "expert_recomposition_relative_l2_mean": float(relative_l2.mean()),
                    "legacy_hidden_equals_layer_output": bool(
                        np.array_equal(
                            dump["hidden_layers"][layer_index], dump["layer_output"][layer_index]
                        )
                    ),
                }
                row["passes"] = bool(
                    row["router_logits_max_abs_error"] < 5e-5
                    and row["router_probabilities_max_abs_error"] < 5e-6
                    and row["topk_indices_exact"]
                    and row["topk_weights_max_abs_error"] < 5e-6
                    and row["expert_norms_max_abs_error"] < 5e-3
                    and row["expert_recomposition_cosine_mean"] > 0.999
                    and row["legacy_hidden_equals_layer_output"]
                )
                layer_rows.append(row)
            stage_result = {
                "path": str(domain_dir / f"{stage}.npz"),
                "tokens": int(len(dump["token_ids"])),
                "sample_count": int(len(sample_ids)),
                "tokens_per_sample_min": int(sample_counts.min()),
                "tokens_per_sample_max": int(sample_counts.max()),
                "all_numeric_finite": bool(
                    all(
                        np.isfinite(value).all()
                        for key, value in dump.items()
                        if np.issubdtype(value.dtype, np.number) and key != "router_topk_margin"
                    )
                ),
                "layers": layer_rows,
            }
            stage_result["passes"] = bool(
                stage_result["all_numeric_finite"]
                and all(row["passes"] for row in layer_rows)
                and stage_result["tokens_per_sample_min"] == stage_result["tokens_per_sample_max"]
            )
            domain_result["stages"][stage] = stage_result
            result["all_checks_pass"] &= stage_result["passes"]

        for stage, dump in dumps.items():
            exact = {
                key: bool(np.array_equal(reference[key], dump[key]))
                for key in ("layer_numbers", "sample_indices", "positions", "token_ids")
            }
            domain_result["aligned_stage_pairs"][stage] = exact
            result["all_checks_pass"] &= all(exact.values())
        result["domains"][domain_dir.name] = domain_result

    out = root / "inventory" / "dump_semantic_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "all_checks_pass": result["all_checks_pass"]}, indent=2))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
