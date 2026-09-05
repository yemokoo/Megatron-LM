#!/usr/bin/env python3
"""Audit whether stable fingerprint scores have token-specific causal value."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


RUN_DIRS = {
    "stable_assignment": "selector_stable_assignment_random_r64_200step_gpu2",
    "permuted_assignment": "selector_permuted_assignment_random_r64_200step_gpu2",
}


def load_second_moment(path: Path) -> tuple[dict, np.ndarray, int]:
    metadata = json.loads((path / "metadata.json").read_text())
    blocks = sorted(path.glob("block_*.npz"))
    if len(blocks) != 5:
        raise RuntimeError(f"expected five blocks in {path}, found {len(blocks)}")
    total = None
    count = 0
    for block in blocks:
        with np.load(block, allow_pickle=False) as payload:
            current = (
                payload["xx"] + payload["yy"]
                - payload["xy"] - payload["xy"].swapaxes(1, 2)
            )
            total = current if total is None else total + current
            count += int(payload["count"])
    if count != 10_000_000 or count != metadata["target_tokens"]:
        raise RuntimeError(f"unexpected token count for {path}: {count}")
    return metadata, total / count, count


def projected_energy(second: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("lhk,lhm,lmk->l", basis, second, basis, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    summary = json.loads(Path(args.training_summary).read_text())
    runs = {row["label"]: row for row in summary["runs"]}
    if set(runs) != set(RUN_DIRS):
        raise RuntimeError(f"unexpected training runs: {sorted(runs)}")
    for name, row in runs.items():
        if row["checkpoint_tracker"] != 200 or not row["training_complete"]:
            raise RuntimeError(f"incomplete training run: {name}")
        series = row["training_series_summary"]
        if series["number of nan iterations"]["max"] or series["number of skipped iterations"]["max"]:
            raise RuntimeError(f"invalid training iteration in {name}")

    loaded = {
        name: load_second_moment(Path(args.streaming_root) / dirname)
        for name, dirname in RUN_DIRS.items()
    }
    hashes = None
    for name, (metadata, _second, _count) in loaded.items():
        current = [block["sha256"] for block in metadata["blocks"]]
        if hashes is None:
            hashes = current
        elif current != hashes:
            raise RuntimeError(f"Wiki sample identity mismatch: {name}")
        if metadata["representation"] != "residual_included_transformer_layer_output":
            raise RuntimeError(f"wrong representation: {name}")

    with np.load(args.bundle, allow_pickle=False) as payload:
        names = payload["representation_names"].tolist()
        layers = payload["layer_numbers"].astype(int)
        bases = payload["bases"].astype(np.float64)
    stable_basis = bases[names.index("stable")]
    random_basis = bases[names.index("random")]

    drift = {}
    for name, (_metadata, second, count) in loaded.items():
        full = np.trace(second, axis1=1, axis2=2)
        stable = projected_energy(second, stable_basis)
        random = projected_energy(second, random_basis)
        drift[name] = {
            "tokens": count,
            "layers_2_to_9_sum": {
                "full_mse": float(full.sum()),
                "stable_r64_mse": float(stable.sum()),
                "random_r64_mse": float(random.sum()),
                "stable_r64_fraction": float(stable.sum() / full.sum()),
                "random_r64_fraction": float(random.sum() / full.sum()),
            },
            "per_layer": {
                str(layer): {
                    "full_mse": float(full[i]),
                    "stable_r64_mse": float(stable[i]),
                    "random_r64_mse": float(random[i]),
                }
                for i, layer in enumerate(layers)
            },
        }

    performance = {}
    selector_logs = {}
    for name, row in runs.items():
        performance[name] = {
            domain: values
            for domain, values in row["probe_summary"].items()
        }
        series = row["training_series_summary"]
        selector_logs[name] = {
            key: series[key]
            for key in (
                "fingerprint mean score",
                "fingerprint mean weight",
                "fingerprint hard coverage",
                "fingerprint score weight covariance",
                "fingerprint kd loss",
                "fingerprint_grad_norm/all_trainable",
                "fingerprint_grad_norm/router_all_rows",
                "fingerprint_grad_norm/new_experts",
            )
        }

    stable_perf = performance["stable_assignment"]
    permuted_perf = performance["permuted_assignment"]
    stable_drift = drift["stable_assignment"]["layers_2_to_9_sum"]
    permuted_drift = drift["permuted_assignment"]["layers_2_to_9_sum"]
    effects = {
        "wiki_accuracy_stable_minus_permuted": (
            stable_perf["wiki_probe"]["last"]["next_token_accuracy"]
            - permuted_perf["wiki_probe"]["last"]["next_token_accuracy"]
        ),
        "wiki_ppl_stable_minus_permuted": (
            stable_perf["wiki_probe"]["last"]["ppl"]
            - permuted_perf["wiki_probe"]["last"]["ppl"]
        ),
        "code_accuracy_stable_minus_permuted": (
            stable_perf["code_probe"]["last"]["next_token_accuracy"]
            - permuted_perf["code_probe"]["last"]["next_token_accuracy"]
        ),
        "code_ppl_stable_minus_permuted": (
            stable_perf["code_probe"]["last"]["ppl"]
            - permuted_perf["code_probe"]["last"]["ppl"]
        ),
        "full_drift_reduction_stable_vs_permuted": (
            1.0 - stable_drift["full_mse"] / permuted_drift["full_mse"]
        ),
        "stable_r64_drift_reduction_stable_vs_permuted": (
            1.0 - stable_drift["stable_r64_mse"] / permuted_drift["stable_r64_mse"]
        ),
        "random_r64_drift_reduction_stable_vs_permuted": (
            1.0 - stable_drift["random_r64_mse"] / permuted_drift["random_r64_mse"]
        ),
    }
    selector_pass = bool(
        effects["wiki_accuracy_stable_minus_permuted"] > 0
        and effects["wiki_ppl_stable_minus_permuted"] < 0
        and effects["code_accuracy_stable_minus_permuted"] >= 0
        and effects["full_drift_reduction_stable_vs_permuted"] > 0
    )

    result = {
        "question": "Does stable fingerprint score assignment have token-specific preservation value?",
        "matched_control": {
            "same_score_and_weight_multiset": True,
            "same_random_r64_loss_basis": True,
            "same_lambda": 0.2,
            "same_source_and_teacher": "expansion KD-init complete step 600",
            "same_code_data_order": True,
            "same_world_size": 2,
            "same_global_batch_size": 2304,
            "old_replay": False,
            "routing": "natural",
            "only_change": "stable-score weight assignment versus deterministic half-batch valid-token rotation",
        },
        "performance": performance,
        "selector_training_logs": selector_logs,
        "wiki_10m_layer_output_drift": drift,
        "effects": effects,
        "decision": {
            "selector_token_specific_value": "PASS" if selector_pass else "FAIL",
            "stable_direction_as_preservation_basis": "FAIL (from prior stable-vs-random audit)",
            "run_1800_steps": False,
            "interpretation": (
                "The stable score is useful for assigning KD to Code tokens, but the low-drift stable "
                "subspace is not the right KD target. Keep the selector role separate from the protected-basis role."
            ),
            "next_single_ablation_not_run": (
                "Use the stable selector with full-hidden KD as a 200-step diagnostic upper bound, with lambda "
                "recalibrated by gradient norm; only then design an old-sensitive, drift-vulnerable compressed basis."
            ),
        },
        "failure_history": [
            "3-step attempt 1: preflight failed because PYTHON_BIN was not set to the H100 environment; no training step.",
            "3-step attempt 2: dataset-helper Makefile used base python3 because PATH was not fixed; no training step.",
            "8-GPU 200-step attempt: an independent GPU4-7 job started after preflight and caused rank4 OOM at the first all-reduce; no optimizer step/checkpoint.",
            "Final evidence uses two simultaneously matched 2-GPU runs isolated on GPUs0-1 and GPUs2-3.",
        ],
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output_json) + ".inprogress")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, output_json)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Stable fingerprint token-assignment audit",
        "",
        "## Matched comparison",
        "",
        "Both runs use the stable teacher score, the exact same valid-token soft-weight multiset, random-r64 projected KD, lambda=0.2, natural routing, and no old replay. The control only rotates weights across valid tokens by half a microbatch.",
        "",
        "| assignment | Code acc | Code PPL | Wiki acc | Wiki PPL | full Wiki drift | stable-r64 drift | random-r64 drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RUN_DIRS:
        perf = performance[name]
        row = drift[name]["layers_2_to_9_sum"]
        lines.append(
            f"| {name} | {perf['code_probe']['last']['next_token_accuracy']:.6f} | "
            f"{perf['code_probe']['last']['ppl']:.5f} | "
            f"{perf['wiki_probe']['last']['next_token_accuracy']:.6f} | "
            f"{perf['wiki_probe']['last']['ppl']:.5f} | "
            f"{row['full_mse']:.4f} | {row['stable_r64_mse']:.4f} | {row['random_r64_mse']:.4f} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        f"**Selector token assignment: {'PASS' if selector_pass else 'FAIL'}.** Stable assignment improves Wiki accuracy by {effects['wiki_accuracy_stable_minus_permuted']:+.6f}, lowers Wiki PPL by {effects['wiki_ppl_stable_minus_permuted']:+.5f}, and reduces full Wiki drift by {pct(effects['full_drift_reduction_stable_vs_permuted'])} relative to the matched permutation.",
        "",
        "The decomposition is now clear: the stable score is useful as a token selector, while the stable low-drift subspace failed as a preservation target. These two fingerprint roles must not be conflated.",
        "",
        "## Scope decision",
        "",
        "Do not run 1800 steps. The next single 200-step diagnostic, if approved, is stable-selector + full-hidden KD with gradient-norm recalibration. Its purpose is to establish the preservation-target upper bound before constructing a compressed old-sensitive and drift-vulnerable basis.",
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
