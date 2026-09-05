#!/usr/bin/env python3
"""Diagnose stable-vs-random fingerprint KD from matched Wiki streaming moments.

The four training runs intentionally share the *stable* token selector.  Only the
KD loss basis differs between soft-stable and soft-random.  This script keeps
those two roles explicit and measures where the Wiki layer-output drift lands.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


RUN_DIRS = {
    "lm_only": "fingerprint_lm_only_200step",
    "soft_stable": "fingerprint_soft_stable_200step",
    "soft_random": "fingerprint_soft_random_200step",
    "hard_stable": "fingerprint_hard_stable_200step",
}


def load_moments(path: Path) -> tuple[dict, np.ndarray, int]:
    metadata = json.loads((path / "metadata.json").read_text())
    blocks = sorted(path.glob("block_*.npz"))
    if len(blocks) != 5:
        raise RuntimeError(f"expected five 2M-token blocks in {path}, found {len(blocks)}")
    count = 0
    delta_second = None
    for block in blocks:
        with np.load(block, allow_pickle=False) as payload:
            n = int(payload["count"])
            current = payload["xx"] + payload["yy"] - payload["xy"] - payload["xy"].swapaxes(1, 2)
            delta_second = current if delta_second is None else delta_second + current
            count += n
    if count != metadata["target_tokens"]:
        raise RuntimeError(f"token-count mismatch for {path}: {count} != {metadata['target_tokens']}")
    return metadata, delta_second / count, count


def project(second: np.ndarray, basis: np.ndarray) -> np.ndarray:
    # trace(U^T E[delta delta^T] U), one value per layer.
    return np.einsum("lhk,lhm,lmk->l", basis, second, basis, optimize=True)


def reduction(current: float, baseline: float) -> float:
    return 1.0 - current / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(args.streaming_root)
    loaded = {name: load_moments(root / dirname) for name, dirname in RUN_DIRS.items()}
    reference_hashes = [row["sha256"] for row in loaded["lm_only"][0]["blocks"]]
    for name, (metadata, _second, count) in loaded.items():
        hashes = [row["sha256"] for row in metadata["blocks"]]
        if hashes != reference_hashes:
            raise RuntimeError(f"sample identity mismatch: {name}")
        if metadata["representation"] != "residual_included_transformer_layer_output":
            raise RuntimeError(f"wrong representation for {name}: {metadata['representation']}")
        if count != 10_000_000:
            raise RuntimeError(f"expected Wiki 10M tokens for {name}, got {count}")

    with np.load(args.bundle, allow_pickle=False) as payload:
        names = payload["representation_names"].tolist()
        layers = payload["layer_numbers"].astype(int)
        bases = payload["bases"].astype(np.float64)
    stable = bases[names.index("stable")]
    random = bases[names.index("random")]
    overlap = np.linalg.svd(
        np.einsum("lhk,lhm->lkm", stable, random), compute_uv=False
    ) ** 2

    training = json.loads(Path(args.training_summary).read_text())
    performance = {}
    selector_logs = {}
    for run in training["runs"]:
        label = run["label"]
        if label not in RUN_DIRS:
            continue
        performance[label] = {
            domain: {
                "accuracy": values["last"]["next_token_accuracy"],
                "ppl": values["last"]["ppl"],
                "accuracy_delta_from_initial": values["accuracy_delta"],
                "ppl_relative_change_from_initial": values["ppl_relative_change"],
            }
            for domain, values in run["probe_summary"].items()
        }
        series = run.get("training_series_summary", {})
        if "fingerprint mean score" in series:
            selector_logs[label] = {
                "mean_score_over_logged_steps": series["fingerprint mean score"]["mean"],
                "mean_weight_over_logged_steps": series["fingerprint mean weight"]["mean"],
                "hard_coverage_over_logged_steps": series["fingerprint hard coverage"]["mean"],
            }

    drift = {}
    for name, (_metadata, second, count) in loaded.items():
        total = np.trace(second, axis1=1, axis2=2)
        stable_energy = project(second, stable)
        random_energy = project(second, random)
        drift[name] = {
            "tokens": count,
            "per_layer": {
                str(int(layer)): {
                    "full_mse": float(total[i]),
                    "stable_r64_mse": float(stable_energy[i]),
                    "stable_r64_fraction": float(stable_energy[i] / total[i]),
                    "random_r64_mse": float(random_energy[i]),
                    "random_r64_fraction": float(random_energy[i] / total[i]),
                }
                for i, layer in enumerate(layers)
            },
            "layers_2_to_9_sum": {
                "full_mse": float(total.sum()),
                "stable_r64_mse": float(stable_energy.sum()),
                "stable_r64_fraction": float(stable_energy.sum() / total.sum()),
                "random_r64_mse": float(random_energy.sum()),
                "random_r64_fraction": float(random_energy.sum() / total.sum()),
            },
        }

    baseline = drift["lm_only"]["layers_2_to_9_sum"]
    changes = {}
    for name in ("soft_stable", "soft_random", "hard_stable"):
        current = drift[name]["layers_2_to_9_sum"]
        changes[name] = {
            "full_drift_reduction_vs_lm_only": reduction(current["full_mse"], baseline["full_mse"]),
            "stable_r64_drift_reduction_vs_lm_only": reduction(
                current["stable_r64_mse"], baseline["stable_r64_mse"]
            ),
            "random_r64_drift_reduction_vs_lm_only": reduction(
                current["random_r64_mse"], baseline["random_r64_mse"]
            ),
        }

    result = {
        "question": "Why did stable-selector + random-basis KD outperform stable-selector + stable-basis KD?",
        "experimental_identity": {
            "token_selector_all_kd_runs": "stable rank-64 score from frozen-teacher residual layer outputs, layers 2-9",
            "soft_stable_loss_basis": "stable rank-64",
            "soft_random_loss_basis": "fixed random orthonormal rank-64",
            "routing": "natural",
            "old_replay": False,
            "wiki_eval_tokens_per_run": 10_000_000,
            "sample_hashes_identical": True,
        },
        "stable_random_basis_overlap": {
            "mean_squared_canonical_cosine_by_layer": overlap.mean(axis=1).tolist(),
            "random_expectation_rank_over_hidden": 64 / 1024,
        },
        "performance": performance,
        "selector_training_logs": selector_logs,
        "wiki_layer_output_drift": drift,
        "drift_reduction_vs_lm_only": changes,
        "phase4_decision": {
            "status": "FAIL",
            "failed_gate": "stable KD must outperform the dimension-matched random-basis control",
            "stable_minus_random_wiki_accuracy": (
                performance["soft_stable"]["wiki_probe"]["accuracy"]
                - performance["soft_random"]["wiki_probe"]["accuracy"]
            ),
            "stable_minus_random_wiki_ppl": (
                performance["soft_stable"]["wiki_probe"]["ppl"]
                - performance["soft_random"]["wiki_probe"]["ppl"]
            ),
            "do_not_run_1800_steps": True,
        },
        "root_cause": {
            "implementation_did_preserve_requested_basis": (
                changes["soft_stable"]["stable_r64_drift_reduction_vs_lm_only"]
                > changes["soft_random"]["stable_r64_drift_reduction_vs_lm_only"]
            ),
            "stable_basis_was_already_less_exposed_than_random_in_lm_only": (
                drift["lm_only"]["layers_2_to_9_sum"]["stable_r64_fraction"]
                < drift["lm_only"]["layers_2_to_9_sum"]["random_r64_fraction"]
            ),
            "random_basis_reduced_more_full_drift": (
                changes["soft_random"]["full_drift_reduction_vs_lm_only"]
                > changes["soft_stable"]["full_drift_reduction_vs_lm_only"]
            ),
            "interpretation": (
                "The discovery objective selected high-Wiki-variance directions with low observed Code drift. "
                "Those directions exist and are preserved by stable KD, but low drift makes them a low-leverage "
                "forgetting target. Random rank-64 intersects more vulnerable directions and behaves as broader "
                "teacher regularization, improving Wiki slightly more while costing Code slightly more."
            ),
            "selector_status": (
                "UNRESOLVED: Wiki-vs-Code AUROC proves domain separation, not that high-score Code tokens "
                "causally carry old knowledge. Stable and random loss-basis runs used identical stable scores."
            ),
            "next_single_ablation": (
                "Keep the random-r64 KD basis and exact per-batch soft-weight multiset, then compare stable-score "
                "token assignment with a deterministic within-batch permutation. This isolates selector-token "
                "correlation without changing KD scale, basis, coverage, optimizer, or data order."
            ),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output_json) + ".inprogress")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output_json)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# 200-step fingerprint KD root-cause audit",
        "",
        "## Experimental distinction",
        "",
        "All KD runs used the same stable-fingerprint token selector. The random run changed only the projected KD loss basis. It is therefore not a random-selector control.",
        "",
        "| run | Wiki acc | Wiki PPL | Code acc | Code PPL | full drift reduction | stable-r64 drift reduction | random-r64 drift reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RUN_DIRS:
        perf = performance[name]
        if name == "lm_only":
            reductions = (0.0, 0.0, 0.0)
        else:
            row = changes[name]
            reductions = (
                row["full_drift_reduction_vs_lm_only"],
                row["stable_r64_drift_reduction_vs_lm_only"],
                row["random_r64_drift_reduction_vs_lm_only"],
            )
        lines.append(
            f"| {name} | {perf['wiki_probe']['accuracy']:.6f} | {perf['wiki_probe']['ppl']:.5f} | "
            f"{perf['code_probe']['accuracy']:.6f} | {perf['code_probe']['ppl']:.5f} | "
            f"{pct(reductions[0])} | {pct(reductions[1])} | {pct(reductions[2])} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "**FAIL.** Stable KD does not beat the dimension-matched random-basis control, so the 1800-step extension is not justified.",
        "",
        f"In LM-only, stable-r64 contains {pct(drift['lm_only']['layers_2_to_9_sum']['stable_r64_fraction'])} of Wiki drift versus {pct(drift['lm_only']['layers_2_to_9_sum']['random_r64_fraction'])} in random-r64. Stable KD then removes more of the stable drift, proving the implementation acts on its requested target, but random KD removes more full drift and preserves Wiki slightly better.",
        "",
        "The failed inference is `invariant direction -> causally important preservation direction`. The stable discovery objective explicitly favors low-drift directions, whereas forgetting prevention needs old-task-sensitive directions that are also vulnerable to the new-task update.",
        "",
        "## Interpretation guardrail",
        "",
        "This audit can reject the stable subspace as the best preservation target under the current selector. It cannot by itself establish that the stable selector identifies causally old-knowledge-bearing Code tokens; its earlier AUROC establishes domain separation only.",
        "",
        "## Next single ablation (not executed)",
        "",
        "Use the same random-r64 loss basis and exactly permute the existing soft weights within each batch. Stable-score assignment versus permuted assignment isolates whether the old-like selector has token-specific preservation value while matching basis, weight histogram, KD scale, data, and optimizer.",
        "",
        f"Machine-readable result: `{output_json}`",
    ])
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    temporary_md = Path(str(output_md) + ".inprogress")
    temporary_md.write_text("\n".join(lines) + "\n")
    os.replace(temporary_md, output_md)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
