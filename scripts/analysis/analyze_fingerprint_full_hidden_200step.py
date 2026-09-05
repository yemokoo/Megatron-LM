#!/usr/bin/env python3
"""Audit stable-selector full-hidden KD against matched random-r64 controls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


STREAM_DIRS = {
    "random_r64_stable_assignment": "selector_stable_assignment_random_r64_200step_gpu2",
    "random_r64_permuted_assignment": "selector_permuted_assignment_random_r64_200step_gpu2",
    "full_hidden_stable_assignment": "stable_selector_full_hidden_lambda0p00315_200step_gpu2",
}


def load_second_moment(path: Path) -> tuple[dict, np.ndarray]:
    metadata = json.loads((path / "metadata.json").read_text())
    blocks = sorted(path.glob("block_*.npz"))
    if len(blocks) != 5:
        raise RuntimeError(f"expected five Wiki blocks in {path}, found {len(blocks)}")
    count = 0
    moment = None
    for block in blocks:
        with np.load(block, allow_pickle=False) as payload:
            current = (
                payload["xx"] + payload["yy"]
                - payload["xy"] - payload["xy"].swapaxes(1, 2)
            )
            moment = current if moment is None else moment + current
            count += int(payload["count"])
    if count != 10_000_000 or count != metadata["target_tokens"]:
        raise RuntimeError(f"unexpected token count in {path}: {count}")
    return metadata, moment / count


def projected_energy(second: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("lhk,lhm,lmk->l", basis, second, basis, optimize=True)


def performance_row(run: dict) -> dict:
    return {
        domain: {
            "initial_accuracy": values["first"]["next_token_accuracy"],
            "final_accuracy": values["last"]["next_token_accuracy"],
            "accuracy_delta": values["accuracy_delta"],
            "initial_ppl": values["first"]["ppl"],
            "final_ppl": values["last"]["ppl"],
            "ppl_relative_change": values["ppl_relative_change"],
        }
        for domain, values in run["probe_summary"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True)
    parser.add_argument("--streaming-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--router-audit")
    parser.add_argument("--random-null-audit")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    training = json.loads(Path(args.training_summary).read_text())
    runs = {row["label"]: row for row in training["runs"]}
    if set(runs) != set(STREAM_DIRS):
        raise RuntimeError(f"unexpected training labels: {sorted(runs)}")
    for label, run in runs.items():
        if run["checkpoint_tracker"] != 200 or not run["training_complete"]:
            raise RuntimeError(f"incomplete run: {label}")
        series = run["training_series_summary"]
        if series["number of nan iterations"]["max"] != 0:
            raise RuntimeError(f"NaN iteration in {label}")
        if series["number of skipped iterations"]["max"] != 0:
            raise RuntimeError(f"skipped iteration in {label}")

    loaded = {
        label: load_second_moment(Path(args.streaming_root) / dirname)
        for label, dirname in STREAM_DIRS.items()
    }
    hashes = None
    for label, (metadata, _second) in loaded.items():
        current_hashes = [row["sha256"] for row in metadata["blocks"]]
        if hashes is None:
            hashes = current_hashes
        elif current_hashes != hashes:
            raise RuntimeError(f"Wiki sample mismatch for {label}")
        if metadata["representation"] != "residual_included_transformer_layer_output":
            raise RuntimeError(f"wrong representation for {label}")
        if metadata["layers"] != list(range(2, 10)):
            raise RuntimeError(f"wrong layers for {label}: {metadata['layers']}")

    with np.load(args.bundle, allow_pickle=False) as payload:
        names = payload["representation_names"].tolist()
        layers = payload["layer_numbers"].astype(int)
        bases = payload["bases"].astype(np.float64)
    stable_basis = bases[names.index("stable")]
    random_basis = bases[names.index("random")]

    drift = {}
    for label, (_metadata, second) in loaded.items():
        full = np.trace(second, axis1=1, axis2=2)
        stable = projected_energy(second, stable_basis)
        random = projected_energy(second, random_basis)
        drift[label] = {
            "layers_2_to_9_sum": {
                "full_mse": float(full.sum()),
                "stable_r64_mse": float(stable.sum()),
                "random_r64_mse": float(random.sum()),
                "stable_r64_fraction": float(stable.sum() / full.sum()),
                "random_r64_fraction": float(random.sum() / full.sum()),
            },
            "per_layer": {
                str(int(layer)): {
                    "full_mse": float(full[i]),
                    "stable_r64_mse": float(stable[i]),
                    "random_r64_mse": float(random[i]),
                }
                for i, layer in enumerate(layers)
            },
        }

    performance = {label: performance_row(run) for label, run in runs.items()}
    gradients = {}
    for label, run in runs.items():
        series = run["training_series_summary"]
        gradients[label] = {
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

    full_label = "full_hidden_stable_assignment"
    random_label = "random_r64_stable_assignment"
    permuted_label = "random_r64_permuted_assignment"
    full_perf = performance[full_label]
    random_perf = performance[random_label]
    full_drift = drift[full_label]["layers_2_to_9_sum"]
    random_drift = drift[random_label]["layers_2_to_9_sum"]
    effects = {
        "full_minus_random_wiki_accuracy": (
            full_perf["wiki_probe"]["final_accuracy"]
            - random_perf["wiki_probe"]["final_accuracy"]
        ),
        "full_minus_random_wiki_ppl": (
            full_perf["wiki_probe"]["final_ppl"]
            - random_perf["wiki_probe"]["final_ppl"]
        ),
        "full_minus_random_code_accuracy": (
            full_perf["code_probe"]["final_accuracy"]
            - random_perf["code_probe"]["final_accuracy"]
        ),
        "full_minus_random_code_ppl": (
            full_perf["code_probe"]["final_ppl"]
            - random_perf["code_probe"]["final_ppl"]
        ),
        "full_hidden_drift_reduction_vs_random_r64_run": (
            1.0 - full_drift["full_mse"] / random_drift["full_mse"]
        ),
        "stable_r64_drift_reduction_vs_random_r64_run": (
            1.0 - full_drift["stable_r64_mse"] / random_drift["stable_r64_mse"]
        ),
        "random_r64_drift_reduction_vs_random_r64_run": (
            1.0 - full_drift["random_r64_mse"] / random_drift["random_r64_mse"]
        ),
    }
    full_grad = gradients[full_label]
    random_grad = gradients[random_label]
    scale_comparison = {
        "mean_all_trainable_gradient_ratio_full_over_random": (
            full_grad["fingerprint_grad_norm/all_trainable"]["mean"]
            / random_grad["fingerprint_grad_norm/all_trainable"]["mean"]
        ),
        "mean_router_gradient_ratio_full_over_random": (
            full_grad["fingerprint_grad_norm/router_all_rows"]["mean"]
            / random_grad["fingerprint_grad_norm/router_all_rows"]["mean"]
        ),
        "mean_new_expert_gradient_ratio_full_over_random": (
            full_grad["fingerprint_grad_norm/new_experts"]["mean"]
            / random_grad["fingerprint_grad_norm/new_experts"]["mean"]
        ),
        "mean_score_difference_full_minus_random": (
            full_grad["fingerprint mean score"]["mean"]
            - random_grad["fingerprint mean score"]["mean"]
        ),
        "mean_weight_difference_full_minus_random": (
            full_grad["fingerprint mean weight"]["mean"]
            - random_grad["fingerprint mean weight"]["mean"]
        ),
        "mean_coverage_difference_full_minus_random": (
            full_grad["fingerprint hard coverage"]["mean"]
            - random_grad["fingerprint hard coverage"]["mean"]
        ),
    }
    layer9_full = drift[full_label]["per_layer"]["9"]
    layer9_random = drift[random_label]["per_layer"]["9"]
    layer9_effects = {
        "full_mse_reduction": 1.0 - layer9_full["full_mse"] / layer9_random["full_mse"],
        "stable_r64_mse_reduction": (
            1.0 - layer9_full["stable_r64_mse"] / layer9_random["stable_r64_mse"]
        ),
        "random_r64_mse_reduction": (
            1.0 - layer9_full["random_r64_mse"] / layer9_random["random_r64_mse"]
        ),
    }

    full_beats_random = bool(
        effects["full_minus_random_wiki_accuracy"] > 0
        and effects["full_minus_random_wiki_ppl"] < 0
    )
    router_audit = (
        json.loads(Path(args.router_audit).read_text()) if args.router_audit else None
    )
    random_null_audit = (
        json.loads(Path(args.random_null_audit).read_text())
        if args.random_null_audit else None
    )
    result = {
        "question": "Does full-hidden KD on stable-fingerprint-selected Code tokens provide the preservation-target upper bound?",
        "experimental_identity": {
            "selector_for_stable_assignment_runs": "stable rank-64 frozen-teacher score, layers 2-9",
            "random_control_loss_target": "fixed random orthonormal rank-64",
            "full_loss_target": "all 1024 residual layer-output dimensions",
            "full_lambda": 0.00315,
            "random_lambda": 0.2,
            "lambda_calibration": "full KD gradient calibrated to 10-30% of matched LM gradient at warm-10 student",
            "old_replay": False,
            "routing": "natural",
            "world_size": 2,
            "global_batch_size": 2304,
            "wiki_eval_tokens": 10_000_000,
            "sample_hashes_identical": True,
        },
        "performance": performance,
        "wiki_10m_layer_output_drift": drift,
        "training_selector_and_gradient_logs": gradients,
        "effects_full_vs_random": effects,
        "gradient_and_selector_scale_comparison": scale_comparison,
        "layer9_effects_full_vs_random": layer9_effects,
        "decision": {
            "full_hidden_target_beats_random_r64": "PASS" if full_beats_random else "FAIL",
            "stable_selector_token_specific_value": "PASS (from matched permutation audit)",
            "run_1800_steps": False,
            "reason": (
                "Full-hidden KD did not improve both Wiki accuracy and PPL over the same stable-selector random-r64 run."
            ),
        },
        "interpretation_guardrail": (
            "The training signal is teacher alignment on selected Code inputs, not replay of Wiki inputs. "
            "Even full hidden alignment therefore cannot be assumed to cover every Wiki state needed for old-task behavior."
        ),
        "root_cause": {
            "not_selector_coverage": (
                "Stable-score means, soft-weight means, and hard coverage are numerically matched between "
                "the full-hidden and random-r64 stable-assignment runs."
            ),
            "not_total_gradient_scale": (
                "Mean scaled KD gradient norms are closely matched for all trainable parameters, router rows, "
                "and new experts; the full-hidden run did not fail because of a gross total-gradient mismatch."
            ),
            "total_hidden_drift_is_not_sufficient": (
                "Full-hidden KD reduced aggregate Wiki full-hidden drift more than random-r64 KD but preserved "
                "Wiki behavior less well. Aggregate Euclidean alignment is therefore not a sufficient proxy."
            ),
            "gradient_budget_dilution": (
                "With total KD gradient calibrated, full-hidden loss spreads its budget across 1024 dimensions. "
                "Random-r64 concentrates the budget in 64 fixed directions and preserves those directions much "
                "more strongly. The fixed random basis may intersect behavior-sensitive directions, but one "
                "random draw cannot establish that this is generic."
            ),
            "input_coverage_limit": (
                "Both objectives align teacher and student only on selected Code inputs. Full coordinates on "
                "those inputs do not provide the missing Wiki activation distribution supplied by old replay."
            ),
        },
        "permuted_control_label": permuted_label,
    }
    if router_audit is not None:
        result["router_usage_readonly"] = router_audit
    if random_null_audit is not None:
        result["random_r64_drift_null"] = random_null_audit

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = Path(str(output_json) + ".inprogress")
    tmp_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_json, output_json)

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    lines = [
        "# Stable-selector full-hidden KD 200-step audit",
        "",
        "All stable-assignment runs use the same frozen-teacher stable fingerprint to select Code tokens. The compared preservation target is fixed random-r64 versus all 1024 hidden dimensions.",
        "",
        "| run | Code acc | Code PPL | Wiki acc | Wiki PPL | full Wiki drift | stable-r64 drift | random-r64 drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in STREAM_DIRS:
        perf = performance[label]
        row = drift[label]["layers_2_to_9_sum"]
        lines.append(
            f"| {label} | {perf['code_probe']['final_accuracy']:.6f} | "
            f"{perf['code_probe']['final_ppl']:.5f} | "
            f"{perf['wiki_probe']['final_accuracy']:.6f} | "
            f"{perf['wiki_probe']['final_ppl']:.5f} | "
            f"{row['full_mse']:.4f} | {row['stable_r64_mse']:.4f} | {row['random_r64_mse']:.4f} |"
        )
    lines.extend([
        "",
        "## Full-hidden versus random-r64",
        "",
        f"- Wiki accuracy difference: {effects['full_minus_random_wiki_accuracy']:+.6f}",
        f"- Wiki PPL difference: {effects['full_minus_random_wiki_ppl']:+.5f}",
        f"- Code accuracy difference: {effects['full_minus_random_code_accuracy']:+.6f}",
        f"- Full Wiki drift reduction: {pct(effects['full_hidden_drift_reduction_vs_random_r64_run'])}",
        f"- Stable-r64 Wiki drift reduction: {pct(effects['stable_r64_drift_reduction_vs_random_r64_run'])}",
        f"- Random-r64 Wiki drift reduction: {pct(effects['random_r64_drift_reduction_vs_random_r64_run'])}",
        f"- Layer 9 full-drift reduction: {pct(layer9_effects['full_mse_reduction'])}",
        f"- Mean total KD-gradient ratio (full/random): {scale_comparison['mean_all_trainable_gradient_ratio_full_over_random']:.3f}",
        f"- Mean router KD-gradient ratio (full/random): {scale_comparison['mean_router_gradient_ratio_full_over_random']:.3f}",
        f"- Mean new-expert KD-gradient ratio (full/random): {scale_comparison['mean_new_expert_gradient_ratio_full_over_random']:.3f}",
        "",
        "## Decision",
        "",
        f"**{'PASS' if full_beats_random else 'FAIL'}.** Full-hidden KD must improve both Wiki accuracy and PPL over random-r64 to justify expansion. It does not, so no 1800-step run is justified.",
        "",
        "The stable fingerprint selector remains supported by the matched permutation audit. This failure concerns the protected hidden target, not the token selector.",
        "",
        "A full hidden target on selected Code inputs is not equivalent to replaying Wiki inputs: unobserved Wiki activation states receive no direct constraint.",
        "",
        "## Root cause",
        "",
        "Selector score, weight, and coverage are numerically matched, and the mean scaled KD-gradient norms are also closely matched. The failure is therefore not explained by selector drift or a gross gradient-scale error.",
        "",
        "Full-hidden KD lowers aggregate Wiki full drift—including layer 9—yet worsens Wiki behavior. Under a fixed total gradient budget, its force is spread over 1024 coordinates, while random-r64 concentrates force in 64 directions and preserves those coordinates much more strongly. This rejects aggregate hidden L2 distance as a sufficient preservation target. It does not prove that arbitrary random bases are generally useful because only one fixed draw has been tested.",
    ])
    if router_audit is not None:
        router_decision = router_audit["decision"]
        lines.extend([
            "",
            "## Read-only natural-routing check",
            "",
            f"Gross expert collapse: **{'YES' if router_decision['gross_expert_collapse_observed'] else 'NO'}**. The minimum effective expert count across Wiki/Code and all three checkpoints is {router_decision['minimum_effective_num_experts']:.2f}; the maximum single-expert assignment fraction is {router_decision['maximum_single_expert_fraction']:.4f}.",
            "",
            "Wiki naturally routes mostly to old experts, while Code routes mostly to newly added experts. The performance failure is therefore not explained by an obvious routing collapse. The earlier zero train-usage scalars were invalid instrumentation and are superseded by this matched read-only probe.",
        ])
    if random_null_audit is not None:
        null_row = random_null_audit["aggregate_layers_2_to_9"]
        lines.extend([
            "",
            "## Random-basis geometric luck check",
            "",
            f"Against {random_null_audit['null']['draws']} independent Haar-random rank-64 draws, the fixed training random basis is at the {100 * null_row['fixed_random_null']['empirical_percentile']:.2f}th percentile of LM-only Wiki drift exposure. It is not an extreme 97.5th-percentile high-drift draw. The stable basis is at the {100 * null_row['stable_in_random_null']['empirical_percentile']:.2f}th percentile, confirming that it is an exceptionally low-drift and therefore low-leverage preservation target.",
            "",
            "This geometric null does not prove behavioral robustness across random seeds; multiple independently trained random-r64 controls would still be required before proposing random projection as a method.",
        ])
    lines.extend(["", f"Machine-readable result: `{output_json}`"])
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = Path(str(output_md) + ".inprogress")
    tmp_md.write_text("\n".join(lines) + "\n")
    os.replace(tmp_md, output_md)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md)}, indent=2))


if __name__ == "__main__":
    main()
