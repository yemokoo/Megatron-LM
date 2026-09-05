#!/usr/bin/env python3
"""Requirement-by-requirement status audit for the sequential fingerprint-KD goal."""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def tracker(path: Path, expected: int) -> bool:
    target = path / "latest_checkpointed_iteration.txt"
    return target.is_file() and int(target.read_text().strip()) == expected


def training_log_valid(path: Path) -> bool:
    log = path / "logs" / "a_to_b_freeze.log"
    if not log.is_file():
        return False
    text = log.read_text(errors="replace")
    return (
        "training complete" in text
        and "number of skipped iterations:   0" in text
        and "number of nan iterations:   0" in text
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repo = Path(args.repo).resolve()

    phase0_path = root / "audits/phase0.json"
    phase1_path = root / "analysis/phase1_token_score.json"
    junit_path = root / "audits/fingerprint_kd_unit_tests.xml"
    phase4_path = root / "analysis/phase4_200step_training_summary.json"
    phase4_cause_path = root / "analysis/phase4_200step_root_cause.json"
    selector_path = root / "analysis/selector_assignment_200step_audit.json"
    full_path = root / "analysis/full_hidden_200step_audit.json"
    router_path = root / "analysis/router_usage_readonly_200step_audit.json"
    random_null_path = root / "analysis/random_r64_drift_null_1024.json"

    phase0 = read_json(phase0_path)
    phase1 = read_json(phase1_path)
    phase4 = read_json(phase4_path)
    phase4_cause = read_json(phase4_cause_path)
    selector = read_json(selector_path)
    full = read_json(full_path)
    router = read_json(router_path)
    random_null = read_json(random_null_path)

    junit = ET.parse(junit_path).getroot()
    suite = junit if junit.tag == "testsuite" else junit.find("testsuite")
    tests = int(suite.attrib["tests"])
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))

    gradient_names = (
        "fingerprint-grad-kdonly-stableselector-randomr64-lambda1-3batch-gpu2-20260810",
        "fingerprint-grad-kdonly-stableselector-full-lambda1-3batch-gpu2-20260810",
        "fingerprint-grad-lmonly-matched-gbs2304-3batch-gpu2-20260810",
        "fingerprint-grad-additive-stableselector-full-lambda0p00315-3batch-gpu2-20260810",
    )
    gradient_dirs = [root / "gradient_scale/final" / name for name in gradient_names]
    phase4_runs = {row["label"]: row for row in phase4["runs"]}
    phase4_runs_valid = all(
        row["checkpoint_tracker"] == 200
        and row["training_complete"]
        and row["training_series_summary"]["number of nan iterations"]["max"] == 0
        and row["training_series_summary"]["number of skipped iterations"]["max"] == 0
        for row in phase4_runs.values()
    )

    full_run = root / (
        "full_hidden/200step/final/"
        "fingerprint-stable-selector-full-hidden-lambda0p00315-200step-gpu2-20260810"
    )
    requirements = [
        {
            "phase": 0,
            "status": "PASS" if phase0.get("passed") else "FAIL",
            "requirement": "Residual-included layer_output, layers 2-9, checkpoint lineage, basis shapes and natural routing audited",
            "evidence": [str(phase0_path)],
        },
        {
            "phase": 1,
            "status": "PASS" if phase1.get("decision") == "PASS" else "FAIL",
            "requirement": "Wiki 10M/Code 10M stable fingerprint selector separates domains and has usable Code coverage",
            "evidence": [str(phase1_path)],
            "metrics": {
                "auroc": phase1["selected_configuration"]["auroc"],
                "auprc": phase1["selected_configuration"]["auprc"],
                "code_selected_fraction": phase1["selected_configuration"]["code_selected_fraction"],
            },
        },
        {
            "phase": 2,
            "status": "PASS" if tests >= 5 and failures == 0 and errors == 0 else "FAIL",
            "requirement": "Fingerprint KD masking, layer 9, full-hidden dimensions, frozen teacher, and standard-MoE router probe path tested",
            "evidence": [str(junit_path)],
            "metrics": {"tests": tests, "failures": failures, "errors": errors},
        },
        {
            "phase": 3,
            "status": "PASS" if all(tracker(path, 3) and training_log_valid(path) for path in gradient_dirs) else "FAIL",
            "requirement": "Matched LM/random/full KD-only gradient calibration and additive smoke completed without NaN/skip",
            "evidence": [str(path) for path in gradient_dirs],
            "metrics": {
                "full_lambda": 0.00315,
                "target_kd_to_lm_gradient_range": "10-30%",
            },
        },
        {
            "phase": 4,
            "status": "FAIL",
            "requirement": "200-step stable fingerprint KD must outperform LM-only and dimension-matched random control while preserving Code",
            "evidence": [str(phase4_path), str(phase4_cause_path), str(selector_path), str(full_path)],
            "subchecks": {
                "all_original_runs_valid": phase4_runs_valid,
                "stable_basis_beats_random": phase4_cause["phase4_decision"]["status"] != "FAIL",
                "selector_assignment_value": selector["decision"]["selector_token_specific_value"],
                "full_hidden_beats_random": full["decision"]["full_hidden_target_beats_random_r64"],
                "gross_router_collapse": router["decision"]["gross_expert_collapse_observed"],
            },
        },
        {
            "phase": 5,
            "status": "NOT_RUN_BY_GATE",
            "requirement": "1800-step extension only after Phase-4 PASS and explicit user approval",
            "evidence": [str(full_path)],
            "reason": "Phase 4 failed; report explicitly sets run_1800_steps=false.",
        },
    ]

    evidence_checks = {
        "full_200step_checkpoint_valid": tracker(full_run, 200) and training_log_valid(full_run),
        "wiki_10m_full_hidden_blocks": len(list((Path(
            "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
            "layer_output_stable_subspace_20260810/streaming_stats/wiki/"
            "stable_selector_full_hidden_lambda0p00315_200step_gpu2"
        )).glob("block_*.npz"))) == 5,
        "router_readonly_audit_no_gross_collapse": not router["decision"]["gross_expert_collapse_observed"],
        "random_null_draws": random_null["null"]["draws"],
        "launch_script_present": (repo / "scripts/experiment/a100/run_fingerprint_kd_code_mha.sh").is_file(),
        "streaming_script_present": (repo / "scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh").is_file(),
        "router_probe_script_present": (repo / "scripts/analysis/run_fingerprint_checkpoint_router_usage_mha.sh").is_file(),
    }

    commands = {
        "unit_tests": (
            "CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 "
            "MASTER_ADDR=127.0.0.1 MASTER_PORT=<free-port> "
            "/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python "
            "-m pytest Megatron-LM/tests/unit_tests/test_fingerprint_kd.py -q"
        ),
        "full_hidden_200step": (
            "CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 FP_MODE=soft_full TRAIN_ITERS=200 "
            "MICRO_BATCH_SIZE=48 GLOBAL_BATCH_SIZE=2304 FINGERPRINT_KD_COEFF=0.00315 "
            "FINGERPRINT_KD_LM_LOSS_COEFF=1 "
            "TRAIN_WEIGHTS=<semantic-.inprogress-path> "
            "bash scripts/experiment/a100/run_fingerprint_kd_code_mha.sh"
        ),
        "wiki_10m_drift": (
            "GPU=0 TARGET_LABEL=stable_selector_full_hidden_lambda0p00315_200step_gpu2 "
            "TARGET_LOAD=<full-hidden-final-checkpoint> DOMAIN=wiki TARGET_TOKENS=10000000 "
            "BLOCK_TOKENS=2000000 bash scripts/analysis/run_layer_output_stable_subspace_stats_mha.sh"
        ),
        "router_usage": (
            "GPU=<0-3> LABEL=<semantic-label> LOAD_DIR=<checkpoint> EXPECTED_STEP=200 "
            "DOMAIN=<wiki|code> PROBE_EVAL_ITERS=25 "
            "bash scripts/analysis/run_fingerprint_checkpoint_router_usage_mha.sh"
        ),
        "random_null": (
            "/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python "
            "scripts/analysis/analyze_random_subspace_drift_null.py --draws 1024 --seed 20260811 ..."
        ),
    }

    result = {
        "status": "STOPPED_AT_FAILED_200STEP_GATE",
        "goal_success_conditions_met": False,
        "scientific_question_answered_at_current_gate": True,
        "requires_user_choice_for_new_direction": True,
        "requirements": requirements,
        "evidence_checks": evidence_checks,
        "reproduction_commands": commands,
        "final_decision": {
            "stable_selector": "SUPPORTED",
            "stable_low_drift_basis_as_preservation_target": "REJECTED",
            "full_hidden_on_selected_code_tokens": "REJECTED_AT_200_STEPS",
            "old_replay_fully_replaced": False,
            "run_1800_steps": False,
            "next_actions_requiring_user_choice": [
                "Run multiple independently trained random-r64 seeds to test behavioral robustness.",
                "Start a new method-design goal for an old-sensitive, drift-vulnerable compressed basis.",
                "Stop this line and retain replay or selected replay as the practical fallback.",
            ],
        },
    }

    output_json = root / "audits/GOAL_STATUS_AUDIT.json"
    output_md = root / "report/GOAL_STATUS_AUDIT.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = Path(str(output_json) + ".inprogress")
    tmp_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_json, output_json)

    lines = [
        "# Fingerprint KD sequential-goal status audit",
        "",
        "| Phase | Status | Requirement |",
        "|---:|---|---|",
    ]
    for row in requirements:
        lines.append(f"| {row['phase']} | **{row['status']}** | {row['requirement']} |")
    lines.extend([
        "",
        "## Status",
        "",
        "**STOPPED_AT_FAILED_200STEP_GATE.** The current scientific gate has been answered, but the goal's success conditions are not met. Phase 5 was correctly not run.",
        "",
        "- Stable fingerprint selector: supported.",
        "- Stable low-drift basis as preservation target: rejected.",
        "- Full hidden on selected Code inputs: rejected at 200 steps.",
        "- Old replay fully replaced: no.",
        "- 1800-step extension: no.",
        "",
        "A new training direction now requires an explicit user choice; automatically tuning rank, threshold, lambda, or launching 1800 steps would violate the sequential stopping rule.",
        "",
        "## Reproduction commands",
        "",
    ])
    for name, command in commands.items():
        lines.extend([f"### {name}", "", "```bash", command, "```", ""])
    lines.append(f"Machine-readable audit: `{output_json}`")
    tmp_md = Path(str(output_md) + ".inprogress")
    tmp_md.write_text("\n".join(lines) + "\n")
    os.replace(tmp_md, output_md)
    print(json.dumps({"json": str(output_json), "markdown": str(output_md), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
