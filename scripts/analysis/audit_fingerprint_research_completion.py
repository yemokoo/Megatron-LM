#!/usr/bin/env python3
"""Audit the fingerprint/router-geometry deliverables against the requested scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    repo = Path(args.repo).resolve()
    all_root = root / "extended_all_layers"
    checks = []

    def check(name, passed, evidence):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    inventory_path = root / "inventory/checkpoint_inventory_A_to_I.json"
    inventory = read_json(inventory_path)
    records = inventory["records"]
    check(
        "checkpoint_inventory_A_to_I",
        set(records) == set("ABCDEFGHI"),
        str(inventory_path),
    )
    physical = {key for key, row in records.items() if row["physical_checkpoint_present"]}
    check(
        "physical_checkpoint_lineage",
        physical == {"A", "B", "C", "F", "G", "H"}
        and all(records[key].get("step_matches") for key in physical),
        {key: records[key]["path"] for key in sorted(physical)},
    )
    unavailable = {key for key in "DEI" if not records[key]["physical_checkpoint_present"]}
    check(
        "unavailable_controls_declared",
        unavailable == {"D", "E", "I"}
        and all("missing" in records[key]["comparison_status"] or "reference" in records[key]["comparison_status"] for key in unavailable),
        {key: records[key]["comparison_status"] for key in sorted(unavailable)},
    )

    smoke_validation_path = root / "inventory/dump_semantic_validation.json"
    all_validation_path = all_root / "inventory/dump_semantic_validation.json"
    smoke_validation = read_json(smoke_validation_path)
    all_validation = read_json(all_validation_path)
    check("smoke_tensor_semantics", smoke_validation["all_checks_pass"], str(smoke_validation_path))
    check("all_moe_layer_tensor_semantics", all_validation["all_checks_pass"], str(all_validation_path))

    expected_stages = {
        "wiki": {"A", "B", "C_vocabkl", "G", "H"},
        "code": {"A", "B", "C_vocabkl", "G", "H"},
        "conversation": {"F", "G", "H"},
    }
    for label, manifest_path, layers, samples, tokens_per_sample in (
        ("smoke", root / "inventory/dump_manifest.json", [2, 5, 9], 64, 16),
        (
            "all_moe_layers",
            all_root / "inventory/dump_manifest.json",
            list(range(2, 10)),
            128,
            8,
        ),
    ):
        manifest = read_json(manifest_path)
        coverage = {domain: set(stages) for domain, stages in manifest.items()}
        metadata_ok = all(
            row["metadata"]["layer_numbers"] == layers
            and row["metadata"]["encountered_samples"] == samples
            and row["metadata"]["tokens_per_sample"] == tokens_per_sample
            and row["metadata"]["captured_tokens"] == 1024
            and all(row["identity_exact"].values())
            for stages in manifest.values()
            for row in stages.values()
        )
        check(
            f"{label}_domain_stage_alignment_coverage",
            coverage == expected_stages and metadata_ok,
            str(manifest_path),
        )

    drift_path = root / "metrics/representation_routing_drift.json"
    all_drift_path = all_root / "metrics/representation_routing_drift.json"
    drift = read_json(drift_path)
    all_drift = read_json(all_drift_path)
    required_comparisons = {
        "wiki": {"A_to_B", "B_to_C_vocabkl", "F_to_G", "G_to_H"},
        "code": {"A_to_B", "B_to_C_vocabkl", "F_to_G", "G_to_H"},
        "conversation": {"F_to_G", "G_to_H"},
    }
    check(
        "required_matched_drift_comparisons",
        all(required_comparisons[domain] <= set(all_drift[domain]) for domain in required_comparisons),
        str(all_drift_path),
    )
    decomposition_keys = {
        "representation_stable_routing_stable",
        "representation_stable_routing_changed",
        "representation_changed_routing_stable",
        "representation_changed_routing_changed",
    }
    check(
        "representation_boundary_four_group_decomposition",
        all(
            set(row["drift_decomposition"]["four_token_groups"]) == decomposition_keys
            and "representation_only_common_router" in row["drift_decomposition"]
            and "boundary_only_common_experts" in row["drift_decomposition"]
            for domain in all_drift.values()
            for comparison in domain.values()
            for row in comparison["layers"]
        ),
        str(all_drift_path),
    )

    suff_path = all_root / "metrics/fingerprint_sufficiency.json"
    suff = read_json(suff_path)
    required_bases = {
        "stable",
        "stable_routing_sensitive",
        "top_variance_pca",
        "router_row_space",
        "routing_discriminative",
        "random",
    }
    basis_ok = True
    intervention_ok = True
    for layer in suff["layers"]:
        basis_ok &= required_bases <= {row["basis"] for row in layer["curves"]}
        for row in layer["curves"]:
            if row["basis"] == "random":
                intervention_ok &= "fingerprint_only_topk_agreement_mean" in row
            else:
                intervention_ok &= {
                    "fingerprint_only",
                    "fingerprint_removed",
                } <= set(row)
                intervention_ok &= "margin_recovery" in row["fingerprint_only"]
    check("candidate_and_baseline_coverage", basis_ok, str(suff_path))
    check("analytic_only_removed_and_margin_interventions", intervention_ok, str(suff_path))

    forward_path = root / "metrics/actual_forward_interventions.json"
    forward = read_json(forward_path)
    required_conditions = {
        "baseline",
        "teacher_full",
        "stable_sensitive32_only",
        "stable_sensitive32_removed",
        "pca32_only",
        "random32_only",
        "rowspace16_only",
        "rowspace16_removed",
    }
    check(
        "actual_forward_intervention_coverage",
        forward["status"] == "complete"
        and set(forward["conditions"]) == required_conditions
        and all(
            {"wiki", "code"} <= set(row["domains"])
            for row in forward["conditions"].values()
        ),
        str(forward_path),
    )

    required_plots = {
        "common_pca_router_input_density.png",
        "model_wiki_code_joint_density.png",
        "token_displacement_routing_stability.png",
        "code_conversation_token_displacement.png",
        "expert_assignment_by_layer.png",
        "stable_orthogonal_and_routing_restoration.png",
        "actual_forward_intervention_accuracy.png",
    }
    plot_evidence = {name: str(root / "plots" / name) for name in sorted(required_plots)}
    check(
        "required_visualizations",
        all(Path(path).is_file() and Path(path).stat().st_size > 10_000 for path in plot_evidence.values()),
        plot_evidence,
    )
    check(
        "all_layer_visualizations",
        (all_root / "plots/common_pca_router_input_density.png").stat().st_size > 10_000
        and (all_root / "plots/stable_orthogonal_and_routing_restoration.png").stat().st_size > 10_000,
        str(all_root / "plots"),
    )

    hook_sources = {
        "dump": repo / "Megatron-LM/pretrain_gpt.py",
        "router": repo / "Megatron-LM/megatron/core/transformer/moe/router.py",
        "args": repo / "Megatron-LM/megatron/training/arguments.py",
    }
    hook_text = "\n".join(path.read_text(encoding="utf-8") for path in hook_sources.values())
    hook_terms = {
        "router_input",
        "post_attention_hidden",
        "expert_outputs",
        "router_fingerprint_intervention_mode",
        "fingerprint_only",
        "fingerprint_removed",
    }
    check(
        "hook_and_forward_intervention_implementation",
        all(term in hook_text for term in hook_terms),
        {key: str(path) for key, path in hook_sources.items()},
    )

    report_path = root / "report/fingerprint_router_geometry_report_ko.md"
    report = report_path.read_text(encoding="utf-8")
    report_terms = {
        "Stability",
        "Selectivity",
        "Sufficiency intervention",
        "Representation drift",
        "router-boundary drift",
        "raw-data-free loss",
        "메모리 scaling",
        "최소 후속 실험",
        "실패",
        "New-data-only training",
    }
    check("final_report_scope", all(term in report for term in report_terms), str(report_path))
    check(
        "outputs_confined_to_data2",
        str(root).startswith("/data2/")
        and all(str(Path(row["path"])).startswith("/data2/") for domain in read_json(root / "inventory/dump_manifest.json").values() for row in domain.values()),
        str(root),
    )

    failed = [row["name"] for row in checks if not row["passed"]]
    result = {
        "status": "complete_with_declared_unavailable_controls" if not failed else "incomplete",
        "all_checks_pass": not failed,
        "failed_checks": failed,
        "declared_limitations": [
            "Matched D checkpoint is absent; only its exact repository recipe is available.",
            "E and I are repository references whose physical checkpoints are absent and whose sources are not matched to A/F.",
            "Only one physical checkpoint family/seed exists, so seed variance is not measurable.",
            "New-data-only preservation training was not run; the requested no-large-retraining constraint is respected and a minimal pilot is specified.",
        ],
        "checks": checks,
    }
    out = root / "report/completion_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "status": result["status"], "all_checks_pass": result["all_checks_pass"]}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
