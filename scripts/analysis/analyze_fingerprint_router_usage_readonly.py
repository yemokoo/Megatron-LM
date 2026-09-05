#!/usr/bin/env python3
"""Summarize read-only natural-routing probes for fingerprint KD checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LABELS = (
    "random_r64_stable_assignment",
    "random_r64_permuted_assignment",
    "full_hidden_stable_assignment",
)
DOMAINS = ("wiki", "code")
PROBE_RE = re.compile(
    r"probe (?P<name>\S+) at iteration \d+ \| local_iteration: \d+ \| "
    r"next_token_acc: (?P<acc>[0-9.E+-]+) \| ppl: (?P<ppl>[0-9.E+-]+)"
)


def scalar_values(probe_dir: Path, domain: str) -> dict[str, float]:
    event_files = sorted((probe_dir / "tensorboard").glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"expected one TensorBoard event in {probe_dir}, got {len(event_files)}")
    events = EventAccumulator(str(event_files[0]))
    events.Reload()
    prefix = f"{domain}_router_usage/router/"
    values = {}
    for tag in events.Tags()["scalars"]:
        if not tag.startswith(prefix):
            continue
        rows = events.Scalars(tag)
        if len(rows) != 1:
            raise RuntimeError(f"expected one scalar for {tag}, got {len(rows)}")
        values[tag[len(prefix):]] = float(rows[0].value)
    return values


def distribution_stats(values: list[float]) -> dict:
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise RuntimeError(f"expert fractions do not sum to one: {total}")
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    return {
        "fractions": values,
        "max_fraction": max(values),
        "min_fraction": min(values),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(len(values)),
        "effective_num_experts": math.exp(entropy),
        "experts_above_1pct": sum(value >= 0.01 for value in values),
    }


def load_probe(path: Path, domain: str) -> dict:
    log_text = (path / "probe.log").read_text(errors="replace")
    matches = list(PROBE_RE.finditer(log_text))
    if len(matches) != 1:
        raise RuntimeError(f"expected one performance probe in {path}, got {len(matches)}")
    values = scalar_values(path, domain)
    for required in ("old_expert_fraction", "new_expert_fraction", "new_expert_prob_mass"):
        if required not in values:
            raise RuntimeError(f"missing {required} in {path}")
    if not math.isclose(
        values["old_expert_fraction"] + values["new_expert_fraction"],
        1.0,
        rel_tol=0.0,
        abs_tol=2e-5,
    ):
        raise RuntimeError(f"old/new fractions do not sum to one in {path}")

    overall = [values[f"expert_{idx}_usage"] for idx in range(16)]
    layers = {}
    for layer in range(2, 10):
        prefix = f"layer_{layer}/"
        fractions = [values[f"{prefix}expert_{idx}_usage"] for idx in range(16)]
        layers[str(layer)] = {
            "old_expert_fraction": values[f"{prefix}old_expert_fraction"],
            "new_expert_fraction": values[f"{prefix}new_expert_fraction"],
            "new_expert_prob_mass": values[f"{prefix}new_expert_prob_mass"],
            "expert_distribution": distribution_stats(fractions),
        }
    match = matches[0]
    return {
        "tokens": 25 * 8 * 512,
        "next_token_accuracy": float(match.group("acc")),
        "ppl": float(match.group("ppl")),
        "old_expert_fraction": values["old_expert_fraction"],
        "new_expert_fraction": values["new_expert_fraction"],
        "new_expert_prob_mass": values["new_expert_prob_mass"],
        "expert_distribution": distribution_stats(overall),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(args.probe_root)
    probes = {
        label: {
            domain: load_probe(root / domain / label, domain)
            for domain in DOMAINS
        }
        for label in LABELS
    }

    effects = {}
    for comparison, left, right in (
        ("stable_random_vs_permuted", "random_r64_stable_assignment", "random_r64_permuted_assignment"),
        ("full_vs_stable_random", "full_hidden_stable_assignment", "random_r64_stable_assignment"),
    ):
        effects[comparison] = {
            domain: {
                "accuracy_difference": (
                    probes[left][domain]["next_token_accuracy"]
                    - probes[right][domain]["next_token_accuracy"]
                ),
                "old_expert_fraction_difference": (
                    probes[left][domain]["old_expert_fraction"]
                    - probes[right][domain]["old_expert_fraction"]
                ),
                "new_expert_prob_mass_difference": (
                    probes[left][domain]["new_expert_prob_mass"]
                    - probes[right][domain]["new_expert_prob_mass"]
                ),
                "effective_num_experts_difference": (
                    probes[left][domain]["expert_distribution"]["effective_num_experts"]
                    - probes[right][domain]["expert_distribution"]["effective_num_experts"]
                ),
            }
            for domain in DOMAINS
        }

    minimum_effective = min(
        probes[label][domain]["expert_distribution"]["effective_num_experts"]
        for label in LABELS for domain in DOMAINS
    )
    maximum_fraction = max(
        probes[label][domain]["expert_distribution"]["max_fraction"]
        for label in LABELS for domain in DOMAINS
    )
    result = {
        "question": "Did fingerprint KD cause a gross natural-routing or expert-usage collapse?",
        "method": {
            "read_only": True,
            "natural_routing": True,
            "forced_routing": False,
            "domains": list(DOMAINS),
            "test_tokens_per_probe": 25 * 8 * 512,
            "layers": list(range(2, 10)),
            "experts": 16,
            "topk": 4,
            "old_experts": "0-7",
            "new_experts": "8-15",
            "same_seed_data_and_world_size": True,
        },
        "probes": probes,
        "effects": effects,
        "decision": {
            "gross_expert_collapse_observed": bool(
                minimum_effective < 4.0 or maximum_fraction > 0.5
            ),
            "minimum_effective_num_experts": minimum_effective,
            "maximum_single_expert_fraction": maximum_fraction,
            "guardrail": (
                "This is a matched 102,400-token read-only probe, not the broken train-time capture. "
                "It can detect gross routing collapse but does not reconstruct every training-step trajectory."
            ),
        },
        "instrumentation_history": {
            "train_time_capture": "INVALID: standard MoE routers produced zero assignments",
            "root_cause": "probe/train instrumentation originally collected only shared_expert_router modules",
            "fix": "probe now uses the existing standard-MoE mlp.router input hook fallback",
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_json = Path(str(output_json) + ".inprogress")
    tmp_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp_json, output_json)

    lines = [
        "# Fingerprint KD read-only natural-routing audit",
        "",
        "Each row uses the same 102,400-token test subset, world size 1, natural top-4 routing, and no parameter update.",
        "",
        "| run | domain | accuracy | old assignment | new assignment | new prob mass | max expert | effective experts |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        for domain in DOMAINS:
            row = probes[label][domain]
            dist = row["expert_distribution"]
            lines.append(
                f"| {label} | {domain} | {row['next_token_accuracy']:.6f} | "
                f"{row['old_expert_fraction']:.4f} | {row['new_expert_fraction']:.4f} | "
                f"{row['new_expert_prob_mass']:.4f} | {dist['max_fraction']:.4f} | "
                f"{dist['effective_num_experts']:.2f} |"
            )
    collapsed = result["decision"]["gross_expert_collapse_observed"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"Gross expert collapse observed: **{'YES' if collapsed else 'NO'}** under the stated diagnostic threshold. The minimum effective expert count is {minimum_effective:.2f}, and the maximum single-expert assignment fraction is {maximum_fraction:.4f}.",
        "",
        "The earlier train-time zero-assignment scalars were an instrumentation failure, not a routing result. This read-only probe uses the standard MoE router hook and supersedes those invalid scalars.",
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
