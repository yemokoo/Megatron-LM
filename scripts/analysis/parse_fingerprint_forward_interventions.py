#!/usr/bin/env python3
"""Parse actual-forward fingerprint intervention probe logs into JSON and bars."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LINE = re.compile(
    r"probe (?P<domain>wiki|code)_fingerprint_forward at iteration\s+\d+\s+\|\s+"
    r"local_iteration:\s+\d+\s+\|\s+next_token_acc:\s+(?P<accuracy>[0-9.]+)\s+"
    r"\|\s+ppl:\s+(?P<ppl>[0-9.E+-]+)"
)
ORDER = (
    "baseline",
    "teacher_full",
    "stable_sensitive32_only",
    "stable_sensitive32_removed",
    "pca32_only",
    "random32_only",
    "rowspace16_only",
    "rowspace16_removed",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    logs = root / "logs" / "forward_interventions"
    records = {}
    for condition in ORDER:
        path = logs / f"{condition}.log"
        if not path.is_file():
            continue
        parsed = {}
        for match in LINE.finditer(path.read_text(errors="replace")):
            parsed[match.group("domain")] = {
                "next_token_accuracy": float(match.group("accuracy")),
                "ppl": float(match.group("ppl")),
            }
        if parsed:
            records[condition] = {"log": str(path), "domains": parsed}

    baseline = records.get("baseline", {}).get("domains", {})
    for condition, record in records.items():
        for domain, values in record["domains"].items():
            base = baseline.get(domain, {}).get("next_token_accuracy")
            values["accuracy_fraction_of_C_baseline"] = (
                values["next_token_accuracy"] / base if base else None
            )
    result = {
        "checkpoint": "C_vocabkl",
        "intervened_layers": [2, 5, 9],
        "samples": 64,
        "tokens_per_sample": 512,
        "probe_micro_batch_size": 8,
        "probe_eval_iters": 8,
        "status": "complete" if all(condition in records for condition in ORDER) else "partial",
        "completed_conditions": list(records),
        "conditions": records,
        "interpretation_scope": (
            "Actual forward pass with expert dispatch changed by the stored B teacher router anchor "
            "at layers 2/5/9. It tests routing-context sufficiency, not training-time preservation."
        ),
    }
    metrics_path = root / "metrics" / "actual_forward_interventions.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    conditions = [condition for condition in ORDER if condition in records]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    for domain_index, domain in enumerate(("wiki", "code")):
        values = [
            records[condition]["domains"].get(domain, {}).get("next_token_accuracy", np.nan)
            for condition in conditions
        ]
        colors = ["#111827" if condition == "baseline" else "#7c3aed" for condition in conditions]
        axes[domain_index].bar(np.arange(len(conditions)), values, color=colors, alpha=0.85)
        axes[domain_index].set_xticks(np.arange(len(conditions)), conditions, rotation=40, ha="right")
        axes[domain_index].set_title(f"C actual-forward intervention | {domain}")
        axes[domain_index].set_ylabel("next-token accuracy")
        axes[domain_index].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    plot_path = root / "plots" / "actual_forward_intervention_accuracy.png"
    fig.savefig(plot_path, dpi=190)
    plt.close(fig)
    print(json.dumps({"metrics": str(metrics_path), "plot": str(plot_path), "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
