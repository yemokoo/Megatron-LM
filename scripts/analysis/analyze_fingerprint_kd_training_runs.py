#!/usr/bin/env python3
"""Summarize matched fingerprint-KD training logs without loading checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


PROBE_RE = re.compile(
    r"probe (?P<name>\S+) at iteration (?P<global>\d+) \| "
    r"local_iteration: (?P<local>\d+) \| next_token_acc: (?P<acc>[0-9.E+-]+) "
    r"\| ppl: (?P<ppl>[0-9.E+-]+)"
)
TRAIN_RE = re.compile(r"iteration\s+(?P<step>\d+)/\s*(?P<total>\d+)")
SCALAR_RE = re.compile(r"\| ([^|:]+):\s*([0-9.E+-]+)")


def parse_run(label: str, run_dir: Path) -> dict:
    log_path = run_dir / "logs" / "a_to_b_freeze.log"
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    probes: dict[str, list[dict]] = {}
    train_rows = []
    for line in log_path.read_text(errors="replace").splitlines():
        probe = PROBE_RE.search(line)
        if probe:
            row = {
                "global_iteration": int(probe.group("global")),
                "local_iteration": int(probe.group("local")),
                "next_token_accuracy": float(probe.group("acc")),
                "ppl": float(probe.group("ppl")),
            }
            probes.setdefault(probe.group("name"), []).append(row)
        train = TRAIN_RE.search(line)
        if train:
            row = {"step": int(train.group("step")), "total_steps": int(train.group("total"))}
            for name, value in SCALAR_RE.findall(line):
                try:
                    row[name.strip()] = float(value)
                except ValueError:
                    pass
            train_rows.append(row)

    tracker_path = run_dir / "latest_checkpointed_iteration.txt"
    tracker = int(tracker_path.read_text().strip()) if tracker_path.is_file() else None
    probe_summary = {}
    for name, rows in probes.items():
        first, last = rows[0], rows[-1]
        probe_summary[name] = {
            "first": first,
            "last": last,
            "accuracy_delta": last["next_token_accuracy"] - first["next_token_accuracy"],
            "ppl_relative_change": last["ppl"] / first["ppl"] - 1.0,
            "num_measurements": len(rows),
        }

    def series(name: str) -> list[float]:
        return [row[name] for row in train_rows if name in row and math.isfinite(row[name])]

    series_summary = {}
    for name in (
        "lm loss",
        "fingerprint kd loss",
        "fingerprint mean score",
        "fingerprint mean weight",
        "fingerprint hard coverage",
        "fingerprint score weight covariance",
        "fingerprint_grad_norm/all_trainable",
        "fingerprint_grad_norm/router_all_rows",
        "fingerprint_grad_norm/router_new_rows",
        "fingerprint_grad_norm/new_experts",
        "grad norm",
        "number of skipped iterations",
        "number of nan iterations",
    ):
        values = series(name)
        if values:
            series_summary[name] = {
                "first": values[0],
                "last": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "count": len(values),
            }
    layer_last = {
        name: values[-1]
        for layer in range(2, 10)
        for name in [f"fingerprint layer {layer} projected mse"]
        if (values := series(name))
    }
    return {
        "label": label,
        "path": str(run_dir),
        "checkpoint_tracker": tracker,
        "training_complete": "training complete" in log_path.read_text(errors="replace"),
        "probes": probes,
        "probe_summary": probe_summary,
        "training_series_summary": series_summary,
        "last_layer_projected_mse": layer_last,
        "logged_training_steps": [row["step"] for row in train_rows],
    }


def markdown_report(payload: dict) -> str:
    lines = ["# Fingerprint KD training summary", "", "| run | step | Code acc | Code PPL | Wiki acc | Wiki PPL |", "|---|---:|---:|---:|---:|---:|"]
    for run in payload["runs"]:
        code = run["probe_summary"].get("code_probe", {}).get("last", {})
        wiki = run["probe_summary"].get("wiki_probe", {}).get("last", {})
        lines.append(
            "| {label} | {step} | {ca:.6f} | {cp:.6f} | {wa:.6f} | {wp:.6f} |".format(
                label=run["label"],
                step=run["checkpoint_tracker"] or 0,
                ca=code.get("next_token_accuracy", float("nan")),
                cp=code.get("ppl", float("nan")),
                wa=wiki.get("next_token_accuracy", float("nan")),
                wp=wiki.get("ppl", float("nan")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=/absolute/run/path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    runs = []
    for spec in args.run:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            raise SystemExit(f"invalid --run value: {spec!r}")
        runs.append(parse_run(label, Path(path)))
    payload = {"schema_version": 1, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(payload))


if __name__ == "__main__":
    main()
