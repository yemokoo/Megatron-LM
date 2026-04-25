#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


PROBE_LINE_RE = re.compile(
    r"probe\s+(?P<name>\S+)\s+at iteration\s+(?P<step>\d+)\s+\|\s+"
    r"local_iteration:\s+(?P<local_step>\d+)\s+\|\s+"
    r"next_token_acc:\s+(?P<acc>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.Ee+-]+)"
)


def find_event_files(run_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in run_dir.rglob("events.out.tfevents*")
        if "wandb" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(f"No events.out.tfevents* found under {run_dir}")
    return sorted(candidates)


def load_scalar_history(run_dir: Path) -> dict[str, list[tuple[int, float, float]]]:
    history = defaultdict(list)
    for event_file in find_event_files(run_dir):
        acc = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for event in acc.Scalars(tag):
                history[tag].append((int(event.step), float(event.value), float(event.wall_time)))

    merged = {}
    for tag, events in history.items():
        events.sort(key=lambda item: (item[0], item[2]))
        by_step = {}
        for step, value, wall_time in events:
            by_step[step] = (value, wall_time)
        merged[tag] = [(step, by_step[step][0], by_step[step][1]) for step in sorted(by_step)]
    return merged


def payload_at_or_before_step(history: dict[str, list[tuple[int, float, float]]], step: int) -> dict[str, float]:
    payload = {}
    for tag, events in history.items():
        chosen = None
        for event_step, value, _wall_time in events:
            if event_step <= step:
                chosen = value
            else:
                break
        if chosen is not None:
            payload[tag] = chosen
    return payload


def max_step(history: dict[str, list[tuple[int, float, float]]]) -> int:
    best = 0
    for events in history.values():
        if events:
            best = max(best, events[-1][0])
    return best


def history_to_step_payloads(
    history: dict[str, list[tuple[int, float, float]]],
    *,
    step_offset: int = 0,
) -> dict[int, dict[str, float]]:
    payloads = defaultdict(dict)
    for tag, events in history.items():
        for step, value, _wall_time in events:
            payloads[step + step_offset][tag] = value
    return dict(sorted(payloads.items()))


def parse_probe_history(log_file: Path) -> dict[int, dict[str, float]]:
    payloads = defaultdict(dict)
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            match = PROBE_LINE_RE.search(raw_line)
            if match is None:
                continue
            step = int(match.group("step"))
            probe_name = match.group("name")
            payloads[step][f"{probe_name}/next_token_accuracy"] = float(match.group("acc"))
            payloads[step][f"{probe_name}/ppl"] = float(match.group("ppl"))
    if not payloads:
        raise RuntimeError(f"No probe history found in {log_file}")
    return dict(sorted(payloads.items()))


def collapse_probe_payload_at_or_before_step(payloads: dict[int, dict[str, float]], step: int) -> dict[str, float]:
    collapsed = {}
    for event_step, payload in sorted(payloads.items()):
        if event_step > step:
            break
        collapsed.update(payload)
    return collapsed


def merge_payloads(base: dict[int, dict[str, float]], overlay: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    merged = {step: dict(payload) for step, payload in base.items()}
    for step, payload in overlay.items():
        merged.setdefault(step, {})
        merged[step].update(payload)
    return dict(sorted(merged.items()))


def filter_payloads_after_step(payloads: dict[int, dict[str, float]], step: int) -> dict[int, dict[str, float]]:
    return {event_step: payload for event_step, payload in payloads.items() if event_step > step}


def filter_payloads_at_or_after_step(payloads: dict[int, dict[str, float]], step: int) -> dict[int, dict[str, float]]:
    return {event_step: payload for event_step, payload in payloads.items() if event_step >= step}


def replay_payloads(run, payloads: dict[int, dict[str, float]]) -> None:
    for step, payload in sorted(payloads.items()):
        wandb.log(payload, step=step)


def infer_continual_step_offset(continual_history: dict[str, list[tuple[int, float, float]]], baseline_step: int) -> int:
    continual_max = max_step(continual_history)
    if continual_max <= baseline_step:
        return baseline_step
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Replay full scalar history to a new W&B run, with optional source baseline stitching."
    )
    parser.add_argument("--run-dir", type=Path, help="Standalone run directory whose scalar history will be replayed as-is.")
    parser.add_argument("--source-run-dir", type=Path, help="Source run directory used to provide baseline values.")
    parser.add_argument("--source-log", type=Path, help="Optional source log file used to add source probe metrics to the baseline step.")
    parser.add_argument("--continual-run-dir", type=Path, help="Continual run directory whose full scalar history will be replayed.")
    parser.add_argument("--continual-log", type=Path, help="Optional continual log file used to overlay probe values by step.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--mode", default="online", choices=["online", "offline"])
    parser.add_argument("--baseline-step", type=int, help="Step at which source baseline values should be inserted.")
    args = parser.parse_args()

    standalone_mode = args.run_dir is not None
    stitched_mode = args.source_run_dir is not None and args.continual_run_dir is not None
    if standalone_mode == stitched_mode:
        raise SystemExit("Choose exactly one mode: either --run-dir, or both --source-run-dir and --continual-run-dir.")
    if stitched_mode and args.baseline_step is None:
        raise SystemExit("--baseline-step is required in stitched mode.")

    init_kwargs = {
        "project": args.project,
        "name": args.run_name,
        "id": args.run_id,
        "resume": "never",
        "mode": args.mode,
    }
    if args.entity:
        init_kwargs["entity"] = args.entity
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs["dir"] = str(args.save_dir)

    if standalone_mode:
        payloads = history_to_step_payloads(load_scalar_history(args.run_dir))
        if args.continual_log is not None:
            payloads = merge_payloads(payloads, parse_probe_history(args.continual_log))
    else:
        source_history = load_scalar_history(args.source_run_dir)
        continual_history = load_scalar_history(args.continual_run_dir)
        payloads = {args.baseline_step: payload_at_or_before_step(source_history, args.baseline_step)}
        if args.source_log is not None:
            source_probe_history = parse_probe_history(args.source_log)
            payloads[args.baseline_step].update(
                collapse_probe_payload_at_or_before_step(source_probe_history, args.baseline_step)
            )
        step_offset = infer_continual_step_offset(continual_history, args.baseline_step)
        continual_payloads = filter_payloads_after_step(
            history_to_step_payloads(continual_history, step_offset=step_offset),
            args.baseline_step,
        )
        payloads = merge_payloads(payloads, continual_payloads)
        if args.continual_log is not None:
            continual_probe_payloads = filter_payloads_at_or_after_step(
                parse_probe_history(args.continual_log),
                args.baseline_step,
            )
            payloads = merge_payloads(payloads, continual_probe_payloads)

    run = wandb.init(**init_kwargs)
    try:
        replay_payloads(run, payloads)
    finally:
        run.finish()

    if standalone_mode:
        print(f"Replayed full scalar history from {args.run_dir}")
    else:
        print(
            f"Replayed stitched history: baseline step {args.baseline_step} from {args.source_run_dir}, "
            f"continual history from {args.continual_run_dir}"
        )


if __name__ == "__main__":
    main()
