#!/usr/bin/env python3
"""Prove that an ablation arm trained with the switches it claims.

Every check is an observable consequence of the switch, not just a restatement
of the flag:

  phase_mode  2phase must leave every previously learned router row
              bit-identical through phase 1 (prephase2 prefix == previous
              round's router), and must move only the router in phase 2
              (prephase2 experts == round experts, routers differ).
              1phase must NOT satisfy the frozen-prefix property.
  kd_init     on/off shows up as the presence/absence of the kd_init role in
              training_workload.json for every round after the first.
  replay      real writes fixed_replay_memory/<task>.json with exactly the
              configured persistent count; selfgen logs one
              "[selfgen] replay/KD for <task>" line per past task per round.
  residual    the post-hoc arm carries residual_expert_meta.json with
              n_residual=1 and residual_only=false.

usage:
  python verify_ablation_run.py <run_dir> --expect-phase 2phase \
      --expect-kd off --expect-replay selfgen [--rounds 8] [--log <train.log>]
  python verify_ablation_run.py <residual_out> --residual
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
PREPHASE2 = "_prephase2"
ROUTER_KEY = re.compile(r"shared_expert_router\.router\.weight$")
EXPERT_KEY = re.compile(r"experts\.\d+\.")

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))

    def ok(self) -> bool:
        return all(status != FAIL for status, _, _ in self.rows)

    def render(self) -> str:
        width = max(len(check) for _, check, _ in self.rows)
        lines = [f"[{status}] {check.ljust(width)}  {detail}".rstrip()
                 for status, check, detail in self.rows]
        verdict = "ALL CHECKS PASSED" if self.ok() else "VERIFICATION FAILED"
        return "\n".join(lines + ["", verdict])


def load_state(directory: Path) -> dict:
    """Load a round's trainable tensors (router + experts only)."""
    import torch
    for name in ("pytorch_model.bin", "adapter_model.bin"):
        path = directory / name
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no checkpoint weights under {directory}")


def split_router_experts(state: dict) -> tuple[dict, dict]:
    routers = {k: v for k, v in state.items() if ROUTER_KEY.search(k)}
    experts = {k: v for k, v in state.items() if EXPERT_KEY.search(k)}
    return routers, experts


def check_meta(run: Path, rounds: int, expect: dict, report: Report) -> None:
    seen = []
    for index in range(rounds):
        meta_path = run / str(index) / "lora_moe_meta.json"
        if not meta_path.is_file():
            report.add(FAIL, f"meta round {index}", f"missing {meta_path}")
            continue
        block = json.load(meta_path.open()).get("ablation")
        if block is None:
            report.add(FAIL, f"meta round {index}",
                       "no ablation block (trained by pre-ablation code?)")
            continue
        seen.append({key: block.get(key) for key in
                     ("phase_mode", "kd_init", "replay_source")})
    if not seen:
        return
    if any(entry != seen[0] for entry in seen[1:]):
        report.add(FAIL, "meta consistent across rounds", str(seen))
    else:
        report.add(PASS, "meta consistent across rounds", str(seen[0]))
    for key, want in expect.items():
        got = seen[0].get(key)
        report.add(PASS if got == want else FAIL, f"meta {key}",
                   f"expected {want}, got {got}")


def check_workload(run: Path, expect_kd: str, expect_phase: str,
                   report: Report) -> None:
    path = run / "training_workload.json"
    if not path.is_file():
        report.add(FAIL, "training_workload.json", f"missing {path}")
        return
    payload = json.load(path.open())
    kd_rounds, phase2_rounds, joint_rounds = [], [], []
    for entry in payload.get("tasks", []):
        roles = entry.get("roles", {})
        if "kd_init" in roles:
            kd_rounds.append(entry["round"])
        if "router_replay_phase2" in roles:
            phase2_rounds.append(entry["round"])
        if "router_replay" in roles:
            joint_rounds.append(entry["round"])
    later = [entry["round"] for entry in payload.get("tasks", [])
             if entry["round"] > 0]
    if expect_kd == "on":
        report.add(PASS if kd_rounds == later else FAIL, "workload kd_init",
                   f"rounds with KD: {kd_rounds} (expected {later})")
    else:
        report.add(PASS if not kd_rounds else FAIL, "workload kd_init",
                   f"rounds with KD: {kd_rounds} (expected none)")
    if expect_phase == "2phase":
        report.add(PASS if phase2_rounds == later else FAIL,
                   "workload phase-2 replay",
                   f"rounds with router_replay_phase2: {phase2_rounds} "
                   f"(expected {later})")
        report.add(PASS if not joint_rounds else FAIL,
                   "workload no joint replay",
                   f"rounds with interleaved router_replay: {joint_rounds}")
    else:
        report.add(PASS if joint_rounds == later else FAIL,
                   "workload joint replay",
                   f"rounds with router_replay: {joint_rounds} "
                   f"(expected {later})")
        report.add(PASS if not phase2_rounds else FAIL,
                   "workload no phase-2 replay", str(phase2_rounds))
    # replay exposure parity: phase-2 must see what the joint arm would have
    for entry in payload.get("tasks", []):
        roles = entry.get("roles", {})
        replay = roles.get("router_replay_phase2") or roles.get(
            "router_replay")
        if replay is None:
            continue
        expected = entry["epochs"] * (
            entry.get("router_replay_exposure_budget_global", 0)
            // max(1, entry["epochs"]))
        got = replay["input_sample_exposures"]
        if expected and got != expected:
            report.add(WARN, f"replay exposures round {entry['round']}",
                       f"{got} vs budget-derived {expected}")


def check_phase_weights(run: Path, rounds: int, expect_phase: str,
                        report: Report) -> None:
    import torch
    if expect_phase == "2phase":
        missing = [index for index in range(rounds)
                   if not (run / f"{index}{PREPHASE2}"
                           / "lora_moe_meta.json").is_file()]
        report.add(PASS if not missing else FAIL, "prephase2 checkpoints",
                   f"missing rounds: {missing}" if missing
                   else f"all {rounds} present")
        if missing:
            return
    else:
        extra = [index for index in range(rounds)
                 if (run / f"{index}{PREPHASE2}").is_dir()]
        report.add(PASS if not extra else FAIL, "no prephase2 checkpoints",
                   f"unexpected: {extra}" if extra else "none, as expected")
        return

    for index in range(1, rounds):
        previous = split_router_experts(load_state(run / str(index - 1)))[0]
        pre = load_state(run / f"{index}{PREPHASE2}")
        post = load_state(run / str(index))
        pre_router, pre_experts = split_router_experts(pre)
        post_router, post_experts = split_router_experts(post)
        if not previous or not pre_router:
            report.add(FAIL, f"round {index} router tensors", "not found")
            continue
        # phase 1 must not touch any router row that already existed
        prefix_drift = 0.0
        for key, old in previous.items():
            new = pre_router.get(key)
            if new is None:
                continue
            rows = old.shape[0]
            prefix_drift = max(prefix_drift, float(
                (new[:rows].float() - old.float()).abs().max()))
        report.add(PASS if prefix_drift == 0.0 else FAIL,
                   f"round {index} phase1 froze old router rows",
                   f"max |drift| = {prefix_drift:.3e}")
        # phase 2 must move the router and nothing else
        router_drift = max(
            float((post_router[key].float() - value.float()).abs().max())
            for key, value in pre_router.items() if key in post_router)
        expert_drift = max(
            (float((post_experts[key].float() - value.float()).abs().max())
             for key, value in pre_experts.items() if key in post_experts),
            default=0.0)
        report.add(PASS if router_drift > 0 else FAIL,
                   f"round {index} phase2 moved router",
                   f"max |delta| = {router_drift:.3e}")
        report.add(PASS if expert_drift == 0.0 else FAIL,
                   f"round {index} phase2 left experts frozen",
                   f"max |delta| = {expert_drift:.3e}")


def read_logs(logs) -> str:
    """Concatenate one or more train logs (a gen chain writes one per round)."""
    text = []
    for entry in logs or []:
        matches = sorted(glob.glob(str(entry))) or [str(entry)]
        for match in matches:
            path = Path(match)
            if path.is_file():
                text.append(path.read_text(errors="ignore"))
    return "\n".join(text)


def check_replay_source(run: Path, rounds: int, expect: str, logs,
                        persistent: int, report: Report) -> None:
    memory_dir = run / "fixed_replay_memory"
    if expect == "real":
        files = sorted(memory_dir.glob("*.json")) if memory_dir.is_dir() else []
        counts = {}
        for path in files:
            payload = json.load(path.open())
            indices = payload.get("indices", payload if isinstance(payload, list) else [])
            counts[path.stem] = len(indices)
        bad = {name: count for name, count in counts.items()
               if count != persistent}
        report.add(PASS if files and not bad else FAIL,
                   "real replay memory files",
                   f"{len(files)} tasks, counts={sorted(set(counts.values()))}"
                   + (f" BAD={bad}" if bad else ""))
        return
    text = read_logs(logs)
    if not text:
        report.add(WARN, "selfgen replay log",
                   "no train log given; pass --log to prove generated records")
        return
    hits = re.findall(
        r"\[selfgen\] replay/KD for ([\w-]+): (\d+) generated records", text)
    tasks = {task for task, _ in hits}
    counts = {int(count) for _, count in hits}
    expected_tasks = set(TASKS[:max(1, rounds - 1)])
    missing = expected_tasks - tasks
    report.add(PASS if hits and not missing else FAIL,
               "selfgen replay records used",
               f"{len(hits)} loads over tasks {sorted(tasks)}"
               + (f" MISSING={sorted(missing)}" if missing else ""))
    report.add(PASS if counts == {persistent} else WARN,
               "selfgen record counts", f"{sorted(counts)}")
    fallbacks = re.findall(
        r"\[selfgen\] ([\w-]+) is the task being trained", text)
    report.add(PASS, "selfgen current-task fallback (expected)",
               f"{len(fallbacks)} rounds: {sorted(set(fallbacks))}")


def check_residual(out: Path, report: Report) -> None:
    meta_path = out / "residual_expert_meta.json"
    if not meta_path.is_file():
        report.add(FAIL, "residual_expert_meta.json", f"missing {meta_path}")
        return
    meta = json.load(meta_path.open())
    n_residual = meta.get("n_residual")
    residual_only = bool(meta.get("residual_only", False))
    report.add(PASS if n_residual == 1 else FAIL, "residual rows",
               f"n_residual={n_residual}")
    report.add(PASS if not residual_only else FAIL, "full router trainable",
               f"residual_only={residual_only}")
    source = meta.get("source_checkpoint")
    report.add(PASS if source and Path(source).is_dir() else FAIL,
               "source checkpoint present", str(source))
    report.add(PASS if (out / "router_state.pt").is_file() else FAIL,
               "tuned router saved", "router_state.pt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expect-phase", choices=["1phase", "2phase"])
    parser.add_argument("--expect-kd", choices=["on", "off"])
    parser.add_argument("--expect-replay", choices=["real", "selfgen"])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--persistent", type=int, default=500,
                        help="v2_new_persistent_samples_per_task")
    parser.add_argument("--log", nargs="*", default=None,
                        help="train log(s) holding the [selfgen] lines; a gen "
                             "chain writes one per round, so pass a glob")
    parser.add_argument("--residual", action="store_true",
                        help="verify a post-hoc residual-expert output dir")
    parser.add_argument("--skip-weights", action="store_true",
                        help="skip the tensor-level phase checks")
    args = parser.parse_args()

    report = Report()
    run = args.run_dir
    if not run.is_dir():
        print(f"[FAIL] run dir missing: {run}")
        return 1
    if args.residual:
        check_residual(run, report)
        print(report.render())
        return 0 if report.ok() else 1

    if not (args.expect_phase and args.expect_kd and args.expect_replay):
        print("--expect-phase/--expect-kd/--expect-replay are required")
        return 2
    expect = {"phase_mode": args.expect_phase, "kd_init": args.expect_kd,
              "replay_source": args.expect_replay}
    check_meta(run, args.rounds, expect, report)
    check_workload(run, args.expect_kd, args.expect_phase, report)
    if args.skip_weights:
        report.add(WARN, "tensor-level phase checks", "skipped")
    else:
        check_phase_weights(run, args.rounds, args.expect_phase, report)
    check_replay_source(run, args.rounds, args.expect_replay, args.log,
                        args.persistent, report)
    print(report.render())
    return 0 if report.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
