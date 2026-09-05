#!/usr/bin/env python3
"""One-screen progress view of the self-generated-replay CL chain.

Renders every stage in the order the driver runs them:

    round t:  [KD-init]  ->  [1-phase train]  ->  [generate replay for round t+1]

Round 0 has no KD-init (no past experts) and round 7 no generation (no next task).
State comes from the artefacts themselves, so it is correct even if the driver
was restarted: a saved checkpoint means the round trained, a records.jsonl means
that task's replay was generated, and the live tqdm bar in the training log gives
the in-flight percentage.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/selfgen_cl_20260831")
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
BAR_W = 22
DONE, RUNNING, PENDING, FAILED = "\033[32m", "\033[36m", "\033[90m", "\033[31m"
RESET, BOLD = "\033[0m", "\033[1m"

# "395/395 [04:49<00:00" -- tqdm writes with \r, so scan the raw bytes
STEP_RE = re.compile(rb"(\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+)")


def tqdm_bars(path: Path):
    """Every (cur, total, elapsed, eta) the log has, in order."""
    if not path.is_file():
        return []
    data = path.read_bytes()
    return [(int(a), int(b), c.decode(), d.decode())
            for a, b, c, d in STEP_RE.findall(data)]


def split_phases(bars, has_kd):
    """KD-init bar then the 1-phase bar; both run the task's full step count.

    Training logs also carry short probe/eval bars (e.g. 4/4), so only bars whose
    total equals the task's step count are considered.
    """
    if not bars:
        return None, None
    total = max(b[1] for b in bars)
    big = [b for b in bars if b[1] == total]
    if not big:
        return None, None
    if not has_kd:
        return None, big[-1]
    # tqdm reprints its final line, so work by position: the KD phase ends at the
    # first completed bar; everything after it belongs to the 1-phase run.
    first_done = next((i for i, b in enumerate(big) if b[0] == b[1]), None)
    if first_done is None:
        return big[-1], None                       # still inside KD-init
    kd = big[first_done]
    rest = [b for b in big[first_done + 1:] if b != kd]
    return kd, (rest[-1] if rest else None)


def bar(frac, width=BAR_W):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def phase_line(label, phase, color_done=True):
    if phase is None:
        return f"{PENDING}{label:<10} {'░' * BAR_W}   대기{RESET}"
    cur, total, elapsed, eta = phase
    frac = cur / total if total else 0
    if cur >= total:
        return f"{DONE}{label:<10} {bar(1.0)} 100%  {elapsed}{RESET}"
    return (f"{RUNNING}{label:<10} {bar(frac)} {frac*100:3.0f}%  "
            f"{cur}/{total}  경과 {elapsed} 남은 {eta}{RESET}")


def gen_state(round_index, upto_task):
    """Generation that FEEDS round_index: tasks 0..upto_task."""
    dest = RUN / "gen" / f"round_{round_index}"
    done, running, pending = [], [], []
    for i in range(upto_task + 1):
        task = TASKS[i]
        d = dest / task
        rec = d / "records.jsonl"
        if rec.is_file() and rec.stat().st_size > 0:
            done.append(f"{task}({sum(1 for _ in rec.open())})")
            continue
        # generation is sharded: stageA.shard<k> dirs and records.part<k>.jsonl
        parts = sorted(d.glob("records.part*.jsonl")) if d.is_dir() else []
        shards = sorted(d.glob("stageA.shard*")) + ([d / "stageA"] if (d / "stageA").is_dir() else [])
        if parts or shards:
            got = sum(sum(1 for _ in p_.open()) for p_ in parts)
            # A part file grows while its answer pass runs, so a bare count reads like a
            # finished total and looks (twice now) like a shortfall.  Show stage B's own
            # progress from its log instead.
            prog = ""
            logs = sorted((RUN / "logs").glob(f"genB_r{round_index}_{task}_s*.log"))
            cur = tot = 0
            for lg in logs:
                m = None
                for m in re.finditer(rb"\[ans\] (\d+)/(\d+),", lg.read_bytes()):
                    pass
                if m:
                    cur += int(m.group(1)); tot += int(m.group(2))
            if tot:
                prog = f" 답변 {cur}/{tot}"
            running.append(f"{task}[{len(parts)}/{len(shards)}샤드, {got}rec{prog}]")
        else:
            pending.append(task)
    return done, running, pending


def main():
    ck = lambda t: (RUN / "model" / str(t) / "lora_moe_meta.json").is_file()
    print(f"\n{BOLD}TRACE self-generated replay CL{RESET}   "
          f"{RUN.name}   {time.strftime('%F %H:%M:%S')}")
    driver = "ALIVE"
    try:
        import subprocess
        driver = ("ALIVE" if subprocess.run(
            ["pgrep", "-f", "run_selfgen_cl"], capture_output=True).returncode == 0
            else "DEAD")
    except Exception:
        driver = "?"
    print(f"driver: {driver}\n" + "─" * 78)

    for t, task in enumerate(TASKS):
        trained = ck(t)
        bars = tqdm_bars(RUN / "logs" / f"train_r{t}.log")
        has_kd = t > 0
        kd, main_bar = split_phases(bars, has_kd)
        mark = DONE + "✅" + RESET if trained else (
            RUNNING + "🔄" + RESET if bars else PENDING + "⏳" + RESET)
        print(f"{BOLD}round {t}  {task:<12}{RESET} {mark}")
        if has_kd:
            print("   " + phase_line("KD-init", kd))
        else:
            print(f"   {PENDING}{'KD-init':<10} {'─' * BAR_W}   해당 없음 (과거 expert 없음){RESET}")
        print("   " + phase_line("train", main_bar))

        if t < len(TASKS) - 1:
            done, running, pending = gen_state(t + 1, t)
            n_all = t + 1
            if len(done) == n_all:
                print(f"   {DONE}{'gen→r' + str(t+1):<10} {bar(1.0)} 100%  "
                      f"{', '.join(done)}{RESET}")
            elif done or running:
                print(f"   {RUNNING}{'gen→r' + str(t+1):<10} {bar(len(done)/n_all)} "
                      f"{len(done)}/{n_all}  완료[{', '.join(done) or '-'}] "
                      f"생성중[{', '.join(running) or '-'}]{RESET}")
            else:
                print(f"   {PENDING}{'gen→r' + str(t+1):<10} {'░' * BAR_W}   대기 "
                      f"({n_all}개 task){RESET}")
        else:
            print(f"   {PENDING}{'gen':<10} {'─' * BAR_W}   해당 없음 (마지막 task){RESET}")
        print()

    done_rounds = sum(1 for t in range(8) if ck(t))
    print("─" * 78)
    print(f"{BOLD}전체 {bar(done_rounds/8, 40)} {done_rounds}/8 라운드 완료{RESET}\n")


if __name__ == "__main__":
    main()
