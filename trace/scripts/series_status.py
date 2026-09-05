#!/usr/bin/env python3
"""One-screen status for the v3 replay series: what is done, what is running.

Round counts come from the checkpoints on disk rather than the logs, because a
run that was killed and restarted leaves stale progress lines behind.  The
in-round bar is read from the live tqdm line, which is the only place the
current epoch's position exists.
"""
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

RUNS = "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace"
LANES = f"{RUNS}/series_lanes"
EPOCHS = [5, 3, 7, 5, 3, 5, 5, 7]
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
# slug, version, one-line config, short column label.  The label is separate
# because the run names are far too long to head a results column.
V3 = [
    # Two different things were both being called "KD", so the labels split them:
    #   KDi   = expansion KD-init, the phase that seeds a new expert from the
    #           old model before joint training
    #   hMSE  = the joint replay objective, hidden-state MSE against the
    #           post-KD-init teacher, as opposed to lm (causal-LM on replay)
    # Percentages are of one primary epoch (5,000 samples).
    ("v3_replay1to1", "v3_new_replay1to1",
     "replay lm 100%, KDi 100%, mem 10%", "lm"),
    ("v3_hidden_mse_1to1", "v3_new_hidden_mse_1to1",
     "replay hMSE 100%, KDi 100%, mem 10%", "hMSE"),
    ("v3_hidden_mse_1to1_p5k", "v3_new_hidden_mse_1to1_p5k",
     "replay hMSE 100%, KDi 4%, mem 100%", "hMSE.m100"),
    ("v3_replay1to1_p5k", "v3_new_replay1to1_p5k",
     "replay lm 100%, KDi 4%, mem 100%", "lm.m100"),
    ("v3_replay1to1_recency", "v3_new_replay1to1_recency",
     "replay lm 100% recency, KDi 4%, mem 10%", "lm.rec"),
    ("v3_kd35k", "v3_new_kd35k",
     "replay lm 100%, KDi 20%, mem 10%", "KDi20"),
    ("v3_recency_kd175k", "v3_new_recency_kd175k",
     "replay lm 100% recency, KDi 100%, mem 10%", "rec.KDi100"),
    ("v3_p5k_kd175k", "v3_new_p5k_kd175k",
     "replay lm 100%, KDi 100%, mem 100%", "mem100"),
    ("v3_r20_kd100", "v3_new_r20_kd100",
     "replay lm 20%, KDi 100%, mem 10%", "rep20"),
    ("v3_kd200", "v3_new_kd200",
     "replay lm 100%, KDi 200%, mem 10%", "KDi200"),
    ("v3_hmse_kd200", "v3_new_hmse_kd200",
     "replay hMSE 100%, KDi 200%, mem 10%", "hMSE.KDi200"),
]



def bar(done, total, width=24):
    if total <= 0:
        return " " * width
    filled = int(round(width * done / total))
    return "█" * filled + "░" * (width - filled)


def rounds_done(slug, version):
    root = f"{RUNS}/{slug}/{version}_st_top1"
    return sum(1 for i in range(8)
               if os.path.exists(f"{root}/{i}/lora_moe_meta.json"))


def tail_text(path, size=4000):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - size))
            return handle.read().decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


LIVE = re.compile(
    r"(?P<task>[A-Za-z0-9-]+) \[(?P<phase>v3 [^\]]*)\][^\n]*?"
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r"(?: \[(?P<el>[0-9:]+)<(?P<rem>[0-9:]+))?")


def live_line(log):
    matches = list(LIVE.finditer(tail_text(log)))
    return matches[-1].groupdict() if matches else None


SLORA_LIVE = re.compile(
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<cur>\d+)/(?P<tot>\d+)"
    r" \[(?P<el>[0-9:]+)<(?P<rem>[0-9:]+)")


def slora_live_line(log):
    """SFTTrainer emits a bare tqdm bar -- no task or phase prefix."""
    matches = list(SLORA_LIVE.finditer(tail_text(log)))
    return matches[-1].groupdict() if matches else None


def slora_gpus():
    """SLoRA runs outside the lanes, so read the devices off the live trainer."""
    try:
        pids = subprocess.run(
            ["pgrep", "-f", "cl_train_slora.py"],
            capture_output=True, text=True, timeout=20).stdout.split()
        for pid in pids:
            with open(f"/proc/{pid}/environ", "rb") as handle:
                for entry in handle.read().split(b"\0"):
                    if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                        value = entry.split(b"=", 1)[1].decode()
                        if value:
                            return value
    except Exception:
        pass
    return None


GPUS_LINE = re.compile(r"GPUs=([0-9,]+)")


def gpu_hint(log, version):
    """Which GPUs a run holds, for runs started outside a lane.

    The launcher's "GPUs=" banner goes to the launch log, not the run's own
    train.log, so read CUDA_VISIBLE_DEVICES off the live trainer instead.
    """
    match = GPUS_LINE.search(tail_text(log, 200000))
    if match:
        return match.group(1)
    try:
        pids = subprocess.run(
            ["pgrep", "-f", f"training_version {version}"],
            capture_output=True, text=True, timeout=20).stdout.split()
        for pid in pids:
            with open(f"/proc/{pid}/environ", "rb") as handle:
                for entry in handle.read().split(b"\0"):
                    if entry.startswith(b"CUDA_VISIBLE_DEVICES="):
                        value = entry.split(b"=", 1)[1].decode()
                        if value:
                            return value
    except Exception:
        pass
    return None


def lane_current(log):
    """Which unit this lane last started, and whether it is still open."""
    started, ended = None, None
    try:
        with open(log, errors="replace") as handle:
            for line in handle:
                m = re.search(r"START (\S+) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
                if m:
                    started = m.group(1)
                m = re.search(r"\] (\S+) exit=", line)
                if m:
                    ended = m.group(1)
    except OSError:
        return None
    return None if started is None or started == ended else started


METRIC = {
    "C-STANCE": "accuracy", "FOMC": "accuracy", "MeetingBank": "rouge-l",
    "Py150": "similarity", "ScienceQA": "accuracy", "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy", "20Minuten": "sari",
}


def scale(value):
    """Result files mix 0-1 and 0-100 conventions; normalise to 0-100."""
    value = float(value)
    return value * 100.0 if value <= 1.0 else value


def v3_cell(slug, version, round_id, task):
    path = (f"{RUNS}/{slug}/{version}_st_top1/evaluation/"
            f"order{round_id}/results-{task}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            scores = json.load(handle).get("eval") or {}
    except (OSError, ValueError):
        return None
    wanted = METRIC[task]
    for key, value in scores.items():
        if key.lower() == wanted and isinstance(value, (int, float)):
            return scale(value)
    return None


SLORA = object()   # marks the SLoRA rows, which lay out their run dir differently
SLORA_LINE = re.compile(r"In \S+: (\{.*\})")


def slora_cell(round_id, task):
    log = (f"{RUNS}/slora_pre_upstream/llama31/pre/evaluation/"
           f"order{round_id}/{task}/eval.log")
    if not os.path.isfile(log):
        return None
    match = None
    for line in open(log, errors="replace"):
        found = SLORA_LINE.search(line)
        if found:
            match = found.group(1)
    if match is None:
        return None
    try:
        scores = ast.literal_eval(match)
    except (ValueError, SyntaxError):
        return None
    wanted = METRIC[task]
    for key, value in scores.items():
        if key.lower() == wanted and isinstance(value, (int, float)):
            return scale(value)
    return None


def reference_columns():
    """The two published baselines every new run is read against."""
    columns = []
    base = (f"{RUNS}/../instruct_priority_fourway_20260812/v3_new/"
            "sparse15_summary.json")
    if os.path.isfile(base):
        with open(base) as handle:
            payload = json.load(handle)
        diag = payload["diagonal_scores_rounds_1_to_7"]
        final = payload["final_scores_round_8"]
        columns.append(("v3_new*", [(diag[i] if i < 7 else None, final[i])
                                    for i in range(8)]))
    # slora_pre_released, from trace/EXPERIMENTS.md; no run directory survives.
    released_diag = [58.10, 65.32, 58.93, 62.06, 93.25, 62.96, 69.54, None]
    released_final = [52.50, 63.31, 48.04, 57.11, 91.30, 64.20, 68.62, 41.64]
    columns.append(("slora_rel*", list(zip(released_diag, released_final))))
    return columns


def print_results():
    columns = reference_columns()
    for slug, version, _note, label in V3:
        cells = [(v3_cell(slug, version, i + 1, t) if i < 7 else None,
                  v3_cell(slug, version, 8, t)) for i, t in enumerate(TASKS)]
        if any(a is not None or b is not None for a, b in cells):
            columns.append((label, cells))
    slora = [(slora_cell(i + 1, t) if i < 7 else None, slora_cell(8, t))
             for i, t in enumerate(TASKS)]
    if any(a is not None or b is not None for a, b in slora):
        columns.append(("slora_new", slora))

    # The two published references head the first block only.  Repeating them
    # in every block cost two columns each time and the numbers never change,
    # so they read fine as the top rows of the table.
    references, runs = columns[:2], columns[2:]
    print("\n  RESULTS — final score (delta from acquisition)"
          "   * = published reference   > = acquisition only")
    # Fit as many columns as the terminal allows instead of a hard six.  The
    # old stride also had no block for the final one or two runs, so a newly
    # finished run could sit at 15/15 and still be missing from the table.
    # STATUS_COLS overrides when there is no tty width to read.
    width = shutil.get_terminal_size((100, 24)).columns
    per_block = int(os.environ.get("STATUS_COLS") or max(4, (width - 17) // 13))
    blocks, taken, first = [], 0, True
    while taken < len(runs) or first:
        take = max(per_block - len(references), 1) if first else per_block
        blocks.append((references if first else []) + runs[taken:taken + take])
        taken += take
        first = False

    for block in blocks:
        header = f"  {'task':<13}" + "".join(f"{name[:12]:>13}" for name, _ in block)
        print()
        print(header)
        print("  " + "-" * (len(header) - 2))
        for index, task in enumerate(TASKS):
            row = f"  {task:<13}"
            for _, cells in block:
                acq, final = cells[index]
                if final is not None and acq is not None:
                    row += f"{final:>8.1f}{final - acq:>+5.1f}"
                elif final is not None:
                    row += f"{final:>8.1f}     "
                elif acq is not None:
                    row += f"{acq:>7.1f}>     "
                else:
                    row += f"{'·':>13}"
            print(row)
        print("  " + "-" * (len(header) - 2))
        for label, kind in (("AA", "aa"), ("Forgetting", "fm")):
            row = f"  {label:<13}"
            for _, cells in block:
                finals = [f for _, f in cells]
                drops = [a - f for a, f in cells[:7]
                         if a is not None and f is not None]
                if kind == "aa":
                    # A partial mean is not an AA; the easy cells finish first,
                    # so it would read high and mean nothing.
                    row += (f"{sum(finals) / 8:>13.2f}"
                            if all(f is not None for f in finals) else f"{'·':>13}")
                else:
                    row += (f"{sum(drops) / 7:>13.2f}" if len(drops) == 7
                            else f"{'·':>13}")
            print(row)


def main():
    print(f"\n  v3 replay series — {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # Which lane, if any, is driving each run.  Only used for the GPU label --
    # the live progress itself comes from the run's own train.log below, so a
    # run started by hand outside any lane still shows its position.
    running = {}
    for label, log in (("0-3", f"{LANES}/train_lane0.log"),
                       ("4-7", f"{LANES}/train_lane_extra.log"),
                       ("abl", f"{RUNS}/ablation_lanes/lane.log")):
        unit = lane_current(log)
        if unit:
            running[unit] = label

    # One row per run instead of a TRAINING block and an EVALUATION block:
    # ten runs across two sections did not fit on a screen, and the two numbers
    # people compare -- rounds trained, cells scored -- belong side by side.
    print(f"  {'run':<24}{'TRAIN':<14}{'EVAL':<16}live")
    print("  " + "-" * 84)
    rows = list(V3) + [
        ("slora_pre_upstream", SLORA, "SLoRA-Pre reproduction", None),
        ("slora_pre_replay", SLORA, "SLoRA-Pre + v3 replay (compute control)",
         None),
    ]
    for slug, version, note, _label in rows:
        if version is SLORA:
            root = f"{RUNS}/{slug}/llama31/pre"
            done = sum(1 for i in range(1, 9)
                       if os.path.exists(f"{root}/order{i}/max.safetensors"))
            cells = len(glob.glob(f"{root}/evaluation/order*/*/infer.jsonl"))
            # SLoRA writes one log per task, so the newest names the round in
            # flight; the v3 branch cannot be reused, it reads a single log.
            logs = sorted(glob.glob(f"{root}/order*.train.log"),
                          key=lambda q: os.path.getmtime(q))
            live = None
            if logs and done < 8:
                newest = logs[-1]
                if time.time() - os.path.getmtime(newest) < 300:
                    order = os.path.basename(newest).split(".")[0]
                    info = slora_live_line(newest)
                    gpus = running.get(slug) or slora_gpus() or "?"
                    live = (f"gpu {gpus}  {order} {info['pct']}%"
                            + (f" -{info['rem']}" if info["rem"] else "")
                            if info else f"gpu {gpus}  {order} loading")
        else:
            done = rounds_done(slug, version)
            cells = len([path for path in glob.glob(
                f"{RUNS}/{slug}/{version}_st_top1/evaluation/order*/results-*.json")
                if ".shard" not in os.path.basename(path)])
            live = None
            log = f"{RUNS}/{slug}/{version}_st_top1/train.log"
            try:
                fresh = time.time() - os.path.getmtime(log) < 300
            except OSError:
                fresh = False
            if fresh and done < 8:
                gpus = (running.get(slug)
                        or running.get(slug[3:] if slug.startswith("v3_") else slug)
                        or gpu_hint(log, version) or "?")
                info = live_line(log)
                live = (f"gpu {gpus}  {info['task']} {info['pct']}%"
                        + (f" -{info['rem']}" if info and info["rem"] else "")
                        if info else f"gpu {gpus}  loading")
        name = slug[3:] if slug.startswith("v3_") else slug
        print(f"  {name:<24}{bar(done, 8, 8)} {done}/8   "
              f"{bar(cells, 15, 8)} {cells:>2}/15   {live or ''}")

    if os.environ.get("STATUS_VERBOSE"):
        print("\n  CONFIG")
        for slug, version, note, _ in V3:
            print(f"  {slug[3:] if slug.startswith('v3_') else slug:<24}{note}")

    print_results()

    print("\n  QUEUE")
    try:
        with open(f"{LANES}/queue.txt") as handle:
            units = [l.split("|", 1)[0] for l in handle if l.strip()]
        print("  " + (", ".join(units) if units else "(empty)"))
    except OSError:
        print("  (no queue file)")

    # A chain is work that is committed but invisible to the queue file: it
    # waits on a pid, so nothing on disk says it is coming.
    chains = subprocess.run(
        ["pgrep", "-af", "chain_.*\\.sh"],
        capture_output=True, text=True).stdout.strip()
    if chains:
        print("  CHAINED (waiting, not in queue)")
        for line in chains.splitlines():
            pid, _, cmd = line.partition(" ")
            print(f"    {pid:>8}  {os.path.basename(cmd.split()[-1])}")

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        print("\n  GPU")
        for row in out.splitlines():
            idx, mem, util = [x.strip() for x in row.split(",")]
            print(f"  {idx}  {bar(int(util), 100, 20)} {util:>3}%  "
                  f"{int(mem) / 1024:5.1f} GB")
    except Exception:
        pass

    fails = f"{LANES}/failures.txt"
    if os.path.exists(fails) and os.path.getsize(fails):
        print("\n  FAILURES (historical; fixed units were requeued)")
        with open(fails) as handle:
            for line in handle:
                print("  " + line.rstrip())
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
