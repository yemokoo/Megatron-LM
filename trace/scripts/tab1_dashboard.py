#!/usr/bin/env python3
"""Live terminal view of the TRACE Table-1 pipeline.

  python scripts/tab1_dashboard.py            # refresh every 10s
  python scripts/tab1_dashboard.py --once     # print one frame and exit
  python scripts/tab1_dashboard.py -n 30      # refresh every 30s

State comes from the filesystem (which checkpoints exist) rather than from
parsing logs for completion, so a killed and restarted job still reports
correctly. Logs are only read for the progress of whatever is in flight.
"""
import argparse
import ast
import glob
import json
import os
import re
import subprocess
import time
from datetime import datetime

RUNS = "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace"
SLORA = f"{RUNS}/slora_pre_released_gb64_20260912"
SW = f"{RUNS}/tab1_sweep_20260912"
FIXED = f"{RUNS}/tab1_fixed_20260913"
EVALQ = f"{RUNS}/tab1_eval_20260913"

TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
BAR_W = 22
TQDM = re.compile(r"(\d+)/(\d+) \[[\d:]+<([\d:?]+)")
DENOISE = re.compile(r"Denoising shard (\d)/(\d):\s*(\d+)%")

C = {"g": "\033[32m", "y": "\033[33m", "d": "\033[90m", "b": "\033[1m",
     "c": "\033[36m", "r": "\033[31m", "0": "\033[0m"}


def tail(path, nbytes=65536):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - nbytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def last_progress(path):
    """(done, total, eta) from the most recent tqdm line, or None."""
    text = tail(path).replace("\r", "\n")
    hit = None
    for line in text.split("\n"):
        found = TQDM.search(line)
        if found:
            hit = found
    if hit is None:
        return None
    return int(hit.group(1)), int(hit.group(2)), hit.group(3)


def denoise_pct(path):
    text = tail(path).replace("\r", "\n")
    hit = None
    for line in text.split("\n"):
        found = DENOISE.search(line)
        if found:
            hit = found
    return int(hit.group(3)) if hit else None


PHASE = re.compile(r"^(" + "|".join(TASKS) + r") \[[a-z0-9_-]+[^\]]*\]")


RESUMED = re.compile(r"continuing from task (\d+)")


def head(path, nbytes=65536):
    try:
        with open(path, "rb") as handle:
            return handle.read(nbytes).decode("utf-8", "replace")
    except OSError:
        return ""


def resumed_from(path):
    """Tasks a resumed cell inherited; their checkpoints live in the old run.

    Read from the HEAD: the banner is printed once at startup, and a training
    log grows past any tail window within minutes.
    """
    hit = RESUMED.findall(head(path))
    return int(hit[-1]) if hit else 0


def current_task(path):
    """Task named by the newest phase banner, so a resumed cell reads right."""
    text = tail(path).replace("\r", "\n")
    hit = None
    for line in text.split("\n"):
        found = PHASE.match(line.strip())
        if found:
            hit = found.group(1)
    return hit


def bar(done, total, width=BAR_W, colour="g"):
    total = max(total, 1)
    filled = int(width * min(done, total) / total)
    body = "█" * filled + "░" * (width - filled)
    return f"{C[colour]}{body}{C['0']}"


def row(label, done, total, note="", state="run"):
    colour = {"done": "g", "run": "y", "wait": "d"}[state]
    tag = {"done": "done", "run": "    ", "wait": "wait"}[state]
    dim = C["d"] if state == "wait" else ""
    reset = C["0"] if state == "wait" else ""
    return (f"  {dim}{label:<20}{reset} {bar(done, total, colour=colour)} "
            f"{done:>2}/{total:<2} {C['d']}{tag}{C['0']} {note}")


# ---------------------------------------------------------------- collectors

def slora_train(base=f"{SLORA}/llama31/pre", label="SLoRA-Pre"):
    done = sum(os.path.isfile(f"{base}/order{i}/max.safetensors")
               for i in range(1, 9))
    note, state = "", "done" if done == 8 else "run"
    if done < 8:
        cur = done + 1
        log = f"{base}/order{cur}.train.log"
        task = TASKS[cur - 1]
        pct = denoise_pct(log)
        prog = last_progress(log)
        if pct is not None and prog and prog[0] == prog[1]:
            note = f"order{cur} {task}  denoise {pct}%"
        elif prog:
            note = (f"order{cur} {task}  train "
                    f"{100 * prog[0] // max(prog[1], 1)}%  eta {prog[2]}")
        else:
            note = f"order{cur} {task}  starting"
    return row(label, done, 8, note, state)


def latest_log(patterns):
    """Most recently modified file matching any of several globs, or None.

    A killed-and-relaunched job logs to a new file (log, log.retry, log.retry2,
    ...) rather than overwriting the old one, so picking by name alone can
    point at a stale, no-longer-updating log from a prior attempt. Several
    exact patterns (not one loose wildcard) avoid e.g. "moelpr_g0*.log"
    matching "moelpr_g0.1*.log" as well.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    matches = [m for p in patterns for m in glob.glob(p)]
    return max(matches, key=os.path.getmtime) if matches else None


def tab1_train(label, run_dir, log_patterns):
    root = f"{run_dir}"
    done = sum(os.path.isfile(f"{root}/{i}/tab1_meta.json") for i in range(8))
    log = latest_log(log_patterns)
    if log and os.path.isfile(log):
        # A resumed cell writes only the rounds it runs; the inherited ones sit
        # in the previous run directory, so counting this one alone under-reports.
        done = max(done, resumed_from(log))
    if done == 8:
        return row(label, 8, 8, "", "done")
    if not log:
        return row(label, 0, 8, "", "wait")
    prog = last_progress(log)
    task = current_task(log) or TASKS[min(done, 7)]
    note = task
    if prog:
        note += f"  {100 * prog[0] // max(prog[1], 1)}%  eta {prog[2]}"
    return row(label, done, 8, note, "run")


# ------------------------------------------------------------------ AA / FM
#
# AA = mean of the 8 final-round (order8) primary scores.
# FM = mean over the 7 non-final tasks of (diagonal score - final score),
#      i.e. "forgetting" already divided by 7 (T-1 = 7 tasks).
# Both evaluators (evaluate_Ours_LoRA_MoE.py for tab1 rows, eval_trace.py's
# printed dict for SLoRA) use the same metric-per-task convention and the
# same 0-100 vs 0-1 scaling, so one primary_scalar() covers both sources.

PRIMARY_METRIC = {
    "C-STANCE": "accuracy", "FOMC": "accuracy", "ScienceQA": "accuracy",
    "NumGLUE-cm": "accuracy", "NumGLUE-ds": "accuracy",
    "Py150": "similarity", "MeetingBank": "rouge-L", "20Minuten": "sari",
}
PRIMARY_SCALE = {t: (1.0 if t in ("Py150", "20Minuten") else 100.0)
                 for t in PRIMARY_METRIC}


def primary_scalar(task, result):
    if not isinstance(result, dict):
        return None
    key = PRIMARY_METRIC.get(task)
    if not key:
        return None
    val = None
    for k, v in result.items():
        if k.lower() == key.lower():  # eval_trace.py prints "rouge-l" (lowercase)
            val = v
            break
    if isinstance(val, (int, float)):
        return val * PRIMARY_SCALE.get(task, 1.0)
    return None


def tab1_score(run_dir, order, task):
    path = f"{run_dir}/evaluation/order{order}/{task}.summary.json"
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return primary_scalar(task, data.get(task) if isinstance(data, dict) else None)


SLORA_LOG_LINE = re.compile(r"^In ([^:]+):\s*(\{.*\})\s*$")


def make_slora_score(eval_base=f"{SLORA}/llama31/pre/evaluation"):
    def score(order, task):
        path = f"{eval_base}/order{order}/{task}/eval.log"
        hit = None
        for line in tail(path, nbytes=8192).split("\n"):
            found = SLORA_LOG_LINE.match(line.strip())
            if found and found.group(1) == task:
                hit = found.group(2)
        if hit is None:
            return None
        try:
            result = ast.literal_eval(hit)
        except (ValueError, SyntaxError):
            return None
        return primary_scalar(task, result)
    return score


slora_score = make_slora_score()


def aa_fm(score_fn):
    """score_fn(order, task) -> float|None. Returns (aa, fm), either may be None."""
    final = {t: score_fn(8, t) for t in TASKS}
    if any(v is None for v in final.values()):
        return None, None
    aa = sum(final.values()) / len(TASKS)
    terms = []
    for i, t in enumerate(TASKS[:-1], start=1):
        diag = score_fn(i, t)
        if diag is None:
            return aa, None
        terms.append(diag - final[t])
    fm = sum(terms) / len(terms) if terms else None
    return aa, fm


def aa_fm_note(score_fn):
    aa, fm = aa_fm(score_fn)
    if aa is None:
        return ""
    note = f"AA {aa:5.2f}"
    if fm is not None:
        note += f"  FM {fm:5.2f}"
    return note


def slora_eval(eval_base=f"{SLORA}/llama31/pre/evaluation",
               label="SLoRA sparse-15", score_fn=None):
    if not os.path.isdir(eval_base):
        return row(label, 0, 15, "", "wait")
    done = sum(1 for r, _d, f in os.walk(eval_base) if "infer.jsonl" in f)
    fn = score_fn or slora_score
    note = aa_fm_note(fn) if done >= 15 else ""
    return row(label, done, 15, note,
               "done" if done >= 15 else "run")


def tab1_eval(label, run_dir):
    base = f"{run_dir}/evaluation"
    if not os.path.isdir(base):
        return row(label, 0, 15, "", "wait")
    done = 0
    for root, _dirs, files in os.walk(base):
        done += sum(1 for f in files
                    if f.endswith(".summary.json") and ".shard" not in f)
    note = aa_fm_note(lambda o, t: tab1_score(run_dir, o, t)) if done >= 15 else ""
    return row(label, done, 15, note, "done" if done >= 15 else "run")


def gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return "  gpu: n/a"
    cells = []
    for line in out.strip().split("\n"):
        idx, mem, util = (x.strip() for x in line.split(","))
        gb = int(mem) // 1024
        colour = C["d"] if gb < 2 else (C["g"] if int(util) > 50 else C["y"])
        cells.append(f"{colour}{idx}:{gb:>2}G/{util:>3}%{C['0']}")
    return "  " + "  ".join(cells)


def _log_patterns(base, name):
    """[name.log, name.retry.log, name.retry2.log, ...] as exact patterns.

    Not a single "name*.log" wildcard: with names like moelpr_g0 vs
    moelpr_g0.1, that would also match the other method's log.
    """
    return [f"{base}/{name}.log"] + [f"{base}/{name}.retry*.log"]


# A killed-and-relaunched job logs to a new file rather than overwriting the
# old one, so tab1_train() picks whichever matching file was modified most
# recently instead of always the first attempt's now-dead log.
SWEEPS = [("lifelong kd0.2", f"{SW}/lifelong_kd0.2_mb8", _log_patterns(f"{SW}/logs", "lifelong_kd0.2")),
          ("lifelong kd1.0", f"{SW}/lifelong_kd1.0_mb8", _log_patterns(f"{SW}/logs", "lifelong_kd1.0")),
          ("lifelong kd3.0", f"{SW}/lifelong_kd3.0_mb8", _log_patterns(f"{SW}/logs", "lifelong_kd3.0")),
          ("moelpr g0", f"{SW}/moelpr_g0_mb16", _log_patterns(f"{SW}/logs", "moelpr_g0")),
          ("moelpr g0.1", f"{SW}/moelpr_g0.1_mb16", _log_patterns(f"{SW}/logs", "moelpr_g0.1")),
          ("moelpr g1.0", f"{SW}/moelpr_g1.0_mb16", _log_patterns(f"{SW}/logs", "moelpr_g1.0")),
          ("olora", f"{FIXED}/olora", _log_patterns(f"{FIXED}/logs", "olora")),
          ("ewc", f"{FIXED}/ewc", _log_patterns(f"{FIXED}/logs", "ewc")),
          ("seq_lora", f"{FIXED}/seq_lora", _log_patterns(f"{FIXED}/logs", "seq_lora"))]

# ------------------------------------------------------------ Table 3 queue
#
# tab3_20260913/queue.sh runs Table-3 (wiki-tuned HP transfer) plus the
# O-LoRA lambda sweep. One job (tab3_slora_r512) is SLoRA-shaped (per-order
# train_trace.sh logs, denoise stages); the rest are tab1-shaped (same
# checkpoint/eval layout as the Table-1 SWEEPS above), so they reuse
# tab1_train/tab1_eval with a per-job score_fn closed over that job's own
# run_dir.
TAB3 = f"{RUNS}/tab3_20260913"
TAB3_SLORA_BASE = f"{TAB3}/tab3_slora_r512/llama31/pre"


def _tab3_log(name):
    # queue.sh's run_job() always writes exactly "<name>.train.log" (it
    # never retries into a new file), unlike the Table-1 SWEEPS convention.
    return [f"{TAB3}/logs/{name}.train.log"]


TAB3_TAB1_JOBS = [
    ("tab3 lifelong_kd1.5", f"{TAB3}/tab3_lifelong_kd1.5",
     _tab3_log("tab3_lifelong_kd1.5")),
    ("tab3 moelpr g0.01", f"{TAB3}/tab3_moelpr_g0.01",
     _tab3_log("tab3_moelpr_g0.01")),
    ("olora l1.05 l2-0", f"{TAB3}/olora_l1-0.05_l2-0",
     _tab3_log("olora_l1-0.05_l2-0")),
    ("olora l1.5 l2-.1", f"{TAB3}/olora_l1-0.5_l2-0.1",
     _tab3_log("olora_l1-0.5_l2-0.1")),
    ("olora l1-5 l2-0", f"{TAB3}/olora_l1-5_l2-0",
     _tab3_log("olora_l1-5_l2-0")),
    ("olora l1.05 l2.1", f"{TAB3}/olora_l1-0.05_l2-0.1",
     _tab3_log("olora_l1-0.05_l2-0.1")),
    ("olora l1-5 l2.1", f"{TAB3}/olora_l1-5_l2-0.1",
     _tab3_log("olora_l1-5_l2-0.1")),
    ("tab3 ewc l1.8e6", f"{TAB3}/tab3_ewc_l1.8e6",
     _tab3_log("tab3_ewc_l1.8e6")),
]


# LPR label = every older task mapped to the whole old-expert group, vs the
# task-label moelpr g0.1 row above (same gamma, same HPs).
LPR_OLD = "/data2/seonghyeonnoh/LLM-continual-learning-runs/lpr_oldlabel_20260915"
LPR_OLD_JOBS = [
    ("moelpr g0.1 old-lbl", f"{LPR_OLD}/trace/moelpr_g0.1_oldlabel",
     [f"{LPR_OLD}/logs/trace_train.log"]),
]


def frame():
    out = [f"{C['b']}TRACE Table-1 pipeline{C['0']}"
           f"{' ' * 26}{datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    out.append(f"{C['c']}TRAINING{C['0']}")
    out.append(slora_train())
    for label, run_dir, log in SWEEPS:
        out.append(tab1_train(label, run_dir, log))
    out.append("")
    out.append(f"{C['c']}EVALUATION{C['0']}  (sparse-15)")
    out.append(slora_eval())
    for label, run_dir, _log in SWEEPS:
        out.append(tab1_eval(label, run_dir))
    out.append("")
    out.append(f"{C['c']}TAB3 TRAINING{C['0']}  (Table 3 + O-LoRA sweep)")
    out.append(slora_train(TAB3_SLORA_BASE, "tab3 slora r512"))
    for label, run_dir, log in TAB3_TAB1_JOBS:
        out.append(tab1_train(label, run_dir, log))
    out.append("")
    out.append(f"{C['c']}TAB3 EVALUATION{C['0']}  (sparse-15)")
    out.append(slora_eval(f"{TAB3_SLORA_BASE}/evaluation", "tab3 slora r512",
                           make_slora_score(f"{TAB3_SLORA_BASE}/evaluation")))
    for label, run_dir, _log in TAB3_TAB1_JOBS:
        out.append(tab1_eval(label, run_dir))
    out.append("")
    out.append(f"{C['c']}LPR OLD-LABEL{C['0']}  (train / sparse-15 eval)")
    for label, run_dir, log in LPR_OLD_JOBS:
        out.append(tab1_train(label, run_dir, log))
        out.append(tab1_eval(label, run_dir))
    out.append(tab1_eval("  vs task-lbl g0.1", f"{SW}/moelpr_g0.1_mb16"))
    out.append("")
    out.append(f"{C['c']}GPU{C['0']}")
    out.append(gpus())
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        print(frame())
        return
    try:
        print("\033[?25l", end="")          # hide cursor
        while True:
            print("\033[H\033[J" + frame(), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="")          # restore cursor
if __name__ == "__main__":
    main()
