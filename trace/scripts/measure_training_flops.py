#!/usr/bin/env python3
"""Measure each v3 variant's training FLOPs without re-running the training.

The counter is disabled in production because FlopCounterMode intercepts every
aten call.  So instead of paying that on a ten-hour run, this replays each
variant's own train.command.txt -- byte for byte, minus the entry point and the
output directory -- for a handful of optimizer updates per phase with the
counter on, and multiplies the measured per-update cost by the update counts the
real run actually recorded in training_workload.json.

Per-phase, not per-round: a round mixes KD-init with the joint primary+replay
loop, and those cost very different amounts per update.

  measure_training_flops.py [--updates N] [--gpus 0,1,2,3] [--runs slug ...]
"""
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "/data2/seonghyeonnoh/LLM-continual-learning-runs/trace"
SCRATCH = f"{RUNS}/flop_probe"
VARIANTS = [
    ("v3_replay1to1", "v3_new_replay1to1"),
    ("v3_hidden_mse_1to1", "v3_new_hidden_mse_1to1"),
    ("v3_hidden_mse_1to1_p5k", "v3_new_hidden_mse_1to1_p5k"),
    ("v3_replay1to1_p5k", "v3_new_replay1to1_p5k"),
    ("v3_replay1to1_recency", "v3_new_replay1to1_recency"),
]


def probe_command(run_dir, out_dir, gpus, port, micro):
    """The measured run's own command, pointed at the probe entry point.

    The micro-batch is shrunk because FlopCounterMode is not free: it wraps
    every aten call in a dispatch mode and holds enough extra state that the
    production batch OOMs under it, which is exactly why training runs with
    --disable_training_flop_counter.  Cost is reported per sample, so a smaller
    probe batch changes nothing about the number that comes out.
    """
    tokens = shlex.split(open(f"{run_dir}/train.command.txt").read())
    tokens[tokens.index("training/main_Ours_LoRA_MoE.py")] = \
        "scripts/flop_probe_entry.py"
    world = len(gpus.split(","))
    tokens = [f"--nproc_per_node={world}" if t.startswith("--nproc_per_node")
              else t for t in tokens]
    tokens = [f"--master_port={port}" if t.startswith("--master_port")
              else t for t in tokens]
    # The counter is what we are here for, and the probe must not touch the
    # measured run's directory.
    tokens = [t for t in tokens if t != "--disable_training_flop_counter"]
    for flag, value in (("--output_dir", out_dir),
                        ("--data_output_path", f"{out_dir}/data_cache"),
                        ("--per_device_train_batch_size", str(micro)),
                        ("--gradient_accumulation_steps", "1")):
        if flag in tokens:
            tokens[tokens.index(flag) + 1] = value
    return tokens, world, micro


def measure(slug, version, gpus, updates, port, micro):
    run_dir = f"{RUNS}/{slug}/{version}_st_top1"
    if not os.path.isfile(f"{run_dir}/train.command.txt"):
        return None
    # Replay exposures are split across ranks, so a probe on fewer GPUs than the
    # training makes one rank absorb the whole replay chunk and OOM in the joint
    # phase.  The probe has to match the world size it is measuring.
    trained_world = int(next(
        token.split("=")[1]
        for token in shlex.split(open(f"{run_dir}/train.command.txt").read())
        if token.startswith("--nproc_per_node")))
    if len(gpus.split(",")) != trained_world:
        print(f"[PROBE] {version} was trained on {trained_world} GPUs; "
              f"--gpus lists {len(gpus.split(','))}. Re-run with that many.",
              file=sys.stderr)
        return None
    out_dir = f"{SCRATCH}/{slug}"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    # V2-new refuses to resume without the authoritative persistent memory, and
    # regenerating it would pick different records than the run being measured.
    if os.path.isdir(f"{run_dir}/fixed_replay_memory"):
        shutil.copytree(f"{run_dir}/fixed_replay_memory",
                        f"{out_dir}/fixed_replay_memory")
    tokens, world, probe_micro = probe_command(
        run_dir, out_dir, gpus, port, micro)
    records = f"{out_dir}/flop_probe.json"
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=gpus,
               FLOP_PROBE_UPDATES=str(updates),
               FLOP_PROBE_OUT=records,
               PYTHONNOUSERSITE="1",
               TOKENIZERS_PARALLELISM="false",
               WANDB_MODE="offline")
    env.setdefault(
        "SLORA_LLAMA31_PATH",
        "/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct")
    # A probe needs the whole card: it loads the same 8B model the training
    # did, so sharing a GPU with a live run just OOMs both.
    for device in gpus.split(","):
        used = subprocess.run(
            ["nvidia-smi", "-i", device, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if used.isdigit() and int(used) > 2000:
            print(f"[PROBE] gpu {device} already holds {used} MiB; "
                  "refusing to share a card", file=sys.stderr)
            return None
    print(f"[PROBE] {version} on gpus {gpus}", flush=True)
    with open(f"{out_dir}/probe.log", "w") as log:
        result = subprocess.run(tokens, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    if not os.path.isfile(records):
        print(f"[PROBE] {version} produced no records (exit {result.returncode});"
              f" see {out_dir}/probe.log", file=sys.stderr)
        return None
    with open(records) as handle:
        probe = json.load(handle)
    return {"world_size": world, "probe_micro": probe_micro,
            "records": probe["phases"], "run_dir": run_dir}


def phase_costs(probe, real_batch):
    """Mean FLOPs per optimizer update of the *measured* run, per phase.

    Records are rank-0 only, so they cover one rank's share of an update:
    probe_micro samples.  Normalising to a per-sample cost and multiplying by
    the measured run's effective batch makes the answer independent of both the
    probe's world size and its batch, which matters because the probe cannot
    use either of the training's values -- the counter's own memory overhead
    forces a smaller batch.
    """
    totals = {}
    for record in probe["records"]:
        if record["optimizer_updates"] < 1:
            continue
        key = record["phase"]
        entry = totals.setdefault(key, {"flops": 0, "updates": 0})
        entry["flops"] += record["local_flops"]
        entry["updates"] += record["optimizer_updates"]
    per_sample = {phase: value["flops"] / value["updates"] / probe["probe_micro"]
                  for phase, value in totals.items()}
    return {phase: cost * real_batch for phase, cost in per_sample.items()}


def real_updates(run_dir):
    """Per-phase optimizer updates the measured run actually ran."""
    with open(f"{run_dir}/training_workload.json") as handle:
        workload = json.load(handle)
    with open(f"{run_dir}/train.command.txt") as handle:
        tokens = shlex.split(handle.read())

    def arg(flag, default):
        return tokens[tokens.index(flag) + 1] if flag in tokens else default

    world = int(next(t.split("=")[1] for t in tokens
                     if t.startswith("--nproc_per_node")))
    micro = int(arg("--per_device_train_batch_size", "8"))
    accum = int(arg("--gradient_accumulation_steps", "1"))
    batch = world * micro * accum
    epochs = [int(x) for x in arg("--num_train_epochs", "5").split(",")]
    cap = int(arg("--v2_new_active_memory_cap", "1000"))
    kd_stream = int(arg("--v2_kd_exposure_samples", "0")) or cap
    kd_epochs_flag = int(arg("--v2_kd_epochs", "0"))
    per_epoch = math.ceil(5000 / batch)
    out = {"_run_v3_primary_epochs": 0, "_run_v2_kd_init": 0,
           "_run_v2_joint_epochs": 0}
    for index, task_epochs in enumerate(epochs):
        if index == 0:
            out["_run_v3_primary_epochs"] += per_epoch * task_epochs
            continue
        out["_run_v2_joint_epochs"] += per_epoch * task_epochs
        kd_epochs = kd_epochs_flag or task_epochs
        out["_run_v2_kd_init"] += math.ceil(kd_stream / batch) * kd_epochs
    seconds = sum(t.get("training_wall_time_seconds", 0)
                  for t in workload.get("tasks", []))
    return out, batch, seconds, workload.get("world_size", world)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--probe-micro-batch", type=int, default=2,
                        help="per-device batch for the probe only; the counter "
                             "needs headroom the training batch does not leave")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--out", default=f"{SCRATCH}/training_flops.json")
    args = parser.parse_args()

    wanted = set(args.runs) if args.runs else None
    expected = [slug for slug, _ in VARIANTS if not wanted or slug in wanted]
    done_marker = args.out + ".done"
    # A stale marker from an earlier invocation would tell a waiting chain the
    # new run is already finished.
    if os.path.exists(done_marker):
        os.remove(done_marker)
    summary, port = {}, 29950
    for slug, version in VARIANTS:
        if wanted and slug not in wanted:
            continue
        probe = measure(slug, version, args.gpus, args.updates, port,
                        args.probe_micro_batch)
        port += 1
        if probe is None:
            continue
        updates, batch, seconds, world = real_updates(probe["run_dir"])
        costs = phase_costs(probe, batch)
        # A phase that recorded no completed update contributes nothing, which
        # would quietly report a fraction of the real cost as the total.  The
        # first probe did exactly that: it OOMed in the joint phase and still
        # printed a "complete" summary covering only primary and KD.
        needed = [phase for phase, count in updates.items()
                  if count > 0 and phase not in costs]
        if needed:
            print(f"[PROBE] {version} measured no cost for {', '.join(needed)};"
                  f" see {SCRATCH}/{slug}/probe.log", file=sys.stderr)
            continue
        total = sum(costs.get(phase, 0.0) * count
                    for phase, count in updates.items())
        summary[slug] = {
            "per_update_flops": costs,
            "real_optimizer_updates": updates,
            "effective_batch": batch,
            "world_size": world,
            "probe_world_size": probe["world_size"],
            "probe_micro_batch": probe["probe_micro"],
            "training_wall_time_seconds": seconds,
            "total_training_flops": total,
        }
        # Written after every run, not once at the end, so a probe that dies
        # part way still leaves the runs it did finish.  That makes the file's
        # existence useless as a completion signal, hence "complete" below and
        # the .done marker.
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"schema_version": 1,
                       "updates_per_phase": args.updates,
                       "method": "measured per-phase cost x recorded updates",
                       "complete": False,
                       "expected_runs": expected,
                       "runs": summary}, handle, indent=1)

    print(f"\n{'run':<26}{'primary':>12}{'KD':>12}{'joint':>12}{'total':>12}")
    print("-" * 74)
    for slug, entry in summary.items():
        costs = entry["per_update_flops"]
        counts = entry["real_optimizer_updates"]
        row = f"{slug:<26}"
        for phase in ("_run_v3_primary_epochs", "_run_v2_kd_init",
                      "_run_v2_joint_epochs"):
            row += f"{costs.get(phase, 0) * counts[phase] / 1e18:>11.2f}E"
        row += f"{entry['total_training_flops'] / 1e18:>11.2f}E"
        print(row)
    complete = all(slug in summary for slug in expected)
    if summary:
        with open(args.out) as handle:
            payload = json.load(handle)
        payload["complete"] = complete
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=1)
    if complete:
        # The signal a waiting chain should key on: the json exists from the
        # first run onward, so only this says every run is measured.
        with open(done_marker, "w") as handle:
            handle.write(f"{len(expected)} runs measured\n")
        print(f"\nwritten to {args.out}  (complete; marker {done_marker})")
    else:
        missing = [slug for slug in expected if slug not in summary]
        print(f"\nwritten to {args.out}  (INCOMPLETE, missing: "
              f"{', '.join(missing)})", file=sys.stderr)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
