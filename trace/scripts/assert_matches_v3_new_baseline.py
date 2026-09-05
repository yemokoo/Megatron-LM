#!/usr/bin/env python3
"""Fail unless a run's train.command.txt matches the v3_new baseline.

The v3_new_replay40 run silently trained on base Llama-3.1-8B instead of the
Instruct checkpoint every published baseline used, and it only surfaced five
hours later as bad accuracy.  Nothing compared the two command lines.  This
does, and it is meant to run right after a launch writes train.command.txt.

Anything not in ALLOWED_TO_DIFFER is a hard failure, so a new run has to
declare exactly which knobs it is turning.

    python3 assert_matches_v3_new_baseline.py <run_dir> [--allow flag ...]
"""
import argparse
import os
import shlex
import sys

BASELINE = ("/data2/seonghyeonnoh/LLM-continual-learning-runs/"
            "instruct_priority_fourway_20260812/v3_new/train.command.txt")

# Paths, ports and process topology legitimately differ between runs; the
# effective batch is checked separately below because it must not.
ALLOWED_TO_DIFFER = {
    "--output_dir", "--data_output_path", "--master_port", "--nproc_per_node",
    "--training_version", "--gradient_accumulation_steps",
    "--per_device_train_batch_size",
}


def parse(path):
    tokens = shlex.split(open(path).read())
    pairs, index = {}, 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            if "=" in token:
                flag, value = token.split("=", 1)
                pairs[flag] = value
                index += 1
                continue
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                pairs[token] = tokens[index + 1]
                index += 2
                continue
            pairs[token] = ""
            index += 1
            continue
        index += 1
    return pairs


# Flags that did not exist when the baseline ran.  They are inert at these
# values, so a run only has to justify them when it sets something else.
NEW_FLAG_DEFAULTS = {
    "--v2_joint_replay_objective": "lm",
    "--v2_hidden_mse_loss_coeff": "1.0",
    "--v2_replay_forward_batch_size": "8",
    "--v2_new_active_unique_cap": "0",
}


def same_value(flag, before, after):
    if before == after:
        return True
    if before is None and NEW_FLAG_DEFAULTS.get(flag) == after:
        return True
    # data_path and friends are symlinked into the repo; compare the targets
    # so a path spelled two ways is not reported as a difference.
    if before and after and (before.startswith("/") and after.startswith("/")):
        try:
            return os.path.realpath(before) == os.path.realpath(after)
        except OSError:
            return False
    return False


def integer(pairs, flag, default):
    try:
        return int(pairs.get(flag, default))
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--allow", action="append", default=[],
                        help="flag that this run intentionally changes")
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()

    command = os.path.join(args.run_dir, "train.command.txt")
    if not os.path.isfile(command):
        print(f"FAIL: no train.command.txt under {args.run_dir}", file=sys.stderr)
        return 2

    base = parse(args.baseline)
    run = parse(command)
    allowed = ALLOWED_TO_DIFFER | set(args.allow)

    problems = []
    for flag in sorted(set(base) | set(run)):
        if flag in allowed:
            continue
        before, after = base.get(flag), run.get(flag)
        if not same_value(flag, before, after):
            problems.append(f"  {flag}: baseline={before!r} run={after!r}")

    # Effective batch = gpus x micro-batch x accumulation.  Splitting the same
    # batch differently across GPUs is fine; changing its size is not.
    def effective(pairs):
        gpus = integer(pairs, "--nproc_per_node", 0)
        micro = integer(pairs, "--per_device_train_batch_size", 0)
        accum = integer(pairs, "--gradient_accumulation_steps", 0)
        return None if None in (gpus, micro, accum) else gpus * micro * accum

    base_batch, run_batch = effective(base), effective(run)
    if base_batch != run_batch:
        problems.append(
            f"  effective batch: baseline={base_batch} run={run_batch}")

    intended = sorted(set(args.allow))
    if intended:
        print("intentionally changed:")
        for flag in intended:
            print(f"  {flag}: baseline={base.get(flag)!r} run={run.get(flag)!r}")
    if problems:
        print("\nFAIL: unintended differences from the v3_new baseline:",
              file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"\nOK: {args.run_dir} matches the v3_new baseline "
          f"(effective batch {run_batch})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
