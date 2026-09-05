"""Feed SLoRA the *same* replay records the v3 runs use.

The point of the control is to remove replay volume as an explanation for
v3's margin, so this does not sample its own memory.  It reads the artifacts
a v3 run already wrote:

  fixed_replay_memory/task_<i>_<name>.json
      500 stored indices into <data_root>/<name>/train.json, plus the seed
      and a sha256 over the index list.
  replay_plans/round_<k-1>_v2_new_replay_active_memory.json
      how many exposures each old task contributes at round k-1, summing to
      one primary epoch (5,000).

Round k-1 is the pool in front of training task k, so task_id 2 reads round 1
and task_id 8 reads round 7.  Task 1 has no past and gets no replay.

Only the destination of the gradient differs from v3: there the replay loss
reaches routers alone, here it reaches the whole new LoRA.
"""
import glob
import hashlib
import json
import os
import random

from datasets import Dataset, concatenate_datasets

from src.trace_data import to_sft_messages

PLAN_PHASE = "v2_new_replay_active_memory"


def _load_plan(v3_run_dir, task_id):
    path = os.path.join(
        v3_run_dir, "replay_plans", f"round_{task_id - 1}_{PLAN_PHASE}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no v3 replay plan for task_id {task_id}: {path}")
    with open(path) as handle:
        return json.load(handle)


def _load_memory_indices(v3_run_dir, task_name):
    matches = sorted(glob.glob(os.path.join(
        v3_run_dir, "fixed_replay_memory", f"task_*_{task_name}.json")))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one stored memory file for {task_name}, "
            f"found {matches}")
    with open(matches[0]) as handle:
        memory = json.load(handle)
    indices = list(memory["indices"])
    # The v3 artifact carries a digest of its own index list.  Recomputing it
    # here is what makes "the same records" a checked claim rather than a
    # naming convention.
    digest = hashlib.sha256(
        ",".join(map(str, indices)).encode("utf-8")).hexdigest()
    if digest != memory["indices_sha256"]:
        raise ValueError(
            f"{matches[0]} index digest mismatch: {digest} != "
            f"{memory['indices_sha256']}")
    if len(indices) != memory["unique_samples"]:
        raise ValueError(
            f"{matches[0]} holds {len(indices)} indices but claims "
            f"{memory['unique_samples']} unique samples")
    return indices, int(memory["resolved_seed"])


def _expose(indices, seed, count):
    """Walk a shuffled pool, wrapping when exposures exceed stored records.

    v3 draws 5,000 exposures per epoch from 500 stored records per task, so
    repetition is part of the setting being reproduced, not an accident.
    """
    order = list(indices)
    random.Random(seed).shuffle(order)
    return [order[position % len(order)] for position in range(count)]


def build_replay_dataset(v3_run_dir, data_root, task_id, num_proc=8):
    plan = _load_plan(v3_run_dir, task_id)
    shards = []
    manifest = []
    for entry in plan["tasks"]:
        name = entry["task"]
        exposures = int(entry["exposure_samples"])
        if exposures < 1:
            continue
        indices, seed = _load_memory_indices(v3_run_dir, name)
        source = Dataset.from_json(os.path.join(data_root, name, "train.json"))
        picked = _expose(indices, seed, exposures)
        out_of_range = [i for i in picked if i >= len(source)]
        if out_of_range:
            raise IndexError(
                f"{name}: stored indices exceed train.json length "
                f"{len(source)} (e.g. {out_of_range[0]})")
        shards.append(source.select(picked))
        manifest.append({
            "task": name, "unique": len(indices), "exposures": exposures})

    if not shards:
        raise ValueError(f"replay plan for task_id {task_id} is empty")

    merged = concatenate_datasets(shards)
    expected = int(plan["planned_exposure_samples"])
    if len(merged) != expected:
        raise ValueError(
            f"built {len(merged)} replay exposures but the v3 plan says "
            f"{expected}")
    merged = merged.map(
        to_sft_messages, remove_columns=merged.column_names,
        num_proc=num_proc, desc="Converting replay records to messages")
    return merged, {
        "round": plan["round"],
        "distribution": plan["distribution"],
        "planned_exposure_samples": expected,
        "tasks": manifest,
    }
