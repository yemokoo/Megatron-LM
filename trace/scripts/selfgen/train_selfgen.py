#!/usr/bin/env python3
"""Run the stock v3 trainer, but serve replay/KD from self-generated records.

Both `_build_v2_kd_loader` and `_build_v2_replay_loader` reach their data through
`_ensure_fixed_task_subset`, so replacing that one method switches BOTH the
KD-init pass and the 1-phase joint replay onto generated data.

IMPORTANT (bug fixed 2026-08-31): the method is defined THREE times --
`Ours_LoRA_MoE` (base) and `Ours_LoRA_MoE_V2_New`, which the concrete trainer
`Ours_LoRA_MoE_V3_New` inherits.  Patching only the base class is silently
shadowed by the V2_New override: the run then builds the REAL replay memory and
writes fixed_replay_memory/*.json, with no error.  The patch is therefore
applied to the concrete trainer class and verified afterwards.

The trainer keeps an exact `v2_new_persistent_samples_per_task` (500) memory per
task, so the generated records are truncated to that count -- the same replay
budget the `lm` baseline had, only self-generated.

env:
  SELFGEN_ROOT   directory holding <task>/records.jsonl for every past task
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMPL = REPO / "implementations" / "llmcl_benchmark"
sys.path.insert(0, str(IMPL))
os.chdir(IMPL)

from torch.utils.data import Dataset                                  # noqa: E402
from utils.data.data_collator import SLoRATraceDataCollator           # noqa: E402
from model.Ours_LoRA_MoE import (Ours_LoRA_MoE, Ours_LoRA_MoE_V2,     # noqa: E402
                                 Ours_LoRA_MoE_V2_New)
from model.Ours_LoRA_MoE_V3 import (Ours_LoRA_MoE_V3,                 # noqa: E402
                                    Ours_LoRA_MoE_V3_New)

TARGETS = (Ours_LoRA_MoE, Ours_LoRA_MoE_V2, Ours_LoRA_MoE_V2_New,
           Ours_LoRA_MoE_V3, Ours_LoRA_MoE_V3_New)
# What the concrete trainer resolved to before patching -- the fallback used for the
# task being trained right now, whose generated records only exist after this round.
ORIGINAL = Ours_LoRA_MoE_V3_New._ensure_fixed_task_subset


class GeneratedRecords(Dataset):
    """Generated replay records, served in whichever shape the collator wants.

    With --tokenized_train_cache_dir the trainer uses
    PreTokenizedSLoRATraceDataCollator, which reads `input_ids`; without it the
    SLoRA collator reads `prompt`/`answer`.  Items carry all three, encoded
    exactly as SLoRATraceDataCollator._encode does (label_scope "full", the
    scope the training path uses), so either collator behaves identically to a
    real TRACE record.
    """

    def __init__(self, path, limit=None, tokenizer=None, max_length=1024):
        records = [json.loads(line) for line in Path(path).open()]
        for r in records:
            if "prompt" not in r or "answer" not in r:
                raise ValueError(f"malformed generated record in {path}: {list(r)}")
        if limit is not None:
            if len(records) < limit:
                raise ValueError(
                    f"{path} has {len(records)} generated records, need {limit}")
            records = records[:limit]
        if not records:
            raise ValueError(f"no generated records in {path}")
        self.records = records
        self._encoder = (
            SLoRATraceDataCollator(tokenizer, max_length=max_length, label_scope="full")
            if tokenizer is not None else None)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        if self._encoder is None:
            return record
        input_ids, _ = self._encoder._encode(record)
        return {"prompt": record["prompt"], "answer": record["answer"],
                "input_ids": input_ids}


def install_selfgen_replay(root):
    root = Path(root)

    def _ensure_fixed_task_subset(self, task):
        cache = self._fixed_task_subsets
        if task in cache:
            return cache[task]
        path = root / task / "records.jsonl"
        if not path.is_file():
            # `_memory_task_names` is strictly the PAST tasks, so the only task that can
            # legitimately have no generated records yet is the one being trained now --
            # the trainer still asks for its subset to persist as memory for later rounds.
            # Any other task missing means the chain is out of order: fail rather than
            # silently fall back to real data.
            current = os.environ.get("SELFGEN_CURRENT_TASK", "")
            if task != current:
                raise FileNotFoundError(
                    f"self-generated replay for {task!r} is missing: {path} "
                    f"(current task is {current!r}); refusing to fall back to real data")
            if getattr(self.args, "global_rank", 0) in (0, -1):
                print(f"[selfgen] {task} is the task being trained; its memory comes from "
                      f"the real set this round and is regenerated afterwards", flush=True)
            return ORIGINAL(self, task)
        limit = None
        getter = getattr(self, "_v2_new_persistent_samples_per_task", None)
        if callable(getter):
            limit = int(getter())
        max_length = int(getattr(self.args, "max_train_len", 0) or (
            getattr(self.args, "max_prompt_len", 512)
            + getattr(self.args, "max_ans_len", 512)))
        dataset = GeneratedRecords(path, limit=limit,
                                   tokenizer=self.tokenizer, max_length=max_length)
        cache[task] = dataset
        # `_build_fixed_memory_loader` reads this to write provenance strings
        # ("<task>:<index>") into the replay plan.  For generated data the record's
        # position in the file is its identity.
        self._fixed_task_subset_indices[task] = list(range(len(dataset)))
        if getattr(self.args, "global_rank", 0) in (0, -1):
            print(f"[selfgen] replay/KD for {task}: {len(dataset)} generated records "
                  f"from {path}", flush=True)
        return dataset

    for cls in TARGETS:
        cls._ensure_fixed_task_subset = _ensure_fixed_task_subset
    # fail loudly rather than silently falling back to the real replay memory
    resolved = Ours_LoRA_MoE_V3_New._ensure_fixed_task_subset
    if getattr(resolved, "__qualname__", "") != _ensure_fixed_task_subset.__qualname__:
        raise RuntimeError(
            "self-generated replay patch did not take effect on the concrete trainer; "
            f"resolved to {resolved}")
    print(f"[selfgen] replay source patched on {len(TARGETS)} classes -> {root}", flush=True)


def main():
    root = os.environ.get("SELFGEN_ROOT", "").strip()
    if not root:
        raise SystemExit("SELFGEN_ROOT must point at the generated-replay directory")
    install_selfgen_replay(root)
    import runpy
    sys.argv[0] = str(IMPL / "training" / "main_Ours_LoRA_MoE.py")
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
