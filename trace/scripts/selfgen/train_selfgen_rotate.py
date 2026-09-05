#!/usr/bin/env python3
"""Self-generated replay with a ROTATING exposure window.

Problem this solves.  The V2-new memory stream draws, once per round, a fixed
``active_cap / n_old`` slice of each old task's memory and then repeats that same
slice for every primary epoch.  With the published 500-record memory the slice was
never smaller than the memory (5000/7 = 714 >= 500), so every record was seen.  A
larger memory (2500) is therefore silently truncated: at round 7 only 714 of the
2500 records ever reach the router.

Change.  Two patches, both on top of train_selfgen's generated-replay source:

1. ``_v2_new_active_memory`` uses the *wide* unique cap for KD as well as replay,
   so each task's active pool is the whole persisted memory rather than an
   ``active_cap``-sized prefix shared across tasks.
2. The per-pass hooks rotate each task's exposure window: pass p reads
   ``perm[p*E : (p+1)*E]`` (mod U) of a fixed permutation of the task's U records,
   where E is the unchanged per-pass exposure count.  Volume per pass, loader
   length, sampler order and the pass-count contract are untouched; only *which*
   records fill the same-sized stream changes, so the union over the round's
   epochs covers the whole memory (needs ceil(U/E) <= epochs, true for every
   TRACE round at U=2500).

env:
  SELFGEN_ROOT / SELFGEN_CURRENT_TASK   as in train_selfgen.py
  ROTATE_LOG=1                          print per-pass cumulative unique coverage
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_selfgen as base                                       # noqa: E402
from model.Ours_LoRA_MoE import (Ours_LoRA_MoE, Ours_LoRA_MoE_V2,  # noqa: E402
                                 Ours_LoRA_MoE_V2_New, RepeatedSubsetDataset)
from model.Ours_LoRA_MoE_V3 import (Ours_LoRA_MoE_V3,              # noqa: E402
                                    Ours_LoRA_MoE_V3_New)

TARGETS = (Ours_LoRA_MoE, Ours_LoRA_MoE_V2, Ours_LoRA_MoE_V2_New,
           Ours_LoRA_MoE_V3, Ours_LoRA_MoE_V3_New)
LOG = os.environ.get("ROTATE_LOG", "") == "1"


# ---------------------------------------------------------------- wide pool
_orig_active_memory = Ours_LoRA_MoE_V2_New._v2_new_active_memory


def _wide_active_memory(self, i_task, unique_cap=None):
    """KD and replay both draw from the full persisted memory."""
    if unique_cap is None:
        unique_cap = self._v2_new_active_unique_cap()
    return _orig_active_memory(self, i_task, unique_cap)


# ------------------------------------------------------------- rotation
def _streams(dataloader):
    dataset = getattr(dataloader, "dataset", None)
    parts = getattr(dataset, "datasets", None)
    if parts is None:
        return []
    return [part for part in parts if isinstance(part, RepeatedSubsetDataset)]


def _install_rotation(dataloader):
    """Attach a fixed per-task permutation and the original window size."""
    state = getattr(dataloader, "_rotation_state", None)
    if state is not None:
        return state
    stream = getattr(dataloader, "_lora_moe_memory_stream", {}) or {}
    seed = int(stream.get("sampler_base_seed", 0) or 0) or 1013
    state = []
    for offset, part in enumerate(_streams(dataloader)):
        unique = len(part.subset)
        window = len(part.exposure_indices)
        generator = torch.Generator().manual_seed(seed + 7919 * offset)
        permutation = torch.randperm(unique, generator=generator).tolist()
        state.append({"part": part, "unique": unique, "window": window,
                      "perm": permutation, "seen": set()})
    dataloader._rotation_state = state
    return state


def _rotate(dataloader, pass_index, role):
    state = _install_rotation(dataloader)
    if not state:
        return
    for entry in state:
        unique, window, permutation = entry["unique"], entry["window"], entry["perm"]
        start = (pass_index * window) % unique
        indices = [permutation[(start + i) % unique] for i in range(window)]
        entry["part"].exposure_indices = indices
        entry["seen"].update(indices)
    if LOG and int(getattr(dataloader, "_rotation_rank", 0) or 0) == 0:
        covered = ", ".join(
            f"{len(e['seen'])}/{e['unique']}(win {e['window']})" for e in state)
        print(f"[rotate] {role} pass {pass_index}: cumulative unique {covered}",
              flush=True)


def _patch_pass_hooks():
    orig_replay = Ours_LoRA_MoE_V2_New._set_v2_replay_memory_sampler_pass
    orig_kd = Ours_LoRA_MoE_V2_New._set_v2_kd_memory_sampler_pass

    def replay_pass(self, dataloader, pass_index):
        dataloader._rotation_rank = int(getattr(self.args, "global_rank", 0) or 0)
        _rotate(dataloader, pass_index, "replay")
        return orig_replay(self, dataloader, pass_index)

    def kd_pass(self, dataloader, pass_index):
        dataloader._rotation_rank = int(getattr(self.args, "global_rank", 0) or 0)
        _rotate(dataloader, pass_index, "kd")
        return orig_kd(self, dataloader, pass_index)

    for cls in TARGETS:
        if hasattr(cls, "_set_v2_replay_memory_sampler_pass"):
            cls._set_v2_replay_memory_sampler_pass = replay_pass
        if hasattr(cls, "_set_v2_kd_memory_sampler_pass"):
            cls._set_v2_kd_memory_sampler_pass = kd_pass


def install():
    for cls in TARGETS:
        if hasattr(cls, "_v2_new_active_memory"):
            cls._v2_new_active_memory = _wide_active_memory
    _patch_pass_hooks()
    print("[rotate] wide unique pool + rotating exposure window installed",
          flush=True)


if __name__ == "__main__":
    install()
    base.main()
