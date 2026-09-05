"""Process-local hooks that turn a normal training launch into a FLOP probe.

``install(updates_per_phase)`` caps each V3 training phase at a fixed number of
optimizer updates and wraps each one in its own FlopCounterMode.  Everything
else -- model construction, expert growth, the real dataloaders, the real
gradient scoping -- runs exactly as production does, which is the point: the
per-update cost has to come from the same code path the measured run used, not
from a reconstruction of it.

Per phase rather than per round, because a round mixes KD-init with the joint
primary+replay loop and those cost very different amounts per update.  Dividing
a round total by its update count would smear one into the other.

FlopCounterMode counts what the dispatcher actually executed, so an expert that
top-1 routing never selected contributes nothing -- which is the whole reason
not to compute this from a shape formula.

Nothing here is imported by training.  It lives outside the trainer so a probe
can never change the behaviour of a production run.
"""
import functools
import json
import os

import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:  # pragma: no cover - older PyTorch
    FlopCounterMode = None

PHASES = ("_run_v3_primary_epochs", "_run_v2_kd_init", "_run_v2_joint_epochs")
RECORDS = []


class _PhaseBudgetReached(Exception):
    """Raised inside a phase once its probe budget is spent."""


def _write(path):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    with open(path, "w") as handle:
        json.dump({"schema_version": 1,
                   "updates_per_phase": int(os.environ.get(
                       "FLOP_PROBE_UPDATES", "0")),
                   "phases": RECORDS}, handle, indent=1)


def install(updates_per_phase, out_path=None):
    if updates_per_phase < 1:
        raise ValueError("probe budget must be at least one update")
    if FlopCounterMode is None:
        raise RuntimeError("this PyTorch has no torch.utils.flop_counter")

    from model import Ours_LoRA_MoE as base_module
    from model.Ours_LoRA_MoE_V3 import Ours_LoRA_MoE_V3_New

    out_path = out_path or os.environ.get(
        "FLOP_PROBE_OUT", "/tmp/flop_probe.json")
    original_count = base_module.Ours_LoRA_MoE._count_workload_update

    def counted(self):
        original_count(self)
        left = getattr(self, "_probe_updates_left", None)
        if left is None:
            return
        left -= 1
        self._probe_updates_left = left
        if left <= 0:
            raise _PhaseBudgetReached
    base_module.Ours_LoRA_MoE._count_workload_update = counted

    def cap(name, method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            self._probe_updates_left = updates_per_phase
            counter = FlopCounterMode(
                display=False,
                custom_mapping=base_module._flop_counter_custom_mapping())
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            try:
                with counter:
                    return method(self, *args, **kwargs)
            except _PhaseBudgetReached:
                # A truncated phase is the point; the caller moves on to the
                # next one so a single round yields KD *and* joint costs.
                return None
            finally:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                spent = updates_per_phase - max(
                    0, getattr(self, "_probe_updates_left", 0))
                RECORDS.append({
                    "round": int(getattr(self, "_probe_round", -1)),
                    "phase": name,
                    "optimizer_updates": spent,
                    "local_flops": int(counter.get_total_flops()),
                    "rank": int(os.environ.get("RANK", "0")),
                })
                _write(out_path)
                self._probe_updates_left = None
        return wrapper

    for name in PHASES:
        setattr(Ours_LoRA_MoE_V3_New, name,
                cap(name, getattr(Ours_LoRA_MoE_V3_New, name)))

    original_train = Ours_LoRA_MoE_V3_New.train_one_task

    def tagged(self, task, i_task, epochs):
        self._probe_round = i_task
        return original_train(self, task, i_task, epochs)
    Ours_LoRA_MoE_V3_New.train_one_task = tagged

    # A truncated model would look like a real checkpoint to every other script
    # in the tree, so the probe never writes one.
    Ours_LoRA_MoE_V3_New.save_model = lambda self, i_task: None
    print(f"[FLOP-PROBE] phases capped at {updates_per_phase} updates; "
          f"records -> {out_path}", flush=True)
