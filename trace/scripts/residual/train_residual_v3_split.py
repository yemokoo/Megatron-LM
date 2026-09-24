#!/usr/bin/env python3
"""Residual expert from task 0, split-gradient variant.

Builds on train_residual_v3.py (residual router row from the first task,
second-choice dispatch, residual-aware freezing/saving) and changes how the
residual row is trained so it cannot become a shortcut for the current task:

  * new expert init: LoRA A random / B zero (the stock LoRAPair init), so a
    freshly added expert is a no-op when selected -- the effect expansion
    KD-init was providing.  KD-init must therefore be off
    (--ablation_kd_init off; enforced);
  * new router row: copied from the residual row at every growth, so the new
    expert and the residual start with identical routing probability;
  * primary (new-task) forward/backward: the residual is MASKED OUT of routing
    (its logit is -inf, so real experts alone compete for every token) and its
    row receives no gradient; the new expert and every other router row train;
  * router-FT forward/backward, in the same optimizer update, router rows only
    and ALL rows including the residual: the stock replay stream (past-task
    records, 500/task, cycled) widened with a pseudo-task of backbone BoS
    generations (500 records), plus a slice of the current primary batch
    forwarded a second time -- one replay source's share (local batch //
    (past tasks + BoS)), each record weighted like a replay record.  The
    pseudo-task exists from task 0, so task 0 also runs the joint loop.

Every non-residual router row therefore accumulates two gradients per update
(primary + router-FT); the residual row only the router-FT one.

Run exactly like scripts/selfgen/train_selfgen.py (same argv; SELFGEN_ROOT /
SELFGEN_CURRENT_TASK select self-generated replay, otherwise the real
fixed_replay_memory records are used).  Requires the pretokenized train cache
(--tokenized_train_cache_dir) because BoS records are served as token ids.

env
  RESIDUAL_BOS_JSONL      backbone BoS records
                          (default scripts/residual/assets/backbone_bos_500.jsonl)
  RESIDUAL_RAMP_FRAC      task-0 residual ramp; default 0 here (the frozen
                          residual row replaces the ramp)
  RESIDUAL_SECOND_CHOICE  1 (default) / 0
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMPL = REPO / "implementations" / "llmcl_benchmark"
for path in (IMPL, HERE, REPO / "scripts" / "selfgen"):
    sys.path.insert(0, str(path))
os.environ.setdefault("RESIDUAL_RAMP_FRAC", "0")

import torch                                   # noqa: E402
from torch.utils.data import Dataset           # noqa: E402

from contextlib import contextmanager         # noqa: E402
import train_residual_v3 as TR                 # noqa: E402  installs the residual patches
import mass_reservoir as MR                    # noqa: E402
from model import Ours_LoRA_MoE_V3 as V3       # noqa: E402

BOS_TASK = "__backbone_bos__"
BOS_JSONL = os.environ.get("RESIDUAL_BOS_JSONL") or str(
    HERE / "assets" / "backbone_bos_500.jsonl")
NEW_EXPERT_INIT = os.environ.get(
    "RESIDUAL_NEW_EXPERT_INIT", "copy_router_zero_b").strip()
if NEW_EXPERT_INIT not in {"copy_router_zero_b", "random_router_zero_b"}:
    raise ValueError(
        "RESIDUAL_NEW_EXPERT_INIT must be copy_router_zero_b or "
        f"random_router_zero_b, got {NEW_EXPERT_INIT!r}")
Trainer = V3.Ours_LoRA_MoE_V3_New


def _rank0(args):
    return getattr(args, "global_rank", 0) in (0, -1)


# ------------------------------------------------ growth: copy residual row
_prev_add_experts = V3.add_v3_experts


def add_experts_from_residual(model, count):
    layers = V3.shared_router_layers(model)
    old = layers[0].num_experts if layers else 0
    if MR.ACTIVE.enabled:
        _add_experts_mass_reservoir(model, count, old)
        return
    _prev_add_experts(model, count)          # grows + attaches the residual
    # random_router_zero_b keeps the stock init (random router row, A random,
    # B zero), i.e. the original Ours expansion; only copy_router_zero_b
    # overrides the new row.
    if NEW_EXPERT_INIT != "copy_router_zero_b":
        return
    for layer in layers:
        router = layer.shared_expert_router
        with torch.no_grad():
            router.router.weight[old:old + count].copy_(
                router.residual_router.weight.expand(count, -1))


def _add_experts_mass_reservoir(model, count, old):
    """New task: new rows := clone(r_res) (task 0 keeps the stock random row), alpha 1 -> 0,
    with an old-task probe before/after to verify the forward is unchanged.  A rebuild
    (resume / checkpoint load) only grows the structure; the state dict supplies the rows."""
    exp = MR.EXPANDING
    probe = exp.get("probe") if (exp["on"] and old > 0) else None
    before = MR.run_probe(model, probe) if probe is not None else None
    _prev_add_experts(model, count)          # grows + residual + reservoir row
    if not exp["on"]:
        return
    if old > 0:
        MR.copy_reservoir_into_rows(model, old, count)
    MR.set_alpha(model, 0.0)
    if before is None:
        return
    report = MR.compare_probes(before, MR.run_probe(model, probe))
    if exp.get("rank0"):
        MR.write_json(exp["path"], report)
        sm = report["summary"]
        print(f"[mres] expansion check round {exp['round']}: denominator rel diff max "
              f"{sm['denominator_rel_diff_max']:.2e}, Top-K change (margin-ok tokens) "
              f"{sm['topk_change_rate_margin_ok_max']:.4f}, Top-K change (all) "
              f"{sm['topk_change_rate_all_max']:.4f}, margin-satisfied min "
              f"{sm['margin_satisfied_rate_min']:.4f}, hidden rel L2 max {sm['hidden_rel_l2_max']:.2e}, "
              f"logits max abs diff {sm['logits_max_abs_diff']:.2e}", flush=True)


V3.add_v3_experts = add_experts_from_residual


# ------------------------------------------- split gradient (joint loop hooks)
# _begin_gradient_memory_batch runs right before the primary forward and
# _after_primary_backward right after its backward: between them the residual
# logit is -inf (never selected, zero gradient); replay/extra forwards, the
# epoch probe and evaluation see the residual normally.
_prev_begin_batch = Trainer._begin_gradient_memory_batch


def _begin_gradient_memory_batch(self):
    TR._set_log_alpha(self.raw_model, float("-inf"))
    if MR.ACTIVE.enabled:
        warm = bool(getattr(self, "_mres_in_warmup", lambda: False)())
        MR.set_flag(self.raw_model, "_mres_primary", True)       # reservoir out of the denominator
        MR.set_flag(self.raw_model, "_mres_force_new", warm)     # warm-up: new expert only
        MR.enable_stats(self.raw_model, "primary_warmup" if warm else "primary")
    return _prev_begin_batch(self)


def _after_primary_backward(self):
    TR._set_log_alpha(self.raw_model, 0.0)
    if MR.ACTIVE.enabled:
        MR.set_flag(self.raw_model, "_mres_primary", False)
        MR.set_flag(self.raw_model, "_mres_force_new", False)
        MR.disable_stats(self.raw_model)
    for layer in V3.shared_router_layers(self.raw_model):
        residual = getattr(layer.shared_expert_router, "residual_router", None)
        if residual is not None and residual.weight.grad is not None:
            residual.weight.grad.zero_()


def _extra_router_replay_batches(self, primary):
    """Re-forward a slice of the current primary batch router-only.

    The slice is one replay source's share: local batch // (past tasks + BoS),
    so the current task enters router-FT with the same weight as each replay
    source (each record also carries the replay per-sample loss scale).
    """
    per_rank = max(1, int(primary["input_ids"].shape[0]) // int(self._router_ft_sources))
    return _CurrentSlices(self, ({key: value[:per_rank] if torch.is_tensor(value) else value
                                  for key, value in primary.items()},))


class _CurrentSlices(tuple):
    """The current-task router-FT slice(s); iterating marks the trainer so the mass-reservoir
    margin can tell the current slice from replay (MRES_MARGIN_CURRENT)."""

    def __new__(cls, trainer, items):
        obj = super().__new__(cls, items)
        obj._trainer = trainer
        return obj

    def __iter__(self):
        self._trainer._mres_current_slice = True
        try:
            yield from super().__iter__()
        finally:
            self._trainer._mres_current_slice = False


Trainer._begin_gradient_memory_batch = _begin_gradient_memory_batch

# ------------------------------- mass reservoir: router-correction margin + stats
_prev_router_only_replay = Trainer._router_only_replay
_stock_per_sample_losses = Trainer._per_sample_causal_lm_losses


@contextmanager
def _router_only_replay_mres(self):
    with _prev_router_only_replay(self):
        if not MR.ACTIVE.enabled:
            yield
            return
        MR.set_flag(self.raw_model, "_mres_margin", True)
        MR.enable_stats(self.raw_model, "router_ft")
        try:
            yield
        finally:
            MR.set_flag(self.raw_model, "_mres_margin", False)
            MR.disable_stats(self.raw_model)
            for router in MR.routers(self.raw_model):
                router._mres_margin_buf.clear()


def _per_sample_losses_mres(self, logits, labels, ignore_index=-100):
    losses = _stock_per_sample_losses(logits, labels, ignore_index)
    if not MR.ACTIVE.enabled:
        return losses
    margin = MR.pop_margin_per_sample(self.raw_model, logits.shape[0])
    if margin is None:
        return losses
    if getattr(self, "_mres_current_slice", False) and not MR.ACTIVE.margin_current:
        return losses
    return losses + MR.ACTIVE.margin_weight * margin.to(losses.dtype)


Trainer._router_only_replay = _router_only_replay_mres
Trainer._per_sample_causal_lm_losses = _per_sample_losses_mres
Trainer._after_primary_backward = _after_primary_backward
Trainer._extra_router_replay_batches = _extra_router_replay_batches


# ------------------------------------------------ backbone BoS pseudo-task
class BackboneBoSRecords(Dataset):
    """Raw backbone generations as full-label token ids (PreTokenized collator)."""

    def __init__(self, path, tokenizer, max_length, limit):
        rows = [json.loads(line) for line in Path(path).open()][:limit]
        if len(rows) < limit:
            raise ValueError(f"{path} has {len(rows)} BoS records, need {limit}")
        doc_start = tokenizer.bos_token_id
        if doc_start is None:
            doc_start = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.items = []
        for index, row in enumerate(rows):
            body = tokenizer(row["text"], add_special_tokens=False,
                             truncation=True, max_length=max_length - 2)["input_ids"]
            ids = ([doc_start] + body + [tokenizer.eos_token_id])[:max_length]
            self.items.append({"prompt": f"{BOS_TASK}:{index}",
                               "answer": row["text"], "input_ids": ids})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def install_bos_memory():
    prev_ensure = Trainer._ensure_fixed_task_subset
    prev_names = Trainer._memory_task_names

    def _ensure_fixed_task_subset(self, task):
        if task != BOS_TASK:
            return prev_ensure(self, task)
        cache = self._fixed_task_subsets
        if task in cache:
            return cache[task]
        if not getattr(self.args, "use_pretokenized_train_cache", False):
            raise RuntimeError(
                "residual-split training needs --tokenized_train_cache_dir "
                "(BoS records are served as token ids)")
        limit = int(self._v2_new_persistent_samples_per_task())
        max_length = int(getattr(self.args, "max_train_len", 0) or (
            getattr(self.args, "max_prompt_len", 512)
            + getattr(self.args, "max_ans_len", 512)))
        dataset = BackboneBoSRecords(BOS_JSONL, self.tokenizer, max_length, limit)
        cache[task] = dataset
        self._fixed_task_subset_indices[task] = list(range(len(dataset)))
        if _rank0(self.args):
            print(f"[residual-split] router-FT backbone BoS: {len(dataset)} records "
                  f"from {BOS_JSONL}", flush=True)
        return dataset

    def _memory_task_names(self, i_task):
        return list(prev_names(self, i_task)) + [BOS_TASK]

    Trainer._ensure_fixed_task_subset = _ensure_fixed_task_subset
    Trainer._memory_task_names = _memory_task_names


# ---------------------------------------------------- guards + metadata
_prev_train_one_task = Trainer.train_one_task


def train_one_task(self, task, i_task, epochs):
    args = self.args
    kd_init = getattr(args, "ablation_kd_init", "on")
    if NEW_EXPERT_INIT == "copy_router_zero_b" and kd_init != "off":
        raise ValueError("residual-split training replaces KD-init with the "
                         "zero-B expert init: pass --ablation_kd_init off")
    if getattr(args, "ablation_phase_mode", "1phase") != "1phase":
        raise ValueError("residual-split training is a 1-phase joint recipe")
    self._router_ft_sources = len(self._memory_task_names(i_task))
    if _rank0(args):
        local = int(args.batch_by_task[task])
        print(f"[residual-split] round {i_task}: router-FT sources={self._router_ft_sources} "
              f"(past tasks + BoS); primary reuse per rank = {max(1, local // self._router_ft_sources)}"
              f"/{local}", flush=True)
    if not MR.ACTIVE.enabled:
        return _prev_train_one_task(self, task, i_task, epochs)
    MR.EXPANDING.update(on=True, round=i_task, rank0=_rank0(args), probe=None,
                        path=os.path.join(args.output_dir, f"mres_expansion_check_round{i_task}.json"))
    if i_task > 0:
        MR.EXPANDING["probe"] = _old_task_probe(self, i_task)
    try:
        return _prev_train_one_task(self, task, i_task, epochs)
    finally:
        MR.EXPANDING.update(on=False, probe=None)


def _old_task_probe(self, i_task):
    """First batch of the previous task's loader, on device; the global RNG is restored so
    the probe does not shift the new expert's initialisation or the data order."""
    prev = list(self.train_task_list)[i_task - 1]
    cpu, cuda = torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        batch = next(iter(self.train_task_list[prev]))
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
    device = next(self.raw_model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()
            if torch.is_tensor(v) and k in ("input_ids", "attention_mask", "labels")}


Trainer.train_one_task = train_one_task

_prev_save_meta = V3.save_v3_meta


def save_meta_split(model, output_dir, args, trainer=None):
    _prev_save_meta(model, output_dir, args, trainer=trainer)
    path = os.path.join(output_dir, V3.V3_META_NAME)
    with open(path) as handle:
        meta = json.load(handle)
    meta["residual_expert"].update({
        "variant": "split_gradient",
        "new_row_init": (
            "copy_of_residual_row" if NEW_EXPERT_INIT == "copy_router_zero_b"
            else "pytorch_linear_random"),
        "new_expert_init": "lora_A_random_B_zero",
        "primary_routing": "residual_masked_out",
        "primary_gradient_rows": "all_but_residual",
        "router_ft_sources": ["past_task_replay", "backbone_bos", "primary_reuse"],
        "primary_reuse_share": "local_batch // (past_tasks + 1)",
        "backbone_bos_jsonl": BOS_JSONL,
    })
    with open(path, "w") as handle:
        json.dump(meta, handle, indent=2)


V3.save_v3_meta = save_meta_split


def main():
    root = os.environ.get("SELFGEN_ROOT", "").strip()
    if root:
        import train_selfgen
        train_selfgen.install_selfgen_replay(root)
    install_bos_memory()      # after selfgen so the BoS wrapper delegates to it
    import runpy
    os.chdir(IMPL)
    script = str(IMPL / "training" / "main_Ours_LoRA_MoE.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
