#!/usr/bin/env python3
"""Stock V3 TRACE trainer (KD-init + 1-phase joint replay, shared-router
QKVO+FFN experts) with a residual (no-op) expert present from the first task.

Run exactly like training/main_Ours_LoRA_MoE.py (same argv); this file patches
the trainer in-process and then executes main_Ours_LoRA_MoE.py via runpy.

Residual expert
  * every SharedExpertRouter gets a separate 1-row ``residual_router``
    (zero init).  Its logit is appended AFTER the active expert prefix, so real
    expert ids, ``limit_v3_experts`` (KD teacher = old experts + residual) and
    old-row freezing keep their stock meaning;
  * a token that selects the residual gets no expert delta in that layer
    (the RoutingContext reports only real experts, so no projection touches it);
  * gradient paths: the full-softmax denominator (tokens on real experts) and,
    in training, a zero-valued second-choice dispatch for residual tokens
    (weight p - p.detach() on their best real expert) -- see residual_expert.py.

Ramp (first task only, env RESIDUAL_RAMP_FRAC, default 0.2)
  the residual probability is scaled by alpha (logit + log alpha), alpha rising
  linearly 0 -> 1 over that fraction of the first task's optimizer updates.
  Later tasks start from a trained residual + KD-init, so no ramp there.

KD-init: the residual row is an "old" row and is frozen together with the old
expert rows while the new expert is distilled.

env
  RESIDUAL_RAMP_FRAC   fraction of task-0 updates for the ramp (default 0.2)
  RESIDUAL_SECOND_CHOICE  1 (default) / 0
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMPL = REPO / "implementations" / "llmcl_benchmark"
sys.path.insert(0, str(IMPL))

import torch                                   # noqa: E402
import torch.nn.functional as F                # noqa: E402

# torch>=2.6 defaults torch.load to weights_only=True, which rejects the pickled
# PromptDataset that upstream TRACE create_prompt_dataset() caches and reloads.
_stock_torch_load = torch.load


def _torch_load_weights_only_default_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _stock_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_default_false

from model import Ours_LoRA_MoE_V3 as V3       # noqa: E402
import mass_reservoir as MR                    # noqa: E402  (inert unless MRES_ENABLE=1)

RAMP_FRAC = float(os.environ.get("RESIDUAL_RAMP_FRAC", "0.2"))
SECOND_CHOICE = os.environ.get("RESIDUAL_SECOND_CHOICE", "1") == "1"
RESIDUAL_KEY = ".shared_expert_router.residual_router."


# ----------------------------------------------------------------- router
_stock_router_forward = V3.SharedExpertRouter.forward


def attach_residual(model):
    for layer in V3.shared_router_layers(model):
        router = layer.shared_expert_router
        if getattr(router, "residual_router", None) is not None:
            continue
        anchor = router._placement_anchor
        ref = router.router.weight if router.router is not None else anchor
        router.residual_router = torch.nn.Linear(
            router.hidden_size, 1, bias=False, device=ref.device, dtype=ref.dtype)
        with torch.no_grad():
            router.residual_router.weight.zero_()
        router._residual_log_alpha = 0.0


def residual_forward(self, hidden_states):
    if getattr(self, "residual_router", None) is None:
        return _stock_router_forward(self, hidden_states)
    if MR.enabled(self):
        return MR.router_forward(self, hidden_states, V3)
    real = self._active_count()
    flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    expert_logits = self.router(flat_hidden)[..., :real]
    res_logit = self.residual_router(flat_hidden).to(expert_logits.dtype)
    if self._residual_log_alpha != 0.0:
        res_logit = res_logit + self._residual_log_alpha
    logits = torch.cat([expert_logits, res_logit], dim=-1)
    logit_bias = getattr(self, "_logit_bias", None)
    if logit_bias is not None:                 # task-conditioning bias (bos_token/train_bos_token.py)
        positions = getattr(self, "_logit_bias_positions", None)   # callable -> flat bool mask or None
        if positions is None:
            logits = logits + logit_bias.to(logits.dtype)
        else:
            pos = positions()                  # None: no flagged position in this forward -> bias off
            if pos is not None:
                # torch.where, not multiplication: a -inf mask entry times a 0 position weight is NaN
                pos = pos.to(logits.device).bool()[:, None]
                logits = torch.where(pos, logits + logit_bias.to(logits.dtype), logits)
    total = real + 1
    k = min(self.top_k, total)
    topk_logits, topk_indices = logits.topk(k, dim=-1)
    full_probs = F.softmax(logits, dim=-1, dtype=torch.float)
    selected = full_probs.gather(-1, topk_indices)
    if self.routing_weight_mode == "straight_through_topk":
        weights = F.softmax(topk_logits, dim=-1).detach() + selected - selected.detach()
    elif self.routing_weight_mode == "full_softmax":
        weights = selected
    else:
        weights = F.softmax(topk_logits, dim=-1)

    is_residual = topk_indices == real
    probe_indices = topk_indices
    if SECOND_CHOICE and self.training and torch.is_grad_enabled() and bool(is_residual.any()):
        best_real = expert_logits.argmax(-1, keepdim=True).expand_as(topk_indices)
        real_prob = full_probs.gather(-1, best_real)
        topk_indices = torch.where(is_residual, best_real, topk_indices)
        weights = torch.where(is_residual, real_prob - real_prob.detach(), weights)

    valid_mask = self._router_token_mask
    if valid_mask is not None:
        valid_mask = valid_mask.reshape(-1).to(device=logits.device, dtype=torch.bool)
        if valid_mask.numel() != logits.shape[0]:
            raise ValueError("router mask/token count mismatch")
    self._last_moe_loss = (
        self._router_loss(logits, topk_indices, full_probs, valid_mask)
        if self.training and not self._suppress_router_loss else None)
    if getattr(self, "_residual_stats", None) is not None:
        rec = is_residual if valid_mask is None else is_residual[valid_mask]
        self._residual_stats[0] += int(rec.sum())
        self._residual_stats[1] += int(rec.numel())
    # probe stats count top1 >= old_count as "new expert"; report residual as -1
    self._last_probe_indices = (torch.where(is_residual, -1, probe_indices).detach()
                                if self._capture_probe_routing else None)
    return V3.RoutingContext(expert_indices=topk_indices, expert_weights=weights,
                             valid_token_mask=valid_mask, num_experts=real)


V3.SharedExpertRouter.forward = residual_forward


# --------------------------------------------------- freezing / snapshots
_stock_freeze_routers = V3.freeze_v3_routers


def freeze_routers_with_residual(model, trainable):
    _stock_freeze_routers(model, trainable)
    for layer in V3.shared_router_layers(model):
        rr = getattr(layer.shared_expert_router, "residual_router", None)
        if rr is not None:
            rr.weight.requires_grad = trainable
        mr = getattr(layer.shared_expert_router, "mres_reservoir", None)
        if mr is not None:
            mr.weight.requires_grad = trainable


V3.freeze_v3_routers = freeze_routers_with_residual

Trainer = V3.Ours_LoRA_MoE_V3_New

if os.environ.get("GRAD_CKPT") == "0":
    # The trainer forces checkpointing whenever replay is present; this is a pure
    # memory/compute trade (identical weights), so allow turning it off on large GPUs.
    _stock_set_grad_ckpt = Trainer._set_grad_ckpt
    Trainer._set_grad_ckpt = lambda self, enable: _stock_set_grad_ckpt(self, False)
_stock_snapshot = V3.Ours_LoRA_MoE_V3._snapshot_old_router_rows
_stock_restore = V3.Ours_LoRA_MoE_V3._freeze_old_router_row_update


def snapshot_with_residual(model, old_expert_count):
    rows = _stock_snapshot(model, old_expert_count)
    res = [layer.shared_expert_router.residual_router.weight.detach().clone()
           for layer in V3.shared_router_layers(model)]
    return {"rows": rows, "residual": res}


def restore_with_residual(model, snapshots, old_expert_count):
    _stock_restore(model, snapshots["rows"], old_expert_count)
    for layer, snap in zip(V3.shared_router_layers(model), snapshots["residual"]):
        w = layer.shared_expert_router.residual_router.weight
        if w.grad is not None:
            w.grad.zero_()
        with torch.no_grad():
            w.copy_(snap)


for cls in (V3.Ours_LoRA_MoE_V3, Trainer):
    cls._snapshot_old_router_rows = staticmethod(snapshot_with_residual)
    cls._freeze_old_router_row_update = staticmethod(restore_with_residual)

if RESIDUAL_KEY not in V3.Ours_LoRA_MoE_V3.save_key_substrings:
    V3.Ours_LoRA_MoE_V3.save_key_substrings.append(RESIDUAL_KEY)
if MR.RESERVOIR_KEY not in V3.Ours_LoRA_MoE_V3.save_key_substrings:
    V3.Ours_LoRA_MoE_V3.save_key_substrings.append(MR.RESERVOIR_KEY)


# ------------------------------------------------------ task hook + ramp
_stock_train_one_task = Trainer.train_one_task
_stock_reinit = Trainer._reinit_engine


def _set_log_alpha(model, value):
    for layer in V3.shared_router_layers(model):
        layer.shared_expert_router._residual_log_alpha = value


_stock_add_experts = V3.add_v3_experts


def add_experts_and_residual(model, count):
    # every growth (fresh task or --resume_checkpoint rebuild) keeps a residual
    _stock_add_experts(model, count)
    attach_residual(model)
    MR.attach(model)                # no-op unless the mass reservoir is enabled


V3.add_v3_experts = add_experts_and_residual


def train_one_task(self, task, i_task, epochs):
    self._residual_ramp_pending = (i_task == 0 and RAMP_FRAC > 0)
    try:
        return _stock_train_one_task(self, task, i_task, epochs)
    finally:
        self._residual_ramp_pending = False
        _set_log_alpha(self.raw_model, 0.0)
        if MR.ACTIVE.enabled:
            _finish_mres_task(self, task, i_task)


def _finish_mres_task(self, task, i_task):
    """Task end: alpha = 1 (the reservoir is ready for the next expansion); dump stats."""
    MR.set_alpha(self.raw_model, 1.0)
    MR.set_flag(self.raw_model, "_mres_force_new", False)
    MR.set_flag(self.raw_model, "_mres_primary", False)
    MR.disable_stats(self.raw_model)
    summary = MR.summarize(self.raw_model)
    if self.args.global_rank in (0, -1):
        MR.write_json(os.path.join(self.args.output_dir, f"mres_stats_round{i_task}.json"),
                      {"task": task, "round": i_task, "alpha_end": 1.0, "passes": summary,
                       "rows": MR.row_report(self.raw_model),
                       "schedule": getattr(self, "_mres_schedule", None)})
        for pass_name, rows in summary.items():
            viol = sum(r["violation"] for r in rows) / max(1, len(rows))
            newr = sum(r["new_rate"] for r in rows) / max(1, len(rows))
            print(f"[mres] round {i_task} {pass_name}: margin violation {viol:.4f} "
                  f"new-expert top1 {newr:.4f}", flush=True)


def reinit_engine(self, num_training_steps, learning_rate=None):
    _stock_reinit(self, num_training_steps, learning_rate)
    if getattr(self, "_residual_ramp_pending", False):
        # task 0 has no KD-init, so its first engine is the primary phase
        self._residual_ramp_pending = False
        ramp_updates = max(1, int(round(num_training_steps * RAMP_FRAC)))
        state = {"step": 0, "n": ramp_updates}
        _set_log_alpha(self.raw_model, math.log(1e-9))
        sched_step = self.lr_scheduler.step

        def step_with_ramp(*a, **k):
            out = sched_step(*a, **k)
            state["step"] += 1
            alpha = min(1.0, state["step"] / state["n"])
            _set_log_alpha(self.raw_model, math.log(max(alpha, 1e-9)))
            return out
        self.lr_scheduler.step = step_with_ramp
        if self.args.global_rank == 0:
            print(f"[residual] task-0 ramp over {ramp_updates}/{num_training_steps} updates",
                  flush=True)
    if MR.ACTIVE.enabled:
        _install_mres_schedule(self, num_training_steps)


MRES_LOG_EVERY = int(os.environ.get("MRES_LOG_EVERY", "50"))


def _install_mres_schedule(self, total):
    """alpha: 0 through the warm-up, then linear to 1 at the last update; warm-up flags."""
    cfg = MR.ACTIVE
    warm = MR.warmup_updates(total, cfg.warmup_frac)
    state = {"step": 0, "total": total, "warmup": warm}
    self._mres_schedule = state
    MR.set_alpha(self.raw_model, 0.0)
    sched_step, opt_step = self.lr_scheduler.step, self.optimizer.step

    def in_warmup():
        return state["step"] < state["warmup"]

    def optimizer_step(*a, **k):
        if in_warmup() and cfg.warmup_freeze_router:
            # warm-up: only the new expert (LoRA + its router row) moves; old rows, skip and
            # r_res keep their values (fresh AdamW moments are zero, so a zero grad is no update)
            for router in MR.routers(self.raw_model):
                n = router.router.weight.shape[0]
                if router.router.weight.grad is not None:
                    router.router.weight.grad[:n - 1].zero_()
                for p in (router.residual_router.weight, router.mres_reservoir.weight):
                    if p.grad is not None:
                        p.grad.zero_()
        return opt_step(*a, **k)

    def scheduler_step(*a, **k):
        out = sched_step(*a, **k)
        state["step"] += 1
        MR.set_alpha(self.raw_model, MR.alpha_schedule(state["step"], total, warm))
        if self.args.global_rank in (0, -1) and MRES_LOG_EVERY and state["step"] % MRES_LOG_EVERY == 0:
            rows = MR.summarize(self.raw_model, reset=False).get("router_ft", [])
            if rows:
                mean = lambda key: sum(r[key] for r in rows) / len(rows)
                print(f"[mres] step {state['step']}/{total} alpha {MR.alpha_of(self.raw_model):.3f} "
                      f"warmup {in_warmup()} router-FT: violation {mean('violation'):.4f} "
                      f"z_res {mean('z_res'):.3f} thr {mean('threshold'):.3f} "
                      f"res_mass {mean('res_mass'):.4f} new {mean('new_rate'):.4f} "
                      f"skip {mean('skip_rate'):.4f}", flush=True)
        return out

    self.optimizer.step = optimizer_step
    self.lr_scheduler.step = scheduler_step
    self._mres_in_warmup = in_warmup
    if self.args.global_rank in (0, -1):
        print(f"[mres] schedule: {total} updates, warm-up {warm} (forced new expert, alpha 0), "
              f"refill alpha 0->1 over the remaining {total - warm}", flush=True)


Trainer.train_one_task = train_one_task
Trainer._reinit_engine = reinit_engine


# ------------------------------------------------------------ metadata
_stock_save_meta = V3.save_v3_meta


def save_meta_with_residual(model, output_dir, args, trainer=None):
    _stock_save_meta(model, output_dir, args, trainer=trainer)
    import json
    path = os.path.join(output_dir, V3.V3_META_NAME)
    with open(path) as f:
        meta = json.load(f)
    meta["residual_expert"] = {"count": 1, "init": "zeros", "ramp_frac_task0": RAMP_FRAC,
                               "second_choice": SECOND_CHOICE,
                               "state_key": RESIDUAL_KEY.strip(".") + ".weight"}
    if MR.ACTIVE.enabled:
        meta[MR.META_KEY] = MR.ACTIVE.to_meta(alpha_end=MR.alpha_of(model))
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


V3.save_v3_meta = save_meta_with_residual


def main():
    import runpy
    os.chdir(IMPL)
    script = str(IMPL / "training" / "main_Ours_LoRA_MoE.py")
    sys.argv[0] = script
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
