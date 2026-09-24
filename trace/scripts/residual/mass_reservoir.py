"""Deterministic mass-transfer reservoir for the expert-expanding V3 LoRA-MoE.

Every shared router carries, besides the physical expert rows and the permanent skip row,
one trainable reservoir row r_res.  The reservoir is NEVER a Top-K candidate: it only adds
alpha * exp(z_res), z_res = r_res.x, to the softmax denominator, reserving the mass the next
expert will take.

Routing (requires routing_weight_mode="full_softmax"):
  candidates = [physical experts, skip]            Top-K is taken over these only
  Z          = sum exp(candidates) + alpha exp(z_res)   (log-space: z_res + log alpha)
  weight_i   = exp(z_i) / Z for the selected i      (full-softmax probability, no renormalisation)

Task cycle
  expansion (task t >= 1): new router row := clone(r_res) (exact, no noise), LoRA A random /
      B = 0, alpha 1 -> 0.  Because z_new == z_res, the denominator is unchanged token by token;
      if z_res < K-th candidate logit the Top-K set is unchanged too, so the forward is identical.
      Task 0 keeps the stock random router row (nothing to preserve yet).
  warm-up (first warmup_frac of the task's optimizer updates): the primary pass routes every
      current-task token to the new expert only (old experts and skip masked); alpha stays 0.
  primary pass: skip and reservoir are excluded from both selection and the denominator.
  router-correction pass (router-only replay: past-task replay, backbone BoS, current slice):
      full pool + reservoir mass; margin loss max(0, z_res - stopgrad(z_th) + delta), z_th the
      K-th largest candidate logit.  The new physical expert is not pushed down.
  refill: after the warm-up alpha rises linearly 0 -> 1, reaching 1 at the last update.

Configuration comes from MRES_* for training and from the checkpoint meta when loading.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

RESERVOIR_KEY = ".shared_expert_router.mres_reservoir."
META_KEY = "mass_reservoir"
NEG_INF = float("-inf")


@dataclass(frozen=True)
class MassReservoirConfig:
    enabled: bool = False
    delta: float = 0.5                 # margin below the K-th candidate logit
    margin_weight: float = 0.1         # lambda for the margin term in the router-correction loss
    warmup_frac: float = 0.05          # share of a task's optimizer updates with forced new-expert routing
    margin_current: bool = True        # also apply the margin on the current-task router-correction slice
    warmup_freeze_router: bool = True  # during warm-up, old rows / skip / reservoir get no update

    def to_meta(self, alpha_end):
        return {**asdict(self), "alpha_end": alpha_end,
                "state_key": RESERVOIR_KEY.strip(".") + ".weight",
                "formulation": "deterministic_mass_transfer: reservoir only in the softmax "
                               "denominator (alpha*exp(z_res)), exact row copy at expansion"}


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def config_from_env():
    if not _flag("MRES_ENABLE", "0"):
        return MassReservoirConfig()
    return MassReservoirConfig(
        enabled=True,
        delta=float(os.environ.get("MRES_DELTA", "0.5")),
        margin_weight=float(os.environ.get("MRES_LAMBDA", "0.1")),
        warmup_frac=float(os.environ.get("MRES_WARMUP_FRAC", "0.05")),
        margin_current=_flag("MRES_MARGIN_CURRENT", "1"),
        warmup_freeze_router=_flag("MRES_WARMUP_FREEZE_ROUTER", "1"))


def config_from_meta(meta):
    block = meta.get(META_KEY)
    if not block or not block.get("enabled"):
        return MassReservoirConfig()
    fields = MassReservoirConfig.__dataclass_fields__
    return MassReservoirConfig(**{k: v for k, v in block.items() if k in fields})


ACTIVE = config_from_env()
# True only while a trainer's train_one_task is growing the model for a NEW task; rebuilds on
# resume / checkpoint loading keep the loaded rows and alpha = 1.
EXPANDING = {"on": False, "probe": None}


def set_active(cfg):
    global ACTIVE
    ACTIVE = cfg


def _v3():
    from model import Ours_LoRA_MoE_V3 as V3
    return V3


def routers(model):
    return [layer.shared_expert_router for layer in _v3().shared_router_layers(model)]


def enabled(router):
    return getattr(router, "mres_reservoir", None) is not None


# --------------------------------------------------------------------- state
def attach(model, cfg=None):
    """Give every shared router one reservoir row (idempotent, default Linear init)."""
    cfg = ACTIVE if cfg is None else cfg
    if not cfg.enabled:
        return model
    for router in routers(model):
        if router.routing_weight_mode != "full_softmax":
            raise ValueError("mass reservoir needs --routing_weight_mode full_softmax "
                             f"(got {router.routing_weight_mode!r}): the selected weight must be "
                             "the full-denominator probability, not a renormalised/unit weight")
        router._mres_cfg = cfg
        if enabled(router):
            continue
        ref = router.router.weight if router.router is not None else router._placement_anchor
        router.mres_reservoir = torch.nn.Linear(router.hidden_size, 1, bias=False,
                                                device=ref.device, dtype=ref.dtype)
        router._mres_log_alpha = 0.0     # alpha = 1: the end-of-task state a checkpoint holds
        router._mres_primary = False     # primary pass: reservoir + skip out of the denominator
        router._mres_force_new = False   # warm-up: only the newest expert is a candidate
        router._mres_margin = False      # router-correction pass: collect the margin term
        router._mres_margin_buf = []
        router._mres_stats = None
        router._mres_probe = None
    return model


def log_alpha_of(alpha):
    return NEG_INF if alpha <= 0.0 else math.log(min(alpha, 1.0))


def set_alpha(model, alpha):
    value = log_alpha_of(alpha)
    for router in routers(model):
        if enabled(router):
            router._mres_log_alpha = value


def alpha_of(model):
    rs = [r for r in routers(model) if enabled(r)]
    return math.exp(rs[0]._mres_log_alpha) if rs else None


def set_flag(model, name, value):
    for router in routers(model):
        if enabled(router):
            setattr(router, name, value)


def copy_reservoir_into_rows(model, start, count):
    """New physical rows := exact clone of r_res (no noise)."""
    for router in routers(model):
        with torch.no_grad():
            router.router.weight[start:start + count].copy_(
                router.mres_reservoir.weight.expand(count, -1))


# ------------------------------------------------------------------- forward
def guarded_positions(router, n_tokens):
    """Flat bool mask of the positions the header/BOS guard overrides to skip at this layer
    (bos_guard._bos_guarded_forward: header at every layer, BOS also below the last layer), or
    None when no guard is installed on this model.  Those tokens never use the routing computed
    here, so they are left out of the margin, the stats and the expansion check."""
    if not hasattr(router, "_is_last_layer"):          # tagged only by bos_guard.install_bos_guard
        return None
    import bos_guard
    full, value = bos_guard._bos_mask["full"], bos_guard._bos_mask["value"]
    mask = full if router._is_last_layer else (value if full is None else value | full)
    if mask is None:
        return None
    mask = mask.reshape(-1).bool()
    return mask if mask.numel() == n_tokens else None


def router_forward(router, hidden_states, V3):
    """Selection over [physical, skip]; full-softmax weights with alpha*exp(z_res) in Z."""
    cfg = router._mres_cfg
    real = router._active_count()
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    # one fp32 GEMM over every row: a row copied from r_res gives bit-identical logits
    rows = torch.cat([router.router.weight[:real], router.residual_router.weight,
                      router.mres_reservoir.weight], dim=0)
    z = F.linear(flat.float(), rows.float())                       # [T, real + 2]
    cand = z[:, :real + 1]
    z_res = z[:, real + 1]
    if router._residual_log_alpha != 0.0:                          # skip gating (primary: -inf)
        cand = torch.cat([cand[:, :real], cand[:, real:] + router._residual_log_alpha], dim=-1)
    logit_bias = getattr(router, "_logit_bias", None)
    if logit_bias is not None:
        positions = getattr(router, "_logit_bias_positions", None)
        if positions is None:
            cand = cand + logit_bias.to(cand.dtype)
        else:
            pos = positions()
            if pos is not None:
                pos = pos.to(cand.device).bool()[:, None]
                cand = torch.where(pos, cand + logit_bias.to(cand.dtype), cand)
    if router._mres_force_new:                                     # warm-up: newest expert only
        keep = torch.zeros(real + 1, dtype=torch.bool, device=cand.device)
        keep[real - 1] = True
        cand = cand.masked_fill(~keep, NEG_INF)
    log_alpha = NEG_INF if router._mres_primary else router._mres_log_alpha
    res_col = z_res + log_alpha if log_alpha != NEG_INF else torch.full_like(z_res, NEG_INF)
    k = min(router.top_k, real + 1)
    topk_logits, topk_idx = cand.topk(k, dim=-1)
    full_probs = F.softmax(torch.cat([cand, res_col[:, None]], dim=-1), dim=-1)   # fp32
    weights = full_probs.gather(-1, topk_idx)                      # no renormalisation

    valid = router._router_token_mask
    if valid is not None:
        valid = valid.reshape(-1).to(device=cand.device, dtype=torch.bool)
        if valid.numel() != cand.shape[0]:
            raise ValueError("router mask/token count mismatch")
    router._last_moe_loss = (
        router._router_loss(cand, topk_idx, full_probs[:, :real + 1], valid)
        if router.training and not router._suppress_router_loss else None)

    threshold = topk_logits[:, -1].detach()                        # K-th candidate logit
    # tokens whose routing is actually used: not padding and not overridden by the header guard
    guarded = guarded_positions(router, cand.shape[0])
    routed = valid
    if guarded is not None:
        routed = ~guarded if routed is None else routed & ~guarded
    if router._mres_margin and router.training and torch.is_grad_enabled():
        hinge = F.relu(z_res - threshold + cfg.delta)
        router._mres_margin_buf.append((hinge, routed))
    if router._mres_stats is not None:
        _accumulate(router, cand, z_res, threshold, full_probs, topk_idx, routed, real)
    if router._mres_probe is not None:
        router._mres_probe.append({
            "log_z": torch.logsumexp(torch.cat([cand, res_col[:, None]], -1), -1).detach(),
            "topk": topk_idx.detach(), "z_res": z_res.detach(), "threshold": threshold,
            "real": real, "routed": None if routed is None else routed.detach()})
    if getattr(router, "_residual_stats", None) is not None:
        rec = topk_idx == real
        rec = rec if valid is None else rec[valid]
        router._residual_stats[0] += int(rec.sum())
        router._residual_stats[1] += int(rec.numel())
    router._last_probe_indices = (
        torch.where(topk_idx == real, -1, topk_idx).detach()
        if router._capture_probe_routing else None)
    return V3.RoutingContext(expert_indices=topk_idx, expert_weights=weights,
                             valid_token_mask=valid, num_experts=real)


# -------------------------------------------------------------------- margin
def pop_margin_per_sample(model, batch_size):
    """Mean over layers of the per-sample token-mean hinge; clears the buffers.  None if empty."""
    per_layer = []
    for router in routers(model):
        if not enabled(router) or not router._mres_margin_buf:
            continue
        hinge, valid = router._mres_margin_buf.pop()
        router._mres_margin_buf.clear()
        hinge = hinge.reshape(batch_size, -1)
        mask = (torch.ones_like(hinge, dtype=torch.bool) if valid is None
                else valid.reshape(batch_size, -1))
        mask = mask.to(hinge.dtype)
        per_layer.append((hinge * mask).sum(-1) / mask.sum(-1).clamp_min(1.0))
    if not per_layer:
        return None
    return torch.stack(per_layer).mean(0)


# --------------------------------------------------------------------- stats
STAT_KEYS = ("tokens", "z_res", "threshold", "violation", "res_mass", "new_rate", "skip_rate",
             "old_rate")


def enable_stats(model, pass_name):
    for router in routers(model):
        if enabled(router):
            if not isinstance(getattr(router, "_mres_stats_all", None), dict):
                router._mres_stats_all = {}
            router._mres_stats = router._mres_stats_all.setdefault(
                pass_name, {k: 0.0 for k in STAT_KEYS})


def disable_stats(model):
    for router in routers(model):
        if enabled(router):
            router._mres_stats = None


def _accumulate(router, cand, z_res, threshold, probs, topk_idx, valid, real):
    with torch.no_grad():
        if valid is not None:
            cand, z_res, threshold, probs, topk_idx = (
                cand[valid], z_res[valid], threshold[valid], probs[valid], topk_idx[valid])
        n = int(z_res.numel())
        if n == 0:
            return
        s = router._mres_stats
        top1 = topk_idx[:, 0]
        s["tokens"] += n
        s["z_res"] += float(z_res.sum())
        finite = torch.isfinite(threshold)
        s["threshold"] += float(threshold[finite].sum())
        s["violation"] += float((z_res > threshold - router._mres_cfg.delta).float().sum())
        s["res_mass"] += float(probs[:, -1].sum())
        s["new_rate"] += float((top1 == real - 1).float().sum())
        s["skip_rate"] += float((top1 == real).float().sum())
        s["old_rate"] += float((top1 < real - 1).float().sum())


def summarize(model, reset=True):
    out = {}
    for i, router in enumerate(routers(model)):
        if not enabled(router):
            continue
        for pass_name, s in (getattr(router, "_mres_stats_all", None) or {}).items():
            n = max(s["tokens"], 1.0)
            row = {k: s[k] / n for k in STAT_KEYS if k != "tokens"}
            row["tokens"] = int(s["tokens"])
            out.setdefault(pass_name, []).append({"layer": i, **row})
            if reset:
                for k in STAT_KEYS:
                    s[k] = 0.0
    return out


def row_report(model):
    """Router row norms and cos(r_res, newest physical row) per layer."""
    rep = []
    for i, router in enumerate(routers(model)):
        if not enabled(router):
            continue
        w = router.router.weight.detach().float()
        res = router.mres_reservoir.weight.detach().float()[0]
        rep.append({"layer": i, "row_norms": [float(x) for x in w.norm(dim=-1)],
                    "skip_norm": float(router.residual_router.weight.detach().float().norm()),
                    "res_norm": float(res.norm()),
                    "cos_res_newest": float(F.cosine_similarity(res, w[-1], dim=0))})
    return rep


# ------------------------------------------------------------ expansion check
@torch.no_grad()
def run_probe(model, batch):
    """Eval-mode forward on a fixed batch, capturing per-router routing tensors."""
    V3 = _v3()
    was_training = model.training
    model.eval()
    for router in routers(model):
        router._mres_probe = []
    V3.set_v3_router_token_mask(model, batch.get("attention_mask"))
    try:
        out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"), output_hidden_states=True, use_cache=False)
    finally:
        V3.set_v3_router_token_mask(model, None)
        captured = [router._mres_probe[-1] if router._mres_probe else None for router in routers(model)]
        for router in routers(model):
            router._mres_probe = None
        model.train(was_training)
    valid = batch.get("attention_mask")
    valid = None if valid is None else valid.reshape(-1).bool()
    return {"routers": captured, "hidden": [h.detach().float() for h in out.hidden_states],
            "logits": out.logits.detach().float(), "loss": None if out.loss is None else float(out.loss),
            "valid": valid}


def compare_probes(before, after):
    """Expansion-invariance report: denominator, Top-K membership, hidden and output drift."""
    valid = before["valid"]
    layers = []
    for i, (b, a) in enumerate(zip(before["routers"], after["routers"])):
        if b is None or a is None:
            continue
        # routed tokens only: padding and header/BOS-guarded positions excluded
        sel = b["routed"] if b.get("routed") is not None else (
            valid if valid is not None else torch.ones_like(b["z_res"], dtype=torch.bool))
        rel_z = torch.expm1((a["log_z"] - b["log_z"]).abs())[sel]
        # hidden_states[i] is decoder layer i's input: the denominator is only comparable where
        # that input is unchanged (an earlier Top-K change propagates to every later layer)
        hb, ha = before["hidden"][i], after["hidden"][i]
        same_in = ((ha - hb).reshape(-1, hb.shape[-1]).abs().amax(-1) == 0)[sel]
        rel_same = rel_z[same_in]
        margin_ok = (b["z_res"] < b["threshold"])[sel]
        # skip is index `real` in each state (it shifts by one at expansion): compare it as -1
        tb = torch.where(b["topk"] == b["real"], -1, b["topk"])
        ta = torch.where(a["topk"] == a["real"], -1, a["topk"])
        same_topk = (ta == tb).all(-1)[sel]
        top_b, top_a = b["topk"][:, 0][sel], a["topk"][:, 0][sel]
        layers.append({
            "layer": i,
            "denominator_rel_diff_max": float(rel_z.max()) if rel_z.numel() else 0.0,
            "denominator_rel_diff_mean": float(rel_z.mean()) if rel_z.numel() else 0.0,
            "same_input_rate": float(same_in.float().mean()),
            "denominator_rel_diff_max_same_input": float(rel_same.max()) if rel_same.numel() else 0.0,
            "margin_satisfied_rate": float(margin_ok.float().mean()),
            "topk_change_rate_all": float((~same_topk).float().mean()),
            "topk_change_rate_margin_ok": float((~same_topk[margin_ok]).float().mean()) if margin_ok.any() else 0.0,
            "new_expert_selected_rate_after": float((top_a == a["real"] - 1).float().mean()),
            "old_expert_rate_before": float((top_b < b["real"]).float().mean()),
            "old_expert_rate_after": float((top_a < b["real"]).float().mean()),
            "skip_rate_before": float((top_b == b["real"]).float().mean()),
            "skip_rate_after": float((top_a == a["real"]).float().mean())})
    hid = []
    for i, (hb, ha) in enumerate(zip(before["hidden"], after["hidden"])):
        hb2, ha2 = hb.reshape(-1, hb.shape[-1]), ha.reshape(-1, ha.shape[-1])
        if valid is not None:
            hb2, ha2 = hb2[valid], ha2[valid]
        rel = ((ha2 - hb2).norm(dim=-1) / hb2.norm(dim=-1).clamp_min(1e-12))
        cos = F.cosine_similarity(ha2, hb2, dim=-1)
        hid.append({"layer": i, "rel_l2_max": float(rel.max()), "rel_l2_mean": float(rel.mean()),
                    "cos_min": float(cos.min())})
    lb, la = before["logits"], after["logits"]
    return {"layers": layers, "hidden": hid,
            "logits_max_abs_diff": float((la - lb).abs().max()),
            "loss_before": before["loss"], "loss_after": after["loss"],
            "summary": {
                "denominator_rel_diff_max": max((l["denominator_rel_diff_max"] for l in layers), default=0.0),
                "denominator_rel_diff_max_same_input": max(
                    (l["denominator_rel_diff_max_same_input"] for l in layers), default=0.0),
                "topk_change_rate_margin_ok_max": max((l["topk_change_rate_margin_ok"] for l in layers), default=0.0),
                "topk_change_rate_all_max": max((l["topk_change_rate_all"] for l in layers), default=0.0),
                "margin_satisfied_rate_min": min((l["margin_satisfied_rate"] for l in layers), default=1.0),
                "hidden_rel_l2_max": max((h["rel_l2_max"] for h in hid), default=0.0),
                "logits_max_abs_diff": float((la - lb).abs().max())}}


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)


# ------------------------------------------------------------------ schedule
def alpha_schedule(step, total, warmup):
    """alpha after `step` optimizer updates: 0 during warm-up, then linear to 1 at `total`."""
    if step <= warmup:
        return 0.0
    return min(1.0, (step - warmup) / max(1, total - warmup))


def warmup_updates(total, frac):
    return 0 if frac <= 0 else max(1, int(round(total * frac)))
