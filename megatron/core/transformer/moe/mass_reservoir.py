"""Mass-transfer reservoir for the expert-expanding shared-router MoE.

Port of trace/scripts/residual/mass_reservoir.py (feature/mass-reservoir) to the FLAME
shared-router hybrid, without the skip expert.

Every TopKRouter carries one extra row r_res.  The reservoir is never a Top-K candidate: it only
adds alpha * exp(z_res), z_res = r_res . x, to the softmax denominator, holding the mass that the
next task's experts will take.  Routing is softmax-then-topk without renormalisation
(--moe-router-pre-softmax), so a selected expert's weight is exp(z_i) / Z with

    Z = sum_i exp(z_i) + alpha * exp(z_res).

Task cycle (alpha_end = experts added per task, e.g. 8):
  expansion  new rows := r_res + eps_j, eps_j ~ N(0, sigma^2 I) (--mres-sigma; 0 = exact copies), new FFN
             experts get a zero down projection (the attention LoRA experts already start with B = 0),
             alpha: alpha_end -> 0.  The reservoir term carries the expected mass of such rows,
             alpha * exp(z_res + sigma^2 |x|^2 / 2), so Z is preserved in expectation (exactly at sigma 0);
             with the margin delta(x) = sigma |x| sqrt(2 ln(m / eta)) (m = alpha_end) no new row enters the
             Top-K with probability >= 1 - eta.  The noise lets the m identical rows diverge (m > k).
  warm-up    first warmup_frac of the updates: the primary pass may only pick new experts, old
             router rows and r_res get no update, alpha stays 0.
  primary    reservoir out of the denominator (stock routing).
  replay     router-only correction pass: reservoir mass in Z plus the margin
             lambda * relu(z_res - stopgrad(z_(K)) + delta), z_(K) the K-th largest candidate;
             delta = --mres-delta at sigma 0, otherwise delta(x) above.
  refill     after the warm-up alpha rises linearly 0 -> alpha_end, reaching it on the last update.
Checkpoints hold the end-of-task state, so loading sets alpha = alpha_end.
"""
from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn.functional as F

from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    switch_load_balancing_loss_func,
)

PARAM_NAME = "mres_reservoir"
NEG_INF = float("-inf")

STATE = {
    "alpha": None,             # None until configured: alpha_end at load, 0 right after expansion
    "alpha_end": 1.0,
    "delta": 0.5,
    "sigma": 0.0,              # reservoir prior N(r_res, sigma^2 I); 0 = exact copies
    "eta": None,               # Top-K failure probability bound (required when sigma > 0)
    "noise_seed": 0,
    "lam": 0.1,
    "warmup_frac": 0.05,
    "warmup_freeze_router": True,
    "pass": "eval",            # "primary" | "replay" | "eval" (training passes set it)
    "force_new_from": None,    # warm-up: candidates restricted to experts >= this index
    "freeze_old_from": None,   # warm-up: router rows < this index get no update
    "freeze_reservoir": False,
    "num_layers": 1,
    "grad_scale": 1.0,
}
CAPTURE = {"maps": None}     # list while an expansion check records routing maps, else None


def configure_from_args(args):
    STATE["alpha_end"] = float(args.mres_alpha_end)
    STATE["delta"] = float(args.mres_delta)
    STATE["sigma"] = float(getattr(args, "mres_sigma", 0.0) or 0.0)
    STATE["eta"] = getattr(args, "mres_eta", None)
    if STATE["sigma"] < 0:
        raise ValueError(f"--mres-sigma must be >= 0, got {STATE['sigma']}")
    if STATE["sigma"] > 0 and (STATE["eta"] is None or not 0 < float(STATE["eta"]) < 1):
        raise ValueError("--mres-sigma > 0 needs --mres-eta in (0, 1)")
    seed = getattr(args, "mres_noise_seed", None)
    STATE["noise_seed"] = int(seed if seed is not None else getattr(args, "seed", 0))
    STATE["lam"] = float(args.mres_lambda)
    STATE["warmup_frac"] = float(args.mres_warmup_frac)
    STATE["warmup_freeze_router"] = not bool(args.mres_no_warmup_freeze_router)
    if STATE["alpha"] is None:
        STATE["alpha"] = STATE["alpha_end"]


def enabled(router) -> bool:
    return getattr(router, PARAM_NAME, None) is not None


def attach(router) -> None:
    """Add the reservoir row to a TopKRouter (called from its __init__)."""
    config = router.config
    weight = torch.empty((1, config.hidden_size), dtype=torch.float32)
    if config.perform_initialization:
        config.init_method(weight)
    reservoir = torch.nn.Parameter(weight.to(dtype=config.params_dtype))
    setattr(reservoir, "sequence_parallel", config.sequence_parallel)
    router.register_parameter(PARAM_NAME, reservoir)
    router.weight.register_hook(_freeze_old_rows_hook)
    reservoir.register_hook(_freeze_reservoir_hook)


def _freeze_old_rows_hook(grad):
    n = STATE["freeze_old_from"]
    if n is None or n <= 0:
        return grad
    grad = grad.clone()
    grad[:n].zero_()
    return grad


def _freeze_reservoir_hook(grad):
    return torch.zeros_like(grad) if STATE["freeze_reservoir"] else grad


def _router_dtype(config, input_dtype):
    if config.moe_router_dtype == "fp32":
        return torch.float32
    if config.moe_router_dtype == "fp64":
        return torch.float64
    return input_dtype


def route(router, input: torch.Tensor):
    """Replacement for TopKRouter.forward after input jitter: returns (probs, routing_map)."""
    config = router.config
    if router.weight.device.type == "cpu":
        router.weight.data = router.weight.data.to(device=torch.cuda.current_device())
    if getattr(router, "_fingerprint_intervention_mode", None):
        raise RuntimeError("mass reservoir does not support router fingerprint interventions")
    if not (config.moe_router_pre_softmax and router.score_function == "softmax"):
        raise RuntimeError("mass reservoir needs --moe-router-pre-softmax with softmax scores")
    if config.moe_expert_capacity_factor is not None or router.expert_bias is not None:
        raise RuntimeError("mass reservoir does not support capacity factors or expert bias")
    if STATE["alpha"] is None:
        from megatron.training import get_args
        configure_from_args(get_args())

    num_experts = config.num_moe_experts
    dtype = _router_dtype(config, input.dtype)
    # one GEMM over every row: a row copied from r_res gives a bit-identical logit
    rows = torch.cat([router.weight, getattr(router, PARAM_NAME)], dim=0).to(dtype)
    z = F.linear(input.to(dtype), rows).view(-1, num_experts + 1)
    logits, z_res = z[:, :num_experts], z[:, num_experts]

    training = router.training
    phase = STATE["pass"] if training else "eval"
    if training and phase == "primary" and STATE["force_new_from"]:
        keep = torch.arange(num_experts, device=logits.device) >= STATE["force_new_from"]
        logits = logits.masked_fill(~keep, NEG_INF)
    logits = router.apply_z_loss(logits)

    alpha = STATE["alpha"]
    if training and phase == "primary":
        alpha = 0.0
    sigma = STATE["sigma"]
    x_norm = None
    if sigma > 0.0:
        x_sq = input.reshape(-1, input.shape[-1]).float().pow(2).sum(-1)
        x_norm = x_sq.sqrt()
    if alpha > 0.0:
        res_logit = z_res + math.log(alpha)
        if sigma > 0.0:   # expected mass of a row drawn from N(r_res, sigma^2 I): exp(z_res + sigma^2 |x|^2 / 2)
            res_logit = res_logit + (0.5 * sigma * sigma * x_sq).to(res_logit.dtype)
        full = torch.cat([logits, res_logit[:, None]], dim=-1)
    else:
        full = logits
    scores = torch.softmax(full, dim=-1, dtype=torch.float32)[:, :num_experts]
    top_probs, top_idx = torch.topk(scores, k=router.topk, dim=1)
    probs = torch.zeros_like(scores).scatter(1, top_idx, top_probs.type_as(scores))
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(1, top_idx, True)
    if CAPTURE["maps"] is not None:
        CAPTURE["maps"].append(routing_map.detach())

    if training and config.moe_aux_loss_coeff:
        aux_scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        probs = router.apply_load_balancing_loss(
            activation=probs,
            load_balancing_loss_func=partial(
                switch_load_balancing_loss_func,
                probs=aux_scores,
                tokens_per_expert=routing_map.sum(dim=0),
                topk=router.topk,
            ),
        )

    if training and phase == "replay" and STATE["lam"] > 0 and torch.is_grad_enabled():
        threshold = logits.gather(1, top_idx)[:, -1].detach()        # K-th candidate logit
        if sigma > 0.0:   # delta(x) = sigma |x| sqrt(2 ln(m / eta)), m = experts drawn per expansion
            m = max(1.0, STATE["alpha_end"])
            delta = (sigma * x_norm * math.sqrt(2.0 * math.log(m / float(STATE["eta"])))).to(z_res.dtype)
        else:
            delta = STATE["delta"]
        hinge = F.relu(z_res - threshold + delta)
        margin = hinge.mean() * (STATE["lam"] / STATE["num_layers"] * STATE["grad_scale"])
        probs = MoEAuxLossAutoScaler.apply(probs, margin)
        with torch.no_grad():
            save_to_aux_losses_tracker(
                "mres_margin_hinge", hinge.mean(), router.layer_number, config.num_layers)
            save_to_aux_losses_tracker(
                "mres_violation_rate", (z_res > threshold - delta).float().mean(),
                router.layer_number, config.num_layers)
            if alpha > 0.0:
                save_to_aux_losses_tracker(
                    "mres_reservoir_mass",
                    torch.softmax(full, dim=-1, dtype=torch.float32)[:, -1].mean(),
                    router.layer_number, config.num_layers)
    return probs, routing_map


# ----------------------------------------------------------------------------- task cycle
def count_routers(model_chunks) -> int:
    n = 0
    for chunk in model_chunks:
        for module in chunk.modules():
            if enabled(module):
                n += 1
    return n


def expand(target_model, num_existing_experts: int) -> dict:
    """After the stock expansion copy: new rows := r_res + eps (eps ~ N(0, sigma^2 I) from a generator seeded by
    (noise seed, first new row, layer) only, so every rank draws the same rows; sigma 0 = exact copies), new FFN
    down projections := 0."""
    from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
    from megatron.core.transformer.moe.router import Router

    rows = ffn = lora = 0
    row_diff = w2_max = lora_b_max = 0.0
    with torch.no_grad():
        for module in target_model.modules():
            if isinstance(module, Router) and enabled(module):
                total = module.weight.shape[0]
                count = total - num_existing_experts
                reservoir = getattr(module, PARAM_NAME)
                new = reservoir.detach().float().expand(count, -1).clone()
                if STATE["sigma"] > 0.0:
                    layer = int(getattr(module, "layer_number", 0) or 0)
                    gen = torch.Generator().manual_seed(
                        STATE["noise_seed"] + 7919 * int(num_existing_experts) + 104729 * layer)
                    eps = torch.randn(count, new.shape[1], generator=gen, dtype=torch.float32)
                    new = new + (STATE["sigma"] * eps).to(new.device)
                module.weight[num_existing_experts:total].copy_(new.to(module.weight.dtype))
                row_diff = max(row_diff, float(
                    (module.weight[num_existing_experts:total].float() - new.to(module.weight.dtype).float())
                    .abs().max()))
                rows += 1
            elif isinstance(module, GroupedMLP):
                w2 = module.weight2.view(module.num_local_experts, -1, module.config.hidden_size)
                w2[num_existing_experts:].zero_()
                w2_max = max(w2_max, float(w2[num_existing_experts:].abs().max()))
                ffn += 1
            elif isinstance(module, SequentialMLP):
                for expert in module.local_experts[num_existing_experts:]:
                    fc2 = expert.linear_fc2
                    fc2.weight.zero_()
                    if getattr(fc2, "bias", None) is not None:
                        fc2.bias.zero_()
                    w2_max = max(w2_max, float(fc2.weight.abs().max()))
                ffn += 1
            else:
                for name in ("qkv_lora_b", "q_lora_b", "k_lora_b", "v_lora_b", "proj_lora_b"):
                    b = getattr(module, name, None)
                    if isinstance(b, torch.Tensor) and b.dim() >= 1 and b.shape[0] > num_existing_experts:
                        lora_b_max = max(lora_b_max, float(b[num_existing_experts:].abs().max()))
                        lora += 1
    alpha_before = STATE["alpha"]
    STATE["alpha"] = 0.0
    report = {"routers": rows, "ffn_expert_modules": ffn, "attn_lora_b_tensors": lora,
              "num_existing_experts": num_existing_experts,
              "sigma": STATE["sigma"], "eta": STATE["eta"], "noise_seed": STATE["noise_seed"],
              "max|new_row - (r_res + eps)|": row_diff, "max|new FFN down|": w2_max,
              "max|new attn LoRA B|": lora_b_max, "alpha_before": alpha_before, "alpha_after": 0.0}
    if rows == 0 or ffn == 0:
        raise RuntimeError(f"mass reservoir expansion found no routers or FFN experts: {report}")
    if row_diff != 0.0 or w2_max != 0.0 or lora_b_max != 0.0:
        raise RuntimeError(f"mass reservoir expansion is not exact: {report}")
    return report


def begin_step(args, iteration: int, num_existing_experts):
    """Set alpha and the warm-up flags for update `iteration` (0-based) of args.train_iters."""
    total = int(args.train_iters)
    warm = max(0, math.ceil(STATE["warmup_frac"] * total))
    if iteration < warm:
        STATE["alpha"] = 0.0
        STATE["force_new_from"] = num_existing_experts or None
        STATE["freeze_old_from"] = (
            num_existing_experts if (num_existing_experts and STATE["warmup_freeze_router"])
            else None)
        STATE["freeze_reservoir"] = STATE["warmup_freeze_router"]
    else:
        STATE["alpha"] = STATE["alpha_end"] * (iteration - warm + 1) / max(1, total - warm)
        STATE["force_new_from"] = None
        STATE["freeze_old_from"] = None
        STATE["freeze_reservoir"] = False
    return {"alpha": STATE["alpha"], "warmup": iteration < warm, "warmup_updates": warm}


def set_pass(name: str, grad_scale: float = 1.0) -> None:
    STATE["pass"] = name
    STATE["grad_scale"] = float(grad_scale)
