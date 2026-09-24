"""LLaVA-DyMoE (Zhao et al., CVPR 2026) ported to the TRACE LLM contract.

Upstream: https://github.com/zhaoc5/DyMoE  (arXiv 2603.27481)
Reference file: ``llava/peft/tuners/incmoelora.py`` (IncMoELinear, TAG, RSR),
``llava/train/llava_trainer.py`` (RSR schedule) and
``llava/peft/utils/save_and_load.py`` (task-end fold).

Nothing in the method depends on the vision tower: the MoE sits on every linear
projection of the language model, and TAG/RSR only read router logits.  What is
ported, and how it maps onto upstream:

* ``DyMoELinear`` = ``IncMoELinear``.  Each wrapped projection owns a frozen
  bank (``dymoe_A_old`` / ``dymoe_B_old`` / ``dymoe_router_old`` = upstream
  ``lora_A/B``, ``lora_router``) holding every earlier task's experts, and a
  trainable bank (``dymoe_*_new`` = ``new_lora_*``) holding this task's
  ``expert_num`` experts.  An expert is a contiguous rank slice of the bank:
  ``r`` is split evenly, so upstream's default ``r=64, expert_num=16`` gives 16
  rank-4 experts.  The router is cosine similarity (x_hat . w_hat) times
  ``cosine_similarity_scale``, softmaxed at ``router_temperature`` over the
  top-k logits.  Training routes over old+new; eval routes over the old bank
  only, which by then contains the folded new experts.
* ``token_assignment_guidance`` and ``routing_score_regularization`` are
  copied from upstream, with the (B, L, E) layout flattened to (N, E).
* The task-end fold (``fold_new_into_old``) is upstream's
  ``get_peft_model_state_dict`` concatenation, done in place so the in-memory
  model and the saved checkpoint have the same layout.
* ``DyMoETab1`` reproduces ``LLaVATrainer.compute_loss``: the per-projection
  RSR terms are averaged over every adapted projection (upstream divides the
  sum by ``num_layers * 7``), and weighted by ``coeff * rsr_scale`` where
  ``rsr_scale`` is 0 for the first half of the task and ramps linearly to 1.
  RSR and TAG are inactive on the first task (no old group).

Two deliberate differences from upstream, both forced by the TRACE contract:

* Everything that is not the method (template, epochs, global batch 64,
  optimizer, bf16 adapters, seed) is the Table-1 contract, like every other
  row.  Upstream's CoIN recipe is 1 epoch at global batch 128.
* The paper's Eq. 8 also lists a load-balancing loss on the new experts;
  upstream never implements it (``compute_loss`` adds only exc/spe), so it is
  not ported.  With the default top-16 over 16 new experts it would be
  degenerate anyway: a new-group token always activates all 16.

``incmoelora`` (the paper's IncMoELoRA baseline) is the same layer and trainer
with TAG off and both RSR weights at zero.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from model.continual_lora import _FrozenLinearWrapper
from model.tab1_baselines import Tab1BaseTrainer
from model.tab1_lora import wrap_targets
from utils.utils import print_rank_0

DYMOE_STATE_KEY_SUBSTRINGS = [".dymoe_"]


# ---------------------------------------------------------------------------
# Upstream regularizers (llava/peft/tuners/incmoelora.py), (N, E) layout
# ---------------------------------------------------------------------------

def routing_score_regularization(z: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 expert_num: int = 16, k: int = 16,
                                 temp: float = 1.0, eps: float = 1e-8
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """RSR: exclusivity ``g_old * g_new`` and specialization BCE(g_new, 1 - max old)."""
    E = z.size(-1)
    assert 0 < expert_num <= E, "expert_num must be in (0, E]"
    num_E_old = E - expert_num

    # float32: at the low temperatures used here bf16 exponentials overflow.
    z_f32 = z.float()
    k_eff = E if k < 0 else min(k, E)
    _, topk_idx = torch.topk(z_f32, k=k_eff, dim=-1)
    sel_mask = torch.zeros_like(z_f32, dtype=torch.bool).scatter(-1, topk_idx, True)
    z_selected = z_f32.masked_fill(~sel_mask, -1e9)
    w = F.softmax(z_selected / max(temp, eps), dim=-1)

    g_new = w[..., num_E_old:].sum(dim=-1)
    g_old = w[..., :num_E_old].sum(dim=-1)
    L_exc_tok = g_old * g_new

    g_tilde_old = torch.max(w[..., :num_E_old], dim=-1).values.detach()
    y = (1.0 - g_tilde_old).clamp(0.0, 1.0)
    L_spe_tok = -(y * torch.log(g_new.clamp(min=eps))
                  + (1.0 - y) * torch.log((1.0 - g_new).clamp(min=eps)))

    if mask is None:
        mask = torch.ones_like(g_new)
    else:
        mask = mask.to(dtype=torch.float32, device=z.device).reshape(g_new.shape)
    denom = mask.sum().clamp_min(1.0)
    return (L_exc_tok * mask).sum() / denom, (L_spe_tok * mask).sum() / denom


def token_assignment_guidance(router: torch.Tensor, expert_num: int,
                              conflict_ratio: float) -> torch.Tensor:
    """TAG: only unambiguous, new-leaning tokens may reach the new group."""
    num_E_old = router.size(-1) - expert_num
    if num_E_old <= 0:
        return router
    z_old = router[..., :num_E_old]
    z_new = router[..., num_E_old:]
    old_max = z_old.max(dim=-1, keepdim=True).values
    new_max = z_new.max(dim=-1, keepdim=True).values
    denominator = torch.max(torch.abs(old_max), torch.abs(new_max))
    relative_difference = torch.where(
        denominator == 0, torch.zeros_like(old_max),
        torch.abs(old_max - new_max) / denominator)
    use_new = (relative_difference > conflict_ratio) & (new_max > old_max)
    z_old_masked = z_old.masked_fill(use_new, float("-inf"))
    z_new_masked = z_new.masked_fill(~use_new, float("-inf"))
    return torch.cat([z_old_masked, z_new_masked], dim=-1)


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

def _empty_parameter(rows: int, cols: int, like: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(torch.zeros(rows, cols, device=like.device,
                                    dtype=like.dtype), requires_grad=False)


class DyMoELinear(_FrozenLinearWrapper):
    """Frozen base + router-weighted LoRA experts over a growing bank (IncMoELinear)."""

    # Counted by tab1_lora.adapter_parameter_count.
    _tab1_adapter = True

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float,
                 expert_num: int, top_k: int, router_temperature: float,
                 cosine_similarity_scale: float):
        super().__init__(base)
        if r <= 0 or expert_num <= 0 or r % expert_num:
            raise ValueError(
                f"DyMoE splits rank {r} into {expert_num} equal experts; "
                "r must be a positive multiple of expert_num")
        self.r = int(r)
        self.expert_num = int(expert_num)
        self.top_k = int(top_k)
        self.router_temperature = float(router_temperature)
        self.cosine_similarity_scale = float(cosine_similarity_scale)
        # Upstream: ``2 if r == 0 else lora_alpha / r`` with r = this task's
        # bank rank; the same factor multiplies the old bank.
        self.scaling = float(alpha) / self.r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        weight = base.weight
        self.dymoe_A_old = _empty_parameter(0, self.in_features, weight)
        self.dymoe_B_old = _empty_parameter(self.out_features, 0, weight)
        self.dymoe_router_old = _empty_parameter(0, self.in_features, weight)
        self.dymoe_A_new: Optional[nn.Parameter] = None
        self.dymoe_B_new: Optional[nn.Parameter] = None
        self.dymoe_router_new: Optional[nn.Parameter] = None
        # Set per forward by the trainer (upstream injects training_args and
        # attention_mask onto the module for the same reason).
        self.tag_conflict_ratio: Optional[float] = None
        self.rsr_temperature: Optional[float] = None
        self.token_mask: Optional[torch.Tensor] = None
        self.last_rsr: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # -- bank bookkeeping ---------------------------------------------------
    @property
    def total_r(self) -> int:
        return self.dymoe_A_old.shape[0]

    @property
    def total_expert_num(self) -> int:
        return self.dymoe_router_old.shape[0]

    @property
    def has_new_bank(self) -> bool:
        return self.dymoe_A_new is not None

    def grow_old_bank(self, total_expert_num: int) -> None:
        """Size the frozen bank for ``total_expert_num`` experts (checkpoint load)."""
        per_expert = self.r // self.expert_num
        weight = self.base.weight
        self.dymoe_A_old = _empty_parameter(total_expert_num * per_expert,
                                            self.in_features, weight)
        self.dymoe_B_old = _empty_parameter(self.out_features,
                                            total_expert_num * per_expert, weight)
        self.dymoe_router_old = _empty_parameter(total_expert_num,
                                                 self.in_features, weight)

    def add_new_bank(self) -> None:
        """Upstream ``reset_lora_parameters`` for the new bank + nn.Linear router init."""
        if self.has_new_bank:
            raise RuntimeError("new bank already present; fold it first")
        weight = self.base.weight
        A = torch.empty(self.r, self.in_features, device=weight.device,
                        dtype=torch.float32)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        router = torch.empty(self.expert_num, self.in_features,
                             device=weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(router, a=math.sqrt(5))
        self.dymoe_A_new = nn.Parameter(A.to(weight.dtype))
        self.dymoe_B_new = nn.Parameter(torch.zeros(
            self.out_features, self.r, device=weight.device, dtype=weight.dtype))
        self.dymoe_router_new = nn.Parameter(router.to(weight.dtype))

    @torch.no_grad()
    def fold_new_into_old(self) -> None:
        """Upstream task-end fold: append the new experts to the frozen bank."""
        if not self.has_new_bank:
            return
        self.dymoe_A_old = nn.Parameter(
            torch.cat([self.dymoe_A_old, self.dymoe_A_new], dim=0),
            requires_grad=False)
        self.dymoe_B_old = nn.Parameter(
            torch.cat([self.dymoe_B_old, self.dymoe_B_new], dim=1),
            requires_grad=False)
        self.dymoe_router_old = nn.Parameter(
            torch.cat([self.dymoe_router_old, self.dymoe_router_new], dim=0),
            requires_grad=False)
        self.dymoe_A_new = None
        self.dymoe_B_new = None
        self.dymoe_router_new = None

    def new_parameters(self) -> List[nn.Parameter]:
        if not self.has_new_bank:
            return []
        return [self.dymoe_A_new, self.dymoe_B_new, self.dymoe_router_new]

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in (
            self.dymoe_A_old, self.dymoe_B_old, self.dymoe_router_old,
            *self.new_parameters()))

    # -- forward --------------------------------------------------------------
    def _score(self, h: torch.Tensor, router: torch.Tensor) -> torch.Tensor:
        if self.cosine_similarity_scale > 0:
            return self.cosine_similarity_scale * torch.matmul(
                F.normalize(h, p=2, dim=-1), F.normalize(router, p=2, dim=-1).T)
        return torch.matmul(h, router.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        use_new = self.training and self.has_new_bank
        if self.total_expert_num == 0 and not use_new:
            return result

        lead = x.shape[:-1]
        h = x.reshape(-1, x.shape[-1]).to(self.dymoe_A_old.dtype)

        router = self._score(h, self.dymoe_router_old)
        if use_new:
            router = torch.cat([router, self._score(h, self.dymoe_router_new)],
                               dim=-1)
            # RSR sees the unmasked scores, the experts the TAG-masked ones.
            if self.rsr_temperature is not None and self.total_expert_num > 0:
                self.last_rsr = routing_score_regularization(
                    router, mask=self.token_mask, expert_num=self.expert_num,
                    k=self.top_k, temp=self.rsr_temperature)
            if self.tag_conflict_ratio is not None:
                router = token_assignment_guidance(
                    router, self.expert_num, self.tag_conflict_ratio)

        num_experts = router.shape[-1]
        if self.top_k < 0 or self.top_k >= num_experts:
            router = torch.softmax(router / self.router_temperature, dim=-1)
        else:
            topk_values, topk_indices = torch.topk(router, k=self.top_k, dim=-1)
            masked = torch.full_like(router, float("-inf"))
            masked.scatter_(-1, topk_indices, topk_values)
            router = torch.softmax(masked / self.router_temperature, dim=-1)

        h_drop = self.dropout(h)
        x_a = F.linear(h_drop, self.dymoe_A_old)
        if use_new:
            x_a = torch.cat([x_a, F.linear(h_drop, self.dymoe_A_new)], dim=-1)
        x_a = (x_a.reshape(x_a.shape[0], num_experts, -1)
               * router.unsqueeze(-1)).reshape(x_a.shape[0], -1)

        out = F.linear(x_a[:, :self.total_r], self.dymoe_B_old)
        if use_new:
            out = out + F.linear(x_a[:, self.total_r:], self.dymoe_B_new)
        out = out * self.scaling
        return result + out.reshape(*lead, self.out_features).to(result.dtype)


# ---------------------------------------------------------------------------
# Model-level helpers
# ---------------------------------------------------------------------------

def attach_dymoe_targets(model: nn.Module, targets: Sequence[str], r: int,
                         alpha: float, dropout: float, expert_num: int,
                         top_k: int, router_temperature: float,
                         cosine_similarity_scale: float) -> int:
    return wrap_targets(
        model, targets,
        lambda linear, _: DyMoELinear(
            linear, r, alpha, dropout, expert_num, top_k, router_temperature,
            cosine_similarity_scale))


def iter_dymoe(model: nn.Module) -> Iterable[DyMoELinear]:
    return (module for module in model.modules()
            if isinstance(module, DyMoELinear))


def dymoe_total_experts(model: nn.Module) -> int:
    counts = {layer.total_expert_num for layer in iter_dymoe(model)}
    if len(counts) != 1:
        raise RuntimeError(f"inconsistent DyMoE old-bank sizes: {sorted(counts)}")
    return counts.pop()


def grow_dymoe_old_banks(model: nn.Module, total_expert_num: int) -> None:
    for layer in iter_dymoe(model):
        layer.grow_old_bank(total_expert_num)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DyMoETab1(Tab1BaseTrainer):
    """Per task: add a new expert bank, train it under TAG + RSR, fold it in.

    Old experts and old router rows are frozen (``requires_grad=False``), which
    is upstream's name-based freezing of ``.lora_*`` vs ``.new_lora_*``.
    """

    method_name = "dymoe"
    save_key_substrings = list(DYMOE_STATE_KEY_SUBSTRINGS)

    def __init__(self, *args_, **kwargs):
        super().__init__(*args_, **kwargs)
        self._task_id = 0
        self._task_updates = 1
        self._rsr_scale = 0.0
        self._last_terms: Optional[Tuple[float, float, float]] = None
        self._layers = list(iter_dymoe(self.raw_model))
        if not self._layers:
            raise RuntimeError("DyMoE trainer found no DyMoELinear layers")
        # Upstream pushes attention_mask onto every MoE layer once per forward
        # (llava_llama.py). It must stay installed through backward: gradient
        # checkpointing re-runs the layer forwards, RSR included.
        self._hook = self.raw_model.register_forward_pre_hook(
            self._before_forward, with_kwargs=True)

    # -- per-forward state --------------------------------------------------
    def _rsr_active(self) -> bool:
        args = self.args
        return (self._task_id > 1 and self._rsr_scale > 0
                and (args.dymoe_exc_coeff or args.dymoe_spe_coeff))

    def _before_forward(self, module, args_, kwargs):
        if not module.training:
            return None
        # llava_trainer.compute_loss: progress = global_step / max_steps, RSR
        # off below RSR_START_FRACTION, then a linear ramp to 1.
        start = self.args.dymoe_rsr_start_fraction
        progress = min(1.0, self.lr_scheduler.last_epoch / self._task_updates)
        self._rsr_scale = (0.0 if progress < start
                           else (progress - start) / max(1e-12, 1.0 - start))
        mask = kwargs.get("attention_mask")
        rsr_temperature = (self.args.dymoe_rsr_temperature
                           if self._rsr_active() else None)
        tag = (self.args.dymoe_conflict_ratio
               if self.args.dymoe_tag and self._task_id > 1 else None)
        for layer in self._layers:
            layer.token_mask = mask
            layer.rsr_temperature = rsr_temperature
            layer.tag_conflict_ratio = tag
            layer.last_rsr = None
        return None

    def extra_loss(self, batch, outputs):
        if not self._rsr_active():
            self._last_terms = None
            return None
        terms = [layer.last_rsr for layer in self._layers
                 if layer.last_rsr is not None]
        if len(terms) != len(self._layers):
            raise RuntimeError(
                f"RSR produced by {len(terms)} of {len(self._layers)} layers")
        # Upstream sums per-projection losses and divides by
        # num_hidden_layers * 7, i.e. the mean over adapted projections.
        loss_exc = torch.stack([exc for exc, _ in terms]).mean()
        loss_spe = torch.stack([spe for _, spe in terms]).mean()
        args = self.args
        term = (args.dymoe_exc_coeff * loss_exc
                + args.dymoe_spe_coeff * loss_spe) * self._rsr_scale
        self._last_terms = (float(loss_exc.detach()), float(loss_spe.detach()),
                            self._rsr_scale)
        return term.to(outputs.loss.dtype)

    def _extra_loss_log(self):
        if not self._last_terms:
            return ""
        exc, spe, scale = self._last_terms
        return f" | exc {exc:.4f} spe {spe:.4f} rsr_scale {scale:.2f}"

    # -- task boundaries ----------------------------------------------------
    def _optimizer_update_count(self, dataloader, epochs):
        count = super()._optimizer_update_count(dataloader, epochs)
        cap = int(getattr(self.args, "max_train_steps_per_task", 0))
        if cap:
            accum = max(1, self.args.gradient_accumulation_steps)
            count = min(count, int(epochs) * math.ceil(cap / accum))
        self._task_updates = max(1, count)
        return count

    def before_task(self, task, i_task):
        self._task_id = i_task + 1
        for parameter in self.raw_model.parameters():
            parameter.requires_grad = False
        for layer in self._layers:
            layer.add_new_bank()
            for parameter in layer.new_parameters():
                parameter.requires_grad = True
        # New banks are drawn from each rank's RNG; make rank 0's authoritative
        # before DDP is (re)built so no rank starts from a different init.
        if dist.is_initialized():
            for layer in self._layers:
                for parameter in layer.new_parameters():
                    dist.broadcast(parameter.data, src=0)
        layer = self._layers[0]
        print_rank_0(
            f"  [dymoe] task {self._task_id}: +{layer.expert_num} experts "
            f"(rank {layer.r // layer.expert_num} each) on "
            f"{len(self._layers)} projections, old experts "
            f"{layer.total_expert_num}, top_k {layer.top_k}, "
            f"TAG {'on' if self.args.dymoe_tag and self._task_id > 1 else 'off'} "
            f"(tau={self.args.dymoe_conflict_ratio}), RSR exc/spe "
            f"{self.args.dymoe_exc_coeff}/{self.args.dymoe_spe_coeff}",
            self.args.global_rank)

    def after_task(self, task, i_task):
        for layer in self._layers:
            layer.fold_new_into_old()
        print_rank_0(
            f"  [dymoe] folded: {dymoe_total_experts(self.raw_model)} experts, "
            f"total rank {self._layers[0].total_r}", self.args.global_rank)

    def write_meta(self, round_index, **extra):
        args = self.args
        layer = self._layers[0]
        super().write_meta(
            round_index,
            dymoe_variant=args.dymoe_variant,
            expert_num=layer.expert_num,
            total_expert_num=dymoe_total_experts(self.raw_model),
            total_r=layer.total_r,
            top_k=layer.top_k,
            router_temperature=layer.router_temperature,
            cosine_similarity_scale=layer.cosine_similarity_scale,
            tag=bool(args.dymoe_tag),
            conflict_ratio=args.dymoe_conflict_ratio,
            exc_coeff=args.dymoe_exc_coeff,
            spe_coeff=args.dymoe_spe_coeff,
            rsr_temperature=args.dymoe_rsr_temperature,
            rsr_start_fraction=args.dymoe_rsr_start_fraction,
            **extra)
