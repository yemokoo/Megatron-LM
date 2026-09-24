"""
Growing FFN LoRA-MoE for continual learning.

Each transformer layer's FFN (gate/up/down proj) keeps its original dense
weights frozen forever, and gets a token-routed mixture of small LoRA
"experts" bolted on top. At the start of every new task, `experts_per_task`
brand-new experts are appended (to every layer, simultaneously) and the
router's output grows to cover them. Only the just-added experts + the
router train on the new task's data (phase 1); afterwards, ALL experts
(including the brand-new ones) are frozen and only the router is re-tuned
on a capped pool of all seen-task fixed subsets, including current data
(phase 2), so routing stays calibrated as the expert pool grows.

Attention is untouched -- no LoRA there, dense and frozen like the rest of
the backbone.
"""
import copy
import hashlib
import json
import math
import os
import re
import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              RandomSampler, Sampler, Subset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import get_scheduler

try:
    from torch.utils.flop_counter import FlopCounterMode
except ImportError:  # pragma: no cover - older PyTorch fallback
    FlopCounterMode = None


def _gqa_sdpa_flop(query_shape, key_shape, value_shape, *args,
                   out_shape=None, **kwargs):
    """Count SDPA FLOPs for regular MHA and grouped-query attention."""
    batch, query_heads, query_tokens, query_dim = query_shape
    key_batch, key_heads, key_tokens, key_dim = key_shape
    value_batch, value_heads, value_tokens, value_dim = value_shape
    if not (batch == key_batch == value_batch
            and key_heads == value_heads
            and query_heads % key_heads == 0
            and query_dim == key_dim
            and key_tokens == value_tokens):
        raise ValueError(
            "unsupported SDPA shapes for FLOP counting: "
            f"q={query_shape} k={key_shape} v={value_shape}")
    return (
        2 * batch * query_heads * query_tokens * key_tokens * query_dim
        + 2 * batch * query_heads * query_tokens * key_tokens * value_dim)


def _flop_counter_custom_mapping():
    if FlopCounterMode is None:
        return {}
    aten = torch.ops.aten
    return {
        aten._scaled_dot_product_efficient_attention: _gqa_sdpa_flop,
        aten._scaled_dot_product_flash_attention: _gqa_sdpa_flop,
        aten._scaled_dot_product_cudnn_attention: _gqa_sdpa_flop,
    }

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0, to_device, get_optimizer_grouped_parameters
from utils.data.data_collator import (DataCollator, SLoRATraceDataCollator,
                                      PreTokenizedSLoRATraceDataCollator)


V2_NEW_TRAINING_VERSIONS = frozenset({
    "v2_new", "v2_new_top4", "v3_new", "v3_new_top4",
})
V2_NEW_MEMORY_TRAINING_VERSIONS = frozenset({
    "v2_new", "v2_new_top4", "v3_new", "v3_new_top4",
    "v1_expert_first",
})


def _is_v2_new_training_version(training_version):
    return training_version in V2_NEW_TRAINING_VERSIONS


def _uses_v2_new_memory(training_version):
    return training_version in V2_NEW_MEMORY_TRAINING_VERSIONS


class RepeatedSubsetDataset(Dataset):
    """Deterministic exposure stream drawn only from one fixed task subset."""

    def __init__(self, subset, exposure_indices):
        self.subset = subset
        self.exposure_indices = list(exposure_indices)

    def __len__(self):
        return len(self.exposure_indices)

    def __getitem__(self, index):
        return self.subset[self.exposure_indices[index]]


class DeterministicPassRandomSampler(Sampler):
    """Single-process analogue of DistributedSampler's seed+epoch order."""

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(
            len(self.data_source), generator=generator).tolist())

    def __len__(self):
        return len(self.data_source)


class LoRAPair(nn.Module):
    """One rank-r LoRA delta: out = (x @ A^T) @ B^T * (alpha / r)."""

    def __init__(self, in_features, out_features, r, alpha, dropout=0.0):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)  # B stays zero -> new expert starts as a no-op

    def forward(self, x):
        return (self.dropout(x) @ self.A.T) @ self.B.T * self.scaling


def _make_expert(hidden_size, intermediate_size, r, alpha, dropout=0.0):
    return nn.ModuleDict({
        "gate": LoRAPair(hidden_size, intermediate_size, r, alpha, dropout),
        "up":   LoRAPair(hidden_size, intermediate_size, r, alpha, dropout),
        "down": LoRAPair(intermediate_size, hidden_size, r, alpha, dropout),
    })


def _quota_top1_dispatch(flat_logits, natural_top1_idx, expert_index,
                         quota_fraction, valid_token_mask=None):
    """Return top-1 indices with the smallest margin-ranked quota injection."""
    if natural_top1_idx.ndim != 2 or natural_top1_idx.shape[1] != 1:
        raise ValueError("quota dispatch requires [tokens, 1] top-1 indices")
    if not 0.0 <= quota_fraction <= 1.0:
        raise ValueError(
            f"quota fraction must be in [0, 1], got {quota_fraction}")
    if not 0 <= expert_index < flat_logits.shape[1]:
        raise ValueError(
            f"quota expert {expert_index} is outside {flat_logits.shape[1]} logits")
    if valid_token_mask is None:
        valid_token_mask = torch.ones(
            flat_logits.shape[0], dtype=torch.bool,
            device=flat_logits.device)
    natural_new_mask = natural_top1_idx[:, 0].eq(expert_index)
    valid_count = int(valid_token_mask.sum().detach().item())
    natural_new_count = int(
        (natural_new_mask & valid_token_mask).sum().detach().item())
    requested_count = int(math.ceil(quota_fraction * valid_count))
    inject_count = max(0, requested_count - natural_new_count)
    dispatch_idx = natural_top1_idx.clone()
    injected_token_indices = natural_top1_idx.new_empty((0,))
    if inject_count > 0:
        candidate_indices = torch.where(
            valid_token_mask & ~natural_new_mask)[0]
        inject_count = min(inject_count, int(candidate_indices.numel()))
        candidate_logits = flat_logits[candidate_indices]
        competitor_logits = candidate_logits.clone()
        competitor_logits[:, expert_index] = -torch.inf
        margins = (
            candidate_logits[:, expert_index]
            - competitor_logits.max(dim=-1).values)
        selected_candidates = margins.topk(
            inject_count, largest=True, sorted=False).indices
        injected_token_indices = candidate_indices[selected_candidates]
        dispatch_idx[injected_token_indices, 0] = expert_index
    dispatched_new_count = int(
        (dispatch_idx[:, 0].eq(expert_index) & valid_token_mask)
        .sum().detach().item())
    diagnostic = {
        "valid_tokens": valid_count,
        "natural_selected_tokens": natural_new_count,
        "dispatched_selected_tokens": dispatched_new_count,
        "injected_tokens": dispatched_new_count - natural_new_count,
        "quota_fraction": float(quota_fraction),
    }
    return dispatch_idx, injected_token_indices, diagnostic


class LoRAMoEMLP(nn.Module):
    """Wraps a frozen dense FFN (base_mlp) with a growing pool of routed LoRA experts."""

    def __init__(self, base_mlp, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
                 routing_weight_mode="topk_softmax", dropout=0.0):
        super().__init__()
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        self.hidden_size = base_mlp.gate_proj.in_features
        self.intermediate_size = base_mlp.gate_proj.out_features
        self.r, self.alpha = r, alpha
        self.dropout = dropout
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        if routing_weight_mode not in {
                "topk_softmax", "full_softmax", "straight_through_topk"}:
            raise ValueError(f"unknown routing_weight_mode: {routing_weight_mode}")
        self.routing_weight_mode = routing_weight_mode

        self.experts = nn.ModuleList()
        self.router = None  # nn.Linear(hidden_size, num_experts, bias=False); grown via add_experts()
        self._last_moe_loss = None
        # Replay optimizes only causal LM loss.  A trainer-owned context may
        # suppress construction of aux/z router-loss graphs for that forward;
        # normal primary/KD forwards leave this disabled.
        self._suppress_router_loss = False
        self._router_token_mask = None
        # v2 KD uses the expanded module as its own frozen pre-expansion teacher.
        # Limiting the active prefix exactly recovers the old expert/router shape
        # without keeping a second 8B backbone in memory.
        self._active_expert_count = None
        # Evaluation-only diagnostic: route every token to exactly one expert.
        # None preserves normal learned routing.
        self._forced_expert_index = None
        # Opt-in training diagnostics. These fields never affect dispatch or
        # gradients; a sampled forward only stores detached scalar summaries.
        self._diagnostic_expert_index = None
        self._last_route_diagnostic = None
        # Opt-in training-only dispatch override.  The natural top-1 route is
        # still computed first; a quota branch may then redirect only enough
        # valid tokens to the target expert to reach the requested floor.
        self._training_quota_expert_index = None
        self._training_quota_fraction = 0.0
        self._last_quota_diagnostic = None
        # Opt-in V2-new auxiliary acquisition route.  The ordinary forward is
        # run separately and remains untouched.  During the auxiliary forward
        # a hard top-1 old route is interpolated with the newly-added expert;
        # router parameters are frozen by the trainer and aux/z are omitted.
        self._training_aux_expert_index = None
        self._training_aux_mix = 0.0
        self._last_aux_diagnostic = None

    @property
    def num_experts(self):
        return len(self.experts)

    def add_experts(self, n):
        device = self.base_mlp.gate_proj.weight.device
        dtype = self.base_mlp.gate_proj.weight.dtype
        for _ in range(n):
            self.experts.append(
                _make_expert(
                    self.hidden_size, self.intermediate_size, self.r,
                    self.alpha, self.dropout).to(device=device, dtype=dtype)
            )

        old_n = self.router.out_features if self.router is not None else 0
        new_router = nn.Linear(self.hidden_size, old_n + n, bias=False).to(device=device, dtype=dtype)
        if self.router is not None:
            with torch.no_grad():
                new_router.weight[:old_n] = self.router.weight
        self.router = new_router

    def forward(self, x):
        active_experts = (
            self.num_experts
            if self._active_expert_count is None
            else int(self._active_expert_count)
        )
        if active_experts == 0:
            self._last_moe_loss = x.new_zeros(())
            return self.base_mlp(x)
        if not 0 < active_experts <= self.num_experts:
            raise ValueError(
                f"active expert prefix {active_experts} is invalid for "
                f"{self.num_experts} experts")

        gate = self.base_mlp.gate_proj(x)
        up = self.base_mlp.up_proj(x)

        # Slice the WEIGHT before the GEMM, not the expanded router output.
        # Besides excluding new rows from softmax/top-k, this reproduces the
        # original pre-expansion BF16 GEMM shape. Computing all expanded rows
        # and slicing the output can select a different CUDA GEMM kernel and
        # numerically drift from the frozen teacher even for identical prefix
        # weights.
        flat_logits = F.linear(
            x, self.router.weight[:active_experts], bias=None
        ).reshape(-1, active_experts)                                      # [N, E]
        k = min(self.top_k, active_experts)
        forced_expert = self._forced_expert_index
        if forced_expert is not None:
            forced_expert = int(forced_expert)
            if not 0 <= forced_expert < active_experts:
                raise ValueError(
                    f"forced expert {forced_expert} is outside active prefix "
                    f"of {active_experts} experts")
            if k != 1:
                raise ValueError(
                    "single-expert diagnostic routing requires top_k=1")
            topk_idx = torch.full(
                (flat_logits.shape[0], 1), forced_expert,
                dtype=torch.long, device=flat_logits.device)
            topk_val = flat_logits.gather(-1, topk_idx)
        else:
            topk_val, topk_idx = flat_logits.topk(k, dim=-1)

        valid_token_mask = self._router_token_mask
        if valid_token_mask is not None:
            valid_token_mask = valid_token_mask.reshape(-1).to(
                device=flat_logits.device, dtype=torch.bool)
            if valid_token_mask.numel() != flat_logits.shape[0]:
                raise ValueError(
                    f"router mask has {valid_token_mask.numel()} tokens but "
                    f"hidden states have {flat_logits.shape[0]}")

        natural_topk_idx = topk_idx
        quota_expert = self._training_quota_expert_index
        quota_fraction = float(self._training_quota_fraction)
        if quota_expert is not None and quota_fraction > 0:
            if not self.training:
                raise RuntimeError("expert quota dispatch is training-only")
            if forced_expert is not None:
                raise RuntimeError(
                    "expert quota and forced diagnostic routing cannot overlap")
            if k != 1:
                raise ValueError("expert quota dispatch currently requires top_k=1")
            quota_expert = int(quota_expert)
            if not 0 <= quota_expert < active_experts:
                raise ValueError(
                    f"quota expert {quota_expert} is outside active prefix "
                    f"of {active_experts} experts")
            if not 0.0 <= quota_fraction <= 1.0:
                raise ValueError(
                    f"quota fraction must be in [0, 1], got {quota_fraction}")
            topk_idx, _, self._last_quota_diagnostic = (
                _quota_top1_dispatch(
                    flat_logits, natural_topk_idx, quota_expert,
                    quota_fraction, valid_token_mask))
            topk_val = flat_logits.gather(-1, topk_idx)
        else:
            self._last_quota_diagnostic = None

        aux_expert = self._training_aux_expert_index
        aux_mix = float(self._training_aux_mix)
        if aux_expert is not None and aux_mix > 0:
            if not self.training:
                raise RuntimeError(
                    "auxiliary expert interpolation is training-only")
            if forced_expert is not None or quota_expert is not None:
                raise RuntimeError(
                    "auxiliary expert interpolation cannot overlap forced "
                    "or quota routing")
            if k != 1 or self.routing_weight_mode != "straight_through_topk":
                raise ValueError(
                    "auxiliary expert interpolation requires straight-through "
                    "top-1 routing")
            aux_expert = int(aux_expert)
            if not 0 <= aux_expert < active_experts:
                raise ValueError(
                    f"auxiliary expert {aux_expert} is outside active prefix "
                    f"of {active_experts} experts")
            if not 0.0 < aux_mix <= 1.0:
                raise ValueError(
                    f"auxiliary expert mix must be in (0, 1], got {aux_mix}")
            aux_valid_mask = valid_token_mask
            if aux_valid_mask is None:
                aux_valid_mask = torch.ones(
                    flat_logits.shape[0], dtype=torch.bool,
                    device=flat_logits.device)
            aux_natural_mask = natural_topk_idx[:, 0].eq(aux_expert)
            aux_exposure_mask = aux_valid_mask & ~aux_natural_mask
            # Keep tensor counters detached so the hot path does not perform a
            # per-layer CUDA synchronization. Tests/logging may materialize
            # them later.
            self._last_aux_diagnostic = {
                "valid_tokens": aux_valid_mask.sum().detach(),
                "natural_selected_tokens": (
                    aux_valid_mask & aux_natural_mask).sum().detach(),
                "auxiliary_exposed_tokens": aux_exposure_mask.sum().detach(),
                "mix": aux_mix,
            }
        else:
            aux_expert = None
            aux_exposure_mask = None
            self._last_aux_diagnostic = None

        full_probs = None
        if forced_expert is not None:
            # Diagnostic measures the selected task expert at its full LoRA
            # strength and intentionally bypasses learned router confidence.
            topk_weight = torch.ones_like(topk_val, dtype=torch.float)
        elif self.routing_weight_mode in {
                "full_softmax", "straight_through_topk"}:
            # Gather probabilities from the full router distribution without
            # renormalizing over the selected experts. For top-k=1 the dispatch
            # weight therefore does not collapse to the constant 1, and the LM
            # objective can train the router.
            full_probs = F.softmax(flat_logits, dim=-1, dtype=torch.float)
            selected_probs = full_probs.gather(-1, topk_idx)
            if self.routing_weight_mode == "straight_through_topk":
                # Forward uses a normalized top-k mixture (exactly 1 for
                # top-k=1), while backward keeps the full-softmax surrogate
                # so the LM objective can still train the router.
                normalized = F.softmax(topk_val, dim=-1)
                topk_weight = (
                    normalized.detach() + selected_probs
                    - selected_probs.detach())
            else:
                topk_weight = selected_probs
        else:
            # Legacy behavior retained for loading existing checkpoints.
            topk_weight = F.softmax(topk_val, dim=-1)

        diagnostic_expert = self._diagnostic_expert_index
        if diagnostic_expert is not None:
            diagnostic_expert = int(diagnostic_expert)
            if not 0 <= diagnostic_expert < active_experts:
                raise ValueError(
                    f"diagnostic expert {diagnostic_expert} is outside "
                    f"active prefix {active_experts}")
            diagnostic_mask = valid_token_mask
            if diagnostic_mask is None:
                diagnostic_mask = torch.ones(
                    flat_logits.shape[0], dtype=torch.bool,
                    device=flat_logits.device)
            diagnostic_logits = flat_logits[diagnostic_mask]
            diagnostic_selected = topk_idx[diagnostic_mask]
            diagnostic_probs = (
                full_probs[diagnostic_mask] if full_probs is not None
                else F.softmax(diagnostic_logits, dim=-1, dtype=torch.float))
            if diagnostic_logits.shape[0] == 0:
                self._last_route_diagnostic = None
            else:
                competitor_logits = diagnostic_logits.clone()
                competitor_logits[:, diagnostic_expert] = -torch.inf
                competitor_max = competitor_logits.max(dim=-1).values
                target_logits = diagnostic_logits[:, diagnostic_expert]
                self._last_route_diagnostic = {
                    "valid_tokens": int(diagnostic_logits.shape[0]),
                    "selected_tokens": int((
                        diagnostic_selected == diagnostic_expert
                    ).any(dim=-1).sum().detach().item()),
                    "probability_sum": float(
                        diagnostic_probs[:, diagnostic_expert].sum()
                        .detach().float().item()),
                    "margin_sum": float(
                        (target_logits - competitor_max).sum()
                        .detach().float().item()),
                    "target_logit_sum": float(
                        target_logits.sum().detach().float().item()),
                    "competitor_logit_sum": float(
                        competitor_max.sum().detach().float().item()),
                }

        # Dispatch only the tokens selected for each expert. The previous path
        # applied every active expert to the complete [batch, seq] tensor and
        # multiplied non-routed positions by zero afterwards. With a long prefill,
        # nearly every expert is active somewhere, so that turned sparse top-k
        # routing into E dense LoRA forwards. It also used torch.any() in Python,
        # introducing a GPU synchronization for every expert in every layer.
        flat_x = x.reshape(-1, x.shape[-1])
        flat_gate_delta = gate.new_zeros((flat_x.shape[0], gate.shape[-1]))
        flat_up_delta = up.new_zeros((flat_x.shape[0], up.shape[-1]))
        # Only experts selected by at least one (valid) token are visited: one
        # host sync for the id set instead of a where()+numel() sync per expert
        # per layer per decode step. Skipped experts had no tokens anyway, and
        # ascending order keeps index_add_ accumulation identical.
        route_idx = topk_idx if valid_token_mask is None else topk_idx[valid_token_mask]
        active_ids = [
            e for e in torch.unique(route_idx).tolist() if 0 <= e < active_experts
        ]
        routes = {}
        for e in active_ids:
            slot, token_idx = torch.where(topk_idx.transpose(0, 1) == e)
            if valid_token_mask is not None:
                keep = valid_token_mask[token_idx]
                slot, token_idx = slot[keep], token_idx[keep]
            routes[e] = (slot, token_idx)
        for e in active_ids:
            expert = self.experts[e]
            slot, token_idx = routes[e]
            routed_x = flat_x[token_idx]
            if aux_expert is None:
                weight = topk_weight[token_idx, slot, None].to(routed_x.dtype)
            else:
                # In the auxiliary pass the hard natural top-1 expert keeps
                # 1-mix of the adapter slot. Tokens already routed to the new
                # expert retain the exact natural value but are detached so
                # this branch only adds exposure that natural routing missed.
                natural_weight = 1.0 if e == aux_expert else 1.0 - aux_mix
                weight = routed_x.new_full(
                    (token_idx.numel(), 1), natural_weight)
            gate_contribution = expert["gate"](routed_x) * weight
            up_contribution = expert["up"](routed_x) * weight
            if aux_expert is not None and e == aux_expert:
                gate_contribution = (
                    gate_contribution.detach() + gate_contribution * 0.0)
                up_contribution = (
                    up_contribution.detach() + up_contribution * 0.0)
            flat_gate_delta.index_add_(
                0, token_idx, gate_contribution)
            flat_up_delta.index_add_(
                0, token_idx, up_contribution)

        if aux_expert is not None:
            aux_token_idx = torch.where(aux_exposure_mask)[0]
            if aux_token_idx.numel() > 0:
                aux_x = flat_x[aux_token_idx]
                aux_weight = aux_x.new_full(
                    (aux_token_idx.numel(), 1), aux_mix)
                aux_module = self.experts[aux_expert]
                flat_gate_delta.index_add_(
                    0, aux_token_idx,
                    aux_module["gate"](aux_x) * aux_weight)
                flat_up_delta.index_add_(
                    0, aux_token_idx,
                    aux_module["up"](aux_x) * aux_weight)

        gate_delta = flat_gate_delta.reshape_as(gate)
        up_delta = flat_up_delta.reshape_as(up)
        h = F.silu(gate + gate_delta) * (up + up_delta)
        down = self.base_mlp.down_proj(h)

        flat_h = h.reshape(-1, h.shape[-1])
        flat_down_delta = down.new_zeros((flat_h.shape[0], down.shape[-1]))
        for e in active_ids:
            expert = self.experts[e]
            slot, token_idx = routes[e]
            routed_h = flat_h[token_idx]
            if aux_expert is None:
                weight = topk_weight[token_idx, slot, None].to(routed_h.dtype)
            else:
                natural_weight = 1.0 if e == aux_expert else 1.0 - aux_mix
                weight = routed_h.new_full(
                    (token_idx.numel(), 1), natural_weight)
            down_contribution = expert["down"](routed_h) * weight
            if aux_expert is not None and e == aux_expert:
                down_contribution = (
                    down_contribution.detach() + down_contribution * 0.0)
            flat_down_delta.index_add_(
                0, token_idx, down_contribution)
        if aux_expert is not None:
            aux_token_idx = torch.where(aux_exposure_mask)[0]
            if aux_token_idx.numel() > 0:
                aux_h = flat_h[aux_token_idx]
                aux_weight = aux_h.new_full(
                    (aux_token_idx.numel(), 1), aux_mix)
                flat_down_delta.index_add_(
                    0, aux_token_idx,
                    self.experts[aux_expert]["down"](aux_h) * aux_weight)
        down = down + flat_down_delta.reshape_as(down)

        # Generation never consumes the auxiliary router loss. Avoid a full
        # softmax, dense hard-routing mask and z-loss on every decode token.
        self._last_moe_loss = (
            self._router_loss(flat_logits, natural_topk_idx,
                              full_probs=full_probs,
                              token_mask=valid_token_mask)
            if (self.training and aux_expert is None
                and not self._suppress_router_loss) else None)
        return down

    def _router_loss(self, flat_logits, topk_idx, full_probs=None, token_mask=None):
        # Switch-Transformer aux loss + router z-loss, same formulas/coefficients
        # as LLM-continual-learning's Megatron router (moe_utils.py).
        if token_mask is not None:
            flat_logits = flat_logits[token_mask]
            topk_idx = topk_idx[token_mask]
            if full_probs is not None:
                full_probs = full_probs[token_mask]
        num_tokens, num_experts = flat_logits.shape
        if num_tokens == 0:
            return flat_logits.sum() * 0.0
        if full_probs is None:
            full_probs = F.softmax(flat_logits, dim=-1, dtype=torch.float)
        tokens_per_expert = torch.bincount(
            topk_idx.reshape(-1), minlength=num_experts).to(full_probs.dtype)
        aggregated_probs_per_expert = full_probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
            num_experts * self.aux_loss_coeff /
            (num_tokens * num_tokens * topk_idx.shape[-1])
        )
        z_loss = torch.mean(torch.square(
            torch.logsumexp(flat_logits.float(), dim=-1))) * self.z_loss_coeff
        return aux_loss + z_loss


def attach_lora_moe(model, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
                    routing_weight_mode="topk_softmax", dropout=0.0):
    """Replace every decoder layer's .mlp with an (initially expert-less) LoRAMoEMLP.

    Then freeze the WHOLE backbone -- attention, embeddings, norms, lm_head and
    every FFN's dense base_mlp stay frozen forever. Only the routed LoRA experts
    and the routers ever train; both are created later with requires_grad=True and
    are (de)activated per phase by freeze_lora_moe_experts / freeze_lora_moe_routers.
    Without this, get_optimizer_grouped_parameters (which filters on requires_grad)
    would happily train the entire backbone every task -- the exact forgetting this
    method is designed to avoid, and it would make phase-2 not router-only.
    """
    layers = model.model.layers
    for layer in layers:
        layer.mlp = LoRAMoEMLP(
            layer.mlp, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
            routing_weight_mode=routing_weight_mode, dropout=dropout)
    for p in model.parameters():  # no experts/routers exist yet -> freezes pure backbone
        p.requires_grad = False
    return model


def _lora_moe_layers(model):
    return [m for m in model.modules() if isinstance(m, LoRAMoEMLP)]


def set_lora_moe_diagnostic_expert(model, expert_index=None):
    for layer in _lora_moe_layers(model):
        layer._diagnostic_expert_index = expert_index
        if expert_index is not None:
            layer._last_route_diagnostic = None


@contextmanager
def quota_lora_moe_expert(model, expert_index, quota_fraction):
    """Training-only context that floors one expert's valid-token share."""
    layers = _lora_moe_layers(model)
    previous = [
        (layer._training_quota_expert_index,
         layer._training_quota_fraction)
        for layer in layers
    ]
    quota_fraction = float(quota_fraction)
    if not 0.0 <= quota_fraction <= 1.0:
        raise ValueError(
            f"quota fraction must be in [0, 1], got {quota_fraction}")
    for layer in layers:
        if not 0 <= expert_index < layer.num_experts:
            raise ValueError(
                f"expert_index={expert_index}, layer experts={layer.num_experts}")
        layer._training_quota_expert_index = int(expert_index)
        layer._training_quota_fraction = quota_fraction
        layer._last_quota_diagnostic = None
    try:
        yield
    finally:
        for layer, (index, fraction) in zip(layers, previous):
            layer._training_quota_expert_index = index
            layer._training_quota_fraction = fraction


@contextmanager
def auxiliary_lora_moe_expert(model, expert_index, mix):
    """Interpolate a new expert into a separate training-only top-1 pass.

    This context does not change router logits or natural selections.  The
    trainer freezes router parameters around the auxiliary forward, so only
    the already-trainable new expert receives its LM gradient.
    """
    layers = _lora_moe_layers(model)
    previous = [
        (layer._training_aux_expert_index, layer._training_aux_mix)
        for layer in layers
    ]
    mix = float(mix)
    if not 0.0 < mix <= 1.0:
        raise ValueError(f"auxiliary expert mix must be in (0, 1], got {mix}")
    for layer in layers:
        if not 0 <= expert_index < layer.num_experts:
            raise ValueError(
                f"expert_index={expert_index}, layer experts={layer.num_experts}")
        layer._training_aux_expert_index = int(expert_index)
        layer._training_aux_mix = mix
        layer._last_aux_diagnostic = None
    try:
        yield
    finally:
        for layer, (index, previous_mix) in zip(layers, previous):
            layer._training_aux_expert_index = index
            layer._training_aux_mix = previous_mix


@contextmanager
def limit_lora_moe_experts(model, active_expert_count):
    """Temporarily expose only the pre-expansion expert/router prefix.

    v2 calls this under ``torch.no_grad()`` to obtain a fixed teacher target.
    Old experts and old router rows remain frozen during KD, so this teacher is
    identical across every KD optimizer step without duplicating the backbone.
    """
    layers = _lora_moe_layers(model)
    previous = [layer._active_expert_count for layer in layers]
    for layer in layers:
        if not 0 <= active_expert_count <= layer.num_experts:
            raise ValueError(
                f"active_expert_count={active_expert_count}, "
                f"layer experts={layer.num_experts}")
        layer._active_expert_count = active_expert_count
    try:
        yield
    finally:
        for layer, value in zip(layers, previous):
            layer._active_expert_count = value


@contextmanager
def force_lora_moe_expert(model, expert_index):
    """Evaluation-only context that dispatches every token to one expert."""
    layers = _lora_moe_layers(model)
    previous = [layer._forced_expert_index for layer in layers]
    for layer in layers:
        if not 0 <= expert_index < layer.num_experts:
            raise ValueError(
                f"expert_index={expert_index}, layer experts={layer.num_experts}")
        layer._forced_expert_index = int(expert_index)
    try:
        yield
    finally:
        for layer, value in zip(layers, previous):
            layer._forced_expert_index = value


def add_experts_to_all_layers(model, n):
    for layer in _lora_moe_layers(model):
        layer.add_experts(n)


def freeze_lora_moe_experts(model, trainable_expert_indices=None):
    """trainable_expert_indices=None freezes every expert; else only those indices train."""
    for layer in _lora_moe_layers(model):
        for i, expert in enumerate(layer.experts):
            requires_grad = trainable_expert_indices is not None and i in trainable_expert_indices
            for p in expert.parameters():
                p.requires_grad = requires_grad


def freeze_lora_moe_routers(model, trainable):
    for layer in _lora_moe_layers(model):
        if layer.router is not None:
            for p in layer.router.parameters():
                p.requires_grad = trainable


def collect_moe_losses(model):
    total = None
    for layer in _lora_moe_layers(model):
        if layer._last_moe_loss is not None:
            total = layer._last_moe_loss if total is None else total + layer._last_moe_loss
    return total


def set_router_token_mask(model, attention_mask):
    """Mask padding in LoRA dispatch and router aux/z objectives.

    Keep the mask installed through backward because gradient checkpointing
    re-runs layer forwards during backward; clear it afterwards with None.
    """
    for layer in _lora_moe_layers(model):
        layer._router_token_mask = attention_mask


# ``_allocate_memory_counts`` is a staticmethod shared by every variant, so the
# recency exponent is published here by the trainer at construction time rather
# than threaded through six call sites.
RECENCY_POWER = [1.0]


class Ours_LoRA_MoE(CL_Base_Model):
    """Per task: (1) grow experts_per_task new experts + train them with the router
    on the new task, then (2) freeze every expert and retune only the router on
    a capped pool of all seen-task fixed subsets, including current data.

    self.raw_model is the plain nn.Module (survives across tasks; add_experts/
    freeze mutate it directly). self.model is a DistributedDataParallel wrapper
    around it (or raw_model itself if not distributed) -- rebuilt via
    _reinit_engine() at the start of EVERY phase, since (a) growing experts
    creates brand-new nn.Parameters a stale DDP wrapper/optimizer never
    registered, and (b) get_optimizer_grouped_parameters filters by
    p.requires_grad, so each phase needs its own optimizer built from the
    freeze pattern that's active *at that moment*. (Previously used
    deepspeed.initialize() here, but repeatedly re-initializing a DeepSpeed
    engine on the same model leaks GPU memory without bound -- confirmed via a
    synthetic repro that grew every re-init regardless of ZeRO stage,
    destroy()+gc.collect(), or bucket size, and eventually OOM'd mid-run. Our
    trainable set is tiny (LoRA experts + router) so ZeRO sharding buys nothing
    anyway; plain DDP + AdamW re-created the same way showed zero growth across
    repeated re-inits.)
    """

    # Save only the grown params (LoRA experts + router), not the frozen 16GB
    # backbone -- the eval loader rebuilds the base from base_model_name_or_path
    # and loads these on top. ".mlp.experts." / ".mlp.router." exclude the frozen
    # dense ".mlp.base_mlp." and everything else in the backbone.
    save_key_substrings = [".mlp.experts.", ".mlp.router."]

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list,
                test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.raw_model = model
        self._dist_initialized = False
        self._fixed_task_subsets = {}
        self._fixed_task_subset_indices = {}
        self._replay_manifest = None
        replay_manifest_path = getattr(args, "replay_manifest_path", "")
        if replay_manifest_path:
            with open(replay_manifest_path, encoding="utf-8") as handle:
                self._replay_manifest = json.load(handle)
            if not isinstance(self._replay_manifest.get("tasks"), dict):
                raise ValueError(
                    f"invalid replay manifest: {replay_manifest_path}")
            self._replay_manifest["_path"] = replay_manifest_path

        self._active_task_workload = None
        self._workload_records = self._load_workload_records()

    def _acquisition_diagnostics_enabled(self):
        return int(getattr(
            self.args, "v2_acquisition_diagnostic_interval", 0)) > 0

    def _acquisition_diagnostic_due(self, update_index):
        interval = int(getattr(
            self.args, "v2_acquisition_diagnostic_interval", 0))
        return interval > 0 and int(update_index) % interval == 0

    def _route_diagnostic_summary(self):
        per_layer = []
        total_tokens = 0
        total_selected = 0
        probability_sum = 0.0
        margin_sum = 0.0
        target_logit_sum = 0.0
        competitor_logit_sum = 0.0
        for layer_index, layer in enumerate(_lora_moe_layers(self.raw_model)):
            item = layer._last_route_diagnostic
            if item is None:
                continue
            tokens = int(item["valid_tokens"])
            selected = int(item["selected_tokens"])
            per_layer.append({
                "layer": layer_index,
                "valid_tokens": tokens,
                "selected_share": selected / max(1, tokens),
                "mean_probability": (
                    item["probability_sum"] / max(1, tokens)),
                "mean_logit_margin": item["margin_sum"] / max(1, tokens),
            })
            total_tokens += tokens
            total_selected += selected
            probability_sum += item["probability_sum"]
            margin_sum += item["margin_sum"]
            target_logit_sum += item["target_logit_sum"]
            competitor_logit_sum += item["competitor_logit_sum"]
        if total_tokens == 0:
            return None
        return {
            "valid_layer_tokens": total_tokens,
            "selected_share": total_selected / total_tokens,
            "mean_probability": probability_sum / total_tokens,
            "mean_logit_margin": margin_sum / total_tokens,
            "mean_target_logit": target_logit_sum / total_tokens,
            "mean_best_competitor_logit": competitor_logit_sum / total_tokens,
            "per_layer": per_layer,
        }

    def _quota_dispatch_summary(self):
        per_layer = []
        totals = {
            "valid_tokens": 0,
            "natural_selected_tokens": 0,
            "dispatched_selected_tokens": 0,
            "injected_tokens": 0,
        }
        quota_fraction = None
        for layer_index, layer in enumerate(_lora_moe_layers(self.raw_model)):
            item = layer._last_quota_diagnostic
            if item is None:
                continue
            layer_item = {"layer": layer_index, **item}
            per_layer.append(layer_item)
            quota_fraction = float(item["quota_fraction"])
            for name in totals:
                totals[name] += int(item[name])
        if not per_layer:
            return None
        valid_tokens = max(1, totals["valid_tokens"])
        return {
            **totals,
            "quota_fraction": quota_fraction,
            "natural_selected_share": (
                totals["natural_selected_tokens"] / valid_tokens),
            "dispatch_selected_share": (
                totals["dispatched_selected_tokens"] / valid_tokens),
            "per_layer": per_layer,
        }

    def _gradient_diagnostic_summary(self, expert_index):
        expert_grad_sq = 0.0
        expert_parameter_sq = 0.0
        router_grad_sq = 0.0
        target_router_grad_sq = 0.0
        old_router_grad_sq = 0.0
        target_router_parameter_sq = 0.0
        for layer in _lora_moe_layers(self.raw_model):
            for parameter in layer.experts[int(expert_index)].parameters():
                expert_parameter_sq += float(
                    parameter.detach().float().square().sum().item())
                if parameter.grad is not None:
                    expert_grad_sq += float(
                        parameter.grad.detach().float().square().sum().item())
            router_parameter = layer.router.weight
            target_router_parameter_sq += float(
                router_parameter[int(expert_index)].detach().float()
                .square().sum().item())
            if router_parameter.grad is None:
                continue
            gradient = router_parameter.grad.detach().float()
            router_grad_sq += float(gradient.square().sum().item())
            target_router_grad_sq += float(
                gradient[int(expert_index)].square().sum().item())
            if expert_index > 0:
                old_router_grad_sq += float(
                    gradient[:int(expert_index)].square().sum().item())
        return {
            "new_expert_grad_norm": math.sqrt(expert_grad_sq),
            "new_expert_parameter_norm": math.sqrt(expert_parameter_sq),
            "router_grad_norm": math.sqrt(router_grad_sq),
            "new_router_row_grad_norm": math.sqrt(target_router_grad_sq),
            "old_router_rows_grad_norm": math.sqrt(old_router_grad_sq),
            "new_router_row_parameter_norm": math.sqrt(
                target_router_parameter_sq),
        }

    def _snapshot_router_gradients(self):
        snapshots = []
        for layer in _lora_moe_layers(self.raw_model):
            gradient = layer.router.weight.grad
            snapshots.append(
                None if gradient is None else gradient.detach().float().clone())
        return snapshots

    def _router_gradient_delta_summary(self, primary_gradients):
        primary_sq = 0.0
        replay_sq = 0.0
        dot = 0.0
        for layer, primary in zip(
                _lora_moe_layers(self.raw_model), primary_gradients):
            combined = layer.router.weight.grad
            if combined is None:
                continue
            combined = combined.detach().float()
            if primary is None:
                primary = torch.zeros_like(combined)
            replay = combined - primary
            primary_sq += float(primary.square().sum().item())
            replay_sq += float(replay.square().sum().item())
            dot += float((primary * replay).sum().item())
        denominator = math.sqrt(primary_sq * replay_sq)
        return {
            "new_router_grad_norm": math.sqrt(primary_sq),
            "replay_router_grad_norm": math.sqrt(replay_sq),
            "new_replay_router_grad_cosine": (
                dot / denominator if denominator > 0 else None),
        }

    def _write_acquisition_diagnostic(self, payload):
        if self.args.global_rank != 0:
            return
        record = {
            "schema_version": 1,
            "scope": "rank0_local_pre_ddp_average",
            **payload,
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(
            self.args.output_dir, "v2_acquisition_diagnostics.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _run_post_kd_acquisition_probe(
            self, primary_loader, device, task, expert_index):
        if not self._acquisition_diagnostics_enabled():
            return
        try:
            source_batch = next(iter(primary_loader))
        except StopIteration:
            raise ValueError("acquisition probe requires non-empty primary data")
        batch = dict(source_batch)
        batch.pop("sources", None)
        batch = to_device(batch, device)
        set_lora_moe_diagnostic_expert(self.raw_model, expert_index)
        was_training = self.raw_model.training
        self.raw_model.eval()
        set_router_token_mask(self.raw_model, batch.get("attention_mask"))
        try:
            with torch.no_grad():
                output = self.raw_model(**batch, use_cache=False)
            self._write_acquisition_diagnostic({
                "task": task,
                "phase": "post_kd_pre_primary_probe",
                "expert_index": int(expert_index),
                "loss": float(output.loss.detach().float().item()),
                "routing": self._route_diagnostic_summary(),
            })
        finally:
            set_router_token_mask(self.raw_model, None)
            set_lora_moe_diagnostic_expert(self.raw_model, None)
            self.raw_model.train(was_training)

    def _load_workload_records(self):
        path = os.path.join(self.args.output_dir, "training_workload.json")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, encoding="utf-8") as handle:
                return list(json.load(handle).get("tasks", []))
        except (OSError, ValueError, TypeError):
            return []

    @staticmethod
    def _empty_workload_role():
        return {
            "input_sample_exposures": 0,
            "input_nonpad_token_exposures": 0,
            "forward_sample_instances": 0,
            "forward_nonpad_token_instances": 0,
            "forward_microbatches": 0,
            "backward_microbatches": 0,
        }

    def _count_workload_batch(self, role, batch, forward_passes=1,
                              backward_passes=1):
        if self._active_task_workload is None:
            return
        mask = batch.get("attention_mask")
        if mask is None:
            samples = int(batch["input_ids"].shape[0])
            tokens = int(batch["input_ids"].numel())
        else:
            samples = int(mask.shape[0])
            tokens = int(mask.detach().sum().item())
        metrics = self._active_task_workload["roles"].setdefault(
            role, self._empty_workload_role())
        metrics["input_sample_exposures"] += samples
        metrics["input_nonpad_token_exposures"] += tokens
        metrics["forward_sample_instances"] += samples * forward_passes
        metrics["forward_nonpad_token_instances"] += tokens * forward_passes
        metrics["forward_microbatches"] += forward_passes
        metrics["backward_microbatches"] += backward_passes

    def _count_workload_update(self):
        if self._active_task_workload is not None:
            self._active_task_workload["optimizer_updates"] += 1

    def _reduce_scalar(self, value, operation="sum"):
        if not torch.distributed.is_initialized():
            return value
        tensor = torch.tensor(
            float(value), dtype=torch.float64,
            device=torch.device("cuda", self.args.local_rank))
        reduce_op = (torch.distributed.ReduceOp.MAX if operation == "max"
                     else torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(tensor, op=reduce_op)
        return tensor.item()

    def _finalize_task_workload(self, elapsed_seconds, local_flops,
                                flop_counter_status):
        workload = self._active_task_workload
        reduced_roles = {}
        for role in sorted(workload["roles"]):
            reduced_roles[role] = {
                key: int(self._reduce_scalar(value))
                for key, value in workload["roles"][role].items()
            }
        workload["roles"] = reduced_roles
        workload["optimizer_updates"] = int(self._reduce_scalar(
            workload["optimizer_updates"], operation="max"))
        workload["training_wall_time_seconds"] = self._reduce_scalar(
            elapsed_seconds, operation="max")
        workload["counted_operator_flops_global"] = (
            int(self._reduce_scalar(local_flops))
            if local_flops is not None else None)
        workload["flop_counter_status"] = flop_counter_status
        workload["flop_counter_scope"] = (
            "PyTorch-supported operator FLOPs summed across ranks; "
            "communication and unsupported operators excluded")
        return workload

    def _write_workload_records(self):
        if self.args.global_rank != 0:
            return
        by_round = {
            int(record["round"]): record for record in self._workload_records}
        tasks = [by_round[index] for index in sorted(by_round)]
        role_totals = {}
        for task in tasks:
            for role, metrics in task["roles"].items():
                total = role_totals.setdefault(role, self._empty_workload_role())
                for key, value in metrics.items():
                    total[key] += value
        flops = [task.get("counted_operator_flops_global") for task in tasks]
        payload = {
            "schema_version": 1,
            "method": f"ours_lora_moe_{self.args.training_version}",
            "world_size": (torch.distributed.get_world_size()
                           if torch.distributed.is_initialized() else 1),
            "tasks": tasks,
            "totals": {
                "roles": role_totals,
                "optimizer_updates": sum(
                    task["optimizer_updates"] for task in tasks),
                "training_wall_time_seconds": sum(
                    task["training_wall_time_seconds"] for task in tasks),
                "counted_operator_flops_global": (
                    sum(flops) if all(value is not None for value in flops)
                    else None),
            },
        }
        os.makedirs(self.args.output_dir, exist_ok=True)
        path = os.path.join(self.args.output_dir, "training_workload.json")
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)

    def train_continual(self):
        """Train every task and persist task/global compute and time totals."""
        for i_task, task in enumerate(self.train_task_list):
            if i_task < getattr(self.args, "start_task", 0):
                print_rank_0(
                    f"Skipping completed task {i_task}: {task}",
                    self.args.global_rank)
                continue
            epochs = int(self.args.num_train_epochs[i_task])
            self._active_task_workload = {
                "round": i_task,
                "task": task,
                "epochs": epochs,
                "router_replay_exposure_budget_global": (
                    self._workload_replay_exposure_budget(epochs)),
                "joint_new_to_replay_sample_ratio": getattr(
                    self.args, "v2_joint_new_to_replay_ratio", 0),
                "roles": {},
                "optimizer_updates": 0,
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            track_flops = not self.args.disable_training_flop_counter
            flop_counter = (
                FlopCounterMode(
                    display=False,
                    custom_mapping=_flop_counter_custom_mapping())
                if track_flops and FlopCounterMode is not None else None)
            flop_status = ("enabled" if flop_counter is not None else
                           "unavailable" if track_flops else "disabled")
            with (flop_counter if flop_counter is not None else nullcontext()):
                self.train_one_task(task, i_task, epochs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            local_flops = (flop_counter.get_total_flops()
                           if flop_counter is not None else None)
            record = self._finalize_task_workload(
                elapsed, local_flops, flop_status)

            save_started = time.perf_counter()
            self.save_model(i_task)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            record["checkpoint_save_wall_time_seconds"] = self._reduce_scalar(
                time.perf_counter() - save_started, operation="max")
            self._workload_records = [
                item for item in self._workload_records
                if int(item["round"]) != i_task]
            self._workload_records.append(record)
            self._write_workload_records()
            self._active_task_workload = None
            if task == getattr(self.args, "stop_after_task", ""):
                print_rank_0(
                    f"Stopping after requested task {task}",
                    self.args.global_rank)
                break

    def _workload_replay_exposure_budget(self, epochs):
        """Declared global replay budget for one continual task round."""
        del epochs
        return int(self.args.router_replay_exposure_samples)

    def _reinit_engine(self, num_training_steps, learning_rate=None):
        args = self.args
        if self._dist_initialized:
            del self.model
            torch.cuda.empty_cache()
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.raw_model, args.weight_decay)
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate if learning_rate is None else learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon)
        warmup_steps = args.num_warmup_steps
        if getattr(args, "warmup_ratio", 0.0) > 0:
            warmup_steps = math.ceil(
                num_training_steps * args.warmup_ratio)
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler_type, self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, num_training_steps))
        if args.local_rank != -1:
            self.model = DDP(self.raw_model, device_ids=[args.local_rank],
                             output_device=args.local_rank, find_unused_parameters=True,
                             broadcast_buffers=False, gradient_as_bucket_view=True)
        else:
            self.model = self.raw_model
        self._dist_initialized = True

    def _set_grad_ckpt(self, enable):
        """Toggle gradient checkpointing on raw_model for the upcoming phase. Pure
        memory/compute tradeoff -- weights are identical either way. Always clears any
        prior input-require-grads hook first so repeated ON transitions across tasks
        don't stack orphaned hooks on the input embeddings."""
        m = self.raw_model
        if getattr(m, "_require_grads_hook", None) is not None:
            m.disable_input_require_grads()
            m._require_grads_hook = None
        if enable:
            # Frozen backbone -> embedding output has requires_grad=False, so a
            # checkpointed segment has no grad_fn; force it on via the input hook.
            m.enable_input_require_grads()
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            m.gradient_checkpointing_disable()
        if self.args.global_rank == 0:
            print_rank_0(f"  [grad_ckpt] {'ON' if enable else 'off'}", self.args.global_rank)

    def _optimizer_update_count(self, dataloader, epochs):
        """Scheduler length in optimizer updates, matching epoch-local accumulation."""
        accum = max(1, self.args.gradient_accumulation_steps)
        return max(1, int(epochs) * math.ceil(len(dataloader) / accum))

    def _gradient_accumulation_for_batch_size(self, batch_size, phase):
        """Return the accumulation contract without mutating phase state."""
        world_size = (torch.distributed.get_world_size()
                      if torch.distributed.is_initialized() else 1)
        denominator = int(batch_size) * world_size
        global_batch = int(self.args.effective_global_batch)
        if global_batch % denominator != 0:
            raise ValueError(
                f"{phase}: effective global batch {global_batch} is not "
                f"divisible by micro-batch {batch_size} * world size {world_size}")
        return global_batch // denominator

    def _set_phase_gradient_accumulation(self, batch_size, phase):
        accumulation = self._gradient_accumulation_for_batch_size(
            batch_size, phase)
        self.args.gradient_accumulation_steps = accumulation
        print_rank_0(
            f"  [batch contract] {phase}: micro_batch={batch_size} "
            f"grad_accum={accumulation} "
            f"effective_global_batch={self.args.effective_global_batch}",
            self.args.global_rank)
        return accumulation

    @staticmethod
    def _allocate_memory_counts(lengths, total, mode):
        """Allocate one global exposure budget without growing with task count."""
        if not lengths or total <= 0:
            return [0] * len(lengths)
        total = int(total)
        if mode == "proportional":
            raw = [total * length / sum(lengths) for length in lengths]
        elif mode == "equal_task":
            raw = [total / len(lengths)] * len(lengths)
        elif mode == "recency_weighted":
            # ``lengths`` is ordered oldest-first, so index+1 is the task's
            # recency rank.  v3_new forgets its two most recent tasks by 7.4
            # and 9.9 points while its oldest loses 0.35, yet equal_task pays
            # them all the same; this tilts the same budget toward the tasks
            # that actually decay.  power 1.0 is a linear ramp; larger values
            # concentrate harder on the newest task.
            power = float(RECENCY_POWER[0])
            weights = [float(index + 1) ** power
                       for index in range(len(lengths))]
            scale = total / sum(weights)
            raw = [weight * scale for weight in weights]
        else:
            raise ValueError(f"unknown replay distribution: {mode}")
        counts = [int(value) for value in raw]
        remainder = total - sum(counts)
        order = sorted(
            range(len(lengths)),
            key=lambda index: (raw[index] - int(raw[index]), lengths[index]),
            reverse=True)
        for index in order[:remainder]:
            counts[index] += 1
        return counts

    def _fixed_subset_seed(self):
        configured = self.args.replay_subset_seed
        return self.args.seed if configured < 0 else configured

    def _ensure_fixed_task_subset(self, task):
        """Create one persistent deterministic 1%/10% subset for a task."""
        if task in self._fixed_task_subsets:
            return self._fixed_task_subsets[task]
        task_names = list(self.train_task_list)
        task_index = task_names.index(task)
        dataset = self.train_task_list[task].dataset
        unique_samples = max(
            1, min(len(dataset), round(
                len(dataset) * self.args.replay_subset_ratio)))
        # Gradient-selected memories are written at task completion. Prefer
        # them on resume so a later process does not silently fall back to the
        # original random manifest for already-scored tasks.
        saved_gradient_entry = None
        if getattr(self.args, "replay_selection_mode", "random") == "router_gradient":
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task)
            saved_path = os.path.join(
                self.args.output_dir, "fixed_replay_memory",
                f"task_{task_index}_{safe_task}.json")
            if os.path.isfile(saved_path):
                with open(saved_path, encoding="utf-8") as handle:
                    candidate = json.load(handle)
                if candidate.get("selection_mode") == "router_gradient":
                    saved_gradient_entry = candidate
        if saved_gradient_entry is not None:
            indices = [int(index) for index in
                       saved_gradient_entry.get("indices", [])]
            if (saved_gradient_entry.get("source_samples") != len(dataset)
                    or len(indices) != unique_samples
                    or len(set(indices)) != len(indices)
                    or any(index < 0 or index >= len(dataset)
                           for index in indices)):
                raise ValueError(
                    f"invalid saved router-gradient memory for {task}: "
                    f"{saved_path}")
            seed = None
        elif getattr(self, "_replay_manifest", None) is not None:
            entry = self._replay_manifest["tasks"].get(task)
            if entry is None:
                raise KeyError(f"replay manifest has no task entry for {task}")
            indices = [int(index) for index in entry.get("indices", [])]
            if entry.get("source_samples") != len(dataset):
                raise ValueError(
                    f"replay source size mismatch for {task}: "
                    f"{entry.get('source_samples')} != {len(dataset)}")
            if len(indices) != unique_samples or len(set(indices)) != len(indices):
                raise ValueError(
                    f"replay manifest for {task} must contain "
                    f"{unique_samples} unique indices, got {len(indices)}")
            if any(index < 0 or index >= len(dataset) for index in indices):
                raise ValueError(f"replay manifest has out-of-range index for {task}")
            seed = int(entry["seed"])
        else:
            seed = self._fixed_subset_seed() + task_index * 1009
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(
                len(dataset), generator=generator)[:unique_samples].tolist()
        subset = Subset(dataset, indices)
        self._fixed_task_subsets[task] = subset
        self._fixed_task_subset_indices[task] = indices

        if self.args.global_rank == 0 and saved_gradient_entry is None:
            digest = hashlib.sha256(
                ",".join(map(str, indices)).encode("utf-8")).hexdigest()
            output_dir = os.path.join(
                self.args.output_dir, "fixed_replay_memory")
            os.makedirs(output_dir, exist_ok=True)
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task)
            metadata = {
                "schema_version": 1,
                "task_index": task_index,
                "task": task,
                "source_samples": len(dataset),
                "subset_ratio": self.args.replay_subset_ratio,
                "unique_samples": unique_samples,
                "seed": seed,
                "selection_mode": "random",
                "indices_sha256": digest,
                "manifest_path": (
                    getattr(self, "_replay_manifest", None) or {}).get("_path"),
                "indices": indices,
                "available_from_next_task_for_v2_past_replay": True,
            }
            with open(os.path.join(
                    output_dir, f"task_{task_index}_{safe_task}.json"),
                    "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
        return subset

    @staticmethod
    def _deterministic_exposure_indices(unique_samples, exposure_samples, seed):
        """Repeat/shuffle a fixed subset until exactly exposure_samples are drawn."""
        if unique_samples <= 0 or exposure_samples <= 0:
            return []
        generator = torch.Generator().manual_seed(seed)
        indices = []
        while len(indices) < exposure_samples:
            permutation = torch.randperm(
                unique_samples, generator=generator).tolist()
            indices.extend(permutation[:exposure_samples - len(indices)])
        return indices

    def _build_fixed_memory_loader(self, task_names, round_index, phase,
                                   exposure_samples=None, batch_size=None,
                                   active_unique_counts=None,
                                   stream_seed_phase=None,
                                   deterministic_pass_sampler=False):
        """Build replay only from accumulated task-fixed subsets.

        ``exposure_samples=None`` exposes each unique memory item once; callers
        such as joint replay may cycle that loader. An integer builds an exact
        total exposure stream distributed across tasks, repeating small subsets
        as needed without increasing the total budget. ``active_unique_counts``
        optionally exposes stable prefixes of the persisted task memories.  A
        caller can therefore shrink a task's active memory in later rounds
        without replacing any sample that was active in an earlier round.
        """
        if not task_names:
            return None
        stored_subsets = [
            self._ensure_fixed_task_subset(task) for task in task_names]
        stored_unique_counts = [len(subset) for subset in stored_subsets]
        if active_unique_counts is None:
            active_unique_counts = list(stored_unique_counts)
        else:
            active_unique_counts = [
                int(count) for count in active_unique_counts]
            if len(active_unique_counts) != len(stored_subsets):
                raise ValueError(
                    "active memory counts must match the number of tasks: "
                    f"{len(active_unique_counts)} != {len(stored_subsets)}")
            for task, active, stored in zip(
                    task_names, active_unique_counts, stored_unique_counts):
                if not 0 <= active <= stored:
                    raise ValueError(
                        f"invalid active memory prefix for {task}: "
                        f"{active}/{stored}")
        subsets = [
            subset if active == stored else Subset(subset, range(active))
            for subset, active, stored in zip(
                stored_subsets, active_unique_counts, stored_unique_counts)
        ]
        unique_counts = [len(subset) for subset in subsets]
        if exposure_samples is None:
            exposure_counts = list(unique_counts)
        else:
            exposure_counts = self._allocate_memory_counts(
                unique_counts, int(exposure_samples),
                self.args.replay_distribution)

        seed_phase = stream_seed_phase or phase
        phase_seed = sum(ord(character) for character in seed_phase)
        stream_datasets = []
        plan_tasks = []
        ordered_identities = []
        base_seed = self._fixed_subset_seed() + round_index * 100003 + phase_seed
        for offset, (task, subset, stored_count, unique_count,
                     exposure_count) in enumerate(zip(
                         task_names, subsets, stored_unique_counts,
                         unique_counts, exposure_counts)):
            if exposure_count > 0:
                exposure_indices = self._deterministic_exposure_indices(
                    unique_count, exposure_count, base_seed + offset * 1009)
                stream_datasets.append(RepeatedSubsetDataset(
                    subset, exposure_indices))
                stored_indices = self._fixed_task_subset_indices[task]
                ordered_identities.extend(
                    f"{task}:{stored_indices[index]}"
                    for index in exposure_indices)
            plan_tasks.append({
                "task": task,
                "stored_unique_samples": stored_count,
                "active_unique_samples": unique_count,
                "unique_samples": unique_count,
                "exposure_samples": exposure_count,
            })
        if not stream_datasets:
            return None
        combined = ConcatDataset(stream_datasets)
        if len(ordered_identities) != len(combined):
            raise RuntimeError(
                "fixed-memory identity plan length mismatch: "
                f"{len(ordered_identities)}/{len(combined)}")
        if batch_size is None or batch_size <= 0:
            batch_size = min(self.args.batch_by_task[task] for task in task_names)
        max_train_len = getattr(self.args, "max_train_len", 0)
        if getattr(self.args, "train_format", "raw_answer") == "slora_chat_full":
            if getattr(self.args, "use_pretokenized_train_cache", False):
                collator = PreTokenizedSLoRATraceDataCollator(self.tokenizer)
            else:
                collator = SLoRATraceDataCollator(
                    self.tokenizer, max_length=(max_train_len or (
                        self.args.max_prompt_len + self.args.max_ans_len)))
        else:
            collator = DataCollator(
                self.tokenizer, padding="longest",
                max_prompt_len=(max_train_len or self.args.max_prompt_len),
                max_ans_len=(0 if max_train_len else self.args.max_ans_len),
                pad_to_multiple_of=8, inference=False)
        if self.args.local_rank == -1:
            sampler = (
                DeterministicPassRandomSampler(combined, seed=base_seed)
                if deterministic_pass_sampler else RandomSampler(combined))
        else:
            sampler = DistributedSampler(
                combined, shuffle=True, seed=base_seed)
        loader = DataLoader(
            combined, collate_fn=collator, sampler=sampler,
            batch_size=batch_size, num_workers=4, pin_memory=True)
        ordered_identity_sha256 = hashlib.sha256(
            "\n".join(ordered_identities).encode("utf-8")).hexdigest()
        loader._lora_moe_memory_stream = {
            "round": round_index,
            "phase": phase,
            "stream_seed_phase": seed_phase,
            "planned_exposure_samples": len(combined),
            "ordered_identity_sha256": ordered_identity_sha256,
            "active_unique_counts": list(unique_counts),
            "sampler_base_seed": base_seed,
            "sampler_type": type(sampler).__name__,
            "sampler_pass_contract": (
                "seed_plus_pass" if hasattr(sampler, "set_epoch") else
                "legacy_sampler_state"),
        }

        if self.args.global_rank == 0:
            plan_dir = os.path.join(self.args.output_dir, "replay_plans")
            os.makedirs(plan_dir, exist_ok=True)
            plan = {
                "schema_version": 1,
                "round": round_index,
                "phase": phase,
                "stream_seed_phase": seed_phase,
                "subset_ratio_per_task": self.args.replay_subset_ratio,
                "distribution": self.args.replay_distribution,
                "stored_unique_pool_samples": sum(stored_unique_counts),
                "active_unique_pool_samples": sum(unique_counts),
                "unique_pool_samples": sum(unique_counts),
                "planned_exposure_samples": sum(exposure_counts),
                "ordered_identity_sha256": ordered_identity_sha256,
                "sampler_base_seed": base_seed,
                "sampler_type": type(sampler).__name__,
                "sampler_pass_contract": (
                    "seed_plus_pass" if hasattr(sampler, "set_epoch") else
                    "legacy_sampler_state"),
                "batch_size_per_rank": batch_size,
                "tasks": plan_tasks,
            }
            with open(os.path.join(
                    plan_dir, f"round_{round_index}_{phase}.json"),
                    "w", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2)
        return loader

    @staticmethod
    def _snapshot_old_router_rows(model, old_expert_count):
        return [
            layer.router.weight[:old_expert_count].detach().clone()
            for layer in _lora_moe_layers(model)
        ]

    @staticmethod
    def _freeze_old_router_row_update(model, snapshots, old_expert_count):
        """Keep a router prefix bit-identical across an optimizer update."""
        for layer, snapshot in zip(_lora_moe_layers(model), snapshots):
            if layer.router.weight.grad is not None:
                layer.router.weight.grad[:old_expert_count].zero_()
            with torch.no_grad():
                layer.router.weight[:old_expert_count].copy_(snapshot)

    def train_one_task(self, task, i_task, epochs):
        device = torch.device("cuda", self.args.local_rank) if self.args.local_rank != -1 else torch.device("cuda")
        args = self.args

        # Each task contributes one immutable 1%/10% memory subset. v1 may use
        # the current task immediately in its post-training router phase; v2
        # exposes it only to later tasks as past memory.
        self._ensure_fixed_task_subset(task)
        old_expert_count = len(_lora_moe_layers(self.raw_model)[0].experts)
        add_experts_to_all_layers(self.raw_model, args.experts_per_task)
        new_indices = set(range(
            old_expert_count, old_expert_count + args.experts_per_task))
        old_router_rows = (
            self._snapshot_old_router_rows(self.raw_model, old_expert_count)
            if old_expert_count > 0 else None)

        # Phase 1: checkpointing per the task's survey (short=off/fast, long=on).
        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} phase1")
        self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
        freeze_lora_moe_experts(self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=True)
        self._reinit_engine(self._optimizer_update_count(
            self.train_task_list[task], epochs))
        self._run_epochs(
            self.train_task_list[task], epochs, device,
            f"{task} [phase1 new-expert+new-router-row]",
            frozen_router_prefix=(old_expert_count, old_router_rows)
            if old_router_rows is not None else None)

        if args.router_retune_epochs > 0:
            replay_loader = self._build_replay_loader(i_task)
            if replay_loader is not None:
                # v1 is a two-phase method: after the current-task phase, its
                # router retune uses every seen fixed subset, including the
                # current task. A long sequence may land in any batch, so
                # checkpoint the whole router-retune phase.
                self._set_grad_ckpt(True)
                freeze_lora_moe_experts(
                    self.raw_model, trainable_expert_indices=None)
                freeze_lora_moe_routers(self.raw_model, trainable=True)
                self._set_phase_gradient_accumulation(
                    replay_loader.batch_size, f"{task} phase2 router retune")
                # The replay stream itself contains exactly the configured
                # global exposure budget, so it is consumed once. For a first
                # 5,000-sample task this is the 50-item 1% subset repeated 20
                # times, i.e. exactly 1,000 router-finetune exposures.
                self._reinit_engine(self._optimizer_update_count(
                    replay_loader, 1))
                self._run_epochs(
                    replay_loader, 1, device,
                    f"{task} [phase2 router retune exact seen replay]")

    def _run_epochs(self, dataloader, epochs, device, phase_name,
                    frozen_router_prefix=None, include_moe_loss=True,
                    workload_role=None):
        args = self.args
        total_steps = epochs * len(dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(f"{phase_name}: epoch {epoch+1}/{epochs}, {len(dataloader)} steps",
                        args.global_rank)
            self.model.train()
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(dataloader):
                active_workload_role = workload_role or (
                    "router_replay" if "phase2 router retune" in phase_name
                    else "new_task")
                self._count_workload_batch(active_workload_role, batch)
                del batch['sources']
                batch = to_device(batch, device)
                accum_steps = args.gradient_accumulation_steps
                window_start = (step // accum_steps) * accum_steps
                window_size = min(accum_steps, len(dataloader) - window_start)
                is_window_end = (step - window_start + 1) == window_size
                sync_context = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not is_window_end
                    else nullcontext())
                set_router_token_mask(self.raw_model, batch.get("attention_mask"))
                try:
                    with sync_context:
                        outputs = self.model(**batch, use_cache=False)
                        moe_loss = (
                            collect_moe_losses(self.raw_model)
                            if include_moe_loss else None)
                        loss = outputs.loss if moe_loss is None else outputs.loss + moe_loss
                        (loss / window_size).backward()
                finally:
                    set_router_token_mask(self.raw_model, None)
                if args.global_rank == 0:
                    progress_bar.update(1)
                    if (step % args.loss_log_interval == 0 or
                            step + 1 == len(dataloader)):
                        progress_bar.set_description(
                            f"{phase_name} | epoch {epoch+1} step {step} "
                            f"loss {loss.detach().float().cpu().item():.4f}",
                            refresh=False)
                if is_window_end:
                    if frozen_router_prefix is not None:
                        old_count, snapshots = frozen_router_prefix
                        self._freeze_old_router_row_update(
                            self.raw_model, snapshots, old_count)
                    trainable_params = [
                        p for p in self.raw_model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    self.optimizer.step()
                    if frozen_router_prefix is not None:
                        self._freeze_old_router_row_update(
                            self.raw_model, snapshots, old_count)
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()

    def save_model(self, round):
        # Persist the HF-format checkpoint (base_mlp + experts + router are all in
        # the state_dict) plus a lora_moe_meta.json so load_lora_moe_checkpoint can
        # rebuild the exact expert-pool size / hyperparameters at eval time.
        super().save_model(round)
        if self.args.global_rank == 0:
            output_dir = os.path.join(self.args.output_dir, str(round))
            extra = {
                "training_version": getattr(self.args, "training_version", "v1"),
                "experts_per_task": self.args.experts_per_task,
                "training_profile": {
                    "format": self.args.train_format,
                    "max_length": self.args.max_train_len or (
                        self.args.max_prompt_len + self.args.max_ans_len),
                    "adam_beta1": self.args.adam_beta1,
                    "adam_beta2": self.args.adam_beta2,
                    "adam_epsilon": self.args.adam_epsilon,
                },
                "replay_memory": {
                    "subset_ratio_per_task": self.args.replay_subset_ratio,
                    "exposure_samples_per_round":
                        self.args.router_replay_exposure_samples,
                    "v1_router_retune_enabled":
                        self.args.router_retune_epochs > 0,
                    "distribution": self.args.replay_distribution,
                    "subset_seed": self.args.replay_subset_seed,
                },
            }
            if _uses_v2_new_memory(extra["training_version"]):
                persistent_samples = int(getattr(
                    self.args, "v2_new_persistent_samples_per_task", 500))
                active_stream_samples = int(getattr(
                    self.args, "v2_new_active_memory_cap", 1000))
                memory_records = getattr(
                    self, "_v2_new_persistent_memory_records", {})
                sampler_pass_digests = getattr(
                    self, "_v2_new_sampler_pass_digests", {})
                persisted_identities = {
                    task: {
                        "resolved_seed": int(record["resolved_seed"]),
                        "indices_sha256": record["indices_sha256"],
                    }
                    for task, record in memory_records.items()
                }
                extra["replay_memory"] = {
                    "persistent_samples_per_task": persistent_samples,
                    "persistent_selection_mode": getattr(
                        self.args, "replay_selection_mode", "random"),
                    "configured_subset_seed": self.args.replay_subset_seed,
                    "resolved_base_subset_seed": self._fixed_subset_seed(),
                    "persisted_identities": persisted_identities,
                    "active_stream_samples_per_primary_epoch":
                        active_stream_samples,
                    "distribution": self.args.replay_distribution,
                    "v1_router_retune_enabled": (
                        extra["training_version"] == "v1_expert_first"
                        and self.args.router_retune_epochs > 0),
                }
            if extra["training_version"] in (
                    "v2", "v2_5", "v2_new", "v2_new_top4",
                    "v1_expert_first"):
                is_v2_new = _is_v2_new_training_version(
                    extra["training_version"])
                extra["v2"] = {
                    "memory_batch_size": self.args.v2_memory_batch_size,
                    "kd_memory_batch_size": getattr(
                        self.args, "v2_kd_memory_batch_size", 0),
                    "effective_kd_memory_batch_size": (
                        getattr(self.args, "v2_kd_memory_batch_size", 0)
                        or self.args.v2_memory_batch_size or 1),
                    "effective_replay_memory_batch_size": (
                        self.args.v2_memory_batch_size or 1),
                    "replay_forward_batch_size": int(getattr(
                        self.args, "v2_replay_forward_batch_size", 1)),
                    "kd_exposure_samples_per_round":
                        self.args.router_replay_exposure_samples,
                    "kd_loss_coeff": self.args.v2_kd_loss_coeff,
                    "kd_temperature": self.args.v2_kd_temperature,
                    "kd_learning_rate": self.args.v2_kd_learning_rate,
                    "kd_chunk_tokens": self.args.v2_kd_chunk_tokens,
                    "kd_token_scope": self.args.v2_kd_token_scope,
                    "joint_replay_loss_coeff": self.args.v2_joint_replay_loss_coeff,
                    "joint_replay_objective": self._joint_replay_objective(),
                    "hidden_mse_loss_coeff": float(getattr(
                        self.args, "v2_hidden_mse_loss_coeff", 1.0)),
                    "hidden_mse_teacher": "expanded_post_kd_init",
                    "hidden_mse_targets": "all_decoder_layer_outputs",
                    "hidden_mse_reduction":
                        "active_sample_mean_equal_layer_mean",
                    "joint_new_to_replay_sample_ratio": getattr(
                        self.args, "v2_joint_new_to_replay_ratio", 0),
                    "stored_replay_pool_exposures_per_round":
                        self.args.router_replay_exposure_samples,
                    "joint_replay_schedule":
                        ("every_optimizer_update_active_stream_per_primary_epoch"
                         if is_v2_new else
                         "every_optimizer_update_fixed_total_no_epoch_multiplier"),
                    "joint_replay_reduction": "active_sample_mean",
                    "joint_replay_forward_reduction":
                        "packed_per_sample_token_mean_then_sum",
                    "max_replay_batches_per_step":
                        self.args.v2_max_replay_batches_per_step,
                }
                if is_v2_new:
                    kd_pass_multiplier = int(getattr(
                        self.args, "v2_kd_pass_multiplier", 1))
                    extra["v2"].pop("kd_exposure_samples_per_round")
                    extra["v2"].pop(
                        "stored_replay_pool_exposures_per_round")
                    extra["v2"].update({
                        "kd_active_stream_samples_per_pass":
                            active_stream_samples,
                        "kd_active_stream_passes":
                            "match_primary_epochs",
                        "kd_total_exposure_strategy":
                            "samples_per_pass_times_primary_epochs",
                        "joint_replay_active_stream_samples_per_primary_epoch":
                            active_stream_samples,
                        "joint_replay_total_exposure_strategy":
                            "samples_per_primary_epoch_times_primary_epochs",
                    })
                    aux_mix = float(getattr(
                        self.args, "v2_new_expert_aux_mix", 0.0))
                    if aux_mix > 0:
                        extra["v2"].update({
                            "new_expert_auxiliary_route":
                                "detached_router_hard_top1_interpolation",
                            "new_expert_aux_mix": aux_mix,
                            "new_expert_aux_loss_coeff": float(getattr(
                                self.args,
                                "v2_new_expert_aux_loss_coeff", 1.0)),
                            "new_expert_aux_optimizer_schedule":
                                "same_primary_update_single_optimizer_step",
                        })
                    extra["v2_new"] = {
                        "persistent_subset_policy":
                            "validated_output_dir_json_else_deterministic_random",
                        "persistent_samples_per_task": persistent_samples,
                        "persistent_memory_integrity":
                            "source_count_unique_range_sha256",
                        "persistent_memory_resume_source":
                            "output_dir/fixed_replay_memory",
                        "configured_subset_seed":
                            self.args.replay_subset_seed,
                        "resolved_base_subset_seed":
                            self._fixed_subset_seed(),
                        "persisted_identities": persisted_identities,
                        "selection_mode": getattr(
                            self.args, "replay_selection_mode", "random"),
                        "active_memory_cap_unique": int(getattr(
                            self.args, "v2_new_active_memory_cap", 1000)),
                        "active_memory_distribution": "equal_task",
                        "active_memory_selection":
                            "stable_nested_task_prefix",
                        "active_stream_samples_per_pass":
                            active_stream_samples,
                        "active_stream_identity_order":
                            "shared_between_kd_and_replay",
                        "sampler_pass_order_contract":
                            "shared_seed_plus_pass_rank_local_order",
                        "sampler_pass_order_sha256":
                            sampler_pass_digests,
                        "active_stream_seed_phase":
                            "v2_new_shared_active_memory",
                        "kd_stream_passes": "match_primary_epochs",
                        "joint_replay_stream_passes":
                            "match_primary_epochs",
                    }
                    if kd_pass_multiplier != 1:
                        extra["v2"].update({
                            "kd_active_stream_passes":
                                "primary_epochs_times_multiplier",
                            "kd_pass_multiplier": kd_pass_multiplier,
                            "kd_total_exposure_strategy":
                                ("samples_per_pass_times_primary_epochs_"
                                 "times_multiplier"),
                        })
                        extra["v2_new"].update({
                            "kd_stream_passes":
                                "primary_epochs_times_multiplier",
                            "kd_stream_pass_multiplier":
                                kd_pass_multiplier,
                        })
                elif extra["training_version"] == "v1_expert_first":
                    kd_pass_multiplier = int(getattr(
                        self.args, "v2_kd_pass_multiplier", 1))
                    extra["v2"] = {
                        "memory_batch_size": self.args.v2_memory_batch_size,
                        "kd_memory_batch_size": getattr(
                            self.args, "v2_kd_memory_batch_size", 0),
                        "effective_kd_memory_batch_size": (
                            getattr(self.args, "v2_kd_memory_batch_size", 0)
                            or self.args.v2_memory_batch_size or 1),
                        "kd_loss_coeff": self.args.v2_kd_loss_coeff,
                        "kd_temperature": self.args.v2_kd_temperature,
                        "kd_learning_rate": self.args.v2_kd_learning_rate,
                        "kd_chunk_tokens": self.args.v2_kd_chunk_tokens,
                        "kd_token_scope": self.args.v2_kd_token_scope,
                        "kd_active_stream_samples_per_pass":
                            active_stream_samples,
                        "kd_active_stream_passes":
                            "primary_epochs_times_multiplier",
                        "kd_pass_multiplier": kd_pass_multiplier,
                        "router_ft_schedule":
                            "post_expert_training_router_only",
                        "router_ft_seen_memory_exposures":
                            active_stream_samples,
                    }
                    extra["v2_new"] = {
                        "persistent_subset_policy":
                            "validated_output_dir_json_else_deterministic_random",
                        "persistent_samples_per_task": persistent_samples,
                        "persistent_memory_integrity":
                            "source_count_unique_range_sha256",
                        "persistent_memory_resume_source":
                            "output_dir/fixed_replay_memory",
                        "configured_subset_seed":
                            self.args.replay_subset_seed,
                        "resolved_base_subset_seed":
                            self._fixed_subset_seed(),
                        "persisted_identities": persisted_identities,
                        "selection_mode": getattr(
                            self.args, "replay_selection_mode", "random"),
                        "active_memory_cap_unique": int(getattr(
                            self.args, "v2_new_active_memory_cap", 1000)),
                        "active_memory_distribution": "equal_task",
                        "active_memory_selection":
                            "stable_nested_task_prefix",
                        "active_stream_samples_per_pass":
                            active_stream_samples,
                        "active_stream_seed_phase":
                            "v2_new_shared_active_memory",
                        "kd_stream_passes":
                            "primary_epochs_times_multiplier",
                        "kd_stream_pass_multiplier": kd_pass_multiplier,
                        "router_ft_stream_passes": "one_seen_task_stream",
                    }
            save_lora_moe_meta(self.raw_model, output_dir, extra=extra)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_replay_loader(self, i_task):
        """v1: exactly one fixed global exposure budget over all seen tasks.

        Every 5,000-sample task contributes one immutable 1% (50-record)
        subset. The current task is included, counts are allocated equally
        across all seen tasks, and small subsets repeat deterministically until
        the per-round global exposure total is exactly the configured budget.
        """
        args = self.args
        replay_tasks = list(self.train_task_list)[:i_task + 1]
        replay_batch_size = min(
            args.batch_by_task[name] for name in replay_tasks)
        return self._build_fixed_memory_loader(
            replay_tasks, round_index=i_task,
            phase="v1_router_retune",
            exposure_samples=args.router_replay_exposure_samples,
            batch_size=replay_batch_size)



class Ours_LoRA_MoE_V2(Ours_LoRA_MoE):
    """KD-initialized growth plus fixed-pool joint router replay.

    For task t>0, KD and joint replay draw from the same deterministic 1%
    subsets of strictly past tasks. KD consumes the deterministic 1,000-sample
    global stream once. Joint replay cycles that same stored stream as needed
    to enforce the configured new:replay sample ratio on every optimizer
    update. A replay backward freezes every expert and adds only router
    gradients before the normal shared update.
    """

    def train_one_task(self, task, i_task, epochs):
        device = (
            torch.device("cuda", self.args.local_rank)
            if self.args.local_rank != -1 else torch.device("cuda")
        )
        args = self.args
        self._ensure_fixed_task_subset(task)
        old_expert_count = len(_lora_moe_layers(self.raw_model)[0].experts)
        add_experts_to_all_layers(self.raw_model, args.experts_per_task)
        new_indices = set(range(
            old_expert_count, old_expert_count + args.experts_per_task))

        # Separate DataLoader instances use the same phase seed, subset indices
        # and exposure allocation, so KD and router replay consume the same
        # deterministic 1,000-record stream. KD may batch that stream more
        # densely; replay keeps its own fine-grained interleaving batch size.
        primary_loader = self.train_task_list[task]
        kd_loader = self._build_v2_kd_loader(
            i_task, primary_loader, epochs)
        replay_loader = self._build_v2_replay_loader(
            i_task, primary_loader, epochs)
        ran_kd_init = (
            old_expert_count > 0 and kd_loader is not None
            and args.v2_kd_loss_coeff > 0)
        if ran_kd_init:
            self._run_v2_kd_init(
                kd_loader, old_expert_count, new_indices, device, task,
                kd_epochs=self._v2_kd_epochs(epochs))

        if self._acquisition_diagnostics_enabled():
            if len(new_indices) != 1:
                raise ValueError(
                    "V2 acquisition diagnostics currently require exactly "
                    "one newly added expert")
            self._run_post_kd_acquisition_probe(
                primary_loader, device, task, next(iter(new_indices)))

        hidden_mse_teacher = None
        if (replay_loader is not None
                and self._joint_replay_objective() == "hidden_mse"):
            if not ran_kd_init:
                raise RuntimeError(
                    "hidden-MSE replay requires a completed expansion KD-init "
                    "before the post-KD teacher snapshot")
            hidden_mse_teacher = self._make_post_kd_hidden_mse_teacher()

        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} v2 primary")

        # One optimizer owns the new experts and every router row. Every
        # optimizer update also receives a past-data backward with all experts
        # frozen, so its router-only gradient is combined with the new-data
        # gradient before the shared step.
        self._set_grad_ckpt(
            replay_loader is not None
            or task in getattr(args, "ckpt_tasks", set()))
        freeze_lora_moe_experts(
            self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=True)
        self._reinit_engine(self._optimizer_update_count(
            primary_loader, epochs))
        try:
            if replay_loader is None:
                self._run_epochs(
                    primary_loader, epochs, device,
                    f"{task} [v2 primary-only; no prior memory]")
            else:
                objective = self._joint_replay_objective()
                self._run_v2_joint_epochs(
                    primary_loader, replay_loader, epochs, device,
                    f"{task} [v2 joint every-update router "
                    f"{objective} replay]",
                    hidden_mse_teacher=hidden_mse_teacher)
        finally:
            if hidden_mse_teacher is not None:
                del hidden_mse_teacher
                torch.cuda.empty_cache()

    def _memory_task_names(self, i_task):
        # Current data already supplies expert+router gradients. It enters the
        # strictly past replay/KD pool only from the next round.
        return list(self.train_task_list)[:i_task]

    def _build_v2_kd_loader(self, i_task, primary_loader, epochs):
        """Hook for V2 variants that derive KD length from primary training."""
        del primary_loader, epochs
        return self._build_v2_memory_loader(i_task, role="kd")

    def _build_v2_replay_loader(self, i_task, primary_loader, epochs):
        """Hook for V2 variants that use a distinct replay-memory stream."""
        del primary_loader, epochs
        return self._build_v2_memory_loader(i_task, role="replay")

    def _v2_kd_epochs(self, primary_epochs):
        """Number of complete memory-stream passes used for KD initialization."""
        del primary_epochs
        return 1

    def _set_v2_kd_memory_sampler_pass(
            self, dataloader, pass_index):
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(pass_index)

    def _set_v2_replay_memory_sampler_pass(
            self, dataloader, pass_index):
        # Legacy V2 deliberately restarts the same replay permutation whenever
        # its fixed stream cycles. V2-new overrides this pass policy.
        del pass_index
        sampler = getattr(dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)

    def _validate_v2_replay_memory_sampler_passes(
            self, dataloader, completed_passes, primary_epochs):
        del dataloader, completed_passes, primary_epochs

    def _build_v2_memory_loader(self, i_task, role="replay",
                                exposure_samples=None):
        args = self.args
        if role not in {"kd", "replay"}:
            raise ValueError(f"unknown v2 memory-loader role: {role}")
        task_names = self._memory_task_names(i_task)
        if not task_names:
            return None
        # Batch size one per rank spreads replay broadly. KD can use a larger
        # microbatch without changing its exact 1,000-record exposure stream.
        memory_batch_size = args.v2_memory_batch_size or 1
        if role == "kd":
            memory_batch_size = (
                getattr(args, "v2_kd_memory_batch_size", 0)
                or memory_batch_size)
        if exposure_samples is None:
            exposure_samples = args.router_replay_exposure_samples
        return self._build_fixed_memory_loader(
            task_names, round_index=i_task,
            phase="v2_shared_exact_memory",
            exposure_samples=exposure_samples,
            batch_size=memory_batch_size)

    def _joint_replay_exposure_budget(self, primary_loader, epochs):
        """Return the global replay budget without multiplying by epochs.

        The ratio is defined against one pass over the new-task dataset.  New
        data may be trained for several epochs, but the fixed replay pool is
        exposed only once across the complete joint phase.
        """
        ratio = int(getattr(
            self.args, "v2_joint_new_to_replay_ratio", 0))
        if ratio == 0:
            return int(self.args.router_replay_exposure_samples)
        sampler = getattr(primary_loader, "sampler", None)
        samples_per_epoch = getattr(sampler, "total_size", None)
        if samples_per_epoch is None:
            dataset = getattr(primary_loader, "dataset", None)
            if dataset is None:
                raise ValueError(
                    "sample-ratio replay requires a primary loader dataset")
            samples_per_epoch = len(dataset)
        new_samples = int(samples_per_epoch)
        if new_samples % ratio != 0:
            raise ValueError(
                "fixed joint replay budget requires the one-epoch global "
                f"new-sample count to be divisible by {ratio}: "
                f"{new_samples} % {ratio} != 0")
        return new_samples // ratio

    def _expected_joint_replay_exposures(
            self, consumed_global_new_exposures, epochs, replay_ratio):
        """Validate legacy V2's one-epoch replay-ratio contract."""
        if consumed_global_new_exposures % epochs != 0:
            raise RuntimeError(
                "joint phase new exposure count is not divisible by "
                f"epochs: new={consumed_global_new_exposures}, "
                f"epochs={epochs}")
        one_epoch_new = consumed_global_new_exposures // epochs
        if one_epoch_new % replay_ratio != 0:
            raise RuntimeError(
                "one-epoch new sample count does not match the fixed "
                f"replay ratio: new={one_epoch_new}, "
                f"ratio={replay_ratio}:1")
        return one_epoch_new // replay_ratio


    @staticmethod
    def _kd_kl_loss(student_logits, teacher_logits, batch, args):
        if args.v2_kd_token_scope == "labels":
            mask = batch["labels"].ne(-100)
        else:
            mask = batch["attention_mask"].bool()
        student = student_logits[mask]
        teacher = teacher_logits[mask]
        if student.numel() == 0:
            return student_logits.sum() * 0.0
        temperature = args.v2_kd_temperature
        total = student.new_zeros((), dtype=torch.float)
        chunk_tokens = max(1, args.v2_kd_chunk_tokens)
        for start in range(0, student.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, student.shape[0])
            student_log_probs = F.log_softmax(
                student[start:stop].float() / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(
                teacher[start:stop].float() / temperature, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            total = total + torch.sum(
                teacher_probs * (teacher_log_probs - student_log_probs))
        return total * (temperature ** 2) / student.shape[0]

    def _run_v2_kd_init(self, dataloader, old_expert_count, new_indices,
                        device, task, kd_epochs=1):
        args = self.args
        kd_epochs = int(kd_epochs)
        if kd_epochs < 1:
            raise ValueError(f"KD epochs must be positive, got {kd_epochs}")
        self._set_phase_gradient_accumulation(
            dataloader.batch_size, f"{task} v2 KD init")
        self._set_grad_ckpt(True)
        freeze_lora_moe_experts(
            self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=True)
        # Each pass consumes the exact loader exposure stream.  Legacy V2 uses
        # one pass; variants may deliberately repeat the fixed active stream.
        total_microsteps = kd_epochs * len(dataloader)
        updates = self._optimizer_update_count(dataloader, kd_epochs)
        kd_lr = args.v2_kd_learning_rate or args.learning_rate
        self._reinit_engine(updates, learning_rate=kd_lr)
        old_router_rows = self._snapshot_old_router_rows(
            self.raw_model, old_expert_count)
        progress = tqdm(
            total=total_microsteps, leave=True,
            disable=args.global_rank != 0)
        self.optimizer.zero_grad(set_to_none=True)
        completed_microsteps = 0
        completed_updates = 0
        consumed_local_samples = 0
        for kd_epoch in range(kd_epochs):
            self._set_v2_kd_memory_sampler_pass(dataloader, kd_epoch)
            for step, source_batch in enumerate(dataloader):
                completed_microsteps += 1
                consumed_local_samples += int(
                    source_batch["input_ids"].shape[0])
                # The same KD record is evaluated once by the frozen teacher
                # and once by the expanded student, then backpropagated once.
                self._count_workload_batch(
                    "kd_init", source_batch, forward_passes=2,
                    backward_passes=1)
                batch = dict(source_batch)
                batch.pop("sources", None)
                batch = to_device(batch, device)
                accum = max(1, args.gradient_accumulation_steps)
                window_start = (step // accum) * accum
                window_size = min(accum, len(dataloader) - window_start)
                window_index = step - window_start
                should_step = window_index + 1 == window_size
                diagnostic_due = (
                    should_step
                    and self._acquisition_diagnostic_due(completed_updates))
                sync = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not should_step
                    else nullcontext())
                set_router_token_mask(
                    self.raw_model, batch.get("attention_mask"))
                try:
                    self.raw_model.eval()
                    with torch.no_grad(), limit_lora_moe_experts(
                            self.raw_model, old_expert_count):
                        teacher_logits = self.raw_model(
                            **batch, use_cache=False).logits.detach()
                    self.model.train()
                    if diagnostic_due:
                        set_lora_moe_diagnostic_expert(
                            self.raw_model, min(new_indices))
                    with sync:
                        student_logits = self.model(
                            **batch, use_cache=False).logits
                        kd_loss = self._kd_kl_loss(
                            student_logits, teacher_logits, batch, args)
                        loss = args.v2_kd_loss_coeff * kd_loss
                        (loss / window_size).backward()
                finally:
                    set_router_token_mask(self.raw_model, None)
                    if diagnostic_due:
                        set_lora_moe_diagnostic_expert(
                            self.raw_model, None)
                if args.global_rank == 0:
                    progress.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(dataloader)):
                        global_step = kd_epoch * len(dataloader) + step
                        progress.set_description(
                            f"{task} [v2 KD-init "
                            f"{len(dataloader.dataset)} samples x "
                            f"{kd_epochs} epochs] s{global_step} "
                            f"kl={kd_loss.detach().float().item():.5f}",
                            refresh=False)
                if should_step:
                    if diagnostic_due:
                        self._write_acquisition_diagnostic({
                            "task": task,
                            "phase": "kd_init",
                            "kd_epoch": kd_epoch,
                            "optimizer_update": completed_updates,
                            "expert_index": min(new_indices),
                            "loss": float(
                                kd_loss.detach().float().item()),
                            "routing": self._route_diagnostic_summary(),
                            "gradients": self._gradient_diagnostic_summary(
                                min(new_indices)),
                        })
                    # Old rows define the fixed pre-expansion teacher. Zero
                    # their gradients and restore exact values after AdamW.
                    self._freeze_old_router_row_update(
                        self.raw_model, old_router_rows, old_expert_count)
                    trainable = [
                        p for p in self.raw_model.parameters()
                        if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    self._freeze_old_router_row_update(
                        self.raw_model, old_router_rows, old_expert_count)
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()
                    completed_updates += 1
        expected_local_samples = kd_epochs * len(dataloader.sampler)
        if completed_microsteps != total_microsteps:
            raise RuntimeError(
                f"V2 KD microsteps {completed_microsteps}/{total_microsteps}")
        if completed_updates != updates:
            raise RuntimeError(
                f"V2 KD optimizer updates {completed_updates}/{updates}")
        if consumed_local_samples != expected_local_samples:
            raise RuntimeError(
                "V2 KD local sample exposure mismatch: "
                f"{consumed_local_samples}/{expected_local_samples}")
        progress.close()

    def _joint_replay_objective(self):
        return str(getattr(
            self.args, "v2_joint_replay_objective", "lm"))

    def _release_phase_engine_for_teacher_snapshot(self):
        """Drop KD optimizer/DDP state before cloning the post-KD teacher."""
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        self.optimizer = None
        self.lr_scheduler = None
        if getattr(self, "_dist_initialized", False):
            del self.model
            self.model = self.raw_model
            self._dist_initialized = False
        # Remove the input-require-grad hook before deepcopy. The primary
        # phase's _set_grad_ckpt call recreates it on the student only.
        if getattr(self.raw_model, "_require_grads_hook", None) is not None:
            self.raw_model.disable_input_require_grads()
            self.raw_model._require_grads_hook = None
        self.raw_model.gradient_checkpointing_disable()
        for module in self.raw_model.modules():
            for name in (
                    "_last_moe_loss", "_last_probe_indices",
                    "_last_route_diagnostic", "_last_quota_diagnostic",
                    "_last_aux_diagnostic", "_routing_context"):
                if hasattr(module, name):
                    setattr(module, name, None)
            if hasattr(module, "_sample_grad_scores"):
                module._sample_grad_scores = []
        torch.cuda.empty_cache()

    def _make_post_kd_hidden_mse_teacher(self):
        """Freeze an exact expanded-model snapshot immediately after KD-init."""
        self._release_phase_engine_for_teacher_snapshot()
        teacher = copy.deepcopy(self.raw_model)
        teacher.eval()
        teacher.gradient_checkpointing_disable()
        if getattr(teacher, "_require_grads_hook", None) is not None:
            teacher.disable_input_require_grads()
            teacher._require_grads_hook = None
        for parameter in teacher.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        print_rank_0(
            "  [hidden-MSE replay] frozen teacher = expanded post-KD-init "
            "model; targets = every decoder-layer output",
            self.args.global_rank)
        return teacher

    @staticmethod
    @contextmanager
    def _capture_decoder_layer_outputs(model):
        decoder = getattr(model, "model", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise TypeError(
                "hidden-MSE replay expects a causal LM with model.layers")
        captured = [None] * len(layers)
        handles = []

        def make_hook(index):
            def hook(_module, _inputs, output):
                captured[index] = output[0] if isinstance(
                    output, (tuple, list)) else output
            return hook

        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(index)))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _per_sample_layer_hidden_mse(
            student_layers, teacher_layers, attention_mask):
        """Equal-layer mean of each sample's valid-token hidden-state MSE."""
        if len(student_layers) != len(teacher_layers) or not student_layers:
            raise ValueError(
                "student/teacher hidden layer counts must match and be nonzero")
        valid = attention_mask.bool()
        valid_counts = valid.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("hidden-MSE replay record has no valid token")
        sample_losses = None
        for index, (student, teacher) in enumerate(zip(
                student_layers, teacher_layers)):
            if student is None or teacher is None:
                raise RuntimeError(
                    f"decoder layer {index} did not emit a captured output")
            if student.shape != teacher.shape or student.shape[:2] != valid.shape:
                raise ValueError(
                    f"hidden-MSE layer {index} shape mismatch: student="
                    f"{tuple(student.shape)} teacher={tuple(teacher.shape)} "
                    f"mask={tuple(valid.shape)}")
            token_mse = (student.float() - teacher.float()).square().mean(dim=-1)
            layer_sample = (
                (token_mse * valid).sum(dim=1)
                / valid_counts.to(token_mse.dtype))
            sample_losses = (
                layer_sample if sample_losses is None
                else sample_losses + layer_sample)
        return sample_losses / len(student_layers)

    def _hidden_mse_replay_losses(self, teacher, replay):
        teacher_inputs = dict(replay)
        teacher_inputs.pop("labels", None)
        with torch.no_grad(), self._capture_decoder_layer_outputs(
                teacher) as teacher_layers:
            teacher.model(**teacher_inputs, use_cache=False, return_dict=True)
        with self._capture_decoder_layer_outputs(
                self.raw_model) as student_layers:
            # Bypass lm_head: the objective consumes decoder-layer outputs,
            # so materializing [batch, tokens, vocabulary] logits would waste
            # substantial memory without changing the loss.
            self.raw_model.model(
                **teacher_inputs, use_cache=False, return_dict=True)
        return self._per_sample_layer_hidden_mse(
            student_layers, teacher_layers, replay["attention_mask"])

    @contextmanager
    def _router_only_replay(self):
        expert_parameters = [
            parameter
            for layer in _lora_moe_layers(self.raw_model)
            for expert in layer.experts
            for parameter in expert.parameters()
        ]
        previous = [parameter.requires_grad for parameter in expert_parameters]
        for parameter in expert_parameters:
            parameter.requires_grad = False
        try:
            yield
        finally:
            for parameter, requires_grad in zip(expert_parameters, previous):
                parameter.requires_grad = requires_grad

    @contextmanager
    def _suppress_replay_router_losses(self):
        """Skip router aux/z bookkeeping only for replay LM forwards."""
        layers = _lora_moe_layers(self.raw_model)
        previous = [layer._suppress_router_loss for layer in layers]
        for layer in layers:
            layer._suppress_router_loss = True
        try:
            yield
        finally:
            for layer, value in zip(layers, previous):
                layer._suppress_router_loss = value

    @contextmanager
    def _router_frozen(self):
        router_parameters = [
            parameter
            for layer in _lora_moe_layers(self.raw_model)
            for parameter in layer.router.parameters()
        ]
        previous = [parameter.requires_grad for parameter in router_parameters]
        for parameter in router_parameters:
            parameter.requires_grad = False
        try:
            yield
        finally:
            for parameter, requires_grad in zip(router_parameters, previous):
                parameter.requires_grad = requires_grad

    @staticmethod
    def _manual_average_gradients(model, bucket_bytes=32 * 1024 * 1024):
        """Average local joint gradients with a few coalesced collectives.

        The joint V2 path deliberately performs every backward under DDP
        ``no_sync`` because only a subset of ranks may receive replay records.
        The old implementation then issued one all-reduce per trainable
        tensor.  Flattening same-device/same-dtype gradients into bounded
        buckets preserves the elementwise world mean while avoiding hundreds
        of tiny NCCL launches per optimizer update.
        """
        if not torch.distributed.is_initialized():
            return
        if bucket_bytes < 1:
            raise ValueError(
                f"gradient synchronization bucket must be positive, got "
                f"{bucket_bytes}")
        world_size = torch.distributed.get_world_size()
        dense_groups = {}
        sparse_gradients = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            gradient = parameter.grad
            if gradient.is_sparse:
                sparse_gradients.append(gradient)
            else:
                dense_groups.setdefault(
                    (gradient.device, gradient.dtype), []).append(gradient)

        for gradients in dense_groups.values():
            bucket = []
            bucket_size = 0
            for gradient in gradients:
                gradient_bytes = gradient.numel() * gradient.element_size()
                if bucket and bucket_size + gradient_bytes > bucket_bytes:
                    Ours_LoRA_MoE_V2._all_reduce_gradient_bucket(
                        bucket, world_size)
                    bucket = []
                    bucket_size = 0
                bucket.append(gradient)
                bucket_size += gradient_bytes
            if bucket:
                Ours_LoRA_MoE_V2._all_reduce_gradient_bucket(
                    bucket, world_size)

        # LoRA/router gradients are dense, but retain the old behavior for an
        # unexpected sparse trainable parameter rather than densifying it.
        for gradient in sparse_gradients:
            torch.distributed.all_reduce(gradient)
            gradient.div_(world_size)

    @staticmethod
    def _all_reduce_gradient_bucket(gradients, world_size):
        if not gradients:
            return
        flat = torch._utils._flatten_dense_tensors(gradients)
        torch.distributed.all_reduce(flat)
        flat.div_(world_size)
        for gradient, averaged in zip(
                gradients,
                torch._utils._unflatten_dense_tensors(flat, gradients)):
            gradient.copy_(averaged)

    @staticmethod
    def _valid_token_count(batch):
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            raise ValueError(
                "v2 token-ratio replay requires an attention_mask")
        return int(attention_mask.sum().item())

    @staticmethod
    def _replay_exposure_assignment(
            total_exposures, total_updates, update_index, world_size, rank):
        """Assign an exact global replay budget to every optimizer update.

        Exposure ids are spread by cumulative floor and round-robin rank.  If
        ``total_exposures >= total_updates`` every update gets a non-empty
        global replay gradient, while the complete stream is consumed once.
        """
        if total_updates < 1 or not 0 <= update_index < total_updates:
            raise ValueError("invalid replay update index")
        if total_exposures < total_updates:
            raise ValueError(
                "every-update replay needs at least one global exposure per "
                f"optimizer update: {total_exposures} < {total_updates}")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed replay rank")
        start = update_index * total_exposures // total_updates
        stop = (update_index + 1) * total_exposures // total_updates
        local_count = sum(
            exposure_index % world_size == rank
            for exposure_index in range(start, stop))
        return start, stop, int(local_count)

    @staticmethod
    def _ratio_replay_exposure_assignment(
            new_exposures_before, new_exposures_after, ratio,
            world_size, rank):
        """Assign replay ids from cumulative new-sample exposure boundaries."""
        if ratio < 1:
            raise ValueError("joint new:replay ratio must be positive")
        if not 0 <= new_exposures_before < new_exposures_after:
            raise ValueError("invalid cumulative new-sample exposure interval")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid distributed replay rank")
        start = new_exposures_before // ratio
        stop = new_exposures_after // ratio
        if stop <= start:
            raise ValueError(
                "every-update replay ratio produced an empty replay update: "
                f"new=[{new_exposures_before}, {new_exposures_after}), "
                f"ratio={ratio}:1")
        local_count = sum(
            exposure_index % world_size == rank
            for exposure_index in range(start, stop))
        return start, stop, int(local_count)

    @staticmethod
    def _replay_loss_scale(world_size, global_replay_count):
        """Keep replay coefficient semantics independent of active ranks.

        Manual gradient synchronization divides by ``world_size``.  Sparse
        replay only has ``global_replay_count`` non-zero rank gradients, so
        scaling those local losses by world/count makes the synchronized result
        the mean replay gradient over the records assigned to this update.
        """
        if global_replay_count < 1:
            raise ValueError(
                "global replay count must be positive, got "
                f"{global_replay_count}")
        return world_size / global_replay_count

    def _merge_replay_batches(self, batches):
        """Extend loader records without moving their existing token positions.

        These inputs have already been collated as MB=1 tensors.  Any leading
        padding is therefore part of the original layout; extending only on
        the right preserves every existing token index and causal target.
        """
        if not batches:
            raise ValueError("cannot merge an empty replay batch list")
        if len(batches) == 1:
            return dict(batches[0])
        max_length = max(batch["input_ids"].shape[1] for batch in batches)
        merged = {}
        tensor_keys = set.intersection(*[
            {key for key, value in batch.items() if torch.is_tensor(value)}
            for batch in batches
        ])
        for key in tensor_keys:
            values = []
            if key == "input_ids":
                pad_value = self.tokenizer.pad_token_id
            elif key == "labels":
                pad_value = -100
            else:
                pad_value = 0
            for batch in batches:
                value = batch[key]
                if value.ndim != 2:
                    raise ValueError(
                        f"unsupported replay tensor shape for {key}: "
                        f"{tuple(value.shape)}")
                pad_length = max_length - value.shape[1]
                values.append(F.pad(
                    value, (0, pad_length), value=pad_value))
            merged[key] = torch.cat(values, dim=0)
        merged["sources"] = [
            source for batch in batches
            for source in batch.get("sources", [])
        ]
        return merged

    @staticmethod
    def _per_sample_causal_lm_losses(logits, labels, ignore_index=-100):
        """Return the original batch-size-one CE for every packed record.

        Hugging Face's causal-LM loss averages over all valid tokens in a
        batch.  Directly increasing replay batch size would therefore weight
        long records more heavily than the historical MB=1 loop.  Computing
        each row's token mean explicitly keeps the joint replay objective
        exactly as a mean over sample losses while sharing the expensive
        backbone forward.
        """
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError(
                "packed replay expects logits [batch, tokens, vocab] and "
                f"labels [batch, tokens], got {tuple(logits.shape)} and "
                f"{tuple(labels.shape)}")
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                "packed replay logits/labels shape mismatch: "
                f"{tuple(logits.shape[:2])} != {tuple(labels.shape)}")
        shift_labels = F.pad(
            labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
        vocab_size = logits.shape[-1]
        shift_labels = shift_labels.to(logits.device)
        valid_mask = shift_labels.ne(ignore_index)
        valid_counts = valid_mask.sum(dim=1)
        token_losses = F.cross_entropy(
            logits.float().reshape(-1, vocab_size),
            shift_labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="none").view_as(shift_labels)
        return ((token_losses * valid_mask).sum(dim=1)
                / valid_counts.to(token_losses.dtype))

    def _run_v2_joint_epochs(
            self, primary_loader, memory_loader, epochs, device, phase_name,
            hidden_mse_teacher=None):
        """Pair every optimizer update with an exact-budget replay gradient."""
        args = self.args
        replay_objective = self._joint_replay_objective()
        if replay_objective not in {"lm", "hidden_mse"}:
            raise ValueError(
                f"unsupported V2 joint replay objective: {replay_objective}")
        if ((replay_objective == "hidden_mse")
                != (hidden_mse_teacher is not None)):
            raise ValueError(
                "hidden-MSE replay objective and post-KD teacher must be "
                "enabled together")
        quota_schedule = list(getattr(
            args, "v2_new_expert_quota_schedule", []) or [])
        if quota_schedule:
            if not _is_v2_new_training_version(args.training_version):
                raise ValueError("expert quota schedule is V2-new only")
            if args.experts_per_task != 1 or args.top_k != 1:
                raise ValueError(
                    "expert quota schedule requires one added expert and top_k=1")
            if len(quota_schedule) < epochs:
                raise ValueError(
                    "expert quota schedule must cover every primary epoch: "
                    f"{len(quota_schedule)} < {epochs}")
        aux_mix = float(getattr(
            args, "v2_new_expert_aux_mix", 0.0))
        aux_loss_coeff = float(getattr(
            args, "v2_new_expert_aux_loss_coeff", 1.0))
        if aux_mix > 0:
            if not _is_v2_new_training_version(args.training_version):
                raise ValueError("expert auxiliary branch is V2-new only")
            if args.experts_per_task != 1 or args.top_k != 1:
                raise ValueError(
                    "expert auxiliary branch requires one added expert and "
                    "top_k=1")
            if args.routing_weight_mode != "straight_through_topk":
                raise ValueError(
                    "expert auxiliary branch requires straight-through top-1")
            if aux_loss_coeff <= 0:
                raise ValueError(
                    "expert auxiliary loss coefficient must be positive")
            if quota_schedule:
                raise ValueError(
                    "expert auxiliary branch and quota schedule cannot overlap")
        total_steps = epochs * len(primary_loader)
        accum = max(1, args.gradient_accumulation_steps)
        total_updates = epochs * math.ceil(len(primary_loader) / accum)
        stored_memory_exposures = len(memory_loader.dataset)
        if total_steps < 1 or stored_memory_exposures < 1:
            raise ValueError("v2 joint training requires non-empty loaders")
        replay_batch_size = getattr(memory_loader, "batch_size", None)
        if replay_batch_size != 1:
            raise ValueError(
                "the exact replay identity stream must yield one record at a "
                f"time before forward packing, got loader batch "
                f"{replay_batch_size}")
        replay_forward_batch_size = max(1, int(getattr(
            args, "v2_replay_forward_batch_size", 1)))
        distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        replay_ratio = int(getattr(
            args, "v2_joint_new_to_replay_ratio", 0))
        total_memory_exposures = (
            self._joint_replay_exposure_budget(primary_loader, epochs)
            if replay_ratio > 0 else stored_memory_exposures)
        if stored_memory_exposures % world_size != 0:
            raise ValueError(
                "stored replay pool must be divisible by world size: "
                f"{stored_memory_exposures} % {world_size} != 0")
        if total_memory_exposures % world_size != 0:
            raise ValueError(
                "joint replay exposure target must be divisible by world size: "
                f"{total_memory_exposures} % {world_size} != 0")
        expected_local_exposures = total_memory_exposures // world_size
        expected_local_pool_size = stored_memory_exposures // world_size
        if len(memory_loader) != expected_local_pool_size:
            raise ValueError(
                "replay loader does not expose one sharded sample per batch: "
                f"len={len(memory_loader)}, expected={expected_local_pool_size}")
        # The complete fixed budget is spread over the complete multi-epoch
        # phase.  This preserves replay on every optimizer update without
        # repeating the 1,000-record exposure budget once per epoch.
        self._replay_exposure_assignment(
            total_memory_exposures, total_updates, 0, world_size, rank)
        progress = tqdm(
            total=total_steps, leave=True, disable=args.global_rank != 0)

        memory_sampler = getattr(memory_loader, "sampler", None)
        memory_pass_index = 0
        self._set_v2_replay_memory_sampler_pass(
            memory_loader, memory_pass_index)
        memory_iterator = iter(memory_loader)
        consumed_global_exposures = 0
        consumed_local_exposures = 0
        global_update = 0
        total_primary_tokens = 0
        total_replay_tokens = 0
        total_replay_steps = 0
        consumed_global_new_exposures = 0
        cumulative_quota_routed_tokens = 0
        diagnostic_expert = len(
            _lora_moe_layers(self.raw_model)[0].experts) - 1
        # Round 0 has no competing expert and already routes every token to the
        # sole new expert. Preserve the exact baseline there; auxiliary
        # exposure starts only after a previous expert exists.
        aux_enabled = aux_mix > 0 and diagnostic_expert > 0

        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            quota_fraction = (
                float(quota_schedule[epoch])
                if quota_schedule else 0.0)
            primary_sampler = getattr(primary_loader, "sampler", None)
            if hasattr(primary_sampler, "set_epoch"):
                primary_sampler.set_epoch(epoch)
            epoch_primary_tokens = 0
            epoch_replay_tokens = 0
            epoch_replay_steps = 0
            epoch_quota_totals = {
                "valid_tokens": 0,
                "natural_selected_tokens": 0,
                "dispatched_selected_tokens": 0,
                "injected_tokens": 0,
            }

            for step, source_batch in enumerate(primary_loader):
                local_primary_samples = int(source_batch["input_ids"].shape[0])
                global_primary_samples = local_primary_samples * world_size
                consumed_global_new_exposures += global_primary_samples
                primary_tokens = self._valid_token_count(source_batch)
                epoch_primary_tokens += primary_tokens
                total_primary_tokens += primary_tokens
                self._count_workload_batch(
                    "new_task", source_batch,
                    forward_passes=(
                        2 if quota_fraction > 0 or aux_enabled else 1),
                    backward_passes=(
                        2 if quota_fraction > 0 or aux_enabled else 1))
                primary = dict(source_batch)
                primary.pop("sources", None)
                primary = to_device(primary, device)
                window_start = (step // accum) * accum
                window_size = min(accum, len(primary_loader) - window_start)
                window_index = step - window_start
                should_step = window_index + 1 == window_size
                diagnostic_due = (
                    should_step
                    and self._acquisition_diagnostic_due(global_update))
                replay_log_due = (
                    should_step
                    and (diagnostic_due
                         or step % args.loss_log_interval == 0
                         or step + 1 == len(primary_loader)))
                if diagnostic_due:
                    set_lora_moe_diagnostic_expert(
                        self.raw_model, diagnostic_expert)

                # Both backward calls remain local; the combined gradient is
                # explicitly averaged once at the accumulation boundary.
                no_sync = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) else nullcontext())
                quota_expert_loss = None
                quota_summary = None
                auxiliary_expert_loss = None
                natural_routing_summary = None
                if quota_fraction > 0:
                    # Branch A: reproduce the ordinary natural-routing router
                    # objective while blocking every expert parameter.
                    with self._router_only_replay():
                        set_router_token_mask(
                            self.raw_model, primary.get("attention_mask"))
                        try:
                            with no_sync:
                                primary_outputs = self.model(
                                    **primary, use_cache=False)
                                moe_loss = collect_moe_losses(self.raw_model)
                                primary_loss = primary_outputs.loss
                                if moe_loss is not None:
                                    primary_loss = primary_loss + moe_loss
                                (primary_loss / window_size).backward()
                            if diagnostic_due:
                                natural_routing_summary = (
                                    self._route_diagnostic_summary())
                        finally:
                            set_router_token_mask(self.raw_model, None)
                            if diagnostic_due:
                                set_lora_moe_diagnostic_expert(
                                    self.raw_model, None)

                    # Branch B: use the same new batch, freeze every router
                    # row, and train only the newly-added expert under the
                    # quota dispatch. Aux/z are deliberately not collected.
                    with self._router_frozen(), quota_lora_moe_expert(
                            self.raw_model, diagnostic_expert,
                            quota_fraction):
                        set_router_token_mask(
                            self.raw_model, primary.get("attention_mask"))
                        try:
                            quota_outputs = self.raw_model(
                                **primary, use_cache=False)
                            quota_expert_loss = quota_outputs.loss
                            (quota_expert_loss / window_size).backward()
                            quota_summary = self._quota_dispatch_summary()
                        finally:
                            set_router_token_mask(self.raw_model, None)
                    if quota_summary is None:
                        raise RuntimeError(
                            "quota branch did not emit dispatch diagnostics")
                    for name in epoch_quota_totals:
                        epoch_quota_totals[name] += int(quota_summary[name])
                else:
                    # Exact V2-new main path.  The opt-in auxiliary branch is a
                    # second backward below; this forward, its router gradient,
                    # and its ordinary new-expert gradient remain unchanged.
                    set_router_token_mask(
                        self.raw_model, primary.get("attention_mask"))
                    try:
                        with no_sync:
                            primary_outputs = self.model(
                                **primary, use_cache=False)
                            moe_loss = collect_moe_losses(self.raw_model)
                            primary_loss = primary_outputs.loss
                            if moe_loss is not None:
                                primary_loss = primary_loss + moe_loss
                            (primary_loss / window_size).backward()
                    finally:
                        set_router_token_mask(self.raw_model, None)
                        if diagnostic_due:
                            set_lora_moe_diagnostic_expert(
                                self.raw_model, None)

                    if aux_enabled:
                        # The same new-task batch is evaluated with the hard
                        # natural top-1 adapter slot interpolated toward the
                        # newly-added expert. Router parameters are frozen and
                        # LoRAMoEMLP omits aux/z, so this real LM loss can add
                        # gradient only to the already-trainable new expert.
                        with self._router_frozen(), auxiliary_lora_moe_expert(
                                self.raw_model, diagnostic_expert, aux_mix):
                            set_router_token_mask(
                                self.raw_model,
                                primary.get("attention_mask"))
                            try:
                                auxiliary_outputs = self.raw_model(
                                    **primary, use_cache=False)
                                auxiliary_expert_loss = auxiliary_outputs.loss
                                (aux_loss_coeff * auxiliary_expert_loss
                                 / window_size).backward()
                            finally:
                                set_router_token_mask(self.raw_model, None)

                primary_gradient_summary = None
                primary_router_gradients = None
                if diagnostic_due:
                    primary_gradient_summary = (
                        self._gradient_diagnostic_summary(
                            diagnostic_expert))
                    primary_router_gradients = (
                        self._snapshot_router_gradients())

                replay_loss = None
                replay_loss_for_log = None
                replay_token_count = 0
                global_replay_count = 0
                replay_loss_scale = None
                if should_step:
                    exposure_start, exposure_stop, local_replay_count = (
                        self._replay_exposure_assignment(
                            total_memory_exposures, total_updates,
                            global_update, world_size, rank))
                    global_replay_count = exposure_stop - exposure_start
                    replay_loss_scale = self._replay_loss_scale(
                        world_size, global_replay_count)
                    if (args.v2_max_replay_batches_per_step > 0
                            and global_replay_count
                            > args.v2_max_replay_batches_per_step):
                        raise RuntimeError(
                            "every-update replay needs "
                            f"{global_replay_count} global samples on one "
                            "update, above --v2_max_replay_batches_per_step="
                            f"{args.v2_max_replay_batches_per_step}")
                    replay_source_batches = []
                    for _ in range(local_replay_count):
                        try:
                            replay_source_batch = next(memory_iterator)
                        except StopIteration:
                            # The persisted/global pool remains exactly 1,000
                            # samples. Ratio mode cycles that same deterministic
                            # shard rather than storing a larger replay stream.
                            memory_pass_index += 1
                            self._set_v2_replay_memory_sampler_pass(
                                memory_loader, memory_pass_index)
                            memory_iterator = iter(memory_loader)
                            replay_source_batch = next(memory_iterator)
                        replay_source_batches.append(replay_source_batch)

                    local_replay_loss_sum = None
                    for chunk_start in range(
                            0, local_replay_count,
                            replay_forward_batch_size):
                        replay_source_chunk = replay_source_batches[
                            chunk_start:
                            chunk_start + replay_forward_batch_size]
                        merged_replay_source = self._merge_replay_batches(
                            replay_source_chunk)
                        shifted_valid_counts = (
                            merged_replay_source["labels"][:, 1:]
                            .ne(-100).sum(dim=1))
                        if torch.any(shifted_valid_counts == 0):
                            raise ValueError(
                                "packed replay record has no supervised "
                                "causal-LM token")
                        replay_token_count += self._valid_token_count(
                            merged_replay_source)
                        self._count_workload_batch(
                            "router_replay", merged_replay_source,
                            forward_passes=(
                                2 if replay_objective == "hidden_mse" else 1))
                        replay = dict(merged_replay_source)
                        replay.pop("sources", None)
                        replay = to_device(replay, device)
                        replay_labels = replay.pop("labels")
                        with (self._router_only_replay(),
                              self._suppress_replay_router_losses()):
                            set_router_token_mask(
                                self.raw_model, replay.get("attention_mask"))
                            try:
                                # Only ranks assigned real replay records run
                                # this forward.  Each row keeps its historical
                                # MB=1 token-mean CE; summing those row losses
                                # before the scaled backward preserves the
                                # exact global active-sample mean objective.
                                if replay_objective == "hidden_mse":
                                    replay_sample_losses = (
                                        self._hidden_mse_replay_losses(
                                            hidden_mse_teacher, replay))
                                else:
                                    replay_output = self.raw_model(
                                        **replay, use_cache=False)
                                    replay_sample_losses = (
                                        self._per_sample_causal_lm_losses(
                                            replay_output.logits,
                                            replay_labels))
                                replay_loss = replay_sample_losses.sum()
                                detached_loss_sum = (
                                    replay_sample_losses.detach().float().sum())
                                local_replay_loss_sum = (
                                    detached_loss_sum
                                    if local_replay_loss_sum is None else
                                    local_replay_loss_sum + detached_loss_sum)
                                replay_coefficient = (
                                    args.v2_hidden_mse_loss_coeff
                                    if replay_objective == "hidden_mse" else
                                    args.v2_joint_replay_loss_coeff)
                                (replay_coefficient
                                 * replay_loss_scale * replay_loss).backward()
                            finally:
                                set_router_token_mask(self.raw_model, None)
                        consumed_local_exposures += len(replay_source_chunk)

                    # Loss statistics do not affect training.  Synchronize
                    # them only when a progress log or diagnostic consumes
                    # them instead of imposing an extra barrier every update.
                    if replay_log_due:
                        if local_replay_loss_sum is None:
                            local_replay_loss_sum = torch.zeros(
                                (), device=device, dtype=torch.float32)
                        replay_stats = torch.stack((
                            local_replay_loss_sum,
                            local_replay_loss_sum.new_tensor(
                                float(local_replay_count))))
                        if distributed:
                            torch.distributed.all_reduce(replay_stats)
                        replay_loss_for_log = (
                            replay_stats[0]
                            / replay_stats[1].clamp_min(1.0))
                    consumed_global_exposures = exposure_stop
                    epoch_replay_tokens += replay_token_count
                    total_replay_tokens += replay_token_count
                    epoch_replay_steps += 1
                    total_replay_steps += 1

                if diagnostic_due:
                    self._write_acquisition_diagnostic({
                        "task": phase_name.split(" [", 1)[0],
                        "phase": "joint_primary_replay",
                        "epoch": epoch,
                        "optimizer_update": global_update,
                        "expert_index": diagnostic_expert,
                        "new_loss": float(
                            primary_loss.detach().float().item()),
                        "quota_expert_loss": (
                            None if quota_expert_loss is None else float(
                                quota_expert_loss.detach().float().item())),
                        "quota_fraction": quota_fraction,
                        "auxiliary_expert_loss": (
                            None if auxiliary_expert_loss is None else float(
                                auxiliary_expert_loss.detach().float().item())),
                        "auxiliary_expert_mix": (
                            aux_mix if aux_enabled else 0.0),
                        "auxiliary_expert_loss_coeff": (
                            aux_loss_coeff if aux_enabled else 0.0),
                        "replay_loss": (
                            None if replay_loss_for_log is None else float(
                                replay_loss_for_log.detach().float().item())),
                        "routing": (
                            natural_routing_summary
                            if natural_routing_summary is not None
                            else self._route_diagnostic_summary()),
                        "quota_dispatch": quota_summary,
                        "gradients_after_new": primary_gradient_summary,
                        "router_gradient_pair": (
                            self._router_gradient_delta_summary(
                                primary_router_gradients)),
                    })

                if args.global_rank == 0:
                    progress.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(primary_loader)):
                        replay_text = (
                            f"{replay_loss_for_log.item():.4f}"
                            if replay_loss_for_log is not None else "accum")
                        replay_scale_text = (
                            f"{replay_loss_scale:.3f}"
                            if replay_loss_scale is not None else "-")
                        quota_text = (
                            f"quota={quota_expert_loss.detach().float().item():.4f} "
                            if quota_expert_loss is not None else "")
                        auxiliary_text = (
                            "aux="
                            f"{auxiliary_expert_loss.detach().float().item():.4f} "
                            f"mix={aux_mix:.3f} "
                            if auxiliary_expert_loss is not None else "")
                        progress.set_description(
                            f"{phase_name} e{epoch + 1} s{step} "
                            f"new={primary_loss.detach().float().item():.4f} "
                            f"{quota_text}"
                            f"{auxiliary_text}"
                            f"replay={replay_text} n={global_replay_count} "
                            f"scale={replay_scale_text} "
                            f"budget={consumed_global_exposures}/"
                            f"{total_memory_exposures}",
                            refresh=False)
                if should_step:
                    self._manual_average_gradients(self.raw_model)
                    trainable = [
                        p for p in self.raw_model.parameters()
                        if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()
                    global_update += 1

            quota_epoch_tensor = torch.tensor([
                float(epoch_quota_totals[name])
                for name in (
                    "valid_tokens", "natural_selected_tokens",
                    "dispatched_selected_tokens", "injected_tokens")
            ], device=device, dtype=torch.float64)
            if distributed:
                torch.distributed.all_reduce(quota_epoch_tensor)
            quota_epoch_values = [
                int(value) for value in quota_epoch_tensor.cpu().tolist()]
            cumulative_quota_routed_tokens += quota_epoch_values[2]
            if quota_fraction > 0:
                valid_layer_tokens = max(1, quota_epoch_values[0])
                self._write_acquisition_diagnostic({
                    "task": phase_name.split(" [", 1)[0],
                    "phase": "quota_epoch_summary",
                    "epoch": epoch,
                    "expert_index": diagnostic_expert,
                    "quota_fraction": quota_fraction,
                    "valid_layer_tokens": quota_epoch_values[0],
                    "natural_selected_tokens": quota_epoch_values[1],
                    "dispatched_selected_tokens": quota_epoch_values[2],
                    "injected_tokens": quota_epoch_values[3],
                    "natural_selected_share": (
                        quota_epoch_values[1] / valid_layer_tokens),
                    "dispatch_selected_share": (
                        quota_epoch_values[2] / valid_layer_tokens),
                    "cumulative_quota_routed_tokens": (
                        cumulative_quota_routed_tokens),
                })
            if args.global_rank == 0:
                quota_epoch_text = (
                    "quota="
                    f"{quota_epoch_values[2] / max(1, quota_epoch_values[0]):.4f} "
                    f"natural={quota_epoch_values[1] / max(1, quota_epoch_values[0]):.4f} "
                    f"injected={quota_epoch_values[3]}, "
                    if quota_fraction > 0 else "")
                print_rank_0(
                    f"{phase_name} epoch {epoch + 1}: valid tokens "
                    f"new={epoch_primary_tokens}, replay={epoch_replay_tokens}, "
                    f"replay_updates={epoch_replay_steps}, "
                    f"{quota_epoch_text}"
                    f"global_exposures={consumed_global_exposures}/"
                    f"{total_memory_exposures}",
                    args.global_rank)

        if global_update != total_updates:
            raise RuntimeError(
                f"v2 joint updates {global_update}/{total_updates}")
        if consumed_global_exposures != total_memory_exposures:
            raise RuntimeError(
                "v2 exact replay schedule consumed "
                f"{consumed_global_exposures}/{total_memory_exposures} "
                "global exposures")
        if replay_ratio > 0:
            expected_replay = self._expected_joint_replay_exposures(
                consumed_global_new_exposures, epochs, replay_ratio)
            if expected_replay != total_memory_exposures:
                raise RuntimeError(
                    "joint replay loader budget does not match actual primary "
                    f"exposure: replay={total_memory_exposures}, "
                    f"expected={expected_replay}")
        if consumed_local_exposures != expected_local_exposures:
            raise RuntimeError(
                "v2 local replay shard consumed "
                f"{consumed_local_exposures}/{expected_local_exposures}")
        self._validate_v2_replay_memory_sampler_passes(
            memory_loader, memory_pass_index + 1, epochs)
        if replay_ratio == 0:
            try:
                next(memory_iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "v2 replay stream has unconsumed local records")
        if args.global_rank == 0:
            print_rank_0(
                f"{phase_name} complete: new_tokens={total_primary_tokens}, "
                f"replay_tokens={total_replay_tokens}, "
                f"token_ratio={total_primary_tokens / max(1, total_replay_tokens):.3f}:1, "
                f"replay_updates={total_replay_steps}, "
                f"global_new_exposures={consumed_global_new_exposures}, "
                f"global_replay_exposures={consumed_global_exposures}, "
                f"stored_replay_pool={stored_memory_exposures}, "
                f"sample_ratio={consumed_global_new_exposures / max(1, consumed_global_exposures):.3f}:1",
                args.global_rank)


class Ours_LoRA_MoE_V2_New(Ours_LoRA_MoE_V2):
    """V2 with persistent large memories and a nested 1,000-sample pool.

    Each task persists exactly ``v2_new_persistent_samples_per_task=500``
    records, independently of ``replay_subset_ratio``. Before a later task,
    strictly old memories are reduced to stable, equal-task prefixes whose
    union is capped by ``v2_new_active_memory_cap=1000``. The 1,000-record stream is
    consumed once per primary epoch for joint router replay. KD may repeat
    that exact stream according to ``v2_kd_pass_multiplier`` without adding
    unique records or changing replay:

    * KD at multiplier 1: 1,000 records x 3/5/7 epochs -> 48/80/112
      optimizer updates at effective global batch 64. Multiplier 2 reuses
      those exact records for 96/160/224 updates.
    * Joint replay: 3,000/5,000/7,000 records spread over every primary update.

    Prefixes are nested across rounds.  For example, with 500 stored records
    per task, active quotas are [500], [500, 500], [334, 333, 333], then
    [250, 250, 250, 250].  Shrinking never introduces a replacement record.
    """

    def _v2_new_persistent_samples_per_task(self):
        persistent = int(getattr(
            self.args, "v2_new_persistent_samples_per_task", 500))
        if self._uses_fixed_v2_new_contract() and persistent != 500:
            raise ValueError(
                "V2-new's fixed TRACE contract requires exactly 500 "
                "persistent samples per task, got "
                f"{persistent}")
        if persistent < 1:
            raise ValueError("persistent samples per task must be positive")
        return persistent

    def _uses_fixed_v2_new_contract(self):
        """Whether this run is pinned to the published 1000 / 5:1 contract."""
        return getattr(self.args, "training_version", "") not in {
            "v3_new_replay40", "v3_new_hidden_mse_full",
            "v3_new_replay1to1", "v3_new_hidden_mse_1to1",
            "v3_new_replay1to1_recency", "v3_new_replay1to1_p5k",
            "v3_new_hidden_mse_1to1_p5k",
    "v3_new_kd35k", "v3_new_recency_kd175k", "v3_new_p5k_kd175k", "v3_new_r20_kd100",
    "v3_new_kd200", "v3_new_recency_p2", "v3_new_hmse_kd200"}

    def _v2_new_active_memory_cap(self):
        active_cap = int(getattr(
            self.args, "v2_new_active_memory_cap", 1000))
        if self._uses_fixed_v2_new_contract() and active_cap != 1000:
            raise ValueError(
                "V2-new's fixed TRACE contract requires an active-memory "
                f"cap of exactly 1000, got {active_cap}")
        if active_cap < 1:
            raise ValueError("active-memory cap must be positive")
        world_size = (torch.distributed.get_world_size()
                      if torch.distributed.is_initialized() else 1)
        if active_cap % world_size != 0:
            raise ValueError(
                "V2-new active-memory cap must be divisible by world size: "
                f"{active_cap} % {world_size} != 0")
        return active_cap

    def _v2_new_kd_exposure_samples(self):
        """How many memory records expansion KD-init consumes per pass.

        The active-memory cap was serving double duty: it sized the joint
        replay stream *and* the KD stream, so raising replay from 1,000 to
        5,000 quietly made KD-init five times longer too -- 33 minutes a round
        on Py150.  Only router finetuning was meant to grow.  This lets KD keep
        the published 1,000-record stream while replay scales independently.

        Defaults to the cap, so runs that do not set it are unchanged.
        """
        samples = int(getattr(self.args, "v2_kd_exposure_samples", 0) or 0)
        if samples < 1:
            return self._v2_new_active_memory_cap()
        if self._uses_fixed_v2_new_contract():
            raise ValueError(
                "the published V2-new contract ties KD to the active-memory "
                "cap; --v2_kd_exposure_samples needs a relaxed version")
        return samples

    def _v2_new_active_unique_cap(self):
        """Cap on *distinct* replay records, independent of exposure volume.

        ``v2_new_active_memory_cap`` historically fixed both how many unique
        old records joint replay may draw from and how many exposures it
        spends per primary epoch, because the two were equal by construction.
        Separating them lets a run keep its exposure budget while drawing from
        the whole persisted memory instead of a small prefix of it -- the
        regime the a100 wiki/code/conv chain ran in, where 829,440 distinct
        replay sequences were each seen less than once.

        Defaults to the exposure cap, so runs that do not set it behave
        exactly as before.
        """
        unique_cap = int(getattr(
            self.args, "v2_new_active_unique_cap", 0) or 0)
        if unique_cap < 1:
            return self._v2_new_active_memory_cap()
        if self._uses_fixed_v2_new_contract():
            raise ValueError(
                "the published V2-new contract ties the unique pool to the "
                "1000-record active cap; --v2_new_active_unique_cap requires "
                "a relaxed training version")
        if unique_cap < self._v2_new_active_memory_cap():
            raise ValueError(
                "the unique replay pool cannot be smaller than the exposure "
                f"cap: {unique_cap} < {self._v2_new_active_memory_cap()}")
        return unique_cap

    @staticmethod
    def _v2_new_indices_sha256(indices):
        return hashlib.sha256(
            ",".join(map(str, indices)).encode("utf-8")).hexdigest()

    def _v2_new_memory_path(self, task, task_index):
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task)
        return os.path.join(
            self.args.output_dir, "fixed_replay_memory",
            f"task_{task_index}_{safe_task}.json")

    def _validate_v2_new_saved_memory(
            self, entry, path, task, task_index, source_samples,
            persistent_samples):
        prefix = f"invalid V2-new persistent memory {path}: "
        if entry.get("task") != task:
            raise ValueError(
                prefix + f"task={entry.get('task')!r}, expected={task!r}")
        for field, expected in (
                ("task_index", task_index),
                ("source_samples", source_samples),
                ("unique_samples", persistent_samples)):
            value = entry.get(field)
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value != expected):
                raise ValueError(
                    prefix + f"{field}={value!r}, expected={expected}")
        if entry.get("selection_mode") != "random":
            raise ValueError(
                prefix + f"selection_mode={entry.get('selection_mode')!r}, "
                "expected='random'")
        raw_indices = entry.get("indices")
        if (not isinstance(raw_indices, list)
                or any(not isinstance(index, int) or isinstance(index, bool)
                       for index in raw_indices)):
            raise ValueError(prefix + "indices must be a list of integers")
        indices = list(raw_indices)
        if len(indices) != persistent_samples:
            raise ValueError(
                prefix + f"index count={len(indices)}, "
                f"expected={persistent_samples}")
        if len(set(indices)) != len(indices):
            raise ValueError(prefix + "indices are not unique")
        if any(index < 0 or index >= source_samples for index in indices):
            raise ValueError(prefix + "indices contain an out-of-range value")
        actual_digest = self._v2_new_indices_sha256(indices)
        stored_digest = entry.get("indices_sha256")
        if stored_digest != actual_digest:
            raise ValueError(
                prefix + f"indices_sha256={stored_digest!r}, "
                f"computed={actual_digest}")
        resolved_seed = entry.get("resolved_seed", entry.get("seed"))
        if (not isinstance(resolved_seed, int)
                or isinstance(resolved_seed, bool)):
            raise ValueError(prefix + "resolved seed is missing or invalid")
        return indices, int(resolved_seed)

    def _crosscheck_v2_new_resume_identity(
            self, task, task_index, resolved_seed, indices):
        checkpoint_identities = getattr(
            self.args, "v2_new_resume_persisted_identities", None)
        if checkpoint_identities is None:
            return
        if not isinstance(checkpoint_identities, dict):
            raise ValueError(
                "v2_new_resume_persisted_identities must be a task mapping")
        is_past_task = task_index < int(getattr(
            self.args, "start_task", 0))
        if not is_past_task:
            return
        checkpoint_entry = checkpoint_identities.get(task)
        if not isinstance(checkpoint_entry, dict):
            raise ValueError(
                "V2-new checkpoint has no persisted identity for completed "
                f"task {task_index} ({task})")
        checkpoint_seed = checkpoint_entry.get("resolved_seed")
        checkpoint_digest = checkpoint_entry.get("indices_sha256")
        actual_digest = self._v2_new_indices_sha256(indices)
        if (not isinstance(checkpoint_seed, int)
                or isinstance(checkpoint_seed, bool)
                or checkpoint_seed != resolved_seed):
            raise ValueError(
                "V2-new saved-memory seed disagrees with checkpoint for "
                f"{task}: json={resolved_seed}, "
                f"checkpoint={checkpoint_seed!r}")
        if checkpoint_digest != actual_digest:
            raise ValueError(
                "V2-new saved-memory identity hash disagrees with checkpoint "
                f"for {task}: json={actual_digest}, "
                f"checkpoint={checkpoint_digest!r}")

    def _ensure_fixed_task_subset(self, task):
        """Load-or-create V2-new's exact persistent 500-record memory.

        A saved output-dir JSON is authoritative on resume.  In particular,
        changing the command seed cannot replace identities that were already
        selected; malformed or tampered saved indices fail closed.
        """
        if task in self._fixed_task_subsets:
            subset = self._fixed_task_subsets[task]
            expected = self._v2_new_persistent_samples_per_task()
            if len(subset) != expected:
                raise RuntimeError(
                    f"cached V2-new memory for {task} has "
                    f"{len(subset)}/{expected} records")
            return subset
        selection_mode = getattr(
            self.args, "replay_selection_mode", "random")
        if selection_mode != "random":
            raise ValueError(
                "V2-new currently requires deterministic random persistent "
                f"memory, got replay_selection_mode={selection_mode!r}")
        task_names = list(self.train_task_list)
        task_index = task_names.index(task)
        dataset = self.train_task_list[task].dataset
        source_samples = len(dataset)
        persistent_samples = self._v2_new_persistent_samples_per_task()
        if source_samples < persistent_samples:
            raise ValueError(
                f"V2-new task {task} has only {source_samples} samples; "
                f"the persistent-memory contract requires {persistent_samples}")
        path = self._v2_new_memory_path(task, task_index)
        resumed_past_task = (
            bool(getattr(self.args, "resume_checkpoint", ""))
            and task_index < int(getattr(self.args, "start_task", 0)))
        if resumed_past_task and not os.path.isfile(path):
            raise FileNotFoundError(
                "V2-new resume is missing the authoritative persistent "
                f"memory for completed task {task_index} ({task}): {path}")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    entry = json.load(handle)
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(
                    f"cannot read V2-new persistent memory {path}") from error
            indices, resolved_seed = self._validate_v2_new_saved_memory(
                entry, path, task, task_index, source_samples,
                persistent_samples)
        else:
            manifest = getattr(self, "_replay_manifest", None)
            manifest_entry = (
                manifest["tasks"].get(task) if manifest is not None else None)
            if manifest is not None and manifest_entry is None:
                raise KeyError(
                    f"V2-new replay manifest has no task entry for {task}")
            if manifest_entry is not None:
                if manifest_entry.get("source_samples") != source_samples:
                    raise ValueError(
                        f"V2-new replay manifest source mismatch for {task}: "
                        f"{manifest_entry.get('source_samples')} != "
                        f"{source_samples}")
                raw_indices = manifest_entry.get("indices")
                if (not isinstance(raw_indices, list)
                        or any(not isinstance(index, int)
                               or isinstance(index, bool)
                               for index in raw_indices)):
                    raise ValueError(
                        f"V2-new replay manifest indices for {task} must be "
                        "a list of integers")
                indices = list(raw_indices)
                if (len(indices) != persistent_samples
                        or len(set(indices)) != persistent_samples
                        or any(index < 0 or index >= source_samples
                               for index in indices)):
                    raise ValueError(
                        f"V2-new replay manifest for {task} must contain "
                        f"exactly {persistent_samples} unique in-range indices")
                resolved_seed = manifest_entry.get("seed")
                if (not isinstance(resolved_seed, int)
                        or isinstance(resolved_seed, bool)):
                    raise ValueError(
                        f"V2-new replay manifest seed is invalid for {task}")
            else:
                resolved_seed = self._fixed_subset_seed() + task_index * 1009
                generator = torch.Generator().manual_seed(resolved_seed)
                indices = torch.randperm(
                    source_samples, generator=generator
                )[:persistent_samples].tolist()
            entry = {
                "schema_version": 3,
                "task_index": task_index,
                "task": task,
                "source_samples": source_samples,
                "persistent_samples_per_task": persistent_samples,
                "unique_samples": persistent_samples,
                "resolved_seed": resolved_seed,
                # Retain the old key so earlier audit tools remain readable.
                "seed": resolved_seed,
                "selection_mode": "random",
                "manifest_path": (
                    manifest.get("_path") if manifest is not None else None),
                "indices_sha256": self._v2_new_indices_sha256(indices),
                "indices": indices,
                "available_from_next_task_for_v2_past_replay": True,
            }
            if self.args.global_rank == 0:
                output_dir = os.path.dirname(path)
                os.makedirs(output_dir, exist_ok=True)
                temporary = path + ".tmp"
                with open(temporary, "w", encoding="utf-8") as handle:
                    json.dump(entry, handle, indent=2)
                os.replace(temporary, path)
        self._crosscheck_v2_new_resume_identity(
            task, task_index, resolved_seed, indices)
        subset = Subset(dataset, indices)
        self._fixed_task_subsets[task] = subset
        self._fixed_task_subset_indices[task] = indices
        records = getattr(self, "_v2_new_persistent_memory_records", None)
        if records is None:
            records = self._v2_new_persistent_memory_records = {}
        records[task] = {
            "path": path,
            "resolved_seed": resolved_seed,
            "indices_sha256": self._v2_new_indices_sha256(indices),
        }
        return subset

    @staticmethod
    def _allocate_bounded_equal_task_prefixes(stored_counts, total_cap):
        """Water-fill an equal-task cap without exceeding stored memories."""
        stored_counts = [int(count) for count in stored_counts]
        if any(count < 0 for count in stored_counts):
            raise ValueError("stored memory counts cannot be negative")
        if not stored_counts:
            return []
        target = min(int(total_cap), sum(stored_counts))
        if target < 1:
            raise ValueError(
                f"active V2-new memory cap must be positive, got {total_cap}")
        allocated = [0] * len(stored_counts)
        remaining = target
        while remaining:
            active = [
                index for index, stored in enumerate(stored_counts)
                if allocated[index] < stored]
            if not active:
                raise RuntimeError(
                    "bounded active-memory allocation exhausted early")
            share = remaining // len(active)
            if share == 0:
                for index in active[:remaining]:
                    allocated[index] += 1
                remaining = 0
                continue
            distributed = 0
            for index in active:
                addition = min(
                    share, stored_counts[index] - allocated[index])
                allocated[index] += addition
                distributed += addition
            if distributed < 1:
                raise RuntimeError(
                    "bounded active-memory allocation made no progress")
            remaining -= distributed
        return allocated

    def _v2_new_active_memory(self, i_task, unique_cap=None):
        task_names = self._memory_task_names(i_task)
        if not task_names:
            return task_names, []
        if self.args.replay_distribution not in (
                "equal_task", "recency_weighted"):
            raise ValueError(
                "V2-new replay must be equal_task or recency_weighted, got "
                f"{self.args.replay_distribution}")
        stored_counts = [
            len(self._ensure_fixed_task_subset(task)) for task in task_names]
        persistent = self._v2_new_persistent_samples_per_task()
        allow_short = os.environ.get("SELFGEN_ALLOW_SHORT", "0") == "1"
        if allow_short:
            # a self-generated replay pool that came in under the target count (e.g. a task whose
            # generation yield was too low to reach `persistent`) is expected and not a drift bug;
            # only an OVER-count (more stored than the contract) still indicates a real problem.
            over = [(t, c) for t, c in zip(task_names, stored_counts) if c > persistent]
            if over:
                raise RuntimeError(
                    "V2-new persistent-memory count drift (over target): "
                    f"{over}, expected_each<={persistent}")
        elif any(count != persistent for count in stored_counts):
            raise RuntimeError(
                "V2-new persistent-memory count drift: "
                f"stored={stored_counts}, expected_each={persistent}")
        if unique_cap is None:
            unique_cap = self._v2_new_active_memory_cap()
        unique_cap = int(unique_cap)
        active_counts = self._allocate_bounded_equal_task_prefixes(
            stored_counts, unique_cap)
        if allow_short:
            expected_active = min(unique_cap, sum(min(persistent, c) for c in stored_counts))
        else:
            expected_active = min(unique_cap, persistent * len(task_names))
        if sum(active_counts) != expected_active:
            raise RuntimeError(
                "V2-new active-memory allocation mismatch: "
                f"{sum(active_counts)}/{expected_active}")
        return task_names, active_counts

    def _build_v2_new_active_loader(self, i_task, role):
        if role not in {"kd", "replay"}:
            raise ValueError(f"unknown V2-new memory role: {role}")
        # Only joint replay widens its unique pool.  KD keeps the exposure
        # cap as its pool so a diversity experiment stays a replay-only change
        # and does not silently alter expansion KD-init as well.
        task_names, active_counts = self._v2_new_active_memory(
            i_task,
            unique_cap=(self._v2_new_active_unique_cap()
                        if role == "replay" else None))
        if not task_names:
            return None
        # The pool cap sizes the *unique* records; the exposure cap sizes one
        # stream, i.e. how many replay forwards a primary epoch spends.  They
        # are the same number unless --v2_new_replay_exposure_cap separates
        # them, and the stream must follow the exposure cap or the joint loop
        # consumes a non-integer number of passes per epoch.
        exposure_samples = (self._v2_new_kd_exposure_samples()
                            if role == "kd"
                            else self._v2_new_replay_exposure_cap())
        memory_batch_size = self.args.v2_memory_batch_size or 1
        if role == "kd":
            memory_batch_size = (
                getattr(self.args, "v2_kd_memory_batch_size", 0)
                or memory_batch_size)
        loader = self._build_fixed_memory_loader(
            task_names,
            round_index=i_task,
            phase=f"v2_new_{role}_active_memory",
            # Even with only one old task, cycle its persisted 500-record
            # prefix twice to form the common 1,000-record stream.
            exposure_samples=exposure_samples,
            batch_size=memory_batch_size,
            active_unique_counts=active_counts,
            stream_seed_phase="v2_new_shared_active_memory",
            deterministic_pass_sampler=True)
        stream = loader._lora_moe_memory_stream
        if (len(loader.dataset) != exposure_samples
                or stream["planned_exposure_samples"] != exposure_samples):
            raise RuntimeError(
                f"V2-new {role} stream is not exactly {exposure_samples} "
                f"records: dataset={len(loader.dataset)}, "
                f"plan={stream['planned_exposure_samples']}")
        signatures = getattr(self, "_v2_new_stream_signatures", None)
        if signatures is None:
            signatures = self._v2_new_stream_signatures = {}
        signature = stream["ordered_identity_sha256"]
        round_signatures = signatures.setdefault(i_task, {})
        round_signatures[role] = signature
        # KD and replay are drawn from one stream, so their identity order
        # must match -- unless the run deliberately gives KD a shorter one, in
        # which case KD is a prefix of a different length and cannot match.
        if (self._v2_new_kd_exposure_samples()
                == self._v2_new_replay_exposure_cap()
                and len(set(round_signatures.values())) != 1):
            raise RuntimeError(
                "V2-new KD/replay active streams do not have literal "
                f"identity order equality: {round_signatures}")
        return loader

    def _build_v2_kd_loader(self, i_task, primary_loader, epochs):
        del primary_loader, epochs
        return self._build_v2_new_active_loader(i_task, role="kd")

    def _build_v2_replay_loader(self, i_task, primary_loader, epochs):
        del primary_loader, epochs
        return self._build_v2_new_active_loader(i_task, role="replay")

    @staticmethod
    def _v2_new_sampler_order(dataloader, pass_index):
        """Return the flattened rank-local order without consuming the pass."""
        sampler = getattr(dataloader, "sampler", None)
        if not hasattr(sampler, "set_epoch"):
            raise RuntimeError(
                "V2-new memory sampler must implement set_epoch")
        sampler.set_epoch(int(pass_index))
        order = tuple(int(index) for index in sampler)
        if len(order) != len(sampler):
            raise RuntimeError(
                "V2-new sampler order length mismatch: "
                f"{len(order)}/{len(sampler)}")
        return order

    @classmethod
    def _v2_new_sampler_order_sha256(cls, dataloader, pass_index):
        order = cls._v2_new_sampler_order(dataloader, pass_index)
        digest = hashlib.sha256(
            ",".join(map(str, order)).encode("utf-8")).hexdigest()
        return order, digest

    def _record_v2_new_sampler_pass(
            self, role, dataloader, pass_index):
        order, digest = self._v2_new_sampler_order_sha256(
            dataloader, pass_index)
        stream = dataloader._lora_moe_memory_stream
        by_pass = stream.setdefault("sampler_order_sha256_by_pass", {})
        by_pass[str(pass_index)] = digest
        registry = getattr(self, "_v2_new_sampler_pass_digests", None)
        if registry is None:
            registry = self._v2_new_sampler_pass_digests = {}
        round_registry = registry.setdefault(int(stream["round"]), {})
        role_registry = round_registry.setdefault(role, {})
        role_registry[int(pass_index)] = {
            "sha256": digest,
            "local_samples": len(order),
        }
        other_role = "replay" if role == "kd" else "kd"
        other = round_registry.get(other_role, {}).get(int(pass_index))
        # KD and replay normally consume one shared stream, so their per-pass
        # sampler orders must be byte-identical.  A run that deliberately gives
        # KD a shorter stream draws a different number of records and can never
        # match, so the invariant only applies while the two are the same size.
        streams_are_shared = (
            self._v2_new_kd_exposure_samples()
            == self._v2_new_replay_exposure_cap())
        if streams_are_shared and other is not None and other["sha256"] != digest:
            raise RuntimeError(
                "V2-new KD/replay sampler order mismatch at round/pass "
                f"{stream['round']}/{pass_index}: "
                f"{role}={digest}, {other_role}={other['sha256']}")
        # _v2_new_sampler_order installed the actual pass for the next iter().

    def _set_v2_kd_memory_sampler_pass(
            self, dataloader, pass_index):
        self._record_v2_new_sampler_pass("kd", dataloader, pass_index)

    def _set_v2_replay_memory_sampler_pass(
            self, dataloader, pass_index):
        self._record_v2_new_sampler_pass("replay", dataloader, pass_index)

    def _validate_v2_replay_memory_sampler_passes(
            self, dataloader, completed_passes, primary_epochs):
        if completed_passes != int(primary_epochs):
            raise RuntimeError(
                "V2-new replay did not consume exactly one stream per "
                f"primary epoch: {completed_passes}/{primary_epochs}")
        stream = dataloader._lora_moe_memory_stream
        registry = self._v2_new_sampler_pass_digests[int(stream["round"])]
        expected_replay = set(range(int(primary_epochs)))
        replay_passes = set(registry.get("replay", {}))
        if replay_passes != expected_replay:
            raise RuntimeError(
                "V2-new replay sampler passes "
                f"{replay_passes}/{expected_replay}")
        kd_passes = set(registry.get("kd", {}))
        # Ask the same helper KD-init used, so an explicit --v2_kd_epochs is
        # validated against what it asked for rather than against the primary
        # schedule it deliberately no longer follows.
        expected_kd = set(range(int(self._v2_kd_epochs(primary_epochs))))
        kd_fraction = float(getattr(self.args, "v3_kd_init_step_fraction", 1.0))
        # A truncated KD-init (fraction < 1) deliberately stops before some
        # passes are ever started, so it only owes a contiguous prefix of
        # expected_kd, not the full set.
        kd_ok = (
            kd_passes == expected_kd
            or (kd_fraction < 1.0
                and kd_passes == set(range(len(kd_passes)))
                and kd_passes.issubset(expected_kd)))
        if self.args.v2_kd_loss_coeff > 0 and not kd_ok:
            raise RuntimeError(
                f"V2-new KD sampler passes {kd_passes}/{expected_kd}")
        # Per-pass orders can only match while KD and replay draw the same
        # stream; a run that shortens KD breaks that by construction.
        if (self._v2_new_kd_exposure_samples()
                == self._v2_new_replay_exposure_cap()):
            for pass_index in kd_passes & replay_passes:
                kd_digest = registry["kd"][pass_index]["sha256"]
                replay_digest = registry["replay"][pass_index]["sha256"]
                if kd_digest != replay_digest:
                    raise RuntimeError(
                        "V2-new final sampler-order mismatch at pass "
                        f"{pass_index}: {kd_digest} != {replay_digest}")

    def _workload_replay_exposure_budget(self, epochs):
        # The declared budget must follow the stream the run actually consumes,
        # i.e. the exposure cap (--v2_new_replay_exposure_cap when set), not the
        # pool cap.  Otherwise a run that holds replay compute fixed while the
        # pool grows records a budget it never spends, and
        # scripts/ablation/verify_ablation_run.py flags the parity check.
        return self._v2_new_replay_exposure_cap() * int(epochs)

    def _v2_kd_epochs(self, primary_epochs):
        # An epoch boundary is intentional: 1,000 samples at global batch 64
        # is 16 updates per pass, rather than 3,000 samples becoming 47 updates
        # after a single partially filled final batch.
        override = int(getattr(self.args, "v2_kd_epochs", 0) or 0)
        if override > 0:
            if self._uses_fixed_v2_new_contract():
                raise ValueError(
                    "the published V2-new contract derives KD epochs from the "
                    "primary schedule; --v2_kd_epochs needs a relaxed version")
            return override
        multiplier = int(getattr(
            self.args, "v2_kd_pass_multiplier", 1))
        if multiplier < 1:
            raise ValueError(
                f"V2-new KD pass multiplier must be positive: {multiplier}")
        return int(primary_epochs) * multiplier

    def _v2_new_replay_exposure_cap(self):
        """Replay exposures per primary epoch: the explicit knob when set, else the pool cap."""
        explicit = int(getattr(self.args, "v2_new_replay_exposure_cap", 0) or 0)
        return explicit if explicit > 0 else self._v2_new_active_memory_cap()

    def _joint_replay_exposure_budget(self, primary_loader, epochs):
        """Consume one exact active-memory stream per primary epoch."""
        active_cap = self._v2_new_replay_exposure_cap()
        ratio = int(getattr(
            self.args, "v2_joint_new_to_replay_ratio", 0))
        if ratio < 1:
            raise ValueError(
                "V2-new requires a positive new:replay sample ratio")
        sampler = getattr(primary_loader, "sampler", None)
        samples_per_epoch = getattr(sampler, "total_size", None)
        if samples_per_epoch is None:
            dataset = getattr(primary_loader, "dataset", None)
            if dataset is None:
                raise ValueError(
                    "V2-new replay requires a primary loader dataset")
            samples_per_epoch = len(dataset)
        expected_new = active_cap * ratio
        if self._uses_fixed_v2_new_contract():
            if int(samples_per_epoch) != expected_new:
                raise ValueError(
                    "V2-new fixed-memory ratio contract mismatch: primary "
                    f"samples/epoch={samples_per_epoch}, active_memory_cap="
                    f"{active_cap}, ratio={ratio}:1, expected={expected_new}")
        else:
            # Relaxed profile: the replay stream size is the exposure cap
            # (--v2_new_replay_exposure_cap, or the active-memory cap when it is
            # not set), and the ratio is reported rather than pinned.  One exact
            # stream per primary epoch is consumed, so replay exposure stays
            # cap x epochs and remains resume-reproducible.  Keeping the
            # exposure cap fixed while the pool cap varies is what lets a
            # replay-size ablation hold the replay compute constant.
            if int(samples_per_epoch) < active_cap:
                raise ValueError(
                    "relaxed V2-new replay needs primary samples/epoch >= "
                    f"replay exposure cap: {samples_per_epoch} < {active_cap}")
        return active_cap * int(epochs)

    def _expected_joint_replay_exposures(
            self, consumed_global_new_exposures, epochs, replay_ratio):
        """Validate 20% replay against all primary epochs, not one epoch."""
        if not self._uses_fixed_v2_new_contract():
            # Relaxed profile: replay is cap-driven, so derive the expected
            # exposure from the stream actually scheduled instead of the
            # nominal ratio.
            return self._v2_new_replay_exposure_cap() * int(epochs)
        del epochs
        if consumed_global_new_exposures % replay_ratio != 0:
            raise RuntimeError(
                "total new sample exposure does not match V2-new replay "
                f"ratio: new={consumed_global_new_exposures}, "
                f"ratio={replay_ratio}:1")
        return consumed_global_new_exposures // replay_ratio


class Ours_LoRA_MoE_V1_Expert_First(Ours_LoRA_MoE_V2_New):
    """V1 phase order with V2-new initialization and memory identities.

    Growth is physically performed before KD because the expanded student is
    required by V2-new KD.  During the primary task phase, however, the new
    row is logically detached from routing: every valid task token is sent to
    the one new expert at unit weight, every router parameter is frozen, and
    router aux/z objectives are omitted.  Only after that expert-only phase is
    complete do we expose normal learned top-1 routing and run a separate
    router-only retune over V2-new's capped, equal-task memory of all seen
    tasks (including the current one).
    """

    def _build_expert_first_router_loader(self, i_task):
        task_names = list(self.train_task_list)[:i_task + 1]
        if not task_names:
            return None
        stored_counts = [
            len(self._ensure_fixed_task_subset(task)) for task in task_names]
        persistent = self._v2_new_persistent_samples_per_task()
        if any(count != persistent for count in stored_counts):
            raise RuntimeError(
                "V1 expert-first persistent-memory count drift: "
                f"stored={stored_counts}, expected_each={persistent}")
        active_cap = self._v2_new_active_memory_cap()
        active_counts = self._allocate_bounded_equal_task_prefixes(
            stored_counts, active_cap)
        expected_active = min(active_cap, persistent * len(task_names))
        if sum(active_counts) != expected_active:
            raise RuntimeError(
                "V1 expert-first active-memory allocation mismatch: "
                f"{sum(active_counts)}/{expected_active}")
        replay_batch_size = min(
            self.args.batch_by_task[name] for name in task_names)
        return self._build_fixed_memory_loader(
            task_names,
            round_index=i_task,
            phase="v1_expert_first_router_seen_memory",
            exposure_samples=active_cap,
            batch_size=replay_batch_size,
            active_unique_counts=active_counts,
            stream_seed_phase="v1_expert_first_router_seen_memory",
            deterministic_pass_sampler=True)

    def _workload_replay_exposure_budget(self, epochs):
        del epochs
        if self.args.router_retune_epochs <= 0:
            return 0
        return self._v2_new_active_memory_cap()

    def train_one_task(self, task, i_task, epochs):
        device = (
            torch.device("cuda", self.args.local_rank)
            if self.args.local_rank != -1 else torch.device("cuda")
        )
        args = self.args
        if args.experts_per_task != 1 or args.top_k != 1:
            raise ValueError(
                "v1_expert_first requires --experts_per_task 1 --top_k 1")

        self._ensure_fixed_task_subset(task)
        old_expert_count = len(
            _lora_moe_layers(self.raw_model)[0].experts)
        add_experts_to_all_layers(self.raw_model, 1)
        new_expert_index = old_expert_count
        new_indices = {new_expert_index}

        primary_loader = self.train_task_list[task]
        kd_loader = self._build_v2_kd_loader(
            i_task, primary_loader, epochs)
        if (old_expert_count > 0 and kd_loader is not None
                and args.v2_kd_loss_coeff > 0):
            self._run_v2_kd_init(
                kd_loader, old_expert_count, new_indices, device, task,
                kd_epochs=self._v2_kd_epochs(epochs))

        # Phase 1: all valid new-task tokens train the new LoRA expert.  The
        # forced route uses unit expert weight, exactly matching top-1's
        # normalized forward, while the frozen router and omitted aux/z loss
        # make this a genuinely expert-only acquisition phase.
        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} expert-first phase1")
        self._set_grad_ckpt(task in getattr(args, "ckpt_tasks", set()))
        freeze_lora_moe_experts(
            self.raw_model, trainable_expert_indices=new_indices)
        freeze_lora_moe_routers(self.raw_model, trainable=False)
        self._reinit_engine(self._optimizer_update_count(
            primary_loader, epochs))
        with force_lora_moe_expert(self.raw_model, new_expert_index):
            self._run_epochs(
                primary_loader, epochs, device,
                f"{task} [phase1 standalone new expert full-token]",
                include_moe_loss=False,
                workload_role="new_task_expert_only")

        # Phase 2: attach logically by restoring natural top-1 dispatch, freeze
        # every expert, and calibrate all router rows on V2-new's persistent
        # memories from both old and current tasks.
        if args.router_retune_epochs > 0:
            replay_loader = self._build_expert_first_router_loader(i_task)
            if replay_loader is not None:
                self._set_grad_ckpt(True)
                freeze_lora_moe_experts(
                    self.raw_model, trainable_expert_indices=None)
                freeze_lora_moe_routers(self.raw_model, trainable=True)
                self._set_phase_gradient_accumulation(
                    replay_loader.batch_size,
                    f"{task} expert-first phase2 router retune")
                self._reinit_engine(self._optimizer_update_count(
                    replay_loader, 1))
                self._run_epochs(
                    replay_loader, 1, device,
                    f"{task} [phase2 router retune V2-new seen memory]",
                    workload_role="router_replay")


# ---------------------------------------------------------------------------
# Checkpoint round-trip: save_hf_format stores the full state_dict (base_mlp +
# experts + router) under the *base* Qwen config.json, so a plain
# from_pretrained / vLLM load can't reconstruct the custom LoRAMoEMLP. These
# helpers write a small self-describing meta at save time and rebuild the exact
# grown model at load time.
# ---------------------------------------------------------------------------
LORA_MOE_META_NAME = "lora_moe_meta.json"


def save_lora_moe_meta(model, output_dir, extra=None):
    """Dump the hyperparameters + current expert count next to the checkpoint."""
    layers = _lora_moe_layers(model)
    if not layers:
        return
    layer = layers[0]
    meta = {
        "r": layer.r,
        "alpha": layer.alpha,
        "dropout": layer.dropout,
        "top_k": layer.top_k,
        "aux_loss_coeff": layer.aux_loss_coeff,
        "z_loss_coeff": layer.z_loss_coeff,
        "routing_weight_mode": layer.routing_weight_mode,
        "num_experts": layer.num_experts,
    }
    if extra:
        meta.update(extra)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, LORA_MOE_META_NAME), "w") as f:
        json.dump(meta, f, indent=2)


def num_experts_from_state_dict(state_dict):
    """Infer the grown expert-pool size straight from checkpoint keys (fallback
    when no lora_moe_meta.json is present)."""
    pat = re.compile(r"\.mlp\.experts\.(\d+)\.")
    max_idx = -1
    for k in state_dict:
        m = pat.search(k)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def load_lora_moe_checkpoint(checkpoint_dir, tokenizer, base_model_name_or_path=None,
                             device="cuda", dtype=torch.bfloat16,
                             device_map=None):
    """Rebuild a trained growing-LoRA-MoE model from a save_hf_format checkpoint.

    checkpoint_dir contains pytorch_model.bin (the FULL state_dict: base weights,
    every layer's mlp.base_mlp.* frozen dense FFN, mlp.experts.*, mlp.router.*),
    config.json (base architecture), the tokenizer, and lora_moe_meta.json.

    base_model_name_or_path (recommended): the original pretrained model. We build
    the skeleton from it (correct base weights + config incl. the training-time
    embedding resize), then strict-load the checkpoint on top -- so any key
    mismatch surfaces loudly instead of silently leaving a tensor random. If it is
    None we build an empty skeleton from the checkpoint's own config.json and rely
    on the checkpoint holding every weight.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model

    meta_path = os.path.join(checkpoint_dir, LORA_MOE_META_NAME)
    weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # No meta (e.g. checkpoints saved before save_lora_moe_meta existed):
        # recover the expert count from the weights and fall back to defaults
        # for the routing hyperparameters.
        meta = {"r": None, "alpha": None, "top_k": 2,
                "aux_loss_coeff": 0.01, "z_loss_coeff": 0.001,
                "dropout": 0.0,
                "routing_weight_mode": "topk_softmax",
                "num_experts": num_experts_from_state_dict(state_dict)}

    # Existing checkpoints were trained with softmax over selected top-k logits.
    # New runs explicitly record full_softmax when requested.
    meta.setdefault("routing_weight_mode", "topk_softmax")
    meta.setdefault("dropout", 0.0)

    n_experts = meta["num_experts"]
    ckpt_experts = num_experts_from_state_dict(state_dict)
    if ckpt_experts and ckpt_experts != n_experts:
        raise ValueError(
            f"lora_moe_meta.json says num_experts={n_experts} but the checkpoint "
            f"holds {ckpt_experts} experts ({checkpoint_dir}).")

    # r/alpha only affect LoRAPair.scaling (= alpha / r), which is baked into the
    # forward, not the weights. If meta lacks them, infer r from an A tensor and
    # keep scaling==1.0 by setting alpha==r (any factor is absorbed into B anyway
    # for a *loaded* expert, but we reproduce training exactly when meta is present).
    if meta["r"] is None:
        a_key = next(k for k in state_dict if re.search(r"\.experts\.0\.gate\.A$", k))
        meta["r"] = state_dict[a_key].shape[0]
        meta["alpha"] = meta["r"]

    # Partial checkpoint (trainable_only save): only mlp.experts.* / mlp.router.*
    # keys, no backbone -> the base weights MUST come from base_model_name_or_path,
    # and we load with strict=False (backbone keys are simply absent).
    is_partial = all((".mlp.experts." in k) or (".mlp.router." in k) for k in state_dict)

    if base_model_name_or_path is not None:
        model = create_hf_model(
            AutoModelForCausalLM,
            base_model_name_or_path,
            tokenizer,
            disable_dropout=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            forbid_vocab_growth=is_partial,
            device_map=device_map,
        )
    elif is_partial:
        raise ValueError(
            f"{checkpoint_dir} is a trainable-only checkpoint (experts+router, no "
            f"backbone); pass base_model_name_or_path so the base can be rebuilt.")
    else:
        if device_map is not None:
            raise ValueError(
                "device_map evaluation requires base_model_name_or_path so the "
                "checkpoint can be loaded and dispatched by from_pretrained().")
        config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.config.pad_token_id = model.config.eos_token_id

    attach_lora_moe(model, r=meta["r"], alpha=meta["alpha"], top_k=meta["top_k"],
                    aux_loss_coeff=meta["aux_loss_coeff"], z_loss_coeff=meta["z_loss_coeff"],
                    routing_weight_mode=meta["routing_weight_mode"],
                    dropout=meta["dropout"])
    add_experts_to_all_layers(model, n_experts)

    # Full checkpoint -> strict; partial -> non-strict (backbone came from the base,
    # only experts/router are overlaid). Guard against silently-random experts by
    # asserting the checkpoint's grown keys all matched.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"unexpected keys when loading {checkpoint_dir}: {unexpected[:5]}"
    if not is_partial:
        assert not missing, f"missing keys when loading full checkpoint {checkpoint_dir}: {missing[:5]}"
    else:
        # every grown key present; the only 'missing' should be backbone (from base)
        grown_missing = [m for m in missing if ".mlp.experts." in m or ".mlp.router." in m]
        assert not grown_missing, f"partial ckpt missing grown keys: {grown_missing[:5]}"
    # A dispatched model already has layers placed across its device map; calling
    # .to() would collapse it back onto one GPU and may OOM.
    if device_map is None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, meta
