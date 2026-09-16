"""Ours V3: one pre-attention router shared by QKVO and FFN LoRA experts.

V3 intentionally leaves the V2 continual-learning schedule unchanged.  The
only conceptual change is inside each decoder layer:

    input -> input_layernorm -> shared router
                                  |-> Q/K/V/O LoRA experts
                                  `-> gate/up/down LoRA experts

The FFN consumes the post-attention normalized states, but it reuses the
expert indices and weights computed from the pre-attention normalized states.
Every task grows the attention experts, FFN experts, and the single router by
the same number of rows.
"""

import json
import math
import os
import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from tqdm import tqdm

from model.base_model import CL_Base_Model
from model.Ours_LoRA_MoE import (
    LoRAPair,
    Ours_LoRA_MoE_V2,
    Ours_LoRA_MoE_V2_New,
)
from utils.utils import print_rank_0, to_device


V3_ARCHITECTURE = "shared_router_qkvo_ffn"
V3_NEW_TRAINING_VERSIONS = frozenset({
    "v3_new", "v3_new_top4", "v3_new_replay40", "v3_new_hidden_mse_full",
    "v3_new_replay1to1", "v3_new_hidden_mse_1to1",
    "v3_new_replay1to1_recency", "v3_new_replay1to1_p5k",
    "v3_new_hidden_mse_1to1_p5k",
    "v3_new_kd35k", "v3_new_recency_kd175k", "v3_new_p5k_kd175k", "v3_new_r20_kd100",
    "v3_new_kd200", "v3_new_recency_p2", "v3_new_hmse_kd200"})
V3_META_NAME = "lora_moe_meta.json"
ATTENTION_TARGETS = ("q", "k", "v", "o")


@dataclass
class RoutingContext:
    """One layer's routing decision, reused by all seven projections."""

    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    valid_token_mask: torch.Tensor | None
    num_experts: int
    # A RoutingContext is created for exactly one decoder-layer forward and is
    # cleared from every projection before that forward returns.  Cache the
    # integer dispatch indices for that short lifetime so Q/K/V/O and the FFN
    # do not each launch the same expert-wise torch.where kernels.  These
    # tensors are routing metadata (not differentiable values), so sharing
    # them leaves both the forward computation and gradient graph unchanged.
    _route_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False, compare=False)
    _active_cache: list[int] | None = field(
        default=None, init=False, repr=False, compare=False)

    def active_expert_ids(self):
        """Expert ids owning at least one live token this forward.

        Experts with no tokens contributed nothing before -- every projection
        loop hit ``continue`` on them -- but reaching that ``continue`` still
        cost a compare plus a ``torch.where`` per unused expert.  During
        decoding that is the whole bill: top-1 routing over a single new token
        leaves one expert live, so seven of eight iterations were pure kernel
        launch overhead, repeated for every layer of every step.  Reading the
        live ids once per layer replaces those launches with a single unique().

        Ascending order matches the old ``range(num_experts)`` iteration, so
        the ``index_add_`` accumulation order -- and therefore the exact
        floating-point result -- is unchanged.
        """
        if self._active_cache is None:
            indices = self.expert_indices
            if self.valid_token_mask is not None:
                indices = indices[self.valid_token_mask]
            ids = torch.unique(indices.reshape(-1)).tolist()
            self._active_cache = [e for e in ids if 0 <= e < self.num_experts]
        return self._active_cache

    def routes_for(self, expert_index):
        """Return (slot, token_index), optionally excluding padded tokens."""
        cached = self._route_cache.get(expert_index)
        if cached is not None:
            return cached
        slot, token_index = torch.where(
            self.expert_indices.transpose(0, 1) == expert_index)
        if self.valid_token_mask is not None and token_index.numel() > 0:
            keep = self.valid_token_mask[token_index]
            slot, token_index = slot[keep], token_index[keep]
        routes = (slot, token_index)
        self._route_cache[expert_index] = routes
        return routes


class SharedExpertRouter(nn.Module):
    """Layer-level router evaluated immediately after input_layernorm."""

    def __init__(self, hidden_size, top_k, aux_loss_coeff, z_loss_coeff,
                 routing_weight_mode="full_softmax", device=None, dtype=None):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if routing_weight_mode not in {
                "topk_softmax", "full_softmax", "straight_through_topk"}:
            raise ValueError(
                f"unknown routing_weight_mode: {routing_weight_mode}")
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.routing_weight_mode = routing_weight_mode
        # A non-persistent placement anchor follows later model.to(...) calls,
        # unlike a cached torch.device value captured at construction time.
        self.register_buffer(
            "_placement_anchor",
            torch.empty(0, device=device, dtype=dtype),
            persistent=False)
        self.router = None
        self._active_expert_count = None
        self._router_token_mask = None
        self._last_moe_loss = None
        # Replay consumes only causal LM loss; its trainer context disables
        # aux/z graph construction without changing primary/KD forwards.
        self._suppress_router_loss = False
        self._capture_probe_routing = False
        self._last_probe_indices = None
        self._capture_sample_grad = False
        self._sample_grad_scores = []

    @property
    def num_experts(self):
        return 0 if self.router is None else self.router.out_features

    @property
    def weight(self):
        """Compatibility-shaped access for V2 router-row snapshots."""
        if self.router is None:
            raise RuntimeError("router has no experts")
        return self.router.weight

    def add_experts(self, count):
        if count < 1:
            raise ValueError("expert growth must be positive")
        old_count = self.num_experts
        new_router = nn.Linear(
            self.hidden_size, old_count + count, bias=False,
            device=self._placement_anchor.device,
            dtype=self._placement_anchor.dtype)
        if self.router is not None:
            with torch.no_grad():
                new_router.weight[:old_count].copy_(self.router.weight)
        self.router = new_router

    def _active_count(self):
        count = (
            self.num_experts if self._active_expert_count is None
            else int(self._active_expert_count))
        if not 0 < count <= self.num_experts:
            raise ValueError(
                f"active expert prefix {count} is invalid for "
                f"{self.num_experts} experts")
        return count

    def forward(self, hidden_states):
        active_count = self._active_count()
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_logits = self.router(flat_hidden)[..., :active_count]
        if self._capture_sample_grad and flat_logits.requires_grad:
            # The gradient arriving at router logits is still separated by
            # token.  Combining it with the corresponding router input gives
            # the exact per-example gradient of this layer's bias-free linear
            # router: dW_b = sum_t dlogits[b,t]^T hidden[b,t].  Computing it in
            # this hook reuses the normal training backward; no second forward
            # or backward is needed.
            batch_size = int(hidden_states.shape[0])
            hidden_by_sample = hidden_states.detach().reshape(
                batch_size, -1, hidden_states.shape[-1])
            score_mask = self._router_token_mask
            if score_mask is not None:
                score_mask = score_mask.detach().reshape(batch_size, -1).bool()

            def capture_router_gradient(gradient):
                gradient = gradient.detach().reshape(
                    batch_size, -1, active_count)
                scores = torch.zeros(
                    batch_size, device=gradient.device, dtype=torch.float32)
                for sample_index in range(batch_size):
                    sample_gradient = gradient[sample_index]
                    sample_hidden = hidden_by_sample[sample_index]
                    if score_mask is not None:
                        keep = score_mask[sample_index]
                        sample_gradient = sample_gradient[keep]
                        sample_hidden = sample_hidden[keep]
                    if sample_gradient.numel() == 0:
                        continue
                    weight_gradient = sample_gradient.float().transpose(
                        0, 1).matmul(sample_hidden.float())
                    scores[sample_index] = weight_gradient.square().sum()
                self._sample_grad_scores.append(scores)

            flat_logits.register_hook(capture_router_gradient)
        k = min(self.top_k, active_count)
        topk_logits, topk_indices = flat_logits.topk(k, dim=-1)
        full_probs = None
        if self.routing_weight_mode in {
                "full_softmax", "straight_through_topk"}:
            full_probs = F.softmax(flat_logits, dim=-1, dtype=torch.float)
            selected_probs = full_probs.gather(-1, topk_indices)
            if self.routing_weight_mode == "straight_through_topk":
                normalized = F.softmax(topk_logits, dim=-1)
                topk_weights = (
                    normalized.detach() + selected_probs
                    - selected_probs.detach())
            else:
                topk_weights = selected_probs
        else:
            topk_weights = F.softmax(topk_logits, dim=-1)

        valid_mask = self._router_token_mask
        if valid_mask is not None:
            valid_mask = valid_mask.reshape(-1).to(
                device=flat_logits.device, dtype=torch.bool)
            if valid_mask.numel() != flat_logits.shape[0]:
                raise ValueError(
                    f"router mask has {valid_mask.numel()} tokens but hidden "
                    f"states have {flat_logits.shape[0]}")

        self._last_moe_loss = (
            self._router_loss(
                flat_logits, topk_indices, full_probs, valid_mask)
            if self.training and not self._suppress_router_loss else None)
        self._last_probe_indices = (
            topk_indices.detach() if self._capture_probe_routing else None)
        return RoutingContext(
            expert_indices=topk_indices,
            expert_weights=topk_weights,
            valid_token_mask=valid_mask,
            num_experts=active_count,
        )

    def _router_loss(self, logits, indices, full_probs, token_mask):
        if token_mask is not None:
            logits = logits[token_mask]
            indices = indices[token_mask]
            if full_probs is not None:
                full_probs = full_probs[token_mask]
        num_tokens, num_experts = logits.shape
        if num_tokens == 0:
            return logits.sum() * 0.0
        if full_probs is None:
            full_probs = F.softmax(logits, dim=-1, dtype=torch.float)
        tokens_per_expert = torch.bincount(
            indices.reshape(-1), minlength=num_experts).to(full_probs.dtype)
        aggregated_probs = full_probs.sum(dim=0)
        aux_loss = torch.sum(aggregated_probs * tokens_per_expert) * (
            num_experts * self.aux_loss_coeff /
            (num_tokens * num_tokens * indices.shape[-1]))
        z_loss = torch.mean(torch.square(
            torch.logsumexp(logits.float(), dim=-1))) * self.z_loss_coeff
        return aux_loss + z_loss


class RoutedLoRALinear(nn.Module):
    """Frozen linear projection plus sparse shared-router LoRA experts."""

    def __init__(self, base_layer, r, alpha, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "V3 attention projections must be torch.nn.Linear, got "
                f"{type(base_layer).__name__}")
        self.base_layer = base_layer
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.experts = nn.ModuleList()
        self._routing_context = None

    def add_experts(self, count):
        device = self.base_layer.weight.device
        dtype = self.base_layer.weight.dtype
        for _ in range(count):
            self.experts.append(LoRAPair(
                self.in_features, self.out_features, self.r, self.alpha,
                self.dropout).to(device=device, dtype=dtype))

    def set_routing_context(self, context):
        self._routing_context = context

    def forward(self, inputs):
        output = self.base_layer(inputs)
        context = self._routing_context
        if context is None:
            if self.experts:
                raise RuntimeError(
                    "attention expert projection used without routing context")
            return output
        if context.num_experts > len(self.experts):
            raise ValueError(
                f"routing exposes {context.num_experts} experts but projection "
                f"contains {len(self.experts)}")
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        if flat_inputs.shape[0] != context.expert_indices.shape[0]:
            raise ValueError(
                "attention projection token count does not match shared routing: "
                f"{flat_inputs.shape[0]} != {context.expert_indices.shape[0]}")
        flat_delta = output.new_zeros((flat_inputs.shape[0], self.out_features))
        for expert_index in context.active_expert_ids():
            expert = self.experts[expert_index]
            slot, token_index = context.routes_for(expert_index)
            if token_index.numel() == 0:
                continue
            expert_input = flat_inputs.index_select(0, token_index)
            weights = context.expert_weights[
                token_index, slot, None].to(expert_input.dtype)
            flat_delta.index_add_(
                0, token_index, expert(expert_input) * weights)
        return output + flat_delta.reshape_as(output)


def _make_ffn_expert(hidden_size, intermediate_size, r, alpha, dropout):
    return nn.ModuleDict({
        "gate": LoRAPair(
            hidden_size, intermediate_size, r, alpha, dropout),
        "up": LoRAPair(
            hidden_size, intermediate_size, r, alpha, dropout),
        "down": LoRAPair(
            intermediate_size, hidden_size, r, alpha, dropout),
    })


class RoutedLoRAMLP(nn.Module):
    """Existing FFN LoRA experts driven by a layer-level routing context."""

    def __init__(self, base_mlp, r, alpha, dropout=0.0):
        super().__init__()
        self.base_mlp = base_mlp
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad = False
        self.hidden_size = base_mlp.gate_proj.in_features
        self.intermediate_size = base_mlp.gate_proj.out_features
        self.activation = getattr(base_mlp, "act_fn", F.silu)
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.experts = nn.ModuleList()
        self._routing_context = None

    def add_experts(self, count):
        device = self.base_mlp.gate_proj.weight.device
        dtype = self.base_mlp.gate_proj.weight.dtype
        for _ in range(count):
            self.experts.append(_make_ffn_expert(
                self.hidden_size, self.intermediate_size, self.r,
                self.alpha, self.dropout).to(device=device, dtype=dtype))

    def set_routing_context(self, context):
        self._routing_context = context

    def forward(self, inputs):
        context = self._routing_context
        if context is None:
            if self.experts:
                raise RuntimeError("FFN experts used without routing context")
            return self.base_mlp(inputs)
        if context.num_experts > len(self.experts):
            raise ValueError(
                f"routing exposes {context.num_experts} experts but FFN "
                f"contains {len(self.experts)}")

        gate = self.base_mlp.gate_proj(inputs)
        up = self.base_mlp.up_proj(inputs)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        if flat_inputs.shape[0] != context.expert_indices.shape[0]:
            raise ValueError(
                "FFN token count does not match shared routing: "
                f"{flat_inputs.shape[0]} != {context.expert_indices.shape[0]}")
        flat_gate_delta = gate.new_zeros(
            (flat_inputs.shape[0], gate.shape[-1]))
        flat_up_delta = up.new_zeros((flat_inputs.shape[0], up.shape[-1]))
        routes = {}
        for expert_index in context.active_expert_ids():
            expert = self.experts[expert_index]
            slot, token_index = context.routes_for(expert_index)
            if token_index.numel() == 0:
                continue
            routes[expert_index] = (slot, token_index)
            expert_input = flat_inputs.index_select(0, token_index)
            weights = context.expert_weights[
                token_index, slot, None].to(expert_input.dtype)
            flat_gate_delta.index_add_(
                0, token_index, expert["gate"](expert_input) * weights)
            flat_up_delta.index_add_(
                0, token_index, expert["up"](expert_input) * weights)

        gate = gate + flat_gate_delta.reshape_as(gate)
        up = up + flat_up_delta.reshape_as(up)
        intermediate = self.activation(gate) * up
        output = self.base_mlp.down_proj(intermediate)
        flat_intermediate = intermediate.reshape(-1, intermediate.shape[-1])
        flat_down_delta = output.new_zeros(
            (flat_intermediate.shape[0], output.shape[-1]))
        for expert_index, (slot, token_index) in routes.items():
            expert = self.experts[expert_index]
            expert_input = flat_intermediate.index_select(0, token_index)
            weights = context.expert_weights[
                token_index, slot, None].to(expert_input.dtype)
            flat_down_delta.index_add_(
                0, token_index, expert["down"](expert_input) * weights)
        return output + flat_down_delta.reshape_as(output)


class SharedRouterDecoderLayer(nn.Module):
    """Llama/Qwen2 decoder layer with routing before self-attention."""

    def __init__(self, base_layer, r, alpha, top_k, aux_loss_coeff,
                 z_loss_coeff, routing_weight_mode, dropout):
        super().__init__()
        required = (
            "self_attn", "mlp", "input_layernorm",
            "post_attention_layernorm")
        missing = [name for name in required if not hasattr(base_layer, name)]
        if missing:
            raise TypeError(
                f"unsupported decoder layer {type(base_layer).__name__}; "
                f"missing {missing}")
        self.hidden_size = base_layer.hidden_size
        self.self_attn = base_layer.self_attn
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm

        projection_names = {
            "q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
        for target, name in projection_names.items():
            if not hasattr(self.self_attn, name):
                raise TypeError(
                    f"{type(self.self_attn).__name__} has no {name}")
            setattr(self.self_attn, name, RoutedLoRALinear(
                getattr(self.self_attn, name), r, alpha, dropout))
        self.mlp = RoutedLoRAMLP(base_layer.mlp, r, alpha, dropout)

        reference = self.self_attn.q_proj.base_layer.weight
        self.shared_expert_router = SharedExpertRouter(
            hidden_size=self.hidden_size,
            top_k=top_k,
            aux_loss_coeff=aux_loss_coeff,
            z_loss_coeff=z_loss_coeff,
            routing_weight_mode=routing_weight_mode,
            device=reference.device,
            dtype=reference.dtype,
        )

    @property
    def attention_expert_projections(self):
        return tuple(
            getattr(self.self_attn, f"{target}_proj")
            for target in ATTENTION_TARGETS)

    @property
    def num_experts(self):
        return self.shared_expert_router.num_experts

    def add_experts(self, count):
        old_count = self.num_experts
        self.shared_expert_router.add_experts(count)
        for projection in self.attention_expert_projections:
            projection.add_experts(count)
        self.mlp.add_experts(count)
        expected = old_count + count
        counts = [len(projection.experts)
                  for projection in self.attention_expert_projections]
        counts.append(len(self.mlp.experts))
        if any(value != expected for value in counts):
            raise RuntimeError(
                f"V3 expert pools grew out of sync: {counts}, "
                f"router={self.num_experts}")

    def _install_context(self, context):
        for projection in self.attention_expert_projections:
            projection.set_routing_context(context)
        self.mlp.set_routing_context(context)

    def _clear_context(self):
        self._install_context(None)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        routing_context = self.shared_expert_router(normalized)
        self._install_context(routing_context)
        try:
            attention_output, self_attn_weights = self.self_attn(
                hidden_states=normalized,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + attention_output
            residual = hidden_states
            ffn_input = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + self.mlp(ffn_input)
        finally:
            self._clear_context()

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


def shared_router_layers(model):
    return [module for module in model.modules()
            if isinstance(module, SharedRouterDecoderLayer)]


def attach_shared_qkvo_lora_moe(
    model, r, alpha, top_k, aux_loss_coeff, z_loss_coeff,
    routing_weight_mode="full_softmax", dropout=0.0,
):
    """Replace every decoder layer and freeze the complete backbone."""
    if r < 1:
        raise ValueError("LoRA rank must be positive")
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("V3 expects a causal LM with model.layers")
    wrapped_layers = []
    for layer in model.model.layers:
        if isinstance(layer, SharedRouterDecoderLayer):
            raise ValueError("model is already wrapped for V3")
        wrapped_layers.append(SharedRouterDecoderLayer(
            layer, r=r, alpha=alpha, top_k=top_k,
            aux_loss_coeff=aux_loss_coeff, z_loss_coeff=z_loss_coeff,
            routing_weight_mode=routing_weight_mode, dropout=dropout))
    model.model.layers = nn.ModuleList(wrapped_layers)
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def add_v3_experts(model, count):
    layers = shared_router_layers(model)
    if not layers:
        raise ValueError("model has no V3 shared-router layers")
    for layer in layers:
        layer.add_experts(count)


def freeze_v3_experts(model, trainable_expert_indices=None):
    for layer in shared_router_layers(model):
        pools = [layer.mlp.experts]
        pools.extend(
            projection.experts
            for projection in layer.attention_expert_projections)
        for pool in pools:
            for index, expert in enumerate(pool):
                trainable = (
                    trainable_expert_indices is not None
                    and index in trainable_expert_indices)
                for parameter in expert.parameters():
                    parameter.requires_grad = trainable


def freeze_v3_routers(model, trainable):
    for layer in shared_router_layers(model):
        router = layer.shared_expert_router.router
        if router is not None:
            for parameter in router.parameters():
                parameter.requires_grad = trainable


def set_v3_router_token_mask(model, attention_mask):
    for layer in shared_router_layers(model):
        layer.shared_expert_router._router_token_mask = attention_mask


def begin_v3_sample_router_gradient(model):
    """Capture per-example router gradients in the upcoming backward."""
    for layer in shared_router_layers(model):
        router = layer.shared_expert_router
        router._sample_grad_scores.clear()
        router._capture_sample_grad = True


def end_v3_sample_router_gradient(model):
    """Return the concatenated-router gradient norm for each batch row."""
    squared_norm = None
    for layer in shared_router_layers(model):
        router = layer.shared_expert_router
        router._capture_sample_grad = False
        if router._sample_grad_scores:
            layer_score = torch.stack(router._sample_grad_scores).sum(dim=0)
            squared_norm = (
                layer_score if squared_norm is None
                else squared_norm + layer_score)
        router._sample_grad_scores.clear()
    if squared_norm is None:
        return None
    return squared_norm.clamp_min(0).sqrt()


def collect_v3_moe_losses(model):
    total = None
    for layer in shared_router_layers(model):
        loss = layer.shared_expert_router._last_moe_loss
        if loss is not None:
            total = loss if total is None else total + loss
    return total


@contextmanager
def limit_v3_experts(model, active_expert_count):
    layers = shared_router_layers(model)
    previous = [
        layer.shared_expert_router._active_expert_count for layer in layers]
    for layer in layers:
        if not 0 <= active_expert_count <= layer.num_experts:
            raise ValueError(
                f"active_expert_count={active_expert_count}, "
                f"layer experts={layer.num_experts}")
        layer.shared_expert_router._active_expert_count = active_expert_count
    try:
        yield
    finally:
        for layer, value in zip(layers, previous):
            layer.shared_expert_router._active_expert_count = value


def _v3_expert_parameters(model):
    parameters = []
    for layer in shared_router_layers(model):
        for expert in layer.mlp.experts:
            parameters.extend(expert.parameters())
        for projection in layer.attention_expert_projections:
            for expert in projection.experts:
                parameters.extend(expert.parameters())
    return parameters


class Ours_LoRA_MoE_V3(Ours_LoRA_MoE_V2):
    """V2 training mechanics with shared-router QKVO+FFN experts."""

    save_key_substrings = [
        ".shared_expert_router.router.",
        ".self_attn.q_proj.experts.",
        ".self_attn.k_proj.experts.",
        ".self_attn.v_proj.experts.",
        ".self_attn.o_proj.experts.",
        ".mlp.experts.",
    ]

    def _gradient_memory_enabled(self):
        return getattr(
            self.args, "replay_selection_mode", "random") == "router_gradient"

    def _begin_gradient_memory_batch(self):
        if self._gradient_memory_enabled():
            begin_v3_sample_router_gradient(self.raw_model)

    def _record_gradient_memory_batch(
            self, task, source_batch, model_batch, window_size):
        if not self._gradient_memory_enabled():
            return
        norms = end_v3_sample_router_gradient(self.raw_model)
        if norms is None:
            return
        sources = source_batch.get("sources", [])
        if len(sources) != norms.numel():
            raise RuntimeError(
                "router-gradient replay scoring requires one source_index "
                "per batch row")
        try:
            source_indices = [int(source) for source in sources]
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "router-gradient replay scoring requires the pretokenized "
                "TRACE cache whose sources are original source_index values") from error

        labels = model_batch.get("labels")
        if labels is None:
            token_counts = torch.ones_like(norms)
            total_tokens = float(norms.numel())
        else:
            token_counts = labels.ne(-100).sum(dim=1).to(norms.dtype)
            total_tokens = float(token_counts.sum().item())
        # Causal-LM loss is reduced over all supervised tokens in the batch.
        # Undo that cross-sample denominator so scores approximate each
        # example's mean-token router gradient and do not merely select the
        # longest sequences. Undo gradient-accumulation scaling as well.
        normalized = norms * (
            total_tokens / token_counts.clamp_min(1)) * float(window_size)
        stats = getattr(self, "_router_gradient_memory_stats", None)
        if stats is None:
            stats = self._router_gradient_memory_stats = {}
        task_stats = stats.setdefault(task, {})
        for source_index, score in zip(
                source_indices, normalized.detach().float().cpu().tolist()):
            aggregate = task_stats.setdefault(
                source_index, {"sum": 0.0, "count": 0, "max": 0.0})
            aggregate["sum"] += float(score)
            aggregate["count"] += 1
            aggregate["max"] = max(aggregate["max"], float(score))

    def _finalize_gradient_replay_memory(self, task):
        """Replace this task's random memory with top mean-gradient samples."""
        if not self._gradient_memory_enabled():
            self._ensure_fixed_task_subset(task)
            return
        local = getattr(self, "_router_gradient_memory_stats", {}).pop(task, {})
        distributed = torch.distributed.is_initialized()
        if distributed:
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]

        selected = None
        selected_records = None
        dataset = self.train_task_list[task].dataset
        unique_samples = max(1, min(
            len(dataset), round(len(dataset) * self.args.replay_subset_ratio)))
        if self.args.global_rank == 0:
            merged = {}
            for shard in gathered:
                for source_index, values in shard.items():
                    aggregate = merged.setdefault(
                        int(source_index), {"sum": 0.0, "count": 0, "max": 0.0})
                    aggregate["sum"] += float(values["sum"])
                    aggregate["count"] += int(values["count"])
                    aggregate["max"] = max(
                        aggregate["max"], float(values["max"]))
            if len(merged) < unique_samples:
                raise RuntimeError(
                    f"router-gradient replay scoring saw only {len(merged)} "
                    f"unique {task} samples; need {unique_samples}")
            ranking = sorted(
                merged.items(),
                key=lambda item: (
                    -(item[1]["sum"] / max(1, item[1]["count"])),
                    item[0]))
            selected = [index for index, _ in ranking[:unique_samples]]
            selected_records = [{
                "source_index": index,
                "mean_router_gradient_norm": (
                    values["sum"] / max(1, values["count"])),
                "max_router_gradient_norm": values["max"],
                "observations": values["count"],
            } for index, values in ranking[:unique_samples]]
        if distributed:
            payload = [selected, selected_records]
            torch.distributed.broadcast_object_list(payload, src=0)
            selected, selected_records = payload

        self._fixed_task_subset_indices[task] = selected
        self._fixed_task_subsets[task] = Subset(dataset, selected)
        if self.args.global_rank == 0:
            task_index = list(self.train_task_list).index(task)
            safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task)
            output_dir = os.path.join(
                self.args.output_dir, "fixed_replay_memory")
            os.makedirs(output_dir, exist_ok=True)
            metadata = {
                "schema_version": 2,
                "task_index": task_index,
                "task": task,
                "source_samples": len(dataset),
                "subset_ratio": self.args.replay_subset_ratio,
                "unique_samples": unique_samples,
                "selection_mode": "router_gradient",
                "score": "mean_per_sample_router_gradient_norm",
                "length_normalization": "mean_supervised_token",
                "indices": selected,
                "scores": selected_records,
                "available_from_next_task_for_v2_past_replay": True,
            }
            with open(os.path.join(
                    output_dir, f"task_{task_index}_{safe_task}.json"),
                    "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)
            print_rank_0(
                f"[router-gradient memory] {task}: selected top "
                f"{unique_samples}/{len(dataset)} samples; "
                f"score range={selected_records[-1]['mean_router_gradient_norm']:.6g}"
                f"..{selected_records[0]['mean_router_gradient_norm']:.6g}",
                self.args.global_rank)

    @staticmethod
    def _snapshot_old_router_rows(model, old_expert_count):
        return [
            layer.shared_expert_router.weight[:old_expert_count]
            .detach().clone()
            for layer in shared_router_layers(model)
        ]

    @staticmethod
    def _freeze_old_router_row_update(model, snapshots, old_expert_count):
        for layer, snapshot in zip(shared_router_layers(model), snapshots):
            weight = layer.shared_expert_router.weight
            if weight.grad is not None:
                weight.grad[:old_expert_count].zero_()
            with torch.no_grad():
                weight[:old_expert_count].copy_(snapshot)

    @staticmethod
    def _slice_probe_batch(batch, start, stop):
        sliced = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                sliced[key] = value[start:stop]
            elif key == "sources":
                sliced[key] = list(value[start:stop])
        return sliced

    def _merge_probe_batches(self, batches):
        batches = [batch for batch in batches
                   if batch and batch["input_ids"].shape[0] > 0]
        if not batches:
            raise ValueError("epoch probe cannot merge an empty batch list")
        maximum = max(batch["input_ids"].shape[1] for batch in batches)
        merged = {"sources": []}
        for key in ("input_ids", "attention_mask", "labels"):
            values = []
            pad_value = (
                self.tokenizer.pad_token_id if key == "input_ids"
                else -100 if key == "labels" else 0)
            for batch in batches:
                value = batch[key]
                values.append(F.pad(
                    value, (0, maximum - value.shape[1]), value=pad_value))
            merged[key] = torch.cat(values, dim=0)
        for batch in batches:
            merged["sources"].extend(batch.get("sources", []))
        return merged

    def _fixed_eval_rows(self, task, count):
        if count < 1:
            return None
        loader = self.eval_task_list[task]
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(0)
        pieces = []
        remaining = count
        for batch in loader:
            take = min(remaining, int(batch["input_ids"].shape[0]))
            pieces.append(self._slice_probe_batch(batch, 0, take))
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise ValueError(
                f"eval split for {task} has fewer than {count} local samples")
        return self._merge_probe_batches(pieces)

    @staticmethod
    def _answer_loss_sums(logits, labels, row_start, row_stop):
        shifted_logits = logits[row_start:row_stop, :-1].float()
        shifted_labels = labels[row_start:row_stop, 1:]
        token_losses = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1), ignore_index=-100,
            reduction="none").reshape_as(shifted_labels)
        valid = shifted_labels.ne(-100)
        return token_losses[valid].sum(), valid.sum()

    def _run_epoch_probe(self, task, i_task, epoch, device):
        total_global = int(getattr(
            self.args, "v3_epoch_probe_samples", 64))
        if total_global == 0:
            return
        distributed = torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        if total_global % world_size != 0:
            raise ValueError(
                "V3 epoch probe samples must be divisible by world size: "
                f"{total_global} % {world_size}")
        local_total = total_global // world_size
        local_current = local_total if i_task == 0 else local_total // 2
        local_past = local_total - local_current
        if i_task > 0 and (local_current < 1 or local_past < 1):
            raise ValueError(
                "V3 epoch probe needs at least one current and past sample "
                "per rank")

        current = self._fixed_eval_rows(task, local_current)
        pieces = [current]
        if local_past:
            old_tasks = list(self.train_task_list)[:i_task]
            counts = {name: 0 for name in old_tasks}
            for local_slot in range(local_past):
                global_slot = rank * local_past + local_slot
                counts[old_tasks[global_slot % len(old_tasks)]] += 1
            for name in old_tasks:
                if counts[name]:
                    pieces.append(self._fixed_eval_rows(name, counts[name]))
        probe = self._merge_probe_batches(pieces)
        probe.pop("sources", None)
        probe = to_device(probe, device)

        layers = shared_router_layers(self.raw_model)
        for layer in layers:
            layer.shared_expert_router._capture_probe_routing = True
        self.raw_model.eval()
        set_v3_router_token_mask(self.raw_model, probe["attention_mask"])
        try:
            with torch.no_grad():
                outputs = self.raw_model(**probe, use_cache=False)
            current_loss, current_tokens = self._answer_loss_sums(
                outputs.logits, probe["labels"], 0, local_current)
            past_loss, past_tokens = self._answer_loss_sums(
                outputs.logits, probe["labels"], local_current, local_total)

            old_expert_count = max(
                0, layers[0].num_experts - self.args.experts_per_task)
            route_values = []
            valid = probe["attention_mask"].bool()
            for row_start, row_stop in (
                    (0, local_current), (local_current, local_total)):
                selected_new = valid.new_zeros((), dtype=torch.long)
                selected_total = valid.new_zeros((), dtype=torch.long)
                row_valid = valid[row_start:row_stop]
                for layer in layers:
                    indices = layer.shared_expert_router._last_probe_indices
                    if indices is None:
                        raise RuntimeError("router probe capture was not populated")
                    top1 = indices[:, 0].reshape_as(valid)[row_start:row_stop]
                    selected_new += (top1[row_valid] >= old_expert_count).sum()
                    selected_total += row_valid.sum()
                route_values.extend((selected_new, selected_total))

            stats = torch.stack([
                current_loss, current_tokens.float(),
                past_loss, past_tokens.float(),
                route_values[0].float(), route_values[1].float(),
                route_values[2].float(), route_values[3].float(),
                current_loss.new_tensor(float(local_current)),
                current_loss.new_tensor(float(local_past)),
            ])
            if distributed:
                torch.distributed.all_reduce(stats)
        finally:
            set_v3_router_token_mask(self.raw_model, None)
            for layer in layers:
                router = layer.shared_expert_router
                router._capture_probe_routing = False
                router._last_probe_indices = None
            self.model.train()

        if self.args.global_rank == 0:
            def ratio(numerator, denominator):
                return float((numerator / denominator.clamp_min(1)).item())

            record = {
                "schema_version": 1,
                "task": task,
                "round": int(i_task),
                "epoch": int(epoch + 1),
                "global_samples": int(stats[8].item() + stats[9].item()),
                "current_samples": int(stats[8].item()),
                "past_samples": int(stats[9].item()),
                "current_answer_ce": ratio(stats[0], stats[1]),
                "past_answer_ce": (
                    ratio(stats[2], stats[3]) if stats[3].item() else None),
                "current_new_expert_route_fraction": ratio(stats[4], stats[5]),
                "past_new_expert_route_fraction": (
                    ratio(stats[6], stats[7]) if stats[7].item() else None),
            }
            path = os.path.join(self.args.output_dir, "epoch_probe.jsonl")
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print_rank_0(
                "[epoch probe] " + json.dumps(record, sort_keys=True),
                self.args.global_rank)

    def train_one_task(self, task, i_task, epochs):
        device = (
            torch.device("cuda", self.args.local_rank)
            if self.args.local_rank != -1 else torch.device("cuda"))
        args = self.args
        # Gradient-selected memory is constructed from this task's ordinary
        # training backwards and only becomes available to the next task.
        if not self._gradient_memory_enabled():
            self._ensure_fixed_task_subset(task)
        layers = shared_router_layers(self.raw_model)
        old_expert_count = layers[0].num_experts
        add_v3_experts(self.raw_model, args.experts_per_task)
        new_indices = set(range(
            old_expert_count, old_expert_count + args.experts_per_task))

        primary_loader = self.train_task_list[task]
        if args.training_version in V3_NEW_TRAINING_VERSIONS:
            kd_loader = self._build_v2_kd_loader(
                i_task, primary_loader, epochs)
            replay_loader = self._build_v2_replay_loader(
                i_task, primary_loader, epochs)
            kd_epochs = self._v2_kd_epochs(epochs)
        else:
            kd_loader = self._build_v2_memory_loader(i_task, role="kd")
            replay_loader = self._build_v2_memory_loader(
                i_task, role="replay")
            kd_epochs = 1
        ran_kd_init = (
            old_expert_count > 0 and kd_loader is not None
            and args.v2_kd_loss_coeff > 0)
        if ran_kd_init:
            self._run_v2_kd_init(
                kd_loader, old_expert_count, new_indices, device, task,
                kd_epochs=kd_epochs)

        hidden_mse_teacher = None
        if (replay_loader is not None
                and self._joint_replay_objective() == "hidden_mse"):
            if not ran_kd_init:
                raise RuntimeError(
                    "hidden-MSE replay requires a completed expansion KD-init "
                    "before the post-KD teacher snapshot")
            hidden_mse_teacher = self._make_post_kd_hidden_mse_teacher()

        self._set_phase_gradient_accumulation(
            args.batch_by_task[task], f"{task} v3 primary")
        self._set_grad_ckpt(
            replay_loader is not None
            or task in getattr(args, "ckpt_tasks", set()))
        freeze_v3_experts(self.raw_model, new_indices)
        freeze_v3_routers(self.raw_model, True)
        self._reinit_engine(self._optimizer_update_count(
            primary_loader, epochs))
        try:
            if replay_loader is None:
                self._run_v3_primary_epochs(
                    primary_loader, epochs, device,
                    f"{task} [v3 primary-only; no prior memory]",
                    task=task, i_task=i_task)
            else:
                objective = self._joint_replay_objective()
                self._run_v2_joint_epochs(
                    primary_loader, replay_loader, epochs, device,
                    f"{task} [v3 joint every-update router "
                    f"{objective} replay]",
                    task=task, i_task=i_task,
                    hidden_mse_teacher=hidden_mse_teacher)
        finally:
            if hidden_mse_teacher is not None:
                del hidden_mse_teacher
                torch.cuda.empty_cache()
        self._finalize_gradient_replay_memory(task)

    def _run_v3_primary_epochs(
            self, dataloader, epochs, device, phase_name,
            task=None, i_task=None):
        args = self.args
        total_steps = epochs * len(dataloader)
        progress = tqdm(
            total=total_steps, leave=True, disable=args.global_rank != 0)
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            self.model.train()
            for step, source_batch in enumerate(dataloader):
                self._count_workload_batch("new_task", source_batch)
                batch = dict(source_batch)
                batch.pop("sources", None)
                batch = to_device(batch, device)
                accum = max(1, args.gradient_accumulation_steps)
                window_start = (step // accum) * accum
                window_size = min(accum, len(dataloader) - window_start)
                window_index = step - window_start
                should_step = window_index + 1 == window_size
                sync = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not should_step
                    else nullcontext())
                set_v3_router_token_mask(
                    self.raw_model, batch.get("attention_mask"))
                self._begin_gradient_memory_batch()
                try:
                    with sync:
                        outputs = self.model(**batch, use_cache=False)
                        moe_loss = collect_v3_moe_losses(self.raw_model)
                        loss = outputs.loss
                        if moe_loss is not None:
                            loss = loss + moe_loss
                        (loss / window_size).backward()
                finally:
                    set_v3_router_token_mask(self.raw_model, None)
                self._record_gradient_memory_batch(
                    task, source_batch, batch, window_size)
                if args.global_rank == 0:
                    progress.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(dataloader)):
                        progress.set_description(
                            f"{phase_name} e{epoch + 1} s{step} "
                            f"loss={loss.detach().float().item():.4f}",
                            refresh=False)
                if should_step:
                    trainable = [parameter for parameter in
                                 self.raw_model.parameters()
                                 if parameter.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()
            if task is not None and i_task is not None:
                self._run_epoch_probe(task, i_task, epoch, device)
        progress.close()

    def _run_v2_kd_init(self, dataloader, old_expert_count, new_indices,
                        device, task, kd_epochs=1):
        args = self.args
        kd_epochs = int(kd_epochs)
        if kd_epochs < 1:
            raise ValueError(f"KD epochs must be positive, got {kd_epochs}")
        self._set_phase_gradient_accumulation(
            dataloader.batch_size, f"{task} v3 KD init")
        self._set_grad_ckpt(True)
        freeze_v3_experts(self.raw_model, new_indices)
        freeze_v3_routers(self.raw_model, True)
        total_microsteps = kd_epochs * len(dataloader)
        updates = self._optimizer_update_count(dataloader, kd_epochs)
        self._reinit_engine(
            updates,
            learning_rate=args.v2_kd_learning_rate or args.learning_rate)
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
                sync = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) and not should_step
                    else nullcontext())
                set_v3_router_token_mask(
                    self.raw_model, batch.get("attention_mask"))
                try:
                    self.raw_model.eval()
                    with torch.no_grad(), limit_v3_experts(
                            self.raw_model, old_expert_count):
                        teacher_logits = self.raw_model(
                            **batch, use_cache=False).logits.detach()
                    self.model.train()
                    with sync:
                        student_logits = self.model(
                            **batch, use_cache=False).logits
                        kd_loss = self._kd_kl_loss(
                            student_logits, teacher_logits, batch, args)
                        loss = args.v2_kd_loss_coeff * kd_loss
                        (loss / window_size).backward()
                finally:
                    set_v3_router_token_mask(self.raw_model, None)
                if args.global_rank == 0:
                    progress.update(1)
                    if (step % args.loss_log_interval == 0
                            or step + 1 == len(dataloader)):
                        global_step = kd_epoch * len(dataloader) + step
                        progress.set_description(
                            f"{task} [v3 KD-init "
                            f"{len(dataloader.dataset)} samples x "
                            f"{kd_epochs} epochs] s{global_step} "
                            f"kl={kd_loss.detach().float().item():.5f}",
                            refresh=False)
                if should_step:
                    self._freeze_old_router_row_update(
                        self.raw_model, old_router_rows, old_expert_count)
                    trainable = [parameter for parameter in
                                 self.raw_model.parameters()
                                 if parameter.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    self._freeze_old_router_row_update(
                        self.raw_model, old_router_rows, old_expert_count)
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()
                    completed_updates += 1
        try:
            expected_local_samples = kd_epochs * len(dataloader.sampler)
        except TypeError:
            # Tiny unit loaders may expose only a set_epoch stub. Production
            # distributed samplers always define their exact local length.
            expected_local_samples = consumed_local_samples
        if completed_microsteps != total_microsteps:
            raise RuntimeError(
                f"V3 KD microsteps {completed_microsteps}/{total_microsteps}")
        if completed_updates != updates:
            raise RuntimeError(
                f"V3 KD optimizer updates {completed_updates}/{updates}")
        if consumed_local_samples != expected_local_samples:
            raise RuntimeError(
                "V3 KD local sample exposure mismatch: "
                f"{consumed_local_samples}/{expected_local_samples}")
        progress.close()

    @contextmanager
    def _router_only_replay(self):
        expert_parameters = _v3_expert_parameters(self.raw_model)
        previous = [parameter.requires_grad
                    for parameter in expert_parameters]
        for parameter in expert_parameters:
            parameter.requires_grad = False
        try:
            yield
        finally:
            for parameter, requires_grad in zip(
                    expert_parameters, previous):
                parameter.requires_grad = requires_grad

    @contextmanager
    def _suppress_replay_router_losses(self):
        """Skip unused shared-router aux/z losses on replay forwards."""
        routers = [
            layer.shared_expert_router
            for layer in shared_router_layers(self.raw_model)]
        previous = [router._suppress_router_loss for router in routers]
        for router in routers:
            router._suppress_router_loss = True
        try:
            yield
        finally:
            for router, value in zip(routers, previous):
                router._suppress_router_loss = value

    def _run_v2_joint_epochs(
            self, primary_loader, memory_loader, epochs, device, phase_name,
            task=None, i_task=None, hidden_mse_teacher=None):
        """Pair every optimizer update with V3 router-only replay."""
        args = self.args
        replay_objective = self._joint_replay_objective()
        if replay_objective not in {"lm", "hidden_mse"}:
            raise ValueError(
                f"unsupported V3 joint replay objective: {replay_objective}")
        if ((replay_objective == "hidden_mse")
                != (hidden_mse_teacher is not None)):
            raise ValueError(
                "hidden-MSE replay objective and post-KD teacher must be "
                "enabled together")
        is_v3_new = getattr(
            args, "training_version", "v3") in V3_NEW_TRAINING_VERSIONS
        total_steps = epochs * len(primary_loader)
        accum = max(1, args.gradient_accumulation_steps)
        total_updates = epochs * math.ceil(len(primary_loader) / accum)
        stored_memory_exposures = len(memory_loader.dataset)
        if total_steps < 1 or stored_memory_exposures < 1:
            raise ValueError("v3 joint training requires non-empty loaders")
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
        # Legacy V3 spreads one fixed stream over the complete phase. V3-new
        # consumes one active-memory stream per primary epoch; the polymorphic
        # budget above captures that contract without changing gradient flow.
        self._replay_exposure_assignment(
            total_memory_exposures, total_updates, 0, world_size, rank)
        progress = tqdm(
            total=total_steps, leave=True, disable=args.global_rank != 0)

        memory_sampler = getattr(memory_loader, "sampler", None)
        replay_pass_index = 0
        if is_v3_new:
            self._set_v2_replay_memory_sampler_pass(
                memory_loader, replay_pass_index)
        elif hasattr(memory_sampler, "set_epoch"):
            memory_sampler.set_epoch(0)
        memory_iterator = iter(memory_loader)
        consumed_global_exposures = 0
        consumed_local_exposures = 0
        global_update = 0
        total_primary_tokens = 0
        total_replay_tokens = 0
        total_replay_steps = 0
        consumed_global_new_exposures = 0

        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(epochs):
            primary_sampler = getattr(primary_loader, "sampler", None)
            if hasattr(primary_sampler, "set_epoch"):
                primary_sampler.set_epoch(epoch)
            epoch_primary_tokens = 0
            epoch_replay_tokens = 0
            epoch_replay_steps = 0

            for step, source_batch in enumerate(primary_loader):
                local_primary_samples = int(source_batch["input_ids"].shape[0])
                global_primary_samples = local_primary_samples * world_size
                consumed_global_new_exposures += global_primary_samples
                primary_tokens = self._valid_token_count(source_batch)
                epoch_primary_tokens += primary_tokens
                total_primary_tokens += primary_tokens
                self._count_workload_batch("new_task", source_batch)
                primary = dict(source_batch)
                primary.pop("sources", None)
                primary = to_device(primary, device)
                window_start = (step // accum) * accum
                window_size = min(accum, len(primary_loader) - window_start)
                window_index = step - window_start
                should_step = window_index + 1 == window_size
                replay_log_due = (
                    should_step
                    and (step % args.loss_log_interval == 0
                         or step + 1 == len(primary_loader)))

                no_sync = (
                    self.model.no_sync()
                    if isinstance(self.model, DDP) else nullcontext())
                set_v3_router_token_mask(
                    self.raw_model, primary.get("attention_mask"))
                self._begin_gradient_memory_batch()
                try:
                    with no_sync:
                        primary_outputs = self.model(
                            **primary, use_cache=False)
                        moe_loss = collect_v3_moe_losses(self.raw_model)
                        primary_loss = primary_outputs.loss
                        if moe_loss is not None:
                            primary_loss = primary_loss + moe_loss
                        (primary_loss / window_size).backward()
                finally:
                    set_v3_router_token_mask(self.raw_model, None)
                self._record_gradient_memory_batch(
                    task, source_batch, primary, window_size)

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
                            replay_pass_index += 1
                            if is_v3_new:
                                self._set_v2_replay_memory_sampler_pass(
                                    memory_loader, replay_pass_index)
                            elif hasattr(memory_sampler, "set_epoch"):
                                memory_sampler.set_epoch(0)
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
                            set_v3_router_token_mask(
                                self.raw_model, replay.get("attention_mask"))
                            try:
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
                                set_v3_router_token_mask(self.raw_model, None)
                        consumed_local_exposures += len(replay_source_chunk)

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
                        progress.set_description(
                            f"{phase_name} e{epoch + 1} s{step} "
                            f"new={primary_loss.detach().float().item():.4f} "
                            f"replay={replay_text} n={global_replay_count} "
                            f"scale={replay_scale_text} "
                            f"budget={consumed_global_exposures}/"
                            f"{total_memory_exposures}", refresh=False)
                if should_step:
                    self._manual_average_gradients(self.raw_model)
                    trainable = [parameter for parameter in
                                 self.raw_model.parameters()
                                 if parameter.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self._count_workload_update()
                    global_update += 1

            if args.global_rank == 0:
                print_rank_0(
                    f"{phase_name} epoch {epoch + 1}: valid tokens "
                    f"new={epoch_primary_tokens}, replay={epoch_replay_tokens}, "
                    f"replay_updates={epoch_replay_steps}, "
                    f"global_exposures={consumed_global_exposures}/"
                    f"{total_memory_exposures}", args.global_rank)
            if task is not None and i_task is not None:
                self._run_epoch_probe(task, i_task, epoch, device)

        progress.close()
        if global_update != total_updates:
            raise RuntimeError(
                f"v3 joint updates {global_update}/{total_updates}")
        if consumed_global_exposures != total_memory_exposures:
            raise RuntimeError(
                "v3 exact replay schedule consumed "
                f"{consumed_global_exposures}/{total_memory_exposures} "
                "global exposures")
        if replay_ratio > 0:
            expected_replay = self._expected_joint_replay_exposures(
                consumed_global_new_exposures, epochs, replay_ratio)
            if expected_replay != total_memory_exposures:
                raise RuntimeError(
                    "joint replay target does not match actual primary "
                    f"exposure: replay={total_memory_exposures}, "
                    f"expected={expected_replay}")
        if consumed_local_exposures != expected_local_exposures:
            raise RuntimeError(
                "v3 local replay shard consumed "
                f"{consumed_local_exposures}/{expected_local_exposures}")
        if is_v3_new:
            completed_replay_passes = replay_pass_index + 1
            self._validate_v2_replay_memory_sampler_passes(
                memory_loader, completed_replay_passes, epochs)
        if replay_ratio == 0:
            try:
                next(memory_iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError(
                    "v3 replay stream has unconsumed local records")
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

    def save_model(self, round):
        CL_Base_Model.save_model(self, round)
        if self.args.global_rank == 0:
            output_dir = os.path.join(self.args.output_dir, str(round))
            save_v3_meta(
                self.raw_model, output_dir, self.args, trainer=self)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


class Ours_LoRA_MoE_V3_New(Ours_LoRA_MoE_V3, Ours_LoRA_MoE_V2_New):
    """V3 QKVO+FFN experts under the strict V2-new memory contract.

    The multiple-inheritance order deliberately keeps every V3 forward,
    gradient-mask, KD, and router-only replay implementation authoritative,
    while resolving persistent-memory construction, active-stream budgets,
    sampler-pass identity checks, and epoch-scaled exposure accounting from
    :class:`Ours_LoRA_MoE_V2_New` before the shared legacy V2 base.
    """

    def _gradient_memory_enabled(self):
        selection = getattr(
            self.args, "replay_selection_mode", "random")
        if selection != "random":
            raise ValueError(
                "v3_new currently requires deterministic random persistent "
                "memory; router_gradient does not implement stable nested "
                "V2-new identities")
        return False


def save_v3_meta(model, output_dir, args, trainer=None):
    layers = shared_router_layers(model)
    if not layers:
        raise ValueError("cannot save V3 metadata without V3 layers")
    layer = layers[0]
    router = layer.shared_expert_router
    payload = {
        "schema_version": 3,
        "architecture": V3_ARCHITECTURE,
        "training_version": getattr(args, "training_version", "v3"),
        "r": layer.mlp.r,
        "attention_rank": layer.self_attn.q_proj.r,
        "alpha": layer.mlp.alpha,
        "dropout": layer.mlp.dropout,
        "attention_targets": list(ATTENTION_TARGETS),
        "top_k": router.top_k,
        "aux_loss_coeff": router.aux_loss_coeff,
        "z_loss_coeff": router.z_loss_coeff,
        "routing_weight_mode": router.routing_weight_mode,
        "num_experts": layer.num_experts,
        "experts_per_task": args.experts_per_task,
        "router_position": "post_input_layernorm_pre_self_attention",
        "training_profile": {
            "format": args.train_format,
            "chat_template_source": getattr(
                args, "chat_template_source", None),
            "max_length": args.max_train_len or (
                args.max_prompt_len + args.max_ans_len),
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_epsilon": args.adam_epsilon,
        },
        "replay_memory": {
            "subset_ratio_per_task": args.replay_subset_ratio,
            "selection_mode": getattr(
                args, "replay_selection_mode", "random"),
            "exposure_samples_per_round":
                args.router_replay_exposure_samples,
            "v1_router_retune_enabled": False,
            "distribution": args.replay_distribution,
            "subset_seed": args.replay_subset_seed,
        },
        "v2": {
            "memory_batch_size": args.v2_memory_batch_size,
            "kd_memory_batch_size": getattr(
                args, "v2_kd_memory_batch_size", 0),
            "effective_kd_memory_batch_size": (
                getattr(args, "v2_kd_memory_batch_size", 0)
                or args.v2_memory_batch_size or 1),
            "effective_replay_memory_batch_size": (
                args.v2_memory_batch_size or 1),
            "replay_forward_batch_size": int(getattr(
                args, "v2_replay_forward_batch_size", 1)),
            "kd_exposure_samples_per_round":
                args.router_replay_exposure_samples,
            "kd_loss_coeff": args.v2_kd_loss_coeff,
            "kd_temperature": args.v2_kd_temperature,
            "kd_learning_rate": args.v2_kd_learning_rate,
            "kd_chunk_tokens": args.v2_kd_chunk_tokens,
            "kd_token_scope": args.v2_kd_token_scope,
            "joint_replay_loss_coeff": args.v2_joint_replay_loss_coeff,
            "joint_replay_objective": str(getattr(
                args, "v2_joint_replay_objective", "lm")),
            "hidden_mse_loss_coeff": float(getattr(
                args, "v2_hidden_mse_loss_coeff", 1.0)),
            "hidden_mse_teacher": "expanded_post_kd_init",
            "hidden_mse_targets": "all_decoder_layer_outputs",
            "hidden_mse_reduction":
                "active_sample_mean_equal_layer_mean",
            "joint_new_to_replay_sample_ratio": getattr(
                args, "v2_joint_new_to_replay_ratio", 0),
            "stored_replay_pool_exposures_per_round":
                args.router_replay_exposure_samples,
            "joint_replay_schedule":
                "every_optimizer_update_fixed_total_no_epoch_multiplier",
            "joint_replay_reduction": "active_sample_mean",
            "joint_replay_forward_reduction":
                "packed_per_sample_token_mean_then_sum",
            "max_replay_batches_per_step":
                args.v2_max_replay_batches_per_step,
        },
    }
    if (getattr(args, "training_version", "v3")
            in V3_NEW_TRAINING_VERSIONS):
        if trainer is None:
            raise ValueError("v3_new metadata requires its trainer state")
        active = int(args.v2_new_active_memory_cap)
        persistent_records = getattr(
            trainer, "_v2_new_persistent_memory_records", {})
        persisted_identities = {
            task: {
                "resolved_seed": record["resolved_seed"],
                "indices_sha256": record["indices_sha256"],
            }
            for task, record in persistent_records.items()
        }
        sampler_digests = getattr(
            trainer, "_v2_new_sampler_pass_digests", {})
        payload["replay_memory"] = {
            "persistent_samples_per_task":
                args.v2_new_persistent_samples_per_task,
            "persistent_selection_mode": args.replay_selection_mode,
            "configured_subset_seed": args.replay_subset_seed,
            "resolved_base_subset_seed": trainer._fixed_subset_seed(),
            "persisted_identities": persisted_identities,
            "active_stream_samples_per_primary_epoch": active,
            "distribution": "equal_task",
            "v1_router_retune_enabled": False,
        }
        payload["v2"] = {
            "memory_batch_size": args.v2_memory_batch_size,
            "kd_memory_batch_size": args.v2_kd_memory_batch_size,
            "effective_kd_memory_batch_size": (
                args.v2_kd_memory_batch_size
                or args.v2_memory_batch_size or 1),
            "effective_replay_memory_batch_size":
                args.v2_memory_batch_size or 1,
            "replay_forward_batch_size": int(getattr(
                args, "v2_replay_forward_batch_size", 1)),
            "kd_loss_coeff": args.v2_kd_loss_coeff,
            "kd_temperature": args.v2_kd_temperature,
            "kd_learning_rate": args.v2_kd_learning_rate,
            "kd_chunk_tokens": args.v2_kd_chunk_tokens,
            "kd_token_scope": args.v2_kd_token_scope,
            "joint_replay_loss_coeff": args.v2_joint_replay_loss_coeff,
            "joint_replay_objective": str(getattr(
                args, "v2_joint_replay_objective", "lm")),
            "hidden_mse_loss_coeff": float(getattr(
                args, "v2_hidden_mse_loss_coeff", 1.0)),
            "hidden_mse_teacher": "expanded_post_kd_init",
            "hidden_mse_targets": "all_decoder_layer_outputs",
            "hidden_mse_reduction":
                "active_sample_mean_equal_layer_mean",
            "joint_new_to_replay_sample_ratio":
                args.v2_joint_new_to_replay_ratio,
            "joint_replay_schedule":
                "every_optimizer_update_active_stream_per_primary_epoch",
            "joint_replay_reduction": "active_sample_mean",
            "joint_replay_forward_reduction":
                "packed_per_sample_token_mean_then_sum",
            "max_replay_batches_per_step":
                args.v2_max_replay_batches_per_step,
            "kd_active_stream_samples_per_pass": active,
            "kd_active_stream_passes": "match_primary_epochs",
            "kd_total_exposure_strategy":
                "samples_per_pass_times_primary_epochs",
            "joint_replay_active_stream_samples_per_primary_epoch": active,
            "joint_replay_total_exposure_strategy":
                "samples_per_primary_epoch_times_primary_epochs",
        }
        multiplier = int(args.v2_kd_pass_multiplier)
        if multiplier != 1:
            payload["v2"].update({
                "kd_active_stream_passes":
                    "primary_epochs_times_multiplier",
                "kd_pass_multiplier": multiplier,
                "kd_total_exposure_strategy":
                    "samples_per_pass_times_primary_epochs_times_multiplier",
            })
        payload["v2_new"] = {
            "persistent_subset_policy":
                "validated_output_dir_json_else_deterministic_random",
            "persistent_samples_per_task":
                args.v2_new_persistent_samples_per_task,
            "persistent_memory_integrity":
                "source_count_unique_range_sha256",
            "persistent_memory_resume_source":
                "output_dir/fixed_replay_memory",
            "configured_subset_seed": args.replay_subset_seed,
            "resolved_base_subset_seed": trainer._fixed_subset_seed(),
            "persisted_identities": persisted_identities,
            "selection_mode": args.replay_selection_mode,
            "active_memory_cap_unique": active,
            "active_memory_distribution": "equal_task",
            "active_memory_selection": "stable_nested_task_prefix",
            "active_stream_samples_per_pass": active,
            "active_stream_identity_order": "shared_between_kd_and_replay",
            "sampler_pass_order_contract":
                "shared_seed_plus_pass_rank_local_order",
            "sampler_pass_order_sha256": sampler_digests,
            "active_stream_seed_phase": "v2_new_shared_active_memory",
            "kd_stream_passes": "match_primary_epochs",
            "joint_replay_stream_passes": "match_primary_epochs",
        }
        if multiplier != 1:
            payload["v2_new"].update({
                "kd_stream_passes": "primary_epochs_times_multiplier",
                "kd_stream_pass_multiplier": multiplier,
            })
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, V3_META_NAME), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def num_v3_experts_from_state_dict(state_dict):
    pattern = re.compile(r"\.mlp\.experts\.(\d+)\.")
    maximum = -1
    for key in state_dict:
        match = pattern.search(key)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def load_v3_checkpoint(
    checkpoint_dir, tokenizer, base_model_name_or_path,
    device="cuda", dtype=torch.bfloat16, device_map=None,
):
    """Rebuild a V3 partial checkpoint over its frozen pretrained base."""
    from transformers import AutoModelForCausalLM
    from utils.model.model_utils import create_hf_model

    if base_model_name_or_path is None:
        raise ValueError(
            "V3 checkpoints contain experts/router only; "
            "base_model_name_or_path is required")
    with open(os.path.join(checkpoint_dir, V3_META_NAME),
              encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("architecture") != V3_ARCHITECTURE:
        raise ValueError(
            f"not a V3 checkpoint: {meta.get('architecture')}")
    if meta.get("attention_targets") != list(ATTENTION_TARGETS):
        raise ValueError(
            f"V3 requires QKVO targets, got {meta.get('attention_targets')}")
    if meta.get("attention_rank") != meta.get("r"):
        raise ValueError("V3 attention and FFN ranks must match")

    model = create_hf_model(
        AutoModelForCausalLM, base_model_name_or_path, tokenizer,
        disable_dropout=True, torch_dtype=dtype, low_cpu_mem_usage=True,
        forbid_vocab_growth=True, device_map=device_map)
    attach_shared_qkvo_lora_moe(
        model, r=meta["r"], alpha=meta["alpha"],
        top_k=meta["top_k"],
        aux_loss_coeff=meta["aux_loss_coeff"],
        z_loss_coeff=meta["z_loss_coeff"],
        routing_weight_mode=meta["routing_weight_mode"],
        dropout=meta.get("dropout", 0.0))
    add_v3_experts(model, meta["num_experts"])
    state = torch.load(
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        map_location="cpu", weights_only=False)
    inferred = num_v3_experts_from_state_dict(state)
    if inferred != meta["num_experts"]:
        raise ValueError(
            f"metadata has {meta['num_experts']} experts but state has "
            f"{inferred}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"unexpected V3 checkpoint keys: {unexpected[:5]}")
    grown_missing = [key for key in missing if any(
        substring in key
        for substring in Ours_LoRA_MoE_V3.save_key_substrings)]
    if grown_missing:
        raise RuntimeError(
            f"V3 checkpoint is missing grown keys: {grown_missing[:5]}")
    if device_map is None:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, meta
