# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Callable

import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sequence_load_balancing_loss_func,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


_TRAINING_NEW_EXPERT_QUOTA = None


@contextmanager
def training_new_expert_quota(
    num_existing_experts: int, quota: float, min_new_slots: int | None = None
):
    """Force a minimum new-expert token share for an expert-only training pass.

    This context changes dispatch only.  The caller is responsible for restoring
    router gradients after the pass so the forced assignments cannot update the
    router.  Keeping this as a runtime context makes evaluation and ordinary
    training bitwise unchanged when the feature is disabled.
    """
    if num_existing_experts <= 0:
        raise ValueError("num_existing_experts must be positive")
    if not 0.0 < quota <= 1.0:
        raise ValueError(f"new-expert quota must be in (0, 1], got {quota}")

    global _TRAINING_NEW_EXPERT_QUOTA
    previous = _TRAINING_NEW_EXPERT_QUOTA
    state = {
        'boundary': int(num_existing_experts),
        'quota': float(quota),
        'min_new_slots': min_new_slots,
        'total_tokens': 0,
        'natural_new_group_tokens': 0,
        'dispatched_new_group_tokens': 0,
        'injected_tokens': 0,
    }
    _TRAINING_NEW_EXPERT_QUOTA = state
    try:
        yield state
    finally:
        _TRAINING_NEW_EXPERT_QUOTA = previous


def _apply_training_new_expert_quota(
    logits, scores, routing_map, boundary, quota, min_new_slots=None
):
    """Give a quota of tokens at least ``min_new_slots`` new-group routes.

    The group is every expert row at or after ``boundary``. No member receives
    a fixed preference: inserted routes always use the highest-logit available
    members of the whole new group. By default ``min_new_slots`` equals top-k,
    preserving the original fully-new-group quota behavior.
    """
    if boundary <= 0 or boundary >= logits.shape[-1]:
        raise ValueError(
            f"new-expert quota boundary must be in [1, {logits.shape[-1] - 1}], "
            f"got {boundary}"
        )
    if not 0.0 < quota <= 1.0:
        raise ValueError(f"new-expert quota must be in (0, 1], got {quota}")
    if logits.ndim != 2 or scores.shape != logits.shape or routing_map.shape != logits.shape:
        raise ValueError("quota routing expects matching [tokens, experts] tensors")

    assignments_per_token = routing_map.sum(dim=-1)
    if assignments_per_token.numel() == 0:
        return scores, routing_map, (0, 0, 0, 0)
    topk = int(assignments_per_token[0].item())
    if topk <= 0 or not torch.all(assignments_per_token == topk):
        raise ValueError("quota routing requires a constant positive top-k per token")
    if logits.shape[-1] - boundary < topk:
        raise ValueError(
            f"new-task expert group has {logits.shape[-1] - boundary} experts, "
            f"fewer than top-k={topk}"
        )

    if min_new_slots is None:
        min_new_slots = topk
    min_new_slots = int(min_new_slots)
    if min_new_slots <= 0 or min_new_slots > topk:
        raise ValueError(
            f"min_new_slots must be in [1, top-k={topk}], got {min_new_slots}"
        )

    token_count = logits.shape[0]
    target = min(token_count, int(torch.ceil(logits.new_tensor(quota * token_count)).item()))
    natural_new_slots = routing_map[:, boundary:].sum(dim=-1)
    natural_new_group = natural_new_slots >= min_new_slots
    inject_count = max(0, target - int(natural_new_group.sum().item()))
    if inject_count == 0:
        natural_count = int(natural_new_group.sum().item())
        stats = (token_count, natural_count, natural_count, 0)
        return scores, routing_map, stats

    candidates = torch.nonzero(~natural_new_group, as_tuple=False).flatten()
    inject_count = min(inject_count, int(candidates.numel()))
    if inject_count == 0:
        natural_count = int(natural_new_group.sum().item())
        stats = (token_count, natural_count, natural_count, 0)
        return scores, routing_map, stats

    candidate_logits = logits[candidates]
    candidate_maps = routing_map[candidates]
    missing_slots = min_new_slots - natural_new_slots[candidates]
    replacement_pairs = None
    if min_new_slots == 1:
        # Every candidate has zero selected new experts. Vectorize the common
        # one-slot bootstrap path because this runs for every token/layer.
        selected_old_logits = candidate_logits[:, :boundary].masked_fill(
            ~candidate_maps[:, :boundary], torch.inf
        )
        remove_old_indices = selected_old_logits.argmin(dim=-1)
        add_new_offsets = candidate_logits[:, boundary:].argmax(dim=-1)
        add_new_indices = add_new_offsets + boundary
        rows = torch.arange(candidates.numel(), device=logits.device)
        margins = (
            candidate_logits[rows, add_new_indices]
            - candidate_logits[rows, remove_old_indices]
        )
    else:
        margins = candidate_logits.new_zeros(candidates.numel())
        replacement_pairs = []
        for row in range(candidates.numel()):
            missing = int(missing_slots[row].item())
            selected_old = torch.nonzero(
                candidate_maps[row, :boundary], as_tuple=False
            ).flatten()
            available_new = torch.nonzero(
                ~candidate_maps[row, boundary:], as_tuple=False
            ).flatten() + boundary
            remove_old = selected_old[
                torch.topk(
                    candidate_logits[row, selected_old], k=missing, largest=False
                ).indices
            ]
            add_new = available_new[
                torch.topk(
                    candidate_logits[row, available_new], k=missing, largest=True
                ).indices
            ]
            margins[row] = (
                candidate_logits[row, add_new].sum()
                - candidate_logits[row, remove_old].sum()
            )
            replacement_pairs.append((remove_old, add_new))
    chosen_offsets = torch.topk(margins, k=inject_count, largest=True, sorted=False).indices
    chosen_tokens = candidates[chosen_offsets]

    quota_map = routing_map.clone()
    if min_new_slots == 1:
        quota_map[chosen_tokens, remove_old_indices[chosen_offsets]] = False
        quota_map[chosen_tokens, add_new_indices[chosen_offsets]] = True
    else:
        for chosen_offset, token in zip(chosen_offsets.tolist(), chosen_tokens.tolist()):
            remove_old, add_new = replacement_pairs[chosen_offset]
            quota_map[token, remove_old] = False
            quota_map[token, add_new] = True

    quota_scores = scores.clone()
    selected_logits = logits[chosen_tokens].masked_fill(~quota_map[chosen_tokens], -torch.inf)
    quota_scores[chosen_tokens] = torch.softmax(selected_logits, dim=-1).to(scores.dtype)
    stats = (
        token_count,
        int(natural_new_group.sum().item()),
        int((quota_map[:, boundary:].sum(dim=-1) >= min_new_slots).sum().item()),
        inject_count,
    )
    return quota_scores, quota_map, stats


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        router_input = input.to(router_dtype)
        intervention_mode = getattr(self, '_fingerprint_intervention_mode', None)
        intervention_weight = getattr(self, '_fingerprint_teacher_weight', None)
        if intervention_mode:
            mean = self._fingerprint_mean.to(dtype=router_dtype)
            basis = self._fingerprint_basis.to(dtype=router_dtype)
            centered = router_input - mean
            projection = torch.matmul(torch.matmul(centered, basis), basis.transpose(0, 1))
            if intervention_mode == 'fingerprint_only':
                router_input = mean + projection
            elif intervention_mode == 'fingerprint_removed':
                router_input = mean + centered - projection
            elif intervention_mode != 'teacher_full':
                raise RuntimeError(
                    f"Unsupported router fingerprint intervention mode: {intervention_mode}"
                )
            if intervention_weight is None:
                raise RuntimeError('Router fingerprint intervention has no teacher router weight.')
            weight = intervention_weight.to(dtype=router_dtype)
        else:
            weight = self.weight.to(router_dtype)
        logits = torch.nn.functional.linear(router_input, weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mask of token to experts assignment.
        """
        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            # Apply load balancing loss
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                switch_load_balancing_loss_func,
                probs=scores,
                tokens_per_expert=tokens_per_expert,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )
        return probs, routing_map

    def seq_aux_loss_load_balancing(self, logits: torch.Tensor, bsz: int, seq_length: int):
        """Apply loss-based load balancing to the logits tensor."""

        probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )

        if self.training:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            aux_loss_func = partial(
                sequence_load_balancing_loss_func,
                probs=scores,
                routing_map=routing_map,
                batch_size=bsz,
                seq_length=seq_length,
                topk=self.topk,
            )
            probs = self.apply_load_balancing_loss(
                activation=probs, load_balancing_loss_func=aux_loss_func
            )

        return probs, routing_map

    def apply_load_balancing_loss(
        self, activation: torch.Tensor, load_balancing_loss_func: Callable
    ):
        """Calculate auxiliary loss, attach gradient function to activation and add to logging."""
        moe_aux_loss_coeff = self.config.moe_aux_loss_coeff
        if moe_aux_loss_coeff == 0:
            return activation
        sequence_partition_group = None
        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            sequence_partition_group = parallel_state.get_context_parallel_group()
            moe_aux_loss_coeff /= parallel_state.get_tensor_model_parallel_world_size()
        elif parallel_state.get_tensor_and_context_parallel_world_size() > 1:
            sequence_partition_group = parallel_state.get_tensor_and_context_parallel_group()

        aux_loss = load_balancing_loss_func(
            moe_aux_loss_coeff=moe_aux_loss_coeff, sequence_partition_group=sequence_partition_group
        )
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
            reduce_group=sequence_partition_group,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training:
            moe_z_loss_coeff = (
                self.config.moe_z_loss_coeff
                / parallel_state.get_tensor_and_context_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, self.config.num_layers
            )
        return logits

    def apply_train_mask_existing_experts(self, logits: torch.Tensor):
        """Mask old experts only during training for code-only expert routing.

        This experiment forces Code batches to train and route through newly
        added experts, while evaluation/probe runs keep all experts available.
        """
        if not self.training or not self.config.shared_router_train_mask_existing_experts:
            return logits

        num_existing = self.config.shared_router_train_mask_existing_experts_from_num_experts
        if num_existing is None or num_existing <= 0:
            return logits
        if num_existing >= logits.shape[-1]:
            raise ValueError(
                "shared_router_train_mask_existing_experts_from_num_experts must be "
                f"smaller than num experts ({logits.shape[-1]}), got {num_existing}."
            )
        if logits.shape[-1] - num_existing < self.topk:
            raise ValueError(
                "Code-only expert routing leaves fewer unmasked experts than top-k: "
                f"num_experts={logits.shape[-1]}, masked={num_existing}, topk={self.topk}."
            )

        logits = logits.clone()
        logits[:, :num_existing] = torch.finfo(logits.dtype).min
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Train-only code expert masking: Code continual batches cannot route to
        # copied Wiki experts, but eval/probe/inference can still use all experts.
        logits = self.apply_train_mask_existing_experts(logits)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if self.config.moe_token_dispatcher_type == "alltoall_seq":
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.routing_type == "sinkhorn":
            scores, routing_map = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, routing_map = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "seq_aux_loss":
            scores, routing_map = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, routing_map, _ = topk_softmax_with_capacity(
                logits,
                self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
                drop_policy=self.config.moe_token_drop_policy,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                deterministic_mode=self.config.deterministic_mode,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
            )
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        quota_config = _TRAINING_NEW_EXPERT_QUOTA
        if self.training and quota_config is not None:
            if self.score_function != "softmax" or not self.config.moe_router_pre_softmax:
                raise RuntimeError(
                    "training new-expert quota currently requires softmax routing with "
                    "--moe-router-pre-softmax"
                )
            scores, routing_map, quota_stats = _apply_training_new_expert_quota(
                logits,
                scores,
                routing_map,
                quota_config['boundary'],
                quota_config['quota'],
                quota_config['min_new_slots'],
            )
            for key, value in zip(
                (
                    'total_tokens',
                    'natural_new_group_tokens',
                    'dispatched_new_group_tokens',
                    'injected_tokens',
                ),
                quota_stats,
            ):
                quota_config[key] += value
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return scores, routing_map

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        scores, routing_map = self.routing(logits)

        return scores, routing_map
