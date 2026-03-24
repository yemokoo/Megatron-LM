# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from .attention import SelfAttention, SelfAttentionSubmodules
from .transformer_config import TransformerConfig


class QVLoraExpertRouter(torch.nn.Module): #lora expert routing class
    """Top-1 routed LoRA experts for attention Q/V projections."""

    def __init__(
        self,
        config: TransformerConfig,
        input_size: int,
        query_output_size: int,
        value_output_size: int,
    ):
        super().__init__()
        if config.attn_lora_num_experts <= 0:
            raise ValueError('attn_lora_num_experts must be > 0')
        if config.attn_lora_rank <= 0:
            raise ValueError('attn_lora_rank must be > 0')
        if config.attn_lora_topk != 1:
            raise NotImplementedError('Only top-1 attention LoRA routing is currently supported.')

        self.config = config
        self.input_size = input_size
        self.query_output_size = query_output_size
        self.value_output_size = value_output_size
        self.num_experts = config.attn_lora_num_experts
        self.rank = config.attn_lora_rank
        self.scale = float(config.attn_lora_alpha) / float(max(self.rank, 1))

        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        params_dtype = config.params_dtype

        self.router_weight = Parameter(
            torch.empty(self.num_experts, input_size, device=device, dtype=params_dtype)
        )
        self.q_lora_a = Parameter(
            torch.empty(self.num_experts, input_size, self.rank, device=device, dtype=params_dtype)
        )
        self.q_lora_b = Parameter(
            torch.empty(self.num_experts, self.rank, query_output_size, device=device, dtype=params_dtype)
        )
        self.v_lora_a = Parameter(
            torch.empty(self.num_experts, input_size, self.rank, device=device, dtype=params_dtype)
        )
        self.v_lora_b = Parameter(
            torch.empty(self.num_experts, self.rank, value_output_size, device=device, dtype=params_dtype)
        )

        if config.perform_initialization:
            self.reset_parameters()

    def reset_parameters(self):
        std = self.config.init_method_std
        with torch.no_grad():
            torch.nn.init.normal_(self.router_weight, mean=0.0, std=std)
            torch.nn.init.normal_(self.q_lora_a, mean=0.0, std=std)
            torch.nn.init.zeros_(self.q_lora_b)
            torch.nn.init.normal_(self.v_lora_a, mean=0.0, std=std)
            torch.nn.init.zeros_(self.v_lora_b)

    def _compute_grouped_qv_deltas(
        self,
        hidden_states: torch.Tensor,
        original_shape: Tuple[int, ...],
        expert_idx: torch.Tensor,
        expert_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.q_lora_a.dtype)
        num_tokens = hidden_states.shape[0]
        q_delta = hidden_states.new_zeros((num_tokens, self.query_output_size))
        v_delta = hidden_states.new_zeros((num_tokens, self.value_output_size))
        scale = hidden_states.new_tensor(self.scale)

        for expert_id in range(self.num_experts):
            token_indices = torch.nonzero(expert_idx == expert_id, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue

            expert_hidden = hidden_states.index_select(0, token_indices)
            expert_scale = (
                expert_scores.index_select(0, token_indices).to(hidden_states.dtype).unsqueeze(-1) * scale
            )

            q_low_rank = expert_hidden @ self.q_lora_a[expert_id]
            v_low_rank = expert_hidden @ self.v_lora_a[expert_id]

            q_delta.index_copy_(0, token_indices, (q_low_rank @ self.q_lora_b[expert_id]) * expert_scale)
            v_delta.index_copy_(0, token_indices, (v_low_rank @ self.v_lora_b[expert_id]) * expert_scale)
        return (
            q_delta.reshape(*original_shape, self.query_output_size),
            v_delta.reshape(*original_shape, self.value_output_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape[:-1]
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_input = hidden_flat.to(self.router_weight.dtype)
        router_logits = F.linear(router_input, self.router_weight)
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_scores, expert_idx = torch.max(router_probs, dim=-1)
        return self._compute_grouped_qv_deltas(
            hidden_flat, original_shape, expert_idx, expert_scores
        )


class QVLoraSelfAttention(SelfAttention):
    """Self-attention with routed LoRA experts on Q and V projections."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )
        if config.attn_lora_num_experts > 0:
            query_output_size = (
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
            )
            value_output_size = (
                self.num_query_groups_per_partition * self.hidden_size_per_attention_head
            )
            self.qv_lora_experts = QVLoraExpertRouter(
                config=config,
                input_size=self.config.hidden_size,
                query_output_size=query_output_size,
                value_output_size=value_output_size,
            )
        else:
            self.qv_lora_experts = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states)
        if self.qv_lora_experts is None:
            return query, key, value

        q_delta, v_delta = self.qv_lora_experts(hidden_states)
        query = query + q_delta.to(query.dtype).reshape_as(query)
        value = value + v_delta.to(value.dtype).reshape_as(value)
        return query, key, value
