# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from .attention import SelfAttention, SelfAttentionSubmodules
from .transformer_config import TransformerConfig


class QVLoraExpertRouter(torch.nn.Module): #lora expert routing class
    """Top-k routed LoRA experts for attention Q/V projections and optional output projection."""

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
        if config.attn_lora_topk <= 0:
            raise ValueError('attn_lora_topk must be > 0')
        if config.attn_lora_topk > config.attn_lora_num_experts:
            raise ValueError('attn_lora_topk must be <= attn_lora_num_experts')

        self.config = config
        self.input_size = input_size
        self.query_output_size = query_output_size
        self.value_output_size = value_output_size
        self.include_proj = config.attn_lora_include_proj
        self.num_experts = config.attn_lora_num_experts
        self.rank = config.attn_lora_rank
        self.topk = config.attn_lora_topk
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
        if self.include_proj:
            self.o_lora_a = Parameter(
                torch.empty(self.num_experts, input_size, self.rank, device=device, dtype=params_dtype)
            )
            self.o_lora_b = Parameter(
                torch.empty(self.num_experts, self.rank, input_size, device=device, dtype=params_dtype)
            )
        else:
            self.register_parameter('o_lora_a', None)
            self.register_parameter('o_lora_b', None)

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
            if self.include_proj:
                torch.nn.init.normal_(self.o_lora_a, mean=0.0, std=std)
                torch.nn.init.zeros_(self.o_lora_b)

    def _compute_grouped_qv_deltas(
        self,
        hidden_states: torch.Tensor,
        original_shape: Tuple[int, ...],
        expert_idx: torch.Tensor,
        expert_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.to(self.q_lora_a.dtype)
        scale = self.scale

        if expert_idx.dim() == 1:
            expert_idx = expert_idx.unsqueeze(1)
            expert_scores = expert_scores.unsqueeze(1)

        num_tokens = hidden_states.shape[0]
        q_delta = hidden_states.new_zeros((num_tokens, self.query_output_size))
        v_delta = hidden_states.new_zeros((num_tokens, self.value_output_size))

        for expert_id in range(self.num_experts):
            matched = torch.nonzero(expert_idx == expert_id, as_tuple=False)
            if matched.numel() == 0:
                continue

            token_indices = matched[:, 0]
            expert_hidden = hidden_states.index_select(0, token_indices)
            expert_scale = (
                expert_scores[matched[:, 0], matched[:, 1]].to(hidden_states.dtype).unsqueeze(-1) * scale
            )

            q_low_rank = expert_hidden @ self.q_lora_a[expert_id]
            v_low_rank = expert_hidden @ self.v_lora_a[expert_id]
            q_out = (q_low_rank @ self.q_lora_b[expert_id]) * expert_scale
            v_out = (v_low_rank @ self.v_lora_b[expert_id]) * expert_scale

            q_delta.index_add_(0, token_indices, q_out)
            v_delta.index_add_(0, token_indices, v_out)

        return (
            q_delta.reshape(*original_shape, self.query_output_size),
            v_delta.reshape(*original_shape, self.value_output_size),
        )

    def _compute_grouped_output_delta(
        self,
        proj_input_states: torch.Tensor,
        router_states: torch.Tensor,
        original_shape: Tuple[int, ...],
        expert_idx: torch.Tensor,
        expert_scores: torch.Tensor,
    ) -> torch.Tensor:
        if not self.include_proj:
            raise RuntimeError('output projection LoRA is disabled')

        proj_input_states = proj_input_states.to(self.o_lora_a.dtype)
        router_states = router_states.to(self.q_lora_a.dtype)
        scale = self.scale

        if expert_idx.dim() == 1:
            expert_idx = expert_idx.unsqueeze(1)
            expert_scores = expert_scores.unsqueeze(1)

        num_tokens = proj_input_states.shape[0]
        output_delta = proj_input_states.new_zeros((num_tokens, self.input_size))

        for expert_id in range(self.num_experts):
            matched = torch.nonzero(expert_idx == expert_id, as_tuple=False)
            if matched.numel() == 0:
                continue

            token_indices = matched[:, 0]
            expert_proj_in = proj_input_states.index_select(0, token_indices)
            expert_scale = (
                expert_scores[matched[:, 0], matched[:, 1]].to(proj_input_states.dtype).unsqueeze(-1) * scale
            )

            o_low_rank = expert_proj_in @ self.o_lora_a[expert_id]
            o_out = (o_low_rank @ self.o_lora_b[expert_id]) * expert_scale
            output_delta.index_add_(0, token_indices, o_out)

        return output_delta.reshape(*original_shape, self.input_size)

    def _route(self, hidden_states: torch.Tensor):
        router_input = hidden_states.to(self.router_weight.dtype)
        router_logits = F.linear(router_input, self.router_weight)
        router_probs = torch.softmax(router_logits, dim=-1)
        if self.topk == 1:
            expert_scores, expert_idx = torch.max(router_probs, dim=-1)
        else:
            expert_scores, expert_idx = torch.topk(router_probs, k=self.topk, dim=-1)
            expert_scores = expert_scores / (expert_scores.sum(dim=-1, keepdim=True) + 1e-20)
        return expert_idx, expert_scores

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape[:-1]
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        expert_idx, expert_scores = self._route(hidden_flat)
        return self._compute_grouped_qv_deltas(
            hidden_flat, original_shape, expert_idx, expert_scores
        )

    def forward_output(self, proj_input_states: torch.Tensor, router_states: torch.Tensor) -> torch.Tensor:
        if not self.include_proj:
            return torch.zeros_like(proj_input_states)

        original_shape = proj_input_states.shape[:-1]
        proj_input_flat = proj_input_states.reshape(-1, proj_input_states.shape[-1])
        router_flat = router_states.reshape(-1, router_states.shape[-1])
        expert_idx, expert_scores = self._route(router_flat)
        return self._compute_grouped_output_delta(
            proj_input_flat, router_flat, original_shape, expert_idx, expert_scores
        )


class QVLoraSelfAttention(SelfAttention):
    """Self-attention with routed LoRA experts on Q/V and optional output projection."""

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
        # Keep the first dense layer free of attention-LoRA experts so expertized
        # attention starts from the second transformer layer onward.
        enable_qv_lora = config.attn_lora_num_experts > 0 and self.layer_number > 1
        if enable_qv_lora:
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

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        output, bias = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            key_value_states=key_value_states,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        if self.qv_lora_experts is None or not self.config.attn_lora_include_proj:
            return output, bias

        output_delta = self.qv_lora_experts.forward_output(output, hidden_states)
        output = output + output_delta.to(output.dtype)
        return output, bias
