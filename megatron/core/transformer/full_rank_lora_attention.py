# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy
from typing import Tuple

import torch
from torch.nn import Parameter

from .attention import AttnMaskType, SelfAttention, SelfAttentionSubmodules, apply_rotary_pos_emb
from .transformer_config import TransformerConfig


class FullRankLoraAdapter(torch.nn.Module):
    """LoRA adapter that can reach full-rank updates when rank=min(in,out)."""

    def __init__(
        self,
        config: TransformerConfig,
        input_size: int,
        output_size: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError('attn_full_rank_lora_rank must be > 0')

        self.input_size = input_size
        self.output_size = output_size
        self.rank = rank
        self.scale = float(alpha) / float(max(rank, 1))

        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        params_dtype = config.params_dtype

        self.lora_a = Parameter(
            torch.empty(input_size, rank, device=device, dtype=params_dtype)
        )
        self.lora_b = Parameter(
            torch.empty(rank, output_size, device=device, dtype=params_dtype)
        )

        if config.perform_initialization:
            self.reset_parameters(config.init_method_std)

    def reset_parameters(self, std: float):
        with torch.no_grad():
            torch.nn.init.normal_(self.lora_a, mean=0.0, std=std)
            torch.nn.init.zeros_(self.lora_b)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape: Tuple[int, ...] = hidden_states.shape[:-1]
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(self.lora_a.dtype)
        low_rank = hidden_flat @ self.lora_a
        delta = (low_rank @ self.lora_b) * self.scale
        return delta.reshape(*original_shape, self.output_size)


class FullRankLoraSelfAttention(SelfAttention):
    """Self-attention with full-rank LoRA on packed QKV and output projection."""

    @staticmethod
    def _parse_targets(targets: str) -> set[str]:
        normalized = (targets or "").lower()
        invalid = sorted(set(normalized) - set("qkvo"))
        if invalid:
            raise ValueError(
                f"Unsupported attn_full_rank_lora_targets={targets!r}; invalid entries: {''.join(invalid)}"
            )
        if not normalized:
            raise ValueError("attn_full_rank_lora_targets must contain at least one of q, k, v, o")
        return set(normalized)

    @classmethod
    def _parse_active_targets(cls, active_targets: str, instantiated_targets: set[str]) -> set[str]:
        if not active_targets:
            return set(instantiated_targets)
        normalized_active = cls._parse_targets(active_targets)
        if not normalized_active.issubset(instantiated_targets):
            missing = ''.join(sorted(normalized_active - instantiated_targets))
            raise ValueError(
                f"attn_full_rank_lora_active_targets={active_targets!r} is not a subset of "
                f"attn_full_rank_lora_targets={''.join(sorted(instantiated_targets))!r}; "
                f"invalid active entries: {missing}"
            )
        return normalized_active

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

        enable_full_rank_lora = config.attn_full_rank_lora_rank > 0 and self.layer_number > 1
        if enable_full_rank_lora:
            targets = self._parse_targets(config.attn_full_rank_lora_targets)
            active_targets = self._parse_active_targets(
                config.attn_full_rank_lora_active_targets, targets
            )
            use_packed_qkv = {"q", "k", "v"}.issubset(targets)
            self.active_targets = active_targets
            self.qkv_full_rank_lora = None
            self.q_full_rank_lora = None
            self.k_full_rank_lora = None
            self.v_full_rank_lora = None

            if use_packed_qkv:
                rank = min(
                    config.attn_full_rank_lora_rank,
                    self.config.hidden_size,
                    self.query_projection_size + 2 * self.kv_projection_size,
                )
                self.qkv_full_rank_lora = FullRankLoraAdapter(
                    config=config,
                    input_size=self.config.hidden_size,
                    output_size=self.query_projection_size + 2 * self.kv_projection_size,
                    rank=rank,
                    alpha=config.attn_full_rank_lora_alpha,
                )
            else:
                if "q" in targets:
                    self.q_full_rank_lora = FullRankLoraAdapter(
                        config=config,
                        input_size=self.config.hidden_size,
                        output_size=self.query_projection_size,
                        rank=min(config.attn_full_rank_lora_rank, self.config.hidden_size, self.query_projection_size),
                        alpha=config.attn_full_rank_lora_alpha,
                    )
                if "k" in targets:
                    self.k_full_rank_lora = FullRankLoraAdapter(
                        config=config,
                        input_size=self.config.hidden_size,
                        output_size=self.kv_projection_size,
                        rank=min(config.attn_full_rank_lora_rank, self.config.hidden_size, self.kv_projection_size),
                        alpha=config.attn_full_rank_lora_alpha,
                    )
                if "v" in targets:
                    self.v_full_rank_lora = FullRankLoraAdapter(
                        config=config,
                        input_size=self.config.hidden_size,
                        output_size=self.kv_projection_size,
                        rank=min(config.attn_full_rank_lora_rank, self.config.hidden_size, self.kv_projection_size),
                        alpha=config.attn_full_rank_lora_alpha,
                    )

            if "o" in targets:
                self.proj_full_rank_lora = FullRankLoraAdapter(
                    config=config,
                    input_size=self.query_projection_size,
                    output_size=self.config.hidden_size,
                    rank=min(config.attn_full_rank_lora_rank, self.query_projection_size),
                    alpha=config.attn_full_rank_lora_alpha,
                )
            else:
                self.proj_full_rank_lora = None

            for param in self.linear_qkv.parameters():
                param.requires_grad = False
            for param in self.linear_proj.parameters():
                param.requires_grad = False

            def remove_full_rank_lora_missing_keys(self, incompatible_keys):
                keys = deepcopy(incompatible_keys.missing_keys)
                for key in keys:
                    if 'full_rank_lora' in key:
                        incompatible_keys.missing_keys.remove(key)

            self.register_load_state_dict_post_hook(remove_full_rank_lora_missing_keys)
        else:
            self.active_targets = set()
            self.qkv_full_rank_lora = None
            self.q_full_rank_lora = None
            self.k_full_rank_lora = None
            self.v_full_rank_lora = None
            self.proj_full_rank_lora = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        lora_qkv_delta = None
        if self.qkv_full_rank_lora is not None:
            lora_qkv_delta = self.qkv_full_rank_lora(hidden_states).to(mixed_qkv.dtype)

        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        if lora_qkv_delta is not None:
            lora_qkv_delta = lora_qkv_delta.view(*new_tensor_shape)
            q_delta, k_delta, v_delta = torch.split(lora_qkv_delta, split_arg_list, dim=3)
            q_delta = q_delta.reshape(
                query.size(0),
                query.size(1),
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            k_delta = k_delta.reshape(
                key.size(0),
                key.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            v_delta = v_delta.reshape(
                value.size(0),
                value.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
            if "q" in self.active_targets:
                query = query + q_delta
            if "k" in self.active_targets:
                key = key + k_delta
            if "v" in self.active_targets:
                value = value + v_delta
        if self.q_full_rank_lora is not None and "q" in self.active_targets:
            q_delta = self.q_full_rank_lora(hidden_states).to(query.dtype)
            query = query + q_delta.view(
                query.size(0),
                query.size(1),
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
        if self.k_full_rank_lora is not None and "k" in self.active_targets:
            k_delta = self.k_full_rank_lora(hidden_states).to(key.dtype)
            key = key + k_delta.view(
                key.size(0),
                key.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )
        if self.v_full_rank_lora is not None and "v" in self.active_targets:
            v_delta = self.v_full_rank_lora(hidden_states).to(value.dtype)
            value = value + v_delta.view(
                value.size(0),
                value.size(1),
                self.num_query_groups_per_partition,
                self.hidden_size_per_attention_head,
            )

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

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
        if rotary_pos_cos is not None and rotary_pos_sin is not None:
            assert self.config.flash_decode
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
            )
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)
        if self.proj_full_rank_lora is not None and "o" in self.active_targets:
            output = output + self.proj_full_rank_lora(core_attn_out).to(output.dtype)

        return output, bias
