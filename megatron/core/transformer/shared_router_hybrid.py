# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import torch
from torch.nn import Parameter

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.moe_layer import BaseMoELayer, MoESubmodules
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    BaseTransformerLayer,
    get_transformer_layer_offset,
)
from megatron.core.utils import make_viewless_tensor


@dataclass
class SharedRoutingContext:
    scores: torch.Tensor
    routing_map: torch.Tensor


@dataclass
class SharedRouterHybridLayerSubmodules:
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    dense_mlp: Union[ModuleSpec, type] = IdentityOp
    moe_mlp: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


def _is_moe_layer(config: TransformerConfig, layer_number: int) -> bool:
    layer_index = layer_number - 1
    if isinstance(config.moe_layer_freq, int):
        return layer_index % config.moe_layer_freq == 0
    if isinstance(config.moe_layer_freq, list):
        return bool(config.moe_layer_freq[layer_index])
    raise ValueError(f"Unsupported moe_layer_freq: {config.moe_layer_freq}")


class SharedQVLoraExperts(MegatronModule):
    """Attention Q/V LoRA experts that consume routing from an external shared router."""

    def __init__(
        self,
        config: TransformerConfig,
        input_size: int,
        query_output_size: int,
        value_output_size: int,
    ):
        super().__init__(config=config)
        if config.attn_lora_num_experts <= 0:
            raise ValueError('attn_lora_num_experts must be > 0')
        if config.attn_lora_rank <= 0:
            raise ValueError('attn_lora_rank must be > 0')

        self.config = config
        self.input_size = input_size
        self.query_output_size = query_output_size
        self.value_output_size = value_output_size
        self.num_experts = config.attn_lora_num_experts
        self.rank = config.attn_lora_rank
        self.scale = float(config.attn_lora_alpha) / float(max(self.rank, 1))

        device = None if config.use_cpu_initialization else torch.cuda.current_device()
        params_dtype = config.params_dtype

        self.q_lora_a = Parameter(
            torch.empty(self.num_experts, input_size, self.rank, device=device, dtype=params_dtype)
        )
        self.q_lora_b = Parameter(
            torch.empty(
                self.num_experts,
                self.rank,
                query_output_size,
                device=device,
                dtype=params_dtype,
            )
        )
        self.v_lora_a = Parameter(
            torch.empty(self.num_experts, input_size, self.rank, device=device, dtype=params_dtype)
        )
        self.v_lora_b = Parameter(
            torch.empty(
                self.num_experts,
                self.rank,
                value_output_size,
                device=device,
                dtype=params_dtype,
            )
        )

        if config.perform_initialization:
            self.reset_parameters()

    def reset_parameters(self):
        std = self.config.init_method_std
        with torch.no_grad():
            torch.nn.init.normal_(self.q_lora_a, mean=0.0, std=std)
            torch.nn.init.zeros_(self.q_lora_b)
            torch.nn.init.normal_(self.v_lora_a, mean=0.0, std=std)
            torch.nn.init.zeros_(self.v_lora_b)

    def forward(
        self, hidden_states: torch.Tensor, routing_context: SharedRoutingContext
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape[:-1]
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(self.q_lora_a.dtype)
        scores = routing_context.scores
        routing_map = routing_context.routing_map
        if scores.shape[0] != hidden_flat.shape[0]:
            raise ValueError(
                f"Shared routing token count {scores.shape[0]} does not match hidden states "
                f"{hidden_flat.shape[0]}"
            )

        q_delta = hidden_flat.new_zeros((hidden_flat.shape[0], self.query_output_size))
        v_delta = hidden_flat.new_zeros((hidden_flat.shape[0], self.value_output_size))

        for expert_id in range(self.num_experts):
            token_indices = torch.nonzero(routing_map[:, expert_id], as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue

            expert_hidden = hidden_flat.index_select(0, token_indices)
            expert_scale = (
                scores.index_select(0, token_indices)[:, expert_id]
                .to(hidden_flat.dtype)
                .unsqueeze(-1)
                * self.scale
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


class SharedRouterQVLoraSelfAttention(SelfAttention):
    """Self-attention whose LoRA experts consume routing from a shared router."""

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
            self.shared_qv_lora_experts = SharedQVLoraExperts(
                config=config,
                input_size=self.config.hidden_size,
                query_output_size=query_output_size,
                value_output_size=value_output_size,
            )
        else:
            self.shared_qv_lora_experts = None
        self._active_routing_context: Optional[SharedRoutingContext] = None

    def forward(self, *args, routing_context: Optional[SharedRoutingContext] = None, **kwargs):
        self._active_routing_context = routing_context
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._active_routing_context = None

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states)
        if self.shared_qv_lora_experts is None or self._active_routing_context is None:
            return query, key, value

        q_delta, v_delta = self.shared_qv_lora_experts(hidden_states, self._active_routing_context)
        query = query + q_delta.to(query.dtype).reshape_as(query)
        value = value + v_delta.to(value.dtype).reshape_as(value)
        return query, key, value


class SharedRouterMoELayer(BaseMoELayer):
    """MoE layer that reuses routing decisions supplied by the hybrid layer."""

    def __init__(
        self, config: TransformerConfig, submodules: MoESubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super().__init__(config=config, layer_number=layer_number)
        if config.moe_layer_recompute:
            raise ValueError(
                "Shared-router hybrid MoE does not support --moe-layer-recompute because "
                "routing is computed once before attention and reused for the FFN branch."
            )

        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number

    def forward(
        self, hidden_states: torch.Tensor, routing_context: Optional[SharedRoutingContext] = None
    ):
        if routing_context is None:
            raise ValueError("SharedRouterMoELayer requires routing_context from the hybrid layer.")
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism "
                "are enabled without also enabling sequence parallelism."
            )

        probs = routing_context.scores
        routing_map = routing_context.routing_map
        dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if self.use_shared_expert and not self.shared_expert_overlap:
            output = output + self.shared_experts(hidden_states)
        return output, mlp_bias


class SharedRouterHybridTransformerLayer(MegatronModule, BaseTransformerLayer):
    """Transformer layer with a single shared router for attention-LoRA and FFN experts."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SharedRouterHybridLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
    ):
        super().__init__(config=config)
        if config.num_moe_experts is None:
            raise ValueError("Shared-router hybrid model requires --num-experts.")
        if config.attn_lora_num_experts <= 0:
            raise ValueError("Shared-router hybrid model requires --attn-lora-num-experts > 0.")
        if config.num_moe_experts != config.attn_lora_num_experts:
            raise ValueError(
                "Shared-router hybrid model requires --num-experts == --attn-lora-num-experts."
            )
        if config.moe_router_topk != config.attn_lora_topk:
            raise ValueError(
                "Shared-router hybrid model requires --moe-router-topk == --attn-lora-topk."
            )

        self.submodules_config = submodules
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_moe_layer = _is_moe_layer(self.config, self.layer_number)

        self.input_layernorm = build_module(
            submodules.input_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        self.shared_expert_router = TopKRouter(config=self.config)
        self.shared_expert_router.set_layer_number(self.layer_number)

        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        self.pre_cross_attn_layernorm = build_module(
            submodules.pre_cross_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        mlp_spec = submodules.moe_mlp if self.is_moe_layer else submodules.dense_mlp
        self.mlp = build_module(mlp_spec, config=self.config)
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)
        self.mlp_bda = build_module(submodules.mlp_bda)
        self.bias_dropout_add_exec_handler = torch.enable_grad

    def _compute_shared_routing(self, hidden_states: torch.Tensor) -> SharedRoutingContext:
        scores, routing_map = self.shared_expert_router(hidden_states)
        return SharedRoutingContext(scores=scores, routing_map=routing_map)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ):
        residual = hidden_states
        input_layernorm_output = self.input_layernorm(hidden_states)
        routing_context = self._compute_shared_routing(input_layernorm_output)

        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            routing_context=routing_context,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        residual = hidden_states
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        residual = hidden_states
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
        if self.is_moe_layer:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, routing_context=routing_context)
        else:
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        if self.config.external_cuda_graph and self.training:
            return output
        return output, context

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def __call__(self, *args, **kwargs):
        return super(MegatronModule, self).__call__(*args, **kwargs)
