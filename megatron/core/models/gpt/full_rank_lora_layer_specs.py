# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import LNImpl, get_mlp_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.transformer.full_rank_lora_attention import FullRankLoraSelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig


def get_gpt_full_rank_lora_layer_local_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    qk_layernorm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    mlp = get_mlp_module_spec(
        use_te=False,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LNImpl,
            self_attention=ModuleSpec(
                module=FullRankLoraSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=LNImpl if qk_layernorm else IdentityOp,
                    k_layernorm=LNImpl if qk_layernorm else IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LNImpl,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
                'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            },
        ),
    )


def get_gpt_full_rank_lora_decoder_block_spec(
    config: TransformerConfig,
) -> TransformerBlockSubmodules:
    dense_layer_spec = get_gpt_full_rank_lora_layer_local_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=config.qk_layernorm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
    )
    moe_layer_spec = get_gpt_full_rank_lora_layer_local_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=config.qk_layernorm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
    )

    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [
            1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.num_layers)
        ]
    elif isinstance(config.moe_layer_freq, list):
        moe_layer_pattern = config.moe_layer_freq
    else:
        raise ValueError(
            f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
        )

    layer_specs = []
    for layer_number in range(config.num_layers):
        layer_specs.append(moe_layer_spec if moe_layer_pattern[layer_number] == 1 else dense_layer_spec)

    offset = get_transformer_layer_offset(config)
    num_layers_to_build = get_num_layers_to_build(config)
    layer_specs = layer_specs[offset : offset + num_layers_to_build]
    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=LNImpl)
