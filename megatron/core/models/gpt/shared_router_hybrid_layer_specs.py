# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import LNImpl, get_mlp_module_spec
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.shared_router_hybrid import (
    SharedRouterHybridLayerSubmodules,
    SharedRouterHybridTransformerLayer,
    SharedRouterMoELayer,
    SharedRouterQVLoraSelfAttention,
)
from megatron.core.transformer.spec_utils import ModuleSpec


def get_shared_router_moe_module_spec():
    base_spec = get_moe_module_spec(
        use_te=False,
        num_experts=1,
        moe_grouped_gemm=False,
        moe_use_legacy_grouped_gemm=False,
    )
    return ModuleSpec(
        module=SharedRouterMoELayer,
        submodules=base_spec.submodules,
    )


gpt_shared_router_hybrid_local_spec = ModuleSpec(
    module=SharedRouterHybridTransformerLayer,
    submodules=SharedRouterHybridLayerSubmodules(
        input_layernorm=LNImpl,
        self_attention=ModuleSpec(
            module=SharedRouterQVLoraSelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=LNImpl,
        dense_mlp=get_mlp_module_spec(use_te=False, num_experts=None),
        moe_mlp=get_shared_router_moe_module_spec(),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        },
    ),
)
