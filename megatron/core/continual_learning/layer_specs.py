"""Layer specifications for the matched dense, SLoRA, and O-LoRA models."""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import LNImpl
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules, apply_rotary_pos_emb
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from .lora_adapter import ContinualLowRankAdapter


def _remove_continual_missing_keys(_module, incompatible_keys) -> None:
    for key in list(incompatible_keys.missing_keys):
        if "continual_" in key:
            incompatible_keys.missing_keys.remove(key)


class ContinualSelfAttention(SelfAttention):
    """Self attention with either one SLoRA adapter or three O-LoRA adapters."""

    def __init__(self, config, submodules, layer_number, attn_mask_type, cp_comm_type=None):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )
        method = getattr(config, "continual_method", "none")
        in_scope = (
            getattr(config, "continual_layer_start", 2)
            <= self.layer_number
            <= getattr(config, "continual_layer_end", 9)
        )
        self.continual_method = method if in_scope else "none"
        self.continual_q_adapters = torch.nn.ModuleList()
        self.continual_k_adapters = torch.nn.ModuleList()
        self.continual_v_adapters = torch.nn.ModuleList()
        self.continual_o_adapters = torch.nn.ModuleList()

        if self.continual_method == "slora_pre":
            max_rank = int(getattr(config, "continual_slora_max_rank", 256))
            alpha = float(getattr(config, "continual_slora_alpha", 128.0))
            self.continual_q_adapters.append(
                ContinualLowRankAdapter(config, config.hidden_size, self.query_projection_size, max_rank, alpha)
            )
            self.continual_k_adapters.append(
                ContinualLowRankAdapter(config, config.hidden_size, self.kv_projection_size, max_rank, alpha)
            )
            self.continual_v_adapters.append(
                ContinualLowRankAdapter(config, config.hidden_size, self.kv_projection_size, max_rank, alpha)
            )
            self.continual_o_adapters.append(
                ContinualLowRankAdapter(config, self.query_projection_size, config.hidden_size, max_rank, alpha)
            )
        elif self.continual_method == "olora":
            rank = int(getattr(config, "continual_olora_rank", 352))
            alpha = float(getattr(config, "continual_olora_alpha", 352.0))
            dropout = float(getattr(config, "continual_olora_dropout", 0.1))
            for _ in range(3):
                self.continual_q_adapters.append(
                    ContinualLowRankAdapter(
                        config, config.hidden_size, self.query_projection_size, rank, alpha, dropout
                    )
                )
                self.continual_v_adapters.append(
                    ContinualLowRankAdapter(
                        config, config.hidden_size, self.kv_projection_size, rank, alpha, dropout
                    )
                )
        if self.continual_method != "none":
            self.register_load_state_dict_post_hook(_remove_continual_missing_keys)

    def _sum_adapter_outputs(self, adapters, hidden_states):
        active = [adapter(hidden_states) for adapter in adapters if adapter.active_rank > 0]
        if not active:
            return None
        return torch.stack(active).sum(dim=0)

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        query, key, value = super().get_query_key_value_tensors(hidden_states, key_value_states)
        q_delta = self._sum_adapter_outputs(self.continual_q_adapters, hidden_states)
        k_delta = self._sum_adapter_outputs(self.continual_k_adapters, hidden_states)
        v_delta = self._sum_adapter_outputs(self.continual_v_adapters, hidden_states)
        if q_delta is not None:
            query = query + q_delta.to(query.dtype).view_as(query)
        if k_delta is not None:
            key = key + k_delta.to(key.dtype).view_as(key)
        if v_delta is not None:
            value = value + v_delta.to(value.dtype).view_as(value)
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
            query, key, value = query.squeeze(1), key.squeeze(1), value.squeeze(1)
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if packed_seq_params is not None:
                cu_q = (
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params.cu_seqlens_q_padded is not None
                    else packed_seq_params.cu_seqlens_q
                )
                cu_kv = (
                    packed_seq_params.cu_seqlens_kv_padded
                    if packed_seq_params.cu_seqlens_kv_padded is not None
                    else packed_seq_params.cu_seqlens_kv
                )
            else:
                cu_q = cu_kv = None
            query = apply_rotary_pos_emb(query, q_pos_emb, config=self.config, cu_seqlens=cu_q)
            key = apply_rotary_pos_emb(key, k_pos_emb, config=self.config, cu_seqlens=cu_kv)
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
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        output, bias = self.linear_proj(core_attn_out)
        o_delta = self._sum_adapter_outputs(self.continual_o_adapters, core_attn_out)
        if o_delta is not None:
            output = output + o_delta.to(output.dtype)
        return output, bias


class ScopedDenseMLP(MegatronModule):
    """SwiGLU MLP with a per-layer width and optional SLoRA gate/up/down."""

    def __init__(self, config, submodules: MLPSubmodules, continual_layer_number: int):
        super().__init__(config=config)
        self.layer_number = int(continual_layer_number)
        width = (
            int(getattr(config, "continual_dense_ffn_hidden_size", 1408))
            if self.layer_number >= int(getattr(config, "continual_layer_start", 2))
            else int(config.ffn_hidden_size)
        )
        self.ffn_hidden_size = width
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            config.hidden_size,
            width * 2,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc1",
        )
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            width,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="fc2",
        )
        self.continual_gate_adapter = None
        self.continual_up_adapter = None
        self.continual_down_adapter = None
        in_scope = (
            getattr(config, "continual_layer_start", 2)
            <= self.layer_number
            <= getattr(config, "continual_layer_end", 9)
        )
        if getattr(config, "continual_method", "none") == "slora_pre" and in_scope:
            max_rank = int(getattr(config, "continual_slora_max_rank", 256))
            alpha = float(getattr(config, "continual_slora_alpha", 128.0))
            self.continual_gate_adapter = ContinualLowRankAdapter(
                config, config.hidden_size, width, max_rank, alpha
            )
            self.continual_up_adapter = ContinualLowRankAdapter(
                config, config.hidden_size, width, max_rank, alpha
            )
            self.continual_down_adapter = ContinualLowRankAdapter(
                config, width, config.hidden_size, max_rank, alpha
            )
            self.register_load_state_dict_post_hook(_remove_continual_missing_keys)

    def forward(self, hidden_states):
        intermediate, bias = self.linear_fc1(hidden_states)
        if self.continual_gate_adapter is not None:
            gate = self.continual_gate_adapter(hidden_states)
            up = self.continual_up_adapter(hidden_states)
            intermediate = intermediate + torch.cat((gate, up), dim=-1).to(intermediate.dtype)
        if bias is not None:
            intermediate = intermediate + bias
        gate, up = torch.chunk(intermediate, 2, dim=-1)
        activated = F.silu(gate) * up
        output, output_bias = self.linear_fc2(activated)
        if self.continual_down_adapter is not None:
            output = output + self.continual_down_adapter(activated).to(output.dtype)
        return output, output_bias


def get_continual_dense_decoder_block_spec(config) -> TransformerBlockSubmodules:
    """Build all nine local layers with exact Layer-1/Layer-2--9 widths."""
    if config.pipeline_model_parallel_size != 1:
        raise ValueError("The six-baseline continual spec currently requires pipeline parallel size 1")
    use_adapted_attention = getattr(config, "continual_method", "none") in {"slora_pre", "olora"}
    attention_module = ContinualSelfAttention if use_adapted_attention else SelfAttention
    layer_specs = []
    for layer_number in range(1, config.num_layers + 1):
        mlp = ModuleSpec(
            module=ScopedDenseMLP,
            params={"continual_layer_number": layer_number},
            submodules=MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
        layer_specs.append(
            ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=LNImpl,
                    self_attention=ModuleSpec(
                        module=attention_module,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=DotProductAttention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=LNImpl if config.qk_layernorm else IdentityOp,
                            k_layernorm=LNImpl if config.qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=LNImpl,
                    mlp=mlp,
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                        "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
                    },
                ),
            )
        )
    return TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=LNImpl)
