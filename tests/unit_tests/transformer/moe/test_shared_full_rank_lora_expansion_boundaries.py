# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import _get_param_groups
from megatron.core.transformer.moe.continual_learning_utils import (
    expand_moe_model,
    freeze_all_but_new_moe_params,
)
from megatron.core.transformer.shared_router_hybrid import SharedFullRankLoraExperts
from megatron.core.transformer.transformer_config import TransformerConfig


class SharedFullRankAttentionExpertModel(torch.nn.Module):
    """Minimal production-named shared QKVO expert container for CPU tests."""

    def __init__(self, num_experts: int, hidden_size: int = 4, rank: int = 2):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_router_topk=min(4, num_experts),
            attn_lora_num_experts=num_experts,
            attn_full_rank_lora_rank=rank,
            attn_full_rank_lora_alpha=rank,
            attn_full_rank_lora_targets="qkvo",
            attn_full_rank_lora_active_targets="",
            attn_lora_grouped_gemm=False,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        # Use the production attribute name. In particular, checkpoint filtering
        # sees ``shared_full_rank_lora_experts`` in every corresponding DCP key.
        self.shared_full_rank_lora_experts = SharedFullRankLoraExperts(
            config,
            input_size=hidden_size,
            query_output_size=hidden_size,
            value_output_size=hidden_size,
        )
        self.ddp_config = SimpleNamespace(use_custom_fsdp=False)


def _attention_params(model):
    return dict(model.shared_full_rank_lora_experts.named_parameters())


def _fill_source_rows_with_unique_sentinels(model):
    with torch.no_grad():
        for param_index, param in enumerate(_attention_params(model).values(), start=1):
            for expert_index in range(param.shape[0]):
                param[expert_index].fill_(1000.0 * param_index + expert_index)


@pytest.mark.parametrize(
    "num_existing_experts,num_target_experts",
    ((8, 16), (16, 24)),
)
def test_shared_full_rank_expansion_copies_old_rows_and_preserves_new_initialization(
    num_existing_experts,
    num_target_experts,
):
    torch.manual_seed(1234 + num_existing_experts)
    source = SharedFullRankAttentionExpertModel(num_existing_experts)
    target = SharedFullRankAttentionExpertModel(num_target_experts)
    _fill_source_rows_with_unique_sentinels(source)

    source_params = _attention_params(source)
    target_params = _attention_params(target)
    new_rows_before_expand = {
        name: param[num_existing_experts:].detach().clone()
        for name, param in target_params.items()
    }

    expand_moe_model(target, source, num_existing_experts=num_existing_experts)

    for name, target_param in target_params.items():
        assert torch.equal(
            target_param[:num_existing_experts],
            source_params[name],
        ), f"old attention expert rows were not copied exactly for {name}"
        assert torch.equal(
            target_param[num_existing_experts:],
            new_rows_before_expand[name],
        ), f"new attention expert initialization was overwritten for {name}"

    for name in ("qkv_lora_a", "proj_lora_a"):
        assert torch.count_nonzero(target_params[name][num_existing_experts:]) > 0
    for name in ("qkv_lora_b", "proj_lora_b"):
        assert torch.count_nonzero(target_params[name][num_existing_experts:]) == 0


@pytest.mark.parametrize(
    "num_existing_experts,num_target_experts",
    ((8, 16), (16, 24)),
)
def test_shared_full_rank_expansion_masks_all_old_attention_expert_grad_rows(
    num_existing_experts,
    num_target_experts,
):
    model = SharedFullRankAttentionExpertModel(num_target_experts)
    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=num_existing_experts,
        freeze_existing_experts=True,
        freeze_existing_router=True,
        train_dense_attention_lora=False,
    )

    model.zero_grad(set_to_none=True)
    sum(param.sum() for param in _attention_params(model).values()).backward()

    for name, param in _attention_params(model).items():
        assert param.requires_grad
        assert param.grad is not None
        assert torch.count_nonzero(param.grad[:num_existing_experts]) == 0, name
        assert torch.all(param.grad[num_existing_experts:] == 1), name


@pytest.mark.parametrize(
    "num_existing_experts,num_target_experts",
    ((8, 16), (16, 24)),
)
def test_partial_row_attention_params_disable_weight_decay_and_keep_old_rows_immutable(
    num_existing_experts,
    num_target_experts,
):
    """Partial-row freezing must also suppress parameter-wide AdamW decay."""

    model = SharedFullRankAttentionExpertModel(num_target_experts)
    with torch.no_grad():
        for param in _attention_params(model).values():
            # Keep both A and B old rows nonzero so decoupled weight decay is observable.
            param.fill_(1.0)

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=num_existing_experts,
        freeze_existing_experts=True,
        freeze_existing_router=True,
        train_dense_attention_lora=False,
    )
    old_rows_before_step = {
        name: param[:num_existing_experts].detach().clone()
        for name, param in _attention_params(model).items()
    }
    new_rows_before_step = {
        name: param[num_existing_experts:].detach().clone()
        for name, param in _attention_params(model).items()
    }

    param_groups = _get_param_groups(
        [model],
        no_weight_decay_cond=None,
        scale_lr_cond=None,
        lr_mult=1.0,
        lr=0.1,
        min_lr=0.0,
        decoupled_lr=None,
        decoupled_min_lr=None,
    )
    group_by_param_id = {
        id(param): group
        for group in param_groups
        for param in group["params"]
    }
    for name, param in _attention_params(model).items():
        assert getattr(param, "_exclude_from_weight_decay_for_frozen_rows", False), name
        assert group_by_param_id[id(param)]["wd_mult"] == 0.0, name

    # Mirror Megatron's effective per-group weight decay: base WD multiplied by
    # the grouping policy's wd_mult.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": group["params"],
                "weight_decay": 0.01 * group["wd_mult"],
            }
            for group in param_groups
        ],
        lr=0.1,
    )

    optimizer.zero_grad(set_to_none=True)
    sum(param.sum() for param in _attention_params(model).values()).backward()
    for name, param in _attention_params(model).items():
        assert torch.count_nonzero(param.grad[:num_existing_experts]) == 0, name
    optimizer.step()

    for name, param in _attention_params(model).items():
        assert torch.equal(
            param[:num_existing_experts],
            old_rows_before_step[name],
        ), f"AdamW changed frozen old attention expert rows for {name}"
        assert not torch.equal(
            param[num_existing_experts:],
            new_rows_before_step[name],
        ), f"AdamW did not update trainable new attention expert rows for {name}"
