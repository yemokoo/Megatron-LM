# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.moe.continual_learning_utils import (
    allow_existing_router_grads,
    freeze_all_but_new_moe_params,
    teacher_student_router_kl,
)
from megatron.core.transformer.moe.experts import GroupedMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.shared_router_hybrid import SharedFullRankLoraExperts
from megatron.core.transformer.transformer_config import TransformerConfig


class RouterOnlyModel(torch.nn.Module):
    def __init__(self, num_experts=4, hidden_size=3):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        self.router = TopKRouter(config)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


class MinimalGroupedMLP(GroupedMLP):
    def __init__(self, num_experts=4, hidden_size=3, expert_width=2):
        torch.nn.Module.__init__(self)
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_ffn_hidden_size=expert_width,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        self.num_local_experts = num_experts
        self.weight1 = torch.nn.Parameter(torch.ones(hidden_size, num_experts * expert_width))
        self.weight2 = torch.nn.Parameter(torch.ones(num_experts * expert_width, hidden_size))


class RouterLoraAndGroupedExpertsModel(torch.nn.Module):
    def __init__(self, num_experts=4, hidden_size=3):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=1,
            num_moe_experts=num_experts,
            moe_router_topk=2,
            attn_lora_num_experts=num_experts,
            attn_full_rank_lora_rank=2,
            attn_full_rank_lora_alpha=2,
            attn_full_rank_lora_targets="qkvo",
            attn_full_rank_lora_active_targets="",
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        self.router = TopKRouter(config)
        self.attn_lora_experts = SharedFullRankLoraExperts(
            config,
            input_size=hidden_size,
            query_output_size=hidden_size,
            value_output_size=hidden_size,
        )
        self.ffn_experts = MinimalGroupedMLP(num_experts=num_experts, hidden_size=hidden_size)
        self.q_full_rank_lora = torch.nn.Linear(hidden_size, hidden_size)
        self.dense = torch.nn.Linear(hidden_size, hidden_size)


def _backward_router_weight_sum(model):
    model.zero_grad(set_to_none=True)
    model.router.weight.sum().backward()
    return model.router.weight.grad.detach()


def _trainable_expert_and_router_loss(model):
    loss = model.router.weight.sum()
    loss = loss + model.ffn_experts.weight1.sum() + model.ffn_experts.weight2.sum()
    for param in model.attn_lora_experts.parameters():
        loss = loss + param.sum()
    return loss


def test_freeze_all_but_new_moe_params_masks_existing_router_rows_by_default():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=True,
    )

    grad = _backward_router_weight_sum(model)

    assert torch.count_nonzero(grad[:2]) == 0
    assert torch.all(grad[2:] == 1)


def test_freeze_all_but_new_moe_params_can_train_all_router_rows():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=False,
    )

    grad = _backward_router_weight_sum(model)

    assert torch.all(grad == 1)
    assert not model.dense.weight.requires_grad


def test_freeze_all_but_new_moe_params_can_train_all_experts_and_router_rows():
    model = RouterLoraAndGroupedExpertsModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=False,
        freeze_existing_router=False,
        train_dense_attention_lora=False,
    )

    model.zero_grad(set_to_none=True)
    _trainable_expert_and_router_loss(model).backward()

    assert torch.all(model.router.weight.grad == 1)
    assert torch.all(model.ffn_experts.weight1.grad == 1)
    assert torch.all(model.ffn_experts.weight2.grad == 1)
    for param in model.attn_lora_experts.parameters():
        assert param.requires_grad
        assert torch.all(param.grad == 1)
    assert not model.q_full_rank_lora.weight.requires_grad
    assert model.q_full_rank_lora.weight.grad is None
    assert not model.dense.weight.requires_grad
    assert model.dense.weight.grad is None


def test_allow_existing_router_grads_temporarily_bypasses_existing_row_mask():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=True,
    )

    model.zero_grad(set_to_none=True)
    with allow_existing_router_grads():
        model.router.weight.sum().backward()
    grad = model.router.weight.grad.detach()

    assert torch.all(grad == 1)


def test_zero_weight_router_kd_does_not_change_accumulated_lm_router_grads():
    model = RouterOnlyModel()

    freeze_all_but_new_moe_params(
        model,
        num_existing_experts=2,
        freeze_existing_experts=True,
        freeze_existing_router=True,
    )

    model.zero_grad(set_to_none=True)
    model.router.weight.sum().backward()
    lm_grad = model.router.weight.grad.detach().clone()

    with allow_existing_router_grads():
        (model.router.weight.sum() * 0.0).backward()
    grad_after_zero_weight_kd = model.router.weight.grad.detach()

    assert torch.equal(grad_after_zero_weight_kd, lm_grad)
    assert torch.count_nonzero(grad_after_zero_weight_kd[:2]) == 0
    assert torch.all(grad_after_zero_weight_kd[2:] == 1)


def test_teacher_student_router_kl_zero_padding_updates_new_student_logits():
    student_logits = torch.tensor([[0.3, -0.2, 1.1, -0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[1.0, -0.5]])

    loss = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=False,
    )
    loss.backward()

    assert torch.count_nonzero(student_logits.grad[:, 2:]) == 2


def test_teacher_student_router_kl_existing_only_matches_manual_wiki_row_kl():
    student_logits = torch.tensor(
        [
            [0.3, -0.2, 1.1, -0.7],
            [-0.4, 0.8, 0.5, 2.0],
        ]
    )
    teacher_logits = torch.tensor(
        [
            [1.0, -0.5],
            [-0.3, 0.2],
        ]
    )

    actual = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=True,
    )

    teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
    student_wiki_log_probs = torch.log_softmax(student_logits.float()[:, :2], dim=-1)
    expected = torch.nn.functional.kl_div(
        student_wiki_log_probs,
        teacher_probs,
        reduction="batchmean",
    )

    assert torch.allclose(actual, expected)


def test_teacher_student_router_kl_existing_only_is_invariant_to_new_logits():
    teacher_logits = torch.tensor([[1.0, -0.5]])
    student_logits = torch.tensor([[0.3, -0.2, 1.1, -0.7]])
    changed_new_logits = torch.tensor([[0.3, -0.2, 100.0, -100.0]])

    original_loss = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=True,
    )
    changed_loss = teacher_student_router_kl(
        changed_new_logits,
        teacher_logits,
        existing_experts_only=True,
    )

    assert torch.equal(original_loss, changed_loss)


def test_teacher_student_router_kl_existing_only_ignores_new_student_logits():
    student_logits = torch.tensor([[0.3, -0.2, 1.1, -0.7]], requires_grad=True)
    teacher_logits = torch.tensor([[1.0, -0.5]])

    loss = teacher_student_router_kl(
        student_logits,
        teacher_logits,
        existing_experts_only=True,
    )
    loss.backward()

    assert torch.count_nonzero(student_logits.grad[:, :2]) == 2
    assert torch.count_nonzero(student_logits.grad[:, 2:]) == 0
