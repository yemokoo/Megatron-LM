# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.moe.continual_learning_utils import (
    allow_existing_router_grads,
    freeze_all_but_new_moe_params,
    teacher_student_router_kl,
)
from megatron.core.transformer.moe.router import TopKRouter
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


def _backward_router_weight_sum(model):
    model.zero_grad(set_to_none=True)
    model.router.weight.sum().backward()
    return model.router.weight.grad.detach()


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
