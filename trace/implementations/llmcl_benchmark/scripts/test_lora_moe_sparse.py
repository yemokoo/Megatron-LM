#!/usr/bin/env python
"""CPU regression test for LoRAMoEMLP sparse token dispatch."""

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from model.Ours_LoRA_MoE import LoRAMoEMLP


class TinyMLP(nn.Module):
    def __init__(self, hidden_size=7, intermediate_size=11):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, value):
        return self.down_proj(
            F.silu(self.gate_proj(value)) * self.up_proj(value))


def dense_reference(module, value):
    gate = module.base_mlp.gate_proj(value)
    up = module.base_mlp.up_proj(value)
    logits = module.router(value).reshape(-1, module.num_experts)
    topk_value, topk_index = logits.topk(module.top_k, dim=-1)
    if module.routing_weight_mode in {
            "full_softmax", "straight_through_topk"}:
        selected_probs = F.softmax(
            logits, dim=-1).gather(-1, topk_index)
        if module.routing_weight_mode == "straight_through_topk":
            normalized = F.softmax(topk_value, dim=-1)
            topk_weight = (
                normalized.detach() + selected_probs
                - selected_probs.detach())
        else:
            topk_weight = selected_probs
    else:
        topk_weight = F.softmax(topk_value, dim=-1)

    combined_weight = torch.zeros_like(logits)
    combined_weight.scatter_(-1, topk_index, topk_weight)
    combined_weight = combined_weight.reshape(
        *value.shape[:-1], module.num_experts)

    gate_delta = torch.zeros_like(gate)
    up_delta = torch.zeros_like(up)
    for index, expert in enumerate(module.experts):
        weight = combined_weight[..., index, None]
        if not torch.any(weight != 0):
            continue
        gate_delta = gate_delta + weight * expert["gate"](value)
        up_delta = up_delta + weight * expert["up"](value)

    hidden = F.silu(gate + gate_delta) * (up + up_delta)
    output = module.base_mlp.down_proj(hidden)
    down_delta = torch.zeros_like(output)
    for index, expert in enumerate(module.experts):
        weight = combined_weight[..., index, None]
        if not torch.any(weight != 0):
            continue
        down_delta = down_delta + weight * expert["down"](hidden)
    return output + down_delta, module._router_loss(logits, topk_index)


def assert_close(name, actual, expected, atol=2e-6, rtol=2e-5):
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        difference = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} differs; max_abs_diff={difference}")


def run_case(top_k, routing_weight_mode):
    torch.manual_seed(1234 + top_k)
    reference = LoRAMoEMLP(
        TinyMLP(), r=3, alpha=6, top_k=top_k,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode=routing_weight_mode)
    reference.add_experts(4)
    # The production initialization has zero B matrices. Make them nonzero so
    # the regression exercises every gate/up/down LoRA contribution and gradient.
    with torch.no_grad():
        for expert in reference.experts:
            for pair in expert.values():
                pair.B.normal_(mean=0.0, std=0.05)
    sparse = copy.deepcopy(reference)

    input_reference = torch.randn(3, 5, 7, requires_grad=True)
    input_sparse = input_reference.detach().clone().requires_grad_(True)
    output_reference, router_loss_reference = dense_reference(
        reference, input_reference)
    loss_reference = output_reference.square().mean() + router_loss_reference
    loss_reference.backward()

    output_sparse = sparse(input_sparse)
    router_loss_sparse = sparse._last_moe_loss
    loss_sparse = output_sparse.square().mean() + router_loss_sparse
    loss_sparse.backward()

    case = f"{routing_weight_mode}/top{top_k}"
    assert_close(f"{case} output", output_sparse, output_reference)
    assert_close(
        f"{case} router_loss", router_loss_sparse, router_loss_reference)
    assert_close(f"{case} input_grad", input_sparse.grad, input_reference.grad)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in sparse.named_parameters():
        expected_gradient = reference_parameters[name].grad
        if parameter.grad is None or expected_gradient is None:
            if parameter.grad is not expected_gradient:
                raise AssertionError(f"{case} gradient presence differs: {name}")
            continue
        assert_close(
            f"{case} parameter_grad {name}",
            parameter.grad,
            expected_gradient,
        )

    sparse.eval()
    with torch.inference_mode():
        sparse(input_sparse.detach())
    if sparse._last_moe_loss is not None:
        raise AssertionError("eval forward should not compute router auxiliary loss")


def lm_router_gradient_norm(top_k, routing_weight_mode):
    torch.manual_seed(4321 + top_k)
    module = LoRAMoEMLP(
        TinyMLP(), r=3, alpha=6, top_k=top_k,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode=routing_weight_mode)
    module.add_experts(4)
    with torch.no_grad():
        for expert in module.experts:
            for pair in expert.values():
                pair.B.normal_(mean=0.0, std=0.05)
    output = module(torch.randn(3, 5, 7))
    # Deliberately exclude _last_moe_loss: this measures whether the task/LM
    # objective itself can train the router through the selected weights.
    output.square().mean().backward()
    gradient = module.router.weight.grad
    return 0.0 if gradient is None else gradient.norm().item()


def padding_mask_audit():
    torch.manual_seed(9876)
    module = LoRAMoEMLP(
        TinyMLP(), r=3, alpha=6, top_k=1,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="full_softmax")
    module.add_experts(3)
    with torch.no_grad():
        for expert in module.experts:
            for pair in expert.values():
                pair.B.normal_(mean=0.0, std=0.05)
    value = torch.randn(2, 4, 7)
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.bool)
    module._router_token_mask = mask
    output = module(value)

    logits = module.router(value).reshape(-1, module.num_experts)
    _, selected = logits.topk(1, dim=-1)
    expected_router_loss = module._router_loss(
        logits[mask.reshape(-1)], selected[mask.reshape(-1)])
    assert_close("padding-masked router loss", module._last_moe_loss,
                 expected_router_loss)

    # Padding receives only the frozen dense FFN path; no LoRA expert delta.
    base_output = module.base_mlp(value)
    assert_close("padding dispatch", output[~mask], base_output[~mask])


def main():
    for mode in (
            "topk_softmax", "full_softmax", "straight_through_topk"):
        run_case(top_k=1, routing_weight_mode=mode)
        run_case(top_k=2, routing_weight_mode=mode)
    print("LORA_MOE_SPARSE_EQUIVALENCE=PASS")
    legacy_top1 = lm_router_gradient_norm(1, "topk_softmax")
    full_top1 = lm_router_gradient_norm(1, "full_softmax")
    full_top2 = lm_router_gradient_norm(2, "full_softmax")
    st_top1 = lm_router_gradient_norm(1, "straight_through_topk")
    st_top2 = lm_router_gradient_norm(2, "straight_through_topk")
    if (legacy_top1 != 0.0 or full_top1 <= 0.0 or full_top2 <= 0.0
            or st_top1 <= 0.0 or st_top2 <= 0.0):
        raise AssertionError(
            "unexpected LM-only router gradients: "
            f"legacy_top1={legacy_top1}, full_top1={full_top1}, "
            f"full_top2={full_top2}, st_top1={st_top1}, "
            f"st_top2={st_top2}")
    print(
        "LORA_ROUTER_LM_GRAD_AUDIT=PASS "
        f"legacy_top1={legacy_top1:.3g} full_top1={full_top1:.3g} "
        f"full_top2={full_top2:.3g} st_top1={st_top1:.3g} "
        f"st_top2={st_top2:.3g}")
    padding_mask_audit()
    print("LORA_ROUTER_PADDING_MASK=PASS")


if __name__ == "__main__":
    main()
