#!/usr/bin/env python
"""CPU-only invariants for V2-new dual-route expert quota dispatch."""

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.Ours_LoRA_MoE import (  # noqa: E402
    LoRAMoEMLP,
    _quota_top1_dispatch,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    quota_lora_moe_expert,
    set_router_token_mask,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden=4, intermediate=6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


def test_selection():
    logits = torch.tensor([
        [3.0, 0.0, 2.0],
        [3.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [3.0, 0.0, 100.0],
    ])
    natural = torch.tensor([[0], [0], [1], [2]])
    valid = torch.tensor([True, True, True, False])
    dispatch, injected, diagnostic = _quota_top1_dispatch(
        logits, natural, 2, 2 / 3, valid)
    assert set(injected.tolist()) == {0, 1}
    assert dispatch[:, 0].tolist() == [2, 2, 1, 2]
    assert diagnostic["valid_tokens"] == 3
    assert diagnostic["dispatched_selected_tokens"] == 2
    assert diagnostic["injected_tokens"] == 2

    enough = natural.clone()
    enough[0, 0] = 2
    unchanged, injected, diagnostic = _quota_top1_dispatch(
        logits, enough, 2, 0.2, valid)
    torch.testing.assert_close(unchanged, enough, rtol=0, atol=0)
    assert injected.numel() == 0
    assert diagnostic["injected_tokens"] == 0


def test_zero_and_gradient_masks():
    torch.manual_seed(19)
    layer = LoRAMoEMLP(
        TinyMLP(), r=2, alpha=4, top_k=1,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    layer.add_experts(3)
    model = nn.Module()
    model.mlp = layer
    x = torch.randn(2, 4, 4)
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    set_router_token_mask(model, mask)
    try:
        with torch.no_grad():
            natural = layer(x)
            with quota_lora_moe_expert(model, 2, 0.0):
                zero_quota = layer(x)
        torch.testing.assert_close(zero_quota, natural, rtol=0, atol=0)

        freeze_lora_moe_experts(model, None)
        freeze_lora_moe_routers(model, True)
        natural_output = layer(x)
        (natural_output.square().mean() + layer._last_moe_loss).backward()
        assert layer.router.weight.grad is not None
        assert torch.count_nonzero(layer.router.weight.grad).item() > 0
        assert all(
            parameter.grad is None
            for expert in layer.experts for parameter in expert.parameters())

        freeze_lora_moe_experts(model, {2})
        freeze_lora_moe_routers(model, False)
        for parameter in model.parameters():
            parameter.grad = None
        with quota_lora_moe_expert(model, 2, 1.0):
            quota_output = layer(x)
            quota_output.square().mean().backward()
        assert layer.router.weight.grad is None
        assert all(
            parameter.grad is None
            for expert in layer.experts[:2]
            for parameter in expert.parameters())
        assert sum(
            float(parameter.grad.square().sum().item())
            for parameter in layer.experts[2].parameters()
            if parameter.grad is not None) > 0
        assert layer._last_quota_diagnostic["valid_tokens"] == 7
        assert layer._last_quota_diagnostic[
            "dispatched_selected_tokens"] == 7
    finally:
        set_router_token_mask(model, None)


if __name__ == "__main__":
    test_selection()
    print("PASS test_selection")
    test_zero_and_gradient_masks()
    print("PASS test_zero_and_gradient_masks")
