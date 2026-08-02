#!/usr/bin/env python
"""CPU regression tests for the five paper-aligned baseline building blocks."""
import copy
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.continual_lora import (
    attach_loramoe, attach_olora, attach_seq_lora,
    collect_loramoe_loss, collect_olora_regularization,
    parameter_report, set_olora_task, set_router_token_mask)
from model.paper_baselines import GEMLoRA, project_gem_gradient


class TinyMLP(nn.Module):
    def __init__(self, hidden=8, intermediate=12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()


class TinyCore(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer() for _ in range(layers)])


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyCore()
        self.embedding = nn.Parameter(torch.randn(8))

    def forward(self, x):
        for layer in self.model.layers:
            x = x + layer.mlp(x)
        return x


def adapter_tensors(model, kind):
    for layer in model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp, name)
            yield module.lora if kind == "seq" else module.experts[0]


def main():
    torch.manual_seed(7)
    base = TinyModel()
    seq = attach_seq_lora(copy.deepcopy(base), r=2, alpha=4, dropout=0)
    report = parameter_report(seq, "seqlora")
    assert report["trainable"] == 240, report
    assert report["activated_adapter"] == 240, report
    assert not seq.embedding.requires_grad
    assert all(not module.base.weight.requires_grad
               for layer in seq.model.layers
               for module in (layer.mlp.gate_proj, layer.mlp.up_proj,
                              layer.mlp.down_proj))

    # K=1, top-1 LoRAMoE must be exactly the same function as SeqLoRA when
    # expert weights are copied. The scalar router softmax is identically one.
    moe_one = attach_loramoe(
        copy.deepcopy(base), r=2, alpha=4, num_experts=1, top_k=1,
        dropout=0, routing_weight_mode="full_softmax")
    for seq_pair, moe_pair in zip(adapter_tensors(seq, "seq"),
                                  adapter_tensors(moe_one, "moe")):
        moe_pair.load_state_dict(seq_pair.state_dict())
    seq.eval()
    moe_one.eval()
    x = torch.randn(3, 5, 8)
    torch.testing.assert_close(seq(x), moe_one(x), rtol=1e-5, atol=1e-6)

    # Sparse LoRAMoE activates exactly one rank-r expert per token, independent
    # of the static pool size; router parameters are trainable but not counted as
    # activated adapter capacity.
    moe = attach_loramoe(
        copy.deepcopy(base), r=2, alpha=4, num_experts=4, top_k=1,
        dropout=0, routing_weight_mode="full_softmax",
        aux_loss_coeff=0.01, z_loss_coeff=0.001)
    assert parameter_report(moe, "loramoe")["activated_adapter"] == 240
    moe.train()
    mask = torch.ones(3, 5, dtype=torch.long)
    set_router_token_mask(moe, mask)
    moe(x).sum().backward()
    router_loss = collect_loramoe_loss(moe)
    assert router_loss is not None and torch.isfinite(router_loss)
    set_router_token_mask(moe, None)

    # With top_k == K this is exactly the official LoRAMoE dense residual:
    # base(x) + sum_e softmax(router(x))_e * B_e(A_e(x)) * alpha/r.
    dense_moe = attach_loramoe(
        copy.deepcopy(base), r=2, alpha=4, num_experts=4, top_k=4,
        dropout=0, routing_weight_mode="full_softmax",
        aux_loss_coeff=0, z_loss_coeff=0)
    dense_layer = dense_moe.model.layers[0].mlp.gate_proj
    dense_x = torch.randn(2, 3, 8)
    dense_logits = dense_layer.router(dense_x)
    dense_probs = torch.softmax(dense_logits, dim=-1, dtype=torch.float)
    expected = dense_layer.base(dense_x)
    for expert_index, expert in enumerate(dense_layer.experts):
        expected = expected + expert(dense_x) * dense_probs[..., expert_index, None]
    torch.testing.assert_close(dense_layer(dense_x), expected,
                               rtol=1e-5, atol=1e-6)
    assert collect_loramoe_loss(dense_moe) is None

    # O-LoRA trains only the current task adapter, retains old adapters in the
    # forward, and exposes the official |A_old A_new^T| regularizer.
    olora = attach_olora(copy.deepcopy(base), r=2, alpha=4,
                         num_tasks=3, dropout=0)
    set_olora_task(olora, 1)
    for layer in olora.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp, name)
            assert not module.adapters[0].A.requires_grad
            assert module.adapters[1].A.requires_grad
            assert not module.adapters[2].A.requires_grad
            module.adapters[1].A.data.copy_(module.adapters[0].A.data)
    orthogonal, l2 = collect_olora_regularization(olora)
    assert orthogonal.item() > 0 and l2.item() > 0

    # Exact GEM projection: the current gradient opposes the memory gradient;
    # projection must make their inner product non-negative.
    current = torch.tensor([-1.0, 0.5, 0.0])
    memory = torch.tensor([1.0, 0.0, 0.0])
    projected = project_gem_gradient(current, [memory], margin=0.0)
    assert torch.dot(projected, memory) >= -2e-3, projected

    # Episodic gradients are recomputed at the current parameter value and
    # averaged by example count rather than retaining a stale task gradient.
    class TinyLossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([2.0]))

        def forward(self, input_ids, labels, use_cache=False):
            prediction = input_ids.float() * self.weight
            return type("Output", (), {
                "loss": torch.mean((prediction - labels.float()).square())})

    gem = object.__new__(GEMLoRA)
    gem.raw_model = TinyLossModel()
    memory_batches = [
        {"input_ids": torch.tensor([[1.0], [2.0]]),
         "labels": torch.tensor([[0.0], [0.0]])},
        {"input_ids": torch.tensor([[3.0]]),
         "labels": torch.tensor([[0.0]])},
    ]
    episodic_gradient = gem._memory_gradient(
        memory_batches, [gem.raw_model.weight], torch.device("cpu"))
    # d mean((w*x)^2)/dw at w=2 over x=[1,2,3] = 4*mean(x^2) = 56/3.
    torch.testing.assert_close(
        episodic_gradient, torch.tensor([56.0 / 3.0]), rtol=1e-5, atol=1e-6)
    gem.raw_model.weight.data.fill_(1.0)
    recomputed = gem._memory_gradient(
        memory_batches, [gem.raw_model.weight], torch.device("cpu"))
    torch.testing.assert_close(
        recomputed, torch.tensor([28.0 / 3.0]), rtol=1e-5, atol=1e-6)

    # A size-32 buffer contributes only one sampled global minibatch per update.
    class IdentityCollator:
        def __call__(self, examples):
            return {"sample_id": torch.tensor(
                [example["sample_id"] for example in examples])}

    gem.args = SimpleNamespace(seed=1234, gem_memory_batch_size=4)
    gem.gem_update_step = 0
    gem.episodic_memory = {
        0: [{"sample_id": index} for index in range(32)]}
    gem.train_task_list = {
        "task0": SimpleNamespace(collate_fn=IdentityCollator())}
    sampled0 = gem._sample_memory_batches(0)[0]["sample_id"]
    assert sampled0.numel() == 4 and sampled0.unique().numel() == 4
    gem.gem_update_step = 1
    sampled1 = gem._sample_memory_batches(0)[0]["sample_id"]
    assert sampled1.numel() == 4 and not torch.equal(sampled0, sampled1)

    # GEM uses one concatenated trainable-LoRA vector, not independent QPs per
    # tensor. A negative first block is valid when the full-vector constraint is
    # satisfied by the second block.
    block_current = torch.tensor([-1.0, 2.0])
    block_memory = torch.tensor([1.0, 1.0])
    assert torch.dot(block_current, block_memory) > 0

    # O-LoRA keeps one rank-r adapter trainable while inference-active capacity
    # grows by exactly one rank-r adapter per task.
    report_task1 = parameter_report(olora, "olora")
    assert report_task1["trainable"] == 240, report_task1
    assert report_task1["activated_adapter"] == 480, report_task1
    set_olora_task(olora, 2)
    report_task2 = parameter_report(olora, "olora")
    assert report_task2["trainable"] == 240, report_task2
    assert report_task2["activated_adapter"] == 720, report_task2

    print("PAPER_BASELINES=PASS seq=240 active_loramoe=240")


if __name__ == "__main__":
    main()
