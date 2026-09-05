#!/usr/bin/env python
"""CPU invariants for V1 expert-first acquisition and router integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.Ours_LoRA_MoE import (  # noqa: E402
    LoRAMoEMLP,
    Ours_LoRA_MoE_V1_Expert_First,
    force_lora_moe_expert,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    set_router_token_mask,
)
from training.main_Ours_LoRA_MoE import (  # noqa: E402
    resolve_training_version_defaults,
    validate_v2_new_args,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden=4, intermediate=7):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class RangeDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, index):
        return {"prompt": str(index), "answer": "x"}


def test_full_token_phase_updates_only_new_expert():
    torch.manual_seed(11)
    layer = LoRAMoEMLP(
        TinyMLP(), r=2, alpha=4, top_k=1,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    layer.add_experts(2)
    model = nn.Module()
    model.mlp = layer
    freeze_lora_moe_experts(model, {1})
    freeze_lora_moe_routers(model, False)

    inputs = torch.randn(2, 4, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    set_router_token_mask(model, mask)
    try:
        with force_lora_moe_expert(model, 1):
            output = layer(inputs)
            # Expert-first deliberately excludes layer._last_moe_loss.
            output.square().mean().backward()
    finally:
        set_router_token_mask(model, None)

    assert layer.router.weight.grad is None
    assert all(
        parameter.grad is None
        for parameter in layer.experts[0].parameters())
    new_grad_sq = sum(
        float(parameter.grad.square().sum().item())
        for parameter in layer.experts[1].parameters()
        if parameter.grad is not None)
    assert new_grad_sq > 0


def test_router_ft_uses_v2_new_seen_memory():
    task_names = ["task0", "task1", "task2"]
    trainer = object.__new__(Ours_LoRA_MoE_V1_Expert_First)
    trainer.train_task_list = {
        name: SimpleNamespace(dataset=RangeDataset()) for name in task_names}
    trainer._fixed_task_subsets = {}
    trainer._fixed_task_subset_indices = {}
    trainer._replay_manifest = None
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    trainer.args = SimpleNamespace(
        replay_subset_ratio=0.1,
        replay_distribution="equal_task",
        replay_subset_seed=2025,
        replay_selection_mode="random",
        seed=2025,
        global_rank=1,
        output_dir="/tmp/v1_expert_first_nonzero_rank",
        local_rank=-1,
        batch_by_task={name: 8 for name in task_names},
        max_prompt_len=32,
        max_ans_len=8,
        max_train_len=0,
        train_format="raw_answer",
        use_pretokenized_train_cache=False,
        router_retune_epochs=1,
        router_replay_exposure_samples=1000,
        v2_new_persistent_samples_per_task=500,
        v2_new_active_memory_cap=1000,
        v2_memory_batch_size=0,
        v2_kd_memory_batch_size=8,
        gradient_accumulation_steps=1,
    )
    loader = trainer._build_expert_first_router_loader(i_task=2)
    assert len(loader.dataset) == 1000
    assert loader.batch_size == 8
    assert [len(stream) for stream in loader.dataset.datasets] == [334, 333, 333]


def test_profile_defaults_and_shape_contract():
    args = SimpleNamespace(
        training_version="v1_expert_first",
        replay_subset_ratio=None,
        v2_joint_new_to_replay_ratio=None,
        routing_weight_mode=None,
        replay_subset_seed=-1,
        seed=2025,
        v2_new_persistent_samples_per_task=500,
        v2_new_active_memory_cap=1000,
        replay_selection_mode="random",
        replay_distribution="equal_task",
        lora_moe_rank=8,
        experts_per_task=1,
        top_k=1,
        v2_new_expert_quota_schedule=[],
    )
    resolve_training_version_defaults(args)
    validate_v2_new_args(args)
    assert args.replay_subset_ratio == 0.1
    assert args.v2_joint_new_to_replay_ratio == 0
    assert args.routing_weight_mode == "straight_through_topk"
    assert args.replay_subset_seed == 2025


if __name__ == "__main__":
    test_full_token_phase_updates_only_new_expert()
    print("PASS test_full_token_phase_updates_only_new_expert")
    test_router_ft_uses_v2_new_seen_memory()
    print("PASS test_router_ft_uses_v2_new_seen_memory")
    test_profile_defaults_and_shape_contract()
    print("PASS test_profile_defaults_and_shape_contract")
