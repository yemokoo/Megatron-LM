#!/usr/bin/env python
"""CPU invariants for V2-new's expert-only auxiliary LM branch."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.Ours_LoRA_MoE import (  # noqa: E402
    LoRAMoEMLP,
    Ours_LoRA_MoE_V2,
    auxiliary_lora_moe_expert,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    set_router_token_mask,
)
from training.main_Ours_LoRA_MoE import (  # noqa: E402
    resolve_training_version_defaults,
    v2_new_v2_metadata_contract,
    validate_v2_new_args,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden=4, intermediate=6):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states):
        hidden = torch.nn.functional.silu(
            self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(hidden)


def make_layer():
    layer = LoRAMoEMLP(
        TinyMLP(), r=2, alpha=4, top_k=1,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    layer.add_experts(2)
    for expert in layer.experts:
        for parameter in expert.parameters():
            parameter.data.normal_(mean=0.0, std=0.1)
    return layer


def route_every_token_to(layer, expert_index):
    with torch.no_grad():
        layer.router.weight.fill_(-1.0)
        layer.router.weight[expert_index].fill_(1.0)


def manual_auxiliary_output(layer, x, mix):
    old = layer.experts[0]
    new = layer.experts[1]
    gate = layer.base_mlp.gate_proj(x)
    up = layer.base_mlp.up_proj(x)
    mixed_gate = gate + (1.0 - mix) * old["gate"](x) + mix * new["gate"](x)
    mixed_up = up + (1.0 - mix) * old["up"](x) + mix * new["up"](x)
    hidden = torch.nn.functional.silu(mixed_gate) * mixed_up
    return (layer.base_mlp.down_proj(hidden)
            + (1.0 - mix) * old["down"](hidden)
            + mix * new["down"](hidden))


def test_auxiliary_forward_is_real_interpolation_and_context_restores():
    torch.manual_seed(101)
    layer = make_layer()
    model = nn.Module()
    model.mlp = layer
    route_every_token_to(layer, 0)
    x = torch.ones(2, 3, 4)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    set_router_token_mask(model, mask)
    natural_before = layer(x).detach().clone()
    with auxiliary_lora_moe_expert(model, 1, 0.1):
        auxiliary = layer(x)
        expected = manual_auxiliary_output(layer, x, 0.1)
        # The masked padding token receives no adapter in the production path.
        expected[1, 2] = layer.base_mlp(x)[1, 2]
        torch.testing.assert_close(auxiliary, expected)
        diagnostic = layer._last_aux_diagnostic
        assert int(diagnostic["valid_tokens"]) == 5
        assert int(diagnostic["natural_selected_tokens"]) == 0
        assert int(diagnostic["auxiliary_exposed_tokens"]) == 5
        assert diagnostic["mix"] == 0.1
        assert layer._last_moe_loss is None
    natural_after = layer(x).detach()
    torch.testing.assert_close(natural_after, natural_before, rtol=0, atol=0)
    assert layer._training_aux_expert_index is None
    assert layer._last_aux_diagnostic is None
    set_router_token_mask(model, None)


def test_auxiliary_backward_changes_only_new_expert_gradient():
    torch.manual_seed(103)
    baseline = make_layer()
    route_every_token_to(baseline, 0)
    treatment = copy.deepcopy(baseline)
    baseline_model = nn.Module()
    baseline_model.mlp = baseline
    treatment_model = nn.Module()
    treatment_model.mlp = treatment
    for model in (baseline_model, treatment_model):
        freeze_lora_moe_experts(model, {1})
        freeze_lora_moe_routers(model, True)

    x = torch.ones(2, 3, 4)
    baseline_loss = baseline(x).square().mean() + baseline._last_moe_loss
    baseline_loss.backward()
    baseline_router_grad = baseline.router.weight.grad.detach().clone()
    baseline_new_grads = [
        (torch.zeros_like(parameter) if parameter.grad is None
         else parameter.grad.detach().clone())
        for parameter in baseline.experts[1].parameters()
    ]

    natural_loss = treatment(x).square().mean() + treatment._last_moe_loss
    natural_loss.backward()
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.raw_model = treatment_model
    with trainer._router_frozen(), auxiliary_lora_moe_expert(
            treatment_model, 1, 0.1):
        auxiliary_loss = treatment(x).square().mean()
        auxiliary_loss.backward()

    # The auxiliary backward cannot alter router gradients; the complete
    # router gradient is exactly the ordinary V2-new main-branch gradient.
    torch.testing.assert_close(
        treatment.router.weight.grad, baseline_router_grad, rtol=0, atol=0)
    assert all(
        parameter.grad is None
        for parameter in treatment.experts[0].parameters())
    treatment_new_grads = [
        (torch.zeros_like(parameter) if parameter.grad is None
         else parameter.grad.detach().clone())
        for parameter in treatment.experts[1].parameters()
    ]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(baseline_new_grads, treatment_new_grads))
    assert all(parameter.requires_grad
               for parameter in treatment.experts[1].parameters())
    assert treatment.router.weight.requires_grad


def test_naturally_selected_new_tokens_get_no_duplicate_aux_gradient():
    torch.manual_seed(107)
    layer = make_layer()
    model = nn.Module()
    model.mlp = layer
    route_every_token_to(layer, 1)
    freeze_lora_moe_experts(model, {1})
    freeze_lora_moe_routers(model, False)
    x = torch.ones(2, 3, 4)
    natural = layer(x).detach()
    with auxiliary_lora_moe_expert(model, 1, 0.1):
        auxiliary = layer(x)
        torch.testing.assert_close(auxiliary, natural, rtol=0, atol=0)
        auxiliary.square().mean().backward()
        diagnostic = layer._last_aux_diagnostic
        assert int(diagnostic["natural_selected_tokens"]) == 6
        assert int(diagnostic["auxiliary_exposed_tokens"]) == 0
    for parameter in layer.experts[1].parameters():
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad).item() == 0


def test_joint_loop_adds_aux_backward_but_keeps_one_optimizer_step():
    class TinyLoader:
        def __init__(self, batches, batch_size=1):
            self.batches = list(batches)
            self.batch_size = batch_size
            self.dataset = range(len(self.batches) * batch_size)
            self.sampler = SimpleNamespace()

        def __len__(self):
            return len(self.batches)

        def __iter__(self):
            return iter(self.batches)

    class TinyTrainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 4)
            self.lm_head = nn.Linear(4, 32, bias=False)
            self.mlp = make_layer()
            route_every_token_to(self.mlp, 0)
            self.forward_calls = 0

        def forward(self, input_ids, attention_mask=None, labels=None,
                    use_cache=False):
            self.forward_calls += 1
            hidden = self.embedding(input_ids)
            logits = self.lm_head(self.mlp(hidden))
            loss = None
            if labels is not None:
                loss = Ours_LoRA_MoE_V2._per_sample_causal_lm_losses(
                    logits, labels).mean()
            return SimpleNamespace(loss=loss, logits=logits)

    class CountingSGD(torch.optim.SGD):
        def __init__(self, parameters):
            super().__init__(parameters, lr=0.01)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    class Scheduler:
        def __init__(self):
            self.step_calls = 0

        def step(self):
            self.step_calls += 1

    torch.manual_seed(109)
    model = TinyTrainModel()
    model.embedding.weight.requires_grad = False
    model.lm_head.weight.requires_grad = False
    freeze_lora_moe_experts(model, {1})
    freeze_lora_moe_routers(model, True)
    old_before = [
        parameter.detach().clone()
        for parameter in model.mlp.experts[0].parameters()
    ]
    new_before = [
        parameter.detach().clone()
        for parameter in model.mlp.experts[1].parameters()
    ]
    router_before = model.mlp.router.weight.detach().clone()
    optimizer = CountingSGD([
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ])
    scheduler = Scheduler()
    args = SimpleNamespace(
        global_rank=1,
        gradient_accumulation_steps=1,
        v2_max_replay_batches_per_step=0,
        v2_replay_forward_batch_size=8,
        v2_joint_replay_loss_coeff=1.0,
        v2_joint_new_to_replay_ratio=2,
        loss_log_interval=1,
        training_version="v2_new",
        experts_per_task=1,
        top_k=1,
        routing_weight_mode="straight_through_topk",
        v2_new_expert_aux_mix=0.1,
        v2_new_expert_aux_loss_coeff=1.0,
        v2_new_expert_quota_schedule=[],
    )
    primary_loader = TinyLoader([
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.ones(2, 3, dtype=torch.long),
            "sources": ["p1", "p2"],
        },
        {
            "input_ids": torch.tensor([[7, 8, 9], [10, 11, 12]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "labels": torch.ones(2, 3, dtype=torch.long),
            "sources": ["p3", "p4"],
        },
    ], batch_size=2)
    memory_loader = TinyLoader([{
        "input_ids": torch.tensor([[13, 14, 15]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.ones(1, 3, dtype=torch.long),
        "sources": ["replay"],
    }])
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.raw_model = model
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.lr_scheduler = scheduler
    trainer.args = args
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    trainer._active_task_workload = None
    trainer._run_v2_joint_epochs(
        primary_loader, memory_loader, epochs=1,
        device=torch.device("cpu"), phase_name="aux-test")

    # Two primary batches each run natural+aux, and each optimizer update has
    # one replay forward: 2 * (2 + 1) == 6 forwards but only 2 steps.
    assert model.forward_calls == 6
    assert optimizer.step_calls == 2
    assert scheduler.step_calls == 2
    for parameter, snapshot in zip(
            model.mlp.experts[0].parameters(), old_before):
        torch.testing.assert_close(parameter, snapshot, rtol=0, atol=0)
    assert any(
        not torch.equal(parameter, snapshot)
        for parameter, snapshot in zip(
            model.mlp.experts[1].parameters(), new_before))
    assert not torch.equal(model.mlp.router.weight, router_before)


def make_args(**overrides):
    values = {
        "training_version": "v2_new",
        "replay_subset_ratio": None,
        "v2_joint_new_to_replay_ratio": None,
        "routing_weight_mode": None,
        "replay_subset_seed": -1,
        "seed": 2025,
        "v2_new_persistent_samples_per_task": 500,
        "v2_new_active_memory_cap": 1000,
        "replay_selection_mode": "random",
        "replay_distribution": "equal_task",
        "lora_moe_rank": 64,
        "lora_moe_alpha": 128,
        "experts_per_task": 1,
        "top_k": 1,
        "v2_memory_batch_size": 0,
        "v2_replay_forward_batch_size": 8,
        "v2_kd_memory_batch_size": 0,
        "v2_kd_loss_coeff": 1.0,
        "v2_kd_pass_multiplier": 1,
        "v2_kd_temperature": 1.0,
        "v2_kd_learning_rate": 0.0,
        "v2_kd_chunk_tokens": 256,
        "v2_kd_token_scope": "nonpad",
        "v2_joint_replay_loss_coeff": 1.0,
        "v2_max_replay_batches_per_step": 0,
        "v2_new_expert_quota_schedule": [],
        "v2_new_expert_aux_mix": 0.1,
        "v2_new_expert_aux_loss_coeff": 1.0,
    }
    values.update(overrides)
    return resolve_training_version_defaults(SimpleNamespace(**values))


def expect_value_error(callable_):
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_auxiliary_cli_contract_is_opt_in_and_resume_auditable():
    args = make_args()
    validate_v2_new_args(args)
    contract = v2_new_v2_metadata_contract(args)
    assert contract["new_expert_aux_mix"] == 0.1
    assert contract["new_expert_aux_loss_coeff"] == 1.0
    assert contract["new_expert_aux_optimizer_schedule"] == (
        "same_primary_update_single_optimizer_step")

    baseline = make_args(
        v2_new_expert_aux_mix=0.0,
        v2_new_expert_aux_loss_coeff=1.0)
    validate_v2_new_args(baseline)
    baseline_contract = v2_new_v2_metadata_contract(baseline)
    assert not any("aux" in key for key in baseline_contract)

    expect_value_error(lambda: validate_v2_new_args(make_args(
        v2_new_expert_aux_mix=1.1)))
    expect_value_error(lambda: validate_v2_new_args(make_args(
        v2_new_expert_aux_loss_coeff=0.0)))
    expect_value_error(lambda: validate_v2_new_args(make_args(
        v2_new_expert_quota_schedule=[0.5])))
    expect_value_error(lambda: validate_v2_new_args(make_args(
        training_version="v2")))


if __name__ == "__main__":
    tests = [
        test_auxiliary_forward_is_real_interpolation_and_context_restores,
        test_auxiliary_backward_changes_only_new_expert_gradient,
        test_naturally_selected_new_tokens_get_no_duplicate_aux_gradient,
        test_joint_loop_adds_aux_backward_but_keeps_one_optimizer_step,
        test_auxiliary_cli_contract_is_opt_in_and_resume_auditable,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
