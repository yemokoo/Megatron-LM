#!/usr/bin/env python
"""CPU invariants for Ours LoRA-MoE v2; no backbone or dataset required."""
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from types import SimpleNamespace
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data.data_collator import SLoRATraceDataCollator  # noqa: E402
from model.Ours_LoRA_MoE import (  # noqa: E402
    LoRAMoEMLP,
    Ours_LoRA_MoE,
    Ours_LoRA_MoE_V2,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    limit_lora_moe_experts,
)


class TinyMLP(nn.Module):
    def __init__(self, hidden=4, intermediate=7):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


def make_layer():
    return LoRAMoEMLP(
        TinyMLP(), r=2, alpha=4, top_k=2,
        aux_loss_coeff=0.0, z_loss_coeff=0.0,
        routing_weight_mode="full_softmax")


def test_pre_expansion_teacher_prefix():
    torch.manual_seed(7)
    layer = make_layer()
    layer.add_experts(1)
    for parameter in layer.experts[0].parameters():
        if parameter.ndim == 2:
            parameter.data.normal_(mean=0.0, std=0.1)
    x = torch.randn(2, 3, 4)
    before = layer(x).detach()
    layer.add_experts(1)
    with limit_lora_moe_experts(layer, 1):
        recovered = layer(x).detach()
    torch.testing.assert_close(recovered, before)


def test_fixed_memory_budget_does_not_grow_with_task_count():
    counts = Ours_LoRA_MoE_V2._allocate_memory_counts(
        [5000, 5000, 5000], total=1000, mode="equal_task")
    assert sum(counts) == 1000, counts
    assert max(counts) - min(counts) <= 1, counts
    tiny = Ours_LoRA_MoE_V2._allocate_memory_counts(
        [5000, 5000, 5000], total=5, mode="equal_task")
    assert sum(tiny) == 5, tiny


def test_accumulated_fixed_subsets_but_capped_replay_exposure():
    class RangeDataset(Dataset):
        def __len__(self):
            return 5000

        def __getitem__(self, index):
            return {"prompt": str(index), "answer": "x"}

    task_names = ["task0", "task1", "task2", "task3"]
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.train_task_list = {
        name: SimpleNamespace(dataset=RangeDataset()) for name in task_names
    }
    trainer._fixed_task_subsets = {}
    trainer._fixed_task_subset_indices = {}
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    trainer.args = SimpleNamespace(
        replay_subset_ratio=0.01,
        replay_distribution="equal_task",
        replay_subset_seed=17,
        seed=2025,
        global_rank=1,
        output_dir="/tmp/not-written-on-nonzero-rank",
        local_rank=-1,
        batch_by_task={name: 10 for name in task_names},
        max_prompt_len=32,
        max_ans_len=8,
        router_retune_epochs=1,
        router_replay_exposure_samples=1000,
        v2_memory_batch_size=0,
        gradient_accumulation_steps=8,
    )
    subsets = [trainer._ensure_fixed_task_subset(name) for name in task_names]
    assert [len(subset) for subset in subsets] == [50, 50, 50, 50]

    loader = trainer._build_fixed_memory_loader(
        task_names, round_index=3, phase="test",
        exposure_samples=1000, batch_size=10)
    assert len(loader.dataset) == 1000
    assert [len(dataset) for dataset in loader.dataset.datasets] == [250] * 4

    # v1 always consumes exactly 1,000 global router-replay exposures,
    # independently of new-task epochs. Four seen tasks receive 250 each.
    v1_loader = trainer._build_replay_loader(i_task=3)
    assert len(v1_loader.dataset) == 1000
    assert [len(dataset) for dataset in v1_loader.dataset.datasets] == [250] * 4

    # v1 is two-phase even on the first task: its 50 unique records repeat
    # exactly 20 times to produce the same 1,000-exposure phase budget.
    first_v1_loader = trainer._build_replay_loader(i_task=0)
    assert len(first_v1_loader.dataset) == 1000
    assert len(first_v1_loader.dataset.datasets) == 1
    assert len(first_v1_loader.dataset.datasets[0]) == 1000

    # v2 KD and joint replay are separate loaders over the exact same
    # deterministic past-only 1,000-record exposure stream.
    kd_loader = trainer._build_v2_memory_loader(i_task=3)
    replay_loader = trainer._build_v2_memory_loader(i_task=3)
    assert len(kd_loader.dataset) == 1000
    assert len(replay_loader.dataset) == 1000
    kd_streams = [dataset.exposure_indices for dataset in kd_loader.dataset.datasets]
    replay_streams = [dataset.exposure_indices for dataset in replay_loader.dataset.datasets]
    assert kd_streams == replay_streams
    assert [len(dataset) for dataset in kd_loader.dataset.datasets] == [334, 333, 333]

    fake_loader = type("Loader", (), {"__len__": lambda self: 625})()
    assert trainer._optimizer_update_count(fake_loader, epochs=5) == 395


def test_valid_token_ratio_accounting():
    batch = {"attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])}
    assert Ours_LoRA_MoE_V2._valid_token_count(batch) == 5
    assert int(torch.ceil(torch.tensor(100 / 5.0)).item()) == 20


def test_replay_sources_merge_into_one_forward_batch():
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    first = {
        "input_ids": torch.tensor([[7, 8, 9, 10]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        "labels": torch.tensor([[-100, -100, 9, 10]]),
        "sources": ["a"],
    }
    second = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, -100, 5, 6, 7, 8]]),
        "sources": ["b"],
    }
    merged = trainer._merge_replay_batches([first, second])
    assert merged["input_ids"].shape == (2, 8)
    assert merged["input_ids"][0].tolist() == [0, 0, 0, 0, 7, 8, 9, 10]
    assert merged["attention_mask"][0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert merged["labels"][0].tolist()[:4] == [-100] * 4
    assert merged["sources"] == ["a", "b"]
    assert Ours_LoRA_MoE_V2._valid_token_count(merged) == 12


def test_joint_loop_consumes_exact_replay_stream_and_updates_primary_steps():
    class TinyTrainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(32, 4)
            self.mlp = make_layer()
            self.mlp.add_experts(2)
            for expert in self.mlp.experts:
                for parameter in expert.parameters():
                    parameter.data.normal_(mean=0.0, std=0.1)
            self.forward_calls = 0

        def forward(self, input_ids, attention_mask=None, labels=None,
                    use_cache=False):
            self.forward_calls += 1
            hidden = self.embedding(input_ids)
            return SimpleNamespace(loss=self.mlp(hidden).square().mean())

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

    torch.manual_seed(19)
    model = TinyTrainModel()
    for parameter in model.embedding.parameters():
        parameter.requires_grad = False
    freeze_lora_moe_experts(model, trainable_expert_indices={1})
    freeze_lora_moe_routers(model, trainable=True)
    optimizer = CountingSGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad])
    scheduler = Scheduler()
    args = SimpleNamespace(
        global_rank=1,
        gradient_accumulation_steps=1,
        v2_max_replay_batches_per_step=0,
        v2_joint_replay_loss_coeff=1.0,
        loss_log_interval=1,
    )
    primary_loader = [
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
    ]
    memory_loader = [{
        "input_ids": torch.tensor([[13, 14, 15]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.ones(1, 3, dtype=torch.long),
        "sources": ["replay"],
    }]
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
        device=torch.device("cpu"), phase_name="test")
    # Two primary forwards plus the one-item replay stream consumed exactly
    # once across the task. Optimizer cadence remains tied to primary steps.
    assert model.forward_calls == len(primary_loader) + len(memory_loader), model.forward_calls
    assert optimizer.step_calls == len(primary_loader), optimizer.step_calls
    assert scheduler.step_calls == len(primary_loader), scheduler.step_calls


def test_v1_phase1_updates_only_new_router_row_then_phase2_can_update_all():
    torch.manual_seed(23)
    layer = make_layer()
    layer.add_experts(1)
    for parameter in layer.experts[0].parameters():
        parameter.data.normal_(mean=0.0, std=0.1)
    layer.router.weight.data.normal_(mean=0.0, std=0.1)
    layer.add_experts(1)
    for parameter in layer.experts[1].parameters():
        parameter.data.normal_(mean=0.0, std=0.1)
    freeze_lora_moe_experts(layer, trainable_expert_indices={1})
    freeze_lora_moe_routers(layer, trainable=True)

    old_snapshot = Ours_LoRA_MoE._snapshot_old_router_rows(layer, 1)
    old_before = layer.router.weight[0].detach().clone()
    new_before = layer.router.weight[1].detach().clone()
    optimizer = torch.optim.SGD(
        [parameter for parameter in layer.parameters() if parameter.requires_grad],
        lr=0.1)
    layer(torch.randn(3, 4, 4)).square().mean().backward()
    assert torch.count_nonzero(layer.router.weight.grad[0])
    Ours_LoRA_MoE._freeze_old_router_row_update(layer, old_snapshot, 1)
    optimizer.step()
    Ours_LoRA_MoE._freeze_old_router_row_update(layer, old_snapshot, 1)
    torch.testing.assert_close(layer.router.weight[0], old_before, rtol=0, atol=0)
    assert not torch.equal(layer.router.weight[1], new_before)

    # Phase 2 omits the prefix guard, so the same old row can now change.
    optimizer.zero_grad(set_to_none=True)
    layer(torch.randn(3, 4, 4)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(layer.router.weight[0], old_before)


def test_two_backwards_add_router_only_replay_gradient():
    torch.manual_seed(11)
    layer = make_layer()
    layer.add_experts(2)
    for expert in layer.experts:
        for parameter in expert.parameters():
            parameter.data.normal_(mean=0.0, std=0.1)
    freeze_lora_moe_experts(layer, trainable_expert_indices={1})
    freeze_lora_moe_routers(layer, trainable=True)

    primary_x = torch.randn(2, 3, 4)
    primary_loss = layer(primary_x).square().mean()
    primary_loss.backward()
    expert_grads_before = [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in layer.experts[1].parameters()
    ]
    router_grad_before = layer.router.weight.grad.detach().clone()
    assert any(grad is not None and torch.count_nonzero(grad) for grad in expert_grads_before)

    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.raw_model = layer
    with trainer._router_only_replay():
        assert not any(
            parameter.requires_grad
            for expert in layer.experts for parameter in expert.parameters())
        replay_x = torch.randn(1, 5, 4)
        layer(replay_x).square().mean().backward()

    for parameter, before in zip(layer.experts[1].parameters(), expert_grads_before):
        if before is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, before)
    assert not torch.equal(layer.router.weight.grad, router_grad_before)
    assert all(parameter.requires_grad for parameter in layer.experts[1].parameters())
    assert not any(parameter.requires_grad for parameter in layer.experts[0].parameters())


def test_slora_trace_collator_full_labels_and_right_padding():
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def __init__(self):
            self.messages = []

        def apply_chat_template(self, messages, tokenize=False):
            assert tokenize is False
            assert [item["role"] for item in messages] == [
                "system", "user", "assistant"]
            assert messages[0]["content"] == "You are a helpful assistant."
            self.messages.append(messages)
            return "|".join(item["content"] for item in messages)

        def __call__(self, text, truncation, max_length, padding,
                     return_tensors):
            ids = [(ord(char) % 50) + 1 for char in text][:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer = FakeTokenizer()
    collator = SLoRATraceDataCollator(tokenizer, max_length=128)
    batch = collator([
        {"prompt": "a", "answer": "x"},
        {"prompt": "a longer prompt", "answer": "a longer answer"},
    ])
    assert len(tokenizer.messages) == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["labels"][0, -1].item() == -100
    for row in range(2):
        active = batch["attention_mask"][row].bool()
        torch.testing.assert_close(
            batch["labels"][row][active], batch["input_ids"][row][active])
        assert torch.all(batch["labels"][row][~active].eq(-100))


def test_workload_accounting_json():
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    with tempfile.TemporaryDirectory() as output_dir:
        trainer.args = SimpleNamespace(
            output_dir=output_dir, global_rank=0, training_version="v2")
        trainer._active_task_workload = {
            "round": 1,
            "task": "task1",
            "epochs": 3,
            "router_replay_exposure_budget_global": 1000,
            "roles": {},
            "optimizer_updates": 0,
        }
        batch = {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
        trainer._count_workload_batch(
            "kd_init", batch, forward_passes=2, backward_passes=1)
        trainer._count_workload_update()
        record = trainer._finalize_task_workload(
            elapsed_seconds=1.25, local_flops=1234,
            flop_counter_status="enabled")
        assert record["roles"]["kd_init"]["input_sample_exposures"] == 2
        assert record["roles"]["kd_init"]["forward_sample_instances"] == 4
        assert record["roles"]["kd_init"]["forward_nonpad_token_instances"] == 10
        trainer._workload_records = [record]
        trainer._write_workload_records()
        payload = json.loads(
            (Path(output_dir) / "training_workload.json").read_text())
        assert payload["totals"]["optimizer_updates"] == 1
        assert payload["totals"]["counted_operator_flops_global"] == 1234


def main():
    tests = [
        test_slora_trace_collator_full_labels_and_right_padding,
        test_pre_expansion_teacher_prefix,
        test_fixed_memory_budget_does_not_grow_with_task_count,
        test_accumulated_fixed_subsets_but_capped_replay_exposure,
        test_valid_token_ratio_accounting,
        test_replay_sources_merge_into_one_forward_batch,
        test_joint_loop_consumes_exact_replay_stream_and_updates_primary_steps,
        test_v1_phase1_updates_only_new_router_row_then_phase2_can_update_all,
        test_two_backwards_add_router_only_replay_gradient,
        test_workload_accounting_json,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
