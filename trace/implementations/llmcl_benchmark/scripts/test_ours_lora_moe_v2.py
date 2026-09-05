#!/usr/bin/env python
"""CPU invariants for Ours LoRA-MoE v2; no backbone or dataset required."""
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        v2_kd_memory_batch_size=8,
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
    kd_loader = trainer._build_v2_memory_loader(i_task=3, role="kd")
    replay_loader = trainer._build_v2_memory_loader(i_task=3, role="replay")
    assert len(kd_loader.dataset) == 1000
    assert len(replay_loader.dataset) == 1000
    assert kd_loader.batch_size == 8
    assert replay_loader.batch_size == 1
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
    trainer.tokenizer = type(
        "Tokenizer", (), {"pad_token_id": 0, "padding_side": "right"})()
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
    assert merged["input_ids"][0].tolist() == [7, 8, 9, 10, 0, 0, 0, 0]
    assert merged["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert merged["labels"][0].tolist()[-4:] == [-100] * 4
    assert merged["sources"] == ["a", "b"]
    assert Ours_LoRA_MoE_V2._valid_token_count(merged) == 12
    assert merged["labels"][:, 1:].ne(-100).sum(dim=1).tolist() == [2, 4]


def test_packed_replay_preserves_per_sample_ce_and_gradient():
    """Packed variable-length replay equals a sum of singleton objectives."""
    torch.manual_seed(2025)
    reference = nn.Linear(5, 17, bias=False)
    packed = nn.Linear(5, 17, bias=False)
    packed.load_state_dict(reference.state_dict())
    features = [
        torch.randn(1, 4, 5),
        torch.randn(1, 7, 5),
        torch.randn(1, 5, 5),
    ]
    labels = [
        torch.tensor([[1, 2, 3, 4]]),
        torch.tensor([[5, 6, 7, 8, 9, 10, 11]]),
        torch.tensor([[-100, -100, 12, 13, 14]]),
    ]

    singleton_losses = []
    for row_features, row_labels in zip(features, labels):
        row_logits = reference(row_features)
        row_loss = Ours_LoRA_MoE_V2._per_sample_causal_lm_losses(
            row_logits, row_labels).sum()
        singleton_losses.append(row_loss.detach())
        row_loss.backward()

    max_length = max(row.shape[1] for row in features)
    packed_features = torch.cat([
        F.pad(row, (0, 0, 0, max_length - row.shape[1]))
        for row in features
    ], dim=0)
    packed_labels = torch.cat([
        F.pad(row, (0, max_length - row.shape[1]), value=-100)
        for row in labels
    ], dim=0)
    packed_losses = Ours_LoRA_MoE_V2._per_sample_causal_lm_losses(
        packed(packed_features), packed_labels)
    torch.testing.assert_close(
        packed_losses.detach(), torch.stack(singleton_losses))
    packed_losses.sum().backward()
    torch.testing.assert_close(
        packed.weight.grad, reference.weight.grad,
        rtol=1e-5, atol=1e-6)

    # The naive Hugging Face-style token-global mean is intentionally not the
    # same objective when rows have different supervised lengths.
    shift_labels = F.pad(
        packed_labels, (0, 1), value=-100)[..., 1:].contiguous()
    naive_loss = F.cross_entropy(
        packed(packed_features).reshape(-1, 17),
        shift_labels.reshape(-1), ignore_index=-100)
    assert not torch.isclose(
        naive_loss * len(features), packed_losses.sum(),
        rtol=1e-4, atol=1e-5)


def test_layer_hidden_mse_is_sample_and_layer_mean_with_padding_mask():
    student_a = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
        [[2.0, 0.0], [4.0, 2.0], [6.0, 4.0]],
    ], requires_grad=True)
    student_b = (student_a * 0.5).detach().requires_grad_(True)
    teacher_a = torch.zeros_like(student_a)
    teacher_b = torch.ones_like(student_b)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    losses = Ours_LoRA_MoE_V2._per_sample_layer_hidden_mse(
        [student_a, student_b], [teacher_a, teacher_b], mask)

    expected = []
    for row, valid_count in enumerate((2, 3)):
        layer_a = student_a[row, :valid_count].square().mean()
        layer_b = (student_b[row, :valid_count] - 1).square().mean()
        expected.append((layer_a + layer_b) / 2)
    torch.testing.assert_close(losses, torch.stack(expected))
    losses.sum().backward()
    assert student_a.grad is not None and student_a.grad.abs().sum() > 0
    assert student_b.grad is not None and student_b.grad.abs().sum() > 0
    assert teacher_a.grad is None and teacher_b.grad is None
    assert torch.count_nonzero(student_a.grad[0, 2]) == 0


def test_manual_gradient_average_coalesces_without_changing_values():
    model = nn.Sequential(
        nn.Linear(4, 6, bias=False),
        nn.Linear(6, 3, bias=False))
    parameters = list(model.parameters())
    parameters[0].grad = torch.randn_like(parameters[0])
    parameters[1].grad = None
    expected = parameters[0].grad.clone()
    calls = []
    originals = (
        torch.distributed.is_initialized,
        torch.distributed.get_world_size,
        torch.distributed.all_reduce,
    )
    try:
        torch.distributed.is_initialized = lambda: True
        torch.distributed.get_world_size = lambda: 4

        def fake_all_reduce(tensor):
            calls.append(tensor.numel())
            # Simulate four identical ranks. The subsequent world division
            # must recover the exact local values.
            tensor.mul_(4)

        torch.distributed.all_reduce = fake_all_reduce
        Ours_LoRA_MoE_V2._manual_average_gradients(
            model, bucket_bytes=1024 * 1024)
    finally:
        (torch.distributed.is_initialized,
         torch.distributed.get_world_size,
         torch.distributed.all_reduce) = originals
    assert len(calls) == 1, calls
    torch.testing.assert_close(parameters[0].grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        parameters[1].grad, torch.zeros_like(parameters[1]),
        rtol=0, atol=0)


def test_joint_loop_consumes_exact_replay_stream_and_updates_primary_steps():
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
            self.mlp.add_experts(2)
            for expert in self.mlp.experts:
                for parameter in expert.parameters():
                    parameter.data.normal_(mean=0.0, std=0.1)
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

    torch.manual_seed(19)
    model = TinyTrainModel()
    for parameter in model.embedding.parameters():
        parameter.requires_grad = False
    for parameter in model.lm_head.parameters():
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
        v2_replay_forward_batch_size=8,
        v2_joint_replay_loss_coeff=1.0,
        v2_joint_new_to_replay_ratio=2,
        loss_log_interval=1,
    )
    primary_loader = TinyLoader([
        {
            "input_ids": torch.tensor([
                [1, 2, 3], [4, 5, 6], [2, 4, 6], [3, 5, 7]]),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "labels": torch.ones(4, 3, dtype=torch.long),
            "sources": ["p1", "p2", "p3", "p4"],
        },
        {
            "input_ids": torch.tensor([
                [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]]),
            "attention_mask": torch.ones(4, 3, dtype=torch.long),
            "labels": torch.ones(4, 3, dtype=torch.long),
            "sources": ["p5", "p6", "p7", "p8"],
        },
    ], batch_size=4)
    memory_loader = TinyLoader([
        {
            "input_ids": torch.tensor([[19, 20, 21]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.ones(1, 3, dtype=torch.long),
            "sources": ["replay-1"],
        },
        {
            "input_ids": torch.tensor([[22, 23, 24, 25, 26]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "labels": torch.ones(1, 5, dtype=torch.long),
            "sources": ["replay-2"],
        },
    ])
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
    # Every optimizer update contains two replay records packed into one
    # variable-length forward. The two-record pool cycles once to obtain the
    # exact 8:4 == 2:1 exposure ratio.
    assert model.forward_calls == len(primary_loader) + 2, model.forward_calls
    assert optimizer.step_calls == len(primary_loader), optimizer.step_calls
    assert scheduler.step_calls == len(primary_loader), scheduler.step_calls


def test_every_update_replay_assignment_is_nonempty_and_exact():
    # The actual 8-GPU task shapes use 1,000 global replay records. MeetingBank
    # has 553 optimizer updates; each receives one or two replay records.
    assignments = [
        Ours_LoRA_MoE_V2._replay_exposure_assignment(
            1000, 553, update, 8, rank)
        for update in range(553)
        for rank in range(8)
    ]
    per_update = []
    per_rank = [0] * 8
    for update in range(553):
        row = assignments[update * 8:(update + 1) * 8]
        starts = {start for start, _, _ in row}
        stops = {stop for _, stop, _ in row}
        assert len(starts) == len(stops) == 1
        count = next(iter(stops)) - next(iter(starts))
        assert count >= 1
        assert sum(local for _, _, local in row) == count
        per_update.append(count)
        for rank, (_, _, local) in enumerate(row):
            per_rank[rank] += local
    assert sum(per_update) == 1000
    assert per_update.count(1) == 106
    assert per_update.count(2) == 447
    assert per_rank == [125] * 8
    assert Ours_LoRA_MoE_V2._replay_loss_scale(8, 4) == 2.0
    assert Ours_LoRA_MoE_V2._replay_loss_scale(8, 5) == 1.6
    assert Ours_LoRA_MoE_V2._replay_loss_scale(8, 8) == 1.0


def test_five_to_one_budget_is_not_multiplied_by_epochs():
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.args = SimpleNamespace(
        v2_joint_new_to_replay_ratio=5,
        router_replay_exposure_samples=1000,
    )
    primary_loader = SimpleNamespace(
        sampler=SimpleNamespace(total_size=5000))
    assert trainer._joint_replay_exposure_budget(
        primary_loader, epochs=3) == 1000
    assert trainer._joint_replay_exposure_budget(
        primary_loader, epochs=5) == 1000
    assert trainer._joint_replay_exposure_budget(
        primary_loader, epochs=7) == 1000

    # The one fixed 1,000-exposure budget is spread across every update in a
    # five-epoch phase rather than restarted at each epoch boundary.
    total_updates = 5 * 79
    per_rank = [0] * 8
    per_update = []
    for update in range(total_updates):
        row = [
            Ours_LoRA_MoE_V2._replay_exposure_assignment(
                1000, total_updates, update, 8, rank)
            for rank in range(8)
        ]
        start, stop = row[0][:2]
        count = stop - start
        assert sum(local for _, _, local in row) == count
        per_update.append(count)
        for rank, (_, _, local) in enumerate(row):
            per_rank[rank] += local
    assert sum(per_update) == 1000
    assert all(count >= 1 for count in per_update)
    assert per_rank == [125] * 8


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


def test_slora_trace_collator_answer_only_probe_labels():
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def apply_chat_template(
                self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            text = "|".join(
                f"{item['role']}:{item['content']}" for item in messages)
            if add_generation_prompt:
                text += "|assistant:"
            return text

        def __call__(self, text, truncation, max_length, padding,
                     return_tensors):
            ids = [(ord(char) % 50) + 1 for char in text][:max_length]
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    tokenizer = FakeTokenizer()
    collator = SLoRATraceDataCollator(
        tokenizer, max_length=256, label_scope="answer")
    batch = collator([
        {"prompt": "question", "answer": "answer"},
        {"prompt": "q", "answer": "x"},
    ])
    for row in range(2):
        labels = batch["labels"][row]
        active = batch["attention_mask"][row].bool()
        assert labels[active].eq(-100).any()
        assert labels[active].ne(-100).any()
        torch.testing.assert_close(
            labels[labels.ne(-100)], batch["input_ids"][row][labels.ne(-100)])


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
        test_slora_trace_collator_answer_only_probe_labels,
        test_pre_expansion_teacher_prefix,
        test_fixed_memory_budget_does_not_grow_with_task_count,
        test_accumulated_fixed_subsets_but_capped_replay_exposure,
        test_valid_token_ratio_accounting,
        test_replay_sources_merge_into_one_forward_batch,
        test_packed_replay_preserves_per_sample_ce_and_gradient,
        test_layer_hidden_mse_is_sample_and_layer_mean_with_padding_mask,
        test_manual_gradient_average_coalesces_without_changing_values,
        test_joint_loop_consumes_exact_replay_stream_and_updates_primary_steps,
        test_every_update_replay_assignment_is_nonempty_and_exact,
        test_five_to_one_budget_is_not_multiplied_by_epochs,
        test_v1_phase1_updates_only_new_router_row_then_phase2_can_update_all,
        test_two_backwards_add_router_only_replay_gradient,
        test_workload_accounting_json,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
