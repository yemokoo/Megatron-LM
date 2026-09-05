#!/usr/bin/env python
"""CPU invariants for the isolated FFN-only Ours LoRA-MoE V2-new variant."""

import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.Ours_LoRA_MoE import (  # noqa: E402
    LoRAMoEMLP,
    Ours_LoRA_MoE_V2,
    Ours_LoRA_MoE_V2_New,
    add_experts_to_all_layers,
    attach_lora_moe,
    _quota_top1_dispatch,
    freeze_lora_moe_experts,
    freeze_lora_moe_routers,
    limit_lora_moe_experts,
    load_lora_moe_checkpoint,
    quota_lora_moe_expert,
    save_lora_moe_meta,
    set_router_token_mask,
)
from training.main_Ours_LoRA_MoE import (  # noqa: E402
    metadata_contract_mismatches,
    resolve_training_version_defaults,
    v2_new_metadata_contract,
    v2_new_replay_memory_contract,
    v2_new_v2_metadata_contract,
    v2_resume_metadata_mismatches,
    validate_v2_new_resume_persisted_identities,
    validate_v2_new_args,
)


class RangeDataset(Dataset):
    def __init__(self, size=5000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {"prompt": str(index), "answer": "x"}


class TinyMLP(nn.Module):
    def __init__(self, hidden=32, intermediate=48):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, hidden_states):
        gated = torch.nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gated * self.up_proj(hidden_states))


def make_trainer(seed=2025, replay_subset_seed=-1):
    task_names = [f"task{index}" for index in range(8)]
    trainer = object.__new__(Ours_LoRA_MoE_V2_New)
    trainer.train_task_list = {
        name: SimpleNamespace(dataset=RangeDataset()) for name in task_names
    }
    trainer._fixed_task_subsets = {}
    trainer._fixed_task_subset_indices = {}
    trainer._replay_manifest = None
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    trainer.args = SimpleNamespace(
        replay_subset_ratio=0.1,
        replay_distribution="equal_task",
        replay_subset_seed=replay_subset_seed,
        replay_selection_mode="random",
        seed=seed,
        global_rank=1,
        output_dir=(
            f"/tmp/v2_new_cpu_test_nonzero_rank_{seed}_"
            f"{replay_subset_seed}"),
        local_rank=-1,
        batch_by_task={name: 8 for name in task_names},
        max_prompt_len=32,
        max_ans_len=8,
        max_train_len=0,
        train_format="raw_answer",
        use_pretokenized_train_cache=False,
        router_replay_exposure_samples=1000,
        v2_new_persistent_samples_per_task=500,
        v2_new_active_memory_cap=1000,
        v2_memory_batch_size=0,
        v2_kd_memory_batch_size=8,
        v2_kd_pass_multiplier=1,
        v2_kd_loss_coeff=1.0,
        v2_joint_new_to_replay_ratio=5,
        gradient_accumulation_steps=1,
    )
    return trainer, task_names


def make_contract_args(training_version="v2_new", **overrides):
    values = {
        "training_version": training_version,
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
    }
    values.update(overrides)
    return resolve_training_version_defaults(SimpleNamespace(**values))


def expect_value_error(callable_):
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


EXPECTED_PREFIX_COUNTS = {
    1: [500],
    2: [500, 500],
    3: [334, 333, 333],
    4: [250, 250, 250, 250],
    5: [200, 200, 200, 200, 200],
    6: [167, 167, 167, 167, 166, 166],
    7: [143, 143, 143, 143, 143, 143, 142],
}


def test_version_dependent_defaults_preserve_legacy_behavior():
    v2_new = make_contract_args()
    assert v2_new.replay_subset_ratio == 0.1
    assert v2_new.v2_joint_new_to_replay_ratio == 5
    assert v2_new.routing_weight_mode == "straight_through_topk"
    assert v2_new.replay_subset_seed == 2025

    top4 = make_contract_args(
        "v2_new_top4", lora_moe_rank=16, lora_moe_alpha=128,
        experts_per_task=4, top_k=4, v2_kd_pass_multiplier=2)
    assert top4.replay_subset_ratio == v2_new.replay_subset_ratio == 0.1
    assert top4.v2_joint_new_to_replay_ratio == 5
    assert top4.routing_weight_mode == "straight_through_topk"
    assert top4.replay_subset_seed == 2025
    assert top4.v2_kd_pass_multiplier == 2

    v3_top4 = make_contract_args(
        "v3_new_top4", lora_moe_rank=16, lora_moe_alpha=128,
        experts_per_task=4, top_k=4, v2_kd_pass_multiplier=2)
    assert v3_top4.replay_subset_ratio == 0.1
    assert v3_top4.v2_joint_new_to_replay_ratio == 5
    assert v3_top4.routing_weight_mode == "straight_through_topk"
    assert v3_top4.replay_subset_seed == 2025
    assert v3_top4.v2_kd_pass_multiplier == 2

    for version in ("v1", "v2", "v2_5", "v3"):
        legacy = make_contract_args(version)
        assert legacy.replay_subset_ratio == 0.01
        assert legacy.v2_joint_new_to_replay_ratio == 0
        assert legacy.routing_weight_mode == "full_softmax"
        # Keep the historical sentinel in legacy metadata and commands.
        assert legacy.replay_subset_seed == -1

    explicit = make_contract_args(
        routing_weight_mode="full_softmax",
        replay_subset_ratio=0.1,
        v2_joint_new_to_replay_ratio=7,
        replay_subset_seed=99)
    assert explicit.routing_weight_mode == "full_softmax"
    assert explicit.v2_joint_new_to_replay_ratio == 7
    assert explicit.replay_subset_seed == 99


def test_v2_new_validation_enforces_the_exact_named_contract():
    valid = make_contract_args()
    validate_v2_new_args(valid)
    invalid_values = {
        "v2_new_persistent_samples_per_task": 499,
        "replay_subset_ratio": 0.01,
        "v2_new_active_memory_cap": 999,
        "replay_selection_mode": "router_gradient",
        "replay_distribution": "proportional",
        "v2_joint_new_to_replay_ratio": 0,
        "lora_moe_rank": 0,
        "experts_per_task": 0,
        "top_k": 0,
    }
    for field, value in invalid_values.items():
        candidate = SimpleNamespace(**vars(valid))
        setattr(candidate, field, value)
        expect_value_error(lambda candidate=candidate:
                           validate_v2_new_args(candidate))

    too_many_routes = SimpleNamespace(**vars(valid))
    too_many_routes.top_k = 2
    too_many_routes.experts_per_task = 1
    expect_value_error(lambda: validate_v2_new_args(too_many_routes))

    wrong_ratio = SimpleNamespace(**vars(valid))
    wrong_ratio.v2_joint_new_to_replay_ratio = 7
    expect_value_error(lambda: validate_v2_new_args(wrong_ratio))

    # The new exact contract is deliberately scoped to V2-new.  It cannot
    # reject an old V2 command/checkpoint with historical settings.
    legacy = SimpleNamespace(**vars(valid))
    legacy.training_version = "v2"
    legacy.v2_new_persistent_samples_per_task = 1
    legacy.replay_subset_ratio = 0.01
    legacy.top_k = 4
    legacy.experts_per_task = 1
    validate_v2_new_args(legacy)


def test_v2_new_top4_validation_is_an_isolated_exact_profile():
    for version in ("v2_new_top4", "v3_new_top4"):
        valid = make_contract_args(
            version, lora_moe_rank=16, lora_moe_alpha=128,
            experts_per_task=4, top_k=4, v2_kd_pass_multiplier=2)
        validate_v2_new_args(valid)
        for field, value in {
                "lora_moe_rank": 64,
                "lora_moe_alpha": 32,
                "experts_per_task": 1,
                "top_k": 1,
                "routing_weight_mode": "full_softmax",
                "v2_kd_pass_multiplier": 1,
        }.items():
            candidate = SimpleNamespace(**vars(valid))
            setattr(candidate, field, value)
            expect_value_error(lambda candidate=candidate:
                               validate_v2_new_args(candidate))

    # The original V2-new profile remains configurable and checkpoint-compatible.
    original = make_contract_args()
    validate_v2_new_args(original)


def test_v2_new_resume_persisted_identity_metadata_is_exact_and_well_formed():
    tasks = ["task0", "task1", "task2", "task3"]
    identities = {
        task: {
            "resolved_seed": 2025 + index * 1009,
            "indices_sha256": format(index + 1, "064x"),
        }
        for index, task in enumerate(tasks[:3])
    }
    meta = {
        "replay_memory": {"persisted_identities": dict(identities)},
        "v2_new": {
            # Deliberately use another insertion order: task-key identity is
            # exact, while JSON object order is not semantic.
            "persisted_identities": dict(reversed(list(identities.items()))),
        },
    }
    validated = validate_v2_new_resume_persisted_identities(
        meta, completed_round=2, task_names=tasks)
    assert list(validated) == tasks[:3]
    assert validated == identities

    missing = json.loads(json.dumps(meta))
    missing["v2_new"]["persisted_identities"].pop("task1")
    expect_value_error(lambda: validate_v2_new_resume_persisted_identities(
        missing, 2, tasks))

    extra = json.loads(json.dumps(meta))
    extra["replay_memory"]["persisted_identities"]["task3"] = {
        "resolved_seed": 9,
        "indices_sha256": "a" * 64,
    }
    expect_value_error(lambda: validate_v2_new_resume_persisted_identities(
        extra, 2, tasks))

    invalid_seed = json.loads(json.dumps(meta))
    invalid_seed["replay_memory"]["persisted_identities"]["task0"][
        "resolved_seed"] = True
    expect_value_error(lambda: validate_v2_new_resume_persisted_identities(
        invalid_seed, 2, tasks))

    invalid_hash = json.loads(json.dumps(meta))
    invalid_hash["replay_memory"]["persisted_identities"]["task0"][
        "indices_sha256"] = "A" * 64
    expect_value_error(lambda: validate_v2_new_resume_persisted_identities(
        invalid_hash, 2, tasks))

    divergent = json.loads(json.dumps(meta))
    divergent["v2_new"]["persisted_identities"]["task0"][
        "indices_sha256"] = "f" * 64
    expect_value_error(lambda: validate_v2_new_resume_persisted_identities(
        divergent, 2, tasks))


def test_v2_new_metadata_contract_uses_saved_identity_not_scalar_seed():
    args = make_contract_args()
    assert v2_new_replay_memory_contract(args) == {
        "persistent_samples_per_task": 500,
        "persistent_selection_mode": "random",
        "active_stream_samples_per_primary_epoch": 1000,
        "distribution": "equal_task",
        "v1_router_retune_enabled": False,
    }
    assert v2_new_metadata_contract(args) == {
        "persistent_subset_policy":
            "validated_output_dir_json_else_deterministic_random",
        "persistent_samples_per_task": 500,
        "persistent_memory_integrity":
            "source_count_unique_range_sha256",
        "persistent_memory_resume_source":
            "output_dir/fixed_replay_memory",
        "selection_mode": "random",
        "active_memory_cap_unique": 1000,
        "active_memory_distribution": "equal_task",
        "active_memory_selection": "stable_nested_task_prefix",
        "active_stream_samples_per_pass": 1000,
        "active_stream_identity_order": "shared_between_kd_and_replay",
        "active_stream_seed_phase": "v2_new_shared_active_memory",
        "kd_stream_passes": "match_primary_epochs",
        "joint_replay_stream_passes": "match_primary_epochs",
    }
    v2_contract = v2_new_v2_metadata_contract(args)
    assert v2_contract["joint_new_to_replay_sample_ratio"] == 5
    assert v2_contract["joint_replay_schedule"] == (
        "every_optimizer_update_active_stream_per_primary_epoch")
    assert v2_contract["kd_active_stream_samples_per_pass"] == 1000
    assert v2_contract["joint_replay_active_stream_samples_per_primary_epoch"] == 1000

    top4 = make_contract_args(
        "v2_new_top4", lora_moe_rank=16, lora_moe_alpha=128,
        experts_per_task=4, top_k=4, v2_kd_pass_multiplier=2)
    top4_v2_new = v2_new_metadata_contract(top4)
    assert top4_v2_new["kd_stream_passes"] == (
        "primary_epochs_times_multiplier")
    assert top4_v2_new["kd_stream_pass_multiplier"] == 2
    top4_v2 = v2_new_v2_metadata_contract(top4)
    assert top4_v2["kd_active_stream_passes"] == (
        "primary_epochs_times_multiplier")
    assert top4_v2["kd_pass_multiplier"] == 2
    assert top4_v2["joint_replay_active_stream_samples_per_primary_epoch"] == 1000
    missing_multiplier = dict(top4_v2)
    missing_multiplier.pop("kd_pass_multiplier")
    assert v2_resume_metadata_mismatches(
        top4, missing_multiplier, completed_round=1)[
            "v2.kd_pass_multiplier"] == (None, 2)

    changed_seed = make_contract_args(seed=2026)
    checkpoint_replay = v2_new_replay_memory_contract(args)
    assert metadata_contract_mismatches(
        v2_new_replay_memory_contract(changed_seed), checkpoint_replay,
        "replay_memory") == {}
    checkpoint_v2_new = v2_new_metadata_contract(args)
    assert metadata_contract_mismatches(
        v2_new_metadata_contract(changed_seed), checkpoint_v2_new,
        "v2_new") == {}

    # Once selected, the output-dir JSON owns identity.  Changing the command
    # seed loads the same indices; corrupting its hash fails closed.
    with tempfile.TemporaryDirectory() as temporary:
        first, _ = make_trainer(seed=2025)
        first.args.global_rank = 0
        first.args.output_dir = temporary
        first._ensure_fixed_task_subset("task0")
        expected_indices = first._fixed_task_subset_indices["task0"]
        saved_path = first._v2_new_memory_path("task0", 0)
        with open(saved_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        assert saved["resolved_seed"] == 2025
        assert saved["indices"] == expected_indices
        assert saved["indices_sha256"] == (
            first._v2_new_indices_sha256(expected_indices))

        resumed, _ = make_trainer(seed=2026)
        resumed.args.global_rank = 0
        resumed.args.output_dir = temporary
        resumed.args.resume_checkpoint = str(Path(temporary) / "0")
        resumed.args.start_task = 1
        resumed.args.v2_new_resume_persisted_identities = {
            "task0": {
                "resolved_seed": saved["resolved_seed"],
                "indices_sha256": saved["indices_sha256"],
            },
        }
        resumed._ensure_fixed_task_subset("task0")
        assert resumed._fixed_task_subset_indices["task0"] == expected_indices

        checkpoint_mismatch, _ = make_trainer(seed=2026)
        checkpoint_mismatch.args.global_rank = 0
        checkpoint_mismatch.args.output_dir = temporary
        checkpoint_mismatch.args.resume_checkpoint = str(
            Path(temporary) / "0")
        checkpoint_mismatch.args.start_task = 1
        checkpoint_mismatch.args.v2_new_resume_persisted_identities = {
            "task0": {
                "resolved_seed": saved["resolved_seed"],
                "indices_sha256": "f" * 64,
            },
        }
        expect_value_error(
            lambda: checkpoint_mismatch._ensure_fixed_task_subset("task0"))

        saved["indices_sha256"] = "0" * 64
        with open(saved_path, "w", encoding="utf-8") as handle:
            json.dump(saved, handle)
        corrupted, _ = make_trainer(seed=2026)
        corrupted.args.global_rank = 0
        corrupted.args.output_dir = temporary
        expect_value_error(
            lambda: corrupted._ensure_fixed_task_subset("task0"))


def test_v2_new_resume_is_strict_but_legacy_missing_fields_remain_loadable():
    v2_new = make_contract_args()
    exact = v2_new_v2_metadata_contract(v2_new)
    assert v2_resume_metadata_mismatches(v2_new, exact, 0) == {}
    missing_ratio = dict(exact)
    missing_ratio.pop("joint_new_to_replay_sample_ratio")
    assert v2_resume_metadata_mismatches(
        v2_new, missing_ratio, 0)[
            "v2.joint_new_to_replay_sample_ratio"] == (None, 5)

    for version in ("v2", "v2_5"):
        legacy = make_contract_args(version)
        old_checkpoint = {
            "memory_batch_size": legacy.v2_memory_batch_size,
            "kd_loss_coeff": legacy.v2_kd_loss_coeff,
            "kd_temperature": legacy.v2_kd_temperature,
            "kd_learning_rate": legacy.v2_kd_learning_rate,
            "kd_chunk_tokens": legacy.v2_kd_chunk_tokens,
            "kd_token_scope": legacy.v2_kd_token_scope,
            "joint_replay_loss_coeff": legacy.v2_joint_replay_loss_coeff,
            "max_replay_batches_per_step":
                legacy.v2_max_replay_batches_per_step,
        }
        # No ratio, schedule, reduction, or split KD batch metadata: this is
        # the pre-extension checkpoint shape and must still resume unchanged.
        assert v2_resume_metadata_mismatches(
            legacy, old_checkpoint, completed_round=6) == {}


def test_internal_memory_is_deterministic_500_per_task():
    first, task_names = make_trainer()
    second, _ = make_trainer()
    for task in task_names:
        assert len(first._ensure_fixed_task_subset(task)) == 500
        assert len(second._ensure_fixed_task_subset(task)) == 500
        first_indices = first._fixed_task_subset_indices[task]
        second_indices = second._fixed_task_subset_indices[task]
        assert first_indices == second_indices
        assert len(first_indices) == len(set(first_indices)) == 500


def test_active_memory_uses_nested_equal_task_prefixes():
    trainer, task_names = make_trainer()
    previous = {}
    for old_task_count, expected in EXPECTED_PREFIX_COUNTS.items():
        names, counts = trainer._v2_new_active_memory(old_task_count)
        assert names == task_names[:old_task_count]
        assert counts == expected, (old_task_count, counts)
        assert sum(counts) == min(1000, 500 * old_task_count)
        current = {
            name: tuple(trainer._fixed_task_subset_indices[name][:count])
            for name, count in zip(names, counts)
        }
        for name in set(previous).intersection(current):
            assert current[name] == previous[name][:len(current[name])]
        previous = current


def test_old1_repeats_500_twice_and_old2_uses_500_each():
    trainer, _ = make_trainer()
    old1 = trainer._build_v2_new_active_loader(1, role="replay")
    assert len(old1.dataset) == 1000
    assert len(old1.dataset.datasets) == 1
    old1_stream = old1.dataset.datasets[0]
    assert len(old1_stream.subset) == 500
    assert Counter(old1_stream.exposure_indices) == Counter(
        {index: 2 for index in range(500)})

    old2 = trainer._build_v2_new_active_loader(2, role="replay")
    assert len(old2.dataset) == 1000
    assert [len(stream.subset) for stream in old2.dataset.datasets] == [500, 500]
    assert [len(stream) for stream in old2.dataset.datasets] == [500, 500]
    for stream in old2.dataset.datasets:
        assert set(stream.exposure_indices) == set(range(500))


def test_kd_and_replay_actual_sampler_pass_order_is_identical():
    trainer, _ = make_trainer()
    kd_loader = trainer._build_v2_new_active_loader(3, role="kd")
    replay_loader = trainer._build_v2_new_active_loader(3, role="replay")
    for sampler_pass in range(2):
        trainer._set_v2_kd_memory_sampler_pass(kd_loader, sampler_pass)
        trainer._set_v2_replay_memory_sampler_pass(
            replay_loader, sampler_pass)
        kd_order = list(iter(kd_loader.sampler))
        replay_order = list(iter(replay_loader.sampler))
        assert kd_order == replay_order
        # Check the actual records addressed by the samplers, not only the
        # pre-sampler stream-plan digest.
        assert [kd_loader.dataset[index]["prompt"] for index in kd_order] == [
            replay_loader.dataset[index]["prompt"]
            for index in replay_order
        ]

    # Production uses DistributedSampler.  Exercise its rank-local flattened
    # seed+pass contract without initializing a process group.
    base_seed = kd_loader._lora_moe_memory_stream["sampler_base_seed"]
    for rank in (0, 3, 7):
        kd_sampler = DistributedSampler(
            kd_loader.dataset, num_replicas=8, rank=rank,
            shuffle=True, seed=base_seed)
        replay_sampler = DistributedSampler(
            replay_loader.dataset, num_replicas=8, rank=rank,
            shuffle=True, seed=base_seed)
        for sampler_pass in range(7):
            kd_sampler.set_epoch(sampler_pass)
            replay_sampler.set_epoch(sampler_pass)
            assert list(kd_sampler) == list(replay_sampler)


def test_kd_and_replay_repeat_one_active_stream_per_primary_epoch():
    trainer, _ = make_trainer()
    primary_loader = SimpleNamespace(
        sampler=SimpleNamespace(total_size=5000),
        dataset=range(5000),
    )
    kd_loader = trainer._build_v2_new_active_loader(3, role="kd")
    replay_loader = trainer._build_v2_new_active_loader(3, role="replay")
    assert len(kd_loader.dataset) == len(replay_loader.dataset) == 1000
    for kd_stream, replay_stream in zip(
            kd_loader.dataset.datasets, replay_loader.dataset.datasets):
        assert len(kd_stream.subset) == len(replay_stream.subset)
        assert set(kd_stream.exposure_indices) == set(
            replay_stream.exposure_indices)
    for epochs in (3, 5, 7):
        kd_exposures = len(kd_loader.dataset) * trainer._v2_kd_epochs(epochs)
        replay_exposures = trainer._joint_replay_exposure_budget(
            primary_loader, epochs)
        assert kd_exposures == replay_exposures == 1000 * epochs
        # With the production effective global batch 64, the deliberate epoch
        # boundary yields 16 KD updates per active-memory pass.
        assert math.ceil(1000 / 64) * epochs == {3: 48, 5: 80, 7: 112}[epochs]

    # Exercise the trainer's actual scheduler-update helper with the 8-GPU KD
    # loader length (ceil(1000 / 64) == 16 batches per stream pass).
    fake_eight_gpu_kd_loader = type(
        "EightGpuKdLoader", (), {"__len__": lambda self: 16})()
    assert trainer._optimizer_update_count(
        fake_eight_gpu_kd_loader, 3) == 48
    assert trainer._optimizer_update_count(
        fake_eight_gpu_kd_loader, 5) == 80
    assert trainer._optimizer_update_count(
        fake_eight_gpu_kd_loader, 7) == 112

    # The actual exposure assignment used by the joint loop must put replay
    # into every update and consume exactly one 1,000-record stream per epoch.
    for epochs, updates in ((3, 237), (5, 395), (7, 553)):
        budget = 1000 * epochs
        per_rank = [0] * 8
        assigned = 0
        for update in range(updates):
            row = [
                trainer._replay_exposure_assignment(
                    budget, updates, update, 8, rank)
                for rank in range(8)
            ]
            starts = {start for start, _, _ in row}
            stops = {stop for _, stop, _ in row}
            assert len(starts) == len(stops) == 1
            global_count = next(iter(stops)) - next(iter(starts))
            assert global_count >= 1
            assert sum(local for _, _, local in row) == global_count
            assigned += global_count
            for rank, (_, _, local) in enumerate(row):
                per_rank[rank] += local
        assert assigned == budget
        assert per_rank == [budget // 8] * 8


def test_top4_kd_repeats_the_same_stream_twice_without_changing_replay():
    trainer, _ = make_trainer()
    trainer.args.v2_kd_pass_multiplier = 2
    primary_loader = SimpleNamespace(
        sampler=SimpleNamespace(total_size=5000),
        dataset=range(5000),
    )
    kd_loader = trainer._build_v2_new_active_loader(1, role="kd")
    replay_loader = trainer._build_v2_new_active_loader(1, role="replay")
    assert len(kd_loader.dataset) == len(replay_loader.dataset) == 1000
    assert kd_loader._lora_moe_memory_stream["ordered_identity_sha256"] == (
        replay_loader._lora_moe_memory_stream["ordered_identity_sha256"])

    primary_epochs = 3
    assert trainer._v2_kd_epochs(primary_epochs) == 6
    assert len(kd_loader.dataset) * trainer._v2_kd_epochs(primary_epochs) == 6000
    # Joint replay remains exactly 1,000 exposures per primary epoch (5:1).
    assert trainer._joint_replay_exposure_budget(
        primary_loader, primary_epochs) == 3000

    for pass_index in range(6):
        trainer._set_v2_kd_memory_sampler_pass(kd_loader, pass_index)
    for pass_index in range(3):
        trainer._set_v2_replay_memory_sampler_pass(
            replay_loader, pass_index)
    trainer._validate_v2_replay_memory_sampler_passes(
        replay_loader, completed_passes=3, primary_epochs=3)


def test_existing_v2_keeps_single_fixed_1000_stream():
    trainer = object.__new__(Ours_LoRA_MoE_V2)
    trainer.args = SimpleNamespace(
        v2_joint_new_to_replay_ratio=5,
        router_replay_exposure_samples=1000,
    )
    primary_loader = SimpleNamespace(
        sampler=SimpleNamespace(total_size=5000), dataset=range(5000))
    assert trainer._v2_kd_epochs(7) == 1
    for epochs in (3, 5, 7):
        assert trainer._joint_replay_exposure_budget(
            primary_loader, epochs) == 1000


def test_top4_active_lora_parameter_count_matches_rank64_top1():
    rank64 = LoRAMoEMLP(
        TinyMLP(), r=64, alpha=128, top_k=1,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    rank64.add_experts(1)
    top4 = LoRAMoEMLP(
        TinyMLP(), r=16, alpha=128, top_k=4,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    top4.add_experts(4)
    rank64_lora_parameters = sum(
        parameter.numel() for expert in rank64.experts
        for parameter in expert.parameters())
    top4_lora_parameters = sum(
        parameter.numel() for expert in top4.experts
        for parameter in expert.parameters())
    assert top4_lora_parameters == rank64_lora_parameters
    assert top4.top_k * top4.r == rank64.top_k * rank64.r == 64
    assert top4.alpha / top4.r == 8
    assert rank64.alpha / rank64.r == 2
    assert (top4.alpha / top4.r) / top4.top_k == 2


def test_four_experts_top4_rank16_growth_and_kd_boundaries():
    torch.manual_seed(31)
    layer = LoRAMoEMLP(
        TinyMLP(), r=16, alpha=128, top_k=4,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    layer.add_experts(4)
    assert layer.num_experts == 4
    assert layer.top_k == 4
    for expert in layer.experts:
        assert expert.gate.A.shape[0] == 16
        assert expert.up.A.shape[0] == 16
        assert expert.down.A.shape[0] == 16
    x = torch.ones(2, 3, 32)
    for expert in layer.experts:
        for pair in expert.values():
            pair.B.data.normal_(mean=0.0, std=0.02)
    with torch.no_grad():
        layer.router.weight.fill_(-0.25)
        teacher_before_growth = layer(x).clone()
    old_expert_snapshots = [
        [parameter.detach().clone() for parameter in expert.parameters()]
        for expert in layer.experts
    ]

    output = layer(x)
    assert output.shape == (2, 3, 32)

    layer.add_experts(4)
    assert layer.num_experts == 8
    assert layer.top_k == 4
    model = nn.Module()
    model.mlp = layer
    new_indices = set(range(4, 8))
    freeze_lora_moe_experts(model, trainable_expert_indices=new_indices)
    freeze_lora_moe_routers(model, trainable=True)
    assert all(
        not parameter.requires_grad
        for expert in layer.experts[:4] for parameter in expert.parameters())
    assert all(
        parameter.requires_grad
        for expert in layer.experts[4:] for parameter in expert.parameters())

    # Force all four new rows into top-4. The prefix-limited teacher must still
    # be exactly the pre-growth model, while all four new experts receive KD.
    with torch.no_grad():
        layer.router.weight[:4].fill_(-0.25)
        layer.router.weight[4:].fill_(0.25)
        with limit_lora_moe_experts(model, 4):
            teacher_after_growth = layer(x).clone()
    torch.testing.assert_close(
        teacher_after_growth, teacher_before_growth, rtol=0, atol=0)

    old_router_rows = Ours_LoRA_MoE_V2._snapshot_old_router_rows(model, 4)
    new_router_before = layer.router.weight[4:].detach().clone()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad], lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    student = layer(x)
    loss = torch.nn.functional.mse_loss(student, teacher_after_growth)
    loss = loss + layer._last_moe_loss
    loss.backward()
    for expert in layer.experts[4:]:
        assert all(pair.B.grad is not None for pair in expert.values())
        assert any(torch.count_nonzero(pair.B.grad).item() > 0
                   for pair in expert.values())
    Ours_LoRA_MoE_V2._freeze_old_router_row_update(
        model, old_router_rows, 4)
    optimizer.step()
    Ours_LoRA_MoE_V2._freeze_old_router_row_update(
        model, old_router_rows, 4)
    torch.testing.assert_close(
        layer.router.weight[:4], old_router_rows[0], rtol=0, atol=0)
    assert not torch.equal(layer.router.weight[4:], new_router_before)
    for expert, snapshots in zip(layer.experts[:4], old_expert_snapshots):
        for parameter, snapshot in zip(expert.parameters(), snapshots):
            torch.testing.assert_close(parameter, snapshot, rtol=0, atol=0)
    with torch.no_grad(), limit_lora_moe_experts(model, 4):
        teacher_after_update = layer(x)
    torch.testing.assert_close(
        teacher_after_update, teacher_before_growth, rtol=0, atol=0)


def test_teacher_prefix_slices_router_weight_before_the_gemm():
    layer = LoRAMoEMLP(
        TinyMLP(), r=16, alpha=128, top_k=4,
        aux_loss_coeff=0.01, z_loss_coeff=0.001,
        routing_weight_mode="straight_through_topk", dropout=0.0)
    layer.add_experts(8)
    model = nn.Module()
    model.mlp = layer

    # A prefix-limited forward must use F.linear with router.weight[:prefix].
    # Calling the expanded nn.Linear and slicing its output is numerically
    # different on real BF16 CUDA kernels when out_features grows 4 -> 8.
    def expanded_router_forward_is_forbidden(*args, **kwargs):
        raise AssertionError("expanded router GEMM was called")

    layer.router.forward = expanded_router_forward_is_forbidden
    with torch.no_grad(), limit_lora_moe_experts(model, 4):
        output = layer(torch.randn(2, 3, 32))
    assert output.shape == (2, 3, 32)


def test_quota_top1_dispatch_is_exact_margin_ranked_and_padding_safe():
    logits = torch.tensor([
        [3.0, 0.0, 2.0],
        [3.0, 0.0, 1.0],
        [0.0, 3.0, 0.0],
        [3.0, 0.0, 100.0],
    ])
    natural = torch.tensor([[0], [0], [1], [2]])
    valid = torch.tensor([True, True, True, False])

    dispatch, injected, diagnostic = _quota_top1_dispatch(
        logits, natural, expert_index=2, quota_fraction=2 / 3,
        valid_token_mask=valid)
    assert set(injected.tolist()) == {0, 1}
    assert dispatch[:, 0].tolist() == [2, 2, 1, 2]
    assert diagnostic["valid_tokens"] == 3
    assert diagnostic["natural_selected_tokens"] == 0
    assert diagnostic["dispatched_selected_tokens"] == 2
    assert diagnostic["injected_tokens"] == 2

    natural_with_enough = natural.clone()
    natural_with_enough[0, 0] = 2
    unchanged, injected, diagnostic = _quota_top1_dispatch(
        logits, natural_with_enough, expert_index=2, quota_fraction=0.2,
        valid_token_mask=valid)
    torch.testing.assert_close(unchanged, natural_with_enough, rtol=0, atol=0)
    assert injected.numel() == 0
    assert diagnostic["injected_tokens"] == 0


def test_quota_zero_is_bitwise_and_branch_gradients_are_disjoint():
    torch.manual_seed(19)
    layer = LoRAMoEMLP(
        TinyMLP(hidden=4, intermediate=6), r=2, alpha=4, top_k=1,
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

        # Natural branch: experts are values in the forward but receive no grad.
        freeze_lora_moe_experts(model, trainable_expert_indices=None)
        freeze_lora_moe_routers(model, trainable=True)
        for parameter in model.parameters():
            parameter.grad = None
        natural_output = layer(x)
        (natural_output.square().mean() + layer._last_moe_loss).backward()
        assert layer.router.weight.grad is not None
        assert torch.count_nonzero(layer.router.weight.grad).item() > 0
        assert all(
            parameter.grad is None
            for expert in layer.experts for parameter in expert.parameters())

        # Quota branch: router and old experts are frozen; only expert 2 moves.
        freeze_lora_moe_experts(model, trainable_expert_indices={2})
        freeze_lora_moe_routers(model, trainable=False)
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
        new_grad_norm = sum(
            float(parameter.grad.square().sum().item())
            for parameter in layer.experts[2].parameters()
            if parameter.grad is not None)
        assert new_grad_norm > 0
        assert layer._last_quota_diagnostic["valid_tokens"] == 7
        assert layer._last_quota_diagnostic[
            "dispatched_selected_tokens"] == 7
    finally:
        set_router_token_mask(model, None)


def test_v2_new_checkpoint_uses_existing_ffn_eval_loader():
    from transformers import LlamaConfig, LlamaForCausalLM

    class TinyTokenizer:
        eos_token_id = 2

        def __len__(self):
            return 32

    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=2, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    with tempfile.TemporaryDirectory() as temporary:
        base_dir = Path(temporary) / "base"
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        source = LlamaForCausalLM(config).eval()
        source.save_pretrained(base_dir)
        attach_lora_moe(
            source, r=2, alpha=4, top_k=1,
            aux_loss_coeff=0.01, z_loss_coeff=0.001,
            routing_weight_mode="full_softmax", dropout=0.0)
        add_experts_to_all_layers(source, 2)
        for name, parameter in source.named_parameters():
            if ".mlp.experts." in name or ".mlp.router." in name:
                parameter.data.normal_(mean=0.0, std=0.05)
        partial = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
            if ".mlp.experts." in key or ".mlp.router." in key
        }
        torch.save(partial, checkpoint_dir / "pytorch_model.bin")
        save_lora_moe_meta(
            source, checkpoint_dir,
            extra={"training_version": "v2_new", "experts_per_task": 1})

        loaded, meta = load_lora_moe_checkpoint(
            checkpoint_dir, TinyTokenizer(), str(base_dir),
            device="cpu", dtype=torch.float32)
        assert meta["training_version"] == "v2_new"
        assert "architecture" not in meta  # evaluator selects the FFN loader
        for key, expected in partial.items():
            torch.testing.assert_close(loaded.state_dict()[key], expected)

        input_ids = torch.tensor([[1, 3, 5, 7]])
        attention_mask = torch.ones_like(input_ids)
        set_router_token_mask(source, attention_mask)
        set_router_token_mask(loaded, attention_mask)
        try:
            with torch.no_grad():
                expected = source(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False).logits
                actual = loaded(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False).logits
        finally:
            set_router_token_mask(source, None)
            set_router_token_mask(loaded, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v2_new_top4_checkpoint_roundtrip_records_exact_profile():
    from transformers import LlamaConfig, LlamaForCausalLM

    class TinyTokenizer:
        eos_token_id = 2

        def __len__(self):
            return 32

    assert [(round_index + 1) * 4 for round_index in range(8)] == [
        4, 8, 12, 16, 20, 24, 28, 32]
    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=64,
        pad_token_id=2, bos_token_id=1, eos_token_id=2,
        attention_dropout=0.0)
    with tempfile.TemporaryDirectory() as temporary:
        base_dir = Path(temporary) / "base"
        checkpoint_dir = Path(temporary) / "checkpoint"
        checkpoint_dir.mkdir()
        source = LlamaForCausalLM(config).eval()
        source.save_pretrained(base_dir)
        attach_lora_moe(
            source, r=16, alpha=128, top_k=4,
            aux_loss_coeff=0.01, z_loss_coeff=0.001,
            routing_weight_mode="straight_through_topk", dropout=0.0)
        add_experts_to_all_layers(source, 8)
        for name, parameter in source.named_parameters():
            if ".mlp.experts." in name or ".mlp.router." in name:
                parameter.data.normal_(mean=0.0, std=0.02)
        partial = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
            if ".mlp.experts." in key or ".mlp.router." in key
        }
        torch.save(partial, checkpoint_dir / "pytorch_model.bin")
        save_lora_moe_meta(
            source, checkpoint_dir,
            extra={
                "training_version": "v2_new_top4",
                "experts_per_task": 4,
            })

        loaded, meta = load_lora_moe_checkpoint(
            checkpoint_dir, TinyTokenizer(), str(base_dir),
            device="cpu", dtype=torch.float32)
        assert meta["training_version"] == "v2_new_top4"
        assert meta["experts_per_task"] == 4
        assert meta["r"] == 16
        assert meta["alpha"] == 128
        assert meta["top_k"] == 4
        assert meta["routing_weight_mode"] == "straight_through_topk"
        assert meta["num_experts"] == 8
        for key, expected in partial.items():
            torch.testing.assert_close(loaded.state_dict()[key], expected)

        input_ids = torch.tensor([[1, 3, 5, 7]])
        attention_mask = torch.ones_like(input_ids)
        set_router_token_mask(source, attention_mask)
        set_router_token_mask(loaded, attention_mask)
        try:
            with torch.no_grad():
                expected = source(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False).logits
                actual = loaded(
                    input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=False).logits
        finally:
            set_router_token_mask(source, None)
            set_router_token_mask(loaded, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main():
    tests = [
        test_version_dependent_defaults_preserve_legacy_behavior,
        test_v2_new_validation_enforces_the_exact_named_contract,
        test_v2_new_top4_validation_is_an_isolated_exact_profile,
        test_v2_new_resume_persisted_identity_metadata_is_exact_and_well_formed,
        test_v2_new_metadata_contract_uses_saved_identity_not_scalar_seed,
        test_v2_new_resume_is_strict_but_legacy_missing_fields_remain_loadable,
        test_internal_memory_is_deterministic_500_per_task,
        test_active_memory_uses_nested_equal_task_prefixes,
        test_old1_repeats_500_twice_and_old2_uses_500_each,
        test_kd_and_replay_actual_sampler_pass_order_is_identical,
        test_kd_and_replay_repeat_one_active_stream_per_primary_epoch,
        test_top4_kd_repeats_the_same_stream_twice_without_changing_replay,
        test_existing_v2_keeps_single_fixed_1000_stream,
        test_top4_active_lora_parameter_count_matches_rank64_top1,
        test_four_experts_top4_rank16_growth_and_kd_boundaries,
        test_teacher_prefix_slices_router_weight_before_the_gemm,
        test_quota_top1_dispatch_is_exact_margin_ranked_and_padding_safe,
        test_quota_zero_is_bitwise_and_branch_gradients_are_disjoint,
        test_v2_new_checkpoint_uses_existing_ffn_eval_loader,
        test_v2_new_top4_checkpoint_roundtrip_records_exact_profile,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
