from types import SimpleNamespace

import pytest
import torch

from megatron.training.training import (
    _MoeJointReplayDataIterator,
    _moe_joint_replay_batches_before_step,
    _moe_joint_replay_old_like_objective,
    _moe_joint_replay_old_like_budget,
    _moe_joint_replay_batches_for_step,
    _validate_joint_replay_old_data_kd,
)
from pretrain_gpt import (
    _apply_old_like_gt_objective_mask,
    _masked_layer_hidden_mse,
)


def _args(*, replay=False, hidden_mse=False, hidden_kl=False, vocab_kl=False, per_token=False):
    return SimpleNamespace(
        moe_joint_replay_old_like_gt_path="/gt",
        moe_joint_replay_lm=True,
        moe_joint_replay_old_data_kd=vocab_kl,
        moe_joint_replay_old_data_hidden_kl=hidden_kl,
        moe_joint_replay_old_data_hidden_mse=hidden_mse,
        moe_old_model_kl_coeff=1.0 if vocab_kl else 0.0,
        moe_old_hidden_kl_coeff=1.0,
        moe_old_hidden_mse_coeff=1.0,
        moe_old_model_kl_load="/teacher" if (hidden_mse or hidden_kl or vocab_kl) else None,
        moe_old_hidden_kl_layers="2,3,4,5,6,7,8,9",
        moe_old_hidden_mse_layers="2,3,4,5,6,7,8,9",
        moe_expansion_distill_lm_loss_coeff=1.0,
        num_layers=9,
        calculate_per_token_loss=per_token,
        _moe_joint_replay_active=replay,
    )


def _batch():
    return {
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "old_like_mask": torch.tensor([[False, True, True, False]]),
        "old_like_sample_id": torch.tensor([7]),
    }


def test_primary_uses_gt_complement_and_replay_uses_gt_positions():
    primary = _batch()
    _apply_old_like_gt_objective_mask(primary, _args(replay=False))
    assert torch.equal(primary["loss_mask"], torch.tensor([[1.0, 0.0, 0.0, 1.0]]))

    replay = _batch()
    replay_args = _args(replay=True)
    _apply_old_like_gt_objective_mask(replay, replay_args)
    assert torch.equal(replay["loss_mask"], torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    stats = replay_args._moe_joint_replay_old_like_batch_stats
    assert stats["base_valid_tokens"].item() == 3
    assert stats["gt_positive_tokens"].item() == 1
    assert stats["selected_tokens"].item() == 1


def test_old_like_config_accepts_default_per_microbatch_normalization():
    args = _args(per_token=False)
    _validate_joint_replay_old_data_kd(args)
    assert _moe_joint_replay_old_like_objective(args) == "lm"


def test_old_like_config_rejects_global_per_token_mode_for_two_finalize_passes():
    with pytest.raises(ValueError, match="does not support --calculate-per-token-loss"):
        _validate_joint_replay_old_data_kd(_args(per_token=True))


def test_old_like_hidden_mse_requires_layers_two_through_final_layer():
    args = _args(hidden_mse=True)
    _validate_joint_replay_old_data_kd(args)
    assert _moe_joint_replay_old_like_objective(args) == "layer_output_hidden_mse"

    args.moe_old_hidden_mse_layers = "2,3,4,5,6,7,8"
    with pytest.raises(ValueError, match="final layer 9"):
        _validate_joint_replay_old_data_kd(args)


def test_old_like_accepts_hidden_kl_and_vocab_kl_objectives():
    hidden_kl = _args(hidden_kl=True)
    _validate_joint_replay_old_data_kd(hidden_kl)
    assert _moe_joint_replay_old_like_objective(hidden_kl) == "layer_output_hidden_kl"

    vocab_kl = _args(vocab_kl=True)
    _validate_joint_replay_old_data_kd(vocab_kl)
    assert _moe_joint_replay_old_like_objective(vocab_kl) == "output_vocab_kl"


def test_twenty_percent_budget_is_deterministic_and_resume_addressable():
    args = _args()
    args.train_iters = 1800
    args.global_batch_size = 2304
    args.moe_joint_replay_old_like_unit = "positive_sequence"
    args.moe_joint_replay_old_like_target_train_fraction = 0.2
    args.moe_joint_replay_old_like_selected_token_count = 2_658_787
    args.moe_joint_replay_old_like_positive_sample_count = 798_138
    args.moe_joint_replay_old_like_full_train_token_count = 2_123_366_400
    budget = _moe_joint_replay_old_like_budget(args)
    per_step = [_moe_joint_replay_batches_for_step(args, i) for i in range(1, 1801)]
    assert budget["total_batches"] == 55_331
    assert sum(per_step) == 55_331
    assert set(per_step) == {30, 31}
    assert per_step.count(31) == 1_331
    assert budget["expected_fraction"] == pytest.approx(0.2, abs=1e-5)


def test_packed_miniset_explicit_twenty_percent_batch_budget(monkeypatch):
    monkeypatch.setattr(
        "megatron.training.training.mpu.get_data_parallel_world_size", lambda: 4
    )
    args = _args()
    args.train_iters = 1800
    args.global_batch_size = 2304
    args.micro_batch_size = 48
    args.moe_joint_replay_old_like_gt_path = None
    args.moe_joint_replay_total_samples = 829_440
    args.moe_joint_replay_micro_batch_size = 24
    budget = _moe_joint_replay_old_like_budget(args)
    per_step = [_moe_joint_replay_batches_for_step(args, i) for i in range(1, 1801)]
    assert budget["total_microbatches"] == 8_640
    assert budget["total_samples"] == 829_440
    assert budget["expected_fraction"] == pytest.approx(0.2)
    assert sum(per_step) == 8_640
    assert set(per_step) == {4, 5}
    assert per_step.count(4) == 360
    assert per_step.count(5) == 1_440
    assert all(
        _moe_joint_replay_batches_before_step(args, step)
        == sum(per_step[:step])
        for step in (0, 1, 4, 5, 899, 900, 1799, 1800)
    )


def test_packed_miniset_eight_gpu_mb96_replay_schedule(monkeypatch):
    monkeypatch.setattr(
        "megatron.training.training.mpu.get_data_parallel_world_size", lambda: 8
    )
    args = _args()
    args.train_iters = 1800
    args.global_batch_size = 2304
    args.micro_batch_size = 96
    args.moe_joint_replay_old_like_gt_path = None
    args.moe_joint_replay_total_samples = 829_440
    args.moe_joint_replay_micro_batch_size = 48
    budget = _moe_joint_replay_old_like_budget(args)
    per_step = [_moe_joint_replay_batches_for_step(args, i) for i in range(1, 1801)]
    assert budget["total_microbatches"] == 2_160
    assert budget["replay_global_micro_batch_size"] == 384
    assert sum(per_step) == 2_160
    assert set(per_step) == {1, 2}
    assert per_step.count(1) == 1_440
    assert per_step.count(2) == 360
    assert sum(count * 384 for count in per_step) == 829_440


def test_paired_identity_checks_ids_tokens_labels_and_gt_mask():
    batch = {
        "old_like_sample_id": torch.tensor([3, 4]),
        "old_like_mask": torch.tensor([[False, True], [True, False]]),
        "tokens": torch.tensor([[10, 11], [12, 13]]),
        "labels": torch.tensor([[11, 12], [13, 14]]),
    }
    primary = _MoeJointReplayDataIterator._old_like_batch_identity(batch, "primary")
    replay = _MoeJointReplayDataIterator._old_like_batch_identity(batch, "replay")
    _MoeJointReplayDataIterator._validate_old_like_pair(primary, replay)

    replay["tokens"][1, 0] = 999
    with pytest.raises(RuntimeError, match="identity mismatch for tokens"):
        _MoeJointReplayDataIterator._validate_old_like_pair(primary, replay)


def test_independent_replay_subset_does_not_require_primary_batch_identity():
    primary_batch = {
        "old_like_sample_id": torch.tensor([10]),
        "old_like_mask": torch.tensor([[False, True]]),
        "tokens": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[2, 3]]),
    }
    replay_batch = {
        "old_like_sample_id": torch.tensor([999]),
        "old_like_mask": torch.tensor([[True, False]]),
        "tokens": torch.tensor([[8, 9]]),
        "labels": torch.tensor([[9, 10]]),
    }
    iterator = _MoeJointReplayDataIterator(
        iter([primary_batch]), iter([replay_batch]), old_like_pairing=False
    )
    assert next(iterator)["old_like_sample_id"].item() == 10
    assert iterator.next_replay()["old_like_sample_id"].item() == 999
    iterator.finish_old_like_pairing()


def test_hidden_mse_is_selected_token_sum_averaged_across_layers_including_last():
    labels = torch.zeros((1, 3), dtype=torch.long)
    mask = torch.tensor([[0.0, 1.0, 0.0]])
    teacher = {
        2: torch.zeros((3, 1, 2)),
        9: torch.zeros((3, 1, 2)),
    }
    student = {
        2: torch.tensor([[[0.0, 0.0]], [[1.0, 1.0]], [[0.0, 0.0]]]),
        9: torch.tensor([[[0.0, 0.0]], [[3.0, 3.0]], [[0.0, 0.0]]]),
    }
    # Selected-token MSE is 1 at L2 and 9 at L9; layer mean is 5.
    assert _masked_layer_hidden_mse(student, teacher, labels, mask).item() == 5.0
