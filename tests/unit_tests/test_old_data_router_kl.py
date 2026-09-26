"""Old-data per-layer router KL as the router-only replay objective
(--moe-joint-replay-old-data-router-kl)."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.training.training import (
    _joint_replay_objective_coefficient,
    _moe_joint_replay_old_like_objective,
    _old_moe_distill_teacher_requested,
    _validate_joint_replay_old_data_kd,
)
from pretrain_gpt import (
    _effective_lm_loss_coeff,
    _masked_router_prob_kl,
    _old_data_router_kl_enabled_for_current_branch,
    _teacher_kd_enabled_for_current_branch,
)


def _args(**kw):
    base = dict(
        moe_joint_replay_old_data_kd=False,
        moe_joint_replay_old_data_hidden_kl=False,
        moe_joint_replay_old_data_hidden_mse=False,
        moe_joint_replay_old_data_router_kl=True,
        moe_joint_replay_lm=True,
        moe_old_model_kl_coeff=0.0,
        moe_old_hidden_kl_coeff=1.0,
        moe_old_hidden_mse_coeff=1.0,
        moe_old_router_kl_coeff=1.0,
        moe_old_model_kl_load="/teacher",
        pipeline_model_parallel_size=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_router_kl_configuration_is_valid_and_needs_the_teacher():
    args = _args()
    _validate_joint_replay_old_data_kd(args)
    assert _old_moe_distill_teacher_requested(args)
    assert _joint_replay_objective_coefficient(args) == 1.0


@pytest.mark.parametrize("other", ["moe_joint_replay_old_data_kd",
                                   "moe_joint_replay_old_data_hidden_kl",
                                   "moe_joint_replay_old_data_hidden_mse"])
def test_router_kl_is_mutually_exclusive(other):
    args = _args(**{other: True, "moe_old_model_kl_coeff": 1.0})
    with pytest.raises(ValueError, match="exactly one"):
        _validate_joint_replay_old_data_kd(args)


def test_router_kl_rejects_nonpositive_coeff_missing_teacher_and_pp():
    with pytest.raises(ValueError, match="old-router-kl-coeff"):
        _validate_joint_replay_old_data_kd(_args(moe_old_router_kl_coeff=0.0))
    with pytest.raises(ValueError, match="kl-load"):
        _validate_joint_replay_old_data_kd(_args(moe_old_model_kl_load=None))
    with pytest.raises(ValueError, match="pipeline"):
        _validate_joint_replay_old_data_kd(_args(pipeline_model_parallel_size=2))


def test_router_kl_is_pure_and_only_on_the_replay_branch():
    args = _args(moe_expansion_distill_lm_loss_coeff=1.0)
    teacher = object()
    args._moe_joint_replay_old_data_kd_active = False          # primary pass
    assert not _old_data_router_kl_enabled_for_current_branch(args)
    assert not _teacher_kd_enabled_for_current_branch(args, teacher, False)
    assert _effective_lm_loss_coeff(args) == 1.0
    args._moe_joint_replay_old_data_kd_active = True           # replay pass: KL only, no LM
    assert _old_data_router_kl_enabled_for_current_branch(args)
    assert _teacher_kd_enabled_for_current_branch(args, teacher, False)
    assert _effective_lm_loss_coeff(args) == 0.0


def test_old_like_objective_name():
    args = _args(moe_joint_replay_old_like_gt_path="/gt")
    assert _moe_joint_replay_old_like_objective(args) == "router_prob_kl"


class _Router(torch.nn.Module):
    def __init__(self, experts, hidden, gen):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden, generator=gen))

    def gating(self, x):
        return F.linear(x, self.weight)


def test_masked_router_prob_kl_matches_manual_zero_padded_per_layer_kl():
    gen = torch.Generator().manual_seed(0)
    S, B, H, old, new = 5, 2, 8, 3, 6
    labels = torch.zeros(B, S, dtype=torch.long)
    loss_mask = torch.ones(B, S)
    loss_mask[0, 3:] = 0                                          # two masked tokens
    layers = (1, 2)
    s_in = {l: torch.randn(S, B, H, generator=gen, requires_grad=True) for l in layers}
    t_in = {l: torch.randn(S, B, H, generator=gen) for l in layers}
    s_r = {l: _Router(new, H, gen) for l in layers}
    t_r = {l: _Router(old, H, gen) for l in layers}
    got = _masked_router_prob_kl(s_in, t_in, s_r, t_r, labels, loss_mask)

    keep = loss_mask.reshape(-1).bool()
    per_layer = []
    for l in layers:
        s = s_in[l].permute(1, 0, 2).reshape(-1, H)[keep]
        t = t_in[l].permute(1, 0, 2).reshape(-1, H)[keep]
        p_t = torch.softmax(t_r[l].gating(t), -1)
        p_t = torch.cat([p_t, torch.zeros(p_t.shape[0], new - old)], -1)   # zero-padded teacher
        log_s = torch.log_softmax(s_r[l].gating(s), -1)
        kl_tok = (p_t * (torch.log(p_t.clamp_min(1e-30)) - log_s)).sum(-1)
        per_layer.append(kl_tok.sum())                               # token-sum numerator
    torch.testing.assert_close(got, torch.stack(per_layer).mean(), rtol=1e-5, atol=1e-6)

    got.backward()
    assert all(s_r[l].weight.grad is not None and s_r[l].weight.grad.abs().sum() > 0 for l in layers)
    assert all(t_r[l].weight.grad is None for l in layers)            # teacher frozen


# ---------------------------------------------------- per-sample split: new task LM, old task KD
from pretrain_gpt import _replay_objective_token_masks  # noqa: E402


def test_split_masks_partition_the_loss_mask_by_sample():
    loss_mask = torch.tensor([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])
    ids = torch.tensor([0, 1, 1])                                      # wiki, code, code
    lm, kd = _replay_objective_token_masks(loss_mask, ids, current_task_dataset_id=1)
    torch.testing.assert_close(lm, torch.tensor([[0., 0., 0.], [1., 1., 1.], [0., 1., 1.]]))
    torch.testing.assert_close(kd, torch.tensor([[1., 1., 0.], [0., 0., 0.], [0., 0., 0.]]))
    torch.testing.assert_close(lm + kd, loss_mask)


def test_split_masks_require_dataset_ids():
    with pytest.raises(RuntimeError, match="dataset_id"):
        _replay_objective_token_masks(torch.ones(2, 3), None, 1)
    with pytest.raises(RuntimeError, match="shape"):
        _replay_objective_token_masks(torch.ones(2, 3), torch.tensor([0, 1, 1]), 1)


def test_router_kl_over_old_samples_only():
    gen = torch.Generator().manual_seed(1)
    S, B, H, old, new = 4, 3, 8, 3, 6
    labels = torch.zeros(B, S, dtype=torch.long)
    loss_mask = torch.ones(B, S)
    ids = torch.tensor([0, 1, 0])
    _, kd = _replay_objective_token_masks(loss_mask, ids, 1)
    s_in = {1: torch.randn(S, B, H, generator=gen)}
    t_in = {1: torch.randn(S, B, H, generator=gen)}
    s_r, t_r = {1: _Router(new, H, gen)}, {1: _Router(old, H, gen)}
    split = _masked_router_prob_kl(s_in, t_in, s_r, t_r, labels, kd)
    # same as the KL over a batch holding only the two old-task samples
    keep = torch.tensor([0, 2])
    only_old = _masked_router_prob_kl({1: s_in[1][:, keep]}, {1: t_in[1][:, keep]}, s_r, t_r,
                                      labels[keep], loss_mask[keep])
    torch.testing.assert_close(split, only_old, rtol=1e-5, atol=1e-6)


def test_split_needs_exactly_one_old_data_objective():
    args = _args(moe_joint_replay_current_task_dataset_id=1)
    _validate_joint_replay_old_data_kd(args)
    args = _args(moe_joint_replay_old_data_router_kl=False, moe_joint_replay_current_task_dataset_id=1)
    with pytest.raises(ValueError, match="current-task-dataset-id"):
        _validate_joint_replay_old_data_kd(args)
    args = _args(moe_joint_replay_current_task_dataset_id=0)          # id 0 is a valid index
    _validate_joint_replay_old_data_kd(args)


def test_posthoc_objective_on_primary_needs_no_joint_replay_and_excludes_it():
    args = _args(moe_joint_replay_lm=False, moe_old_data_objective_on_primary=True,
                 moe_joint_replay_current_task_dataset_id=1)
    _validate_joint_replay_old_data_kd(args)                 # post-hoc retune: no joint replay
    args = _args(moe_joint_replay_lm=False)                  # neither joint replay nor post-hoc
    with pytest.raises(ValueError, match="joint-replay-lm"):
        _validate_joint_replay_old_data_kd(args)
    args = _args(moe_old_data_objective_on_primary=True)     # both at once
    with pytest.raises(ValueError, match="post-hoc"):
        _validate_joint_replay_old_data_kd(args)


def test_masked_new_experts_kl_over_old_experts_only_and_no_grad_on_new_rows():
    gen = torch.Generator().manual_seed(3)
    S, B, H, old, new = 4, 2, 8, 3, 6
    labels = torch.zeros(B, S, dtype=torch.long)
    loss_mask = torch.ones(B, S)
    s_in = {1: torch.randn(S, B, H, generator=gen)}
    t_in = {1: torch.randn(S, B, H, generator=gen)}
    s_r, t_r = {1: _Router(new, H, gen)}, {1: _Router(old, H, gen)}
    got = _masked_router_prob_kl(s_in, t_in, s_r, t_r, labels, loss_mask, existing_experts_only=True)
    s = s_in[1].permute(1, 0, 2).reshape(-1, H)
    t = t_in[1].permute(1, 0, 2).reshape(-1, H)
    p_t = torch.softmax(t_r[1].gating(t), -1)
    log_s_old = torch.log_softmax(s_r[1].gating(s)[:, :old], -1)       # new experts masked, re-normalised
    expected = (p_t * (torch.log(p_t) - log_s_old)).sum(-1).sum()
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    got.backward()
    g = s_r[1].weight.grad
    assert g[:old].abs().sum() > 0 and torch.all(g[old:] == 0)          # expanded rows: no KD gradient


def test_default_new_expert_handling_is_zero_pad():
    import megatron.training.arguments as A
    import inspect
    assert ("'--moe-old-router-kl-new-experts', choices=['zero_pad', 'mask'], default='zero_pad'"
            in inspect.getsource(A))
