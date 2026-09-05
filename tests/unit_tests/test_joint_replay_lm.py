import torch
from types import SimpleNamespace
import pytest
from megatron.training.training import (
    _blend_joint_replay_non_router_grads,
    _joint_replay_router_grad_metrics,
    _moe_joint_new_expert_quota,
    _old_moe_distill_teacher_requested,
    _restore_joint_replay_non_router_grads,
    _restore_joint_replay_router_grads,
    _snapshot_joint_replay_non_router_grads,
    _validate_joint_replay_old_data_kd,
    _zero_joint_replay_non_router_grads,
)
from megatron.core.transformer.moe.router import _apply_training_new_expert_quota
from pretrain_gpt import (
    _effective_old_hidden_kl_coeff,
    _effective_lm_loss_coeff,
    _masked_layer_hidden_mse,
    _masked_layer_hidden_kl,
    _old_data_hidden_mse_enabled_for_current_branch,
    _teacher_kd_enabled_for_current_branch,
    _token_mean_to_loss_numerator,
)


def _kd_args(enabled=True, joint=True, coeff=1.0, load="/teacher"):
    return SimpleNamespace(
        moe_joint_replay_old_data_kd=enabled,
        moe_joint_replay_old_data_hidden_kl=False,
        moe_joint_replay_old_data_hidden_mse=False,
        moe_joint_replay_lm=joint,
        moe_old_model_kl_coeff=coeff,
        moe_old_hidden_kl_coeff=1.0,
        moe_old_hidden_mse_coeff=1.0,
        moe_old_model_kl_load=load,
    )


def test_old_data_kd_replay_accepts_complete_configuration():
    _validate_joint_replay_old_data_kd(_kd_args())


def test_old_data_kd_is_disabled_on_primary_and_pure_kd_on_replay():
    args = _kd_args()
    args.moe_expansion_distill_lm_loss_coeff = 1.0
    args._moe_joint_replay_old_data_kd_active = False
    teacher = object()
    assert _effective_lm_loss_coeff(args) == 1.0
    assert not _teacher_kd_enabled_for_current_branch(args, teacher, False)

    args._moe_joint_replay_old_data_kd_active = True
    assert _effective_lm_loss_coeff(args) == 0.0
    assert _teacher_kd_enabled_for_current_branch(args, teacher, False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"joint": False}, "joint-replay-lm"),
        ({"coeff": 0.0}, "kl-coeff"),
        ({"load": None}, "kl-load"),
    ],
)
def test_old_data_kd_replay_rejects_incomplete_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _validate_joint_replay_old_data_kd(_kd_args(**kwargs))

def test_primary_expert_grad_is_kept_and_router_grad_accumulates():
    expert=torch.nn.Parameter(torch.zeros(2));router=torch.nn.Parameter(torch.zeros(2))
    expert.main_grad=torch.tensor([1.,2.]);router.main_grad=torch.tensor([3.,4.])
    snapshot=_snapshot_joint_replay_non_router_grads([expert,router],{id(router)})
    expert.main_grad.add_(torch.tensor([10.,20.]));router.main_grad.add_(torch.tensor([30.,40.]))
    _restore_joint_replay_non_router_grads(snapshot)
    assert torch.equal(expert.main_grad,torch.tensor([1.,2.]))
    assert torch.equal(router.main_grad,torch.tensor([33.,44.]))

def test_replay_only_expert_grad_is_zeroed():
    expert=torch.nn.Parameter(torch.zeros(2));expert.main_grad=None
    snapshot=_snapshot_joint_replay_non_router_grads([expert],set())
    expert.main_grad=torch.tensor([7.,8.]);_restore_joint_replay_non_router_grads(snapshot)
    assert torch.equal(expert.main_grad,torch.zeros(2))

def test_frozen_parameter_is_ignored():
    frozen=torch.nn.Parameter(torch.zeros(1),requires_grad=False);frozen.main_grad=torch.ones(1)
    assert _snapshot_joint_replay_non_router_grads([frozen],set())=={}


def test_quota_dispatch_injects_exact_minimum_without_changing_topk():
    logits = torch.tensor(
        [
            [4.0, 3.0, 2.0],
            [4.0, 3.0, 2.5],
            [4.0, 3.0, 5.0],
            [4.0, 3.0, 1.0],
        ]
    )
    routing_map = torch.tensor(
        [[True, False, False], [True, False, False], [False, False, True], [True, False, False]]
    )
    scores = routing_map.float()
    quota_scores, quota_map, stats = _apply_training_new_expert_quota(
        logits, scores, routing_map, boundary=2, quota=0.5
    )
    assert quota_map[:, 2].sum().item() == 2
    assert torch.equal(quota_map.sum(dim=-1), routing_map.sum(dim=-1))
    # The highest-margin candidate is token 1 (new logit 2.5).
    assert quota_map[1, 2]
    assert torch.allclose(quota_scores.sum(dim=-1), torch.ones(4))
    assert stats == (4, 1, 2, 1)


def test_quota_dispatch_uses_topk_inside_new_group_without_per_expert_priority():
    # Experts 0:2 are old; 2:5 are the newly added task group.  The injected
    # token must use two members of the new group and no old expert.
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0],
            [6.0, 5.0, 3.0, 7.0, 4.0],
        ]
    )
    routing_map = torch.tensor(
        [
            [True, True, False, False, False],
            [True, False, False, True, False],
        ]
    )
    scores = torch.softmax(logits.masked_fill(~routing_map, -torch.inf), dim=-1)
    quota_scores, quota_map, stats = _apply_training_new_expert_quota(
        logits, scores, routing_map, boundary=2, quota=0.5
    )

    fully_new = ~quota_map[:, :2].any(dim=-1)
    assert fully_new.sum().item() == 1
    chosen = torch.nonzero(fully_new, as_tuple=False).item()
    expected_new_top2 = torch.topk(logits[chosen, 2:], k=2).indices + 2
    actual = torch.nonzero(quota_map[chosen], as_tuple=False).flatten()
    assert torch.equal(actual.sort().values, expected_new_top2.sort().values)
    assert quota_map[chosen, :2].sum().item() == 0
    assert torch.allclose(quota_scores.sum(dim=-1), torch.ones(2))
    assert stats == (2, 0, 1, 1)


def test_quota_dispatch_can_replace_only_one_old_slot():
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0],
            [6.0, 5.0, 4.5, 3.0, 2.0],
            [6.0, 2.0, 5.0, 1.0, 0.0],
            [6.0, 5.0, 1.0, 0.0, -1.0],
        ]
    )
    routing_map = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, True, False, False, False],
        ]
    )
    scores = torch.softmax(logits.masked_fill(~routing_map, -torch.inf), dim=-1)
    quota_scores, quota_map, stats = _apply_training_new_expert_quota(
        logits,
        scores,
        routing_map,
        boundary=2,
        quota=0.5,
        min_new_slots=1,
    )

    has_new = quota_map[:, 2:].any(dim=-1)
    assert has_new.sum().item() == 2
    # Token 1 is the most competitive zero-new candidate, so only its weakest
    # old route is replaced by its best new-group route.
    assert quota_map[1, 0]
    assert not quota_map[1, 1]
    assert quota_map[1, 2]
    assert torch.equal(quota_map.sum(dim=-1), routing_map.sum(dim=-1))
    assert torch.allclose(quota_scores.sum(dim=-1), torch.ones(4))
    assert stats == (4, 1, 2, 1)


def test_quota_schedule_bootstraps_only_early_without_optimizer_phase_boundary():
    args = SimpleNamespace(
        moe_joint_new_expert_quota=0.0,
        moe_joint_new_expert_quota_schedule='300:0.5,600:0.3',
        iteration=0,
    )
    assert _moe_joint_new_expert_quota(args, 0) == 0.5
    assert _moe_joint_new_expert_quota(args, 299) == 0.5
    assert _moe_joint_new_expert_quota(args, 300) == 0.3
    assert _moe_joint_new_expert_quota(args, 599) == 0.3
    assert _moe_joint_new_expert_quota(args, 600) == 0.0
    assert _moe_joint_new_expert_quota(args, 1799) == 0.0


def test_quota_gradient_assembly_keeps_only_expert_quota_and_router_natural_replay():
    expert = torch.nn.Parameter(torch.zeros(2))
    router = torch.nn.Parameter(torch.zeros(2))
    expert.main_grad = torch.tensor([5.0, 5.0])
    router.main_grad = torch.tensor([3.0, 3.0])

    router_snapshot = {
        id(router): {'param': router, 'layer_number': 1, 'primary': router.main_grad.clone()}
    }
    _zero_joint_replay_non_router_grads([expert, router], {id(router)})
    # Quota backward: both receive gradients, but its router contribution is discarded.
    expert.main_grad.add_(torch.tensor([7.0, 7.0]))
    router.main_grad.add_(torch.tensor([100.0, 100.0]))
    _restore_joint_replay_router_grads(router_snapshot)

    expert_snapshot = _snapshot_joint_replay_non_router_grads(
        [expert, router], {id(router)}
    )
    # Replay backward: its expert contribution is discarded; router accumulates.
    expert.main_grad.add_(torch.tensor([13.0, 13.0]))
    router.main_grad.add_(torch.tensor([11.0, 11.0]))
    _restore_joint_replay_non_router_grads(expert_snapshot)

    assert torch.equal(expert.main_grad, torch.tensor([7.0, 7.0]))
    assert torch.equal(router.main_grad, torch.tensor([14.0, 14.0]))


def test_quota_gradient_can_be_added_weakly_without_dropping_natural_expert_grad():
    expert = torch.nn.Parameter(torch.zeros(2))
    expert.main_grad = torch.tensor([5.0, 7.0])
    snapshot = _snapshot_joint_replay_non_router_grads([expert], set())
    # The second backward accumulates a quota contribution of [10, 20].
    expert.main_grad.add_(torch.tensor([10.0, 20.0]))
    _blend_joint_replay_non_router_grads(snapshot, 0.1)
    assert torch.allclose(expert.main_grad, torch.tensor([6.0, 9.0]))


def test_hidden_kl_is_zero_for_identical_layer_outputs():
    labels = torch.zeros((2, 3), dtype=torch.long)
    mask = torch.ones((2, 3))
    hidden = {1: torch.randn(3, 2, 5), 2: torch.randn(3, 2, 5)}
    loss = _masked_layer_hidden_kl(hidden, hidden, labels, mask, temperature=2.0)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_hidden_kl_masks_padding_and_backpropagates_to_student_only():
    labels = torch.zeros((1, 2), dtype=torch.long)
    mask = torch.tensor([[1.0, 0.0]])
    student = torch.zeros((2, 1, 3), requires_grad=True)
    teacher = torch.zeros((2, 1, 3))
    teacher[0, 0, 0] = 2.0
    teacher[1, 0, 1] = 100.0
    loss = _masked_layer_hidden_kl({1: student}, {1: teacher}, labels, mask, 1.0)
    loss.backward()
    assert loss.item() > 0
    assert student.grad[0].abs().sum().item() > 0
    assert student.grad[1].abs().sum().item() == 0


def test_hidden_kl_is_a_token_mean_before_loss_func_scaling():
    """The helper stays invariant to duplicated valid tokens.

    loss_func is responsible for converting this mean back to a token-sum
    numerator before Megatron applies its global token normalization.
    """
    student_one = torch.tensor([[[0.0, 0.0, 0.0]]])
    teacher_one = torch.tensor([[[2.0, 0.0, 0.0]]])
    one = _masked_layer_hidden_kl(
        {1: student_one}, {1: teacher_one},
        torch.zeros((1, 1), dtype=torch.long), torch.ones((1, 1)), 1.0,
    )

    student_two = student_one.repeat(2, 1, 1)
    teacher_two = teacher_one.repeat(2, 1, 1)
    two = _masked_layer_hidden_kl(
        {1: student_two}, {1: teacher_two},
        torch.zeros((1, 2), dtype=torch.long), torch.ones((1, 2)), 1.0,
    )
    assert torch.allclose(one, two)


def test_token_mean_is_restored_to_megatron_loss_numerator():
    mean_loss = torch.tensor(0.02, requires_grad=True)
    numerator = _token_mean_to_loss_numerator(mean_loss, torch.tensor(18432.0))
    normalized = numerator / 18432.0
    normalized.backward()
    assert normalized.item() == pytest.approx(0.02)
    assert mean_loss.grad.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(0, 100.0), (100, 85.0), (300, 55.0), (600, 10.0), (900, 10.0)],
)
def test_hidden_kl_coeff_linearly_decays_then_holds(iteration, expected):
    args = SimpleNamespace(
        moe_old_hidden_kl_coeff=10.0,
        moe_old_hidden_kl_coeff_start=100.0,
        moe_old_hidden_kl_coeff_decay_steps=600,
        iteration=iteration,
    )
    assert _effective_old_hidden_kl_coeff(args) == pytest.approx(expected)


def test_hidden_kl_coeff_schedule_uses_current_iteration_and_can_be_disabled():
    args = SimpleNamespace(
        moe_old_hidden_kl_coeff=10.0,
        moe_old_hidden_kl_coeff_start=100.0,
        moe_old_hidden_kl_coeff_decay_steps=600,
        iteration=0,
        curr_iteration=400,
    )
    assert _effective_old_hidden_kl_coeff(args) == pytest.approx(40.0)
    args.moe_old_hidden_kl_coeff_decay_steps = 0
    assert _effective_old_hidden_kl_coeff(args) == pytest.approx(10.0)


def test_hidden_kl_replay_configuration_is_valid_and_mutually_exclusive():
    args = _kd_args(enabled=False, coeff=0.0)
    args.moe_joint_replay_old_data_hidden_kl = True
    _validate_joint_replay_old_data_kd(args)
    assert _old_moe_distill_teacher_requested(args)
    args.moe_joint_replay_old_data_kd = True
    with pytest.raises(ValueError, match="exactly one"):
        _validate_joint_replay_old_data_kd(args)


def test_hidden_mse_replay_configuration_is_valid():
    args = _kd_args(enabled=False, coeff=0.0)
    args.moe_joint_replay_old_data_hidden_mse = True
    args.moe_old_hidden_mse_coeff = 10.0
    _validate_joint_replay_old_data_kd(args)


def test_targeted_cka_gt_requests_frozen_teacher_even_without_kd_loss():
    args = _kd_args(enabled=False, coeff=0.0)
    args.cka_gt_targeted_path = "/tmp/exact-targeted"
    assert _old_moe_distill_teacher_requested(args)


@pytest.mark.parametrize(
    "enabled_objectives",
    [
        ("moe_joint_replay_old_data_kd", "moe_joint_replay_old_data_hidden_mse"),
        ("moe_joint_replay_old_data_hidden_kl", "moe_joint_replay_old_data_hidden_mse"),
        (
            "moe_joint_replay_old_data_kd",
            "moe_joint_replay_old_data_hidden_kl",
            "moe_joint_replay_old_data_hidden_mse",
        ),
    ],
)
def test_old_data_replay_objectives_are_three_way_mutually_exclusive(enabled_objectives):
    args = _kd_args(enabled=False)
    for objective in enabled_objectives:
        setattr(args, objective, True)
    with pytest.raises(ValueError, match="exactly one"):
        _validate_joint_replay_old_data_kd(args)


def test_hidden_mse_replay_rejects_nonpositive_coefficient():
    args = _kd_args(enabled=False)
    args.moe_joint_replay_old_data_hidden_mse = True
    args.moe_old_hidden_mse_coeff = 0.0
    with pytest.raises(ValueError, match="old-hidden-mse-coeff"):
        _validate_joint_replay_old_data_kd(args)


def test_hidden_mse_teacher_and_pure_replay_activate_only_on_replay_branch():
    args = _kd_args(enabled=False, coeff=0.0)
    args.moe_joint_replay_old_data_hidden_mse = True
    args.moe_old_hidden_mse_coeff = 10.0
    args.moe_expansion_distill_lm_loss_coeff = 1.0
    args._moe_joint_replay_old_data_kd_active = False
    teacher = object()

    assert _old_moe_distill_teacher_requested(args)
    assert _effective_lm_loss_coeff(args) == 1.0
    assert not _teacher_kd_enabled_for_current_branch(args, teacher, False)
    assert not _old_data_hidden_mse_enabled_for_current_branch(args)

    args._moe_joint_replay_old_data_kd_active = True
    assert _effective_lm_loss_coeff(args) == 0.0
    assert _teacher_kd_enabled_for_current_branch(args, teacher, False)
    assert _old_data_hidden_mse_enabled_for_current_branch(args)


def test_hidden_mse_is_token_sum_and_is_normalized_exactly_once():
    """The helper returns Megatron's numerator, not a per-token mean."""
    student_one = torch.tensor([[[0.0, 1.0]]])
    teacher_one = torch.tensor([[[2.0, 1.0]]])
    one_token_sum = _masked_layer_hidden_mse(
        {2: student_one},
        {2: teacher_one},
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1)),
    )

    student_two = student_one.repeat(2, 1, 1)
    teacher_two = teacher_one.repeat(2, 1, 1)
    two_token_sum = _masked_layer_hidden_mse(
        {2: student_two},
        {2: teacher_two},
        torch.zeros((1, 2), dtype=torch.long),
        torch.ones((1, 2)),
    )

    assert torch.allclose(two_token_sum, 2 * one_token_sum)
    assert torch.allclose(two_token_sum / 2, one_token_sum)


def test_hidden_mse_masks_padding_and_backpropagates_to_student_only():
    labels = torch.zeros((1, 2), dtype=torch.long)
    mask = torch.tensor([[1.0, 0.0]])
    student = torch.zeros((2, 1, 3), requires_grad=True)
    teacher = torch.zeros((2, 1, 3))
    teacher[0, 0, 0] = 2.0
    teacher[1, 0, 1] = 100.0

    loss = _masked_layer_hidden_mse({2: student}, {2: teacher}, labels, mask)
    loss.backward()

    assert loss.item() > 0
    assert student.grad[0].abs().sum().item() > 0
    assert student.grad[1].abs().sum().item() == 0
    assert teacher.grad is None


def test_joint_replay_grad_metrics_include_last_layer_and_old_new_rows():
    """The diagnostic must prove that replay reaches router layer 9."""
    router_weight = torch.nn.Parameter(torch.zeros(2, 3))
    primary = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    replay_scaled = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    router_weight.main_grad = primary + replay_scaled
    snapshot = {
        id(router_weight): {
            "param": router_weight,
            "layer_number": 9,
            "primary": primary,
        }
    }

    metrics = _joint_replay_router_grad_metrics(
        model=None,
        primary_snapshot=snapshot,
        replay_coefficient=10.0,
        boundary=1,
    )

    prefix = "joint_replay/router_grad/layer_9"
    assert metrics[f"{prefix}/all_rows/replay_scaled_norm"].item() == pytest.approx(5.0)
    assert metrics[f"{prefix}/old_rows/replay_scaled_norm"].item() == pytest.approx(3.0)
    assert metrics[f"{prefix}/new_rows/replay_scaled_norm"].item() == pytest.approx(4.0)
    assert metrics[f"{prefix}/all_rows/replay_raw_norm"].item() == pytest.approx(0.5)
    assert metrics["joint_replay/router_grad/replay_objective_coefficient"].item() == 10.0
