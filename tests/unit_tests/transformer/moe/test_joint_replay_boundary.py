from types import SimpleNamespace

import pytest

from megatron.training.training import _debug_param_kind, _resolve_joint_replay_existing_experts


def _args(moe=None, shared=None):
    return SimpleNamespace(
        moe_resume_from_num_experts=moe,
        shared_router_hybrid_resume_from_num_experts=shared,
    )


def test_joint_replay_boundary_accepts_standard_moe_resume():
    assert _resolve_joint_replay_existing_experts(_args(moe=8)) == 8


def test_joint_replay_boundary_accepts_shared_router_resume():
    assert _resolve_joint_replay_existing_experts(_args(shared=8)) == 8


def test_joint_replay_boundary_accepts_matching_resume_modes():
    assert _resolve_joint_replay_existing_experts(_args(moe=8, shared=8)) == 8


def test_joint_replay_boundary_rejects_conflicting_resume_modes():
    with pytest.raises(ValueError, match="conflicting"):
        _resolve_joint_replay_existing_experts(_args(moe=8, shared=16))


def test_joint_replay_boundary_requires_a_resume_mode():
    with pytest.raises(ValueError, match="requires"):
        _resolve_joint_replay_existing_experts(_args())


@pytest.mark.parametrize(
    "name",
    (
        "decoder.layers.1.self_attention.attn_lora_experts.q_lora_a",
        "decoder.layers.1.self_attention.shared_qv_lora_experts.q_lora_a",
        "decoder.layers.1.self_attention.shared_full_rank_lora_experts.qkv_lora_a",
    ),
)
def test_debug_param_kind_recognizes_attention_expert_variants(name):
    assert _debug_param_kind(name) == "attention_expert"
