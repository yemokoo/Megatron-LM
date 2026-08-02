# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

from megatron.training.checkpointing import (
    _strip_missing_attn_full_rank_lora_from_sharded_state_dict,
)


def _sharded_value(key):
    return SimpleNamespace(key=key)


def test_full_rank_lora_filter_keeps_checkpoint_backed_tensors():
    present_key = (
        "decoder.layers.1.self_attention.shared_full_rank_lora_experts.qkv_lora_a"
    )
    missing_key = (
        "decoder.layers.1.self_attention.shared_full_rank_lora_experts.proj_lora_b"
    )
    ordinary_key = "decoder.layers.1.self_attention.linear_qkv.weight"
    optimizer_key = f"optimizer.state.exp_avg.{missing_key}"
    present_value = _sharded_value(present_key)
    ordinary_value = _sharded_value(ordinary_key)
    optimizer_value = _sharded_value(optimizer_key)
    state_dict = {
        "model": {
            "attention": {
                "present_adapter": present_value,
                "missing_adapter": _sharded_value(missing_key),
                "ordinary_weight": ordinary_value,
            }
        },
        "optimizer": {"adapter_state": optimizer_value},
    }

    filtered, stripped_keys = (
        _strip_missing_attn_full_rank_lora_from_sharded_state_dict(
            state_dict,
            checkpoint_tensor_keys={present_key, ordinary_key},
        )
    )

    assert filtered is state_dict
    assert filtered["model"]["attention"]["present_adapter"] is present_value
    assert "missing_adapter" not in filtered["model"]["attention"]
    assert filtered["model"]["attention"]["ordinary_weight"] is ordinary_value
    assert filtered["optimizer"]["adapter_state"] is optimizer_value
    assert stripped_keys == [missing_key]


def test_full_rank_lora_filter_retains_legacy_blanket_behavior_without_metadata():
    adapter_key = (
        "decoder.layers.1.self_attention.shared_full_rank_lora_experts.qkv_lora_b"
    )
    ordinary_value = _sharded_value("decoder.layers.1.mlp.linear_fc1.weight")
    state_dict = {
        "model": {
            "adapter": _sharded_value(adapter_key),
            "ordinary_weight": ordinary_value,
        }
    }

    filtered, stripped_keys = (
        _strip_missing_attn_full_rank_lora_from_sharded_state_dict(state_dict)
    )

    assert "adapter" not in filtered["model"]
    assert filtered["model"]["ordinary_weight"] is ordinary_value
    assert stripped_keys == [adapter_key]
