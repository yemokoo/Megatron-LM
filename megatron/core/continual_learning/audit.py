"""Architecture and active-parameter invariants for the matched controls."""

from __future__ import annotations

import json
from pathlib import Path


def expected_active_counts(hidden_size=1024, dense_ffn=1408, expert_ffn=352, top_k=4, olora_rank=352):
    dense = 3 * hidden_size * dense_ffn
    moe = top_k * 3 * hidden_size * expert_ffn
    olora = 3 * 2 * (hidden_size * olora_rank + olora_rank * hidden_size)
    if dense != moe:
        raise AssertionError(f"dense/MoE active FFN mismatch: {dense} != {moe}")
    if olora != moe:
        raise AssertionError(f"O-LoRA/MoE active projection mismatch: {olora} != {moe}")
    return {
        "dense_ffn_projection_parameters_per_layer": dense,
        "fixed_moe_top4_projection_parameters_per_layer": moe,
        "olora_final_three_qv_projection_parameters_per_layer": olora,
        "fixed_moe_router_overhead_per_layer": hidden_size * 24,
    }


def write_audit(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
