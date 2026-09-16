"""Architecture and active-parameter invariants for the matched controls."""

from __future__ import annotations

import json
from pathlib import Path


def expected_active_counts(
    hidden_size=1024,
    dense_ffn=1408,
    expert_ffn=352,
    top_k=4,
    olora_rank=352,
    enforce_dense_moe_match=True,
    enforce_olora_moe_match=True,
):
    dense = 3 * hidden_size * dense_ffn
    moe = top_k * 3 * hidden_size * expert_ffn
    olora = 3 * 2 * (hidden_size * olora_rank + olora_rank * hidden_size)
    # The matched-control experiment requires equality.  A dense-width DoF
    # sweep deliberately violates it, but should still record both counts.
    if enforce_dense_moe_match and dense != moe:
        raise AssertionError(f"dense/MoE active FFN mismatch: {dense} != {moe}")
    # Ranks other than the matched 352 are deliberate in the O-LoRA rank DoF
    # sweep and must be audited, not rejected -- same policy as the dense width.
    if enforce_olora_moe_match and olora != moe:
        raise AssertionError(f"O-LoRA/MoE active projection mismatch: {olora} != {moe}")
    return {
        "dense_ffn_projection_parameters_per_layer": dense,
        "fixed_moe_top4_projection_parameters_per_layer": moe,
        "olora_final_three_qv_projection_parameters_per_layer": olora,
        "fixed_moe_router_overhead_per_layer": hidden_size * 24,
        "dense_moe_active_parameters_matched": dense == moe,
        "olora_moe_active_parameters_matched": olora == moe,
    }


def write_audit(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
