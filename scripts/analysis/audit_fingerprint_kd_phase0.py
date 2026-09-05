#!/usr/bin/env python3
"""Fail-fast Phase-0 audit for token-gated layer-output fingerprint KD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MEGATRON = REPO / "Megatron-LM"
STABILITY_ROOT = Path(
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/"
    "layer_output_stable_subspace_20260810"
)
REFERENCE = Path(
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/"
    "01_common_kd_init/code_e8_to_e16_wiki_kd_step600"
)
CODE_ONLY = Path(
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/fingerprint_router_geometry_20260809/"
    "checkpoints/D_B_to_Code_only_no_olddata_mb48_gbs2304_step1800"
)
MIXED = Path(
    "/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/"
    "02_nine_runs/local/weights/a100/mha/g2-checkpoints/code/joint_old_data_hidden_kl/"
    "post_kd_teacher_c10_fixed_l2to9/"
    "r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100"
)
BASIS = STABILITY_ROOT / (
    "analysis/stable_subspace_results_bases/"
    "code_only_no_replay_no_router_ft.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_step(path: Path, expected: int) -> dict:
    tracker = path / "latest_checkpointed_iteration.txt"
    actual = int(tracker.read_text().strip())
    if actual != expected:
        raise RuntimeError(f"checkpoint mismatch: {path}: {actual} != {expected}")
    return {"path": str(path), "step": actual, "tracker_sha256": sha256(tracker)}


def source_evidence(path: Path, required: list[str]) -> dict:
    text = path.read_text()
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise RuntimeError(f"missing source evidence in {path}: {missing}")
    lines = text.splitlines()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "matches": {
            needle: [index + 1 for index, line in enumerate(lines) if needle in line]
            for needle in required
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    validation_path = STABILITY_ROOT / "analysis/streaming_validation.json"
    validation = json.loads(validation_path.read_text())
    if validation.get("passed") is not True:
        raise RuntimeError("previous streaming validation did not pass")

    with np.load(BASIS, allow_pickle=False) as payload:
        layers = payload["layer_numbers"]
        bases = payload["stable_bases_rank256"]
        means = payload["old_means"]
        if layers.tolist() != list(range(2, 10)):
            raise RuntimeError(f"wrong basis layer order: {layers.tolist()}")
        if bases.shape != (8, 1024, 256) or means.shape != (8, 1024):
            raise RuntimeError(f"wrong fingerprint shapes: bases={bases.shape}, means={means.shape}")
        if not np.isfinite(bases).all() or not np.isfinite(means).all():
            raise RuntimeError("non-finite fingerprint basis or mean")
        gram_error = float(
            max(np.max(np.abs(basis.T @ basis - np.eye(256))) for basis in bases)
        )
        if gram_error > 1e-4:
            raise RuntimeError(f"basis is not orthonormal: max gram error={gram_error}")

    pretrain = MEGATRON / "pretrain_gpt.py"
    transformer_layer = MEGATRON / "megatron/core/transformer/transformer_layer.py"
    audit = {
        "passed": True,
        "phase": 0,
        "representation": "residual-included Transformer layer_output",
        "layers": list(range(2, 10)),
        "last_layer_included": True,
        "natural_routing_only": True,
        "checkpoints": {
            "expansion_kd_init_complete_teacher": checkpoint_step(REFERENCE, 600),
            "code_only_no_replay_no_router_ft": checkpoint_step(CODE_ONLY, 1800),
            "code_lm_plus_wiki_layer_output_hidden_kl": checkpoint_step(MIXED, 1800),
        },
        "fingerprint": {
            "path": str(BASIS),
            "sha256": sha256(BASIS),
            "basis_shape": list(bases.shape),
            "mean_shape": list(means.shape),
            "dtype": str(bases.dtype),
            "finite": True,
            "max_orthonormal_gram_error": gram_error,
        },
        "previous_streaming_validation": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
            "passed": True,
        },
        "source_evidence": {
            "hook_and_hidden_kd": source_evidence(
                pretrain,
                [
                    "def _capture_transformer_layer_outputs",
                    "def _masked_layer_hidden_kl",
                    "student_hidden_ctx",
                    "teacher_hidden_ctx",
                ],
            ),
            "residual_included_return": source_evidence(
                transformer_layer,
                [
                    "mlp_output_with_bias = self.mlp",
                    "mlp_output_with_bias, residual, self.hidden_dropout",
                    "inp=hidden_states",
                    "return output, context",
                ],
            ),
        },
        "scope": {
            "old_replay_during_fingerprint_kd": False,
            "forced_routing": False,
            "conversation": False,
            "automatic_1800_step_run": False,
            "stop_after_200_step_report_without_user_approval": True,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".inprogress")
    temporary.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
