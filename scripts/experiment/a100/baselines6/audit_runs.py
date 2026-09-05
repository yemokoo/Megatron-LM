#!/usr/bin/env python3
"""Fail-fast audit for a completed six-baseline output tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ACTIVE = 4_325_376


def read_json(path: Path):
    if not path.is_file():
        raise AssertionError(f"missing audit: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_sidecar(stage: Path, method: str):
    path = stage / f"continual_state_{method}" / "state_tp00_pp00.pt"
    if not path.is_file():
        raise AssertionError(f"missing {method} sidecar: {path}")
    wrapper = torch.load(path, map_location="cpu", weights_only=False)
    if wrapper["method"] != method:
        raise AssertionError(f"sidecar method mismatch in {path}: {wrapper['method']}")
    return wrapper["payload"]


def audit_stage(root: Path, relative: str, method: str, task: str, train_iters: int):
    stage = root / relative
    tracker = stage / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != str(train_iters):
        raise AssertionError(f"incomplete tracker: {tracker}")
    audit = read_json(stage / "continual_audit" / "final.json")
    if audit["method"] != method or audit["task"] != task:
        raise AssertionError(f"method/task mismatch in {relative}")
    if audit["layer_scope"] != [2, 9]:
        raise AssertionError(f"layer scope mismatch in {relative}")
    if audit["old_model_kd_coefficient"] != 0.0:
        raise AssertionError(f"KD is enabled in {relative}")
    counts = audit["active_counts"]
    matched = (
        counts["dense_ffn_projection_parameters_per_layer"],
        counts["fixed_moe_top4_projection_parameters_per_layer"],
        counts["olora_final_three_qv_projection_parameters_per_layer"],
    )
    if matched != (ACTIVE, ACTIVE, ACTIVE):
        raise AssertionError(f"active projection mismatch in {relative}: {matched}")
    if task != "wiki" and audit.get("frozen_bit_identical") is not True:
        raise AssertionError(f"frozen checksum changed in {relative}")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--train-iters", type=int, required=True)
    parser.add_argument("--slora-ranks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--require-regularizer-audit", action="store_true")
    parser.add_argument("--require-positive-olora", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    audits = {}
    stages = [
        ("common_dense/wiki", "ewc", "wiki"),
        ("ewc/code", "ewc", "code"),
        ("ewc/conversation", "ewc", "conversation"),
        ("trace_gem/code", "trace_gem", "code"),
        ("trace_gem/conversation", "trace_gem", "conversation"),
        ("olora/wiki", "olora", "wiki"),
        ("olora/code", "olora", "code"),
        ("olora/conversation", "olora", "conversation"),
        ("sequential_dense/code", "sequential_dense", "code"),
        ("sequential_dense/conversation", "sequential_dense", "conversation"),
        ("fixed_moe/wiki", "fixed_moe", "wiki"),
        ("fixed_moe/code", "fixed_moe", "code"),
        ("fixed_moe/conversation", "fixed_moe", "conversation"),
    ]
    for rank in args.slora_ranks:
        stages.extend(
            [
                (f"slora_pre/rank{rank}/code", "slora_pre", "code"),
                (f"slora_pre/rank{rank}/conversation", "slora_pre", "conversation"),
            ]
        )
    for relative, method, task in stages:
        audits[relative] = audit_stage(root, relative, method, task, args.train_iters)

    common_ewc = read_sidecar(root / "common_dense/wiki", "ewc")
    code_ewc = read_sidecar(root / "ewc/code", "ewc")
    for payload in (common_ewc, code_ewc):
        if payload["fisher_definition"] != "task_boundary_diagonal_empirical_fisher_pure_nll":
            raise AssertionError("non-canonical EWC Fisher definition")
        if not 1 <= payload["fisher_batches"] <= 100:
            raise AssertionError(f"invalid Fisher batch count: {payload['fisher_batches']}")
        if len(payload["fisher_sum"]) != 48 or len(payload["mean"]) != 48:
            raise AssertionError("EWC state must cover 48 Layer-2--9 tensors")

    for relative, expected_count in (
        ("common_dense/wiki", 1),
        ("trace_gem/code", 2),
        ("trace_gem/conversation", 3),
    ):
        payload = read_sidecar(root / relative, "trace_gem")
        if len(payload["memories"]) != expected_count:
            raise AssertionError(f"TRACE memory count mismatch in {relative}")
        if payload["semantics"] != "trace_terminal_gradient_per_parameter_corrected_qp_sign":
            raise AssertionError(f"TRACE projection semantics mismatch in {relative}")

    for rank in args.slora_ranks:
        code = read_sidecar(root / f"slora_pre/rank{rank}/code", "slora_pre")
        conv = read_sidecar(root / f"slora_pre/rank{rank}/conversation", "slora_pre")
        if code["code_rank"] != rank or conv["code_rank"] != rank:
            raise AssertionError(f"SLoRA Code rank mismatch for rank {rank}")
        if code["conversation_rank"] != 64 or conv["conversation_rank"] != 64:
            raise AssertionError(f"SLoRA Conversation rank mismatch for rank {rank}")
        if code["reference_checksum"] != conv["reference_checksum"]:
            raise AssertionError(f"SLoRA immutable reference changed for rank {rank}")
        if len(code["reference"]) != 56 or len(code["retained_ranks"]) != 56:
            raise AssertionError(f"SLoRA target count mismatch for rank {rank}")

    for task, expected_count in (("wiki", 1), ("code", 2), ("conversation", 3)):
        payload = read_sidecar(root / f"olora/{task}", "olora")
        if payload["active_adapter_count"] != expected_count:
            raise AssertionError(f"O-LoRA active slot mismatch in {task}")
        audit = audits[f"olora/{task}"]
        if args.require_regularizer_audit and audit.get("regularizer", {}).get("calls", 0) <= 0:
            raise AssertionError(f"O-LoRA regularizer was not audited in {task}")
        if args.require_positive_olora and task != "wiki":
            if audit.get("regularizer", {}).get("last_value", 0.0) <= 0.0:
                raise AssertionError(f"O-LoRA regularizer is not positive in {task}")
        if args.require_regularizer_audit:
            norms = audit.get("olora_adapter_norms")
            if norms is None or len(norms) != 3:
                raise AssertionError(f"O-LoRA slot norms are missing in {task}")
            for index, slot in enumerate(norms):
                should_be_active = index < expected_count
                if should_be_active and (slot["a_l2"] <= 0.0 or slot["b_l2"] <= 0.0):
                    raise AssertionError(f"O-LoRA active slot {index} is zero in {task}")
                if not should_be_active and (slot["a_l2"] != 0.0 or slot["b_l2"] != 0.0):
                    raise AssertionError(f"O-LoRA inactive slot {index} is nonzero in {task}")

    if args.require_regularizer_audit:
        for task in ("code", "conversation"):
            if audits[f"ewc/{task}"].get("regularizer", {}).get("calls", 0) <= 0:
                raise AssertionError(f"EWC regularizer was not audited in {task}")

    print(
        f"OK: {len(stages)} stages, ranks={args.slora_ranks}, "
        f"train_iters={args.train_iters}, active={ACTIVE:,}"
    )


if __name__ == "__main__":
    main()
