#!/usr/bin/env python3
"""Organize persistent FlameMoE checkpoint roots without copying checkpoint data.

The operation is a same-filesystem rename. Scratch/staging/non-persistent/failed
outputs are deliberately not considered checkpoint artifacts. Compatibility
symlinks are opt-in so a normal move removes the old checkpoint location.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/home/seonghyeonnoh/yemokoo/data2_llmcl/LLM-continual-learning-runs"
)
SKIP_PART_MARKERS = (
    "scratch",
    "staging",
    "non_persistent",
    ".inprogress",
    "failed",
    "smoke",
    "parity",
    "flamemoe",
)


def contains_marker(path: Path) -> bool:
    return any(
        marker in part.lower()
        for part in path.parts
        for marker in SKIP_PART_MARKERS
    )


def load_metadata(checkpoint: Path) -> dict[str, Any]:
    candidates = [checkpoint / "logs" / "run_metadata.json"]
    candidates.extend(checkpoint.glob("**/logs/run_metadata.json"))
    for candidate in candidates:
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def architecture(rel: Path, metadata: dict[str, Any]) -> str:
    rank = int(metadata.get("attn_full_rank_lora_rank", 0) or 0)
    active = str(metadata.get("attn_full_rank_lora_active_targets", "") or "")
    trains_attention = bool(metadata.get("train_attention_with_new_experts", False))
    name = str(rel).lower()
    explicit_attention = any(
        term in name
        for term in (
            "attention_expert",
            "attn_expert",
            "ffn-attn",
            "shared_router",
            "full_rank_lora",
            "fullrank_qkvo",
        )
    )
    if rank > 0 or active or trains_attention or explicit_attention:
        return "attention_experts"
    return "ffn_experts_only"


def task(rel: Path) -> str:
    name = str(rel).lower()
    if "conversation" in name or "conv-" in name or "/conv" in name:
        return "conversation"
    if "wiki_ffn" in name or "/wiki/" in f"/{name}/" or name.endswith("wiki"):
        return "wiki"
    return "code"


def objective(rel: Path) -> str:
    name = str(rel).lower()
    leaf = rel.name.lower()
    if "hidden_mse" in leaf or "hiddenmse" in leaf:
        return "hidden_mse"
    if "hidden_kl" in leaf or "hiddenkl" in leaf:
        return "hidden_kl"
    if any(term in leaf for term in ("vocab_kl", "vocabkl", "logits_kd")):
        return "vocab_kl"
    if "joint_old_data_kd" in name:
        return "vocab_kl"
    if (
        ("fingerprint" in leaf and ("kd" in leaf or "selector" in leaf))
        or rel.parts[0].lower().startswith("layer_output_fingerprint_kd")
    ):
        return "fingerprint_kd"
    if stage(rel) == "expansion_kd_init":
        return "kd_init"
    return "lm"


def stage(rel: Path) -> str:
    name = str(rel).lower()
    padded = f"/{name}/"
    if "from_kdinit" in name or "from_kd_init" in name:
        return "no_replay"
    if any(
        term in name
        for term in (
            "expansion_distill_init",
            "expand_",
            "expand-",
            "e8_to_e16",
            "e16_to_e24",
            "e16to24",
        )
    ) or "/kd_init/" in padded:
        return "expansion_kd_init"
    if any(
        term in name
        for term in (
            "joint_old_data",
            "replay",
            "oldlike",
            "old_like",
            "router_only_lm",
            "codewiki_router",
        )
    ):
        return "replay"
    if rel.parts[0].lower().startswith("layer_output_fingerprint_kd"):
        return "fingerprint_replay"
    if task(rel) == "wiki":
        return "pretrain"
    return "no_replay"


def tier(step: int, rel: Path) -> str:
    name = str(rel).lower()
    if step < 600 or any(term in name for term in ("200step", "3batch", "selector_ablation", "gradient_scale")):
        return "short_validation"
    return "full_training"


def directory_size(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for file_name in files:
            try:
                total += (Path(base) / file_name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def destination_name(rel: Path) -> str:
    top = rel.parts[0]
    leaf = rel.name
    readable = f"{top}__{leaf}"
    if len(readable) <= 180:
        return readable
    digest = hashlib.sha256(str(rel).encode()).hexdigest()[:12]
    return f"{top}__{leaf[:130]}__{digest}"


def discover(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for tracker in sorted(root.rglob("latest_checkpointed_iteration.txt")):
        checkpoint = tracker.parent
        rel = checkpoint.relative_to(root)
        if contains_marker(rel) or checkpoint.is_symlink():
            continue
        try:
            step = int(tracker.read_text().strip())
        except (OSError, ValueError):
            continue
        metadata = load_metadata(checkpoint)
        arch = architecture(rel, metadata)
        task_name = task(rel)
        stage_name = stage(rel)
        objective_name = objective(rel)
        tier_name = tier(step, rel)
        destination = (
            root
            / "flamemoe"
            / arch
            / task_name
            / stage_name
            / objective_name
            / tier_name
            / destination_name(rel)
        )
        records.append(
            {
                "source": str(checkpoint),
                "source_relative": str(rel),
                "destination": str(destination),
                "architecture": arch,
                "task": task_name,
                "stage": stage_name,
                "objective": objective_name,
                "tier": tier_name,
                "latest_iteration": step,
                "iteration_directories": sorted(p.name for p in checkpoint.glob("iter_*") if p.is_dir()),
                "bytes": directory_size(checkpoint),
                "metadata_evidence": {
                    key: metadata.get(key)
                    for key in (
                        "attn_full_rank_lora_rank",
                        "attn_full_rank_lora_active_targets",
                        "train_attention_with_new_experts",
                    )
                    if key in metadata
                },
            }
        )
    return records


def validate(records: list[dict[str, Any]], root: Path) -> None:
    destinations: set[str] = set()
    for record in records:
        source = Path(record["source"])
        destination = Path(record["destination"])
        if not source.is_dir() or source.is_symlink():
            raise RuntimeError(f"invalid source: {source}")
        latest = source / f"iter_{record['latest_iteration']:07d}"
        if not latest.is_dir():
            raise RuntimeError(f"latest iteration directory missing: {latest}")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError(f"destination already exists: {destination}")
        if str(destination) in destinations:
            raise RuntimeError(f"duplicate destination: {destination}")
        destinations.add(str(destination))
        if source.stat().st_dev != root.stat().st_dev:
            raise RuntimeError(f"source is not on destination filesystem: {source}")


def write_manifests(records: list[dict[str, Any]], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "checkpoint_manifest.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    with (target / "checkpoint_manifest.tsv").open("w", newline="") as handle:
        columns = [
            "architecture",
            "task",
            "stage",
            "objective",
            "tier",
            "latest_iteration",
            "bytes",
            "source",
            "destination",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def execute(records: list[dict[str, Any]], leave_compatibility_symlinks: bool) -> None:
    for number, record in enumerate(records, 1):
        source = Path(record["source"])
        destination = Path(record["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{number:02d}/{len(records):02d}] {source} -> {destination}", flush=True)
        source.rename(destination)
        if leave_compatibility_symlinks:
            source.symlink_to(destination, target_is_directory=True)


def repair_existing(root: Path) -> None:
    target = root / "flamemoe"
    manifest_path = target / "checkpoint_manifest.json"
    records = json.loads(manifest_path.read_text())
    changes = 0
    for record in records:
        rel = Path(record["source_relative"])
        new_stage = stage(rel)
        new_objective = objective(rel)
        new_destination = (
            target
            / record["architecture"]
            / record["task"]
            / new_stage
            / new_objective
            / record["tier"]
            / destination_name(rel)
        )
        old_destination = Path(record["destination"])
        if old_destination == new_destination:
            continue
        if not old_destination.is_dir() or new_destination.exists():
            raise RuntimeError(
                f"cannot repair {old_destination} -> {new_destination}"
            )
        new_destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"repair: {old_destination} -> {new_destination}")
        old_destination.rename(new_destination)
        source = Path(record["source"])
        if not source.is_symlink():
            raise RuntimeError(f"compatibility link missing: {source}")
        source.unlink()
        source.symlink_to(new_destination, target_is_directory=True)
        record["stage"] = new_stage
        record["objective"] = new_objective
        record["destination"] = str(new_destination)
        changes += 1
    write_manifests(records, target)
    print(f"repaired records: {changes}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repair-existing", action="store_true")
    parser.add_argument("--leave-compatibility-symlinks", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.repair_existing:
        repair_existing(root)
        return
    records = discover(root)
    validate(records, root)
    total = sum(record["bytes"] for record in records)
    counts: dict[tuple[str, str, str, str, str], int] = {}
    for record in records:
        key = tuple(record[name] for name in ("architecture", "task", "stage", "objective", "tier"))
        counts[key] = counts.get(key, 0) + 1
    print(f"checkpoint roots: {len(records)}")
    print(f"logical bytes: {total} ({total / 2**30:.2f} GiB)")
    for key, count in sorted(counts.items()):
        print(f"{count:3d}  {' / '.join(key)}")
    if not args.execute:
        print("dry-run only; pass --execute to rename checkpoint roots")
        return
    target = root / "flamemoe"
    execute(records, args.leave_compatibility_symlinks)
    write_manifests(records, target)
    (target / "attention_experts" / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (target / "attention_experts" / "README.md").write_text(
        "# Attention-expert FlameMoE checkpoints\n\n"
        "No persistent attention-expert checkpoint was found in this run root during organization.\n"
    )
    print(f"manifest: {target / 'checkpoint_manifest.tsv'}")


if __name__ == "__main__":
    main()
