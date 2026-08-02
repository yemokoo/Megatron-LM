#!/usr/bin/env python3
"""Validate indexed safetensors snapshots without loading model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_safetensors(path: Path) -> tuple[set[str], int]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: missing safetensors header length")
        header_length = int.from_bytes(raw_length, "little")
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            raise ValueError(f"{path}: invalid safetensors header length {header_length}")
        header = json.loads(handle.read(header_length))
    tensors = {name for name in header if name != "__metadata__"}
    if not tensors:
        raise ValueError(f"{path}: no tensors in safetensors header")
    maximum_offset = 0
    for name in tensors:
        entry = header[name]
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{path}: invalid data_offsets for {name}")
        start, end = offsets
        if start < 0 or end < start:
            raise ValueError(f"{path}: invalid tensor range for {name}")
        maximum_offset = max(maximum_offset, end)
    expected_file_size = 8 + header_length + maximum_offset
    if path.stat().st_size != expected_file_size:
        raise ValueError(
            f"{path}: expected {expected_file_size} bytes from header, "
            f"found {path.stat().st_size}"
        )
    return tensors, maximum_offset


def verify_model(path: Path) -> dict:
    errors: list[str] = []
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return {"path": str(path), "ok": False, "errors": [f"missing {index_path}"]}
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "ok": False,
            "errors": [f"invalid index: {type(exc).__name__}: {exc}"],
        }

    weight_map = index.get("weight_map", {})
    indexed_tensors = set(weight_map)
    shard_names = sorted(set(weight_map.values()))
    header_tensors: set[str] = set()
    tensor_bytes = 0
    shards = []
    for shard_name in shard_names:
        shard_path = path / shard_name
        if not shard_path.is_file():
            errors.append(f"missing shard: {shard_path}")
            continue
        try:
            tensors, data_bytes = inspect_safetensors(shard_path)
            header_tensors.update(tensors)
            tensor_bytes += data_bytes
            shards.append(
                {
                    "name": shard_name,
                    "file_bytes": shard_path.stat().st_size,
                    "tensor_bytes": data_bytes,
                    "tensors": len(tensors),
                }
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if header_tensors and header_tensors != indexed_tensors:
        missing = sorted(indexed_tensors - header_tensors)
        extra = sorted(header_tensors - indexed_tensors)
        errors.append(
            f"index/header tensor mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    expected_tensor_bytes = index.get("metadata", {}).get("total_size")
    if expected_tensor_bytes is not None and tensor_bytes != expected_tensor_bytes:
        errors.append(
            f"tensor byte mismatch: expected {expected_tensor_bytes}, got {tensor_bytes}"
        )

    for required in ("config.json", "tokenizer_config.json"):
        if not (path / required).is_file():
            errors.append(f"missing model metadata: {path / required}")

    return {
        "path": str(path),
        "ok": not errors,
        "index": str(index_path),
        "indexed_tensors": len(indexed_tensors),
        "expected_tensor_bytes": expected_tensor_bytes,
        "verified_tensor_bytes": tensor_bytes,
        "shards": shards,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [verify_model(path) for path in args.models]
    print(json.dumps({"models": reports}, indent=2))
    return 0 if all(report["ok"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
