"""Rank-aware sidecar state for continual-learning statistics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from megatron.core import parallel_state


STATE_VERSION = 1


def _rank_suffix() -> str:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tp = parallel_state.get_tensor_model_parallel_rank()
        pp = parallel_state.get_pipeline_model_parallel_rank()
        return f"tp{tp:02d}_pp{pp:02d}"
    return "tp00_pp00"


def _is_data_parallel_writer() -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return True
    return parallel_state.get_data_parallel_rank() == 0


def save_sidecar(directory: str, method: str, task: str, payload: Dict[str, Any]) -> Optional[str]:
    """Save one file per TP/PP shard and one human-readable manifest."""
    if not _is_data_parallel_writer():
        return None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"state_{_rank_suffix()}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    envelope = {
        "version": STATE_VERSION,
        "method": method,
        "task": task,
        "rank_suffix": _rank_suffix(),
        "payload": payload,
    }
    torch.save(envelope, tmp)
    os.replace(tmp, path)
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": STATE_VERSION,
                "method": method,
                "task": task,
                "state_pattern": "state_tpXX_ppXX.pt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(path)


def load_sidecar(directory: Optional[str], expected_method: Optional[str] = None) -> Optional[dict]:
    if not directory:
        return None
    root = Path(directory)
    path = root if root.is_file() else root / f"state_{_rank_suffix()}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"continual state shard not found: {path}")
    envelope = torch.load(path, map_location="cpu", weights_only=False)
    if int(envelope.get("version", -1)) != STATE_VERSION:
        raise RuntimeError(f"unsupported continual state version in {path}: {envelope.get('version')}")
    if expected_method and envelope.get("method") != expected_method:
        raise RuntimeError(
            f"continual state method mismatch: {envelope.get('method')} != {expected_method}"
        )
    return envelope
