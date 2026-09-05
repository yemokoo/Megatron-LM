"""Streaming sufficient statistics for paired hidden-representation analysis.

This module intentionally knows nothing about Megatron.  The model-side hook passes
aligned [token, hidden] tensors here; only block moments and a bounded deterministic
reservoir are retained.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch


def _new_layer_accumulator(hidden_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "sum_x": torch.zeros(hidden_size, dtype=torch.float64, device=device),
        "sum_y": torch.zeros(hidden_size, dtype=torch.float64, device=device),
        "sum_delta": torch.zeros(hidden_size, dtype=torch.float64, device=device),
        "xx": torch.zeros((hidden_size, hidden_size), dtype=torch.float64, device=device),
        "yy": torch.zeros((hidden_size, hidden_size), dtype=torch.float64, device=device),
        "xy": torch.zeros((hidden_size, hidden_size), dtype=torch.float64, device=device),
        "cosine_hist": torch.zeros(2000, dtype=torch.int64, device=device),
        "relative_l2_hist": torch.zeros(2002, dtype=torch.int64, device=device),
        "scalar_sums": torch.zeros(6, dtype=torch.float64, device=device),
        "stable_counts": torch.zeros(6, dtype=torch.int64, device=device),
    }


def _update_scalar(acc: Dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> None:
    xf, yf = x.float(), y.float()
    delta = yf - xf
    xnorm = xf.norm(dim=-1)
    ynorm = yf.norm(dim=-1)
    dnorm = delta.norm(dim=-1)
    cosine = torch.nn.functional.cosine_similarity(xf, yf, dim=-1, eps=1e-12)
    relative = dnorm / xnorm.clamp_min(1e-12)
    acc["cosine_hist"] += torch.histc(cosine, bins=2000, min=-1.0, max=1.0).to(torch.int64)
    clipped = relative.clamp(0.0, 2.0)
    base_hist = torch.histc(clipped, bins=2000, min=0.0, max=2.0).to(torch.int64)
    acc["relative_l2_hist"][1:-1] += base_hist
    acc["relative_l2_hist"][0] += (relative < 0.0).sum()
    acc["relative_l2_hist"][-1] += (relative > 2.0).sum()
    acc["scalar_sums"] += torch.stack(
        (cosine.sum(), relative.sum(), xnorm.sum(), ynorm.sum(), dnorm.sum(), delta.square().sum())
    ).double()
    acc["stable_counts"] += torch.stack(
        (
            (cosine >= 0.99).sum(),
            (cosine >= 0.999).sum(),
            (relative <= 0.01).sum(),
            (relative <= 0.05).sum(),
            (relative <= 0.10).sum(),
            ((cosine >= 0.99) & (relative <= 0.05)).sum(),
        )
    ).to(torch.int64)


@dataclass
class Reservoir:
    size: int
    layers: Iterable[int]
    hidden_size: int
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.keys = np.empty((0,), dtype=np.float64)
        self.global_indices = np.empty((0,), dtype=np.int64)
        self.token_ids = np.empty((0,), dtype=np.int32)
        self.positions = np.empty((0,), dtype=np.int16)
        self.delta = {int(layer): np.empty((0, self.hidden_size), dtype=np.float16) for layer in self.layers}

    def update(self, global_start: int, token_ids: torch.Tensor, positions: torch.Tensor,
               deltas: Dict[int, torch.Tensor]) -> None:
        if self.size <= 0 or token_ids.numel() == 0:
            return
        n = int(token_ids.numel())
        new_keys = self.rng.random(n)
        all_keys = np.concatenate((self.keys, new_keys))
        keep = np.argpartition(all_keys, min(self.size, all_keys.size) - 1)[: self.size]
        old_n = self.keys.size
        current_rows = keep[keep >= old_n] - old_n
        old_rows = keep[keep < old_n]
        # Preserve the selected order consistently for every field/layer.
        merged_global = np.concatenate((self.global_indices, np.arange(global_start, global_start + n)))
        merged_token = np.concatenate((self.token_ids, token_ids.detach().cpu().numpy().astype(np.int32)))
        merged_pos = np.concatenate((self.positions, positions.detach().cpu().numpy().astype(np.int16)))
        self.global_indices = merged_global[keep]
        self.token_ids = merged_token[keep]
        self.positions = merged_pos[keep]
        for layer in self.delta:
            current = deltas[layer][torch.as_tensor(current_rows, device=deltas[layer].device)].float().cpu().numpy().astype(np.float16)
            merged = np.concatenate((self.delta[layer][old_rows], current), axis=0)
            # `keep` interleaves old/current; rebuild the same interleaving.
            out = np.empty((keep.size, self.hidden_size), dtype=np.float16)
            old_cursor = current_cursor = 0
            for out_row, source_row in enumerate(keep):
                if source_row < old_n:
                    out[out_row] = self.delta[layer][source_row]
                    old_cursor += 1
                else:
                    out[out_row] = current[current_cursor]
                    current_cursor += 1
            self.delta[layer] = out
        self.keys = all_keys[keep]


class PairedStreamingStats:
    """Accumulate exact block moments and bounded token-level diagnostics."""

    def __init__(self, output_dir: str, layers: Iterable[int], hidden_size: int,
                 block_tokens: int, target_tokens: int, reservoir_size: int,
                 seed: int = 1234, metadata: dict | None = None) -> None:
        if block_tokens <= 0 or target_tokens <= 0 or target_tokens % block_tokens:
            raise ValueError("target_tokens must be a positive multiple of block_tokens")
        self.output_dir = os.path.abspath(output_dir)
        self.layers = tuple(int(x) for x in layers)
        self.hidden_size = int(hidden_size)
        self.block_tokens = int(block_tokens)
        self.target_tokens = int(target_tokens)
        self.metadata = dict(metadata or {})
        self.total = 0
        self.block_index = 0
        self.block_count = 0
        self.acc = None
        self.ffn_acc = None
        self.block_hash = hashlib.sha256()
        self.block_hashes = []
        self.sample_records = []
        self.sample_count = 0
        self.sample_manifest_tmp = os.path.join(self.output_dir, "samples.jsonl.inprogress")
        self.reservoir = Reservoir(reservoir_size, self.layers, hidden_size, seed)
        os.makedirs(self.output_dir, exist_ok=True)
        if os.path.exists(self.sample_manifest_tmp):
            os.remove(self.sample_manifest_tmp)

    def record_samples(self, tokens: torch.Tensor, loss_mask: torch.Tensor) -> None:
        """Record stable sample identities without retaining corpus text or activations."""
        token_rows = tokens.detach().cpu().numpy()
        mask_rows = loss_mask.detach().cpu().numpy().astype(bool)
        with open(self.sample_manifest_tmp, "a", encoding="utf-8") as handle:
            for row, mask in zip(token_rows, mask_rows):
                valid_tokens = np.asarray(row[mask], dtype=np.int32)
                valid_positions = np.flatnonzero(mask).astype(np.int16)
                record = {
                    "sample_index": self.sample_count,
                    "valid_tokens": int(valid_tokens.size),
                    "token_sha256": hashlib.sha256(valid_tokens.tobytes()).hexdigest(),
                    "position_sha256": hashlib.sha256(valid_positions.tobytes()).hexdigest(),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                self.sample_count += 1

    def _ensure(self, device: torch.device, ffn: bool) -> None:
        if self.acc is None:
            self.acc = {layer: _new_layer_accumulator(self.hidden_size, device) for layer in self.layers}
        if ffn and self.ffn_acc is None:
            self.ffn_acc = {layer: _new_layer_accumulator(self.hidden_size, device) for layer in self.layers}

    def remaining(self) -> int:
        return self.target_tokens - self.total

    def update(self, before: Dict[int, torch.Tensor], after: Dict[int, torch.Tensor],
               token_ids: torch.Tensor, positions: torch.Tensor,
               ffn_before: Dict[int, torch.Tensor] | None = None,
               ffn_after: Dict[int, torch.Tensor] | None = None) -> None:
        offset = 0
        n_total = min(int(token_ids.numel()), self.remaining())
        while offset < n_total:
            take = min(n_total - offset, self.block_tokens - self.block_count)
            sl = slice(offset, offset + take)
            device = before[self.layers[0]].device
            self._ensure(device, ffn_before is not None)
            deltas = {}
            for layer in self.layers:
                x = before[layer][sl].float()
                y = after[layer][sl].float()
                d = y - x
                deltas[layer] = d
                acc = self.acc[layer]
                acc["sum_x"] += x.sum(dim=0).double()
                acc["sum_y"] += y.sum(dim=0).double()
                acc["sum_delta"] += d.sum(dim=0).double()
                acc["xx"] += (x.T @ x).double()
                acc["yy"] += (y.T @ y).double()
                acc["xy"] += (x.T @ y).double()
                _update_scalar(acc, x, y)
                if ffn_before is not None and ffn_after is not None:
                    _update_scalar(self.ffn_acc[layer], ffn_before[layer][sl], ffn_after[layer][sl])
            ids = token_ids[sl]
            pos = positions[sl]
            self.block_hash.update(ids.detach().cpu().numpy().astype(np.int32).tobytes())
            self.block_hash.update(pos.detach().cpu().numpy().astype(np.int16).tobytes())
            self.reservoir.update(self.total, ids, pos, deltas)
            self.total += take
            self.block_count += take
            offset += take
            if self.block_count == self.block_tokens:
                self._flush_block()

    def _flush_block(self) -> None:
        payload = {
            "layer_numbers": np.asarray(self.layers, dtype=np.int16),
            "count": np.asarray(self.block_count, dtype=np.int64),
        }
        for name in ("sum_x", "sum_y", "sum_delta", "xx", "yy", "xy",
                     "cosine_hist", "relative_l2_hist", "scalar_sums", "stable_counts"):
            payload[name] = np.stack([self.acc[layer][name].cpu().numpy() for layer in self.layers])
        if self.ffn_acc is not None:
            for name in ("cosine_hist", "relative_l2_hist", "scalar_sums", "stable_counts"):
                payload[f"ffn_{name}"] = np.stack(
                    [self.ffn_acc[layer][name].cpu().numpy() for layer in self.layers]
                )
        path = os.path.join(self.output_dir, f"block_{self.block_index:03d}.npz")
        np.savez(path, **payload)
        digest = self.block_hash.hexdigest()
        self.block_hashes.append({"block": self.block_index, "tokens": self.block_count, "sha256": digest})
        self.block_index += 1
        self.block_count = 0
        self.acc = None
        self.ffn_acc = None
        self.block_hash = hashlib.sha256()

    def finalize(self) -> dict:
        if self.total != self.target_tokens or self.block_count:
            raise RuntimeError(f"incomplete stream: captured={self.total}, target={self.target_tokens}")
        reservoir_payload = {
            "layer_numbers": np.asarray(self.layers, dtype=np.int16),
            "priority": self.reservoir.keys,
            "global_indices": self.reservoir.global_indices,
            "token_ids": self.reservoir.token_ids,
            "positions": self.reservoir.positions,
            "delta": np.stack([self.reservoir.delta[layer] for layer in self.layers]),
        }
        np.savez(os.path.join(self.output_dir, "delta_reservoir.npz"), **reservoir_payload)
        os.replace(self.sample_manifest_tmp, os.path.join(self.output_dir, "samples.jsonl"))
        result = {
            **self.metadata,
            "layers": list(self.layers),
            "hidden_size": self.hidden_size,
            "target_tokens": self.target_tokens,
            "block_tokens": self.block_tokens,
            "blocks": self.block_hashes,
            "reservoir_size": int(self.reservoir.keys.size),
            "encountered_samples": self.sample_count,
        }
        with open(os.path.join(self.output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result


def validate_or_write_manifest(metadata: dict, manifest_path: str) -> None:
    """Make every checkpoint pair prove it used identical token IDs/positions."""
    canonical = {key: metadata[key] for key in ("target_tokens", "block_tokens", "blocks")}
    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != canonical:
            raise RuntimeError(f"probe manifest mismatch: {manifest_path}")
    else:
        temporary = manifest_path + ".inprogress"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(canonical, handle, indent=2)
        os.replace(temporary, manifest_path)
