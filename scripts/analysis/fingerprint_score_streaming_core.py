"""Streaming token-score statistics for stable/PCA/random hidden subspaces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


class FingerprintScoreStats:
    def __init__(self, output_dir, representation_names, ranks, selector_names,
                 target_tokens, block_tokens, vocab_size, sequence_length,
                 reservoir_size=8192, bins=4096, seed=1234, metadata=None):
        if target_tokens <= 0 or block_tokens <= 0 or target_tokens % block_tokens:
            raise ValueError("target_tokens must be a positive multiple of block_tokens")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.representation_names = tuple(representation_names)
        self.ranks = tuple(int(value) for value in ranks)
        self.selector_names = tuple(selector_names)
        self.target_tokens = int(target_tokens)
        self.block_tokens = int(block_tokens)
        self.vocab_size = int(vocab_size)
        self.sequence_length = int(sequence_length)
        self.reservoir_size = int(reservoir_size)
        self.bins = int(bins)
        self.metadata = dict(metadata or {})
        self.total = 0
        self.block_index = 0
        self.block_count = 0
        self.sample_count = 0
        self.block_hash = hashlib.sha256()
        self.block_hashes = []
        self.rng = np.random.default_rng(seed)
        self.reservoir_keys = np.empty(0, dtype=np.float64)
        self.reservoir_global = np.empty(0, dtype=np.int64)
        self.reservoir_token = np.empty(0, dtype=np.int32)
        self.reservoir_position = np.empty(0, dtype=np.int16)
        shape = (len(self.representation_names), len(self.ranks), len(self.selector_names), 0)
        self.reservoir_scores = np.empty(shape, dtype=np.float16)
        self.sample_tmp = self.output_dir / "samples.jsonl.inprogress"
        if self.sample_tmp.exists():
            self.sample_tmp.unlink()
        self._reset_block()

    def _reset_block(self):
        shape = (len(self.representation_names), len(self.ranks), len(self.selector_names))
        self.hist = np.zeros(shape + (self.bins,), dtype=np.int64)
        self.score_sum = np.zeros(shape, dtype=np.float64)
        self.score_sumsq = np.zeros(shape, dtype=np.float64)
        self.token_counts = np.zeros(self.vocab_size, dtype=np.int64)
        # Full-vocabulary bias summaries are retained for stable scores only.
        self.stable_token_score_sum = np.zeros(
            (len(self.ranks), len(self.selector_names), self.vocab_size), dtype=np.float64
        )
        self.position_counts = np.zeros(self.sequence_length, dtype=np.int64)
        self.stable_position_score_sum = np.zeros(
            (len(self.ranks), len(self.selector_names), self.sequence_length), dtype=np.float64
        )

    def remaining(self):
        return self.target_tokens - self.total

    def record_samples(self, tokens, loss_mask):
        rows = tokens.detach().cpu().numpy()
        masks = loss_mask.detach().cpu().numpy().astype(bool)
        with self.sample_tmp.open("a", encoding="utf-8") as handle:
            for row, mask in zip(rows, masks):
                valid = np.asarray(row[mask], dtype=np.int32)
                positions = np.flatnonzero(mask).astype(np.int16)
                record = {
                    "sample_index": self.sample_count,
                    "valid_tokens": int(valid.size),
                    "token_sha256": hashlib.sha256(valid.tobytes()).hexdigest(),
                    "position_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                self.sample_count += 1

    def _update_reservoir(self, token_ids, positions, scores):
        n = int(token_ids.numel())
        if self.reservoir_size <= 0 or n == 0:
            return
        new_keys = self.rng.random(n)
        all_keys = np.concatenate((self.reservoir_keys, new_keys))
        keep_count = min(self.reservoir_size, all_keys.size)
        keep = np.argpartition(all_keys, keep_count - 1)[:keep_count]
        old_n = self.reservoir_keys.size
        current_scores = scores.detach().cpu().numpy().astype(np.float16)
        merged_scores = np.concatenate((self.reservoir_scores, current_scores), axis=-1)
        merged_global = np.concatenate((self.reservoir_global, np.arange(self.total, self.total + n)))
        merged_token = np.concatenate((self.reservoir_token, token_ids.detach().cpu().numpy().astype(np.int32)))
        merged_position = np.concatenate((self.reservoir_position, positions.detach().cpu().numpy().astype(np.int16)))
        self.reservoir_keys = all_keys[keep]
        self.reservoir_scores = merged_scores[..., keep]
        self.reservoir_global = merged_global[keep]
        self.reservoir_token = merged_token[keep]
        self.reservoir_position = merged_position[keep]

    def update(self, token_ids, positions, scores):
        """scores: [representation, rank, selector, token], values expected in [0,1]."""
        offset = 0
        n_total = min(int(token_ids.numel()), self.remaining())
        while offset < n_total:
            take = min(n_total - offset, self.block_tokens - self.block_count)
            sl = slice(offset, offset + take)
            ids = token_ids[sl]
            pos = positions[sl]
            values = scores[..., sl].float().clamp(0.0, 1.0)
            values_np = values.detach().cpu().numpy()
            for rep in range(values.shape[0]):
                for rank in range(values.shape[1]):
                    for selector in range(values.shape[2]):
                        row = values[rep, rank, selector]
                        buckets = torch.clamp((row * self.bins).long(), max=self.bins - 1)
                        self.hist[rep, rank, selector] += torch.bincount(
                            buckets, minlength=self.bins
                        ).cpu().numpy()
                        self.score_sum[rep, rank, selector] += float(row.double().sum().item())
                        self.score_sumsq[rep, rank, selector] += float(row.double().square().sum().item())
            ids_np = ids.detach().cpu().numpy().astype(np.int64)
            pos_np = pos.detach().cpu().numpy().astype(np.int64)
            self.token_counts += np.bincount(ids_np, minlength=self.vocab_size)[:self.vocab_size]
            self.position_counts += np.bincount(pos_np, minlength=self.sequence_length)[:self.sequence_length]
            stable = values_np[0]
            for rank in range(stable.shape[0]):
                for selector in range(stable.shape[1]):
                    np.add.at(self.stable_token_score_sum[rank, selector], ids_np, stable[rank, selector])
                    np.add.at(self.stable_position_score_sum[rank, selector], pos_np, stable[rank, selector])
            self.block_hash.update(ids_np.astype(np.int32).tobytes())
            self.block_hash.update(pos_np.astype(np.int16).tobytes())
            self._update_reservoir(ids, pos, values)
            self.total += take
            self.block_count += take
            offset += take
            if self.block_count == self.block_tokens:
                self._flush_block()

    def _flush_block(self):
        path = self.output_dir / f"block_{self.block_index:03d}.npz"
        np.savez(
            path,
            count=np.asarray(self.block_count, dtype=np.int64),
            hist=self.hist,
            score_sum=self.score_sum,
            score_sumsq=self.score_sumsq,
            token_counts=self.token_counts,
            stable_token_score_sum=self.stable_token_score_sum,
            position_counts=self.position_counts,
            stable_position_score_sum=self.stable_position_score_sum,
        )
        self.block_hashes.append({
            "block": self.block_index,
            "tokens": self.block_count,
            "sha256": self.block_hash.hexdigest(),
        })
        self.block_index += 1
        self.block_count = 0
        self.block_hash = hashlib.sha256()
        self._reset_block()

    def finalize(self):
        if self.total != self.target_tokens or self.block_count:
            raise RuntimeError(f"incomplete score stream: {self.total}/{self.target_tokens}")
        np.savez(
            self.output_dir / "score_reservoir.npz",
            priority=self.reservoir_keys,
            global_indices=self.reservoir_global,
            token_ids=self.reservoir_token,
            positions=self.reservoir_position,
            scores=self.reservoir_scores,
            representation_names=np.asarray(self.representation_names),
            ranks=np.asarray(self.ranks, dtype=np.int16),
            selector_names=np.asarray(self.selector_names),
        )
        os.replace(self.sample_tmp, self.output_dir / "samples.jsonl")
        result = {
            **self.metadata,
            "target_tokens": self.target_tokens,
            "block_tokens": self.block_tokens,
            "blocks": self.block_hashes,
            "representations": list(self.representation_names),
            "ranks": list(self.ranks),
            "selectors": list(self.selector_names),
            "bins": self.bins,
            "reservoir_size": int(self.reservoir_keys.size),
            "encountered_samples": self.sample_count,
        }
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        return result
