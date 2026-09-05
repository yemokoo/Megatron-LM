"""Token-occurrence metrics for paired residual layer-output forwards.

The model-side runner supplies the exact same Code batch to a frozen reference
checkpoint and a current checkpoint.  This module computes float32 metrics,
writes restartable sample-contiguous shards, and keeps only a deterministic
bounded reservoir of raw hidden vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


SCHEMA_VERSION = 2
METRIC_NAMES = (
    "cosine",
    "relative_l2",
    "symmetric_relative_l2",
    "log_norm_ratio",
    "delta_mse",
    "feature_centered_cosine",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_token_pair_metrics(
    reference: torch.Tensor,
    current: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Compute the six explicitly requested shard metrics in float32."""
    if reference.shape != current.shape or reference.ndim != 2:
        raise ValueError(
            f"expected aligned [token, hidden] tensors, got "
            f"reference={tuple(reference.shape)} current={tuple(current.shape)}"
        )
    x = reference.float()
    y = current.float()
    delta = y - x
    xnorm = torch.linalg.vector_norm(x, dim=-1)
    ynorm = torch.linalg.vector_norm(y, dim=-1)
    dnorm = torch.linalg.vector_norm(delta, dim=-1)
    cosine = (x * y).sum(dim=-1) / (xnorm * ynorm + eps)

    xc = x - x.mean(dim=-1, keepdim=True)
    yc = y - y.mean(dim=-1, keepdim=True)
    xcnorm = torch.linalg.vector_norm(xc, dim=-1)
    ycnorm = torch.linalg.vector_norm(yc, dim=-1)
    centered_cosine = (xc * yc).sum(dim=-1) / (xcnorm * ycnorm + eps)

    result = {
        "cosine": cosine.clamp(-1.0, 1.0),
        "relative_l2": dnorm / (xnorm + eps),
        "symmetric_relative_l2": 2.0 * dnorm / (xnorm + ynorm + eps),
        "log_norm_ratio": torch.log((ynorm + eps) / (xnorm + eps)),
        "delta_mse": delta.square().mean(dim=-1),
        "feature_centered_cosine": centered_cosine.clamp(-1.0, 1.0),
    }
    return {name: value.float() for name, value in result.items()}


def derive_requested_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Recover the user-facing redundant metrics without approximation."""
    norm_ratio = np.exp(metrics["log_norm_ratio"].astype(np.float64)).astype(np.float32)
    return {
        "cosine": metrics["cosine"],
        "cosine_distance": 1.0 - metrics["cosine"],
        "relative_l2": metrics["relative_l2"],
        "symmetric_relative_l2": metrics["symmetric_relative_l2"],
        "norm_ratio": norm_ratio,
        "log_norm_ratio": metrics["log_norm_ratio"],
        "delta_mse": metrics["delta_mse"],
        "feature_centered_cosine": metrics["feature_centered_cosine"],
    }


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic priorities independent of batching, workers, and resume."""
    with np.errstate(over="ignore"):
        z = (
            values.astype(np.uint64, copy=False)
            + np.uint64(seed)
            + np.uint64(0x9E3779B97F4A7C15)
        )
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


class DeterministicHiddenReservoir:
    """Keep the globally smallest deterministic token priorities per worker."""

    def __init__(self, size: int, layers: Iterable[int], hidden_size: int, seed: int) -> None:
        self.size = int(size)
        self.layers = tuple(int(layer) for layer in layers)
        self.hidden_size = int(hidden_size)
        self.seed = int(seed)
        self.priority = np.empty(0, dtype=np.uint64)
        self.sample_ids = np.empty(0, dtype=np.int64)
        self.positions = np.empty(0, dtype=np.uint16)
        self.token_ids = np.empty(0, dtype=np.int32)
        shape = (0, len(self.layers), self.hidden_size)
        self.reference = np.empty(shape, dtype=np.float16)
        self.current = np.empty(shape, dtype=np.float16)
        self.delta = np.empty(shape, dtype=np.float16)

    def update(
        self,
        sample_ids: np.ndarray,
        positions: np.ndarray,
        token_ids: np.ndarray,
        reference_by_layer: Dict[int, torch.Tensor],
        current_by_layer: Dict[int, torch.Tensor],
        valid_flat_indices: np.ndarray,
        sequence_length: int,
    ) -> None:
        if self.size <= 0 or valid_flat_indices.size == 0:
            return
        occurrence = (
            np.repeat(sample_ids, sequence_length)[valid_flat_indices].astype(np.uint64)
            * np.uint64(sequence_length)
            + positions.astype(np.uint64)
        )
        new_priority = _splitmix64(occurrence, self.seed)
        old_count = self.priority.size
        combined_priority = np.concatenate((self.priority, new_priority))
        keep_count = min(self.size, combined_priority.size)
        if keep_count == 0:
            return
        keep = np.argpartition(combined_priority, keep_count - 1)[:keep_count]
        keep = keep[np.argsort(combined_priority[keep], kind="stable")]
        new_selected = keep[keep >= old_count] - old_count

        if new_selected.size:
            source_rows = valid_flat_indices[new_selected]
            source_tensor = torch.as_tensor(
                source_rows, dtype=torch.long, device=next(iter(reference_by_layer.values())).device
            )
            reference_rows = []
            current_rows = []
            for layer in self.layers:
                ref = reference_by_layer[layer]
                cur = current_by_layer[layer]
                reference_rows.append(ref.index_select(0, source_tensor).float().cpu())
                if cur.device != source_tensor.device:
                    cur_index = source_tensor.to(cur.device)
                else:
                    cur_index = source_tensor
                current_rows.append(cur.index_select(0, cur_index).float().cpu())
            stacked_reference = torch.stack(reference_rows, dim=1)
            stacked_current = torch.stack(current_rows, dim=1)
            new_reference = stacked_reference.numpy().astype(np.float16)
            new_current = stacked_current.numpy().astype(np.float16)
            new_delta = (stacked_current - stacked_reference).numpy().astype(np.float16)
        else:
            new_reference = np.empty((0, len(self.layers), self.hidden_size), dtype=np.float16)
            new_current = np.empty_like(new_reference)
            new_delta = np.empty_like(new_reference)

        merged_sample_ids = np.concatenate(
            (self.sample_ids, np.repeat(sample_ids, sequence_length)[valid_flat_indices])
        )
        merged_positions = np.concatenate((self.positions, positions))
        merged_token_ids = np.concatenate((self.token_ids, token_ids))
        self.priority = combined_priority[keep]
        self.sample_ids = merged_sample_ids[keep]
        self.positions = merged_positions[keep]
        self.token_ids = merged_token_ids[keep]
        output_shape = (keep_count, len(self.layers), self.hidden_size)
        output_reference = np.empty(output_shape, dtype=np.float16)
        output_current = np.empty(output_shape, dtype=np.float16)
        output_delta = np.empty(output_shape, dtype=np.float16)
        old_mask = keep < old_count
        if old_mask.any():
            output_reference[old_mask] = self.reference[keep[old_mask]]
            output_current[old_mask] = self.current[keep[old_mask]]
            output_delta[old_mask] = self.delta[keep[old_mask]]
        if (~old_mask).any():
            # new_reference/current were materialized in the order in which new
            # rows occur in `keep`, so they fill the complementary rows exactly.
            output_reference[~old_mask] = new_reference
            output_current[~old_mask] = new_current
            output_delta[~old_mask] = new_delta
        self.reference = output_reference
        self.current = output_current
        self.delta = output_delta

    def restore(self, payload: dict) -> None:
        expected_layers = np.asarray(self.layers, dtype=np.int16)
        if not np.array_equal(payload["layer_numbers"], expected_layers):
            raise RuntimeError("reservoir layer mismatch on resume")
        self.priority = payload["priority"].astype(np.uint64, copy=True)
        self.sample_ids = payload["sample_ids"].astype(np.int64, copy=True)
        self.positions = payload["positions"].astype(np.uint16, copy=True)
        self.token_ids = payload["token_ids"].astype(np.int32, copy=True)
        self.reference = payload["reference"].astype(np.float16, copy=True)
        self.current = payload["current"].astype(np.float16, copy=True)
        self.delta = payload["delta"].astype(np.float16, copy=True)
        count = self.priority.size
        expected = (count, len(self.layers), self.hidden_size)
        if (
            self.reference.shape != expected
            or self.current.shape != expected
            or self.delta.shape != expected
        ):
            raise RuntimeError(
                f"reservoir hidden shape mismatch on resume: "
                f"reference={self.reference.shape} current={self.current.shape} expected={expected}"
            )

    def payload(self) -> dict:
        return {
            "layer_numbers": np.asarray(self.layers, dtype=np.int16),
            "priority": self.priority,
            "sample_ids": self.sample_ids,
            "positions": self.positions,
            "token_ids": self.token_ids,
            "reference": self.reference,
            "current": self.current,
            "delta": self.delta,
        }


class TokenMetricShardWriter:
    """Restartable, sample-contiguous token metric writer."""

    def __init__(
        self,
        output_dir: str,
        *,
        layers: Iterable[int],
        hidden_size: int,
        sequence_length: int,
        partition_start_sample: int,
        partition_samples: int,
        total_samples: int,
        shard_samples: int,
        reservoir_size: int,
        seed: int,
        metadata: dict,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.layers = tuple(int(layer) for layer in layers)
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        self.partition_start_sample = int(partition_start_sample)
        self.partition_samples = int(partition_samples)
        self.total_samples = int(total_samples)
        self.shard_samples = int(shard_samples)
        if not self.layers or self.layers != tuple(sorted(set(self.layers))):
            raise ValueError(f"layers must be sorted and unique: {self.layers}")
        if self.partition_start_sample < 0 or self.partition_samples <= 0:
            raise ValueError("partition sample range must be positive")
        if self.partition_start_sample + self.partition_samples > self.total_samples:
            raise ValueError("partition exceeds total samples")
        if self.shard_samples <= 0:
            raise ValueError("shard_samples must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.output_dir / "token_metrics"
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.reservoir_dir = self.output_dir / "reservoir"
        self.reservoir_dir.mkdir(parents=True, exist_ok=True)
        self.run_config = {
            "schema_version": SCHEMA_VERSION,
            "metric_names": list(METRIC_NAMES),
            "metric_dtype": "float32",
            "derived_metrics": {
                "cosine_distance": "1-cosine",
                "norm_ratio": "exp(log_norm_ratio)",
            },
            "layers": list(self.layers),
            "hidden_size": self.hidden_size,
            "sequence_length": self.sequence_length,
            "partition_start_sample": self.partition_start_sample,
            "partition_samples": self.partition_samples,
            "total_samples": self.total_samples,
            "shard_samples": self.shard_samples,
            "seed": int(seed),
            **metadata,
        }
        self.config_hash = _json_hash(self.run_config)
        self._initialize_or_validate_config()
        self.completed_samples, self.next_shard = self._scan_committed_shards()
        self.buffer = {
            "sample_ids": [],
            "input_token_ids": [],
            "label_token_ids": [],
            "valid_mask": [],
            **{name: [] for name in METRIC_NAMES},
        }
        self.buffered_samples = 0
        self.valid_tokens = self._valid_tokens_from_sidecars()
        metric_shape = (len(METRIC_NAMES), len(self.layers))
        self.metric_count = np.zeros(metric_shape, dtype=np.int64)
        self.metric_sum = np.zeros(metric_shape, dtype=np.float64)
        self.metric_min = np.full(metric_shape, np.inf, dtype=np.float64)
        self.metric_max = np.full(metric_shape, -np.inf, dtype=np.float64)
        self.output_bytes = 0
        self._restore_metric_summaries()
        self.nonfinite_count = 0
        self.started_at = time.time()
        self.reservoir = DeterministicHiddenReservoir(
            reservoir_size, self.layers, self.hidden_size, seed
        )
        self._restore_reservoir_progress()
        self.initial_completed_samples = self.completed_samples

    @property
    def remaining_samples(self) -> int:
        return self.partition_samples - self.completed_samples - self.buffered_samples

    @property
    def next_global_sample(self) -> int:
        return self.partition_start_sample + self.completed_samples + self.buffered_samples

    @property
    def is_complete(self) -> bool:
        return self.completed_samples == self.partition_samples

    def _initialize_or_validate_config(self) -> None:
        path = self.output_dir / "run_metadata.json"
        if path.exists():
            with open(path, encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != self.run_config:
                raise RuntimeError(
                    f"resume metadata mismatch in {path}: "
                    f"existing_hash={_json_hash(existing)} requested_hash={self.config_hash}"
                )
        else:
            _atomic_json(path, self.run_config)
        for temporary in sorted(self.shard_dir.glob("*.inprogress*")):
            failed = temporary.with_name(
                temporary.name + ".failed." + time.strftime("%Y%m%d-%H%M%S")
            )
            os.replace(temporary, failed)

    def _scan_committed_shards(self) -> tuple[int, int]:
        sidecar_stems = {path.stem for path in self.shard_dir.glob("shard_*.json")}
        for data_path in sorted(self.shard_dir.glob("shard_*.npz")):
            if data_path.stem not in sidecar_stems:
                failed = data_path.with_name(
                    data_path.name + ".failed." + time.strftime("%Y%m%d-%H%M%S")
                )
                os.replace(data_path, failed)
        completed = 0
        expected_shard = 0
        expected_sample = self.partition_start_sample
        for sidecar_path in sorted(self.shard_dir.glob("shard_*.json")):
            shard_number = int(sidecar_path.stem.split("_")[-1])
            if shard_number != expected_shard:
                raise RuntimeError(
                    f"non-contiguous shard manifests: expected {expected_shard}, got {shard_number}"
                )
            with open(sidecar_path, encoding="utf-8") as handle:
                row = json.load(handle)
            data_path = self.shard_dir / row["file"]
            if not data_path.is_file():
                raise RuntimeError(f"manifest references missing shard: {data_path}")
            if row["config_sha256"] != self.config_hash:
                raise RuntimeError(f"config hash mismatch: {sidecar_path}")
            if row["start_sample"] != expected_sample:
                raise RuntimeError(
                    f"sample gap/overlap at {sidecar_path}: "
                    f"expected={expected_sample} got={row['start_sample']}"
                )
            expected_sample = row["end_sample"]
            completed += row["samples"]
            expected_shard += 1
        if completed > self.partition_samples:
            raise RuntimeError("committed shards exceed partition size")
        return completed, expected_shard

    def _valid_tokens_from_sidecars(self) -> int:
        total = 0
        for sidecar_path in self.shard_dir.glob("shard_*.json"):
            with open(sidecar_path, encoding="utf-8") as handle:
                total += int(json.load(handle)["valid_tokens"])
        return total

    def _restore_reservoir_progress(self) -> None:
        path = self.reservoir_dir / "reservoir_progress.npz"
        if not path.exists():
            if self.completed_samples:
                # A process can die after the token-shard commit marker is
                # written but before its reservoir checkpoint is replaced.
                # Roll the token transaction back and deterministically
                # re-forward it instead of silently losing raw reservoir rows.
                self._rollback_committed_shards(0)
            return
        with np.load(path, allow_pickle=False) as data:
            completed = int(data["completed_samples"])
            if completed < self.completed_samples:
                self._rollback_committed_shards(completed)
            elif completed > self.completed_samples:
                raise RuntimeError(
                    f"reservoir is ahead of committed token shards: reservoir={completed} "
                    f"token_shards={self.completed_samples}"
                )
            if completed != self.completed_samples:
                raise RuntimeError(
                    f"reservoir/token progress mismatch: reservoir={completed} "
                    f"token_shards={self.completed_samples}"
                )
            self.reservoir.restore({name: data[name] for name in data.files})

    def _rollback_committed_shards(self, completed_samples: int) -> None:
        """Quarantine shard commits newer than a durable reservoir checkpoint."""
        completed_samples = int(completed_samples)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        retained = 0
        for sidecar_path in sorted(self.shard_dir.glob("shard_*.json")):
            with open(sidecar_path, encoding="utf-8") as handle:
                row = json.load(handle)
            relative_end = int(row["end_sample"]) - self.partition_start_sample
            if relative_end <= completed_samples:
                retained = relative_end
                continue
            data_path = self.shard_dir / row["file"]
            if data_path.exists():
                os.replace(data_path, data_path.with_name(data_path.name + f".rollback.{timestamp}"))
            os.replace(
                sidecar_path,
                sidecar_path.with_name(sidecar_path.name + f".rollback.{timestamp}"),
            )
        if retained != completed_samples:
            raise RuntimeError(
                f"reservoir progress {completed_samples} is not a committed shard boundary "
                f"(last retained={retained})"
            )
        self.completed_samples, self.next_shard = self._scan_committed_shards()
        self.valid_tokens = self._valid_tokens_from_sidecars()
        self.metric_count.fill(0)
        self.metric_sum.fill(0.0)
        self.metric_min.fill(np.inf)
        self.metric_max.fill(-np.inf)
        self.output_bytes = 0
        self._restore_metric_summaries()

    def _restore_metric_summaries(self) -> None:
        for sidecar_path in sorted(self.shard_dir.glob("shard_*.json")):
            with open(sidecar_path, encoding="utf-8") as handle:
                row = json.load(handle)
            self.output_bytes += int(row.get("bytes", 0))
            summary = row.get("metric_summary")
            if summary is None:
                continue
            self._accumulate_metric_summary(summary)

    def _accumulate_metric_summary(self, summary: dict) -> None:
        for metric_index, name in enumerate(METRIC_NAMES):
            row = summary[name]
            count = int(row["count"])
            if count == 0:
                continue
            self.metric_count[metric_index] += count
            self.metric_sum[metric_index] += np.asarray(row["sum"], dtype=np.float64)
            self.metric_min[metric_index] = np.minimum(
                self.metric_min[metric_index], np.asarray(row["min"], dtype=np.float64)
            )
            self.metric_max[metric_index] = np.maximum(
                self.metric_max[metric_index], np.asarray(row["max"], dtype=np.float64)
            )

    def _metric_summary_for_payload(self, payload: dict) -> dict:
        valid = payload["valid_mask"].astype(bool, copy=False)
        summary = {}
        for name in METRIC_NAMES:
            values = payload[name][valid]
            if values.size == 0:
                zeros = [0.0] * len(self.layers)
                summary[name] = {"count": 0, "sum": zeros, "min": zeros, "max": zeros}
                continue
            summary[name] = {
                "count": int(values.shape[0]),
                "sum": values.astype(np.float64).sum(axis=0).tolist(),
                "min": values.min(axis=0).astype(np.float64).tolist(),
                "max": values.max(axis=0).astype(np.float64).tolist(),
            }
        return summary

    def _save_reservoir_progress(self) -> None:
        path = self.reservoir_dir / "reservoir_progress.npz"
        temporary = path.with_name(path.name + ".inprogress")
        payload = {
            **self.reservoir.payload(),
            "completed_samples": np.asarray(self.completed_samples, dtype=np.int64),
        }
        with open(temporary, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def update(
        self,
        sample_ids: torch.Tensor,
        input_token_ids: torch.Tensor,
        label_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        reference_by_layer: Dict[int, torch.Tensor],
        current_by_layer: Dict[int, torch.Tensor],
    ) -> None:
        if sample_ids.ndim != 1 or input_token_ids.ndim != 2:
            raise ValueError("sample_ids must be [batch] and tokens must be [batch, sequence]")
        batch, sequence = input_token_ids.shape
        if sequence != self.sequence_length or sample_ids.numel() != batch:
            raise ValueError("batch/sample/sequence shape mismatch")
        if label_token_ids.shape != input_token_ids.shape or valid_mask.shape != input_token_ids.shape:
            raise ValueError("token/label/mask shape mismatch")
        expected = torch.arange(
            self.next_global_sample,
            self.next_global_sample + batch,
            dtype=sample_ids.dtype,
            device=sample_ids.device,
        )
        if not torch.equal(sample_ids, expected):
            raise RuntimeError(
                f"non-contiguous sample IDs: expected start={self.next_global_sample}, "
                f"got={sample_ids.detach().cpu().tolist()[:4]}"
            )
        if batch > self.remaining_samples:
            raise RuntimeError(
                f"batch overruns partition: batch={batch} remaining={self.remaining_samples}"
            )

        flat_count = batch * sequence
        metric_cpu = {name: [] for name in METRIC_NAMES}
        for layer in self.layers:
            if layer not in reference_by_layer or layer not in current_by_layer:
                raise RuntimeError(f"missing captured layer {layer}")
            reference = reference_by_layer[layer]
            current = current_by_layer[layer]
            if reference.shape != (flat_count, self.hidden_size):
                raise RuntimeError(
                    f"reference layer {layer} shape mismatch: {tuple(reference.shape)}"
                )
            if current.shape != reference.shape:
                raise RuntimeError(
                    f"current layer {layer} shape mismatch: {tuple(current.shape)}"
                )
            if reference.device != current.device:
                reference = reference.to(current.device, non_blocking=True)
            layer_metrics = compute_token_pair_metrics(reference, current)
            for name in METRIC_NAMES:
                metric_cpu[name].append(layer_metrics[name].detach().cpu())

        mask_cpu = valid_mask.detach().cpu().bool()
        metric_arrays = {}
        for name in METRIC_NAMES:
            values = torch.stack(metric_cpu[name], dim=-1).reshape(
                batch, sequence, len(self.layers)
            )
            invalid = ~mask_cpu.unsqueeze(-1)
            values.masked_fill_(invalid, 0.0)
            nonfinite = int((~torch.isfinite(values)).sum().item())
            self.nonfinite_count += nonfinite
            if nonfinite:
                raise RuntimeError(f"non-finite {name} values: {nonfinite}")
            metric_arrays[name] = values.numpy().astype(np.float32, copy=False)

        sample_np = sample_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        input_np = input_token_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        label_np = label_token_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        mask_np = mask_cpu.numpy().astype(np.uint8, copy=False)
        # Apply buffer and reservoir updates in shard-aligned prefixes.  This
        # matters for crash recovery: every reservoir_progress checkpoint must
        # contain exactly the token range committed by the matching sidecars,
        # never an uncommitted suffix from a batch that crossed a boundary.
        cursor = 0
        while cursor < batch:
            room = self.shard_samples - self.buffered_samples
            segment_samples = min(room, batch - cursor)
            end = cursor + segment_samples
            self.buffer["sample_ids"].append(sample_np[cursor:end])
            self.buffer["input_token_ids"].append(input_np[cursor:end])
            self.buffer["label_token_ids"].append(label_np[cursor:end])
            self.buffer["valid_mask"].append(mask_np[cursor:end])
            for name in METRIC_NAMES:
                self.buffer[name].append(metric_arrays[name][cursor:end])

            segment_mask = mask_np[cursor:end]
            valid_flat_indices = np.flatnonzero(segment_mask.reshape(-1)).astype(np.int64)
            positions = (valid_flat_indices % sequence).astype(np.uint16)
            valid_input = input_np[cursor:end].reshape(-1)[valid_flat_indices]
            flat_start = cursor * sequence
            flat_end = end * sequence
            segment_reference = {
                layer: reference_by_layer[layer][flat_start:flat_end] for layer in self.layers
            }
            segment_current = {
                layer: current_by_layer[layer][flat_start:flat_end] for layer in self.layers
            }
            self.reservoir.update(
                sample_np[cursor:end],
                positions,
                valid_input,
                segment_reference,
                segment_current,
                valid_flat_indices,
                sequence,
            )
            self.buffered_samples += segment_samples
            cursor = end
            if self.buffered_samples == self.shard_samples:
                self._flush(self.shard_samples)

    def _take_buffer(self, count: int) -> dict:
        result = {}
        for name, chunks in self.buffer.items():
            joined = np.concatenate(chunks, axis=0)
            result[name] = joined[:count]
            remaining = joined[count:]
            self.buffer[name] = [remaining] if remaining.shape[0] else []
        self.buffered_samples -= count
        return result

    def _flush(self, count: int) -> None:
        payload = self._take_buffer(count)
        start_sample = int(payload["sample_ids"][0])
        expected_ids = np.arange(start_sample, start_sample + count, dtype=np.int64)
        if not np.array_equal(payload["sample_ids"], expected_ids):
            raise RuntimeError("sample IDs are not contiguous inside shard")
        layer_numbers = np.asarray(self.layers, dtype=np.int16)
        payload = {"layer_numbers": layer_numbers, **payload}
        final_name = f"shard_{self.next_shard:06d}.npz"
        final_path = self.shard_dir / final_name
        temporary = self.shard_dir / (final_name + ".inprogress")
        with open(temporary, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._validate_shard(temporary, count)
        os.replace(temporary, final_path)

        token_hash = hashlib.sha256()
        token_hash.update(payload["input_token_ids"].tobytes())
        token_hash.update(payload["label_token_ids"].tobytes())
        token_hash.update(payload["valid_mask"].tobytes())
        valid_tokens = int(payload["valid_mask"].sum())
        metric_summary = self._metric_summary_for_payload(payload)
        sidecar = {
            "schema_version": SCHEMA_VERSION,
            "file": final_name,
            "shard": self.next_shard,
            "config_sha256": self.config_hash,
            "start_sample": start_sample,
            "end_sample": start_sample + count,
            "samples": count,
            "valid_tokens": valid_tokens,
            "logical_token_sha256": token_hash.hexdigest(),
            "bytes": final_path.stat().st_size,
            "metric_summary": metric_summary,
        }
        _atomic_json(self.shard_dir / f"shard_{self.next_shard:06d}.json", sidecar)
        self.completed_samples += count
        self.valid_tokens += valid_tokens
        self.output_bytes += int(sidecar["bytes"])
        self._accumulate_metric_summary(metric_summary)
        self.next_shard += 1
        self._save_reservoir_progress()
        self._write_progress()

    def _validate_shard(self, path: Path, samples: int) -> None:
        with np.load(path, allow_pickle=False) as data:
            expected_dense = (samples, self.sequence_length)
            expected_metric = expected_dense + (len(self.layers),)
            if not np.array_equal(data["layer_numbers"], np.asarray(self.layers, dtype=np.int16)):
                raise RuntimeError(f"layer mismatch in {path}")
            if data["sample_ids"].shape != (samples,):
                raise RuntimeError(f"sample ID shape mismatch in {path}")
            for name in ("input_token_ids", "label_token_ids", "valid_mask"):
                if data[name].shape != expected_dense:
                    raise RuntimeError(f"{name} shape mismatch in {path}: {data[name].shape}")
            for name in METRIC_NAMES:
                values = data[name]
                if values.shape != expected_metric or values.dtype != np.float32:
                    raise RuntimeError(
                        f"{name} shape/dtype mismatch in {path}: {values.shape}/{values.dtype}"
                    )
                if not np.isfinite(values).all():
                    raise RuntimeError(f"non-finite {name} in {path}")
                if name in ("cosine", "feature_centered_cosine") and (
                    float(values.min()) < -1.00001 or float(values.max()) > 1.00001
                ):
                    raise RuntimeError(f"invalid cosine range in {path}: {name}")
                if name in ("relative_l2", "symmetric_relative_l2", "delta_mse") and float(values.min()) < 0:
                    raise RuntimeError(f"negative {name} in {path}")
                if name == "symmetric_relative_l2" and float(values.max()) > 2.00001:
                    raise RuntimeError(f"invalid symmetric relative L2 range in {path}")

    def _write_progress(self) -> None:
        payload = self.progress_snapshot()
        payload["updated_at_unix"] = time.time()
        _atomic_json(self.output_dir / "progress.json", payload)

    def progress_snapshot(self) -> dict:
        elapsed = max(time.time() - self.started_at, 1e-9)
        processed_this_run = max(self.completed_samples - self.initial_completed_samples, 0)
        rate = processed_this_run * self.sequence_length / elapsed
        remaining_tokens = (self.partition_samples - self.completed_samples) * self.sequence_length
        running = {}
        for metric_index, name in enumerate(METRIC_NAMES):
            count = self.metric_count[metric_index]
            mean = np.divide(
                self.metric_sum[metric_index],
                count,
                out=np.zeros(len(self.layers), dtype=np.float64),
                where=count > 0,
            )
            minimum = np.where(count > 0, self.metric_min[metric_index], 0.0)
            maximum = np.where(count > 0, self.metric_max[metric_index], 0.0)
            running[name] = {
                "mean": mean.tolist(),
                "min": minimum.tolist(),
                "max": maximum.tolist(),
            }
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                "gpu_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            }
        return {
            "config_sha256": self.config_hash,
            "completed_samples": self.completed_samples,
            "partition_samples": self.partition_samples,
            "completed_valid_tokens": self.valid_tokens,
            "next_global_sample": self.partition_start_sample + self.completed_samples,
            "next_shard": self.next_shard,
            "tokens_per_second_this_process": rate,
            "eta_seconds": remaining_tokens / rate if rate > 0 else None,
            "nonfinite_count": self.nonfinite_count,
            "output_bytes": self.output_bytes,
            "metric_running_by_layer": running,
            **gpu_memory,
        }

    def finalize(self) -> dict:
        if self.buffered_samples:
            self._flush(self.buffered_samples)
        if self.completed_samples != self.partition_samples:
            raise RuntimeError(
                f"incomplete partition: {self.completed_samples}/{self.partition_samples} samples"
            )
        reservoir_path = self.reservoir_dir / "hidden_reservoir.npz"
        temporary = reservoir_path.with_name(reservoir_path.name + ".inprogress")
        with open(temporary, "wb") as handle:
            np.savez(handle, **self.reservoir.payload())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, reservoir_path)
        result = {
            **self.run_config,
            "config_sha256": self.config_hash,
            "completed_samples": self.completed_samples,
            "completed_valid_tokens": self.valid_tokens,
            "shards": self.next_shard,
            "reservoir_tokens": int(self.reservoir.priority.size),
            "completed": True,
        }
        _atomic_json(self.output_dir / "metadata.json", result)
        self._write_progress()
        return result


def storage_estimate(total_tokens: int, layers: int) -> dict:
    metric_bytes = int(total_tokens) * int(layers) * len(METRIC_NAMES) * 4
    identity_bytes = int(total_tokens) * (4 + 4 + 1)
    return {
        "tokens": int(total_tokens),
        "layers": int(layers),
        "stored_metric_arrays": len(METRIC_NAMES),
        "metric_bytes": metric_bytes,
        "dense_identity_upper_bound_bytes": identity_bytes,
        "total_upper_bound_bytes": metric_bytes + identity_bytes,
        "metric_gib": metric_bytes / (1024 ** 3),
        "total_upper_bound_gib": (metric_bytes + identity_bytes) / (1024 ** 3),
    }
