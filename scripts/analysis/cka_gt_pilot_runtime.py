"""CPU-testable runtime components for the CKA GT pilot.

The production entrypoint in ``pretrain_gpt.py`` owns checkpoint loading,
distributed model forwarding, and hook registration.  This module owns the
deterministic document-window reader, fp32 metric processing, router summaries,
restartable Arrow storage, and the two-pass reference statistics.  It never
starts training and never stores raw full-corpus hidden states.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from .cka_gt_pilot_core import (
        centered_linear_cka_metrics,
        random_pair_indices,
        random_pair_null_metrics,
        uncentered_token_metrics,
    )
    from .cka_gt_pilot_windows import (
        DOMAIN_IDS,
        SPLIT_IDS,
        STAGE_COVARIANCE_SUBSAMPLE,
        STAGE_KMEANS_RESERVOIR,
        WINDOW_DTYPE,
        chunk_starts,
        load_document_window,
        seeded_rng,
        window_chunk_layout,
    )
except ImportError:  # Direct ``python scripts/analysis/...`` execution.
    from cka_gt_pilot_core import (
        centered_linear_cka_metrics,
        random_pair_indices,
        random_pair_null_metrics,
        uncentered_token_metrics,
    )
    from cka_gt_pilot_windows import (
        DOMAIN_IDS,
        SPLIT_IDS,
        STAGE_COVARIANCE_SUBSAMPLE,
        STAGE_KMEANS_RESERVOIR,
        WINDOW_DTYPE,
        chunk_starts,
        load_document_window,
        seeded_rng,
        window_chunk_layout,
    )


RUNTIME_SCHEMA_VERSION = 1
LAYERS = tuple(range(2, 10))
OLD_EXPERT_COUNT = 8
ROUTER_TOPK = 4
KMEANS_DEFAULTS = {
    "algorithm": "repository_deterministic_minibatch_kmeans_v1",
    "version": 1,
    "n_clusters": 64,
    "reservoir_size": 200_000,
    "batch_size": 4_096,
    "n_init": 1,
    "max_iter": 100,
    "reassignment_ratio": 0.01,
    "random_state": 1234,
    "init": "k-means++",
}

RANDOM_PAIR_VALID = 0
RANDOM_PAIR_NO_CACHE = 1
RANDOM_PAIR_NO_NONSELF_DONOR = 2
RANDOM_PAIR_METRIC_INVALID = 3
RANDOM_PAIR_REASON_NAMES = {
    RANDOM_PAIR_VALID: "valid",
    RANDOM_PAIR_NO_CACHE: "singleton_no_donor_cache",
    RANDOM_PAIR_NO_NONSELF_DONOR: "singleton_no_nonself_full_window_donor",
    RANDOM_PAIR_METRIC_INVALID: "paired_cka_metric_invalid",
}
RANDOM_PAIR_DONOR_CACHE_SIZE = 8


@contextlib.contextmanager
def fp32_metric_context() -> Iterator[dict[str, Any]]:
    """Disable CUDA TF32 for metric matmuls and restore the caller's state."""

    previous = bool(torch.backends.cuda.matmul.allow_tf32)
    audit = {
        "torch_dtype": "float32",
        "cuda_matmul_allow_tf32_before": previous,
        "cuda_matmul_allow_tf32_during": False,
        "cuda_matmul_allow_tf32_restored": None,
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield audit
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
        audit["cuda_matmul_allow_tf32_restored"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )


def build_attention_position_tensors(
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one shared causal mask, restart-at-zero positions, and valid mask."""

    if tokens.ndim != 2 or tokens.shape[1] <= 0:
        raise ValueError(f"tokens must be [batch, length], got {tuple(tokens.shape)}")
    batch, length = tokens.shape
    position_ids = torch.arange(length, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand(batch, -1)
    # Megatron boolean attention masks use True for positions that are masked.
    attention_mask = torch.triu(
        torch.ones((1, 1, length, length), dtype=torch.bool, device=tokens.device),
        diagonal=1,
    )
    valid_mask = torch.ones((batch, length), dtype=torch.bool, device=tokens.device)
    return attention_mask, position_ids, valid_mask


@dataclass(frozen=True)
class WindowBatch:
    batch_index: int
    manifest_indices: np.ndarray
    rows: np.ndarray
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    valid_mask: torch.Tensor


class DocumentWindowBatchReader:
    """Read manifest windows directly from IndexedDataset without padding.

    The deterministic plan groups equal lengths and preserves ``sample_order``
    inside each group.  Length groups are processed largest first, so ordinary
    512-token windows receive full batches while each unusual tail length is
    still forwarded at its exact unpadded length.
    """

    def __init__(
        self,
        dataset: Any,
        manifest_rows: np.ndarray,
        *,
        batch_size: int,
        device: torch.device | str = "cpu",
        start_batch_index: int = 0,
    ) -> None:
        self.dataset = dataset
        self.rows = np.asarray(manifest_rows)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.start_batch_index = int(start_batch_index)
        if self.rows.dtype != WINDOW_DTYPE or self.rows.ndim != 1:
            raise ValueError(
                f"manifest_rows must be one-dimensional WINDOW_DTYPE, got "
                f"shape={self.rows.shape} dtype={self.rows.dtype}"
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.rows.size and np.unique(self.rows["sample_order"]).size != self.rows.size:
            raise ValueError("manifest sample_order values must be unique")

        plan: list[np.ndarray] = []
        lengths = np.unique(self.rows["window_length"])[::-1]
        for length in lengths.tolist():
            indices = np.flatnonzero(self.rows["window_length"] == length)
            order = np.argsort(self.rows["sample_order"][indices], kind="stable")
            indices = indices[order]
            for start in range(0, indices.size, self.batch_size):
                plan.append(indices[start : start + self.batch_size].astype(np.int64))
        self._plan = tuple(plan)
        if not 0 <= self.start_batch_index <= len(self._plan):
            raise ValueError("start_batch_index is outside the deterministic batch plan")

    def __len__(self) -> int:
        return len(self._plan)

    @property
    def plan(self) -> tuple[np.ndarray, ...]:
        return self._plan

    def __iter__(self) -> Iterator[WindowBatch]:
        for batch_index in range(self.start_batch_index, len(self._plan)):
            indices = self._plan[batch_index]
            rows = self.rows[indices]
            unique_lengths = np.unique(rows["window_length"])
            if unique_lengths.size != 1:
                raise AssertionError("same-length batch plan mixed window lengths")
            expected_length = int(unique_lengths[0])
            token_rows = [
                load_document_window(
                    self.dataset,
                    int(row["document_id"]),
                    int(row["window_offset"]),
                    int(row["window_length"]),
                )
                for row in rows
            ]
            if any(value.shape != (expected_length,) for value in token_rows):
                raise RuntimeError("IndexedDataset returned a malformed window")
            token_array = np.stack(token_rows, axis=0).astype(np.int64, copy=False)
            tokens = torch.from_numpy(token_array).to(device=self.device, dtype=torch.long)
            attention_mask, position_ids, valid_mask = build_attention_position_tensors(tokens)
            yield WindowBatch(
                batch_index=batch_index,
                manifest_indices=indices.copy(),
                rows=rows.copy(),
                tokens=tokens,
                attention_mask=attention_mask,
                position_ids=position_ids,
                valid_mask=valid_mask,
            )


def router_diagnostics_from_outputs(
    logits: torch.Tensor,
    scores: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    old_expert_count: int = OLD_EXPERT_COUNT,
    topk: int = ROUTER_TOPK,
    leading_shape: Sequence[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize actual dispatch and full-softmax old-expert preference."""

    if logits.shape != scores.shape or logits.shape != routing_map.shape:
        raise ValueError("logits, scores, and routing_map must have identical shapes")
    if logits.ndim < 2:
        raise ValueError("router tensors must end in an expert dimension")
    expert_count = int(logits.shape[-1])
    if not 0 < old_expert_count < expert_count:
        raise ValueError("old_expert_count must split the expert axis")
    if not 0 < topk <= expert_count:
        raise ValueError("topk must be within the expert axis")

    flat_logits = logits.reshape(-1, expert_count).float()
    flat_scores = scores.reshape(-1, expert_count).float()
    flat_map = routing_map.reshape(-1, expert_count).bool()
    assignments = flat_map.sum(dim=-1)
    if not torch.all(assignments == topk):
        observed = torch.unique(assignments).detach().cpu().tolist()
        raise RuntimeError(
            f"standard router probe expected exactly topk={topk} assignments, got {observed}"
        )
    selected_scores = flat_scores.masked_fill(~flat_map, -torch.inf)
    top_weights, top_ids = torch.topk(selected_scores, k=topk, dim=-1, sorted=True)
    if not torch.isfinite(top_weights).all():
        raise RuntimeError("selected router weights contain non-finite values")
    full_probabilities = torch.softmax(flat_logits, dim=-1, dtype=torch.float32)
    old_full_mass = full_probabilities[:, :old_expert_count].sum(dim=-1)
    old_selected_mass = flat_scores[:, :old_expert_count].sum(dim=-1)

    if leading_shape is None:
        leading_shape = logits.shape[:-1]
    leading_shape = tuple(int(value) for value in leading_shape)
    if math.prod(leading_shape) != flat_logits.shape[0]:
        raise ValueError("leading_shape does not match the flattened router token count")
    return {
        "top4_id": top_ids.reshape(*leading_shape, topk).to(torch.int16),
        "top4_weight": top_weights.reshape(*leading_shape, topk).float(),
        "old_full_mass": old_full_mass.reshape(*leading_shape).float(),
        "old_selected_mass": old_selected_mass.reshape(*leading_shape).float(),
    }


def router_diagnostics_from_input(
    router: Any,
    router_input: torch.Tensor,
    *,
    old_expert_count: int = OLD_EXPERT_COUNT,
    topk: int = ROUTER_TOPK,
) -> dict[str, torch.Tensor]:
    """Run a read-only standard ``mlp.router`` probe from captured input."""

    if getattr(router, "training", False):
        raise RuntimeError("router probe requires eval mode")
    if not hasattr(router, "gating") or not hasattr(router, "routing"):
        raise TypeError("captured module is not a standard router with gating/routing")
    jitter = getattr(getattr(router, "config", None), "moe_input_jitter_eps", None)
    if jitter is not None:
        raise RuntimeError("read-only router probe refuses stochastic input jitter")
    leading_shape = tuple(router_input.shape[:-1])
    with torch.no_grad():
        logits = router.gating(router_input)
        scores, routing_map = router.routing(logits)
    return router_diagnostics_from_outputs(
        logits.reshape(-1, logits.shape[-1]),
        scores,
        routing_map,
        old_expert_count=old_expert_count,
        topk=topk,
        leading_shape=leading_shape,
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".inprogress", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _import_arrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - production dependency
        raise RuntimeError("pyarrow is required for CKA pilot metric shards") from error
    return pa, pq


def _arrow_table(columns: Mapping[str, Any]):
    pa, _ = _import_arrow()
    arrays: dict[str, Any] = {}
    row_count: int | None = None
    for name, values in columns.items():
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        if isinstance(values, (list, tuple)) and values and isinstance(values[0], np.ndarray):
            # Chunk-token raw units have variable lengths across 128/256/tail
            # rows, but their leaves remain explicitly float32.
            array = pa.array(
                [np.asarray(value, dtype=np.float32).tolist() for value in values],
                type=pa.list_(pa.float32()),
            )
            count = len(values)
            values = None
        else:
            values = np.asarray(values) if not isinstance(values, (list, tuple)) else values
            array = None
        if isinstance(values, np.ndarray) and values.ndim > 1:
            # Preserve both the NumPy leaf dtype and every fixed list width.
            array = pa.array(values.reshape(-1))
            for width in reversed(values.shape[1:]):
                array = pa.FixedSizeListArray.from_arrays(array, int(width))
            count = int(values.shape[0])
        elif array is None:
            array = pa.array(values)
            count = len(array)
        if row_count is None:
            row_count = count
        elif count != row_count:
            raise ValueError(f"Arrow column {name} has {count} rows, expected {row_count}")
        arrays[name] = array
    if row_count is None:
        raise ValueError("cannot build an Arrow table without columns")
    return pa.table(arrays)


def _arrow_tables_equivalent(left: Any, right: Any) -> bool:
    """Bitwise Arrow equality that treats identically encoded NaNs as equal."""
    if left.num_rows != right.num_rows or not left.schema.equals(right.schema):
        return False
    for name in left.column_names:
        left_array = left[name].combine_chunks()
        right_array = right[name].combine_chunks()
        left_buffers = left_array.buffers()
        right_buffers = right_array.buffers()
        if len(left_buffers) != len(right_buffers):
            return False
        for left_buffer, right_buffer in zip(left_buffers, right_buffers):
            if left_buffer is None or right_buffer is None:
                if left_buffer is not None or right_buffer is not None:
                    return False
            elif left_buffer.size != right_buffer.size or left_buffer.to_pybytes() != right_buffer.to_pybytes():
                return False
    return True


class AtomicArrowShardWriter:
    """Restartable Parquet shards assembled from Arrow tables."""

    required_columns: tuple[str, ...] = ()

    def __init__(
        self,
        output_dir: str | Path,
        *,
        stream_kind: str,
        rows_per_shard: int = 131_072,
        schema: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_kind = str(stream_kind)
        self.rows_per_shard = int(rows_per_shard)
        if self.rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.manifest_path = self.output_dir / "manifest.json"
        self.schema = schema
        self.metadata = dict(metadata or {})
        self.shards: list[dict[str, Any]] = []
        self.committed_rows = 0
        self._buffer: list[Any] = []
        self._buffered_rows = 0
        if self.manifest_path.exists():
            self._restore()
        elif any(self.output_dir.glob("shard_*.parquet")):
            raise RuntimeError("Parquet shards exist without a resume manifest")

    @property
    def next_row(self) -> int:
        return self.committed_rows + self._buffered_rows

    def _schema_b64(self) -> str | None:
        if self.schema is None:
            return None
        return base64.b64encode(self.schema.serialize().to_pybytes()).decode("ascii")

    def _manifest(self, *, complete: bool = False) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "stream_kind": self.stream_kind,
            "rows_per_shard": self.rows_per_shard,
            "committed_rows": self.committed_rows,
            "next_shard": len(self.shards),
            "schema_base64": self._schema_b64(),
            "shards": self.shards,
            "complete": bool(complete),
            "metadata": self.metadata,
        }

    def _restore(self) -> None:
        pa, pq = _import_arrow()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise RuntimeError("unsupported Arrow shard manifest version")
        if payload.get("stream_kind") != self.stream_kind:
            raise RuntimeError("Arrow shard stream kind changed on resume")
        if int(payload.get("rows_per_shard")) != self.rows_per_shard:
            raise RuntimeError("rows_per_shard changed on resume")
        encoded = payload.get("schema_base64")
        restored_schema = (
            pa.ipc.read_schema(pa.BufferReader(base64.b64decode(encoded))) if encoded else None
        )
        if self.schema is not None and restored_schema is not None and not self.schema.equals(restored_schema):
            raise RuntimeError("Arrow schema changed on resume")
        self.schema = restored_schema if self.schema is None else self.schema
        self.shards = list(payload.get("shards", []))
        total = 0
        for index, record in enumerate(self.shards):
            path = self.output_dir / record["file"]
            if not path.is_file() or _file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"Parquet shard missing or corrupt: {path}")
            sidecar = self.output_dir / record["sidecar_file"]
            if not sidecar.is_file() or _file_sha256(sidecar) != record["sidecar_sha256"]:
                raise RuntimeError(f"Parquet shard sidecar missing or corrupt: {sidecar}")
            table = pq.read_table(path)
            if table.num_rows != int(record["rows"]):
                raise RuntimeError(f"Parquet shard row mismatch: {path}")
            if self.schema is not None and not table.schema.equals(self.schema):
                raise RuntimeError(f"Parquet shard schema mismatch: {path}")
            if record["file"] != f"shard_{index:06d}.parquet":
                raise RuntimeError("Parquet shard sequence is not contiguous")
            total += table.num_rows
        if total != int(payload.get("committed_rows", -1)):
            raise RuntimeError("Arrow resume manifest committed_rows mismatch")
        self.committed_rows = total
        self.metadata = dict(payload.get("metadata", {}))

    def append(
        self,
        columns_or_table: Mapping[str, Any] | Any,
        *,
        global_start_row: int | None = None,
    ) -> None:
        pa, _ = _import_arrow()
        table = (
            columns_or_table
            if isinstance(columns_or_table, pa.Table)
            else _arrow_table(columns_or_table)
        )
        missing = set(self.required_columns) - set(table.column_names)
        if missing:
            raise ValueError(f"missing required {self.stream_kind} columns: {sorted(missing)}")
        if global_start_row is not None and int(global_start_row) != self.next_row:
            raise RuntimeError(
                f"non-contiguous append: start={global_start_row}, expected={self.next_row}"
            )
        if self.schema is None:
            self.schema = table.schema
        elif not table.schema.equals(self.schema):
            raise ValueError("Arrow table schema changed within a metric stream")
        self._buffer.append(table)
        self._buffered_rows += table.num_rows
        self._flush_full_shards()

    def _flush_full_shards(self) -> None:
        pa, _ = _import_arrow()
        if self._buffered_rows < self.rows_per_shard:
            return
        # An append is one complete model/window batch.  Never slice through
        # that boundary: resume can then restart at the last committed batch
        # without duplicating a suffix of an already-written window.
        combined = pa.concat_tables(self._buffer)
        self._commit(combined)
        self._buffer = []
        self._buffered_rows = 0

    def _commit(self, table: Any) -> None:
        _, pq = _import_arrow()
        index = len(self.shards)
        final = self.output_dir / f"shard_{index:06d}.parquet"
        temporary = final.with_name(final.name + ".inprogress")
        if final.exists():
            raise RuntimeError(f"refusing to overwrite committed Parquet shard {final}")
        try:
            with temporary.open("wb") as handle:
                pq.write_table(table, handle, compression="zstd")
                handle.flush()
                os.fsync(handle.fileno())
            checked = pq.read_table(temporary)
            if not _arrow_tables_equivalent(checked, table):
                raise RuntimeError("Parquet inprogress round-trip failed")
            os.replace(temporary, final)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise
        record = {
            "file": final.name,
            "rows": int(table.num_rows),
            "bytes": int(final.stat().st_size),
            "sha256": _file_sha256(final),
        }
        if "window_uid" in table.column_names and table.num_rows:
            window_values = np.asarray(
                table["window_uid"].combine_chunks().to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            unique_windows = np.unique(window_values)
            record.update(
                {
                    "unique_window_count": int(unique_windows.size),
                    "window_uid_min": int(unique_windows.min()),
                    "window_uid_max": int(unique_windows.max()),
                }
            )
        sidecar = final.with_suffix(".json")
        _atomic_json(
            sidecar,
            {
                "schema_version": RUNTIME_SCHEMA_VERSION,
                "stream_kind": self.stream_kind,
                "shard_index": index,
                **record,
                "column_names": table.column_names,
            },
        )
        record["sidecar_file"] = sidecar.name
        record["sidecar_sha256"] = _file_sha256(sidecar)
        self.shards.append(record)
        self.committed_rows += int(table.num_rows)
        _atomic_json(self.manifest_path, self._manifest(complete=False))

    def finalize(self) -> dict[str, Any]:
        if self._buffered_rows:
            pa, _ = _import_arrow()
            self._commit(pa.concat_tables(self._buffer))
            self._buffer = []
            self._buffered_rows = 0
        payload = self._manifest(complete=True)
        _atomic_json(self.manifest_path, payload)
        return payload

    def committed_window_uids(self) -> np.ndarray:
        """Read the exact unique window IDs covered by committed shards."""
        _, pq = _import_arrow()
        values: list[np.ndarray] = []
        for record in self.shards:
            path = self.output_dir / record["file"]
            table = pq.read_table(path, columns=["window_uid"])
            values.append(
                np.asarray(
                    table["window_uid"].combine_chunks().to_numpy(zero_copy_only=False),
                    dtype=np.int64,
                )
            )
        if not values:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(values))


class TokenArrowShardWriter(AtomicArrowShardWriter):
    """Wide one-row-per-contextual-token Arrow stream."""

    required_columns = (
        "domain",
        "split",
        "window_uid",
        "sample_order",
        "document_id",
        "window_offset",
        "position",
        "token_id",
        "eligible",
        "cosine",
        "relative_l2",
        "symmetric_relative_l2",
        "log_r",
        "ref_rms",
        "maha_mean",
        "proto_mean",
        "cka_min_128",
        "cka_min_256",
        "s_min_128",
        "s_min_256",
        "worst_diag_ratio_128",
        "worst_diag_ratio_256",
    )

    def __init__(self, output_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(output_dir, stream_kind="token_metrics_wide", **kwargs)


class ChunkArrowShardWriter(AtomicArrowShardWriter):
    """Long one-row-per-window/chunk/layer Arrow stream."""

    required_columns = (
        "domain",
        "split",
        "chunk_uid",
        "window_uid",
        "sample_order",
        "document_id",
        "window_offset",
        "scale",
        "chunk_start",
        "chunk_length",
        "layer",
        "cka",
        "cka_permutation",
        "cka_random_pair",
        "random_pair_invalid_reason",
        "random_pair_donor_window_uid",
        "cka_off",
        "diag_ratio",
        "invalid_reason",
    )

    def __init__(self, output_dir: str | Path, **kwargs: Any) -> None:
        super().__init__(output_dir, stream_kind="chunk_metrics_long", **kwargs)


@dataclass(frozen=True)
class PairedMetricStreamPaths:
    """Non-overlapping physical roots for open and sealed-test metrics."""

    stream: str
    journal_dir: Path
    token_dir: Path
    chunk_dir: Path


def paired_metric_stream_paths(
    *,
    pilot_root: str | Path,
    worker_output: str | Path,
    domain: str,
    requested_split: str,
    worker_index: int,
    stream: str,
) -> PairedMetricStreamPaths:
    """Return canonical physical paths for one visibility transaction.

    Calibration/selection rows use ``stream='open'``.  Test rows use
    ``stream='test'`` and are rooted below ``sealed_test/raw``; no writer or
    shard can therefore contain both visibility classes.
    """

    if stream not in ("open", "test"):
        raise ValueError("paired metric stream must be 'open' or 'test'")
    if domain not in ("code", "wiki"):
        raise ValueError("paired metric domain must be 'code' or 'wiki'")
    if requested_split not in ("calibration", "selection", "test", "all"):
        raise ValueError("invalid requested split")
    pilot_root = Path(pilot_root)
    worker_output = Path(worker_output)
    label = f"{domain}_{requested_split}_{stream}_worker_{int(worker_index):03d}"
    if stream == "open":
        token_root = pilot_root / "token_metrics"
        chunk_root = pilot_root / "chunk_metrics"
    else:
        token_root = pilot_root / "sealed_test" / "raw" / "token_metrics"
        chunk_root = pilot_root / "sealed_test" / "raw" / "chunk_metrics"
    return PairedMetricStreamPaths(
        stream=stream,
        journal_dir=worker_output / "paired_progress" / stream,
        token_dir=token_root / label,
        chunk_dir=chunk_root / label,
    )


class PairedParquetMetricWriter:
    """Commit token/chunk shards as one resumable model-batch transaction.

    A crash after only one Parquet rename leaves an uncommitted orphan at the
    deterministic next shard name.  On reprocessing, the orphan is validated
    against the regenerated table and reused; common progress advances only
    after both files and their sidecar are durable.  No truncation or deletion
    is needed and a token stream can never advance independently of chunks.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        token_dir: str | Path | None = None,
        chunk_dir: str | Path | None = None,
        target_token_rows_per_shard: int = 131_072,
        metadata: Mapping[str, Any] | None = None,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.token_dir = Path(token_dir) if token_dir is not None else self.output_dir / "token_metrics"
        self.chunk_dir = Path(chunk_dir) if chunk_dir is not None else self.output_dir / "chunk_metrics"
        self.token_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.json"
        self.target_token_rows = int(target_token_rows_per_shard)
        if self.target_token_rows <= 0:
            raise ValueError("target_token_rows_per_shard must be positive")
        self.metadata = dict(metadata or {})
        self.failure_injector = failure_injector
        self.shards: list[dict[str, Any]] = []
        self.committed_batches = 0
        self.cumulative_elapsed_seconds = 0.0
        self._committed_windows: set[int] = set()
        self._token_buffer: list[Any] = []
        self._chunk_buffer: list[Any] = []
        self._buffered_token_rows = 0
        self._buffered_batch_count = 0
        self._buffered_active_seconds = 0.0
        self._buffered_windows: list[int] = []
        self.token_schema: Any | None = None
        self.chunk_schema: Any | None = None
        if self.progress_path.exists():
            self._restore()

    def _progress(self, *, complete: bool = False) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "stream_kind": "paired_token_chunk_parquet",
            # These streams may intentionally live outside output_dir (the
            # canonical pilot token_metrics/chunk_metrics roots).  Persist the
            # resolved locations so a resume cannot silently point the same
            # transaction journal at a different pair of datasets.
            "token_dir": str(self.token_dir.resolve()),
            "chunk_dir": str(self.chunk_dir.resolve()),
            "target_token_rows_per_shard": self.target_token_rows,
            "committed_batches": self.committed_batches,
            "committed_window_count": len(self._committed_windows),
            "cumulative_elapsed_seconds": self.cumulative_elapsed_seconds,
            "shards": self.shards,
            "complete": bool(complete),
            "metadata": self.metadata,
        }

    def _restore(self) -> None:
        _, pq = _import_arrow()
        payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise RuntimeError("unsupported paired progress schema")
        for key, requested in (
            ("token_dir", self.token_dir),
            ("chunk_dir", self.chunk_dir),
        ):
            stored = payload.get(key)
            if stored is None:
                raise RuntimeError(f"paired progress is missing canonical {key}")
            if Path(stored).resolve() != requested.resolve():
                raise RuntimeError(
                    f"paired {key} changed on resume: "
                    f"stored={Path(stored).resolve()} requested={requested.resolve()}"
                )
        if int(payload.get("target_token_rows_per_shard")) != self.target_token_rows:
            raise RuntimeError("paired shard target changed on resume")
        self.shards = list(payload.get("shards", []))
        self.committed_batches = int(payload.get("committed_batches", 0))
        self.metadata = dict(payload.get("metadata", {}))
        total_batches = 0
        total_active_seconds = 0.0
        for index, record in enumerate(self.shards):
            token_path = Path(record["token_file"])
            chunk_path = Path(record["chunk_file"])
            sidecar_path = Path(record["sidecar_file"])
            if not token_path.is_absolute():
                token_path = self.token_dir / token_path
            if not chunk_path.is_absolute():
                chunk_path = self.chunk_dir / chunk_path
            if not sidecar_path.is_absolute():
                sidecar_path = self.output_dir / sidecar_path
            for path, digest in (
                (token_path, record["token_sha256"]),
                (chunk_path, record["chunk_sha256"]),
                (sidecar_path, record["sidecar_sha256"]),
            ):
                if not path.is_file() or _file_sha256(path) != digest:
                    raise RuntimeError(f"paired committed artifact corrupt: {path}")
            token_table = pq.read_table(token_path, columns=["window_uid"])
            chunk_table = pq.read_table(chunk_path, columns=["window_uid"])
            full_token_schema = pq.read_schema(token_path)
            full_chunk_schema = pq.read_schema(chunk_path)
            if self.token_schema is None:
                self.token_schema = full_token_schema
                self.chunk_schema = full_chunk_schema
            elif not self.token_schema.equals(full_token_schema) or not self.chunk_schema.equals(
                full_chunk_schema
            ):
                raise RuntimeError("paired Parquet schema changed across committed shards")
            token_windows = set(token_table["window_uid"].to_pylist())
            chunk_windows = set(chunk_table["window_uid"].to_pylist())
            recorded = set(int(value) for value in record["window_uids"])
            if token_windows != recorded or chunk_windows != recorded:
                raise RuntimeError(f"paired window identity mismatch at shard {index}")
            if self._committed_windows & recorded:
                raise RuntimeError("window duplicated across paired committed shards")
            self._committed_windows.update(recorded)
            total_batches += int(record["batch_count"])
            active_seconds = float(record.get("active_seconds", float("nan")))
            if not math.isfinite(active_seconds) or active_seconds < 0.0:
                raise RuntimeError(
                    f"paired shard {index} has invalid active_seconds"
                )
            total_active_seconds += active_seconds
        if total_batches != self.committed_batches:
            raise RuntimeError("paired committed batch count mismatch")
        recorded_elapsed = float(
            payload.get("cumulative_elapsed_seconds", float("nan"))
        )
        if not math.isfinite(recorded_elapsed) or not math.isclose(
            recorded_elapsed, total_active_seconds, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise RuntimeError("paired cumulative elapsed time mismatch")
        self.cumulative_elapsed_seconds = total_active_seconds

    @staticmethod
    def _validate_required(table: Any, required: Sequence[str], kind: str) -> None:
        missing = set(required) - set(table.column_names)
        if missing:
            raise ValueError(f"missing required {kind} columns: {sorted(missing)}")

    def append_batch(
        self,
        token_columns_or_table: Mapping[str, Any] | Any,
        chunk_columns_or_table: Mapping[str, Any] | Any,
        *,
        active_seconds: float = 0.0,
    ) -> None:
        active_seconds = float(active_seconds)
        if not math.isfinite(active_seconds) or active_seconds < 0.0:
            raise ValueError("active_seconds must be finite and non-negative")
        pa, _ = _import_arrow()
        token_table = (
            token_columns_or_table
            if isinstance(token_columns_or_table, pa.Table)
            else _arrow_table(token_columns_or_table)
        )
        chunk_table = (
            chunk_columns_or_table
            if isinstance(chunk_columns_or_table, pa.Table)
            else _arrow_table(chunk_columns_or_table)
        )
        self._validate_required(
            token_table, TokenArrowShardWriter.required_columns, "token"
        )
        self._validate_required(
            chunk_table, ChunkArrowShardWriter.required_columns, "chunk"
        )
        if self.token_schema is None:
            self.token_schema = token_table.schema
            self.chunk_schema = chunk_table.schema
        elif not self.token_schema.equals(token_table.schema) or not self.chunk_schema.equals(
            chunk_table.schema
        ):
            raise ValueError("paired token/chunk schema changed within a worker stream")
        token_windows = set(int(value) for value in token_table["window_uid"].to_pylist())
        chunk_windows = set(int(value) for value in chunk_table["window_uid"].to_pylist())
        if not token_windows or token_windows != chunk_windows:
            raise ValueError("token/chunk batch window_uid sets differ or are empty")
        allowed_splits = self.metadata.get("stream_splits")
        if allowed_splits is not None:
            allowed = {str(value) for value in allowed_splits}
            token_splits = {str(value) for value in token_table["split"].to_pylist()}
            chunk_splits = {str(value) for value in chunk_table["split"].to_pylist()}
            if token_splits != chunk_splits or not token_splits.issubset(allowed):
                raise RuntimeError(
                    "paired metric batch violates physical split isolation: "
                    f"token={sorted(token_splits)} chunk={sorted(chunk_splits)} "
                    f"allowed={sorted(allowed)}"
                )
        pending = set(self._buffered_windows)
        if token_windows & (self._committed_windows | pending):
            raise RuntimeError("attempted to append an already buffered/committed window")
        self._token_buffer.append(token_table)
        self._chunk_buffer.append(chunk_table)
        self._buffered_token_rows += token_table.num_rows
        self._buffered_batch_count += 1
        self._buffered_active_seconds += active_seconds
        self._buffered_windows.extend(sorted(token_windows))
        if self._buffered_token_rows >= self.target_token_rows:
            self._commit_buffer()

    @staticmethod
    def _write_or_validate(path: Path, table: Any) -> None:
        _, pq = _import_arrow()
        if path.exists():
            existing = pq.read_table(path)
            if not _arrow_tables_equivalent(existing, table):
                raise RuntimeError(f"uncommitted orphan differs on deterministic replay: {path}")
            return
        temporary = path.with_name(path.name + ".inprogress")
        try:
            with temporary.open("wb") as handle:
                pq.write_table(table, handle, compression="zstd")
                handle.flush()
                os.fsync(handle.fileno())
            checked = pq.read_table(temporary)
            if not _arrow_tables_equivalent(checked, table):
                raise RuntimeError(f"paired Parquet round-trip failed: {path}")
            os.replace(temporary, path)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _fail(self, stage: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(stage)

    def _commit_buffer(self) -> None:
        if not self._token_buffer:
            return
        pa, _ = _import_arrow()
        token_table = pa.concat_tables(self._token_buffer)
        chunk_table = pa.concat_tables(self._chunk_buffer)
        index = len(self.shards)
        token_path = self.token_dir / f"shard_{index:06d}.parquet"
        chunk_path = self.chunk_dir / f"shard_{index:06d}.parquet"
        sidecar_path = self.output_dir / f"shard_{index:06d}.json"
        windows = sorted(set(self._buffered_windows))
        self._write_or_validate(token_path, token_table)
        self._fail("after_token_rename")
        self._write_or_validate(chunk_path, chunk_table)
        self._fail("after_chunk_rename")
        sidecar_payload = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "shard_index": index,
            "batch_count": self._buffered_batch_count,
            "active_seconds": self._buffered_active_seconds,
            "window_uids": windows,
            "token_file": str(token_path.resolve()),
            "token_rows": token_table.num_rows,
            "token_sha256": _file_sha256(token_path),
            "chunk_file": str(chunk_path.resolve()),
            "chunk_rows": chunk_table.num_rows,
            "chunk_sha256": _file_sha256(chunk_path),
        }
        _atomic_json(sidecar_path, sidecar_payload)
        self._fail("after_sidecar")
        record = {
            **sidecar_payload,
            "sidecar_file": str(sidecar_path.resolve()),
            "sidecar_sha256": _file_sha256(sidecar_path),
        }
        self.shards.append(record)
        self.committed_batches += self._buffered_batch_count
        self.cumulative_elapsed_seconds += self._buffered_active_seconds
        self._committed_windows.update(windows)
        _atomic_json(self.progress_path, self._progress(complete=False))
        self._token_buffer = []
        self._chunk_buffer = []
        self._buffered_token_rows = 0
        self._buffered_batch_count = 0
        self._buffered_active_seconds = 0.0
        self._buffered_windows = []

    def committed_window_uids(self) -> np.ndarray:
        return np.asarray(sorted(self._committed_windows), dtype=np.int64)

    def finalize(self) -> dict[str, Any]:
        self._commit_buffer()
        payload = self._progress(complete=True)
        _atomic_json(self.progress_path, payload)
        return payload


class _RawUnitSpillBank:
    def __init__(self, root: Path) -> None:
        self.root = root
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"raw-unit spill directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.handles: dict[tuple[Any, ...], Any] = {}
        self.paths: dict[tuple[Any, ...], Path] = {}
        self.counts: dict[tuple[Any, ...], int] = defaultdict(int)

    @staticmethod
    def _name(key: tuple[Any, ...]) -> str:
        return "__".join("none" if value is None else str(value) for value in key)

    def add(self, key: tuple[Any, ...], values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        if key not in self.handles:
            path = self.root / (self._name(key) + ".f32.inprogress")
            self.paths[key] = path
            self.handles[key] = path.open("wb")
        finite.astype("<f4", copy=False).tofile(self.handles[key])
        self.counts[key] += int(finite.size)

    def finalize(self) -> dict[tuple[Any, ...], Path]:
        result: dict[tuple[Any, ...], Path] = {}
        for key, handle in self.handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temporary = self.paths[key]
            final = temporary.with_suffix("")
            os.replace(temporary, final)
            expected_bytes = self.counts[key] * np.dtype("<f4").itemsize
            if final.stat().st_size != expected_bytes:
                raise RuntimeError(f"raw-unit spill byte count mismatch: {final}")
            result[key] = final
        self.handles = {}
        return result


def _parquet_files(paths: str | Path | Sequence[str | Path]) -> list[str]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files: list[str] = []
    for value in paths:
        path = Path(value)
        if path.is_file() and path.suffix == ".parquet":
            files.append(str(path))
        elif path.is_dir():
            files.extend(str(item) for item in sorted(path.rglob("*.parquet")))
    if not files:
        raise FileNotFoundError(f"no Parquet shards found below {paths}")
    return files


def build_exact_raw_unit_quantiles(
    *,
    token_metric_paths: str | Path | Sequence[str | Path],
    chunk_metric_paths: str | Path | Sequence[str | Path],
    output_path: str | Path,
    spill_dir: str | Path,
    layers: Sequence[int] = LAYERS,
) -> dict[str, Any]:
    """Build exact pre-aggregation calibration quantiles via disk spills.

    The raw ``s_i`` values come from the long chunk table, never from token
    minima.  Each metric/group is spilled as a finite little-endian float32
    stream, then exact NumPy linear quantiles are evaluated over a read-only
    memmap.  The spill files and their hashes remain as audit artifacts.
    """

    pa, pq = _import_arrow()
    import pyarrow.dataset as pads

    token_dataset = pads.dataset(_parquet_files(token_metric_paths), format="parquet")
    chunk_dataset = pads.dataset(_parquet_files(chunk_metric_paths), format="parquet")
    bank = _RawUnitSpillBank(Path(spill_dir))
    layer_values = tuple(int(layer) for layer in layers)

    token_columns = ["domain", "split", "cosine", "relative_l2", "log_r"]
    for batch in token_dataset.scanner(columns=token_columns, batch_size=65_536).to_batches():
        domains = np.asarray(batch.column(0).to_pylist(), dtype=object)
        splits = np.asarray(batch.column(1).to_pylist(), dtype=object)
        cosine = np.asarray(batch.column(2).to_pylist(), dtype=np.float32)
        relative_l2 = np.asarray(batch.column(3).to_pylist(), dtype=np.float32)
        log_r = np.abs(np.asarray(batch.column(4).to_pylist(), dtype=np.float32))
        wiki_cal = (domains == "wiki") & (splits == "calibration")
        for layer_index, layer in enumerate(layer_values):
            bank.add(
                ("wiki", "calibration", "relative_l2", None, layer),
                relative_l2[wiki_cal, layer_index],
            )
            bank.add(
                ("wiki", "calibration", "abs_log_r", None, layer),
                log_r[wiki_cal, layer_index],
            )
            for code_split in ("calibration", "selection"):
                mask = (domains == "code") & (splits == code_split)
                bank.add(
                    ("code", code_split, "cosine", None, layer),
                    cosine[mask, layer_index],
                )

    chunk_columns = ["domain", "split", "scale", "layer", "cka", "s_i"]
    for batch in chunk_dataset.scanner(columns=chunk_columns, batch_size=16_384).to_batches():
        domains = np.asarray(batch.column(0).to_pylist(), dtype=object)
        splits = np.asarray(batch.column(1).to_pylist(), dtype=object)
        scales = np.asarray(batch.column(2).to_numpy(zero_copy_only=False), dtype=np.int16)
        chunk_layers = np.asarray(
            batch.column(3).to_numpy(zero_copy_only=False), dtype=np.int16
        )
        cka = np.asarray(batch.column(4).to_numpy(zero_copy_only=False), dtype=np.float32)
        raw_s = batch.column(5).to_pylist()
        for scale in (128, 256):
            for layer in layer_values:
                mask = (
                    (domains == "wiki")
                    & (splits == "calibration")
                    & (scales == scale)
                    & (chunk_layers == layer)
                )
                if not np.any(mask):
                    continue
                bank.add(
                    ("wiki", "calibration", "cka", scale, layer), cka[mask]
                )
                selected_rows = np.flatnonzero(mask)
                if selected_rows.size:
                    bank.add(
                        ("wiki", "calibration", "s_i", scale, layer),
                        np.concatenate(
                            [np.asarray(raw_s[index], dtype=np.float32) for index in selected_rows]
                        ),
                    )

    spill_paths = bank.finalize()
    rows: list[dict[str, Any]] = []
    for key, path in sorted(spill_paths.items(), key=lambda item: repr(item[0])):
        domain, split, metric, scale, layer = key
        count = bank.counts[key]
        values = np.memmap(path, dtype="<f4", mode="r", shape=(count,))
        quantiles = (
            (0.01, 0.03, 0.05)
            if metric in ("cka", "s_i")
            else (0.95, 0.97, 0.99)
            if metric in ("relative_l2", "abs_log_r")
            else (0.99,)
        )
        results = np.quantile(values, quantiles, method="linear")
        for quantile, value in zip(quantiles, np.atleast_1d(results).tolist()):
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "metric": metric,
                    "scale": scale,
                    "layer": int(layer),
                    "quantile": float(quantile),
                    "value": float(value),
                    "count": int(count),
                    "method": "exact_disk_backed",
                }
            )
        del values

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".inprogress")
    table = pa.Table.from_pylist(rows)
    with temporary.open("wb") as handle:
        pq.write_table(table, handle, compression="zstd")
        handle.flush()
        os.fsync(handle.fileno())
    checked = pq.read_table(temporary)
    if not _arrow_tables_equivalent(checked, table):
        raise RuntimeError("raw-unit quantile Parquet round-trip failed")
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "output": str(output_path),
        "rows": len(rows),
        "sha256": _file_sha256(output_path),
        "method": "exact_disk_backed",
        "spills": [
            {
                "key": list(key),
                "file": str(path),
                "count": bank.counts[key],
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for key, path in sorted(spill_paths.items(), key=lambda item: repr(item[0]))
        ],
    }
    _atomic_json(output_path.with_suffix(".json"), manifest)
    return manifest


def _sample_indices(
    total: int,
    count: int,
    *,
    base_seed: int,
    domain_id: int,
    stage_id: int,
    split_id: int,
) -> np.ndarray:
    """Uniform indices from the prepared-config SeedSequence namespace."""

    total = int(total)
    count = min(int(count), total)
    if total <= 0 or count <= 0:
        return np.empty(0, dtype=np.int64)
    rng = seeded_rng(
        base_seed=int(base_seed),
        domain_id=int(domain_id),
        stage_id=int(stage_id),
        split_id=int(split_id),
    )
    selected = rng.choice(total, size=count, replace=False).astype(np.int64)
    selected.sort()
    return selected


class _RawMomentAccumulator:
    """Sufficient raw moments for exact centered Ledoit-Wolf shrinkage."""

    def __init__(self, hidden_size: int, *, buffer_rows: int = 16_384) -> None:
        self.hidden_size = int(hidden_size)
        self.buffer_rows = int(buffer_rows)
        if self.buffer_rows <= 0:
            raise ValueError("buffer_rows must be positive")
        self.count = 0
        self.sum_x = np.zeros(self.hidden_size, dtype=np.float64)
        self.xtx = np.zeros((self.hidden_size, self.hidden_size), dtype=np.float64)
        self.sum_norm2_x = np.zeros(self.hidden_size, dtype=np.float64)
        self.sum_norm4 = 0.0
        self._mode: str | None = None
        self._torch_device: torch.device | None = None
        self._torch_buffer: list[torch.Tensor] = []
        self._torch_buffer_rows = 0
        self._torch_sum_x: torch.Tensor | None = None
        self._torch_xtx: torch.Tensor | None = None
        self._torch_sum_norm2_x: torch.Tensor | None = None
        self._torch_sum_norm4: torch.Tensor | None = None

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        if isinstance(values, torch.Tensor):
            tensor = values.detach().float()
            if tensor.ndim != 2 or tensor.shape[1] != self.hidden_size:
                raise ValueError(
                    f"moment values must be [tokens,{self.hidden_size}], got {tuple(tensor.shape)}"
                )
            if tensor.numel() == 0:
                return
            if not torch.isfinite(tensor).all():
                raise ValueError("non-finite hidden value in membership covariance sample")
            if self._mode not in (None, "torch"):
                raise RuntimeError("cannot mix NumPy and torch moment updates")
            if self._torch_device is not None and tensor.device != self._torch_device:
                raise RuntimeError("moment tensor device changed within a stream")
            self._mode = "torch"
            self._torch_device = tensor.device
            self._torch_buffer.append(tensor)
            self._torch_buffer_rows += int(tensor.shape[0])
            self.count += int(tensor.shape[0])
            if self._torch_buffer_rows >= self.buffer_rows:
                self._flush_torch()
            return
        else:
            array = np.asarray(values, dtype=np.float64)
        if self._mode not in (None, "numpy"):
            raise RuntimeError("cannot mix torch and NumPy moment updates")
        self._mode = "numpy"
        if array.ndim != 2 or array.shape[1] != self.hidden_size:
            raise ValueError(
                f"moment values must be [tokens,{self.hidden_size}], got {array.shape}"
            )
        if array.size == 0:
            return
        if not np.isfinite(array).all():
            raise ValueError("non-finite hidden value in membership covariance sample")
        norm2 = np.einsum("ij,ij->i", array, array, optimize=True)
        self.count += int(array.shape[0])
        self.sum_x += array.sum(axis=0, dtype=np.float64)
        self.xtx += array.T @ array
        self.sum_norm2_x += np.einsum("i,ij->j", norm2, array, optimize=True)
        self.sum_norm4 += float(np.dot(norm2, norm2))

    def _flush_torch(self) -> None:
        if not self._torch_buffer:
            return
        values = torch.cat(self._torch_buffer, dim=0)
        device = values.device
        if self._torch_sum_x is None:
            self._torch_sum_x = torch.zeros(
                self.hidden_size, dtype=torch.float32, device=device
            )
            self._torch_xtx = torch.zeros(
                (self.hidden_size, self.hidden_size), dtype=torch.float32, device=device
            )
            self._torch_sum_norm2_x = torch.zeros(
                self.hidden_size, dtype=torch.float32, device=device
            )
            self._torch_sum_norm4 = torch.zeros((), dtype=torch.float32, device=device)
        assert self._torch_xtx is not None
        assert self._torch_sum_norm2_x is not None
        assert self._torch_sum_norm4 is not None
        norm2 = torch.einsum("ij,ij->i", values, values)
        self._torch_sum_x.add_(values.sum(dim=0))
        self._torch_xtx.add_(values.T @ values)
        self._torch_sum_norm2_x.add_(torch.einsum("i,ij->j", norm2, values))
        self._torch_sum_norm4.add_(torch.dot(norm2, norm2))
        self._torch_buffer = []
        self._torch_buffer_rows = 0

    def _raw_numpy_moments(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self._mode == "torch":
            self._flush_torch()
            assert self._torch_sum_x is not None
            assert self._torch_xtx is not None
            assert self._torch_sum_norm2_x is not None
            assert self._torch_sum_norm4 is not None
            return (
                self._torch_sum_x.double().cpu().numpy(),
                self._torch_xtx.double().cpu().numpy(),
                self._torch_sum_norm2_x.double().cpu().numpy(),
                float(self._torch_sum_norm4.double().cpu().item()),
            )
        return self.sum_x, self.xtx, self.sum_norm2_x, self.sum_norm4

    def centered_statistics(self) -> dict[str, np.ndarray | float | int]:
        if self.count < 2:
            raise RuntimeError("at least two covariance samples are required")
        sum_x, xtx, sum_norm2_x, sum_norm4 = self._raw_numpy_moments()
        n = float(self.count)
        mean = sum_x / n
        centered_xtx = xtx - n * np.outer(mean, mean)
        centered_xtx = 0.5 * (centered_xtx + centered_xtx.T)
        covariance = centered_xtx / n

        # Exact expansion of sum_i ||x_i - mean||^4 using only raw moments.
        mean_norm2 = float(np.dot(mean, mean))
        raw_trace = float(np.trace(xtx))
        centered_norm4 = (
            sum_norm4
            + 4.0 * float(mean @ xtx @ mean)
            + n * mean_norm2**2
            - 4.0 * float(mean @ sum_norm2_x)
            + 2.0 * mean_norm2 * raw_trace
            - 4.0 * mean_norm2 * float(mean @ sum_x)
        )
        centered_norm4 = max(centered_norm4, 0.0)
        return {
            "count": self.count,
            "sample_mean": mean,
            "covariance": covariance,
            "centered_norm4_sum": centered_norm4,
        }


def ledoit_wolf_from_moments(
    moments: _RawMomentAccumulator,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return sklearn-equivalent Ledoit-Wolf covariance from raw moments.

    The empirical covariance uses the maximum-likelihood ``1/n`` denominator,
    matching ``sklearn.covariance.ledoit_wolf``.  No sklearn dependency is
    required in the model runtime.
    """

    stats = moments.centered_statistics()
    covariance = np.asarray(stats["covariance"], dtype=np.float64)
    n = float(stats["count"])
    p = covariance.shape[0]
    if p == 1:
        return covariance.copy(), 0.0, covariance
    trace = float(np.trace(covariance))
    mu = trace / float(p)
    delta_raw = float(np.square(covariance).sum(dtype=np.float64))
    beta = (float(stats["centered_norm4_sum"]) / n - delta_raw) / (p * n)
    delta = (delta_raw - 2.0 * mu * trace + p * mu**2) / p
    beta = max(0.0, min(beta, delta))
    shrinkage = 0.0 if delta <= 0.0 else float(beta / delta)
    shrunk = (1.0 - shrinkage) * covariance
    shrunk.flat[:: p + 1] += shrinkage * mu
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk, shrinkage, covariance


def covariance_inverse_sqrt(
    covariance: np.ndarray,
    *,
    eigenvalue_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    floor = max(float(eigenvalue_floor), float(eigenvalues.max(initial=0.0)) * 1.0e-12)
    clipped = np.maximum(eigenvalues, floor)
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(clipped))[None, :]) @ eigenvectors.T
    return inverse_sqrt.astype(np.float32), eigenvalues.astype(np.float64)


def _kmeans_plusplus_initial_indices(
    values: np.ndarray,
    *,
    n_clusters: int,
    rng: np.random.Generator,
    init_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose deterministic D-squared centers inside a bounded init pool.

    MiniBatchKMeans does not need to run D-squared sampling over the complete
    200K-token reservoir.  We first draw a uniform no-replacement init pool,
    then run exact sequential k-means++ within that pool.  This CPU helper is
    shared by the CPU and CUDA update backends, so both start from the exact
    same reservoir row indices.
    """

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("k-means++ values must be two-dimensional")
    sample_count = int(values.shape[0])
    n_clusters = int(n_clusters)
    init_size = min(int(init_size), sample_count)
    if n_clusters <= 0 or init_size < n_clusters:
        raise ValueError("k-means++ init_size must be at least n_clusters")

    pool_indices = rng.choice(
        sample_count, size=init_size, replace=False
    ).astype(np.int64)
    pool = values[pool_indices]
    pool_norm2 = np.einsum("ij,ij->i", pool, pool, optimize=True).astype(
        np.float32, copy=False
    )
    selected_positions = np.empty(n_clusters, dtype=np.int64)
    selected_mask = np.zeros(init_size, dtype=bool)
    first = int(rng.integers(0, init_size))
    selected_positions[0] = first
    selected_mask[first] = True
    closest_distance2 = np.full(init_size, np.inf, dtype=np.float32)

    for center_number in range(1, n_clusters):
        previous = pool[selected_positions[center_number - 1]]
        distance2 = pool_norm2 + np.dot(previous, previous) - 2.0 * (pool @ previous)
        np.maximum(distance2, 0.0, out=distance2)
        np.minimum(closest_distance2, distance2, out=closest_distance2)
        closest_distance2[selected_mask] = 0.0
        weights = closest_distance2.astype(np.float64, copy=False)
        total = float(weights.sum(dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            # Degenerate duplicate-valued pools still receive unique row IDs.
            selected = int(np.flatnonzero(~selected_mask)[0])
        else:
            target = float(rng.random()) * total
            cumulative = np.cumsum(weights, dtype=np.float64)
            selected = int(np.searchsorted(cumulative, target, side="right"))
            if selected >= init_size or selected_mask[selected] or weights[selected] <= 0.0:
                available = np.flatnonzero((~selected_mask) & (weights > 0.0))
                if available.size:
                    selected = int(available[np.argmax(weights[available])])
                else:
                    selected = int(np.flatnonzero(~selected_mask)[0])
        selected_positions[center_number] = selected
        selected_mask[selected] = True

    initial_indices = pool_indices[selected_positions]
    if np.unique(initial_indices).size != n_clusters:
        raise AssertionError("k-means++ produced duplicate reservoir row indices")
    return initial_indices, pool_indices


def deterministic_minibatch_kmeans(
    samples: torch.Tensor | np.ndarray,
    *,
    n_clusters: int = 64,
    batch_size: int = 4_096,
    n_init: int = 1,
    max_iter: int = 100,
    reassignment_ratio: float = 0.01,
    seed: int = 1234,
    init: str = "k-means++",
    init_size: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic repository MiniBatchKMeans implementation.

    ``max_iter`` is the number of minibatch updates (not full reservoir epochs).
    Empty/underused centers are deterministically reassigned to farthest points
    in the current minibatch according to ``reassignment_ratio``.
    """

    if isinstance(samples, torch.Tensor):
        values = samples.detach().float().cpu().numpy()
    else:
        values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < n_clusters:
        raise ValueError("k-means samples must be [n,p] with n >= n_clusters")
    if batch_size <= 0 or n_init <= 0 or max_iter <= 0:
        raise ValueError("batch_size, n_init, and max_iter must be positive")
    if not 0.0 <= reassignment_ratio < 1.0:
        raise ValueError("reassignment_ratio must be in [0,1)")
    if init != "k-means++":
        raise ValueError("repository deterministic MiniBatchKMeans requires init='k-means++'")
    init_pool_policy = (
        "min(n_samples,max(3*batch_size,3*n_clusters))"
        if init_size is None
        else "explicit"
    )
    resolved_init_size = min(
        int(values.shape[0]),
        int(init_size) if init_size is not None else max(3 * batch_size, 3 * n_clusters),
    )
    if resolved_init_size < n_clusters:
        raise ValueError("resolved k-means++ init_size is smaller than n_clusters")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    if target_device.type != "cpu":
        return _deterministic_minibatch_kmeans_torch(
            values,
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=n_init,
            max_iter=max_iter,
            reassignment_ratio=reassignment_ratio,
            seed=seed,
            init=init,
            init_size=resolved_init_size,
            init_pool_policy=init_pool_policy,
            device=target_device,
        )

    best_centers: np.ndarray | None = None
    best_inertia = math.inf
    best_counts: np.ndarray | None = None
    initial_indices_by_run: list[list[int]] = []
    init_pool_indices_by_run: list[list[int]] = []
    for initialization in range(n_init):
        rng = np.random.default_rng(np.random.SeedSequence([seed, initialization]))
        initial, init_pool = _kmeans_plusplus_initial_indices(
            values,
            n_clusters=n_clusters,
            rng=rng,
            init_size=resolved_init_size,
        )
        initial_indices_by_run.append(initial.tolist())
        init_pool_indices_by_run.append(init_pool.tolist())
        centers = values[initial].astype(np.float64, copy=True)
        counts = np.zeros(n_clusters, dtype=np.int64)
        permutation = rng.permutation(values.shape[0])
        cursor = 0
        for _ in range(max_iter):
            if cursor + batch_size > permutation.size:
                permutation = rng.permutation(values.shape[0])
                cursor = 0
            indices = permutation[cursor : cursor + min(batch_size, permutation.size)]
            cursor += indices.size
            batch = values[indices].astype(np.float64, copy=False)
            x2 = np.einsum("ij,ij->i", batch, batch)[:, None]
            c2 = np.einsum("ij,ij->i", centers, centers)[None, :]
            distance2 = np.maximum(x2 + c2 - 2.0 * (batch @ centers.T), 0.0)
            assignments = distance2.argmin(axis=1)
            for cluster in np.unique(assignments).tolist():
                members = batch[assignments == cluster]
                old_count = int(counts[cluster])
                new_count = old_count + members.shape[0]
                centers[cluster] = (
                    centers[cluster] * old_count + members.sum(axis=0)
                ) / float(new_count)
                counts[cluster] = new_count

            if reassignment_ratio > 0.0 and counts.max(initial=0) > 0:
                underused = np.flatnonzero(counts < reassignment_ratio * counts.max())
                if underused.size:
                    nearest = distance2.min(axis=1)
                    farthest = np.argsort(nearest, kind="stable")[::-1]
                    for cluster, row in zip(underused.tolist(), farthest.tolist()):
                        centers[cluster] = batch[row]
                        counts[cluster] = max(1, int(counts.max() * reassignment_ratio))

        # Compare initializations on the whole finite reservoir in blocks.
        inertia = 0.0
        for start in range(0, values.shape[0], batch_size):
            block = values[start : start + batch_size].astype(np.float64, copy=False)
            x2 = np.einsum("ij,ij->i", block, block)[:, None]
            c2 = np.einsum("ij,ij->i", centers, centers)[None, :]
            distance2 = np.maximum(x2 + c2 - 2.0 * (block @ centers.T), 0.0)
            inertia += float(distance2.min(axis=1).sum(dtype=np.float64))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.astype(np.float32)
            best_counts = counts.copy()
    assert best_centers is not None and best_counts is not None
    return best_centers, {
        "algorithm": KMEANS_DEFAULTS["algorithm"],
        "version": KMEANS_DEFAULTS["version"],
        "n_clusters": int(n_clusters),
        "batch_size": int(batch_size),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "reassignment_ratio": float(reassignment_ratio),
        "seed": int(seed),
        "init": init,
        "init_size": int(resolved_init_size),
        "init_pool_policy": init_pool_policy,
        "initial_indices_by_run": initial_indices_by_run,
        "init_pool_indices_sha256_by_run": [
            hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
            for indices in init_pool_indices_by_run
        ],
        "initialization_compute": "shared_cpu_numpy_float32_d_squared",
        "inertia": float(best_inertia),
        "cluster_update_counts": best_counts.tolist(),
        "compute_device": "cpu",
    }


def _deterministic_minibatch_kmeans_torch(
    values: np.ndarray,
    *,
    n_clusters: int,
    batch_size: int,
    n_init: int,
    max_iter: int,
    reassignment_ratio: float,
    seed: int,
    init: str,
    init_size: int,
    init_pool_policy: str,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    """CUDA-capable fp32 backend with NumPy-defined deterministic ordering."""

    best_centers: torch.Tensor | None = None
    best_counts: torch.Tensor | None = None
    best_inertia = math.inf
    initial_indices_by_run: list[list[int]] = []
    init_pool_indices_by_run: list[list[int]] = []
    for initialization in range(n_init):
        rng = np.random.default_rng(np.random.SeedSequence([seed, initialization]))
        initial, init_pool = _kmeans_plusplus_initial_indices(
            values,
            n_clusters=n_clusters,
            rng=rng,
            init_size=init_size,
        )
        initial_indices_by_run.append(initial.tolist())
        init_pool_indices_by_run.append(init_pool.tolist())
        centers = torch.as_tensor(values[initial], dtype=torch.float32, device=device).clone()
        counts = torch.zeros(n_clusters, dtype=torch.float32, device=device)
        permutation = rng.permutation(values.shape[0])
        cursor = 0
        for _ in range(max_iter):
            if cursor + batch_size > permutation.size:
                permutation = rng.permutation(values.shape[0])
                cursor = 0
            indices = permutation[cursor : cursor + min(batch_size, permutation.size)]
            cursor += indices.size
            batch = torch.as_tensor(values[indices], dtype=torch.float32, device=device)
            distance2 = (
                batch.square().sum(dim=1, keepdim=True)
                + centers.square().sum(dim=1).unsqueeze(0)
                - 2.0 * batch @ centers.T
            ).clamp_min_(0.0)
            assignments = distance2.argmin(dim=1)
            batch_counts = torch.bincount(assignments, minlength=n_clusters).float()
            batch_sums = torch.zeros_like(centers)
            batch_sums.index_add_(0, assignments, batch)
            new_counts = counts + batch_counts
            present = batch_counts > 0
            centers[present] = (
                centers[present] * counts[present, None] + batch_sums[present]
            ) / new_counts[present, None]
            counts = new_counts
            if reassignment_ratio > 0.0 and float(counts.max().item()) > 0.0:
                underused = torch.nonzero(
                    counts < reassignment_ratio * counts.max(), as_tuple=False
                ).flatten()
                if underused.numel():
                    nearest = distance2.min(dim=1).values
                    farthest = torch.topk(
                        nearest,
                        k=min(int(underused.numel()), int(nearest.numel())),
                        largest=True,
                        sorted=True,
                    ).indices
                    underused = underused[: farthest.numel()]
                    centers[underused] = batch[farthest]
                    counts[underused] = torch.clamp(
                        counts.max() * reassignment_ratio, min=1.0
                    )

        inertia = 0.0
        for start in range(0, values.shape[0], batch_size):
            block = torch.as_tensor(
                values[start : start + batch_size], dtype=torch.float32, device=device
            )
            distance2 = (
                block.square().sum(dim=1, keepdim=True)
                + centers.square().sum(dim=1).unsqueeze(0)
                - 2.0 * block @ centers.T
            ).clamp_min_(0.0)
            inertia += float(distance2.min(dim=1).values.sum().item())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.detach().clone()
            best_counts = counts.detach().clone()
    assert best_centers is not None and best_counts is not None
    return best_centers.cpu().numpy().astype(np.float32), {
        "algorithm": KMEANS_DEFAULTS["algorithm"],
        "version": KMEANS_DEFAULTS["version"],
        "n_clusters": int(n_clusters),
        "batch_size": int(batch_size),
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "reassignment_ratio": float(reassignment_ratio),
        "seed": int(seed),
        "init": init,
        "init_size": int(init_size),
        "init_pool_policy": init_pool_policy,
        "initial_indices_by_run": initial_indices_by_run,
        "init_pool_indices_sha256_by_run": [
            hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
            for indices in init_pool_indices_by_run
        ],
        "initialization_compute": "shared_cpu_numpy_float32_d_squared",
        "inertia": float(best_inertia),
        "cluster_update_counts": best_counts.cpu().tolist(),
        "compute_device": str(device),
    }


class Pass1MembershipAccumulator:
    """All-token mean plus deterministic covariance/k-means subsamples."""

    def __init__(
        self,
        *,
        layers: Sequence[int],
        hidden_size: int,
        total_tokens: int,
        covariance_sample_size: int = 2_000_000,
        kmeans_reservoir_size: int = 200_000,
        seed: int = 1234,
        kmeans_storage_dtype: np.dtype = np.float16,
    ) -> None:
        self.layers = tuple(int(layer) for layer in layers)
        self.hidden_size = int(hidden_size)
        self.total_tokens = int(total_tokens)
        self.seed = int(seed)
        if not self.layers or self.hidden_size <= 0 or self.total_tokens <= 0:
            raise ValueError("layers, hidden_size, and total_tokens must be positive")
        # Pass 1 is fixed to Wiki/calibration.  These IDs are part of the
        # prepared config's SeedSequence contract and must not be replaced by
        # ad-hoc arithmetic on the base seed.
        self.membership_domain_id = int(DOMAIN_IDS["wiki"])
        self.membership_split_id = int(SPLIT_IDS["calibration"])
        self.covariance_indices = _sample_indices(
            self.total_tokens,
            covariance_sample_size,
            base_seed=self.seed,
            domain_id=self.membership_domain_id,
            stage_id=STAGE_COVARIANCE_SUBSAMPLE,
            split_id=self.membership_split_id,
        )
        self.kmeans_indices = _sample_indices(
            self.total_tokens,
            kmeans_reservoir_size,
            base_seed=self.seed,
            domain_id=self.membership_domain_id,
            stage_id=STAGE_KMEANS_RESERVOIR,
            split_id=self.membership_split_id,
        )
        self.sampling_provenance = {
            "sampler": "numpy.PCG64_SeedSequence_choice_without_replacement_sorted",
            "seed_sequence_entropy_order": [
                "base_seed",
                "domain_id",
                "stage_id",
                "split_id",
            ],
            "base_seed": self.seed,
            "domain": "wiki",
            "domain_id": self.membership_domain_id,
            "split": "calibration",
            "split_id": self.membership_split_id,
            "covariance_stage_id": int(STAGE_COVARIANCE_SUBSAMPLE),
            "kmeans_reservoir_stage_id": int(STAGE_KMEANS_RESERVOIR),
            "covariance_entropy": [
                self.seed,
                self.membership_domain_id,
                int(STAGE_COVARIANCE_SUBSAMPLE),
                self.membership_split_id,
            ],
            "kmeans_reservoir_entropy": [
                self.seed,
                self.membership_domain_id,
                int(STAGE_KMEANS_RESERVOIR),
                self.membership_split_id,
            ],
            "covariance_sample_count": int(self.covariance_indices.size),
            "kmeans_reservoir_count": int(self.kmeans_indices.size),
            "covariance_indices_sha256": hashlib.sha256(
                self.covariance_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "kmeans_indices_sha256": hashlib.sha256(
                self.kmeans_indices.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        }
        self.kmeans_storage_dtype = np.dtype(kmeans_storage_dtype)
        self.processed_tokens = 0
        self.all_count = 0
        self.all_sum = {
            layer: np.zeros(self.hidden_size, dtype=np.float64) for layer in self.layers
        }
        self.covariance_moments = {
            layer: _RawMomentAccumulator(self.hidden_size) for layer in self.layers
        }
        self.kmeans_values = {
            layer: np.empty(
                (self.kmeans_indices.size, self.hidden_size),
                dtype=self.kmeans_storage_dtype,
            )
            for layer in self.layers
        }
        self.kmeans_written = 0

    @staticmethod
    def _local_positions(selected: np.ndarray, start: int, stop: int) -> np.ndarray:
        left = int(np.searchsorted(selected, start, side="left"))
        right = int(np.searchsorted(selected, stop, side="left"))
        return selected[left:right] - start

    def update(self, hidden_by_layer: Mapping[int, torch.Tensor]) -> None:
        if set(hidden_by_layer) != set(self.layers):
            raise ValueError("hidden_by_layer keys do not match configured layers")
        sizes = {tuple(value.shape) for value in hidden_by_layer.values()}
        if len(sizes) != 1:
            raise ValueError("all layer hidden batches must have the same shape")
        shape = next(iter(sizes))
        if len(shape) < 2 or shape[-1] != self.hidden_size:
            raise ValueError("layer hidden must end in hidden_size")
        count = int(math.prod(shape[:-1]))
        start = self.processed_tokens
        stop = start + count
        if stop > self.total_tokens:
            raise RuntimeError("Pass1 received more tokens than configured")
        covariance_local = self._local_positions(self.covariance_indices, start, stop)
        kmeans_local = self._local_positions(self.kmeans_indices, start, stop)

        for layer in self.layers:
            flat = hidden_by_layer[layer].detach().reshape(-1, self.hidden_size).float()
            if not torch.isfinite(flat).all():
                raise ValueError(f"non-finite Pass1 hidden at layer {layer}")
            self.all_sum[layer] += flat.sum(dim=0).double().cpu().numpy()
            if covariance_local.size:
                index = torch.as_tensor(covariance_local, dtype=torch.long, device=flat.device)
                self.covariance_moments[layer].update(flat.index_select(0, index))
            if kmeans_local.size:
                index = torch.as_tensor(kmeans_local, dtype=torch.long, device=flat.device)
                selected = flat.index_select(0, index).cpu().numpy()
                self.kmeans_values[layer][
                    self.kmeans_written : self.kmeans_written + kmeans_local.size
                ] = selected.astype(self.kmeans_storage_dtype)
        self.kmeans_written += int(kmeans_local.size)
        self.processed_tokens = stop
        self.all_count += count

    def finalize(
        self,
        *,
        kmeans_params: Mapping[str, Any] | None = None,
        kmeans_device: torch.device | str | None = None,
    ) -> dict[int, dict[str, Any]]:
        if self.processed_tokens != self.total_tokens:
            raise RuntimeError(
                f"incomplete Pass1 token stream: {self.processed_tokens}/{self.total_tokens}"
            )
        if self.kmeans_written != self.kmeans_indices.size:
            raise RuntimeError("k-means deterministic sample was not completely filled")
        params = dict(KMEANS_DEFAULTS)
        if kmeans_params:
            params.update(kmeans_params)
        params.pop("algorithm", None)
        params.pop("version", None)
        params.pop("reservoir_size", None)
        params["seed"] = params.pop("random_state", self.seed)

        result: dict[int, dict[str, Any]] = {}
        for layer in self.layers:
            shrunk, shrinkage, empirical = ledoit_wolf_from_moments(
                self.covariance_moments[layer]
            )
            inverse_sqrt, eigenvalues = covariance_inverse_sqrt(shrunk)
            prototypes, kmeans_report = deterministic_minibatch_kmeans(
                self.kmeans_values[layer].astype(np.float32),
                device=kmeans_device,
                **params,
            )
            result[layer] = {
                "mean": (self.all_sum[layer] / float(self.all_count)).astype(np.float32),
                "covariance": shrunk.astype(np.float32),
                "empirical_covariance": empirical.astype(np.float32),
                "inverse_sqrt": inverse_sqrt,
                "covariance_eigenvalues": eigenvalues,
                "ledoit_wolf_shrinkage": float(shrinkage),
                "prototypes": prototypes,
                "all_token_count": int(self.all_count),
                "covariance_sample_count": int(self.covariance_moments[layer].count),
                "kmeans_sample_count": int(self.kmeans_indices.size),
                "kmeans_report": kmeans_report,
                "sampling_provenance": dict(self.sampling_provenance),
            }
        return result


def save_membership_statistics_atomic(
    path: str | Path,
    statistics: Mapping[int, Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write the exact ``args.cka_gt_pilot_membership_stats`` NPZ atomically."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    layers = np.asarray(sorted(statistics), dtype=np.int16)
    if layers.size == 0:
        raise ValueError("cannot save empty membership statistics")
    payload: dict[str, Any] = {"layers": layers}
    summary: dict[str, Any] = {"metadata": dict(metadata or {}), "layers": {}}
    provenance_values = [
        statistics[int(layer)].get("sampling_provenance") for layer in layers.tolist()
    ]
    if any(value is not None for value in provenance_values):
        if any(value != provenance_values[0] for value in provenance_values[1:]):
            raise ValueError("membership sampling provenance differs across layers")
        summary["sampling_provenance"] = provenance_values[0]
    for layer in layers.tolist():
        values = statistics[int(layer)]
        prefix = f"layer_{int(layer)}_"
        for name in (
            "mean",
            "covariance",
            "empirical_covariance",
            "inverse_sqrt",
            "covariance_eigenvalues",
            "prototypes",
        ):
            payload[prefix + name] = np.asarray(values[name])
        summary["layers"][str(int(layer))] = {
            "ledoit_wolf_shrinkage": float(values["ledoit_wolf_shrinkage"]),
            "all_token_count": int(values["all_token_count"]),
            "covariance_sample_count": int(values["covariance_sample_count"]),
            "kmeans_sample_count": int(values["kmeans_sample_count"]),
            "kmeans_report": values["kmeans_report"],
        }
    payload["metadata_json"] = np.asarray(json.dumps(summary, sort_keys=True))
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as check:
            if not np.array_equal(check["layers"], layers):
                raise RuntimeError("membership NPZ layer round-trip failed")
            for layer in layers.tolist():
                if check[f"layer_{int(layer)}_mean"].shape != np.asarray(
                    statistics[int(layer)]["mean"]
                ).shape:
                    raise RuntimeError("membership NPZ shape round-trip failed")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class LoadedMembershipStatistics(dict[int, dict[str, Any]]):
    """Layer statistics plus immutable file and source provenance."""

    def __init__(
        self,
        values: Mapping[int, Mapping[str, Any]],
        *,
        saved_metadata: Mapping[str, Any],
        membership_identity: Mapping[str, Any],
    ) -> None:
        super().__init__((int(layer), dict(stats)) for layer, stats in values.items())
        self.saved_metadata = dict(saved_metadata)
        self.membership_identity = dict(membership_identity)


def _membership_saved_metadata_summary(
    metadata: Mapping[str, Any], layers: Sequence[int]
) -> dict[str, Any]:
    source = metadata.get("metadata", {})
    if not isinstance(source, Mapping):
        raise ValueError("membership metadata_json.metadata must be an object")
    reference = source.get("reference_load")
    return {
        "analysis": source.get("analysis"),
        "mode": source.get("mode"),
        "source_domain": source.get("domain"),
        "source_split": source.get("requested_split"),
        "source_dataset_prefix": source.get("dataset_prefix"),
        "reference_checkpoint": os.path.realpath(str(reference)) if reference else None,
        "reference_step": source.get("reference_step"),
        "prepared_config_content_sha256": source.get(
            "prepared_config_content_sha256"
        ),
        "source_window_count": source.get("total_windows"),
        "source_token_count": source.get("total_tokens"),
        "seed": source.get("seed"),
        "max_windows": source.get("max_windows"),
        "layers": [int(layer) for layer in layers],
        "sampling_provenance": metadata.get("sampling_provenance"),
    }


def validate_membership_identity(
    identity: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_path: str | Path | None = None,
    expected_saved_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Reject a Pass1 NPZ that was produced for a different pilot source."""

    if identity.get("schema") != "cka_gt_pilot_membership_identity_v1":
        raise RuntimeError("unsupported membership identity schema")
    if expected_identity is not None and dict(identity) != dict(expected_identity):
        differing = sorted(
            key
            for key in set(identity) | set(expected_identity)
            if identity.get(key) != expected_identity.get(key)
        )
        raise RuntimeError(
            f"membership file identity mismatch for fields: {differing}"
        )
    if expected_path is not None and Path(str(identity.get("path"))).resolve() != Path(
        expected_path
    ).resolve():
        raise RuntimeError(
            "membership identity path mismatch: "
            f"saved={identity.get('path')} expected={Path(expected_path).resolve()}"
        )
    saved = identity.get("saved_metadata_summary")
    if not isinstance(saved, Mapping):
        raise RuntimeError("membership identity omits saved metadata summary")
    for key, expected in (expected_saved_metadata or {}).items():
        actual = saved.get(key)
        if actual != expected:
            raise RuntimeError(
                f"membership provenance mismatch for {key}: "
                f"saved={actual!r} expected={expected!r}"
            )


def load_membership_statistics(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_saved_metadata: Mapping[str, Any] | None = None,
) -> LoadedMembershipStatistics:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        layers = data["layers"].astype(np.int16).tolist()
        metadata = json.loads(str(data["metadata_json"].item()))
        result: dict[int, dict[str, Any]] = {}
        for layer in layers:
            prefix = f"layer_{int(layer)}_"
            result[int(layer)] = {
                name: data[prefix + name].copy()
                for name in (
                    "mean",
                    "covariance",
                    "empirical_covariance",
                    "inverse_sqrt",
                    "covariance_eigenvalues",
                    "prototypes",
                )
            }
            result[int(layer)].update(metadata["layers"][str(int(layer))])
            if "sampling_provenance" in metadata:
                result[int(layer)]["sampling_provenance"] = metadata[
                    "sampling_provenance"
                ]
    identity = {
        "schema": "cka_gt_pilot_membership_identity_v1",
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
        "saved_metadata_summary": _membership_saved_metadata_summary(
            metadata, layers
        ),
    }
    validate_membership_identity(
        identity,
        expected_identity=expected_identity,
        expected_path=path,
        expected_saved_metadata=expected_saved_metadata,
    )
    return LoadedMembershipStatistics(
        result, saved_metadata=metadata, membership_identity=identity
    )


def recover_pass1_worker_metadata_from_stats(
    statistics_path: str | Path,
    worker_metadata_path: str | Path,
    *,
    expected_saved_metadata: Mapping[str, Any],
    expected_runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover the narrow NPZ-committed/worker-metadata-missing crash case.

    The membership NPZ is already an atomic immutable commit and embeds the
    complete Pass1 model-driver metadata.  Recovery validates its frozen source
    identity, then atomically reconstructs only the missing completion record.
    An existing worker metadata file is never overwritten here.
    """

    statistics_path = Path(statistics_path).resolve()
    worker_metadata_path = Path(worker_metadata_path).resolve()
    if worker_metadata_path.exists():
        raise FileExistsError(
            f"Pass1 worker metadata already exists: {worker_metadata_path}"
        )
    loaded = load_membership_statistics(
        statistics_path,
        expected_saved_metadata=expected_saved_metadata,
    )
    embedded = loaded.saved_metadata.get("metadata")
    if not isinstance(embedded, Mapping):
        raise RuntimeError("membership NPZ omits embedded Pass1 metadata")
    for key, expected in (expected_runtime_metadata or {}).items():
        if embedded.get(key) != expected:
            raise RuntimeError(
                f"embedded Pass1 runtime metadata mismatch for {key}: "
                f"saved={embedded.get(key)!r} expected={expected!r}"
            )
    elapsed_raw = embedded.get("cumulative_elapsed_seconds")
    elapsed = (
        float(elapsed_raw)
        if elapsed_raw is not None and math.isfinite(float(elapsed_raw))
        else None
    )
    if elapsed is not None and elapsed <= 0.0:
        raise RuntimeError("embedded Pass1 cumulative elapsed time is non-positive")
    final = {
        **dict(embedded),
        "completed": True,
        "membership_statistics": str(statistics_path),
        "membership_identity": loaded.membership_identity,
        "elapsed_seconds": elapsed,
        "cumulative_elapsed_seconds": elapsed,
        "resume_recovered_after_stats_commit": True,
        "timing_available": elapsed is not None,
    }
    _atomic_json(worker_metadata_path, final)
    return final


def membership_scores(
    hidden: torch.Tensor,
    statistics: Mapping[str, Any],
    *,
    token_block_size: int = 8_192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Square-root whitened Mahalanobis and nearest-prototype distances."""

    original_shape = hidden.shape[:-1]
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    device = flat.device
    mean = torch.as_tensor(statistics["mean"], dtype=torch.float32, device=device)
    inverse_sqrt = torch.as_tensor(
        statistics["inverse_sqrt"], dtype=torch.float32, device=device
    )
    prototypes = torch.as_tensor(
        statistics["prototypes"], dtype=torch.float32, device=device
    )
    if mean.shape != (flat.shape[-1],) or inverse_sqrt.shape != (
        flat.shape[-1],
        flat.shape[-1],
    ):
        raise ValueError("membership statistics hidden dimension mismatch")
    maha_blocks = []
    proto_blocks = []
    for start in range(0, flat.shape[0], token_block_size):
        block = flat[start : start + token_block_size]
        centered = block - mean
        whitened = centered @ inverse_sqrt
        maha_blocks.append(torch.linalg.vector_norm(whitened, dim=-1))
        x2 = centered.square().sum(dim=-1, keepdim=True)
        centered_prototypes = prototypes - mean
        p2 = centered_prototypes.square().sum(dim=-1).unsqueeze(0)
        distance2 = (x2 + p2 - 2.0 * centered @ centered_prototypes.T).clamp_min(0.0)
        proto_blocks.append(torch.sqrt(distance2.min(dim=-1).values))
    maha = torch.cat(maha_blocks).reshape(original_shape).float()
    proto = torch.cat(proto_blocks).reshape(original_shape).float()
    return maha, proto


def _nanmin_update(target: torch.Tensor, candidate: torch.Tensor) -> None:
    finite = torch.isfinite(candidate)
    replace = finite & ((~torch.isfinite(target)) | (candidate < target))
    target[replace] = candidate[replace]


def _nanmax_update(target: torch.Tensor, candidate: torch.Tensor) -> None:
    finite = torch.isfinite(candidate)
    replace = finite & ((~torch.isfinite(target)) | (candidate > target))
    target[replace] = candidate[replace]


def _nanmin_update_with_id(
    target: torch.Tensor,
    target_id: torch.Tensor,
    candidate: torch.Tensor,
    candidate_id: int,
) -> torch.Tensor:
    finite = torch.isfinite(candidate)
    replace = finite & ((~torch.isfinite(target)) | (candidate < target))
    target[replace] = candidate[replace]
    target_id[replace] = int(candidate_id)
    return replace


def _window_uids(domain: str, split: str, sample_orders: np.ndarray) -> np.ndarray:
    domain_ids = {"code": 0, "wiki": 1}
    split_ids = {"calibration": 0, "selection": 1, "test": 2}
    if domain not in domain_ids or split not in split_ids:
        raise ValueError(f"unsupported domain/split identity: {domain}/{split}")
    # Fixed decimal fields keep IDs globally unique and human-auditable while
    # staying far below int64 limits for the 100K-window pilot.
    return (
        np.int64(domain_ids[domain]) * np.int64(1_000_000_000)
        + np.int64(split_ids[split]) * np.int64(100_000_000)
        + np.asarray(sample_orders, dtype=np.int64)
    )


class RandomPairDonorCache:
    """Memory-only deterministic AFTER-window donors for singleton nulls.

    The candidate identity is frozen from the first ``max_windows`` full
    512-token windows in one split/worker partition.  Hidden states are added
    only for those candidates and are never serialized.  Re-forwarding the
    same candidates after a restart reconstructs the identical cache.
    """

    def __init__(
        self,
        *,
        domain: str,
        split: str,
        layers: Sequence[int],
        manifest_rows: np.ndarray,
        max_windows: int = RANDOM_PAIR_DONOR_CACHE_SIZE,
    ) -> None:
        rows = np.asarray(manifest_rows)
        if rows.dtype != WINDOW_DTYPE or rows.ndim != 1:
            raise ValueError("donor manifest_rows must be one-dimensional WINDOW_DTYPE")
        if domain not in DOMAIN_IDS or split not in SPLIT_IDS:
            raise ValueError("unsupported donor cache domain/split")
        self.domain = str(domain)
        self.split = str(split)
        self.layers = tuple(sorted(int(layer) for layer in layers))
        self.max_windows = int(max_windows)
        if not self.layers or self.max_windows <= 0:
            raise ValueError("donor cache layers/max_windows must be non-empty/positive")
        full = rows[rows["window_length"] == 512]
        if full.size:
            full = full[
                np.argsort(full["sample_order"].astype(np.int64), kind="stable")
            ]
        self.candidate_rows = full[: self.max_windows].copy()
        self.candidate_window_uids = tuple(
            int(value)
            for value in _window_uids(
                self.domain,
                self.split,
                self.candidate_rows["sample_order"].astype(np.int64),
            ).tolist()
        )
        self._candidate_set = set(self.candidate_window_uids)
        self._after_by_uid: dict[int, dict[int, torch.Tensor]] = {}

    @property
    def cached_window_uids(self) -> tuple[int, ...]:
        return tuple(
            uid for uid in self.candidate_window_uids if uid in self._after_by_uid
        )

    @property
    def complete(self) -> bool:
        return len(self._after_by_uid) == len(self.candidate_window_uids)

    def rows_needed_from(self, rows: np.ndarray) -> np.ndarray:
        """Mask rows whose candidate AFTER hidden is not cached yet."""

        values = np.asarray(rows)
        if values.dtype != WINDOW_DTYPE or values.ndim != 1:
            raise ValueError("rows must be one-dimensional WINDOW_DTYPE")
        uids = _window_uids(
            self.domain, self.split, values["sample_order"].astype(np.int64)
        )
        return np.asarray(
            [
                int(uid) in self._candidate_set
                and int(uid) not in self._after_by_uid
                for uid in uids.tolist()
            ],
            dtype=bool,
        )

    def add(
        self,
        after_by_layer: Mapping[int, torch.Tensor],
        manifest_rows: np.ndarray,
    ) -> None:
        rows = np.asarray(manifest_rows)
        if rows.dtype != WINDOW_DTYPE or rows.ndim != 1:
            raise ValueError("donor rows must be one-dimensional WINDOW_DTYPE")
        if tuple(sorted(int(layer) for layer in after_by_layer)) != self.layers:
            raise ValueError("donor AFTER layer set differs from cache layer set")
        batch = int(rows.shape[0])
        for layer in self.layers:
            values = after_by_layer[layer]
            if values.ndim != 3 or values.shape[:2] != (batch, 512):
                raise ValueError("donor AFTER values must be [batch,512,hidden]")
        uids = _window_uids(
            self.domain, self.split, rows["sample_order"].astype(np.int64)
        )
        for row_index, raw_uid in enumerate(uids.tolist()):
            uid = int(raw_uid)
            if uid not in self._candidate_set:
                continue
            candidate = {
                layer: after_by_layer[layer][row_index].detach().clone()
                for layer in self.layers
            }
            previous = self._after_by_uid.get(uid)
            if previous is not None:
                if any(
                    not torch.equal(previous[layer], candidate[layer])
                    for layer in self.layers
                ):
                    raise RuntimeError(
                        "deterministic donor cache rebuild changed an AFTER hidden"
                    )
                continue
            self._after_by_uid[uid] = candidate

    @staticmethod
    def _choice_index(
        window_uid: int, scale: int, chunk_start: int, count: int
    ) -> int:
        identity = np.asarray(
            [int(window_uid), int(scale), int(chunk_start)], dtype="<i8"
        )
        digest = hashlib.sha256(identity.tobytes()).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % int(count)

    def donor_chunk(
        self,
        *,
        window_uid: int,
        scale: int,
        chunk_start: int,
        chunk_length: int,
        layer: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, int, int]:
        """Return a deterministic non-self donor slice, UID, and reason code."""

        if int(layer) not in self.layers:
            raise ValueError(f"donor cache omits layer {layer}")
        available = [
            uid
            for uid in self.candidate_window_uids
            if uid in self._after_by_uid and uid != int(window_uid)
        ]
        if not available:
            return None, -1, RANDOM_PAIR_NO_NONSELF_DONOR
        choice = self._choice_index(window_uid, scale, chunk_start, len(available))
        donor_uid = int(available[choice])
        stop = int(chunk_start) + int(chunk_length)
        if chunk_start < 0 or chunk_length <= 0 or stop > 512:
            raise ValueError("donor chunk slice lies outside a full 512-token window")
        donor = self._after_by_uid[donor_uid][int(layer)][chunk_start:stop]
        return donor.to(device=device), donor_uid, RANDOM_PAIR_VALID

    def metadata(self) -> dict[str, Any]:
        return {
            "policy": "first_8_full_after_windows_per_split_worker_memory_only",
            "max_windows": self.max_windows,
            "candidate_window_uids": list(self.candidate_window_uids),
            "cached_window_uids": list(self.cached_window_uids),
            "raw_hidden_disk_storage": False,
        }


def _seed_from_identity(base_seed: int, *values: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), *(int(value) for value in values)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _within_chunk_permutations(
    sample_orders: np.ndarray,
    *,
    scale: int,
    chunk_start: int,
    chunk_length: int,
    base_seed: int,
    device: torch.device,
) -> torch.Tensor:
    permutations = []
    for sample_order in sample_orders.tolist():
        seed = _seed_from_identity(base_seed, sample_order, scale, chunk_start, 17)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutations.append(
            torch.randperm(chunk_length, generator=generator, device="cpu")
        )
    return torch.stack(permutations, dim=0).to(device=device)


def _batched_permutation_null_cka(
    before: torch.Tensor,
    after: torch.Tensor,
    permutations: torch.Tensor,
) -> torch.Tensor:
    if before.ndim != 3 or before.shape != after.shape:
        raise ValueError("batched permutation null expects [batch,tokens,hidden]")
    if permutations.shape != before.shape[:2]:
        raise ValueError("permutation shape mismatch")
    gather = permutations.unsqueeze(-1).expand(-1, -1, before.shape[-1])
    shuffled = torch.gather(after, dim=1, index=gather)
    return centered_linear_cka_metrics(before, shuffled)["cka"]


def _flatten_router_diagnostics(
    diagnostics_by_layer: Mapping[int, Mapping[str, torch.Tensor]] | None,
    *,
    batch: int,
    length: int,
) -> dict[str, np.ndarray]:
    if diagnostics_by_layer is None:
        return {}
    layers = sorted(diagnostics_by_layer)
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for layer in layers:
        diagnostics = diagnostics_by_layer[layer]
        for name in ("top4_id", "top4_weight", "old_full_mass", "old_selected_mass"):
            if name not in diagnostics:
                raise ValueError(f"router layer {layer} omitted {name}")
            value = diagnostics[name]
            if value.shape[:2] == (length, batch):
                value = value.transpose(0, 1)
            if value.shape[:2] != (batch, length):
                raise ValueError(
                    f"router layer {layer} {name} is not aligned to [batch,length]"
                )
            outputs[name].append(value)
    result: dict[str, np.ndarray] = {
        "router_layer_numbers": np.tile(
            np.asarray(layers, dtype=np.int16), (batch * length, 1)
        )
    }
    for name, values in outputs.items():
        # [B,L,moe_layer,(topk)] -> one wide list value per token row.
        stacked = torch.stack(values, dim=2)
        result[name] = stacked.reshape(batch * length, *stacked.shape[2:]).detach().cpu().numpy()
    return result


def process_pass2_metric_batch(
    *,
    before_by_layer: Mapping[int, torch.Tensor],
    after_by_layer: Mapping[int, torch.Tensor],
    token_ids: torch.Tensor,
    manifest_rows: np.ndarray,
    domain: str,
    split: str,
    membership_statistics: Mapping[int, Mapping[str, Any]] | None = None,
    router_before: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    router_after: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
    random_pair_donor_cache: RandomPairDonorCache | None = None,
    seed: int = 1234,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute all scalar Pass2 metrics for one same-length window batch.

    Returns wide token columns and long chunk columns ready for the two atomic
    shard writers.  Raw hidden states and RSMs are released by the caller after
    this function returns.
    """

    layers = tuple(sorted(int(layer) for layer in before_by_layer))
    if layers != tuple(sorted(int(layer) for layer in after_by_layer)):
        raise ValueError("before/after layer sets differ")
    if not layers:
        raise ValueError("at least one residual layer is required")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be [batch,length]")
    batch, length = token_ids.shape
    rows = np.asarray(manifest_rows)
    if rows.dtype != WINDOW_DTYPE or rows.shape != (batch,):
        raise ValueError("manifest_rows must align one-to-one with the batch")
    if np.any(rows["window_length"] != length):
        raise ValueError("manifest length differs from unpadded token batch length")

    hidden_size: int | None = None
    for layer in layers:
        before = before_by_layer[layer]
        after = after_by_layer[layer]
        if before.shape != after.shape or before.shape[:2] != (batch, length):
            raise ValueError(f"layer {layer} hidden is not [batch,length,hidden]")
        if hidden_size is None:
            hidden_size = int(before.shape[-1])
        elif before.shape[-1] != hidden_size:
            raise ValueError("hidden size differs across layers")

    layer_count = len(layers)
    device = next(iter(before_by_layer.values())).device
    uncentered_names = ("cosine", "rel_l2", "sym_rel_l2", "log_r", "ref_rms")
    uncentered = {
        name: torch.empty((batch, length, layer_count), dtype=torch.float32, device=device)
        for name in uncentered_names
    }
    maha = torch.full_like(uncentered["cosine"], float("nan"))
    proto = torch.full_like(uncentered["cosine"], float("nan"))
    for layer_index, layer in enumerate(layers):
        metrics = uncentered_token_metrics(before_by_layer[layer], after_by_layer[layer])
        for name in uncentered_names:
            uncentered[name][..., layer_index] = metrics[name]
        if membership_statistics is not None:
            if layer not in membership_statistics:
                raise ValueError(f"membership statistics omit layer {layer}")
            maha_layer, proto_layer = membership_scores(
                before_by_layer[layer], membership_statistics[layer]
            )
            maha[..., layer_index] = maha_layer
            proto[..., layer_index] = proto_layer

    aggregate: dict[int, dict[str, torch.Tensor]] = {}
    for scale in (128, 256):
        shape = (batch, length, layer_count)
        aggregate[scale] = {
            "cka_min": torch.full(shape, float("nan"), device=device),
            "s_min": torch.full(shape, float("nan"), device=device),
            "r_min": torch.full(shape, float("nan"), device=device),
            "r_max": torch.full(shape, float("nan"), device=device),
            "r_sum": torch.zeros(shape, device=device),
            "r_count": torch.zeros(shape, dtype=torch.int16, device=device),
            "worst_diag_ratio": torch.full(shape, float("nan"), device=device),
            "worst_chunk_id": torch.full(
                shape, -1, dtype=torch.int32, device=device
            ),
            "s_at_worst_cka": torch.full(shape, float("nan"), device=device),
            "worst_s_chunk_id": torch.full(
                shape, -1, dtype=torch.int32, device=device
            ),
            "neg_contrib": torch.zeros(shape, dtype=torch.bool, device=device),
            "offdiag_warning": torch.zeros(shape, dtype=torch.bool, device=device),
        }

    chunk_records: dict[str, list[Any]] = defaultdict(list)
    sample_orders = rows["sample_order"].astype(np.int64)
    batch_window_uids = _window_uids(domain, split, sample_orders)
    layout = window_chunk_layout(length)
    for scale in (128, 256, 512):
        if scale == 512:
            chunk_length = length
            starts = np.asarray([0], dtype=np.int32)
        else:
            chunk_length = scale
            starts = chunk_starts(length, scale, scale // 2)
        for chunk_start_value in starts.tolist():
            chunk_start_value = int(chunk_start_value)
            chunk_stop = chunk_start_value + chunk_length
            permutations = _within_chunk_permutations(
                sample_orders,
                scale=scale,
                chunk_start=chunk_start_value,
                chunk_length=chunk_length,
                base_seed=seed,
                device=device,
            )
            pair_indices = (
                random_pair_indices(
                    batch,
                    seed=_seed_from_identity(seed, scale, chunk_start_value, 29),
                ).to(device=device)
                if batch >= 2
                else None
            )
            within_batch_donor_uids = (
                batch_window_uids[pair_indices.detach().cpu().numpy()]
                if pair_indices is not None
                else None
            )
            for layer_index, layer in enumerate(layers):
                x = before_by_layer[layer][:, chunk_start_value:chunk_stop, :]
                y = after_by_layer[layer][:, chunk_start_value:chunk_stop, :]
                values = centered_linear_cka_metrics(x, y)
                permutation_cka = _batched_permutation_null_cka(x, y, permutations)
                if pair_indices is not None:
                    random_pair_cka = random_pair_null_metrics(
                        x, y, pair_indices=pair_indices
                    )["cka"]
                    random_pair_reason = torch.full(
                        (batch,),
                        RANDOM_PAIR_VALID,
                        dtype=torch.uint8,
                        device=device,
                    )
                    invalid_null = ~torch.isfinite(random_pair_cka)
                    random_pair_reason[invalid_null] = RANDOM_PAIR_METRIC_INVALID
                    random_pair_donor_uids = torch.as_tensor(
                        within_batch_donor_uids,
                        dtype=torch.int64,
                        device=device,
                    )
                elif random_pair_donor_cache is not None:
                    window_uid = int(batch_window_uids[0])
                    donor, donor_uid, reason = random_pair_donor_cache.donor_chunk(
                        window_uid=window_uid,
                        scale=scale,
                        chunk_start=chunk_start_value,
                        chunk_length=chunk_length,
                        layer=layer,
                        device=device,
                    )
                    random_pair_reason = torch.full(
                        (1,), reason, dtype=torch.uint8, device=device
                    )
                    random_pair_donor_uids = torch.full(
                        (1,), donor_uid, dtype=torch.int64, device=device
                    )
                    if donor is not None:
                        random_pair_cka = centered_linear_cka_metrics(
                            x, donor.unsqueeze(0)
                        )["cka"]
                        if not bool(torch.isfinite(random_pair_cka[0]).item()):
                            random_pair_reason[0] = RANDOM_PAIR_METRIC_INVALID
                    else:
                        random_pair_cka = torch.full(
                            (1,), float("nan"), dtype=torch.float32, device=device
                        )
                else:
                    random_pair_cka = torch.full(
                        (batch,), float("nan"), dtype=torch.float32, device=device
                    )
                    random_pair_reason = torch.full(
                        (batch,),
                        RANDOM_PAIR_NO_CACHE,
                        dtype=torch.uint8,
                        device=device,
                    )
                    random_pair_donor_uids = torch.full(
                        (batch,), -1, dtype=torch.int64, device=device
                    )

                if scale in aggregate:
                    token_slice = slice(chunk_start_value, chunk_stop)
                    cka_expanded = values["cka"][:, None].expand(-1, chunk_length)
                    diag_expanded = values["diag_ratio"][:, None].expand(-1, chunk_length)
                    warning_expanded = values["offdiag_warning"][:, None].expand(
                        -1, chunk_length
                    )
                    target = aggregate[scale]
                    cka_replaced = _nanmin_update_with_id(
                        target["cka_min"][:, token_slice, layer_index],
                        target["worst_chunk_id"][:, token_slice, layer_index],
                        cka_expanded,
                        chunk_start_value,
                    )
                    diag_target = target["worst_diag_ratio"][:, token_slice, layer_index]
                    diag_target[cka_replaced] = diag_expanded[cka_replaced]
                    s_at_worst = target["s_at_worst_cka"][:, token_slice, layer_index]
                    s_at_worst[cka_replaced] = values["s_i"][cka_replaced]
                    _nanmin_update_with_id(
                        target["s_min"][:, token_slice, layer_index],
                        target["worst_s_chunk_id"][:, token_slice, layer_index],
                        values["s_i"],
                        chunk_start_value,
                    )
                    _nanmin_update(
                        target["r_min"][:, token_slice, layer_index], values["r_i"]
                    )
                    _nanmax_update(
                        target["r_max"][:, token_slice, layer_index], values["r_i"]
                    )
                    finite_r = torch.isfinite(values["r_i"])
                    target["r_sum"][:, token_slice, layer_index] += torch.nan_to_num(
                        values["r_i"], nan=0.0
                    )
                    target["r_count"][:, token_slice, layer_index] += finite_r.to(
                        torch.int16
                    )
                    target["neg_contrib"][:, token_slice, layer_index] |= values[
                        "neg_contrib"
                    ]
                    target["offdiag_warning"][:, token_slice, layer_index] |= warning_expanded

                for batch_index in range(batch):
                    chunk_records["domain"].append(domain)
                    chunk_records["split"].append(split)
                    window_uid = int(batch_window_uids[batch_index])
                    chunk_uid = (
                        window_uid * 100_000
                        + scale * 100
                        + chunk_start_value
                        + layer
                    )
                    chunk_records["chunk_uid"].append(chunk_uid)
                    chunk_records["window_uid"].append(window_uid)
                    chunk_records["sample_order"].append(int(rows[batch_index]["sample_order"]))
                    chunk_records["document_id"].append(int(rows[batch_index]["document_id"]))
                    chunk_records["window_offset"].append(int(rows[batch_index]["window_offset"]))
                    chunk_records["scale"].append(scale)
                    chunk_records["chunk_start"].append(chunk_start_value)
                    chunk_records["chunk_length"].append(chunk_length)
                    chunk_records["layer"].append(layer)
                    for name in (
                        "cka",
                        "cka_off",
                        "diag_ratio",
                        "centered_var_x",
                        "centered_var_y",
                        "k_norm",
                        "l_norm",
                    ):
                        chunk_records[name].append(float(values[name][batch_index].item()))
                    chunk_records["invalid_reason"].append(
                        int(values["invalid_reason"][batch_index].item())
                    )
                    chunk_records["t_invalid_reason"].append(
                        int(values["t_invalid_reason"][batch_index].item())
                    )
                    chunk_records["offdiag_warning"].append(
                        bool(values["offdiag_warning"][batch_index].item())
                    )
                    chunk_records["cka_permutation"].append(
                        float(permutation_cka[batch_index].item())
                    )
                    chunk_records["cka_random_pair"].append(
                        float(random_pair_cka[batch_index].item())
                    )
                    chunk_records["random_pair_invalid_reason"].append(
                        int(random_pair_reason[batch_index].item())
                    )
                    chunk_records["random_pair_donor_window_uid"].append(
                        int(random_pair_donor_uids[batch_index].item())
                    )
                    # Raw-unit diagnostic arrays required by the approved s_i
                    # safety check; these are scalar lists, not hidden vectors.
                    for name in ("c_i", "c_i_off", "s_i", "r_i"):
                        chunk_records[name].append(
                            values[name][batch_index].detach().cpu().numpy().astype(np.float32)
                        )

    positions = torch.arange(length, device=token_ids.device).unsqueeze(0).expand(batch, -1)
    eligible_counts = torch.as_tensor(
        rows["eligible_token_count"].astype(np.int64), device=token_ids.device
    )
    eligible = positions < eligible_counts[:, None]
    row_count = batch * length
    window_uids = _window_uids(domain, split, rows["sample_order"])
    token_columns: dict[str, Any] = {
        "domain": np.repeat(np.asarray([domain], dtype=object), row_count),
        "split": np.repeat(np.asarray([split], dtype=object), row_count),
        "window_uid": np.repeat(window_uids, length),
        "sample_order": np.repeat(rows["sample_order"].astype(np.int64), length),
        "source_window_index": np.repeat(
            rows["source_window_index"].astype(np.int64), length
        ),
        "document_id": np.repeat(rows["document_id"].astype(np.int64), length),
        "window_offset": np.repeat(rows["window_offset"].astype(np.int64), length),
        "window_length": np.repeat(rows["window_length"].astype(np.int32), length),
        "position": np.tile(np.arange(length, dtype=np.int16), batch),
        "document_token_offset": np.repeat(
            rows["window_offset"].astype(np.int64), length
        ) + np.tile(np.arange(length, dtype=np.int64), batch),
        "token_id": token_ids.detach().cpu().numpy().reshape(-1).astype(np.int32),
        "eligible": eligible.detach().cpu().numpy().reshape(-1),
        "layer_numbers": np.tile(np.asarray(layers, dtype=np.int16), (row_count, 1)),
        "maha": maha.detach().cpu().numpy().reshape(row_count, layer_count),
        "proto": proto.detach().cpu().numpy().reshape(row_count, layer_count),
    }
    for name, values in (("maha_mean", maha), ("proto_mean", proto)):
        finite = torch.isfinite(values)
        count = finite.sum(dim=-1)
        mean = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).sum(
            dim=-1
        ) / count.clamp_min(1)
        mean = torch.where(count > 0, mean, torch.full_like(mean, float("nan")))
        token_columns[name] = mean.detach().cpu().numpy().reshape(row_count)
    canonical_uncentered_names = {
        "cosine": "cosine",
        "rel_l2": "relative_l2",
        "sym_rel_l2": "symmetric_relative_l2",
        "log_r": "log_r",
        "ref_rms": "ref_rms",
    }
    for source_name, canonical_name in canonical_uncentered_names.items():
        value = uncentered[source_name]
        token_columns[canonical_name] = value.detach().cpu().numpy().reshape(
            row_count, layer_count
        )
    for scale, values in aggregate.items():
        for name in (
            "cka_min",
            "s_min",
            "r_min",
            "r_max",
            "worst_diag_ratio",
            "neg_contrib",
            "offdiag_warning",
        ):
            token_columns[f"{name}_{scale}"] = (
                values[name].detach().cpu().numpy().reshape(row_count, layer_count)
            )
        token_columns[f"worst_chunk_id_{scale}"] = (
            values["worst_chunk_id"]
            .detach()
            .cpu()
            .numpy()
            .reshape(row_count, layer_count)
        )
        token_columns[f"s_at_worst_cka_{scale}"] = (
            values["s_at_worst_cka"]
            .detach()
            .cpu()
            .numpy()
            .reshape(row_count, layer_count)
        )
        token_columns[f"worst_s_chunk_id_{scale}"] = (
            values["worst_s_chunk_id"]
            .detach()
            .cpu()
            .numpy()
            .reshape(row_count, layer_count)
        )
        count = values["r_count"]
        r_mean = values["r_sum"] / count.clamp_min(1)
        r_mean = torch.where(count > 0, r_mean, torch.full_like(r_mean, float("nan")))
        token_columns[f"r_mean_{scale}"] = (
            r_mean.detach().cpu().numpy().reshape(row_count, layer_count)
        )
        token_columns[f"valid_cka_{scale}"] = (
            torch.isfinite(values["cka_min"])
            .detach()
            .cpu()
            .numpy()
            .reshape(row_count, layer_count)
        )
        token_columns[f"valid_s_{scale}"] = (
            torch.isfinite(values["s_min"])
            .detach()
            .cpu()
            .numpy()
            .reshape(row_count, layer_count)
        )

    for prefix, diagnostics in (("before", router_before), ("after", router_after)):
        for name, value in _flatten_router_diagnostics(
            diagnostics, batch=batch, length=length
        ).items():
            canonical_router_name = {
                "top4_id": "top4_ids",
                "top4_weight": "top4_weight",
                "old_full_mass": "old_full_mass",
                "old_selected_mass": "old_selected_mass",
                "router_layer_numbers": "router_layer_numbers",
            }[name]
            token_columns[f"{prefix}_{canonical_router_name}"] = value

    chunk_columns: dict[str, Any] = {}
    chunk_float_columns = {
        "cka",
        "cka_off",
        "diag_ratio",
        "centered_var_x",
        "centered_var_y",
        "k_norm",
        "l_norm",
        "cka_permutation",
        "cka_random_pair",
    }
    chunk_int_dtypes = {
        "chunk_uid": np.int64,
        "window_uid": np.int64,
        "sample_order": np.int64,
        "document_id": np.int64,
        "window_offset": np.int64,
        "scale": np.int16,
        "chunk_start": np.int16,
        "chunk_length": np.int16,
        "layer": np.int16,
        "invalid_reason": np.uint8,
        "t_invalid_reason": np.uint8,
        "random_pair_invalid_reason": np.uint8,
        "random_pair_donor_window_uid": np.int64,
    }
    for name, values in chunk_records.items():
        if name in ("c_i", "c_i_off", "s_i", "r_i"):
            chunk_columns[name] = values
        elif name in chunk_float_columns:
            chunk_columns[name] = np.asarray(values, dtype=np.float32)
        elif name in chunk_int_dtypes:
            chunk_columns[name] = np.asarray(values, dtype=chunk_int_dtypes[name])
        else:
            chunk_columns[name] = np.asarray(values)
    return token_columns, chunk_columns


def run_model_pilot(
    *,
    model: Any,
    teacher: Any,
    args: Any,
    hooks: Mapping[str, Callable[..., Any]],
    print_fn: Callable[[str], None] = print,
) -> Any:
    """Stable integration surface for the model-side CKA pilot driver.

    The distributed checkpoint owner supplies a single ``driver`` callback in
    ``hooks``.  Keeping model forwarding in that owner avoids duplicating
    pipeline/tensor-parallel semantics here, while the callback receives this
    module's tested components through ordinary imports.  The exact public
    signature is intentionally small and will remain stable:

    ``run_model_pilot(model=model, teacher=teacher, args=args,
    hooks={"driver": callback}, print_fn=print_rank_0)``.
    """

    driver = hooks.get("driver")
    if driver is None or not callable(driver):
        raise ValueError("run_model_pilot requires callable hooks['driver']")
    with fp32_metric_context() as precision_audit:
        print_fn(
            "CKA GT pilot metric context: float32, "
            f"CUDA TF32 disabled (previous={precision_audit['cuda_matmul_allow_tf32_before']})"
        )
        result = driver(
            model=model,
            teacher=teacher,
            args=args,
            precision_audit=precision_audit,
            print_fn=print_fn,
        )
    return result


__all__ = [
    "RUNTIME_SCHEMA_VERSION",
    "KMEANS_DEFAULTS",
    "fp32_metric_context",
    "build_attention_position_tensors",
    "WindowBatch",
    "DocumentWindowBatchReader",
    "router_diagnostics_from_outputs",
    "router_diagnostics_from_input",
    "AtomicArrowShardWriter",
    "TokenArrowShardWriter",
    "ChunkArrowShardWriter",
    "PairedMetricStreamPaths",
    "paired_metric_stream_paths",
    "PairedParquetMetricWriter",
    "build_exact_raw_unit_quantiles",
    "ledoit_wolf_from_moments",
    "covariance_inverse_sqrt",
    "deterministic_minibatch_kmeans",
    "Pass1MembershipAccumulator",
    "save_membership_statistics_atomic",
    "LoadedMembershipStatistics",
    "load_membership_statistics",
    "validate_membership_identity",
    "recover_pass1_worker_metadata_from_stats",
    "membership_scores",
    "RandomPairDonorCache",
    "RANDOM_PAIR_REASON_NAMES",
    "RANDOM_PAIR_DONOR_CACHE_SIZE",
    "process_pass2_metric_batch",
    "run_model_pilot",
]
