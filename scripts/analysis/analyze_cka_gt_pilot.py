#!/usr/bin/env python3
"""Post-process the CKA old-like-token pilot scalar artifacts.

This program deliberately does *not* choose a final threshold and does not
train a model.  It consumes the wide, one-row-per-contextual-token Parquet
artifacts produced by the paired-forward pilot and creates the candidate
threshold tables, selector comparisons, histograms, sealed test aggregates,
and ``REPORT.md`` required by the v1 specification.

Canonical token schema
----------------------
Identity columns are scalar::

    domain, split, window_uid, document_id, window_offset, window_length,
    position, document_token_offset, token_id

The following columns are fixed-size list[8], ordered by layers 2--9::

    cosine, relative_l2, symmetric_relative_l2, log_r, ref_rms, maha, proto,
    cka_min_128, cka_min_256, s_min_128, s_min_256,
    r_min_128, r_mean_128, r_max_128,
    r_min_256, r_mean_256, r_max_256,
    valid_cka_128, valid_cka_256, valid_s_128, valid_s_256,
    worst_chunk_id_128, worst_chunk_id_256,
    s_at_worst_cka_128, s_at_worst_cka_256,
    worst_s_chunk_id_128, worst_s_chunk_id_256

Routing IDs/weights are fixed-size list[8,4], and routing masses list[8].
The analyzer only requires the identity, cosine, relative_l2, log_r,
cka_min_*, and s_min_* columns.  Optional fields enrich diagnostics.

The chunk table is long and contains domain/split/scale/layer plus ``cka``,
``cka_permutation``, ``cka_random_pair``, ``diag_ratio`` and invalid-reason
columns. It also carries scalar ``cka_off``/``offdiag_warning`` and the raw
chunk-token ``s_i`` list used by the safety diagnostics. Exact raw-unit calibration quantiles are read from
``threshold_tables/raw_unit_quantiles.parquet``.  That table must identify
metric, domain, split, scale (nullable for magnitude metrics), layer,
quantile, value, count, and method.  Quantiles must have been computed over
the pre-aggregation units specified by the pilot, especially raw chunk-token
``s_i`` values rather than the token-level minimum.

The implementation uses streaming Arrow batches.  It never materializes the
full token table in RAM and it writes selector assignments incrementally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import html
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

# ``python /absolute/path/to/analyze_cka_gt_pilot.py ...`` puts only this
# file's ``scripts/analysis`` directory on ``sys.path``.  The context
# enrichment step intentionally imports the lightweight indexed-dataset
# reader through its repository-qualified name, so make that name resolvable
# independently of the caller's current working directory.  This is derived
# from ``__file__`` rather than cwd and therefore cannot accidentally select a
# different checkout.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.dataset as pads
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "analyze_cka_gt_pilot.py requires numpy and pyarrow. Run it with the "
        "flame-megatron-h100 Python environment."
    ) from exc


SCHEMA = "cka_gt_pilot_postprocess_v1"
LAYERS = tuple(range(2, 10))
SCALES = (128, 256)
# Scale 512 is the whole-window CKA control.  It is intentionally diagnostic
# only: candidate thresholds and GT consensus continue to use 128/256.
CHUNK_DIAGNOSTIC_SCALES = (128, 256, 512)
BUNDLES = (95, 97, 99)
DEFAULT_SEED = 1234
DEFAULT_BINS = 256
DEFAULT_BATCH_SIZE = 65_536
EPS = 1.0e-12
DEFAULT_LOCAL_TOKENIZER_PATH = Path(
    "/data2/seonghyeonnoh/homecache/huggingface/hub/"
    "models--EleutherAI--pythia-12b/snapshots/"
    "bb1e3e710cdf6b524461d543cfb5ba773f0a81b6"
)
INPUT_INVENTORY_SCHEMA = "cka_gt_pilot_parquet_inventory_v1"
ANALYSIS_BINDING_SCHEMA = "cka_gt_pilot_analysis_input_binding_v1"

IDENTITY_COLUMNS = (
    "domain",
    "split",
    "window_uid",
    "document_id",
    "window_offset",
    "window_length",
    "position",
    "document_token_offset",
    "token_id",
)

REQUIRED_TOKEN_COLUMNS = (
    "domain",
    "split",
    "window_uid",
    "document_id",
    "window_offset",
    "position",
    "token_id",
    "cosine",
    "relative_l2",
    "log_r",
    "cka_min_128",
    "cka_min_256",
    "s_min_128",
    "s_min_256",
)

OPTIONAL_TOKEN_COLUMNS = (
    "window_length",
    "document_token_offset",
    "symmetric_relative_l2",
    "ref_rms",
    "maha",
    "proto",
    "maha_mean",
    "proto_mean",
    "r_min_128",
    "r_mean_128",
    "r_max_128",
    "r_min_256",
    "r_mean_256",
    "r_max_256",
    "valid_cka_128",
    "valid_cka_256",
    "valid_s_128",
    "valid_s_256",
    "worst_chunk_id_128",
    "worst_chunk_id_256",
    "s_at_worst_cka_128",
    "s_at_worst_cka_256",
    "worst_s_chunk_id_128",
    "worst_s_chunk_id_256",
    "worst_diag_ratio_128",
    "worst_diag_ratio_256",
    "before_top4_ids",
    "after_top4_ids",
    "before_top4_weight",
    "after_top4_weight",
    "before_old_full_mass",
    "after_old_full_mass",
    "before_old_selected_mass",
    "after_old_selected_mass",
    "context_token_ids",
    "context_text",
)


def _json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_clean(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_clean(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepared_input_provenance(artifact_root: Path) -> dict[str, Any]:
    """Bind analysis outputs to the exact prepared datasets/checkpoints.

    The deep pre-analysis validator recomputes the physical identities.  This
    compact copy records those immutable identities in the analysis binding so
    a config upgrade or source/checkpoint replacement invalidates launcher
    resume even when the scalar Parquet inventory itself is unchanged.
    """

    config_path = artifact_root / "config.json"
    if not config_path.is_file():
        return {"available": False, "reason": "prepared config not supplied"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    unhashed = dict(config)
    expected_content_hash = unhashed.pop("config_content_sha256", None)
    canonical = (
        json.dumps(
            _json_clean(unhashed),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    actual_content_hash = hashlib.sha256(canonical).hexdigest()
    if expected_content_hash != actual_content_hash:
        raise RuntimeError("prepared config content SHA256 is invalid")

    sources = config.get("source_dataset_identity")
    checkpoints = config.get("checkpoint_identity")
    if not isinstance(sources, Mapping) or set(sources) != {"code", "wiki"}:
        raise RuntimeError("prepared config omits exact Code/Wiki source identities")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != {"before", "after"}:
        raise RuntimeError("prepared config omits exact before/after checkpoint identities")

    source_summary: dict[str, Any] = {}
    for domain in ("code", "wiki"):
        identity = sources[domain]
        if identity.get("schema") != "cka_gt_pilot_source_dataset_identity_v1":
            raise RuntimeError(f"invalid prepared source identity schema: {domain}")
        source_summary[domain] = {
            "schema": identity.get("schema"),
            "storage_kind": identity.get("storage_kind"),
            "resolved_prefix": identity.get("resolved_prefix"),
            "idx_size_bytes": identity.get("idx", {}).get("size_bytes"),
            "idx_sha256": identity.get("idx", {}).get("sha256"),
            "bin_size_bytes": identity.get("bin", {}).get("size_bytes"),
            "bin_sha256": identity.get("bin", {}).get("sha256"),
        }
        if not source_summary[domain]["idx_sha256"] or not source_summary[domain]["bin_sha256"]:
            raise RuntimeError(f"prepared source identity is incomplete: {domain}")

    checkpoint_summary: dict[str, Any] = {}
    for name in ("before", "after"):
        identity = checkpoints[name]
        if identity.get("schema") != "cka_gt_pilot_checkpoint_identity_v1":
            raise RuntimeError(f"invalid prepared checkpoint identity schema: {name}")
        checkpoint_summary[name] = {
            "schema": identity.get("schema"),
            "storage_kind": identity.get("storage_kind"),
            "resolved_root": identity.get("resolved_root"),
            "tracker_step": identity.get("tracker_step"),
            "iteration_dir": identity.get("iteration_dir"),
            "total_bytes": identity.get("total_bytes"),
            "content_sha256": identity.get("content_sha256"),
        }
        if not checkpoint_summary[name]["content_sha256"]:
            raise RuntimeError(f"prepared checkpoint identity is incomplete: {name}")

    return {
        "available": True,
        "config_relative_path": str(config_path.relative_to(artifact_root)),
        "config_file_sha256": _sha256(config_path),
        "config_content_sha256": expected_content_hash,
        "source_dataset_identity": source_summary,
        "checkpoint_identity": checkpoint_summary,
    }


def _parquet_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".parquet" else []
    return sorted(candidate for candidate in path.rglob("*.parquet") if candidate.is_file())


def _input_parquet_inventory(
    artifact_root: Path,
    token_metrics: Path,
    chunk_metrics: Path,
    sealed_token_metrics: Path | None = None,
    sealed_chunk_metrics: Path | None = None,
) -> dict[str, Any]:
    """Hash the exact scalar Parquet inputs consumed by this analysis.

    Paths in the digest are logical, relative paths rooted at ``token_metrics``
    or ``chunk_metrics``.  Absolute deployment paths are metadata only and do
    not make an otherwise identical inventory hash machine-specific.
    """

    files: list[dict[str, Any]] = []
    sources: list[tuple[str, Path, bool]] = [
        ("token_metrics", token_metrics, True),
        ("chunk_metrics", chunk_metrics, True),
    ]
    if sealed_token_metrics is not None:
        sources.append(("sealed_test_token_metrics", sealed_token_metrics, False))
    if sealed_chunk_metrics is not None:
        sources.append(("sealed_test_chunk_metrics", sealed_chunk_metrics, False))
    present_datasets: list[str] = []
    for dataset, root, required in sources:
        paths = _parquet_paths(root)
        if not paths and required:
            raise RuntimeError(f"input inventory found no Parquet files under {root}")
        if not paths:
            continue
        present_datasets.append(dataset)
        relative_base = root if root.is_dir() else root.parent
        for path in paths:
            try:
                relative = path.relative_to(relative_base).as_posix()
            except ValueError:  # defensive; rglob/file paths normally share the base
                relative = path.name
            metadata = pq.ParquetFile(path).metadata
            files.append(
                {
                    "dataset": dataset,
                    "relative_path": f"{dataset}/{relative}",
                    "bytes": int(path.stat().st_size),
                    "rows": int(metadata.num_rows),
                    "row_groups": int(metadata.num_row_groups),
                    "sha256": _sha256(path),
                }
            )
    files.sort(key=lambda row: (row["dataset"], row["relative_path"]))
    digest_payload = {"schema": INPUT_INVENTORY_SCHEMA, "files": files}
    totals = {
        dataset: {
            "files": sum(row["dataset"] == dataset for row in files),
            "bytes": sum(row["bytes"] for row in files if row["dataset"] == dataset),
            "rows": sum(row["rows"] for row in files if row["dataset"] == dataset),
        }
        for dataset in present_datasets
    }
    return {
        **digest_payload,
        "inventory_digest_sha256": _canonical_digest(digest_payload),
        "totals": totals,
        "source_roots": {
            "artifact_root": str(artifact_root),
            "token_metrics": str(token_metrics),
            "chunk_metrics": str(chunk_metrics),
            "sealed_test_token_metrics": str(sealed_token_metrics) if sealed_token_metrics is not None else None,
            "sealed_test_chunk_metrics": str(sealed_chunk_metrics) if sealed_chunk_metrics is not None else None,
        },
    }


def _raw_quantiles_bound_to_inventory(
    raw_quantiles: Path,
    inventory_digest: str,
) -> bool:
    manifest_path = raw_quantiles.with_name("raw_unit_quantile_accumulator_manifest.json")
    if not raw_quantiles.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema") == "cka_gt_raw_unit_exact_quantiles_v1"
        and manifest.get("input_inventory_digest_sha256") == inventory_digest
        and manifest.get("output_sha256") == _sha256(raw_quantiles)
    )


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def _json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _splitmix64(values: np.ndarray, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Stable vectorized 64-bit priority hash."""

    with np.errstate(over="ignore"):
        z = values.astype(np.uint64, copy=False) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _token_priorities(window_uid: np.ndarray, position: np.ndarray, seed: int) -> np.ndarray:
    mixed = window_uid.astype(np.uint64, copy=False) ^ (
        (position.astype(np.uint64, copy=False) + np.uint64(1)) * np.uint64(0xD6E8FEB86659FD93)
    )
    return _splitmix64(mixed, seed)


def _arrow_scalar_numpy(batch: pa.RecordBatch, name: str, dtype: Any | None = None) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    array = batch.column(index)
    result = array.to_numpy(zero_copy_only=False)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _arrow_list_numpy(
    batch: pa.RecordBatch,
    name: str,
    width: int,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Convert fixed/list Arrow columns to a dense [rows,width] ndarray."""

    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    array = batch.column(index)
    value_type = array.type.value_type if isinstance(array, pa.FixedSizeListArray) else None
    nested = bool(
        value_type is not None
        and (pa.types.is_list(value_type) or pa.types.is_large_list(value_type) or pa.types.is_fixed_size_list(value_type))
    )
    if isinstance(array, pa.FixedSizeListArray) and array.offset == 0 and not nested:
        values = array.values.to_numpy(zero_copy_only=False)
        result = values.reshape(len(array), width).astype(dtype, copy=False)
        if array.null_count:
            valid = array.is_valid().to_numpy(zero_copy_only=False)
            result = result.copy()
            result[~valid] = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
        return result
    values = array.to_pylist()
    fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    result = np.full((len(values), width), fill, dtype=dtype)
    for row_index, row in enumerate(values):
        if row is None:
            continue
        flat = np.asarray(row).reshape(-1)
        if flat.size != width:
            raise ValueError(f"{name} row {row_index} has {flat.size} values, expected {width}")
        result[row_index] = flat.astype(dtype, copy=False)
    return result


def _optional_list(
    batch: pa.RecordBatch,
    name: str,
    width: int,
    dtype: Any = np.float32,
) -> np.ndarray | None:
    return _arrow_list_numpy(batch, name, width, dtype) if name in batch.schema.names else None


class ParquetArtifactReader:
    def __init__(
        self,
        token_path: Path,
        chunk_path: Path | None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        token_files = _parquet_paths(token_path)
        if not token_files:
            raise FileNotFoundError(f"token metrics not found: {token_path}")
        self.token = pads.dataset(
            [str(path) for path in token_files],
            format="parquet",
            exclude_invalid_files=True,
        )
        chunk_files = _parquet_paths(chunk_path) if chunk_path is not None else []
        self.chunk = (
            pads.dataset(
                [str(path) for path in chunk_files],
                format="parquet",
                exclude_invalid_files=True,
            )
            if chunk_files
            else None
        )
        self.batch_size = int(batch_size)
        missing = sorted(set(REQUIRED_TOKEN_COLUMNS) - set(self.token.schema.names))
        if missing:
            raise RuntimeError(f"token metric schema is missing required columns: {missing}")

    @property
    def token_columns(self) -> set[str]:
        return set(self.token.schema.names)

    def token_batches(
        self,
        domain: str,
        split: str,
        columns: Sequence[str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        if columns is None:
            columns = [*REQUIRED_TOKEN_COLUMNS, *OPTIONAL_TOKEN_COLUMNS]
        selected = [name for name in dict.fromkeys(columns) if name in self.token.schema.names]
        expression = (pads.field("domain") == domain) & (pads.field("split") == split)
        batches = self.token.scanner(
            columns=selected,
            filter=expression,
            batch_size=self.batch_size,
            use_threads=True,
        ).to_batches()
        for batch in batches:
            if len(batch):
                yield batch

    def chunk_batches(
        self,
        domain: str,
        split: str,
        columns: Sequence[str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        if self.chunk is None:
            return
        if columns is None:
            columns = list(self.chunk.schema.names)
        selected = [name for name in columns if name in self.chunk.schema.names]
        expression = (pads.field("domain") == domain) & (pads.field("split") == split)
        batches = self.chunk.scanner(
            columns=selected,
            filter=expression,
            batch_size=self.batch_size,
            use_threads=True,
        ).to_batches()
        for batch in batches:
            if len(batch):
                yield batch


@dataclass(frozen=True)
class CandidateBundle:
    level: int
    b: Mapping[int, np.ndarray]
    t: Mapping[int, np.ndarray]
    rel_l2: np.ndarray
    abs_log_r: np.ndarray

    def serializable(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "B_lower_threshold": {str(scale): self.b[scale].tolist() for scale in SCALES},
            "T_lower_threshold": {str(scale): self.t[scale].tolist() for scale in SCALES},
            "relative_l2_upper_threshold": self.rel_l2.tolist(),
            "abs_log_r_upper_threshold": self.abs_log_r.tolist(),
        }


def _normalize_quantile(value: Any) -> float:
    result = float(value)
    return result / 100.0 if result > 1.0 else result


def _metric_matches(value: str, candidates: Sequence[str]) -> bool:
    normalized = value.strip().lower().replace("-", "_")
    return normalized in {candidate.lower().replace("-", "_") for candidate in candidates}


class RawQuantileTable:
    """Strict accessor for exact, pre-aggregation calibration quantiles."""

    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"raw-unit quantile table not found: {path}. Do not substitute token-min s_i quantiles."
            )
        table = pq.read_table(path)
        self.rows = table.to_pylist()
        required = {"metric", "layer", "quantile", "value", "count", "method"}
        missing = required - set(table.schema.names)
        if missing:
            raise RuntimeError(f"raw quantile table missing columns: {sorted(missing)}")
        bad = sorted({str(row.get("method")) for row in self.rows if row.get("method") != "exact_disk_backed"})
        if bad:
            raise RuntimeError(f"non-exact raw-unit quantiles are forbidden in v1: methods={bad}")

    def get(
        self,
        metrics: Sequence[str],
        quantile: float,
        layer: int,
        *,
        domain: str | None = None,
        split: str | None = None,
        scale: int | None = None,
    ) -> float:
        candidates: list[dict[str, Any]] = []
        for row in self.rows:
            if not _metric_matches(str(row.get("metric", "")), metrics):
                continue
            if int(row["layer"]) != int(layer):
                continue
            if not math.isclose(_normalize_quantile(row["quantile"]), quantile, abs_tol=1e-9):
                continue
            if domain is not None and row.get("domain") not in (None, domain):
                continue
            if split is not None and row.get("split") not in (None, split):
                continue
            row_scale = row.get("scale")
            if scale is not None and row_scale is not None and int(row_scale) != int(scale):
                continue
            if scale is None and row_scale not in (None, 0, "", float("nan")):
                try:
                    if math.isfinite(float(row_scale)):
                        continue
                except (TypeError, ValueError):
                    continue
            candidates.append(row)
        if len(candidates) != 1:
            raise RuntimeError(
                "expected exactly one raw-unit quantile row for "
                f"metrics={metrics}, q={quantile}, layer={layer}, domain={domain}, "
                f"split={split}, scale={scale}; found {len(candidates)}"
            )
        value = float(candidates[0]["value"])
        if not math.isfinite(value) or int(candidates[0]["count"]) <= 0:
            raise RuntimeError(f"invalid quantile row: {candidates[0]}")
        return value

    def build_bundles(self) -> dict[int, CandidateBundle]:
        result: dict[int, CandidateBundle] = {}
        for level in BUNDLES:
            lower_q = 1.0 - level / 100.0
            upper_q = level / 100.0
            b = {
                scale: np.asarray(
                    [
                        self.get(
                            ("cka", "chunk_cka", "b_cka"),
                            lower_q,
                            layer,
                            domain="wiki",
                            split="calibration",
                            scale=scale,
                        )
                        for layer in LAYERS
                    ],
                    dtype=np.float32,
                )
                for scale in SCALES
            }
            t = {
                scale: np.asarray(
                    [
                        self.get(
                            ("s_i", "s", "token_contribution_share"),
                            lower_q,
                            layer,
                            domain="wiki",
                            split="calibration",
                            scale=scale,
                        )
                        for layer in LAYERS
                    ],
                    dtype=np.float32,
                )
                for scale in SCALES
            }
            rel_l2 = np.asarray(
                [
                    self.get(
                        ("relative_l2", "rel_l2"),
                        upper_q,
                        layer,
                        domain="wiki",
                        split="calibration",
                    )
                    for layer in LAYERS
                ],
                dtype=np.float32,
            )
            abs_log_r = np.asarray(
                [
                    self.get(
                        ("abs_log_r", "absolute_log_r"),
                        upper_q,
                        layer,
                        domain="wiki",
                        split="calibration",
                    )
                    for layer in LAYERS
                ],
                dtype=np.float32,
            )
            result[level] = CandidateBundle(level, b, t, rel_l2, abs_log_r)
        return result

    def legacy_cosine(self, split: str) -> np.ndarray:
        metric_names = ("code_cosine",) if not any("domain" in row for row in self.rows) else ("cosine", "code_cosine")
        return np.asarray(
            [
                self.get(
                    metric_names,
                    0.99,
                    layer,
                    domain="code",
                    split=split,
                )
                for layer in LAYERS
            ],
            dtype=np.float32,
        )


class _ExactDiskGroup:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("wb")
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            finite.tofile(self.handle)
            self.count += int(finite.size)

    def exact_quantiles(self, quantiles: Sequence[float]) -> dict[float, float]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.count <= 0:
            raise RuntimeError(f"empty exact quantile group: {self.path.name}")
        values = np.memmap(self.path, mode="r+", dtype=np.float32, shape=(self.count,))
        positions = {float(q): (self.count - 1) * float(q) for q in quantiles}
        indices = sorted(
            {
                index
                for position in positions.values()
                for index in (int(math.floor(position)), int(math.ceil(position)))
            }
        )
        values.partition(indices)
        result: dict[float, float] = {}
        for quantile, position in positions.items():
            lower = int(math.floor(position))
            upper = int(math.ceil(position))
            fraction = position - lower
            result[quantile] = float(
                float(values[lower]) * (1.0 - fraction) + float(values[upper]) * fraction
            )
        values.flush()
        del values
        return result


def _flatten_arrow_list(batch: pa.RecordBatch, name: str) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    array = batch.column(index)
    if isinstance(array, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        flattened = array.flatten()
        return flattened.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    raise TypeError(f"{name} must be an Arrow list column, got {array.type}")


def _atomic_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    pq.write_table(table, temporary, compression="zstd")
    checked = pq.read_table(temporary)
    if checked.num_rows != table.num_rows or not checked.schema.equals(table.schema):
        raise RuntimeError(f"Parquet round-trip validation failed: {path}")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_raw_unit_quantiles_exact(
    reader: ParquetArtifactReader,
    output_path: Path,
    *,
    work_root: Path,
    input_inventory_digest: str,
) -> Path:
    """Build all v1 calibration candidates from their exact raw units.

    In particular, T thresholds flatten raw per-chunk ``s_i`` list values
    from chunk metrics. They never use ``s_min_128/256`` token aggregates.
    Each group is appended to its own float32 file and exact NumPy-linear
    quantiles are computed by in-place memmap partitioning.
    """

    if reader.chunk is None:
        raise RuntimeError("cannot build raw-unit quantiles without chunk_metrics")
    chunk_names = set(reader.chunk.schema.names)
    s_column = next((name for name in ("s_i", "raw_s_i", "token_s_i") if name in chunk_names), None)
    if s_column is None:
        raise RuntimeError(
            "chunk_metrics must store the raw per-chunk s_i list to build exact T thresholds; "
            "token-level s_min is not a valid substitute"
        )
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str, str, int | None, int], _ExactDiskGroup] = {}

    def group(key: tuple[str, str, str, int | None, int]) -> _ExactDiskGroup:
        if key not in groups:
            digest = hashlib.sha256(repr(key).encode()).hexdigest()[:20]
            groups[key] = _ExactDiskGroup(work_root / f"{digest}.float32.bin")
        return groups[key]

    # Token raw units: Wiki magnitude calibration and both Code cosine axes.
    for domain, split in (
        ("wiki", "calibration"),
        ("code", "calibration"),
        ("code", "selection"),
    ):
        columns = ["domain", "split"]
        if domain == "wiki":
            columns += ["relative_l2", "log_r"]
        else:
            columns += ["cosine"]
        for batch in reader.token_batches(domain, split, columns):
            if domain == "wiki":
                rel = _arrow_list_numpy(batch, "relative_l2", 8)
                abs_log = np.abs(_arrow_list_numpy(batch, "log_r", 8))
                for layer_index, layer in enumerate(LAYERS):
                    group((domain, split, "relative_l2", None, layer)).add(rel[:, layer_index])
                    group((domain, split, "abs_log_r", None, layer)).add(abs_log[:, layer_index])
            else:
                cosine = _arrow_list_numpy(batch, "cosine", 8)
                for layer_index, layer in enumerate(LAYERS):
                    group((domain, split, "cosine", None, layer)).add(cosine[:, layer_index])

    # Chunk raw units: Wiki actual chunk CKA and every raw token contribution share.
    chunk_columns = ["domain", "split", "scale", "layer", "cka", s_column]
    for batch in reader.chunk_batches("wiki", "calibration", chunk_columns):
        scale_values = _arrow_scalar_numpy(batch, "scale", np.int16)
        layer_values = _arrow_scalar_numpy(batch, "layer", np.int16)
        cka_values = _arrow_scalar_numpy(batch, "cka", np.float32)
        # Flatten each row separately because chunk lengths differ between scales/tails.
        s_rows = batch.column(batch.schema.get_field_index(s_column)).to_pylist()
        for row_index in range(len(batch)):
            scale = int(scale_values[row_index])
            layer = int(layer_values[row_index])
            if scale not in SCALES or layer not in LAYERS:
                continue
            group(("wiki", "calibration", "cka", scale, layer)).add(
                np.asarray([cka_values[row_index]], dtype=np.float32)
            )
            row_s = s_rows[row_index]
            if row_s is not None:
                group(("wiki", "calibration", "s_i", scale, layer)).add(
                    np.asarray(row_s, dtype=np.float32)
                )

    requested: dict[str, tuple[float, ...]] = {
        "cka": (0.01, 0.03, 0.05),
        "s_i": (0.01, 0.03, 0.05),
        "relative_l2": (0.95, 0.97, 0.99),
        "abs_log_r": (0.95, 0.97, 0.99),
        "cosine": (0.99,),
    }
    rows: list[dict[str, Any]] = []
    manifest_groups: list[dict[str, Any]] = []
    for key in sorted(groups, key=repr):
        domain, split, metric, scale, layer = key
        accumulator = groups[key]
        quantile_values = accumulator.exact_quantiles(requested[metric])
        accumulator_hash = _sha256(accumulator.path)
        manifest_groups.append(
            {
                "domain": domain,
                "split": split,
                "metric": metric,
                "scale": scale,
                "layer": layer,
                "count": accumulator.count,
                "accumulator_bytes": accumulator.path.stat().st_size,
                "accumulator_sha256_after_partition": accumulator_hash,
                "quantiles": list(requested[metric]),
            }
        )
        for quantile in requested[metric]:
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "metric": metric,
                    "scale": scale,
                    "layer": layer,
                    "quantile": quantile,
                    "value": quantile_values[quantile],
                    "count": accumulator.count,
                    "method": "exact_disk_backed",
                }
            )

    # Fail closed if any required group was absent.
    expected_groups = {
        *{
            ("wiki", "calibration", metric, scale, layer)
            for metric in ("cka", "s_i")
            for scale in SCALES
            for layer in LAYERS
        },
        *{
            ("wiki", "calibration", metric, None, layer)
            for metric in ("relative_l2", "abs_log_r")
            for layer in LAYERS
        },
        *{
            ("code", split, "cosine", None, layer)
            for split in ("calibration", "selection")
            for layer in LAYERS
        },
    }
    missing = sorted(expected_groups - set(groups), key=repr)
    if missing:
        raise RuntimeError(f"raw exact quantile stream is missing required groups: {missing}")
    table = pa.Table.from_pylist(rows)
    _atomic_parquet(output_path, table)
    manifest_path = output_path.with_name("raw_unit_quantile_accumulator_manifest.json")
    _atomic_json(
        manifest_path,
        {
            "schema": "cka_gt_raw_unit_exact_quantiles_v1",
            "output": output_path.name,
            "output_sha256": _sha256(output_path),
            "input_inventory_schema": INPUT_INVENTORY_SCHEMA,
            "input_inventory_digest_sha256": input_inventory_digest,
            "raw_s_source_column": s_column,
            "raw_s_used_token_min": False,
            "groups": manifest_groups,
            "temporary_accumulators_retained": False,
        },
    )
    shutil.rmtree(work_root)
    return output_path


@dataclass
class ConsensusResult:
    eligible: np.ndarray
    passed: np.ndarray
    n_valid: np.ndarray
    pass_count: np.ndarray


def condition_consensus(
    values: np.ndarray,
    thresholds: np.ndarray,
    lower_is_better: bool,
    *,
    forced_fail: np.ndarray | None = None,
) -> ConsensusResult:
    if values.ndim != 2 or values.shape[1] != len(LAYERS):
        raise ValueError(f"expected [tokens,8], got {values.shape}")
    valid = np.isfinite(values)
    n_valid = valid.sum(axis=1, dtype=np.int16)
    comparison = values <= thresholds[None, :] if lower_is_better else values >= thresholds[None, :]
    comparison &= valid
    if forced_fail is not None:
        if forced_fail.shape != values.shape:
            raise ValueError(f"forced_fail shape {forced_fail.shape} does not match values {values.shape}")
        comparison &= ~forced_fail
    pass_count = comparison.sum(axis=1, dtype=np.int16)
    eligible = n_valid >= 6
    passed = eligible & ((pass_count >= 7) | ((n_valid == 6) & (pass_count == 6)))
    return ConsensusResult(eligible, passed, n_valid, pass_count)


@dataclass
class BatchSelectors:
    by_level: dict[int, dict[str, np.ndarray]]
    legacy_primary: np.ndarray
    legacy_reproduction: np.ndarray


def evaluate_batch(
    batch: pa.RecordBatch,
    bundles: Mapping[int, CandidateBundle],
    legacy_primary_thresholds: np.ndarray,
    legacy_reproduction_thresholds: np.ndarray,
) -> BatchSelectors:
    cosine = _arrow_list_numpy(batch, "cosine", 8)
    rel_l2 = _arrow_list_numpy(batch, "relative_l2", 8)
    abs_log_r = np.abs(_arrow_list_numpy(batch, "log_r", 8))
    b_values = {scale: _arrow_list_numpy(batch, f"cka_min_{scale}", 8) for scale in SCALES}
    t_values = {scale: _arrow_list_numpy(batch, f"s_min_{scale}", 8) for scale in SCALES}
    for scale in SCALES:
        valid_b = _optional_list(batch, f"valid_cka_{scale}", 8, np.bool_)
        valid_t = _optional_list(batch, f"valid_s_{scale}", 8, np.bool_)
        if valid_b is not None:
            b_values[scale] = np.where(valid_b, b_values[scale], np.nan)
        if valid_t is not None:
            t_values[scale] = np.where(valid_t, t_values[scale], np.nan)

    by_level: dict[int, dict[str, np.ndarray]] = {}
    for level, bundle in bundles.items():
        b_consensus = {
            scale: condition_consensus(b_values[scale], bundle.b[scale], lower_is_better=False)
            for scale in SCALES
        }
        t_consensus = {
            scale: condition_consensus(
                t_values[scale],
                bundle.t[scale],
                lower_is_better=False,
                forced_fail=t_values[scale] < 0,
            )
            for scale in SCALES
        }
        l2 = condition_consensus(rel_l2, bundle.rel_l2, lower_is_better=True)
        logr = condition_consensus(abs_log_r, bundle.abs_log_r, lower_is_better=True)
        b_eligible = b_consensus[128].eligible & b_consensus[256].eligible
        b_pass = b_eligible & b_consensus[128].passed & b_consensus[256].passed
        t_eligible = t_consensus[128].eligible & t_consensus[256].eligible
        t_pass = t_eligible & t_consensus[128].passed & t_consensus[256].passed
        cka_eligible = b_eligible & t_eligible
        full_eligible = cka_eligible & l2.eligible & logr.eligible
        cka_only = cka_eligible & b_pass & t_pass
        full = full_eligible & b_pass & t_pass & l2.passed & logr.passed

        same_valid = np.ones((len(batch), 8), dtype=bool)
        same_pass = np.ones((len(batch), 8), dtype=bool)
        for scale in SCALES:
            vb = np.isfinite(b_values[scale])
            vt = np.isfinite(t_values[scale])
            same_valid &= vb & vt
            same_pass &= vb & vt & (b_values[scale] >= bundle.b[scale][None, :])
            same_pass &= (t_values[scale] >= bundle.t[scale][None, :]) & (t_values[scale] >= 0)
        vm_l2 = np.isfinite(rel_l2)
        vm_r = np.isfinite(abs_log_r)
        same_valid &= vm_l2 & vm_r
        same_pass &= vm_l2 & vm_r & (rel_l2 <= bundle.rel_l2[None, :])
        same_pass &= abs_log_r <= bundle.abs_log_r[None, :]
        same_n_valid = same_valid.sum(axis=1, dtype=np.int16)
        same_count = (same_pass & same_valid).sum(axis=1, dtype=np.int16)
        same_eligible = same_n_valid >= 6
        same_selected = same_eligible & ((same_count >= 7) | ((same_n_valid == 6) & (same_count == 6)))

        by_level[level] = {
            "B_eligible": b_eligible,
            "B": b_pass,
            "T_eligible": t_eligible,
            "T": t_pass,
            "L2_eligible": l2.eligible,
            "L2": l2.passed,
            "R_eligible": logr.eligible,
            "R": logr.passed,
            "cka_eligible": cka_eligible,
            "full_eligible": full_eligible,
            "cka_only": cka_only,
            "full": full,
            "same_eligible": same_eligible,
            "same_selected": same_selected,
        }

    legacy_valid = np.isfinite(cosine).all(axis=1)
    legacy_primary = legacy_valid & (cosine >= legacy_primary_thresholds[None, :]).all(axis=1)
    legacy_reproduction = legacy_valid & (cosine >= legacy_reproduction_thresholds[None, :]).all(axis=1)
    return BatchSelectors(by_level, legacy_primary, legacy_reproduction)


@dataclass
class PriorityReservoir:
    capacity: int
    seed: int
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    priorities: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))
    seen: int = 0

    def add(self, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return
        indices = np.arange(self.seen, self.seen + finite.size, dtype=np.uint64)
        self.seen += finite.size
        priorities = _splitmix64(indices, self.seed)
        combined_v = np.concatenate((self.values, finite))
        combined_p = np.concatenate((self.priorities, priorities))
        if combined_v.size > self.capacity:
            keep = np.argpartition(combined_p, self.capacity - 1)[: self.capacity]
            combined_v = combined_v[keep]
            combined_p = combined_p[keep]
        self.values = combined_v
        self.priorities = combined_p


class HistogramBank:
    """Two-pass, full-count histograms with reservoir-derived common ranges."""

    def __init__(self, bins: int, seed: int):
        self.bins = max(200, int(bins))
        self.seed = int(seed)
        self.reservoirs: dict[tuple[Any, ...], PriorityReservoir] = {}
        self.edges: dict[tuple[Any, ...], np.ndarray] = {}
        self.counts: dict[tuple[tuple[Any, ...], str], np.ndarray] = {}
        self.under_over: dict[tuple[tuple[Any, ...], str], tuple[int, int, int]] = {}

    def observe_range(self, key: tuple[Any, ...], values: np.ndarray) -> None:
        if key not in self.reservoirs:
            digest = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
            self.reservoirs[key] = PriorityReservoir(20_000, self.seed ^ digest)
        self.reservoirs[key].add(values)

    def finalize_ranges(self) -> None:
        for key, reservoir in self.reservoirs.items():
            values = reservoir.values
            if values.size == 0:
                lower, upper = 0.0, 1.0
            elif key[0] == "chunk_cka":
                lower, upper = 0.0, 1.0
            else:
                lower, upper = np.quantile(values, [0.001, 0.999]).tolist()
                if key[0] in {"relative_l2", "abs_log_r", "maha", "proto"}:
                    lower = 0.0
                if not math.isfinite(lower) or not math.isfinite(upper) or lower == upper:
                    center = float(np.nanmean(values)) if values.size else 0.5
                    width = max(abs(center) * 0.1, 1e-6)
                    lower, upper = center - width, center + width
            self.edges[key] = np.linspace(float(lower), float(upper), self.bins + 1, dtype=np.float64)

    def add_counts(self, key: tuple[Any, ...], series: str, values: np.ndarray) -> None:
        edges = self.edges[key]
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        under = int(np.count_nonzero(finite < edges[0]))
        over = int(np.count_nonzero(finite > edges[-1]))
        if finite.size:
            finite = np.clip(finite, edges[0], np.nextafter(edges[-1], edges[0]))
            counts = np.histogram(finite, bins=edges)[0].astype(np.uint64)
        else:
            counts = np.zeros(self.bins, dtype=np.uint64)
        count_key = (key, series)
        if count_key not in self.counts:
            self.counts[count_key] = counts
            self.under_over[count_key] = (under, over, int(finite.size))
        else:
            self.counts[count_key] += counts
            old = self.under_over[count_key]
            self.under_over[count_key] = (old[0] + under, old[1] + over, old[2] + int(finite.size))


def _svg_histogram(
    path: Path,
    title: str,
    xlabel: str,
    edges: np.ndarray,
    series_counts: Mapping[str, np.ndarray],
) -> None:
    """Write a dependency-free SVG with linear and log-y panels."""

    width, height = 1200, 760
    left, right = 90, 30
    plot_width = width - left - right
    panel_height = 255
    panel_tops = (90, 430)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
    fractions: dict[str, np.ndarray] = {}
    for label, counts in series_counts.items():
        total = counts.sum(dtype=np.float64)
        fractions[label] = counts.astype(np.float64) / total if total else np.zeros_like(counts, dtype=np.float64)
    max_linear = max((float(values.max(initial=0.0)) for values in fractions.values()), default=1.0)
    max_linear = max(max_linear, 1e-12)
    positive = np.concatenate([values[values > 0] for values in fractions.values()]) if fractions else np.empty(0)
    log_floor = max(float(positive.min(initial=1e-8)) if positive.size else 1e-8, 1e-9)
    log_top = max(max_linear, log_floor * 10)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="38" text-anchor="middle" font-family="sans-serif" font-size="24">{html.escape(title)}</text>',
    ]
    centers = 0.5 * (edges[:-1] + edges[1:])
    xmin, xmax = float(edges[0]), float(edges[-1])
    for panel_index, top in enumerate(panel_tops):
        bottom = top + panel_height
        parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="none" stroke="#333"/>')
        for tick in range(6):
            x = left + plot_width * tick / 5
            value = xmin + (xmax - xmin) * tick / 5
            parts.append(f'<line x1="{x:.1f}" y1="{bottom}" x2="{x:.1f}" y2="{bottom+6}" stroke="#333"/>')
            parts.append(f'<text x="{x:.1f}" y="{bottom+23}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.4g}</text>')
        for tick in range(5):
            frac = tick / 4
            y = bottom - panel_height * frac
            if panel_index == 0:
                value = max_linear * frac
                label = f"{value:.3g}"
            else:
                log_value = math.log10(log_floor) + (math.log10(log_top) - math.log10(log_floor)) * frac
                label = f"1e{log_value:.1f}"
            parts.append(f'<line x1="{left-6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>')
        for series_index, (label, values) in enumerate(fractions.items()):
            points: list[str] = []
            for xvalue, value in zip(centers, values):
                x = left + (xvalue - xmin) / max(xmax - xmin, EPS) * plot_width
                if panel_index == 0:
                    y = bottom - value / max_linear * panel_height
                else:
                    clipped = max(value, log_floor)
                    yfrac = (math.log10(clipped) - math.log10(log_floor)) / (
                        math.log10(log_top) - math.log10(log_floor)
                    )
                    y = bottom - yfrac * panel_height
                points.append(f"{x:.2f},{y:.2f}")
            parts.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[series_index % len(colors)]}" stroke-width="1.6"/>'
            )
        ylabel = "fraction / bin" if panel_index == 0 else "fraction / bin (log10)"
        parts.append(f'<text x="20" y="{top+panel_height/2}" transform="rotate(-90 20 {top+panel_height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">{ylabel}</text>')
    parts.append(f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(xlabel)}</text>')
    legend_x, legend_y = 760, 55
    for index, label in enumerate(fractions):
        x = legend_x + (index % 3) * 145
        y = legend_y + (index // 3) * 20
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+24}" y2="{y}" stroke="{colors[index % len(colors)]}" stroke-width="3"/>')
        parts.append(f'<text x="{x+30}" y="{y+4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    parts.append("</svg>\n")
    _atomic_text(path, "\n".join(parts))


def _write_histogram_bank(bank: HistogramBank, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: list[dict[str, Any]] = []
    for key, edges in sorted(bank.edges.items(), key=lambda item: repr(item[0])):
        series = {
            series_name: counts
            for (count_key, series_name), counts in bank.counts.items()
            if count_key == key
        }
        if not series:
            continue
        metric = str(key[0])
        suffix = "_".join(str(value) for value in key[1:])
        stem = metric + ("_" + suffix if suffix else "")
        csv_rows = []
        for index in range(len(edges) - 1):
            row: dict[str, Any] = {"bin_left": edges[index], "bin_right": edges[index + 1]}
            for label, counts in series.items():
                total = counts.sum(dtype=np.float64)
                row[label] = float(counts[index] / total) if total else 0.0
                row[label + "_count"] = int(counts[index])
            csv_rows.append(row)
        _write_csv(output_dir / f"{stem}.csv", csv_rows)
        _svg_histogram(output_dir / f"{stem}.svg", stem.replace("_", " "), metric, edges, series)
        entry = {
            "key": list(key),
            "range": [float(edges[0]), float(edges[-1])],
            "bins": len(edges) - 1,
            "series": {},
        }
        for label in series:
            under, over, count = bank.under_over[(key, label)]
            entry["series"][label] = {"count": count, "underflow": under, "overflow": over}
        metadata.append(entry)
    _atomic_json(output_dir / "histogram_metadata.json", metadata)
    return metadata


def _histogram_values(batch: pa.RecordBatch) -> Iterator[tuple[tuple[Any, ...], np.ndarray]]:
    for scale in SCALES:
        values = _arrow_list_numpy(batch, f"s_min_{scale}", 8)
        for index, layer in enumerate(LAYERS):
            yield ("s_i_min", scale, layer), values[:, index]
    for name, source in (("relative_l2", "relative_l2"), ("abs_log_r", "log_r"), ("maha", "maha"), ("proto", "proto")):
        if source not in batch.schema.names:
            continue
        values = _arrow_list_numpy(batch, source, 8)
        if name == "abs_log_r":
            values = np.abs(values)
        for index, layer in enumerate(LAYERS):
            yield (name, layer), values[:, index]


@dataclass
class CountSummary:
    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    intersections: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, flags: Mapping[str, np.ndarray]) -> None:
        size = len(next(iter(flags.values()))) if flags else 0
        self.total += size
        names = list(flags)
        for name, values in flags.items():
            self.counts[name] = self.counts.get(name, 0) + int(np.count_nonzero(values))
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                key = (first, second)
                self.intersections[key] = self.intersections.get(key, 0) + int(
                    np.count_nonzero(flags[first] & flags[second])
                )


class SmallestPrioritySamples:
    def __init__(self, limit: int, seed: int):
        self.limit = int(limit)
        self.seed = int(seed)
        self.heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
        self.counter = 0

    def add(self, selector: str, priorities: np.ndarray, rows: Sequence[dict[str, Any]]) -> None:
        heap = self.heaps.setdefault(selector, [])
        for priority, row in zip(priorities.tolist(), rows):
            self.counter += 1
            item = (-int(priority), -self.counter, row)
            if len(heap) < self.limit:
                heapq.heappush(heap, item)
            elif -item[0] < -heap[0][0]:
                heapq.heapreplace(heap, item)

    def rows(self, selector: str) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self.heaps.get(selector, []), key=lambda item: -item[0])]


class RandomEligibleCollector:
    """Disk-backed priorities used to make the matched random arm exact."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "eligible_priorities.uint64.bin"
        self.count = 0
        self._handle = self.path.open("wb")

    def add(self, priorities: np.ndarray) -> None:
        array = np.asarray(priorities, dtype=np.uint64)
        array.tofile(self._handle)
        self.count += int(array.size)

    def cutoff(self, count: int) -> tuple[int, int]:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        if count < 0 or count > self.count:
            raise ValueError(f"cannot select {count} from eligible pool {self.count}")
        if count == 0:
            return 0, 0
        values = np.memmap(self.path, mode="r+", dtype=np.uint64, shape=(self.count,))
        values.partition(count - 1)
        cutoff = int(values[count - 1])
        less = int(np.count_nonzero(values < np.uint64(cutoff)))
        del values
        return cutoff, count - less


def _random_mask(priorities: np.ndarray, cutoff: int, tie_allowance: list[int]) -> np.ndarray:
    selected = priorities < np.uint64(cutoff)
    equal = priorities == np.uint64(cutoff)
    need = tie_allowance[0]
    if need > 0 and np.any(equal):
        indices = np.flatnonzero(equal)[:need]
        selected[indices] = True
        tie_allowance[0] -= len(indices)
    return selected


def _available_token_columns(reader: ParquetArtifactReader) -> list[str]:
    return [
        name
        for name in [*REQUIRED_TOKEN_COLUMNS, *OPTIONAL_TOKEN_COLUMNS]
        if name in reader.token_columns
    ]


def _first_pass(
    reader: ParquetArtifactReader,
    bundles: Mapping[int, CandidateBundle],
    legacy_primary: np.ndarray,
    legacy_repro: np.ndarray,
    histogram_bank: HistogramBank,
    random_collector: RandomEligibleCollector,
    seed: int,
) -> tuple[dict[str, dict[int, CountSummary]], int]:
    summaries: dict[str, dict[int, CountSummary]] = {domain: {level: CountSummary() for level in BUNDLES} for domain in ("code", "wiki")}
    cka_m_count = 0
    columns = _available_token_columns(reader)
    for domain in ("code", "wiki"):
        for batch in reader.token_batches(domain, "selection", columns):
            evaluated = evaluate_batch(batch, bundles, legacy_primary, legacy_repro)
            for key, values in _histogram_values(batch):
                histogram_bank.observe_range(key, values)
            for level in BUNDLES:
                fields = evaluated.by_level[level]
                flags = {
                    "B_eligible": fields["B_eligible"],
                    "B": fields["B"],
                    "T_eligible": fields["T_eligible"],
                    "T": fields["T"],
                    "L2_eligible": fields["L2_eligible"],
                    "L2": fields["L2"],
                    "R_eligible": fields["R_eligible"],
                    "R": fields["R"],
                    "full_eligible": fields["full_eligible"],
                    "cka_only": fields["cka_only"],
                    "full": fields["full"],
                    "same_eligible": fields["same_eligible"],
                    "same_selected": fields["same_selected"],
                }
                summaries[domain][level].add(flags)
            if domain == "code":
                full_95 = evaluated.by_level[95]
                cka_m_count += int(np.count_nonzero(full_95["full"]))
                window_uid = _arrow_scalar_numpy(batch, "window_uid", np.int64)
                position = _arrow_scalar_numpy(batch, "position", np.int64)
                priorities = _token_priorities(window_uid, position, seed)
                random_collector.add(priorities[full_95["full_eligible"]])
    histogram_bank.finalize_ranges()
    return summaries, cka_m_count


def _selector_names() -> tuple[str, ...]:
    return ("legacy_cosine", "cka_only95", "cka_plus_m95", "matched_random")


def _routing_batch(batch: pa.RecordBatch) -> dict[str, np.ndarray] | None:
    needed = {"before_top4_ids", "after_top4_ids", "before_old_full_mass", "after_old_full_mass"}
    if not needed.issubset(batch.schema.names):
        return None
    before = _arrow_list_numpy(batch, "before_top4_ids", 32, np.int32).reshape(-1, 8, 4)
    after = _arrow_list_numpy(batch, "after_top4_ids", 32, np.int32).reshape(-1, 8, 4)
    overlap = np.zeros((len(batch), 8), dtype=np.uint8)
    for slot in range(4):
        overlap += (before[:, :, slot, None] == after).any(axis=-1)
    before_full = _arrow_list_numpy(batch, "before_old_full_mass", 8)
    after_full = _arrow_list_numpy(batch, "after_old_full_mass", 8)
    result = {"top4_overlap": overlap, "old_full_mass_delta": after_full - before_full}
    if {"before_old_selected_mass", "after_old_selected_mass"}.issubset(batch.schema.names):
        before_selected = _arrow_list_numpy(batch, "before_old_selected_mass", 8)
        after_selected = _arrow_list_numpy(batch, "after_old_selected_mass", 8)
        result["old_selected_mass_delta"] = after_selected - before_selected
    return result


@dataclass
class RoutingSummary:
    overlap_hist: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.uint64))
    full_sum: float = 0.0
    full_count: int = 0
    selected_sum: float = 0.0
    selected_count: int = 0

    def add(self, routing: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
        overlaps = routing["top4_overlap"][mask].reshape(-1)
        self.overlap_hist += np.bincount(overlaps, minlength=5).astype(np.uint64)
        full = routing["old_full_mass_delta"][mask]
        finite = np.isfinite(full)
        self.full_sum += float(full[finite].sum(dtype=np.float64))
        self.full_count += int(np.count_nonzero(finite))
        if "old_selected_mass_delta" in routing:
            selected = routing["old_selected_mass_delta"][mask]
            finite_selected = np.isfinite(selected)
            self.selected_sum += float(selected[finite_selected].sum(dtype=np.float64))
            self.selected_count += int(np.count_nonzero(finite_selected))

    def serialize(self) -> dict[str, Any]:
        return {
            "top4_overlap_hist": self.overlap_hist.tolist(),
            "old_full_mass_delta_mean": _json_number(_ratio(self.full_sum, self.full_count)),
            "old_selected_mass_delta_mean": _json_number(_ratio(self.selected_sum, self.selected_count)),
            "layer_occurrences": int(self.overlap_hist.sum()),
        }


def _context_row(batch: pa.RecordBatch, index: int, evaluated: BatchSelectors) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for name in IDENTITY_COLUMNS:
        if name in batch.schema.names:
            value = batch.column(batch.schema.get_field_index(name))[index].as_py()
            row[name] = value
    for name in (
        "cosine",
        "relative_l2",
        "log_r",
        "maha",
        "proto",
        "maha_mean",
        "proto_mean",
        "cka_min_128",
        "cka_min_256",
        "s_min_128",
        "s_min_256",
    ):
        if name in batch.schema.names:
            row[name] = batch.column(batch.schema.get_field_index(name))[index].as_py()
    for name in (
        "r_min_128",
        "r_mean_128",
        "r_max_128",
        "r_min_256",
        "r_mean_256",
        "r_max_256",
        "worst_chunk_id_128",
        "worst_chunk_id_256",
        "s_at_worst_cka_128",
        "s_at_worst_cka_256",
        "worst_s_chunk_id_128",
        "worst_s_chunk_id_256",
        "worst_diag_ratio_128",
        "worst_diag_ratio_256",
        "before_old_full_mass",
        "after_old_full_mass",
        "before_old_selected_mass",
        "after_old_selected_mass",
        "before_top4_ids",
        "after_top4_ids",
    ):
        if name in batch.schema.names:
            row[name] = batch.column(batch.schema.get_field_index(name))[index].as_py()
    if "context_token_ids" in batch.schema.names:
        row["context_token_ids"] = batch.column(batch.schema.get_field_index("context_token_ids"))[index].as_py()
    if "context_text" in batch.schema.names:
        row["context_text"] = batch.column(batch.schema.get_field_index("context_text"))[index].as_py()
    row["context_available"] = "context_token_ids" in row or "context_text" in row
    return row


def _second_pass(
    reader: ParquetArtifactReader,
    output_root: Path,
    bundles: Mapping[int, CandidateBundle],
    legacy_primary: np.ndarray,
    legacy_repro: np.ndarray,
    histogram_bank: HistogramBank,
    cutoff: int,
    ties: int,
    seed: int,
) -> tuple[dict[str, CountSummary], dict[str, Any], dict[str, Any]]:
    selector_dir = output_root / "selector_comparison"
    context_dir = output_root / "context_samples"
    selector_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    summaries = {domain: CountSummary() for domain in ("code", "wiki")}
    routing = {
        "selected": RoutingSummary(),
        "nonselected": RoutingSummary(),
    }
    profile_values: dict[str, dict[str, list[np.ndarray]]] = {
        "legacy_only": {"relative_l2": [], "maha": [], "diag_ratio": []},
        "cka_plus_m_only": {"relative_l2": [], "maha": [], "diag_ratio": []},
    }
    samples = SmallestPrioritySamples(100, seed)
    writer: pq.ParquetWriter | None = None
    tie_allowance = [int(ties)]
    columns = _available_token_columns(reader)
    try:
        for domain in ("code", "wiki"):
            for batch in reader.token_batches(domain, "selection", columns):
                evaluated = evaluate_batch(batch, bundles, legacy_primary, legacy_repro)
                base = evaluated.by_level[95]
                window_uid = _arrow_scalar_numpy(batch, "window_uid", np.int64)
                position = _arrow_scalar_numpy(batch, "position", np.int64)
                priorities = _token_priorities(window_uid, position, seed)
                if domain == "code":
                    random_selected = np.zeros(len(batch), dtype=bool)
                    eligible_indices = np.flatnonzero(base["full_eligible"])
                    if eligible_indices.size:
                        random_selected[eligible_indices] = _random_mask(
                            priorities[eligible_indices], cutoff, tie_allowance
                        )
                else:
                    random_selected = np.zeros(len(batch), dtype=bool)
                flags = {
                    "legacy_cosine": evaluated.legacy_primary,
                    "cka_only95": base["cka_only"],
                    "cka_plus_m95": base["full"],
                    "matched_random": random_selected,
                    "legacy_in_split_reproduction": evaluated.legacy_reproduction,
                }
                summaries[domain].add(flags)
                for key, values in _histogram_values(batch):
                    histogram_bank.add_counts(key, "Code" if domain == "code" else "Wiki", values)

                if domain != "code":
                    continue
                routing_values = _routing_batch(batch)
                if routing_values is not None:
                    routing["selected"].add(routing_values, flags["cka_plus_m95"])
                    routing["nonselected"].add(routing_values, ~flags["cka_plus_m95"])

                legacy_only = flags["legacy_cosine"] & ~flags["cka_plus_m95"]
                cka_only = flags["cka_plus_m95"] & ~flags["legacy_cosine"]
                rel = _arrow_list_numpy(batch, "relative_l2", 8)
                maha = _optional_list(batch, "maha", 8)
                diag128 = _arrow_list_numpy(batch, "worst_diag_ratio_128", 8)
                diag256 = _arrow_list_numpy(batch, "worst_diag_ratio_256", 8)
                if maha is not None and np.any(flags["cka_plus_m95"]):
                    for layer_index, layer in enumerate(LAYERS):
                        histogram_bank.add_counts(
                            ("maha", layer),
                            "CKA+M95 selected Code",
                            maha[flags["cka_plus_m95"], layer_index],
                        )
                for group_name, mask in (("legacy_only", legacy_only), ("cka_plus_m_only", cka_only)):
                    if np.any(mask):
                        profile_values[group_name]["relative_l2"].append(np.nanmedian(rel[mask], axis=1))
                        if maha is not None:
                            profile_values[group_name]["maha"].append(np.nanmean(maha[mask], axis=1))
                        profile_values[group_name]["diag_ratio"].append(
                            np.nanmedian(np.concatenate((diag128[mask], diag256[mask]), axis=1), axis=1)
                        )

                for selector_name in _selector_names():
                    mask = flags[selector_name]
                    indices = np.flatnonzero(mask)
                    if indices.size:
                        rows = [_context_row(batch, int(index), evaluated) for index in indices]
                        samples.add(selector_name, priorities[indices], rows)

                output: dict[str, Any] = {
                    name: batch.column(batch.schema.get_field_index(name))
                    for name in IDENTITY_COLUMNS
                    if name in batch.schema.names
                }
                for name, values in flags.items():
                    output[name] = pa.array(values)
                output["full_eligible95"] = pa.array(base["full_eligible"])
                output["random_priority"] = pa.array(priorities)
                output_table = pa.table(output)
                if writer is None:
                    writer = pq.ParquetWriter(
                        selector_dir / "code_selection_assignments.parquet.inprogress",
                        output_table.schema,
                        compression="zstd",
                    )
                writer.write_table(output_table)
    finally:
        if writer is not None:
            writer.close()
    temporary = selector_dir / "code_selection_assignments.parquet.inprogress"
    if temporary.exists():
        os.replace(temporary, selector_dir / "code_selection_assignments.parquet")

    for selector_name in _selector_names():
        path = context_dir / f"{selector_name}.jsonl"
        lines = "".join(
            json.dumps(_json_clean(row), sort_keys=True, allow_nan=False) + "\n"
            for row in samples.rows(selector_name)
        )
        _atomic_text(path, lines)

    profiles: dict[str, Any] = {}
    for group, metrics in profile_values.items():
        profiles[group] = {}
        for metric, chunks in metrics.items():
            values = np.concatenate(chunks) if chunks else np.empty(0)
            profiles[group][metric] = {
                "count": int(values.size),
                "median": _json_number(float(np.nanmedian(values))) if values.size else None,
                "p05": _json_number(float(np.nanquantile(values, 0.05))) if values.size else None,
                "p95": _json_number(float(np.nanquantile(values, 0.95))) if values.size else None,
            }
    routing_payload = {name: value.serialize() for name, value in routing.items()}
    _atomic_json(selector_dir / "routing_preservation.json", routing_payload)
    _atomic_json(selector_dir / "exclusive_set_profiles.json", profiles)
    return summaries, routing_payload, profiles


def _pairwise_rows(summary: CountSummary) -> list[dict[str, Any]]:
    names = list(_selector_names())
    rows = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            intersection = summary.intersections.get((first, second), summary.intersections.get((second, first), 0))
            union = summary.counts.get(first, 0) + summary.counts.get(second, 0) - intersection
            rows.append(
                {
                    "selector_a": first,
                    "selector_b": second,
                    "count_a": summary.counts.get(first, 0),
                    "count_b": summary.counts.get(second, 0),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": _ratio(intersection, union),
                }
            )
    return rows


def _enrich_context_samples(artifact_root: Path, output_root: Path, domain: str = "code") -> dict[str, Any]:
    """Attach exact +/-32 document tokens after selector priority sampling.

    Context is intentionally reconstructed here rather than duplicated on
    every bulk token row.  A real prepared pilot always has ``config.json``;
    fixtures without it retain explicit unavailable reasons.
    """

    context_dir = output_root / "context_samples"
    paths = sorted(context_dir.glob("*.jsonl"))
    config_path = artifact_root / "config.json"
    if not config_path.exists():
        reason = "prepared pilot config.json is absent (synthetic fixture or incomplete artifact)"
        for path in paths:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            for row in rows:
                row["context_available"] = False
                row["context_unavailable_reason"] = reason
            _atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        payload = {"available": False, "reason": reason, "files": len(paths)}
        _atomic_json(context_dir / "context_enrichment.json", payload)
        return payload

    config = json.loads(config_path.read_text(encoding="utf-8"))
    prefix = config.get(f"{domain}_prefix")
    if not prefix:
        # Domain manifests are an authoritative fallback.
        candidates = [
            artifact_root / "splits" / f"{domain}_manifest.json",
            artifact_root / "splits" / domain / "manifest.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                prefix = json.loads(candidate.read_text(encoding="utf-8")).get("dataset_prefix")
                if prefix:
                    break
    if not prefix:
        raise RuntimeError(f"cannot enrich contexts: {domain} dataset prefix absent from prepared config/manifests")
    from scripts.analysis.cka_gt_pilot_windows import MMapIndexedDatasetLite

    dataset = MMapIndexedDatasetLite(str(prefix))
    tokenizer = None
    configured_tokenizer = config.get("tokenizer")
    tokenizer_path = config.get("tokenizer_path") or config.get("tokenizer_name_or_path")
    if not tokenizer_path and isinstance(configured_tokenizer, dict):
        tokenizer_path = configured_tokenizer.get("path") or configured_tokenizer.get("name_or_path")
    if tokenizer_path:
        tokenizer_source = "prepared_config"
    else:
        tokenizer_path = str(DEFAULT_LOCAL_TOKENIZER_PATH)
        tokenizer_source = "runner_default_local_pythia12b_snapshot_fallback"
    tokenizer_reason = ""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    except Exception as error:  # decoding is optional; token IDs are mandatory
        tokenizer_reason = f"local tokenizer decode unavailable: {type(error).__name__}: {error}"

    record_count = 0
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for row in rows:
            document_id = int(row["document_id"])
            center = int(
                row.get(
                    "document_token_offset",
                    int(row.get("window_offset", 0)) + int(row["position"]),
                )
            )
            document_length = int(dataset.sequence_lengths[document_id])
            if center < 0 or center >= document_length:
                raise RuntimeError(
                    f"context center outside document: doc={document_id} offset={center} length={document_length}"
                )
            start = max(0, center - 32)
            end = min(document_length, center + 33)
            ids = np.asarray(dataset.get(document_id, offset=start, length=end - start), dtype=np.int64)
            target_offset = center - start
            if int(ids[target_offset]) != int(row["token_id"]):
                raise RuntimeError(
                    "context/token identity mismatch: "
                    f"doc={document_id} offset={center} artifact={row['token_id']} dataset={ids[target_offset]}"
                )
            row["context_start_document_offset"] = start
            row["context_token_ids"] = [int(value) for value in ids]
            row["target_offset_in_context"] = target_offset
            row["context_radius"] = 32
            row["context_available"] = True
            row["context_tokenizer_source"] = tokenizer_source
            row["context_tokenizer_path"] = str(tokenizer_path)
            if tokenizer is not None:
                row["context_text"] = tokenizer.decode(row["context_token_ids"], skip_special_tokens=False)
                row["target_token_text"] = tokenizer.decode([int(row["token_id"])], skip_special_tokens=False)
            else:
                row["context_text_unavailable_reason"] = tokenizer_reason
            row.pop("context_unavailable_reason", None)
            record_count += 1
        _atomic_text(
            path,
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        )
    payload = {
        "available": True,
        "dataset_prefix": str(prefix),
        "records": record_count,
        "radius": 32,
        "token_identity_verified": True,
        "decoded_text_available": tokenizer is not None,
        "tokenizer_source": tokenizer_source,
        "tokenizer_path": str(tokenizer_path),
        "decode_note": tokenizer_reason or "decoded with local tokenizer",
    }
    _atomic_json(context_dir / "context_enrichment.json", payload)
    return payload


def _threshold_rows(first_pass: Mapping[str, Mapping[int, CountSummary]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in BUNDLES:
        code = first_pass["code"][level]
        wiki = first_pass["wiki"][level]
        code_selected = code.counts.get("full", 0)
        wiki_selected = wiki.counts.get("full", 0)
        code_rate = _ratio(code_selected, code.total)
        wiki_rate = _ratio(wiki_selected, wiki.total)
        precision = wiki_rate / (wiki_rate + code_rate) if wiki_rate + code_rate else float("nan")
        rows.append(
            {
                "candidate_bundle": level,
                "combined_wiki_recall": wiki_rate,
                "code_coverage_percent": 100.0 * code_rate,
                "code_selected_tokens": code_selected,
                "equal_prior_operational_precision": precision,
                "code_ineligible_fraction": 1.0 - _ratio(code.counts.get("full_eligible", 0), code.total),
                "wiki_ineligible_fraction": 1.0 - _ratio(wiki.counts.get("full_eligible", 0), wiki.total),
                "code_B_pass_fraction": _ratio(code.counts.get("B", 0), code.total),
                "code_T_pass_fraction": _ratio(code.counts.get("T", 0), code.total),
                "code_L2_pass_fraction": _ratio(code.counts.get("L2", 0), code.total),
                "code_log_r_pass_fraction": _ratio(code.counts.get("R", 0), code.total),
            }
        )
    return rows


def _same_layer_rows(first_pass: Mapping[str, Mapping[int, CountSummary]]) -> list[dict[str, Any]]:
    rows = []
    for domain in ("code", "wiki"):
        for level in BUNDLES:
            summary = first_pass[domain][level]
            condition_count = summary.counts.get("full", 0)
            same_count = summary.counts.get("same_selected", 0)
            intersection = summary.intersections.get(("full", "same_selected"), 0)
            union = condition_count + same_count - intersection
            jaccard = _ratio(intersection, union)
            rows.append(
                {
                    "domain": domain,
                    "candidate_bundle": level,
                    "condition_specific_count": condition_count,
                    "same_layer_count": same_count,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": jaccard,
                    "manual_review_required": bool(math.isfinite(jaccard) and jaccard < 0.9),
                }
            )
    return rows


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in columns) + " |\n"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = "NA" if not math.isfinite(value) else f"{value:.6g}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |\n")
    return header + separator + "".join(body)


def _flatten_arrow_list_where(batch: pa.RecordBatch, name: str, mask: np.ndarray) -> np.ndarray:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise KeyError(name)
    array = batch.column(index)
    if not isinstance(array, (pa.ListArray, pa.LargeListArray, pa.FixedSizeListArray)):
        raise TypeError(f"{name} must be an Arrow list column, got {array.type}")
    filtered = array.filter(pa.array(np.asarray(mask, dtype=np.bool_)))
    flattened = filtered.flatten()
    return flattened.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)


class _RawSSafetyAccumulator:
    def __init__(self, seed: int) -> None:
        self.total = 0
        self.finite = 0
        self.nonfinite = 0
        self.negative = 0
        self.abs_gt_10 = 0
        self.abs_gt_100 = 0
        self.abs_gt_1000 = 0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.max_abs = 0.0
        self.reservoir = PriorityReservoir(50_000, seed)

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        self.total += int(values.size)
        finite_mask = np.isfinite(values)
        self.nonfinite += int(values.size - np.count_nonzero(finite_mask))
        finite = values[finite_mask]
        self.finite += int(finite.size)
        if not finite.size:
            return
        absolute = np.abs(finite)
        self.negative += int(np.count_nonzero(finite < 0))
        self.abs_gt_10 += int(np.count_nonzero(absolute > 10.0))
        self.abs_gt_100 += int(np.count_nonzero(absolute > 100.0))
        self.abs_gt_1000 += int(np.count_nonzero(absolute > 1000.0))
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.max_abs = max(self.max_abs, float(absolute.max()))
        self.reservoir.add(finite)

    def serialize(self) -> dict[str, Any]:
        sample = self.reservoir.values
        return {
            "total_values": self.total,
            "finite_values": self.finite,
            "nonfinite_values": self.nonfinite,
            "negative_count": self.negative,
            "negative_fraction_of_finite": _json_number(_ratio(self.negative, self.finite)),
            "abs_gt_10_count": self.abs_gt_10,
            "abs_gt_100_count": self.abs_gt_100,
            "abs_gt_1000_count": self.abs_gt_1000,
            "minimum": _json_number(self.minimum) if self.finite else None,
            "maximum": _json_number(self.maximum) if self.finite else None,
            "max_abs": _json_number(self.max_abs) if self.finite else None,
            "sample_count": int(sample.size),
            "sample_p01": _json_number(float(np.quantile(sample, 0.01))) if sample.size else None,
            "sample_median": _json_number(float(np.quantile(sample, 0.5))) if sample.size else None,
            "sample_p99": _json_number(float(np.quantile(sample, 0.99))) if sample.size else None,
            "sample_method": "deterministic_priority_reservoir_up_to_50k",
        }


class _OffdiagAccumulator:
    def __init__(self, seed: int) -> None:
        self.chunks = 0
        self.warning = 0
        self.finite_cka_off = 0
        self.nonfinite_cka_off = 0
        self.nonpositive_cka_off = 0
        self.minimum_cka_off = float("inf")
        self.reservoir = PriorityReservoir(50_000, seed)

    def add(self, cka_off: np.ndarray, warning: np.ndarray) -> None:
        values = np.asarray(cka_off, dtype=np.float32).reshape(-1)
        warnings = np.asarray(warning, dtype=np.bool_).reshape(-1)
        if values.size != warnings.size:
            raise ValueError("CKA_off/offdiag_warning length mismatch")
        self.chunks += int(values.size)
        self.warning += int(np.count_nonzero(warnings))
        finite_mask = np.isfinite(values)
        finite = values[finite_mask]
        self.finite_cka_off += int(finite.size)
        self.nonfinite_cka_off += int(values.size - finite.size)
        if finite.size:
            self.nonpositive_cka_off += int(np.count_nonzero(finite <= 0))
            self.minimum_cka_off = min(self.minimum_cka_off, float(finite.min()))
            self.reservoir.add(finite)

    def serialize(self) -> dict[str, Any]:
        sample = self.reservoir.values
        return {
            "chunk_count": self.chunks,
            "offdiag_warning_count": self.warning,
            "offdiag_warning_fraction": _json_number(_ratio(self.warning, self.chunks)),
            "finite_cka_off_count": self.finite_cka_off,
            "nonfinite_cka_off_count": self.nonfinite_cka_off,
            "nonpositive_cka_off_count": self.nonpositive_cka_off,
            "minimum_cka_off": _json_number(self.minimum_cka_off) if self.finite_cka_off else None,
            "sample_p01_cka_off": _json_number(float(np.quantile(sample, 0.01))) if sample.size else None,
            "sample_median_cka_off": _json_number(float(np.quantile(sample, 0.5))) if sample.size else None,
            "sample_method": "deterministic_priority_reservoir_up_to_50k",
        }


def _analyze_chunks(
    reader: ParquetArtifactReader,
    output_root: Path,
    bins: int,
    seed: int,
    bundles: Mapping[int, CandidateBundle],
) -> dict[str, Any]:
    if reader.chunk is None:
        payload = {"available": False, "reason": "chunk_metrics dataset missing"}
        _atomic_json(output_root / "histograms" / "chunk_summary.json", payload)
        return payload
    required = ["domain", "split", "scale", "layer", "cka", "cka_off", "offdiag_warning", "s_i"]
    missing = [name for name in required if name not in reader.chunk.schema.names]
    if missing:
        raise RuntimeError(f"chunk metric schema missing: {missing}")
    available = set(reader.chunk.schema.names)
    bank = HistogramBank(bins, seed ^ 0xCA11)
    quantile_reservoirs: dict[tuple[int, int, str], PriorityReservoir] = {}
    null_completeness: dict[tuple[str, str, int, int, str], dict[str, int]] = {}
    random_pair_reasons: dict[tuple[str, str, int, int, int], int] = {}
    raw_s_summaries: dict[tuple[str, int, int], _RawSSafetyAccumulator] = {}
    offdiag_summaries: dict[tuple[str, int, int], _OffdiagAccumulator] = {}

    def raw_s_accumulator(domain: str, scale: int, layer: int) -> _RawSSafetyAccumulator:
        key = (domain, scale, layer)
        if key not in raw_s_summaries:
            digest = int(hashlib.sha256(repr(("raw_s", key)).encode()).hexdigest()[:8], 16)
            raw_s_summaries[key] = _RawSSafetyAccumulator(seed ^ digest)
        return raw_s_summaries[key]

    def offdiag_accumulator(domain: str, scale: int, layer: int) -> _OffdiagAccumulator:
        key = (domain, scale, layer)
        if key not in offdiag_summaries:
            digest = int(hashlib.sha256(repr(("offdiag", key)).encode()).hexdigest()[:8], 16)
            offdiag_summaries[key] = _OffdiagAccumulator(seed ^ digest)
        return offdiag_summaries[key]

    def observe_quantile(scale_value: int, layer_value: int, series: str, values: np.ndarray) -> None:
        key = (scale_value, layer_value, series)
        if key not in quantile_reservoirs:
            digest = int(hashlib.sha256(repr(key).encode()).hexdigest()[:8], 16)
            quantile_reservoirs[key] = PriorityReservoir(200_000, seed ^ digest)
        quantile_reservoirs[key].add(values)

    def observe_null_completeness(
        domain: str,
        split: str,
        scale_value: int,
        layer_value: int,
        null_kind: str,
        values: np.ndarray,
    ) -> None:
        key = (domain, split, scale_value, layer_value, null_kind)
        entry = null_completeness.setdefault(key, {"total_count": 0, "finite_count": 0})
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        entry["total_count"] += int(values.size)
        entry["finite_count"] += int(np.count_nonzero(np.isfinite(values)))
    invalid: dict[str, int] = {}
    total = 0
    rows_by_scale = {scale: 0 for scale in CHUNK_DIAGNOSTIC_SCALES}
    columns = [
        name
        for name in (
            "domain",
            "split",
            "window_uid",
            "scale",
            "layer",
            "cka",
            "cka_permutation",
            "cka_random_pair",
            "random_pair_invalid_reason",
            "random_pair_donor_window_uid",
            "diag_ratio",
            "cka_off",
            "offdiag_warning",
            "s_i",
            "invalid_reason",
        )
        if name in available
    ]
    for domain in ("code", "wiki"):
        for batch in reader.chunk_batches(domain, "selection", columns):
            scale = _arrow_scalar_numpy(batch, "scale", np.int16)
            layer = _arrow_scalar_numpy(batch, "layer", np.int16)
            cka = _arrow_scalar_numpy(batch, "cka", np.float32)
            cka_off = _arrow_scalar_numpy(batch, "cka_off", np.float32)
            offdiag_warning = _arrow_scalar_numpy(batch, "offdiag_warning", np.bool_)
            random_pair_reason = (
                _arrow_scalar_numpy(batch, "random_pair_invalid_reason", np.uint8)
                if "random_pair_invalid_reason" in batch.schema.names
                else None
            )
            random_pair_donor = (
                _arrow_scalar_numpy(batch, "random_pair_donor_window_uid", np.int64)
                if "random_pair_donor_window_uid" in batch.schema.names
                else None
            )
            window_uid = (
                _arrow_scalar_numpy(batch, "window_uid", np.int64)
                if "window_uid" in batch.schema.names
                else None
            )
            random_pair_values = (
                _arrow_scalar_numpy(batch, "cka_random_pair", np.float32)
                if "cka_random_pair" in batch.schema.names
                else None
            )
            if random_pair_reason is not None:
                if random_pair_donor is None or window_uid is None or random_pair_values is None:
                    raise RuntimeError(
                        "random_pair_invalid_reason requires donor UID, window UID, and random-pair CKA"
                    )
                valid_reason = random_pair_reason == 0
                if np.any(valid_reason & ~np.isfinite(random_pair_values)):
                    raise RuntimeError("reason-0 random-pair rows must have finite CKA")
                if np.any(valid_reason & (random_pair_donor == window_uid)):
                    raise RuntimeError("reason-0 random-pair donor must differ from source window")
            total += len(batch)
            for scale_value in CHUNK_DIAGNOSTIC_SCALES:
                rows_by_scale[scale_value] += int(np.count_nonzero(scale == scale_value))
            if "invalid_reason" in batch.schema.names:
                for reason in batch.column(batch.schema.get_field_index("invalid_reason")).to_pylist():
                    key = str(reason or "valid")
                    invalid[key] = invalid.get(key, 0) + 1
            for scale_value in CHUNK_DIAGNOSTIC_SCALES:
                for layer_value in LAYERS:
                    mask = (scale == scale_value) & (layer == layer_value)
                    if not np.any(mask):
                        continue
                    if random_pair_reason is not None:
                        for reason_code in range(4):
                            count = int(np.count_nonzero(mask & (random_pair_reason == reason_code)))
                            key_reason = (
                                domain,
                                "selection",
                                scale_value,
                                layer_value,
                                reason_code,
                            )
                            random_pair_reasons[key_reason] = (
                                random_pair_reasons.get(key_reason, 0) + count
                            )
                        unexpected = np.unique(random_pair_reason[mask])
                        if any(int(code) not in (0, 1, 2, 3) for code in unexpected):
                            raise RuntimeError(
                                f"unknown random_pair_invalid_reason code(s): {unexpected.tolist()}"
                            )
                    key = ("chunk_cka", scale_value, layer_value)
                    bank.observe_range(key, cka[mask])
                    observe_quantile(
                        scale_value,
                        layer_value,
                        "Code real" if domain == "code" else "Wiki real",
                        cka[mask],
                    )
                    for column, label in (("cka_permutation", "Permutation null"), ("cka_random_pair", "Random-pair null")):
                        if column in batch.schema.names:
                            null_values = _arrow_scalar_numpy(batch, column, np.float32)[mask]
                            observe_quantile(
                                scale_value,
                                layer_value,
                                label,
                                null_values,
                            )
                            observe_null_completeness(
                                domain,
                                "selection",
                                scale_value,
                                layer_value,
                                "permutation" if column == "cka_permutation" else "random_pair",
                                null_values,
                            )
                    if "diag_ratio" in batch.schema.names:
                        diag = _arrow_scalar_numpy(batch, "diag_ratio", np.float32)
                        bank.observe_range(("diag_ratio", scale_value, layer_value), diag[mask])
                    raw_s = _flatten_arrow_list_where(batch, "s_i", mask)
                    bank.observe_range(("raw_s_i", scale_value, layer_value), raw_s)
                    bank.observe_range(("cka_off", scale_value, layer_value), cka_off[mask])
                    raw_s_accumulator(domain, scale_value, layer_value).add(raw_s)
                    offdiag_accumulator(domain, scale_value, layer_value).add(
                        cka_off[mask], offdiag_warning[mask]
                    )
    bank.finalize_ranges()
    for domain in ("code", "wiki"):
        for batch in reader.chunk_batches(domain, "selection", columns):
            scale = _arrow_scalar_numpy(batch, "scale", np.int16)
            layer = _arrow_scalar_numpy(batch, "layer", np.int16)
            cka_off = _arrow_scalar_numpy(batch, "cka_off", np.float32)
            for scale_value in CHUNK_DIAGNOSTIC_SCALES:
                for layer_value in LAYERS:
                    mask = (scale == scale_value) & (layer == layer_value)
                    if not np.any(mask):
                        continue
                    key = ("chunk_cka", scale_value, layer_value)
                    bank.add_counts(key, "Code" if domain == "code" else "Wiki", _arrow_scalar_numpy(batch, "cka", np.float32)[mask])
                    for column, label in (("cka_permutation", "Permutation null"), ("cka_random_pair", "Random-pair null")):
                        if column in batch.schema.names:
                            bank.add_counts(key, label, _arrow_scalar_numpy(batch, column, np.float32)[mask])
                    if "diag_ratio" in batch.schema.names:
                        bank.add_counts(
                            ("diag_ratio", scale_value, layer_value),
                            "Code" if domain == "code" else "Wiki",
                            _arrow_scalar_numpy(batch, "diag_ratio", np.float32)[mask],
                        )
                    bank.add_counts(
                        ("raw_s_i", scale_value, layer_value),
                        "Code" if domain == "code" else "Wiki",
                        _flatten_arrow_list_where(batch, "s_i", mask),
                    )
                    bank.add_counts(
                        ("cka_off", scale_value, layer_value),
                        "Code" if domain == "code" else "Wiki",
                        cka_off[mask],
                    )
    missing_control_scales = [
        scale for scale, count in rows_by_scale.items() if count <= 0
    ]
    if missing_control_scales:
        raise RuntimeError(
            "selection chunk metrics omit required diagnostic scale(s): "
            f"{missing_control_scales}; scale 512 is a diagnostic-only whole-window control"
        )
    _write_histogram_bank(bank, output_root / "histograms")
    null_rows: list[dict[str, Any]] = []
    for scale_value in CHUNK_DIAGNOSTIC_SCALES:
        for layer_index, layer_value in enumerate(LAYERS):
            row: dict[str, Any] = {
                "scale": scale_value,
                "layer": layer_value,
                "selector_threshold_role": (
                    "candidate_bundle_source"
                    if scale_value in SCALES
                    else "diagnostic_only_no_bundle_threshold"
                ),
            }
            for series in ("Code real", "Wiki real", "Permutation null", "Random-pair null"):
                values = quantile_reservoirs.get((scale_value, layer_value, series))
                sample = values.values if values is not None else np.empty(0)
                stem = series.lower().replace(" ", "_").replace("-", "_")
                row[f"{stem}_count_seen"] = int(values.seen) if values is not None else 0
                row[f"{stem}_median"] = float(np.quantile(sample, 0.5)) if sample.size else float("nan")
                row[f"{stem}_p95"] = float(np.quantile(sample, 0.95)) if sample.size else float("nan")
                if series in ("Permutation null", "Random-pair null"):
                    null_kind = "permutation" if series == "Permutation null" else "random_pair"
                    group_values = [
                        null_completeness.get(
                            (domain, "selection", scale_value, layer_value, null_kind),
                            {"total_count": 0, "finite_count": 0},
                        )
                        for domain in ("code", "wiki")
                    ]
                    total_count = sum(value["total_count"] for value in group_values)
                    finite_count = sum(value["finite_count"] for value in group_values)
                    missing_count = total_count - finite_count
                    row[f"{stem}_total_count"] = total_count
                    row[f"{stem}_finite_count"] = finite_count
                    row[f"{stem}_missing_count"] = missing_count
                    row[f"{stem}_missing_fraction"] = _json_number(
                        _ratio(missing_count, total_count)
                    )
            null_candidates = np.asarray(
                [
                    row.get("permutation_null_p95", float("nan")),
                    row.get("random_pair_null_p95", float("nan")),
                ],
                dtype=np.float64,
            )
            null95 = float(np.nanmax(null_candidates)) if np.isfinite(null_candidates).any() else float("nan")
            row["max_null_p95"] = null95
            if scale_value in SCALES:
                for level in BUNDLES:
                    threshold = float(bundles[level].b[scale_value][layer_index])
                    row[f"B{level}_threshold"] = threshold
                    row[f"B{level}_above_max_null_p95"] = bool(
                        math.isfinite(null95) and threshold > null95
                    )
            row["quantile_method"] = "deterministic_priority_reservoir_up_to_200k_per_series"
            null_rows.append(row)
    _write_csv(output_root / "histograms" / "chunk_cka_null_summary.csv", null_rows)
    null_completeness_rows: list[dict[str, Any]] = []
    for domain in ("code", "wiki"):
        for split in ("selection",):
            for scale_value in CHUNK_DIAGNOSTIC_SCALES:
                for layer_value in LAYERS:
                    for null_kind in ("permutation", "random_pair"):
                        counts = null_completeness.get(
                            (domain, split, scale_value, layer_value, null_kind),
                            {"total_count": 0, "finite_count": 0},
                        )
                        total_count = int(counts["total_count"])
                        finite_count = int(counts["finite_count"])
                        missing_count = total_count - finite_count
                        null_completeness_rows.append(
                            {
                                "domain": domain,
                                "split": split,
                                "scale": scale_value,
                                "layer": layer_value,
                                "null_kind": null_kind,
                                "total_count": total_count,
                                "finite_count": finite_count,
                                "missing_count": missing_count,
                                "missing_fraction": _json_number(
                                    _ratio(missing_count, total_count)
                                ),
                                "diagnostic_only": True,
                            }
                        )
    _write_csv(
        output_root / "histograms" / "chunk_cka_null_completeness.csv",
        null_completeness_rows,
    )
    random_pair_reason_rows: list[dict[str, Any]] = []
    reason_labels = {
        0: "valid",
        1: "singleton_no_cache",
        2: "no_nonself_full_donor",
        3: "paired_cka_invalid",
    }
    if "random_pair_invalid_reason" in available:
        for domain in ("code", "wiki"):
            for split in ("selection",):
                for scale_value in CHUNK_DIAGNOSTIC_SCALES:
                    for layer_value in LAYERS:
                        total_group = sum(
                            random_pair_reasons.get(
                                (domain, split, scale_value, layer_value, reason_code), 0
                            )
                            for reason_code in reason_labels
                        )
                        for reason_code, reason_label in reason_labels.items():
                            count = random_pair_reasons.get(
                                (domain, split, scale_value, layer_value, reason_code), 0
                            )
                            random_pair_reason_rows.append(
                                {
                                    "domain": domain,
                                    "split": split,
                                    "scale": scale_value,
                                    "layer": layer_value,
                                    "reason_code": reason_code,
                                    "reason": reason_label,
                                    "count": count,
                                    "fraction": _json_number(_ratio(count, total_group)),
                                    "diagnostic_only": True,
                                }
                            )
        _write_csv(
            output_root / "histograms" / "random_pair_invalid_reason_summary.csv",
            random_pair_reason_rows,
        )

    raw_s_rows: list[dict[str, Any]] = []
    offdiag_rows: list[dict[str, Any]] = []
    explosion_groups: list[dict[str, Any]] = []
    watch_groups: list[dict[str, Any]] = []
    for domain in ("code", "wiki"):
        for scale_value in CHUNK_DIAGNOSTIC_SCALES:
            for layer_value in LAYERS:
                key = (domain, scale_value, layer_value)
                raw = raw_s_summaries.get(key)
                offdiag = offdiag_summaries.get(key)
                if raw is None or offdiag is None:
                    raise RuntimeError(f"missing raw-s/offdiag selection diagnostics for {key}")
                raw_row = {
                    "domain": domain,
                    "scale": scale_value,
                    "layer": layer_value,
                    **raw.serialize(),
                }
                offdiag_row = {
                    "domain": domain,
                    "scale": scale_value,
                    "layer": layer_value,
                    **offdiag.serialize(),
                }
                raw_s_rows.append(raw_row)
                offdiag_rows.append(offdiag_row)
                identity = {"domain": domain, "scale": scale_value, "layer": layer_value}
                if raw.abs_gt_10:
                    watch_groups.append({**identity, "abs_gt_10_count": raw.abs_gt_10})
                if raw.abs_gt_100:
                    explosion_groups.append({**identity, "abs_gt_100_count": raw.abs_gt_100})
    _write_csv(output_root / "histograms" / "raw_s_i_summary.csv", raw_s_rows)
    _write_csv(output_root / "histograms" / "offdiag_warning_summary.csv", offdiag_rows)

    finite_total = sum(value.finite for value in raw_s_summaries.values())
    negative_total = sum(value.negative for value in raw_s_summaries.values())
    abs_gt_10_total = sum(value.abs_gt_10 for value in raw_s_summaries.values())
    abs_gt_100_total = sum(value.abs_gt_100 for value in raw_s_summaries.values())
    abs_gt_1000_total = sum(value.abs_gt_1000 for value in raw_s_summaries.values())
    chunk_total = sum(value.chunks for value in offdiag_summaries.values())
    warning_total = sum(value.warning for value in offdiag_summaries.values())
    raw_s_safety = {
        "negative": {
            "count": negative_total,
            "fraction_of_finite": _json_number(_ratio(negative_total, finite_total)),
        },
        "extreme_absolute_counts": {
            "abs_gt_10": abs_gt_10_total,
            "abs_gt_100": abs_gt_100_total,
            "abs_gt_1000": abs_gt_1000_total,
        },
        "explosion_warning": {
            "triggered": bool(abs_gt_100_total),
            "rule": "any finite raw chunk-token s_i with abs(s_i) > 100",
            "diagnostic_only": True,
            "trigger_groups": explosion_groups,
            "watch_groups_abs_gt_10": watch_groups,
        },
        "offdiag_warning": {
            "chunk_count": chunk_total,
            "warning_count": warning_total,
            "incidence": _json_number(_ratio(warning_total, chunk_total)),
            "definition": "runtime offdiag_warning; selector-independent safety diagnostic",
        },
        "group_table": "histograms/raw_s_i_summary.csv",
        "offdiag_group_table": "histograms/offdiag_warning_summary.csv",
    }
    payload = {
        "available": True,
        "selection_chunk_layer_rows": total,
        "selection_chunk_layer_rows_by_scale": {
            str(scale): rows_by_scale[scale] for scale in CHUNK_DIAGNOSTIC_SCALES
        },
        "diagnostic_scales": list(CHUNK_DIAGNOSTIC_SCALES),
        "selector_threshold_scales": list(SCALES),
        "scale_512_role": "whole_window_diagnostic_control_only",
        "raw_s_i_safety": raw_s_safety,
        "invalid_reason_counts": invalid,
        "invalid_fraction": _ratio(total - invalid.get("valid", total), total),
        "null_overlay_columns": [name for name in ("cka_permutation", "cka_random_pair") if name in available],
        "null_missingness": {
            "group_table": "histograms/chunk_cka_null_completeness.csv",
            "group_keys": ["domain", "split", "scale", "layer", "null_kind"],
            "diagnostic_only": True,
            "reason": (
                "A random-pair donor is unavailable for a singleton exact-length batch; "
                "nonfinite null values are counted, never imputed, and never affect selectors."
            ),
            "missing_count": int(
                sum(row["missing_count"] for row in null_completeness_rows)
            ),
            "total_count": int(sum(row["total_count"] for row in null_completeness_rows)),
        },
        "random_pair_invalid_reasons": {
            "available": bool(random_pair_reason_rows),
            "group_table": (
                "histograms/random_pair_invalid_reason_summary.csv"
                if random_pair_reason_rows
                else None
            ),
            "reason_codes": reason_labels,
            "diagnostic_only": True,
            "valid_count": int(
                sum(row["count"] for row in random_pair_reason_rows if row["reason_code"] == 0)
            ),
            "invalid_count": int(
                sum(row["count"] for row in random_pair_reason_rows if row["reason_code"] != 0)
            ),
        },
    }
    _atomic_json(output_root / "histograms" / "chunk_summary.json", payload)
    return payload


def _sealed_test(
    reader: ParquetArtifactReader,
    output_root: Path,
    bundles: Mapping[int, CandidateBundle],
    legacy_primary: np.ndarray,
    legacy_repro: np.ndarray,
) -> Path:
    result: dict[str, Any] = {
        "schema": "cka_gt_pilot_sealed_test_v1",
        "sealed": True,
        "warning": "Do not inspect before a human records the final threshold choice. Any later threshold change contaminates test.",
        "domains": {},
    }
    columns = _available_token_columns(reader)
    for domain in ("code", "wiki"):
        summaries = {level: CountSummary() for level in BUNDLES}
        legacy_count = 0
        total = 0
        for batch in reader.token_batches(domain, "test", columns):
            evaluated = evaluate_batch(batch, bundles, legacy_primary, legacy_repro)
            total += len(batch)
            legacy_count += int(np.count_nonzero(evaluated.legacy_primary))
            for level in BUNDLES:
                fields = evaluated.by_level[level]
                summaries[level].add(
                    {
                        "full_eligible": fields["full_eligible"],
                        "cka_only": fields["cka_only"],
                        "full": fields["full"],
                        "same_selected": fields["same_selected"],
                    }
                )
        result["domains"][domain] = {
            "total_tokens": total,
            "legacy_primary_count": legacy_count,
            "candidate_bundles": {
                str(level): {
                    "eligible_count": summaries[level].counts.get("full_eligible", 0),
                    "cka_only_count": summaries[level].counts.get("cka_only", 0),
                    "cka_plus_m_count": summaries[level].counts.get("full", 0),
                    "same_layer_count": summaries[level].counts.get("same_selected", 0),
                }
                for level in BUNDLES
            },
        }
    sealed_dir = output_root / "sealed_test"
    path = sealed_dir / "test_metrics.DO_NOT_OPEN_BEFORE_THRESHOLD_LOCK.json"
    _atomic_json(path, result)
    _atomic_json(
        sealed_dir / "manifest.json",
        {
            "schema": "cka_gt_pilot_sealed_manifest_v1",
            "metrics_file": path.name,
            "sha256": _sha256(path),
            "included_in_report_v1": False,
            "unseal_rule": "Only after the human records one final threshold bundle; no threshold changes after inspection.",
        },
    )
    return path


def _window_summary(root: Path) -> dict[str, Any]:
    if (root / "splits").exists():
        manifests = sorted(
            set((root / "splits").glob("*_manifest.json"))
            | set((root / "splits").glob("*/manifest.json"))
        )
    else:
        manifests = []
    return {
        "manifest_paths": [str(path.relative_to(root)) for path in manifests],
        "manifests": [json.loads(path.read_text(encoding="utf-8")) for path in manifests],
    }


def _window_summary_rows(window_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten authoritative domain manifests for a compact REPORT table.

    The prepared manifests deliberately retain substantially more provenance
    than belongs in REPORT v1.  These are the construction statistics needed
    to audit document boundedness, the sampled split balance, and tail loss
    without opening the manifests separately.  Missing fields remain blank so
    synthetic/minimal fixtures do not acquire invented values.
    """

    rows: list[dict[str, Any]] = []
    for manifest in window_summary.get("manifests", []):
        if not isinstance(manifest, Mapping):
            continue
        splits = manifest.get("splits", {})
        if not isinstance(splits, Mapping):
            splits = {}
        stats = manifest.get("all_documents_window_stats", {})
        if not isinstance(stats, Mapping):
            stats = {}

        def split_windows(name: str) -> Any:
            value = splits.get(name, {})
            return value.get("sampled_window_count", "") if isinstance(value, Mapping) else ""

        split_counts = [split_windows(name) for name in ("calibration", "selection", "test")]
        sampled_total: Any = ""
        if all(isinstance(value, (int, np.integer)) for value in split_counts):
            sampled_total = int(sum(int(value) for value in split_counts))
        rows.append(
            {
                "domain": manifest.get("domain", ""),
                "document_count": manifest.get("document_count", ""),
                "sampled_windows": sampled_total,
                "full_windows": stats.get("full_window_count", ""),
                "tail_windows": stats.get("tail_window_count", ""),
                "discarded_tail_fraction": stats.get(
                    "discarded_tail_fraction_of_document_tokens", ""
                ),
                "ineligible_suffix_fraction": stats.get(
                    "ineligible_fraction_of_retained_tokens", ""
                ),
                "calibration_windows": split_counts[0],
                "selection_windows": split_counts[1],
                "test_windows": split_counts[2],
            }
        )
    return rows


def _report_validation_evidence(artifact_root: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    probe_candidates = (
        output_root / "probe_validation.json",
        artifact_root / "probe_validation.json",
        artifact_root
        / "runtime"
        / "router_smoke"
        / "code"
        / "calibration"
        / "worker_000"
        / "metadata.json",
        artifact_root
        / "runtime"
        / "pass2"
        / "code"
        / "all"
        / "worker_000"
        / "metadata.json",
        artifact_root
        / "runtime"
        / "pass1"
        / "wiki"
        / "calibration"
        / "worker_000"
        / "metadata.json",
    )
    probe: dict[str, Any] = {"status": "NOT_SUPPLIED", "searched": [str(path) for path in probe_candidates]}
    for path in probe_candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtime_fallback = path.name == "metadata.json" and "runtime" in path.parts
        passed = (
            payload.get("completed") is True
            and payload.get("router_probe_verified") is True
            and payload.get("natural_routing") is True
        ) if runtime_fallback else not (
            payload.get("ok") is False or payload.get("passed") is False
        )
        if runtime_fallback and "router_smoke" in path.parts:
            source_kind = "runtime_router_smoke_metadata"
        elif runtime_fallback and "pass2" in path.parts:
            source_kind = "runtime_pass2_router_probe_metadata"
        elif runtime_fallback and "pass1" in path.parts:
            source_kind = "runtime_pass1_router_probe_metadata"
        else:
            source_kind = "dedicated_probe_validation"
        # A completed Pass-2 metadata document also contains the paired stream
        # journals, including the physically sealed test artifact paths.  REPORT
        # v1 is intentionally open-split only, so use the runtime document as
        # evidence but serialize only the router/representation facts needed by
        # this section.  The complete document remains hash-bound separately by
        # the input inventory and validators.
        report_payload: Mapping[str, Any]
        if runtime_fallback:
            report_payload = {
                key: payload.get(key)
                for key in (
                    "analysis",
                    "mode",
                    "domain",
                    "requested_split",
                    "completed",
                    "router_probe_verified",
                    "natural_routing",
                    "standard_router_layers",
                    "layers",
                    "representation",
                    "raw_hidden_stored",
                    "training_run",
                )
            }
        else:
            report_payload = payload
        probe = {
            "status": "PASS" if passed else "FAIL",
            "source": str(path),
            "source_kind": source_kind,
            "payload": report_payload,
        }
        break

    contribution_candidates = (
        output_root / "validation" / "cka_contribution_validation.json",
        artifact_root / "validation" / "cka_contribution_validation.json",
        artifact_root / "validation" / "07_full_pre_analysis.json",
    )
    contribution: dict[str, Any] = {
        "status": "NOT_SUPPLIED",
        "searched": [str(path) for path in contribution_candidates],
    }
    for path in contribution_candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        full_validator = path.name == "07_full_pre_analysis.json"
        if full_validator:
            passed = (
                payload.get("ok") is True
                and payload.get("deep") is True
                and not payload.get("errors")
            )
            report_payload: Mapping[str, Any] = {
                "schema": payload.get("schema"),
                "ok": payload.get("ok"),
                "deep": payload.get("deep"),
                "error_count": payload.get("error_count", len(payload.get("errors", []))),
                "warning_count": payload.get("warning_count", len(payload.get("warnings", []))),
                "checks": payload.get("checks", []),
            }
        else:
            passed = not (payload.get("ok") is False or payload.get("passed") is False)
            report_payload = payload
        contribution = {
            "status": "PASS" if passed else "FAIL",
            "source": str(path),
            "source_kind": "full_pre_analysis_deep_validator" if full_validator else "dedicated_contribution_validation",
            "payload": report_payload,
        }
        break
    return probe, contribution


def _report(
    artifact_root: Path,
    output_root: Path,
    threshold_rows: Sequence[Mapping[str, Any]],
    same_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    chunk_summary: Mapping[str, Any],
    routing: Mapping[str, Any],
    window_summary: Mapping[str, Any],
    input_inventory: Mapping[str, Any],
    prepared_provenance: Mapping[str, Any],
) -> None:
    threshold_md = _markdown_table(
        threshold_rows,
        (
            "candidate_bundle",
            "combined_wiki_recall",
            "code_coverage_percent",
            "code_selected_tokens",
            "equal_prior_operational_precision",
            "code_ineligible_fraction",
            "wiki_ineligible_fraction",
        ),
    )
    same_md = _markdown_table(
        same_rows,
        ("domain", "candidate_bundle", "condition_specific_count", "same_layer_count", "jaccard", "manual_review_required"),
    )
    pair_md = _markdown_table(
        pairwise_rows,
        ("selector_a", "selector_b", "count_a", "count_b", "intersection", "union", "jaccard"),
    )
    probe, contribution = _report_validation_evidence(artifact_root, output_root)
    _atomic_json(
        output_root / "report_validation_evidence.json",
        {
            "schema": "cka_gt_pilot_report_validation_evidence_v1",
            "probe": probe,
            "contribution": contribution,
        },
    )
    window_rows = _window_summary_rows(window_summary)
    window_md = _markdown_table(
        window_rows,
        (
            "domain",
            "document_count",
            "sampled_windows",
            "full_windows",
            "tail_windows",
            "discarded_tail_fraction",
            "ineligible_suffix_fraction",
            "calibration_windows",
            "selection_windows",
            "test_windows",
        ),
    )
    manual = [row for row in same_rows if row.get("manual_review_required")]
    s_safety = chunk_summary.get("raw_s_i_safety", {})
    explosion = s_safety.get("explosion_warning", {}) if isinstance(s_safety, Mapping) else {}
    explosion_banner = (
        "**WARNING: s_i explosion candidates were observed. Inspect `histograms/raw_s_i_summary.csv` "
        "and the off-diagonal incidence table before choosing a threshold.**"
        if explosion.get("triggered")
        else "No s_i explosion candidate crossed the explicit diagnostic rule; this is diagnostic only, not a selector pass."
    )
    text = f"""# CKA GT Pilot v1 — REPORT v1

> Label semantics: **replay-stable relational anchor**. This report presents candidate thresholds only; it does not choose a final threshold and contains no test-derived metrics. Prepared split sizes below are construction metadata, not unsealed metric outcomes.

Input Parquet inventory digest: `{input_inventory.get('inventory_digest_sha256')}`. `analysis_input_binding.json` machine-binds this report and the raw quantiles to the complete open+sealed input inventory. REPORT v1 does not enumerate sealed-test paths or counts.

Prepared input provenance: `{json.dumps(prepared_provenance, sort_keys=True)}`

## 1. Probe verification

```json
{json.dumps(probe, indent=2, sort_keys=True)}
```

## 2. Window construction statistics

Split manifests: {', '.join(window_summary.get('manifest_paths', [])) or 'not supplied'}.

{window_md}

The document-overlap assertion and tail-ineligible rate are authoritative in the split manifests. Tail tokens without both fixed-stride scales remain ineligible; no right-aligned chunk was added.

## 3. Contribution identity and invalid chunks

Contribution validation: `{json.dumps(contribution, sort_keys=True)}`

Chunk summary: `{json.dumps(chunk_summary, sort_keys=True)}`

## 4. Chunk CKA distributions

See the per-scale/per-layer `histograms/chunk_cka_*.svg` files and matching CSV files. Code, Wiki, permutation-null, and random-pair-null distributions share the same raw CKA bins (>=200). Scales 128 and 256 feed candidate selectors; scale 512 is retained strictly as a whole-window diagnostic control and receives no bundle threshold. Bimodality and any valley cut remain human decisions; no automated valley threshold was promoted.

Null finite/missing counts are reported for every domain × selection split × scale × layer × null kind in `histograms/chunk_cka_null_completeness.csv`. A singleton exact-length batch can have no valid random-pair donor; its NaN is counted rather than imputed. This missingness is diagnostic-only and never changes a selector.

When runtime reason codes are present, `histograms/random_pair_invalid_reason_summary.csv` separates valid pairs, singleton/no-cache cases, missing non-self donors, and invalid paired CKA. Valid rows are checked to have a finite score and a donor window different from the source; these remain null diagnostics only.

## 5. Token contribution share s_i

See `histograms/s_i_min_scale_layer.svg` for the fixed min-over-overlapping-chunks token aggregate, and per-scale/per-layer `histograms/raw_s_i_*.svg` for the raw chunk-token contributions, including the scale-512 diagnostic control. Thresholds themselves come from the exact **pre-aggregation raw-unit** quantile table, not the token minimum. Exact negative/extreme counts are in `histograms/raw_s_i_summary.csv`.

{explosion_banner}

```json
{json.dumps(s_safety, indent=2, sort_keys=True)}
```

## 6. Diagonal dominance

See the per-scale/per-layer `histograms/diag_ratio_*.svg` and `histograms/cka_off_*.svg` files, including the scale-512 control. Exact `offdiag_warning` incidence is in `histograms/offdiag_warning_summary.csv`. No automatic exclusion uses these diagnostics.

## 7. Candidate threshold table

{threshold_md}
`equal_prior_operational_precision` is source-domain identification precision under equal Code/Wiki priors. Code can contain genuine anchors, so it is not ground-truth semantic precision.

Condition-specific consensus is primary: `B 7/8 AND T 7/8 AND rel-L2 7/8 AND |log-r| 7/8`, with the specified 6/6 exception. Same-layer consensus is diagnostic only:

{same_md}
Same-layer Jaccard below 0.9 requires human review. Flagged rows: {len(manual)}.

## 8. Selector comparison

{pair_md}
Primary legacy thresholds are Code-calibration top-1% per layer. `legacy_in_split_reproduction` is emitted separately only to reproduce the earlier in-split construction. Exclusive-set profiles are in `selector_comparison/exclusive_set_profiles.json`; deterministic context candidates are in `context_samples/`. Each context row keeps `worst_chunk_id_*` with `s_at_worst_cka_*` and separately keeps `worst_s_chunk_id_*`, so the minimum-CKA and minimum-s provenance cannot be conflated.

## 9. Routing preservation

```json
{json.dumps(routing, indent=2, sort_keys=True)}
```

The three routing quantities remain distinct: top-4 overlap, full 16-way old-expert probability mass change, and selected-top-4 old-expert mass change.

## 10. Decisions intentionally deferred

- Select one 95/97/99 candidate bundle only after inspecting selection outputs.
- Decide whether an observed CKA valley deserves a separate candidate; none was selected automatically.
- Decide whether membership should be promoted from purity diagnostic to a GT condition.
- Review any condition-specific versus same-layer Jaccard below 0.9.
- Record the final choice before opening the sealed test artifact. REPORT v1 intentionally omits its raw path and all test counts.
- If the threshold changes after test inspection, record test contamination explicitly; do not report it as an untouched test.

REPORT v1 deliberately contains no test-derived selector count, recall, coverage, or precision.
"""
    _atomic_text(output_root / "REPORT.md", text)


def _write_analysis_input_binding(
    output_root: Path,
    input_inventory: Mapping[str, Any],
    raw_quantiles: Path,
    prepared_provenance: Mapping[str, Any],
    *,
    stale_prior_outputs_detected: bool,
) -> dict[str, Any]:
    inventory_path = output_root / "input_parquet_inventory.json"
    raw_manifest = raw_quantiles.with_name("raw_unit_quantile_accumulator_manifest.json")
    raw_payload = json.loads(raw_manifest.read_text(encoding="utf-8"))
    inventory_digest = str(input_inventory["inventory_digest_sha256"])
    if raw_payload.get("input_inventory_digest_sha256") != inventory_digest:
        raise RuntimeError("raw quantile manifest is not bound to the current Parquet inventory")
    outputs = {}
    for label, path in (
        ("analysis_config", output_root / "analysis_config.json"),
        ("report_v1", output_root / "REPORT.md"),
        ("report_validation_evidence", output_root / "report_validation_evidence.json"),
        ("selector_assignments", output_root / "selector_comparison" / "code_selection_assignments.parquet"),
        ("sealed_test_manifest", output_root / "sealed_test" / "manifest.json"),
    ):
        if not path.is_file():
            raise RuntimeError(f"cannot bind missing analysis output: {path}")
        outputs[label] = {
            "relative_path": str(path.relative_to(output_root)),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    payload = {
        "schema": ANALYSIS_BINDING_SCHEMA,
        "status": "BOUND_COMPLETE",
        "input_inventory": {
            "relative_path": inventory_path.name,
            "file_sha256": _sha256(inventory_path),
            "inventory_digest_sha256": inventory_digest,
            "token_rows": input_inventory["totals"]["token_metrics"]["rows"],
            "chunk_rows": input_inventory["totals"]["chunk_metrics"]["rows"],
            "sealed_test_raw_bound": "sealed_test_token_metrics" in input_inventory["totals"],
            "sealed_test_token_rows": input_inventory["totals"].get(
                "sealed_test_token_metrics", {}
            ).get("rows", 0),
            "sealed_test_chunk_rows": input_inventory["totals"].get(
                "sealed_test_chunk_metrics", {}
            ).get("rows", 0),
        },
        "raw_quantiles": {
            "path": str(raw_quantiles),
            "sha256": _sha256(raw_quantiles),
            "manifest_path": str(raw_manifest),
            "manifest_sha256": _sha256(raw_manifest),
            "input_inventory_digest_sha256": raw_payload.get("input_inventory_digest_sha256"),
        },
        "prepared_input": dict(prepared_provenance),
        "analysis_outputs": outputs,
        "stale_prior_outputs_detected_and_rebuilt": bool(stale_prior_outputs_detected),
        "verification_contract": (
            "Recompute current token/chunk Parquet inventory digest and require equality with "
            "input_inventory.inventory_digest_sha256; require the current prepared config file/content "
            "hash and compact source/checkpoint identities to match prepared_input; then verify every "
            "listed output SHA256."
        ),
    }
    _atomic_json(output_root / "analysis_input_binding.json", payload)
    return payload


def run_analysis(
    artifact_root: Path,
    output_root: Path | None = None,
    *,
    token_metrics: Path | None = None,
    chunk_metrics: Path | None = None,
    raw_quantiles: Path | None = None,
    bins: int = DEFAULT_BINS,
    seed: int = DEFAULT_SEED,
    batch_size: int = DEFAULT_BATCH_SIZE,
    keep_work: bool = False,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    output_root = (output_root or artifact_root).resolve()
    token_metrics = token_metrics or artifact_root / "token_metrics"
    chunk_metrics = chunk_metrics or artifact_root / "chunk_metrics"
    sealed_token_metrics = artifact_root / "sealed_test" / "raw" / "token_metrics"
    sealed_chunk_metrics = artifact_root / "sealed_test" / "raw" / "chunk_metrics"
    raw_quantiles = raw_quantiles or artifact_root / "threshold_tables" / "raw_unit_quantiles.parquet"
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("threshold_tables", "selector_comparison", "context_samples", "histograms", "sealed_test"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)

    input_inventory = _input_parquet_inventory(
        artifact_root,
        token_metrics,
        chunk_metrics,
        sealed_token_metrics,
        sealed_chunk_metrics,
    )
    prepared_provenance = _prepared_input_provenance(artifact_root)
    inventory_path = output_root / "input_parquet_inventory.json"
    prior_binding_path = output_root / "analysis_input_binding.json"
    prior_inventory_digest = None
    if prior_binding_path.is_file():
        try:
            prior_inventory_digest = json.loads(
                prior_binding_path.read_text(encoding="utf-8")
            ).get("input_inventory", {}).get("inventory_digest_sha256")
        except (OSError, json.JSONDecodeError):
            prior_inventory_digest = "UNREADABLE"
    stale_prior_outputs_detected = (
        prior_inventory_digest is not None
        and prior_inventory_digest != input_inventory["inventory_digest_sha256"]
    )
    _atomic_json(inventory_path, input_inventory)

    # Keep sealed test shards out of every calibration/selection scanner by
    # construction, not merely by a split predicate.  Only `_sealed_test`
    # receives the isolated sealed reader below.
    reader = ParquetArtifactReader(token_metrics, chunk_metrics, batch_size)
    sealed_raw_available = bool(_parquet_paths(sealed_token_metrics))
    sealed_reader = (
        ParquetArtifactReader(sealed_token_metrics, sealed_chunk_metrics, batch_size)
        if sealed_raw_available
        else reader  # backward-compatible synthetic/legacy artifact layout
    )
    missing_diag = {"worst_diag_ratio_128", "worst_diag_ratio_256"} - reader.token_columns
    if missing_diag:
        raise RuntimeError(
            "production selector diagnostics require per-token worst-chunk diag_ratio lists; "
            f"missing {sorted(missing_diag)}. Emit both fields from the runner rather than reporting diag_ratio unavailable."
        )
    required_context_provenance = {
        "s_at_worst_cka_128",
        "s_at_worst_cka_256",
        "worst_s_chunk_id_128",
        "worst_s_chunk_id_256",
    }
    missing_context_provenance = required_context_provenance - reader.token_columns
    if missing_context_provenance:
        raise RuntimeError(
            "production context audit requires distinct worst-CKA and minimum-s provenance; "
            f"missing {sorted(missing_context_provenance)}"
        )
    raw_quantiles_rebuilt = not _raw_quantiles_bound_to_inventory(
        raw_quantiles, input_inventory["inventory_digest_sha256"]
    )
    if raw_quantiles_rebuilt:
        build_raw_unit_quantiles_exact(
            reader,
            raw_quantiles,
            work_root=output_root / ".raw_unit_quantile_work",
            input_inventory_digest=input_inventory["inventory_digest_sha256"],
        )
    if not _raw_quantiles_bound_to_inventory(
        raw_quantiles, input_inventory["inventory_digest_sha256"]
    ):
        raise RuntimeError("raw quantiles failed current-input binding verification after build")
    quantiles = RawQuantileTable(raw_quantiles)
    bundles = quantiles.build_bundles()
    legacy_primary = quantiles.legacy_cosine("calibration")
    legacy_repro = quantiles.legacy_cosine("selection")

    analysis_config = {
        "schema": SCHEMA,
        "artifact_root": str(artifact_root),
        "output_root": str(output_root),
        "token_metrics": str(token_metrics),
        "chunk_metrics": str(chunk_metrics),
        "sealed_test_raw_reader_isolated": sealed_raw_available,
        "raw_quantiles": str(raw_quantiles),
        "input_inventory_file": str(inventory_path),
        "input_inventory_digest_sha256": input_inventory["inventory_digest_sha256"],
        "prepared_input_provenance": prepared_provenance,
        "raw_quantiles_rebuilt_for_current_inventory": raw_quantiles_rebuilt,
        "seed": seed,
        "layers": list(LAYERS),
        "scales_used_for_gt": list(SCALES),
        "chunk_diagnostic_scales": list(CHUNK_DIAGNOSTIC_SCALES),
        "scale_512_role": "whole-window CKA/diag/null diagnostic only; never thresholded",
        "context_provenance_fields": sorted(required_context_provenance),
        "histogram_bins": max(200, int(bins)),
        "consensus": "condition-specific >=7/8, except n_valid=6 requires 6/6; both scales AND",
        "same_layer_consensus": "diagnostic only; manual review when Jaccard < 0.9",
        "tail_policy": "fixed-stride uncovered tokens are ineligible; no right-aligned chunks",
        "test_policy": "metrics saved sealed; excluded from REPORT v1; one opening after human threshold lock",
        "candidate_thresholds": {str(level): bundle.serializable() for level, bundle in bundles.items()},
        "legacy_primary_thresholds": legacy_primary.tolist(),
        "legacy_selection_reproduction_thresholds": legacy_repro.tolist(),
        "matched_random_pool": "Code selection tokens eligible for all B/T/M95 measurements",
    }
    _atomic_json(output_root / "analysis_config.json", analysis_config)

    work_dir = output_root / ".postprocess_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    hist_bank = HistogramBank(bins, seed)
    random_collector = RandomEligibleCollector(work_dir)
    first_pass, cka_m_count = _first_pass(
        reader,
        bundles,
        legacy_primary,
        legacy_repro,
        hist_bank,
        random_collector,
        seed,
    )
    cutoff, ties = random_collector.cutoff(cka_m_count)
    second_summaries, routing, profiles = _second_pass(
        reader,
        output_root,
        bundles,
        legacy_primary,
        legacy_repro,
        hist_bank,
        cutoff,
        ties,
        seed,
    )
    context_enrichment = _enrich_context_samples(artifact_root, output_root)
    if second_summaries["code"].counts.get("matched_random", 0) != cka_m_count:
        raise AssertionError(
            "matched-random count mismatch: "
            f"random={second_summaries['code'].counts.get('matched_random', 0)} CKA+M={cka_m_count}"
        )
    _write_histogram_bank(hist_bank, output_root / "histograms")
    threshold_rows = _threshold_rows(first_pass)
    same_rows = _same_layer_rows(first_pass)
    pairwise_rows = _pairwise_rows(second_summaries["code"])
    _write_csv(output_root / "threshold_tables" / "candidate_bundles.csv", threshold_rows)
    _atomic_text(
        output_root / "threshold_tables" / "candidate_bundles.md",
        _markdown_table(threshold_rows, tuple(threshold_rows[0].keys()) if threshold_rows else ()),
    )
    _write_csv(output_root / "threshold_tables" / "same_layer_diagnostic.csv", same_rows)
    _write_csv(output_root / "selector_comparison" / "pairwise_jaccard.csv", pairwise_rows)
    _atomic_text(
        output_root / "selector_comparison" / "pairwise_jaccard.md",
        _markdown_table(pairwise_rows, tuple(pairwise_rows[0].keys()) if pairwise_rows else ()),
    )
    _atomic_json(
        output_root / "selector_comparison" / "selector_counts.json",
        {
            domain: {"total": summary.total, "counts": summary.counts}
            for domain, summary in second_summaries.items()
        },
    )
    chunk_summary = _analyze_chunks(reader, output_root, bins, seed, bundles)
    sealed_path = _sealed_test(sealed_reader, output_root, bundles, legacy_primary, legacy_repro)
    windows = _window_summary(artifact_root)
    _report(
        artifact_root,
        output_root,
        threshold_rows,
        same_rows,
        pairwise_rows,
        chunk_summary,
        routing,
        windows,
        input_inventory,
        prepared_provenance,
    )
    binding = _write_analysis_input_binding(
        output_root,
        input_inventory,
        raw_quantiles,
        prepared_provenance,
        stale_prior_outputs_detected=stale_prior_outputs_detected,
    )
    if not keep_work:
        shutil.rmtree(work_dir)
    result = {
        "schema": SCHEMA,
        "output_root": str(output_root),
        "threshold_rows": threshold_rows,
        "same_layer_rows": same_rows,
        "pairwise_rows": pairwise_rows,
        "matched_random_count": cka_m_count,
        "context_enrichment": context_enrichment,
        "input_inventory_digest_sha256": input_inventory["inventory_digest_sha256"],
        "prepared_input_config_content_sha256": prepared_provenance.get(
            "config_content_sha256"
        ),
        "analysis_input_binding": str(output_root / "analysis_input_binding.json"),
        "analysis_input_binding_status": binding["status"],
        "raw_quantiles_rebuilt_for_current_inventory": raw_quantiles_rebuilt,
        "sealed_test_path": str(sealed_path),
        "report": str(output_root / "REPORT.md"),
    }
    _atomic_json(output_root / "postprocess_summary.json", result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path, help="CKA pilot root containing scalar artifacts")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--token-metrics", type=Path, default=None)
    parser.add_argument("--chunk-metrics", type=Path, default=None)
    parser.add_argument("--raw-quantiles", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_analysis(
        args.artifact_root,
        args.output_root,
        token_metrics=args.token_metrics,
        chunk_metrics=args.chunk_metrics,
        raw_quantiles=args.raw_quantiles,
        bins=args.bins,
        seed=args.seed,
        batch_size=args.batch_size,
        keep_work=args.keep_work,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
