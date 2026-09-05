"""Attach a contextual old-like pseudo-GT mask to GPTDataset samples.

The GT artifact is intentionally indexed by the *outer* dataset index.  This
keeps the mapping correct when Megatron's sampler shards indices over data
parallel ranks, shuffles them, or resumes from ``consumed_train_samples``.
The packed masks remain memory-mapped; only the requested 64-byte row is
unpacked for each 512-token sample.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy
import torch


OLD_LIKE_GT_SCHEMA = "old_like_gt_l2_l9_all_code_top1_raw_v1"
OLD_LIKE_TASK_GT_SCHEMA = "old_like_gt_l2_l9_all_task_top1_raw_v1"
# CKA-selected occurrences keep their original document-bounded window, so the
# outer index is the contextual miniset's own sample index rather than a
# GPTDataset index over the full task corpus.  The mask layout is identical.
CKA_GT_CONTEXTUAL_WINDOW_SCHEMA = "cka_gt_contextual_window_miniset_v1"
# The random control writes the identical mask layout; only the selector that
# chose the positions differs, so it must be accepted on the same path.
MATCHED_RANDOM_GT_WINDOW_SCHEMA = "matched_random_gt_window_miniset_v1"
CKA_GT_TOKEN_PACK_SCHEMA = "cka_gt_token_pack_miniset_v1"
OLD_LIKE_GT_SCHEMAS = frozenset((
    OLD_LIKE_GT_SCHEMA, OLD_LIKE_TASK_GT_SCHEMA,
    CKA_GT_CONTEXTUAL_WINDOW_SCHEMA, MATCHED_RANDOM_GT_WINDOW_SCHEMA,
    CKA_GT_TOKEN_PACK_SCHEMA,
))
OLD_LIKE_REPLAY_SUBSET_SCHEMA = "old_like_gt_positive_sample_subset_v1"
OLD_LIKE_REPLAY_SUBSET_FILE = "replay_positive_sample_ids.npy"
OLD_LIKE_REPLAY_SUBSET_METADATA = "replay_subset_metadata.json"
OLD_LIKE_OCCURRENCE_SUBSET_SCHEMA = "old_like_gt_token_occurrence_subset_v1"
OLD_LIKE_OCCURRENCE_SAMPLE_FILE = "replay_occurrence_sample_ids.npy"
OLD_LIKE_OCCURRENCE_POSITION_FILE = "replay_occurrence_positions.npy"
OLD_LIKE_OCCURRENCE_METADATA = "replay_occurrence_metadata.json"
_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"old-like GT metadata does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read old-like GT metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"old-like GT metadata must be a JSON object: {path}")
    return payload


def _normalize_data_paths(paths: Optional[Iterable[Union[str, os.PathLike]]]) -> set[str]:
    """Return path-like entries, ignoring numeric blend weights."""

    normalized: set[str] = set()
    if isinstance(paths, (str, os.PathLike)):
        paths = (paths,)
    for value in paths or ():
        try:
            text = os.fspath(value)
        except TypeError:
            # Programmatic configs may store blend weights as floats rather
            # than argparse strings.
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid runtime dataset path entry: {value!r}") from exc
            continue
        try:
            float(text)
        except ValueError:
            normalized.add(os.path.realpath(text))
    return normalized


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[key] = value
    return value


class OldLikeGTDataset(torch.utils.data.Dataset):
    """Dataset wrapper that appends ``old_like_mask`` and its outer index.

    Args:
        dataset: The already-built train dataset. Its outer index must be the
            same GPTDataset sample axis used to construct the GT artifact.
        gt_root: Directory containing root ``metadata.json`` and partitioned
            ``rank_NNN/old_like_gt_packed.npy`` files.
        expected_sequence_length: Runtime sequence length to validate.
        expected_seed: Runtime dataset seed. Validated when the GT metadata (or
            its extraction source metadata) records it.
        expected_data_paths: Runtime data paths. Validated when the extraction
            source metadata records ``code_data_path``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        gt_root: Union[str, os.PathLike],
        *,
        expected_sequence_length: Optional[int] = None,
        expected_seed: Optional[int] = None,
        expected_data_paths: Optional[Sequence[Union[str, os.PathLike]]] = None,
        replay_subset: bool = False,
        replay_unit: str = "positive_sequence",
        virtual_length: Optional[int] = None,
    ) -> None:
        if dataset is None:
            raise ValueError("old-like GT can only wrap a non-null train dataset")
        self.dataset = dataset
        self.gt_root = Path(gt_root).expanduser().resolve()
        self.metadata_path = self.gt_root / "metadata.json"
        self.metadata = _load_json(self.metadata_path)

        self._validate_root_metadata(expected_sequence_length)
        self.gt_schema = self.metadata["schema"]
        self.sequence_length = int(self.metadata["sequence_length"])
        self.packed_bytes = (self.sequence_length + 7) // 8
        self.total_samples = int(self.metadata["total_samples"])
        self.replay_subset = bool(replay_subset)
        if replay_unit not in ("positive_sequence", "token_occurrence"):
            raise ValueError(f"unsupported old-like replay unit: {replay_unit!r}")
        self.replay_unit = replay_unit

        dataset_samples = len(self.dataset)
        if dataset_samples < self.total_samples:
            raise ValueError(
                "old-like GT source sample count exceeds the underlying train dataset: "
                f"GT={self.total_samples}, dataset={dataset_samples}, "
                f"replay_subset={self.replay_subset}"
            )
        self.virtual_length = (
            dataset_samples if virtual_length is None else int(virtual_length)
        )
        if self.virtual_length <= 0:
            raise ValueError(
                f"old-like GT virtual length must be positive, got {self.virtual_length}"
            )

        self._partition_starts: list[int] = []
        self._partition_ends: list[int] = []
        self._partition_paths: list[Path] = []
        self._validate_partitions()
        self._validate_dataset_identity(expected_seed, expected_data_paths)

        # DataLoader worker processes open their own mmap handles lazily. This
        # also prevents a parent process's mmap objects from being pickled.
        self._mmap_pid: Optional[int] = None
        self._mmap_arrays: Optional[list[numpy.ndarray]] = None
        self._replay_source_sample_ids: Optional[numpy.ndarray] = None
        self._replay_occurrence_positions: Optional[numpy.ndarray] = None
        if self.replay_subset:
            if self.replay_unit == "positive_sequence":
                self._replay_source_sample_ids = self._load_replay_subset()
            else:
                (
                    self._replay_source_sample_ids,
                    self._replay_occurrence_positions,
                ) = self._load_occurrence_subset()

    def _load_replay_subset(self) -> numpy.ndarray:
        """Load the fixed GT-positive contextual-sample replay subset.

        The wrapped dataset keeps the full virtual training length so the
        replay branch consumes exactly the same number of sequence slots as
        the primary branch.  Virtual replay index ``i`` maps to the fixed,
        shuffled positive-sample list at ``i % subset_size``; this is the same
        repeated-miniset semantics used by the existing router-FT experiments.
        Only GT-positive token positions in the returned source sequence are
        supervised by the objective mask.
        """

        metadata_path = self.gt_root / OLD_LIKE_REPLAY_SUBSET_METADATA
        subset_path = self.gt_root / OLD_LIKE_REPLAY_SUBSET_FILE
        subset_metadata = _load_json(metadata_path)
        expected = {
            "schema": OLD_LIKE_REPLAY_SUBSET_SCHEMA,
            "complete": True,
            "source_gt_schema": self.gt_schema,
            "source_gt_config_sha256": self.metadata.get("config_sha256"),
            "source_total_samples": self.total_samples,
            "subset_file": OLD_LIKE_REPLAY_SUBSET_FILE,
            "dtype": "int32",
        }
        for key, value in expected.items():
            if subset_metadata.get(key) != value:
                raise ValueError(
                    f"old-like replay subset metadata {key}="
                    f"{subset_metadata.get(key)!r}, expected {value!r}"
                )

        try:
            source_ids = numpy.load(subset_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"could not mmap old-like replay subset {subset_path}: {exc}"
            ) from exc
        expected_count = subset_metadata.get("positive_sample_count")
        if (
            source_ids.dtype != numpy.int32
            or source_ids.ndim != 1
            or source_ids.shape != (expected_count,)
            or source_ids.size <= 0
        ):
            raise ValueError(
                "old-like replay subset shape/dtype mismatch: "
                f"got {source_ids.shape}/{source_ids.dtype}, expected "
                f"({expected_count},)/int32"
            )
        if int(source_ids.min()) < 0 or int(source_ids.max()) >= self.total_samples:
            raise ValueError("old-like replay subset contains an out-of-range sample ID")
        if numpy.unique(source_ids).size != source_ids.size:
            raise ValueError("old-like replay subset contains duplicate source sample IDs")
        recorded_sha = subset_metadata.get("subset_sha256")
        if not isinstance(recorded_sha, str) or _file_sha256(subset_path) != recorded_sha:
            raise ValueError("old-like replay subset SHA256 mismatch")

        # A deterministic sentinel check ties the subset IDs back to the
        # packed occurrence masks without scanning every row on every rank.
        sentinel_positions = sorted({0, source_ids.size // 2, source_ids.size - 1})
        for position in sentinel_positions:
            source_id = int(source_ids[position])
            if not bool(self._mask_for_index(source_id).any()):
                raise ValueError(
                    "old-like replay subset contains a non-positive sentinel: "
                    f"position={position}, source_sample_id={source_id}"
                )
        return source_ids

    def _load_occurrence_subset(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Load one replay item per contextual GT token occurrence."""

        metadata_path = self.gt_root / OLD_LIKE_OCCURRENCE_METADATA
        sample_path = self.gt_root / OLD_LIKE_OCCURRENCE_SAMPLE_FILE
        position_path = self.gt_root / OLD_LIKE_OCCURRENCE_POSITION_FILE
        metadata = _load_json(metadata_path)
        expected = {
            "schema": OLD_LIKE_OCCURRENCE_SUBSET_SCHEMA,
            "complete": True,
            "source_gt_schema": self.gt_schema,
            "source_gt_config_sha256": self.metadata.get("config_sha256"),
            "source_total_samples": self.total_samples,
            "sample_file": OLD_LIKE_OCCURRENCE_SAMPLE_FILE,
            "position_file": OLD_LIKE_OCCURRENCE_POSITION_FILE,
            "sample_dtype": "int32",
            "position_dtype": "uint16",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"old-like occurrence metadata {key}={metadata.get(key)!r}, "
                    f"expected {value!r}"
                )
        try:
            sample_ids = numpy.load(sample_path, mmap_mode="r", allow_pickle=False)
            positions = numpy.load(position_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not mmap old-like occurrence subset: {exc}") from exc
        count = metadata.get("occurrence_count")
        if (
            sample_ids.shape != (count,)
            or positions.shape != (count,)
            or sample_ids.dtype != numpy.int32
            or positions.dtype != numpy.uint16
            or sample_ids.size <= 0
        ):
            raise ValueError(
                "old-like occurrence subset shape/dtype mismatch: "
                f"samples={sample_ids.shape}/{sample_ids.dtype}, "
                f"positions={positions.shape}/{positions.dtype}, count={count}"
            )
        if int(sample_ids.min()) < 0 or int(sample_ids.max()) >= self.total_samples:
            raise ValueError("old-like occurrence subset contains an out-of-range sample ID")
        if int(positions.max()) >= self.sequence_length:
            raise ValueError("old-like occurrence subset contains an out-of-range position")
        if _file_sha256(sample_path) != metadata.get("sample_sha256"):
            raise ValueError("old-like occurrence sample-ID SHA256 mismatch")
        if _file_sha256(position_path) != metadata.get("position_sha256"):
            raise ValueError("old-like occurrence position SHA256 mismatch")

        sentinels = sorted({0, sample_ids.size // 2, sample_ids.size - 1})
        for occurrence_index in sentinels:
            source_id = int(sample_ids[occurrence_index])
            position = int(positions[occurrence_index])
            if not bool(self._mask_for_index(source_id)[position]):
                raise ValueError(
                    "old-like occurrence sentinel is not GT-positive: "
                    f"occurrence={occurrence_index}, sample={source_id}, position={position}"
                )
        return sample_ids, positions

    def _validate_root_metadata(self, expected_sequence_length: Optional[int]) -> None:
        metadata = self.metadata
        if metadata.get("schema") not in OLD_LIKE_GT_SCHEMAS:
            raise ValueError(
                f"unsupported old-like GT schema {metadata.get('schema')!r}; "
                f"expected one of {sorted(OLD_LIKE_GT_SCHEMAS)!r}"
            )
        if metadata.get("complete") is not True:
            raise ValueError(f"old-like GT artifact is not complete: {self.metadata_path}")

        total_samples = metadata.get("total_samples")
        sequence_length = metadata.get("sequence_length")
        if not isinstance(total_samples, int) or total_samples <= 0:
            raise ValueError(f"invalid old-like GT total_samples: {total_samples!r}")
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            raise ValueError(f"invalid old-like GT sequence_length: {sequence_length!r}")
        if expected_sequence_length is not None and sequence_length != int(
            expected_sequence_length
        ):
            raise ValueError(
                "old-like GT sequence length mismatch: "
                f"GT={sequence_length}, runtime={expected_sequence_length}"
            )
        if sequence_length % 8 != 0:
            raise ValueError(
                "old-like GT sequence length must be byte-packable (a multiple of 8), "
                f"got {sequence_length}"
            )
        if not isinstance(metadata.get("partitions"), list) or not metadata["partitions"]:
            raise ValueError("old-like GT root metadata has no partitions")

    def _validate_partitions(self) -> None:
        expected_start = 0
        root_config_hash = self.metadata.get("config_sha256")
        for partition in self.metadata["partitions"]:
            if not isinstance(partition, dict):
                raise ValueError("old-like GT partition metadata must be JSON objects")
            rank_name = partition.get("rank")
            if not isinstance(rank_name, str) or Path(rank_name).name != rank_name:
                raise ValueError(f"invalid old-like GT partition name: {rank_name!r}")
            start = partition.get("partition_start_sample")
            count = partition.get("partition_samples")
            mask_file = partition.get("mask_file")
            if start != expected_start or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    "old-like GT partitions are not positive, contiguous outer-index ranges: "
                    f"expected start {expected_start}, got start={start!r}, count={count!r}"
                )
            if not isinstance(mask_file, str) or Path(mask_file).name != mask_file:
                raise ValueError(f"invalid old-like GT mask filename: {mask_file!r}")

            rank_dir = self.gt_root / rank_name
            rank_metadata_path = rank_dir / "metadata.json"
            rank_metadata = _load_json(rank_metadata_path)
            expected_shape = [count, self.packed_bytes]
            checks = {
                "schema": self.gt_schema,
                "complete": True,
                "partition_start_sample": start,
                "partition_samples": count,
                "sequence_length": self.sequence_length,
                "mask_file": mask_file,
                "mask_shape": expected_shape,
                "mask_dtype": "uint8",
                "bitorder": "little",
            }
            if root_config_hash is not None:
                checks["config_sha256"] = root_config_hash
            for key, expected in checks.items():
                if rank_metadata.get(key) != expected:
                    raise ValueError(
                        f"old-like GT partition {rank_name} has {key}="
                        f"{rank_metadata.get(key)!r}, expected {expected!r}"
                    )

            mask_path = rank_dir / mask_file
            try:
                array = numpy.load(mask_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"could not mmap old-like GT mask {mask_path}: {exc}") from exc
            if array.shape != tuple(expected_shape) or array.dtype != numpy.uint8:
                raise ValueError(
                    f"old-like GT mask {mask_path} has shape/dtype {array.shape}/{array.dtype}, "
                    f"expected {tuple(expected_shape)}/uint8"
                )
            del array

            end = start + count
            self._partition_starts.append(start)
            self._partition_ends.append(end)
            self._partition_paths.append(mask_path)
            expected_start = end

        if expected_start != self.total_samples:
            raise ValueError(
                "old-like GT partitions do not cover total_samples: "
                f"covered={expected_start}, total={self.total_samples}"
            )

    def _source_dataset_metadata(self) -> Optional[Dict[str, Any]]:
        first_rank = self.metadata["partitions"][0]["rank"]
        rank_metadata = _load_json(self.gt_root / first_rank / "metadata.json")
        source_rank_dir = rank_metadata.get("source_rank_dir")
        if not source_rank_dir:
            return None
        source_path = Path(source_rank_dir) / "metadata.json"
        if not source_path.is_file():
            return None
        return _load_json(source_path)

    def _validate_dataset_identity(
        self,
        expected_seed: Optional[int],
        expected_data_paths: Optional[Sequence[Union[str, os.PathLike]]],
    ) -> None:
        # Newer artifacts may copy identity directly into root metadata. The
        # current artifact records it in the paired-extraction rank metadata,
        # so use that as a backwards-compatible fallback.
        identity = self.metadata.get("dataset_identity")
        if identity is not None and not isinstance(identity, dict):
            raise ValueError("old-like GT dataset_identity must be a JSON object")
        identity = dict(identity or {})
        source = self._source_dataset_metadata()
        if source:
            identity.setdefault("seed", source.get("seed"))
            identity.setdefault("sequence_length", source.get("sequence_length"))
            identity.setdefault("total_samples", source.get("total_samples"))
            identity.setdefault("data_path", source.get("code_data_path"))
            identity.setdefault("split", source.get("dataset_split"))

        if identity.get("total_samples") not in (None, self.total_samples):
            raise ValueError(
                "old-like GT dataset identity total_samples disagrees with mask metadata: "
                f"{identity.get('total_samples')} != {self.total_samples}"
            )
        if identity.get("sequence_length") not in (None, self.sequence_length):
            raise ValueError(
                "old-like GT dataset identity sequence length disagrees with mask metadata: "
                f"{identity.get('sequence_length')} != {self.sequence_length}"
            )
        if expected_seed is not None and identity.get("seed") not in (None, int(expected_seed)):
            raise ValueError(
                f"old-like GT seed mismatch: GT={identity.get('seed')}, runtime={expected_seed}"
            )
        if identity.get("split") not in (None, "100,0,0"):
            raise ValueError(
                f"old-like GT was not built from the Code train split: {identity.get('split')!r}"
            )

        runtime_paths = _normalize_data_paths(expected_data_paths)
        recorded_shards = identity.get("shards")
        if recorded_shards is not None:
            if not isinstance(recorded_shards, list) or not recorded_shards:
                raise ValueError("old-like GT dataset_identity shards must be a nonempty list")
            if len(runtime_paths) != len(recorded_shards):
                raise ValueError(
                    "old-like GT shard-count mismatch: "
                    f"GT={len(recorded_shards)}, runtime={len(runtime_paths)}"
                )
            unused_runtime = set(runtime_paths)
            for shard in recorded_shards:
                if not isinstance(shard, dict) or not shard.get("prefix"):
                    raise ValueError(f"invalid old-like GT shard identity: {shard!r}")
                recorded_prefix = os.path.realpath(os.fspath(shard["prefix"]))
                candidates = sorted(
                    unused_runtime,
                    key=lambda value: (
                        os.path.basename(value) != os.path.basename(recorded_prefix),
                        value != recorded_prefix,
                    ),
                )
                matched = None
                reasons = []
                for candidate in candidates:
                    try:
                        self._validate_physical_dataset_identity(candidate, shard)
                    except ValueError as exc:
                        reasons.append(f"{candidate}: {exc}")
                    else:
                        matched = candidate
                        break
                if matched is None:
                    raise ValueError(
                        "no runtime shard matches old-like GT source "
                        f"{recorded_prefix}: " + "; ".join(reasons)
                    )
                unused_runtime.remove(matched)
            if unused_runtime:
                raise ValueError(
                    f"unmatched runtime old-like GT shards: {sorted(unused_runtime)}"
                )
            return

        artifact_path = identity.get("data_path") or identity.get(
            "paired_extraction_dataset_prefix"
        )
        if artifact_path:
            artifact_path = os.path.realpath(os.fspath(artifact_path))

        # Training rsyncs the indexed dataset to per-run NVMe scratch. Permit a
        # relocated prefix only when the artifact carries enough physical
        # identity to prove that the runtime copy is byte-identical.
        physical_identity_recorded = any(
            identity.get(key) is not None
            for key in ("idx_bytes", "bin_bytes", "idx_sha256")
        )
        if runtime_paths:
            if artifact_path in runtime_paths:
                runtime_artifact_path = artifact_path
            elif artifact_path and not physical_identity_recorded:
                raise ValueError(
                    "old-like GT data path mismatch and no physical identity permits "
                    f"relocation: GT={artifact_path}, runtime={sorted(runtime_paths)}"
                )
            else:
                runtime_artifact_path = None
                mismatch_reasons = []
                for candidate in sorted(runtime_paths):
                    try:
                        self._validate_physical_dataset_identity(candidate, identity)
                    except ValueError as exc:
                        mismatch_reasons.append(f"{candidate}: {exc}")
                    else:
                        runtime_artifact_path = candidate
                        break
                if runtime_artifact_path is None:
                    raise ValueError(
                        "no runtime data prefix matches the old-like GT physical identity: "
                        + "; ".join(mismatch_reasons)
                    )
        else:
            runtime_artifact_path = artifact_path

        # If the artifact records physical IndexedDataset identity, validate it
        # against the selected runtime prefix. The .idx digest is small enough
        # to verify at startup and is cached across primary/replay construction.
        if runtime_artifact_path and physical_identity_recorded:
            self._validate_physical_dataset_identity(runtime_artifact_path, identity)

    @staticmethod
    def _validate_physical_dataset_identity(
        dataset_prefix: Union[str, os.PathLike], identity: Dict[str, Any]
    ) -> None:
        dataset_prefix = os.path.realpath(os.fspath(dataset_prefix))
        if not any(
            identity.get(key) is not None
            for key in ("idx_bytes", "bin_bytes", "idx_sha256")
        ):
            raise ValueError("artifact records no .idx/.bin size or hash")
        try:
            for suffix, size_key in ((".idx", "idx_bytes"), (".bin", "bin_bytes")):
                recorded_size = identity.get(size_key)
                if recorded_size is None:
                    continue
                data_file = Path(dataset_prefix + suffix)
                try:
                    actual_size = data_file.stat().st_size
                except OSError as exc:
                    raise ValueError(
                        f"could not stat old-like GT source dataset file {data_file}: {exc}"
                    ) from exc
                if actual_size != int(recorded_size):
                    raise ValueError(
                        f"old-like GT {suffix} size mismatch: GT={recorded_size}, "
                        f"runtime={actual_size} ({data_file})"
                    )
            recorded_idx_sha256 = identity.get("idx_sha256")
            if recorded_idx_sha256 is not None:
                idx_path = Path(dataset_prefix + ".idx")
                try:
                    actual_idx_sha256 = _file_sha256(idx_path)
                except OSError as exc:
                    raise ValueError(
                        f"could not hash old-like GT source dataset index {idx_path}: {exc}"
                    ) from exc
                if actual_idx_sha256 != recorded_idx_sha256:
                    raise ValueError(
                        "old-like GT IndexedDataset .idx SHA256 mismatch: "
                        f"GT={recorded_idx_sha256}, runtime={actual_idx_sha256}"
                    )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"invalid old-like GT physical identity metadata: {exc}") from exc

    def _ensure_mmaps(self) -> list[numpy.ndarray]:
        pid = os.getpid()
        if self._mmap_arrays is None or self._mmap_pid != pid:
            self._mmap_arrays = [
                numpy.load(path, mmap_mode="r", allow_pickle=False)
                for path in self._partition_paths
            ]
            self._mmap_pid = pid
        return self._mmap_arrays

    def _mask_for_index(self, idx: int) -> torch.Tensor:
        partition_index = bisect.bisect_right(self._partition_starts, idx) - 1
        if partition_index < 0 or idx >= self._partition_ends[partition_index]:
            raise IndexError(f"old-like GT outer dataset index is out of range: {idx}")
        local_index = idx - self._partition_starts[partition_index]
        packed = self._ensure_mmaps()[partition_index][local_index]
        unpacked = numpy.unpackbits(
            packed, count=self.sequence_length, bitorder="little"
        ).astype(numpy.bool_, copy=False)
        return torch.from_numpy(unpacked)

    def __len__(self) -> int:
        return self.virtual_length

    def __getitem__(self, idx: Optional[int]) -> Dict[str, Any]:
        if idx is None:
            sample = dict(self.dataset[idx])
            sample["old_like_mask"] = torch.zeros(self.sequence_length, dtype=torch.bool)
            sample["old_like_sample_id"] = torch.tensor(-1, dtype=torch.int64)
            return sample

        outer_index = int(idx)
        if outer_index < 0 or outer_index >= self.virtual_length:
            raise IndexError(
                f"old-like GT outer dataset index {outer_index} is outside "
                f"[0, {self.virtual_length})"
            )
        # The GT is tied to one deterministic exhaustive source epoch.  A
        # longer primary run repeats that exact contextual-sample axis rather
        # than rebuilding a differently shuffled multi-epoch GPTDataset.
        source_index = outer_index % self.total_samples
        if self._replay_source_sample_ids is not None:
            replay_index = outer_index % self._replay_source_sample_ids.size
            source_index = int(self._replay_source_sample_ids[replay_index])
        sample = dict(self.dataset[source_index])
        old_like_mask = self._mask_for_index(source_index)
        if self._replay_occurrence_positions is not None:
            position = int(self._replay_occurrence_positions[replay_index])
            occurrence_mask = torch.zeros_like(old_like_mask)
            occurrence_mask[position] = True
            old_like_mask = occurrence_mask
        sample["old_like_mask"] = old_like_mask
        sample["old_like_sample_id"] = torch.tensor(source_index, dtype=torch.int64)
        return sample

    def __getattr__(self, name: str) -> Any:
        # Preserve useful GPTDataset attributes without inheriting from its
        # concrete type. Avoid delegation while unpickling ``dataset`` itself.
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_mmap_pid"] = None
        state["_mmap_arrays"] = None
        return state


def maybe_wrap_train_dataset_with_old_like_gt(
    train_dataset: Optional[torch.utils.data.Dataset],
    gt_root: Optional[Union[str, os.PathLike]],
    *,
    expected_sequence_length: Optional[int] = None,
    expected_seed: Optional[int] = None,
    expected_data_paths: Optional[Sequence[Union[str, os.PathLike]]] = None,
    replay_subset: bool = False,
    replay_unit: str = "positive_sequence",
    virtual_length: Optional[int] = None,
) -> Optional[torch.utils.data.Dataset]:
    """Wrap only the train dataset when an old-like GT path is configured."""

    # Core datasets are built only on the pipeline endpoint / TP-source ranks;
    # all other ranks receive ``None`` and obtain batches via TP broadcast.
    if train_dataset is None or not gt_root:
        return train_dataset
    return OldLikeGTDataset(
        train_dataset,
        gt_root,
        expected_sequence_length=expected_sequence_length,
        expected_seed=expected_seed,
        expected_data_paths=expected_data_paths,
        replay_subset=replay_subset,
        replay_unit=replay_unit,
        virtual_length=virtual_length,
    )


def old_like_gt_source_sample_count(
    gt_root: Union[str, os.PathLike],
    *,
    expected_sequence_length: Optional[int] = None,
) -> int:
    """Return the immutable contextual source-sample axis recorded by a GT."""

    root = Path(gt_root).expanduser().resolve()
    metadata = _load_json(root / "metadata.json")
    if metadata.get("schema") not in OLD_LIKE_GT_SCHEMAS:
        raise ValueError(
            f"unsupported old-like GT schema {metadata.get('schema')!r}: {root}"
        )
    if metadata.get("complete") is not True:
        raise ValueError(f"old-like GT artifact is incomplete: {root}")
    total_samples = metadata.get("total_samples")
    sequence_length = metadata.get("sequence_length")
    if not isinstance(total_samples, int) or total_samples <= 0:
        raise ValueError(f"invalid old-like GT total_samples: {total_samples!r}")
    if expected_sequence_length is not None and sequence_length != int(
        expected_sequence_length
    ):
        raise ValueError(
            "old-like GT sequence length mismatch: "
            f"GT={sequence_length}, runtime={expected_sequence_length}"
        )
    return total_samples


def old_like_gt_source_blend(
    gt_root: Union[str, os.PathLike],
) -> Tuple[List[str], None]:
    """Return the exact exhaustive shard blend used to assign GT sample IDs.

    Contextual occurrence IDs are positions on the paired-extraction dataset
    axis, not IDs of the runtime task blend.  In particular, the Conversation
    extraction used a path-only (length-proportional, one-epoch) blend whereas
    ordinary training uses explicit equal shard weights.  Reusing the latter
    silently maps a valid ``sample_id`` to a different token sequence.
    """

    root = Path(gt_root).expanduser().resolve()
    metadata = _load_json(root / "metadata.json")
    if metadata.get("schema") not in OLD_LIKE_GT_SCHEMAS:
        raise ValueError(
            f"unsupported old-like GT schema {metadata.get('schema')!r}: {root}"
        )
    if metadata.get("complete") is not True:
        raise ValueError(f"old-like GT artifact is incomplete: {root}")
    identity = metadata.get("dataset_identity")
    if not isinstance(identity, dict) or identity.get("blend_mode") != "exhaustive":
        raise ValueError(
            "old-like contextual GT requires an exhaustive source blend; "
            f"got {None if not isinstance(identity, dict) else identity.get('blend_mode')!r}"
        )
    shards = identity.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"old-like GT has no source shard identity: {root}")
    prefixes: List[str] = []
    for index, shard in enumerate(shards):
        if not isinstance(shard, dict) or not shard.get("prefix"):
            raise ValueError(f"invalid old-like GT source shard {index}: {shard!r}")
        prefix = str(Path(shard["prefix"]).expanduser().resolve())
        for suffix in (".bin", ".idx"):
            if not Path(prefix + suffix).is_file():
                raise FileNotFoundError(
                    f"old-like GT source shard is unavailable: {prefix + suffix}"
                )
        prefixes.append(prefix)
    # ``None`` weights is intentional: BlendedMegatronDatasetBuilder then
    # reproduces the length-proportional exhaustive ordering used at extraction.
    return prefixes, None
