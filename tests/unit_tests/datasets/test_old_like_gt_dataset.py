import hashlib
import json
import pickle
from pathlib import Path

import numpy
import pytest
import torch

from megatron.core.datasets.old_like_gt_dataset import (
    OLD_LIKE_GT_SCHEMA,
    OLD_LIKE_REPLAY_SUBSET_FILE,
    OLD_LIKE_REPLAY_SUBSET_METADATA,
    OLD_LIKE_REPLAY_SUBSET_SCHEMA,
    OLD_LIKE_OCCURRENCE_METADATA,
    OLD_LIKE_OCCURRENCE_POSITION_FILE,
    OLD_LIKE_OCCURRENCE_SAMPLE_FILE,
    OLD_LIKE_OCCURRENCE_SUBSET_SCHEMA,
    OldLikeGTDataset,
    maybe_wrap_train_dataset_with_old_like_gt,
)
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, length, sequence_length):
        self.length = length
        self.sequence_length = sequence_length
        self.marker = "delegated"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        value = 0 if idx is None else int(idx)
        return {
            "tokens": torch.full((self.sequence_length,), value, dtype=torch.int64),
            "labels": torch.full((self.sequence_length,), value + 1, dtype=torch.int64),
            "loss_mask": torch.ones(self.sequence_length, dtype=torch.float32),
            "position_ids": torch.arange(self.sequence_length, dtype=torch.int64),
        }


def _write_gt(root: Path, masks: numpy.ndarray, partition_sizes=(3, 5)) -> Path:
    sequence_length = masks.shape[1]
    packed = numpy.packbits(masks, axis=1, bitorder="little")
    config_hash = "test-config"
    partitions = []
    start = 0
    for rank_index, count in enumerate(partition_sizes):
        rank = f"rank_{rank_index:03d}"
        rank_dir = root / rank
        rank_dir.mkdir(parents=True)
        mask_file = "old_like_gt_packed.npy"
        numpy.save(rank_dir / mask_file, packed[start : start + count])
        rank_metadata = {
            "schema": OLD_LIKE_GT_SCHEMA,
            "complete": True,
            "config_sha256": config_hash,
            "partition_start_sample": start,
            "partition_samples": count,
            "sequence_length": sequence_length,
            "mask_file": mask_file,
            "mask_shape": [count, packed.shape[1]],
            "mask_dtype": "uint8",
            "bitorder": "little",
        }
        (rank_dir / "metadata.json").write_text(json.dumps(rank_metadata))
        partitions.append(
            {
                "rank": rank,
                "partition_start_sample": start,
                "partition_samples": count,
                "mask_file": mask_file,
            }
        )
        start += count

    metadata = {
        "schema": OLD_LIKE_GT_SCHEMA,
        "complete": True,
        "config_sha256": config_hash,
        "total_samples": masks.shape[0],
        "sequence_length": sequence_length,
        "dataset_identity": {
            "seed": 1234,
            "sequence_length": sequence_length,
            "total_samples": masks.shape[0],
            "data_path": "/dataset/code/train",
            "split": "100,0,0",
        },
        "partitions": partitions,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root


@pytest.fixture
def gt_fixture(tmp_path):
    masks = numpy.zeros((8, 16), dtype=numpy.bool_)
    for sample_index in range(8):
        masks[sample_index, sample_index] = True
        masks[sample_index, 15 - sample_index] = True
    root = _write_gt(tmp_path / "gt", masks)
    return root, masks


def test_wrapper_uses_outer_index_across_partitions_and_unpacks_little_endian(gt_fixture):
    root, masks = gt_fixture
    base = _ToyDataset(8, 16)
    wrapped = OldLikeGTDataset(
        base,
        root,
        expected_sequence_length=16,
        expected_seed=1234,
        expected_data_paths=["1.0", "/dataset/code/train"],
    )

    for index in (0, 2, 3, 7):
        sample = wrapped[index]
        assert sample["old_like_sample_id"].dtype == torch.int64
        assert sample["old_like_sample_id"].item() == index
        assert sample["old_like_mask"].dtype == torch.bool
        numpy.testing.assert_array_equal(sample["old_like_mask"].numpy(), masks[index])
        assert sample["tokens"][0].item() == index

    assert wrapped.marker == "delegated"
    assert isinstance(wrapped._mmap_arrays[0], numpy.memmap)


def test_primary_virtual_axis_repeats_exact_context_and_next_token_label(gt_fixture):
    root, masks = gt_fixture
    wrapped = OldLikeGTDataset(
        _ToyDataset(8, 16),
        root,
        expected_sequence_length=16,
        virtual_length=13,
    )

    assert len(wrapped) == 13
    repeated = wrapped[10]
    assert repeated["old_like_sample_id"].item() == 2
    assert repeated["tokens"][0].item() == 2
    assert repeated["labels"][0].item() == 3
    numpy.testing.assert_array_equal(repeated["old_like_mask"].numpy(), masks[2])


def test_padding_index_has_no_gt_and_negative_sample_id(gt_fixture):
    root, _ = gt_fixture
    wrapped = OldLikeGTDataset(_ToyDataset(8, 16), root, expected_sequence_length=16)
    sample = wrapped[None]
    assert not sample["old_like_mask"].any()
    assert sample["old_like_sample_id"].item() == -1


def test_sampler_dp_sharding_and_resume_preserve_outer_sample_ids(gt_fixture):
    root, masks = gt_fixture
    wrapped = OldLikeGTDataset(_ToyDataset(8, 16), root, expected_sequence_length=16)
    samplers = [
        MegatronPretrainingSampler(
            total_samples=8,
            consumed_samples=4,
            micro_batch_size=1,
            data_parallel_rank=rank,
            data_parallel_size=2,
        )
        for rank in (0, 1)
    ]
    sampled_indices = [
        [batch[0] for batch in sampler]
        for sampler in samplers
    ]
    assert sampled_indices == [[4, 6], [5, 7]]
    for indices in sampled_indices:
        for index in indices:
            sample = wrapped[index]
            assert sample["old_like_sample_id"].item() == index
            numpy.testing.assert_array_equal(sample["old_like_mask"].numpy(), masks[index])


def test_wrapper_reopens_mmaps_after_pickling(gt_fixture):
    root, masks = gt_fixture
    wrapped = OldLikeGTDataset(_ToyDataset(8, 16), root, expected_sequence_length=16)
    _ = wrapped[0]
    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored._mmap_arrays is None
    numpy.testing.assert_array_equal(restored[7]["old_like_mask"].numpy(), masks[7])


def test_wrapper_rejects_length_sequence_and_identity_mismatches(gt_fixture):
    root, _ = gt_fixture
    with pytest.raises(ValueError, match="sample count"):
        OldLikeGTDataset(_ToyDataset(7, 16), root, expected_sequence_length=16)
    with pytest.raises(ValueError, match="sequence length mismatch"):
        OldLikeGTDataset(_ToyDataset(8, 16), root, expected_sequence_length=512)
    with pytest.raises(ValueError, match="seed mismatch"):
        OldLikeGTDataset(
            _ToyDataset(8, 16), root, expected_sequence_length=16, expected_seed=7
        )
    with pytest.raises(ValueError, match="data path mismatch"):
        OldLikeGTDataset(
            _ToyDataset(8, 16),
            root,
            expected_sequence_length=16,
            expected_data_paths=["/dataset/wiki/train"],
        )


def test_maybe_wrap_only_changes_dataset_when_path_is_set(gt_fixture):
    root, _ = gt_fixture
    base = _ToyDataset(8, 16)
    assert maybe_wrap_train_dataset_with_old_like_gt(base, None) is base
    assert maybe_wrap_train_dataset_with_old_like_gt(None, root) is None
    assert isinstance(
        maybe_wrap_train_dataset_with_old_like_gt(
            base, root, expected_sequence_length=16
        ),
        OldLikeGTDataset,
    )


def test_replay_subset_cycles_fixed_positive_contexts_and_keeps_gt_masks(gt_fixture):
    root, masks = gt_fixture
    source_ids = numpy.asarray([7, 2, 5], dtype=numpy.int32)
    subset_path = root / OLD_LIKE_REPLAY_SUBSET_FILE
    numpy.save(subset_path, source_ids)
    subset_sha = hashlib.sha256(subset_path.read_bytes()).hexdigest()
    (root / OLD_LIKE_REPLAY_SUBSET_METADATA).write_text(
        json.dumps(
            {
                "schema": OLD_LIKE_REPLAY_SUBSET_SCHEMA,
                "complete": True,
                "source_gt_schema": OLD_LIKE_GT_SCHEMA,
                "source_gt_config_sha256": "test-config",
                "source_total_samples": 8,
                "positive_sample_count": 3,
                "subset_file": OLD_LIKE_REPLAY_SUBSET_FILE,
                "dtype": "int32",
                "subset_sha256": subset_sha,
            }
        )
    )

    wrapped = OldLikeGTDataset(
        _ToyDataset(8, 16), root, expected_sequence_length=16, replay_subset=True
    )
    assert len(wrapped) == 8
    expected_source_ids = [7, 2, 5, 7, 2, 5, 7, 2]
    for virtual_index, source_id in enumerate(expected_source_ids):
        sample = wrapped[virtual_index]
        assert sample["old_like_sample_id"].item() == source_id
        assert sample["tokens"][0].item() == source_id
        numpy.testing.assert_array_equal(
            sample["old_like_mask"].numpy(), masks[source_id]
        )


def test_token_occurrence_subset_cycles_one_gt_position_per_item(gt_fixture):
    root, masks = gt_fixture
    sample_ids = numpy.asarray([7, 2, 7], dtype=numpy.int32)
    positions = numpy.asarray([7, 2, 8], dtype=numpy.uint16)
    sample_path = root / OLD_LIKE_OCCURRENCE_SAMPLE_FILE
    position_path = root / OLD_LIKE_OCCURRENCE_POSITION_FILE
    numpy.save(sample_path, sample_ids)
    numpy.save(position_path, positions)
    (root / OLD_LIKE_OCCURRENCE_METADATA).write_text(
        json.dumps(
            {
                "schema": OLD_LIKE_OCCURRENCE_SUBSET_SCHEMA,
                "complete": True,
                "source_gt_schema": OLD_LIKE_GT_SCHEMA,
                "source_gt_config_sha256": "test-config",
                "source_total_samples": 8,
                "occurrence_count": 3,
                "sample_file": OLD_LIKE_OCCURRENCE_SAMPLE_FILE,
                "position_file": OLD_LIKE_OCCURRENCE_POSITION_FILE,
                "sample_dtype": "int32",
                "position_dtype": "uint16",
                "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                "position_sha256": hashlib.sha256(position_path.read_bytes()).hexdigest(),
            }
        )
    )

    wrapped = OldLikeGTDataset(
        _ToyDataset(8, 16),
        root,
        expected_sequence_length=16,
        replay_subset=True,
        replay_unit="token_occurrence",
        virtual_length=4,
    )
    assert len(wrapped) == 4
    expected = [(7, 7), (2, 2), (7, 8), (7, 7)]
    for virtual_index, (source_id, position) in enumerate(expected):
        sample = wrapped[virtual_index]
        assert sample["old_like_sample_id"].item() == source_id
        assert sample["old_like_mask"].sum().item() == 1
        assert sample["old_like_mask"][position]
        assert masks[source_id, position]
        # The wrapper retains the original sequence and its original shifted
        # next-token label; only the selected hidden position is supervised.
        assert sample["tokens"][position].item() == source_id
        assert sample["labels"][position].item() == source_id + 1


def test_wrapper_validates_recorded_indexed_dataset_identity(gt_fixture):
    root, _ = gt_fixture
    prefix = root.parent / "code_text_document"
    idx_payload = b"deterministic-index"
    bin_payload = b"token-bytes"
    Path(str(prefix) + ".idx").write_bytes(idx_payload)
    Path(str(prefix) + ".bin").write_bytes(bin_payload)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dataset_identity"].update(
        {
            # The artifact was built at a different prefix; runtime stages an
            # identical rsync copy at ``prefix``.
            "data_path": "/original/code_text_document",
            "idx_bytes": len(idx_payload),
            "bin_bytes": len(bin_payload),
            "idx_sha256": hashlib.sha256(idx_payload).hexdigest(),
        }
    )
    metadata_path.write_text(json.dumps(metadata))

    OldLikeGTDataset(
        _ToyDataset(8, 16),
        root,
        expected_sequence_length=16,
        expected_data_paths=[str(prefix)],
    )
    Path(str(prefix) + ".idx").write_bytes(idx_payload + b"corrupt")
    with pytest.raises(ValueError, match=r"\.idx size mismatch"):
        OldLikeGTDataset(
            _ToyDataset(8, 16),
            root,
            expected_sequence_length=16,
            expected_data_paths=[str(prefix)],
        )
