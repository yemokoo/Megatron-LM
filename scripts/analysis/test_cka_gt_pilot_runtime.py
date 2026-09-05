#!/usr/bin/env python3
"""Focused CPU-only tests for the CKA pilot runtime components."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from cka_gt_pilot_runtime import (  # noqa: E402
    ChunkArrowShardWriter,
    DocumentWindowBatchReader,
    PairedParquetMetricWriter,
    Pass1MembershipAccumulator,
    RandomPairDonorCache,
    TokenArrowShardWriter,
    build_attention_position_tensors,
    build_exact_raw_unit_quantiles,
    deterministic_minibatch_kmeans,
    fp32_metric_context,
    load_membership_statistics,
    membership_scores,
    paired_metric_stream_paths,
    process_pass2_metric_batch,
    recover_pass1_worker_metadata_from_stats,
    router_diagnostics_from_input,
    run_model_pilot,
    save_membership_statistics_atomic,
    validate_membership_identity,
    _deterministic_minibatch_kmeans_torch,
    _RawMomentAccumulator,
    ledoit_wolf_from_moments,
)
from cka_gt_pilot_windows import WINDOW_DTYPE  # noqa: E402


class FakeIndexedDataset:
    def __init__(self, documents: list[np.ndarray]) -> None:
        self.documents = [np.asarray(document, dtype=np.int32) for document in documents]
        self.sequence_lengths = np.asarray([value.size for value in self.documents], dtype=np.int32)
        self.document_indices = np.arange(len(documents) + 1, dtype=np.int64)

    def get(self, idx: int, offset: int = 0, length: int | None = None) -> np.ndarray:
        value = self.documents[int(idx)]
        if length is None:
            length = value.size - offset
        return value[int(offset) : int(offset) + int(length)]


def manifest_rows(lengths: list[int], sample_orders: list[int] | None = None) -> np.ndarray:
    rows = np.empty(len(lengths), dtype=WINDOW_DTYPE)
    if sample_orders is None:
        sample_orders = list(range(len(lengths)))
    for index, (length, order) in enumerate(zip(lengths, sample_orders)):
        eligible = 512 if length == 512 else 256 if length < 384 else 384
        rows[index] = (
            order,
            index,
            index,
            0,
            length,
            eligible,
            length - eligible,
            int(length < 512),
        )
    return rows


class ReaderAndInputTest(unittest.TestCase):
    def test_same_length_batches_are_unpadded_and_positions_restart(self) -> None:
        documents = [
            np.arange(0, 512),
            np.arange(1_000, 1_512),
            np.arange(2_000, 2_300),
        ]
        rows = manifest_rows([512, 512, 300], sample_orders=[2, 0, 1])
        reader = DocumentWindowBatchReader(
            FakeIndexedDataset(documents), rows, batch_size=2, device="cpu"
        )
        batches = list(reader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(tuple(batches[0].tokens.shape), (2, 512))
        self.assertEqual(tuple(batches[1].tokens.shape), (1, 300))
        self.assertEqual(batches[0].rows["sample_order"].tolist(), [0, 2])
        self.assertEqual(batches[0].position_ids[0].tolist(), list(range(512)))
        self.assertEqual(tuple(batches[0].attention_mask.shape), (1, 1, 512, 512))
        self.assertTrue(batches[0].attention_mask[0, 0, 0, 1])
        self.assertFalse(batches[0].attention_mask[0, 0, 1, 0])
        self.assertTrue(batches[1].valid_mask.all())

    def test_input_builder_rejects_padding_shaped_rank(self) -> None:
        with self.assertRaises(ValueError):
            build_attention_position_tensors(torch.ones(8, dtype=torch.long))


class FakeRouter:
    class Config:
        moe_input_jitter_eps = None

    def __init__(self) -> None:
        self.training = False
        self.config = self.Config()

    def gating(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def routing(self, logits: torch.Tensor):
        flat = logits.reshape(-1, logits.shape[-1])
        full = torch.softmax(flat.float(), dim=-1)
        weights, ids = torch.topk(full, k=4, dim=-1)
        scores = torch.zeros_like(full).scatter(1, ids, weights)
        routing_map = torch.zeros_like(full, dtype=torch.bool).scatter(1, ids, True)
        return scores, routing_map


class RouterTest(unittest.TestCase):
    def test_three_router_mass_views_from_standard_input(self) -> None:
        logits = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(3, 2, 16) / 20.0
        result = router_diagnostics_from_input(FakeRouter(), logits)
        self.assertEqual(tuple(result["top4_id"].shape), (3, 2, 4))
        self.assertEqual(result["top4_id"].dtype, torch.int16)
        expected_full = torch.softmax(logits, dim=-1)[..., :8].sum(dim=-1)
        torch.testing.assert_close(result["old_full_mass"], expected_full)
        # All four largest experts are new (12--15), so actual selected old mass is zero.
        torch.testing.assert_close(result["old_selected_mass"], torch.zeros(3, 2))
        self.assertTrue(torch.all(result["top4_id"] >= 12))


def canonical_tables(window_uid: int = 11, split: str = "selection"):
    import pyarrow as pa

    layer_values = [[1.0] * 8, [0.9] * 8]
    token = pa.table(
        {
            "domain": ["code", "code"],
            "split": [split, split],
            "window_uid": [window_uid, window_uid],
            "sample_order": [0, 0],
            "document_id": [4, 4],
            "window_offset": [0, 0],
            "position": [0, 1],
            "token_id": [10, 11],
            "eligible": [True, True],
            "cosine": layer_values,
            "relative_l2": [[0.0] * 8] * 2,
            "symmetric_relative_l2": [[0.0] * 8] * 2,
            "log_r": [[0.0] * 8] * 2,
            "ref_rms": layer_values,
            "maha_mean": [0.5, 0.6],
            "proto_mean": [0.7, 0.8],
            "cka_min_128": layer_values,
            "cka_min_256": layer_values,
            "s_min_128": layer_values,
            "s_min_256": layer_values,
            "worst_diag_ratio_128": [[0.2] * 8] * 2,
            "worst_diag_ratio_256": [[0.2] * 8] * 2,
        }
    )
    chunk = pa.table(
        {
            "domain": ["code"],
            "split": [split],
            "chunk_uid": [window_uid * 100_000 + 1],
            "window_uid": [window_uid],
            "sample_order": [0],
            "document_id": [4],
            "window_offset": [0],
            "scale": [128],
            "chunk_start": [0],
            "chunk_length": [128],
            "layer": [2],
            "cka": [1.0],
            "cka_permutation": [0.2],
            "cka_random_pair": [0.1],
            "random_pair_invalid_reason": [0],
            "random_pair_donor_window_uid": [window_uid + 1],
            "cka_off": [0.8],
            "diag_ratio": [0.2],
            "invalid_reason": [0],
        }
    )
    return token, chunk


class StorageTest(unittest.TestCase):
    def test_open_and_test_metrics_are_physically_disjoint_transactions(self) -> None:
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "runtime" / "pass2" / "code" / "all" / "worker_000"
            open_paths = paired_metric_stream_paths(
                pilot_root=root,
                worker_output=worker,
                domain="code",
                requested_split="all",
                worker_index=0,
                stream="open",
            )
            test_paths = paired_metric_stream_paths(
                pilot_root=root,
                worker_output=worker,
                domain="code",
                requested_split="all",
                worker_index=0,
                stream="test",
            )
            self.assertNotEqual(open_paths.journal_dir, test_paths.journal_dir)
            self.assertEqual(open_paths.token_dir.parents[0], root / "token_metrics")
            self.assertIn(root / "sealed_test" / "raw", test_paths.token_dir.parents)

            writers = {
                "open": PairedParquetMetricWriter(
                    open_paths.journal_dir,
                    token_dir=open_paths.token_dir,
                    chunk_dir=open_paths.chunk_dir,
                    target_token_rows_per_shard=1,
                    metadata={"stream_splits": ["calibration", "selection"]},
                ),
                "test": PairedParquetMetricWriter(
                    test_paths.journal_dir,
                    token_dir=test_paths.token_dir,
                    chunk_dir=test_paths.chunk_dir,
                    target_token_rows_per_shard=1,
                    metadata={"stream_splits": ["test"]},
                ),
            }
            writers["open"].append_batch(
                *canonical_tables(window_uid=41, split="selection")
            )
            with self.assertRaisesRegex(RuntimeError, "physical split isolation"):
                writers["test"].append_batch(
                    *canonical_tables(window_uid=43, split="selection")
                )
            writers["test"].append_batch(
                *canonical_tables(window_uid=42, split="test")
            )
            manifests = {name: writer.finalize() for name, writer in writers.items()}
            open_table = pq.read_table(manifests["open"]["shards"][0]["token_file"])
            test_table = pq.read_table(manifests["test"]["shards"][0]["token_file"])
            self.assertEqual(set(open_table["split"].to_pylist()), {"selection"})
            self.assertEqual(set(test_table["split"].to_pylist()), {"test"})
            self.assertNotEqual(
                manifests["open"]["shards"][0]["token_file"],
                manifests["test"]["shards"][0]["token_file"],
            )

    def test_single_stream_commits_whole_append_not_row_slice(self) -> None:
        token, _ = canonical_tables()
        # Duplicate the complete two-row window: rows_per_shard=3 must commit
        # all four rows together rather than slicing the second append.
        with tempfile.TemporaryDirectory() as temporary:
            writer = TokenArrowShardWriter(Path(temporary), rows_per_shard=3)
            writer.append(token)
            writer.append(token)
            payload = writer.finalize()
            self.assertEqual(len(payload["shards"]), 1)
            self.assertEqual(payload["shards"][0]["rows"], 4)
            resumed = TokenArrowShardWriter(Path(temporary), rows_per_shard=3)
            self.assertEqual(resumed.committed_window_uids().tolist(), [11])

    def test_crash_between_paired_renames_replays_without_duplicate(self) -> None:
        token, chunk = canonical_tables(window_uid=21)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = {"done": False}

            def inject(stage: str) -> None:
                if stage == "after_token_rename" and not failed["done"]:
                    failed["done"] = True
                    raise RuntimeError("synthetic crash")

            writer = PairedParquetMetricWriter(
                root, target_token_rows_per_shard=1, failure_injector=inject
            )
            with self.assertRaisesRegex(RuntimeError, "synthetic crash"):
                writer.append_batch(token, chunk, active_seconds=0.4)
            self.assertTrue((root / "token_metrics" / "shard_000000.parquet").is_file())
            self.assertFalse((root / "progress.json").exists())

            resumed = PairedParquetMetricWriter(root, target_token_rows_per_shard=1)
            resumed.append_batch(token, chunk, active_seconds=0.5)
            progress = resumed.finalize()
            self.assertEqual(progress["committed_batches"], 1)
            self.assertEqual(resumed.committed_window_uids().tolist(), [21])
            self.assertEqual(len(list((root / "token_metrics").glob("*.parquet"))), 1)
            self.assertEqual(len(list((root / "chunk_metrics").glob("*.parquet"))), 1)
            self.assertAlmostEqual(progress["cumulative_elapsed_seconds"], 0.5)

    def test_paired_cumulative_timing_survives_partial_and_final_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = PairedParquetMetricWriter(
                root, target_token_rows_per_shard=1
            )
            writer.append_batch(
                *canonical_tables(window_uid=51), active_seconds=1.25
            )
            first = writer.finalize()
            self.assertAlmostEqual(first["cumulative_elapsed_seconds"], 1.25)

            resumed = PairedParquetMetricWriter(
                root, target_token_rows_per_shard=1
            )
            self.assertAlmostEqual(resumed.cumulative_elapsed_seconds, 1.25)
            resumed.append_batch(
                *canonical_tables(window_uid=52), active_seconds=0.75
            )
            final = resumed.finalize()
            self.assertAlmostEqual(final["cumulative_elapsed_seconds"], 2.0)

            # This is the post-final-shard/pre-worker-metadata recovery path:
            # reopening only the durable paired journal retains non-zero time.
            recovered = PairedParquetMetricWriter(
                root, target_token_rows_per_shard=1
            )
            self.assertAlmostEqual(recovered.cumulative_elapsed_seconds, 2.0)

    def test_paired_progress_freezes_external_canonical_directories(self) -> None:
        token, chunk = canonical_tables(window_uid=31)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "worker" / "paired_progress"
            token_dir = root / "token_metrics" / "code_selection_worker_000"
            chunk_dir = root / "chunk_metrics" / "code_selection_worker_000"
            writer = PairedParquetMetricWriter(
                journal,
                token_dir=token_dir,
                chunk_dir=chunk_dir,
                target_token_rows_per_shard=1,
            )
            writer.append_batch(token, chunk)
            progress = writer.finalize()
            self.assertEqual(progress["token_dir"], str(token_dir.resolve()))
            self.assertEqual(progress["chunk_dir"], str(chunk_dir.resolve()))

            resumed = PairedParquetMetricWriter(
                journal,
                token_dir=token_dir,
                chunk_dir=chunk_dir,
                target_token_rows_per_shard=1,
            )
            self.assertEqual(resumed.committed_window_uids().tolist(), [31])
            with self.assertRaisesRegex(RuntimeError, "token_dir changed on resume"):
                PairedParquetMetricWriter(
                    journal,
                    token_dir=root / "token_metrics" / "wrong_worker",
                    chunk_dir=chunk_dir,
                    target_token_rows_per_shard=1,
                )

    def test_exact_raw_unit_quantiles_use_chunk_s_not_token_minimum(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            token_dir = root / "token_metrics"
            chunk_dir = root / "chunk_metrics"
            token_dir.mkdir()
            chunk_dir.mkdir()
            domains = ["wiki"] * 4 + ["code"] * 4 + ["code"] * 4
            splits = ["calibration"] * 8 + ["selection"] * 4
            base = np.arange(12 * 8, dtype=np.float32).reshape(12, 8) / 100.0
            token = pa.table(
                {
                    "domain": domains,
                    "split": splits,
                    "cosine": pa.FixedSizeListArray.from_arrays(
                        pa.array(base.reshape(-1)), 8
                    ),
                    "relative_l2": pa.FixedSizeListArray.from_arrays(
                        pa.array((base + 0.1).reshape(-1)), 8
                    ),
                    "log_r": pa.FixedSizeListArray.from_arrays(
                        pa.array((base - 0.2).reshape(-1)), 8
                    ),
                }
            )
            pq.write_table(token, token_dir / "part.parquet")
            chunk = pa.table(
                {
                    "domain": ["wiki", "wiki"],
                    "split": ["calibration", "calibration"],
                    "scale": [128, 256],
                    "layer": [2, 2],
                    "cka": [0.8, 0.9],
                    # Raw chunk-token values deliberately differ from any
                    # token-level minimum and are the only T calibration input.
                    "s_i": [[0.2, 0.4, 0.6], [1.0, 1.2, 1.4]],
                }
            )
            pq.write_table(chunk, chunk_dir / "part.parquet")
            output = root / "threshold_tables" / "raw_unit_quantiles.parquet"
            manifest = build_exact_raw_unit_quantiles(
                token_metric_paths=token_dir,
                chunk_metric_paths=chunk_dir,
                output_path=output,
                spill_dir=root / "threshold_tables" / "raw_unit_spills",
                layers=(2,),
            )
            result = pq.read_table(output).to_pylist()
            self.assertEqual(manifest["method"], "exact_disk_backed")
            s128 = [
                row
                for row in result
                if row["metric"] == "s_i"
                and row["scale"] == 128
                and abs(row["quantile"] - 0.05) < 1e-9
            ]
            self.assertEqual(len(s128), 1)
            self.assertAlmostEqual(s128[0]["value"], np.quantile([0.2, 0.4, 0.6], 0.05))
            self.assertEqual(s128[0]["count"], 3)


class Pass1StatisticsTest(unittest.TestCase):
    def test_membership_sampling_uses_prepared_stage_seed_sequences(self) -> None:
        accumulator = Pass1MembershipAccumulator(
            layers=(2,),
            hidden_size=2,
            total_tokens=100,
            covariance_sample_size=20,
            kmeans_reservoir_size=15,
            seed=1234,
        )

        def expected(stage_id: int, count: int) -> np.ndarray:
            generator = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([1234, 1, stage_id, 0]))
            )
            result = generator.choice(100, size=count, replace=False).astype(np.int64)
            result.sort()
            return result

        np.testing.assert_array_equal(accumulator.covariance_indices, expected(2, 20))
        np.testing.assert_array_equal(accumulator.kmeans_indices, expected(3, 15))
        provenance = accumulator.sampling_provenance
        self.assertEqual(provenance["covariance_stage_id"], 2)
        self.assertEqual(provenance["kmeans_reservoir_stage_id"], 3)
        self.assertEqual(provenance["covariance_entropy"], [1234, 1, 2, 0])
        self.assertEqual(provenance["kmeans_reservoir_entropy"], [1234, 1, 3, 0])

    def test_kmeans_plus_plus_initialization_is_deterministic_and_unique(self) -> None:
        rng = np.random.default_rng(91)
        values = rng.normal(size=(96, 5)).astype(np.float32)
        kwargs = {
            "n_clusters": 6,
            "batch_size": 12,
            "n_init": 2,
            "max_iter": 8,
            "reassignment_ratio": 0.0,
            "seed": 44,
        }
        centers_a, report_a = deterministic_minibatch_kmeans(values, **kwargs)
        centers_b, report_b = deterministic_minibatch_kmeans(values, **kwargs)
        np.testing.assert_array_equal(centers_a, centers_b)
        self.assertEqual(report_a, report_b)
        self.assertEqual(report_a["init"], "k-means++")
        self.assertEqual(report_a["init_size"], 36)
        self.assertEqual(
            report_a["init_pool_policy"],
            "min(n_samples,max(3*batch_size,3*n_clusters))",
        )
        for initial_indices in report_a["initial_indices_by_run"]:
            self.assertEqual(len(initial_indices), 6)
            self.assertEqual(len(set(initial_indices)), 6)
            self.assertEqual(np.unique(values[initial_indices], axis=0).shape[0], 6)

    def test_torch_kmeans_backend_preserves_resolved_init_pool_policy(self) -> None:
        rng = np.random.default_rng(92)
        values = rng.normal(size=(24, 4)).astype(np.float32)
        _, report = _deterministic_minibatch_kmeans_torch(
            values,
            n_clusters=3,
            batch_size=8,
            n_init=1,
            max_iter=3,
            reassignment_ratio=0.0,
            seed=45,
            init="k-means++",
            init_size=24,
            init_pool_policy="min(n_samples,max(3*batch_size,3*n_clusters))",
            device=torch.device("cpu"),
        )
        self.assertEqual(report["init_size"], 24)
        self.assertEqual(
            report["init_pool_policy"],
            "min(n_samples,max(3*batch_size,3*n_clusters))",
        )

    def test_streaming_mean_ledoit_wolf_kmeans_and_atomic_stats(self) -> None:
        rng = np.random.default_rng(4)
        left = rng.normal(-2.0, 0.2, size=(40, 4)).astype(np.float32)
        right = rng.normal(2.0, 0.2, size=(40, 4)).astype(np.float32)
        values = np.concatenate((left, right), axis=0)
        accumulator = Pass1MembershipAccumulator(
            layers=(2, 3),
            hidden_size=4,
            total_tokens=80,
            covariance_sample_size=80,
            kmeans_reservoir_size=40,
            seed=1234,
        )
        accumulator.update(
            {2: torch.from_numpy(values[:31]), 3: torch.from_numpy(2.0 * values[:31])}
        )
        accumulator.update(
            {2: torch.from_numpy(values[31:]), 3: torch.from_numpy(2.0 * values[31:])}
        )
        stats = accumulator.finalize(
            kmeans_params={
                "n_clusters": 2,
                "batch_size": 16,
                "n_init": 1,
                "max_iter": 30,
                "reassignment_ratio": 0.0,
                "random_state": 7,
            }
        )
        np.testing.assert_allclose(
            stats[2]["mean"], values.mean(axis=0), rtol=5e-5, atol=5e-7
        )
        empirical = np.cov(values.astype(np.float64), rowvar=False, bias=True)
        np.testing.assert_allclose(stats[2]["empirical_covariance"], empirical, rtol=2e-5, atol=2e-6)
        self.assertTrue(0.0 <= stats[2]["ledoit_wolf_shrinkage"] <= 1.0)
        self.assertEqual(stats[2]["prototypes"].shape, (2, 4))
        self.assertLess(float(stats[2]["prototypes"].min(axis=0).mean()), -1.0)
        self.assertGreater(float(stats[2]["prototypes"].max(axis=0).mean()), 1.0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wiki_calibration_membership.npz"
            pass1_metadata = {
                "analysis": "cka_gt_pilot_v1",
                "mode": "pass1",
                "domain": "wiki",
                "requested_split": "calibration",
                "dataset_prefix": "/synthetic/wiki",
                "reference_load": "/synthetic/reference",
                "reference_step": 600,
                "prepared_config_content_sha256": "config-a",
                "total_windows": 2,
                "total_tokens": 80,
                "seed": 1234,
                "max_windows": 0,
                "elapsed_seconds": 1.5,
                "cumulative_elapsed_seconds": 1.5,
                "timing_available": True,
            }
            save_membership_statistics_atomic(path, stats, metadata=pass1_metadata)
            expected = {
                "source_domain": "wiki",
                "source_split": "calibration",
                "prepared_config_content_sha256": "config-a",
                "source_window_count": 2,
                "source_token_count": 80,
                "seed": 1234,
                "layers": [2, 3],
            }
            loaded = load_membership_statistics(
                path, expected_saved_metadata=expected
            )
            np.testing.assert_array_equal(loaded[2]["mean"], stats[2]["mean"])
            self.assertEqual(
                loaded[2]["sampling_provenance"], stats[2]["sampling_provenance"]
            )
            self.assertEqual(loaded.membership_identity["path"], str(path.resolve()))
            self.assertEqual(
                loaded.membership_identity["size_bytes"], path.stat().st_size
            )
            self.assertEqual(len(loaded.membership_identity["sha256"]), 64)
            validate_membership_identity(
                loaded.membership_identity,
                expected_path=path,
                expected_saved_metadata=expected,
            )
            recovered_metadata_path = Path(temporary) / "worker" / "metadata.json"
            recovered = recover_pass1_worker_metadata_from_stats(
                path,
                recovered_metadata_path,
                expected_saved_metadata=expected,
                expected_runtime_metadata={
                    "mode": "pass1",
                    "reference_load": "/synthetic/reference",
                    "total_tokens": 80,
                },
            )
            self.assertTrue(recovered["completed"])
            self.assertTrue(recovered["resume_recovered_after_stats_commit"])
            self.assertAlmostEqual(recovered["cumulative_elapsed_seconds"], 1.5)
            self.assertEqual(
                json.loads(recovered_metadata_path.read_text()), recovered
            )
            with self.assertRaisesRegex(
                RuntimeError, "prepared_config_content_sha256"
            ):
                load_membership_statistics(
                    path,
                    expected_saved_metadata={
                        **expected,
                        "prepared_config_content_sha256": "swapped-config",
                    },
                )
            maha, proto = membership_scores(torch.from_numpy(values[:5]), loaded[2])
            self.assertEqual(maha.shape, (5,))
            self.assertEqual(proto.shape, (5,))
            self.assertTrue(torch.isfinite(maha).all() and torch.isfinite(proto).all())
            original_identity = dict(loaded.membership_identity)
            swapped_stats = {
                layer: {
                    **layer_stats,
                    "mean": np.asarray(layer_stats["mean"]) + np.float32(0.25),
                }
                for layer, layer_stats in stats.items()
            }
            save_membership_statistics_atomic(
                path, swapped_stats, metadata=pass1_metadata
            )
            with self.assertRaisesRegex(RuntimeError, "file identity mismatch"):
                load_membership_statistics(
                    path,
                    expected_identity=original_identity,
                    expected_saved_metadata=expected,
                )

    def test_covariance_moment_buffer_is_batching_invariant(self) -> None:
        rng = np.random.default_rng(17)
        values = torch.from_numpy(rng.normal(size=(53, 6)).astype(np.float32))
        one = _RawMomentAccumulator(6, buffer_rows=128)
        one.update(values)
        many = _RawMomentAccumulator(6, buffer_rows=7)
        for start in range(0, values.shape[0], 3):
            many.update(values[start : start + 3])
        one_shrunk, one_alpha, one_empirical = ledoit_wolf_from_moments(one)
        many_shrunk, many_alpha, many_empirical = ledoit_wolf_from_moments(many)
        np.testing.assert_allclose(many_empirical, one_empirical, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(many_shrunk, one_shrunk, rtol=2e-6, atol=2e-7)
        self.assertAlmostEqual(many_alpha, one_alpha, places=5)


class Pass2Test(unittest.TestCase):
    def test_singleton_tail_random_pair_uses_deterministic_full_window_donor(
        self,
    ) -> None:
        torch.manual_seed(41)
        layers = (2, 3)
        candidates = manifest_rows([512] * 8, sample_orders=list(range(8)))
        donor_after = {
            layer: torch.randn(8, 512, 8) + float(layer)
            for layer in layers
        }

        first = RandomPairDonorCache(
            domain="code",
            split="selection",
            layers=layers,
            manifest_rows=candidates,
        )
        first.add(donor_after, candidates)
        # Simulate an interrupted worker rebuilding the same memory-only cache
        # in a different forwarding batch partition/order.
        rebuilt = RandomPairDonorCache(
            domain="code",
            split="selection",
            layers=layers,
            manifest_rows=candidates,
        )
        rebuilt.add(
            {layer: values[4:] for layer, values in donor_after.items()},
            candidates[4:],
        )
        rebuilt.add(
            {layer: values[:4] for layer, values in donor_after.items()},
            candidates[:4],
        )
        self.assertEqual(first.cached_window_uids, rebuilt.cached_window_uids)
        self.assertTrue(first.complete)
        self.assertTrue(rebuilt.complete)

        for length in (128, 256, 512):
            target_rows = manifest_rows([length], sample_orders=[99])
            before = {
                layer: torch.randn(1, length, 8) + float(layer)
                for layer in layers
            }
            after = {
                layer: torch.randn(1, length, 8) - float(layer)
                for layer in layers
            }
            kwargs = {
                "before_by_layer": before,
                "after_by_layer": after,
                "token_ids": torch.arange(length).reshape(1, length),
                "manifest_rows": target_rows,
                "domain": "code",
                "split": "selection",
                "seed": 1234,
            }
            _, chunk_first = process_pass2_metric_batch(
                **kwargs, random_pair_donor_cache=first
            )
            _, chunk_rebuilt = process_pass2_metric_batch(
                **kwargs, random_pair_donor_cache=rebuilt
            )
            self.assertTrue(np.isfinite(chunk_first["cka_random_pair"]).all())
            self.assertTrue(
                np.all(chunk_first["random_pair_invalid_reason"] == 0)
            )
            target_uid = 100_000_000 + 99
            self.assertTrue(
                np.all(chunk_first["random_pair_donor_window_uid"] != target_uid)
            )
            np.testing.assert_array_equal(
                chunk_first["random_pair_donor_window_uid"],
                chunk_rebuilt["random_pair_donor_window_uid"],
            )
            np.testing.assert_allclose(
                chunk_first["cka_random_pair"],
                chunk_rebuilt["cka_random_pair"],
                rtol=0.0,
                atol=0.0,
            )

        no_donor_rows = manifest_rows([128], sample_orders=[200])
        empty_cache = RandomPairDonorCache(
            domain="code",
            split="selection",
            layers=layers,
            manifest_rows=no_donor_rows,
        )
        before = {layer: torch.randn(1, 128, 8) for layer in layers}
        after = {layer: torch.randn(1, 128, 8) for layer in layers}
        _, no_donor_chunk = process_pass2_metric_batch(
            before_by_layer=before,
            after_by_layer=after,
            token_ids=torch.arange(128).reshape(1, 128),
            manifest_rows=no_donor_rows,
            domain="code",
            split="selection",
            random_pair_donor_cache=empty_cache,
        )
        self.assertTrue(np.isnan(no_donor_chunk["cka_random_pair"]).all())
        self.assertTrue(
            np.all(no_donor_chunk["random_pair_invalid_reason"] == 2)
        )
        self.assertTrue(
            np.all(no_donor_chunk["random_pair_donor_window_uid"] == -1)
        )

    def test_identity_pair_produces_canonical_wide_and_long_tables(self) -> None:
        torch.manual_seed(5)
        batch, length, hidden = 2, 256, 8
        tokens = torch.arange(batch * length).reshape(batch, length)
        rows = manifest_rows([length, length])
        before = {
            2: torch.randn(batch, length, hidden),
            3: torch.randn(batch, length, hidden),
        }
        after = {layer: value.clone() for layer, value in before.items()}
        token, chunk = process_pass2_metric_batch(
            before_by_layer=before,
            after_by_layer=after,
            token_ids=tokens,
            manifest_rows=rows,
            domain="code",
            split="selection",
            seed=1234,
        )
        for name in (
            "window_uid",
            "relative_l2",
            "symmetric_relative_l2",
            "cka_min_128",
            "cka_min_256",
            "s_min_128",
            "s_min_256",
            "worst_diag_ratio_128",
            "worst_diag_ratio_256",
            "maha_mean",
            "proto_mean",
        ):
            self.assertIn(name, token)
        self.assertEqual(token["relative_l2"].shape, (batch * length, 2))
        np.testing.assert_allclose(token["relative_l2"], 0.0, atol=1e-6)
        np.testing.assert_allclose(token["cka_min_128"], 1.0, atol=2e-5)
        np.testing.assert_allclose(token["cka_min_256"], 1.0, atol=2e-5)
        self.assertEqual(token["worst_chunk_id_128"].shape, (batch * length, 2))
        self.assertEqual(token["worst_diag_ratio_128"].shape, (batch * length, 2))
        self.assertTrue(np.isnan(token["maha_mean"]).all())
        self.assertTrue(np.isnan(token["proto_mean"]).all())

        # Per window/layer: 3 x 128 chunks + 1 x 256 + 1 whole-window control.
        self.assertEqual(len(chunk["cka"]), batch * 2 * 5)
        self.assertIn("cka_permutation", chunk)
        self.assertIn("cka_random_pair", chunk)
        token_table = TokenArrowShardWriter  # schema assertion occurs in paired writer below
        del token_table
        with tempfile.TemporaryDirectory() as temporary:
            paired = PairedParquetMetricWriter(
                temporary, target_token_rows_per_shard=1
            )
            paired.append_batch(token, chunk)
            paired.finalize()


class ContextAndEntrypointTest(unittest.TestCase):
    def test_tf32_state_restores_and_driver_signature_is_stable(self) -> None:
        previous = bool(torch.backends.cuda.matmul.allow_tf32)
        with fp32_metric_context() as audit:
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(audit["cuda_matmul_allow_tf32_before"], previous)
        self.assertEqual(bool(torch.backends.cuda.matmul.allow_tf32), previous)
        self.assertEqual(audit["cuda_matmul_allow_tf32_restored"], previous)

        seen = {}

        def driver(**kwargs):
            seen.update(kwargs)
            return "ok"

        result = run_model_pilot(
            model="student",
            teacher="reference",
            args=object(),
            hooks={"driver": driver},
            print_fn=lambda _: None,
        )
        self.assertEqual(result, "ok")
        self.assertEqual(seen["model"], "student")
        self.assertEqual(seen["teacher"], "reference")


if __name__ == "__main__":
    unittest.main(verbosity=2)
