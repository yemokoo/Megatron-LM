from __future__ import annotations

import json
import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

try:
    from scripts.analysis.analyze_cka_gt_pilot import (
        BUNDLES,
        CHUNK_DIAGNOSTIC_SCALES,
        LAYERS,
        SCALES,
        _context_row,
        _enrich_context_samples,
        _prepared_input_provenance,
        _report,
        _report_validation_evidence,
        condition_consensus,
        run_analysis,
    )
except ModuleNotFoundError:  # direct CLI execution from scripts/analysis
    from analyze_cka_gt_pilot import (
        BUNDLES,
        CHUNK_DIAGNOSTIC_SCALES,
        LAYERS,
        SCALES,
        _context_row,
        _enrich_context_samples,
        _prepared_input_provenance,
        _report,
        _report_validation_evidence,
        condition_consensus,
        run_analysis,
    )


def _fixed(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def _nested_fixed(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values)
    inner = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[2])
    return pa.FixedSizeListArray.from_arrays(inner, values.shape[1])


def _token_table(windows_per_domain: int = 1_000, seed: int = 1234) -> pa.Table:
    rng = np.random.default_rng(seed)
    domains: list[str] = []
    splits: list[str] = []
    window_uid: list[int] = []
    document_id: list[int] = []
    window_offset: list[int] = []
    window_length: list[int] = []
    position: list[int] = []
    document_token_offset: list[int] = []
    token_id: list[int] = []
    matrices: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "cosine",
            "relative_l2",
            "symmetric_relative_l2",
            "log_r",
            "ref_rms",
            "maha",
            "proto",
            "cka_min_128",
            "cka_min_256",
            "s_min_128",
            "s_min_256",
            "r_min_128",
            "r_mean_128",
            "r_max_128",
            "r_min_256",
            "r_mean_256",
            "r_max_256",
            "before_old_full_mass",
            "after_old_full_mass",
            "before_old_selected_mass",
            "after_old_selected_mass",
            "worst_diag_ratio_128",
            "worst_diag_ratio_256",
            "s_at_worst_cka_128",
            "s_at_worst_cka_256",
        )
    }
    bool_matrices: dict[str, list[np.ndarray]] = {
        name: [] for name in ("valid_cka_128", "valid_cka_256", "valid_s_128", "valid_s_256")
    }
    integer_matrices: dict[str, list[np.ndarray]] = {
        "worst_chunk_id_128": [],
        "worst_chunk_id_256": [],
        "worst_s_chunk_id_128": [],
        "worst_s_chunk_id_256": [],
    }
    routing: dict[str, list[np.ndarray]] = {
        "before_top4_ids": [],
        "after_top4_ids": [],
        "before_top4_weight": [],
        "after_top4_weight": [],
    }

    uid_base = 0
    split_counts = {
        "calibration": (windows_per_domain * 40) // 100,
        "selection": (windows_per_domain * 30) // 100,
        "test": windows_per_domain - (windows_per_domain * 40) // 100 - (windows_per_domain * 30) // 100,
    }
    for domain in ("code", "wiki"):
        for split in ("calibration", "selection", "test"):
            count = split_counts[split]
            anchor = rng.random(count) < (0.12 if domain == "code" else 0.96)
            base_cka = np.where(anchor, 0.97, 0.55)[:, None]
            base_s = np.where(anchor, 1.10, 0.20)[:, None]
            base_rel = np.where(anchor, 0.045, 0.40)[:, None]
            base_log = np.where(anchor, 0.018, 0.22)[:, None]
            cosine = np.where(anchor, 0.995, 0.75)[:, None] + rng.normal(0, 0.004, (count, 8))
            cka128 = base_cka + rng.normal(0, 0.012, (count, 8))
            cka256 = base_cka + rng.normal(0, 0.010, (count, 8))
            s128 = base_s + rng.normal(0, 0.06, (count, 8))
            s256 = base_s + rng.normal(0, 0.05, (count, 8))
            rel = np.maximum(base_rel + rng.normal(0, 0.008, (count, 8)), 0)
            logr = base_log + rng.normal(0, 0.005, (count, 8))
            sym = 2 * rel / (1 + np.maximum(1 - rel, 0.05))
            maha = np.where(anchor, 20.0, 45.0)[:, None] + rng.normal(0, 2, (count, 8))
            proto = np.where(anchor, 4.0, 9.0)[:, None] + rng.normal(0, 0.5, (count, 8))
            rmean128 = np.where(anchor, 0.95, 0.25)[:, None] + rng.normal(0, 0.02, (count, 8))
            rmean256 = np.where(anchor, 0.96, 0.28)[:, None] + rng.normal(0, 0.02, (count, 8))
            valid = np.ones((count, 8), dtype=bool)
            # A deterministic small ineligible tail population.
            valid[np.arange(count) % 97 == 0, :3] = False
            for array in (cka128, cka256, s128, s256):
                array[~valid] = np.nan

            domains.extend([domain] * count)
            splits.extend([split] * count)
            local = np.arange(count, dtype=np.int64)
            # Schema smoke: one representative contextual token per unique
            # synthetic window. This tests window identity/partition flow but
            # intentionally does not emulate 512-token hidden computation.
            windows = uid_base + local
            positions = np.zeros(count, dtype=np.int64)
            window_uid.extend(windows.tolist())
            document_id.extend((windows + 100_000).tolist())
            window_offset.extend((local * 512).tolist())
            window_length.extend([512] * count)
            position.extend(positions.tolist())
            document_token_offset.extend((local * 512 + positions).tolist())
            token_id.extend((local % 32_000).tolist())
            uid_base += count + 10

            values = {
                "cosine": cosine,
                "relative_l2": rel,
                "symmetric_relative_l2": sym,
                "log_r": logr,
                "ref_rms": np.ones((count, 8)),
                "maha": maha,
                "proto": proto,
                "cka_min_128": cka128,
                "cka_min_256": cka256,
                "s_min_128": s128,
                "s_min_256": s256,
                "r_min_128": rmean128 - 0.03,
                "r_mean_128": rmean128,
                "r_max_128": rmean128 + 0.03,
                "r_min_256": rmean256 - 0.03,
                "r_mean_256": rmean256,
                "r_max_256": rmean256 + 0.03,
                "before_old_full_mass": np.full((count, 8), 0.72),
                "after_old_full_mass": np.where(anchor[:, None], 0.71, 0.55) * np.ones((count, 8)),
                "before_old_selected_mass": np.full((count, 8), 0.68),
                "after_old_selected_mass": np.where(anchor[:, None], 0.67, 0.50) * np.ones((count, 8)),
                "worst_diag_ratio_128": np.where(anchor[:, None], 0.16, 0.42) * np.ones((count, 8)),
                "worst_diag_ratio_256": np.where(anchor[:, None], 0.14, 0.38) * np.ones((count, 8)),
                "s_at_worst_cka_128": s128 + 0.125,
                "s_at_worst_cka_256": s256 + 0.250,
            }
            for name, value in values.items():
                matrices[name].append(value.astype(np.float32))
            for name in bool_matrices:
                bool_matrices[name].append(valid.copy())
            integer_matrices["worst_chunk_id_128"].append(np.tile(np.arange(8), (count, 1)) + 1000)
            integer_matrices["worst_chunk_id_256"].append(np.tile(np.arange(8), (count, 1)) + 2000)
            integer_matrices["worst_s_chunk_id_128"].append(np.tile(np.arange(8), (count, 1)) + 3000)
            integer_matrices["worst_s_chunk_id_256"].append(np.tile(np.arange(8), (count, 1)) + 4000)
            ids_before = np.tile(np.asarray([0, 1, 8, 9], dtype=np.int16), (count, 8, 1))
            ids_after = ids_before.copy()
            ids_after[~anchor, :, -1] = 10
            routing["before_top4_ids"].append(ids_before)
            routing["after_top4_ids"].append(ids_after)
            routing["before_top4_weight"].append(np.full((count, 8, 4), 0.25, dtype=np.float32))
            routing["after_top4_weight"].append(np.full((count, 8, 4), 0.25, dtype=np.float32))

    payload: dict[str, pa.Array] = {
        "domain": pa.array(domains),
        "split": pa.array(splits),
        "window_uid": pa.array(window_uid, type=pa.int64()),
        "document_id": pa.array(document_id, type=pa.int64()),
        "window_offset": pa.array(window_offset, type=pa.int64()),
        "window_length": pa.array(window_length, type=pa.int32()),
        "position": pa.array(position, type=pa.int32()),
        "document_token_offset": pa.array(document_token_offset, type=pa.int64()),
        "token_id": pa.array(token_id, type=pa.int32()),
    }
    for name, chunks in matrices.items():
        payload[name] = _fixed(np.concatenate(chunks).astype(np.float32))
    for name, chunks in bool_matrices.items():
        payload[name] = _fixed(np.concatenate(chunks).astype(bool))
    for name, chunks in integer_matrices.items():
        payload[name] = _fixed(np.concatenate(chunks).astype(np.int64))
    for name, chunks in routing.items():
        payload[name] = _nested_fixed(np.concatenate(chunks))
    return pa.table(payload)


def _chunk_table(rows_per_group: int = 80, seed: int = 99) -> pa.Table:
    rng = np.random.default_rng(seed)
    rows = []
    chunk_uid = 0
    for domain in ("code", "wiki"):
        for split in ("calibration", "selection", "test"):
            for scale in CHUNK_DIAGNOSTIC_SCALES:
                for layer in LAYERS:
                    actual = np.clip(
                        rng.normal(0.96 if domain == "wiki" else 0.72, 0.04, rows_per_group), 0, 1
                    )
                    perm = np.clip(rng.normal(0.22, 0.05, rows_per_group), 0, 1)
                    random_pair = np.clip(rng.normal(0.18, 0.05, rows_per_group), 0, 1)
                    # Model the known diagnostic-only singleton donor miss.
                    random_pair[0] = np.nan
                    diag = np.clip(rng.normal(0.18, 0.04, rows_per_group), 0, 1)
                    cka_off = np.clip(actual * (1.0 - diag), 0, 1)
                    for index in range(rows_per_group):
                        raw_s = rng.normal(
                            1.05 if domain == "wiki" else 0.25, 0.08, 12
                        ).astype(np.float32)
                        if index == 0:
                            raw_s[0] = -0.5
                        if (
                            domain == "code"
                            and split == "selection"
                            and scale == 512
                            and layer == 2
                            and index == 0
                        ):
                            raw_s[1] = 150.0
                        rows.append(
                            {
                                "domain": domain,
                                "split": split,
                                "chunk_uid": chunk_uid,
                                "window_uid": chunk_uid // 5,
                                "scale": scale,
                                "layer": layer,
                                "cka": float(actual[index]),
                                "cka_permutation": float(perm[index]),
                                "cka_random_pair": float(random_pair[index]),
                                "random_pair_invalid_reason": 1 if index == 0 else 0,
                                "random_pair_donor_window_uid": (
                                    -1 if index == 0 else (chunk_uid // 5) + 100_000
                                ),
                                "diag_ratio": float(diag[index]),
                                "cka_off": float(cka_off[index]),
                                "offdiag_warning": bool(index % 37 == 0),
                                "s_i": raw_s.tolist(),
                                "invalid_reason": "valid",
                            }
                        )
                        chunk_uid += 1
    return pa.Table.from_pylist(rows)


def _raw_quantiles(tokens: pa.Table, chunks: pa.Table) -> pa.Table:
    rows = []
    token = tokens.to_pydict()
    domains = np.asarray(token["domain"])
    splits = np.asarray(token["split"])
    chunk = chunks.to_pydict()
    chunk_domain = np.asarray(chunk["domain"])
    chunk_split = np.asarray(chunk["split"])
    chunk_scale = np.asarray(chunk["scale"])
    chunk_layer = np.asarray(chunk["layer"])
    chunk_cka = np.asarray(chunk["cka"])

    def add(domain: str, split: str, metric: str, scale: int | None, layer: int, q: float, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        rows.append(
            {
                "domain": domain,
                "split": split,
                "metric": metric,
                "scale": scale,
                "layer": layer,
                "quantile": q,
                "value": float(np.quantile(finite, q)),
                "count": int(finite.size),
                "method": "exact_disk_backed",
            }
        )

    wiki_cal = (domains == "wiki") & (splits == "calibration")
    for layer_index, layer in enumerate(LAYERS):
        for scale in SCALES:
            mask = (
                (chunk_domain == "wiki")
                & (chunk_split == "calibration")
                & (chunk_scale == scale)
                & (chunk_layer == layer)
            )
            for q in (0.01, 0.03, 0.05):
                add("wiki", "calibration", "cka", scale, layer, q, chunk_cka[mask])
                # Synthetic raw-unit s; deliberately not the token minimum array.
                raw_s = np.asarray(token[f"s_min_{scale}"].to_pylist() if hasattr(token[f"s_min_{scale}"], "to_pylist") else token[f"s_min_{scale}"])
                add("wiki", "calibration", "s_i", scale, layer, q, raw_s[wiki_cal, layer_index] + 0.02)
        rel = np.asarray(token["relative_l2"])[wiki_cal, layer_index]
        abs_log = np.abs(np.asarray(token["log_r"])[wiki_cal, layer_index])
        for q in (0.95, 0.97, 0.99):
            add("wiki", "calibration", "relative_l2", None, layer, q, rel)
            add("wiki", "calibration", "abs_log_r", None, layer, q, abs_log)
        for split in ("calibration", "selection"):
            mask = (domains == "code") & (splits == split)
            cosine = np.asarray(token["cosine"])[mask, layer_index]
            add("code", split, "cosine", None, layer, 0.99, cosine)
    return pa.Table.from_pylist(rows)


def _write_synthetic(
    root: Path,
    windows_per_domain: int = 1_000,
    *,
    write_raw_quantiles: bool = True,
    seal_test_raw: bool = False,
) -> None:
    token_dir = root / "token_metrics"
    chunk_dir = root / "chunk_metrics"
    threshold_dir = root / "threshold_tables"
    token_dir.mkdir(parents=True)
    chunk_dir.mkdir(parents=True)
    threshold_dir.mkdir(parents=True)
    tokens = _token_table(windows_per_domain=windows_per_domain)
    chunks = _chunk_table()
    if seal_test_raw:
        token_is_test = np.asarray(tokens["split"].to_pylist()) == "test"
        chunk_is_test = np.asarray(chunks["split"].to_pylist()) == "test"
        pq.write_table(tokens.filter(pa.array(~token_is_test)), token_dir / "part-00000.parquet", compression="zstd")
        pq.write_table(chunks.filter(pa.array(~chunk_is_test)), chunk_dir / "part-00000.parquet", compression="zstd")
        sealed_token_dir = root / "sealed_test" / "raw" / "token_metrics"
        sealed_chunk_dir = root / "sealed_test" / "raw" / "chunk_metrics"
        sealed_token_dir.mkdir(parents=True)
        sealed_chunk_dir.mkdir(parents=True)
        pq.write_table(tokens.filter(pa.array(token_is_test)), sealed_token_dir / "part-00000.parquet", compression="zstd")
        pq.write_table(chunks.filter(pa.array(chunk_is_test)), sealed_chunk_dir / "part-00000.parquet", compression="zstd")
    else:
        pq.write_table(tokens, token_dir / "part-00000.parquet", compression="zstd")
        pq.write_table(chunks, chunk_dir / "part-00000.parquet", compression="zstd")
    if write_raw_quantiles:
        pq.write_table(_raw_quantiles(tokens, chunks), threshold_dir / "raw_unit_quantiles.parquet")
    for domain in ("code", "wiki"):
        split_dir = root / "splits" / domain
        split_dir.mkdir(parents=True)
        manifest = {
            "domain": domain,
            "document_count": 2_000,
            "splits": {
                "calibration": {"sampled_window_count": 400},
                "selection": {"sampled_window_count": 300},
                "test": {"sampled_window_count": 300},
            },
            "all_documents_window_stats": {
                "full_window_count": 3_000,
                "tail_window_count": 200,
                "discarded_tail_fraction_of_document_tokens": 0.004,
                "ineligible_fraction_of_retained_tokens": 0.012,
            },
            "overlap_validation": {"passed": True, "pairwise_overlap": 0},
            "tail": {"fixed_stride_ineligible_fraction": 0.012},
        }
        (split_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    router_metadata = (
        root
        / "runtime"
        / "router_smoke"
        / "code"
        / "calibration"
        / "worker_000"
        / "metadata.json"
    )
    router_metadata.parent.mkdir(parents=True)
    router_metadata.write_text(
        json.dumps(
            {
                "completed": True,
                "router_probe_verified": True,
                "natural_routing": True,
                "mode": "router_smoke",
            }
        ),
        encoding="utf-8",
    )
    validation = root / "validation" / "07_full_pre_analysis.json"
    validation.parent.mkdir(parents=True)
    validation.write_text(
        json.dumps(
            {
                "schema": "cka_gt_pilot_validation_v1",
                "ok": True,
                "deep": True,
                "error_count": 0,
                "warning_count": 0,
                "errors": [],
                "warnings": [],
                "checks": ["contribution identity sum(c_i)=CKA passed", "deep scalar validation passed"],
            }
        ),
        encoding="utf-8",
    )
    (root / "synthetic_smoke_metadata.json").write_text(
        json.dumps(
            {
                "schema": "cka_gt_pilot_synthetic_schema_smoke_v1",
                "windows_per_domain": windows_per_domain,
                "split_counts_per_domain": {
                    "calibration": (windows_per_domain * 40) // 100,
                    "selection": (windows_per_domain * 30) // 100,
                    "test": windows_per_domain - (windows_per_domain * 40) // 100 - (windows_per_domain * 30) // 100,
                },
                "representative_tokens_per_window": 1,
                "limitation": "schema/selector/report E2E only; this fixture does not perform hidden forward or emulate 512 tokens/window",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_condition_consensus_uses_seven_of_eight_and_six_of_six() -> None:
    values = np.asarray(
        [
            [1] * 8,
            [1] * 7 + [0],
            [1] * 6 + [np.nan, np.nan],
            [1] * 6 + [0, np.nan],
            [1] * 5 + [np.nan] * 3,
        ],
        dtype=np.float32,
    )
    result = condition_consensus(values, np.full(8, 0.5), lower_is_better=False)
    assert result.eligible.tolist() == [True, True, True, True, False]
    assert result.passed.tolist() == [True, True, True, False, False]


def test_negative_contribution_forces_t_failure_even_with_negative_threshold() -> None:
    values = np.ones((2, 8), dtype=np.float32)
    values[0, 0] = -0.1
    values[1, :2] = -0.1
    result = condition_consensus(
        values,
        np.full(8, -0.5, dtype=np.float32),
        lower_is_better=False,
        forced_fail=values < 0,
    )
    assert result.n_valid.tolist() == [8, 8]
    assert result.pass_count.tolist() == [7, 6]
    assert result.passed.tolist() == [True, False]


def test_context_row_preserves_distinct_worst_cka_and_worst_s_provenance() -> None:
    batch = _token_table(windows_per_domain=10).to_batches(max_chunksize=10)[0]
    row = _context_row(batch, 1, None)  # evaluated is intentionally unused by serialization
    for scale in SCALES:
        assert row[f"worst_chunk_id_{scale}"][0] == (1000 if scale == 128 else 2000)
        assert row[f"worst_s_chunk_id_{scale}"][0] == (3000 if scale == 128 else 4000)
        assert len(row[f"s_at_worst_cka_{scale}"]) == len(LAYERS)
        assert row[f"s_at_worst_cka_{scale}"][0] != row[f"s_min_{scale}"][0]


def test_probe_evidence_falls_back_to_pass2_router_metadata(tmp_path: Path) -> None:
    metadata = (
        tmp_path / "runtime" / "pass2" / "code" / "all" / "worker_000" / "metadata.json"
    )
    metadata.parent.mkdir(parents=True)
    sealed_token_path = tmp_path / "sealed_test" / "raw" / "token_metrics" / "part-00000.parquet"
    sealed_chunk_path = tmp_path / "sealed_test" / "raw" / "chunk_metrics" / "part-00000.parquet"
    metadata.write_text(
        json.dumps(
            {
                "completed": True,
                "router_probe_verified": True,
                "natural_routing": True,
                "mode": "pass2",
                "domain": "code",
                "requested_split": "all",
                "standard_router_layers": list(LAYERS),
                "representation": "residual_layer_output",
                "raw_hidden_stored": False,
                "paired_manifests": {
                    "open": {"token_path": str(tmp_path / "token_metrics")},
                    "test": {
                        "token_path": str(sealed_token_path),
                        "chunk_path": str(sealed_chunk_path),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    probe, _ = _report_validation_evidence(tmp_path, tmp_path)
    assert probe["status"] == "PASS"
    assert probe["source_kind"] == "runtime_pass2_router_probe_metadata"
    assert probe["payload"]["router_probe_verified"] is True
    assert probe["payload"]["natural_routing"] is True
    assert probe["payload"]["standard_router_layers"] == list(LAYERS)
    serialized_probe = json.dumps(probe, sort_keys=True)
    assert "paired_manifests" not in serialized_probe
    assert "sealed_test/raw" not in serialized_probe
    assert str(sealed_token_path) not in serialized_probe
    assert str(sealed_chunk_path) not in serialized_probe

    _report(
        tmp_path,
        tmp_path,
        threshold_rows=[],
        same_rows=[],
        pairwise_rows=[],
        chunk_summary={"raw_s_i_safety": {}},
        routing={},
        window_summary={"manifests": []},
        input_inventory={"inventory_digest_sha256": "unit-test-inventory"},
        prepared_provenance={},
    )
    evidence = json.loads((tmp_path / "report_validation_evidence.json").read_text())
    assert evidence["probe"]["payload"]["router_probe_verified"] is True
    assert evidence["probe"]["payload"]["natural_routing"] is True
    for report_artifact in (
        (tmp_path / "report_validation_evidence.json").read_text(),
        (tmp_path / "REPORT.md").read_text(),
    ):
        assert "paired_manifests" not in report_artifact
        assert "sealed_test/raw" not in report_artifact
        assert str(sealed_token_path) not in report_artifact
        assert str(sealed_chunk_path) not in report_artifact


def test_prepared_input_provenance_binds_config_sources_and_checkpoints(tmp_path: Path) -> None:
    sources = {
        domain: {
            "schema": "cka_gt_pilot_source_dataset_identity_v1",
            "storage_kind": "physical_indexed_dataset_files",
            "resolved_prefix": f"/data/{domain}",
            "idx": {"size_bytes": 11, "sha256": f"{domain}-idx"},
            "bin": {"size_bytes": 22, "sha256": f"{domain}-bin"},
        }
        for domain in ("code", "wiki")
    }
    checkpoints = {
        name: {
            "schema": "cka_gt_pilot_checkpoint_identity_v1",
            "storage_kind": "megatron_tracker_and_iteration_files",
            "resolved_root": f"/checkpoint/{name}",
            "tracker_step": 600 if name == "before" else 1800,
            "iteration_dir": "iter_0000000",
            "total_bytes": 123,
            "content_sha256": f"{name}-checkpoint",
        }
        for name in ("before", "after")
    }
    config = {
        "schema": "cka_gt_pilot_prepared_inputs_v1",
        "source_dataset_identity": sources,
        "checkpoint_identity": checkpoints,
    }
    canonical = (
        json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    config["config_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    provenance = _prepared_input_provenance(tmp_path)
    assert provenance["available"] is True
    assert provenance["config_content_sha256"] == config["config_content_sha256"]
    assert provenance["source_dataset_identity"]["code"]["bin_sha256"] == "code-bin"
    assert provenance["checkpoint_identity"]["after"]["content_sha256"] == "after-checkpoint"

    config["checkpoint_identity"]["after"]["total_bytes"] = 124
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="config content SHA256"):
        _prepared_input_provenance(tmp_path)


def test_one_thousand_window_identity_artifact_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "cka_gt_pilot_v1"
    _write_synthetic(root)
    result = run_analysis(root, bins=200, batch_size=257)

    assert (root / "REPORT.md").exists()
    assert (root / "report_validation_evidence.json").exists()
    assert (root / "threshold_tables" / "candidate_bundles.csv").exists()
    assert (root / "threshold_tables" / "same_layer_diagnostic.csv").exists()
    assert (root / "selector_comparison" / "pairwise_jaccard.csv").exists()
    assert (root / "selector_comparison" / "code_selection_assignments.parquet").exists()
    assert (root / "histograms" / "chunk_cka_128_2.svg").exists()
    assert (root / "histograms" / "chunk_cka_128_2.csv").exists()
    assert (root / "histograms" / "chunk_cka_512_2.svg").exists()
    assert (root / "histograms" / "chunk_cka_512_2.csv").exists()
    assert (root / "histograms" / "diag_ratio_512_2.svg").exists()
    assert (root / "histograms" / "raw_s_i_512_2.svg").exists()
    assert (root / "histograms" / "cka_off_512_2.svg").exists()
    assert (root / "histograms" / "raw_s_i_summary.csv").exists()
    assert (root / "histograms" / "offdiag_warning_summary.csv").exists()
    assert (root / "histograms" / "chunk_cka_null_completeness.csv").exists()
    assert (root / "histograms" / "random_pair_invalid_reason_summary.csv").exists()
    assert (root / "histograms" / "s_i_min_128_2.svg").exists()
    assert len(result["threshold_rows"]) == len(BUNDLES)
    assert len(result["same_layer_rows"]) == 2 * len(BUNDLES)

    chunk_summary = json.loads((root / "histograms" / "chunk_summary.json").read_text())
    assert chunk_summary["diagnostic_scales"] == [128, 256, 512]
    assert chunk_summary["selector_threshold_scales"] == [128, 256]
    assert int(chunk_summary["selection_chunk_layer_rows_by_scale"]["512"]) > 0
    with (root / "histograms" / "chunk_cka_null_summary.csv").open(newline="") as handle:
        null_rows = list(csv.DictReader(handle))
    scale512 = [row for row in null_rows if row["scale"] == "512"]
    assert len(scale512) == len(LAYERS)
    assert all(row["selector_threshold_role"] == "diagnostic_only_no_bundle_threshold" for row in scale512)
    assert all(not row["B95_threshold"] for row in scale512)
    assert all(int(row["permutation_null_count_seen"]) > 0 for row in scale512)
    assert all(int(row["random_pair_null_count_seen"]) > 0 for row in scale512)
    assert all(int(row["random_pair_null_missing_count"]) > 0 for row in scale512)
    with (root / "histograms" / "chunk_cka_null_completeness.csv").open(newline="") as handle:
        completeness = list(csv.DictReader(handle))
    assert len(completeness) == 2 * len(CHUNK_DIAGNOSTIC_SCALES) * len(LAYERS) * 2
    random_pair_groups = [row for row in completeness if row["null_kind"] == "random_pair"]
    assert all(int(row["missing_count"]) == 1 for row in random_pair_groups)
    assert all(float(row["missing_fraction"]) > 0 for row in random_pair_groups)
    permutation_groups = [row for row in completeness if row["null_kind"] == "permutation"]
    assert all(int(row["missing_count"]) == 0 for row in permutation_groups)
    with (root / "histograms" / "random_pair_invalid_reason_summary.csv").open(newline="") as handle:
        reason_rows = list(csv.DictReader(handle))
    assert len(reason_rows) == 2 * len(CHUNK_DIAGNOSTIC_SCALES) * len(LAYERS) * 4
    singleton_rows = [row for row in reason_rows if row["reason_code"] == "1"]
    assert all(int(row["count"]) == 1 for row in singleton_rows)
    valid_rows = [row for row in reason_rows if row["reason_code"] == "0"]
    assert all(int(row["count"]) == 79 for row in valid_rows)
    assert chunk_summary["raw_s_i_safety"]["negative"]["count"] > 0
    assert chunk_summary["raw_s_i_safety"]["explosion_warning"]["triggered"] is True
    assert chunk_summary["raw_s_i_safety"]["offdiag_warning"]["warning_count"] > 0

    counts = json.loads((root / "selector_comparison" / "selector_counts.json").read_text())
    assert counts["code"]["counts"]["matched_random"] == counts["code"]["counts"]["cka_plus_m95"]

    report = (root / "REPORT.md").read_text(encoding="utf-8")
    assert "REPORT v1 deliberately contains no test-derived selector count" in report
    assert "test_metrics.DO_NOT_OPEN_BEFORE_THRESHOLD_LOCK.json" not in report
    assert "s_i explosion candidates were observed" in report
    assert "runtime_router_smoke_metadata" in report
    assert "full_pre_analysis_deep_validator" in report
    assert '"status": "PASS"' in report
    assert "REPORT v1 does not enumerate sealed-test paths or counts" in report
    assert "singleton exact-length batch" in report
    assert "random_pair_invalid_reason_summary.csv" in report
    assert "| code | 2000 | 1000 | 3000 | 200 | 0.004 | 0.012 | 400 | 300 | 300 |" in report
    assert "| wiki | 2000 | 1000 | 3000 | 200 | 0.004 | 0.012 | 400 | 300 | 300 |" in report
    assert "legacy_in_split_reproduction" in pq.read_table(
        root / "selector_comparison" / "code_selection_assignments.parquet"
    ).schema.names

    sealed = root / "sealed_test" / "test_metrics.DO_NOT_OPEN_BEFORE_THRESHOLD_LOCK.json"
    manifest = json.loads((root / "sealed_test" / "manifest.json").read_text())
    assert sealed.exists()
    assert manifest["included_in_report_v1"] is False
    binding = json.loads((root / "analysis_input_binding.json").read_text())
    inventory = json.loads((root / "input_parquet_inventory.json").read_text())
    raw_manifest = json.loads(
        (root / "threshold_tables" / "raw_unit_quantile_accumulator_manifest.json").read_text()
    )
    assert binding["schema"] == "cka_gt_pilot_analysis_input_binding_v1"
    assert binding["status"] == "BOUND_COMPLETE"
    assert binding["analysis_outputs"]["report_validation_evidence"]["relative_path"] == "report_validation_evidence.json"
    assert binding["input_inventory"]["inventory_digest_sha256"] == inventory["inventory_digest_sha256"]
    assert raw_manifest["input_inventory_digest_sha256"] == inventory["inventory_digest_sha256"]

    token_table = pq.read_table(root / "token_metrics" / "part-00000.parquet")
    for domain in ("code", "wiki"):
        mask = np.asarray(token_table["domain"].to_pylist()) == domain
        uids = np.asarray(token_table["window_uid"])[mask]
        assert uids.size == 1_000
        assert np.unique(uids).size == 1_000


def test_context_enrichment_reads_document_bounded_plus_minus_32(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "pilot"
    context_dir = root / "context_samples"
    context_dir.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"code_prefix": "/synthetic/code"}))
    row = {
        "document_id": 0,
        "window_offset": 0,
        "position": 40,
        "document_token_offset": 40,
        "token_id": 1040,
    }
    (context_dir / "cka_plus_m95.jsonl").write_text(json.dumps(row) + "\n")

    class FakeDataset:
        def __init__(self, prefix: str):
            assert prefix == "/synthetic/code"
            self.sequence_lengths = np.asarray([100], dtype=np.int32)

        def get(self, document_id: int, offset: int, length: int) -> np.ndarray:
            assert document_id == 0
            return np.arange(1000 + offset, 1000 + offset + length, dtype=np.int32)

    import scripts.analysis.cka_gt_pilot_windows as windows
    import scripts.analysis.analyze_cka_gt_pilot as analyzer

    class FakeTokenizer:
        def decode(self, values, skip_special_tokens=False):
            return " ".join(str(value) for value in values)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str, local_files_only: bool):
            assert path == "/synthetic/local/pythia-12b"
            assert local_files_only is True
            return FakeTokenizer()

    monkeypatch.setattr(windows, "MMapIndexedDatasetLite", FakeDataset)
    monkeypatch.setattr(analyzer, "DEFAULT_LOCAL_TOKENIZER_PATH", Path("/synthetic/local/pythia-12b"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))
    result = _enrich_context_samples(root, root)
    enriched = json.loads((context_dir / "cka_plus_m95.jsonl").read_text())
    assert result["token_identity_verified"] is True
    assert enriched["context_token_ids"][0] == 1008
    assert enriched["context_token_ids"][-1] == 1072
    assert enriched["target_offset_in_context"] == 32
    assert enriched["context_token_ids"][32] == enriched["token_id"]
    assert result["decoded_text_available"] is True
    assert result["tokenizer_source"] == "runner_default_local_pythia12b_snapshot_fallback"
    assert enriched["context_tokenizer_source"] == result["tokenizer_source"]
    assert enriched["context_tokenizer_path"] == "/synthetic/local/pythia-12b"


def test_direct_script_context_import_works_from_unrelated_cwd(tmp_path: Path) -> None:
    """Regression test for the CLI's lazy repository-qualified import.

    ``runpy.run_path`` models execution by file path without placing the
    repository root on ``sys.path``.  Reaching the expected missing-dataset
    error proves that context enrichment imported the sibling window module;
    the historical failure instead raised ``No module named 'scripts'``.
    """

    analyzer_path = Path(__file__).with_name("analyze_cka_gt_pilot.py").resolve()
    artifact_root = tmp_path / "artifact"
    context_dir = artifact_root / "context_samples"
    context_dir.mkdir(parents=True)
    (artifact_root / "config.json").write_text(
        json.dumps({"code_prefix": str(tmp_path / "missing_dataset")}),
        encoding="utf-8",
    )
    probe = f"""
import runpy
from pathlib import Path

namespace = runpy.run_path({str(analyzer_path)!r})
try:
    namespace['_enrich_context_samples'](Path({str(artifact_root)!r}), Path({str(artifact_root)!r}))
except FileNotFoundError:
    pass
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_raw_quantile_table_is_built_exactly_from_raw_units(tmp_path: Path) -> None:
    root = tmp_path / "auto_quantiles"
    _write_synthetic(root, windows_per_domain=200, write_raw_quantiles=False)
    assert not (root / "threshold_tables" / "raw_unit_quantiles.parquet").exists()
    run_analysis(root, bins=200, batch_size=113)
    quantile_path = root / "threshold_tables" / "raw_unit_quantiles.parquet"
    manifest_path = root / "threshold_tables" / "raw_unit_quantile_accumulator_manifest.json"
    assert quantile_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["raw_s_source_column"] == "s_i"
    assert manifest["raw_s_used_token_min"] is False
    rows = pq.read_table(quantile_path).to_pylist()
    assert rows
    assert {row["method"] for row in rows} == {"exact_disk_backed"}
    assert any(
        row["domain"] == "wiki"
        and row["metric"] == "s_i"
        and row["scale"] == 128
        and row["layer"] == 2
        and row["quantile"] == 0.05
        for row in rows
    )
    token_rows = pq.read_table(root / "token_metrics" / "part-00000.parquet").to_pydict()
    token_domain = np.asarray(token_rows["domain"])
    token_split = np.asarray(token_rows["split"])
    cosine = np.asarray(token_rows["cosine"], dtype=np.float32)
    expected_cosine = float(
        np.quantile(cosine[(token_domain == "code") & (token_split == "selection"), 0], 0.99)
    )
    actual_cosine = next(
        row["value"]
        for row in rows
        if row["domain"] == "code"
        and row["split"] == "selection"
        and row["metric"] == "cosine"
        and row["layer"] == 2
    )
    assert actual_cosine == pytest.approx(expected_cosine, abs=1e-7)


def test_sealed_raw_test_tree_is_unioned_without_report_disclosure(tmp_path: Path) -> None:
    root = tmp_path / "sealed_union"
    _write_synthetic(root, windows_per_domain=100, seal_test_raw=True)
    run_analysis(root, bins=200, batch_size=61)
    sealed_metrics = json.loads(
        (root / "sealed_test" / "test_metrics.DO_NOT_OPEN_BEFORE_THRESHOLD_LOCK.json").read_text()
    )
    assert sealed_metrics["domains"]["code"]["total_tokens"] == 30
    assert sealed_metrics["domains"]["wiki"]["total_tokens"] == 30
    inventory = json.loads((root / "input_parquet_inventory.json").read_text())
    assert inventory["totals"]["sealed_test_token_metrics"]["rows"] == 60
    assert inventory["totals"]["sealed_test_chunk_metrics"]["rows"] > 0
    report = (root / "REPORT.md").read_text()
    assert "sealed_test/raw" not in report
    assert "test_metrics.DO_NOT_OPEN_BEFORE_THRESHOLD_LOCK.json" not in report


def test_changed_parquet_inventory_rebuilds_stale_quantiles_and_report_binding(tmp_path: Path) -> None:
    root = tmp_path / "inventory_rebind"
    _write_synthetic(root, windows_per_domain=100, write_raw_quantiles=False)
    first = run_analysis(root, bins=200, batch_size=67)
    first_digest = first["input_inventory_digest_sha256"]

    token_path = root / "token_metrics" / "part-00000.parquet"
    table = pq.read_table(token_path)
    cosine = np.asarray(table["cosine"].to_pylist(), dtype=np.float32)
    cosine[0, 0] -= 0.125
    table = table.set_column(table.schema.get_field_index("cosine"), "cosine", _fixed(cosine))
    pq.write_table(table, token_path, compression="zstd")

    second = run_analysis(root, bins=200, batch_size=67)
    second_digest = second["input_inventory_digest_sha256"]
    assert second_digest != first_digest
    assert second["raw_quantiles_rebuilt_for_current_inventory"] is True
    binding = json.loads((root / "analysis_input_binding.json").read_text())
    raw_manifest = json.loads(
        (root / "threshold_tables" / "raw_unit_quantile_accumulator_manifest.json").read_text()
    )
    assert binding["stale_prior_outputs_detected_and_rebuilt"] is True
    assert binding["input_inventory"]["inventory_digest_sha256"] == second_digest
    assert raw_manifest["input_inventory_digest_sha256"] == second_digest
    assert second_digest in (root / "REPORT.md").read_text()


def _smoke_main() -> None:
    parser = argparse.ArgumentParser(description="Run a canonical 1,000-window-per-domain CKA postprocess schema smoke")
    parser.add_argument("--synthetic-smoke-windows", type=int, default=1_000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()
    if args.synthetic_smoke_windows <= 0:
        raise SystemExit("--synthetic-smoke-windows must be positive")
    if args.output_root.exists() and not args.keep_existing:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_synthetic(args.output_root, windows_per_domain=args.synthetic_smoke_windows)
    result = run_analysis(args.output_root, bins=200, batch_size=257)
    print(
        json.dumps(
            {
                "complete": True,
                "windows_per_domain": args.synthetic_smoke_windows,
                "representative_tokens_per_window": 1,
                "limitation": "schema E2E only; no hidden forward",
                "report": result["report"],
                "matched_random_count": result["matched_random_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _smoke_main()
