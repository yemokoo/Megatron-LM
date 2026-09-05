from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analysis.analyze_code_token_cross_layer_stability import (
    _candidate_empty,
    _candidate_merge,
    _pattern_derived,
    build_percentile_lookup,
    cosine_to_percentiles,
    membership_pattern_counts,
    reference_rms_proxy,
)


def test_percentile_lookup_is_midrank_and_monotonic(tmp_path: Path) -> None:
    counts = np.zeros((9, 20_000), dtype=np.uint64)
    counts[:, 10_000] = 2
    counts[:, 15_000] = 2
    edges = np.linspace(-1.0, 1.0, 20_001)
    path = tmp_path / "counts.npz"
    np.savez(path, counts=counts, bin_edges=edges, layer_numbers=np.arange(1, 10))
    lookup, metadata = build_percentile_lookup(path)
    assert lookup.shape == (8, 20_000)
    assert lookup[0, 10_000] == 0.25
    assert lookup[0, 15_000] == 0.75
    cosine = np.asarray([[0.0] * 8, [0.5] * 8], dtype=np.float32)
    percentiles = cosine_to_percentiles(cosine, lookup)
    np.testing.assert_allclose(percentiles[:, 0], [0.25, 0.75])
    assert metadata["tokens_per_layer"] == 4


def test_membership_patterns_and_stable_counts() -> None:
    percentiles = np.asarray(
        [
            [1.0] * 8,
            [0.995, 0.995, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 8,
        ],
        dtype=np.float32,
    )
    patterns, stable = membership_pattern_counts(percentiles)
    assert stable[0].tolist() == [8, 8, 8, 8]
    assert stable[1].tolist() == [2, 2, 2, 2]
    assert stable[2].tolist() == [0, 0, 0, 0]
    assert patterns[0, 255] == 1
    assert patterns[0, 3] == 1
    assert patterns[0, 0] == 1
    derived = _pattern_derived(patterns[0])
    assert derived["total"] == 3
    assert derived["stable_count_hist"].tolist() == [1, 0, 1, 0, 0, 0, 0, 0, 1]
    assert derived["intersections"][0, 1] == 2
    assert derived["late_l7_l9_all"] == 1


def test_reference_rms_proxy_recovers_true_rms() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(32, 8, 64)).astype(np.float32)
    current = reference + rng.normal(scale=0.2, size=reference.shape).astype(np.float32)
    delta = current - reference
    reference_norm = np.linalg.norm(reference, axis=-1)
    delta_norm = np.linalg.norm(delta, axis=-1)
    relative = delta_norm / (reference_norm + 1e-8)
    delta_mse = np.mean(delta * delta, axis=-1)
    proxy, valid = reference_rms_proxy(delta_mse, relative)
    expected = reference_norm / np.sqrt(reference.shape[-1])
    assert valid.all()
    np.testing.assert_allclose(proxy, expected, rtol=2e-6, atol=1e-7)


def test_candidate_merge_keeps_smallest_priorities() -> None:
    def make(priorities: list[int]):
        result = _candidate_empty()
        count = len(priorities)
        result["priority"] = np.asarray(priorities, dtype=np.uint64)
        result["sample_ids"] = np.arange(count, dtype=np.int64)
        result["positions"] = np.zeros(count, dtype=np.uint16)
        result["token_ids"] = np.arange(count, dtype=np.int32)
        result["cosine"] = np.zeros((count, 8), dtype=np.float32)
        result["percentile"] = np.zeros((count, 8), dtype=np.float32)
        result["stable_counts"] = np.zeros((count, 4), dtype=np.uint8)
        result["relative_l2"] = np.zeros((count, 8), dtype=np.float32)
        result["log_norm_ratio"] = np.zeros((count, 8), dtype=np.float32)
        result["reference_rms"] = np.ones((count, 8), dtype=np.float32)
        return result

    merged = _candidate_merge(make([9, 3, 7]), make([2, 8, 1]), 4)
    assert merged["priority"].tolist() == [1, 2, 3, 7]
