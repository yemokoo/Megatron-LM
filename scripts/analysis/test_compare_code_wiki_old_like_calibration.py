from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from compare_code_wiki_old_like_calibration import _auc, _pattern_to_stable_counts, _threshold_rows


def test_binned_auc_perfect_and_tied() -> None:
    assert _auc(np.array([0, 0, 4]), np.array([4, 0, 0])) == 1.0
    assert _auc(np.array([4, 0, 0]), np.array([0, 0, 4]), higher_is_positive=False) == 1.0
    assert _auc(np.array([0, 4, 0]), np.array([0, 4, 0])) == 0.5


def test_pattern_counts_map_to_stable_count() -> None:
    patterns = np.zeros(256, dtype=np.uint64)
    patterns[0] = 2
    patterns[0b00000011] = 3
    patterns[255] = 5
    counts = _pattern_to_stable_counts(patterns)
    assert counts.tolist() == [2, 0, 3, 0, 0, 0, 0, 0, 5]


def test_threshold_curve_uses_balanced_prior() -> None:
    rows = _threshold_rows("x", np.array([0, 10]), np.array([10, 0]), 0.0, 1.0)
    strict = rows[-1]
    assert strict["wiki_recall_tpr"] == 1.0
    assert strict["code_fpr"] == 0.0
    assert strict["balanced_prior_precision"] == 1.0
