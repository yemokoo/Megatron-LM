from __future__ import annotations

import unittest

import numpy as np

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover - lightweight environments omit Arrow
    pa = None

import validate_cka_gt_pilot as validator


@unittest.skipIf(pa is None, "pyarrow is required")
class ValidatorVectorizationTest(unittest.TestCase):
    def test_fixed_list_fast_path_handles_sliced_arrow_offsets(self) -> None:
        expected = np.arange(12 * 8, dtype=np.float32).reshape(12, 8)
        child = pa.array(expected.reshape(-1), type=pa.float32())
        values = pa.FixedSizeListArray.from_arrays(child, 8).slice(3, 5)
        table = pa.table({"metric": values})
        actual = validator._list_numpy(table, "metric", 8, np.float32)
        np.testing.assert_array_equal(actual, expected[3:8])

    def test_variable_list_uses_compatible_fallback(self) -> None:
        expected = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        table = pa.table({"metric": pa.array(expected.tolist(), type=pa.list_(pa.float32()))})
        actual = validator._list_numpy(table, "metric", 8, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_nested_router_fast_path_and_duplicate_check(self) -> None:
        ids = np.tile(np.arange(4, dtype=np.int16), (6, 8, 1))
        leaf = pa.array(ids.reshape(-1), type=pa.int16())
        inner = pa.FixedSizeListArray.from_arrays(leaf, 4)
        outer = pa.FixedSizeListArray.from_arrays(inner, 8)
        table = pa.table({"before_top4_ids": outer})
        actual = validator._nested_router_numpy(
            table, "before_top4_ids", np.int64
        )
        np.testing.assert_array_equal(actual, ids)

        valid_report = validator.ValidationReport("fixture", False, True)
        validator._validate_token_batch(table, valid_report, "fixture")
        self.assertFalse(valid_report.errors)

        duplicate = ids.copy()
        duplicate[4, 6, 3] = duplicate[4, 6, 2]
        leaf = pa.array(duplicate.reshape(-1), type=pa.int16())
        inner = pa.FixedSizeListArray.from_arrays(leaf, 4)
        duplicate_table = pa.table(
            {"before_top4_ids": pa.FixedSizeListArray.from_arrays(inner, 8)}
        )
        bad_report = validator.ValidationReport("fixture", False, True)
        validator._validate_token_batch(duplicate_table, bad_report, "fixture")
        self.assertTrue(
            any("duplicate top-4 IDs" in error for error in bad_report.errors),
            bad_report.errors,
        )


if __name__ == "__main__":
    unittest.main()
