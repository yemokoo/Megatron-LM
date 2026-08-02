import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mock_pipeline import (  # noqa: E402
    denoise_update,
    greedy_generate,
    load_adapter,
    merge_adapter,
    metric_dispatch,
    one_optimizer_step,
    save_adapter,
)


class MockPipelineTests(unittest.TestCase):
    def test_optimizer_adapter_save_reload_and_merge(self):
        base = np.eye(2)
        updated = one_optimizer_step(
            base, np.eye(2), np.zeros((2, 2)), lr=0.1
        )
        self.assertFalse(np.array_equal(updated, base))
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0], [4.0]])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "adapter.npz"
            save_adapter(path, a, b)
            loaded_a, loaded_b = load_adapter(path)
        np.testing.assert_array_equal(loaded_a, a)
        np.testing.assert_array_equal(loaded_b, b)
        np.testing.assert_array_equal(merge_adapter(base, a, b), base + b @ a)

    def test_denoising_generation_and_metric_dispatch(self):
        update = np.diag([3.0, 2.0, 1.0])
        denoised = denoise_update(update, rank=1)
        self.assertEqual(np.linalg.matrix_rank(denoised), 1)
        self.assertEqual(greedy_generate(np.array([[0.0, 2.0], [4.0, 1.0]])), [1, 0])
        for metric in ("accuracy", "rouge_l", "edit_similarity", "sari"):
            self.assertEqual(metric_dispatch(metric, "same text", "same text"), 1.0)


if __name__ == "__main__":
    unittest.main()
