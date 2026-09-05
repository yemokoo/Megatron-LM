import json
import os
import tempfile
import unittest

import numpy as np
import torch

from scripts.analysis.layer_output_streaming_core import (
    PairedStreamingStats,
    validate_or_write_manifest,
)


class PairedStreamingStatsTest(unittest.TestCase):
    def test_exact_blocks_moments_reservoir_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            stats = PairedStreamingStats(
                root, layers=(2, 9), hidden_size=3, block_tokens=4,
                target_tokens=8, reservoir_size=3, seed=7,
                metadata={"label": "unit"},
            )
            x = torch.arange(24, dtype=torch.float32).view(8, 3) / 10
            y = x + torch.tensor([1.0, -0.5, 0.25])
            tokens = torch.arange(8, dtype=torch.int64)
            positions = torch.arange(8, dtype=torch.int64)
            stats.record_samples(tokens.view(2, 4), torch.ones(2, 4))
            stats.update({2: x[:5], 9: 2 * x[:5]}, {2: y[:5], 9: 2 * y[:5]},
                         tokens[:5], positions[:5])
            stats.update({2: x[5:], 9: 2 * x[5:]}, {2: y[5:], 9: 2 * y[5:]},
                         tokens[5:], positions[5:])
            metadata = stats.finalize()
            self.assertEqual(metadata["target_tokens"], 8)
            self.assertEqual(len(metadata["blocks"]), 2)
            with np.load(os.path.join(root, "block_000.npz")) as block:
                np.testing.assert_allclose(block["sum_x"][0], x[:4].sum(0).numpy())
                np.testing.assert_allclose(block["xx"][0], (x[:4].T @ x[:4]).numpy(), rtol=1e-6)
                np.testing.assert_allclose(block["xy"][0], (x[:4].T @ y[:4]).numpy(), rtol=1e-6)
            with np.load(os.path.join(root, "delta_reservoir.npz")) as reservoir:
                self.assertEqual(reservoir["delta"].shape, (2, 3, 3))
                selected = reservoir["global_indices"]
                np.testing.assert_allclose(
                    reservoir["delta"][0].astype(np.float32),
                    (y - x)[selected].numpy(), atol=1e-3,
                )
            manifest = os.path.join(root, "manifest.json")
            validate_or_write_manifest(metadata, manifest)
            validate_or_write_manifest(metadata, manifest)
            changed = json.loads(json.dumps(metadata))
            changed["blocks"][0]["sha256"] = "bad"
            with self.assertRaises(RuntimeError):
                validate_or_write_manifest(changed, manifest)


if __name__ == "__main__":
    unittest.main()
