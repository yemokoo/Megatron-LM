import json
import os
import tempfile
import unittest

import numpy as np
import torch

from scripts.analysis.fingerprint_score_streaming_core import FingerprintScoreStats


class FingerprintScoreStatsTest(unittest.TestCase):
    def test_exact_blocks_hist_bias_and_reservoir(self):
        with tempfile.TemporaryDirectory() as root:
            stats = FingerprintScoreStats(
                root,
                representation_names=("stable", "pca_top", "random"),
                ranks=(16, 32, 64),
                selector_names=("layer_2", "layer_5", "layer_9", "mean"),
                target_tokens=8,
                block_tokens=4,
                vocab_size=32,
                sequence_length=4,
                reservoir_size=3,
                bins=16,
                seed=7,
                metadata={"label": "unit"},
            )
            tokens = torch.arange(8).view(2, 4)
            mask = torch.ones_like(tokens, dtype=torch.float32)
            stats.record_samples(tokens, mask)
            scores = torch.linspace(0, 1, 3 * 3 * 4 * 8).view(3, 3, 4, 8)
            stats.update(tokens.reshape(-1)[:5], torch.arange(5) % 4, scores[..., :5])
            stats.update(tokens.reshape(-1)[5:], torch.arange(5, 8) % 4, scores[..., 5:])
            metadata = stats.finalize()
            self.assertEqual(metadata["target_tokens"], 8)
            self.assertEqual(len(metadata["blocks"]), 2)
            with np.load(os.path.join(root, "block_000.npz")) as block:
                self.assertEqual(int(block["count"]), 4)
                self.assertTrue(np.all(block["hist"].sum(axis=-1) == 4))
                self.assertEqual(int(block["token_counts"].sum()), 4)
                np.testing.assert_allclose(
                    block["score_sum"], scores[..., :4].sum(-1).numpy(), rtol=1e-6
                )
            with np.load(os.path.join(root, "score_reservoir.npz")) as reservoir:
                self.assertEqual(reservoir["scores"].shape, (3, 3, 4, 3))
                selected = reservoir["global_indices"]
                np.testing.assert_allclose(
                    reservoir["scores"].astype(np.float32),
                    scores[..., selected].numpy(),
                    atol=5e-4,
                )
            with open(os.path.join(root, "metadata.json"), encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["reservoir_size"], 3)


if __name__ == "__main__":
    unittest.main()
