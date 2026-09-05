#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_old_like_gt_l2_l9_top1 import THRESHOLDS, config_hash, scan_rank


class TestOldLikeGT(unittest.TestCase):
    def test_all_layers_and_bit_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "source"
            rank = root / "rank_000"
            metrics = rank / "token_metrics"
            metrics.mkdir(parents=True)
            with (rank / "metadata.json").open("w") as handle:
                json.dump({
                    "partition_start_sample": 10, "partition_samples": 2, "shards": 1,
                }, handle)
            cosine = np.ones((2, 512, 9), dtype=np.float32)
            # Exact threshold qualifies; one failing layer rejects only this occurrence.
            cosine[0, 3, 1:] = THRESHOLDS
            cosine[1, 7, 1:] = THRESHOLDS
            cosine[1, 7, 4] = np.nextafter(THRESHOLDS[3], np.float32(-np.inf))
            valid = np.ones((2, 512), dtype=np.uint8)
            valid[0, 9] = 0
            np.savez(metrics / "shard_000000.npz",
                     layer_numbers=np.arange(1, 10), sample_ids=np.array([10, 11]),
                     valid_mask=valid, cosine=cosine)
            output = Path(td) / "output"
            digest = config_hash(root)
            row = scan_rank(str(rank), str(output), digest, False)
            packed = np.load(output / "rank_000" / "old_like_gt_packed.npy")
            mask = np.unpackbits(packed, axis=1, bitorder="little")
            self.assertTrue(mask[0, 3])
            self.assertFalse(mask[1, 7])
            self.assertFalse(mask[0, 9])
            self.assertEqual(row["selected_count"], 1022)
            self.assertEqual(mask.shape, (2, 512))
            # Completed rerun is a cache hit with identical identity.
            cached = scan_rank(str(rank), str(output), digest, False)
            self.assertEqual(cached["mask_sha256"], row["mask_sha256"])


if __name__ == "__main__":
    unittest.main()
