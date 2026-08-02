import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_math import (  # noqa: E402
    gem_project_exact,
    gem_upstream_qpth_1d,
    grassmann_similarity,
    olora_paper_penalty,
    olora_upstream_penalty,
)


class GemProjectionTests(unittest.TestCase):
    def test_reference_projection_satisfies_constraints(self):
        gradient = np.array([-1.0, -2.0])
        memories = np.eye(2)
        projected = gem_project_exact(gradient, memories)
        np.testing.assert_allclose(projected, np.zeros(2), atol=1e-8)
        self.assertTrue(np.all(memories @ projected >= -1e-8))

    def test_trace_qpth_sign_translation_worsens_violation(self):
        upstream = gem_upstream_qpth_1d(-1.0, 1.0, margin=0.5)
        self.assertEqual(upstream, -2.0)
        self.assertLess(upstream * 1.0, 0.0)

    def test_nonconflicting_gradient_is_unchanged(self):
        gradient = np.array([2.0, 1.0])
        memories = np.array([[1.0, 0.0]])
        np.testing.assert_allclose(
            gem_project_exact(gradient, memories), gradient, atol=1e-8
        )


class SloraSimilarityTests(unittest.TestCase):
    def test_similarity_uses_candidate_rank_not_nullspace_padding(self):
        e1, e2, e3 = np.eye(3)
        base_c1 = e1[:, None]
        update_c1 = e1[:, None]
        self.assertEqual(grassmann_similarity(update_c1, base_c1), 1.0)

        # The public code pads a rank-1 candidate to the original LoRA rank.
        # Two arbitrary null-space completions then receive different scores.
        base_rank2 = np.column_stack([e1, e2])
        padded_aligned = np.column_stack([e1, e2])
        padded_rotated = np.column_stack([e1, e3])
        score_aligned = np.linalg.norm(padded_aligned.T @ base_rank2, ord="fro")
        score_rotated = np.linalg.norm(padded_rotated.T @ base_rank2, ord="fro")
        self.assertNotEqual(score_aligned, score_rotated)


class OloraPenaltyTests(unittest.TestCase):
    def test_upstream_can_miss_identical_update_column_spaces(self):
        old_a = np.array([[1.0, 0.0]])
        new_a = np.array([[0.0, 1.0]])
        old_b = np.array([[1.0], [0.0]])
        new_b = old_b.copy()
        self.assertEqual(olora_upstream_penalty([old_a], new_a), 0.0)
        self.assertGreater(olora_paper_penalty([old_b], new_b), 0.0)

    def test_paper_penalty_is_squared_frobenius(self):
        old_b = np.array([[1.0], [0.0]])
        new_b = np.array([[0.5], [np.sqrt(0.75)]])
        self.assertAlmostEqual(olora_paper_penalty([old_b], new_b), 0.25)


if __name__ == "__main__":
    unittest.main()
