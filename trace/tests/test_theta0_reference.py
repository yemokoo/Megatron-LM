import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "implementations" / "SLoRA-repro"))

from src.model.builder import denoising, snapshot_reference_weights  # noqa: E402


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.other = nn.Linear(2, 2, bias=False)


class FixedThetaZeroTests(unittest.TestCase):
    def test_snapshot_is_filtered_and_immutable(self):
        model = ToyModel()
        expected = model.q_proj.weight.detach().clone()
        snapshot = snapshot_reference_weights(model)

        self.assertEqual(set(snapshot), {"q_proj.weight"})
        with torch.no_grad():
            model.q_proj.weight.add_(10)
        torch.testing.assert_close(snapshot["q_proj.weight"], expected)

    def test_denoising_accepts_fixed_reference_mapping(self):
        model = ToyModel()
        snapshot = snapshot_reference_weights(model)
        state = {
            "base_model.model.q_proj.lora_A.weight": torch.tensor([[1.0, 0.0]]),
            "base_model.model.q_proj.lora_B.weight": torch.tensor([[1.0], [0.0]]),
        }

        output = denoising(snapshot, state, mode="max")
        self.assertEqual(set(output), set(state))
        self.assertEqual(output[state_key("A")].shape, (1, 2))
        self.assertEqual(output[state_key("B")].shape, (2, 1))


def state_key(matrix: str) -> str:
    return f"base_model.model.q_proj.lora_{matrix}.weight"


if __name__ == "__main__":
    unittest.main()
