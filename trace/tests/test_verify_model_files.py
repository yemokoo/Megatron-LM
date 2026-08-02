import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from verify_model_files import verify_model  # noqa: E402


def write_fixture(root: Path, truncate: bool = False) -> None:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
        separators=(",", ":"),
    ).encode()
    padding = b" " * ((8 - len(header) % 8) % 8)
    header += padding
    payload = len(header).to_bytes(8, "little") + header + (b"\0" * 8)
    if truncate:
        payload = payload[:-1]
    (root / "model-00001-of-00001.safetensors").write_bytes(payload)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {"weight": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class ModelFileVerificationTests(unittest.TestCase):
    def test_complete_indexed_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root)
            report = verify_model(root)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(report["verified_tensor_bytes"], 8)

    def test_truncated_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, truncate=True)
            report = verify_model(root)
            self.assertFalse(report["ok"])
            self.assertTrue(any("expected" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
