import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PortableLayoutTests(unittest.TestCase):
    def test_supported_entrypoints_have_no_previous_server_paths(self):
        files = [
            ROOT / "scripts/baselines/_run_ours_lora_moe.sh",
            ROOT / "scripts/baselines/_run_loramoe.sh",
            ROOT / "scripts/data/prepare_llama31_trace_cache.sh",
            ROOT / "implementations/SLoRA-repro/scripts/repro/train_trace.sh",
            ROOT / "implementations/SLoRA-repro/scripts/repro/eval_trace.sh",
            ROOT / "implementations/TRACE-repro/scripts/repro/train_trace.sh",
            ROOT / "implementations/TRACE-repro/scripts/repro/eval_trace.sh",
        ]
        forbidden = (
            "/home/work/Agent_HJ",
            "/30_flame_agent/TRACE",
            "/30_flame_agent/slora_repro",
            "/30_flame_agent/llmcl_benchmark",
            "envs/train_env",
        )
        for path in files:
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"{path}: {value}")

    def test_registry_paths_are_project_relative(self):
        for name in ("models.json", "trace_experiments.json", "experiment_matrix.json"):
            text = (ROOT / "config" / name).read_text(encoding="utf-8")
            self.assertNotIn("/home/work/", text)

    def test_replay_manifest_is_portable_and_complete(self):
        path = ROOT / "manifests/replay/trace_seed2025_random50_per_task.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["samples_per_task"], 50)
        self.assertEqual(set(manifest["task_order"]), set(manifest["tasks"]))
        for entry in manifest["tasks"].values():
            self.assertFalse(Path(entry["source_path"]).is_absolute())
            self.assertEqual(len(entry["indices"]), 50)

    def test_vendored_lora_module_starts_with_valid_import(self):
        path = ROOT / "implementations/llmcl_benchmark/model/lora.py"
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(first_line, "from model.base_model import CL_Base_Model")


if __name__ == "__main__":
    unittest.main()
