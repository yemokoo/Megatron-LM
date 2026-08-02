import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "relog_expansion_distill_pipeline",
    ROOT / "scripts" / "analysis" / "relog_expansion_distill_pipeline_to_wandb.py",
)
relog = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = relog
spec.loader.exec_module(relog)


def valid_argv():
    return [
        "relog_expansion_distill_pipeline_to_wandb.py",
        "--teacher-dir",
        ".",
        "--distill-dir",
        ".",
        "--code-dir",
        ".",
        "--retune-dir",
        ".",
        "--run-id",
        "test-run",
        "--run-name",
        "test run",
    ]


def test_required_directory_rejects_empty_shell_expansion(monkeypatch, capsys):
    argv = valid_argv()
    argv[argv.index("--teacher-dir") + 1] = ""
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        relog.parse_args()

    assert exc_info.value.code == 2
    assert "path must not be empty" in capsys.readouterr().err


def test_explicit_log_rejects_empty_shell_expansion(monkeypatch, capsys):
    argv = valid_argv() + ["--distill-log", ""]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        relog.parse_args()

    assert exc_info.value.code == 2
    assert "path must not be empty" in capsys.readouterr().err
