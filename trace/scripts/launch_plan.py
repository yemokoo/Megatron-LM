#!/usr/bin/env python3
"""Render a provenance-rich launch plan without starting training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    profiles = read(ROOT / "config" / "run_profiles.json")["profiles"]
    if args.profile not in profiles:
        parser.error(f"unknown profile; choose one of: {', '.join(sorted(profiles))}")
    profile = profiles[args.profile]
    if profile["classification"].startswith("smoke_only"):
        payload = {
            "profile": args.profile,
            **profile,
            "command": [str(ROOT / "scripts" / "smoke.sh")],
            "executable_now": True,
        }
    else:
        models = read(ROOT / "config" / "models.json")["models"]
        model = models[profile["model_registry_key"]]
        model_path = args.model_path or Path(model["planned_path"])
        ready = (model_path / "config.json").is_file() and bool(
            list(model_path.glob("*.safetensors")) + list(model_path.glob("*.index.json"))
        )
        payload = {
            "profile": args.profile,
            **profile,
            "model_hf_id": model["hf_id"],
            "model_path": str(model_path),
            "model_ready": ready,
            "executable_now": False,
            "reason": (
                "render-only by design; apply the listed patches in a disposable "
                "run tree, install the tested full environment, and authorize the "
                "separate model phase before execution"
            ),
            "command_template": [
                "torchrun",
                "--nproc_per_node=<WORLD_SIZE>",
                profile["entrypoint"],
                "--model_name_or_path",
                str(model_path),
                "--train_data_path",
                "<VERIFIED_TASK_TRAIN_JSON>",
                "--output_dir",
                "<PROVENANCE_TAGGED_OUTPUT>",
            ],
        }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(args.output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
