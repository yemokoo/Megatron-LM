#!/usr/bin/env python3
"""Wait for the remaining G2 objective runs and clone them to the target project."""

import subprocess
import sys
import time
from pathlib import Path

import wandb


ENTITY = "yemoyemo010831-korea-university"
SOURCE_PROJECT = "flame-continual"
TARGET_PROJECT = "flame-continual-top2-qv-lora"
RUN_IDS = (
    "g2-3objective-hidden-mse-c10-l2to9-probe3i100",
    "g2-3objective-vocab-kl-c10-probe3i100",
)


def target_exists(api: wandb.Api, run_id: str) -> bool:
    try:
        api.run(f"{ENTITY}/{TARGET_PROJECT}/{run_id}-offset1800")
    except wandb.errors.CommError:
        return False
    return True


def main() -> None:
    clone_script = Path(__file__).with_name("clone_wandb_run.py")
    for run_id in RUN_IDS:
        while True:
            api = wandb.Api(timeout=60)
            if target_exists(api, run_id):
                print(f"[SKIP] target already exists: {run_id}", flush=True)
                break
            try:
                source = api.run(f"{ENTITY}/{SOURCE_PROJECT}/{run_id}")
            except wandb.errors.CommError:
                print(f"[WAIT] source has not started: {run_id}", flush=True)
                time.sleep(60)
                continue
            print(f"[CHECK] {run_id}: state={source.state} step={source.lastHistoryStep}", flush=True)
            if source.state != "finished":
                time.sleep(60)
                continue
            subprocess.run(
                [
                    sys.executable,
                    str(clone_script),
                    "--entity", ENTITY,
                    "--source-project", SOURCE_PROJECT,
                    "--target-project", TARGET_PROJECT,
                    "--run-id", run_id,
                    "--target-run-id", f"{run_id}-offset1800",
                    "--step-offset", "1800",
                ],
                check=True,
            )
            break


if __name__ == "__main__":
    main()
