#!/usr/bin/env python3
import argparse
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5400)
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.project,
        name=args.run_name,
        id=args.run_id,
        resume="never",
        dir=str(args.save_dir),
        config={"kd_conversation_stepwise_baseline": True},
    )
    try:
        wandb.log(
            {
                "conversation_probe/next_token_accuracy": 0.288416,
                "conversation_probe/ppl": 65.47378,
                "code_probe/next_token_accuracy": 0.658937,
                "code_probe/ppl": 6.432723,
                "wiki_probe/next_token_accuracy": 0.462014,
                "wiki_probe/ppl": 18.00381,
            },
            step=args.step,
        )
    finally:
        run.finish()
    print(f"Logged all six KD-final probe metrics at step {args.step}.")


if __name__ == "__main__":
    main()
