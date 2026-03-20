# Routing Similarity Eval

This directory contains the routing-similarity evaluation pipeline used to compare:

- `a-base` vs `a-to-b` (`shared` trainable)
- `a-base` vs `a-to-b-new-only` (`shared` frozen)
- `b-base` vs `b-to-a` (`shared` trainable)
- `b-base` vs `b-to-a-new-only` (`shared` frozen)

## Canonical Eval Protocol

As of March 18, 2026, the canonical protocol is:

- Model checkpoint: always `iter_0001800`
- Train datasets:
  - Wiki task train: `/workspace/FLAME-MoE/.local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact`
  - Code task train: `/workspace/FLAME-MoE/.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact`
- Eval datasets:
  - Wiki task test: `/workspace/FLAME-MoE/.local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact`
  - Code task test: `/workspace/FLAME-MoE/.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact`
- Dataset sizes:
  - Train tokens: `2,123,366,400` (`1800 * 2304 * 512`)
  - Wiki test tokens: `203,881,952` (all remaining tokens after the train cut)
  - Code test tokens: `203,881,952` (matched to wiki test size, remainder discarded)
- Because each dataset directory is already a physically split dataset:
  - `DATASET_SPLIT=100,0,0`
  - `DATASET_SPLIT_NAME=train`
  - `CONSUMED_SAMPLES=0`

## Outputs

Each compare run produces:

- `comparison/summary.json`
- `comparison/routing_overlap.png`
- `comparison/new_expert_usage.png`
- `comparison/expert_group_usage.png`
- `comparison/new_group_slot_share.png`

The grouped plots are the main way to compare old-vs-new expert usage:

- `fraction_all_old_group_in_b`
- `fraction_mixed_old_new_group_in_b`
- `fraction_all_new_group_in_b`
- `mean_new_group_slot_fraction_in_b`

## Recommended Runs

Use one shared parent output root, for example:

- `/workspace/FLAME-MoE/.local/weights/routing-compare-suite-exact1800`

and create four subdirectories:

- `a-base-vs-a-to-b-full`
- `a-base-vs-a-to-b-new-only`
- `b-base-vs-b-to-a-full`
- `b-base-vs-b-to-a-new-only`

All four runs should set:

```bash
TARGET_EVAL_SAMPLES=0
TARGET_EVAL_TOKENS=203881952
DATASET_SPLIT=100,0,0
DATASET_SPLIT_NAME=train
CONSUMED_SAMPLES=0
BASE_ITERATION=1800
TARGET_ITERATION=1800
KEEP_RAW_DUMPS=1
```

## Notes

- `a-to-b` compares against Task A (`wiki`) holdout.
- `a-to-b-new-only` compares against Task A (`wiki`) holdout.
- `b-to-a` compares against Task B (`code`) holdout.
- `b-to-a-new-only` compares against Task B (`code`) holdout.
- `new-only` means the old experts/router rows were frozen during training while newly added experts/router rows were trainable.
