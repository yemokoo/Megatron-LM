# Mask Extra Router Eval

This directory contains the evaluation code for the router-masking experiment.

## Purpose

The question is:

After continual learning on Task 2, what happens to Task 1 performance if the
newly added Task-2-side experts are completely masked out at routing time?

The masking is applied inside the router logits before routing:

- experts with index `>= 4` are set to `-inf`
- they cannot enter softmax/top-k

This lets us compare:

- Task1-only baseline
- continual model
- continual model with extra experts masked

using the canonical post-step-1800 exact test splits.

## Files

- `eval_masked_router_acc.py`
  Evaluates one checkpoint and optionally masks experts `>= source_num_experts`.
- `plot_mask_extra_router_results.py`
  Builds the summary JSON and SVG charts from per-model JSON outputs.
- `run_mask_extra_router_suite.sh`
  Runs the full suite for the default checkpoints.

## Canonical Protocol

- evaluate checkpoints at `iter_0001800`
- use the physically split exact test datasets:
  - wiki: `.../pythia-12b-step1800-test-fullremainder-exact`
  - code: `.../pythia-12b-step1800-test-matchwiki-exact`
- treat each of those directories as the full dataset:
  - `DATASET_SPLIT=100,0,0`
  - `DATASET_SPLIT_NAME=train`
  - `CONSUMED_SAMPLES=0`
- by default the suite evaluates `10M tokens`

## Default checkpoints

Task1-only baselines:

- `continual-stage-A/stage-a-local-fp32-20260310-103742`
- `continual-stage-B/stage-b-first-local-fp32-20260312-043202`

Continual models:

- `continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347` at `iter_1800`
- `continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542`
- `continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645`
- `continual-stage-B-to-A-new-only/stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509`

## Outputs

The suite writes:

- per-model JSON metric files
- `summary.json`
- `task1_accuracy_comparison.svg`
- `task1_accuracy_a_to_b.svg`
- `task1_accuracy_a_to_b_new_only.svg`
- `task1_accuracy_b_to_a.svg`
- `task1_accuracy_b_to_a_new_only.svg`

The overview chart contains all four experiments together.
The per-model charts split them into separate images for easier comparison.

## Example

```bash
docker exec flame-moe-3090 bash -lc '
  cd /workspace/FLAME-MoE &&
  PATH=/workspace/FLAME-MoE/.conda/envs/flame3090/bin:$PATH \
  GPU_DEVICE=2 \
  TARGET_EVAL_TOKENS=10000000 \
  MICRO_BATCH_SIZE=24 \
  OUTPUT_ROOT=/workspace/FLAME-MoE/.local/weights/mask-extra-router-suite-exact1800 \
  bash eval/mask_extra_router/run_mask_extra_router_suite.sh
'
```
