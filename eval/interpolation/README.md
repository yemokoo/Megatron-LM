# Task B Interpolation Eval

This directory contains the evaluation code for checkpoint interpolation on the
Task B dataset (`python-code-full`).

## Purpose

The experiment linearly interpolates between:

- a source checkpoint
- a target checkpoint

and evaluates each interpolation point on Task B using:

- `next_token_acc`
- `ppl`

This was mainly written for the `A -> B` and `A -> B new-only` continual
learning checkpoints.

## Files

- `eval_interpolated_checkpoint.py`
  Evaluates one interpolation sweep for a single source/target pair.
- `run_task_b_interpolation_suite.sh`
  Runs the interpolation sweep for the prepared Task B experiments and writes
  result files.

## Current behavior

The suite expands the source `A` checkpoint from `4 -> 7 experts` before
interpolation so it can be compared against `7-expert` continual models.

For each pair it writes:

- `metrics.csv`
- `metrics.json`
- `next_token_accuracy.svg`
- `ppl.svg`

under the pair-specific output directory.

## Default experiment targets

The runner is configured around these checkpoints by default:

- `continual-stage-A/stage-a-local-fp32-20260310-103742`
- `continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347`
- `continual-stage-A-to-B-new-only/...`

Note:
- the `A -> B resume` path is usable directly
- the `A -> B new-only` path depends on the new-only checkpoint being saved as a
  real `7-expert` checkpoint

## Example

```bash
docker exec flame-moe-3090 bash -lc '
  cd /workspace/FLAME-MoE &&
  PATH=/workspace/FLAME-MoE/.conda/envs/flame3090/bin:$PATH \
  GPU_DEVICE=6 \
  bash eval/interpolation/run_task_b_interpolation_suite.sh
'
```
