# QV LoRA Routing Eval

This directory contains the routing-profile evaluation pipeline for the QV LoRA attention model.

## Purpose

The goal is to evaluate a single QV LoRA checkpoint on multiple datasets and measure:

- which LoRA expert each token routes to at each layer
- how much routing stays in the old expert group vs the newly added expert group
- how routing differs between wiki test and code test

Unlike the older MoE compare pipeline, the current QV LoRA router is top-1 only.
That means:

- `old4 only` is meaningful
- `new3 only` is meaningful
- `mixed old4+new3` is currently always `0`

The plotting code still keeps the same stacked-bar format so the output stays visually comparable to the earlier routing figures.

## Files

- `dump_qv_lora_routing_eval.py`
  Loads one checkpoint, runs eval batches, hooks the QV LoRA router, and writes per-layer routing summary JSON.
- `plot_qv_lora_routing.py`
  Builds per-dataset and wiki-vs-code comparison plots from the dumped summaries.
- `a100/run_qv_lora_routing_profile_bf16.sh`
  A100-oriented bf16 runner that evaluates one checkpoint on wiki test and code test.

## Default flow

1. Resolve the model run directory and checkpoint load directory.
2. Read `source_num_experts` from `logs/run_metadata.json` when available.
3. Run routing eval on:
   - wiki test
   - code test
4. Save:
   - `wiki_test/summary.json`
   - `code_test/summary.json`
   - `comparison/wiki_expert_group_usage.png`
   - `comparison/code_expert_group_usage.png`
   - `comparison/expert_group_usage_comparison.png`
   - `comparison/new_expert_usage_comparison.png`
   - `comparison/summary.json`

## Dynamic dataset size

The A100 runner supports:

- `TARGET_EVAL_TOKENS`
- `TARGET_EVAL_SAMPLES`
- `EVAL_ITERS`

Priority is:

1. `TARGET_EVAL_SAMPLES`
2. `TARGET_EVAL_TOKENS`
3. `EVAL_ITERS`

This makes it easy to run smaller routing probes without scanning the full test split.
