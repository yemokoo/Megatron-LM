# Ours + residual expert — exact recipe (as run 2026-09-15/16, reported gen_res / rep_res)

All paths below assume the run layout of this repo's launchers; adapt `RUN`/`OUT` roots.
Python = `trace/.venv-runtime/bin/python`, cwd = `trace/implementations/llmcl_benchmark`
unless noted. Scripts: `scripts/residual/{train_router_residual.py, residual_expert.py,
eval_trace_router_tuned.py, gen_backbone_bos.py}`; the shell launchers actually used are
copied verbatim under `scripts/residual/launchers/` (paths inside are the 09-15 run dirs).

## 0. Inputs
* Source checkpoint: finished V3 round 7 (`<run>/7`, has `lora_moe_meta.json`). Works for any
  rank — the loader reads r/alpha from the meta.
* TRACE replay, 500/task, 8 tasks (4,000):
  * **rep**: `real:<run>/fixed_replay_memory` (the run's own persistent-memory indices -> train.json)
  * **gen**: `gen:C-STANCE=<gen>/round_7/C-STANCE/records.jsonl,...,NumGLUE-ds=...,20Minuten=<new>`
    — round_7 of the selfgen run has the 7 earlier tasks; 20Minuten must be generated (step 1).
* Backbone BoS: `scripts/residual/assets/backbone_bos_500.jsonl` (first 500 of the 4,000 generated
  from plain Llama-3.1-8B-Instruct with `gen_backbone_bos.py`: prompt = `<|begin_of_text|>` only,
  T=1.0, top_p 0.95, max 512 new tokens, vLLM). Only 500 are used (`--backbone_n 500`), so this
  file is sufficient; no vLLM needed on the new host.

## 1. 20Minuten self-generated records (gen only) — from model/7, same 2-stage pipeline
```
cd trace
pfx=$(python -c "import json;print(json.load(open('scripts/selfgen/assets/anchors.json'))['7']['anchor'],end='')")
CUDA_VISIBLE_DEVICES=G python scripts/analysis/bos_sample_v3.py --checkpoint <gen_run>/model/7 --mode anchor \
  --prefix-text "$pfx" --num-seqs 640 --max-seqs 200000 --max-new-tokens 1024 --batch 128 --no-routing-probe \
  --seed 319 --out-dir <D>/stageA --label oursgen_20Minuten
CUDA_VISIBLE_DEVICES=G python scripts/analysis/answer_pass_v3_fix.py --checkpoint <gen_run>/model/7 \
  --stage-a <D>/stageA --out <D>/records.jsonl --prompt-cue $'\n\nSimplification:' --max-answer-tokens 512 --batch 64
```
(CAP 1024 / BS 128 like MeetingBank/Py150 in run_selfgen_cl_frozen.sh; cue = every real
20Minuten prompt ends with "\n\nSimplification:". 640 -> first 500 used. Stage B can be split
over GPUs by sharding stageA/text.jsonl — see launchers/gen20min_stageB_sharded.sh.)

## 2. Router-only tuning with one residual row (the reported variant = full router trainable)
```
cd trace
CUDA_VISIBLE_DEVICES=G python scripts/residual/train_router_residual.py \
  --source <run>/7 --replay "<REPLAY spec>" --backbone scripts/residual/assets/backbone_bos_500.jsonl \
  --out <OUT>/<arm> --n_residual 1 --replay_per_task 500 --backbone_n 500 \
  --batch 8 --accum 2 --lr 2e-4 --epochs 1
```
Defaults that apply: `--residual_init zeros`, second_choice ON (no `--no_second_choice`),
`--max_len 1024`, seed 2025, LM loss only (aux/z suppressed), fp32 router master weights,
grad-clip 1.0, 3% warmup + cosine. 4,500 samples -> 282 optimizer steps, ~23 min on one H100.
**Do not pass `--residual_only`** for the reported numbers (that is the "residual row only"
ablation, which scored lower on TRACE). Control arm = same command with `--n_residual 0`.
Output: `<OUT>/<arm>/router_state.pt` + `residual_expert_meta.json` (source checkpoint path
is stored inside; the source run must stay in place).

## 3. Evaluation
* **TRACE**: only the final row (order8, 8 tasks) is re-scored with
  `scripts/residual/eval_trace_router_tuned.py --checkpoint_dir <OUT>/<arm> ...` (same args as
  evaluate_Ours_LoRA_MoE: `--max_prompt_len 0 --max_ans_len 1024 --no-task_generation_limits
  --slora_conv_mode llama3 --temperature 0`; batch 16, MeetingBank 1-2, Py150 4). The diagonal
  is the SOURCE run's sparse-15 diagonal (residual tuning is post-hoc, so acquisition scores are
  unchanged); F = mean over tasks 1-7 of (source diag - new final). Long tasks were split with
  `--num_sample_shards/--sample_shard_id` and merged by `scripts/merge_trace_shards.py`
  (launchers/run_trace_rest.sh).
* **General**: `scripts/lmeval/run_lmeval_trace.py --ckpt <OUT>/<arm> --tasks mmlu` and
  `--tasks gsm8k,piqa` (lm-eval venv `LLM-continual-learning-runtime/lmeval-venv`; the copy in
  `scripts/lmeval/` is the current one — the rsynced runtime copy is older). It detects
  `residual_expert_meta.json` and rebuilds source + residual + tuned router; prints per-layer
  residual selection rate.
