# General-ability eval (MMLU / GSM8K / PIQA)

lm-evaluation-harness 0.4.8 on the final TRACE checkpoint of each method, with no further training.
Shot counts are the lm-eval task defaults: **MMLU 0-shot (acc), GSM8K 5-shot (exact_match), PIQA
0-shot (acc)**. The base model on this setup gives MMLU 67.97.

## Setup (once per server)
```bash
LMEVAL_VENV=<venv dir> LMEVAL_DATA=<data dir> bash scripts/lmeval/setup_lmeval.sh
```
- venv: `requirements-lmeval.txt`, the exact package set of the original runs (Python 3.10,
  torch 2.4.1+cu124, lm_eval 0.4.8, datasets 3.6.0, transformers 4.51.3)
- data: `assets/lmeval_datasets.tar.gz` (10 MB, sha256 in `.sha256`) is the offline Hugging Face
  dataset cache the table was scored with (hails/mmlu_no_train, gsm8k main, piqa), unpacked to
  `<data dir>/lmeval_datasets`. `LMEVAL_DOWNLOAD=1` fetches the same datasets from the Hub instead.

## Run one checkpoint
```bash
export LMEVAL_VENV=<venv dir> HF_DATASETS_CACHE=<data dir>/lmeval_datasets SLORA_LLAMA31_PATH=<Llama-3.1-8B-Instruct>
GPU=0 CKPT=<run>/model/7 OUT=<out dir> GUARD=1 bash scripts/lmeval/run_general_ability.sh
GPU=0 CKPT=base          OUT=<out dir>         bash scripts/lmeval/run_general_ability.sh   # backbone row
```
- `run_lmeval_trace.py` loads the checkpoint with the TRACE loaders (Table-1 baselines, V3
  LoRA-MoE, residual / mass-reservoir, SLoRA-Pre) and hands it to lm-eval's HFLM.
- `GUARD=1` = `--bos_guard --guard_header --guard_decision none`, the same switches as the
  header-guarded residual / mass-reservoir TRACE eval; use it for those checkpoints only.
- Output: `<out dir>/results_mmlu_gsm8k_piqa.json` (+ `lmeval.log` with the lm-eval tables).
- `run_lmeval.sh` is the plain-HF variant (a model directory, optionally + a PEFT adapter).
