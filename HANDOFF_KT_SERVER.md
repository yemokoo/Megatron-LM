# KT Server Handoff

## Current Situation

This repo was developed and validated primarily on local 3090-class GPUs, so most of the recent training/eval workflow was standardized around local multi-GPU `fp32` runs.

The next environment is a KT internal server with A100 GPUs. Because of that hardware change:
- we do **not** necessarily need to keep using `fp32`
- moving to `bf16` or `fp16` is now realistic and should be revisited
- the main migration task is to reproduce the current environment and then run small smoke tests on A100 before restarting full training/eval

## Important Repo State

Branch in active use:
- `slurm`

Top-level docs and setup files already updated:
- KT image preset decision: if restricted to the shown NGC presets, start from `24.07` and use the no-conda setup path in `KT_24_07_SETUP.md`
- `README.md`
- `.gitignore`
- `environment.a100.yml`
- `scripts/miscellaneous/install_a100_env.sh`
- `KT_24_07_SETUP.md`
- `scripts/miscellaneous/install_kt_24_07_no_conda.sh`
- `scripts/release/install_hf_tools.sh`
- `scripts/release/upload_models_to_hf.py`

Representative model uploads are intended for:
- HF repo: `YeMoKoo/flamemoe`

Representative model folders configured for upload:
- `base_wiki_a`
- `base_code_b`
- `a_to_b_unfreeze`
- `a_to_b_freeze`
- `b_to_a_unfreeze`
- `b_to_a_freeze`
- `seven_expert_wiki_a`
- `seven_expert_code_after_wiki`

## Canonical Dataset Policy

Exact physical splits are now the standard. Use these directly instead of relying on Megatron internal train/valid splitting.

Wiki:
- train: `.local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact`
- test: `.local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact`

Code:
- train: `.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact`
- test: `.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact`

Common implication for eval:
- use `DATASET_SPLIT=100,0,0`
- use `CONSUMED_SAMPLES=0`
- use the exact split directory itself as the dataset
- for the main continual comparisons, the canonical representative checkpoint is `iter_0001800`

## Representative Checkpoints

Base models:
- wiki-only A: `.local/weights/continual-stage-A/stage-a-local-fp32-20260310-103742`
- code-only B: `.local/weights/continual-stage-B/stage-b-first-local-fp32-20260312-043202`

Continual runs:
- A->B unfreeze: `.local/weights/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347`
- A->B freeze/new-only: `.local/weights/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542`
- B->A unfreeze: `.local/weights/continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645`
- B->A freeze/new-only: `.local/weights/continual-stage-B-to-A-new-only/stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509`

7-expert runs:
- resumed wiki A: `.local/weights/continual-stage-A-7experts-resume-local/stage-a-7experts-resume-local-fp32-gpu0123-r1`
- code-after-wiki: `.local/weights/continual-stage-B-after-A-7experts-no-freeze-local/stage-b-after-a-7experts-no-freeze-local-fp32-gpu0123-r1`

## Recently Confirmed Findings

Shared freeze vs unfreeze:
- `shared freeze` does **not** preserve old-task routing more faithfully
- instead, it tends to reduce the old/new router-logit gap and increases mixed `old+new` routing
- old-task performance can still be preserved because the shared representation is less disrupted and new experts act as auxiliary paths

Interpretation used in discussions:
- `shared unfreeze`: stronger old-vs-new router logit separation, more old-expert-dominant routing
- `shared freeze`: smaller old-vs-new logit gap, more mixed routing
- retention seems better explained by representation preservation plus mixed expert reuse than by literal routing preservation

## Scripts Worth Knowing

Training:
- `scripts/experiment/stage_A_local_fp32.sh`
- `scripts/experiment/stage_B_local_fp32.sh`
- `scripts/experiment/stage_A_7experts_resume_local_fp32.sh`
- `scripts/experiment/stage_B_after_A_7experts_no_freeze_local_fp32.sh`
- `scripts/experiment/pretrain_wiki_dense_local_bf16.sh`
- `scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh`
- `scripts/experiment/pretrain_wiki_qv_lora_local_bf16.sh`
- `scripts/experiment/continual_code_from_wiki_qv_lora_expand_local_bf16.sh`

Eval / analysis:
- `eval/task_a_compare/run_compare_pair_fp32.sh`
- `eval/mask_extra_router/run_mask_extra_router_parallel.sh`
- `eval/interpolation/run_a_to_b_full_code_probe_schedule.sh`
- `analysis/plot_probe_comparison_suite.py`
- `analysis/run_router_logit_stats_suite.sh`

## Hugging Face Upload Notes

The HF upload helper script now uploads only the representative `iter_*` folder, not the whole run directory. This avoids permission issues from `wandb/` and `logs/`.

Useful commands:
```bash
~/.local/bin/hf auth login
~/.local/bin/hf auth whoami
.conda/envs/flame3090/bin/python scripts/release/upload_models_to_hf.py --dry-run
.conda/envs/flame3090/bin/python scripts/release/upload_models_to_hf.py
```

## KT Server Image Decision

If the KT server only allows choosing from the shown NGC PyTorch images (`25.05`, `25.01`, `24.07`, `23.09`), the recommended starting point is:
- `24.07`

Reasoning:
- it keeps `Python 3.10`, which is closest to the currently validated local environment
- it is less disruptive for Megatron-LM + TransformerEngine + Apex than the newer `25.01` / `25.05` images that move to Python `3.12`
- it is newer and more practical than `23.09`, while still staying conservative

Important nuance:
- choose `24.07` as the base image
- do not rely on the image's stock `PyTorch 2.4` as the final runtime
- recreate the KT user-site runtime with `scripts/miscellaneous/install_kt_24_07_no_conda.sh`
- source `scripts/miscellaneous/activate_kt_env.sh` in each new KT session
- that install path now matches the currently observed KT runtime: `/usr/bin/python`, `~/.local` packages, `Python 3.10`, `torch 2.5.1+cu121`, local Apex, and local TransformerEngine

Practical KT-server plan:
1. start from NGC PyTorch `24.07`
2. clone this repo and checkout `slurm`
3. if the session is reclaimed on low utilization, run `python scripts/miscellaneous/session_warmup.py` in another pane while installing
4. default warmup is light (`1s` compute / `4s` sleep); if needed, raise it with `--matrix-size 3072 --compute-seconds 2.0 --sleep-seconds 2.0`
5. run `bash scripts/miscellaneous/install_kt_24_07_no_conda.sh`
6. `source scripts/miscellaneous/activate_kt_env.sh`
7. run smoke tests before full training/eval

GitHub credential restore for KT:
- Use HTTPS remotes, not SSH, because KT often blocks or complicates SSH auth.
- Use `credential.helper store` and write a personal access token once to `~/.git-credentials`.
- Always unset code-server askpass variables before CLI git operations.

```bash
git config --global credential.helper store
git config --global credential.useHttpPath false

unset GIT_ASKPASS SSH_ASKPASS \
  VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_EXTRA_ARGS \
  VSCODE_GIT_IPC_HANDLE VSCODE_GIT_ASKPASS_MAIN

read -p "GitHub username: " GITHUB_USER
read -s -p "GitHub token: " GITHUB_TOKEN
echo

printf "https://%s:%s@github.com\n" "$GITHUB_USER" "$GITHUB_TOKEN" > ~/.git-credentials
chmod 600 ~/.git-credentials
unset GITHUB_TOKEN

git remote set-url origin https://github.com/yemokoo/LLM-continual-learning.git
GIT_TERMINAL_PROMPT=0 git pull origin slurm
git submodule update --init --recursive
```

## A100 Migration Priority

When resuming on the KT server, the first questions to revisit are:
1. Can we switch the local training path from `fp32` to `bf16` or `fp16` safely on A100?
2. Do Apex / TransformerEngine / Megatron build cleanly on the KT server?
3. Do the existing local scripts need A100-specific presets or new wrapper scripts?
4. Can we run a short smoke test before full continual training?

Recommended next step on the KT server:
- recreate the environment from `environment.a100.yml`
- run `scripts/miscellaneous/install_a100_env.sh`
- verify PyTorch + CUDA + Apex + TransformerEngine imports
- do a tiny 1-step or 10-step Megatron smoke run
- only then decide whether to keep `fp32` or switch to `bf16`/`fp16`

## Current Sequential Experiment Design

The current requested setup is sequential, not mixed-data training.

Dense-only path:
- stage 1: wiki only
- stage 2: continue on code from the wiki checkpoint
- scripts:
  - `scripts/experiment/pretrain_wiki_dense_local_bf16.sh`
  - `scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh`

Attention-LoRA path:
- stage 1: wiki only with routed attention Q/V LoRA experts
- stage 2: continue on code with attention LoRA expert expansion
- scripts:
  - `scripts/experiment/pretrain_wiki_qv_lora_local_bf16.sh`
  - `scripts/experiment/continual_code_from_wiki_qv_lora_expand_local_bf16.sh`

Important model constraints:
- base backbone is a standard dense transformer
- FFN is not expanded during the attention-LoRA continual experiment
- hidden size and FFN size should match `scripts/experiment/stage_A_local_fp32.sh`
- canonical values:
  - `NUM_LAYERS=9`
  - `HIDDEN_SIZE=1024`
  - `FFN_HIDDEN_SIZE=5472`

Attention-LoRA continual specifics:
- wiki stage uses `4` attention LoRA experts
- code stage expands to `7` experts
- when moving to code:
  - old `4` experts are copied into the new checkpoint
  - old router rows and old experts are frozen
  - new experts are initialized by copying existing trained experts/router rows
  - shared backbone params remain trainable
  - output-level KL regularization is enabled with `lambda=1.0` by default

## KT Smoke Test Order

Do not start with full training. Use this order:

1. environment import smoke
2. dense wiki 1-10 step smoke
3. dense wiki->code resume 1-10 step smoke
4. q/v LoRA wiki 1-10 step smoke
5. q/v LoRA wiki->code expansion 1-10 step smoke
6. only after all four pass, launch the full run

Recommended smoke-test overrides:
- `CUDA_VISIBLE_DEVICES=0`
- `NPROC_PER_NODE=1`
- `MICRO_BATCH_SIZE=1`
- `GLOBAL_BATCH_SIZE=8`
- `TRAIN_ITERS=10`
- `SAVE_INTERVAL=10`
- `EVAL_INTERVAL=10`
- `SEQ_LENGTH=512`

## KT Smoke Commands

Assume:
- repo root is the current working directory
- KT environment is already activated
- dataset paths are edited to match the KT server

Dense wiki smoke:
```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=8 \
TRAIN_ITERS=10 \
SAVE_INTERVAL=10 \
EVAL_INTERVAL=10 \
SEQ_LENGTH=512 \
TRAIN_DATASET=/path/to/wiki-train-exact \
PROBE_DATASET=/path/to/wiki-test-exact \
bash scripts/experiment/pretrain_wiki_dense_local_bf16.sh
```

Dense wiki -> code smoke:
```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=8 \
TRAIN_ITERS=10 \
SAVE_INTERVAL=10 \
EVAL_INTERVAL=10 \
SEQ_LENGTH=512 \
STAGE1_WEIGHTS_DIR=/path/to/wiki-dense-checkpoint \
TRAIN_DATASET=/path/to/code-train-exact \
PROBE_DATASET=/path/to/code-test-exact \
SECONDARY_PROBE_DATASET=/path/to/wiki-test-exact \
bash scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh
```

Q/V LoRA wiki smoke:
```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=8 \
TRAIN_ITERS=10 \
SAVE_INTERVAL=10 \
EVAL_INTERVAL=10 \
SEQ_LENGTH=512 \
ATTN_LORA_NUM_EXPERTS=4 \
TRAIN_DATASET=/path/to/wiki-train-exact \
PROBE_DATASET=/path/to/wiki-test-exact \
bash scripts/experiment/pretrain_wiki_qv_lora_local_bf16.sh
```

Q/V LoRA wiki -> code expansion smoke:
```bash
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
MICRO_BATCH_SIZE=1 \
GLOBAL_BATCH_SIZE=8 \
TRAIN_ITERS=10 \
SAVE_INTERVAL=10 \
EVAL_INTERVAL=10 \
SEQ_LENGTH=512 \
ATTN_LORA_SOURCE_NUM_EXPERTS=4 \
ATTN_LORA_NUM_EXPERTS=7 \
OLD_MODEL_KL_COEFF=1.0 \
STAGE1_WEIGHTS_DIR=/path/to/wiki-qv-lora-checkpoint \
TRAIN_DATASET=/path/to/code-train-exact \
PROBE_DATASET=/path/to/code-test-exact \
SECONDARY_PROBE_DATASET=/path/to/wiki-test-exact \
bash scripts/experiment/continual_code_from_wiki_qv_lora_expand_local_bf16.sh
```

## What To Check During Smoke Tests

Each smoke run should confirm:
- script starts without import/build errors
- dataset copy succeeds
- checkpoint load or resume succeeds
- at least one optimizer step completes
- checkpoint save succeeds
- no immediate NaN or shape mismatch appears

For the Q/V LoRA expansion smoke, also confirm:
- expansion audit JSON is produced under `expansion_audit/`
- the run log prints the attention LoRA expansion message
- old checkpoint loads as the teacher/source checkpoint without failure

## Short Prompt For The Next Assistant

Use the current `slurm` branch state, not the older release docs. This repo was recently reorganized around local exact-split continual-learning experiments. The main migration task is to move the existing 3090-oriented `fp32` workflow to an A100 server, probably with `bf16` or `fp16`, while preserving the exact dataset policy, representative checkpoints, and recent eval scripts.
