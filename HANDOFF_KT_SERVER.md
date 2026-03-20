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
- `README.md`
- `.gitignore`
- `environment.a100.yml`
- `scripts/miscellaneous/install_a100_env.sh`
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

## Short Prompt For The Next Assistant

Use the current `slurm` branch state, not the older release docs. This repo was recently reorganized around local exact-split continual-learning experiments. The main migration task is to move the existing 3090-oriented `fp32` workflow to an A100 server, probably with `bf16` or `fp16`, while preserving the exact dataset policy, representative checkpoints, and recent eval scripts.
