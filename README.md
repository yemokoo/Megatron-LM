# FLAME-MoE

This repository contains our local continual-learning workflow for FLAME-MoE on top of a vendored [Megatron-LM](./Megatron-LM) tree.

The current codebase is centered around:
- local multi-GPU `fp32` training on 3090/A100-class machines
- exact physical dataset splits for train/test
- continual-learning experiments for `A -> B`, `B -> A`, and `7-expert` runs
- routing, masking, interpolation, and probe-curve evaluation suites

## Current Focus

We are currently using the local `fp32` experiment entrypoints in [scripts/experiment](./scripts/experiment), not the older SLURM/DCLM release path described in earlier docs.

Canonical top-level entrypoints:
- [a_local_fp32.sh](./scripts/experiment/a_local_fp32.sh)
- [b_local_fp32.sh](./scripts/experiment/b_local_fp32.sh)
- [a_to_b_local_fp32.sh](./scripts/experiment/a_to_b_local_fp32.sh)
- [b_to_a_local_fp32.sh](./scripts/experiment/b_to_a_local_fp32.sh)
- [a_to_b_new_only_local_fp32.sh](./scripts/experiment/a_to_b_new_only_local_fp32.sh)
- [b_to_a_new_only_local_fp32.sh](./scripts/experiment/b_to_a_new_only_local_fp32.sh)
- [stage_A_7experts_local_fp32.sh](./scripts/experiment/stage_A_7experts_local_fp32.sh)
- [stage_A_7experts_resume_local_fp32.sh](./scripts/experiment/stage_A_7experts_resume_local_fp32.sh)
- [stage_B_after_A_7experts_no_freeze_local_fp32.sh](./scripts/experiment/stage_B_after_A_7experts_no_freeze_local_fp32.sh)

The implementation-detail scripts under `stage_*` are still the real workers. The short aliases above are the preferred entrypoints when possible.

## Dataset Policy

We use exact physical dataset splits under [`.local/dataset`](./.local/dataset):

Wiki:
- train: [pythia-12b-step1800-train-exact](./.local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact)
- test: [pythia-12b-step1800-test-fullremainder-exact](./.local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact)

Code:
- train: [pythia-12b-step1800-train-exact](./.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact)
- test: [pythia-12b-step1800-test-matchwiki-exact](./.local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact)

Exact token policy:
- train prefix: exactly `1800` steps worth of tokens
- wiki test: all remaining tokens after that prefix
- code test: trimmed to match wiki test token count

Important implication:
- for evaluation on continual checkpoints, we standardize on `iter_0001800`

## Representative Checkpoints

These are the representative model directories we have been using for analysis and evaluation.

Base task models:
- wiki-only base A: [stage-a-local-fp32-20260310-103742](./.local/weights/continual-stage-A/stage-a-local-fp32-20260310-103742)
- code-only base B: [stage-b-first-local-fp32-20260312-043202](./.local/weights/continual-stage-B/stage-b-first-local-fp32-20260312-043202)

Continual 4-expert / 7-expert comparisons:
- `A -> B` shared unfreeze: [stage-b-resume-local-fp32-20260311-171347](./.local/weights/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347)
- `A -> B` shared freeze (`new-only`): [stage-b-new-expert-router-only-local-fp32-20260317-113542](./.local/weights/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542)
- `B -> A` shared unfreeze: [stage-a-after-b-local-fp32-20260312-175645](./.local/weights/continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645)
- `B -> A` shared freeze (`new-only`): [stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509](./.local/weights/continual-stage-B-to-A-new-only/stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509)

7-expert runs:
- wiki 7-expert resumed A: [stage-a-7experts-resume-local-fp32-gpu0123-r1](./.local/weights/continual-stage-A-7experts-resume-local/stage-a-7experts-resume-local-fp32-gpu0123-r1)
- code after wiki 7-expert continual: [stage-b-after-a-7experts-no-freeze-local-fp32-gpu0123-r1](./.local/weights/continual-stage-B-after-A-7experts-no-freeze-local/stage-b-after-a-7experts-no-freeze-local-fp32-gpu0123-r1)

## Continual-Learning Semantics

When we say:
- `shared unfreeze`: shared trunk parameters keep training; old router/old experts may still be constrained depending on the experiment
- `shared freeze` / `new-only`: only newly introduced routed experts and router parameters remain trainable; shared trunk is frozen

In our discussion, `shared params` means the common trunk outside the routed expert/router parameters:
- embedding / output path
- attention blocks
- normalization layers
- other shared transformer-path parameters

Note:
- the vendored Megatron MoE stack also includes a per-MoE-layer shared expert branch
- routed expert count and shared expert are different concepts

## Evaluation Suites

Main evaluation/analysis entrypoints:

Routing similarity:
- [eval/task_a_compare/README.md](./eval/task_a_compare/README.md)
- [run_compare_pair_fp32.sh](./eval/task_a_compare/run_compare_pair_fp32.sh)

Masking extra routed experts:
- [eval/mask_extra_router/README.md](./eval/mask_extra_router/README.md)
- [run_mask_extra_router_suite.sh](./eval/mask_extra_router/run_mask_extra_router_suite.sh)
- [run_mask_extra_router_parallel.sh](./eval/mask_extra_router/run_mask_extra_router_parallel.sh)

Interpolation / checkpoint schedule eval:
- [run_task_b_interpolation_suite.sh](./eval/interpolation/run_task_b_interpolation_suite.sh)
- [run_a_to_b_full_code_probe_schedule.sh](./eval/interpolation/run_a_to_b_full_code_probe_schedule.sh)

Probe-comparison plotting:
- [plot_probe_comparison_suite.py](./analysis/plot_probe_comparison_suite.py)

Router logit analysis:
- [run_router_logit_stats_suite.sh](./analysis/run_router_logit_stats_suite.sh)
- [eval_router_logit_stats.py](./analysis/eval_router_logit_stats.py)

Representative output folders:
- routing compare: [routing-compare-suite-test10m-exact1800](./.local/weights/routing-compare-suite-test10m-exact1800)
- masking: [mask-extra-router-suite-parallel-exact1800](./.local/weights/mask-extra-router-suite-parallel-exact1800)
- probe plots: [probe-comparison-suite/plots](./.local/weights/probe-comparison-suite/plots)
- router logits: [router-logit-stats-suite](./.local/weights/router-logit-stats-suite)

## Environment Setup

For a reproducible A100-side setup, use:
- [environment.a100.yml](./environment.a100.yml)
- [install_a100_env.sh](./scripts/miscellaneous/install_a100_env.sh)

Typical flow on a remote server:
```bash
conda activate base
bash scripts/miscellaneous/install_a100_env.sh
```

This creates a conda environment, installs PyTorch/CUDA packages, then builds:
- [apex](./apex)
- [TransformerEngine](./TransformerEngine)

The setup is intended to approximate the current local Docker/conda environment without shipping the Docker image itself.

## Git / Upload Notes

The repository now ignores local-only artifacts via [`.gitignore`](./.gitignore), including:
- `.local`
- `.conda`
- `wandb`
- `build/dist`
- compiled extensions such as `*.so`
- checkpoint shards such as `*.distcp`

Do not commit:
- local checkpoints
- local tokenized datasets
- locally built binaries

Do commit:
- source changes under this repo
- vendored Megatron-LM source modifications that we authored
- environment files and reproducibility scripts

For vendored third-party code such as [Megatron-LM](./Megatron-LM), keep upstream license files intact:
- [Megatron-LM/LICENSE](./Megatron-LM/LICENSE)

## Repository Structure

Important directories:
- [configs](./configs): model and train config fragments
- [scripts/experiment](./scripts/experiment): local training entrypoints
- [eval](./eval): evaluation runners
- [analysis](./analysis): analysis and plotting scripts
- [Megatron-LM](./Megatron-LM): vendored training engine
- [apex](./apex): vendored extension source
- [TransformerEngine](./TransformerEngine): vendored extension source


## Hugging Face Export

To publish the representative checkpoints to a single Hugging Face repo such as `YeMoKoo/flamemoe`:
- install the upload tooling with [install_hf_tools.sh](./scripts/release/install_hf_tools.sh)
- authenticate once with `hf auth login`
- run [upload_models_to_hf.py](./scripts/release/upload_models_to_hf.py)

Typical flow:
```bash
bash scripts/release/install_hf_tools.sh
~/.local/bin/hf auth login
.conda/envs/flame3090/bin/python scripts/release/upload_models_to_hf.py --dry-run
.conda/envs/flame3090/bin/python scripts/release/upload_models_to_hf.py
```

The upload script pushes one shared model repo and stores each representative checkpoint under its own subfolder:
- `base_wiki_a`
- `base_code_b`
- `a_to_b_unfreeze`
- `a_to_b_freeze`
- `b_to_a_unfreeze`
- `b_to_a_freeze`
- `seven_expert_wiki_a`
- `seven_expert_code_after_wiki`

Each subfolder is exported as a minimal Megatron load directory with a single representative `iter_XXXXXXXX/` tree.

## Notes

- The old top-level README described the earlier release/SLURM workflow. This file now reflects the current local continual-learning workflow.
- If a script or doc conflicts with this README, prefer the local fp32 experiment scripts and the per-suite READMEs under `eval/`.
