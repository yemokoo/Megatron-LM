# FLAME-MoE

This repository contains our local continual-learning workflow for FLAME-MoE on top of a vendored [Megatron-LM](./Megatron-LM) tree.

The current codebase is centered around:
- local multi-GPU training on 3090/A100-class machines
- exact physical dataset splits for train/test
- continual-learning experiments for `A -> B`, `B -> A`, and `7-expert` runs
- new dense-plus-attention-LoRA experiments on mixed `wiki + code` data
- routing, masking, interpolation, and probe-curve evaluation suites

The repository also contains a unified TRACE/SLoRA project under
[`trace/`](./trace). It includes the latest local `llmcl_benchmark` training code,
SLoRA/TRACE/O-LoRA reproduction ports, portable launchers, and new-server setup
instructions in [`trace/docs/PORTABILITY_KO.md`](./trace/docs/PORTABILITY_KO.md).

## Current Focus

We currently keep two experiment families side by side under [scripts/experiment](./scripts/experiment):
- [scripts/experiment/3090](./scripts/experiment/3090): the older local `fp32` MoE continual-learning path used for 3090-era runs
- [scripts/experiment/a100](./scripts/experiment/a100): the current A100-oriented `bf16` path for MoE continual-learning plus dense / attention-LoRA work

The older SLURM/DCLM release path described in earlier docs is no longer the primary workflow.

Canonical top-level entrypoints:
- [a_local_fp32.sh](./scripts/experiment/3090/a_local_fp32.sh)
- [b_local_fp32.sh](./scripts/experiment/3090/b_local_fp32.sh)
- [a_to_b_local_fp32.sh](./scripts/experiment/3090/a_to_b_local_fp32.sh)
- [b_to_a_local_fp32.sh](./scripts/experiment/3090/b_to_a_local_fp32.sh)
- [a_to_b_new_only_local_fp32.sh](./scripts/experiment/3090/a_to_b_new_only_local_fp32.sh)
- [b_to_a_new_only_local_fp32.sh](./scripts/experiment/3090/b_to_a_new_only_local_fp32.sh)
- [stage_A_7experts_local_fp32.sh](./scripts/experiment/3090/stage_A_7experts_local_fp32.sh)
- [stage_A_7experts_resume_local_fp32.sh](./scripts/experiment/3090/stage_A_7experts_resume_local_fp32.sh)
- [stage_B_after_A_7experts_no_freeze_local_fp32.sh](./scripts/experiment/3090/stage_B_after_A_7experts_no_freeze_local_fp32.sh)
- [pretrain_mixed_dense_local_bf16.sh](./scripts/experiment/pretrain_mixed_dense_local_bf16.sh)
- [train_mixed_qv_lora_experts_local_bf16.sh](./scripts/experiment/train_mixed_qv_lora_experts_local_bf16.sh)

A100-specific bf16 MoE continual-learning entrypoints:
- [scripts/experiment/a100/README.md](./scripts/experiment/a100/README.md)
- [wiki_a_a100_bf16.sh](./scripts/experiment/a100/wiki_a_a100_bf16.sh)
- [code_b_a100_bf16.sh](./scripts/experiment/a100/code_b_a100_bf16.sh)
- [a_to_b_a100_bf16.sh](./scripts/experiment/a100/a_to_b_a100_bf16.sh)
- [a_to_b_freeze_a100_bf16.sh](./scripts/experiment/a100/a_to_b_freeze_a100_bf16.sh)
- [b_to_a_a100_bf16.sh](./scripts/experiment/a100/b_to_a_a100_bf16.sh)
- [b_to_a_freeze_a100_bf16.sh](./scripts/experiment/a100/b_to_a_freeze_a100_bf16.sh)
- [run_all_moe_a100_bf16_sequential.sh](./scripts/experiment/a100/run_all_moe_a100_bf16_sequential.sh)

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

## Attention LoRA Experiments

We also maintain a new experiment path that does not modify the existing FFN-MoE continual-learning codepath.

Current design:
- Stage 1: train a dense transformer backbone on mixed `wiki + code` train-exact data
- Stage 2: load that dense checkpoint, add routed LoRA experts on attention `Q` and `V`, and continue training
- the Stage 2 default is to freeze the backbone and train only the attention LoRA experts plus their router

Current defaults for this path:
- dense FFN hidden size stays at `5472`
- attention LoRA routing uses `top-1`
- both exact train splits are mixed 1:1 by construction
- probes are still logged on wiki test and code test
- intended precision is `bf16` on A100-class GPUs

Relevant files:
- model config for dense pretrain: [flame-dense.sh](./configs/model/flame-dense.sh)
- model config for attention LoRA experts: [flame-qv-lora-experts.sh](./configs/model/flame-qv-lora-experts.sh)
- attention LoRA module: [qv_lora_attention.py](./Megatron-LM/megatron/core/transformer/qv_lora_attention.py)
- custom GPT layer spec: [qv_lora_layer_specs.py](./Megatron-LM/megatron/core/models/gpt/qv_lora_layer_specs.py)
- stage 1 launcher: [pretrain_mixed_dense_local_bf16.sh](./scripts/experiment/pretrain_mixed_dense_local_bf16.sh)
- stage 2 launcher: [train_mixed_qv_lora_experts_local_bf16.sh](./scripts/experiment/train_mixed_qv_lora_experts_local_bf16.sh)

Typical flow:
```bash
bash scripts/experiment/pretrain_mixed_dense_local_bf16.sh
bash scripts/experiment/train_mixed_qv_lora_experts_local_bf16.sh
```

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

## A100 MoE BF16 Path

The A100 MoE path under [scripts/experiment/a100](./scripts/experiment/a100) is the current clean bf16 continual-learning workflow.

Key properties:
- it uses a dedicated model config: [flame-moe-bf16-no-shared.sh](./scripts/experiment/a100/flame-moe-bf16-no-shared.sh)
- shared experts are disabled by omission of `--moe-shared-expert-intermediate-size`
- routed experts are the only MoE expert path kept active
- base runs use `4` experts, continual runs expand `4 -> 7`
- old routed experts and old router slices are frozen during expansion
- `shared unfreeze` runs keep the shared trunk trainable and apply output-level KL with `lambda=1.0`
- `shared freeze` runs train only new experts and router slices and skip teacher-model loading entirely

Main A100 MoE entrypoints:
- base wiki model: [wiki_a_a100_bf16.sh](./scripts/experiment/a100/wiki_a_a100_bf16.sh)
- base code model: [code_b_a100_bf16.sh](./scripts/experiment/a100/code_b_a100_bf16.sh)
- wiki to code unfreeze: [a_to_b_a100_bf16.sh](./scripts/experiment/a100/a_to_b_a100_bf16.sh)
- wiki to code freeze: [a_to_b_freeze_a100_bf16.sh](./scripts/experiment/a100/a_to_b_freeze_a100_bf16.sh)
- code to wiki unfreeze: [b_to_a_a100_bf16.sh](./scripts/experiment/a100/b_to_a_a100_bf16.sh)
- code to wiki freeze: [b_to_a_freeze_a100_bf16.sh](./scripts/experiment/a100/b_to_a_freeze_a100_bf16.sh)

Sequential launcher:
- [run_all_moe_a100_bf16_sequential.sh](./scripts/experiment/a100/run_all_moe_a100_bf16_sequential.sh) runs all six MoE experiments in order
- default order: `wiki A -> code B -> A to B -> A to B freeze -> B to A -> B to A freeze`
- default continual-friendly settings:
  - `MICRO_BATCH_SIZE=32`
  - `GLOBAL_BATCH_SIZE=2304`
  - `TRAIN_ITERS=1800`
  - `SAVE_INTERVAL=300`
  - `EVAL_INTERVAL=100`
  - dual probes every `40` steps on both wiki and code test sets
  - a `300` second pause between runs

W&B / probe continuity:
- continual runs read the source checkpoint metadata and set `WANDB_STEP_OFFSET`, `PROBE_STEP_OFFSET`, and `SECONDARY_PROBE_STEP_OFFSET`
- this keeps probe curves and W&B step axes aligned when a continual stage starts after the base stage ends at step `1800`

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

FLAME/Megatron and TRACE/SLoRA use separate runtimes. The complete migration
contract and exact commands are in [environments/README_KO.md](./environments/README_KO.md).
For a copy-paste Codex handoff covering GitHub/HF/W&B authentication, datasets,
final checkpoints, grouped-gemm builds, and smoke checks, use
[docs/new_server_codex_bootstrap_prompt_ko.md](./docs/new_server_codex_bootstrap_prompt_ko.md).

For FLAME/G2 Megatron, use the dedicated Conda environment:

```bash
git submodule update --init --recursive
bash scripts/miscellaneous/install_a100_env.sh
conda activate flame-megatron-a100
source scripts/miscellaneous/activate_flame_env.sh
python scripts/miscellaneous/verify_flame_env.py --require-gpu
```

This installs PyTorch 2.4.1+cu124 and builds grouped-gemm, Apex,
TransformerEngine, and flash-attn against one isolated ABI. The observed NGC
24.07 runtime that produced the current G2 checkpoints is recorded in
[reference-runtime.json](./environments/flame-megatron/reference-runtime.json).

For the vendored TRACE/SLoRA project, use its separate project-local venv:

```bash
cd trace
./scripts/setup_runtime.sh
source .venv-runtime/bin/activate
python scripts/runtime_preflight.py --skip-gpu
```

Do not use `pip install --user` for either runtime and do not mix their
`PYTHONPATH`s.

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
- [scripts/experiment/3090](./scripts/experiment/3090): older fp32 MoE continual-learning path
- [scripts/experiment/a100](./scripts/experiment/a100): current bf16 A100 MoE runners and sequential launcher
- [eval](./eval): evaluation runners
- [analysis](./analysis): analysis and plotting scripts
- [Megatron-LM](./Megatron-LM): vendored training engine, including the new attention LoRA experiment path
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
- If a script or doc conflicts with this README, prefer the current local experiment scripts under `scripts/experiment/`, including the `scripts/experiment/3090/` fp32 continual-learning path and the newer bf16 paths.
