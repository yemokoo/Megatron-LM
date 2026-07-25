

# FLAME-MoE Local Continual Learning Guide

This guide explains how the local continual-learning experiment is wired, what is already implemented for MoE expansion, and what to change when you want to reproduce the phenomenon from Figure 3 of the FLAME paper at smaller scale.

## Goal

The local experiment is a reduced version of the paper's staged continual-learning setup:

- Stage A trains a small MoE on a general-domain dataset.
- Stage B loads the Stage A checkpoint, expands the number of experts, freezes the old experts and router rows, and continues training on a shifted domain.
- The main question is whether the same qualitative behavior from the paper appears even when the model is much smaller and the datasets are only proxies for the original private data.

The paper referenced by the experiment is:

- `FLAME: Mixture-of-Experts for Continual Learning and Knowledge Consolidation`
- arXiv: `2305.12281`
- Figure 3 studies continual learning under domain shift while increasing the number of experts.

## What the two local scripts do

### Stage A

File: [scripts/experiment/stage_A_local.sh](/home/yemoyemo010831/nas/FLAME-MoE/scripts/experiment/stage_A_local.sh)

Flow:

1. Set a small FLAME-MoE architecture.
2. Copy tokenized Stage A data to local SSD.
3. Source [configs/model/flame-moe.sh](/home/yemoyemo010831/nas/FLAME-MoE/configs/model/flame-moe.sh) to build `MODEL_ARGS`.
4. Launch [Megatron-LM/pretrain_gpt.py](/home/yemoyemo010831/nas/FLAME-MoE/Megatron-LM/pretrain_gpt.py) with `torchrun`.
5. Periodically sync checkpoints from SSD back to persistent storage.

Default local Stage A dataset:

- `$LOCAL_DATASET/wikipedia/tokenized/EleutherAI/pythia-12b`

Key variables:

- `NUM_EXPERTS` default `4`
- `MOE_ROUTER_TOPK` default `2`
- `TRANSFORMER_IMPL` default `local`

Example:

```bash
LOCAL_BASE=/path/to/local-assets \
LOCAL_SSD_ROOT=/tmp/flame-moe \
NUM_EXPERTS=4 \
MOE_ROUTER_TOPK=2 \
TRANSFORMER_IMPL=local \
NPROC_PER_NODE=8 \
bash scripts/experiment/stage_A_local.sh
```

### Stage B

File: [scripts/experiment/stage_B_local.sh](/home/yemoyemo010831/nas/FLAME-MoE/scripts/experiment/stage_B_local.sh)

Flow:

1. Load the Stage A checkpoint.
2. Build the same base architecture, but with a larger `NUM_EXPERTS`.
3. Pass MoE expansion flags into Megatron-LM.
4. Continue training on a shifted-domain dataset.

Conceptually, Stage B is not a fresh pretraining run.

It is:

- Stage A checkpoint loading
- expert expansion from a smaller MoE to a larger MoE
- optional freezing of the copied experts and router rows
- continual learning on the new domain

So the expected workflow is:

1. Train Stage A on the general-domain proxy data.
2. Save a usable Stage A checkpoint.
3. Launch Stage B from that checkpoint with a larger expert count.

Default local Stage B dataset:

- `$LOCAL_DATASET/python-code/tokenized/EleutherAI/pythia-12b`

Key variables:

- `SOURCE_NUM_EXPERTS` default `4`
- `NUM_EXPERTS` default `7`
- `MOE_ROUTER_TOPK` default `2`
- `TRANSFORMER_IMPL` default `local`

Example:

```bash
LOCAL_BASE=/path/to/local-assets \
LOCAL_SSD_ROOT=/tmp/flame-moe \
STAGE_A_RUN_ID=stage-a-local-20260308-120000 \
SOURCE_NUM_EXPERTS=4 \
NUM_EXPERTS=7 \
TRANSFORMER_IMPL=local \
NPROC_PER_NODE=8 \
bash scripts/experiment/stage_B_local.sh
```

The script validates `SOURCE_NUM_EXPERTS < NUM_EXPERTS`.

## Where the expansion behavior is implemented

The important part is not only in the shell scripts. The actual expansion logic is inside the local Megatron-LM fork.

### CLI arguments

File: [Megatron-LM/megatron/training/arguments.py](/home/yemoyemo010831/nas/FLAME-MoE/Megatron-LM/megatron/training/arguments.py)

Relevant arguments:

- `--moe-expand-from-num-experts`
- `--moe-freeze-existing-experts`
- `--moe-freeze-existing-router`
- `--moe-old-model-kl-coeff`
- `--moe-old-model-kl-temperature`

### Expansion during startup

File: [Megatron-LM/megatron/training/training.py](/home/yemoyemo010831/nas/FLAME-MoE/Megatron-LM/megatron/training/training.py)

What happens:

1. A source MoE model is instantiated with the smaller expert count.
2. The Stage A checkpoint is loaded into that source model.
3. The source experts/router are copied into the target model.
4. Existing experts/router rows are optionally frozen.

This means Stage B is not a placeholder. The expert expansion path is already implemented.

### Old-model distillation

File: [Megatron-LM/pretrain_gpt.py](/home/yemoyemo010831/nas/FLAME-MoE/Megatron-LM/pretrain_gpt.py)

The loss function adds an optional KL term between teacher logits from the old model and student logits from the expanded model. Stage B enables this through:

- `--moe-old-model-kl-coeff`
- `--moe-old-model-kl-temperature`

## Model config flow

The shell scripts do not directly spell out every model argument. They source:

- [configs/model/flame-moe.sh](/home/yemoyemo010831/nas/FLAME-MoE/configs/model/flame-moe.sh)

That file defines:

- hidden size
- number of layers
- MoE FFN size
- number of experts
- top-k routing
- shared expert size
- tokenizer choice

Then `torchrun` calls:

- [Megatron-LM/pretrain_gpt.py](/home/yemoyemo010831/nas/FLAME-MoE/Megatron-LM/pretrain_gpt.py)

Inside `pretrain_gpt.py`, the model spec is chosen from:

- local implementation via `get_gpt_layer_local_spec(...)`
- TransformerEngine implementation via `get_gpt_layer_with_transformer_engine_spec(...)`

The selection is controlled by:

- `--transformer-impl local`
- `--transformer-impl transformer_engine`

## Dataset flow

The Stage A/B local scripts expect Megatron indexed datasets:

- `.bin`
- `.idx`

The runtime constructs `--data-path` by scanning every `.bin` file under the chosen dataset directory and stripping the extension. So the expected directory contents are already compatible with standard Megatron preprocessing.

Relevant files:

- [scripts/dataset/download_wikipedia.sh](/home/yemoyemo010831/nas/FLAME-MoE/scripts/dataset/download_wikipedia.sh)
- [scripts/dataset/download_code.sh](/home/yemoyemo010831/nas/FLAME-MoE/scripts/dataset/download_code.sh)

These scripts now honor `LOCAL_BASE`, so their default output layout matches the local experiment scripts:

- Stage A data: `$LOCAL_BASE/dataset/wikipedia/tokenized/EleutherAI/pythia-12b`
- Stage B data: `$LOCAL_BASE/dataset/python-code/tokenized/EleutherAI/pythia-12b`

## Does this match Figure 3 closely enough?

Partially.

What already matches the paper's mechanism:

- staged continual learning
- expert expansion between stages
- freezing old experts/router rows
- old-model distillation
- fixed top-k routing

What does not match exactly:

- the original datasets are not available
- the current proxy uses only `Wikipedia -> Python code`
- the paper's Figure 3 uses a stronger sequence of domain shifts

So this setup is valid for testing whether the same qualitative phenomenon appears at smaller scale, but it is not an exact reproduction of the paper's full data regime.

## Recommended proxy dataset strategy

If the goal is to maximize similarity to the paper while staying public:

1. Stage A: English Wikipedia or a general web corpus
2. Stage B1: multilingual or i18n-heavy text
3. Stage B2: conversational/instruction data or code, depending on which shift you want to stress

If you keep only two stages, `Wikipedia -> code` is still a useful stress test, but it is a stronger and less paper-faithful domain jump.

## Can I control expert counts freely?

Yes for the local scripts, within normal Megatron constraints.

Stage A:

- set `NUM_EXPERTS`

Stage B:

- set `SOURCE_NUM_EXPERTS`
- set `NUM_EXPERTS`

Constraints:

- `SOURCE_NUM_EXPERTS < NUM_EXPERTS`
- `MOE_ROUTER_TOPK <= NUM_EXPERTS`
- if you later increase `EXPERT_MODEL_PARALLEL_SIZE`, it must be compatible with `NUM_EXPERTS`

Useful examples:

```bash
# 4 -> 8 expansion
NUM_EXPERTS=4 bash scripts/experiment/stage_A_local.sh
STAGE_A_RUN_ID=<run_id> SOURCE_NUM_EXPERTS=4 NUM_EXPERTS=8 bash scripts/experiment/stage_B_local.sh

# 2 -> 6 expansion
NUM_EXPERTS=2 bash scripts/experiment/stage_A_local.sh
STAGE_A_RUN_ID=<run_id> SOURCE_NUM_EXPERTS=2 NUM_EXPERTS=6 bash scripts/experiment/stage_B_local.sh
```

## Is TransformerEngine required?

For this repo as actually validated on the local Docker setup: yes, effectively.

Why the earlier assumption changed:

- `pretrain_gpt.py` does support both `local` and `transformer_engine` layer specs.
- However, in this fork, importing `pretrain_gpt.py` still pulls code paths that import `megatron.core.extensions.transformer_engine`.
- In practice, `python pretrain_gpt.py --help` failed without `transformer_engine`, even when the intended runtime flag was `--transformer-impl local`.

What this means:

- `TRANSFORMER_IMPL=local` still controls which GPT layer spec is selected.
- But the runtime environment still needs the `transformer_engine` Python package installed.
- Apex is not strictly required for basic bring-up; the current validated path falls back to Torch norm when Apex is absent.

Practical recommendation:

- Treat TransformerEngine as a required dependency for local bring-up in this repo.
- Treat Apex as optional until a later runtime error proves otherwise.
- Keep `TRANSFORMER_IMPL=local` for the first sanity-check runs unless you specifically want TE-backed model layers.

## Is CUDA 12.4 required?

No evidence in this repo suggests a hard requirement on CUDA 12.4 specifically.

What I found:

- `scripts/config.sh` on the cluster side loads `cuda/12.0.1`
- the local experiment scripts themselves do not pin CUDA 12.4
- Megatron-LM docs in the repo reference multiple NVIDIA container versions, not a single mandatory CUDA 12.4 stack
- TransformerEngine packaging in-tree is generally aligned with CUDA 12 wheels, not uniquely 12.4

What is actually required in practice:

- a working CUDA stack compatible with your installed PyTorch build
- GPU support for the chosen precision mode
- if you use `--bf16`, hardware and software support for bf16 training

So the real requirement is compatibility between PyTorch, CUDA, NCCL, and optional TransformerEngine, not CUDA 12.4 specifically.

## Environment reality check from the validated local Docker setup

The local bring-up that actually reached `pretrain_gpt.py --help` used:

- Docker container based on `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04`
- container name: `flame-moe-3090`
- Conda installed under `/workspace/FLAME-MoE/.conda`
- Conda env name: `flame3090`
- Python `3.10`
- PyTorch `2.5.1+cu121`
- `torchrun` works
- `transformer_engine` import works
- `python Megatron-LM/pretrain_gpt.py --help` prints help successfully

Useful re-entry commands:

```bash
docker start flame-moe-3090
docker exec -it flame-moe-3090 bash
source /workspace/FLAME-MoE/.conda/etc/profile.d/conda.sh
conda activate flame3090
```

## Precision note for RTX 3090

The local scripts currently pass `--bf16` by default.

For first bring-up on RTX 3090, `fp16` is the safer default:

- Ampere consumer GPUs do not make bf16 the safest assumption for Megatron bring-up.
- The first goal here is not optimal performance, but getting Stage A to run reliably.
- In practice, switching the local scripts from `--bf16` to `--fp16` is the recommended first sanity-check path.

So yes, this recommendation is specifically tied to the current local target hardware:

- RTX 3090 local server bring-up first
- reliability before optimization
- avoid spending time debugging bf16-specific behavior if fp16 already satisfies the smoke test goal

## Wikipedia subset note

For the local smoke test, a small Stage A subset was prepared instead of a large training corpus.

What was done:

- source dataset: `wikimedia/wikipedia`
- config used successfully: `20231101.en`
- subset size: `20,000` streamed English Wikipedia documents
- raw output directory: `$LOCAL_BASE/dataset/wikipedia/raw`
- tokenized output directory: `$LOCAL_BASE/dataset/wikipedia/tokenized/EleutherAI/pythia-12b`

Important implementation notes from the actual run:

- `20220301.en` was not available in the current `datasets` package view; `20231101.en` worked.
- In this repo's `Megatron-LM/tools/preprocess_data.py`, `--chunk-size` was not a valid flag.
- `--json-keys text` worked; `--json-key text` was not the correct CLI spelling for this version.

This subset is only for pipeline validation:

- enough to confirm tokenization and Stage A startup
- far too small to represent a real `10B`-token Stage A corpus

## Suggested next steps

1. Switch the local Stage A script from `--bf16` to `--fp16` or make precision configurable.
2. Run a 4x3090 Stage A sanity-check on GPUs `4,5,6,7`.
3. After Stage A bring-up succeeds, decide whether to keep the small Wikipedia subset only for smoke testing or prepare a larger public Stage A corpus.
4. Expand with Stage B to `NUM_EXPERTS=7` or `8`.
5. Track:
   - Stage A validation perplexity before and after Stage B
   - Stage B in-domain validation perplexity
   - router usage and old-vs-new expert utilization
6. If the goal is closer Figure 3 fidelity, add an intermediate multilingual stage
