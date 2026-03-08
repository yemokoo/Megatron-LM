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

Not strictly for this local experiment.

Why:

- `pretrain_gpt.py` supports both `local` and `transformer_engine`.
- `arguments.py` defaults to `transformer_engine`, but the local scripts now pass `--transformer-impl "$TRANSFORMER_IMPL"` explicitly.
- The local GPT layer path supports MoE without TransformerEngine.

Important nuance:

- TransformerEngine is still useful for performance and for some optimized kernels.
- Some advanced paths in Megatron require it, but the current Stage A/B setup does not.
- Because these scripts do not enable FP8 or grouped GEMM, they do not rely on the main TE-only MoE fast paths.

Practical recommendation:

- Start with `TRANSFORMER_IMPL=local`
- Move to `transformer_engine` only if you specifically want speed or parity with a TE-based cluster environment

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

## Environment reality check from this workspace

In the current shell environment used for inspection:

- `python3` exists
- `torch` is not installed in that interpreter
- `transformer_engine` is not installed in that interpreter

This does not prove your training environment is broken, because the actual training scripts rely on the target runtime environment, not necessarily the inspection shell. But it does mean the current shell cannot launch training as-is.

## Suggested next steps

1. Prepare Stage A proxy data under `$LOCAL_BASE/dataset/wikipedia/tokenized/EleutherAI/pythia-12b`
2. Run a small Stage A baseline such as `NUM_EXPERTS=4`
3. Expand with Stage B to `NUM_EXPERTS=7` or `8`
4. Track:
   - Stage A validation perplexity before and after Stage B
   - Stage B in-domain validation perplexity
   - router usage and old-vs-new expert utilization
5. If the goal is closer Figure 3 fidelity, add an intermediate multilingual stage
