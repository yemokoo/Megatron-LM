# FLAME-MoE: Developer Guide & Codebase Walkthrough

This document explains how the codebase is structured and **exactly where to make changes** when you want to modify the architecture, number of experts, or training data.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Repository Layout](#2-repository-layout)
3. [How a Training Run Actually Works](#3-how-a-training-run-actually-works)
4. [Where to Modify: Architecture](#4-where-to-modify-architecture)
5. [Where to Modify: Number of Experts](#5-where-to-modify-number-of-experts)
6. [Where to Modify: Training Data](#6-where-to-modify-training-data)
7. [Where to Modify: Training Hyperparameters](#7-where-to-modify-training-hyperparameters)
8. [Running a Custom Model](#8-running-a-custom-model)
9. [Evaluation](#9-evaluation)
10. [Empirical Analysis Tools](#10-empirical-analysis-tools)
11. [External Dependencies (Submodules)](#11-external-dependencies-submodules)
12. [Continual Learning Routing Experiment](#12-continual-learning-routing-experiment)

---

## 1. High-Level Architecture

FLAME-MoE is a **Mixture-of-Experts (MoE) language model** trained on top of [Megatron-LM](https://github.com/yuzc19/Megatron-LM). Key design choices baked into the released configs:

| Property | Value |
|---|---|
| Attention heads | 16 (all sizes) |
| Position embeddings | RoPE |
| Normalization | RMSNorm (ε = 1e-6) |
| Activation | SwiGLU |
| Tokenizer | EleutherAI/pythia-12b (vocab ~50277) |
| Context length | 2048 tokens |
| Number of experts | **64** (routed) + 1 shared expert |
| Top-k routing | 6 experts per token |
| MoE layer pattern | Layer 0 = dense, layers 1..N-1 = MoE |
| Training data | DCLM-138B, tokenized with pythia-12b tokenizer |
| Precision | bfloat16 |
| Optimizer | Distributed Adam (via Megatron) |
| LR schedule | WSD (Warmup-Stable-Decay) |

The model is trained using `Megatron-LM/pretrain_gpt.py`, launched via `torchrun` on SLURM.

---

## 2. Repository Layout

```
LLM-continual-learning/
│
├── configs/                    # Reusable config fragments (sourced by training scripts)
│   ├── model/
│   │   ├── flame-moe.sh        # ★ MoE model architecture args
│   │   └── dclm.sh             # ★ Dense (DCLM baseline) model architecture args
│   └── train/
│       ├── flame-moe.sh        # ★ MoE infra + training hyperparameters
│       └── dclm.sh             # ★ Dense infra + training hyperparameters
│
├── scripts/
│   ├── config.sh               # GCP/SSD path setup + conda env activation
│   │
│   ├── release/                # ★ One script per released model size — START HERE
│   │   ├── flame-moe-1.7b.sh   # Sets env vars → calls training/flame-moe.sh
│   │   ├── flame-moe-721m.sh
│   │   ├── flame-moe-419m.sh
│   │   ├── flame-moe-290m.sh
│   │   ├── flame-moe-115m.sh
│   │   ├── flame-moe-98m.sh
│   │   ├── flame-moe-38m.sh
│   │   ├── dclm-1b-1x.sh       # Dense baseline
│   │   ├── dclm-411m-1x.sh
│   │   └── dclm-411m-4x.sh
│   │
│   ├── training/
│   │   ├── flame-moe.sh        # SLURM job header + dataset path + calls step1/step2
│   │   ├── dclm.sh             # Same for dense model
│   │   └── modules/
│   │       ├── flame-moe_step1.sh   # Downloads dataset from GCP → SSD
│   │       ├── flame-moe_step2.sh   # ★ Actual torchrun launch for MoE training
│   │       ├── dclm_step1.sh        # Downloads dataset from GCP → SSD
│   │       └── dclm_step2.sh        # ★ Actual torchrun launch for dense training
│   │
│   ├── dataset/
│   │   ├── download.sh         # Downloads raw DCLM data from S3
│   │   ├── tokenize.sh         # Tokenizes raw text to .bin files
│   │   └── modules/
│   │       ├── download_dclm_step1.sh
│   │       └── tokenize_step1.sh
│   │
│   ├── evaluate.sh             # Runs lm-evaluation-harness on a saved checkpoint
│   │
│   ├── ablation/
│   │   ├── generate.py         # Reads a Google Sheet config → generates search.sh
│   │   ├── search.sh           # SLURM array job for hyperparameter search
│   │   ├── evaluate.sh
│   │   └── testing.sh
│   │
│   └── empirical_analysis/     # Scripts for capturing router activations
│       ├── capture-*.sh        # Capture router outputs for a given model size
│       ├── expert_coactivation-*.sh
│       ├── router_saturation-*.sh
│       └── modules/            # Step-by-step sub-scripts for each analysis
│
├── empirical_analysis/
│   ├── work/
│   │   └── expert_specialization.py  # Compute per-expert token specialization scores
│   └── plot/
│       ├── expert_specialization.ipynb
│       └── router_saturation.ipynb
│
├── analysis/                   # Scaling law analysis notebooks + scripts
│   ├── find_scaling_law.ipynb
│   ├── plot_scaling_law.py
│   ├── plot_fitted_law.py
│   ├── expert_coactivation.py
│   ├── router_saturation.py
│   └── infrastructure.py
│
└── Megatron-LM/                # Git submodule — the actual training engine
    └── pretrain_gpt.py         # Entry point called by torchrun
```

---

## 3. How a Training Run Actually Works

The call chain for a MoE training run (`flame-moe-1.7b.sh`) is:

```
scripts/release/flame-moe-1.7b.sh
  │  Sets: NUM_LAYERS, HIDDEN_SIZE, FFN_HIDDEN_SIZE, MOE_FFN_HIDDEN_SIZE,
  │         MOE_LAYER_FREQ, MICRO_BATCH_SIZE, PIPELINE_MODEL_PARALLEL_SIZE,
  │         EXPERT_MODEL_PARALLEL_SIZE, TRAIN_ITERS, SAVE_INTERVAL
  └─→ sbatch scripts/training/flame-moe.sh
        │  Sets: TRAIN_DATASET path, TRAIN_WEIGHTS path, WandB env vars
        ├─→ srun scripts/training/modules/flame-moe_step1.sh
        │       (downloads tokenized dataset from GCP to local SSD)
        └─→ srun scripts/training/modules/flame-moe_step2.sh
                │  Sources: configs/model/flame-moe.sh  → builds MODEL_ARGS
                │  Sources: configs/train/flame-moe.sh  → builds TORCH_ARGS, INFRA_ARGS, TRAIN_ARGS
                └─→ torchrun Megatron-LM/pretrain_gpt.py
                        (all MODEL_ARGS + INFRA_ARGS + TRAIN_ARGS + DATA_ARGS + SAVE_ARGS)
```

**The `release/` scripts are the single point of control.** They export env vars that cascade through every layer below.

---

## 4. Where to Modify: Architecture

### Option A — Edit an existing release script

Open any file in [scripts/release/](scripts/release/) and change the exported variables:

```bash
# scripts/release/flame-moe-1.7b.sh  (example — current values)
export NUM_LAYERS=18          # total transformer layers
export HIDDEN_SIZE=2048       # attention hidden dim (d_model)
export FFN_HIDDEN_SIZE=10944  # hidden dim of the dense FFN (layer 0 only)
export MOE_FFN_HIDDEN_SIZE=1408  # hidden dim of each MoE expert FFN
export MOE_LAYER_FREQ="[0]*1+[1]*17"  # layer 0 = dense, layers 1-17 = MoE
```

**`MOE_LAYER_FREQ` explained:**
- `"[0]*1+[1]*17"` means: 1 dense layer followed by 17 MoE layers
- To make the first 3 layers dense: `"[0]*3+[1]*15"`
- To make all layers MoE: `"[1]*18"`
- This gets passed directly to Megatron-LM as `--moe-layer-freq`

### Option B — Create a new release script

Copy the closest existing size and adjust:

```bash
cp scripts/release/flame-moe-290m.sh scripts/release/my-custom-model.sh
# then edit NUM_LAYERS, HIDDEN_SIZE, etc.
bash scripts/release/my-custom-model.sh
```

### Where the args go

The env vars you set in the release script are consumed in [configs/model/flame-moe.sh](configs/model/flame-moe.sh):

```bash
# configs/model/flame-moe.sh
--hidden-size $HIDDEN_SIZE
--ffn-hidden-size $FFN_HIDDEN_SIZE
--num-layers $NUM_LAYERS
--num-attention-heads 16          # ← hardcoded; change here to modify
--max-position-embeddings 2048    # ← context length; change here
--moe-ffn-hidden-size $MOE_FFN_HIDDEN_SIZE
--num-experts 64                  # ← SEE SECTION 5
--moe-router-topk 6               # ← how many experts each token uses
--moe-layer-freq $MOE_LAYER_FREQ
```

**To change attention heads or context length**, edit [configs/model/flame-moe.sh](configs/model/flame-moe.sh) directly — those values are hardcoded there, not set via env vars.

### Model size quick reference

| Script | Layers | Hidden | MoE FFN hidden | Approx params |
|---|---|---|---|---|
| `flame-moe-38m.sh` | 9 | 256 | 176 | ~38M |
| `flame-moe-98m.sh` | (see file) | (see file) | (see file) | ~98M |
| `flame-moe-115m.sh` | (see file) | (see file) | (see file) | ~115M |
| `flame-moe-290m.sh` | 9 | 1024 | 704 | ~290M |
| `flame-moe-419m.sh` | (see file) | (see file) | (see file) | ~419M |
| `flame-moe-721m.sh` | 12 | 1536 | 1056 | ~721M |
| `flame-moe-1.7b.sh` | 18 | 2048 | 1408 | ~1.7B |

---

## 5. Where to Modify: Number of Experts

All expert-count settings live in [configs/model/flame-moe.sh](configs/model/flame-moe.sh):

```bash
--num-experts 64                              # total routed experts per MoE layer
--moe-router-topk 6                           # tokens route to this many experts
--moe-shared-expert-intermediate-size $((2 * MOE_FFN_HIDDEN_SIZE))  # 1 shared expert, always active
```

**To change the number of experts**, edit `--num-experts` in [configs/model/flame-moe.sh](configs/model/flame-moe.sh).

**Important constraints when changing expert count:**
- `EXPERT_MODEL_PARALLEL_SIZE` (set in the release script) must evenly divide `num-experts`. Currently `EXPERT_MODEL_PARALLEL_SIZE=8` and `num-experts=64` → 8 experts per GPU.
- If you set `--num-experts 32`, you could use `EXPERT_MODEL_PARALLEL_SIZE=4` or `8`.
- The expert specialization analysis script ([empirical_analysis/work/expert_specialization.py](empirical_analysis/work/expert_specialization.py)) hardcodes `minlength=50277` (vocab size) and `assert 0 <= parsed.expert_index <= 63` — update that bound if you change expert count.

**To change top-k routing** (how many experts each token uses), edit `--moe-router-topk`. The auxiliary loss and balance will be affected.

---

## 6. Where to Modify: Training Data

### Changing the dataset path

The training dataset path is set in [scripts/training/flame-moe.sh](scripts/training/flame-moe.sh):

```bash
export TRAIN_DATASET="${TRAIN_DATASET:-$GCP_DATASET/dclm-138b/tokenized/EleutherAI/pythia-12b}"
```

The `:-` syntax means: use this default **unless** `TRAIN_DATASET` is already set in the environment. So you can override it from outside without editing the file:

```bash
export TRAIN_DATASET=/path/to/your/tokenized/data
bash scripts/release/flame-moe-290m.sh
```

Or just hard-code your path in [scripts/training/flame-moe.sh](scripts/training/flame-moe.sh) line 25.

### What format the data must be in

The training script ([scripts/training/modules/flame-moe_step2.sh](scripts/training/modules/flame-moe_step2.sh)) builds the `--data-path` arg by scanning for `.bin` files:

```bash
--data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
```

This means your data must be in **Megatron-LM's binary format** (`.bin` + `.idx` pairs). Each dataset shard is weighted `1.0` (equal weight). To use different mixture weights, you would modify this `find` command to emit different weights per file.

### Tokenizing your own data

Use [scripts/dataset/tokenize.sh](scripts/dataset/tokenize.sh), which calls [scripts/dataset/modules/tokenize_step1.sh](scripts/dataset/modules/tokenize_step1.sh). The tokenizer is `EleutherAI/pythia-12b` (set in [configs/model/flame-moe.sh](configs/model/flame-moe.sh) via `--tokenizer-model`). If you use a different tokenizer, change both the tokenization step and the `--tokenizer-model` arg.

### Train/val/test split

Set in [scripts/training/modules/flame-moe_step2.sh](scripts/training/modules/flame-moe_step2.sh):

```bash
--split 90,5,5   # 90% train, 5% val, 5% test
```

---

## 7. Where to Modify: Training Hyperparameters

All training hyperparameters are in [configs/train/flame-moe.sh](configs/train/flame-moe.sh):

```bash
# configs/train/flame-moe.sh
--global-batch-size 1024       # effective batch size (tokens = 1024 * 2048)
--lr 3e-4                      # peak learning rate
--min-lr 3e-5                  # final LR after decay
--lr-decay-style WSD           # Warmup-Stable-Decay schedule
--lr-warmup-fraction 0.01      # 1% of training steps = warmup
--lr-wsd-decay-iters $((TRAIN_ITERS / 10))  # last 10% of steps = decay
--train-iters $TRAIN_ITERS     # total gradient steps (set by release script)
```

**`TRAIN_ITERS`** is set per-model in the release scripts. It corresponds roughly to a Chinchilla-optimal token budget for each model size.

**MoE-specific loss coefficients** are in [configs/model/flame-moe.sh](configs/model/flame-moe.sh):

```bash
--moe-aux-loss-coeff 0.01   # load balancing auxiliary loss weight
--moe-z-loss-coeff 0.001    # router z-loss (stabilizes softmax)
```

---

## 8. Running a Custom Model

Here is the minimal set of edits to train a custom MoE configuration:

### Step 1: Create a release script

```bash
cp scripts/release/flame-moe-290m.sh scripts/release/my-model.sh
```

Edit `my-model.sh`:

```bash
#!/bin/bash
export NUM_LAYERS=12              # your layer count
export HIDDEN_SIZE=768            # your hidden dim
export FFN_HIDDEN_SIZE=4096       # dense layer FFN (usually ~4x hidden)
export MOE_FFN_HIDDEN_SIZE=512    # per-expert FFN dim
export MOE_LAYER_FREQ="[0]*1+[1]*11"  # 1 dense + 11 MoE layers
export MICRO_BATCH_SIZE=8
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8   # must divide num-experts (64)
export TRAIN_ITERS=5000
export SAVE_INTERVAL=500
export EVAL_INTERVAL=500
sbatch --job-name=my-model --nodes=4 scripts/training/flame-moe.sh
```

### Step 2: (Optional) Change expert count

Edit [configs/model/flame-moe.sh](configs/model/flame-moe.sh):

```bash
--num-experts 32          # change from 64 to whatever you want
--moe-router-topk 4       # adjust top-k accordingly
```

Also update `EXPERT_MODEL_PARALLEL_SIZE` in your release script to divide evenly into `num-experts`.

### Step 3: (Optional) Point to custom data

```bash
export TRAIN_DATASET=/your/data/path   # before calling the release script
```

### Step 4: Launch

```bash
bash scripts/release/my-model.sh
```

---

## 9. Evaluation

```bash
export JOBID=<slurm_job_id>
export ITER=<checkpoint_iteration>
sbatch scripts/evaluate.sh
```

[scripts/evaluate.sh](scripts/evaluate.sh) runs `lm-evaluation-harness` on:
- 0-shot: `openbookqa`, `winogrande`
- 10-shot: `piqa`, `arc_easy`, `arc_challenge`, `hellaswag`

To add or remove benchmarks, edit the arrays `num_fewshots` and `fewshot_tasks` inside [scripts/evaluate.sh](scripts/evaluate.sh).

---

## 10. Empirical Analysis Tools

These are research tools for studying what the trained experts learn:

| Script / Notebook | What it does |
|---|---|
| [empirical_analysis/work/expert_specialization.py](empirical_analysis/work/expert_specialization.py) | For each expert, counts which token IDs it activates on → specialization score |
| [scripts/empirical_analysis/capture-*.sh](scripts/empirical_analysis/) | Captures router logits/activations from a checkpoint to disk |
| [scripts/empirical_analysis/expert_coactivation-*.sh](scripts/empirical_analysis/) | Analyzes which pairs of experts fire together |
| [scripts/empirical_analysis/router_saturation-*.sh](scripts/empirical_analysis/) | Studies how router load balances across training |
| [analysis/find_scaling_law.ipynb](analysis/find_scaling_law.ipynb) | Fits a scaling law to loss vs. compute |

The specialization analysis script assumes `--num-experts 64` (hardcodes `assert 0 <= parsed.expert_index <= 63`). Update [empirical_analysis/work/expert_specialization.py:73](empirical_analysis/work/expert_specialization.py) if you change the expert count.

---

## 11. External Dependencies (Submodules)

The repo uses four git submodules:

| Submodule | Role |
|---|---|
| `Megatron-LM` (yuzc19 fork, `multi-nodes` branch) | Core training engine — `pretrain_gpt.py` is the main entry point |
| `apex` (NVIDIA) | Fused CUDA kernels for optimizer and layer norm |
| `TransformerEngine` (NVIDIA, v1.11) | FP8/BF16 attention kernels |
| `lm-evaluation-harness` (yuzc19 fork, `megatron` branch) | Evaluation harness with Megatron checkpoint loading |

To update or swap a submodule:

```bash
cd Megatron-LM
git fetch origin
git checkout <new-branch-or-commit>
cd ..
git add Megatron-LM
git commit -m "update Megatron-LM submodule"
```

---

## 12. Continual Learning Routing Experiment

This experiment investigates whether training on a new data distribution (Task B) disrupts the routing patterns formed during pretraining on a previous distribution (Task A). This is the core problem that Lifelong-MoE-style methods aim to solve.

**Hypothesis to test:** After continuing training on Task B (Python code), does the router assign A-domain tokens (Wikipedia) to different experts than it did right after Task A training?

### Datasets

| Stage | Dataset | HuggingFace ID | Distribution |
|---|---|---|---|
| Task A | English Wikipedia | `wikimedia/wikipedia` (20220301.en) | Encyclopedic prose |
| Task B | Python code | `codeparrot/codeparrot-clean` | Source code |

### New Scripts

```
scripts/
├── dataset/
│   ├── download_wikipedia.sh    # Download + tokenize Wikipedia (Task A)
│   └── download_code.sh         # Download + tokenize Python code (Task B)
└── experiment/
    ├── stage_A_train.sh         # Train FLAME-MoE-290M on Wikipedia
    ├── stage_B_train.sh         # Continue training on code from Stage A checkpoint
    └── capture_routing.sh       # Capture router traces (always evaluated on Wikipedia)
```

### Step-by-Step Execution

```bash
# 0. Create log directories
mkdir -p logs/download-wikipedia logs/download-code \
         logs/continual-stage-A logs/continual-stage-B logs/capture-routing

# 1. Prepare datasets (can run in parallel)
sbatch scripts/dataset/download_wikipedia.sh
sbatch scripts/dataset/download_code.sh

# 2. Train Stage A: Wikipedia pretraining (~500 iters, ~1B tokens)
sbatch scripts/experiment/stage_A_train.sh
#   → note the SLURM job ID printed at the end (e.g. 12345)

# 3. Train Stage B: continue from Stage A on Python code
sbatch --export=STAGE_A_JOB_ID=12345 scripts/experiment/stage_B_train.sh
#   → note the SLURM job ID (e.g. 12346)

# 4. Capture routing on Wikipedia samples (A-domain) for both checkpoints
sbatch --export=CAPTURE_JOB_ID=12345,CAPTURE_STAGE=A scripts/experiment/capture_routing.sh
sbatch --export=CAPTURE_JOB_ID=12346,CAPTURE_STAGE=B scripts/experiment/capture_routing.sh
```

### Output

After Step 4, routing traces are saved to:
```
actives/continual-stage-A/<job_id>/   ← routing at each iter of Stage A (on Wikipedia)
actives/continual-stage-B/<job_id>/   ← routing at each iter of Stage B (still on Wikipedia)
```

Compare the two with the existing analysis tools in [empirical_analysis/work/expert_specialization.py](empirical_analysis/work/expert_specialization.py) and [analysis/](analysis/). If routing is disrupted, different experts will be selected for the same Wikipedia tokens after Stage B training.

### Scale Controls

To make the experiment faster/smaller, edit these variables in the stage scripts:

| Variable | Location | Default | Effect |
|---|---|---|---|
| `TRAIN_ITERS` | `stage_A_train.sh`, `stage_B_train.sh` | 500 | Total gradient steps per stage |
| `SAVE_INTERVAL` | both stage scripts | 50–100 | How often to save checkpoints for capture |
| `max_examples` | `download_code.sh` (Python) | 200,000 | Size of Task B dataset |
| `NODES` | `#SBATCH --nodes` | 2 | Compute resources |

---

## Quick Reference: Key Files at a Glance

| What you want to change | File to edit |
|---|---|
| Layer count, hidden size, FFN dim | `scripts/release/<model>.sh` |
| Number of experts (e.g. 64 → 32) | `configs/model/flame-moe.sh` — `--num-experts` |
| Top-k routing | `configs/model/flame-moe.sh` — `--moe-router-topk` |
| Which layers are MoE vs. dense | `scripts/release/<model>.sh` — `MOE_LAYER_FREQ` |
| Attention heads, context length | `configs/model/flame-moe.sh` — `--num-attention-heads`, `--max-position-embeddings` |
| Training data path | `scripts/training/flame-moe.sh` line 25 or `$TRAIN_DATASET` env var |
| Data format / mixture weights | `scripts/training/modules/flame-moe_step2.sh` — `DATA_ARGS` |
| Learning rate, batch size, schedule | `configs/train/flame-moe.sh` |
| Total training iterations | `scripts/release/<model>.sh` — `TRAIN_ITERS` |
| Load-balancing loss strength | `configs/model/flame-moe.sh` — `--moe-aux-loss-coeff` |
| Evaluation benchmarks | `scripts/evaluate.sh` — `fewshot_tasks` array |
| SLURM resource allocation | `scripts/training/flame-moe.sh` — `#SBATCH` headers |
