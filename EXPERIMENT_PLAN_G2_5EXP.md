
↓ lower is better · below 0 = backward transfer · C = undefined# G2 Continual-Learning Experiment Plan (5 experiments)

Last updated: 2026-07-02

Goal: compare, on an **active-parameter-matched** basis and extended to **3 tasks
(wiki → code → conversation)**, how the transformer body (layers 2–9) behaves when
it is dense vs a fixed MoE vs an expandable MoE — and bound the best achievable
score with joint (mixed) training.

All models share **layer 1 = dense FFN 5472** and differ **only in layers 2–9**, so
the comparison isolates the layers-2–9 change. Active FFN width is 1408 everywhere
(top-k × moe_ffn = 4×352 = 1×1408).

Common settings: hidden 1024, 9 layers, `moe_layer_freq=[0,1,1,1,1,1,1,1,1]`,
GQA 16 groups (MHA), seed 1234, global batch 2304, 1800 steps/task.

Environment prerequisite (KT session): restore deps before running (see
`KT_SESSION_RESTART_RUNBOOK.md`). MoE runs (exp 2, 3, 5) need `grouped_gemm`
(`git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`). exp 1 & 4 use a 1-expert
MoE with `MOE_GROUPED_GEMM=0` so they do not need grouped_gemm. For live metrics run
W&B online (`WANDB_MODE=online` + API login); use `WANDB_PROJECT=""` to disable W&B.

--------------------------------------------------------------------------------
## Architecture summary

| exp | model | layer 1 | layers 2–9 | how (config) |
|-----|-------|---------|-----------|--------------|
| 1,4 | dense(active) | dense 5472 | **dense 1408** | num_experts=1, topk=1, moe_ffn=1408, aux/z=0 |
| 2,5 | fixed24 MoE | dense 5472 | **MoE 24×352, top-4** | num_experts=24, topk=4, moe_ffn=352 |
| 3   | FFN-only expand + attn | dense 5472 | **MoE 8→16→24, top-4** | expand, old experts frozen, attention trained |

Total params (FFN-only 24-expert): ≈365M. dense(active) total ≈ (fewer, layers 2–9 are 1408).
All model config comes from `configs/model/flame-moe-ffn-only-no-shared.sh` (no shared
expert, no attention experts). dense(active) = that config with num_experts=1/topk=1.

--------------------------------------------------------------------------------
## A. Sequential continual learning (one dataset at a time, wiki→code→conv)

Total 5400 steps = 3 × 1800. Each stage logs all three probes (wiki/code/conversation).

### Exp 1 — dense(active), sequential
Script: `scripts/experiment/a100/run_g2_dense_active_matched_wiki_code_conversation_mha.sh`
- From scratch. wiki→code→conv, full finetune each stage.
- KD default kl1 (`OLD_MODEL_KL_COEFF=1.0`, adjustable; `0.0` = kl0).
```bash
WANDB_MODE=online \
bash scripts/experiment/a100/run_g2_dense_active_matched_wiki_code_conversation_mha.sh
```
Output root: `.local/weights/a100/mha/dense-active-matched/{wiki,code,conversation}/`

### Exp 2 — fixed24 MoE, sequential
Script: `scripts/experiment/a100/run_g2_fixed24_wiki_code_conversation_mha.sh`
- From scratch, 24 experts from wiki. wiki→code→conv full finetune (no freeze).
- Needs grouped_gemm (or set `MOE_GROUPED_GEMM=0` for SequentialMLP).
```bash
WANDB_MODE=online \
bash scripts/experiment/a100/run_g2_fixed24_wiki_code_conversation_mha.sh
```
Output root: `.local/weights/a100/mha/fixed24/{wiki,code,conversation}/`

### Exp 3 — FFN-only expandable MoE, conversation with ATTENTION UNFREEZE
Script: `scripts/experiment/a100/run_g2_exp3_ffn_only_attn_unfreeze_conversation_mha.sh`
- Continues from the wiki→code router-retuned 16-expert checkpoint:
  `FFN_SOURCE=.local/weights/a100/mha/g2-checkpoints/code/phase3/g2matched-attn-freeze-phase3-router-only-retune-wikicode-no-reinit-mb96-1800` (iter 3600).
- Expands 16→24; **old experts/router FROZEN**, **attention (shared) TRAINED**.
- Only the conversation stage (1800 steps). The FREEZE counterpart already exists
  as a baseline; this run adds only the attention-unfreeze variant.
- Implemented as `RUN_ONLY_STAGE=ffn_only_attn_unfreeze` in the phase4 chain
  (`FREEZE_SHARED=1` + `TRAIN_ATTENTION_WITH_NEW_EXPERTS=1`).
```bash
WANDB_MODE=online \
bash scripts/experiment/a100/run_g2_exp3_ffn_only_attn_unfreeze_conversation_mha.sh
```
Output: `.local/weights/a100/mha/g2-checkpoints/conversation/phase4/g2-ffn-only-attn-unfreeze-phase4-conversation-from-router-retuned-e16to24-mb96-1800`

--------------------------------------------------------------------------------
## B. Joint / mixed upper bound (all 3 datasets shuffled, 1:1:1)

Single stage, **5400 steps** (= sequential total). Blend is 1:1:1 (each dataset shard
gets weight 1.0 in `--data-path`). No forgetting → performance ceiling. Same
architectures as exp 1 / exp 2. Generalized trainer:
`scripts/experiment/a100/pretrain_mixed_3way_local_bf16.sh`.

### Exp 4 — dense(active), mixed
Script: `scripts/experiment/a100/run_g2_exp4_dense_active_mixed_wiki_code_conv_mha.sh`
```bash
WANDB_MODE=online \
bash scripts/experiment/a100/run_g2_exp4_dense_active_mixed_wiki_code_conv_mha.sh
```
Output: `.local/weights/a100/mha/mixed-3way/dense-active/`

### Exp 5 — fixed24 MoE, mixed
Script: `scripts/experiment/a100/run_g2_exp5_fixed24_mixed_wiki_code_conv_mha.sh`
```bash
WANDB_MODE=online \
bash scripts/experiment/a100/run_g2_exp5_fixed24_mixed_wiki_code_conv_mha.sh
```
Output: `.local/weights/a100/mha/mixed-3way/fixed24/`

--------------------------------------------------------------------------------
## Run order / notes

- Exp 1, 2 are self-contained (from scratch). Exp 3 needs the existing `FFN_SOURCE`
  code checkpoint (already present, iter 3600). Exp 4, 5 are single-stage.
- Only one training job per GPU pair; scripts default to GPUs 0,1 (override
  `CUDA_VISIBLE_DEVICES`, and `MICRO_BATCH_SIZE` per memory).
- Micro-batch defaults: dense(active) 72, fixed24 48 (override as needed).
- KD (sequential code/conv): default kl1; set `OLD_MODEL_KL_COEFF=0.0` for kl0.
- `/tmp/flame-moe` is used for dataset/checkpoint staging — clean stale copies.

## Result extraction (probe accuracy per stage)

Parse `logs/*.log` for lines:
`probe (wiki_probe|code_probe|conversation_probe) at iteration N | local_iteration: L | next_token_acc: A | ppl: P`
(see `RESEARCH_CODE_HANDOFF.md` §8 for the extractor). Build the 3×3 table
(training stage × probe) per model; compare sequential (1,2,3) against the mixed
upper bounds (4,5).

--------------------------------------------------------------------------------
## Files created for this plan

- `configs/model/flame-moe-ffn-only-no-shared.sh` — FFN-only MoE config (no shared expert)
- `scripts/experiment/a100/run_g2_fixed24_wiki_code_conversation_mha.sh` — exp 2
- `scripts/experiment/a100/run_g2_dense_active_matched_wiki_code_conversation_mha.sh` — exp 1 (wraps exp 2 engine, e1/top1)
- `scripts/experiment/a100/run_g2_exp3_ffn_only_attn_unfreeze_conversation_mha.sh` — exp 3
- `scripts/experiment/a100/pretrain_mixed_3way_local_bf16.sh` — 3-way mixed trainer
- `scripts/experiment/a100/run_g2_exp4_dense_active_mixed_wiki_code_conv_mha.sh` — exp 4
- `scripts/experiment/a100/run_g2_exp5_fixed24_mixed_wiki_code_conv_mha.sh` — exp 5
- edited `scripts/experiment/a100/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh`
  (added `ffn_only_attn_unfreeze` stage; freeze flags now overridable)

Status: implemented + unit/arch-verified (args, py_compile, pytest 19 passed).
Actual training pending KT env restore (grouped_gemm for exp 2/3/5).
