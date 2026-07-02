# LLM Continual Learning Research and Code Handoff

Last updated: 2026-07-02

This document is a detailed handoff for the current MoE continual-learning project.
It is written so that a fresh Codex/KT session can understand the research direction,
the major code paths, the experiment naming conventions, the important results, and
the operational traps that have repeatedly mattered.

For KT environment recovery commands, see `KT_SESSION_RESTART_RUNBOOK.md`.
For planned experiment details, see `todoexp`.
For older broad experiment indexing, see `experiments.md`.

## 1. Research Goal

The project studies continual learning with Mixture-of-Experts (MoE) style model
growth. The central goal is:

- Add new task/domain knowledge while changing old useful parameters as little as possible.
- Preserve old-task performance, especially Wiki, while learning Code and later Conversation.
- Understand whether task-specific capacity expansion can reduce forgetting better than dense
  full-model finetuning.
- Diagnose whether the observed behavior is caused by expert capacity, router alignment,
  shared backbone co-adaptation, or attention-side experts.

The main task order used in recent G2 experiments is:

```text
wiki -> code -> conversation
```

The most important recurring measurements are:

- `wiki_probe/next_token_accuracy`
- `code_probe/next_token_accuracy`
- `conversation_probe/next_token_accuracy`
- corresponding probe perplexities
- LM loss when useful
- router/expert usage distributions
- hidden-state latent-space drift/recovery

## 2. Core G2 Model Conventions

Most recent experiments use the G2 setting.

Important defaults:

- Transformer hidden size: `1024`
- Number of layers: `9`
- MoE layer frequency: `[0,1,1,1,1,1,1,1,1]`
- This means layer 1 is dense/non-MoE and layers 2-9 are the main MoE/shared-router layers.
- FFN expert hidden size: `352`
- Router top-k: `4`
- Wiki stage expert count: `8`
- Code-expanded expert count: `16`
- Conversation-expanded expert count: `24`
- Global batch size: usually `2304`
- Standard seed policy for new runs: `1234`

Capacity arithmetic used throughout:

```text
1 FFN expert hidden size = 352
top-4 active FFN capacity = 4 * 352 = 1408
8 expert total capacity = 8 * 352 = 2816
16 expert total capacity = 16 * 352 = 5632
24 expert total capacity = 24 * 352 = 8448
```

Attention-side expert convention:

- Attention experts are QKVO LoRA-style experts attached to attention projections.
- The recent G2 shared-router attention expert rank is usually `256`.
- In shared-router hybrid models, FFN experts and attention experts share the same router decision.
- In attention-only expert models, FFN is dense and only attention QKVO has expert structure.

## 3. Terminology Used in Discussions

Use these terms consistently to avoid ambiguity:

- `FFN-only baseline`: MoE experts are in FFN only. Attention is frozen/no expert.
- `FFN + attention expert`: shared-router hybrid model with FFN experts and QKVO attention experts.
- `attention-only expert`: dense FFN, QKVO attention experts only.
- `dense baseline`: no MoE expert expansion; dense FFN is finetuned through tasks.
- `fixed16 MoE`: starts with 16 experts from Wiki stage; no 8-to-16 expansion.
- `expanded old-freeze shared-unfreeze`: Wiki learns 8 experts, Code expands to 16; old experts/router are frozen but shared trunk can train.
- `router finetune` or `router retune`: train router rows only; all experts and dense/shared parameters frozen.
- `KD` or `KL regularization`: old-model logits regularization used during full/shared finetuning baselines.
- `expanded capacity MoE`: task growth by adding experts as tasks increase.
- `unexpanded/fixed-capacity MoE`: all experts are available from the beginning.

## 4. Main Experiment Families

### 4.1 G2 FFN-Only Baseline

Purpose:

- Provide the cleanest task-expansion baseline.
- Existing parameters are frozen during new-task learning.
- New FFN experts and new router rows are trained for the new task.
- Attention is not expertized.

Typical path examples on KT:

```text
.local/weights/a100/mha/g2-checkpoints/code/g2matched/
.local/weights/a100/mha/g2-checkpoints/code/phase3/
.local/weights/a100/mha/g2-checkpoints/conversation/phase4/
.local/weights/a100/mha/g2-checkpoints/conversation/phase5_router_finetune_fullmix/
```

Representative code:

- `scripts/experiment/a100/wiki_ffn_moe_g2matched_mha_a100_bf16.sh`
- `scripts/experiment/a100/code_from_wiki_ffn_moe_g2matched_attn_freeze_mha_a100_bf16.sh`
- `scripts/experiment/a100/run_g2matched_phase3_router_only_retune_mha.sh`
- `scripts/experiment/a100/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh`
- `scripts/experiment/a100/run_g2_phase5_router_only_retune_after_conversation_fullmix_offline_chain_mha.sh`

Important observation:

- After Code, Wiki performance can drop.
- Router-only retuning on Wiki+Code often recovers much of Wiki while keeping Code high.
- After adding Conversation, FFN-only was surprisingly robust: Wiki sometimes improved after Conversation learning,
  likely because only new task-specific FFN capacity is added while the old useful subnetwork remains accessible.

### 4.2 G2 FFN + Attention Expert Shared-Router Models

Purpose:

- Test whether adding QKVO attention experts improves continual learning beyond FFN-only.
- The key structure is a shared router controlling both FFN expert selection and attention expert selection.

Representative code:

- `scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh`
- `scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh`
- `scripts/experiment/a100/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh`
- `configs/model/flame-shared-router-hybrid-experts.sh`
- `Megatron-LM/megatron/core/transformer/shared_router_hybrid.py`
- `Megatron-LM/megatron/core/models/gpt/shared_router_hybrid_layer_specs.py`

Common G2 shared-router settings:

```text
SOURCE_NUM_EXPERTS=8
NUM_EXPERTS=16
MOE_FFN_HIDDEN_SIZE=352
MOE_ROUTER_TOPK=4
ATTN_FULL_RANK_LORA_TARGETS=qkvo
ATTN_FULL_RANK_LORA_RANK=256
MOE_GROUPED_GEMM=1
ATTN_LORA_GROUPED_GEMM=1
SEED=1234
```

Important hypothesis:

- Attention experts did not always help as expected.
- The shared router may force FFN and attention experts into the same specialization pattern.
- This could be good if the two modules need aligned specialization, but bad if FFN and attention need different routing.

### 4.3 Conversation Extension

Purpose:

- Extend the task sequence to `wiki -> code -> conversation`.
- Test whether the tradeoff between old-task retention and new-task learning changes when a third task is added.
- See whether router retuning over `wiki+code+conversation` can recover all tasks after Conversation training.

Representative code:

- `scripts/experiment/a100/run_g2_phase4_conversation_after_router_finetune_offline_chain_mha.sh`
- `scripts/experiment/a100/run_g2_phase5_router_only_retune_after_conversation_fullmix_offline_chain_mha.sh`
- `scripts/analysis/relog_phase4_phase5_conversation_to_wandb.py`

Key result snapshot from logs:

```text
FFN-only phase4 after conversation:
  wiki acc 0.407037, code acc 0.347415, conversation acc 0.385033
FFN-only phase5 after router finetune:
  wiki acc 0.449889, code acc 0.651510, conversation acc 0.384467

Exp1 freeze-wiki phase4 after conversation:
  wiki acc 0.355520, code acc 0.300280, conversation acc 0.393481
Exp1 freeze-wiki phase5 after router finetune:
  wiki acc 0.397031, code acc 0.514123, conversation acc 0.393096

Exp2 unfreeze-wiki phase4 after conversation:
  wiki acc 0.330243, code acc 0.306725, conversation acc 0.407043
Exp2 unfreeze-wiki phase5 after router finetune:
  wiki acc 0.358788, code acc 0.415874, conversation acc 0.406458
```

Interpretation:

- FFN-only remained the strongest among these conversation extension runs.
- Router-only retuning can recover Code strongly for FFN-only while preserving Conversation.
- FFN+attention variants learned Conversation, but post-retune Code/Wiki recovery was weaker than FFN-only.

### 4.4 Router-Only Finetune Data Budget and Miniset Repeats

Purpose:

- Reduce the amount of data needed for router finetune.
- Previous full-data-budget experiments showed that around 20 percent of the full router-retune budget was already near saturation.
- New question: if the same tiny miniset is repeated for many epochs, can it match larger data budgets?

Implemented scripts:

- `scripts/experiment/a100/prepare_g2_router_finetune_miniset_repeats_mha.sh`
- `scripts/experiment/a100/run_g2_router_finetune_miniset_repeats_mha.sh`
- `scripts/analysis/relog_router_finetune_with_source_baseline_to_wandb.py`

Supported fixed-miniset variants:

```text
0.01 percent x 2000 epochs
0.1 percent x 200 epochs
1 percent x 20 epochs
5 percent x 4 epochs
10 percent x 2 epochs
100 percent full reference
```

Default semantics:

- Full 100 percent router-finetune budget: `3600` steps.
- 20 percent equivalent repeated-miniset budget: `720` steps.
- Fixed sampling seed: `1234`.
- Router finetune loss: LM loss only.
- Aux loss: `0.0`
- Z loss: `0.0`

Recent log-scale plot values used:

```text
No router finetune:
  wiki acc 0.3139
  code acc 0.6896

0.01 percent:
  wiki acc 0.3845
  code acc 0.6774

0.1 percent:
  wiki acc 0.3928
  code acc 0.6811

1 percent:
  wiki acc 0.3942
  code acc 0.6825

10 percent:
  wiki acc 0.3947
  code acc 0.6825

100 percent:
  wiki acc 0.3954
  code acc 0.6838
```

Interpretation:

- Router finetune is extremely data-efficient for Wiki recovery.
- Even very small repeated minisets recover most of the achievable Wiki gain.
- Code accuracy is slightly lower than the no-finetune checkpoint but remains close.

### 4.5 Interleaved Code Training and Router Finetune

Purpose:

- Test whether periodic router retuning during Code training prevents task manifolds/routing from drifting too far.
- Instead of training Code for all 1800 steps and then retuning router, alternate:

```text
code n steps -> router k steps -> code n steps -> router k steps -> ...
```

Implemented script:

- `scripts/experiment/a100/run_g2_exp2_interleaved_code_router_finetune_mha.sh`

Default intended setting:

```text
CODE_TOTAL_STEPS=1800
INTERLEAVE_CODE_STEPS=100
INTERLEAVE_ROUTER_STEPS=100
RUN_ROUTER_AFTER_FINAL=1
```

Important implementation detail:

- This is not one Python process with internal mode switching.
- Each chunk relaunches `torchrun`.
- Continuity is checkpoint-based.
- Code phase trains new experts and new router rows.
- Router phase trains all router rows only.
- Optimizer/RNG state is intentionally not preserved across phase transitions.

Operational trap:

- The script copies/resumes checkpoints through `/tmp/flame-moe`.
- Repeated chunking can create very large temp checkpoint copies.
- Clean `/tmp/flame-moe/<RUN_ID>-*` when runs crash or when KT warns about local disk usage.

### 4.6 Aux/Z-Loss-Off Expert Specialization Experiments

Purpose:

- Standard MoE uses aux/load-balancing loss to encourage even expert use.
- For continual learning, even usage may be counterproductive if we want a small set of important old experts to specialize and be frozen.
- Experiments test whether turning off aux loss, and then aux+z loss, produces more concentrated expert usage.

Implemented scripts:

- `scripts/experiment/a100/run_g2_aux0_wiki_shared_router_mha.sh`
- `scripts/experiment/a100/run_g2_aux0_z0_cumulative80_partial_freeze_code_mha.sh`
- `scripts/experiment/a100/run_g2_shared_router_partial_old_top4_freeze_code_mha.sh`
- `scripts/experiment/a100/run_g2_wiki_router_softmax_importance_1m_mha.sh`
- `scripts/analysis/plot_wiki_router_softmax_importance.py`

Important completed wiki checkpoints:

```text
.local/weights/a100/mha/g2-checkpoints/wiki_aux0/0619-g2-exp3-aux0-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800
.local/weights/a100/mha/g2-checkpoints/wiki_aux0/0620-g2-exp3-aux0-z0-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800
```

Final aux0 wiki run:

```text
wiki acc 0.467981
code acc 0.414915
```

Final aux0+z0 wiki run:

```text
wiki acc 0.468997
code acc 0.413909
```

Router usage analysis correction:

- Earlier router softmax importance used mean softmax over all experts before top-k masking.
- That made usage look too uniform.
- The more accurate CL-freeze criterion is:
  - compute router softmax probabilities;
  - apply top-k selection;
  - selected experts keep their probability;
  - unselected experts are treated as zero;
  - aggregate by layer/expert.
- This measures effective routed probability mass, not just pre-topk preference.

Partial-freeze idea:

- Choose the minimal set of old experts per layer that covers about 80 percent of routed mass.
- Freeze those old important experts.
- Train the remaining old experts plus new Code experts.
- For aux0+z0 cumulative-80 runs, many layers still needed about six experts to cover 80 percent, suggesting specialization was not as extreme as hoped.

### 4.7 Attention-Only Expert Baseline

Purpose:

- Isolate how much attention experts matter.
- Keep FFN dense, but give attention QKVO expert structure controlled by shared router.
- This checks whether shared-router hybrid issues are caused by attention-side expertization itself.

Implemented script:

- `scripts/experiment/a100/run_g2_attention_only_expert_mha.sh`
- Model config: `configs/model/flame-attn-only-shared-router-qkvo-experts.sh`

Default structure:

```text
FFN_HIDDEN_SIZE=5632
WIKI_NUM_EXPERTS=8
SOURCE_NUM_EXPERTS=8
NUM_EXPERTS=16
MOE_ROUTER_TOPK=4
ATTN_FULL_RANK_LORA_TARGETS=qkvo
ATTN_FULL_RANK_LORA_RANK=256
```

Important clarification:

- `dense5632` means FFN dense hidden size is 16-expert total capacity: `16 * 352 = 5632`.
- This is not 24-expert capacity.
- The previous dense4/dense24 experiments are dense baselines, not attention-only expert models.

KT checkpoint paths:

```text
phase1 wiki:
.local/weights/a100/mha/g2-checkpoints/wiki_attn_only/g2-attn-only-dense5632-top4-e8-r256-wiki-qkvo-mha-a100-bf16-mb72-1800

phase2 code:
.local/weights/a100/mha/g2-checkpoints/code/phase1/g2-attn-only-dense5632-top4-e8to16-r256-wiki-to-code-qkvo-mha-a100-bf16-mb72-1800

phase3 router finetune:
.local/weights/a100/mha/g2-checkpoints/code/phase3/g2-attn-only-dense5632-top4-e16-r256-phase3-router-only-retune-wikicode-qkvo-mha-a100-bf16-mb72-1800
```

Result snapshot from discussion:

```text
after wiki:
  wiki acc 0.4763
  code acc 0.4181

after code:
  wiki acc 0.3365
  code acc 0.6616

after router finetune:
  wiki acc 0.3937
  code acc 0.6591
```

Interpretation:

- Attention-only expert can learn Wiki and Code.
- Forgetting after Code exists but is less severe than dense full finetune baselines.
- Router finetune recovers Wiki while keeping Code roughly stable.
- It does not clearly dominate the FFN-only baseline.

### 4.8 Dense Baselines

Purpose:

- Compare expert expansion against dense full finetuning.
- Dense baselines change shared/dense parameters directly, so forgetting can be severe.
- Code and Conversation stages can use old-model KL regularization.

Implemented scripts:

- `scripts/experiment/a100/run_g2_dense_capacity_active_wiki_code_conversation_chain_mha.sh`
- `scripts/experiment/a100/run_g2_dense24_wiki_code_conversation_mha.sh`
- `scripts/experiment/a100/pretrain_wiki_dense_local_bf16.sh`
- `scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh`

Dense variants:

```text
dense4_active:
  FFN_HIDDEN_SIZE=1408 = top4 active expert capacity

dense16:
  FFN_HIDDEN_SIZE=5632 = 16 experts total capacity

dense24_capacity:
  FFN_HIDDEN_SIZE=8448 = 24 experts total capacity
```

Dense4/dense24 result snapshot:

```text
dense24_capacity after wiki:
  wiki acc 0.477104
  code acc 0.413209
  conversation acc 0.299759

dense24_capacity after code:
  wiki acc 0.300444
  code acc 0.722041
  conversation acc 0.270951

dense24_capacity after conversation:
  wiki acc 0.319383
  code acc 0.354001
  conversation acc 0.414496

dense4_active after wiki:
  wiki acc 0.452854
  code acc 0.398876
  conversation acc 0.284033

dense4_active after code:
  wiki acc 0.285353
  code acc 0.703326
  conversation acc 0.256497

dense4_active after conversation:
  wiki acc 0.304323
  code acc 0.333073
  conversation acc 0.398687
```

Interpretation:

- Dense finetuning adapts well to the current task.
- Forgetting is much more severe than in the best expert-expansion/router-retune setups.
- This supports the hypothesis that keeping old parameters mostly frozen matters for CL.

### 4.9 Fixed16 and Expanded Old-Freeze Shared-Unfreeze KL Sweep

Purpose:

- Compare three alternatives for Wiki->Code only:
  1. dense16 full finetune
  2. fixed16 MoE full finetune
  3. expanded 8->16 MoE where old experts/router are frozen but shared trunk is trainable
- Run with and without old-model KL regularization during Code training.

Implemented script:

- `scripts/experiment/a100/run_g2_wiki_code_baseline_kl_sweep_mha.sh`

Variants:

```text
dense16:
  dense FFN hidden 5632

fixed16:
  16 experts from the Wiki stage, full finetune into Code

expanded_oldfreeze_sharedunfreeze:
  Wiki source has 8 experts
  Code expands to 16
  old experts/router frozen
  shared trunk trainable
```

KD modes:

```text
kl1: OLD_MODEL_KL_COEFF=1.0
kl0: OLD_MODEL_KL_COEFF=0.0
```

Important operational caveat:

- The wrapper builds run IDs using the same micro batch size for both Wiki and Code.
- If a completed Wiki source is `mb32` and we want to continue Code with `mb48`, do not call the full wrapper directly.
- Instead, call `continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh` directly with:
  - `STAGE1_WEIGHTS_DIR` pointing to the completed `mb32` wiki source.
  - `TRAIN_WEIGHTS` pointing to the desired new `mb48` code destination.
- Otherwise the wrapper starts a new `mb48` Wiki source run, which is not the intended resumed Code run.

Correct Code run should log:

```text
Loaded old shared-router hybrid checkpoint as logits KD teacher with coefficient 1.0.
Expanded shared-router hybrid checkpoint from 8 to 16 experts.
```

## 5. Hidden-Space PCA/KDE Analysis

Purpose:

- Explain why router tuning can recover Wiki performance.
- Hypothesis: after Code training, Wiki hidden representations drift away from the original Wiki-only latent manifold.
- Router retuning re-aligns routing/hidden distribution toward the Wiki-only latent space.

Implemented analysis script:

- `scripts/analysis/plot_hidden_space_ffn_only.py`

Hidden dump directories used:

```text
FFN-only wiki probe:
.local/analysis/g2matched-ffn-only-hidden-space-wiki-probe/

FFN-only code probe:
.local/analysis/g2matched-ffn-only-hidden-space-code-probe/

Exp1 FFN+attention wiki probe:
.local/analysis/g2-exp1-ffn-attn-hidden-space-wiki-probe/

Exp1 FFN+attention code probe:
.local/analysis/g2-exp1-ffn-attn-hidden-space-code-probe/
```

The plot compares three checkpoints:

```text
wiki_only
code_trained
router_retuned
```

For Wiki probe:

```text
left comparison:  wiki_only vs code_trained
right comparison: wiki_only vs router_retuned
```

For Code probe:

```text
left comparison:  code_trained vs wiki_only
right comparison: code_trained vs router_retuned
```

PCA/KDE overlap metric:

- Hidden states are collected layer by layer.
- For each layer and pair of checkpoints, the high-dimensional hidden vectors are projected into 2D by PCA.
- The 2D point cloud is converted into a smooth density using KDE.
- KDE means each point is treated like a small Gaussian kernel/bump; all bumps are summed into a density map.
- The overlap score is approximately:

```text
overlap = integral min(p(x), q(x)) dx
```

where `p` and `q` are the two KDE density maps in the same 2D PCA plane.

Interpretation:

- Higher overlap means the two hidden distributions occupy more similar latent regions in the PCA view.
- It is a quantitative proxy for latent-space alignment, not a full high-dimensional distribution distance.
- It is preferable to GMM for the current plots because the distributions are irregular and non-Gaussian; GMM would require choosing component counts.

Plot style decisions already made:

- Router-retuned color should be purple/magenta (`#9333ea`) for both Wiki and Code plots.
- Code-trained color should be orange.
- Wiki-only color should be blue.
- Layer 1 is usually omitted in compact layer grids.
- Compact layout: layers 2-5 on the left block, layers 6-9 on the right block.
- Layer labels are vertical and large enough to read.
- KDE overlap values should be printed inside each of the 16 pairwise panels.

## 6. W&B Upload and Relogging Strategy

Direct `wandb sync` has often hung on metadata/artifact uploads.
The preferred approach is to parse `run.log` and event files where needed, then create a fresh online W&B run containing only:

- LM loss
- probe accuracies
- probe perplexities
- optionally learning rate / grad norm if reliable

Important relog scripts:

- `scripts/analysis/relog_phase4_phase5_conversation_to_wandb.py`
- `scripts/analysis/relog_phase1_code_with_source_baseline_to_wandb.py`
- `scripts/analysis/relog_router_finetune_with_source_baseline_to_wandb.py`

Graph continuity rule:

- Phase 1 Wiki: display steps `0 -> 1800`
- Phase 2 Code: source marker at `1800`, Code local `0..1800` maps to display `1800..3600`
- Phase 3 Router finetune: source marker at `3600`, retune local `0..1800` maps to display `3600..5400`
- Conversation phase after Code+retune: often starts at `5400`
- Conversation router retune starts after Conversation, often at `7200`

Avoid duplicate/overwriting issue:

- Use unique W&B run IDs.
- When inserting source baseline point, do not let local step 0 overwrite it with a weaker new-run initial probe.
- For the FFN-only conversation upload, the source Code accuracy at step 3600 should be the actual final source value, e.g. around `0.6628`.
- For the Exp1 setting, the source Code accuracy at step 3600 should be around `0.68375` when that is the true previous final value.

## 7. KT Operational Notes

Always start with:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh
source ~/.config/wandb/env 2>/dev/null || true
```

New session setup:

- Follow `KT_SESSION_RESTART_RUNBOOK.md`.
- Flash-attn is currently not reinstalled for these runs.
- `grouped_gemm` often disappears across KT sessions and should be checked/reinstalled from:

```bash
git+https://github.com/fanshiqing/grouped_gemm@v1.1.4
```

Before starting training:

```bash
if ps -ef | grep -E 'pretrain_gpt.py|torchrun|run_guarded_training|rsync' | grep -v grep; then
  echo "[ERROR] another training/copy process is running"
  exit 1
fi
```

Common disk issue:

- Many scripts copy checkpoints to `/tmp/flame-moe`.
- KT has warned that `/tmp` local disk must not be filled.
- If a run crashes during rsync/checkpoint copy, inspect:

```bash
df -h /tmp /home/work
du -sh /tmp/flame-moe 2>/dev/null || true
du -sh /tmp/flame-moe/* 2>/dev/null | sort -h | tail -40
```

Only delete temp copies when no related training/rsync process is running.

Recommended process kill pattern for a specific run:

```bash
RUN_ID="..."
pkill -TERM -f "$RUN_ID" || true
pkill -TERM -f "pretrain_gpt.py" || true
pkill -TERM -f "torchrun" || true
pkill -TERM -f "run_guarded_training" || true
pkill -TERM -f "rsync.*$RUN_ID" || true
```

## 8. Common Result Extraction Command

For any set of logs, this regex pattern is the useful probe extractor:

```python
probe (wiki_probe|code_probe|conversation_probe) at iteration\s+(\d+) \| local_iteration:\s+(\d+) \| next_token_acc:\s+([0-9.]+) \| ppl:\s+([0-9.E+-]+)
```

Minimal shell/Python pattern:

```bash
python - <<'PY'
import re
from pathlib import Path

logs = [
    ("name", Path("path/to/run.log")),
]
pat = re.compile(
    r"probe (wiki_probe|code_probe|conversation_probe) at iteration\s+(\d+) \| local_iteration:\s+(\d+) \| next_token_acc:\s+([0-9.]+) \| ppl:\s+([0-9.E+-]+)"
)

for name, log in logs:
    print("=" * 90)
    print(name)
    print("log:", log)
    last = {}
    if not log.exists():
        print("[MISSING]")
        continue
    for line in log.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            probe, step, local, acc, ppl = m.groups()
            last[probe] = (int(step), int(local), float(acc), float(ppl))
    for probe in ["wiki_probe", "code_probe", "conversation_probe"]:
        if probe in last:
            step, local, acc, ppl = last[probe]
            print(f"{probe:20s} step={step:5d} local={local:5d} acc={acc:.6f} ppl={ppl:.4f}")
        else:
            print(f"{probe:20s} [NO VALUE]")
PY
```

## 9. Current Interpretation Summary

The strongest high-level story so far:

1. Dense full finetuning learns the current task very well, but forgetting is severe.
2. Expert expansion with old-parameter freezing is much better aligned with continual learning.
3. Router-only finetuning is a surprisingly strong and data-efficient recovery mechanism.
4. FFN-only expansion remains a very strong baseline; attention experts have not clearly beaten it.
5. Attention-only experts are useful for isolating attention contribution, but current results do not prove attention experts solve forgetting.
6. Aux/z loss removal did not create dramatically sparse expert specialization under the current measurement; top-k-masked effective usage is the correct analysis target.
7. Hidden-space PCA/KDE analysis supports the idea that router retuning realigns task latent distributions, especially for Wiki recovery.
8. KL regularization is important to test in shared/dense full-finetuning baselines because those baselines modify old parameters directly.

## 10. Files Most Likely Needed by a New Session

Environment and handoff:

- `KT_SESSION_RESTART_RUNBOOK.md`
- `RESEARCH_CODE_HANDOFF.md`
- `todoexp`
- `experiments.md`

Core G2 launchers:

- `scripts/experiment/a100/run_g2_attention_only_expert_mha.sh`
- `scripts/experiment/a100/run_g2_wiki_code_baseline_kl_sweep_mha.sh`
- `scripts/experiment/a100/run_g2_dense_capacity_active_wiki_code_conversation_chain_mha.sh`
- `scripts/experiment/a100/run_g2_router_finetune_miniset_repeats_mha.sh`
- `scripts/experiment/a100/run_g2_exp2_interleaved_code_router_finetune_mha.sh`
- `scripts/experiment/a100/run_g2_aux0_z0_cumulative80_partial_freeze_code_mha.sh`

Core lower-level stage scripts:

- `scripts/experiment/a100/pretrain_wiki_shared_router_hybrid_local_bf16.sh`
- `scripts/experiment/continual_code_from_wiki_shared_router_hybrid_expand_local_bf16.sh`
- `scripts/experiment/a100/phase3_router_only_retune_shared_router_hybrid_mixed_local_bf16.sh`
- `scripts/experiment/a100/pretrain_wiki_dense_local_bf16.sh`
- `scripts/experiment/continual_code_from_wiki_dense_local_bf16.sh`

Analysis/relog scripts:

- `scripts/analysis/plot_hidden_space_ffn_only.py`
- `scripts/analysis/plot_wiki_router_softmax_importance.py`
- `scripts/analysis/relog_phase4_phase5_conversation_to_wandb.py`
- `scripts/analysis/relog_phase1_code_with_source_baseline_to_wandb.py`
- `scripts/analysis/relog_router_finetune_with_source_baseline_to_wandb.py`

Core Megatron implementation areas:

- `Megatron-LM/megatron/core/transformer/shared_router_hybrid.py`
- `Megatron-LM/megatron/core/models/gpt/shared_router_hybrid_layer_specs.py`
- `Megatron-LM/megatron/core/transformer/moe/continual_learning_utils.py`
- `Megatron-LM/megatron/training/arguments.py`
- `Megatron-LM/megatron/training/training.py`

## 11. Safe Local-to-KT Git Flow

On MacBook, after editing docs/code:

```bash
cd "/Users/yemokoo/miil/1. LLM-CL/LLM-continual-learning"
git status --short
git add KT_SESSION_RESTART_RUNBOOK.md RESEARCH_CODE_HANDOFF.md todoexp experiments.md
git commit -m "Update research and KT handoff docs"
git push origin main
```

If push fails with 403, refresh GitHub HTTPS credential:

```bash
git config --global credential.helper store
printf "protocol=https\nhost=github.com\n\n" | git credential reject

echo -n "GitHub token: "
stty -echo
read GITHUB_TOKEN
stty echo
echo

printf "protocol=https\nhost=github.com\nusername=x-access-token\npassword=%s\n\n" "$GITHUB_TOKEN" | git credential approve
unset GITHUB_TOKEN

git push origin main
```

On KT:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
source scripts/miscellaneous/activate_kt_env.sh

git pull --ff-only origin main

cd Megatron-LM
git pull --ff-only origin qv-lora-bf16-fix
cd ..
```

