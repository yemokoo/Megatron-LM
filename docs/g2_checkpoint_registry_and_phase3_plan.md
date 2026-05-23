# G2 Checkpoint Registry And Phase 3 Plan

This note records the current G2 checkpoint registry on the KT server and the
naming/copy policy for router-only retuning runs.

KT repo root:

```bash
/home/work/Agent_HJ/30_flame_agent/LLM-continual-learning
```

Central G2 registry root:

```bash
$PWD/.local/weights/a100/mha/g2-checkpoints
```

The registry is a symlink catalog unless a new run is explicitly saved under it.
The original checkpoint directories are not moved.

## Phase 3 Copy Policy

For router-only retuning from an existing model, do not train in-place on the
source checkpoint. Always make a new copy under:

```bash
$G2_ROOT/code/phase3/<run-id>
```

The run id should include:

- Source experiment: `from-exp1` or `from-exp2`
- Operation: `router-only-retune`
- Data: `wikicode`
- Router init state: `no-reinit` or `router-reinit`
- Important batch/schedule suffix, e.g. `mb72-1800`

Recommended examples:

```bash
g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-mb72-1800
g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-mb72-1800
g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-router-reinit-mb72-1800
```

Copy helper:

```bash
cd /home/work/Agent_HJ/30_flame_agent/LLM-continual-learning

G2_ROOT="$PWD/.local/weights/a100/mha/g2-checkpoints"
PHASE3_ROOT="$G2_ROOT/code/phase3"
mkdir -p "$PHASE3_ROOT"

copy_for_phase3() {
  local src="$1"
  local dst="$2"

  if [ ! -f "$src/latest_checkpointed_iteration.txt" ]; then
    echo "[ERROR] source checkpoint tracker not found: $src/latest_checkpointed_iteration.txt" >&2
    return 1
  fi
  if [ -e "$dst" ]; then
    echo "[ERROR] destination already exists: $dst" >&2
    return 1
  fi

  mkdir -p "$dst"
  rsync -aH --info=progress2 \
    --exclude 'wandb/' \
    "$src/" "$dst/"

  {
    echo "source=$src"
    echo "copied_at=$(date -Is)"
    echo "purpose=phase3 router-only retuning copy"
  } > "$dst/PHASE3_SOURCE.txt"
}
```

Example copy for G2 - experiment 1 Phase 3:

```bash
SRC="$G2_ROOT/code/phase1/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-all-experts-router-mha-a100-bf16-mb72-1800"
DST="$PHASE3_ROOT/g2-exp1-phase3-router-only-retune-wikicode-from-all-experts-router-no-reinit-mb72-1800"
copy_for_phase3 "$SRC" "$DST"
```

Equivalent scripted helper:

```bash
scripts/experiment/a100/prepare_g2_phase3_copy.sh exp1 no-reinit
```

Example copy for G2 - experiment 2 Phase 3 after experiment 2 completes:

```bash
SRC="$G2_ROOT/code/phase1/g2-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-all-router-mha-a100-bf16-mb72-1800"
DST="$PHASE3_ROOT/g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-no-reinit-mb72-1800"
copy_for_phase3 "$SRC" "$DST"
```

Equivalent scripted helper:

```bash
scripts/experiment/a100/prepare_g2_phase3_copy.sh exp2 no-reinit
```

For a router-reinit branch, copy from the same experiment 2 source into a
separate destination:

```bash
SRC="$G2_ROOT/code/phase1/g2-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-all-router-mha-a100-bf16-mb72-1800"
DST="$PHASE3_ROOT/g2-exp2-phase3-router-only-retune-wikicode-from-new-experts-all-router-router-reinit-mb72-1800"
copy_for_phase3 "$SRC" "$DST"
```

Equivalent scripted helper:

```bash
scripts/experiment/a100/prepare_g2_phase3_copy.sh exp2 router-reinit
```

## Phase 3 Training Contract

Phase 3 is not KD and not expert training. It is a router-only retuning pass:

- Train data: full wiki train plus full code train as a Megatron weighted blend.
- Trainable parameters: shared-router weights in each routed layer only.
- Frozen parameters: FFN experts, attention LoRA experts, dense trunk, embeddings,
  and output weights.
- Objective: the normal final language-modeling loss only. Router auxiliary loss,
  z-loss, router-memory KL, and teacher-student KD are disabled.
- Checkpoint handling: train only from a copied Phase 3 checkpoint directory, never
  from the original Phase 1 source.

The code path is:

```bash
--shared-router-hybrid-resume-from-num-experts 8
--shared-router-hybrid-train-router-only
--no-load-optim
--no-load-rng
--moe-aux-loss-coeff 0.0
--moe-z-loss-coeff 0.0
--data-path 1.0 <wiki_train_prefix> 1.0 <code_train_prefix>
```

Run one experiment:

```bash
WANDB_MODE=offline scripts/experiment/a100/run_g2_phase3_router_only_retune_mha.sh exp1
WANDB_MODE=offline scripts/experiment/a100/run_g2_phase3_router_only_retune_mha.sh exp2
```

Run both sequentially:

```bash
WANDB_MODE=offline scripts/experiment/a100/run_g2_phase3_router_only_retune_mha.sh all
```

## Experiment 3 Plan

Experiment 3 starts from the same completed G2 wiki checkpoint:

```bash
$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800
```

The code-training stage expands 8 wiki experts to 16 total experts and uses
full-Wiki teacher-student router KD with KL coefficient 10.0. The Code LM
forward uses a custom shared-router mode:

```bash
--shared-router-hybrid-topk-with-all-new-experts
--shared-router-hybrid-all-new-experts-from-num-experts 8
```

This mode selects top-4 over all 16 experts, unions every Code expert
`[8, 16)`, and weights the active expert outputs with the original 16-way
softmax probabilities without renormalization. It is enabled only during
grad-enabled training forwards, so no-grad KD hidden capture, probe, and eval
forwards keep the normal top-k route.

Default Experiment 3 trainability:

- Train all 16 router rows.
- Train only newly added Code FFN and attention LoRA experts.
- Freeze copied Wiki experts and shared dense/backbone parameters.
- Accumulate Code LM loss and full-Wiki router KD loss in the same optimizer
  step.
- Disable MoE aux/z loss by default so the objective is exactly Code LM +
  Wiki router KD, unless explicitly overridden.

Run:

```bash
WANDB_MODE=offline bash scripts/experiment/a100/run_g2_exp3_router_kd_topk_plus_all_code_experts_mha.sh
```

## Current Completed G2 Registry

These entries are from `$G2_ROOT/MANIFEST.tsv` on 2026-05-22.

| Name | Status | Purpose | Trainable / Frozen | Path |
| --- | --- | --- | --- | --- |
| G2 wiki source | complete, step 1800 | First task wiki training for shared-router G2. Source for code continual runs. | Train wiki model normally. 8 experts, top-k 4, FFN expert hidden 352, attention LoRA rank 256, QKVO full-rank LoRA targets. | `$G2_ROOT/wiki/g2-top4-e8-ffn352-r256-wiki-shared-router-qkvo-mha-a100-bf16-mb72-1800` |
| G2 no-KD baseline | complete, step 1800 | Standard wiki to code expansion baseline without router-memory/KD. | Train newly added FFN experts, newly added attention LoRA experts, newly added router rows. Freeze old wiki experts and shared dense trunk. | `$G2_ROOT/code/baseline/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-1800` |
| G2 - experiment 1 Phase 1 | complete, step 1800 | Code task with maximum plasticity before router-only retuning. | Train all FFN experts, all attention LoRA experts, and all router rows. Freeze shared dense trunk, attention main matrices, dense/output trunk. No KD/router-memory regularization. | `$G2_ROOT/code/phase1/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-all-experts-router-mha-a100-bf16-mb72-1800` |
| G2 freeze-wiki-experts baseline | complete, step 1800 | Code task with first-task experts masked/frozen. Closest previous baseline to experiment 2 style. | Freeze wiki experts. Train code-side/new experts. Router behavior depends on the historical script for this run; use metadata before comparing. | `$G2_ROOT/code/phase1/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-code-train-mask-wiki-experts-1800` |
| G2 - experiment 2 Phase 1 | complete, step 1800 | Code task for experiment 2. | Train newly added FFN experts, newly added attention LoRA experts, and all router rows. Freeze existing wiki experts and shared dense trunk. No KD/router-memory regularization. | `$G2_ROOT/code/phase1/g2-exp2-top4-e8to16-ffn352-r256-wiki-to-code-new-experts-all-router-mha-a100-bf16-mb72-1800` |
| Router-memory fixed5 KL0.1 | complete, step 1800 | Router memory regularization baseline using fixed wiki memory sample. | Standard expansion training with router-memory KL coeff 0.1 on fixed 5 percent memory. | `$G2_ROOT/code/router-memory/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-router-memory-fixed5-kl0p1-1800` |
| Router-memory fixed5 KL0.1 early stop | partial/early-stop, step 900 | Early-stop variant of fixed 5 percent router-memory KL baseline. | Same as router-memory fixed5 KL0.1, stopped at checkpoint 900. | `$G2_ROOT/code/router-memory/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-router-memory-fixed5-kl0p1-earlystop-1800` |
| Teacher-student full-wiki KL1 | partial, step 600 | Teacher-student router KD on full wiki memory. | Shared-router teacher-student router KL, KL coeff 1.0. Partial checkpoint only. | `$G2_ROOT/code/kd-teacher-student/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-teacher-student-router-kd-fullwiki-kl1p0-1800` |
| Existing-only router KD KL10 | partial, step 1260 | Teacher-student router KD where KL is applied to existing/wiki experts only. | Existing experts only router KD, KL coeff 10.0. Partial checkpoint. | `$G2_ROOT/code/kd-existingonly/g2-ts-routerkd-existingonly-kl10p0-log20-save60-1800` |
| Existing-only router KD KL50 | complete, step 1800 | Strong existing-only router KD baseline. | Existing experts only router KD, KL coeff 50.0. | `$G2_ROOT/code/kd-existingonly/g2-ts-routerkd-existingonly-kl50p0-log20-save60-1800-rerun-20260519-225301` |
| All-router teacher-student KD KL1 | complete, step 1800 | Teacher-student router KD on all router rows. | All-router KD, KL coeff 1.0. | `$G2_ROOT/code/kd-allrouter/g2-ts-routerkd-allrouter-kl1p0-log20-save60-1800` |
| All-router teacher-student KD KL10 | complete, step 1800 | Strong all-router teacher-student router KD. | All-router KD, KL coeff 10.0. | `$G2_ROOT/code/kd-allrouter/g2-ts-routerkd-allrouter-kl10p0-log20-save60-1800` |
| All-router teacher-student KD KL1 save20/probe20 | complete, step 1800 | High-frequency checkpoint/probe variant for diagnostics and relogging. | All-router KD, KL coeff 1.0, save interval 20, probe interval 20. | `$G2_ROOT/code/kd-allrouter/g2-ts-routerkd-kl1-save20-probe20-1800` |
| Freeze-old-router KD path KL0 | partial, step 1740 | KD-path/control run with old router frozen or KL path disabled. | KL coeff 0.0 control. Partial checkpoint at 1740. | `$G2_ROOT/code/kd-freezeold-router/g2-freezeoldrouter-kdpath-kl0p0-log20-save60-1800` |
| G2matched wiki FFN-MoE | complete, step 1800 | FFN-only MoE wiki source matched to G2 capacity. | Wiki FFN-MoE source model, no shared-router attention LoRA hybrid. | `$G2_ROOT/wiki/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb128-1800` |
| G2matched code attention-freeze | complete, step 1800 | G2matched code baseline with attention frozen. | FFN-MoE code expansion with attention freeze. | `$G2_ROOT/code/g2matched/g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-freeze-mha-a100-bf16-mb96-1800` |
| G2matched code full-rank QKVO | complete, step 1800 | G2matched code baseline with full-rank LoRA attention adaptation. | FFN-MoE code expansion plus full-rank QKVO LoRA rank 1024. | `$G2_ROOT/code/g2matched/g2matched-top4-e8to16-ffn352-wiki-to-code-ffn-moe-attn-fullrank-qkvo-r1024-mha-a100-bf16-mb96-1800` |

## Incomplete Or Failed Top-Level Runs

These entries were cataloged in `$G2_ROOT/INCOMPLETE_OR_FAILED.tsv`.
They are useful for archaeology but should not be treated as final comparison
models without checking their `logs/run.log`.

| Name | Status | Note | Path |
| --- | --- | --- | --- |
| Teacher-student full-wiki KL0.1 | no tracker | Earlier/incomplete teacher-student full-wiki KD attempt. | `$G2_ROOT/_incomplete_or_failed/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-teacher-student-router-kd-fullwiki-kl0p1-1800__fb282ce1` |
| Teacher-student full-wiki KL1 save20/probe20 | no tracker | Earlier/incomplete high-frequency teacher-student run. | `$G2_ROOT/_incomplete_or_failed/g2-top4-e8to16-ffn352-r256-wiki-to-code-shared-router-qkvo-mha-a100-bf16-mb72-teacher-student-router-kd-fullwiki-kl1p0-save20-probe20-1800__158c5ab2` |
| All-router loss calibration mini 165239 | no tracker | Mini calibration/debug run. | `$G2_ROOT/_incomplete_or_failed/g2-ts-routerkd-allrouter-losscalib-kl1-mini20-20260515-165239__26be170b` |
| All-router loss calibration mini 170333 | no tracker | Mini calibration/debug run. | `$G2_ROOT/_incomplete_or_failed/g2-ts-routerkd-allrouter-losscalib-kl1-mini20-20260515-170333__f684d042` |
| G2matched wiki mb72 | no tracker | Earlier G2matched wiki attempt, superseded by mb128 completed source. | `$G2_ROOT/_incomplete_or_failed/g2matched-top4-e8-ffn352-wiki-ffn-moe-mha-a100-bf16-mb72-1800__3aa94613` |

## Comparison Questions

For G2 - experiment 1 and G2 - experiment 2, collect:

- Before Phase 3 router retune: code probe and wiki probe from the Phase 1 checkpoint.
- After Phase 3 router retune: code probe and wiki probe from the router-only retuned checkpoint.
- Router-only retune training loss curve on mixed wiki+code data.
- Whether wiki performance recovers by router retuning alone.
- Whether code performance is preserved or trades off slightly.

## Notes

- `G2 - experiment 1 Phase 1` is complete and can be used as a Phase 3 source.
- `G2 - experiment 2 Phase 1` should be copied for Phase 3 only after its
  checkpoint reaches step 1800.
- Phase 3 should have an explicit marker file, `PHASE3_SOURCE.txt`, recording
  the source checkpoint path and copy time.
