#!/usr/bin/env bash
set -euo pipefail

# After the mandatory oracle 1800-step stage, extract paired Conversation
# representations, derive task-specific L2--L9 top-1% intersections, and pack
# only GT-positive token IDs into a standalone replay indexed dataset.

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../.." && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812}"
REFERENCE="${CONVERSATION_KD_INIT_CHECKPOINT:-$PIPELINE_ROOT/checkpoints/01_expansion_kd_init_e16_to_e24_step600}"
CURRENT="${CONVERSATION_ORACLE_CHECKPOINT:-$PIPELINE_ROOT/checkpoints/02_conversation_wikicode_replay_lm_oracle_step1800}"
CONVERSATION_DIR="${CONVERSATION_TRAIN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/conversation/train}"
PAIRED_ROOT="${CONVERSATION_PAIRED_ROOT:-$PIPELINE_ROOT/analysis/conversation_token_hidden_pair}"
GT_ROOT="${CONVERSATION_GT_ROOT:-$PIPELINE_ROOT/analysis/conversation_old_like_gt_l2_l9_top1}"
MINISET_DIR="${CONVERSATION_MINISET_DIR:-$PIPELINE_ROOT/data/conversation_old_like_gt_token_miniset/train}"
PYTHON_BIN="${PYTHON_BIN:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
# The 42 Conversation train shards contain 1,403,196,153 uint16 tokens.
# One raw-corpus pass is 2,740,617 full 512-token samples plus a 249-token
# remainder.  Use the largest prefix aligned to 8 workers x micro-batch 192 so
# every worker starts on an exact batch boundary.  The aligned tail omits
# 201,465 tokens total (0.0144%) and, unlike the
# former 4,147,200-sample stream, does not intentionally traverse ~1.51
# corpus-equivalent epochs.
TOTAL_SAMPLES="${TOTAL_SAMPLES:-2740224}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-192}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PLAN_ONLY="${PLAN_ONLY:-0}"

die() { echo "[ERROR] $*" >&2; exit 1; }
checkpoint_at() {
    local root="$1" step="$2"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] &&
        [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == "$step" ]] &&
        [[ -s "$root/iter_$(printf '%07d' "$step")/common.pt" ]]
}

checkpoint_at "$REFERENCE" 600 || die "reference KD-init step 600 is incomplete: $REFERENCE"
checkpoint_at "$CURRENT" 1800 || die "mandatory Conversation oracle step 1800 is incomplete: $CURRENT"
[[ -x "$PYTHON_BIN" ]] || die "Python missing: $PYTHON_BIN"
shopt -s nullglob
conv_bins=("$CONVERSATION_DIR"/*.bin)
(( ${#conv_bins[@]} == 42 )) || die "expected 42 Conversation train shards, got ${#conv_bins[@]}"
available_kb="$(df -Pk "$(dirname "$PIPELINE_ROOT")" | awk 'NR==2 {print $4}')"
(( available_kb >= 650000000 )) || die "paired extraction needs at least 650 GB free; available_kb=$available_kb"

cat <<EOF
[PLAN] mandatory oracle verified: $CURRENT (step 1800)
[PLAN] paired reference: $REFERENCE (24E expansion KD-init step 600)
[PLAN] paired current:   $CURRENT (24E Conversation+WikiCode replay LM step 1800)
[PLAN] stream: $TOTAL_SAMPLES x 512 Conversation contextual occurrences (one corpus epoch, 8xMB192 aligned)
[PLAN] GT: task-specific full-stream cosine top 1% at each residual layer 2..9, all 8 must pass
[PLAN] miniset: GT-positive token IDs only, occurrence order, target replay exposure 20%
[PLAN] paired output: $PAIRED_ROOT
[PLAN] GT output: $GT_ROOT
[PLAN] miniset output: $MINISET_DIR
EOF
[[ "$PLAN_ONLY" == 1 ]] && exit 0

mkdir -p "$PIPELINE_ROOT/analysis" "$PIPELINE_ROOT/data"
exec 9>"$PIPELINE_ROOT/.gt_build_chain.lock"
flock -n 9 || die "GT build chain is already active"

env \
    REFERENCE_LOAD="$REFERENCE" CURRENT_LOAD="$CURRENT" \
    REFERENCE_REQUIRED_STEP=600 CURRENT_REQUIRED_STEP=1800 \
    TASK_NAME=conversation DATASET_DIR="$CONVERSATION_DIR" \
    DATASET_BLEND_MODE=exhaustive \
    NUM_EXPERTS=24 RESUME_FROM_NUM_EXPERTS=16 REFERENCE_NUM_EXPERTS=24 \
    TOTAL_SAMPLES="$TOTAL_SAMPLES" MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
    GPU_LIST="$GPU_LIST" OUT_ROOT="$PAIRED_ROOT" \
    PAIR_LABEL=conversation_train_post_expansion_kd_init_step600_vs_conversation_wikicode_replay_lm_step1800 \
    WAIT_FOR_WORKERS=1 \
    bash "$D/launch_code_token_hidden_pair_8gpu.sh"

"$PYTHON_BIN" - "$PAIRED_ROOT" "$TOTAL_SAMPLES" <<'PY'
import json, sys
from pathlib import Path
root, expected = Path(sys.argv[1]), int(sys.argv[2])
ranks = sorted(root.glob('rank_[0-9][0-9][0-9]'))
assert len(ranks) == 8, len(ranks)
rows = [json.loads((r/'metadata.json').read_text()) for r in ranks]
assert all(r.get('completed') is True for r in rows)
rows.sort(key=lambda r:r['partition_start_sample'])
cursor=0
for r in rows:
    assert r['partition_start_sample']==cursor
    cursor += r['partition_samples']
assert cursor==expected, (cursor,expected)
print(f'[VALID] paired extraction covers {cursor:,} samples')
PY

last_prefix="${conv_bins[${#conv_bins[@]}-1]%.bin}"
"$PYTHON_BIN" "$D/validate_code_token_hidden_pair.py" \
    --root "$PAIRED_ROOT" \
    --output "$PAIRED_ROOT/validation_report.json" \
    --expected-workers 8 --expected-total-samples "$TOTAL_SAMPLES" \
    --expected-reference-step 600 --expected-current-step 1800 \
    --expected-reference-load "$REFERENCE" --expected-current-load "$CURRENT" \
    --expected-dataset "$last_prefix"

if [[ ! -s "$GT_ROOT/metadata.json" ]]; then
    "$PYTHON_BIN" "$D/build_task_old_like_gt_l2_l9_top1.py" \
        --source-root "$PAIRED_ROOT" --output-root "$GT_ROOT" \
        --task conversation --workers 8 --bins 20000 --top-fraction 0.01
fi

if [[ ! -s "$MINISET_DIR/miniset_metadata.json" ]]; then
    mkdir -p "$MINISET_DIR"
    "$PYTHON_BIN" "$D/build_old_like_gt_token_miniset.py" \
        --gt-root "$GT_ROOT" --paired-root "$PAIRED_ROOT" \
        --output-dir "$MINISET_DIR" --target-train-fraction 0.20
fi

"$PYTHON_BIN" - "$GT_ROOT/metadata.json" "$MINISET_DIR/miniset_metadata.json" <<'PY'
import json, sys
g=json.load(open(sys.argv[1])); m=json.load(open(sys.argv[2]))
assert g['complete'] and m['complete']
assert g['task']=='conversation' and g['layers']==list(range(2,10))
assert m['selected_token_count']==g['selected_count']
assert abs(m['target_train_fraction']-0.2)<1e-12
print('[DONE] selected={:,} ({:.6%}) effective_epochs={:.3f}'.format(
    g['selected_count'],g['selected_fraction'],m['effective_token_epochs']))
PY
