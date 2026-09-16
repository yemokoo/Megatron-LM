#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$D/common.sh"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

STAGE="${1:?usage: $0 <code|conversation> <source-checkpoint> [output-checkpoint]}"
SOURCE="${2:?source checkpoint is required}"
case "$STAGE" in
  code)
    NUM_EXPERTS=16; RESUME_FROM_NUM_EXPERTS=8
    DATA_DIRS=("$(dataset_dir_for_task wiki)" "$(dataset_dir_for_task code)")
    LPR_RANGES="0:8,-"
    PROBE_DATASET="$(probe_dir_for_task code)"
    PROBE_NAME="code_probe"
    SECONDARY_PROBE_DATASET="$(probe_dir_for_task wiki)"
    SECONDARY_PROBE_NAME="wiki_probe"
    TERTIARY_PROBE_DATASET=""
    TERTIARY_PROBE_NAME=""
    TERTIARY_PROBE_EVAL_INTERVAL=0
    DEFAULT_ID="g2-ffn-only-code-router-lpr-wiki0to8-equal-token-360"
    ;;
  conversation)
    NUM_EXPERTS=24; RESUME_FROM_NUM_EXPERTS=16
    DATA_DIRS=("$(dataset_dir_for_task wiki)" "$(dataset_dir_for_task code)" "$(dataset_dir_for_task conversation)")
    LPR_RANGES="0:8,8:16,-"
    PROBE_DATASET="$(probe_dir_for_task wiki)"
    PROBE_NAME="wiki_probe"
    SECONDARY_PROBE_DATASET="$(probe_dir_for_task code)"
    SECONDARY_PROBE_NAME="code_probe"
    TERTIARY_PROBE_DATASET="$(probe_dir_for_task conversation)"
    TERTIARY_PROBE_NAME="conversation_probe"
    TERTIARY_PROBE_EVAL_INTERVAL="${TERTIARY_PROBE_EVAL_INTERVAL:-50}"
    DEFAULT_ID="g2-ffn-only-conversation-router-lpr-wiki0to8-code8to16-equal-token-360"
    ;;
  *) echo "ERROR: stage must be code or conversation" >&2; exit 1 ;;
esac

# 라우터 보정용 old 데이터를 고정 서브셋으로 교체한다(ours 1-phase replay 와 동일 풀).
# 현재 태스크(conversation)는 replay 가 아니므로 full pool 유지. 혼합 가중치는 건드리지
# 않으므로 old forward 횟수는 그대로이고 unique 개수만 줄어든다.
if [ -n "${LPR_OLD_SUBSET_ROOT:-}" ]; then
  [ -d "$LPR_OLD_SUBSET_ROOT/wiki/train" ] || { echo "ERROR: no $LPR_OLD_SUBSET_ROOT/wiki/train" >&2; exit 1; }
  [ -d "$LPR_OLD_SUBSET_ROOT/code/train" ] || { echo "ERROR: no $LPR_OLD_SUBSET_ROOT/code/train" >&2; exit 1; }
  DATA_DIRS[0]="$LPR_OLD_SUBSET_ROOT/wiki/train"
  DATA_DIRS[1]="$LPR_OLD_SUBSET_ROOT/code/train"
  echo "[LPR-SUB] old replay pool -> ${DATA_DIRS[0]} , ${DATA_DIRS[1]}"
fi

G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
RUN_ID="${RUN_ID:-$DEFAULT_ID}"
OUTPUT="${3:-$G2_ROOT/lpr/$STAGE/$RUN_ID}"
RETUNE_ITERS="${RETUNE_ITERS:-360}"
LPR_COEFF="${LPR_COEFF:-0.1}"
SOURCE_STEP="$(tr -d '[:space:]' < "$SOURCE/latest_checkpointed_iteration.txt")"
TARGET_STEP="$((SOURCE_STEP + RETUNE_ITERS))"

if [ -f "$OUTPUT/latest_checkpointed_iteration.txt" ] && [ "$(tr -d '[:space:]' < "$OUTPUT/latest_checkpointed_iteration.txt")" = "$TARGET_STEP" ]; then
  echo "[SKIP] completed $STAGE LPR: $OUTPUT"; exit 0
fi
if [ -e "$OUTPUT" ]; then
  [ -f "$OUTPUT/LPR_SOURCE.txt" ] || { echo "ERROR: existing output is not an LPR run: $OUTPUT" >&2; exit 1; }
  grep -Fxq "source=$SOURCE" "$OUTPUT/LPR_SOURCE.txt" || { echo "ERROR: LPR output source mismatch: $OUTPUT" >&2; exit 1; }
  echo "[RESUME] incomplete $STAGE LPR output: $OUTPUT"
else
  mkdir -p "$(dirname "$OUTPUT")" "$OUTPUT"
  rsync -aH --exclude wandb/ "$SOURCE/" "$OUTPUT/"
  {
    echo "source=$SOURCE"
    echo "source_step=$SOURCE_STEP"
    echo "stage=$STAGE"
    echo "retune_iters=$RETUNE_ITERS"
    echo "lpr_coeff=$LPR_COEFF"
    echo "lpr_ranges=$LPR_RANGES"
    echo "data_weight_mode=equal_dataset"
  } > "$OUTPUT/LPR_SOURCE.txt"
fi

PREFIX_COUNTS=""
for dir in "${DATA_DIRS[@]}"; do
  n="$(find "$dir" -maxdepth 1 -name '*.bin' | wc -l)"
  [ "$n" -gt 0 ] || { echo "ERROR: no dataset prefixes in $dir" >&2; exit 1; }
  PREFIX_COUNTS="${PREFIX_COUNTS:+$PREFIX_COUNTS,}$n"
done

TRAIN_DATASET_CONVERSATION=""
[ "$STAGE" = conversation ] && TRAIN_DATASET_CONVERSATION="${DATA_DIRS[2]}"
RUN_ID="$RUN_ID" TRAIN_WEIGHTS="$OUTPUT" SOURCE_STEP="$SOURCE_STEP" \
RETUNE_ITERS="$RETUNE_ITERS" TRAIN_ITERS="$TARGET_STEP" NUM_EXPERTS="$NUM_EXPERTS" \
SOURCE_NUM_EXPERTS="$RESUME_FROM_NUM_EXPERTS" RESUME_FROM_NUM_EXPERTS="$RESUME_FROM_NUM_EXPERTS" \
TRAIN_DATASET_WIKI="${DATA_DIRS[0]}" TRAIN_DATASET_CODE="${DATA_DIRS[1]}" \
TRAIN_DATASET_CONVERSATION="$TRAIN_DATASET_CONVERSATION" MIXED_DATA_WEIGHT_MODE=equal_dataset \
MOE_LPR_LOSS_COEFF="$LPR_COEFF" MOE_LPR_DATASET_PREFIX_COUNTS="$PREFIX_COUNTS" \
MOE_LPR_TASK_EXPERT_RANGES="$LPR_RANGES" MOE_AUX_LOSS_COEFF=0.0 MOE_Z_LOSS_COEFF=0.0 \
PROBE_DATASET="$PROBE_DATASET" PROBE_NAME="$PROBE_NAME" PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-50}" \
SECONDARY_PROBE_DATASET="$SECONDARY_PROBE_DATASET" SECONDARY_PROBE_NAME="$SECONDARY_PROBE_NAME" \
SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-50}" \
TERTIARY_PROBE_DATASET="$TERTIARY_PROBE_DATASET" TERTIARY_PROBE_NAME="$TERTIARY_PROBE_NAME" \
TERTIARY_PROBE_EVAL_INTERVAL="$TERTIARY_PROBE_EVAL_INTERVAL" RUN_INITIAL_PROBE_EVAL=1 \
SAVE_INTERVAL="$RETUNE_ITERS" MASTER_PORT="${MASTER_PORT:-29931}" \
bash "$D/phase3_router_only_retune_moe_mixed_local_bf16.sh"
