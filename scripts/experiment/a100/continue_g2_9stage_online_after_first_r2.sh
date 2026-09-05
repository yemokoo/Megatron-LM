#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"

: "${OLD_CHAIN_PGID:?set OLD_CHAIN_PGID to the currently running offline chain process group}"

PYTHON_BIN="${PYTHON_BIN:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
WANDB_ENTITY="${WANDB_ENTITY:-yemoyemo010831-korea-university}"
CHAIN_SCRIPT="$D/run_g2_9stage_old_replay_3objective_c10_l2to9_postkd_chain_mha.sh"
CHAIN_LOG="$LOCAL_BASE/logs/g2_9stage_old_replay_3objective_c10_l2to9_postkd_probe3i100_launcher.log"
R2_DIR="$LOCAL_BASE/weights/a100/mha/g2-checkpoints/code/joint_old_data_hidden_kl/post_kd_teacher_c10_fixed_l2to9/r2-code-hiddenkl-c10-fixed-l2to9-post16-mb48-1800-probe3i100"
R2_LOG="$R2_DIR/logs/a_to_b_freeze.log"
R2_WANDB_ID="${HKL_WANDB_RUN_ID:-g2-3objective-hidden-kl-c10-l2to9-probe3i100}"

echo "[WATCH] waiting for completed offline hidden-KL R2"
until grep -q '^\[DONE\] hidden_kl_r2_code$' "$CHAIN_LOG" 2>/dev/null; do
    sleep 5
done

echo "[WATCH] R2 complete; stopping old offline chain process group $OLD_CHAIN_PGID"
kill -TERM -- "-$OLD_CHAIN_PGID" 2>/dev/null || true
for _ in $(seq 1 30); do
    ps -g "$OLD_CHAIN_PGID" >/dev/null 2>&1 || break
    sleep 1
done

[ -f "$R2_DIR/latest_checkpointed_iteration.txt" ] || {
    echo "[ERROR] completed R2 checkpoint marker missing: $R2_DIR" >&2
    exit 1
}
[ "$(tr -d '[:space:]' < "$R2_DIR/latest_checkpointed_iteration.txt")" = "1800" ] || {
    echo "[ERROR] R2 checkpoint is not at iteration 1800" >&2
    exit 1
}

echo "[WANDB] backfilling hidden-KL R2 at local steps 0..1800"
"$PYTHON_BIN" "$R/analysis/replay_full_history_to_wandb.py" \
    --run-dir "$R2_DIR" \
    --continual-log "$R2_LOG" \
    --probe-step-source local \
    --project "$WANDB_PROJECT" \
    --run-name "G2 hidden-KL c10 L2-L9" \
    --run-id "$R2_WANDB_ID" \
    --save-dir "$R2_DIR/wandb" \
    --iteration-axis-only \
    --mode online

echo "[WANDB] relaunching chain online; completed R2 will be skipped"
cd "$R"
export WANDB_MODE=online WANDB_PROJECT WANDB_ENTITY
set -o pipefail
bash "$CHAIN_SCRIPT" 2>&1 | tee -a "$CHAIN_LOG"
