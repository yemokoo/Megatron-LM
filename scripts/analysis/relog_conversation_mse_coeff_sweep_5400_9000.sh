#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning}"
ROOT="${MSE_COEFF_SWEEP_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/conversation_old_like_gt_pipeline_20260812/mse_coeff_sweep_20pct_contextual_occurrence_exact_axis_v1_all_steps3600}"
PYTHON_BIN="${PYTHON_BIN:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
ENTITY="${WANDB_ENTITY:-yemoyemo010831-korea-university}"
STATUS="$ROOT/logs/relog_5400_9000_status.tsv"

mkdir -p "$ROOT/logs"
exec 9>"$ROOT/.relog_5400_9000.lock"
flock -n 9 || { echo "[ERROR] relog watcher already active" >&2; exit 1; }

checkpoint_at_1800() {
    local root="$1"
    [[ -s "$root/latest_checkpointed_iteration.txt" ]] &&
        [[ "$(tr -d '[:space:]' < "$root/latest_checkpointed_iteration.txt")" == 1800 ]] &&
        [[ -s "$root/iter_0001800/common.pt" ]] &&
        [[ -s "$root/iter_0001800/.metadata" ]]
}

relog_one() {
    local ordinal="$1" tag="$2" coeff="$3"
    local run_dir="$ROOT/checkpoints/${ordinal}_mse_${tag}"
    local log="$run_dir/logs/code_to_conversation_freeze.log"
    local run_id="conversation_oldlike_hidden_mse_${tag}_l2to9_exactaxis_step5400to9000"
    local run_name="Conversation old-like hidden MSE c${coeff} L2-L9 | step 5400-9000"
    local marker="$ROOT/logs/${run_id}.uploaded"

    if [[ -s "$marker" ]]; then
        echo "$(date -Is) SKIP uploaded $run_id" | tee -a "$STATUS"
        return
    fi
    echo "$(date -Is) WAIT checkpoint=$run_dir" | tee -a "$STATUS"
    until checkpoint_at_1800 "$run_dir"; do sleep 30; done
    [[ -s "$log" ]] || { echo "[ERROR] missing log: $log" >&2; exit 1; }
    echo "$(date -Is) START $run_id" | tee -a "$STATUS"
    (
        cd "$REPO"
        env WANDB_PROJECT="$PROJECT" WANDB_ENTITY="$ENTITY" WANDB_MODE=online \
            "$PYTHON_BIN" scripts/analysis/relog_continual_local_steps_to_wandb.py \
            --run-dir "$run_dir" \
            --log "$log" \
            --run-id "$run_id" \
            --run-name "$run_name" \
            --step-offset 5400 \
            --step-scale 2 \
            --train-iters 1800
    ) >> "$ROOT/logs/${run_id}.relog.log" 2>&1
    printf '%s\n' "https://wandb.ai/$ENTITY/$PROJECT/runs/$run_id" > "$marker.inprogress"
    mv "$marker.inprogress" "$marker"
    echo "$(date -Is) DONE $run_id" | tee -a "$STATUS"
}

relog_one 01 c0p3 0.3
relog_one 02 c0p4 0.4
relog_one 03 c1p0 1.0
echo "$(date -Is) COMPLETE all coefficient relogs" | tee -a "$STATUS"
