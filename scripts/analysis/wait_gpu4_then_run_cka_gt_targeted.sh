#!/usr/bin/env bash
set -euo pipefail

# Wait without disturbing the current owner of physical GPU 4, then run the
# exact CKA targeted pass.  The called runner performs a second collision and
# port check, closing the race between this wait loop and process launch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU="${GPU:-4}"
POLL_SECONDS="${POLL_SECONDS:-15}"
OUT_ROOT="${OUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$OUT_ROOT/gt_candidates_b99_v2}"
TARGET_OUTPUT="${TARGET_OUTPUT:-$OUT_ROOT/full_census/exact_targeted_v2}"
AUTHORITATIVE_B="${AUTHORITATIVE_B:-$OUT_ROOT/gt_candidates_b99_v2_authoritative_b_v1/authoritative_b_masks.npz}"
THRESHOLD_REVIEW="${THRESHOLD_REVIEW:-$OUT_ROOT/full_census/threshold_review_v1}"
EXACT_REVIEW_OUTPUT="${EXACT_REVIEW_OUTPUT:-$TARGET_OUTPUT/exact_review_v1}"
CODE_PREFIX="${CODE_PREFIX:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/code/train/train_text_document}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
PYTHON_BIN="${PYTHON_BIN:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
WAIT_LOG="${WAIT_LOG:-$OUT_ROOT/logs/wait_gpu4_then_targeted.log}"

[[ "$GPU" == 4 ]] || { echo "[ERROR] this guarded chain is pinned to physical GPU 4" >&2; exit 2; }
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] POLL_SECONDS must be positive" >&2; exit 2; }

mkdir -p "$(dirname "$WAIT_LOG")"
exec > >(tee -a "$WAIT_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] waiting for physical GPU 4 compute contexts to clear"
while true; do
    gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
        | awk -F', ' -v g="$GPU" '$1 == g {print $2}')"
    [[ -n "$gpu_uuid" ]] || { echo "[ERROR] physical GPU $GPU not found"; exit 2; }
    gpu_pids="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
        | awk -F', ' -v u="$gpu_uuid" '$1 == u {print $2}')"
    used_mib="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' -v g="$GPU" '$1 == g {print $2}')"
    if [[ -z "$gpu_pids" && "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= 64 )); then
        echo "[$(date --iso-8601=seconds)] GPU 4 free (used=${used_mib}MiB); launching exact targeted pass"
        break
    fi
    echo "[$(date --iso-8601=seconds)] GPU 4 busy (pids=${gpu_pids//$'\n'/,} used=${used_mib}MiB); waiting"
    sleep "$POLL_SECONDS"
done

cd "$REPO_ROOT"
export GPU OUT_ROOT CANDIDATE_ROOT TARGET_OUTPUT AUTHORITATIVE_B
export WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-192}"
export FORWARD_SUBBATCH_SIZE="${FORWARD_SUBBATCH_SIZE:-128}"
export MASTER_PORT="${MASTER_PORT:-35784}"
bash scripts/analysis/run_cka_gt_targeted_gt_mha.sh

if [[ -e "$EXACT_REVIEW_OUTPUT" ]]; then
    echo "[ERROR] refusing to overwrite existing exact review output: $EXACT_REVIEW_OUTPUT" >&2
    exit 2
fi
echo "[$(date --iso-8601=seconds)] exact targeted pass complete; starting exact CPU review"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/analysis/review_cka_gt_exact_targeted.py \
    --exact-dir "$TARGET_OUTPUT" \
    --threshold-review-dir "$THRESHOLD_REVIEW" \
    --dataset-prefix "$CODE_PREFIX" \
    --output-dir "$EXACT_REVIEW_OUTPUT" \
    --tokenizer-path "$TOKENIZER_PATH"
echo "[$(date --iso-8601=seconds)] exact targeted pass and exact review complete"
