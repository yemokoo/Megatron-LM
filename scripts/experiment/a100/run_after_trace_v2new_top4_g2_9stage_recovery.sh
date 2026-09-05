#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"

TRACE_RUN="${TRACE_RUN:-/data2/seonghyeonnoh/LLM-continual-learning-runs/v2_new_top4_kd2x_20260808}"
TRACE_FINAL="$TRACE_RUN/7/lora_moe_meta.json"
POLL_SECONDS="${POLL_SECONDS:-600}"

PERM_ROOT="${PERM_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808}"
SOURCE_ROOT="$PERM_ROOT/00_sources"
COMMON_R1_ROOT="$PERM_ROOT/01_common_kd_init"
RUNS_ROOT="$PERM_ROOT/02_nine_runs"
LOGS_ROOT="$PERM_ROOT/03_logs"
export LOCAL_BASE="${LOCAL_BASE:-$RUNS_ROOT/local}"
export LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-$PERM_ROOT/04_scratch}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
export G2_ROOT="${G2_ROOT:-$LOCAL_WEIGHTS/a100/mha/g2-checkpoints}"
export FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
export TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"
export FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
export PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual-top2-qv-lora}"
export WANDB_ENTITY="${WANDB_ENTITY:-yemoyemo010831-korea-university}"

HF_REPO="${HF_REPO:-YeMoKoo/LLM-continual-learning}"
HF_WIKI_PREFIX="g2_wiki_code_conversation/sources/ffn_only/wiki"
WIKI_OUT="$SOURCE_ROOT/wiki_ffn_only_e8_step1800"
R1_ID="r1-code-expand-wiki-kd-e8to16-mb36-600"
R1_OUT="$COMMON_R1_ROOT/code_e8_to_e16_wiki_kd_step600"
LAUNCH_LOG="$LOGS_ROOT/recovery_chain.log"

checkpoint_at() {
    [ -f "$1/latest_checkpointed_iteration.txt" ] &&
        [ "$(tr -d '[:space:]' < "$1/latest_checkpointed_iteration.txt")" = "$2" ]
}

mkdir -p "$SOURCE_ROOT" "$COMMON_R1_ROOT" "$RUNS_ROOT" "$LOGS_ROOT" "$LOCAL_BASE/logs" "$LOCAL_SSD_ROOT"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

echo "[CHAIN START] $(date '+%F %T %Z')"
echo "[WAIT] TRACE final checkpoint: $TRACE_FINAL"
while [ ! -f "$TRACE_FINAL" ]; do
    echo "[WAIT] $(date '+%F %T %Z') TRACE is not complete; retry in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
done
echo "[TRACE DONE] $(date '+%F %T %Z')"

if checkpoint_at "$WIKI_OUT" 1800; then
    echo "[SKIP] downloaded Wiki source complete: $WIKI_OUT"
else
    echo "[DOWNLOAD] FFN-only Wiki 8-expert source from $HF_REPO -> $WIKI_OUT"
    "$FLAME_ENV/bin/python" - "$HF_REPO" "$HF_WIKI_PREFIX" "$WIKI_OUT" <<'PY'
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

repo, prefix, output = sys.argv[1:]
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
files = [name for name in HfApi().list_repo_files(repo) if name.startswith(prefix + "/")]
if not files:
    raise SystemExit(f"no files found under {repo}/{prefix}")
for name in files:
    source = Path(hf_hub_download(repo, name))
    target = output / name.removeprefix(prefix + "/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
PY
    checkpoint_at "$WIKI_OUT" 1800 || { echo "[ERROR] downloaded Wiki checkpoint incomplete" >&2; exit 1; }
fi

# The HF artifact intentionally contains checkpoint state only, while the
# continual launcher also uses this small local provenance file to recover the
# source step offset. Reconstruct it without changing any checkpoint tensor.
if [ ! -f "$WIKI_OUT/logs/run_metadata.json" ]; then
    mkdir -p "$WIKI_OUT/logs"
    "$FLAME_ENV/bin/python" - "$WIKI_OUT/logs/run_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = {
    "stage": "wiki_a_g2matched_ffn_moe",
    "run_id": "g2-wiki-e8-ffn352-top4-h100-mb72-1800",
    "dataset_name": "wiki_exact",
    "dataset_source": "Wikipedia exact train",
    "train_iters": 1800,
    "micro_batch_size": 72,
    "global_batch_size": 2304,
    "num_layers": 9,
    "hidden_size": 1024,
    "ffn_hidden_size": 5472,
    "num_query_groups": 16,
    "moe_ffn_hidden_size": 352,
    "num_experts": 8,
    "moe_router_topk": 4,
    "precision": "bf16",
    "shared_expert_enabled": False,
    "provenance": "reconstructed for HF checkpoint; tensors unchanged",
}
path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY
fi

if checkpoint_at "$R1_OUT" 600; then
    echo "[SKIP] common R1 complete: $R1_OUT"
else
    echo "[START] common R1 Wiki 8E -> Code-init 16E KD -> $R1_OUT"
    env \
        SOURCE_WEIGHTS_DIR="$WIKI_OUT" SOURCE_REQUIRED_ITERS=1800 \
        TRAIN_ITERS=600 MICRO_BATCH_SIZE=36 GLOBAL_BATCH_SIZE=2304 \
        SAVE_INTERVAL=600 EVAL_INTERVAL=600 RUN_ID="$R1_ID" TRAIN_WEIGHTS="$R1_OUT" \
        LOG_SOURCE_PROBE_BASELINE_BEFORE_EXPAND=0 \
        MASTER_PORT=29782 WANDB_RUN_ID="g2-9stage-recovery-common-r1-20260808" \
        WANDB_EXP_NAME="G2 9stage recovery common R1" \
        bash "$D/run_g2_ffn_only_code_expert_distill_init_mha.sh" logits
    checkpoint_at "$R1_OUT" 600 || { echo "[ERROR] common R1 checkpoint incomplete" >&2; exit 1; }
fi

echo "[START] nine objective stages: hidden-MSE -> hidden softmax-KL -> output-vocab KL"
for objective in hidden_mse hidden_kl vocab_kl; do
    echo "[OBJECTIVE START] $objective $(date '+%F %T %Z')"
    env \
        OBJECTIVE_FILTER="$objective" R1_SOURCE="$R1_OUT" R1_STEP=600 \
        LOCAL_BASE="$LOCAL_BASE" LOCAL_SSD_ROOT="$LOCAL_SSD_ROOT" G2_ROOT="$G2_ROOT" \
        CHAIN_LOG_DIR="$LOGS_ROOT/nine_stages" \
        CODE_PHASE_MB=48 CONV_EXPAND_MB=32 CONV_PHASE_MB=36 GLOBAL_BATCH_SIZE=2304 \
        HKL_WANDB_RUN_ID="g2-recovery-hidden-kl-c10-l2to9-20260808" \
        HMSE_WANDB_RUN_ID="g2-recovery-hidden-mse-c10-l2to9-20260808" \
        VKL_WANDB_RUN_ID="g2-recovery-vocab-kl-c10-20260808" \
        bash "$D/run_g2_9stage_old_replay_3objective_c10_l2to9_postkd_chain_mha.sh"
    echo "[OBJECTIVE DONE] $objective $(date '+%F %T %Z')"
done

echo "[TRACE SPARSE-15 EVAL START] $(date '+%F %T %Z')"
TRACE_ROOT="$R/trace"
env \
    PYTHONNOUSERSITE=1 WANDB_MODE=offline \
    TRACE_DATA_ROOT="$FLAME_DATA_ROOT/trace" \
    SLORA_LLAMA31_PATH=/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B \
    OURS_LORAMOE_OUTPUT_ROOT="$TRACE_RUN" \
    SPARSE15_METHODS=ours_lora_moe_v2_new_top4 \
    SPARSE15_GPUS=0,1,2,3,4,5,6,7 \
    SPARSE15_LOG_ROOT="$TRACE_RUN/eval_queue_logs" \
    SPARSE15_EVAL_BATCH=32 SPARSE15_SCIENCEQA_BATCH=128 \
    SPARSE15_20MINUTEN_BATCH=32 SPARSE15_MEETINGBANK_BATCH=1 \
    SPARSE15_PY150_BATCH=8 SPARSE15_CPU_THREADS=4 \
    "$TRACE_ROOT/.venv-runtime/bin/python" -u "$TRACE_ROOT/scripts/run_ours_sparse15_efficient.py"
[ -s "$TRACE_RUN/sparse15_summary.json" ] || {
    echo "[ERROR] TRACE sparse-15 summary missing" >&2
    exit 1
}
echo "[TRACE SPARSE-15 EVAL DONE] $(date '+%F %T %Z')"

echo "[ALL DONE] $(date '+%F %T %Z')"
