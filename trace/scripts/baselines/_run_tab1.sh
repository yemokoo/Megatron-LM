#!/usr/bin/env bash
# Runner for the Table-1 continual-learning baselines.
#
#   _run_tab1.sh <validate|train> <llama31|qwen25_7b> <method>
#
# methods: seq_lora ewc olora mtl lifelong_moe lifelong_moe_attn
#          moe_lpr moe_lpr_attn dymoe incmoelora
#
# Every method runs under the same contract as Ours (global batch 64, r64 /
# alpha128 / dropout 0.05, lr 2e-4, cosine + 3% warmup, seed 2025, the
# slora_chat_full template and the Instruct token cache), so a Table-1
# difference is a method difference.
#
# GRAD_ACCUM is DERIVED from the requested global batch rather than defaulted.
# The SLoRA scripts hardcode micro/accum per repo, so a run that only changed
# CUDA_VISIBLE_DEVICES silently trained at global batch 16 instead of 64; that
# failure mode is impossible here.
set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OURS_ROOT="${TAB1_IMPL_ROOT:-${ROOT}/implementations/llmcl_benchmark}"
ACTION="${1:?usage: _run_tab1.sh <validate|train> <model> <method>}"
MODEL="${2:?usage: _run_tab1.sh <validate|train> <model> <method>}"
METHOD="${3:?usage: _run_tab1.sh <validate|train> <model> <method>}"

case "${METHOD}" in
  seq_lora|ewc|olora|mtl)          IMPL_METHOD="${METHOD}"; MOE_SCOPE="" ;;
  lifelong_moe)                    IMPL_METHOD="lifelong_moe"; MOE_SCOPE="ffn" ;;
  lifelong_moe_attn)               IMPL_METHOD="lifelong_moe"; MOE_SCOPE="ffn_attn" ;;
  moe_lpr)                         IMPL_METHOD="moe_lpr"; MOE_SCOPE="ffn" ;;
  moe_lpr_attn)                    IMPL_METHOD="moe_lpr"; MOE_SCOPE="ffn_attn" ;;
  dymoe|incmoelora)                IMPL_METHOD="dymoe"; MOE_SCOPE="" ;;
  *) echo "[ERROR] unknown tab1 method: ${METHOD}" >&2; exit 2 ;;
esac

PYTHON_BIN="${TAB1_PYTHON:-${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}}"
DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
TRACE_RUN_ROOT="${TRACE_RUN_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace}"
RUN_DIR="${TAB1_OUTPUT_ROOT:-${TRACE_RUN_ROOT}/tab1/${MODEL}/${METHOD}}"
GPUS="${TAB1_GPUS:-0,1,2,3}"
PORT="${TAB1_PORT:-29671}"
EPOCHS="${TAB1_EPOCHS:-5,3,7,5,3,5,5,7}"
# Smoke knobs. TAB1_TASKS trims the sequence; TAB1_MAX_STEPS caps micro-batches
# per phase. Keep the cap at or above the accumulation window, otherwise the
# run finishes without ever taking an optimizer step and proves nothing.
TASKS="${TAB1_TASKS:-all}"
MAX_STEPS="${TAB1_MAX_STEPS:-0}"
GLOBAL_BATCH="${TAB1_GLOBAL_BATCH:-64}"
MICRO_BATCH="${TAB1_MICRO_BATCH:-8}"
EVAL_BATCH="${TAB1_EVAL_BATCH:-4}"
SEED="${TAB1_SEED:-2025}"
LR="${TAB1_LR:-2e-4}"
RANK="${TAB1_RANK:-64}"
ALPHA="${TAB1_ALPHA:-128}"
DROPOUT="${TAB1_DROPOUT:-0.05}"
# O-LoRA adapts q_proj/v_proj only -- PEFT's Llama default, which the official
# repo inherits by passing no target_modules and which TRACE's port mirrors.
# Every other row uses the seven-projection SLoRA/Seq-LoRA contract.
case "${IMPL_METHOD}" in
  olora) LORA_TARGETS="${TAB1_LORA_TARGETS:-qv}" ;;
  *)     LORA_TARGETS="${TAB1_LORA_TARGETS:-all7}" ;;
esac
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
WORLD_SIZE="${#GPU_ARRAY[@]}"
DENOM=$((MICRO_BATCH * WORLD_SIZE))
if (( GLOBAL_BATCH % DENOM != 0 )); then
  echo "[ERROR] global batch ${GLOBAL_BATCH} is not divisible by micro ${MICRO_BATCH} x world ${WORLD_SIZE}" >&2
  exit 2
fi
GRAD_ACCUM="${TAB1_GRAD_ACCUM:-$((GLOBAL_BATCH / DENOM))}"
if (( MICRO_BATCH * WORLD_SIZE * GRAD_ACCUM != GLOBAL_BATCH )); then
  echo "[ERROR] micro ${MICRO_BATCH} x world ${WORLD_SIZE} x accum ${GRAD_ACCUM} != ${GLOBAL_BATCH}" >&2
  exit 2
fi

case "${MODEL}" in
  llama31)
    MODEL_PATH="${SLORA_LLAMA31_PATH:?export SLORA_LLAMA31_PATH (Llama-3.1-8B-Instruct); there is no default}"
    PREFLIGHT_KEY="llama31_8b"
    TOKEN_CACHE="${TAB1_LLAMA31_TOKEN_CACHE:-/data2/seonghyeonnoh/LLM-continual-learning-caches/llama31_8b_instruct/slora_chat_full_len1024}"
    ;;
  qwen25_7b)
    MODEL_PATH="${SLORA_QWEN25_7B_PATH:-${ROOT}/models/Qwen2.5-7B-Instruct}"
    PREFLIGHT_KEY="qwen25_7b_instruct"
    TOKEN_CACHE="${TAB1_QWEN25_TOKEN_CACHE:-}"
    ;;
  *) echo "[ERROR] unknown model: ${MODEL}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_COMMAND=(
  "${PYTHON_BIN}" -m torch.distributed.run
  "--nproc_per_node=${WORLD_SIZE}"
  "--master_port=${PORT}"
  training/main_tab1.py
  --method "${IMPL_METHOD}"
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_ROOT}"
  --data_output_path "${RUN_DIR}/data_cache"
  --output_dir "${RUN_DIR}"
  --dataset_name "${TASKS}"
  --num_train_epochs "${EPOCHS}"
  --per_device_train_batch_size "${MICRO_BATCH}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --per_device_eval_batch_size "${EVAL_BATCH}"
  --max_prompt_len 1024
  --max_ans_len 512
  --max_train_len 1024
  --train_format slora_chat_full
  --learning_rate "${LR}"
  --weight_decay 0
  --adam_beta1 0.9
  --adam_beta2 0.999
  --adam_epsilon 1e-8
  --lr_scheduler_type cosine
  --num_warmup_steps 0
  --warmup_ratio 0.03
  --gradient_checkpointing
  --seed "${SEED}"
  --lora_targets "${LORA_TARGETS}"
  --lora_rank "${RANK}"
  --lora_alpha "${ALPHA}"
  --lora_dropout "${DROPOUT}"
  --max_train_steps_per_task "${MAX_STEPS}"
)

# Continue an interrupted cell instead of repeating its finished tasks.
if [[ -n "${TAB1_RESUME_FROM:-}" ]]; then
  TRAIN_COMMAND+=(--resume_from "${TAB1_RESUME_FROM}")
fi

if [[ -n "${TOKEN_CACHE}" ]]; then
  TRAIN_COMMAND+=(--tokenized_train_cache_dir "${TOKEN_CACHE}")
fi

case "${IMPL_METHOD}" in
  ewc)
    # Our EWC, not TRACE's: the Fisher is estimated after each task converges,
    # with the penalty off, normalised by samples.  See model/tab1_lora.py.
    TRAIN_COMMAND+=(
      --ewc_lambda "${TAB1_EWC_LAMBDA:-400}"
      --ewc_mode "${TAB1_EWC_MODE:-online}"
      --ewc_fisher_samples "${TAB1_EWC_FISHER_SAMPLES:-1000}"
      --ewc_fisher_batch_size "${TAB1_EWC_FISHER_BATCH_SIZE:-1}")
    ;;
  olora)
    # Official long.sh uses l1=0.5 at every one of the first eight positions
    # (task 1 has nothing to be orthogonal to) and l2=0 outside three
    # hand-tuned slots; l1=5 only appears from task 10.
    TRAIN_COMMAND+=(
      --olora_lambda_orthogonal "${TAB1_OLORA_L1:-0.5}"
      --olora_lambda_l2 "${TAB1_OLORA_L2:-0.0}"
      --olora_merge_at_end "${TAB1_OLORA_MERGE:-1}")
    ;;
  mtl)
    # 5 epochs x 40,000 joint records = the 200,000 samples the sequential
    # schedule consumes, so the ceiling is matched on compute too.
    TRAIN_COMMAND+=(--mtl_epochs "${TAB1_MTL_EPOCHS:-5}")
    ;;
  dymoe)
    # LLaVA-DyMoE (zhaoc5/DyMoE scripts/Train/*.sh): per task 16 rank-4
    # experts (TAB1_RANK 64 split 16 ways) on all seven projections, top-16
    # cosine routing at T=0.01, TAG tau=0.2, RSR exc=spe=1e-3 at T=0.1 ramped
    # in over the second half of each task. incmoelora = the paper's
    # IncMoELoRA baseline: the same layer with TAG and RSR switched off.
    if [[ "${METHOD}" == "incmoelora" ]]; then
      DYMOE_TAG_DEFAULT=0; DYMOE_EXC_DEFAULT=0; DYMOE_SPE_DEFAULT=0
    else
      DYMOE_TAG_DEFAULT=1; DYMOE_EXC_DEFAULT=1e-3; DYMOE_SPE_DEFAULT=1e-3
    fi
    TRAIN_COMMAND+=(
      --dymoe_variant "${METHOD}"
      --dymoe_experts_per_task "${TAB1_DYMOE_EXPERTS:-16}"
      --dymoe_top_k "${TAB1_DYMOE_TOP_K:-16}"
      --dymoe_router_temperature "${TAB1_DYMOE_ROUTER_T:-0.01}"
      --dymoe_cosine_scale "${TAB1_DYMOE_COSINE_SCALE:-1.0}"
      --dymoe_tag "${TAB1_DYMOE_TAG:-${DYMOE_TAG_DEFAULT}}"
      --dymoe_conflict_ratio "${TAB1_DYMOE_TAU:-0.2}"
      --dymoe_exc_coeff "${TAB1_DYMOE_EXC:-${DYMOE_EXC_DEFAULT}}"
      --dymoe_spe_coeff "${TAB1_DYMOE_SPE:-${DYMOE_SPE_DEFAULT}}"
      --dymoe_rsr_temperature "${TAB1_DYMOE_RSR_T:-0.1}"
      --dymoe_rsr_start_fraction "${TAB1_DYMOE_RSR_START:-0.5}")
    ;;
esac

if [[ -n "${MOE_SCOPE}" ]]; then
  TRAIN_COMMAND+=(
    --moe_scope "${MOE_SCOPE}"
    --lora_moe_rank "${TAB1_MOE_RANK:-${RANK}}"
    --lora_moe_alpha "${TAB1_MOE_ALPHA:-${ALPHA}}"
    --lora_moe_dropout "${TAB1_MOE_DROPOUT:-${DROPOUT}}"
    --experts_per_task "${TAB1_EXPERTS_PER_TASK:-1}"
    --top_k "${TAB1_TOP_K:-1}"
    --routing_weight_mode "${TAB1_ROUTING_WEIGHT_MODE:-straight_through_topk}")
  case "${IMPL_METHOD}" in
    lifelong_moe)
      # Lifelong-MoE keeps a shared path that is retrained on every task; that
      # channel is where its forgetting comes from (wiki: FM .1665 -> .2139
      # across the capacity sweep while every other expansion method is flat).
      # Under ffn_attn all seven projections already host routed experts, so
      # there is nowhere for that path to live and the run would silently be a
      # different method.  It therefore needs an explicit opt-in and is an
      # ablation row, not a Lifelong-MoE row.
      if [[ "${MOE_SCOPE}" == "ffn_attn" ]]; then
        SHARED_DEFAULT="none"
        if [[ "${TAB1_LIFELONG_ALLOW_NO_SHARED:-0}" != "1" ]]; then
          echo "[ERROR] lifelong_moe_attn has no free projection for the shared path." >&2
          echo "        Report Lifelong-MoE at --moe_scope ffn, or set" >&2
          echo "        TAB1_LIFELONG_ALLOW_NO_SHARED=1 and label the row an ablation." >&2
          exit 2
        fi
      else
        SHARED_DEFAULT="attn"
      fi
      TRAIN_COMMAND+=(
        --lifelong_shared_targets "${TAB1_LIFELONG_SHARED:-${SHARED_DEFAULT}}"
        --lifelong_train_shared "${TAB1_LIFELONG_TRAIN_SHARED:-1}"
        --lifelong_kd_coeff "${TAB1_LIFELONG_KD:-1.0}"
        --lifelong_kd_temperature "${TAB1_LIFELONG_KD_T:-1.0}"
        --lifelong_l2_coeff "${TAB1_LIFELONG_L2:-0.0}"
        --lifelong_allow_no_shared_path "${TAB1_LIFELONG_ALLOW_NO_SHARED:-0}"
        --moe_aux_loss_coeff "${TAB1_MOE_AUX:-0.01}"
        --moe_z_loss_coeff "${TAB1_MOE_Z:-0.001}")
      ;;
    moe_lpr)
      # Review-phase aux/z are forced to zero inside the trainer: a
      # load-balancing pressure works directly against teaching the router to
      # prefer one expert.  These coefficients apply to phase 1 only.
      TRAIN_COMMAND+=(
        --lpr_gamma "${TAB1_LPR_GAMMA:-0.1}"
        --lpr_review_fraction "${TAB1_LPR_REVIEW_FRACTION:-0.2}"
        --replay_subset_ratio "${TAB1_REPLAY_SUBSET_RATIO:-0.1}"
        --replay_subset_seed "${TAB1_REPLAY_SUBSET_SEED:--1}"
        --router_replay_exposure_samples "${TAB1_ROUTER_REPLAY_SAMPLES:-1000}"
        --moe_aux_loss_coeff "${TAB1_MOE_AUX:-0.01}"
        --moe_z_loss_coeff "${TAB1_MOE_Z:-0.001}")
      if [[ -n "${TAB1_REPLAY_MANIFEST:-}" ]]; then
        TRAIN_COMMAND+=(--replay_manifest_path "${TAB1_REPLAY_MANIFEST}")
      fi
      ;;
  esac
fi

print_command() { printf '%q ' "$@"; printf '\n'; }

preflight() {
  [[ -f "${OURS_ROOT}/training/main_tab1.py" ]] || {
    echo "[ERROR] tab1 entry point missing under ${OURS_ROOT}" >&2; exit 3; }
  [[ -x "${PYTHON_BIN}" ]] || {
    echo "[ERROR] training venv missing: ${PYTHON_BIN}" >&2; exit 4; }
  if [[ -n "${TOKEN_CACHE}" ]]; then
    [[ -f "${TOKEN_CACHE}/manifest.json" ]] || {
      echo "[ERROR] token cache missing: ${TOKEN_CACHE}" >&2; exit 5; }
  fi
  python3 "${ROOT}/scripts/preflight.py" --mode full --models "${PREFLIGHT_KEY}"
}

write_manifest() {
  mkdir -p "${RUN_DIR}"
  {
    echo "classification=tab1-${METHOD}"
    echo "method=${METHOD}"
    echo "impl_method=${IMPL_METHOD}"
    echo "moe_scope=${MOE_SCOPE:-none}"
    echo "model=${MODEL}"
    echo "model_path=${MODEL_PATH}"
    echo "data_root=${DATA_ROOT}"
    echo "tasks=${TASKS}"
    echo "epochs=${EPOCHS}"
    echo "max_train_steps_per_task=${MAX_STEPS}"
    echo "gpus=${GPUS}"
    echo "world_size=${WORLD_SIZE}"
    echo "micro_batch=${MICRO_BATCH}"
    echo "gradient_accumulation=${GRAD_ACCUM}"
    echo "effective_global_batch=${GLOBAL_BATCH}"
    echo "learning_rate=${LR}"
    echo "lora_rank=${RANK}"
    echo "lora_alpha=${ALPHA}"
    echo "lora_dropout=${DROPOUT}"
    echo "lora_targets=${LORA_TARGETS}"
    echo "seed=${SEED}"
    echo "token_cache=${TOKEN_CACHE}"
  } > "${RUN_DIR}/run.env"
  print_command "${TRAIN_COMMAND[@]}" > "${RUN_DIR}/train.command.txt"
}

cd "${OURS_ROOT}"
case "${ACTION}" in
  validate)
    echo "[VALIDATE] tab1 ${METHOD} / ${MODEL}"
    echo "  world=${WORLD_SIZE} micro=${MICRO_BATCH} accum=${GRAD_ACCUM} global=${GLOBAL_BATCH}"
    print_command "${TRAIN_COMMAND[@]}"
    ;;
  train)
    preflight
    write_manifest
    if [[ "${DRY_RUN}" == "1" ]]; then
      print_command "${TRAIN_COMMAND[@]}"
      exit 0
    fi
    echo "[TRAIN] tab1 ${METHOD} / ${MODEL} -> ${RUN_DIR}"
    exec "${TRAIN_COMMAND[@]}"
    ;;
  *)
    echo "[ERROR] action must be validate or train" >&2
    exit 2
    ;;
esac
