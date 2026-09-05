#!/bin/bash
# Retrain SLoRA-Pre from the port that preserves the released code's behaviour.
#
# The previous run used implementations/SLoRA-repro, whose audited similarity
# compares U_c (out x c) against base_U_r[:, :c] (out x c).  That score grows
# monotonically with c, so the rank search picked c = 64 for all 224 modules
# (verified in the train log), and at c = rank the randomized SVD reconstructs
# delta-W exactly -- denoising became an identity re-factorization and
# SLoRA-Pre collapsed onto SeqLoRA (54.34 / 12.65 versus SeqLoRA 54.42 /
# 11.02).  SLoRA-upstream-port re-SVDs the rank-c reconstruction and compares a
# fixed out x rank matrix, so the score is not monotone and a real argmax
# exists; on this run's own adapters it selects c = 44 and c = 12 for modules
# where the audited version selected 64.
#
# Target: AA 60.84 / forgetting 3.59, the slora_pre_released figures in
# trace/EXPERIMENTS.md.
#
# Batch: the released launcher is 1 process x micro 2 x accum 8 = 16.  This
# uses four GPUs for speed but keeps that effective batch (4 x 2 x 2 = 16).
# Reproduction is the entire point of this run, so the batch is not a free
# knob -- the failed run had already changed it to 64.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_DIR="${ROOT}/implementations/SLoRA-upstream-port"

export SLORA_LLAMA31_PATH="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
if ! grep -q chat_template "${SLORA_LLAMA31_PATH}/tokenizer_config.json" 2>/dev/null; then
  echo "[ERROR] ${SLORA_LLAMA31_PATH} has no chat_template; that is the BASE model" >&2
  exit 2
fi
export SLORA_OUTPUT_ROOT="${SLORA_OUTPUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_pre_upstream}"
export TRACE_PYTHON="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
export SLORA_EVAL_PYTHON="${SLORA_EVAL_PYTHON:-${TRACE_PYTHON}}"

GPUS="${SLORA_GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export WORLD_SIZE="${WORLD_SIZE:-4}"
# 4 x 4 x 1 = 16, the released effective batch, but with one forward per
# optimizer step instead of two accumulated ones.  Same optimization math
# and the same 313 steps/epoch; roughly half the wall clock, because at
# micro-batch 2 the GPUs were idling between accumulation micro-steps.
export MICRO_BATCH="${MICRO_BATCH:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"

# train_trace.sh calls bare `torchrun`, `python` and `python3`; none of them
# are on PATH here, which is how the first attempt died with exit 127 two
# seconds in.  Put the runtime venv first so all three resolve to it.
export PATH="$(dirname "${TRACE_PYTHON}"):${PATH}"
for binary in torchrun python python3; do
  command -v "${binary}" >/dev/null 2>&1 || {
    echo "[ERROR] ${binary} not found after adding $(dirname "${TRACE_PYTHON}") to PATH" >&2
    exit 2
  }
done

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

effective=$(( WORLD_SIZE * MICRO_BATCH * GRAD_ACCUM ))
if [[ "${effective}" -ne 16 && "${SLORA_ALLOW_BATCH_CHANGE:-0}" != "1" ]]; then
  echo "[ERROR] effective batch ${effective} != the released 16; set SLORA_ALLOW_BATCH_CHANGE=1 to override" >&2
  exit 2
fi

echo "[SLORA] port=${PORT_DIR}"
echo "[SLORA] gpus=${GPUS} world=${WORLD_SIZE} micro=${MICRO_BATCH} accum=${GRAD_ACCUM} effective=${effective}"
echo "[SLORA] model=${SLORA_LLAMA31_PATH}"
echo "[SLORA] output=${SLORA_OUTPUT_ROOT}"

if [[ "${SLORA_SKIP_TRAIN:-0}" != "1" ]]; then
  bash "${PORT_DIR}/scripts/repro/train_trace.sh" pre llama31
fi

# The series trains every variant before scoring any of them, so the two
# halves have to be runnable on their own.
if [[ "${SLORA_SKIP_EVAL:-0}" == "1" ]]; then
  echo "[SLORA] train-only phase finished"
  exit 0
fi

# Sparse-15: the same fifteen cells the v3 runs are scored on.  eval_trace.sh
# assigns cells round-robin to EVAL_SHARD_INDEX, so one shard per GPU turns the
# serial 15-cell sweep into four concurrent streams on this lane's devices.
export EVAL_SPARSE_15=1
# sparse-15 lives in rounds 1..8: seven diagonal cells plus all eight of
# round 8.  EVAL_ALL_ROUNDS=0 pins START_ROUND to 8, so the diagonals were
# never run -- the sweep produced 8 cells and still reported success.
export EVAL_ALL_ROUNDS=1
# One 8B model per shard on an 80GB H100 leaves ~60GB free, so the released
# default of 4 was leaving the card idle.  Greedy decoding with left padding
# and per-sample stop markers makes this a throughput knob, not a scoring one.
export SLORA_EVAL_BATCH="${SLORA_EVAL_BATCH:-32}"
# Generation peaks well above the KV-cache estimate: prefill materialises
# batch x prompt x 14336 FFN activations at once, and v3 adds the expert
# LoRA on top.  Py150 at batch 48 OOMed on one cell while an adjacent cell
# survived, so leave headroom and let the allocator grow segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EVAL_CONTINUE_ON_CELL_ERROR="${EVAL_CONTINUE_ON_CELL_ERROR:-0}"

# WORLD_SIZE is a training knob for train_trace.sh, but evaluation runs as a
# plain python process -- and transformers turns device_map="auto" into a
# tensor-parallel plan whenever WORLD_SIZE is set, which torch 2.4 rejects
# outright.  Unset it before the eval shards so the loader stays single-process.
unset WORLD_SIZE
unset RANK LOCAL_RANK MASTER_ADDR MASTER_PORT

IFS=',' read -r -a lane_gpus <<< "${GPUS}"
shard_count="${#lane_gpus[@]}"
eval_log_dir="${SLORA_OUTPUT_ROOT}/eval_shard_logs"
mkdir -p "${eval_log_dir}"
shard_pids=()
for ((shard=0; shard<shard_count; shard++)); do
  # Each shard sees exactly one physical GPU, so eval_trace.sh's own default of
  # device 0 lands on a different card per shard.
  CUDA_VISIBLE_DEVICES="${lane_gpus[shard]}" \
  EVAL_SHARD_COUNT="${shard_count}" EVAL_SHARD_INDEX="${shard}" \
    bash "${PORT_DIR}/scripts/repro/eval_trace.sh" pre llama31 \
    > "${eval_log_dir}/shard${shard}.log" 2>&1 &
  shard_pids+=("$!")
done
eval_status=0
for pid in "${shard_pids[@]}"; do
  wait "${pid}" || eval_status=1
done
if [[ "${eval_status}" -ne 0 ]]; then
  echo "[ERROR] at least one eval shard failed; see ${eval_log_dir}" >&2
  exit 1
fi
# A sweep that skips cells must not report success: the EVAL_ALL_ROUNDS=0
# bug produced 8 of 15 cells twice and exited 0 both times.
cells=$(find "${SLORA_OUTPUT_ROOT}/llama31/pre/evaluation" -name infer.jsonl 2>/dev/null | wc -l)
if [[ "${cells}" -ne 15 ]]; then
  echo "[ERROR] sparse-15 produced ${cells}/15 cells" >&2; exit 3
fi
echo "[SLORA] sparse-15 eval complete across ${shard_count} shards (15/15 cells)"
