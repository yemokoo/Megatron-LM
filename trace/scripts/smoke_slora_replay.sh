#!/bin/bash
# Prove the SLoRA joint-replay path runs before it takes six hours of GPU.
#
# The replay dataset builder is already covered on CPU.  What has never
# executed is JointReplaySFTTrainer: _prepare_dataset on a second dataset,
# accelerate-preparing a second loader, and a second forward inside
# compute_loss under DeepSpeed zero2.  Three steps at task 2 exercises all of
# it; denoising is skipped because it proves nothing about replay.
#
# Passes only if replay_loss is logged and finite, and the manifest records
# the full 5,000 exposures the v3 plan calls for.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT_DIR="${ROOT}/implementations/SLoRA-upstream-port"
PY="${TRACE_PYTHON:-${ROOT}/.venv-runtime/bin/python}"
GPUS="${SMOKE_GPUS:-4,5,6,7}"
WORLD=$(awk -F, '{print NF}' <<< "${GPUS}")

BASELINE=/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_pre_upstream/llama31/pre
V3_RUN="${SLORA_REPLAY_V3_RUN_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/v3_replay1to1/v3_new_replay1to1_st_top1}"
DATA_ROOT="${TRACE_DATA_ROOT:-${ROOT}/data/trace}"
MODEL="${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct}"
SCRATCH="${SMOKE_DIR:-/data2/seonghyeonnoh/LLM-continual-learning-runs/trace/slora_replay_smoke}"

rm -rf "${SCRATCH}"; mkdir -p "${SCRATCH}"
# task 2 loads task 1's denoised LoRA; borrow the baseline's so the smoke
# exercises the same code path the real run will.
cp -r "${BASELINE}/order1" "${SCRATCH}/order1"

export PATH="$(dirname "${PY}"):${PATH}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
# train_trace.sh does exactly this; torchrun puts the *script* dir on
# sys.path, not the cwd, so `import src.*` needs the port root.
unset PYTHONPATH
export PYTHONPATH="${PORT_DIR}"
export SLORA_SKIP_DENOISE=1
LOG="${SCRATCH}/smoke.log"

echo "[SMOKE] gpus=${GPUS} world=${WORLD} v3_run=${V3_RUN}"
cd "${PORT_DIR}"
set +e
torchrun --nproc_per_node="${WORLD}" --master_port=29971 src/train/cl_train_slora.py \
  --bf16 True --use_peft True --lora_r 64 --lora_alpha 128 \
  --deepspeed "${PORT_DIR}/scripts/zero2.json" \
  --model_name_or_path "${MODEL}" --model llama3 \
  --dataset_name FOMC \
  --train_data_path "${DATA_ROOT}/FOMC/train.json" \
  --output_dir "${SCRATCH}/order2" \
  --max_steps 3 --num_train_epochs 1 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --eval_strategy no --save_strategy no \
  --learning_rate 2e-4 --weight_decay 0 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine --logging_steps 1 \
  --gradient_checkpointing True --seed 2025 --task_id 2 --mode max \
  --replay_v3_run_dir "${V3_RUN}" --replay_data_root "${DATA_ROOT}" \
  --replay_loss_coeff 1.0 > "${LOG}" 2>&1
status=$?
set -e

fail() { echo "[SMOKE] FAIL: $1"; echo "--- last 40 lines ---"; tail -40 "${LOG}"; exit 1; }

[[ "${status}" -eq 0 ]] || fail "torchrun exited ${status}"
grep -q "replay_loss" "${LOG}" || fail "no replay_loss in the training log -- the replay forward never ran"
grep -qE "replay_loss': (nan|inf|-inf)" "${LOG}" && fail "replay_loss is not finite"
[[ -f "${SCRATCH}/order2/replay_manifest.json" ]] || fail "no replay_manifest.json"
"${PY}" - "${SCRATCH}/order2/replay_manifest.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
total = sum(t["exposures"] for t in m["tasks"])
assert m["round"] == 1, m["round"]
assert total == m["planned_exposure_samples"] == 5000, (total, m)
print(f"[SMOKE] manifest OK: round {m['round']}, {total} exposures, {m['distribution']}")
PYEOF

echo "[SMOKE] replay losses seen:"
grep -o "'replay_loss': [0-9.]*" "${LOG}" | head -5
echo "[SMOKE] PASS"
