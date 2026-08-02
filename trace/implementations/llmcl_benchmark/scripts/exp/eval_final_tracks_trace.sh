#!/bin/bash
# Final TRACE evaluation for both growing-model tracks.
#
# For each model, two GPUs independently load one full model copy and evaluate
# four disjoint tasks each. Qwen finishes and unloads first; OLMoE then runs in
# the same 2x4 layout. RUN=1 is required to launch evaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."

RUN=${RUN:-0}
PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
GPUS=${GPUS:-0,1}
TASKS=${TASKS:-C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}
DATA=${DATA:-data/LLM-CL-Benchmark_5000}
TEMPERATURE=${TEMPERATURE:-0.0}
QWEN_BATCH=${QWEN_BATCH:-32}
OLMOE_BATCH=${OLMOE_BATCH:-8}
WITH_SARI=${WITH_SARI:-0}

QWEN_BASE=${QWEN_BASE:-/home/work/Agent_HJ/00_models/Qwen3-8B}
QWEN_CKPT=${QWEN_CKPT:-/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/output/track1_Qwen3-8B_full/7}
QWEN_OUT=${QWEN_OUT:-/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/eval_out/track1_Qwen3-8B_full_round7_trace_taskcaps_2gpu}

OLMOE_BASE=${OLMOE_BASE:-/home/work/Agent_HJ/00_models/OLMoE-1B-7B-0125}
OLMOE_CKPT=${OLMOE_CKPT:-/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/output/track2_OLMoE_ept1_force_upper_5k_seed1234/7}
OLMOE_OUT=${OLMOE_OUT:-/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/eval_out/track2_OLMoE_ept1_force_upper_5k_seed1234_round7_trace_taskcaps_2gpu}

COMBINED_OUT=${COMBINED_OUT:-/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/eval_out/final_tracks_round7_trace_taskcaps_2gpu_summary.json}

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [ -f "$1" ] || die "missing file: $1"; }
require_dir() { [ -d "$1" ] || die "missing directory: $1"; }

[ -x "$PYTHON_BIN" ] || die "Python is not executable: $PYTHON_BIN"
IFS=',' read -r -a gpu_array <<<"$GPUS"
[ "${#gpu_array[@]}" -eq 2 ] || die "this runner requires exactly two GPU IDs"
require_dir "$QWEN_BASE"
require_dir "$OLMOE_BASE"
require_file "$QWEN_CKPT/pytorch_model.bin"
require_file "$QWEN_CKPT/lora_moe_meta.json"
require_file "$OLMOE_CKPT/pytorch_model.bin"
require_file "$OLMOE_CKPT/moe_ffn_meta.json"

IFS=',' read -r -a task_array <<<"$TASKS"
[ "${#task_array[@]}" -eq 8 ] || die "the 2x4 final evaluation requires exactly 8 tasks"
for task in "${task_array[@]}"; do
  require_file "$DATA/$task/test.json"
done

declare -a gpu_tasks
for index in "${!task_array[@]}"; do
  slot=$((index % 2))
  if [ -n "${gpu_tasks[$slot]:-}" ]; then
    gpu_tasks[$slot]="${gpu_tasks[$slot]},${task_array[$index]}"
  else
    gpu_tasks[$slot]="${task_array[$index]}"
  fi
done

"$PYTHON_BIN" - <<'PY'
for name in ("torch", "transformers", "rouge", "nltk"):
    __import__(name)
from model.Ours_LoRA_MoE import load_lora_moe_checkpoint
from model.Ours_MoE_FFN import load_moe_ffn_checkpoint
from vllm_eval import load_task, normalize_predictions, score, TASK_MAX_NEW_TOKENS
assert TASK_MAX_NEW_TOKENS["C-STANCE"] == 4
print("Evaluator imports and task generation limits: OK")
PY

sari_args=()
if [ "$WITH_SARI" = "1" ]; then sari_args+=(--with_sari); fi

echo "================ TRACE FINAL EVAL SETUP ================"
echo "execution   : Qwen 2x4 parallel -> unload -> OLMoE 2x4 parallel"
echo "GPU ${gpu_array[0]}       : ${gpu_tasks[0]}"
echo "GPU ${gpu_array[1]}       : ${gpu_tasks[1]}"
echo "temperature : $TEMPERATURE"
echo "task caps   : C-STANCE=4 FOMC=4 MeetingBank=512 Py150=160"
echo "              ScienceQA=512 NumGLUE-cm=16 NumGLUE-ds=16 20Minuten=256"
echo "Qwen        : batch $QWEN_BATCH, checkpoint $QWEN_CKPT"
echo "Qwen output : $QWEN_OUT"
echo "OLMoE       : batch $OLMOE_BATCH, checkpoint $OLMOE_CKPT"
echo "OLMoE output: $OLMOE_OUT"

if [ "$RUN" != "1" ]; then
  echo "SETUP CHECK PASSED; evaluation was not started."
  echo "Launch with: GPUS=0,1 RUN=1 bash scripts/exp/eval_final_tracks_trace.sh"
  exit 0
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$QWEN_OUT" "$OLMOE_OUT" "$(dirname "$COMBINED_OUT")"

merge_phase() {
  local out=$1
  "$PYTHON_BIN" - "$out" "${task_array[@]}" <<'PY'
import json, os, sys
out, tasks = sys.argv[1], sys.argv[2:]
summary = {}
for task in tasks:
    path = os.path.join(out, f"results-{task}.json")
    if not os.path.isfile(path):
        raise SystemExit(f"missing evaluation result: {path}")
    summary[task] = json.load(open(path))["eval"]
with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

echo "[1/2] Qwen: launching one model per GPU, four tasks per worker"
pids=()
for slot in 0 1; do
  gpu=${gpu_array[$slot]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" evaluate_Ours_LoRA_MoE.py \
    --checkpoint_dir "$QWEN_CKPT" \
    --base_model_name_or_path "$QWEN_BASE" \
    --data_path "$DATA" --inference_tasks "${gpu_tasks[$slot]}" \
    --inference_output_path "$QWEN_OUT" \
    --summary_filename "summary_gpu${gpu}.json" \
    --per_device_eval_batch_size "$QWEN_BATCH" \
    --temperature "$TEMPERATURE" --task_generation_limits \
    "${sari_args[@]}" >"$QWEN_OUT/eval_gpu${gpu}.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || die "Qwen evaluator failed; inspect $QWEN_OUT/eval_gpu*.log"
merge_phase "$QWEN_OUT"
echo "[1/2] Qwen complete; both worker processes exited and GPUs unloaded"

echo "[2/2] OLMoE: launching one model per GPU, four tasks per worker"
pids=()
for slot in 0 1; do
  gpu=${gpu_array[$slot]}
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" evaluate_Ours_MoE_FFN.py \
    --checkpoint_dir "$OLMOE_CKPT" \
    --base_model_name_or_path "$OLMOE_BASE" \
    --data_path "$DATA" --inference_tasks "${gpu_tasks[$slot]}" \
    --inference_output_path "$OLMOE_OUT" \
    --summary_filename "summary_gpu${gpu}.json" \
    --per_device_eval_batch_size "$OLMOE_BATCH" \
    --temperature "$TEMPERATURE" --task_generation_limits \
    --attn_implementation auto \
    "${sari_args[@]}" >"$OLMOE_OUT/eval_gpu${gpu}.log" 2>&1 &
  pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || die "OLMoE evaluator failed; inspect $OLMOE_OUT/eval_gpu*.log"
merge_phase "$OLMOE_OUT"
echo "[2/2] OLMoE complete; both worker processes exited and GPUs unloaded"

"$PYTHON_BIN" - "$QWEN_OUT/summary.json" "$OLMOE_OUT/summary.json" "$COMBINED_OUT" <<'PY'
import json, sys
qwen_path, olmoe_path, output = sys.argv[1:]
payload = {
    "track1_qwen3_lora_moe_round7": json.load(open(qwen_path)),
    "track2_olmoe_full_ffn_round7": json.load(open(olmoe_path)),
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
print(f"Combined summary: {output}")
PY
