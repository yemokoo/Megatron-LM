#!/bin/bash
# Evaluate Qwen LoRA-MoE ScienceQA with two independent data-parallel replicas.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON_BIN=${PYTHON_BIN:-/home/work/Agent_HJ/30_flame_agent/envs/train_env/bin/python}
GPUS=${GPUS:-0,1}
BATCH=${BATCH:-64}
BASE=${BASE:-/home/work/Agent_HJ/00_models/Qwen3-8B}
CKPT=${CKPT:-/home/work/Agent_HJ/30_flame_agent/llmcl_benchmark/output/track1_Qwen3-8B_full/7}
DATA=${DATA:-data/LLM-CL-Benchmark_5000}
OUT=${OUT:-eval_out/track1_Qwen3-8B_full_round7_trace_taskcaps_2gpu}

IFS=',' read -r -a gpu_array <<<"$GPUS"
[ "${#gpu_array[@]}" -eq 2 ] || {
  echo "exactly two GPU IDs are required" >&2
  exit 2
}
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

pids=()
for shard_id in 0 1; do
  gpu=${gpu_array[$shard_id]}
  suffix=".shard${shard_id}-of-2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" evaluate_Ours_LoRA_MoE.py \
    --checkpoint_dir "$CKPT" \
    --base_model_name_or_path "$BASE" \
    --data_path "$DATA" \
    --inference_tasks ScienceQA \
    --inference_output_path "$OUT" \
    --summary_filename "summary_science_shard${shard_id}.json" \
    --num_sample_shards 2 \
    --sample_shard_id "$shard_id" \
    --result_suffix "$suffix" \
    --per_device_eval_batch_size "$BATCH" \
    --temperature 0.0 \
    --task_generation_limits \
    >"$OUT/eval_science_gpu${gpu}.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "ScienceQA shard failed; inspect $OUT/eval_science_gpu*.log" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/merge_trace_shards.py \
  --input_dir "$OUT" --task ScienceQA --num_shards 2
PYTHONWARNINGS=ignore "$PYTHON_BIN" scripts/rescore_trace_outputs.py \
  --input_dir "$OUT" --write

echo "ScienceQA 2-GPU evaluation complete: $OUT/results-ScienceQA.json"
