#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seonghyeonnoh/yemokoo/30_flame_agent/LLM-continual-learning"
TRACE="$ROOT/trace"
OURS="$TRACE/implementations/llmcl_benchmark"
PY="$TRACE/.venv-runtime/bin/python"
MODEL="$TRACE/models/Llama-3.1-8B"
DATA="$TRACE/data/trace"
RUN_ROOT="$TRACE/results/full_runs/llama31"
V2="$RUN_ROOT/ours_lora_moe_v2_strict_1phase_ratio5"
V3="$RUN_ROOT/ours_lora_moe_v3_strict_1phase"
LOG_ROOT="/data3/seonghyeonnoh/LLM-continual-learning-runs/local/logs"
STATE="$LOG_ROOT/trace_sparse15_repair"
G2="$ROOT/scripts/experiment/a100/run_g2_7run_hidden_kl_c100to10_s600_postfirst_h100_chain_mha.sh"
G2_LOG="$LOG_ROOT/g2_7run_hidden_kl_c100to10_s600_launcher.log"
SENTINEL="$LOG_ROOT/g2_7run_hidden_kl_c100to10_s600_launched.sentinel"
TASKS=(C-STANCE FOMC MeetingBank Py150 ScienceQA NumGLUE-cm NumGLUE-ds 20Minuten)

timestamp() { date '+%F %T %Z'; }
summary_path() { echo "$1/evaluation/order$2/$3.summary.json"; }

enqueue() {
    local queue="$1" version="$2" run="$3" round="$4" task="$5" batch="$6"
    test -s "$(summary_path "$run" "$round" "$task")" ||
        printf '%s|%s|%s|%s|%s\n' "$version" "$run" "$round" "$task" "$batch" >> "$queue"
}

claim() {
    local queue="$1" lock="$2" tmp line
    exec 9>>"$lock"; flock 9
    if ! IFS= read -r line < "$queue" || test -z "$line"; then
        flock -u 9; exec 9>&-; return 1
    fi
    tmp="$queue.$$"; tail -n +2 "$queue" > "$tmp"; mv "$tmp" "$queue"
    flock -u 9; exec 9>&-
    echo "$line"
}

long_worker() {
    local gpus="$1" line version run round task batch
    while line="$(claim "$STATE/long.tsv" "$STATE/long.lock")"; do
        IFS='|' read -r version run round task batch <<< "$line"
        echo "[$(timestamp)] LONG START $version order$round $task gpus=$gpus batch=$batch"
        TRACE_PYTHON="$PY" "$PY" "$TRACE/scripts/run_ours_py150_4way.py" \
            --run-dir "$run" --round "$round" --task "$task" \
            --batch "$batch" --gpus "$gpus"
        echo "[$(timestamp)] LONG DONE $version order$round $task"
    done
}

short_worker() {
    local gpu="$1" line version run round task batch out checkpoint
    while line="$(claim "$STATE/short.tsv" "$STATE/short.lock")"; do
        IFS='|' read -r version run round task batch <<< "$line"
        out="$run/evaluation/order$round"; checkpoint="$run/$((round - 1))"
        mkdir -p "$out"
        echo "[$(timestamp)] SHORT START $version order$round $task gpu=$gpu batch=$batch"
        (
            export CUDA_VISIBLE_DEVICES="$gpu"
            export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
            cd "$OURS"
            "$PY" evaluate_Ours_LoRA_MoE.py \
                --checkpoint_dir "$checkpoint" --base_model_name_or_path "$MODEL" \
                --data_path "$DATA" --inference_tasks "$task" \
                --inference_output_path "$out" --summary_filename "$task.summary.json" \
                --max_prompt_len 0 --max_ans_len 1024 --no-task_generation_limits \
                --slora_conv_mode llama3 --per_device_eval_batch_size "$batch" --temperature 0
        ) > "$out/$task.repair.log" 2>&1
        echo "[$(timestamp)] SHORT DONE $version order$round $task"
    done
}

validate() {
    local run round i task
    for run in "$V2" "$V3"; do
        for round in 1 2 3 4 5 6 7; do
            i=$((round - 1)); test -s "$(summary_path "$run" "$round" "${TASKS[$i]}")"
        done
        for task in "${TASKS[@]}"; do test -s "$(summary_path "$run" 8 "$task")"; done
    done
}

collect() {
    "$PY" "$TRACE/scripts/collect_results.py" --method "ours_lora_moe_$1" \
        --model llama31 --run-dir "$2" --family paper_baseline --sparse-15 \
        --output "$2/sparse15_summary.json"
}

mkdir -p "$STATE"
: > "$STATE/long.tsv"; : > "$STATE/short.tsv"
# Only the two long-form/OOM-prone tasks use the existing four-way sample split.
enqueue "$STATE/long.tsv" v2 "$V2" 3 MeetingBank 2
enqueue "$STATE/long.tsv" v3 "$V3" 4 Py150 8
enqueue "$STATE/long.tsv" v2 "$V2" 8 Py150 8
enqueue "$STATE/long.tsv" v3 "$V3" 8 Py150 8
enqueue "$STATE/long.tsv" v2 "$V2" 8 MeetingBank 2
enqueue "$STATE/long.tsv" v3 "$V3" 3 MeetingBank 2
enqueue "$STATE/long.tsv" v3 "$V3" 8 MeetingBank 2

# Everything else remains one cell per GPU.
enqueue "$STATE/short.tsv" v2 "$V2" 7 NumGLUE-ds 4
enqueue "$STATE/short.tsv" v2 "$V2" 8 ScienceQA 4
enqueue "$STATE/short.tsv" v2 "$V2" 8 NumGLUE-ds 4
enqueue "$STATE/short.tsv" v2 "$V2" 8 20Minuten 4
enqueue "$STATE/short.tsv" v3 "$V3" 7 NumGLUE-ds 4
enqueue "$STATE/short.tsv" v3 "$V3" 8 C-STANCE 4
enqueue "$STATE/short.tsv" v3 "$V3" 8 ScienceQA 4
enqueue "$STATE/short.tsv" v3 "$V3" 8 NumGLUE-ds 4
enqueue "$STATE/short.tsv" v3 "$V3" 8 20Minuten 4

echo "[$(timestamp)] repair start long=$(wc -l < "$STATE/long.tsv") short=$(wc -l < "$STATE/short.tsv")"
long_worker 0,1,2,3 & a=$!
long_worker 4,5,6,7 & b=$!
wait "$a"; wait "$b"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do short_worker "$gpu" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid"; done

validate
collect v2 "$V2"; collect v3 "$V3"
echo "[$(timestamp)] sparse-15 complete"
if ! (set -o noclobber; printf '%s\n' "$(timestamp)" > "$SENTINEL") 2>/dev/null; then
    echo "G2 sentinel exists; refusing duplicate" >&2; exit 1
fi
cd "$ROOT"
exec bash "$G2" >> "$G2_LOG" 2>&1
