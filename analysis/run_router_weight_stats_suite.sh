#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"
DEFAULT_PROJECT_PYTHON="$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    export PYTHON_BIN
elif [[ -x "$DEFAULT_PROJECT_PYTHON" ]]; then
    export PYTHON_BIN="$DEFAULT_PROJECT_PYTHON"
elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="$(command -v python3)"
else
    echo "ERROR: could not find a usable python interpreter"
    exit 1
fi

export A_TO_B_DIR="${A_TO_B_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347}"
export A_TO_B_ITERATION="${A_TO_B_ITERATION:-1800}"
export A_TO_B_NEW_ONLY_DIR="${A_TO_B_NEW_ONLY_DIR:-$LOCAL_WEIGHTS/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542}"
export A_TO_B_NEW_ONLY_ITERATION="${A_TO_B_NEW_ONLY_ITERATION:-1800}"
export B_TO_A_DIR="${B_TO_A_DIR:-$LOCAL_WEIGHTS/continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645}"
export B_TO_A_ITERATION="${B_TO_A_ITERATION:-1800}"
export B_TO_A_NEW_ONLY_DIR="${B_TO_A_NEW_ONLY_DIR:-$LOCAL_WEIGHTS/continual-stage-B-to-A-new-only/stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509}"
export B_TO_A_NEW_ONLY_ITERATION="${B_TO_A_NEW_ONLY_ITERATION:-1800}"

export GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29800}"
export EVAL_LOAD_ROOT="${EVAL_LOAD_ROOT:-$LOCAL_WEIGHTS/router-weight-stats-loads}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$LOCAL_WEIGHTS/router-weight-stats-suite}"
export OLD_EXPERT_COUNT="${OLD_EXPERT_COUNT:-4}"

IFS=',' read -r -a GPU_LIST <<< "$GPU_DEVICES"
if [ "${#GPU_LIST[@]}" -lt 4 ]; then
    echo "ERROR: GPU_DEVICES must contain at least 4 GPU ids, e.g. 0,1,2,3"
    exit 1
fi

prepare_eval_load_dir() {
    local source_dir="$1"
    local requested_iteration="$2"
    local tag="$3"

    if [ ! -f "$source_dir/latest_checkpointed_iteration.txt" ]; then
        echo "ERROR: missing latest_checkpointed_iteration.txt in $source_dir"
        exit 1
    fi

    if [ "$requested_iteration" = "latest" ]; then
        printf '%s\n' "$source_dir"
        return 0
    fi

    local checkpoint_dir="$source_dir/iter_$(printf '%07d' "$requested_iteration")"
    if [ ! -d "$checkpoint_dir" ]; then
        echo "ERROR: missing checkpoint directory $checkpoint_dir"
        exit 1
    fi

    mkdir -p "$EVAL_LOAD_ROOT"
    local load_dir="$EVAL_LOAD_ROOT/${tag}-minimal-iter-$(printf '%07d' "$requested_iteration")"
    local load_checkpoint_dir="$load_dir/iter_$(printf '%07d' "$requested_iteration")"

    if [ -f "$load_dir/latest_checkpointed_iteration.txt" ] && [ -d "$load_checkpoint_dir" ]; then
        local existing_iteration
        existing_iteration="$(tr -d '[:space:]' < "$load_dir/latest_checkpointed_iteration.txt" || true)"
        if [ "$existing_iteration" = "$requested_iteration" ]; then
            printf '%s\n' "$load_dir"
            return 0
        fi
    fi

    mkdir -p "$load_checkpoint_dir"
    rsync -rlptD --delete "$checkpoint_dir/" "$load_checkpoint_dir/"
    printf '%s\n' "$requested_iteration" > "$load_dir/latest_checkpointed_iteration.txt"
    printf '%s\n' "$load_dir"
}

run_export() {
    local gpu_id="$1"
    local load_dir="$2"
    local model_label="$3"
    local output_json="$4"
    local port="$5"

    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -m torch.distributed.run \
        --standalone \
        --nnodes 1 \
        --nproc_per_node 1 \
        --master_port "$port" \
        "$PROJECT_ROOT/analysis/export_router_weight_stats.py" \
        --load "$load_dir" \
        --model-label "$model_label" \
        --output-json "$output_json" \
        --old-expert-count "$OLD_EXPERT_COUNT" \
        --pipeline-model-parallel-size 1 \
        --expert-model-parallel-size 1 \
        --tensor-model-parallel-size 1 \
        --transformer-impl local \
        --no-persist-layer-norm \
        --no-gradient-accumulation-fusion \
        --no-masked-softmax-fusion \
        --attention-softmax-in-fp32 \
        --no-load-optim \
        --no-load-rng \
        --exit-on-missing-checkpoint
}

mkdir -p "$OUTPUT_ROOT"

A_TO_B_LOAD="$(prepare_eval_load_dir "$A_TO_B_DIR" "$A_TO_B_ITERATION" a-to-b-router-stats)"
A_TO_B_NEW_ONLY_LOAD="$(prepare_eval_load_dir "$A_TO_B_NEW_ONLY_DIR" "$A_TO_B_NEW_ONLY_ITERATION" a-to-b-new-only-router-stats)"
B_TO_A_LOAD="$(prepare_eval_load_dir "$B_TO_A_DIR" "$B_TO_A_ITERATION" b-to-a-router-stats)"
B_TO_A_NEW_ONLY_LOAD="$(prepare_eval_load_dir "$B_TO_A_NEW_ONLY_DIR" "$B_TO_A_NEW_ONLY_ITERATION" b-to-a-new-only-router-stats)"

run_export "${GPU_LIST[0]}" "$A_TO_B_LOAD" "a_to_b_full" "$OUTPUT_ROOT/a_to_b_full.json" "$MASTER_PORT_BASE" &
pid0=$!
run_export "${GPU_LIST[1]}" "$A_TO_B_NEW_ONLY_LOAD" "a_to_b_new_only" "$OUTPUT_ROOT/a_to_b_new_only.json" "$((MASTER_PORT_BASE + 1))" &
pid1=$!
run_export "${GPU_LIST[2]}" "$B_TO_A_LOAD" "b_to_a_full" "$OUTPUT_ROOT/b_to_a_full.json" "$((MASTER_PORT_BASE + 2))" &
pid2=$!
run_export "${GPU_LIST[3]}" "$B_TO_A_NEW_ONLY_LOAD" "b_to_a_new_only" "$OUTPUT_ROOT/b_to_a_new_only.json" "$((MASTER_PORT_BASE + 3))" &
pid3=$!

wait "$pid0" "$pid1" "$pid2" "$pid3"

"$PYTHON_BIN" "$PROJECT_ROOT/analysis/plot_router_weight_stats_suite.py" \
    --a-to-b-full "$OUTPUT_ROOT/a_to_b_full.json" \
    --a-to-b-new-only "$OUTPUT_ROOT/a_to_b_new_only.json" \
    --b-to-a-full "$OUTPUT_ROOT/b_to_a_full.json" \
    --b-to-a-new-only "$OUTPUT_ROOT/b_to_a_new_only.json" \
    --output-dir "$OUTPUT_ROOT"

echo "Router weight stats summary: $OUTPUT_ROOT/router_weight_summary.json"
echo "Router abs-mean heatmaps:   $OUTPUT_ROOT/router_abs_mean_heatmaps.svg"
echo "Router L2 heatmaps:         $OUTPUT_ROOT/router_l2_norm_heatmaps.svg"
echo "Router group summary:       $OUTPUT_ROOT/router_group_summary.svg"
