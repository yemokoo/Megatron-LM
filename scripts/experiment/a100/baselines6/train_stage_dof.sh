#!/bin/bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_baselines6_stage() {
    if [ "$#" -lt 7 ]; then
        echo "usage: run_baselines6_stage METHOD TASK OUTPUT SOURCE STATE_LOAD MICRO_BATCH PORT [METHOD_ARGS...]" >&2
        return 2
    fi
    local method="$1" task="$2" output_dir="$3" source_dir="$4"
    local state_load="$5" micro_batch="$6" master_port="$7"
    shift 7
    local method_args=("$@")

    if [ ! -x "$BASELINES6_PYTHON" ]; then
        echo "ERROR: baseline Python not executable: $BASELINES6_PYTHON" >&2
        return 1
    fi
    local train_dir
    train_dir="$(task_train_dir "$task")"
    if [ ! -d "$train_dir" ]; then
        echo "ERROR: missing dataset directory: $train_dir" >&2
        return 1
    fi
    mapfile -t data_path < <(build_data_path_lines "$train_dir")
    if [ "${#data_path[@]}" -eq 0 ]; then
        echo "ERROR: no indexed dataset shards in $train_dir" >&2
        return 1
    fi

    # Optional replay: blend fixed old-task subsets into the training stream.
    # Megatron blend weights are sample-count ratios, so weighting the primary
    # task by its full sample count and each replay subset by (its size x
    # epochs) makes the loader draw exactly that many replay sequences over
    # the run.  Steps grow to cover the extra samples: 4,147,200 + 829,400 =
    # 4,976,600 / 2304 = 2160.  This mirrors the lm-loss router-finetune
    # budget (0.1% x 200 epochs, ~360 steps' worth) that our own method saw.
    #   BASELINES6_REPLAY_DIRS   space-separated dirs, one indexed prefix each
    #   BASELINES6_REPLAY_EPOCHS repeats per subset (default 200)
    local stage_train_iters="$TRAIN_ITERS"
    if [ -n "${BASELINES6_REPLAY_DIRS:-}" ]; then
        local primary_samples=$(( TRAIN_ITERS * GLOBAL_BATCH_SIZE ))
        local replay_epochs="${BASELINES6_REPLAY_EPOCHS:-200}"
        local replay_total=0 rdir rseqs
        local -a blended=()
        # Primary shards keep their relative 1.0 weights but are rescaled so the
        # whole primary block weighs primary_samples.
        local nprim=$(( ${#data_path[@]} / 2 ))
        local i
        for (( i=0; i<${#data_path[@]}; i+=2 )); do
            blended+=("$(awk -v n=$primary_samples -v k=$nprim 'BEGIN{printf "%.6f", n/k}')" "${data_path[i+1]}")
        done
        for rdir in $BASELINES6_REPLAY_DIRS; do
            rseqs="$(replay_subset_sequences "$rdir")"
            [ -n "$rseqs" ] || { echo "ERROR: cannot size replay subset $rdir" >&2; return 1; }
            replay_total=$(( replay_total + rseqs * replay_epochs ))
            blended+=("$(( rseqs * replay_epochs ))" "$rdir/train_text_document")
        done
        data_path=("${blended[@]}")
        # Round the step count to the nearest global batch rather than up:
        # 829,400 (code) and 829,600 (conv) both map to 2160, so both stages
        # of a method run the same schedule.  The <0.03% of replay that falls
        # off the end is immaterial; mismatched step counts are not.
        stage_train_iters=$(( (primary_samples + replay_total + GLOBAL_BATCH_SIZE / 2) / GLOBAL_BATCH_SIZE ))
        echo "[REPLAY] primary=$primary_samples replay=$replay_total (${replay_epochs} epochs) -> train_iters $TRAIN_ITERS -> $stage_train_iters"
    fi
    if [ "${BASELINES6_EVAL_ONLY:-0}" != "1" ] && stage_is_complete "$output_dir" "$stage_train_iters"; then
        echo "[SKIP] $method/$task complete: $output_dir"
        return 0
    fi

    local primary_task="$task" secondary_task tertiary_task
    case "$task" in
        wiki) secondary_task=code; tertiary_task=conversation ;;
        code) secondary_task=wiki; tertiary_task=conversation ;;
        conversation) secondary_task=wiki; tertiary_task=code ;;
    esac
    mapfile -t primary_probe < <(build_data_path_lines "$(task_test_dir "$primary_task")")
    mapfile -t secondary_probe < <(build_data_path_lines "$(task_test_dir "$secondary_task")")
    mapfile -t tertiary_probe < <(build_data_path_lines "$(task_test_dir "$tertiary_task")")

    mkdir -p "$output_dir/logs"
    local run_log="$output_dir/logs/train.log"
    local gpu_log="$output_dir/logs/gpu_usage.csv"
    local resume=0 load_dir="$output_dir"
    local infra_args=(
        --transformer-impl local
        --tensor-model-parallel-size 1
        --pipeline-model-parallel-size 1
        --expert-model-parallel-size 1
        --distributed-timeout-minutes 30
        --no-persist-layer-norm
        --dist-ckpt-strictness log_unexpected
    )
    if [ -f "$output_dir/latest_checkpointed_iteration.txt" ]; then
        resume=1
        load_dir="$output_dir"
    elif [ -n "$source_dir" ]; then
        # The source stage finished with whatever step count *it* ran (Wiki:
        # TRAIN_ITERS; a replay Code stage: its own extended count).  Judge it
        # by its own tracker + final audit, never by this stage's step count.
        if ! source_stage_finished "$source_dir"; then
            echo "ERROR: source stage is not complete: $source_dir" >&2
            return 1
        fi
        load_dir="$source_dir"
        infra_args+=(--finetune --no-load-optim --no-load-rng)
    fi

    local model_args=(
        --hidden-size 1024
        --ffn-hidden-size "${DENSE_FFN_HIDDEN_SIZE:-5472}"
        --num-layers 9
        --num-attention-heads 16
        --group-query-attention
        --num-query-groups 16
        --swiglu
        --max-position-embeddings 2048
        --normalization RMSNorm
        --norm-epsilon 1e-6
        --untie-embeddings-and-output-weights
        --position-embedding-type rope
        --disable-bias-linear
        --hidden-dropout 0.0
        --attention-dropout 0.0
        --init-method-std 0.02
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model EleutherAI/pythia-12b
    )
    if [ "$method" = "fixed_moe" ]; then
        model_args+=(
            --moe-ffn-hidden-size 352
            --num-experts 24
            --moe-router-topk 4
            --moe-layer-freq '[0,1,1,1,1,1,1,1,1]'
            --moe-router-dtype fp32
            --moe-router-pre-softmax
            --moe-router-score-function softmax
            --moe-aux-loss-coeff 0.01
            --moe-z-loss-coeff 0.001
            --moe-grouped-gemm
        )
    fi

    local continual_args=(
        --continual-method "$method"
        --continual-task-name "$task"
        --continual-trainable-layer-start 2
        --continual-trainable-layer-end 9
        # DoF 스윕: dense FFN 폭을 env 로 개방 (미지정 시 기존 1408 그대로)
        --continual-dense-ffn-hidden-size "${CONTINUAL_DENSE_FFN_HIDDEN_SIZE:-1408}"
        --continual-ewc-fisher-batches "$FISHER_BATCHES"
    )
    if [ -n "$state_load" ]; then
        continual_args+=(--continual-state-load "$state_load")
    fi
    continual_args+=("${method_args[@]}")

    local train_args=(
        --bf16
        --dataloader-type cyclic
        # Let stop_chain.sh interrupt a stage without discarding work: on
        # SIGTERM the trainer saves at the next iteration boundary and exits.
        # The incomplete tracker then fails the stage_is_complete check below,
        # so run_all.sh stops instead of advancing to the next baseline.
        --exit-signal-handler
        --micro-batch-size "$micro_batch"
        --global-batch-size "$GLOBAL_BATCH_SIZE"
        --lr "$LR"
        --min-lr "$MIN_LR"
        --lr-decay-style WSD
        --lr-decay-iters "$stage_train_iters"
        --lr-warmup-fraction 0.01
        --lr-wsd-decay-iters "$(( stage_train_iters > 10 ? stage_train_iters / 10 : 1 ))"
        --train-iters "$stage_train_iters"
        --seed "$SEED"
        --seq-length "$SEQ_LENGTH"
    )
    # Evaluation mode: load the checkpoint, run the three probes once, exit.
    # Reuses the exact probe path the training runs used, so the numbers are
    # produced the same way -- only the batch size and sample budget change.
    if [ "${BASELINES6_EVAL_ONLY:-0}" = "1" ]; then
        train_args+=(--skip-train --run-initial-probe-eval)
    fi
    local io_args=(
        --data-path "${data_path[@]}"
        --split 100,0,0
        --log-interval "${LOG_INTERVAL:-20}"
        --log-throughput
        --log-progress
        --save "$output_dir"
        --save-interval "$SAVE_INTERVAL"
        --load "$load_dir"
        --eval-interval "$(( stage_train_iters + 1 ))"
        --tensorboard-dir "$output_dir"
        --probe-name "${primary_task}_probe"
        --probe-eval-iters "$PROBE_ITERS"
        --probe-eval-interval "$PROBE_INTERVAL"
        --probe-step-offset 0
        --probe-data-path "${primary_probe[@]}"
        --secondary-probe-name "${secondary_task}_probe"
        --secondary-probe-eval-iters "$PROBE_ITERS"
        --secondary-probe-eval-interval "$PROBE_INTERVAL"
        --secondary-probe-step-offset 0
        --secondary-probe-data-path "${secondary_probe[@]}"
        --tertiary-probe-name "${tertiary_task}_probe"
        --tertiary-probe-eval-iters "$PROBE_ITERS"
        --tertiary-probe-eval-interval "$PROBE_INTERVAL"
        --tertiary-probe-step-offset 0
        --tertiary-probe-data-path "${tertiary_probe[@]}"
    )

    echo "[START] method=$method task=$task mb=$micro_batch gbs=$GLOBAL_BATCH_SIZE iters=$stage_train_iters resume=$resume"
    echo "[PATH] output=$output_dir source=${source_dir:-none} state=${state_load:-none}"
    (
        while true; do
            nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$gpu_log"
            sleep 30
        done
    ) &
    local gpu_logger_pid=$!
    trap 'kill "$gpu_logger_pid" 2>/dev/null || true' RETURN

    set +e
    PYTHONPATH="$BASELINES6_PYTHONPATH_OVERLAY:$MEGATRON_ROOT" "$BASELINES6_PYTHON" -m torch.distributed.run \
        --standalone --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \
        --master_addr 127.0.0.1 --master_port "$master_port" \
        "$MEGATRON_ROOT/pretrain_gpt_baselines6.py" \
        "${model_args[@]}" "${infra_args[@]}" "${train_args[@]}" \
        "${io_args[@]}" "${continual_args[@]}" 2>&1 | tee -a "$run_log"
    local status=${PIPESTATUS[0]}
    set -e
    kill "$gpu_logger_pid" 2>/dev/null || true
    trap - RETURN
    if [ "$status" -ne 0 ]; then
        echo "[FAIL] method=$method task=$task exit=$status log=$run_log" >&2
        return "$status"
    fi
    if [ "${BASELINES6_EVAL_ONLY:-0}" != "1" ] && ! stage_is_complete "$output_dir" "$stage_train_iters"; then
        echo "ERROR: successful process did not leave a complete checkpoint: $output_dir" >&2
        return 1
    fi
    echo "[DONE] method=$method task=$task output=$output_dir"
    sleep "$PAUSE_SECONDS"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    run_baselines6_stage "$@"
fi
