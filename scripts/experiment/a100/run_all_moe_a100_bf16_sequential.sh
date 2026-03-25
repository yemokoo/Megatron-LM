#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/Megatron-LM:${PYTHONPATH:-}"
export WANDB_FINISH_TIMEOUT="${WANDB_FINISH_TIMEOUT:-30}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-300}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export PROBE_EVAL_INTERVAL="${PROBE_EVAL_INTERVAL:-100}"
export PROBE_EVAL_ITERS="${PROBE_EVAL_ITERS:-25}"
export SECONDARY_PROBE_EVAL_INTERVAL="${SECONDARY_PROBE_EVAL_INTERVAL:-100}"
export SECONDARY_PROBE_EVAL_ITERS="${SECONDARY_PROBE_EVAL_ITERS:-25}"
export SOURCE_REQUIRED_ITERS="${SOURCE_REQUIRED_ITERS:-$TRAIN_ITERS}"
export WANDB_PROJECT="${WANDB_PROJECT:-flame-continual}"
export SLEEP_BETWEEN_RUNS_SECONDS="${SLEEP_BETWEEN_RUNS_SECONDS:-300}"
export LOCAL_BASE="${LOCAL_BASE:-$PROJECT_ROOT/.local}"
export LOCAL_DATASET="${LOCAL_DATASET:-$LOCAL_BASE/dataset}"
export LOCAL_WEIGHTS="${LOCAL_WEIGHTS:-$LOCAL_BASE/weights}"

run_step() {
    local step_name="$1"
    shift

    echo "============================================================"
    echo "Starting $step_name at $(date)"
    echo "============================================================"

    "$@"

    echo "Finished $step_name at $(date)"
    echo
    sleep "$SLEEP_BETWEEN_RUNS_SECONDS"
}

run_step \
    "wiki_a_a100_bf16" \
    env \
        WANDB_EXP_NAME="${WANDB_EXP_NAME_WIKI_A:-wiki-a-moe-bf16-a100}" \
        TRAIN_DATASET="${TRAIN_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}" \
        PROBE_DATASET="${PROBE_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}" \
        SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET_WIKI:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}" \
        bash "$PROJECT_ROOT/scripts/experiment/a100/wiki_a_a100_bf16.sh"

run_step \
    "code_b_a100_bf16" \
    env \
        WANDB_EXP_NAME="${WANDB_EXP_NAME_CODE_B:-code-b-moe-bf16-a100}" \
        TRAIN_DATASET="${TRAIN_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}" \
        PROBE_DATASET="${PROBE_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}" \
        SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET_CODE:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}" \
        bash "$PROJECT_ROOT/scripts/experiment/a100/code_b_a100_bf16.sh"

run_step \
    "a_to_b_a100_bf16" \
    env \
        WANDB_EXP_NAME="${WANDB_EXP_NAME_A_TO_B:-a-to-b-moe-bf16-a100}" \
        TRAIN_DATASET="${TRAIN_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}" \
        PROBE_DATASET="${PROBE_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}" \
        SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET_A_TO_B:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}" \
        SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR_A_TO_B:-a100/wiki-a-moe-bf16}" \
        bash "$PROJECT_ROOT/scripts/experiment/a100/a_to_b_a100_bf16.sh"

run_step \
    "a_to_b_freeze_a100_bf16" \
    env \
        WANDB_EXP_NAME="${WANDB_EXP_NAME_A_TO_B_FREEZE:-a-to-b-freeze-moe-bf16-a100}" \
        TRAIN_DATASET="${TRAIN_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}" \
        PROBE_DATASET="${PROBE_DATASET_CODE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}" \
        SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET_A_TO_B_FREEZE:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}" \
        SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR_A_TO_B_FREEZE:-a100/wiki-a-moe-bf16}" \
        bash "$PROJECT_ROOT/scripts/experiment/a100/a_to_b_freeze_a100_bf16.sh"

run_step \
    "b_to_a_a100_bf16" \
    env \
        WANDB_EXP_NAME="${WANDB_EXP_NAME_B_TO_A:-b-to-a-moe-bf16-a100}" \
        TRAIN_DATASET="${TRAIN_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}" \
        PROBE_DATASET="${PROBE_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}" \
        SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET_B_TO_A:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}" \
        SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR_B_TO_A:-a100/code-b-moe-bf16}" \
        bash "$PROJECT_ROOT/scripts/experiment/a100/b_to_a_a100_bf16.sh"

run_step \
    "b_to_a_freeze_a100_bf16" \
    env \
        WANDB_EXP_NAME="${WANDB_EXP_NAME_B_TO_A_FREEZE:-b-to-a-freeze-moe-bf16-a100}" \
        TRAIN_DATASET="${TRAIN_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact}" \
        PROBE_DATASET="${PROBE_DATASET_WIKI:-$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact}" \
        SECONDARY_PROBE_DATASET="${SECONDARY_PROBE_DATASET_B_TO_A_FREEZE:-$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact}" \
        SOURCE_RUN_SUBDIR="${SOURCE_RUN_SUBDIR_B_TO_A_FREEZE:-a100/code-b-moe-bf16}" \
        bash "$PROJECT_ROOT/scripts/experiment/a100/b_to_a_freeze_a100_bf16.sh"

echo "All sequential A100 MoE bf16 runs completed at $(date)"
