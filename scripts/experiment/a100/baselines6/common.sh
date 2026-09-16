#!/bin/bash

# Shared, isolated runtime for the six continual-learning baselines.
# This file never calls the existing pretrain/continual launchers.

BASELINES6_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BASELINES6_DIR/../../../.." && pwd)"
MEGATRON_ROOT="$PROJECT_ROOT/Megatron-LM"

export BASELINES6_PYTHON="${BASELINES6_PYTHON:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python}"
export BASELINES6_PYTHONPATH_OVERLAY="${BASELINES6_PYTHONPATH_OVERLAY:-$PROJECT_ROOT/trace/.venv-runtime/lib/python3.10/site-packages}"
# Megatron builds its dataset helper with the bare `python3` and
# `python3-config` commands from its existing Makefile.  Keep that build on the
# same interpreter as the isolated baseline process instead of inheriting a
# different base-conda executable from the login shell.
BASELINES6_PYTHON_BIN="$(cd "$(dirname "$BASELINES6_PYTHON")" && pwd)"
export PATH="$BASELINES6_PYTHON_BIN:$PATH"
export BASELINES6_DATA_ROOT="${BASELINES6_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
export BASELINES6_OUTPUT_ROOT="${BASELINES6_OUTPUT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_wiki_code_conversation_20260816}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export SEED="${SEED:-1234}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
export TRAIN_ITERS="${TRAIN_ITERS:-1800}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-600}"
export PROBE_INTERVAL="${PROBE_INTERVAL:-50}"
export PROBE_ITERS="${PROBE_ITERS:-25}"
export SEQ_LENGTH="${SEQ_LENGTH:-512}"
export LR="${LR:-3e-4}"
export MIN_LR="${MIN_LR:-3e-5}"
export PAUSE_SECONDS="${PAUSE_SECONDS:-30}"
export FISHER_BATCHES="${FISHER_BATCHES:-100}"

if [ "${BASELINES6_SMOKE:-0}" = "1" ]; then
    export TRAIN_ITERS="${SMOKE_TRAIN_ITERS:-2}"
    export SAVE_INTERVAL=1
    export PROBE_INTERVAL=2
    export PROBE_ITERS=1
    export FISHER_BATCHES="${SMOKE_FISHER_BATCHES:-2}"
    export BASELINES6_OUTPUT_ROOT="${BASELINES6_SMOKE_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/baselines6_smoke_20260816}"
fi

task_train_dir() {
    printf '%s/%s/train\n' "$BASELINES6_DATA_ROOT" "$1"
}

task_test_dir() {
    printf '%s/%s/test\n' "$BASELINES6_DATA_ROOT" "$1"
}

build_data_path_lines() {
    local dataset_dir="$1"
    local bin_path
    while IFS= read -r -d '' bin_path; do
        printf '1.0\n%s\n' "${bin_path%.bin}"
    done < <(find "$dataset_dir" -maxdepth 1 -type f -name '*.bin' -print0 | sort -z)
}

stage_is_complete() {
    local output_dir="$1"
    local expected_iters="${2:-$TRAIN_ITERS}"
    local tracker="$output_dir/latest_checkpointed_iteration.txt"
    local final_audit="$output_dir/continual_audit/final.json"
    [ -f "$tracker" ] && \
        [ "$(tr -d '[:space:]' < "$tracker")" = "$expected_iters" ] && \
        [ -f "$final_audit" ]
}

# A source checkpoint is usable when its tracker matches its own final audit,
# whatever step count that stage happened to run.
source_stage_finished() {
    local d="$1"
    [ -f "$d/latest_checkpointed_iteration.txt" ] && [ -f "$d/continual_audit/final.json" ] || return 1
    local tracker; tracker="$(tr -d '[:space:]' < "$d/latest_checkpointed_iteration.txt")"
    [ -n "$tracker" ] && [ "$tracker" -gt 0 ] 2>/dev/null
}

# Number of sequences in a fixed replay subset, read from the indexed dataset
# so the blend weight matches what the loader will actually see.
replay_subset_sequences() {
    local dir="$1"
    PYTHONPATH="$BASELINES6_PYTHONPATH_OVERLAY:$MEGATRON_ROOT" "$BASELINES6_PYTHON" - "$dir/train_text_document" <<'PY2' 2>/dev/null
import sys
from megatron.core.datasets import indexed_dataset
print(len(indexed_dataset.IndexedDataset(sys.argv[1], multimodal=False, mmap=True)))
PY2
}

# Micro-batches sized from measured peaks (~0.38 GB/seq for dense, ~0.53 for
# Fixed MoE) against a 65 GB ceiling on the 80 GB cards; each must divide
# GBS/DP = 1152.  Larger mb only shortens gradient accumulation -- the
# optimizer sees the same 2304-sample global batch, so results are unchanged.
#   slora_pre 24->144, olora 32->144, fixed_moe 48->96, ewc 64->128,
#   trace_gem 128, sequential_dense 72->144.  Set the env vars to override.
method_micro_batch() {
    case "$1" in
        ewc|trace_gem) echo "${DENSE_REGULARIZER_MB:-128}" ;;
        gem_episodic) echo "${GEM_EPISODIC_MB:-96}" ;;
        sequential_dense) echo "${SEQUENTIAL_DENSE_MB:-144}" ;;
        slora_pre) echo "${SLORA_MB:-144}" ;;
        olora) echo "${OLORA_MB:-144}" ;;
        fixed_moe) echo "${FIXED_MOE_MB:-96}" ;;
        *) echo "${COMMON_DENSE_WIKI_MB:-64}" ;;
    esac
}

method_port_base() {
    case "$1" in
        ewc) echo 29710 ;;
        trace_gem) echo 29720 ;;
        gem_episodic) echo 29770 ;;
        slora_pre) echo 29730 ;;
        olora) echo 29740 ;;
        sequential_dense) echo 29750 ;;
        fixed_moe) echo 29760 ;;
        *) echo 29700 ;;
    esac
}

state_dir() {
    local checkpoint="$1"
    local method="$2"
    printf '%s/continual_state_%s\n' "$checkpoint" "$method"
}
