#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

resolve_python() {
    if [ -x "$PROJECT_ROOT/.conda/envs/flame3090/bin/python" ]; then
        echo "$PROJECT_ROOT/.conda/envs/flame3090/bin/python"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    command -v python3
}

export PYTHON_BIN="${PYTHON_BIN:-$(resolve_python)}"

default_train_dir_for_task() {
    case "$1" in
        wiki)
            if [ -d "$PROJECT_ROOT/data/wiki/train" ]; then
                echo "$PROJECT_ROOT/data/wiki/train"
            else
                echo "$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact"
            fi
            ;;
        code)
            if [ -d "$PROJECT_ROOT/data/code/train" ]; then
                echo "$PROJECT_ROOT/data/code/train"
            else
                echo "$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact"
            fi
            ;;
        *)
            echo "ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}

default_probe_dir_for_task() {
    case "$1" in
        wiki)
            if [ -d "$PROJECT_ROOT/data/wiki/test" ]; then
                echo "$PROJECT_ROOT/data/wiki/test"
            else
                echo "$LOCAL_DATASET/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact"
            fi
            ;;
        code)
            if [ -d "$PROJECT_ROOT/data/code/test" ]; then
                echo "$PROJECT_ROOT/data/code/test"
            else
                echo "$LOCAL_DATASET/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact"
            fi
            ;;
        *)
            echo "ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}

build_data_path() {
    "$PYTHON_BIN" - "$@" <<'PY'
import sys
from pathlib import Path

parts = []
for dataset_dir in sys.argv[1:]:
    for bin_path in sorted(Path(dataset_dir).glob('*.bin')):
        parts.extend(['1.0', str(bin_path.with_suffix(''))])
print(' '.join(parts))
PY
}

dataset_dir_for_task() {
    default_train_dir_for_task "$1"
}

probe_dir_for_task() {
    default_probe_dir_for_task "$1"
}

weights_subdir_for_task() {
    case "$1" in
        wiki)
            echo "a100/wiki-a-moe-bf16"
            ;;
        code)
            echo "a100/code-b-moe-bf16"
            ;;
        *)
            echo "ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}

stage_label_for_task() {
    case "$1" in
        wiki) echo "wiki_a" ;;
        code) echo "code_b" ;;
        *)
            echo "ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}

dataset_name_for_task() {
    case "$1" in
        wiki) echo "wiki_exact" ;;
        code) echo "code_exact" ;;
        *)
            echo "ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}

dataset_source_for_task() {
    case "$1" in
        wiki) echo "Wikipedia exact train" ;;
        code) echo "Python code exact train" ;;
        *)
            echo "ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}

resolve_completed_run_dir() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

explicit = os.environ.get('SOURCE_WEIGHTS_DIR', '').strip()
if explicit:
    print(explicit)
    raise SystemExit(0)

local_weights = Path(os.environ['LOCAL_WEIGHTS']) / os.environ['SOURCE_RUN_SUBDIR']
required_iters = int(os.environ.get('SOURCE_REQUIRED_ITERS', '1'))
requested_run_id = os.environ.get('SOURCE_RUN_ID', '').strip()

if requested_run_id:
    candidate = local_weights / requested_run_id
    if not candidate.exists():
        raise SystemExit(f"ERROR: requested SOURCE_RUN_ID not found: {candidate}")
    print(candidate)
    raise SystemExit(0)

best = None
best_mtime = -1.0
for candidate in local_weights.iterdir() if local_weights.exists() else []:
    tracker = candidate / 'latest_checkpointed_iteration.txt'
    metadata = candidate / 'logs' / 'run_metadata.json'
    if not candidate.is_dir() or not tracker.exists() or not metadata.exists():
        continue
    try:
        tracker_step = int(tracker.read_text(encoding='utf-8').strip())
        train_iters = int(json.loads(metadata.read_text(encoding='utf-8'))['train_iters'])
    except Exception:
        continue
    if tracker_step < required_iters or train_iters < required_iters:
        continue
    mtime = tracker.stat().st_mtime
    if mtime > best_mtime:
        best = candidate
        best_mtime = mtime

if best is None:
    raise SystemExit(
        f"ERROR: could not find a completed run under {local_weights}. "
        "Set SOURCE_WEIGHTS_DIR or SOURCE_RUN_ID explicitly."
    )

print(best)
PY
}

read_train_iters_from_run() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

metadata = json.loads((Path(os.environ['SOURCE_WEIGHTS_DIR']) / 'logs' / 'run_metadata.json').read_text(encoding='utf-8'))
print(int(metadata['train_iters']))
PY
}

compute_train_iters_from_dataset_dir() {
    "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

dataset_dir = Path(os.environ['SSD_TRAIN_DATASET'])
total_tokens = 0
for idx_path in sorted(dataset_dir.glob('*.idx')):
    ds = indexed_dataset.IndexedDataset(str(idx_path.with_suffix('')), multimodal=False, mmap=True)
    total_tokens += int(ds.sequence_lengths.sum())
seq_length = int(os.environ.get('SEQ_LENGTH', '512'))
global_batch_size = int(os.environ.get('GLOBAL_BATCH_SIZE', '16'))
print(max(1, (total_tokens // seq_length) // global_batch_size))
PY
}

write_base_metadata() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

dataset_dir = Path(os.environ['SSD_TRAIN_DATASET'])
total_tokens = 0
total_documents = 0
shards = []
for idx_path in sorted(dataset_dir.glob('*.idx')):
    prefix = idx_path.with_suffix('')
    ds = indexed_dataset.IndexedDataset(str(prefix), multimodal=False, mmap=True)
    shard_tokens = int(ds.sequence_lengths.sum())
    shard_docs = int(ds.document_indices.shape[0] - 1)
    total_tokens += shard_tokens
    total_documents += shard_docs
    shards.append({'prefix': prefix.name, 'documents': shard_docs, 'tokens': shard_tokens})

metadata = {
    'stage': os.environ['STAGE_NAME'],
    'run_id': os.environ['RUN_ID'],
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'train_dataset': {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents, 'shards': shards},
    'train_iters': int(os.environ['TRAIN_ITERS']),
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'moe_ffn_hidden_size': int(os.environ['MOE_FFN_HIDDEN_SIZE']),
    'num_experts': int(os.environ['NUM_EXPERTS']),
    'moe_router_topk': int(os.environ['MOE_ROUTER_TOPK']),
    'precision': os.environ['PRECISION'],
    'shared_expert_enabled': False,
}
with open(os.environ['RUN_METADATA'], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
PY
}

write_continual_metadata() {
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from megatron.core.datasets import indexed_dataset

dataset_dir = Path(os.environ['SSD_TRAIN_DATASET'])
total_tokens = 0
total_documents = 0
shards = []
for idx_path in sorted(dataset_dir.glob('*.idx')):
    prefix = idx_path.with_suffix('')
    ds = indexed_dataset.IndexedDataset(str(prefix), multimodal=False, mmap=True)
    shard_tokens = int(ds.sequence_lengths.sum())
    shard_docs = int(ds.document_indices.shape[0] - 1)
    total_tokens += shard_tokens
    total_documents += shard_docs
    shards.append({'prefix': prefix.name, 'documents': shard_docs, 'tokens': shard_tokens})

freeze_shared = os.environ.get('TRAIN_NEW_EXPERTS_AND_ROUTER_ONLY', '0') == '1'
metadata = {
    'stage': os.environ['STAGE_NAME'],
    'run_id': os.environ['RUN_ID'],
    'source_weights_dir': os.environ['SOURCE_WEIGHTS_DIR'],
    'source_num_experts': int(os.environ['SOURCE_NUM_EXPERTS']),
    'target_num_experts': int(os.environ['NUM_EXPERTS']),
    'dataset_name': os.environ['DATASET_NAME'],
    'dataset_source': os.environ['DATASET_SOURCE'],
    'train_dataset': {'path': str(dataset_dir), 'tokens': total_tokens, 'documents': total_documents, 'shards': shards},
    'train_iters': int(os.environ['TRAIN_ITERS']),
    'micro_batch_size': int(os.environ['MICRO_BATCH_SIZE']),
    'global_batch_size': int(os.environ['GLOBAL_BATCH_SIZE']),
    'num_layers': int(os.environ['NUM_LAYERS']),
    'hidden_size': int(os.environ['HIDDEN_SIZE']),
    'ffn_hidden_size': int(os.environ['FFN_HIDDEN_SIZE']),
    'moe_ffn_hidden_size': int(os.environ['MOE_FFN_HIDDEN_SIZE']),
    'moe_router_topk': int(os.environ['MOE_ROUTER_TOPK']),
    'precision': os.environ['PRECISION'],
    'shared_expert_enabled': False,
    'shared_frozen': freeze_shared,
    'old_model_kl_enabled': not freeze_shared,
    'old_model_kl_coeff': None if freeze_shared else float(os.environ['OLD_MODEL_KL_COEFF']),
    'old_model_kl_temperature': None if freeze_shared else float(os.environ['OLD_MODEL_KL_TEMPERATURE']),
    'probe_step_offset': int(os.environ['PROBE_STEP_OFFSET']),
}
with open(os.environ['RUN_METADATA'], 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
PY
}
