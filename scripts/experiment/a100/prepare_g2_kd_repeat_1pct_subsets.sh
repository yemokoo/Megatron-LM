#!/bin/bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$D/../../.." && pwd)"
source "$D/common.sh"

SEED="${KD_SUBSET_SEED:-20260725}"
SAMPLES="${KD_SUBSET_SAMPLES:-41472}"
SEQ="${SEQ_LENGTH:-512}"
TOKENIZER="${TOKENIZER_MODEL:-$R/.local/models/pythia-12b-tokenizer}"
ROOT="${KD_SUBSET_ROOT:-$R/data/kd_subsets/g2_repeat_1pct_seed${SEED}}"

prepare_one() {
    local task="$1" source_dir="$2" output_dir="$3" sample_count="$4" purpose="$5" metadata
    metadata="$output_dir/subset_metadata.json"
    if [ -f "$metadata" ] && [ -f "$output_dir/train_text_document.bin" ] && [ -f "$output_dir/train_text_document.idx" ]; then
        "$PYTHON_BIN" - "$metadata" "$task" "$source_dir" "$sample_count" "$SEED" "$SEQ" "$purpose" <<'PY'
import json, sys
from pathlib import Path
metadata, task, source, samples, seed, seq, purpose = sys.argv[1:]
d = json.loads(Path(metadata).read_text())
expected = {"purpose": purpose, "input_dir": source,
            "random_seed": int(seed), "sequence_length": int(seq)}
for key, value in expected.items():
    if d.get(key) != value:
        raise SystemExit(f"ERROR: stale {task} subset: {key}={d.get(key)!r}, expected {value!r}")
if int(d.get("memory", {}).get("samples", -1)) != int(samples):
    raise SystemExit(f"ERROR: stale {task} subset sample count")
PY
        echo "[SKIP] verified $task subset: $output_dir"
        return
    fi
    if [ -d "$output_dir" ] && [ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "ERROR: incomplete non-empty subset directory: $output_dir" >&2
        exit 1
    fi
    mkdir -p "$output_dir"
    "$PYTHON_BIN" "$R/scripts/dataset/materialize_fixed_sample_stream.py" \
        --input-dir "$source_dir" --output-dir "$output_dir" --samples "$sample_count" \
        --sequence-length "$SEQ" --random-seed "$SEED" --tokenizer-model "$TOKENIZER" \
        --purpose "$purpose" --metadata-name subset_metadata.json
}

prepare_one wiki "$(dataset_dir_for_task wiki)" "$ROOT/wiki" "$SAMPLES" kd_fixed_random_1pct_repeat20
prepare_one wiki_half "$(dataset_dir_for_task wiki)" "$ROOT/wiki_half" "$((SAMPLES / 2))" kd_fixed_random_0p5pct_repeat20
prepare_one code "$(dataset_dir_for_task code)" "$ROOT/code" "$SAMPLES" kd_fixed_random_1pct_repeat20
prepare_one code_half "$(dataset_dir_for_task code)" "$ROOT/code_half" "$((SAMPLES / 2))" kd_fixed_random_0p5pct_repeat20
echo "[DONE] fixed KD subsets at $ROOT"
