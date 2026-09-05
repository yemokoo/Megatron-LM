#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Sequential, independent 4-GPU comparison from the same post-expansion
# KD-init checkpoint:
#   1) Code LM + old-like Code-token LM replay (router gradients only)
#   2) Code LM + old-like Code-token layer-output MSE replay (router gradients
#      only, residual-included Layers 2--9, coefficient 10)
#
# This launcher never stops or replaces an existing process.  It refuses to
# start when GPUs 0--3 are busy, a port is occupied, the chain lock is held, or
# an output directory contains anything other than a verified completed stage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

FLAME_ENV="${FLAME_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$FLAME_ENV/bin/python}"
[[ -x "$PYTHON_BIN" ]] || {
    echo "[ERROR] Python environment is unavailable: $PYTHON_BIN" >&2
    exit 1
}
export PATH="$FLAME_ENV/bin:/usr/bin:/bin"
export PYTHON_BIN PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-$FLAME_ENV}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/g2_olddata_kd_9run_20260808/01_common_kd_init/code_e8_to_e16_wiki_kd_step600}"
SOURCE_STEP="${SOURCE_STEP:-600}"
GT_ROOT="${GT_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/code_old_like_gt_l2_l9_top1_raw_20260812}"
GT_CONFIG_SHA256="${GT_CONFIG_SHA256:-c8054ecccca55f0892955cb81065ae1792f7c99c86b58300332ef188e0ccfd9d}"
GT_METADATA_SHA256="${GT_METADATA_SHA256:-4389ff76097121bbcdc63a6811058f93cf06dcdbfe628974003692d63f1c263f}"
GT_SCHEMA="${GT_SCHEMA:-old_like_gt_l2_l9_all_code_top1_raw_v1}"
GT_SELECTED_COUNT="${GT_SELECTED_COUNT:-2658787}"
GT_TOTAL_SAMPLES="${GT_TOTAL_SAMPLES:-4147200}"
GT_SEQUENCE_LENGTH="${GT_SEQUENCE_LENGTH:-512}"
OLD_LIKE_REPLAY_UNIT="${MOE_JOINT_REPLAY_OLD_LIKE_UNIT:-positive_sequence}"
case "$OLD_LIKE_REPLAY_UNIT" in
    positive_sequence)
        DEFAULT_GT_REPLAY_COUNT=798138
        DEFAULT_GT_REPLAY_SAMPLE_FILE=replay_positive_sample_ids.npy
        DEFAULT_GT_REPLAY_SAMPLE_SHA256=b2df3ca918775abcea5e1d1d17ea5842904fb2f2305ac8289a433c5508ffcd37
        DEFAULT_GT_REPLAY_POSITION_FILE=
        DEFAULT_GT_REPLAY_POSITION_SHA256=
        DEFAULT_GT_REPLAY_METADATA_FILE=replay_subset_metadata.json
        DEFAULT_GT_REPLAY_METADATA_SHA256=ad76997b2c700f57d6e8ac9312ef8ae7327b93920c2253c552e4c4224e787fe2
        DEFAULT_STUDY_NAME=code_old_like_gt_token_masked_20pct_router_replay_20260812
        REPLAY_RUN_TAG=token-masked-20pct
        REPLAY_UNIT_DESCRIPTION="GT-positive contextual sequences (all GT positions masked in)"
        ;;
    token_occurrence)
        DEFAULT_GT_REPLAY_COUNT=2658787
        DEFAULT_GT_REPLAY_SAMPLE_FILE=replay_occurrence_sample_ids.npy
        DEFAULT_GT_REPLAY_SAMPLE_SHA256=7b154ceb531410527dcf42717c8ba0844f5f43103e28d909e5c53833a0fa5077
        DEFAULT_GT_REPLAY_POSITION_FILE=replay_occurrence_positions.npy
        DEFAULT_GT_REPLAY_POSITION_SHA256=7cde3898831350ba95bc4024d1e238cc49979805f75ecaf78983c432691f08f9
        DEFAULT_GT_REPLAY_METADATA_FILE=replay_occurrence_metadata.json
        DEFAULT_GT_REPLAY_METADATA_SHA256=576852df39fb064263a296e6c8751e494e4620a6459a7f6e53c40f7a451acf07
        DEFAULT_STUDY_NAME=code_old_like_gt_token_occurrence_router_replay_20260812
        REPLAY_RUN_TAG=token-occurrence
        REPLAY_UNIT_DESCRIPTION="contextual GT token occurrences (exactly one position per replay item)"
        ;;
    *) echo "[ERROR] unsupported old-like replay unit: $OLD_LIKE_REPLAY_UNIT" >&2; exit 1 ;;
esac
GT_REPLAY_SUBSET_COUNT="${GT_REPLAY_SUBSET_COUNT:-$DEFAULT_GT_REPLAY_COUNT}"
GT_REPLAY_SUBSET_SHA256="${GT_REPLAY_SUBSET_SHA256:-$DEFAULT_GT_REPLAY_SAMPLE_SHA256}"
GT_REPLAY_POSITION_SHA256="${GT_REPLAY_POSITION_SHA256:-$DEFAULT_GT_REPLAY_POSITION_SHA256}"
GT_REPLAY_SUBSET_METADATA_SHA256="${GT_REPLAY_SUBSET_METADATA_SHA256:-$DEFAULT_GT_REPLAY_METADATA_SHA256}"
GT_REPLAY_SAMPLE_FILE="${GT_REPLAY_SAMPLE_FILE:-$DEFAULT_GT_REPLAY_SAMPLE_FILE}"
GT_REPLAY_POSITION_FILE="${GT_REPLAY_POSITION_FILE:-$DEFAULT_GT_REPLAY_POSITION_FILE}"
GT_REPLAY_METADATA_FILE="${GT_REPLAY_METADATA_FILE:-$DEFAULT_GT_REPLAY_METADATA_FILE}"

FLAME_DATA_ROOT="${FLAME_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup}"
CODE_TRAIN_DIR="${CODE_TRAIN_DIR:-$FLAME_DATA_ROOT/code/train}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6}"

STUDY_ROOT="${STUDY_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-runs/$DEFAULT_STUDY_NAME}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$STUDY_ROOT/checkpoints}"
LOG_ROOT="${CHAIN_LOG_DIR:-$STUDY_ROOT/logs/lm_then_hidden_mse_4gpu}"
LOCAL_BASE="${LOCAL_BASE:-$STUDY_ROOT/local}"
LOCAL_SSD_ROOT="${LOCAL_SSD_ROOT:-$STUDY_ROOT/scratch}"

TRAIN_ITERS="${TRAIN_ITERS:-1800}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2304}"
HIDDEN_MSE_COEFF="${OLD_HIDDEN_MSE_COEFF:-10}"
HIDDEN_MSE_LAYERS="${OLD_HIDDEN_MSE_LAYERS:-2,3,4,5,6,7,8,9}"
HIDDEN_KL_COEFF="${OLD_HIDDEN_KL_COEFF:-1}"
HIDDEN_KL_LAYERS="${OLD_HIDDEN_KL_LAYERS:-2,3,4,5,6,7,8,9}"
VOCAB_KL_COEFF="${OLD_MODEL_KL_COEFF:-1}"
TARGET_TRAIN_FRACTION="${OLD_LIKE_TARGET_TRAIN_FRACTION:-0.20}"
GT_POSITIVE_SAMPLE_COUNT="${GT_POSITIVE_SAMPLE_COUNT:-798138}"
GT_FULL_TRAIN_TOKEN_COUNT="${GT_FULL_TRAIN_TOKEN_COUNT:-2123366400}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
LM_MASTER_PORT="${LM_MASTER_PORT:-34121}"
MSE_MASTER_PORT="${MSE_MASTER_PORT:-34122}"
HIDDEN_KL_MASTER_PORT="${HIDDEN_KL_MASTER_PORT:-34123}"
VOCAB_KL_MASTER_PORT="${VOCAB_KL_MASTER_PORT:-34124}"

LM_RUN_ID="${LM_RUN_ID:-g2-code-oldlike-${REPLAY_RUN_TAG}-all8-top1-router-lm-mb48-gbs2304-s1800}"
MSE_RUN_ID="${MSE_RUN_ID:-g2-code-oldlike-${REPLAY_RUN_TAG}-all8-top1-router-hiddenmse-c10-l2to9-mb48-gbs2304-s1800}"
HIDDEN_KL_RUN_ID="${HIDDEN_KL_RUN_ID:-g2-code-oldlike-${REPLAY_RUN_TAG}-all8-top1-router-hiddenkl-c1-l2to9-20pct-mb48-gbs2304-s1800}"
VOCAB_KL_RUN_ID="${VOCAB_KL_RUN_ID:-g2-code-oldlike-${REPLAY_RUN_TAG}-all8-top1-router-vocabkl-c1-20pct-mb48-gbs2304-s1800}"
LM_OUTPUT="${LM_OUTPUT:-$CHECKPOINT_ROOT/01_old_like_router_lm/$LM_RUN_ID}"
MSE_OUTPUT="${MSE_OUTPUT:-$CHECKPOINT_ROOT/02_old_like_router_hidden_mse/$MSE_RUN_ID}"
HIDDEN_KL_OUTPUT="${HIDDEN_KL_OUTPUT:-$CHECKPOINT_ROOT/03_old_like_router_hidden_kl/$HIDDEN_KL_RUN_ID}"
VOCAB_KL_OUTPUT="${VOCAB_KL_OUTPUT:-$CHECKPOINT_ROOT/04_old_like_router_vocab_kl/$VOCAB_KL_RUN_ID}"
LM_LOG="${LM_LOG:-$LOG_ROOT/01_old_like_router_lm.log}"
MSE_LOG="${MSE_LOG:-$LOG_ROOT/02_old_like_router_hidden_mse.log}"
HIDDEN_KL_LOG="${HIDDEN_KL_LOG:-$LOG_ROOT/03_old_like_router_hidden_kl.log}"
VOCAB_KL_LOG="${VOCAB_KL_LOG:-$LOG_ROOT/04_old_like_router_vocab_kl.log}"
LOCK_FILE="${CHAIN_LOCK_FILE:-$STUDY_ROOT/.lm_then_hidden_mse_4gpu.lock}"
STATUS_FILE="${CHAIN_STATUS_FILE:-$LOG_ROOT/status.tsv}"

LM_ENTRY_SCRIPT="${LM_ENTRY_SCRIPT:-$SCRIPT_DIR/run_g2_ffn_only_code_wiki_joint_lm_allrouter_mha.sh}"
MSE_ENTRY_SCRIPT="${MSE_ENTRY_SCRIPT:-$SCRIPT_DIR/run_g2_ffn_only_code_wiki_joint_old_data_hidden_mse_allrouter_mha.sh}"
HIDDEN_KL_ENTRY_SCRIPT="${HIDDEN_KL_ENTRY_SCRIPT:-$SCRIPT_DIR/run_g2_ffn_only_code_wiki_joint_old_data_hidden_kl_allrouter_mha.sh}"
VOCAB_KL_ENTRY_SCRIPT="${VOCAB_KL_ENTRY_SCRIPT:-$SCRIPT_DIR/run_g2_ffn_only_code_wiki_joint_old_data_kd_allrouter_mha.sh}"
GT_VALIDATOR="${GT_VALIDATOR:-$REPO_ROOT/scripts/analysis/validate_old_like_gt_l2_l9_top1.py}"

PLAN_ONLY="${PLAN_ONLY:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

record() {
    local state="$1"
    shift
    printf '%s\t%s\t%s\n' "$(date -Is)" "$state" "$*" | tee -a "$STATUS_FILE"
}

checkpoint_payload_at() {
    local root="$1" expected="$2" tracker iteration_dir
    local -a shards
    tracker="$root/latest_checkpointed_iteration.txt"
    iteration_dir="$root/iter_$(printf '%07d' "$expected")"
    [[ -s "$tracker" ]] || return 1
    [[ "$(tr -d '[:space:]' < "$tracker")" == "$expected" ]] || return 1
    [[ -d "$iteration_dir" ]] || return 1
    [[ -s "$iteration_dir/.metadata" ]] || return 1
    [[ -s "$iteration_dir/common.pt" ]] || return 1
    [[ -s "$iteration_dir/metadata.json" ]] || return 1
    shards=("$iteration_dir"/__*.distcp)
    [[ "${#shards[@]}" -gt 0 ]] || return 1
    local shard
    for shard in "${shards[@]}"; do
        [[ -s "$shard" ]] || return 1
    done
}

stage_manifest_matches() {
    local output="$1" objective="$2"
    "$PYTHON_BIN" - "$output/stage_completion_manifest.json" "$objective" \
        "$SOURCE_CHECKPOINT" "$SOURCE_STEP" "$GT_ROOT" "$GT_CONFIG_SHA256" \
        "$GT_METADATA_SHA256" "$GT_SELECTED_COUNT" "$TRAIN_ITERS" \
        "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$HIDDEN_MSE_COEFF" \
        "$HIDDEN_MSE_LAYERS" "$CODE_TRAIN_DIR" "$GT_REPLAY_SUBSET_COUNT" \
        "$GT_REPLAY_SUBSET_SHA256" "$OLD_LIKE_REPLAY_UNIT" \
        "$GT_REPLAY_POSITION_SHA256" "$HIDDEN_KL_COEFF" "$HIDDEN_KL_LAYERS" \
        "$VOCAB_KL_COEFF" "$TARGET_TRAIN_FRACTION" <<'PY'
import json
import sys
from pathlib import Path

(manifest_path, objective, source, source_step, gt_root, gt_config_sha,
 gt_metadata_sha, selected_count, train_iters, micro_batch, global_batch,
 mse_coeff, mse_layers, code_train_dir, subset_count, subset_sha, replay_unit,
 position_sha, hidden_kl_coeff, hidden_kl_layers, vocab_kl_coeff,
 target_fraction) = sys.argv[1:]
path = Path(manifest_path)
if not path.is_file():
    raise SystemExit(1)
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
expected = {
    "schema": "old_like_router_replay_stage_completion_v1",
    "complete": True,
    "objective": objective,
    "source_checkpoint": str(Path(source).resolve()),
    "source_step": int(source_step),
    "gt_root": str(Path(gt_root).resolve()),
    "gt_config_sha256": gt_config_sha,
    "gt_metadata_sha256": gt_metadata_sha,
    "gt_selected_count": int(selected_count),
    "gt_replay_subset_count": int(subset_count),
    "gt_replay_subset_sha256": subset_sha,
    "gt_replay_unit": replay_unit,
    "gt_replay_position_sha256": position_sha,
    "train_iters": int(train_iters),
    "micro_batch_size": int(micro_batch),
    "global_batch_size": int(global_batch),
    "code_train_dir": str(Path(code_train_dir).resolve()),
    "calculate_per_token_loss": False,
    "hidden_mse_coeff": float(mse_coeff) if objective == "old_like_router_hidden_mse" else 0.0,
    "hidden_mse_layers": mse_layers if objective == "old_like_router_hidden_mse" else "",
    "hidden_kl_coeff": float(hidden_kl_coeff) if objective == "old_like_router_hidden_kl" else 0.0,
    "hidden_kl_layers": hidden_kl_layers if objective == "old_like_router_hidden_kl" else "",
    "vocab_kl_coeff": float(vocab_kl_coeff) if objective == "old_like_router_vocab_kl" else 0.0,
    "target_train_fraction": float(target_fraction),
}
if any(value.get(key) != wanted for key, wanted in expected.items()):
    raise SystemExit(1)
PY
}

run_metadata_matches() {
    local output="$1" objective="$2"
    "$PYTHON_BIN" - "$output/logs/run_metadata.json" "$objective" \
        "$SOURCE_CHECKPOINT" "$GT_ROOT" "$TRAIN_ITERS" "$MICRO_BATCH_SIZE" \
        "$GLOBAL_BATCH_SIZE" "$HIDDEN_MSE_COEFF" "$HIDDEN_MSE_LAYERS" \
        "$OLD_LIKE_REPLAY_UNIT" "$GT_REPLAY_SUBSET_COUNT" \
        "$GT_REPLAY_SUBSET_SHA256" "$HIDDEN_KL_COEFF" "$HIDDEN_KL_LAYERS" \
        "$VOCAB_KL_COEFF" "$TARGET_TRAIN_FRACTION" "$GT_POSITIVE_SAMPLE_COUNT" \
        "$GT_FULL_TRAIN_TOKEN_COUNT" <<'PY'
import json
import sys
from pathlib import Path

(metadata_path, objective, source, gt_root, train_iters, micro_batch,
 global_batch, mse_coeff, mse_layers, replay_unit, subset_count, subset_sha,
 hidden_kl_coeff, hidden_kl_layers, vocab_kl_coeff, target_fraction,
 positive_count, full_token_count) = sys.argv[1:]
path = Path(metadata_path)
if not path.is_file():
    raise SystemExit(1)
try:
    value = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
expected = {
    "source_weights_dir": str(Path(source).resolve()),
    "source_num_experts": 8,
    "target_num_experts": 16,
    "train_iters": int(train_iters),
    "micro_batch_size": int(micro_batch),
    "global_batch_size": int(global_batch),
    "train_new_experts_and_router_only": True,
    "moe_new_expert_lr_ramp_steps": 0,
    "moe_router_lr_multiplier": 1.0,
    "lr": 3.0e-4,
    "min_lr": 3.0e-5,
    "lr_decay_style": "WSD",
    "lr_decay_iters": int(train_iters),
    "lr_wsd_decay_iters": 180,
    "lr_warmup_fraction": 0.01,
    "moe_joint_replay_lm": True,
    "moe_joint_replay_old_data_kd": objective == "old_like_router_vocab_kl",
    "moe_joint_replay_old_data_hidden_kl": objective == "old_like_router_hidden_kl",
    "moe_joint_replay_old_data_hidden_mse": objective == "old_like_router_hidden_mse",
    "moe_joint_replay_old_like_gt_path": str(Path(gt_root).resolve()),
    "moe_joint_replay_old_like_unit": replay_unit,
    "moe_joint_replay_old_like_subset_count": int(subset_count),
    "moe_joint_replay_old_like_subset_sha256": subset_sha,
    "moe_joint_replay_old_like_target_train_fraction": float(target_fraction),
    "moe_joint_replay_old_like_selected_token_count": 2658787,
    "moe_joint_replay_old_like_positive_sample_count": int(positive_count),
    "moe_joint_replay_old_like_full_train_token_count": int(full_token_count),
    "old_model_kl_enabled": objective != "old_like_router_lm",
}
if objective == "old_like_router_hidden_kl":
    expected.update({
        "old_hidden_kl_coeff": float(hidden_kl_coeff),
        "old_hidden_kl_layers": hidden_kl_layers,
        "old_model_kl_coeff": 0.0,
    })
if objective == "old_like_router_vocab_kl":
    expected.update({"old_model_kl_coeff": float(vocab_kl_coeff)})
if objective == "old_like_router_hidden_mse":
    expected.update({
        "old_hidden_mse_coeff": float(mse_coeff),
        "old_hidden_mse_layers": mse_layers,
        "old_model_kl_coeff": 0.0,
    })
if any(value.get(key) != wanted for key, wanted in expected.items()):
    raise SystemExit(1)
PY
}

verified_stage_complete() {
    local output="$1" objective="$2"
    checkpoint_payload_at "$output" "$TRAIN_ITERS" &&
        run_metadata_matches "$output" "$objective" &&
        stage_manifest_matches "$output" "$objective"
}

output_is_empty_or_absent() {
    local output="$1"
    [[ ! -e "$output" ]] && return 0
    [[ -d "$output" ]] || return 1
    [[ -z "$(find "$output" -mindepth 1 -print -quit 2>/dev/null)" ]]
}

write_stage_manifest() {
    local output="$1" objective="$2"
    "$PYTHON_BIN" - "$output/stage_completion_manifest.json" "$objective" \
        "$SOURCE_CHECKPOINT" "$SOURCE_STEP" "$GT_ROOT" "$GT_CONFIG_SHA256" \
        "$GT_METADATA_SHA256" "$GT_SELECTED_COUNT" "$TRAIN_ITERS" \
        "$MICRO_BATCH_SIZE" "$GLOBAL_BATCH_SIZE" "$HIDDEN_MSE_COEFF" \
        "$HIDDEN_MSE_LAYERS" "$CUDA_VISIBLE_DEVICES" "$CODE_TRAIN_DIR" \
        "$GT_REPLAY_SUBSET_COUNT" "$GT_REPLAY_SUBSET_SHA256" \
        "$OLD_LIKE_REPLAY_UNIT" "$GT_REPLAY_POSITION_SHA256" \
        "$HIDDEN_KL_COEFF" "$HIDDEN_KL_LAYERS" "$VOCAB_KL_COEFF" \
        "$TARGET_TRAIN_FRACTION" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

(manifest_path, objective, source, source_step, gt_root, gt_config_sha,
 gt_metadata_sha, selected_count, train_iters, micro_batch, global_batch,
 mse_coeff, mse_layers, devices, code_train_dir, subset_count, subset_sha,
 replay_unit, position_sha, hidden_kl_coeff, hidden_kl_layers, vocab_kl_coeff,
 target_fraction) = sys.argv[1:]
path = Path(manifest_path)
payload = {
    "schema": "old_like_router_replay_stage_completion_v1",
    "complete": True,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "objective": objective,
    "source_checkpoint": str(Path(source).resolve()),
    "source_step": int(source_step),
    "gt_root": str(Path(gt_root).resolve()),
    "gt_config_sha256": gt_config_sha,
    "gt_metadata_sha256": gt_metadata_sha,
    "gt_selected_count": int(selected_count),
    "gt_replay_subset_count": int(subset_count),
    "gt_replay_subset_sha256": subset_sha,
    "gt_replay_unit": replay_unit,
    "gt_replay_position_sha256": position_sha,
    "train_iters": int(train_iters),
    "micro_batch_size": int(micro_batch),
    "global_batch_size": int(global_batch),
    "code_train_dir": str(Path(code_train_dir).resolve()),
    "calculate_per_token_loss": False,
    "cuda_visible_devices": devices,
    "hidden_mse_coeff": float(mse_coeff) if objective == "old_like_router_hidden_mse" else 0.0,
    "hidden_mse_layers": mse_layers if objective == "old_like_router_hidden_mse" else "",
    "hidden_kl_coeff": float(hidden_kl_coeff) if objective == "old_like_router_hidden_kl" else 0.0,
    "hidden_kl_layers": hidden_kl_layers if objective == "old_like_router_hidden_kl" else "",
    "vocab_kl_coeff": float(vocab_kl_coeff) if objective == "old_like_router_vocab_kl" else 0.0,
    "target_train_fraction": float(target_fraction),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY
}

preflight_gt() {
    local actual_metadata_sha actual_subset_sha actual_subset_metadata_sha actual_position_sha
    [[ -s "$GT_ROOT/metadata.json" ]] || die "GT metadata is missing: $GT_ROOT/metadata.json"
    actual_metadata_sha="$(sha256sum "$GT_ROOT/metadata.json" | awk '{print $1}')"
    [[ "$actual_metadata_sha" == "$GT_METADATA_SHA256" ]] || {
        die "GT metadata SHA mismatch: expected=$GT_METADATA_SHA256 actual=$actual_metadata_sha"
    }
    [[ -s "$GT_ROOT/$GT_REPLAY_SAMPLE_FILE" ]] || die "GT replay subset is missing: $GT_REPLAY_SAMPLE_FILE"
    [[ -s "$GT_ROOT/$GT_REPLAY_METADATA_FILE" ]] || die "GT replay subset metadata is missing: $GT_REPLAY_METADATA_FILE"
    actual_subset_sha="$(sha256sum "$GT_ROOT/$GT_REPLAY_SAMPLE_FILE" | awk '{print $1}')"
    actual_subset_metadata_sha="$(sha256sum "$GT_ROOT/$GT_REPLAY_METADATA_FILE" | awk '{print $1}')"
    [[ "$actual_subset_sha" == "$GT_REPLAY_SUBSET_SHA256" ]] || {
        die "GT replay subset SHA mismatch: expected=$GT_REPLAY_SUBSET_SHA256 actual=$actual_subset_sha"
    }
    [[ "$actual_subset_metadata_sha" == "$GT_REPLAY_SUBSET_METADATA_SHA256" ]] || {
        die "GT replay subset metadata SHA mismatch: expected=$GT_REPLAY_SUBSET_METADATA_SHA256 actual=$actual_subset_metadata_sha"
    }
    if [[ -n "$GT_REPLAY_POSITION_FILE" ]]; then
        [[ -s "$GT_ROOT/$GT_REPLAY_POSITION_FILE" ]] || die "GT replay positions are missing: $GT_REPLAY_POSITION_FILE"
        actual_position_sha="$(sha256sum "$GT_ROOT/$GT_REPLAY_POSITION_FILE" | awk '{print $1}')"
        [[ "$actual_position_sha" == "$GT_REPLAY_POSITION_SHA256" ]] || {
            die "GT replay position SHA mismatch: expected=$GT_REPLAY_POSITION_SHA256 actual=$actual_position_sha"
        }
    fi
    "$PYTHON_BIN" - "$GT_ROOT/metadata.json" "$GT_SCHEMA" "$GT_CONFIG_SHA256" \
        "$GT_SELECTED_COUNT" "$GT_TOTAL_SAMPLES" "$GT_SEQUENCE_LENGTH" \
        "$TRAIN_ITERS" "$GLOBAL_BATCH_SIZE" \
        "$CODE_TRAIN_DIR/train_text_document" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(path, schema, config_sha, selected_count, total_samples, sequence_length,
 train_iters, global_batch, code_prefix) = sys.argv[1:]
value = json.loads(Path(path).read_text(encoding="utf-8"))
identity = value.get("dataset_identity") or {}
idx_path = Path(code_prefix + ".idx")
bin_path = Path(code_prefix + ".bin")
idx_sha = hashlib.sha256(idx_path.read_bytes()).hexdigest()
checks = {
    "complete": value.get("complete") is True,
    "schema": value.get("schema") == schema,
    "config_sha256": value.get("config_sha256") == config_sha,
    "layers": value.get("layers") == list(range(2, 10)),
    "comparison": value.get("comparison") == ">=",
    "selected_count": value.get("selected_count") == int(selected_count),
    "total_samples": value.get("total_samples") == int(total_samples),
    "sequence_length": value.get("sequence_length") == int(sequence_length),
    "training_sample_count": int(train_iters) * int(global_batch) == int(total_samples),
    "dataset_seed": identity.get("seed") == 1234,
    "dataset_split": identity.get("split") == "100,0,0",
    "dataset_prefix": os.path.realpath(identity.get("paired_extraction_dataset_prefix", "")) == os.path.realpath(code_prefix),
    "dataset_idx_bytes": identity.get("idx_bytes") == idx_path.stat().st_size,
    "dataset_bin_bytes": identity.get("bin_bytes") == bin_path.stat().st_size,
    "dataset_idx_sha256": identity.get("idx_sha256") == idx_sha,
}
failed = [name for name, okay in checks.items() if not okay]
if failed:
    raise SystemExit("GT metadata preflight failed: " + ", ".join(failed))
PY
    "$PYTHON_BIN" - "$GT_ROOT/$GT_REPLAY_METADATA_FILE" \
        "$OLD_LIKE_REPLAY_UNIT" \
        "$GT_REPLAY_SUBSET_COUNT" "$GT_TOTAL_SAMPLES" "$GT_SELECTED_COUNT" \
        "$GT_REPLAY_SUBSET_SHA256" "$GT_REPLAY_POSITION_SHA256" <<'PY'
import json
import sys
from pathlib import Path

path, unit, subset_count, total_samples, selected_count, subset_sha, position_sha = sys.argv[1:]
value = json.loads(Path(path).read_text(encoding="utf-8"))
checks = {
    "complete": value.get("complete") is True,
    "source_total_samples": value.get("source_total_samples") == int(total_samples),
    "ordering_seed": value.get("ordering_seed") == 1234,
    "virtual_samples": value.get("virtual_samples_per_1800_step_run") == int(total_samples),
}
if unit == "positive_sequence":
    checks.update({
        "schema": value.get("schema") == "old_like_gt_positive_sample_subset_v1",
        "positive_sample_count": value.get("positive_sample_count") == int(subset_count),
        "source_selected_token_count": value.get("source_selected_token_count") == int(selected_count),
        "subset_sha256": value.get("subset_sha256") == subset_sha,
    })
elif unit == "token_occurrence":
    checks.update({
        "schema": value.get("schema") == "old_like_gt_token_occurrence_subset_v1",
        "occurrence_count": value.get("occurrence_count") == int(subset_count) == int(selected_count),
        "sample_sha256": value.get("sample_sha256") == subset_sha,
        "position_sha256": value.get("position_sha256") == position_sha,
        "one_position": value.get("direct_supervised_positions_per_replay_item") == 1,
    })
else:
    checks["unit"] = False
failed = [name for name, okay in checks.items() if not okay]
if failed:
    raise SystemExit("GT replay subset preflight failed: " + ", ".join(failed))
PY
    "$PYTHON_BIN" "$GT_VALIDATOR" --gt-root "$GT_ROOT" --workers 8 >/dev/null
    echo "[PREFLIGHT] GT masks and fixed $OLD_LIKE_REPLAY_UNIT replay subset verified"
}

preflight_source_and_data() {
    [[ "$SOURCE_STEP" == "600" ]] || die "this comparison is pinned to source step 600, got $SOURCE_STEP"
    checkpoint_payload_at "$SOURCE_CHECKPOINT" "$SOURCE_STEP" || {
        die "source checkpoint payload is not complete at step $SOURCE_STEP: $SOURCE_CHECKPOINT"
    }
    [[ -s "$CODE_TRAIN_DIR/train_text_document.bin" ]] || die "Code train .bin missing: $CODE_TRAIN_DIR"
    [[ -s "$CODE_TRAIN_DIR/train_text_document.idx" ]] || die "Code train .idx missing: $CODE_TRAIN_DIR"
    "$PYTHON_BIN" - "$GT_ROOT/metadata.json" \
        "$CODE_TRAIN_DIR/train_text_document.idx" \
        "$CODE_TRAIN_DIR/train_text_document.bin" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

metadata_path, idx_name, bin_name = map(Path, sys.argv[1:])
identity = json.loads(metadata_path.read_text(encoding="utf-8"))["dataset_identity"]
digest = hashlib.sha256()
with idx_name.open("rb") as handle:
    while block := handle.read(8 << 20):
        digest.update(block)
checks = {
    "idx_sha256": digest.hexdigest() == identity["idx_sha256"],
    "idx_bytes": idx_name.stat().st_size == identity["idx_bytes"],
    "bin_bytes": bin_name.stat().st_size == identity["bin_bytes"],
    "seed": identity["seed"] == 1234,
    "split": identity["split"] == "100,0,0",
}
failed = [name for name, okay in checks.items() if not okay]
if failed:
    raise SystemExit("Code dataset/GT identity mismatch: " + ", ".join(failed))
PY
    [[ -s "$TOKENIZER_MODEL/tokenizer.json" ]] || die "tokenizer.json missing: $TOKENIZER_MODEL"
    [[ -x "$LM_ENTRY_SCRIPT" || -f "$LM_ENTRY_SCRIPT" ]] || die "LM entry script missing: $LM_ENTRY_SCRIPT"
    [[ -x "$MSE_ENTRY_SCRIPT" || -f "$MSE_ENTRY_SCRIPT" ]] || die "MSE entry script missing: $MSE_ENTRY_SCRIPT"
    [[ -x "$HIDDEN_KL_ENTRY_SCRIPT" || -f "$HIDDEN_KL_ENTRY_SCRIPT" ]] || die "hidden-KL entry script missing: $HIDDEN_KL_ENTRY_SCRIPT"
    [[ -x "$VOCAB_KL_ENTRY_SCRIPT" || -f "$VOCAB_KL_ENTRY_SCRIPT" ]] || die "vocab-KL entry script missing: $VOCAB_KL_ENTRY_SCRIPT"
    [[ -f "$GT_VALIDATOR" ]] || die "GT validator missing: $GT_VALIDATOR"
    [[ "$HIDDEN_MSE_LAYERS" == "2,3,4,5,6,7,8,9" ]] || {
        die "hidden-MSE must include residual layer outputs 2--9, including final Layer 9"
    }
    [[ "$HIDDEN_KL_LAYERS" == "2,3,4,5,6,7,8,9" ]] || {
        die "hidden-KL must include residual layer outputs 2--9, including final Layer 9"
    }
    [[ "$TRAIN_ITERS" == "1800" ]] || die "comparison is pinned to 1800 steps, got $TRAIN_ITERS"
    [[ "$MICRO_BATCH_SIZE" == "48" ]] || die "comparison is pinned to MB=48, got $MICRO_BATCH_SIZE"
    [[ "$GLOBAL_BATCH_SIZE" == "2304" ]] || die "comparison is pinned to GBS=2304, got $GLOBAL_BATCH_SIZE"
    "$PYTHON_BIN" - "$HIDDEN_MSE_COEFF" <<'PY'
import sys
if float(sys.argv[1]) != 10.0:
    raise SystemExit(f"hidden-MSE comparison is pinned to coefficient 10, got {sys.argv[1]}")
PY
    "$PYTHON_BIN" - "$GT_SELECTED_COUNT" "$GT_FULL_TRAIN_TOKEN_COUNT" \
        "$TARGET_TRAIN_FRACTION" "$GT_POSITIVE_SAMPLE_COUNT" "$GLOBAL_BATCH_SIZE" <<'PY'
import sys
selected, full, target, positive, gbs = map(float, sys.argv[1:])
epochs = target * full / selected
batches = round(target * full * positive / (selected * gbs))
actual = batches * gbs * (selected / positive) / full
if abs(target - 0.20) > 1e-12 or batches != 55331 or abs(actual - 0.20) > 1e-5:
    raise SystemExit(
        f"20pct GT budget mismatch: epochs={epochs} batches={batches} actual={actual}"
    )
print(f"[BUDGET] GT epochs={epochs:.6f} replay_global_batches={batches} expected_fraction={actual:.9f}")
PY
    echo "[PREFLIGHT] source checkpoint, Code data, tokenizer, and entry scripts verified"
}

preflight_outputs() {
    local output objective label
    while IFS='|' read -r output objective label; do
        if verified_stage_complete "$output" "$objective"; then
            echo "[PREFLIGHT] $label is already verified complete and may be skipped: $output"
        elif output_is_empty_or_absent "$output"; then
            echo "[PREFLIGHT] $label output is fresh: $output"
        else
            die "$label output exists but is not a verified matching completion; refusing overwrite/resume: $output"
        fi
    done <<EOF
$LM_OUTPUT|old_like_router_lm|old-like LM
$MSE_OUTPUT|old_like_router_hidden_mse|old-like hidden-MSE
$HIDDEN_KL_OUTPUT|old_like_router_hidden_kl|old-like hidden-KL
$VOCAB_KL_OUTPUT|old_like_router_vocab_kl|old-like vocab-KL
EOF
}

port_is_free() {
    "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
PY
}

preflight_gpus_and_ports() {
    [[ "$CUDA_VISIBLE_DEVICES" == "0,1,2,3" ]] || {
        die "this launcher is pinned to physical GPUs 0,1,2,3; got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    }
    [[ "$NPROC_PER_NODE" == "4" ]] || die "NPROC_PER_NODE must be 4, got $NPROC_PER_NODE"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    local gpu descriptor applications
    for gpu in 0 1 2 3; do
        descriptor="$(nvidia-smi -i "$gpu" --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null)" || {
            die "cannot query physical GPU $gpu"
        }
        [[ "$descriptor" == "$gpu,"* ]] || die "physical GPU index mismatch for requested GPU $gpu: $descriptor"
        applications="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)" || {
            die "cannot query compute processes on GPU $gpu"
        }
        applications="$(printf '%s' "$applications" | sed '/^[[:space:]]*$/d; /No running processes found/d')"
        [[ -z "$applications" ]] || {
            echo "[BUSY] GPU $gpu has compute process(es):" >&2
            printf '%s\n' "$applications" >&2
            die "GPUs 0--3 must be idle; this launcher will not terminate existing processes"
        }
    done
    if ! verified_stage_complete "$LM_OUTPUT" old_like_router_lm; then
        port_is_free "$LM_MASTER_PORT" || die "LM master port is occupied: $LM_MASTER_PORT"
    fi
    if ! verified_stage_complete "$MSE_OUTPUT" old_like_router_hidden_mse; then
        [[ "$MSE_MASTER_PORT" != "$LM_MASTER_PORT" ]] || die "LM and MSE ports must be unique"
        port_is_free "$MSE_MASTER_PORT" || die "MSE master port is occupied: $MSE_MASTER_PORT"
    fi
    if ! verified_stage_complete "$HIDDEN_KL_OUTPUT" old_like_router_hidden_kl; then
        port_is_free "$HIDDEN_KL_MASTER_PORT" || die "hidden-KL master port is occupied: $HIDDEN_KL_MASTER_PORT"
    fi
    if ! verified_stage_complete "$VOCAB_KL_OUTPUT" old_like_router_vocab_kl; then
        port_is_free "$VOCAB_KL_MASTER_PORT" || die "vocab-KL master port is occupied: $VOCAB_KL_MASTER_PORT"
    fi
    echo "[PREFLIGHT] physical GPUs 0--3 are idle; required master ports are free"
}

run_stage() {
    local label="$1" objective="$2" output="$3" log="$4"
    shift 4
    if verified_stage_complete "$output" "$objective"; then
        record SKIP "$label verified complete at step $TRAIN_ITERS: $output"
        return
    fi
    output_is_empty_or_absent "$output" || {
        die "$label output became nonempty before launch; refusing overwrite: $output"
    }
    record START "$label -> $output"
    set +e
    "$@" 2>&1 | tee -a "$log"
    local -a pipeline_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${pipeline_status[0]}" -ne 0 || "${pipeline_status[1]}" -ne 0 ]]; then
        record ERROR "$label failed command_rc=${pipeline_status[0]} tee_rc=${pipeline_status[1]} log=$log"
        exit "$([[ "${pipeline_status[0]}" -ne 0 ]] && echo "${pipeline_status[0]}" || echo "${pipeline_status[1]}")"
    fi
    checkpoint_payload_at "$output" "$TRAIN_ITERS" || {
        record ERROR "$label returned successfully but has no complete step-$TRAIN_ITERS checkpoint: $output"
        exit 1
    }
    run_metadata_matches "$output" "$objective" || {
        record ERROR "$label run metadata does not match the requested objective/configuration: $output"
        exit 1
    }
    write_stage_manifest "$output" "$objective"
    verified_stage_complete "$output" "$objective" || {
        record ERROR "$label completion manifest verification failed: $output"
        exit 1
    }
    record DONE "$label verified at step $TRAIN_ITERS: $output"
}

cat <<PLAN
[EXPERIMENT] old-like stability pseudo-GT router-replay comparison
[GT] GT=1 iff the contextual Code occurrence is in Code cosine top-1% at every residual-included Layer 2--9
[GT CLAIM] Wiki-like stability pseudo-GT, not semantic Wiki-domain ground truth
[GT ROOT] $GT_ROOT
[GT COUNT] $GT_SELECTED_COUNT / $((GT_TOTAL_SAMPLES * GT_SEQUENCE_LENGTH)) occurrences (packed-mask valid count is metadata-authoritative)
[REPLAY TOKENS] only GT positions receive direct replay loss; non-GT positions are context only
[BUDGET] target=$TARGET_TRAIN_FRACTION of full train tokens; GT occurrence epochs=$("$PYTHON_BIN" -c "print($TARGET_TRAIN_FRACTION * $GT_FULL_TRAIN_TOKEN_COUNT / $GT_SELECTED_COUNT)"); 55,331 replay global batches
[SOURCE] post-expansion KD-init model at step $SOURCE_STEP: $SOURCE_CHECKPOINT
[COMMON TRAIN] Code, steps=$TRAIN_ITERS, MB=$MICRO_BATCH_SIZE, GBS=$GLOBAL_BATCH_SIZE, GPUs=$CUDA_VISIBLE_DEVICES
[TOKEN ACCOUNTING] default per-microbatch selected-token mean; --calculate-per-token-loss stays disabled because joint replay finalizes gradients separately
[ROUTING] natural routing; primary full Code uses GT complement; independent repeated GT-positive replay subset updates router only
[STAGE 1] GT=1 same-position LM replay -> $LM_OUTPUT
[STAGE 2] GT=1 residual layer-output hidden MSE, coeff=$HIDDEN_MSE_COEFF, layers=$HIDDEN_MSE_LAYERS -> $MSE_OUTPUT
[STAGE 3] GT=1 residual layer-output hidden KL, coeff=$HIDDEN_KL_COEFF, layers=$HIDDEN_KL_LAYERS -> $HIDDEN_KL_OUTPUT
[STAGE 4] GT=1 output-vocabulary KL, coeff=$VOCAB_KL_COEFF -> $VOCAB_KL_OUTPUT
[INDEPENDENCE] all four stages start from the same source step-$SOURCE_STEP checkpoint; no stage loads a previous result
[ORDER] LM -> hidden-MSE -> hidden-KL -> vocab-KL, with checkpoint verification between stages
[SAFETY] no process termination, no output overwrite, nonblocking chain lock, distinct ports $LM_MASTER_PORT/$MSE_MASTER_PORT/$HIDDEN_KL_MASTER_PORT/$VOCAB_KL_MASTER_PORT
PLAN

if [[ "$PLAN_ONLY" == "1" ]]; then
    exit 0
fi

mkdir -p "$STUDY_ROOT" "$CHECKPOINT_ROOT" "$LOG_ROOT" "$LOCAL_BASE" "$LOCAL_SSD_ROOT"
exec 9>>"$LOCK_FILE"
flock -n 9 || die "another old-like LM/MSE chain holds the lock: $LOCK_FILE"
record LOCK "acquired chain lock pid=$$ host=$(hostname)"

preflight_source_and_data
preflight_gt
preflight_outputs

if verified_stage_complete "$LM_OUTPUT" old_like_router_lm && \
   verified_stage_complete "$MSE_OUTPUT" old_like_router_hidden_mse && \
   verified_stage_complete "$HIDDEN_KL_OUTPUT" old_like_router_hidden_kl && \
   verified_stage_complete "$VOCAB_KL_OUTPUT" old_like_router_vocab_kl; then
    record COMPLETE "all four stages were already verified complete"
    exit 0
fi

preflight_gpus_and_ports
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    record PREFLIGHT "all checks passed; no training launched"
    exit 0
fi

COMMON_STAGE_ENV=(
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    NPROC_PER_NODE="$NPROC_PER_NODE"
    FLAME_ENV="$FLAME_ENV"
    PYTHON_BIN="$PYTHON_BIN"
    FLAME_DATA_ROOT="$FLAME_DATA_ROOT"
    TOKENIZER_MODEL="$TOKENIZER_MODEL"
    LOCAL_BASE="$LOCAL_BASE"
    LOCAL_SSD_ROOT="$LOCAL_SSD_ROOT"
    DIRECT_LOCAL_SAVE=1
    SOURCE_WEIGHTS_DIR="$SOURCE_CHECKPOINT"
    SOURCE_REQUIRED_ITERS="$SOURCE_STEP"
    TRAIN_DATASET="$CODE_TRAIN_DIR"
    JOINT_REPLAY_DATASET="$CODE_TRAIN_DIR"
    JOINT_REPLAY_DATA_WEIGHT_MODE=equal_dataset
    MOE_JOINT_REPLAY_OLD_LIKE_GT_PATH="$GT_ROOT"
    MOE_JOINT_REPLAY_OLD_LIKE_UNIT="$OLD_LIKE_REPLAY_UNIT"
    MOE_JOINT_REPLAY_OLD_LIKE_TARGET_TRAIN_FRACTION="$TARGET_TRAIN_FRACTION"
    MOE_JOINT_REPLAY_OLD_LIKE_SELECTED_TOKEN_COUNT="$GT_SELECTED_COUNT"
    MOE_JOINT_REPLAY_OLD_LIKE_POSITIVE_SAMPLE_COUNT="$GT_POSITIVE_SAMPLE_COUNT"
    MOE_JOINT_REPLAY_OLD_LIKE_FULL_TRAIN_TOKEN_COUNT="$GT_FULL_TRAIN_TOKEN_COUNT"
    OLD_LIKE_REPLAY_SUBSET_COUNT="$GT_REPLAY_SUBSET_COUNT"
    OLD_LIKE_REPLAY_SUBSET_SHA256="$GT_REPLAY_SUBSET_SHA256"
    TRAIN_ITERS="$TRAIN_ITERS"
    MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE"
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE"
    SEQ_LENGTH=512
    DATASET_SPLIT=100,0,0
    LR=3e-4
    MIN_LR=3e-5
    LR_DECAY_STYLE=WSD
    LR_DECAY_ITERS="$TRAIN_ITERS"
    LR_WSD_DECAY_ITERS=180
    LR_WARMUP_FRACTION=0.01
    SAVE_INTERVAL="$TRAIN_ITERS"
    EVAL_INTERVAL=600
    LOG_INTERVAL=20
    PROBE_EVAL_INTERVAL=100
    SECONDARY_PROBE_EVAL_INTERVAL=100
    PROBE_EVAL_ITERS=25
    SECONDARY_PROBE_EVAL_ITERS=25
    TERTIARY_PROBE_EVAL_INTERVAL=0
    RUN_INITIAL_PROBE_EVAL=1
    RUN_INITIAL_VALID_EVAL=0
    MOE_JOINT_NEW_EXPERT_QUOTA=0
    MOE_JOINT_NEW_EXPERT_QUOTA_SCHEDULE=200:0.5
    MOE_JOINT_NEW_EXPERT_QUOTA_MIN_NEW_SLOTS=1
    MOE_JOINT_NEW_EXPERT_QUOTA_LOSS_COEFF=0.1
    MOE_JOINT_NEW_EXPERT_QUOTA_PRESERVE_NATURAL_GRADS=1
    MOE_SEPARATE_ROUTER_EXPERT_GRAD_CLIP=1
    MOE_ALLOW_PARTIAL_OPTIMIZER_STATE=0
    RECOVERY_SAVE_INTERVAL=300
    MOE_NEW_EXPERT_LR_RAMP_STEPS=0
    MOE_ROUTER_LR_MULTIPLIER=1.0
    LOG_ROUTER_GRAD_NORM_SOURCES=1
    MOE_AUX_LOSS_COEFF=0.01
    MOE_Z_LOSS_COEFF=0.001
    NO_SAVE_OPTIM=1
    WANDB_MODE="$WANDB_MODE"
)

run_stage old_like_router_lm old_like_router_lm "$LM_OUTPUT" "$LM_LOG" \
    env "${COMMON_STAGE_ENV[@]}" \
        ENABLE_OLD_MODEL_KL=0 OLD_MODEL_KL_COEFF=0 \
        MOE_JOINT_REPLAY_LM=1 \
        MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
        RUN_ID="$LM_RUN_ID" TRAIN_WEIGHTS="$LM_OUTPUT" \
        WANDB_RUN_ID="$LM_RUN_ID" WANDB_EXP_NAME="G2 Code old-like router LM" \
        MASTER_PORT="$LM_MASTER_PORT" \
        bash "$LM_ENTRY_SCRIPT"

run_stage old_like_router_hidden_mse old_like_router_hidden_mse "$MSE_OUTPUT" "$MSE_LOG" \
    env "${COMMON_STAGE_ENV[@]}" \
        ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0 \
        OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE_CHECKPOINT" OLD_MODEL_KL_NUM_EXPERTS=16 \
        MOE_JOINT_REPLAY_LM=1 \
        MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=1 \
        OLD_HIDDEN_MSE_COEFF="$HIDDEN_MSE_COEFF" \
        OLD_HIDDEN_MSE_LAYERS="$HIDDEN_MSE_LAYERS" \
        RUN_ID="$MSE_RUN_ID" TRAIN_WEIGHTS="$MSE_OUTPUT" \
        WANDB_RUN_ID="$MSE_RUN_ID" WANDB_EXP_NAME="G2 Code old-like router hidden-MSE c10 L2-L9" \
        MASTER_PORT="$MSE_MASTER_PORT" \
        bash "$MSE_ENTRY_SCRIPT"

run_stage old_like_router_hidden_kl old_like_router_hidden_kl "$HIDDEN_KL_OUTPUT" "$HIDDEN_KL_LOG" \
    env "${COMMON_STAGE_ENV[@]}" \
        ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF=0 \
        OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE_CHECKPOINT" OLD_MODEL_KL_NUM_EXPERTS=16 \
        MOE_JOINT_REPLAY_LM=1 \
        MOE_JOINT_REPLAY_OLD_DATA_KD=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=1 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
        OLD_HIDDEN_KL_COEFF="$HIDDEN_KL_COEFF" \
        OLD_HIDDEN_KL_LAYERS="$HIDDEN_KL_LAYERS" \
        RUN_ID="$HIDDEN_KL_RUN_ID" TRAIN_WEIGHTS="$HIDDEN_KL_OUTPUT" \
        WANDB_RUN_ID="$HIDDEN_KL_RUN_ID" WANDB_EXP_NAME="G2 Code old-like router hidden-KL c1 L2-L9 20pct" \
        MASTER_PORT="$HIDDEN_KL_MASTER_PORT" \
        bash "$HIDDEN_KL_ENTRY_SCRIPT"

run_stage old_like_router_vocab_kl old_like_router_vocab_kl "$VOCAB_KL_OUTPUT" "$VOCAB_KL_LOG" \
    env "${COMMON_STAGE_ENV[@]}" \
        ENABLE_OLD_MODEL_KL=1 OLD_MODEL_KL_COEFF="$VOCAB_KL_COEFF" \
        OLD_MODEL_KL_WEIGHTS_DIR="$SOURCE_CHECKPOINT" OLD_MODEL_KL_NUM_EXPERTS=16 \
        MOE_JOINT_REPLAY_LM=1 \
        MOE_JOINT_REPLAY_OLD_DATA_KD=1 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_KL=0 \
        MOE_JOINT_REPLAY_OLD_DATA_HIDDEN_MSE=0 \
        RUN_ID="$VOCAB_KL_RUN_ID" TRAIN_WEIGHTS="$VOCAB_KL_OUTPUT" \
        WANDB_RUN_ID="$VOCAB_KL_RUN_ID" WANDB_EXP_NAME="G2 Code old-like router vocab-KL c1 20pct" \
        MASTER_PORT="$VOCAB_KL_MASTER_PORT" \
        bash "$VOCAB_KL_ENTRY_SCRIPT"

record COMPLETE "all four independent old-like token-only 20pct router-replay stages verified complete"
