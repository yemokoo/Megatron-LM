#!/usr/bin/env bash
set -euo pipefail

# Deterministic, restartable CKA-GT pilot chain for physical GPUs 0--3.
#
# Safety contract:
#   * every GPU stage refuses an occupied device (exit 75);
#   * this launcher never terminates another process;
#   * four pass-2 workers are independent world-size-1 processes;
#   * a nonblocking flock prevents two copies of this chain;
#   * committed worker manifests are resumed or verified, never overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_cka_gt_pilot_mha.sh"
WINDOW_TOOL="$SCRIPT_DIR/cka_gt_pilot_windows.py"
VALIDATOR="$SCRIPT_DIR/validate_cka_gt_pilot.py"
ANALYZER="$SCRIPT_DIR/analyze_cka_gt_pilot.py"
PY_ENV="${PY_ENV:-/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100}"
PYTHON_BIN="${PYTHON_BIN:-$PY_ENV/bin/python}"

APPROVED_OUT_ROOT="/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_pilot_20260815/analysis/cka_gt_pilot_v1"
OUT_ROOT="${OUT_ROOT:-$APPROVED_OUT_ROOT}"
PILOT_CONFIG="$OUT_ROOT/config.json"
MEMBERSHIP_STATS="$OUT_ROOT/membership/wiki_calibration_membership.npz"
TOKEN_METRICS="$OUT_ROOT/token_metrics"
CHUNK_METRICS="$OUT_ROOT/chunk_metrics"
RAW_QUANTILES="$OUT_ROOT/threshold_tables/raw_unit_quantiles.parquet"
LOG_ROOT="$OUT_ROOT/logs"
VALIDATION_ROOT="$OUT_ROOT/validation"
STATE_ROOT="$OUT_ROOT/launch_state"
LOCK_FILE="$OUT_ROOT/.launch_cka_gt_pilot_4gpu.lock"
CHAIN_LOG="$LOG_ROOT/launch_cka_gt_pilot_4gpu.log"

# The real-checkpoint smoke is physically isolated from production.  Its
# prepared windows, journals, Parquet, membership statistics, sealed test,
# analysis, and validation all live below this sibling root and therefore
# cannot enter the production inventory or resume state.
APPROVED_SMOKE_ROOT="${APPROVED_OUT_ROOT}_checkpoint_smoke_1000"
SMOKE_ROOT="${SMOKE_ROOT:-$APPROVED_SMOKE_ROOT}"
SMOKE_CONFIG="$SMOKE_ROOT/config.json"
SMOKE_MEMBERSHIP_STATS="$SMOKE_ROOT/membership/wiki_calibration_membership.npz"
SMOKE_TOKEN_METRICS="$SMOKE_ROOT/token_metrics"
SMOKE_CHUNK_METRICS="$SMOKE_ROOT/chunk_metrics"
SMOKE_RAW_QUANTILES="$SMOKE_ROOT/threshold_tables/raw_unit_quantiles.parquet"
SMOKE_LOG_ROOT="$SMOKE_ROOT/logs"
SMOKE_VALIDATION_ROOT="$SMOKE_ROOT/validation"
SMOKE_PROJECTION="$SMOKE_ROOT/smoke_projection.json"
SMOKE_WINDOWS_PER_DOMAIN=1000

PLAN_ONLY="${PLAN_ONLY:-0}"
START_STAGE="${START_STAGE:-prepared_validate}"
STOP_AFTER_STAGE="${STOP_AFTER_STAGE:-analyze}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-32}"
PASS1_BATCH_SIZE="${PASS1_BATCH_SIZE:-32}"
SMOKE_WINDOW_BATCH_SIZE="${SMOKE_WINDOW_BATCH_SIZE:-4}"
SMOKE_PASS1_BATCH_SIZE="${SMOKE_PASS1_BATCH_SIZE:-4}"
SHARD_WINDOWS="${SHARD_WINDOWS:-64}"
SEED="${SEED:-1234}"
PREPARED_VALIDATED_THIS_INVOCATION=0
FULL_VALIDATED_THIS_INVOCATION=0

# Every torchrun gets a unique fixed port, including sequential stages.  This
# makes logs/configs self-identifying and prevents accidental port reuse.
ROUTER_SMOKE_PORT="${ROUTER_SMOKE_PORT:-35610}"
CHECKPOINT_SMOKE_PASS1_PORT="${CHECKPOINT_SMOKE_PASS1_PORT:-35510}"
CHECKPOINT_SMOKE_CODE_PORT="${CHECKPOINT_SMOKE_CODE_PORT:-35520}"
CHECKPOINT_SMOKE_WIKI_PORT="${CHECKPOINT_SMOKE_WIKI_PORT:-35530}"
PASS1_PORT="${PASS1_PORT:-35620}"
CODE_PORT_BASE="${CODE_PORT_BASE:-35630}"   # 35630..35633
WIKI_PORT_BASE="${WIKI_PORT_BASE:-35640}"   # 35640..35643

STAGE_NAMES=(
    prepared_validate
    router_smoke
    checkpoint_smoke_1000
    pass1_wiki_calibration
    membership_validate
    pass2_code
    pass2_wiki
    full_validate
    analyze
)

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

stage_number() {
    local value="$1" index
    if [[ "$value" =~ ^[1-9]$ ]]; then
        echo "$value"
        return
    fi
    for index in "${!STAGE_NAMES[@]}"; do
        if [[ "$value" == "${STAGE_NAMES[$index]}" ]]; then
            echo "$((index + 1))"
            return
        fi
    done
    die "unknown stage '$value'; use 1..9 or: ${STAGE_NAMES[*]}"
}

START_INDEX="$(stage_number "$START_STAGE")"
STOP_INDEX="$(stage_number "$STOP_AFTER_STAGE")"
(( START_INDEX <= STOP_INDEX )) || die "START_STAGE is after STOP_AFTER_STAGE"
[[ "$PLAN_ONLY" == 0 || "$PLAN_ONLY" == 1 ]] || die "PLAN_ONLY must be 0 or 1"
[[ "$OUT_ROOT" == "$APPROVED_OUT_ROOT" ]] || die "OUT_ROOT must remain the approved root: $APPROVED_OUT_ROOT"
[[ "$SMOKE_ROOT" == "$APPROVED_SMOKE_ROOT" ]] \
    || die "SMOKE_ROOT must remain the isolated approved root: $APPROVED_SMOKE_ROOT"
[[ "$SMOKE_ROOT" != "$OUT_ROOT" && "$SMOKE_ROOT" != "$OUT_ROOT/"* ]] \
    || die "checkpoint smoke must be outside the production root"
[[ "$WINDOW_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "WINDOW_BATCH_SIZE must be positive"
[[ "$PASS1_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "PASS1_BATCH_SIZE must be positive"
[[ "$SMOKE_WINDOW_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || die "SMOKE_WINDOW_BATCH_SIZE must be positive"
[[ "$SMOKE_PASS1_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] \
    || die "SMOKE_PASS1_BATCH_SIZE must be positive"
[[ "$SHARD_WINDOWS" =~ ^[1-9][0-9]*$ ]] || die "SHARD_WINDOWS must be positive"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be a nonnegative integer"
[[ "$SEED" == 1234 ]] || die "the frozen CKA pilot manifest requires SEED=1234"

declare -a ALL_PORTS=(
    "$ROUTER_SMOKE_PORT"
    "$CHECKPOINT_SMOKE_PASS1_PORT"
    "$CHECKPOINT_SMOKE_CODE_PORT"
    "$CHECKPOINT_SMOKE_WIKI_PORT"
    "$PASS1_PORT"
)
for worker in 0 1 2 3; do
    ALL_PORTS+=("$((CODE_PORT_BASE + worker))" "$((WIKI_PORT_BASE + worker))")
done
declare -A SEEN_PORT=()
for port in "${ALL_PORTS[@]}"; do
    [[ "$port" =~ ^[1-9][0-9]*$ ]] && (( port < 65536 )) || die "invalid master port: $port"
    [[ -z "${SEEN_PORT[$port]:-}" ]] || die "master ports must be distinct; duplicate $port"
    SEEN_PORT[$port]=1
done

cat <<PLAN
[CKA GT PILOT 4-GPU CHAIN]
output: $OUT_ROOT
physical GPUs: 0,1,2,3 only
start: $START_INDEX (${STAGE_NAMES[$((START_INDEX - 1))]})
stop:  $STOP_INDEX (${STAGE_NAMES[$((STOP_INDEX - 1))]})
seed: $SEED
ports: router=$ROUTER_SMOKE_PORT pass1=$PASS1_PORT code=$CODE_PORT_BASE..$((CODE_PORT_BASE + 3)) wiki=$WIKI_PORT_BASE..$((WIKI_PORT_BASE + 3))
isolated real-checkpoint smoke: $SMOKE_ROOT
smoke ports: pass1=$CHECKPOINT_SMOKE_PASS1_PORT code=$CHECKPOINT_SMOKE_CODE_PORT wiki=$CHECKPOINT_SMOKE_WIKI_PORT
production batches: pass1=$PASS1_BATCH_SIZE pass2=$WINDOW_BATCH_SIZE
checkpoint-smoke batches: pass1=$SMOKE_PASS1_BATCH_SIZE pass2=$SMOKE_WINDOW_BATCH_SIZE

1. prepared config validation (CPU)
2. one-window standard-router smoke (GPU 0)
3. isolated 1,000 Code + 1,000 Wiki checkpoint smoke: Pass 1, Pass 2, deep validation, analysis, ETA/storage projection (GPU 0 + CPU)
4. Wiki-calibration Pass 1 membership statistics (GPU 0)
5. membership validator (CPU)
6. Code Pass 2, four concurrent world-size-1 workers (GPUs 0..3)
7. Wiki Pass 2, four concurrent world-size-1 workers (GPUs 0..3)
8. full scalar/coverage validator before analysis (CPU)
9. threshold/selector/histogram analysis and final validator (CPU)

[SAFETY] occupied GPU or port => fail; no process termination; nonblocking flock; exact worker resume.
PLAN

if [[ "$PLAN_ONLY" == 1 ]]; then
    for index in $(seq "$START_INDEX" "$STOP_INDEX"); do
        echo "[PLAN STAGE $index] ${STAGE_NAMES[$((index - 1))]}"
    done
    echo "[PLAN COMMAND] prepared: $PYTHON_BIN $WINDOW_TOOL --output-root $OUT_ROOT --validate-only"
    echo "[PLAN COMMAND] prepared identity upgrade when missing: $PYTHON_BIN $WINDOW_TOOL --output-root $OUT_ROOT --upgrade-source-identities"
    echo "[PLAN COMMAND] router: MODE=router_smoke DOMAIN=code SPLIT=calibration GPU=0 MAX_WINDOWS=1 MASTER_PORT=$ROUTER_SMOKE_PORT bash $RUNNER"
    echo "[PLAN COMMAND] checkpoint-smoke prepare: $PYTHON_BIN $WINDOW_TOOL --output-root $SMOKE_ROOT --sample-windows-per-domain $SMOKE_WINDOWS_PER_DOMAIN"
    echo "[PLAN COMMAND] checkpoint-smoke pass1: MODE=pass1 DOMAIN=wiki SPLIT=calibration GPU=0 WORKER_COUNT=1 WINDOW_BATCH_SIZE=$SMOKE_PASS1_BATCH_SIZE MASTER_PORT=$CHECKPOINT_SMOKE_PASS1_PORT bash $RUNNER"
    echo "[PLAN COMMAND] checkpoint-smoke pass2-code: MODE=pass2 DOMAIN=code SPLIT=all GPU=0 WORKER_COUNT=1 WINDOW_BATCH_SIZE=$SMOKE_WINDOW_BATCH_SIZE MASTER_PORT=$CHECKPOINT_SMOKE_CODE_PORT bash $RUNNER"
    echo "[PLAN COMMAND] checkpoint-smoke pass2-wiki: MODE=pass2 DOMAIN=wiki SPLIT=all GPU=0 WORKER_COUNT=1 WINDOW_BATCH_SIZE=$SMOKE_WINDOW_BATCH_SIZE MASTER_PORT=$CHECKPOINT_SMOKE_WIKI_PORT bash $RUNNER"
    echo "[PLAN COMMAND] checkpoint-smoke validate/analyze: $PYTHON_BIN $VALIDATOR $SMOKE_ROOT --allow-incomplete --deep; $PYTHON_BIN $ANALYZER $SMOKE_ROOT --output-root $SMOKE_ROOT --bins 256 --seed $SEED"
    echo "[PLAN COMMAND] pass1: MODE=pass1 DOMAIN=wiki SPLIT=calibration GPU=0 WINDOW_BATCH_SIZE=$PASS1_BATCH_SIZE MASTER_PORT=$PASS1_PORT bash $RUNNER"
    for domain in code wiki; do
        base="$CODE_PORT_BASE"
        [[ "$domain" == wiki ]] && base="$WIKI_PORT_BASE"
        for worker in 0 1 2 3; do
            echo "[PLAN COMMAND] pass2-$domain worker=$worker GPU=$worker WINDOW_BATCH_SIZE=$WINDOW_BATCH_SIZE port=$((base + worker))"
        done
    done
    echo "[PLAN COMMAND] full pre-analysis validate: $PYTHON_BIN $VALIDATOR $OUT_ROOT --allow-incomplete --deep"
    echo "[PLAN COMMAND] analyze: $PYTHON_BIN $ANALYZER $OUT_ROOT --output-root $OUT_ROOT --token-metrics $TOKEN_METRICS --chunk-metrics $CHUNK_METRICS --raw-quantiles $RAW_QUANTILES --bins 256 --seed $SEED"
    if (( START_INDEX <= 8 && STOP_INDEX >= 9 )); then
        echo "[PLAN COMMAND] final after-analysis validate (stage8 current => shallow): $PYTHON_BIN $VALIDATOR $OUT_ROOT"
    elif (( START_INDEX == 9 )); then
        echo "[PLAN COMMAND] final after-analysis validate (analyze-only safety fallback): $PYTHON_BIN $VALIDATOR $OUT_ROOT --deep"
    fi
    exit 0
fi

for path in "$PYTHON_BIN" "$RUNNER" "$WINDOW_TOOL" "$VALIDATOR" "$ANALYZER"; do
    [[ -e "$path" ]] || die "missing required executable/file: $path"
done
[[ -x "$RUNNER" ]] || die "runner is not executable: $RUNNER"
[[ -f "$PILOT_CONFIG" ]] || die "prepared config is missing: $PILOT_CONFIG"

mkdir -p "$LOG_ROOT" "$VALIDATION_ROOT" "$STATE_ROOT" "$OUT_ROOT/membership"
exec 9>"$LOCK_FILE"
flock -n 9 || die "another CKA GT pilot chain holds $LOCK_FILE"

record() {
    local kind="$1"; shift
    local line="[$(date '+%Y-%m-%d %H:%M:%S')] [$kind] $*"
    echo "$line" | tee -a "$CHAIN_LOG"
}

mark_stage() {
    local index="$1" status="$2"
    "$PYTHON_BIN" - "$STATE_ROOT/stage_${index}_${STAGE_NAMES[$((index - 1))]}.json" \
        "$index" "${STAGE_NAMES[$((index - 1))]}" "$status" "$PILOT_CONFIG" <<'PY'
import hashlib, json, os, socket, sys, tempfile, time
path, index, name, status, config = sys.argv[1:]
with open(config, "rb") as handle:
    config_hash = hashlib.sha256(handle.read()).hexdigest()
payload = {
    "schema": "cka_gt_pilot_launch_stage_v1",
    "stage_index": int(index),
    "stage": name,
    "status": status,
    "config_sha256": config_hash,
    "host": socket.gethostname(),
    "pid": os.getppid(),
    "completed_unix_time": time.time(),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, path)
PY
}

run_logged() {
    local label="$1" log="$2"; shift 2
    record START "$label log=$log"
    set +e
    "$@" 2>&1 | tee -a "$log"
    local -a status=("${PIPESTATUS[@]}")
    set -e
    if (( status[0] != 0 || status[1] != 0 )); then
        record FAIL "$label command_rc=${status[0]} tee_rc=${status[1]}"
        (( status[0] != 0 )) && exit "${status[0]}"
        exit "${status[1]}"
    fi
    record DONE "$label"
}

port_is_free() {
    "$PYTHON_BIN" - "$1" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
finally:
    sock.close()
PY
}

assert_gpu_idle() {
    local gpu="$1" descriptor applications
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
    [[ "$gpu" =~ ^[0-3]$ ]] || die "launcher GPU must be physical 0..3: $gpu"
    descriptor="$(nvidia-smi -i "$gpu" --query-gpu=index,uuid --format=csv,noheader,nounits 2>/dev/null)" \
        || die "cannot query GPU $gpu"
    [[ "$descriptor" == "$gpu,"* ]] || die "physical GPU mismatch: requested $gpu, got '$descriptor'"
    applications="$(nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)" \
        || die "cannot query compute applications on GPU $gpu"
    applications="$(printf '%s\n' "$applications" | sed '/^[[:space:]]*$/d; /No running processes found/d')"
    if [[ -n "$applications" ]]; then
        echo "[OCCUPIED] GPU $gpu:" >&2
        printf '%s\n' "$applications" >&2
        die "GPU $gpu must be idle; this chain does not terminate occupants"
    fi
}

assert_gpus_0_3_idle() {
    local gpu
    for gpu in 0 1 2 3; do
        assert_gpu_idle "$gpu"
    done
}

verify_worker_metadata() {
    local path="$1" mode="$2" domain="$3" split="$4" worker="$5" count="$6" batch_size="$7"
    local requested_shard_windows="${8:-$SHARD_WINDOWS}"
    local artifact_root="${9:-$OUT_ROOT}"
    local prepared_config="${10:-$PILOT_CONFIG}"
    local membership_stats="${11:-$MEMBERSHIP_STATS}"
    "$PYTHON_BIN" - "$path" "$mode" "$domain" "$split" "$worker" "$count" \
        "$batch_size" "$requested_shard_windows" "$SEED" "$prepared_config" "$artifact_root" \
        "$membership_stats" <<'PY'
import hashlib, json, os, sys
import numpy as np

path, mode, domain, split, worker, count, batch_size, shard_windows, seed, config_path, root, membership_path = sys.argv[1:]
if not os.path.isfile(path):
    raise SystemExit(1)
with open(path, encoding="utf-8") as handle:
    value = json.load(handle)
with open(config_path, encoding="utf-8") as handle:
    config = json.load(handle)
expected = {
    "completed": True,
    "mode": mode,
    "domain": domain,
    "requested_split": split,
    "worker_index": int(worker),
    "worker_count": int(count),
    "window_batch_size": int(batch_size),
    "shard_windows": int(shard_windows),
    "max_windows": 1 if mode == "router_smoke" else 0,
    "seed": int(seed),
    "prepared_config_content_sha256": config.get("config_content_sha256"),
    "reference_step": 600,
    "current_step": 1800,
    "natural_routing": True,
    "raw_hidden_stored": False,
    "training_run": False,
}
for key, wanted in expected.items():
    if value.get(key) != wanted:
        raise SystemExit(f"metadata mismatch {path}: {key}={value.get(key)!r}, expected {wanted!r}")
for key, wanted in (
    ("reference_load", config.get("before_checkpoint")),
    ("current_load", config.get("after_checkpoint")),
):
    if value.get(key) != wanted:
        raise SystemExit(
            f"checkpoint/config mismatch {path}: {key}={value.get(key)!r}, expected {wanted!r}"
        )

# Bind completed metadata to the exact prepared logical partition.  This
# prevents a stale worker from being skipped merely because its filename and
# mode happen to match a newly prepared pilot with different windows.
worker_i, count_i = int(worker), int(count)
split_names = ("calibration", "selection", "test") if split == "all" else (split,)
expected_windows = 0
expected_tokens = 0
partition_by_split = {}
for split_name in split_names:
    rows_path = os.path.join(root, "splits", f"{domain}_{split_name}_windows.npy")
    rows = np.load(rows_path, allow_pickle=False)
    quotient, remainder = divmod(int(rows.shape[0]), count_i)
    start = worker_i * quotient + min(worker_i, remainder)
    size = quotient + int(worker_i < remainder)
    partition = rows[start : start + size]
    if mode == "router_smoke":
        partition = partition[:1]
    domain_id = {"code": 0, "wiki": 1}[domain]
    split_id = {"calibration": 0, "selection": 1, "test": 2}[split_name]
    uids = (
        np.int64(domain_id) * np.int64(1_000_000_000)
        + np.int64(split_id) * np.int64(100_000_000)
        + partition["sample_order"].astype(np.int64)
    )
    partition_by_split[split_name] = {
        "windows": int(partition.shape[0]),
        "tokens": int(partition["window_length"].sum(dtype=np.int64)),
        "uids": {int(uid) for uid in uids.tolist()},
    }
    expected_windows += int(partition.shape[0])
    expected_tokens += int(partition["window_length"].sum(dtype=np.int64))
if int(value.get("total_windows", -1)) != expected_windows:
    raise SystemExit(
        f"prepared-window count mismatch {path}: {value.get('total_windows')!r} != {expected_windows}"
    )
if int(value.get("total_tokens", -1)) != expected_tokens:
    raise SystemExit(
        f"prepared-token count mismatch {path}: {value.get('total_tokens')!r} != {expected_tokens}"
    )

def file_sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def validate_membership_identity(identity, label):
    if not isinstance(identity, dict) or identity.get("schema") != "cka_gt_pilot_membership_identity_v1":
        raise SystemExit(f"{label}: invalid membership identity schema")
    resolved = os.path.realpath(membership_path)
    if os.path.realpath(str(identity.get("path", ""))) != resolved:
        raise SystemExit(f"{label}: membership path mismatch")
    if not os.path.isfile(resolved):
        raise SystemExit(f"{label}: membership file missing: {resolved}")
    if int(identity.get("size_bytes", -1)) != os.path.getsize(resolved):
        raise SystemExit(f"{label}: membership size mismatch")
    if identity.get("sha256") != file_sha256(resolved):
        raise SystemExit(f"{label}: membership SHA256 mismatch")
    wiki_rows = np.load(
        os.path.join(root, "splits", "wiki_calibration_windows.npy"),
        allow_pickle=False,
    )
    saved = identity.get("saved_metadata_summary", {})
    expected_saved = {
        "analysis": "cka_gt_pilot_v1",
        "mode": "pass1",
        "source_domain": "wiki",
        "source_split": "calibration",
        "source_dataset_prefix": config.get("wiki_prefix"),
        "reference_checkpoint": os.path.realpath(str(config.get("before_checkpoint"))),
        "reference_step": 600,
        "prepared_config_content_sha256": config.get("config_content_sha256"),
        "source_window_count": int(wiki_rows.shape[0]),
        "source_token_count": int(wiki_rows["window_length"].sum(dtype=np.int64)),
        "seed": int(seed),
        "max_windows": 0,
        "layers": list(range(2, 10)),
    }
    for key, wanted in expected_saved.items():
        if saved.get(key) != wanted:
            raise SystemExit(
                f"{label}: membership source mismatch {key}={saved.get(key)!r}, expected {wanted!r}"
            )

if mode in ("pass1", "pass2"):
    identity = value.get("membership_identity")
    validate_membership_identity(identity, path)
    pass1_path = os.path.join(
        root, "runtime", "pass1", "wiki", "calibration", "worker_000", "metadata.json"
    )
    if mode == "pass2":
        if not os.path.isfile(pass1_path):
            raise SystemExit(f"Pass2 canonical Pass1 metadata missing: {pass1_path}")
        with open(pass1_path, encoding="utf-8") as handle:
            pass1 = json.load(handle)
        if pass1.get("completed") is not True or pass1.get("membership_identity") != identity:
            raise SystemExit(f"Pass2 membership identity differs from canonical Pass1: {path}")
if mode in ("router_smoke", "pass2") and value.get("router_probe_verified") is not True:
    raise SystemExit(f"router probe not verified: {path}")
if mode == "pass2":
    manifests = value.get("paired_manifests")
    if not isinstance(manifests, dict) or set(manifests) != {"open", "test"}:
        raise SystemExit(f"Pass2 metadata must contain exact open/test paired_manifests: {path}")
    expected_streams = {
        "open": ("calibration", "selection"),
        "test": ("test",),
    }
    seen_stream_uids = set()
    for stream, stream_splits in expected_streams.items():
        manifest = manifests.get(stream)
        if not isinstance(manifest, dict) or manifest.get("stream_kind") != "paired_token_chunk_parquet":
            raise SystemExit(f"missing paired {stream} stream manifest: {path}")
        expected_uids = set().union(
            *(partition_by_split[name]["uids"] for name in stream_splits)
        )
        expected_stream_tokens = sum(partition_by_split[name]["tokens"] for name in stream_splits)
        if seen_stream_uids & expected_uids:
            raise SystemExit(f"prepared open/test UID overlap: {path}")
        seen_stream_uids.update(expected_uids)
        expected_label = f"{domain}_{split}_{stream}_worker_{int(worker):03d}"
        if stream == "open":
            token_dir = os.path.realpath(os.path.join(root, "token_metrics", expected_label))
            chunk_dir = os.path.realpath(os.path.join(root, "chunk_metrics", expected_label))
        else:
            token_dir = os.path.realpath(
                os.path.join(root, "sealed_test", "raw", "token_metrics", expected_label)
            )
            chunk_dir = os.path.realpath(
                os.path.join(root, "sealed_test", "raw", "chunk_metrics", expected_label)
            )
        progress_path = os.path.join(
            os.path.dirname(path), "paired_progress", stream, "progress.json"
        )
        if not os.path.isfile(progress_path):
            raise SystemExit(f"paired {stream} progress journal missing: {progress_path}")
        with open(progress_path, encoding="utf-8") as handle:
            progress = json.load(handle)
        if progress != manifest:
            raise SystemExit(f"paired {stream} progress/metadata mismatch: {path}")
        if manifest.get("complete") is not True:
            raise SystemExit(f"incomplete paired {stream} manifest: {path}")
        if os.path.realpath(str(manifest.get("token_dir", ""))) != token_dir:
            raise SystemExit(f"paired {stream} token_dir is not canonical: {path}")
        if os.path.realpath(str(manifest.get("chunk_dir", ""))) != chunk_dir:
            raise SystemExit(f"paired {stream} chunk_dir is not canonical: {path}")
        if int(manifest.get("target_token_rows_per_shard", -1)) != int(shard_windows) * 512:
            raise SystemExit(f"paired {stream} shard target mismatch: {path}")
        if int(manifest.get("committed_window_count", -1)) != len(expected_uids):
            raise SystemExit(f"paired {stream} committed window count mismatch: {path}")
        if int(manifest.get("committed_batches", 0)) <= 0:
            raise SystemExit(f"paired {stream} manifest has no committed batches: {path}")
        progress_metadata = progress.get("metadata", {})
        metadata_expected = {
            "mode": mode,
            "domain": domain,
            "requested_split": split,
            "worker_index": int(worker),
            "worker_count": int(count),
            "window_batch_size": int(batch_size),
            "shard_windows": int(shard_windows),
            "max_windows": 0,
            "seed": int(seed),
            "prepared_config_content_sha256": config.get("config_content_sha256"),
            "total_windows": int(value["total_windows"]),
            "total_tokens": int(value["total_tokens"]),
            "metric_stream": stream,
            "stream_splits": list(stream_splits),
            "test_metrics_sealed": stream == "test",
            "stream_window_count": len(expected_uids),
            "stream_token_count": expected_stream_tokens,
            "membership_identity": identity,
        }
        for key, wanted in metadata_expected.items():
            if progress_metadata.get(key) != wanted:
                raise SystemExit(
                    f"paired {stream} progress metadata mismatch {progress_path}: "
                    f"{key}={progress_metadata.get(key)!r}, expected {wanted!r}"
                )
        shards = manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise SystemExit(f"paired {stream} manifest has no committed shards: {path}")
        committed_uids = set()
        token_rows = 0
        committed_batches = 0
        for shard_index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise SystemExit(f"malformed paired {stream} shard record: {progress_path}")
            shard_uids = [int(uid) for uid in shard.get("window_uids", [])]
            if len(shard_uids) != len(set(shard_uids)) or committed_uids.intersection(shard_uids):
                raise SystemExit(f"duplicate paired {stream} window UID at shard {shard_index}")
            committed_uids.update(shard_uids)
            token_rows += int(shard.get("token_rows", -1))
            committed_batches += int(shard.get("batch_count", -1))
            artifacts = (
                ("token_file", token_dir),
                ("chunk_file", chunk_dir),
                ("sidecar_file", os.path.dirname(progress_path)),
            )
            for key, expected_parent in artifacts:
                artifact = shard.get(key)
                if not isinstance(artifact, str) or not os.path.isfile(artifact):
                    raise SystemExit(f"missing paired {stream} artifact {key}={artifact!r}")
                if os.path.realpath(os.path.dirname(artifact)) != os.path.realpath(expected_parent):
                    raise SystemExit(f"paired {stream} artifact escaped canonical parent: {artifact}")
        if committed_uids != expected_uids:
            raise SystemExit(f"paired {stream} committed UID set mismatch: {path}")
        if token_rows != expected_stream_tokens:
            raise SystemExit(
                f"paired {stream} token rows {token_rows} != prepared tokens {expected_stream_tokens}"
            )
        if committed_batches != int(manifest.get("committed_batches", -1)):
            raise SystemExit(f"paired {stream} committed batch sum mismatch: {path}")
    if seen_stream_uids != set().union(*(item["uids"] for item in partition_by_split.values())):
        raise SystemExit(f"paired open/test streams do not exactly cover worker partition: {path}")
PY
}

worker_metadata_path() {
    local mode="$1" domain="$2" split="$3" worker="$4" root="${5:-$OUT_ROOT}"
    printf '%s/runtime/%s/%s/%s/worker_%03d/metadata.json\n' \
        "$root" "$mode" "$domain" "$split" "$worker"
}

verify_router_smoke() {
    verify_worker_metadata "$(worker_metadata_path router_smoke code calibration 0)" \
        router_smoke code calibration 0 1 1 1
}

verify_pass1() {
    verify_worker_metadata "$(worker_metadata_path pass1 wiki calibration 0)" \
        pass1 wiki calibration 0 1 "$PASS1_BATCH_SIZE" || return 1
    [[ -s "$MEMBERSHIP_STATS" ]]
}

verify_pass2_worker() {
    local domain="$1" worker="$2"
    verify_worker_metadata "$(worker_metadata_path pass2 "$domain" all "$worker")" \
        pass2 "$domain" all "$worker" 4 "$WINDOW_BATCH_SIZE"
}

verify_pass2_domain() {
    local domain="$1" worker
    for worker in 0 1 2 3; do
        verify_pass2_worker "$domain" "$worker" || return 1
    done
}

verify_pass1_at() {
    local root="$1" config="$2" membership="$3" batch_size="$4"
    verify_worker_metadata \
        "$(worker_metadata_path pass1 wiki calibration 0 "$root")" \
        pass1 wiki calibration 0 1 "$batch_size" "$SHARD_WINDOWS" \
        "$root" "$config" "$membership" || return 1
    [[ -s "$membership" ]]
}

verify_pass2_at() {
    local root="$1" config="$2" membership="$3" domain="$4" batch_size="$5"
    verify_worker_metadata \
        "$(worker_metadata_path pass2 "$domain" all 0 "$root")" \
        pass2 "$domain" all 0 1 "$batch_size" "$SHARD_WINDOWS" \
        "$root" "$config" "$membership"
}

assert_validation_json() {
    local path="$1" kind="$2" expected_stream_count="${3:-16}"
    "$PYTHON_BIN" - "$path" "$kind" "$expected_stream_count" <<'PY'
import json, os, sys
path, kind, expected_stream_count = sys.argv[1:]
expected_stream_count = int(expected_stream_count)
if not os.path.isfile(path):
    raise SystemExit(f"validation JSON missing: {path}")
with open(path, encoding="utf-8") as handle:
    value = json.load(handle)
if value.get("ok") is not True or value.get("errors"):
    raise SystemExit(f"validation failed: {path}")
stats = value.get("stats", {})
if kind == "membership" and "membership" not in stats:
    raise SystemExit("membership validator did not validate membership statistics")
if kind in ("full", "final"):
    if value.get("deep") is not True:
        raise SystemExit(f"{kind} validator was not deep")
    coverage = stats.get("pass2_window_coverage", {})
    required = ("expected", "observed_pass2", "missing", "extra", "missing_chunk_windows", "extra_chunk_windows")
    if any(name not in coverage for name in required):
        raise SystemExit("full validator omitted pass2 coverage")
    if coverage["missing"] or coverage["extra"] or coverage["missing_chunk_windows"] or coverage["extra_chunk_windows"]:
        raise SystemExit(f"incomplete pass2 coverage: {coverage}")
    if coverage["observed_pass2"] != coverage["expected"]:
        raise SystemExit(f"pass2 observed/expected mismatch: {coverage}")
if kind in ("full", "final", "final_shallow"):
    if kind == "final_shallow" and value.get("deep") is not False:
        raise SystemExit("final-after-analysis validator must be shallow")
    streams = stats.get("runtime_streams", [])
    paired = [stream for stream in streams if stream.get("kind") == "paired_token_chunk_parquet"]
    open_streams = [stream for stream in paired if "/paired_progress/open/progress.json" in stream.get("manifest", "")]
    test_streams = [stream for stream in paired if "/paired_progress/test/progress.json" in stream.get("manifest", "")]
    expected_half = expected_stream_count // 2
    if (
        expected_stream_count <= 0
        or expected_stream_count % 2
        or len(paired) != expected_stream_count
        or len(open_streams) != expected_half
        or len(test_streams) != expected_half
    ):
        raise SystemExit(
            f"expected {expected_stream_count} paired streams "
            f"({expected_half} open + {expected_half} sealed test), got "
            f"all={len(paired)} open={len(open_streams)} test={len(test_streams)}"
        )
    if any(stream.get("complete") is not True for stream in paired):
        raise SystemExit("one or more paired open/test streams are incomplete")
if kind in ("final", "final_shallow") and not any(
    "sealed-test" in check for check in value.get("checks", [])
):
    raise SystemExit("final validator did not verify sealed-test/report policy")
PY
}

verify_analysis() {
    local root="${1:-$OUT_ROOT}"
    "$PYTHON_BIN" - "$root" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1]).resolve()

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def load(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"analysis artifact missing/empty: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)

summary_path = root / "postprocess_summary.json"
report_path = root / "REPORT.md"
sealed_manifest_path = root / "sealed_test" / "manifest.json"
assignment_path = root / "selector_comparison" / "code_selection_assignments.parquet"
binding_path = root / "analysis_input_binding.json"
inventory_path = root / "input_parquet_inventory.json"
for artifact in (summary_path, report_path, sealed_manifest_path, assignment_path, binding_path, inventory_path):
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise SystemExit(f"analysis artifact missing/empty: {artifact}")

summary = load(summary_path)
binding = load(binding_path)
inventory = load(inventory_path)
if summary.get("schema") != "cka_gt_pilot_postprocess_v1":
    raise SystemExit("postprocess summary schema mismatch")
if binding.get("schema") != "cka_gt_pilot_analysis_input_binding_v1" or binding.get("status") != "BOUND_COMPLETE":
    raise SystemExit("analysis input binding is not BOUND_COMPLETE")
if inventory.get("schema") != "cka_gt_pilot_parquet_inventory_v1":
    raise SystemExit("input inventory schema mismatch")

digest_payload = {"schema": inventory["schema"], "files": inventory.get("files")}
canonical = json.dumps(digest_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
inventory_digest = hashlib.sha256(canonical).hexdigest()
if inventory_digest != inventory.get("inventory_digest_sha256"):
    raise SystemExit("input inventory canonical digest mismatch")
bound_inventory = binding.get("input_inventory", {})
if bound_inventory.get("inventory_digest_sha256") != inventory_digest:
    raise SystemExit("analysis binding points to a different input inventory")
if bound_inventory.get("file_sha256") != sha256(inventory_path):
    raise SystemExit("analysis binding inventory-file SHA mismatch")
if bound_inventory.get("sealed_test_raw_bound") is not True:
    raise SystemExit("analysis binding did not bind the physical sealed-test raw stream")

prepared = binding.get("prepared_input", {})
config_path = root / "config.json"
if prepared.get("available") is not True or not config_path.is_file():
    raise SystemExit("analysis binding omitted exact prepared-input provenance")
config = load(config_path)
if prepared.get("config_file_sha256") != sha256(config_path):
    raise SystemExit("analysis binding prepared-config file SHA mismatch")
if prepared.get("config_content_sha256") != config.get("config_content_sha256"):
    raise SystemExit("analysis binding prepared-config content hash mismatch")
sources = config.get("source_dataset_identity", {})
checkpoints = config.get("checkpoint_identity", {})
expected_sources = {
    domain: {
        "schema": sources.get(domain, {}).get("schema"),
        "storage_kind": sources.get(domain, {}).get("storage_kind"),
        "resolved_prefix": sources.get(domain, {}).get("resolved_prefix"),
        "idx_size_bytes": sources.get(domain, {}).get("idx", {}).get("size_bytes"),
        "idx_sha256": sources.get(domain, {}).get("idx", {}).get("sha256"),
        "bin_size_bytes": sources.get(domain, {}).get("bin", {}).get("size_bytes"),
        "bin_sha256": sources.get(domain, {}).get("bin", {}).get("sha256"),
    }
    for domain in ("code", "wiki")
}
expected_checkpoints = {
    name: {
        "schema": checkpoints.get(name, {}).get("schema"),
        "storage_kind": checkpoints.get(name, {}).get("storage_kind"),
        "resolved_root": checkpoints.get(name, {}).get("resolved_root"),
        "tracker_step": checkpoints.get(name, {}).get("tracker_step"),
        "iteration_dir": checkpoints.get(name, {}).get("iteration_dir"),
        "total_bytes": checkpoints.get(name, {}).get("total_bytes"),
        "content_sha256": checkpoints.get(name, {}).get("content_sha256"),
    }
    for name in ("before", "after")
}
if prepared.get("source_dataset_identity") != expected_sources:
    raise SystemExit("analysis binding source-dataset identity mismatch")
if prepared.get("checkpoint_identity") != expected_checkpoints:
    raise SystemExit("analysis binding checkpoint identity mismatch")

dataset_roots = {
    "token_metrics": root / "token_metrics",
    "chunk_metrics": root / "chunk_metrics",
    "sealed_test_token_metrics": root / "sealed_test" / "raw" / "token_metrics",
    "sealed_test_chunk_metrics": root / "sealed_test" / "raw" / "chunk_metrics",
}
listed = set()
for record in inventory.get("files", []):
    dataset = record.get("dataset")
    prefix = f"{dataset}/"
    relative = record.get("relative_path", "")
    if dataset not in dataset_roots or not relative.startswith(prefix):
        raise SystemExit(f"invalid inventory logical path: {record}")
    suffix = relative[len(prefix):]
    if not suffix or Path(suffix).is_absolute() or ".." in Path(suffix).parts:
        raise SystemExit(f"unsafe inventory relative path: {relative}")
    path = dataset_roots[dataset] / suffix
    listed.add((dataset, relative))
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise SystemExit(f"inventory file missing/size changed: {path}")
    metadata = pq.ParquetFile(path).metadata
    if metadata.num_rows != int(record.get("rows", -1)) or metadata.num_row_groups != int(record.get("row_groups", -1)):
        raise SystemExit(f"inventory Parquet row metadata changed: {path}")
    if sha256(path) != record.get("sha256"):
        raise SystemExit(f"inventory Parquet SHA changed: {path}")
actual = {
    (dataset, f"{dataset}/{path.relative_to(directory).as_posix()}")
    for dataset, directory in dataset_roots.items()
    if directory.exists()
    for path in directory.rglob("*.parquet")
    if path.is_file()
}
if actual != listed:
    raise SystemExit(f"current Parquet inventory differs: missing={listed-actual} extra={actual-listed}")

raw = binding.get("raw_quantiles", {})
raw_path = Path(str(raw.get("path", "")))
raw_manifest_path = Path(str(raw.get("manifest_path", "")))
if not raw_path.is_file() or not raw_manifest_path.is_file():
    raise SystemExit("bound raw quantile artifact/manifest missing")
if sha256(raw_path) != raw.get("sha256") or sha256(raw_manifest_path) != raw.get("manifest_sha256"):
    raise SystemExit("bound raw quantile hash mismatch")
raw_manifest = load(raw_manifest_path)
if raw_manifest.get("input_inventory_digest_sha256") != inventory_digest:
    raise SystemExit("raw quantiles are stale relative to current Parquet inventory")
if raw_manifest.get("output_sha256") != sha256(raw_path):
    raise SystemExit("raw quantile manifest/output mismatch")

for label, record in binding.get("analysis_outputs", {}).items():
    relative = Path(str(record.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe bound analysis output {label}: {relative}")
    path = root / relative
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise SystemExit(f"bound analysis output missing/size changed: {path}")
    if sha256(path) != record.get("sha256"):
        raise SystemExit(f"bound analysis output SHA changed: {path}")
if summary.get("input_inventory_digest_sha256") != inventory_digest:
    raise SystemExit("postprocess summary inventory digest mismatch")
if summary.get("prepared_input_config_content_sha256") != config.get("config_content_sha256"):
    raise SystemExit("postprocess summary prepared-config identity mismatch")
if summary.get("analysis_input_binding_status") != "BOUND_COMPLETE":
    raise SystemExit("postprocess summary binding status mismatch")

sealed = load(sealed_manifest_path)
metrics = root / "sealed_test" / str(sealed.get("metrics_file", ""))
if sealed.get("included_in_report_v1") is not False or not metrics.is_file():
    raise SystemExit("sealed aggregate manifest malformed")
if sha256(metrics) != sealed.get("sha256"):
    raise SystemExit("sealed aggregate SHA mismatch")
report_text = report_path.read_text(encoding="utf-8")
if "sealed_test/raw" in report_text or str(sealed.get("metrics_file", "")) in report_text:
    raise SystemExit("REPORT v1 leaks a sealed-test raw/aggregate path")
PY
}

verify_smoke_prepared() {
    [[ -f "$SMOKE_CONFIG" ]] || return 1
    "$PYTHON_BIN" - "$PILOT_CONFIG" "$SMOKE_CONFIG" "$SMOKE_WINDOWS_PER_DOMAIN" <<'PY'
import hashlib, json, sys

production_path, smoke_path, expected_windows = sys.argv[1:]
with open(production_path, encoding="utf-8") as handle:
    production = json.load(handle)
with open(smoke_path, encoding="utf-8") as handle:
    smoke = json.load(handle)

def payload_hash(value):
    value = dict(value)
    expected = value.pop("config_content_sha256", None)
    actual = hashlib.sha256(
        (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if expected != actual:
        raise SystemExit(f"prepared config content hash mismatch: {smoke_path}")
    return expected

payload_hash(smoke)
if smoke.get("schema") != "cka_gt_pilot_prepared_inputs_v1":
    raise SystemExit("checkpoint-smoke prepared schema mismatch")
if int(smoke.get("sample_windows_per_domain", -1)) != int(expected_windows):
    raise SystemExit("checkpoint-smoke must contain exactly 1,000 sampled windows per domain")
for key in (
    "code_prefix",
    "wiki_prefix",
    "before_checkpoint",
    "after_checkpoint",
    "base_seed",
    "layers",
    "position_ids",
    "split_policy",
    "test_policy",
    "window_policy",
    "chunk_policy",
    "numeric_policy",
    "representation_policy",
    "routing_policy",
    "membership_pass1",
    "consensus_policy",
    "source_dataset_identity",
    "checkpoint_identity",
):
    if smoke.get(key) != production.get(key):
        raise SystemExit(f"checkpoint-smoke config diverges from production: {key}")
PY
}

prepared_has_exact_identities() {
    local config="${1:-$PILOT_CONFIG}"
    "$PYTHON_BIN" - "$config" <<'PY'
import hashlib, json, sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text(encoding="utf-8"))
unhashed = dict(value)
expected = unhashed.pop("config_content_sha256", None)
canonical = (
    json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    + "\n"
).encode("utf-8")
if hashlib.sha256(canonical).hexdigest() != expected:
    raise SystemExit("prepared config content hash is invalid")
sources = value.get("source_dataset_identity")
checkpoints = value.get("checkpoint_identity")
if not isinstance(sources, dict) or set(sources) != {"code", "wiki"}:
    raise SystemExit(1)
if not isinstance(checkpoints, dict) or set(checkpoints) != {"before", "after"}:
    raise SystemExit(1)
for domain in ("code", "wiki"):
    identity = sources.get(domain, {})
    if identity.get("schema") != "cka_gt_pilot_source_dataset_identity_v1":
        raise SystemExit(1)
    for component in ("idx", "bin"):
        record = identity.get(component, {})
        if not record.get("sha256") or int(record.get("size_bytes", -1)) < 0:
            raise SystemExit(1)
for name in ("before", "after"):
    identity = checkpoints.get(name, {})
    if identity.get("schema") != "cka_gt_pilot_checkpoint_identity_v1":
        raise SystemExit(1)
    if not identity.get("content_sha256") or int(identity.get("total_bytes", -1)) < 0:
        raise SystemExit(1)
PY
}

ensure_prepared_exact_identities() {
    local label="$1" log="$2" upgraded=0
    if ! prepared_has_exact_identities "$PILOT_CONFIG"; then
        record UPGRADE "$label exact dataset/checkpoint identities are absent; prepared-only upgrade required"
        run_logged "${label}_identity_upgrade" "$log" \
            "$PYTHON_BIN" "$WINDOW_TOOL" --output-root "$OUT_ROOT" \
            --upgrade-source-identities
        upgraded=1
    fi
    prepared_has_exact_identities "$PILOT_CONFIG" \
        || die "$label did not produce exact dataset/checkpoint identities"
    if (( upgraded == 1 )); then
        # The upgrade command returns only after validate_prepared_pilot has
        # recomputed every source/checkpoint identity and artifact hash.
        record VERIFIED "$label identity upgrade completed its deep prepared validation"
    else
        run_logged "${label}_validate" "$log" \
            "$PYTHON_BIN" "$WINDOW_TOOL" --output-root "$OUT_ROOT" --validate-only
    fi
}

write_smoke_projection() {
    local pipeline_wall_seconds="$1"
    "$PYTHON_BIN" - "$OUT_ROOT" "$SMOKE_ROOT" "$SMOKE_PROJECTION" \
        "$pipeline_wall_seconds" <<'PY'
import hashlib, json, os, sys, tempfile, time
from pathlib import Path

import numpy as np

production = Path(sys.argv[1]).resolve()
smoke = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
pipeline_wall_seconds = float(sys.argv[4])

def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def split_stats(root, domain, names):
    windows = tokens = 0
    for name in names:
        rows = np.load(root / "splits" / f"{domain}_{name}_windows.npy", allow_pickle=False)
        windows += int(rows.shape[0])
        tokens += int(rows["window_length"].sum(dtype=np.int64))
    return {"windows": windows, "tokens": tokens}

def worker(root, mode, domain, split):
    value = load(root / "runtime" / mode / domain / split / "worker_000" / "metadata.json")
    if value.get("completed") is not True:
        raise SystemExit(f"incomplete checkpoint-smoke worker: {mode}/{domain}/{split}")
    return value

prod_config = production / "config.json"
smoke_config = smoke / "config.json"
pass1 = worker(smoke, "pass1", "wiki", "calibration")
code = worker(smoke, "pass2", "code", "all")
wiki = worker(smoke, "pass2", "wiki", "all")
worker_paths = {
    "pass1_wiki_calibration": smoke / "runtime" / "pass1" / "wiki" / "calibration" / "worker_000" / "metadata.json",
    "pass2_code": smoke / "runtime" / "pass2" / "code" / "all" / "worker_000" / "metadata.json",
    "pass2_wiki": smoke / "runtime" / "pass2" / "wiki" / "all" / "worker_000" / "metadata.json",
}

smoke_counts = {
    "pass1_wiki_calibration": split_stats(smoke, "wiki", ("calibration",)),
    "pass2_code": split_stats(smoke, "code", ("calibration", "selection", "test")),
    "pass2_wiki": split_stats(smoke, "wiki", ("calibration", "selection", "test")),
}
production_counts = {
    "pass1_wiki_calibration": split_stats(production, "wiki", ("calibration",)),
    "pass2_code": split_stats(production, "code", ("calibration", "selection", "test")),
    "pass2_wiki": split_stats(production, "wiki", ("calibration", "selection", "test")),
}
for domain_key in ("pass2_code", "pass2_wiki"):
    if smoke_counts[domain_key]["windows"] != 1000:
        raise SystemExit(f"checkpoint smoke does not contain 1,000 windows: {domain_key}")

def cumulative_time(name, value):
    raw = value.get("cumulative_elapsed_seconds", value.get("elapsed_seconds"))
    if raw is None:
        raise SystemExit(
            f"checkpoint-smoke cumulative timing unavailable for {name}; "
            "the recovered artifact is valid but cannot support an ETA projection"
        )
    return float(raw)

elapsed = {
    "pass1_wiki_calibration": cumulative_time("pass1_wiki_calibration", pass1),
    "pass2_code": cumulative_time("pass2_code", code),
    "pass2_wiki": cumulative_time("pass2_wiki", wiki),
}
if any(value <= 0 for value in elapsed.values()):
    raise SystemExit(f"checkpoint-smoke worker elapsed time missing/nonpositive: {elapsed}")
throughput = {
    name: smoke_counts[name]["tokens"] / elapsed[name]
    for name in elapsed
}

# Production Pass2 uses four equally partitioned world-size-1 workers.  The
# projection therefore scales the single-GPU smoke time by target/smoke token
# count and divides the two Pass2 domain stages by four; Pass1 remains one GPU.
projected_seconds = {
    "pass1_wiki_calibration_one_gpu": (
        elapsed["pass1_wiki_calibration"]
        * production_counts["pass1_wiki_calibration"]["tokens"]
        / smoke_counts["pass1_wiki_calibration"]["tokens"]
    ),
    "pass2_code_four_gpu_wall": (
        elapsed["pass2_code"]
        * production_counts["pass2_code"]["tokens"]
        / smoke_counts["pass2_code"]["tokens"]
        / 4.0
    ),
    "pass2_wiki_four_gpu_wall": (
        elapsed["pass2_wiki"]
        * production_counts["pass2_wiki"]["tokens"]
        / smoke_counts["pass2_wiki"]["tokens"]
        / 4.0
    ),
}
projected_seconds["total_sequential_wall"] = sum(projected_seconds.values())

metric_roots = (
    smoke / "token_metrics",
    smoke / "chunk_metrics",
    smoke / "sealed_test" / "raw" / "token_metrics",
    smoke / "sealed_test" / "raw" / "chunk_metrics",
)
metric_files = [
    path for root in metric_roots if root.exists()
    for path in root.rglob("*.parquet") if path.is_file()
]
metric_bytes = sum(path.stat().st_size for path in metric_files)
smoke_pass2_tokens = smoke_counts["pass2_code"]["tokens"] + smoke_counts["pass2_wiki"]["tokens"]
production_pass2_tokens = (
    production_counts["pass2_code"]["tokens"] + production_counts["pass2_wiki"]["tokens"]
)
projected_metric_bytes = int(round(metric_bytes * production_pass2_tokens / smoke_pass2_tokens))
artifact_bytes = sum(
    path.stat().st_size for path in smoke.rglob("*")
    if path.is_file() and path.name != output.name
)

inventory = load(smoke / "input_parquet_inventory.json")
binding = load(smoke / "analysis_input_binding.json")
payload = {
    "schema": "cka_gt_pilot_checkpoint_smoke_projection_v1",
    "status": "PASS_GATE",
    "generated_unix_time": time.time(),
    "production_root": str(production),
    "smoke_root": str(smoke),
    "production_config_sha256": sha256(prod_config),
    "smoke_config_sha256": sha256(smoke_config),
    "smoke_windows_per_domain": 1000,
    "smoke_counts": smoke_counts,
    "production_counts": production_counts,
    "runtime_model_driver_cumulative_elapsed_seconds": elapsed,
    "runtime_tokens_per_second": throughput,
    "worker_metadata_sha256": {
        name: sha256(path) for name, path in worker_paths.items()
    },
    "pipeline_wall_seconds_current_invocation": pipeline_wall_seconds,
    "timing_basis": (
        "Runtime elapsed is crash-resumable cumulative committed model/metric time "
        "from completed worker metadata. "
        "Pipeline wall time includes preparation, checkpoint startup, validation, and analysis "
        "performed in this invocation; a resumed invocation may be shorter."
    ),
    "projected_full_elapsed_seconds": projected_seconds,
    "storage": {
        "smoke_metric_parquet_bytes": metric_bytes,
        "smoke_metric_parquet_file_count": len(metric_files),
        "smoke_total_artifact_bytes": artifact_bytes,
        "projected_full_metric_parquet_bytes": projected_metric_bytes,
        "projection_basis": "aggregate open+physically-sealed scalar Parquet bytes per processed token",
    },
    "test_policy": {
        "physical_test_seal_verified": True,
        "report_v1_test_metrics_exposed": False,
        "aggregate_only_storage_accounting": True,
    },
    "analysis_binding": {
        "status": binding.get("status"),
        "inventory_digest_sha256": inventory.get("inventory_digest_sha256"),
        "binding_file_sha256": sha256(smoke / "analysis_input_binding.json"),
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    handle.write("\n")
    handle.flush(); os.fsync(handle.fileno())
os.replace(temporary, output)
PY
}

verify_smoke_projection() {
    "$PYTHON_BIN" - "$PILOT_CONFIG" "$SMOKE_CONFIG" "$SMOKE_PROJECTION" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

production_config, smoke_config, projection_path = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

if not projection_path.is_file():
    raise SystemExit(1)
with projection_path.open(encoding="utf-8") as handle:
    value = json.load(handle)
evidence_path = smoke_config.parent / "report_validation_evidence.json"
if not evidence_path.is_file():
    raise SystemExit("checkpoint-smoke REPORT validation evidence is missing")
with evidence_path.open(encoding="utf-8") as handle:
    evidence = json.load(handle)
probe = evidence.get("probe", {})
if probe.get("status") != "PASS" or probe.get("source_kind") != "runtime_pass2_router_probe_metadata":
    raise SystemExit(f"checkpoint-smoke REPORT router probe is not PASS via Pass2: {probe}")
if value.get("schema") != "cka_gt_pilot_checkpoint_smoke_projection_v1" or value.get("status") != "PASS_GATE":
    raise SystemExit("checkpoint-smoke projection is not a PASS gate")
if value.get("smoke_windows_per_domain") != 1000:
    raise SystemExit("checkpoint-smoke projection count mismatch")
if value.get("production_config_sha256") != sha256(production_config):
    raise SystemExit("checkpoint-smoke gate is stale relative to production config")
if value.get("smoke_config_sha256") != sha256(smoke_config):
    raise SystemExit("checkpoint-smoke gate is stale relative to smoke config")
if value.get("analysis_binding", {}).get("status") != "BOUND_COMPLETE":
    raise SystemExit("checkpoint-smoke analysis binding is incomplete")
inventory_path = smoke_config.parent / "input_parquet_inventory.json"
binding_path = smoke_config.parent / "analysis_input_binding.json"
with inventory_path.open(encoding="utf-8") as handle:
    inventory = json.load(handle)
if value.get("analysis_binding", {}).get("inventory_digest_sha256") != inventory.get("inventory_digest_sha256"):
    raise SystemExit("checkpoint-smoke projection is stale relative to the current Parquet inventory")
if value.get("analysis_binding", {}).get("binding_file_sha256") != sha256(binding_path):
    raise SystemExit("checkpoint-smoke projection is stale relative to analysis binding")
worker_paths = {
    "pass1_wiki_calibration": smoke_config.parent / "runtime" / "pass1" / "wiki" / "calibration" / "worker_000" / "metadata.json",
    "pass2_code": smoke_config.parent / "runtime" / "pass2" / "code" / "all" / "worker_000" / "metadata.json",
    "pass2_wiki": smoke_config.parent / "runtime" / "pass2" / "wiki" / "all" / "worker_000" / "metadata.json",
}
if value.get("worker_metadata_sha256") != {
    name: sha256(path) for name, path in worker_paths.items()
}:
    raise SystemExit("checkpoint-smoke projection is stale relative to worker metadata")
metric_roots = (
    smoke_config.parent / "token_metrics",
    smoke_config.parent / "chunk_metrics",
    smoke_config.parent / "sealed_test" / "raw" / "token_metrics",
    smoke_config.parent / "sealed_test" / "raw" / "chunk_metrics",
)
current_metric_bytes = sum(
    path.stat().st_size
    for root in metric_roots if root.exists()
    for path in root.rglob("*.parquet") if path.is_file()
)
if int(value.get("storage", {}).get("smoke_metric_parquet_bytes", -1)) != current_metric_bytes:
    raise SystemExit("checkpoint-smoke projection storage is stale relative to scalar Parquet")
if value.get("test_policy", {}).get("report_v1_test_metrics_exposed") is not False:
    raise SystemExit("checkpoint-smoke test policy was not sealed")
if int(value.get("storage", {}).get("smoke_metric_parquet_bytes", 0)) <= 0:
    raise SystemExit("checkpoint-smoke did not record scalar Parquet storage")
if float(value.get("projected_full_elapsed_seconds", {}).get("total_sequential_wall", 0.0)) <= 0:
    raise SystemExit("checkpoint-smoke did not record a full ETA projection")
PY
}

verify_smoke_postprocess() {
    verify_smoke_prepared || return 1
    verify_pass1_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
        "$SMOKE_PASS1_BATCH_SIZE" || return 1
    verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
        code "$SMOKE_WINDOW_BATCH_SIZE" || return 1
    verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
        wiki "$SMOKE_WINDOW_BATCH_SIZE" || return 1
    [[ -f "$SMOKE_VALIDATION_ROOT/07_full_pre_analysis.json" ]] || return 1
    assert_validation_json "$SMOKE_VALIDATION_ROOT/07_full_pre_analysis.json" full 4 || return 1
    verify_analysis "$SMOKE_ROOT" || return 1
    [[ -f "$SMOKE_VALIDATION_ROOT/08_final_after_analysis.json" ]] || return 1
    assert_validation_json \
        "$SMOKE_VALIDATION_ROOT/08_final_after_analysis.json" final_shallow 4 \
        || return 1
    "$PYTHON_BIN" - "$SMOKE_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
full = root / "validation" / "07_full_pre_analysis.json"
inventory = root / "input_parquet_inventory.json"
binding = root / "analysis_input_binding.json"
final = root / "validation" / "08_final_after_analysis.json"
metric_roots = (
    root / "token_metrics",
    root / "chunk_metrics",
    root / "sealed_test" / "raw" / "token_metrics",
    root / "sealed_test" / "raw" / "chunk_metrics",
)
parquet = [
    path
    for directory in metric_roots
    if directory.exists()
    for path in directory.rglob("*.parquet")
    if path.is_file()
]
if not parquet:
    raise SystemExit("checkpoint-smoke metric inventory is empty")
if full.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in parquet):
    raise SystemExit("checkpoint-smoke deep validation predates a metric shard")
if inventory.stat().st_mtime_ns < full.stat().st_mtime_ns:
    raise SystemExit("checkpoint-smoke analysis inventory predates deep validation")
if binding.stat().st_mtime_ns < inventory.stat().st_mtime_ns:
    raise SystemExit("checkpoint-smoke analysis binding predates its inventory")
if final.stat().st_mtime_ns < binding.stat().st_mtime_ns:
    raise SystemExit("checkpoint-smoke final validation predates analysis binding")
PY
}

verify_smoke_1000() {
    verify_smoke_postprocess || return 1
    verify_smoke_projection
}

stage_prepared_validate() {
    ensure_prepared_exact_identities prepared "$LOG_ROOT/01_prepared_validate.log"
    PREPARED_VALIDATED_THIS_INVOCATION=1
    mark_stage 1 complete
}

stage_router_smoke() {
    if (( PREPARED_VALIDATED_THIS_INVOCATION == 1 )) \
            && prepared_has_exact_identities "$PILOT_CONFIG"; then
        record SKIP "prepared prerequisite was physically validated in this launcher invocation"
    else
        ensure_prepared_exact_identities prepared_prerequisite "$LOG_ROOT/01_prepared_validate.log"
        PREPARED_VALIDATED_THIS_INVOCATION=1
    fi
    if verify_router_smoke; then
        record SKIP "router smoke already verified"
        mark_stage 2 skipped_verified
        return
    fi
    assert_gpu_idle 0
    port_is_free "$ROUTER_SMOKE_PORT" || die "router-smoke port occupied: $ROUTER_SMOKE_PORT"
    run_logged router_smoke "$LOG_ROOT/02_router_smoke.launch.log" \
        env GPU=0 MODE=router_smoke DOMAIN=code SPLIT=calibration \
        WORKER_INDEX=0 WORKER_COUNT=1 MAX_WINDOWS=1 \
        WINDOW_BATCH_SIZE=1 SHARD_WINDOWS=1 MASTER_PORT="$ROUTER_SMOKE_PORT" \
        OUT_ROOT="$OUT_ROOT" PILOT_CONFIG="$PILOT_CONFIG" MEMBERSHIP_STATS="$MEMBERSHIP_STATS" \
        SEED="$SEED" bash "$RUNNER"
    verify_router_smoke || die "router smoke returned without verified metadata"
    mark_stage 2 complete
}

stage_checkpoint_smoke_1000() {
    local started_epoch now_epoch wall_seconds
    local -a source_fields=()
    started_epoch="$(date +%s)"

    if verify_smoke_1000; then
        record SKIP "isolated 1,000+1,000 real-checkpoint smoke already verified"
        mark_stage 3 skipped_verified
        return
    fi
    if verify_smoke_postprocess; then
        now_epoch="$(date +%s)"
        wall_seconds="$((now_epoch - started_epoch))"
        record REUSE \
            "checkpoint-smoke deep validation + bound analysis inventory are current; writing projection only"
        write_smoke_projection "$wall_seconds"
        verify_smoke_1000 || die "checkpoint-smoke projection did not bind to reused evidence"
        mark_stage 3 complete
        return
    fi

    mkdir -p "$SMOKE_LOG_ROOT" "$SMOKE_VALIDATION_ROOT" "$SMOKE_ROOT/membership"
    mapfile -t source_fields < <(
        "$PYTHON_BIN" - "$PILOT_CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)
for key in ("code_prefix", "wiki_prefix", "before_checkpoint", "after_checkpoint"):
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise SystemExit(f"production config field missing: {key}")
    print(item)
PY
    )
    [[ "${#source_fields[@]}" == 4 ]] || die "could not read production inputs for checkpoint smoke"

    run_logged checkpoint_smoke_prepare "$SMOKE_LOG_ROOT/00_prepare.log" \
        "$PYTHON_BIN" "$WINDOW_TOOL" --output-root "$SMOKE_ROOT" \
        --code-prefix "${source_fields[0]}" --wiki-prefix "${source_fields[1]}" \
        --before-checkpoint "${source_fields[2]}" --after-checkpoint "${source_fields[3]}" \
        --seed "$SEED" --sample-windows-per-domain "$SMOKE_WINDOWS_PER_DOMAIN"
    verify_smoke_prepared || die "isolated checkpoint-smoke prepared inputs failed identity validation"

    # Only require idle GPUs when an actual checkpoint forward remains.  A
    # resumed smoke that only needs CPU postprocessing is allowed to finish
    # without claiming GPU 0.
    if ! verify_pass1_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            "$SMOKE_PASS1_BATCH_SIZE" \
        || ! verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            code "$SMOKE_WINDOW_BATCH_SIZE" \
        || ! verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            wiki "$SMOKE_WINDOW_BATCH_SIZE"; then
        assert_gpu_idle 0
    fi

    if verify_pass1_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            "$SMOKE_PASS1_BATCH_SIZE"; then
        record SKIP "checkpoint-smoke Pass 1 already verified"
    else
        port_is_free "$CHECKPOINT_SMOKE_PASS1_PORT" \
            || die "checkpoint-smoke Pass-1 port occupied: $CHECKPOINT_SMOKE_PASS1_PORT"
        run_logged checkpoint_smoke_pass1 "$SMOKE_LOG_ROOT/01_pass1_wiki_calibration.launch.log" \
            env GPU=0 MODE=pass1 DOMAIN=wiki SPLIT=calibration \
            WORKER_INDEX=0 WORKER_COUNT=1 MAX_WINDOWS=0 \
            WINDOW_BATCH_SIZE="$SMOKE_PASS1_BATCH_SIZE" SHARD_WINDOWS="$SHARD_WINDOWS" \
            MASTER_PORT="$CHECKPOINT_SMOKE_PASS1_PORT" \
            OUT_ROOT="$SMOKE_ROOT" PILOT_CONFIG="$SMOKE_CONFIG" \
            MEMBERSHIP_STATS="$SMOKE_MEMBERSHIP_STATS" \
            REFERENCE_LOAD="${source_fields[2]}" CURRENT_LOAD="${source_fields[3]}" \
            CODE_PREFIX="${source_fields[0]}" WIKI_PREFIX="${source_fields[1]}" \
            SEED="$SEED" bash "$RUNNER"
        verify_pass1_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            "$SMOKE_PASS1_BATCH_SIZE" || die "checkpoint-smoke Pass 1 metadata/statistics are incomplete"
    fi

    local smoke_membership="$SMOKE_VALIDATION_ROOT/04_membership_validation.json"
    run_logged checkpoint_smoke_membership_validate "$SMOKE_LOG_ROOT/02_membership_validate.log" \
        "$PYTHON_BIN" "$VALIDATOR" "$SMOKE_ROOT" --allow-incomplete --deep \
        --output-json "$smoke_membership"
    assert_validation_json "$smoke_membership" membership 4

    if verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            code "$SMOKE_WINDOW_BATCH_SIZE"; then
        record SKIP "checkpoint-smoke Code Pass 2 already verified"
    else
        port_is_free "$CHECKPOINT_SMOKE_CODE_PORT" \
            || die "checkpoint-smoke Code port occupied: $CHECKPOINT_SMOKE_CODE_PORT"
        run_logged checkpoint_smoke_pass2_code "$SMOKE_LOG_ROOT/03_pass2_code.launch.log" \
            env GPU=0 MODE=pass2 DOMAIN=code SPLIT=all \
            WORKER_INDEX=0 WORKER_COUNT=1 MAX_WINDOWS=0 \
            WINDOW_BATCH_SIZE="$SMOKE_WINDOW_BATCH_SIZE" SHARD_WINDOWS="$SHARD_WINDOWS" \
            MASTER_PORT="$CHECKPOINT_SMOKE_CODE_PORT" \
            OUT_ROOT="$SMOKE_ROOT" PILOT_CONFIG="$SMOKE_CONFIG" \
            MEMBERSHIP_STATS="$SMOKE_MEMBERSHIP_STATS" \
            REFERENCE_LOAD="${source_fields[2]}" CURRENT_LOAD="${source_fields[3]}" \
            CODE_PREFIX="${source_fields[0]}" WIKI_PREFIX="${source_fields[1]}" \
            SEED="$SEED" bash "$RUNNER"
        verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            code "$SMOKE_WINDOW_BATCH_SIZE" || die "checkpoint-smoke Code Pass 2 is incomplete"
    fi

    if verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            wiki "$SMOKE_WINDOW_BATCH_SIZE"; then
        record SKIP "checkpoint-smoke Wiki Pass 2 already verified"
    else
        port_is_free "$CHECKPOINT_SMOKE_WIKI_PORT" \
            || die "checkpoint-smoke Wiki port occupied: $CHECKPOINT_SMOKE_WIKI_PORT"
        run_logged checkpoint_smoke_pass2_wiki "$SMOKE_LOG_ROOT/04_pass2_wiki.launch.log" \
            env GPU=0 MODE=pass2 DOMAIN=wiki SPLIT=all \
            WORKER_INDEX=0 WORKER_COUNT=1 MAX_WINDOWS=0 \
            WINDOW_BATCH_SIZE="$SMOKE_WINDOW_BATCH_SIZE" SHARD_WINDOWS="$SHARD_WINDOWS" \
            MASTER_PORT="$CHECKPOINT_SMOKE_WIKI_PORT" \
            OUT_ROOT="$SMOKE_ROOT" PILOT_CONFIG="$SMOKE_CONFIG" \
            MEMBERSHIP_STATS="$SMOKE_MEMBERSHIP_STATS" \
            REFERENCE_LOAD="${source_fields[2]}" CURRENT_LOAD="${source_fields[3]}" \
            CODE_PREFIX="${source_fields[0]}" WIKI_PREFIX="${source_fields[1]}" \
            SEED="$SEED" bash "$RUNNER"
        verify_pass2_at "$SMOKE_ROOT" "$SMOKE_CONFIG" "$SMOKE_MEMBERSHIP_STATS" \
            wiki "$SMOKE_WINDOW_BATCH_SIZE" || die "checkpoint-smoke Wiki Pass 2 is incomplete"
    fi

    local smoke_full="$SMOKE_VALIDATION_ROOT/07_full_pre_analysis.json"
    run_logged checkpoint_smoke_full_validate "$SMOKE_LOG_ROOT/05_full_validate.log" \
        "$PYTHON_BIN" "$VALIDATOR" "$SMOKE_ROOT" --allow-incomplete --deep \
        --output-json "$smoke_full"
    assert_validation_json "$smoke_full" full 4

    run_logged checkpoint_smoke_analyze "$SMOKE_LOG_ROOT/06_analyze.log" \
        "$PYTHON_BIN" "$ANALYZER" "$SMOKE_ROOT" --output-root "$SMOKE_ROOT" \
        --token-metrics "$SMOKE_TOKEN_METRICS" --chunk-metrics "$SMOKE_CHUNK_METRICS" \
        --raw-quantiles "$SMOKE_RAW_QUANTILES" --bins 256 --seed "$SEED"
    verify_analysis "$SMOKE_ROOT" || die "checkpoint-smoke analyzer output binding is incomplete"

    local smoke_final="$SMOKE_VALIDATION_ROOT/08_final_after_analysis.json"
    run_logged checkpoint_smoke_final_validate "$SMOKE_LOG_ROOT/07_final_validate.log" \
        "$PYTHON_BIN" "$VALIDATOR" "$SMOKE_ROOT" --output-json "$smoke_final"
    assert_validation_json "$smoke_final" final_shallow 4

    now_epoch="$(date +%s)"
    wall_seconds="$((now_epoch - started_epoch))"
    write_smoke_projection "$wall_seconds"
    verify_smoke_1000 || die "checkpoint-smoke gate did not verify after completion"
    record DONE "checkpoint-smoke gate PASS projection=$SMOKE_PROJECTION"
    mark_stage 3 complete
}

stage_pass1() {
    verify_smoke_1000 \
        || die "production Pass 1 is gated on the isolated 1,000+1,000 checkpoint smoke"
    verify_router_smoke || die "Pass 1 requires completed router smoke; start at router_smoke"
    if verify_pass1; then
        record SKIP "Pass 1 membership artifact already verified by metadata"
        mark_stage 4 skipped_verified
        return
    fi
    assert_gpu_idle 0
    port_is_free "$PASS1_PORT" || die "Pass-1 port occupied: $PASS1_PORT"
    run_logged pass1_wiki_calibration "$LOG_ROOT/03_pass1_wiki_calibration.launch.log" \
        env GPU=0 MODE=pass1 DOMAIN=wiki SPLIT=calibration \
        WORKER_INDEX=0 WORKER_COUNT=1 MAX_WINDOWS=0 \
        WINDOW_BATCH_SIZE="$PASS1_BATCH_SIZE" SHARD_WINDOWS="$SHARD_WINDOWS" MASTER_PORT="$PASS1_PORT" \
        OUT_ROOT="$OUT_ROOT" PILOT_CONFIG="$PILOT_CONFIG" MEMBERSHIP_STATS="$MEMBERSHIP_STATS" \
        SEED="$SEED" bash "$RUNNER"
    verify_pass1 || die "Pass 1 returned without membership stats/metadata"
    mark_stage 4 complete
}

stage_membership_validate() {
    verify_smoke_1000 || die "membership validation is gated on the isolated checkpoint smoke"
    verify_pass1 || die "membership validation requires completed Pass 1"
    local output="$VALIDATION_ROOT/04_membership_validation.json"
    run_logged membership_validate "$LOG_ROOT/04_membership_validate.log" \
        "$PYTHON_BIN" "$VALIDATOR" "$OUT_ROOT" --allow-incomplete --deep --output-json "$output"
    assert_validation_json "$output" membership
    mark_stage 5 complete
}

launch_pass2_domain() {
    local domain="$1" base_port="$2" stage_index="$3" label="$4"
    local worker gpu port log pid overall
    if verify_pass2_domain "$domain"; then
        record SKIP "$label all four workers already verified"
        mark_stage "$stage_index" skipped_verified
        return
    fi
    verify_smoke_1000 || die "$label is gated on the isolated checkpoint smoke"
    verify_pass1 || die "$label requires completed Pass 1"
    [[ -f "$VALIDATION_ROOT/04_membership_validation.json" ]] \
        || die "$label requires membership_validate stage"
    assert_validation_json "$VALIDATION_ROOT/04_membership_validation.json" membership
    assert_gpus_0_3_idle

    declare -a pids=() workers=()
    for worker in 0 1 2 3; do
        if verify_pass2_worker "$domain" "$worker"; then
            record SKIP "$label worker $worker already verified"
            continue
        fi
        gpu="$worker"
        port="$((base_port + worker))"
        port_is_free "$port" || die "$label worker $worker port occupied: $port"
        log="$LOG_ROOT/${stage_index}_${label}_worker_$(printf '%03d' "$worker").launch.log"
        record START "$label worker=$worker GPU=$gpu port=$port log=$log"
        env GPU="$gpu" MODE=pass2 DOMAIN="$domain" SPLIT=all \
            WORKER_INDEX="$worker" WORKER_COUNT=4 MAX_WINDOWS=0 \
            WINDOW_BATCH_SIZE="$WINDOW_BATCH_SIZE" SHARD_WINDOWS="$SHARD_WINDOWS" MASTER_PORT="$port" \
            OUT_ROOT="$OUT_ROOT" PILOT_CONFIG="$PILOT_CONFIG" MEMBERSHIP_STATS="$MEMBERSHIP_STATS" \
            SEED="$SEED" bash "$RUNNER" >"$log" 2>&1 &
        pids+=("$!")
        workers+=("$worker")
    done

    overall=0
    for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        worker="${workers[$index]}"
        if wait "$pid"; then
            record DONE "$label worker=$worker pid=$pid"
        else
            rc=$?
            record FAIL "$label worker=$worker pid=$pid rc=$rc"
            overall=1
        fi
    done
    (( overall == 0 )) || die "$label had failed worker(s); completed workers remain resumable"
    verify_pass2_domain "$domain" || die "$label returned without four verified worker metadata files"
    mark_stage "$stage_index" complete
}

stage_full_validate() {
    verify_smoke_1000 || die "full validator is gated on the isolated checkpoint smoke"
    verify_pass2_domain code || die "full validator requires all Code workers"
    verify_pass2_domain wiki || die "full validator requires all Wiki workers"
    local output="$VALIDATION_ROOT/07_full_pre_analysis.json"
    run_logged full_validate "$LOG_ROOT/07_full_validate.log" \
        "$PYTHON_BIN" "$VALIDATOR" "$OUT_ROOT" --allow-incomplete --deep --output-json "$output"
    assert_validation_json "$output" full
    FULL_VALIDATED_THIS_INVOCATION=1
    mark_stage 8 complete
}

stage_analyze() {
    verify_smoke_1000 || die "analysis is gated on the isolated checkpoint smoke"
    local full="$VALIDATION_ROOT/07_full_pre_analysis.json" final_kind="final"
    local -a final_flags=(--deep)
    if (( FULL_VALIDATED_THIS_INVOCATION == 1 )); then
        final_kind="final_shallow"
        final_flags=()
    fi
    [[ -f "$full" ]] || die "analysis requires full_validate stage"
    assert_validation_json "$full" full
    if verify_analysis && [[ -f "$VALIDATION_ROOT/08_final_after_analysis.json" ]] \
            && assert_validation_json \
                "$VALIDATION_ROOT/08_final_after_analysis.json" "$final_kind"; then
        record SKIP "analysis and final validation already verified"
        mark_stage 9 skipped_verified
        return
    fi
    run_logged analyze "$LOG_ROOT/08_analyze.log" \
        "$PYTHON_BIN" "$ANALYZER" "$OUT_ROOT" --output-root "$OUT_ROOT" \
        --token-metrics "$TOKEN_METRICS" --chunk-metrics "$CHUNK_METRICS" \
        --raw-quantiles "$RAW_QUANTILES" --bins 256 --seed "$SEED"
    verify_analysis || die "analysis returned without complete report/selector/sealed artifacts"
    local final="$VALIDATION_ROOT/08_final_after_analysis.json"
    if (( FULL_VALIDATED_THIS_INVOCATION == 1 )); then
        # Stage 8 deep-scanned every scalar row in this process.  The analysis
        # verifier above re-hashed the complete open + sealed inventory, so
        # only structural/hash/sealed-report checks remain.
        record VERIFIED "current-invocation deep validation permits shallow final validation"
    else
        record VERIFIED "analyze-only resume uses deep final validation for stale-evidence safety"
    fi
    run_logged final_validate "$LOG_ROOT/08_final_validate.log" \
        "$PYTHON_BIN" "$VALIDATOR" "$OUT_ROOT" "${final_flags[@]}" --output-json "$final"
    assert_validation_json "$final" "$final_kind"
    mark_stage 9 complete
}

for stage in $(seq "$START_INDEX" "$STOP_INDEX"); do
    record STAGE "$stage ${STAGE_NAMES[$((stage - 1))]}"
    if (( stage >= 2 && PREPARED_VALIDATED_THIS_INVOCATION == 0 )); then
        ensure_prepared_exact_identities prepared_prerequisite \
            "$LOG_ROOT/01_prepared_validate.log"
        PREPARED_VALIDATED_THIS_INVOCATION=1
    fi
    case "$stage" in
        1) stage_prepared_validate ;;
        2) stage_router_smoke ;;
        3) stage_checkpoint_smoke_1000 ;;
        4) stage_pass1 ;;
        5) stage_membership_validate ;;
        6) launch_pass2_domain code "$CODE_PORT_BASE" 6 pass2_code ;;
        7)
            verify_pass2_domain code || die "Wiki Pass 2 requires completed Code Pass 2"
            launch_pass2_domain wiki "$WIKI_PORT_BASE" 7 pass2_wiki
            ;;
        8) stage_full_validate ;;
        9) stage_analyze ;;
        *) die "internal stage dispatch error: $stage" ;;
    esac
done

record COMPLETE "requested stage interval $START_INDEX..$STOP_INDEX completed"
