#!/bin/bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$D/../../.." && pwd)"

CHAIN_PID="${CHAIN_PID:?set CHAIN_PID to the running LPR chain shell PID}"
POLL_SECONDS="${POLL_SECONDS:-30}"
WATCH_DIR="${WATCH_DIR:-$PROJECT_ROOT/.local/logs/lpr_to_kd_ramp900_watcher}"
NEXT_CHAIN="$D/run_g2_ffn_only_kd_ramp900_code_kd_ramp900_conversation_chain_mha.sh"
AGENT_DATA_SCRIPT="${AGENT_DATA_SCRIPT:-/home/work/Agent_HJ/30_flame_agent/agent_data_make.py}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"

G2_ROOT="${G2_ROOT:-$PROJECT_ROOT/.local/weights/a100/mha/g2-checkpoints}"
LPR_ROOT="${LPR_ROOT:-$G2_ROOT/lpr_chain}"

CODE_OUT="$LPR_ROOT/code_task/g2-ffn-only-e8to16-code-lm-aux-z-mb96-1800-lpr-chain"
CODE_LPR_OUT="$LPR_ROOT/code_router/g2-ffn-only-code-router-lpr-gamma0.1-equal-token-mb96-360"
CONV_OUT="$LPR_ROOT/conversation_task/g2-ffn-only-e16to24-conversation-lm-aux-z-mb64-1800-from-lpr"
CONV_LPR_OUT="$LPR_ROOT/conversation_router/g2-ffn-only-conversation-router-lpr-gamma0.1-equal-token-mb64-360"

RAMP_CODE_OUT="$G2_ROOT/code/joint_lm_replay_ramp/g2-ffn-only-code-wiki-joint-lm-allrouter-newexpert-ramp900-mb96-1800"
RAMP_KD_OUT="$G2_ROOT/conversation/expansion_distill_init_joint_code_ramp/g2-ffn-only-e16to24-conv-init-from-code-ramp900-logits-wikicode-kd-mb32-600"
RAMP_CONV_OUT="$G2_ROOT/conversation/joint_lm_replay_ramp/g2-ffn-only-conv-wikicode-joint-lm-allrouter-112-newexpert-ramp900-mb96-1800"

mkdir -p "$WATCH_DIR"
exec 9>"$WATCH_DIR/watcher.lock"
if ! flock -n 9; then
    echo "[$(date -Is)] ERROR: another LPR-to-ramp900 watcher already holds the lock"
    exit 2
fi
echo "$$" > "$WATCH_DIR/watcher.pid"

log() {
    echo "[$(date -Is)] $*"
}

same_lpr_chain() {
    [ -r "/proc/$CHAIN_PID/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$CHAIN_PID/cmdline" \
        | grep -Fq 'run_g2_ffn_only_lpr_1800_360_1800_360_chain_mha.sh'
}

check_step() {
    local label="$1"
    local checkpoint="$2"
    local expected="$3"
    local tracker="$checkpoint/latest_checkpointed_iteration.txt"
    local actual=""

    if [ -f "$tracker" ]; then
        actual="$(tr -d '\n\r[:space:]' < "$tracker")"
    fi
    if [ "$actual" != "$expected" ]; then
        log "ERROR: $label incomplete; expected=$expected actual=${actual:-missing}"
        log "ERROR: checkpoint=$checkpoint"
        return 1
    fi
    log "OK: $label completed at step $actual"
}

if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    log "ERROR: POLL_SECONDS must be a positive integer"
    exit 2
fi
if [ ! -x "$NEXT_CHAIN" ]; then
    log "ERROR: next chain is missing or not executable: $NEXT_CHAIN"
    exit 2
fi
if [ ! -f "$AGENT_DATA_SCRIPT" ]; then
    log "ERROR: agent data script is missing: $AGENT_DATA_SCRIPT"
    exit 2
fi
if [ ! -x "$PYTHON_BIN" ]; then
    log "ERROR: Python is not executable: $PYTHON_BIN"
    exit 2
fi
if ! same_lpr_chain; then
    log "ERROR: PID $CHAIN_PID is not the active LPR chain"
    exit 2
fi

log "Watching LPR chain PID $CHAIN_PID"
log "Next chain will run with WANDB_MODE=offline"
log "Next chain: $NEXT_CHAIN"
log "After ramp900 completion: $AGENT_DATA_SCRIPT"

while same_lpr_chain; do
    # Do not let the polling child inherit the watcher lock. This allows a
    # stopped watcher to be replaced immediately instead of waiting for sleep.
    sleep "$POLL_SECONDS" 9>&-
done

log "LPR chain exited; verifying all four completed checkpoints"
status=0
check_step code_task "$CODE_OUT" 1800 || status=1
check_step code_router_lpr "$CODE_LPR_OUT" 2160 || status=1
check_step conversation_task "$CONV_OUT" 1800 || status=1
check_step conversation_router_lpr "$CONV_LPR_OUT" 2160 || status=1
if [ "$status" -ne 0 ]; then
    log "ERROR: LPR chain was not fully successful; ramp900 chain will not start"
    exit 1
fi

while pgrep -f '[t]orch.distributed.run.*--master_port 2993[0-3]' >/dev/null; do
    log "Waiting for LPR torchrun workers to exit"
    sleep "$POLL_SECONDS" 9>&-
done

sync
log "Starting KD ramp900 chain on GPUs 0,1,2,3"
cd "$PROJECT_ROOT"
set +e
env \
    WANDB_MODE=offline \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    bash "$NEXT_CHAIN"
ramp_status=$?
set -e

if [ "$ramp_status" -ne 0 ]; then
    log "ERROR: KD ramp900 chain failed with exit=$ramp_status; agent_data_make.py will not start"
    exit "$ramp_status"
fi

log "KD ramp900 chain exited successfully; verifying all three checkpoints"
status=0
check_step ramp_code_joint "$RAMP_CODE_OUT" 1800 || status=1
check_step ramp_conversation_kd "$RAMP_KD_OUT" 600 || status=1
check_step ramp_conversation_joint "$RAMP_CONV_OUT" 1800 || status=1
if [ "$status" -ne 0 ]; then
    log "ERROR: KD ramp900 chain checkpoints are incomplete; agent_data_make.py will not start"
    exit 1
fi

if pgrep -f '[a]gent_data_make.py' >/dev/null; then
    log "agent_data_make.py is already running; refusing to start a duplicate"
    exit 0
fi

sync
log "Starting agent_data_make.py on GPUs 0,1,2,3"
cd "$(dirname "$AGENT_DATA_SCRIPT")"
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$PYTHON_BIN" -u "$AGENT_DATA_SCRIPT" --gpus 4
