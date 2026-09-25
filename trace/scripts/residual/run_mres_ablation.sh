#!/usr/bin/env bash
# Mass-reservoir ablations: one arm = one full 8-task run_residual_chain.sh run.
#
# Every arm is the method (left side) with exactly one factor switched:
#   ARM=main_gen    the method: self-generated replay (GEN_PROTOCOL=v1)
#   ARM=main_real   replay gen -> real (trainer's fixed replay memory, 500 real records/task);
#                   also the reference arm for every arm below, which all use real replay
#   ARM=random_row  new router row: clone(r_res) -> stock nn.Linear random init (MRES_NEW_ROW)
#   ARM=no_margin   margin loss weight lambda -> 0 (MRES_LAMBDA=0; margin stats still logged)
#   ARM=posthoc     joint router correction -> post-hoc: the task trains the primary branch
#                   alone, then a router-only pass replays exactly the joint arm's per-update
#                   router-FT batches (same records, same current slices, same update count)
#   ARM=distill     router-FT objective on replay/BoS records: LM -> per-layer KL to the
#                   pre-expansion router (RESIDUAL_DISTILL_PAD=row: zero teacher row per new
#                   expert; prob: zero teacher probability); current-task slice keeps LM
#
#   ARM=posthoc RUN_ROOT=/data2/.../mres_ablation GPUS=0,1,2,3,4,5,6,7 bash scripts/residual/run_mres_ablation.sh
#
# The method's own knobs (MRES_DELTA, MRES_LAMBDA, MRES_WARMUP_FRAC, MRES_MARGIN_CURRENT,
# GEN_PROTOCOL, V1_ANCHOR_PREFIX_CHARS, EPOCHS, SEED_CKPT, ...) pass through unchanged, so set them
# exactly as in the main run; the arm then overrides only its own factor.
set -uo pipefail
TRACE=${TRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ARM=${ARM:?set ARM=main_gen|main_real|random_row|no_margin|posthoc|distill}
RUN_ROOT=${RUN_ROOT:-/data2/seonghyeonnoh/paper/mres_ablation}
export RUN_DIR=${RUN_DIR:-$RUN_ROOT/$ARM}

# ---- the method (left side of every ablation)
export MRES_ENABLE=1 ROUTING_WEIGHT_MODE=full_softmax
export GEN_PROTOCOL=${GEN_PROTOCOL:-v1}
export MRES_DELTA=${MRES_DELTA:-0.5} MRES_LAMBDA=${MRES_LAMBDA:-0.1} MRES_WARMUP_FRAC=${MRES_WARMUP_FRAC:-0.05}
export MRES_MARGIN_CURRENT=${MRES_MARGIN_CURRENT:-1} MRES_WARMUP_FREEZE_ROUTER=${MRES_WARMUP_FREEZE_ROUTER:-1}
export MRES_NEW_ROW=reservoir RESIDUAL_ROUTER_FT_TIMING=joint RESIDUAL_ROUTER_FT_OBJECTIVE=lm
export RESIDUAL_DISTILL_PAD=${RESIDUAL_DISTILL_PAD:-row}
export REPLAY_SOURCE=real

# ---- the one factor this arm switches
case "$ARM" in
  main_gen)   REPLAY_SOURCE=selfgen ;;
  main_real)  ;;
  random_row) MRES_NEW_ROW=random ;;
  no_margin)  MRES_LAMBDA=0 ;;
  posthoc)    RESIDUAL_ROUTER_FT_TIMING=posthoc ;;
  distill)    RESIDUAL_ROUTER_FT_OBJECTIVE=distill ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
export METHOD_NAME=${METHOD_NAME:-mres_$ARM}

mkdir -p "$RUN_DIR"
python3 - "$RUN_DIR/ablation_arm.json" "$ARM" <<'PY'
import json, os, sys
keys = ["REPLAY_SOURCE", "GEN_PROTOCOL", "V1_ANCHOR_PREFIX_CHARS", "ROUTING_WEIGHT_MODE", "MRES_ENABLE",
        "MRES_NEW_ROW", "MRES_LAMBDA", "MRES_DELTA", "MRES_WARMUP_FRAC", "MRES_MARGIN_CURRENT",
        "MRES_WARMUP_FREEZE_ROUTER", "RESIDUAL_ROUTER_FT_TIMING", "RESIDUAL_ROUTER_FT_OBJECTIVE",
        "RESIDUAL_DISTILL_PAD", "EPOCHS", "GPUS", "SEED_CKPT", "METHOD_NAME", "LAST_ROUND"]
arm = {"arm": sys.argv[2], "env": {k: os.environ.get(k) for k in keys}}
path = sys.argv[1]
if os.path.exists(path):
    prev = json.load(open(path))
    if prev != arm:        # a restart must not silently switch the arm's configuration
        sys.exit(f"{path} records a different configuration:\n{prev}\nvs\n{arm}")
json.dump(arm, open(path, "w"), indent=1)
PY
[ $? = 0 ] || exit 2

# ---- real replay: every real arm uses one pre-sampled memory (500 records/task, seed 2025,
# build_real_replay_subsets.py pct10) instead of each run drawing its own.  The backbone-BoS
# pseudo task is the fixed RESIDUAL_BOS_JSONL file in every arm and is left as is.
if [ "$REPLAY_SOURCE" = real ]; then
  SUBSET=${REPLAY_SUBSET:-$RUN_ROOT/replay_subsets/pct10/fixed_replay_memory}
  [ -f "$SUBSET/task_7_20Minuten.json" ] || { echo "missing pre-sampled replay $SUBSET (run build_real_replay_subsets.py --out-root $RUN_ROOT/replay_subsets)" >&2; exit 2; }
  mem=$RUN_DIR/model/fixed_replay_memory
  mkdir -p "$mem"
  for f in "$SUBSET"/task_*.json; do
    dst=$mem/$(basename "$f")
    if [ -f "$dst" ]; then
      cmp -s "$f" "$dst" || { echo "$dst differs from the shared pre-sampled replay $f" >&2; exit 2; }
    else
      cp "$f" "$dst"
    fi
  done
  echo "[mres-ablation] real replay <- $SUBSET ($(ls "$SUBSET"/task_*.json | wc -l) tasks)"
fi
echo "[mres-ablation] ARM=$ARM RUN_DIR=$RUN_DIR replay=$REPLAY_SOURCE new_row=$MRES_NEW_ROW lambda=$MRES_LAMBDA timing=$RESIDUAL_ROUTER_FT_TIMING objective=$RESIDUAL_ROUTER_FT_OBJECTIVE"
exec bash "$TRACE/scripts/residual/run_residual_chain.sh"
