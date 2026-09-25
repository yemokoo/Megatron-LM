#!/usr/bin/env bash
# Mass-reservoir ablations: one arm = one full 8-task run_residual_chain.sh run.
#
# Every arm is the method (left side) with exactly one factor switched:
#   ARM=main_gen    the method: self-generated replay (GEN_PROTOCOL=v1)
#   ARM=main_real   replay gen -> real (trainer's fixed replay memory, 500 real records/task);
#                   also the reference arm for every arm below, which all use real replay
#   ARM=random_row  new router row: clone(r_res) -> stock nn.Linear random init (MRES_NEW_ROW)
#   ARM=no_margin   margin loss weight lambda -> 0 (MRES_LAMBDA=0; margin stats still logged)
#   ARM=posthoc     joint router correction -> post-hoc (router tuning on 20% of the train):
#                   the task trains the primary branch
#                   alone, then a router-only pass replays exactly the joint arm's per-update
#                   router-FT batches (same records, same current slices, same update count);
#                   <round>_prephase2 = pre-correction checkpoint, used for the diagonal scores
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
  posthoc)    RESIDUAL_ROUTER_FT_TIMING=posthoc
              # router tuning on 20% of the train: 20% of the joint updates, same per-update batch
              export RESIDUAL_POSTHOC_ROUTER_FRAC=${RESIDUAL_POSTHOC_ROUTER_FRAC:-0.2}
              # acquisition score = right after the new task, before the router correction
              # (<round>_prephase2); final row = the corrected final model for all 8 tasks; the
              # last task's acquisition score is scored separately after the chain (see below)
              export SPARSE15_DIAGONAL_CKPT_SUFFIX=_prephase2 SPARSE15_FINAL_ROW_PLAIN=1 ;;
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
        "RESIDUAL_DISTILL_PAD", "RESIDUAL_POSTHOC_ROUTER_FRAC", "SPARSE15_DIAGONAL_CKPT_SUFFIX",
        "SPARSE15_FINAL_ROW_PLAIN", "EPOCHS", "GPUS", "SEED_CKPT", "METHOD_NAME", "LAST_ROUND"]
arm = {"arm": sys.argv[2], "env": {k: os.environ.get(k) for k in keys}}
path = sys.argv[1]
if os.path.exists(path):
    prev = json.load(open(path))
    if prev != arm:        # a restart must not silently switch the arm's configuration
        sys.exit(f"{path} records a different configuration:\n{prev}\nvs\n{arm}")
json.dump(arm, open(path, "w"), indent=1)
PY
[ $? = 0 ] || exit 2

# ---- real replay: every real arm, on every server, uses one pre-sampled memory -- the index
# files committed in scripts/residual/assets/real_replay_pct10_seed2025 (500 records/task, seed
# 2025, build_real_replay_subsets.py pct10) -- instead of each run drawing its own.  The indices
# point into each task's train.json, so that file is checked against the manifest's sha256 first.
# The backbone-BoS pseudo task is the fixed RESIDUAL_BOS_JSONL file in every arm and is left as is.
if [ "$REPLAY_SOURCE" = real ]; then
  SUBSET=${REPLAY_SUBSET:-$TRACE/scripts/residual/assets/real_replay_pct10_seed2025}
  [ -f "$SUBSET/task_7_20Minuten.json" ] || { echo "missing pre-sampled replay $SUBSET" >&2; exit 2; }
  if [ -f "$SUBSET/manifest.json" ]; then
    python3 - "$SUBSET/manifest.json" "${TRACE_DATA_ROOT:-/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup/trace}" <<'PY' || exit 2
import hashlib, json, sys
man, data = json.load(open(sys.argv[1])), sys.argv[2]
bad = [t for t, v in man["tasks"].items()
       if hashlib.sha256(open(f"{data}/{t}/train.json", "rb").read()).hexdigest() != v["train_json_sha256"]]
if bad:
    sys.exit(f"train.json under {data} differs from the one the replay indices were drawn from: {bad}")
print(f"[mres-ablation] train.json sha256 matches the replay manifest for {len(man['tasks'])} tasks")
PY
  fi
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
[ "$ARM" = posthoc ] || exec bash "$TRACE/scripts/residual/run_residual_chain.sh"

# ---- posthoc: chain, then the last task's acquisition score from <last>_prephase2 and an
# 8-task forgetting summary (sparse15_summary.json covers the acquisition scores of tasks 1-7 only)
bash "$TRACE/scripts/residual/run_residual_chain.sh" || exit $?
M=$RUN_DIR/model
[ -f "$M/sparse15_summary.json" ] || exit 0          # LAST_ROUND < 7: nothing to score
PY=${TRACE_PYTHON:-$TRACE/.venv-runtime/bin/python}
VIEW=$RUN_DIR/last_acquisition            # own evaluation/ dir: order8/results-<task>.json would
mkdir -p "$VIEW"                          # otherwise collide with the final-row cell
ln -sfn "$M/7_prephase2" "$VIEW/7_prephase2"
if [ ! -f "$RUN_DIR/posthoc_forgetting_summary.json" ]; then
  echo "[mres-ablation] last-task acquisition score from 7_prephase2"
  ( cd "$TRACE" && SPARSE15_FINAL_ROW_PLAIN=0 SPARSE15_EVAL_EXTRA_ARGS="--bos_guard --guard_header --guard_decision none" SPARSE15_CONV_MODE=llama3_template \
      SLORA_LLAMA31_PATH=${SLORA_LLAMA31_PATH:-/data2/seonghyeonnoh/LLM-continual-learning-models/Llama-3.1-8B-Instruct} \
      $PY scripts/run_ours_sparse15_optimized.py --run-dir "$VIEW" --method "${METHOD_NAME}_last_acq" \
      --gpus "$GPUS" --matrix-mode last_acquisition --diagonal-checkpoint-suffix _prephase2 --no-collect \
  ) > "$RUN_DIR/logs/eval_last_acquisition.log" 2>&1 || { echo "last-task acquisition eval failed" >&2; exit 1; }
  $PY - "$M/sparse15_summary.json" "$VIEW" "$RUN_DIR/posthoc_forgetting_summary.json" <<'PY' | tee -a "$RUN_DIR/chain.log"
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import run_ours_sparse15_optimized as E
s = json.load(open(sys.argv[1]))
tasks = s["tasks"]
acq = list(s["diagonal_scores_rounds_1_to_7"]) + [E.read_primary_score(Path(sys.argv[2]), E.Cell(8, tasks[-1]))]
final = list(s["final_scores_round_8"])
drop = [a - f for a, f in zip(acq, final)]
out = {"tasks": tasks,
       "acquisition_scores_pre_router_correction": acq,
       "final_scores_after_router_correction": final,
       "forgetting_per_task": drop,
       "AA": sum(final) / len(final),
       "F_8tasks": sum(drop) / len(drop),
       "F_7tasks_sparse15": -s["BWT"],
       "note": "acquisition = <round>_prephase2 (new task learned, router not yet corrected); "
               "final = checkpoint 7 (after the last router correction) for every task"}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("[posthoc] AA %.2f  F(8 tasks, pre-correction acquisition) %.2f  F(7 tasks) %.2f"
      % (out["AA"], out["F_8tasks"], out["F_7tasks_sparse15"]))
print("[posthoc] acq  ", {t: round(v, 1) for t, v in zip(tasks, acq)})
print("[posthoc] final", {t: round(v, 1) for t, v in zip(tasks, final)})
PY
fi
