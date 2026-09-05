#!/usr/bin/env bash
set -uo pipefail
# Scale-32 (stride 16) Conversation census over the full window axis on all
# eight GPUs.  Same before/after pair, same manifest, same metrics; only the
# chunk scale changes, so B/T become far more local to the token.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"
C=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817
D=/data2/seonghyeonnoh/LLM-continual-learning-data
OUT="$C/full_census_scale32"
mkdir -p "$OUT/logs" "$C/logs"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"; N=$(echo $GPUS | wc -w)
say(){ echo "$(date -Is) $*" | tee -a "$C/logs/scale32_status.tsv"; }
say "SCALE32-CENSUS-START workers=$N gpus=$GPUS"
w=0
for g in $GPUS; do
  CKA_CENSUS_SCALES=32 GPU=$g WORKER_INDEX=$w WORKER_COUNT=$N MAX_WINDOWS=0 \
  WINDOW_BATCH_SIZE=384 FORWARD_SUBBATCH_SIZE=256 RESERVOIR_SIZE=5000000 \
  ANALYSIS_CONFIG="$C/analysis_config_conversation_scale32.json" \
  DATA_PREFIX="$D/conversation_merged_train_20260817/train_text_document" \
  CENSUS_MANIFEST="$C/manifest/manifest.json" CENSUS_OUTPUT="$OUT" \
  CACHE_PATH="$C/data_cache/s32_w$w" LOG_PATH="$OUT/logs/worker_$(printf %03d $w).log" \
  MASTER_PORT=$((37000+w)) \
  bash scripts/analysis/run_cka_conv_census_mha.sh "$g" > "$OUT/logs/launcher_$w.log" 2>&1 &
  w=$((w+1))
done
wait
say "SCALE32-CENSUS-WORKERS-DONE"
/home/seonghyeonnoh/dohyun/.venv/bin/python scripts/analysis/merge_census_histograms.py --census-root "$OUT" --expected-windows 2745838 >> "$C/logs/scale32_merge.log" 2>&1
say "SCALE32-MERGE rc=$?"
