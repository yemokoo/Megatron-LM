#!/usr/bin/env bash
set -uo pipefail
# 1) wiki + code calibration at scale 32 (150k windows each, 4 GPUs each) — sets
#    the old-domain bands on the 32-scale axis for the Conversation selector.
# 2) full Code-train census at scale 32 (8 GPUs) against the Code before/after
#    pair — the Code-stage counterpart of the Conversation 32 census.
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"
PY=/home/seonghyeonnoh/dohyun/.venv/bin/python
CONV=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_conv_census_20260817
CODE=/data2/seonghyeonnoh/LLM-continual-learning-runs/cka_gt_full_census_20260816/analysis/cka_gt_full_census_v1
V=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
ST=$CONV/logs/scale32_status.tsv
say(){ echo "$(date -Is) $*" | tee -a "$ST"; }

# ---- 1. calibration (conv before/after pair, wiki & code data) ----
say "CAL32-START"
for dom in wiki code; do
  OUT=$CONV/calibration_${dom}_scale32; mkdir -p $OUT/logs; rm -rf $OUT/worker_* $CONV/data_cache/c32_${dom}_*
  if [ $dom = wiki ]; then gpus="0 1 2 3"; base=37100; else gpus="4 5 6 7"; base=37110; fi
  w=0
  for g in $gpus; do
    CKA_CENSUS_SCALES=32 GPU=$g WORKER_INDEX=$w WORKER_COUNT=4 MAX_WINDOWS=150000 \
    WINDOW_BATCH_SIZE=384 FORWARD_SUBBATCH_SIZE=256 RESERVOIR_SIZE=5000000 \
    ANALYSIS_CONFIG=$CONV/analysis_config_cal_${dom}_scale32.json \
    DATA_PREFIX=$V/$dom/train/train_text_document \
    CENSUS_MANIFEST=$CONV/manifest_$dom/manifest.json CENSUS_OUTPUT=$OUT \
    CACHE_PATH=$CONV/data_cache/c32_${dom}_$w LOG_PATH=$OUT/logs/worker_$(printf %03d $w).log \
    MASTER_PORT=$((base + w)) \
    bash scripts/analysis/run_cka_conv_census_mha.sh $g > $OUT/logs/launcher_$w.log 2>&1 &
    w=$((w+1))
  done
done
wait
for dom in wiki code; do $PY scripts/analysis/merge_census_histograms.py --census-root $CONV/calibration_${dom}_scale32 > $CONV/logs/cal32_merge_$dom.log 2>&1; done
say "CAL32-DONE"

# ---- 2. Code full census at scale 32 (Code before/after pair) ----
say "CODE32-CENSUS-START 4161493 windows, 8 workers"
OUT=$CODE/full_census_scale32; mkdir -p $OUT/logs; rm -rf $OUT/worker_* $CODE/data_cache/s32_w*
w=0
for g in 0 1 2 3 4 5 6 7; do
  CKA_CENSUS_SCALES=32 GPU=$g WORKER_INDEX=$w WORKER_COUNT=8 MAX_WINDOWS=0 \
  WINDOW_BATCH_SIZE=384 FORWARD_SUBBATCH_SIZE=256 RESERVOIR_SIZE=5000000 \
  ANALYSIS_CONFIG=$CODE/analysis_config_code_scale32.json \
  DATA_PREFIX=$V/code/train/train_text_document \
  CENSUS_MANIFEST=$CODE/manifest/manifest.json CENSUS_OUTPUT=$OUT \
  CACHE_PATH=$CODE/data_cache/s32_w$w LOG_PATH=$OUT/logs/worker_$(printf %03d $w).log \
  MASTER_PORT=$((37200 + w)) \
  bash scripts/analysis/run_cka_gt_full_census_mha.sh $g > $OUT/logs/launcher_$w.log 2>&1 &
  w=$((w+1))
done
wait
$PY scripts/analysis/merge_census_histograms.py --census-root $OUT --expected-windows 4161493 > $CONV/logs/code32_merge.log 2>&1
say "CODE32-CENSUS-DONE merge rc=$?"
