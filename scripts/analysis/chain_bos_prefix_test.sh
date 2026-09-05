#!/usr/bin/env bash
# Minimal-conditioning test: BoS + one code-ish prefix token, same code16 checkpoint,
# then token-distribution and routing comparison against wiki/code test.  GPUs 0,1 only.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
B=/data2/seonghyeonnoh/LLM-continual-learning-runs/bos_samples_20260827
PY=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python
export PYTHONNOUSERSITE=1
CKPT="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["load"])' "$B/code16_t1.0_p0.95/stats.json")"
N="${N:-1024}"; T="${T:-512}"
declare -A PFX=([pfx_hash]='#' [pfx_import]='import' [pfx_docstr]='"""')
V=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
ts(){ date +%H:%M:%S; }
samp(){ local gpu=$1 lab=$2; echo "[pfx $(ts)] sample $lab prefix=${PFX[$lab]} gpu=$gpu";
  GPU=$gpu CKPT="$CKPT" NE=16 SRC=8 OUT="$B/$lab" N=$N T=$T PREFIX="${PFX[$lab]}" LABEL="$lab" \
    bash scripts/analysis/run_bos_sample_hybrid.sh > "$B/$lab.launch.log" 2>&1; echo "[pfx $(ts)] $lab exit=$? $(tail -1 "$B/$lab.launch.log")"; }
hist(){ local gpu=$1 lab=$2; GPU=$gpu CKPT="$CKPT" NE=16 SRC=8 DATA="$B/$lab/gen_text_document" OUT="$B/router_usage/$lab.json" LABEL="$lab" \
    bash scripts/analysis/run_router_usage_hist.sh > "$B/router_usage/$lab.launch.log" 2>&1; echo "[pfx $(ts)] hist $lab exit=$?"; }
samp 0 pfx_hash & samp 1 pfx_import & wait
samp 0 pfx_docstr & hist 1 pfx_hash & wait
hist 0 pfx_import & hist 1 pfx_docstr & wait
echo "[pfx $(ts)] === token distribution (prefix runs) ==="
$PY scripts/analysis/compare_bos_distribution.py \
  --gen pfx_hash="$B/pfx_hash/gen_text_document" pfx_import="$B/pfx_import/gen_text_document" pfx_docstr="$B/pfx_docstr/gen_text_document" \
  --out "$B/compare_tokens_prefix.json" 2>&1 | grep -vE "pip install|python_path=|mkdir -p|wget -P"
echo "[pfx $(ts)] === routing (prefix runs) ==="
$PY scripts/analysis/compare_router_usage.py --wiki "$B/router_usage/wiki_test.json" --code "$B/router_usage/code_test.json" \
  --gen pfx_hash="$B/router_usage/pfx_hash.json" pfx_import="$B/router_usage/pfx_import.json" pfx_docstr="$B/router_usage/pfx_docstr.json" \
  --out "$B/router_usage/compare_prefix.json"
echo "[pfx $(ts)] ALL DONE"
