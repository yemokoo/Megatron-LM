#!/usr/bin/env bash
# Self-generated replay set, 1% of train per task (41,472 seqs x 512 = 21.2M tokens each), from the
# code-stage KD-init+1phase model.  One anchor token per sequence drawn from the task's document-first-
# token histogram (wiki top-1024, code top-64), KV-cached nucleus sampling.  GPUs 0,1 only.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
B=/data2/seonghyeonnoh/LLM-continual-learning-runs/bos_samples_20260827
R="$B/replay_1pct"; mkdir -p "$R"
PY=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python
export PYTHONNOUSERSITE=1
CKPT="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["load"])' "$B/code16_t1.0_p0.95/stats.json")"
N="${N:-41472}"; T="${T:-512}"; BATCH="${BATCH:-512}"
ts(){ date +%H:%M:%S; }
samp(){ local gpu=$1 task=$2 json=$3; echo "[rp $(ts)] sample $task anchors=$json gpu=$gpu N=$N";
  GPU=$gpu CKPT="$CKPT" NE=16 SRC=8 OUT="$R/${task}_anchor" N=$N T=$T BATCH=$BATCH KV=1 ANCHOR_JSON="$json" LABEL="replay1pct_${task}_anchor" SEED=$((100 + gpu)) PORT=$((42060 + gpu)) \
    bash scripts/analysis/run_bos_sample_hybrid.sh > "$R/${task}_anchor.launch.log" 2>&1; echo "[rp $(ts)] $task exit=$? $(tail -1 "$R/${task}_anchor.launch.log")"; }
hist(){ local gpu=$1 task=$2; GPU=$gpu CKPT="$CKPT" NE=16 SRC=8 DATA="$R/${task}_anchor/gen_text_document" OUT="$B/router_usage/replay1pct_${task}.json" LABEL="replay1pct_${task}" PORT=$((42160 + gpu)) \
    bash scripts/analysis/run_router_usage_hist.sh > "$B/router_usage/replay1pct_${task}.launch.log" 2>&1; echo "[rp $(ts)] hist $task exit=$?"; }
samp 0 wiki "$B/anchors/wiki_first_token_top1024.json" & samp 1 code "$B/anchors/code_first_token_top64.json" & wait
hist 0 wiki & hist 1 code & wait
echo "[rp $(ts)] === token distribution vs wiki/code test (1M-token subsets) ==="
$PY scripts/analysis/compare_bos_distribution.py --gen replay_wiki="$R/wiki_anchor/gen_text_document" replay_code="$R/code_anchor/gen_text_document" \
  --out "$R/compare_tokens.json" 2>&1 | grep -vE "pip install|python_path=|mkdir -p|wget -P|UserWarning|warnings.warn"
echo "[rp $(ts)] === natural routing ==="
$PY scripts/analysis/compare_router_usage.py --wiki "$B/router_usage/wiki_test.json" --code "$B/router_usage/code_test.json" \
  --gen replay_wiki="$B/router_usage/replay1pct_wiki.json" replay_code="$B/router_usage/replay1pct_code.json" --out "$R/compare_router.json" | grep -E "^\S+  \(tokens|layer-avg"
for t in wiki code; do echo "[rp] $t: $(du -sh "$R/${t}_anchor/gen_text_document.bin" | cut -f1) $(grep -oE '"(num_seqs|seq_len|wall_seconds|repeat_4gram_rate|seqs_with_eod)": [0-9.]+' "$R/${t}_anchor/stats.json" | tr '\n' ' ')"; done
echo "[rp $(ts)] ALL DONE"
