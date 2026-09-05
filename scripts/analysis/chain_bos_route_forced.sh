#!/usr/bin/env bash
# Route-forced BoS sampling (no prefix): wiki experts only (0:8), code experts only (8:16),
# all 16 with Gumbel-top-k routing (tau=1).  Then token + routing comparison vs wiki/code test,
# with the plain all-16 argmax BoS run (code16_t1.0_p0.95) as the reference arm.  GPUs 0,1 only.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
B=/data2/seonghyeonnoh/LLM-continual-learning-runs/bos_samples_20260827
PY=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python
export PYTHONNOUSERSITE=1
CKPT="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["load"])' "$B/code16_t1.0_p0.95/stats.json")"
N="${N:-1024}"; T="${T:-512}"
ts(){ date +%H:%M:%S; }
samp(){ local gpu=$1 lab=$2 allow=$3 tau=$4; echo "[rf $(ts)] sample $lab allow=${allow:-all} tau=$tau gpu=$gpu";
  GPU=$gpu CKPT="$CKPT" NE=16 SRC=8 OUT="$B/$lab" N=$N T=$T ROUTE_ALLOW="$allow" ROUTE_TAU="$tau" LABEL="$lab" \
    bash scripts/analysis/run_bos_sample_hybrid.sh > "$B/$lab.launch.log" 2>&1; echo "[rf $(ts)] $lab exit=$? $(tail -1 "$B/$lab.launch.log")"; }
hist(){ local gpu=$1 lab=$2; GPU=$gpu CKPT="$CKPT" NE=16 SRC=8 DATA="$B/$lab/gen_text_document" OUT="$B/router_usage/$lab.json" LABEL="$lab" \
    bash scripts/analysis/run_router_usage_hist.sh > "$B/router_usage/$lab.launch.log" 2>&1; echo "[rf $(ts)] hist $lab exit=$?"; }
samp 0 rf_wiki8 0:8 0 & samp 1 rf_code8 8:16 0 & wait
samp 0 rf_all16_gumbel1 "" 1.0 & hist 1 rf_wiki8 & wait
hist 0 rf_code8 & hist 1 rf_all16_gumbel1 & wait
echo "[rf $(ts)] === forced routing during generation (old-expert fraction, layer avg) ==="
$PY - "$B" <<'PYE'
import json,sys,numpy as np
B=sys.argv[1]
for lab in ("rf_wiki8","rf_code8","rf_all16_gumbel1"):
    try: s=json.load(open(f"{B}/{lab}/stats.json"))
    except Exception as e: print(lab, "no stats", e); continue
    fu=s["forced_usage"]; old=np.mean([sum(v[:8]) for v in fu.values()])
    print(f"  {lab:<18} old%={old:.3f}  eod_rate={s['eod_token_rate']:.4f} seqs_with_eod={s['seqs_with_eod']:.2f} uniq={s['unique_token_ratio']:.3f} rep4={s['repeat_4gram_rate']:.3f}")
PYE
echo "[rf $(ts)] === token distribution ==="
$PY scripts/analysis/compare_bos_distribution.py \
  --gen all16_argmax="$B/code16_t1.0_p0.95/gen_text_document" rf_wiki8="$B/rf_wiki8/gen_text_document" rf_code8="$B/rf_code8/gen_text_document" rf_all16_gumbel1="$B/rf_all16_gumbel1/gen_text_document" \
  --out "$B/compare_tokens_routeforced.json" 2>&1 | grep -vE "pip install|python_path=|mkdir -p|wget -P|UserWarning|warnings.warn"
echo "[rf $(ts)] === natural routing of the generated text (re-routed without forcing) ==="
$PY scripts/analysis/compare_router_usage.py --wiki "$B/router_usage/wiki_test.json" --code "$B/router_usage/code_test.json" \
  --gen all16_argmax="$B/router_usage/gen_t1.0.json" rf_wiki8="$B/router_usage/rf_wiki8.json" rf_code8="$B/router_usage/rf_code8.json" rf_all16_gumbel1="$B/router_usage/rf_all16_gumbel1.json" \
  --out "$B/router_usage/compare_routeforced.json"
echo "[rf $(ts)] ALL DONE"
