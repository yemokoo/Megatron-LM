#!/usr/bin/env bash
# 4th route-forced arm: per token best-2 wiki experts + best-2 code experts (the mix code tokens
# naturally use, old% ~0.45).  GPU 1.  Waits for chain_bos_route_forced to finish before the
# final comparison so every arm is in one table.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
B=/data2/seonghyeonnoh/LLM-continual-learning-runs/bos_samples_20260827
PY=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin/python
export PYTHONNOUSERSITE=1
CKPT="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["load"])' "$B/code16_t1.0_p0.95/stats.json")"
ts(){ date +%H:%M:%S; }; lab=rf_mix2_2
echo "[mix $(ts)] sample $lab groups=0:8=2,8:16=2 gpu=1"
GPU=1 CKPT="$CKPT" NE=16 SRC=8 OUT="$B/$lab" N=1024 T=512 ROUTE_GROUPS="0:8=2,8:16=2" LABEL="$lab" PORT=42011 \
  bash scripts/analysis/run_bos_sample_hybrid.sh > "$B/$lab.launch.log" 2>&1; echo "[mix $(ts)] $lab exit=$? $(tail -1 "$B/$lab.launch.log")"
GPU=1 CKPT="$CKPT" NE=16 SRC=8 DATA="$B/$lab/gen_text_document" OUT="$B/router_usage/$lab.json" LABEL="$lab" PORT=42111 \
  bash scripts/analysis/run_router_usage_hist.sh > "$B/router_usage/$lab.launch.log" 2>&1; echo "[mix $(ts)] hist exit=$?"
until grep -q "ALL DONE" "$B/routeforced_test.log"; do sleep 30; done
echo "[mix $(ts)] === forced routing during generation ==="
$PY - "$B" <<'PYE'
import json,sys,numpy as np
B=sys.argv[1]
for lab in ("rf_wiki8","rf_code8","rf_mix2_2","rf_all16_gumbel1"):
    s=json.load(open(f"{B}/{lab}/stats.json")); fu=s["forced_usage"]; old=np.mean([sum(v[:8]) for v in fu.values()])
    print(f"  {lab:<18} old%={old:.3f}  eod_rate={s['eod_token_rate']:.4f} seqs_with_eod={s['seqs_with_eod']:.2f} uniq={s['unique_token_ratio']:.3f} rep4={s['repeat_4gram_rate']:.3f}")
PYE
echo "[mix $(ts)] === token distribution (all arms) ==="
$PY scripts/analysis/compare_bos_distribution.py \
  --gen all16_argmax="$B/code16_t1.0_p0.95/gen_text_document" rf_wiki8="$B/rf_wiki8/gen_text_document" rf_code8="$B/rf_code8/gen_text_document" rf_mix2_2="$B/$lab/gen_text_document" rf_all16_gumbel1="$B/rf_all16_gumbel1/gen_text_document" pfx_hash="$B/pfx_hash/gen_text_document" \
  --out "$B/compare_tokens_routeforced_all.json" 2>&1 | grep -vE "pip install|python_path=|mkdir -p|wget -P|UserWarning|warnings.warn"
echo "[mix $(ts)] === natural routing of generated text (all arms) ==="
$PY scripts/analysis/compare_router_usage.py --wiki "$B/router_usage/wiki_test.json" --code "$B/router_usage/code_test.json" \
  --gen all16_argmax="$B/router_usage/gen_t1.0.json" rf_wiki8="$B/router_usage/rf_wiki8.json" rf_code8="$B/router_usage/rf_code8.json" rf_mix2_2="$B/router_usage/$lab.json" rf_all16_gumbel1="$B/router_usage/rf_all16_gumbel1.json" pfx_hash="$B/router_usage/pfx_hash.json" \
  --out "$B/router_usage/compare_routeforced_all.json"
echo "[mix $(ts)] ALL DONE"
