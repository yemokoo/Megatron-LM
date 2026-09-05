#!/usr/bin/env bash
# Dump wiki+code probe hidden states (layers 1-9, 20k tokens) for the 2-phase
# A/B/C/D checkpoints (5 stages each: s0 wiki -> s1 +code -> s2 +router FT 1 ->
# s3 +conv -> s4 +router FT 2) plus F (HF kd_1phase one_phase) and Gnew
# (selfgen_replay_Gnew_20260828) at the s2/s4 points only (1-phase has no
# separate phase-1/router-FT split).  Same alignment contract as
# dump_wiki_drift_all.sh: seed 1234, mb 64, gbs 64, probe-iters 2, single
# sampler, --no-load-optim --no-load-rng --diagnostic-override-consumed-train-
# samples 64, 20,000 tokens.  Token alignment asserted in analyze_2phase_drift.py.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin
TOKENIZER=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6

RUNS=/data2/seonghyeonnoh/LLM-continual-learning-runs
R2P=$RUNS/g2_2phase_4runs_20260827
HF=$RUNS/hf_g2_wiki_code_conversation/g2_wiki_code_conversation
GNEW=$RUNS/selfgen_replay_Gnew_20260828
V=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
WIKI_PROBE="$V/wiki/test/test_text_document"; CODE_PROBE="$V/code/test/test_text_document"
OUT="${OUT:-$RUNS/hidden_drift_2phase_20260829}"
mkdir -p "$OUT/hidden" "$OUT/logs"

MB=64; GBS=64; PROBE_ITERS=2; MAX_TOKENS=20000; LAYERS=all; SEED=1234
DUMP_ARGS=(--hidden-space-dump-max-tokens "$MAX_TOKENS" --hidden-space-dump-layers "$LAYERS")

DRY="${DRY:-0}"
GPUS="${GPUS:-0,1}"

if [ "$DRY" != 1 ]; then
  exec 9>"$OUT/.driver.lock"
  flock -n 9 || { echo "[ABORT] another dump_2phase_drift.sh holds $OUT/.driver.lock"; exit 1; }
fi

# ---- job table: id | kind(ours|hyb) | experts:source | checkpoint dir | probe(wiki|code) ----
JOBS=(
  # s0: wiki-only references (shared across A/C and B/D respectively)
  "ffn_s0_wiki    |ours|8:8  |$HF/sources/ffn_only/wiki|wiki"
  "ffn_s0_code    |ours|8:8  |$HF/sources/ffn_only/wiki|code"
  "hyb_s0_wiki    |hyb |8:8  |$HF/sources/ffn_attn_shared_router/wiki|wiki"
  "hyb_s0_code    |hyb |8:8  |$HF/sources/ffn_attn_shared_router/wiki|code"
)
for run in ffn_kd0:ours:A ffn_kd1:ours:C hyb_kd0:hyb:B hyb_kd1:hyb:D; do
  dir="${run%%:*}"; rest="${run#*:}"; kind="${rest%%:*}"; label="${rest#*:}"
  JOBS+=(
    "${label}_s1_wiki|${kind}|16:8 |$R2P/$dir/code_task|wiki"
    "${label}_s1_code|${kind}|16:8 |$R2P/$dir/code_task|code"
    "${label}_s2_wiki|${kind}|16:8 |$R2P/$dir/code_router_ft|wiki"
    "${label}_s2_code|${kind}|16:8 |$R2P/$dir/code_router_ft|code"
    "${label}_s3_wiki|${kind}|24:16|$R2P/$dir/conv_task|wiki"
    "${label}_s3_code|${kind}|24:16|$R2P/$dir/conv_task|code"
    "${label}_s4_wiki|${kind}|24:16|$R2P/$dir/conv_router_ft|wiki"
    "${label}_s4_code|${kind}|24:16|$R2P/$dir/conv_router_ft|code"
  )
done
JOBS+=(
  # F: HF kd_1phase (real full old-data 1-phase) -- s2/s4 points only
  "F_s2_wiki      |hyb |16:8 |$HF/kd_1phase/ffn_attn_shared_router/code/one_phase|wiki"
  "F_s2_code      |hyb |16:8 |$HF/kd_1phase/ffn_attn_shared_router/code/one_phase|code"
  "F_s4_wiki      |hyb |24:16|$HF/kd_1phase/ffn_attn_shared_router/conversation/one_phase|wiki"
  "F_s4_code      |hyb |24:16|$HF/kd_1phase/ffn_attn_shared_router/conversation/one_phase|code"
  # Gnew: fully replay-free 1-phase (BoS+anchor self-generated replay throughout) -- s2/s4 points only
  "G_s2_wiki      |hyb |16:8 |$GNEW/code_1phase|wiki"
  "G_s2_code      |hyb |16:8 |$GNEW/code_1phase|code"
  "G_s4_wiki      |hyb |24:16|$GNEW/conv_1phase|wiki"
  "G_s4_code      |hyb |24:16|$GNEW/conv_1phase|code"
)

trim() { echo "$1" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g'; }
probe_path() { [ "$1" = code ] && echo "$CODE_PROBE" || echo "$WIKI_PROBE"; }
probe_name() { [ "$1" = code ] && echo "code_probe" || echo "wiki_probe"; }

run_ours() {   # id experts:source real probe gpu port
  local id="$1" ne="${2%%:*}" src="${2##*:}" real="$3" probe="$4" gpu="$5" port="$6"
  local pp; pp="$(probe_path "$probe")"; local pn; pn="$(probe_name "$probe")"
  local log="$OUT/logs/${id}.log" npz="$OUT/hidden/${id}.npz"
  [ -s "$npz" ] && { echo "[SKIP] $id"; return; }
  [ -f "$real/latest_checkpointed_iteration.txt" ] || { echo "[MISS] $id $real"; return; }
  local -a resume=(); [ "$ne" -gt 8 ] && resume=(--moe-resume-from-num-experts "$src")
  local -a dump=(--hidden-space-dump-path "$npz" --hidden-space-dump-label "$id" "${DUMP_ARGS[@]}")
  echo "[RUN] $id gpu=$gpu (ours experts=$ne source=$src probe=$probe)"
  if [ "$DRY" = 1 ]; then echo "     torchrun … --load $real ${resume[*]} --probe-name $pn --probe-data-path 1.0 $pp ${dump[*]}"; return; fi
  (
    export TOKENIZER_MODEL="$TOKENIZER"
    export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 \
           MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 NUM_EXPERTS="$ne" SOURCE_NUM_EXPERTS="$src"
    # shellcheck source=/dev/null
    source scripts/experiment/a100/flame-moe-bf16-no-shared.sh
    CUDA_VISIBLE_DEVICES="$gpu" torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "$port" \
      Megatron-LM/pretrain_gpt.py "${MODEL_ARGS[@]}" \
      --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
      --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
      --micro-batch-size "$MB" --global-batch-size "$GBS" --seed "$SEED" \
      --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 \
      --seq-length 512 --data-path 1.0 "$pp" --split 100,0,0 --train-iters 1 --skip-train \
      --load "$real" --no-load-optim --no-load-rng "${resume[@]}" \
      --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
      --eval-interval 1 --probe-name "$pn" --probe-eval-iters "$PROBE_ITERS" --probe-eval-interval 1 \
      --probe-data-path 1.0 "$pp" --run-initial-probe-eval "${dump[@]}"
  ) > "$log" 2>&1
  [ -s "$npz" ] && echo "[DONE] $id" || echo "[FAIL] $id (see $log)"
}

run_hyb() {   # id experts:source real probe gpu port
  local id="$1" ne="${2%%:*}" src="${2##*:}" real="$3" probe="$4" gpu="$5" port="$6"
  local pp; pp="$(probe_path "$probe")"; local pn; pn="$(probe_name "$probe")"
  local log="$OUT/logs/${id}.log" npz="$OUT/hidden/${id}.npz"
  [ -s "$npz" ] && { echo "[SKIP] $id"; return; }
  [ -f "$real/latest_checkpointed_iteration.txt" ] || { echo "[MISS] $id $real"; return; }
  local -a resume=(); [ "$ne" -gt "$src" ] && resume=(--shared-router-hybrid-resume-from-num-experts "$src" --shared-router-hybrid-train-new-experts-and-router-only)
  local -a dump=(--hidden-space-dump-path "$npz" --hidden-space-dump-label "$id" "${DUMP_ARGS[@]}")
  echo "[RUN] $id gpu=$gpu (hybrid experts=$ne source=$src probe=$probe)"
  if [ "$DRY" = 1 ]; then echo "     torchrun(hybrid) … --load $real ${resume[*]} --probe-name $pn --probe-data-path 1.0 $pp ${dump[*]}"; return; fi
  (
    export TOKENIZER_MODEL="$TOKENIZER"
    export NUM_LAYERS=9 HIDDEN_SIZE=1024 FFN_HIDDEN_SIZE=5472 NUM_QUERY_GROUPS=16 MOE_FFN_HIDDEN_SIZE=352 \
           MOE_LAYER_FREQ="[0]*1+[1]*8" MOE_ROUTER_TOPK=4 NUM_EXPERTS="$ne" MOE_ROUTER_DTYPE=fp32 \
           ATTN_LORA_RANK=256 ATTN_LORA_ALPHA=256 ATTN_FULL_RANK_LORA_RANK=256 ATTN_FULL_RANK_LORA_ALPHA=256 \
           ATTN_FULL_RANK_LORA_TARGETS=qkvo MOE_GROUPED_GEMM=1 ATTN_LORA_GROUPED_GEMM=1
    # shellcheck source=/dev/null
    source configs/model/flame-shared-router-hybrid-experts.sh
    CUDA_VISIBLE_DEVICES="$gpu" torchrun --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "$port" \
      Megatron-LM/pretrain_gpt.py "${MODEL_ARGS[@]}" \
      --transformer-impl local --pipeline-model-parallel-size 1 --expert-model-parallel-size 1 \
      --distributed-timeout-minutes 30 --no-persist-layer-norm --bf16 \
      --micro-batch-size "$MB" --global-batch-size "$GBS" --seed "$SEED" \
      --lr 3e-4 --min-lr 3e-5 --lr-decay-style WSD --lr-decay-iters 1 --lr-warmup-fraction 0.0 --lr-wsd-decay-iters 1 \
      --seq-length 512 --data-path 1.0 "$pp" --split 100,0,0 --train-iters 1 --skip-train \
      --load "$real" --no-load-optim --no-load-rng --finetune "${resume[@]}" \
      --eval-interval 1 --eval-iters 0 --probe-name "$pn" --probe-eval-iters "$PROBE_ITERS" --probe-eval-interval 1 \
      --probe-data-path 1.0 "$pp" --run-initial-probe-eval "${dump[@]}"
  ) > "$log" 2>&1
  [ -s "$npz" ] && echo "[DONE] $id" || echo "[FAIL] $id (see $log)"
}

run_job() {   # jobline gpu port
  IFS='|' read -r id kind spec real probe <<< "$1"
  id=$(trim "$id"); kind=$(trim "$kind"); spec=$(trim "$spec"); real=$(trim "$real"); probe=$(trim "$probe")
  case "$kind" in
    ours) run_ours "$id" "$spec" "$real" "$probe" "$2" "$3" ;;
    hyb)  run_hyb  "$id" "$spec" "$real" "$probe" "$2" "$3" ;;
  esac
}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
stream() {   # stream_index gpu
  exec 9>&-
  local si="$1" gpu="$2" i=0 port=$((${PORT_BASE:-38100} + si * 100))
  for job in "${JOBS[@]}"; do
    if [ $(( i % ${#GPU_LIST[@]} )) -eq "$si" ]; then
      run_job "$job" "$gpu" $((port + i))
    fi
    i=$((i + 1))
  done
}
for si in "${!GPU_LIST[@]}"; do stream "$si" "${GPU_LIST[$si]}" & done
wait
echo "[ALL DONE] $(ls "$OUT/hidden" | wc -l) npz in $OUT/hidden"
