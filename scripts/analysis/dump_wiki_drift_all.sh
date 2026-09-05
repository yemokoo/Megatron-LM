#!/usr/bin/env bash
# Dump wiki-probe hidden states (all 9 layers, 20k tokens) and wiki next_token_acc
# for every method at three points -- after wiki, after code, after conversation --
# so per-token drift from the wiki-only model can be measured in the 1024-d space.
#
# Token alignment: every run reads the same wiki test probe with the same seed,
# seq length, micro/global batch and probe iterations, so the probe loader yields
# the same samples in the same order and the dump captures the same 20,000
# loss-mask tokens.  The analysis asserts token_ids/positions/sample_indices match.
#
# Two passes per checkpoint: the hidden dump returns from run_probe_evaluation
# before the probe metrics print, so acc needs its own pass.
#
# Baselines go through train_stage.sh in eval-only mode via a scratch dir of
# symlinks (resume path -> iteration 1800, so O-LoRA does not re-init its slot).
# Ours goes through the flame-moe config directly, as the earlier 3-domain dump did.
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$P"
export PYTHONNOUSERSITE=1
export PATH=/data2/seonghyeonnoh/condatest/miniconda3/envs/flame-megatron-h100/bin:/usr/bin:/bin

RUNS=/data2/seonghyeonnoh/LLM-continual-learning-runs
B=$RUNS/baselines6_wiki_code_conversation_20260816
F=$RUNS/flamemoe/ffn_experts_only
HF=$RUNS/hf_g2_wiki_code_conversation/g2_wiki_code_conversation
LPR=$RUNS/moe_lpr_from_flamemoe_e8_20260826
LPRH=$RUNS/moe_lpr_hybrid_from_hf_e8_20260826
V=/data2/seonghyeonnoh/LLM-continual-learning-data/flamedata2.data2-verified-backup
WIKI_PROBE="$V/wiki/test/test_text_document"
OUT="${OUT:-$RUNS/hidden_drift_wiki_20260826}"
mkdir -p "$OUT/acc" "$OUT/hidden" "$OUT/logs" "$OUT/work"

# ---- alignment contract (identical for all 22 checkpoints) --------------------
MB=64; GBS=64; PROBE_ITERS=2; MAX_TOKENS=20000; LAYERS=all; SEED=1234
DUMP_ARGS=(--hidden-space-dump-max-tokens "$MAX_TOKENS" --hidden-space-dump-layers "$LAYERS")

DRY="${DRY:-0}"
GPUS="${GPUS:-2,3}"

# Single instance only.  Killing the driver bash does not kill its stream
# subshells; a relaunch then raced the orphans on the same GPUs and npz paths
# and left two truncated dumps behind.  The lock is held by this process and
# inherited by the streams, so a second launch refuses until all of them exit.
if [ "$DRY" != 1 ]; then
  exec 9>"$OUT/.driver.lock"
  flock -n 9 || { echo "[ABORT] another dump_wiki_drift_all.sh holds $OUT/.driver.lock"; exit 1; }
fi

# ---- job table: id | kind | spec/task | checkpoint | extra ----------------------
# kind=base : spec = continual method used to *load*, task = continual-task-name
#             (O-LoRA activates slots by task name, so it must be its own stage;
#              dense-spec checkpoints ignore it)
# kind=ours : spec = NUM_EXPERTS:SOURCE_NUM_EXPERTS (resume flag added when >8)
OL="--continual-olora-rank 352 --continual-olora-alpha 352 --continual-olora-dropout 0.1 --continual-olora-orth-lambda 0.5"
JOBS=(
  # wiki-only references
  "dense_wiki    |base|sequential_dense|wiki        |$B/common_dense/wiki|"
  "olora_wiki    |base|olora           |wiki        |$B/olora/wiki|$OL"
  "fmoe_wiki     |base|fixed_moe       |wiki        |$B/fixed_moe/wiki|"
  "ours_wiki     |ours|8:8             |wiki        |$F/wiki/pretrain/lm/full_training/g2_olddata_kd_9run_20260808__wiki_ffn_only_e8_step1800|"
  # after code
  "seq_code      |base|sequential_dense|wiki        |$B/sequential_dense/code|"
  "ewc_code      |base|sequential_dense|wiki        |$B/ewc/code|"
  "gem_code      |base|sequential_dense|wiki        |$B/trace_gem/code|"
  "slora16_code  |base|sequential_dense|wiki        |$B/slora_pre/rank16/code|"
  "slora64_code  |base|sequential_dense|wiki        |$B/slora_pre/rank64/code|"
  "olora_code    |base|olora           |code        |$B/olora/code|$OL"
  "fmoe_code     |base|fixed_moe       |wiki        |$B/fixed_moe/code|"
  "ours_code_hmse|ours|16:8            |wiki        |$F/code/replay/hidden_mse/full_training/code_old_like_gt_token_miniset_20pct_1phase_4objective_v4_8gpu_mb96_probe24_kdfirst_20260812__01_hidden_mse|"
  "ours_code_lm  |ours|16:8            |wiki        |$F/code/replay/lm/full_training/code_old_like_gt_token_miniset_20pct_1phase_4objective_v4_8gpu_mb96_probe24_kdfirst_20260812__04_lm|"
  # after conversation
  "seq_conv      |base|sequential_dense|wiki        |$B/sequential_dense/conversation|"
  "ewc_conv      |base|sequential_dense|wiki        |$B/ewc/conversation|"
  "gem_conv      |base|sequential_dense|wiki        |$B/trace_gem/conversation|"
  "slora16_conv  |base|sequential_dense|wiki        |$B/slora_pre/rank16/conversation|"
  "slora64_conv  |base|sequential_dense|wiki        |$B/slora_pre/rank64/conversation|"
  "olora_conv    |base|olora           |conversation|$B/olora/conversation|$OL"
  "fmoe_conv     |base|fixed_moe       |wiki        |$B/fixed_moe/conversation|"
  "ours_conv     |ours|24:16           |wiki        |$F/conversation/replay/lm/full_training/conversation_old_like_gt_pipeline_20260812__02_conversation_wikicode_replay_lm_oracle_step1800|"
  "ours_conv_norep|ours|24:16          |wiki        |$F/conversation/no_replay/lm/full_training/fingerprint_router_geometry_20260809__Conversation_only_from_KDinit_mb48_gbs2304_step1800|"
  # HF YeMoKoo/LLM-continual-learning: FFN + QKVO attention experts, shared
  # router, KD-init + one-phase lm replay (kd_1phase/ffn_attn_shared_router).
  # Frozen parts verified bit-identical wiki -> code -> conv.
  "oursattn_wiki |hyb |8:8             |wiki        |$HF/sources/ffn_attn_shared_router/wiki|"
  "oursattn_code |hyb |16:8            |wiki        |$HF/kd_1phase/ffn_attn_shared_router/code/one_phase|"
  "oursattn_conv |hyb |24:16           |wiki        |$HF/kd_1phase/ffn_attn_shared_router/conversation/one_phase|"
  # HF baseline/ffn_only: plain expansion + full fine-tune, no KD, no replay,
  # then a 1800-iter router retune on wiki+code (code_router_retune @3600).
  # Its trunk/embedding move, so it is NOT frozen-identical to the e8 source.
  "moeft_code    |ours|16:8            |wiki        |$HF/baseline/ffn_only/code_task|"
  "moeft_retune  |ours|16:8            |wiki        |$HF/baseline/ffn_only/code_router_retune|"
  "moeft_conv    |ours|24:16           |wiki        |$HF/baseline/ffn_only/conversation_task|"
  # MoE-LPR trained here from the flamemoe e8 (run_moe_lpr_from_flamemoe_e8.sh):
  # code 1800 -> LPR router retune 360 -> conv 1800 -> LPR router retune 360.
  "lpr_code_pre  |ours|16:8            |wiki        |$LPR/code_task|"
  "lpr_code      |ours|16:8            |wiki        |$LPR/code_router_lpr|"
  "lpr_conv_pre  |ours|24:16           |wiki        |$LPR/conversation_task|"
  "lpr_conv      |ours|24:16           |wiki        |$LPR/conversation_router_lpr|"
  # MoE-LPR on the FFN+attention shared-router model (run_moe_lpr_hybrid_from_hf_e8.sh),
  # same expert layout as Ours-attn, from the same HF hybrid e8.
  "lprhyb_code_pre|hyb |16:8            |wiki        |$LPRH/code_task|"
  "lprhyb_code   |hyb |16:8            |wiki        |$LPRH/code_router_lpr|"
  "lprhyb_conv_pre|hyb |24:16           |wiki        |$LPRH/conversation_task|"
  "lprhyb_conv   |hyb |24:16           |wiki        |$LPRH/conversation_router_lpr|"
)

trim() { echo "$1" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g'; }

# ---- baselines: eval-only through train_stage.sh ------------------------------
run_base() {   # id spec task real extra pass(acc|dump) gpu port
  local id="$1" spec="$2" task="$3" real="$4" extra="$5" pass="$6" gpu="$7" port="$8"
  local work="$OUT/work/${id}_${pass}" log="$OUT/logs/${id}_${pass}.log"
  local npz="$OUT/hidden/${id}.npz" acc="$OUT/acc/${id}.txt"
  if [ "$pass" = dump ] && [ -s "$npz" ]; then echo "[SKIP] $id dump"; return; fi
  if [ "$pass" = acc ] && [ -s "$acc" ]; then echo "[SKIP] $id acc"; return; fi
  [ -f "$real/latest_checkpointed_iteration.txt" ] || { echo "[MISS] $id $real"; return; }
  local it; it="$(tr -d '[:space:]' < "$real/latest_checkpointed_iteration.txt")"
  rm -rf "$work"; mkdir -p "$work"
  cp "$real/latest_checkpointed_iteration.txt" "$work/"
  ln -sfn "$real/$(printf 'iter_%07d' "$it")" "$work/$(printf 'iter_%07d' "$it")"
  for s in "$real"/continual_state_*; do [ -e "$s" ] && ln -sfn "$s" "$work/$(basename "$s")"; done
  # trailing args land in continual_args, which is last in argv, so the wiki
  # probe override wins over the task's own primary probe (argparse last-wins).
  # --no-load-optim: the resume path otherwise loads the checkpoint's
  # OptimizerParamScheduler and asserts its warmup samples against this run's
  # (gbs 64 vs the training gbs 2304) -- an assert we have no reason to satisfy
  # for a skip-train probe.  Iteration still loads, so O-LoRA sees 1800, not 0.
  # --dataloader-type single: train_stage.sh runs the baselines with the cyclic
  # (random-order) sampler while the ours path uses the sequential one.  Both
  # read the same 128 probe sequences, but in a different order, so the first
  # 20k tokens were different sequences (19,850/20,000 token ids disagreed).
  # Sequential on both sides makes sample i the same sequence everywhere.
  # --diagnostic-override-consumed-train-samples 64: the sequential sampler
  # asserts consumed < total on the (unused, skip-train) train loader, and the
  # checkpoint carries consumed = 1800 x 2304 = 4,147,200.  Overriding to 0 is
  # undone -- Megatron rebuilds a zero as iteration x gbs before building the
  # loaders -- so use one micro-batch's worth: non-zero, and far below any
  # train-set size.  args.iteration stays 1800, so O-LoRA keeps its slot.
  local -a tail=($extra --no-load-optim --no-load-rng --dataloader-type single --diagnostic-override-consumed-train-samples 64 --probe-name wiki_probe --probe-data-path 1.0 "$WIKI_PROBE")
  [ "$pass" = dump ] && tail+=(--hidden-space-dump-path "$npz" --hidden-space-dump-label "$id" "${DUMP_ARGS[@]}")
  echo "[RUN] $id $pass gpu=$gpu (base spec=$spec task=$task iter=$it)"
  if [ "$DRY" = 1 ]; then echo "     run_baselines6_stage $spec $task $work '' '' $MB $port ${tail[*]}"; return; fi
  (
    export BASELINES6_EVAL_ONLY=1 PROBE_ITERS="$PROBE_ITERS" PROBE_INTERVAL=1 \
           GLOBAL_BATCH_SIZE="$GBS" SEED="$SEED" NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES="$gpu" PAUSE_SECONDS=0
    # shellcheck source=/dev/null
    source "$P/scripts/experiment/a100/baselines6/train_stage.sh"
    run_baselines6_stage "$spec" "$task" "$work" "" "" "$MB" "$port" "${tail[@]}"
  ) > "$log" 2>&1
  if [ "$pass" = acc ]; then
    grep -E '^probe wiki_probe ' "$work/logs/train.log" | tail -1 > "$acc"
    [ -s "$acc" ] && echo "[DONE] $id acc: $(grep -oE 'next_token_acc: [0-9.]+' "$acc")" || echo "[FAIL] $id acc (see $log)"
  else
    [ -s "$npz" ] && echo "[DONE] $id dump" || echo "[FAIL] $id dump (see $log)"
  fi
}

# ---- ours: flame-moe config, as dump_conv_chain_hidden_space_gpu23.sh ---------
run_ours() {   # id experts:source real pass gpu port
  local id="$1" ne="${2%%:*}" src="${2##*:}" real="$3" pass="$4" gpu="$5" port="$6"
  local log="$OUT/logs/${id}_${pass}.log" npz="$OUT/hidden/${id}.npz" acc="$OUT/acc/${id}.txt"
  if [ "$pass" = dump ] && [ -s "$npz" ]; then echo "[SKIP] $id dump"; return; fi
  if [ "$pass" = acc ] && [ -s "$acc" ]; then echo "[SKIP] $id acc"; return; fi
  [ -f "$real/latest_checkpointed_iteration.txt" ] || { echo "[MISS] $id $real"; return; }
  local -a resume=(); [ "$ne" -gt 8 ] && resume=(--moe-resume-from-num-experts "$src")
  local -a dump=(); [ "$pass" = dump ] && dump=(--hidden-space-dump-path "$npz" --hidden-space-dump-label "$id" "${DUMP_ARGS[@]}")
  echo "[RUN] $id $pass gpu=$gpu (ours experts=$ne source=$src)"
  if [ "$DRY" = 1 ]; then echo "     torchrun … --load $real ${resume[*]} ${dump[*]}"; return; fi
  (
    export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
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
      --seq-length 512 --data-path 1.0 "$WIKI_PROBE" --split 100,0,0 --train-iters 1 --skip-train \
      --load "$real" --no-load-optim --no-load-rng "${resume[@]}" \
      --diagnostic-override-train-iteration 0 --diagnostic-override-consumed-train-samples 0 \
      --eval-interval 1 --probe-name wiki_probe --probe-eval-iters "$PROBE_ITERS" --probe-eval-interval 1 \
      --probe-data-path 1.0 "$WIKI_PROBE" --run-initial-probe-eval "${dump[@]}"
  ) > "$log" 2>&1
  if [ "$pass" = acc ]; then
    grep -E '^probe wiki_probe ' "$log" | tail -1 > "$acc"
    [ -s "$acc" ] && echo "[DONE] $id acc: $(grep -oE 'next_token_acc: [0-9.]+' "$acc")" || echo "[FAIL] $id acc (see $log)"
  else
    [ -s "$npz" ] && echo "[DONE] $id dump" || echo "[FAIL] $id dump (see $log)"
  fi
}

# ---- ours, FFN + QKVO attention experts with a shared router (HF kd_1phase) ---
# Same probe contract as run_ours; only the model config differs
# (configs/model/flame-shared-router-hybrid-experts.sh, r256 full-rank LoRA
# attention experts) and expanded checkpoints resume from their source count.
run_hyb() {   # id experts:source real pass gpu port
  local id="$1" ne="${2%%:*}" src="${2##*:}" real="$3" pass="$4" gpu="$5" port="$6"
  local log="$OUT/logs/${id}_${pass}.log" npz="$OUT/hidden/${id}.npz" acc="$OUT/acc/${id}.txt"
  if [ "$pass" = dump ] && [ -s "$npz" ]; then echo "[SKIP] $id dump"; return; fi
  if [ "$pass" = acc ] && [ -s "$acc" ]; then echo "[SKIP] $id acc"; return; fi
  [ -f "$real/latest_checkpointed_iteration.txt" ] || { echo "[MISS] $id $real"; return; }
  local -a resume=(); [ "$ne" -gt "$src" ] && resume=(--shared-router-hybrid-resume-from-num-experts "$src" --shared-router-hybrid-train-new-experts-and-router-only)
  local -a dump=(); [ "$pass" = dump ] && dump=(--hidden-space-dump-path "$npz" --hidden-space-dump-label "$id" "${DUMP_ARGS[@]}")
  echo "[RUN] $id $pass gpu=$gpu (hybrid experts=$ne source=$src)"
  if [ "$DRY" = 1 ]; then echo "     torchrun(hybrid) … --load $real ${resume[*]} ${dump[*]}"; return; fi
  (
    export TOKENIZER_MODEL=/data2/seonghyeonnoh/homecache/huggingface/hub/models--EleutherAI--pythia-12b/snapshots/bb1e3e710cdf6b524461d543cfb5ba773f0a81b6
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
      --seq-length 512 --data-path 1.0 "$WIKI_PROBE" --split 100,0,0 --train-iters 1 --skip-train \
      --load "$real" --no-load-optim --no-load-rng --finetune "${resume[@]}" \
      --eval-interval 1 --eval-iters 0 --probe-name wiki_probe --probe-eval-iters "$PROBE_ITERS" --probe-eval-interval 1 \
      --probe-data-path 1.0 "$WIKI_PROBE" --run-initial-probe-eval "${dump[@]}"
  ) > "$log" 2>&1
  if [ "$pass" = acc ]; then
    grep -E '^probe wiki_probe ' "$log" | tail -1 > "$acc"
    [ -s "$acc" ] && echo "[DONE] $id acc: $(grep -oE 'next_token_acc: [0-9.]+' "$acc")" || echo "[FAIL] $id acc (see $log)"
  else
    [ -s "$npz" ] && echo "[DONE] $id dump" || echo "[FAIL] $id dump (see $log)"
  fi
}

run_job() {   # jobline pass gpu port
  IFS='|' read -r id kind spec task real extra <<< "$1"
  id=$(trim "$id"); kind=$(trim "$kind"); spec=$(trim "$spec"); task=$(trim "$task"); real=$(trim "$real"); extra=$(trim "${extra:-}")
  case "$kind" in
    base) run_base "$id" "$spec" "$task" "$real" "$extra" "$2" "$3" "$4" ;;
    ours) run_ours "$id" "$spec" "$real" "$2" "$3" "$4" ;;
    hyb)  run_hyb  "$id" "$spec" "$real" "$2" "$3" "$4" ;;
  esac
}

# ---- two GPU streams, jobs interleaved; acc pass then dump pass per job --------
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
stream() {   # stream_index gpu
  # Drop the lock fd here: train_stage.sh forks a background nvidia-smi logger
  # that would otherwise inherit it and keep the lock alive after the driver is
  # killed, so the next launch aborts on a lock nobody useful holds.
  exec 9>&-
  local si="$1" gpu="$2" i=0 port=$((${PORT_BASE:-37100} + si * 100))
  for job in "${JOBS[@]}"; do
    if [ $(( i % ${#GPU_LIST[@]} )) -eq "$si" ]; then
      run_job "$job" acc  "$gpu" $((port + i * 2))
      run_job "$job" dump "$gpu" $((port + i * 2 + 1))
    fi
    i=$((i + 1))
  done
  echo "[STREAM $si gpu=$gpu] finished $(date '+%F %T')"
}

echo "[CONFIG] out=$OUT gpus=$GPUS mb=$MB gbs=$GBS probe_iters=$PROBE_ITERS max_tokens=$MAX_TOKENS layers=$LAYERS seed=$SEED"
echo "[CONFIG] probe=$WIKI_PROBE jobs=${#JOBS[@]}"
for si in "${!GPU_LIST[@]}"; do stream "$si" "${GPU_LIST[$si]}" & done
wait
echo "[ALL DONE] $(ls "$OUT/hidden"/*.npz 2>/dev/null | wc -l)/${#JOBS[@]} dumps, $(ls "$OUT/acc"/*.txt 2>/dev/null | wc -l)/${#JOBS[@]} acc  $(date '+%F %T')"
