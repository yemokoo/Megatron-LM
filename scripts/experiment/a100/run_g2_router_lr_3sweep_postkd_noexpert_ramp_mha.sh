#!/usr/bin/env bash
set -euo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BASE="${LOCAL_BASE:-/data3/seonghyeonnoh/LLM-continual-learning-runs/local}"
LOG_ROOT="${ROUTER_LR_SWEEP_LOG_ROOT:-$LOCAL_BASE/logs/g2_router_lr_3sweep_postkd_noexpert_ramp}"
mkdir -p "$LOG_ROOT"

# Expert LR stays on the canonical 3e-4 schedule with no expert-only ramp.
# Only the router optimizer group changes across the three runs.
LABELS=(rlr1over3 rlr1over10 rlr1over30)
MULTIPLIERS=(0.3333333333333333 0.1 0.03333333333333333)

for index in "${!LABELS[@]}"; do
    label="${LABELS[$index]}"
    multiplier="${MULTIPLIERS[$index]}"
    experiment_tag="c100to10_s600_normfix_postkd_noeramp_${label}"
    run_tag="c100to10-s600-normfix-postkd-noeramp-${label}"
    variant_log="$LOG_ROOT/${label}.log"

    echo "[SWEEP START] label=$label router_lr_multiplier=$multiplier expert_lr=3e-4 expert_ramp=0"
    env \
        STOP_AFTER_R4=1 \
        OLD_HIDDEN_KL_COEFF=10 \
        OLD_HIDDEN_KL_COEFF_START=100 \
        OLD_HIDDEN_KL_COEFF_DECAY_STEPS=600 \
        HIDDEN_KL_EXPERIMENT_TAG="$experiment_tag" \
        HIDDEN_KL_RUN_TAG="$run_tag" \
        CHAIN_LOG_DIR="$LOG_ROOT/$label" \
        MOE_NEW_EXPERT_LR_RAMP_STEPS=0 \
        MOE_ROUTER_LR_MULTIPLIER="$multiplier" \
        LR=3e-4 \
        MIN_LR=3e-5 \
        bash "$D/run_g2_7run_hidden_kl_c10_normfix_postfirst_h100_chain_mha.sh" \
        2>&1 | tee "$variant_log"
    echo "[SWEEP DONE] label=$label"
done

echo "[ALL DONE] router LR 1/3, 1/10, 1/30 sweep with post-KD teachers"
