#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "[CHAIN] G2 router-LR sweep starts"
bash "$ROOT/scripts/experiment/a100/run_g2_router_lr_3sweep_postkd_noexpert_ramp_mha.sh"

echo "[CHAIN] G2 sweep complete; resuming trace V3 and then evaluating sparse-15"
bash "$ROOT/trace/scripts/resume_v3_then_sparse15_eval.sh"

echo "[CHAIN DONE] router-LR sweep -> trace V3 resume -> sparse-15 eval"
