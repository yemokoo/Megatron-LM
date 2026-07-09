#!/bin/bash
set -euo pipefail

# G2 dense (activation-matched) baseline: wiki -> code -> conversation.
#
# Goal: a DENSE counterpart to the MoE models whose layer 1 is IDENTICAL to the
# MoE models (dense FFN 5472) so that the only thing changing across baselines is
# layers 2-9 (dense vs MoE). A pure dense Megatron model cannot have a per-layer
# FFN size (`--ffn-hidden-size` is a single int), so we realize "dense in layers
# 2-9" via a 1-expert top-1 MoE:
#
#   layer 1      : dense MLP, ffn_hidden_size = 5472           (== every MoE model)
#   layers 2-9   : MoE with num_experts=1, topk=1, moe_ffn=1408
#                  -> router softmax over a single expert = gate 1.0
#                  -> functionally a monolithic dense FFN of width 1408
#                     (the active FFN width of the top-4 MoE: 4 * 352 = 1408)
#
# aux/z load-balancing losses are disabled (meaningless with a single expert).
#
# This is a thin wrapper over run_g2_fixed24_wiki_code_conversation_mha.sh (same
# 3-stage full-finetune engine, same 3-probe logging, same KD option).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- dense(active) architecture: 1-expert top-1 in layers 2-9 ---
export VARIANT_TAG="${VARIANT_TAG:-dense-active-matched}"
export VARIANT_LABEL="${VARIANT_LABEL:-dense(active) layer1-5472 layers2to9-dense1408}"
export NUM_EXPERTS="${NUM_EXPERTS:-1}"
export MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-1}"
export MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-1408}"   # dense width of layers 2-9
export FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-5472}"           # dense width of layer 1 (shared with MoE models)
export MOE_AUX_LOSS_COEFF="${MOE_AUX_LOSS_COEFF:-0.0}"
export MOE_Z_LOSS_COEFF="${MOE_Z_LOSS_COEFF:-0.0}"
# 1 expert: grouped_gemm brings no benefit and adds a fragile dependency.
export MOE_GROUPED_GEMM="${MOE_GROUPED_GEMM:-0}"
# dense(active) is light (top-1 of 1408), but the code/conversation stages run
# old-model logits KD (a full teacher forward + fp32 vocab-size softmax), which at
# mb72 peaked at ~80 GB (OOM-adjacent on 80 GB A100s). Drop to 48 so the KD stages
# have headroom; the wiki stage (no KD) is safe at 48 too.
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-48}"

# Distinct default master ports so it can run without clashing with fixed24.
export WIKI_MASTER_PORT="${WIKI_MASTER_PORT:-29971}"
export CODE_MASTER_PORT="${CODE_MASTER_PORT:-29972}"
export CONV_MASTER_PORT="${CONV_MASTER_PORT:-29973}"

exec bash "$SCRIPT_DIR/run_g2_fixed24_wiki_code_conversation_mha.sh"
