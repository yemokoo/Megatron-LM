#!/bin/bash
# Shared-router hybrid transformer: routed LoRA attention experts plus FFN MoE experts
# sharing one router per layer.

MODEL_ARGS=(
    --spec megatron.core.models.gpt.shared_router_hybrid_layer_specs gpt_shared_router_hybrid_local_spec
    --shared-router-hybrid-model
    --hidden-size "$HIDDEN_SIZE"
    --ffn-hidden-size "$FFN_HIDDEN_SIZE"
    --num-layers "$NUM_LAYERS"
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups "${NUM_QUERY_GROUPS:-16}"
    --swiglu
    --max-position-embeddings 2048
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --init-method-std "${INIT_METHOD_STD:-0.02}"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/pythia-12b

    --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
    --num-experts "${NUM_EXPERTS:-4}"
    --moe-router-topk "${MOE_ROUTER_TOPK:-2}"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.01}"
    --moe-z-loss-coeff "${MOE_Z_LOSS_COEFF:-0.001}"

    --attn-lora-num-experts "${NUM_EXPERTS:-4}"
    --attn-lora-rank "${ATTN_LORA_RANK:-16}"
    --attn-lora-topk "${MOE_ROUTER_TOPK:-2}"
    --attn-lora-alpha "${ATTN_LORA_ALPHA:-16}"
    ${ATTN_LORA_INCLUDE_PROJ:+--attn-lora-include-proj}
    --attn-full-rank-lora-rank "${ATTN_FULL_RANK_LORA_RANK:-0}"
    --attn-full-rank-lora-alpha "${ATTN_FULL_RANK_LORA_ALPHA:-1.0}"
    --attn-full-rank-lora-targets "${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
    --attn-full-rank-lora-active-targets "${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
)
