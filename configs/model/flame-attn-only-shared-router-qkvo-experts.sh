#!/bin/bash
# Dense-FFN transformer with shared-router QKVO attention experts only.
# The router controls attention full-rank LoRA experts on Q/K/V/O; FFN remains dense.

MODEL_ARGS=(
    --spec megatron.core.models.gpt.shared_router_hybrid_layer_specs gpt_shared_router_attention_only_local_spec
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
    --tokenizer-model "${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"

    --num-experts "${NUM_EXPERTS:-8}"
    --moe-router-topk "${MOE_ROUTER_TOPK:-4}"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.01}"
    --moe-z-loss-coeff "${MOE_Z_LOSS_COEFF:-0.001}"

    --attn-lora-num-experts "${NUM_EXPERTS:-8}"
    --attn-lora-rank "${ATTN_LORA_RANK:-256}"
    --attn-lora-topk "${MOE_ROUTER_TOPK:-4}"
    --attn-lora-alpha "${ATTN_LORA_ALPHA:-256}"
    --attn-full-rank-lora-rank "${ATTN_FULL_RANK_LORA_RANK:-256}"
    --attn-full-rank-lora-alpha "${ATTN_FULL_RANK_LORA_ALPHA:-256}"
    --attn-full-rank-lora-targets "${ATTN_FULL_RANK_LORA_TARGETS:-qkvo}"
    --attn-full-rank-lora-active-targets "${ATTN_FULL_RANK_LORA_ACTIVE_TARGETS:-}"
)

if [ "${MOE_ROUTER_DTYPE:-fp32}" != "none" ]; then
    MODEL_ARGS+=(--moe-router-dtype "${MOE_ROUTER_DTYPE:-fp32}")
fi

if [ "${ATTN_LORA_GROUPED_GEMM:-0}" = "1" ]; then
    MODEL_ARGS+=(--attn-lora-grouped-gemm)
fi

if [ "${SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS:-0}" = "1" ]; then
    MODEL_ARGS+=(
        --shared-router-train-mask-existing-experts
        --shared-router-train-mask-existing-experts-from-num-experts "${SHARED_ROUTER_TRAIN_MASK_EXISTING_EXPERTS_FROM_NUM_EXPERTS:-${SOURCE_NUM_EXPERTS:-0}}"
    )
fi

if [ "${SHARED_ROUTER_HYBRID_TOPK_WITH_ALL_NEW_EXPERTS:-0}" = "1" ]; then
    MODEL_ARGS+=(
        --shared-router-hybrid-topk-with-all-new-experts
        --shared-router-hybrid-all-new-experts-from-num-experts "${SHARED_ROUTER_HYBRID_ALL_NEW_EXPERTS_FROM_NUM_EXPERTS:-${SOURCE_NUM_EXPERTS:-0}}"
    )
fi

if [ -n "${MOE_EXPERT_CAPACITY_FACTOR:-}" ]; then
    MODEL_ARGS+=(--moe-expert-capacity-factor "$MOE_EXPERT_CAPACITY_FACTOR")
fi

if [ "${MOE_PAD_EXPERT_INPUT_TO_CAPACITY:-0}" = "1" ]; then
    MODEL_ARGS+=(--moe-pad-expert-input-to-capacity)
fi
