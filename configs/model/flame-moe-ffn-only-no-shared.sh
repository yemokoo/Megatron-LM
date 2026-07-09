#!/bin/bash
# FFN-only MoE (no always-on shared expert, no attention experts).
#
# Used for the fixed-capacity MoE baseline (e.g. fixed24) where all experts are
# available from the start and the whole model is full-finetuned across tasks.
# Because there is no always-on shared expert, the active FFN parameters equal
# MOE_ROUTER_TOPK * MOE_FFN_HIDDEN_SIZE, which matches the dense active-parameter
# baseline (top-4 * 352 = 1408).

MODEL_ARGS=(
    # Network Size
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

    # Mixture of Experts (FFN only, no shared expert)
    --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
    --num-experts "${NUM_EXPERTS:-24}"
    --moe-router-topk "${MOE_ROUTER_TOPK:-4}"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-dtype "${MOE_ROUTER_DTYPE:-fp32}"
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.01}"
    --moe-z-loss-coeff "${MOE_Z_LOSS_COEFF:-0.001}"

    # Regularization
    --hidden-dropout 0.0
    --attention-dropout 0.0

    # Initialization
    --init-method-std "${INIT_METHOD_STD:-0.02}"

    # Tokenizer
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/pythia-12b
)

if [ "${MOE_GROUPED_GEMM:-0}" = "1" ]; then
    MODEL_ARGS+=(--moe-grouped-gemm)
fi
