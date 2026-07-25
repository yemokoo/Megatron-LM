#!/bin/bash

MODEL_ARGS=(
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

    --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE"
    --num-experts "${NUM_EXPERTS:-4}"
    --moe-router-topk "${MOE_ROUTER_TOPK:-2}"
    --moe-layer-freq "$MOE_LAYER_FREQ"
    --moe-router-dtype fp32
    --moe-router-pre-softmax
    --moe-router-score-function softmax
    --moe-aux-loss-coeff "${MOE_AUX_LOSS_COEFF:-0.01}"
    --moe-z-loss-coeff "${MOE_Z_LOSS_COEFF:-0.001}"

    --hidden-dropout 0.0
    --attention-dropout 0.0
    --init-method-std "${INIT_METHOD_STD:-0.02}"

    --tokenizer-type HuggingFaceTokenizer
    # Keep the Hub ID as the default, but allow offline runs to point to a
    # project-local tokenizer snapshot.
    --tokenizer-model "${TOKENIZER_MODEL:-EleutherAI/pythia-12b}"
)
