#!/bin/bash
# Dense transformer configuration for mixed wiki+code pretraining.

MODEL_ARGS=(
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-layers $NUM_LAYERS
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups ${NUM_QUERY_GROUPS:-16}
    --swiglu
    --max-position-embeddings 2048
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --disable-bias-linear
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --init-method-std ${INIT_METHOD_STD:-0.02}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model EleutherAI/pythia-12b
)
