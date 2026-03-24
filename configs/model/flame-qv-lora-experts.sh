#!/bin/bash
# Dense transformer with routed LoRA experts attached to attention Q/V projections.

MODEL_ARGS=(
    --spec megatron.core.models.gpt.qv_lora_layer_specs gpt_qv_lora_local_spec
    --hidden-size $HIDDEN_SIZE
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --num-layers $NUM_LAYERS
    --num-attention-heads 16
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
    --attn-lora-num-experts ${ATTN_LORA_NUM_EXPERTS:-4}
    --attn-lora-rank ${ATTN_LORA_RANK:-16}
    --attn-lora-topk ${ATTN_LORA_TOPK:-1}
    --attn-lora-alpha ${ATTN_LORA_ALPHA:-16}
)
