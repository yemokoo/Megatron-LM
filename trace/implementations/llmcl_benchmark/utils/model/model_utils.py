# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
try:
    from transformers.integrations import HfDeepSpeedConfig   # transformers >= 4.40
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig       # older transformers
from transformers import LlamaForCausalLM, LlamaConfig


def resolve_attention_implementation(requested="auto"):
    """Resolve the safe native Transformers attention backend for this host."""
    if requested is None:
        return None
    if requested != "auto":
        return requested
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            return "flash_attention_2"
    except ImportError:
        pass
    return "sdpa"


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    torch_dtype=None,
                    low_cpu_mem_usage=None,
                    attn_implementation=None,
                    forbid_vocab_growth=False,
                    device_map=None,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    target_vocab_size = int(8 * math.ceil(len(tokenizer) / 8.0))
    # Shrinking away unused padded rows is deterministic; growing would create
    # random embedding/lm-head rows that an expert/router-only delta cannot restore.
    if forbid_vocab_growth and target_vocab_size > model_config.vocab_size:
        raise ValueError(
            "Expert/router-only checkpoints cannot reproduce randomly initialized "
            f"grown embeddings: base vocab={model_config.vocab_size}, rounded "
            f"tokenizer vocab={target_vocab_size}. Use the matching base tokenizer/model.")

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    load_kwargs = {
        "from_tf": bool(".ckpt" in model_name_or_path),
        "config": model_config,
        "trust_remote_code": True,
    }
    # Loading a multi-billion-parameter model directly in its training dtype avoids
    # first materialising an FP32 copy and then allocating another BF16 copy on the
    # GPU. Keep these kwargs optional so every existing caller retains its old
    # behaviour.
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if low_cpu_mem_usage is not None:
        # Transformers 4.51 requires Accelerate for this path. Eval-only installs
        # may intentionally omit it; direct BF16 loading is still valid there.
        try:
            from transformers.utils import is_accelerate_available
            accelerate_available = is_accelerate_available()
        except ImportError:
            accelerate_available = False
        if not low_cpu_mem_usage or accelerate_available:
            load_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        # Evaluation can shard a model across several visible GPUs through
        # Accelerate. Training callers leave this unset and retain their existing
        # explicit model.to(local_device) behaviour.
        load_kwargs["device_map"] = device_map

    model = model_class.from_pretrained(model_name_or_path, **load_kwargs)

    # llama use eos_token_id but not end_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    # compatible with OPT and llama2
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(target_vocab_size)  # make the vocab size multiple of 8

    return model
