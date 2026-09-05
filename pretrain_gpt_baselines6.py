"""Dedicated Wiki -> Code -> Conversation entrypoint for six baselines."""

from megatron.core.enums import ModelType
from megatron.core.continual_learning.branch import add_baselines6_args, install_baselines6_branch

import pretrain_gpt as base


install_baselines6_branch(base)


if __name__ == "__main__":
    base.train_valid_test_datasets_provider.is_distributed = True
    base.pretrain(
        base.train_valid_test_datasets_provider,
        base.model_provider,
        ModelType.encoder_or_decoder,
        base.forward_step,
        probe_eval_func=base.run_probe_evaluation,
        router_memory_step_func=base.router_memory_step,
        router_memory_eval_func=base.router_memory_eval,
        router_memory_accum_func=base.router_memory_accum_step,
        extra_args_provider=add_baselines6_args,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
