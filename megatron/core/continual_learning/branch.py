"""Process-local branch installer for the six-baseline entrypoint.

Nothing in the ordinary ``pretrain_gpt.py`` or ``training.py`` path imports
this module.  The dedicated entrypoint installs wrappers only in its own
Python process, preserving the existing experiment path byte-for-byte.
"""

from __future__ import annotations

import dataclasses

import torch

from megatron.core import mpu
from megatron.core.continual_learning.layer_specs import get_continual_dense_decoder_block_spec
from megatron.core.continual_learning.parameter_scope import (
    apply_parameter_scope,
    named_parameters,
)
from megatron.core.continual_learning.runtime import (
    configure_continual_learning,
    continual_after_optimizer_step,
    continual_before_optimizer_step,
    continual_loss_penalty,
    continual_on_train_end,
)
from megatron.training import get_args
import megatron.training.training as training_module


def add_baselines6_args(parser):
    group = parser.add_argument_group(title="six continual-learning baselines")
    group.add_argument(
        "--continual-method",
        choices=["none", "sequential_dense", "ewc", "trace_gem", "slora_pre", "olora", "fixed_moe"],
        default="none",
    )
    group.add_argument(
        "--continual-task-name", choices=["wiki", "code", "conversation"], required=True
    )
    group.add_argument("--continual-trainable-layer-start", type=int, default=2)
    group.add_argument("--continual-trainable-layer-end", type=int, default=9)
    group.add_argument("--continual-dense-ffn-hidden-size", type=int, default=1408)
    group.add_argument("--continual-state-load", type=str, default=None)
    group.add_argument("--continual-capture-trace-terminal", action="store_true")

    group.add_argument("--continual-ewc-lambda", type=float, default=400.0)
    group.add_argument("--continual-ewc-fisher-batches", type=int, default=100)

    group.add_argument("--continual-trace-gem-margin", type=float, default=0.5)
    group.add_argument("--continual-trace-gem-eps", type=float, default=1.0e-6)

    group.add_argument("--continual-slora-rank", type=int, default=64)
    group.add_argument("--continual-slora-conversation-rank", type=int, default=64)
    group.add_argument("--continual-slora-max-rank", type=int, default=256)
    group.add_argument("--continual-slora-alpha", type=float, default=128.0)
    group.add_argument(
        "--continual-slora-denoise-mode", choices=["max", "min", "minor"], default="max"
    )

    group.add_argument("--continual-olora-rank", type=int, default=352)
    group.add_argument("--continual-olora-alpha", type=float, default=352.0)
    group.add_argument("--continual-olora-dropout", type=float, default=0.1)
    group.add_argument("--continual-olora-orth-lambda", type=float, default=0.5)
    return parser


def _install_model_branch(base_module):
    original_config_factory = base_module.core_transformer_config_from_args
    config_box = {}

    def baseline_config_factory(args):
        config = original_config_factory(args)
        for name in (
            "continual_method",
            "continual_trainable_layer_start",
            "continual_trainable_layer_end",
            "continual_dense_ffn_hidden_size",
            "continual_slora_rank",
            "continual_slora_conversation_rank",
            "continual_slora_max_rank",
            "continual_slora_alpha",
            "continual_olora_rank",
            "continual_olora_alpha",
            "continual_olora_dropout",
        ):
            setattr(config, name, getattr(args, name))
        if args.continual_method != "fixed_moe":
            if args.num_layers != 9:
                raise ValueError("The matched six-baseline architecture requires exactly 9 layers")
            if args.continual_dense_ffn_hidden_size != 1408:
                raise ValueError("The matched dense Layer-2--9 FFN width must be 1408")
            # Layer 1 has FFN width 5472 while Layers 2--9 have width 1408.
            # TransformerBlock uses a list-valued moe_layer_freq as its existing
            # signal that layer checkpoint shapes are heterogeneous.  The all-zero
            # pattern does not create MoE layers in this dedicated dense spec; it
            # only keeps each layer under a distinct distributed-checkpoint key.
            config.moe_layer_freq = [0] * args.num_layers
        if args.continual_method in {"slora_pre", "olora"} and args.tensor_model_parallel_size != 1:
            raise ValueError("SLoRA/O-LoRA baseline adapters currently require tensor parallel size 1")
        config_box["config"] = config
        return config

    def baseline_local_spec(*_args, **_kwargs):
        if "config" not in config_box:
            raise RuntimeError("continual config must be constructed before its layer spec")
        return get_continual_dense_decoder_block_spec(config_box["config"])

    base_module.core_transformer_config_from_args = baseline_config_factory
    base_module.get_gpt_layer_local_spec = baseline_local_spec


def _install_loss_branch(base_module):
    original_loss = base_module.loss_func

    def baseline_loss(*args, **kwargs):
        gradient_loss, local_tokens, reporting = original_loss(*args, **kwargs)
        penalty = continual_loss_penalty()
        if penalty is None:
            return gradient_loss, local_tokens, reporting
        gradient_loss = gradient_loss + penalty * local_tokens.to(penalty.dtype)
        report_value = penalty.detach().clone()
        torch.distributed.all_reduce(report_value, group=mpu.get_data_parallel_group())
        report_value /= mpu.get_data_parallel_world_size()
        reporting["continual penalty"] = report_value
        return gradient_loss.clone(), local_tokens, reporting

    base_module.loss_func = baseline_loss


def _install_probe_branch(base_module):
    """Hold probe evaluation on one fixed, repeatable sample window.

    This branch runs with ``--dataloader-type cyclic`` so the training iterator
    is wrapped in ``cyclic_iter`` and the EWC Fisher pass can keep drawing
    minibatches after the finite training schedule ends.  The same flag also
    hands every probe loader a ``MegatronPretrainingRandomSampler``, whose
    ``consumed_samples`` cursor advances as it yields.  Probe loaders are built
    once, cached, and re-iterated at each probe interval without a cyclic
    wrapper, and the dataloader workers prefetch past what an evaluation
    actually consumes, so that cursor walks off the end of a probe dataset
    sized to exactly ``probe_eval_iters * global_batch_size`` and raises
    ``StopIteration`` partway through the run.  The ``single`` sampler used by
    the existing path restarts from its construction offset on every
    ``__iter__``; restore precisely that behaviour, for probe loaders only.

    The builder is consulted once per probe evaluation, so rewinding here also
    keeps every probe measured on the same samples across iterations.
    """
    original_builder = base_module._build_probe_dataloader
    start_offsets = {}

    def baseline_probe_dataloader(*args, **kwargs):
        dataloader = original_builder(*args, **kwargs)
        sampler = getattr(dataloader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "consumed_samples"):
            start_offsets.setdefault(id(sampler), sampler.consumed_samples)
            sampler.consumed_samples = start_offsets[id(sampler)]
        return dataloader

    base_module._build_probe_dataloader = baseline_probe_dataloader


def _install_training_branch():
    original_setup = training_module.setup_model_and_optimizer
    original_get_model = training_module.get_model
    original_train = training_module.train

    def baseline_get_model(*model_args, **model_kwargs):
        """Apply the Layer 2--9 scope before the optimizer groups are formed.

        Megatron drops ``requires_grad=False`` tensors when it builds optimizer
        parameter groups, but it does so inside setup_model_and_optimizer,
        before this branch used to freeze anything.  Frozen tensors then stayed
        in the groups, and because Megatron's Adam is AdamW -- decoupled weight
        decay -- every step rescaled them by ``1 - lr * weight_decay`` even with
        an exactly zero gradient.  Over a full 1800-step stage that is roughly a
        0.5% shrink of the embeddings, Layer 1, final norm and LM head, well past
        bf16's ~0.2% resolution, which is what trips the end-of-stage frozen
        checksum.  A two-step smoke moves them by ~6e-6 and cannot see it.
        """
        model = original_get_model(*model_args, **model_kwargs)
        runtime_args = get_args()
        if getattr(runtime_args, "continual_method", "none") != "none":
            apply_parameter_scope(
                training_module.unwrap_model(model),
                runtime_args.continual_method,
                runtime_args.continual_task_name,
                int(runtime_args.continual_trainable_layer_start),
                int(runtime_args.continual_trainable_layer_end),
            )
        return model

    def baseline_setup(*args, **kwargs):
        model, optimizer, scheduler = original_setup(*args, **kwargs)
        runtime_args = get_args()
        unwrapped = training_module.unwrap_model(model)
        configure_continual_learning(unwrapped, runtime_args, training_model=model)
        # Fail in seconds if anything frozen still reached the optimizer,
        # instead of discovering it from the checksum 1800 steps later.
        trainable_numel = sum(
            parameter.numel()
            for _name, parameter in named_parameters(unwrapped)
            if parameter.requires_grad
        )
        optimizer_numel = sum(
            parameter.numel()
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if optimizer_numel > trainable_numel:
            raise RuntimeError(
                "frozen tensors reached the optimizer: it holds "
                f"{optimizer_numel:,} parameters but only {trainable_numel:,} are "
                "trainable; the parameter scope must be applied before the "
                "optimizer is built"
            )
        # Adapter initialization/reset happens after checkpoint loading in this
        # isolated branch.  Megatron's BF16 optimizer owns FP32 main-parameter
        # copies, so refresh those copies after any out-of-optimizer adapter
        # initialization; otherwise the first optimizer step would overwrite a
        # fresh LoRA A factor with the stale pre-reset value.
        optimizer.reload_model_params()

        original_step = optimizer.step

        def baseline_optimizer_step(*step_args, **step_kwargs):
            current_step = int(getattr(runtime_args, "curr_iteration", runtime_args.iteration)) + 1
            continual_before_optimizer_step(current_step)
            result = original_step(*step_args, **step_kwargs)
            update_successful = bool(result[0])
            continual_after_optimizer_step(current_step, update_successful)
            return result

        optimizer.step = baseline_optimizer_step
        return model, optimizer, scheduler

    def baseline_train(
        forward_step_func,
        model,
        optimizer,
        opt_param_scheduler,
        train_data_iterator,
        valid_data_iterator,
        probe_eval_func,
        process_non_loss_data_func,
        config,
        checkpointing_context,
        non_loss_data_func,
        router_memory_step_func=None,
        router_memory_eval_func=None,
        router_memory_accum_func=None,
    ):
        result = original_train(
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            probe_eval_func,
            process_non_loss_data_func,
            config,
            checkpointing_context,
            non_loss_data_func,
            router_memory_step_func,
            router_memory_eval_func,
            router_memory_accum_func,
        )
        continual_on_train_end(optimizer, train_data_iterator, forward_step_func, config)
        return result

    training_module.get_model = baseline_get_model
    training_module.setup_model_and_optimizer = baseline_setup
    training_module.train = baseline_train


def install_baselines6_branch(base_module):
    _install_model_branch(base_module)
    _install_loss_branch(base_module)
    _install_probe_branch(base_module)
    _install_training_branch()
