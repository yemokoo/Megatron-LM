# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import math
import os
import json
import torch
import torch.nn.functional as F
from functools import partial
from contextlib import contextmanager, nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import print_rank_last
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import get_rerun_state_machine
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.shared_router_hybrid import capture_shared_router_inputs
from megatron.core.transformer.moe.continual_learning_utils import teacher_student_router_kl
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    unwrap_model,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.training import (
    _collect_current_shared_routers,
    _router_logits,
    accumulate_shared_router_teacher_student_memory_kl,
    evaluate_shared_router_memory_kl,
    get_old_moe_distill_teacher,
    run_shared_router_memory_distillation_step,
)
from megatron.training.global_vars import get_tensorboard_writer, get_wandb_writer
from megatron.legacy.data.data_samplers import build_pretraining_data_loader
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.full_rank_lora_layer_specs import (
    get_gpt_full_rank_lora_decoder_block_spec,
    get_gpt_full_rank_lora_layer_local_spec,
)


stimer = StragglerDetector()
_PROBE_DATALOADER = None
_ROUTER_MEMORY_DATALOADER = None
_ROUTER_MEMORY_ITERATOR = None
_ROUTER_MEMORY_EVAL_DATALOADER = None
_ROUTER_MEMORY_EVAL_ITERATOR = None

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            dump(snapshot, open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'))

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...') #make gpt model from config and spec.
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if config.attn_full_rank_lora_rank > 0:
                if use_te:
                    raise ValueError('Packed attention full-rank LoRA is currently supported only with local transformer_impl.')
                if args.num_experts:
                    transformer_layer_spec = get_gpt_full_rank_lora_decoder_block_spec(config)
                else:
                    transformer_layer_spec = get_gpt_full_rank_lora_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.moe_use_legacy_grouped_gemm,
                    )
            elif args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=use_te)
            else:
                # Define the decoder layer spec
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm,
                        args.qk_layernorm, args.multi_latent_attention, args.moe_use_legacy_grouped_gemm)

        build_model_context = nullcontext
        build_model_context_args = {}
        if args.fp8_param_gather:
            try:
                from transformer_engine.pytorch import fp8_model_init

                build_model_context = fp8_model_init
                build_model_context_args["enabled"] = True

                # Check if fp8_model_init supports preserve_high_precision_init_val
                if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                    build_model_context_args["preserve_high_precision_init_val"] = True
            except:
                raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling
            )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    if getattr(get_args(), "moe_lpr_loss_coeff", 0.0) > 0.0:
        get_args()._moe_lpr_dataset_ids = batch["dataset_id"]
    return (
        batch["tokens"], batch["labels"], batch["loss_mask"],
        batch["attention_mask"], batch["position_ids"],
    )


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    teacher_logits = None
    student_logits = None
    hidden_mse_loss = None
    router_kl_loss = None
    lpr_loss = None
    if isinstance(output_tensor, dict):
        losses = output_tensor["losses"].float()
        teacher_logits = output_tensor.get("teacher_logits")
        student_logits = output_tensor.get("student_logits")
        hidden_mse_loss = output_tensor.get("hidden_mse_loss")
        router_kl_loss = output_tensor.get("router_kl_loss")
        lpr_loss = output_tensor.get("lpr_loss")
    else:
        losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    lm_loss = torch.sum(losses.view(-1) * loss_mask)
    lm_loss = lm_loss * getattr(args, "moe_expansion_distill_lm_loss_coeff", 1.0)
    loss = torch.cat([lm_loss.view(1), total_tokens.view(1)])

    if teacher_logits is not None and student_logits is not None:
        temperature = args.moe_old_model_kl_temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
        kl_per_token = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl_loss_sum = torch.sum(kl_per_token.view(-1) * loss_mask)
        kl_loss = kl_loss_sum / total_tokens.clamp_min(1.0)
        kl_loss = kl_loss * (temperature ** 2)
        if getattr(args, "moe_expansion_distill_mode", "none") != "none":
            loss[0] = loss[0] + args.moe_old_model_kl_coeff * kl_loss_sum * (temperature ** 2)
        else:
            loss[0] = loss[0] + args.moe_old_model_kl_coeff * kl_loss

    if hidden_mse_loss is not None:
        loss[0] = loss[0] + args.moe_expansion_distill_hidden_mse_coeff * hidden_mse_loss

    if router_kl_loss is not None:
        loss[0] = loss[0] + args.moe_expansion_distill_router_kl_coeff * router_kl_loss

    if lpr_loss is not None:
        loss[0] = loss[0] + args.moe_lpr_loss_coeff * lpr_loss

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,        # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    reporting = {'lm loss': (reporting_loss[0], reporting_loss[1])}
    if teacher_logits is not None and student_logits is not None:
        reporting_kl_sum = (kl_loss_sum.detach() * (temperature ** 2)).view(1)
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(
                reporting_kl_sum, group=mpu.get_context_parallel_group()
            )
        torch.distributed.all_reduce(
            reporting_kl_sum, group=mpu.get_data_parallel_group()
        )
        reporting['kd loss'] = (reporting_kl_sum[0], reporting_loss[1])
    if hidden_mse_loss is not None:
        hidden_mse_sum = hidden_mse_loss.detach().view(1)
        torch.distributed.all_reduce(hidden_mse_sum, group=mpu.get_data_parallel_group())
        reporting['hidden mse loss'] = (hidden_mse_sum[0], reporting_loss[1])
    if router_kl_loss is not None:
        router_kl_sum = router_kl_loss.detach().view(1)
        torch.distributed.all_reduce(router_kl_sum, group=mpu.get_data_parallel_group())
        reporting['router prob kl loss'] = (router_kl_sum[0], reporting_loss[1])
    if lpr_loss is not None:
        lpr_sum = lpr_loss.detach().view(1)
        torch.distributed.all_reduce(lpr_sum, group=mpu.get_data_parallel_group())
        reporting['lpr loss'] = (lpr_sum[0], reporting_loss[1])

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        reporting,
    )


def _masked_task_group_lpr(router_inputs, routers, labels, loss_mask, dataset_ids, args):
    """Summed old-task group NLL, averaged across MoE layers."""
    router_inputs = _router_inputs_by_layer(router_inputs)
    common_layers = sorted(set(router_inputs) & set(routers))
    if not common_layers:
        raise RuntimeError("Task-group LPR captured no MoE router inputs.")

    prefix_counts = [int(v) for v in args.moe_lpr_dataset_prefix_counts.split(",")]
    range_specs = args.moe_lpr_task_expert_ranges.split(",")
    expert_ranges = [None if v == "-" else tuple(map(int, v.split(":"))) for v in range_specs]
    if len(prefix_counts) != len(expert_ranges) or any(v <= 0 for v in prefix_counts):
        raise RuntimeError("Invalid LPR task prefix-count/range specification.")
    if dataset_ids.dim() != 1 or dataset_ids.shape[0] != labels.shape[0]:
        raise RuntimeError(f"LPR dataset_id shape mismatch: {tuple(dataset_ids.shape)} vs {tuple(labels.shape)}")

    sample_task_ids = torch.empty_like(dataset_ids)
    lower = 0
    for task_id, count in enumerate(prefix_counts):
        upper = lower + count
        sample_task_ids[(dataset_ids >= lower) & (dataset_ids < upper)] = task_id
        lower = upper
    if int(dataset_ids.max().item()) >= lower:
        raise RuntimeError(f"LPR dataset_id exceeds configured {lower} prefixes.")

    token_task_ids = sample_task_ids[:, None].expand_as(labels).reshape(-1)
    flat_loss_mask = loss_mask.reshape(-1).bool()
    layer_losses = []
    for layer_number in common_layers:
        flat_hidden = _flatten_layer_hidden(router_inputs[layer_number], labels)
        log_probs = torch.log_softmax(routers[layer_number].gating(flat_hidden).float(), dim=-1)
        layer_loss = log_probs.new_zeros(())
        supervised = 0
        for task_id, expert_range in enumerate(expert_ranges):
            if expert_range is None:
                continue
            token_mask = flat_loss_mask & (token_task_ids == task_id)
            count = int(token_mask.sum().item())
            if count == 0:
                continue
            start, end = expert_range
            if not (0 <= start < end <= log_probs.shape[-1]):
                raise RuntimeError(f"LPR expert range {start}:{end} is invalid for {log_probs.shape[-1]} experts.")
            layer_loss = layer_loss - torch.logsumexp(log_probs[token_mask, start:end], dim=-1).sum()
            supervised += count
        if supervised:
            layer_losses.append(layer_loss)
    if not layer_losses:
        raise RuntimeError("Task-group LPR found zero supervised old-task tokens.")
    return torch.stack(layer_losses).mean()


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
            data_iterator)
    timers('batch-generator').stop()

    teacher_model = get_old_moe_distill_teacher()
    distill_mode = getattr(args, "moe_expansion_distill_mode", "none")
    expansion_distill_enabled = distill_mode != "none"
    if expansion_distill_enabled and teacher_model is None:
        raise RuntimeError(
            "--moe-expansion-distill-mode requires a loaded pre-expansion teacher. "
            "Check --moe-old-model-kl-load, --moe-expand-from-num-experts, and the "
            "source checkpoint path."
        )
    teacher_kd_enabled = teacher_model is not None and (
        args.moe_old_model_kl_coeff > 0 or expansion_distill_enabled
    )
    if teacher_kd_enabled:
        if args.moe_old_model_kl_coeff <= 0:
            raise RuntimeError(
                "--moe-expansion-distill-mode requires --moe-old-model-kl-coeff > 0 "
                "because all expansion-distill modes include final-logit KL."
            )
        if args.pipeline_model_parallel_size != 1 and (
            _distill_mode_includes_hidden(args) or _distill_mode_includes_router(args)
        ):
            raise RuntimeError(
                "Hidden/router expansion distillation currently requires "
                "--pipeline-model-parallel-size 1."
            )

        student_modules = _as_module_list(model)
        teacher_modules = _as_module_list(teacher_model[0])
        student_hidden_ctx = (
            _capture_transformer_layer_outputs(
                student_modules, args.moe_expansion_distill_hidden_layers, detach=False
            )
            if _distill_mode_includes_hidden(args)
            else nullcontext({})
        )
        teacher_hidden_ctx = (
            _capture_transformer_layer_outputs(
                teacher_modules, args.moe_expansion_distill_hidden_layers, detach=True
            )
            if _distill_mode_includes_hidden(args)
            else nullcontext({})
        )
        student_router_ctx = (
            _capture_distill_router_inputs(student_modules, detach=False)
            if _distill_mode_includes_router(args)
            else nullcontext(({}, {}))
        )
        teacher_router_ctx = (
            _capture_distill_router_inputs(teacher_modules, detach=True)
            if _distill_mode_includes_router(args)
            else nullcontext(({}, {}))
        )

        with stimer:
            with student_hidden_ctx as student_hidden, student_router_ctx as (
                student_router_inputs,
                student_routers,
            ):
                student_output = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=labels,
                    runtime_gather_output=True,
                    return_loss_and_logits=True,
                )
        output_tensor = student_output["losses"]
        student_logits = student_output["logits"]
        with torch.no_grad():
            with teacher_hidden_ctx as teacher_hidden, teacher_router_ctx as (
                teacher_router_inputs,
                teacher_routers,
            ):
                teacher_logits = teacher_model[0](
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=True,
                )
        output_tensor = {
            "losses": output_tensor,
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
        }
        if _distill_mode_includes_hidden(args):
            output_tensor["hidden_mse_loss"] = _masked_layer_hidden_mse(
                student_hidden,
                teacher_hidden,
                labels,
                loss_mask,
            )
        if _distill_mode_includes_router(args):
            output_tensor["router_kl_loss"] = _masked_router_prob_kl(
                student_router_inputs,
                teacher_router_inputs,
                student_routers,
                teacher_routers,
                labels,
                loss_mask,
            )
    elif args.moe_lpr_loss_coeff > 0.0 and _as_module_list(model)[0].training:
        with _capture_distill_router_inputs(_as_module_list(model), detach=False) as (lpr_inputs, lpr_routers):
            with stimer:
                losses = model(tokens, position_ids, attention_mask, labels=labels)
        dataset_ids = getattr(args, "_moe_lpr_dataset_ids", None)
        if dataset_ids is None:
            raise RuntimeError("LPR batch dataset IDs were not populated.")
        output_tensor = {
            "losses": losses,
            "lpr_loss": _masked_task_group_lpr(
                lpr_inputs, lpr_routers, labels, loss_mask, dataset_ids, args
            ),
        }
    else:
        with stimer:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def _build_probe_dataloader(probe_data_path, probe_eval_iters, cache_key):
    global _PROBE_DATALOADER

    if not isinstance(_PROBE_DATALOADER, dict):
        _PROBE_DATALOADER = {}

    if cache_key in _PROBE_DATALOADER:
        return _PROBE_DATALOADER[cache_key]

    args = get_args()
    if not probe_data_path or probe_eval_iters <= 0:
        return None

    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(probe_data_path),
        blend_per_split=None,
        split="0,1,0",
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )

    dataset_type = MockGPTDataset if args.mock_data else GPTDataset
    _, valid_ds, _ = BlendedMegatronDatasetBuilder(
        dataset_type,
        (0, probe_eval_iters * args.global_batch_size, 0),
        is_dataset_built_on_rank,
        config,
    ).build()
    training_micro_batch_size = args.micro_batch_size
    if args.probe_micro_batch_size is not None:
        args.micro_batch_size = args.probe_micro_batch_size
    try:
        _PROBE_DATALOADER[cache_key] = build_pretraining_data_loader(valid_ds, 0)
    finally:
        args.micro_batch_size = training_micro_batch_size
    return _PROBE_DATALOADER[cache_key]


def _get_router_memory_dataloader():
    global _ROUTER_MEMORY_DATALOADER
    args = get_args()
    if _ROUTER_MEMORY_DATALOADER is not None:
        return _ROUTER_MEMORY_DATALOADER
    if (
        not args.router_memory_data_path
        or (args.router_memory_kl_coeff <= 0 and not args.router_memory_force_enable_zero_coeff)
    ):
        return None

    interval = args.router_memory_interval
    if interval <= 0 and args.router_memory_fraction > 0:
        interval = max(1, int(round(1.0 / args.router_memory_fraction)))
    memory_steps = max(1, math.ceil(max(1, args.train_iters) / max(1, interval)))
    _ROUTER_MEMORY_DATALOADER = _build_probe_dataloader(
        args.router_memory_data_path,
        memory_steps,
        "router_memory",
    )
    return _ROUTER_MEMORY_DATALOADER


def _next_router_memory_batch():
    global _ROUTER_MEMORY_ITERATOR
    dataloader = _get_router_memory_dataloader()
    if dataloader is None:
        raise RuntimeError("Router-memory dataloader is not configured.")
    if _ROUTER_MEMORY_ITERATOR is None:
        _ROUTER_MEMORY_ITERATOR = iter(dataloader)
    try:
        return get_batch(_ROUTER_MEMORY_ITERATOR)
    except StopIteration:
        _ROUTER_MEMORY_ITERATOR = iter(dataloader)
        return get_batch(_ROUTER_MEMORY_ITERATOR)


def router_memory_step(model, optimizer, config, iteration):
    return run_shared_router_memory_distillation_step(
        model,
        optimizer,
        config,
        _next_router_memory_batch,
        iteration,
    )


def router_memory_accum_step(model, config, iteration):
    return accumulate_shared_router_teacher_student_memory_kl(
        model,
        config,
        _next_router_memory_batch,
        iteration,
    )


def _get_router_memory_eval_dataloader():
    global _ROUTER_MEMORY_EVAL_DATALOADER
    args = get_args()
    if _ROUTER_MEMORY_EVAL_DATALOADER is not None:
        return _ROUTER_MEMORY_EVAL_DATALOADER
    if args.router_memory_kl_coeff <= 0 and not args.router_memory_force_enable_zero_coeff:
        return None

    eval_data_path = args.router_memory_eval_data_path or args.router_memory_data_path
    if not eval_data_path:
        return None

    _ROUTER_MEMORY_EVAL_DATALOADER = _build_probe_dataloader(
        eval_data_path,
        max(1, args.router_memory_eval_iters),
        "router_memory_eval",
    )
    return _ROUTER_MEMORY_EVAL_DATALOADER


def _reset_router_memory_eval_iterator():
    global _ROUTER_MEMORY_EVAL_ITERATOR
    dataloader = _get_router_memory_eval_dataloader()
    if dataloader is None:
        raise RuntimeError("Router-memory fixed-probe dataloader is not configured.")
    # Reset on every diagnostic call so fixed-probe KL is measured on the same samples.
    _ROUTER_MEMORY_EVAL_ITERATOR = iter(dataloader)


def _next_router_memory_eval_batch():
    global _ROUTER_MEMORY_EVAL_ITERATOR
    if _ROUTER_MEMORY_EVAL_ITERATOR is None:
        _reset_router_memory_eval_iterator()
    try:
        return get_batch(_ROUTER_MEMORY_EVAL_ITERATOR)
    except StopIteration:
        _reset_router_memory_eval_iterator()
        return get_batch(_ROUTER_MEMORY_EVAL_ITERATOR)


def router_memory_eval(model, config, iteration):
    _reset_router_memory_eval_iterator()
    return evaluate_shared_router_memory_kl(
        model,
        config,
        _next_router_memory_eval_batch,
        iteration,
    )


def _align_logits(logits, labels):
    if logits.dim() != 3:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    if logits.shape[0] == labels.shape[0] and logits.shape[1] == labels.shape[1]:
        return logits
    if logits.shape[0] == labels.shape[1] and logits.shape[1] == labels.shape[0]:
        return logits.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"Could not align logits {tuple(logits.shape)} and labels {tuple(labels.shape)}")


def _probe_router_usage_num_existing_experts(args):
    explicit = getattr(args, "probe_router_usage_num_existing_experts", None)
    if explicit is not None:
        return int(explicit)
    for attr in (
        "shared_router_hybrid_resume_from_num_experts",
        "shared_router_hybrid_expand_from_num_experts",
        "attn_lora_resume_from_num_experts",
        "attn_lora_expand_from_num_experts",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            return int(value)
    if getattr(args, "num_experts", None):
        return max(0, int(args.num_experts) // 2)
    return 0


def _accumulate_probe_router_usage(captured, routers, num_existing_experts, args, totals, hist_totals):
    layer_old_fractions = []
    layer_new_fractions = []
    layer_new_prob_masses = []
    captured_any = False

    for layer_number, hidden_states in captured:
        layer_number = int(layer_number)
        router = routers.get(layer_number)
        if router is None:
            continue

        captured_any = True
        flat_hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        router_logits = _router_logits(router, flat_hidden)
        router_log_probs = torch.log_softmax(router_logits.float(), dim=-1)
        router_probs = router_log_probs.exp()

        num_experts = router_log_probs.shape[-1]
        num_existing = min(max(0, int(num_existing_experts)), num_experts)
        topk = min(args.moe_router_topk, num_experts)
        expert_idx = torch.topk(router_log_probs, k=topk, dim=-1).indices

        old_fraction = (expert_idx < num_existing).float().mean()
        new_fraction = 1.0 - old_fraction
        new_prob_mass = router_probs[:, num_existing:].sum(dim=-1).mean()
        hist = torch.bincount(expert_idx.reshape(-1), minlength=num_experts).float()
        hist = hist / hist.sum().clamp_min(1.0)

        prefix = f"layer_{layer_number}"
        totals[f"{prefix}/old_expert_fraction"] = (
            totals.get(f"{prefix}/old_expert_fraction", 0.0) + old_fraction.detach()
        )
        totals[f"{prefix}/new_expert_fraction"] = (
            totals.get(f"{prefix}/new_expert_fraction", 0.0) + new_fraction.detach()
        )
        totals[f"{prefix}/new_expert_prob_mass"] = (
            totals.get(f"{prefix}/new_expert_prob_mass", 0.0) + new_prob_mass.detach()
        )
        hist_totals[layer_number] = hist_totals.get(layer_number, 0.0) + hist.detach()

        layer_old_fractions.append(old_fraction)
        layer_new_fractions.append(new_fraction)
        layer_new_prob_masses.append(new_prob_mass)

    if layer_old_fractions:
        totals["old_expert_fraction"] = (
            totals.get("old_expert_fraction", 0.0)
            + torch.stack(layer_old_fractions).mean().detach()
        )
        totals["new_expert_fraction"] = (
            totals.get("new_expert_fraction", 0.0)
            + torch.stack(layer_new_fractions).mean().detach()
        )
        totals["new_expert_prob_mass"] = (
            totals.get("new_expert_prob_mass", 0.0)
            + torch.stack(layer_new_prob_masses).mean().detach()
        )

    return captured_any


def _finalize_probe_router_usage(totals, hist_totals, denom):
    reporting = {}
    if denom <= 0:
        return reporting

    denom = float(denom)
    for key, value in totals.items():
        tensor = value / denom
        torch.distributed.all_reduce(tensor, group=mpu.get_data_parallel_group())
        tensor = tensor / mpu.get_data_parallel_world_size()
        reporting[key] = tensor

    for layer_number, hist in hist_totals.items():
        hist = hist / denom
        torch.distributed.all_reduce(hist, group=mpu.get_data_parallel_group())
        hist = hist / mpu.get_data_parallel_world_size()
        for expert_idx, value in enumerate(hist):
            reporting[f"layer_{layer_number}/expert_{expert_idx}_usage"] = value

    if hist_totals:
        overall_hist = torch.stack([hist / denom for hist in hist_totals.values()]).mean(dim=0)
        torch.distributed.all_reduce(overall_hist, group=mpu.get_data_parallel_group())
        overall_hist = overall_hist / mpu.get_data_parallel_world_size()
        for expert_idx, value in enumerate(overall_hist):
            reporting[f"expert_{expert_idx}_usage"] = value

    return reporting


def _parse_hidden_space_layers(layer_spec, available_layers):
    if layer_spec is None or layer_spec == "" or layer_spec.lower() == "all":
        return {int(layer.layer_number) for layer in available_layers}
    selected = set()
    for value in layer_spec.split(","):
        value = value.strip()
        if not value:
            continue
        selected.add(int(value))
    return selected


def _collect_transformer_layers(modules, layer_spec):
    layers = []
    for module in unwrap_model(modules):
        decoder = getattr(module, "decoder", None)
        module_layers = getattr(decoder, "layers", None)
        if module_layers is None:
            continue
        layers.extend(list(module_layers))

    selected = _parse_hidden_space_layers(layer_spec, layers)
    return [(int(layer.layer_number), layer) for layer in layers if int(layer.layer_number) in selected]


def _distill_mode_includes_hidden(args):
    return args.moe_expansion_distill_mode in ("logits_hidden", "logits_hidden_router")


def _distill_mode_includes_router(args):
    return args.moe_expansion_distill_mode == "logits_hidden_router"


def _as_module_list(model):
    return model if isinstance(model, list) else [model]


def _layer_output_tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("hidden_states")
    if not torch.is_tensor(output):
        raise RuntimeError(f"Unsupported transformer layer output type for hidden dump: {type(output)}")
    return output


def _flatten_layer_hidden(hidden_states, labels):
    # Transformer layers use [seq, batch, hidden], while labels use [batch, seq].
    if hidden_states.dim() != 3:
        raise RuntimeError(f"Unexpected hidden state shape: {tuple(hidden_states.shape)}")
    if hidden_states.shape[0] == labels.shape[1] and hidden_states.shape[1] == labels.shape[0]:
        return hidden_states.permute(1, 0, 2).contiguous().view(-1, hidden_states.shape[-1])
    if hidden_states.shape[0] == labels.shape[0] and hidden_states.shape[1] == labels.shape[1]:
        return hidden_states.contiguous().view(-1, hidden_states.shape[-1])
    raise RuntimeError(
        f"Could not align hidden states {tuple(hidden_states.shape)} with labels {tuple(labels.shape)}"
    )


@contextmanager
def _capture_transformer_layer_outputs(modules, layer_spec, *, detach):
    layer_modules = _collect_transformer_layers(modules, layer_spec)
    captured = {}
    handles = []

    def make_hook(layer_number):
        def hook(_module, _inputs, output):
            tensor = _layer_output_tensor(output)
            captured[layer_number] = tensor.detach() if detach else tensor

        return hook

    for layer_number, layer in layer_modules:
        handles.append(layer.register_forward_hook(make_hook(layer_number)))

    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _collect_moe_router_layers(modules):
    router_layers = []
    seen = set()
    for module in unwrap_model(modules):
        decoder = getattr(module, "decoder", None)
        module_layers = getattr(decoder, "layers", None)
        if module_layers is None:
            continue
        for layer in module_layers:
            layer_number = getattr(layer, "layer_number", None)
            mlp = getattr(layer, "mlp", None)
            router = getattr(mlp, "router", None)
            if layer_number is None or router is None or not hasattr(router, "gating"):
                continue
            key = (id(mlp), int(layer_number))
            if key in seen:
                continue
            seen.add(key)
            router_layers.append((int(layer_number), mlp, router))
    return router_layers


@contextmanager
def _capture_moe_router_inputs(modules, *, detach):
    captured = {}
    routers = {}
    handles = []

    def make_hook(layer_number):
        def hook(_module, inputs):
            if not inputs:
                return
            hidden_states = inputs[0]
            if not torch.is_tensor(hidden_states):
                return
            captured[layer_number] = hidden_states.detach() if detach else hidden_states

        return hook

    for layer_number, mlp, router in _collect_moe_router_layers(modules):
        routers[layer_number] = router
        handles.append(mlp.register_forward_pre_hook(make_hook(layer_number)))

    try:
        yield captured, routers
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _capture_distill_router_inputs(modules, *, detach):
    """Capture standard-MoE or shared-router inputs for expansion distillation."""
    shared_routers = _collect_current_shared_routers(modules)
    if shared_routers:
        with capture_shared_router_inputs(detach=detach) as captured:
            yield captured, shared_routers
        return

    with _capture_moe_router_inputs(modules, detach=detach) as captured:
        yield captured


def _router_inputs_by_layer(captured):
    if isinstance(captured, dict):
        return captured
    return {int(layer_number): hidden_states for layer_number, hidden_states in captured}


def _masked_layer_hidden_mse(student_hidden, teacher_hidden, labels, loss_mask):
    common_layers = sorted(set(student_hidden) & set(teacher_hidden))
    if not common_layers:
        raise RuntimeError("Expansion hidden distillation requested, but no common layers were captured.")

    flat_mask = loss_mask.reshape(-1).float()
    layer_losses = []
    for layer_number in common_layers:
        student_flat = _flatten_layer_hidden(student_hidden[layer_number], labels)
        teacher_flat = _flatten_layer_hidden(teacher_hidden[layer_number], labels)
        if student_flat.shape != teacher_flat.shape:
            raise RuntimeError(
                "Hidden distillation shape mismatch at layer "
                f"{layer_number}: student={tuple(student_flat.shape)} "
                f"teacher={tuple(teacher_flat.shape)}"
            )
        per_token_mse = (student_flat.float() - teacher_flat.float()).pow(2).mean(dim=-1)
        layer_losses.append(torch.sum(per_token_mse * flat_mask))
    return torch.stack(layer_losses).mean()


def _masked_router_prob_kl(
    student_router_inputs,
    teacher_router_inputs,
    student_routers,
    teacher_routers,
    labels,
    loss_mask,
):
    student_router_inputs = _router_inputs_by_layer(student_router_inputs)
    teacher_router_inputs = _router_inputs_by_layer(teacher_router_inputs)
    common_layers = sorted(
        set(student_router_inputs)
        & set(teacher_router_inputs)
        & set(student_routers)
        & set(teacher_routers)
    )
    if not common_layers:
        raise RuntimeError("Expansion router distillation requested, but no common routers were captured.")

    flat_mask = loss_mask.reshape(-1).bool()
    layer_losses = []
    for layer_number in common_layers:
        student_flat = _flatten_layer_hidden(student_router_inputs[layer_number], labels)
        teacher_flat = _flatten_layer_hidden(teacher_router_inputs[layer_number], labels)
        if student_flat.shape[:-1] != teacher_flat.shape[:-1]:
            raise RuntimeError(
                "Router distillation token-shape mismatch at layer "
                f"{layer_number}: student={tuple(student_flat.shape)} "
                f"teacher={tuple(teacher_flat.shape)}"
            )
        student_flat = student_flat[flat_mask]
        teacher_flat = teacher_flat[flat_mask]
        if student_flat.numel() == 0:
            continue
        with torch.no_grad():
            teacher_logits = teacher_routers[layer_number].gating(teacher_flat)
        student_logits = student_routers[layer_number].gating(student_flat)
        layer_losses.append(
            teacher_student_router_kl(student_logits, teacher_logits) * student_flat.shape[0]
        )

    if not layer_losses:
        raise RuntimeError("Expansion router distillation found zero valid tokens.")
    return torch.stack(layer_losses).mean()


def _run_hidden_space_dump(model, iteration):
    args = get_args()
    dump_path = getattr(args, "hidden_space_dump_path", None)
    if not dump_path:
        return False
    if args.pipeline_model_parallel_size != 1:
        raise RuntimeError("--hidden-space-dump-path currently requires pipeline parallel size 1.")
    if args.probe_eval_iters <= 0 or not args.probe_data_path:
        raise RuntimeError("Hidden-space dump requires --probe-data-path and --probe-eval-iters > 0.")

    modules = model if isinstance(model, list) else [model]
    layer_modules = _collect_transformer_layers(modules, args.hidden_space_dump_layers)
    if not layer_modules:
        raise RuntimeError("No transformer layers found for hidden-space dump.")

    dataloader = _build_probe_dataloader(args.probe_data_path, args.probe_eval_iters, "hidden_space_dump")
    if dataloader is None:
        raise RuntimeError("Could not build hidden-space probe dataloader.")

    prior_states = [module.training for module in modules]
    for module in modules:
        module.eval()

    captured = {}
    handles = []
    layer_chunks = {layer_number: [] for layer_number, _ in layer_modules}
    selected_token_ids = []
    selected_positions = []
    selected_sample_indices = []
    total_tokens = 0
    max_tokens = max(1, int(args.hidden_space_dump_max_tokens))

    def make_hook(layer_number):
        def hook(_module, _inputs, output):
            captured[layer_number] = _layer_output_tensor(output).detach()
        return hook

    for layer_number, layer in layer_modules:
        handles.append(layer.register_forward_hook(make_hook(layer_number)))

    try:
        with torch.no_grad():
            probe_iterator = iter(dataloader)
            for _ in range(args.probe_eval_iters):
                if total_tokens >= max_tokens:
                    break
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(probe_iterator)
                captured.clear()
                _ = modules[0](
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=True,
                )

                flat_mask = loss_mask.reshape(-1).bool()
                candidate_indices = torch.nonzero(flat_mask, as_tuple=False).view(-1)
                if candidate_indices.numel() == 0:
                    continue
                remaining = max_tokens - total_tokens
                candidate_indices = candidate_indices[:remaining]

                flat_tokens = tokens.reshape(-1)
                batch_size, seq_length = labels.shape
                selected_token_ids.append(flat_tokens[candidate_indices].detach().cpu())
                selected_positions.append((candidate_indices % seq_length).detach().cpu())
                selected_sample_indices.append((candidate_indices // seq_length).detach().cpu())

                for layer_number, _layer in layer_modules:
                    if layer_number not in captured:
                        raise RuntimeError(f"Layer {layer_number} was not captured during hidden dump.")
                    flat_hidden = _flatten_layer_hidden(captured[layer_number], labels)
                    layer_chunks[layer_number].append(
                        flat_hidden[candidate_indices].float().detach().cpu()
                    )
                total_tokens += int(candidate_indices.numel())
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in zip(modules, prior_states):
            if was_training:
                module.train()

    if total_tokens <= 0:
        raise RuntimeError("Hidden-space dump captured zero valid tokens.")

    layer_numbers = [layer_number for layer_number, _ in layer_modules]
    hidden_layers = torch.stack(
        [torch.cat(layer_chunks[layer_number], dim=0) for layer_number in layer_numbers],
        dim=0,
    ).numpy()
    token_ids = torch.cat(selected_token_ids, dim=0).numpy()
    positions = torch.cat(selected_positions, dim=0).numpy()
    sample_indices = torch.cat(selected_sample_indices, dim=0).numpy()

    metadata = {
        "label": args.hidden_space_dump_label or "",
        "iteration": int(iteration),
        "probe_data_path": list(args.probe_data_path),
        "probe_eval_iters": int(args.probe_eval_iters),
        "max_tokens": int(max_tokens),
        "captured_tokens": int(total_tokens),
        "layer_numbers": layer_numbers,
        "hidden_shape": list(hidden_layers.shape),
        "load": getattr(args, "load", None),
    }

    if torch.distributed.get_rank() == 0:
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
        np.savez_compressed(
            dump_path,
            hidden_layers=hidden_layers,
            layer_numbers=np.asarray(layer_numbers, dtype=np.int64),
            token_ids=token_ids,
            positions=positions,
            sample_indices=sample_indices,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
        print_rank_0(
            f"hidden-space dump saved: {dump_path} | "
            f"layers={layer_numbers} | tokens={total_tokens}"
        )
    torch.distributed.barrier()
    return True


def _run_single_probe_evaluation(
    model,
    iteration,
    probe_data_path,
    probe_eval_iters,
    probe_name,
    probe_step_offset,
    cache_key,
):
    args = get_args()
    if not probe_data_path or probe_eval_iters <= 0:
        return
    if args.pipeline_model_parallel_size != 1:
        print_rank_0("Skipping probe evaluation because pipeline parallel size is not 1.")
        return

    dataloader = _build_probe_dataloader(probe_data_path, probe_eval_iters, cache_key)
    if dataloader is None:
        return

    probe_name = probe_name or "probe"
    logged_iteration = iteration + max(probe_step_offset, 0)
    probe_iterator = iter(dataloader)
    modules = model if isinstance(model, list) else [model]
    prior_states = [module.training for module in modules]
    for module in modules:
        module.eval()

    loss_total = torch.zeros(1, device="cuda", dtype=torch.float64)
    correct_total = torch.zeros(1, device="cuda", dtype=torch.float64)
    token_total = torch.zeros(1, device="cuda", dtype=torch.float64)
    router_usage_enabled = bool(getattr(args, "probe_router_usage", False))
    router_usage_totals = {}
    router_usage_hist_totals = {}
    router_usage_batches = 0
    router_usage_existing_experts = _probe_router_usage_num_existing_experts(args)
    routers = _collect_current_shared_routers(modules) if router_usage_enabled else {}
    if router_usage_enabled and not routers:
        print_rank_0("Probe router usage requested, but no shared-router modules were found.")

    with torch.no_grad():
        for _ in range(probe_eval_iters):
            training_micro_batch_size = args.micro_batch_size
            if args.probe_micro_batch_size is not None:
                args.micro_batch_size = args.probe_micro_batch_size
            try:
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(probe_iterator)
            finally:
                args.micro_batch_size = training_micro_batch_size
            if router_usage_enabled and routers:
                with capture_shared_router_inputs() as captured_router_inputs:
                    logits = modules[0](
                        tokens,
                        position_ids,
                        attention_mask,
                        labels=None,
                        runtime_gather_output=True,
                    )
                if _accumulate_probe_router_usage(
                    captured_router_inputs,
                    routers,
                    router_usage_existing_experts,
                    args,
                    router_usage_totals,
                    router_usage_hist_totals,
                ):
                    router_usage_batches += 1
            else:
                logits = modules[0](
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=True,
                )
            logits = _align_logits(logits.float(), labels)
            flat_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                reduction="none",
            ).view_as(labels)
            preds = logits.argmax(dim=-1)
            mask = loss_mask.float()
            loss_total += torch.sum(flat_loss * mask).double()
            correct_total += torch.sum((preds == labels).float() * mask).double()
            token_total += torch.sum(mask).double()

    for module, was_training in zip(modules, prior_states):
        if was_training:
            module.train()

    stats = torch.cat([loss_total, correct_total, token_total])
    torch.distributed.all_reduce(stats, group=mpu.get_data_parallel_group())
    loss_value = (stats[0] / stats[2].clamp_min(1.0)).item()
    accuracy = (stats[1] / stats[2].clamp_min(1.0)).item()
    ppl = math.exp(min(20, loss_value))
    router_usage = _finalize_probe_router_usage(
        router_usage_totals, router_usage_hist_totals, router_usage_batches
    )

    writer = get_tensorboard_writer()
    if writer:
        writer.add_scalar(f"{probe_name}/next_token_accuracy", accuracy, logged_iteration)
        writer.add_scalar(f"{probe_name}/ppl", ppl, logged_iteration)
        for key, value in router_usage.items():
            writer.add_scalar(f"{probe_name}/router/{key}", value.item(), logged_iteration)

    wandb_writer = get_wandb_writer()
    if wandb_writer and torch.distributed.get_rank() == (args.world_size - 1):
        metrics = {
            f"{probe_name}/next_token_accuracy": accuracy,
            f"{probe_name}/ppl": ppl,
        }
        metrics.update(
            {f"{probe_name}/router/{key}": value.item() for key, value in router_usage.items()}
        )
        wandb_writer.log(metrics, logged_iteration)

    print_rank_last(
        f"probe {probe_name} at iteration {logged_iteration} | local_iteration: {iteration} "
        f"| next_token_acc: {accuracy:.6f} | ppl: {ppl:.6E}"
    )
    if router_usage:
        print_rank_last(
            f"probe {probe_name} router usage at iteration {logged_iteration} | "
            f"old_expert_fraction: {router_usage['old_expert_fraction'].item():.6f} | "
            f"new_expert_fraction: {router_usage['new_expert_fraction'].item():.6f} | "
            f"new_expert_prob_mass: {router_usage['new_expert_prob_mass'].item():.6f}"
        )


def run_probe_evaluation(model, iteration):
    args = get_args()
    if _run_hidden_space_dump(model, iteration):
        return
    if args.probe_eval_interval and iteration % args.probe_eval_interval == 0:
        _run_single_probe_evaluation(
            model,
            iteration,
            args.probe_data_path,
            args.probe_eval_iters,
            args.probe_name,
            args.probe_step_offset,
            "primary_probe",
        )

    if (
        args.secondary_probe_eval_interval
        and iteration % args.secondary_probe_eval_interval == 0
    ):
        _run_single_probe_evaluation(
            model,
            iteration,
            args.secondary_probe_data_path,
            args.secondary_probe_eval_iters,
            args.secondary_probe_name,
            args.secondary_probe_step_offset,
            "secondary_probe",
        )

    if (
        getattr(args, "tertiary_probe_eval_interval", 0)
        and iteration % args.tertiary_probe_eval_interval == 0
    ):
        _run_single_probe_evaluation(
            model,
            iteration,
            args.tertiary_probe_data_path,
            args.tertiary_probe_eval_iters,
            args.tertiary_probe_name,
            args.tertiary_probe_step_offset,
            "tertiary_probe",
        )


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        probe_eval_func=run_probe_evaluation,
        router_memory_step_func=router_memory_step,
        router_memory_eval_func=router_memory_eval,
        router_memory_accum_func=router_memory_accum_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
