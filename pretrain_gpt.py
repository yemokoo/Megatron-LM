# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import math
import os
import torch
import torch.nn.functional as F
from functools import partial
from contextlib import nullcontext
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
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.training import get_old_moe_distill_teacher
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

    return batch.values()


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
    if isinstance(output_tensor, dict):
        losses = output_tensor["losses"].float()
        teacher_logits = output_tensor.get("teacher_logits")
        student_logits = output_tensor.get("student_logits")
    else:
        losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    lm_loss = torch.sum(losses.view(-1) * loss_mask)
    loss = torch.cat([lm_loss.view(1), total_tokens.view(1)])

    if teacher_logits is not None and student_logits is not None:
        temperature = args.moe_old_model_kl_temperature
        student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / temperature, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
        kl_loss = torch.sum(kl_loss.view(-1) * loss_mask) / total_tokens.clamp_min(1.0)
        kl_loss = kl_loss * (temperature ** 2)
        loss[0] = loss[0] + args.moe_old_model_kl_coeff * kl_loss

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
        reporting_kl = torch.tensor([kl_loss.detach()], device=reporting_loss.device)
        torch.distributed.all_reduce(reporting_kl, group=mpu.get_data_parallel_group())
        reporting['kd loss'] = (reporting_kl[0], reporting_loss[1])

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        reporting,
    )


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
    if teacher_model is not None and args.moe_old_model_kl_coeff > 0:
        with stimer:
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
    _PROBE_DATALOADER[cache_key] = build_pretraining_data_loader(valid_ds, 0)
    return _PROBE_DATALOADER[cache_key]


def _align_logits(logits, labels):
    if logits.dim() != 3:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    if logits.shape[0] == labels.shape[0] and logits.shape[1] == labels.shape[1]:
        return logits
    if logits.shape[0] == labels.shape[1] and logits.shape[1] == labels.shape[0]:
        return logits.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"Could not align logits {tuple(logits.shape)} and labels {tuple(labels.shape)}")


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

    with torch.no_grad():
        for _ in range(probe_eval_iters):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(probe_iterator)
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

    writer = get_tensorboard_writer()
    if writer:
        writer.add_scalar(f"{probe_name}/next_token_accuracy", accuracy, logged_iteration)
        writer.add_scalar(f"{probe_name}/ppl", ppl, logged_iteration)

    wandb_writer = get_wandb_writer()
    if wandb_writer and torch.distributed.get_rank() == (args.world_size - 1):
        wandb_writer.log(
            {
                f"{probe_name}/next_token_accuracy": accuracy,
                f"{probe_name}/ppl": ppl,
            },
            logged_iteration,
        )

    print_rank_last(
        f"probe {probe_name} at iteration {logged_iteration} | local_iteration: {iteration} "
        f"| next_token_acc: {accuracy:.6f} | ppl: {ppl:.6E}"
    )


def run_probe_evaluation(model, iteration):
    args = get_args()
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


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        probe_eval_func=run_probe_evaluation,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
