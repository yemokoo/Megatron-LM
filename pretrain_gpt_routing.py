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
_FINGERPRINT_KD_BUNDLE_CACHE = {}


def _configure_router_fingerprint_intervention(model, args):
    path = getattr(args, "router_fingerprint_intervention_path", None)
    mode = getattr(args, "router_fingerprint_intervention_mode", None)
    if not path and not mode:
        return
    if not path or not mode:
        raise RuntimeError(
            "Router fingerprint intervention requires both --router-fingerprint-intervention-path "
            "and --router-fingerprint-intervention-mode."
        )
    if args.pipeline_model_parallel_size != 1:
        raise RuntimeError("Router fingerprint intervention currently requires pipeline size 1.")

    import numpy as np

    with np.load(path, allow_pickle=False) as payload:
        required = {"layer_numbers", "means", "bases", "teacher_router_weights"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"Router fingerprint file {path} is missing arrays: {missing}")
        layer_numbers = payload["layer_numbers"].astype(np.int64).tolist()
        means = payload["means"]
        bases = payload["bases"]
        teacher_weights = payload["teacher_router_weights"]

    if not (len(layer_numbers) == len(means) == len(bases) == len(teacher_weights)):
        raise RuntimeError(f"Router fingerprint file {path} has inconsistent layer dimensions.")
    by_layer = {
        int(layer_number): (means[index], bases[index], teacher_weights[index])
        for index, layer_number in enumerate(layer_numbers)
    }
    configured = []
    decoder = getattr(model, "decoder", None)
    for layer in getattr(decoder, "layers", []):
        layer_number = int(getattr(layer, "layer_number", -1))
        if layer_number not in by_layer:
            continue
        router = getattr(getattr(layer, "mlp", None), "router", None)
        if router is None:
            raise RuntimeError(f"Layer {layer_number} has no standard MoE router for intervention.")
        mean, basis, teacher_weight = by_layer[layer_number]
        expected_hidden = int(router.weight.shape[1])
        expected_experts = int(router.weight.shape[0])
        if tuple(mean.shape) != (expected_hidden,):
            raise RuntimeError(
                f"Layer {layer_number} fingerprint mean shape {mean.shape} != {(expected_hidden,)}"
            )
        if basis.ndim != 2 or basis.shape[0] != expected_hidden:
            raise RuntimeError(
                f"Layer {layer_number} fingerprint basis shape {basis.shape} has wrong hidden size."
            )
        if tuple(teacher_weight.shape) != (expected_experts, expected_hidden):
            raise RuntimeError(
                f"Layer {layer_number} teacher router shape {teacher_weight.shape} != "
                f"{(expected_experts, expected_hidden)}"
            )
        router.register_buffer(
            "_fingerprint_mean", torch.as_tensor(mean, dtype=torch.float32), persistent=False
        )
        router.register_buffer(
            "_fingerprint_basis", torch.as_tensor(basis, dtype=torch.float32), persistent=False
        )
        router.register_buffer(
            "_fingerprint_teacher_weight",
            torch.as_tensor(teacher_weight, dtype=torch.float32),
            persistent=False,
        )
        router._fingerprint_intervention_mode = mode
        configured.append(layer_number)

    missing_layers = sorted(set(by_layer) - set(configured))
    if missing_layers:
        raise RuntimeError(f"Could not attach router fingerprint to layers: {missing_layers}")
    print_rank_0(
        f"Configured router fingerprint intervention mode={mode} layers={configured} from {path}"
    )

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

    _configure_router_fingerprint_intervention(model, args)
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # Apply the old-like objective split before context-parallel slicing so the
    # ordinary CP helper can slice the already-masked loss tensor without
    # needing to understand the sidecar GT schema.
    _apply_old_like_gt_objective_mask(batch, get_args())
    batch.pop("old_like_mask", None)
    batch.pop("old_like_sample_id", None)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    if getattr(get_args(), "moe_lpr_loss_coeff", 0.0) > 0.0:
        get_args()._moe_lpr_dataset_ids = batch["dataset_id"]
    return (
        batch["tokens"], batch["labels"], batch["loss_mask"],
        batch["attention_mask"], batch["position_ids"],
    )


def _apply_old_like_gt_objective_mask(batch, args):
    """Split one Code batch into primary and old-like replay objectives.

    ``old_like_mask`` is contextual-occurrence GT aligned to the input token at
    the same sequence position as ``loss_mask``.  The primary pass trains on its
    complement; the replay pass trains only on GT-positive positions.  The
    replay pass is selected by the short-lived marker set in ``train_step``.

    Probe/evaluation batches intentionally carry no GT fields and are left
    unchanged.  Training batches are validated independently by the paired
    iterator in ``training.py`` before this helper is reached.
    """
    args._moe_joint_replay_old_like_batch_stats = None
    if not getattr(args, "moe_joint_replay_old_like_gt_path", None):
        return
    has_mask = "old_like_mask" in batch
    has_sample_id = "old_like_sample_id" in batch
    if not has_mask and not has_sample_id:
        return
    if not (has_mask and has_sample_id):
        raise RuntimeError(
            "Old-like GT batch must contain both old_like_mask and old_like_sample_id."
        )

    base_loss_mask = batch["loss_mask"]
    gt_mask = batch["old_like_mask"]
    if gt_mask.shape != base_loss_mask.shape:
        raise RuntimeError(
            "Old-like GT/loss-mask shape mismatch: "
            f"gt={tuple(gt_mask.shape)} loss_mask={tuple(base_loss_mask.shape)}"
        )
    if gt_mask.dtype != torch.bool:
        raise RuntimeError(
            f"old_like_mask must be torch.bool, got {gt_mask.dtype}."
        )

    valid = base_loss_mask.ne(0)
    replay_branch = bool(getattr(args, "_moe_joint_replay_active", False))
    objective_mask = gt_mask if replay_branch else ~gt_mask
    selected = valid & objective_mask
    batch["loss_mask"] = base_loss_mask * objective_mask.to(base_loss_mask.dtype)
    args._moe_joint_replay_old_like_batch_stats = {
        "branch": "replay" if replay_branch else "primary",
        "base_valid_tokens": valid.sum().detach(),
        "gt_positive_tokens": (valid & gt_mask).sum().detach(),
        "selected_tokens": selected.sum().detach(),
        "zero_selected": selected.sum().eq(0).detach(),
    }


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def _effective_lm_loss_coeff(args):
    if getattr(args, "_moe_joint_replay_old_data_kd_active", False):
        return 0.0
    if _fingerprint_kd_enabled(args):
        return float(getattr(args, "fingerprint_kd_lm_loss_coeff", 1.0))
    return getattr(args, "moe_expansion_distill_lm_loss_coeff", 1.0)


def _token_mean_to_loss_numerator(mean_loss, total_tokens):
    """Convert an auxiliary per-token mean to Megatron's loss numerator."""
    return mean_loss * total_tokens


def _effective_old_hidden_kl_coeff(args):
    """Return the restart-safe, iteration-based old-data hidden-KL scale."""
    end = float(args.moe_old_hidden_kl_coeff)
    start = getattr(args, "moe_old_hidden_kl_coeff_start", None)
    decay_steps = int(getattr(args, "moe_old_hidden_kl_coeff_decay_steps", 0))
    if start is None or decay_steps <= 0:
        return end
    iteration = max(0, int(getattr(args, "curr_iteration", args.iteration)))
    progress = min(float(iteration) / float(decay_steps), 1.0)
    return float(start) + (end - float(start)) * progress


def _teacher_kd_enabled_for_current_branch(args, teacher_model, expansion_distill_enabled):
    if teacher_model is None:
        return False
    if expansion_distill_enabled:
        return True
    replay_distillation_requested = (
        getattr(args, "moe_joint_replay_old_data_kd", False)
        or getattr(args, "moe_joint_replay_old_data_hidden_kl", False)
        or getattr(args, "moe_joint_replay_old_data_hidden_mse", False)
    )
    if replay_distillation_requested:
        return bool(getattr(args, "_moe_joint_replay_old_data_kd_active", False))
    return args.moe_old_model_kl_coeff > 0


def _old_data_hidden_kl_enabled_for_current_branch(args):
    return bool(
        getattr(args, "moe_joint_replay_old_data_hidden_kl", False)
        and getattr(args, "_moe_joint_replay_old_data_kd_active", False)
    )


def _old_data_hidden_mse_enabled_for_current_branch(args):
    return bool(
        getattr(args, "moe_joint_replay_old_data_hidden_mse", False)
        and getattr(args, "_moe_joint_replay_old_data_kd_active", False)
    )


def loss_func(
    loss_mask: torch.Tensor,
    output_tensor: torch.Tensor,
    old_like_stats=None,
):
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
    old_hidden_mse_loss = None
    hidden_kl_loss = None
    router_kl_loss = None
    lpr_loss = None
    fingerprint_kd_loss = None
    fingerprint_mean_score = None
    fingerprint_mean_weight = None
    fingerprint_hard_coverage = None
    fingerprint_score_weight_covariance = None
    fingerprint_layer_loss_means = None
    if isinstance(output_tensor, dict):
        losses = output_tensor["losses"].float()
        teacher_logits = output_tensor.get("teacher_logits")
        student_logits = output_tensor.get("student_logits")
        hidden_mse_loss = output_tensor.get("hidden_mse_loss")
        old_hidden_mse_loss = output_tensor.get("old_hidden_mse_loss")
        hidden_kl_loss = output_tensor.get("hidden_kl_loss")
        router_kl_loss = output_tensor.get("router_kl_loss")
        lpr_loss = output_tensor.get("lpr_loss")
        fingerprint_kd_loss = output_tensor.get("fingerprint_kd_loss")
        fingerprint_mean_score = output_tensor.get("fingerprint_mean_score")
        fingerprint_mean_weight = output_tensor.get("fingerprint_mean_weight")
        fingerprint_hard_coverage = output_tensor.get("fingerprint_hard_coverage")
        fingerprint_score_weight_covariance = output_tensor.get(
            "fingerprint_score_weight_covariance"
        )
        fingerprint_layer_loss_means = output_tensor.get("fingerprint_layer_loss_means")
    else:
        losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    lm_loss = torch.sum(losses.view(-1) * loss_mask)
    lm_loss_coeff = _effective_lm_loss_coeff(args)
    lm_loss = lm_loss * lm_loss_coeff
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
            # Megatron divides this numerator by the returned token count.  The
            # replay KL above is already a per-token mean, so convert it back to
            # a token sum here to avoid normalizing it a second time.
            loss[0] = loss[0] + args.moe_old_model_kl_coeff * (
                _token_mean_to_loss_numerator(kl_loss, total_tokens)
            )

    if hidden_mse_loss is not None:
        loss[0] = loss[0] + args.moe_expansion_distill_hidden_mse_coeff * hidden_mse_loss

    if old_hidden_mse_loss is not None:
        # _masked_layer_hidden_mse already returns a token-sum numerator,
        # averaged only across the selected layers. Megatron divides loss[0]
        # by the returned token count, so multiplying by total_tokens here
        # would incorrectly apply token normalization twice in the other
        # direction.
        loss[0] = loss[0] + args.moe_old_hidden_mse_coeff * old_hidden_mse_loss

    if hidden_kl_loss is not None:
        # _masked_layer_hidden_kl returns a per-token mean.  loss[0], however,
        # is the numerator of Megatron's token-normalized loss.  Without this
        # conversion the hidden-KL gradient is smaller by the number of valid
        # tokens in the microbatch (18,432 in the R2 production run).
        hidden_kl_coeff = _effective_old_hidden_kl_coeff(args)
        loss[0] = (
            loss[0]
            + hidden_kl_coeff
            * _token_mean_to_loss_numerator(hidden_kl_loss, total_tokens)
        )

    if router_kl_loss is not None:
        loss[0] = loss[0] + args.moe_expansion_distill_router_kl_coeff * router_kl_loss

    if lpr_loss is not None:
        loss[0] = loss[0] + args.moe_lpr_loss_coeff * lpr_loss

    if fingerprint_kd_loss is not None:
        loss[0] = loss[0] + args.fingerprint_kd_coeff * _token_mean_to_loss_numerator(
            fingerprint_kd_loss, total_tokens
        )

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
    if old_like_stats is not None:
        old_like_counts = torch.stack(
            (
                old_like_stats["base_valid_tokens"],
                old_like_stats["gt_positive_tokens"],
                old_like_stats["selected_tokens"],
                old_like_stats["zero_selected"].to(
                    old_like_stats["selected_tokens"].dtype
                ),
            )
        ).to(device=loss.device, dtype=torch.float32)
        # These counts were captured before CP slicing and are identical on CP
        # peers, so reduce only over data-parallel ranks.
        torch.distributed.all_reduce(
            old_like_counts, group=mpu.get_data_parallel_group()
        )
        branch = old_like_stats["branch"]
        dp_size = old_like_counts.new_tensor(
            float(mpu.get_data_parallel_world_size())
        )
        reporting['old-like GT coverage'] = (
            old_like_counts[1], old_like_counts[0]
        )
        reporting[f'old-like {branch} objective coverage'] = (
            old_like_counts[2], old_like_counts[0]
        )
        reporting[f'old-like {branch} selected tokens per microbatch'] = (
            old_like_counts[2] / dp_size
        )
        reporting[f'old-like {branch} zero-selected DP-rank fraction'] = (
            old_like_counts[3] / dp_size
        )
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
    if old_hidden_mse_loss is not None:
        old_hidden_mse_sum = old_hidden_mse_loss.detach().view(1)
        torch.distributed.all_reduce(
            old_hidden_mse_sum, group=mpu.get_data_parallel_group()
        )
        reporting['old hidden mse loss'] = (
            old_hidden_mse_sum[0], reporting_loss[1]
        )
        reporting['old hidden mse coeff'] = torch.tensor(
            args.moe_old_hidden_mse_coeff,
            dtype=torch.float,
            device=old_hidden_mse_loss.device,
        )
    if hidden_kl_loss is not None:
        hidden_kl_value = hidden_kl_loss.detach().view(1)
        torch.distributed.all_reduce(hidden_kl_value, group=mpu.get_data_parallel_group())
        reporting['hidden kl loss'] = hidden_kl_value[0] / mpu.get_data_parallel_world_size()
        reporting['hidden kl coeff'] = torch.tensor(
            _effective_old_hidden_kl_coeff(args),
            dtype=torch.float,
            device=hidden_kl_loss.device,
        )
    if router_kl_loss is not None:
        router_kl_sum = router_kl_loss.detach().view(1)
        torch.distributed.all_reduce(router_kl_sum, group=mpu.get_data_parallel_group())
        reporting['router prob kl loss'] = (router_kl_sum[0], reporting_loss[1])
    if lpr_loss is not None:
        lpr_sum = lpr_loss.detach().view(1)
        torch.distributed.all_reduce(lpr_sum, group=mpu.get_data_parallel_group())
        reporting['lpr loss'] = (lpr_sum[0], reporting_loss[1])
    if fingerprint_kd_loss is not None:
        fingerprint_value = fingerprint_kd_loss.detach().view(1)
        torch.distributed.all_reduce(fingerprint_value, group=mpu.get_data_parallel_group())
        reporting['fingerprint kd loss'] = (
            fingerprint_value[0] / mpu.get_data_parallel_world_size()
        )
        reporting['fingerprint kd coeff'] = torch.tensor(
            args.fingerprint_kd_coeff,
            dtype=torch.float,
            device=fingerprint_kd_loss.device,
        )
        fingerprint_stats = torch.cat(
            (
                fingerprint_mean_score.detach().view(1),
                fingerprint_mean_weight.detach().view(1),
                fingerprint_hard_coverage.detach().view(1),
                fingerprint_score_weight_covariance.detach().view(1),
                fingerprint_layer_loss_means.detach().view(-1),
            )
        )
        torch.distributed.all_reduce(fingerprint_stats, group=mpu.get_data_parallel_group())
        fingerprint_stats = fingerprint_stats / mpu.get_data_parallel_world_size()
        reporting['fingerprint mean score'] = fingerprint_stats[0]
        reporting['fingerprint mean weight'] = fingerprint_stats[1]
        reporting['fingerprint hard coverage'] = fingerprint_stats[2]
        reporting['fingerprint score weight covariance'] = fingerprint_stats[3]
        for layer_number, value in zip(
            [int(item) for item in args.fingerprint_kd_layers.split(',')],
            fingerprint_stats[4:],
        ):
            reporting[f'fingerprint layer {layer_number} projected mse'] = value

    replay_gradient_scale = float(
        getattr(args, "_moe_joint_replay_gradient_scale", 1.0) or 1.0
    )
    if not 0.0 < replay_gradient_scale <= 1.0:
        raise RuntimeError(
            f"invalid old-like replay gradient scale: {replay_gradient_scale}"
        )
    gradient_loss = loss[0] * replay_gradient_scale

    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)
    if old_like_stats is not None:
        # At ~0.125% coverage a small microbatch can contain no GT-positive
        # positions.  Keep its objective exactly zero while returning a safe
        # normalization denominator to the pipeline schedule, which otherwise
        # divides 0 by 0 before backward. For nonempty microbatches the exact
        # selected-token count is returned unchanged.
        local_num_tokens = local_num_tokens.clamp_min(1)
    return (
        gradient_loss.clone(),
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
        old_like_stats = getattr(
            args, "_moe_joint_replay_old_like_batch_stats", None
        )
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
    teacher_kd_enabled = _teacher_kd_enabled_for_current_branch(
        args, teacher_model, expansion_distill_enabled
    )
    fingerprint_kd_enabled = _fingerprint_kd_enabled(args)
    if fingerprint_kd_enabled and teacher_model is None:
        raise RuntimeError(
            "Fingerprint KD requires a frozen teacher loaded with --moe-old-model-kl-load."
        )
    if fingerprint_kd_enabled and (teacher_kd_enabled or expansion_distill_enabled):
        raise RuntimeError("Fingerprint KD cannot be combined with existing expansion/replay KD objectives.")
    if fingerprint_kd_enabled:
        if args.pipeline_model_parallel_size != 1:
            raise RuntimeError("Fingerprint KD currently requires pipeline parallel size 1.")
        student_modules = _as_module_list(model)
        teacher_modules = _as_module_list(teacher_model[0])
        layer_spec = args.fingerprint_kd_layers
        with stimer:
            with _capture_transformer_layer_outputs(
                student_modules, layer_spec, detach=False
            ) as student_hidden:
                student_losses = model(tokens, position_ids, attention_mask, labels=labels)
        with torch.no_grad():
            with _capture_transformer_layer_outputs(
                teacher_modules, layer_spec, detach=True
            ) as teacher_hidden:
                _ = teacher_model[0](
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=False,
                )
        bundle = _load_fingerprint_kd_bundle(args, tokens.device)
        fingerprint = _masked_fingerprint_projected_mse(
            student_hidden,
            teacher_hidden,
            labels,
            loss_mask,
            bundle,
            gate_mode=args.fingerprint_kd_gate_mode,
            threshold=args.fingerprint_kd_threshold,
            soft_temperature=args.fingerprint_kd_soft_temperature,
            weight_assignment=args.fingerprint_kd_weight_assignment,
        )
        output_tensor = {
            "losses": student_losses,
            "fingerprint_kd_loss": fingerprint["loss"],
            "fingerprint_mean_score": fingerprint["mean_score"],
            "fingerprint_mean_weight": fingerprint["mean_weight"],
            "fingerprint_hard_coverage": fingerprint["hard_coverage"],
            "fingerprint_score_weight_covariance": fingerprint[
                "score_weight_covariance"
            ],
            "fingerprint_layer_loss_means": fingerprint["layer_loss_means"],
        }
    elif teacher_kd_enabled:
        old_data_hidden_kl_enabled = _old_data_hidden_kl_enabled_for_current_branch(args)
        old_data_hidden_mse_enabled = _old_data_hidden_mse_enabled_for_current_branch(args)
        if old_data_hidden_kl_enabled and old_data_hidden_mse_enabled:
            raise RuntimeError(
                "Select exactly one old-data hidden replay objective: hidden KL or hidden MSE."
            )
        final_logits_kd_enabled = expansion_distill_enabled or not (
            old_data_hidden_kl_enabled or old_data_hidden_mse_enabled
        )
        if final_logits_kd_enabled and args.moe_old_model_kl_coeff <= 0:
            raise RuntimeError(
                "--moe-expansion-distill-mode requires --moe-old-model-kl-coeff > 0 "
                "because all expansion-distill modes include final-logit KL."
            )
        if args.pipeline_model_parallel_size != 1 and (
            _distill_mode_includes_hidden(args)
            or _distill_mode_includes_router(args)
            or old_data_hidden_kl_enabled
            or old_data_hidden_mse_enabled
        ):
            raise RuntimeError(
                "Hidden/router expansion distillation currently requires "
                "--pipeline-model-parallel-size 1."
            )

        student_modules = _as_module_list(model)
        teacher_modules = _as_module_list(teacher_model[0])
        capture_hidden = (
            _distill_mode_includes_hidden(args)
            or old_data_hidden_kl_enabled
            or old_data_hidden_mse_enabled
        )
        if old_data_hidden_kl_enabled:
            hidden_layer_spec = args.moe_old_hidden_kl_layers
        elif old_data_hidden_mse_enabled:
            hidden_layer_spec = args.moe_old_hidden_mse_layers
        else:
            hidden_layer_spec = args.moe_expansion_distill_hidden_layers
        student_hidden_ctx = (
            _capture_transformer_layer_outputs(
                student_modules, hidden_layer_spec, detach=False
            )
            if capture_hidden
            else nullcontext({})
        )
        teacher_hidden_ctx = (
            _capture_transformer_layer_outputs(
                teacher_modules, hidden_layer_spec, detach=True
            )
            if capture_hidden
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
                if final_logits_kd_enabled:
                    student_output = model(
                        tokens, position_ids, attention_mask, labels=labels,
                        runtime_gather_output=True, return_loss_and_logits=True,
                    )
                    output_tensor = student_output["losses"]
                    student_logits = student_output["logits"]
                else:
                    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
                    student_logits = None
        with torch.no_grad():
            with teacher_hidden_ctx as teacher_hidden, teacher_router_ctx as (
                teacher_router_inputs,
                teacher_routers,
            ):
                teacher_output = teacher_model[0](
                    tokens,
                    position_ids,
                    attention_mask,
                    labels=None,
                    runtime_gather_output=final_logits_kd_enabled,
                )
                teacher_logits = teacher_output if final_logits_kd_enabled else None
        output_tensor = {"losses": output_tensor}
        if final_logits_kd_enabled:
            output_tensor.update(student_logits=student_logits, teacher_logits=teacher_logits)
        if _distill_mode_includes_hidden(args):
            output_tensor["hidden_mse_loss"] = _masked_layer_hidden_mse(
                student_hidden,
                teacher_hidden,
                labels,
                loss_mask,
            )
        if old_data_hidden_kl_enabled:
            output_tensor["hidden_kl_loss"] = _masked_layer_hidden_kl(
                student_hidden, teacher_hidden, labels, loss_mask,
                args.moe_old_hidden_kl_temperature,
            )
        if old_data_hidden_mse_enabled:
            output_tensor["old_hidden_mse_loss"] = _masked_layer_hidden_mse(
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

    return output_tensor, partial(
        loss_func, loss_mask, old_like_stats=old_like_stats
    )


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
    if (
        getattr(args, "moe_joint_replay_old_like_gt_path", None)
        and not getattr(args, "moe_joint_replay_lm", False)
    ):
        raise ValueError(
            "--moe-joint-replay-old-like-gt-path requires --moe-joint-replay-lm."
        )

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    # Contextual GT masks are indexed by the exact one-epoch GPTDataset axis
    # used by paired extraction.  Only the replay loader uses that axis.  The
    # primary loader remains the ordinary full task blend and is never replaced
    # by (or masked down to) the GT source dataset.
    requested_num_samples = list(train_val_test_num_samples)
    build_num_samples = requested_num_samples
    old_like_gt_path = getattr(args, "moe_joint_replay_old_like_gt_path", None)
    replay_gt_build = bool(
        old_like_gt_path
        and getattr(args, "_moe_joint_replay_dataset_build_active", False)
    )
    if replay_gt_build:
        from megatron.core.datasets.old_like_gt_dataset import (
            old_like_gt_source_blend,
            old_like_gt_source_sample_count,
        )

        build_num_samples = list(requested_num_samples)
        build_num_samples[0] = old_like_gt_source_sample_count(
            old_like_gt_path, expected_sequence_length=args.seq_length
        )
        # sample_id must address the extraction-time path-only exhaustive
        # blend.  Runtime task training commonly uses explicit equal weights,
        # which gives a different outer sample axis even for identical shards.
        config.blend = old_like_gt_source_blend(old_like_gt_path)
        config.blend_per_split = None

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        build_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    # The GT sidecar is indexed by the deterministic outer training-sample
    # order. Validation/test/probe datasets remain untouched.
    from megatron.core.datasets.old_like_gt_dataset import (
        maybe_wrap_train_dataset_with_old_like_gt,
    )
    train_ds = maybe_wrap_train_dataset_with_old_like_gt(
        train_ds,
        old_like_gt_path if replay_gt_build else None,
        expected_sequence_length=args.seq_length,
        expected_seed=args.seed,
        expected_data_paths=(args.data_path or args.train_data_path),
        replay_subset=replay_gt_build,
        replay_unit=getattr(
            args, "moe_joint_replay_old_like_unit", "positive_sequence"
        ),
        virtual_length=requested_num_samples[0],
    )

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


def _build_code_token_hidden_pair_dataloader(data_path, total_samples, consumed_samples):
    """Rebuild the exact Code train GPTDataset and resume at a global sample index.

    This intentionally uses the training split (``100,0,0``), unlike ordinary
    probe loaders, and every independent worker builds the same total sample
    count before seeking to its own contiguous range.
    """
    args = get_args()
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(data_path),
        blend_per_split=None,
        split="100,0,0",
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
    train_ds, _, _ = BlendedMegatronDatasetBuilder(
        dataset_type,
        (int(total_samples), 0, 0),
        is_dataset_built_on_rank,
        config,
    ).build()
    if train_ds is None:
        raise RuntimeError("Could not build the Code train dataset for paired hidden extraction.")
    return build_pretraining_data_loader(train_ds, int(consumed_samples))


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

    captured_items = captured.items() if isinstance(captured, dict) else captured
    for layer_number, hidden_states in captured_items:
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
    if layer_spec.lower() == "all_but_last":
        numbers = sorted(int(layer.layer_number) for layer in available_layers)
        return set(numbers[:-1])
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
            if router is None or not hasattr(router, "gating"):
                # shared-router hybrid layers route once, before attention, from
                # layer.shared_expert_router rather than layer.mlp.router.
                router = getattr(layer, "shared_expert_router", None)
            if layer_number is None or router is None or not hasattr(router, "gating"):
                continue
            key = (id(router), int(layer_number))
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
        raise RuntimeError("Hidden MSE requested, but no common layers were captured.")

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


def _masked_layer_hidden_kl(student_hidden, teacher_hidden, labels, loss_mask, temperature):
    common_layers = sorted(set(student_hidden) & set(teacher_hidden))
    if not common_layers:
        raise RuntimeError("Old-data hidden KL requested, but no common layers were captured.")
    if temperature <= 0:
        raise ValueError("Old-data hidden KL temperature must be positive.")

    flat_mask = loss_mask.reshape(-1).float()
    token_count = flat_mask.sum().clamp_min(1.0)
    layer_losses = []
    for layer_number in common_layers:
        student_flat = _flatten_layer_hidden(student_hidden[layer_number], labels).float()
        teacher_flat = _flatten_layer_hidden(teacher_hidden[layer_number], labels).float()
        if student_flat.shape != teacher_flat.shape:
            raise RuntimeError(
                "Hidden KL shape mismatch at layer "
                f"{layer_number}: student={tuple(student_flat.shape)} "
                f"teacher={tuple(teacher_flat.shape)}"
            )
        student_log_probs = F.log_softmax(student_flat / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_flat / temperature, dim=-1)
        per_token_kl = F.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1)
        layer_losses.append((per_token_kl * flat_mask).sum() / token_count)
    return torch.stack(layer_losses).mean() * (temperature ** 2)


def _fingerprint_kd_enabled(args):
    return bool(
        getattr(args, "fingerprint_kd_coeff", 0.0) > 0.0
        or getattr(args, "fingerprint_kd_force_enable_zero_coeff", False)
    )


def _load_fingerprint_kd_bundle(args, device):
    """Load immutable means/bases once per process/device."""
    global _FINGERPRINT_KD_BUNDLE_CACHE
    path = getattr(args, "fingerprint_kd_bundle", None)
    if not path:
        raise RuntimeError("Fingerprint KD requires --fingerprint-kd-bundle.")
    key = (
        os.path.abspath(path), str(device), int(args.fingerprint_kd_rank),
        args.fingerprint_kd_score_representation,
        args.fingerprint_kd_loss_representation,
    )
    if key in _FINGERPRINT_KD_BUNDLE_CACHE:
        return _FINGERPRINT_KD_BUNDLE_CACHE[key]
    import numpy as np
    with np.load(path, allow_pickle=False) as payload:
        names = payload["representation_names"].tolist()
        layers = payload["layer_numbers"].astype(np.int64).tolist()
        means_np = payload["means"]
        bases_np = payload["bases"]
    expected_layers = [int(value) for value in args.fingerprint_kd_layers.split(",")]
    if layers != expected_layers:
        raise RuntimeError(f"Fingerprint KD bundle/layer mismatch: {layers} != {expected_layers}")
    rank = int(args.fingerprint_kd_rank)
    score_index = names.index(args.fingerprint_kd_score_representation)
    loss_representation = args.fingerprint_kd_loss_representation
    loss_index = None if loss_representation == "full" else names.index(loss_representation)
    bundle = {
        "layers": layers,
        "means": torch.as_tensor(means_np, dtype=torch.float32, device=device),
        "score_bases": torch.as_tensor(
            bases_np[score_index, :, :, :rank], dtype=torch.float32, device=device
        ),
        # Full-hidden KD is the implicit identity projection.  Do not store or
        # materialize an H x H identity basis: the stable basis is still loaded
        # for teacher-derived token scoring, while the loss uses h_student-h_teacher.
        "loss_bases": None if loss_index is None else torch.as_tensor(
            bases_np[loss_index, :, :, :rank], dtype=torch.float32, device=device
        ),
    }
    for tensor in (bundle["means"], bundle["score_bases"], bundle["loss_bases"]):
        if tensor is not None:
            tensor.requires_grad_(False)
    _FINGERPRINT_KD_BUNDLE_CACHE[key] = bundle
    return bundle


def _masked_fingerprint_projected_mse(
    student_hidden,
    teacher_hidden,
    labels,
    loss_mask,
    bundle,
    *,
    gate_mode,
    threshold,
    soft_temperature,
    weight_assignment="stable",
):
    """Teacher-gated, additive MSE restricted to a stored hidden subspace."""
    layers = bundle["layers"]
    if sorted(student_hidden) != layers or sorted(teacher_hidden) != layers:
        raise RuntimeError(
            "Fingerprint KD hidden layers do not match bundle: "
            f"student={sorted(student_hidden)}, teacher={sorted(teacher_hidden)}, bundle={layers}"
        )
    if gate_mode == "soft" and soft_temperature <= 0:
        raise ValueError("Soft fingerprint KD requires a positive temperature.")
    flat_mask = loss_mask.reshape(-1).float()
    token_count = flat_mask.sum().clamp_min(1.0)
    layer_scores = []
    layer_projected_mse = []
    for layer_index, layer_number in enumerate(layers):
        student = _flatten_layer_hidden(student_hidden[layer_number], labels).float()
        teacher = _flatten_layer_hidden(teacher_hidden[layer_number], labels).float().detach()
        if student.shape != teacher.shape:
            raise RuntimeError(
                f"Fingerprint KD shape mismatch at layer {layer_number}: "
                f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}"
            )
        centered_teacher = teacher - bundle["means"][layer_index]
        score_projection = centered_teacher @ bundle["score_bases"][layer_index]
        layer_scores.append(
            score_projection.square().sum(dim=-1)
            / centered_teacher.square().sum(dim=-1).clamp_min(1e-20)
        )
        delta = student - teacher
        if bundle["loss_bases"] is None:
            delta_projection = delta
        else:
            delta_projection = delta @ bundle["loss_bases"][layer_index]
        layer_projected_mse.append(delta_projection.square().sum(dim=-1))
    score = torch.stack(layer_scores).mean(dim=0).detach()
    if gate_mode == "hard":
        weight = (score >= threshold).float()
    elif gate_mode == "soft":
        weight = torch.sigmoid((score - threshold) / soft_temperature)
    elif gate_mode == "all":
        weight = torch.ones_like(score)
    else:
        raise ValueError(f"Unsupported fingerprint gate mode: {gate_mode}")
    if weight_assignment == "permuted":
        valid = flat_mask.bool()
        valid_weight = weight[valid]
        if valid_weight.numel() > 1:
            # Exact matched control: preserve the valid-token weight multiset
            # and therefore mean/coverage/scale, while breaking its assignment
            # to the token that produced the stable score. A half rotation also
            # avoids retaining the strong local correlation of a one-token roll.
            permuted = torch.roll(valid_weight, shifts=valid_weight.numel() // 2)
            weight = weight.clone()
            weight[valid] = permuted
    elif weight_assignment != "stable":
        raise ValueError(
            f"Unsupported fingerprint weight assignment: {weight_assignment}"
        )
    hard_selected = (score >= threshold).float()
    mean_score = (score * flat_mask).sum().detach() / token_count
    mean_weight = (weight * flat_mask).sum().detach() / token_count
    score_weight_covariance = (
        (score * weight * flat_mask).sum().detach() / token_count
        - mean_score * mean_weight
    )
    per_layer_numerators = torch.stack(
        [(mse * weight * flat_mask).sum() for mse in layer_projected_mse]
    )
    mean_loss = per_layer_numerators.mean() / token_count
    return {
        "loss": mean_loss,
        "mean_score": mean_score,
        "mean_weight": mean_weight,
        "hard_coverage": (hard_selected * flat_mask).sum().detach() / token_count,
        "score_weight_covariance": score_weight_covariance,
        "layer_loss_means": (per_layer_numerators.detach() / token_count),
    }


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


@contextmanager
def _capture_moe_ffn_outputs(modules, layer_spec, *, detach):
    """Capture the MoE branch output before it is added to the residual stream."""
    selected = {number for number, _ in _collect_transformer_layers(modules, layer_spec)}
    captured = {}
    handles = []

    def make_hook(layer_number):
        def hook(_module, _inputs, output):
            tensor = _layer_output_tensor(output)
            captured[layer_number] = tensor.detach() if detach else tensor
        return hook

    for layer_number, mlp, _router in _collect_moe_router_layers(modules):
        if layer_number in selected:
            handles.append(mlp.register_forward_hook(make_hook(layer_number)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def _checkpoint_tracker_step(path):
    if not path:
        return None
    tracker = os.path.join(path, "latest_checkpointed_iteration.txt")
    if not os.path.isfile(tracker):
        return None
    with open(tracker, encoding="utf-8") as handle:
        return int(handle.read().strip())


def _run_cka_gt_full_census(model, iteration):
    """Stream threshold-free B/T/M distributions over document-bounded Code train windows.

    This deliberately keeps checkpoint loading and residual-output hooks in
    Megatron while delegating metric accumulation and transactional resume to
    ``cka_gt_full_census``.  A batch's hidden tensors live only until its
    scalar histograms, compact chunk-CKA rows, and a bounded deterministic
    token-score reservoir are committed.
    """
    args = get_args()
    output_root = getattr(args, "cka_gt_full_census_path", None)
    if not output_root:
        return False
    if args.pipeline_model_parallel_size != 1 or torch.distributed.get_world_size() != 1:
        raise RuntimeError(
            "Full CKA census requires one independent world-size-1 process per GPU and PP=1."
        )

    import time
    from pathlib import Path

    import numpy as np

    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from scripts.analysis.cka_gt_full_census import (
        BTMBatch,
        FullCensusWorker,
        compute_btm_batch,
        load_full_train_manifest,
    )
    from scripts.analysis.cka_gt_pilot_runtime import DocumentWindowBatchReader
    from scripts.analysis.cka_gt_pilot_windows import (
        checkpoint_identity,
        file_sha256,
        source_dataset_identity,
    )

    config_path = Path(args.cka_gt_full_census_config).resolve()
    manifest_path = Path(args.cka_gt_full_census_manifest).resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Full CKA census analysis config is missing: {config_path}")
    if not manifest_path.is_file():
        raise RuntimeError(f"Full CKA census manifest is missing: {manifest_path}")
    with config_path.open(encoding="utf-8") as handle:
        analysis_config = json.load(handle)
    if analysis_config.get("schema") != "cka_gt_pilot_postprocess_v1":
        raise RuntimeError(
            f"Unsupported CKA full-census config schema: {analysis_config.get('schema')}"
        )
    expected_layers = list(range(2, 10))
    if analysis_config.get("layers") != expected_layers:
        raise RuntimeError("Full CKA census requires residual layer outputs 2..9.")
    from scripts.analysis.cka_gt_full_census import SCALES as _CENSUS_SCALES
    if analysis_config.get("scales_used_for_gt") != list(_CENSUS_SCALES):
        raise RuntimeError(
            f"Full CKA census config scales {analysis_config.get('scales_used_for_gt')} "
            f"differ from the census module scales {list(_CENSUS_SCALES)}."
        )
    if sorted(int(key) for key in analysis_config.get("candidate_thresholds", {})) != [95, 97, 99]:
        raise RuntimeError(
            "Full CKA census requires the pilot 95/97/99 lines for histogram overlays only."
        )

    provenance = analysis_config.get("prepared_input_provenance", {})
    checkpoints = provenance.get("checkpoint_identity", {})
    source_identity = provenance.get("source_dataset_identity", {}).get("code", {})
    reference_load = os.path.realpath(str(getattr(args, "moe_old_model_kl_load", "") or ""))
    current_load = os.path.realpath(str(getattr(args, "load", "") or ""))
    expected_reference = os.path.realpath(str(checkpoints.get("before", {}).get("resolved_root", "")))
    expected_current = os.path.realpath(str(checkpoints.get("after", {}).get("resolved_root", "")))
    if reference_load != expected_reference or current_load != expected_current:
        raise RuntimeError(
            "Full CKA census checkpoint pair differs from the frozen pilot pair: "
            f"before={reference_load} expected={expected_reference}; "
            f"after={current_load} expected={expected_current}"
        )
    expected_reference_step = int(checkpoints.get("before", {}).get("tracker_step", -1))
    expected_current_step = int(checkpoints.get("after", {}).get("tracker_step", -1))
    if _checkpoint_tracker_step(reference_load) != expected_reference_step:
        raise RuntimeError("Full CKA census before-checkpoint tracker changed.")
    if _checkpoint_tracker_step(current_load) != expected_current_step:
        raise RuntimeError("Full CKA census after-checkpoint tracker changed.")

    dataset_prefix = os.path.realpath(str(source_identity.get("resolved_prefix", "")))
    if not dataset_prefix:
        raise RuntimeError("Full CKA census config has no bound Code dataset prefix.")
    live_source_identity = source_dataset_identity(dataset_prefix)
    normalized_live_source = {
        "schema": live_source_identity.get("schema"),
        "storage_kind": live_source_identity.get("storage_kind"),
        "resolved_prefix": live_source_identity.get("resolved_prefix"),
        "idx_size_bytes": live_source_identity.get("idx", {}).get("size_bytes"),
        "idx_sha256": live_source_identity.get("idx", {}).get("sha256"),
        "bin_size_bytes": live_source_identity.get("bin", {}).get("size_bytes"),
        "bin_sha256": live_source_identity.get("bin", {}).get("sha256"),
    }
    if normalized_live_source != source_identity:
        raise RuntimeError("Full CKA census Code IndexedDataset content identity changed.")
    live_checkpoint_identities = {
        "before": checkpoint_identity(reference_load),
        "after": checkpoint_identity(current_load),
    }
    for name in ("before", "after"):
        live = live_checkpoint_identities[name]
        expected = checkpoints.get(name, {})
        normalized_live = {
            "schema": live.get("schema"),
            "storage_kind": live.get("storage_kind"),
            "resolved_root": live.get("resolved_root"),
            "tracker_step": live.get("tracker_step"),
            "iteration_dir": live.get("iteration_dir"),
            "total_bytes": live.get("total_bytes"),
            "content_sha256": live.get("content_sha256"),
        }
        if normalized_live != expected:
            raise RuntimeError(f"Full CKA census {name} checkpoint content identity changed.")
    dataset = IndexedDataset(dataset_prefix, mmap=True)
    _manifest_metadata, all_manifest_rows = load_full_train_manifest(manifest_path)
    worker_index = int(args.cka_gt_full_census_worker_index)
    worker_count = int(args.cka_gt_full_census_worker_count)
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise RuntimeError("Invalid full CKA census worker index/count.")
    max_windows = int(args.cka_gt_full_census_max_windows)
    if max_windows < 0:
        raise RuntimeError("--cka-gt-full-census-max-windows cannot be negative.")
    if all_manifest_rows.size == 0:
        raise RuntimeError("Full CKA census manifest is empty.")

    teacher = get_old_moe_distill_teacher()
    if teacher is None:
        raise RuntimeError("Full CKA census requires the frozen before checkpoint as teacher.")
    current_modules = _as_module_list(model)
    teacher_modules = _as_module_list(teacher[0])
    layer_spec = "2,3,4,5,6,7,8,9"
    current_layers = [number for number, _ in _collect_transformer_layers(current_modules, layer_spec)]
    reference_layers = [number for number, _ in _collect_transformer_layers(teacher_modules, layer_spec)]
    if current_layers != expected_layers or reference_layers != expected_layers:
        raise RuntimeError(
            f"Full CKA census residual hook mismatch: before={reference_layers}, after={current_layers}"
        )

    batch_size = int(args.cka_gt_full_census_batch_size)
    forward_subbatch_size = int(
        getattr(args, "cka_gt_full_census_forward_subbatch_size", 0) or batch_size
    )
    checkpoint_every = int(args.cka_gt_full_census_checkpoint_every_batches)
    histogram_bins = int(args.cka_gt_full_census_histogram_bins)
    reservoir_size = int(args.cka_gt_full_census_reservoir_size)
    if (batch_size <= 0 or forward_subbatch_size <= 0
            or forward_subbatch_size > batch_size or checkpoint_every <= 0
            or histogram_bins < 200 or reservoir_size <= 0):
        raise RuntimeError(
            "Full CKA census requires 0 < forward subbatch <= journal batch, "
            "positive checkpoint/reservoir values, and >=200 bins."
        )
    benchmark_spec = str(
        getattr(args, "cka_gt_full_census_benchmark_batch_sizes", "") or ""
    ).strip()
    benchmark_batches = []
    if benchmark_spec:
        try:
            benchmark_batches = [int(value) for value in benchmark_spec.split(",")]
        except ValueError as error:
            raise RuntimeError("Invalid full CKA census benchmark batch list.") from error
        if benchmark_batches[:3] != [64, 96, 128] or benchmark_batches not in (
            [64, 96, 128], [64, 96, 128, 192]
        ):
            raise RuntimeError(
                "In-process CKA benchmark must be 64,96,128 with optional trailing 192."
            )
        if worker_index != 0 or worker_count != 1 or not 1024 <= max_windows <= 2048:
            raise RuntimeError(
                "In-process CKA benchmark requires worker 0/1 and 1024..2048 windows."
            )

    def align_hidden(captured, token_batch):
        batch, length = token_batch.shape
        aligned = {}
        for layer in expected_layers:
            value = captured.get(layer)
            if value is None or value.dim() != 3:
                raise RuntimeError(f"Full CKA hook omitted/malformed layer {layer} output.")
            if value.shape[:2] == (length, batch):
                value = value.permute(1, 0, 2).contiguous()
            elif value.shape[:2] != (batch, length):
                raise RuntimeError(
                    f"Full CKA layer {layer} shape {tuple(value.shape)} cannot align "
                    f"to {(batch, length)}"
                )
            aligned[layer] = value
        return aligned

    def warmup_candidate(rows, requested_batch, warmup_batches=3):
        """Warm forward and B/T/M kernels without touching candidate journals."""
        reader = DocumentWindowBatchReader(
            dataset,
            rows,
            batch_size=requested_batch,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        with torch.inference_mode(), \
                _capture_transformer_layer_outputs(
                    teacher_modules, layer_spec, detach=True
                ) as reference_hidden, \
                _capture_transformer_layer_outputs(
                    current_modules, layer_spec, detach=True
                ) as current_hidden:
            for warmup_index, batch in enumerate(reader):
                if warmup_index >= warmup_batches:
                    break
                reference_hidden.clear()
                current_hidden.clear()
                _ = teacher_modules[0](
                    batch.tokens,
                    batch.position_ids,
                    batch.attention_mask,
                    labels=None,
                    runtime_gather_output=True,
                )
                _ = current_modules[0](
                    batch.tokens,
                    batch.position_ids,
                    batch.attention_mask,
                    labels=None,
                    runtime_gather_output=True,
                )
                before = align_hidden(reference_hidden, batch.tokens)
                after = align_hidden(current_hidden, batch.tokens)
                metrics = compute_btm_batch(
                    before_by_layer=before,
                    after_by_layer=after,
                    manifest_rows=batch.rows,
                )
                del metrics, before, after
                reference_hidden.clear()
                current_hidden.clear()
        torch.cuda.synchronize()

    def write_json_atomic(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".inprogress")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def execute_worker(census_worker, *, label):
        rows = np.asarray(census_worker.manifest_rows)
        resume_start_batch = int(census_worker.next_batch_index)
        reader = DocumentWindowBatchReader(
            dataset,
            rows,
            batch_size=int(census_worker.batch_size),
            device=torch.device("cuda", torch.cuda.current_device()),
            start_batch_index=resume_start_batch,
        )
        resumed_windows = sum(len(indices) for indices in reader.plan[:resume_start_batch])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        invocation_started = time.monotonic()
        processed_this_invocation = 0

        def concatenate_metrics(parts):
            if len(parts) == 1:
                return parts[0]
            return BTMBatch(
                raw_b=torch.cat([part.raw_b for part in parts], dim=0),
                b_min={
                    scale: torch.cat([part.b_min[scale] for part in parts], dim=0)
                    for scale in _CENSUS_SCALES
                },
                t_min={
                    scale: torch.cat([part.t_min[scale] for part in parts], dim=0)
                    for scale in _CENSUS_SCALES
                },
                rel_l2=torch.cat([part.rel_l2 for part in parts], dim=0),
                abs_log_r=torch.cat([part.abs_log_r for part in parts], dim=0),
                eligible=torch.cat([part.eligible for part in parts], dim=0),
            )

        def slice_attention_mask(mask, start, stop, total):
            if mask is not None and mask.dim() > 0 and mask.shape[0] == total:
                return mask[start:stop]
            return mask

        with torch.inference_mode(), \
                _capture_transformer_layer_outputs(
                    teacher_modules, layer_spec, detach=True
                ) as reference_hidden, \
                _capture_transformer_layer_outputs(
                    current_modules, layer_spec, detach=True
            ) as current_hidden:
            for batch in reader:
                metric_compute_seconds = 0.0
                forward_compute_seconds = 0.0
                metric_parts = []
                total = int(batch.tokens.shape[0])
                for start in range(0, total, forward_subbatch_size):
                    stop = min(start + forward_subbatch_size, total)
                    tokens = batch.tokens[start:stop]
                    positions = batch.position_ids[start:stop]
                    attention = slice_attention_mask(
                        batch.attention_mask, start, stop, total
                    )
                    rows_part = batch.rows[start:stop]
                    forward_started = time.monotonic()
                    reference_hidden.clear()
                    current_hidden.clear()
                    teacher_output = teacher_modules[0](
                        tokens, positions, attention, labels=None,
                        runtime_gather_output=True,
                    )
                    del teacher_output
                    current_output = current_modules[0](
                        tokens, positions, attention, labels=None,
                        runtime_gather_output=True,
                    )
                    del current_output
                    before = align_hidden(reference_hidden, tokens)
                    after = align_hidden(current_hidden, tokens)
                    forward_compute_seconds += time.monotonic() - forward_started
                    metric_compute_started = time.perf_counter()
                    metric_parts.append(compute_btm_batch(
                        before_by_layer=before, after_by_layer=after,
                        manifest_rows=rows_part,
                    ))
                    metric_compute_seconds += (
                        time.perf_counter() - metric_compute_started
                    )
                    reference_hidden.clear()
                    current_hidden.clear()
                    del before, after
                metrics = concatenate_metrics(metric_parts)
                census_worker.process_precomputed_batch(
                    batch_index=int(batch.batch_index), rows=batch.rows,
                    token_ids=batch.tokens, metrics=metrics,
                    forward_seconds=forward_compute_seconds,
                    metric_seconds=metric_compute_seconds,
                )
                processed_this_invocation += int(batch.tokens.shape[0])
                reference_hidden.clear()
                current_hidden.clear()
                del metrics, metric_parts
                if processed_this_invocation % 2048 < int(batch.tokens.shape[0]):
                    elapsed = max(time.monotonic() - invocation_started, 1e-9)
                    peak_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
                    print_rank_0(
                        f"CKA {label} progress "
                        f"worker={worker_index}/{worker_count} "
                        f"windows_total={resumed_windows + processed_this_invocation}/{rows.size} "
                        f"windows_this_invocation={processed_this_invocation} "
                        f"windows_per_sec={processed_this_invocation / elapsed:.3f} "
                        f"peak_reserved_gib={peak_gib:.2f}"
                    )
        summary = census_worker.finalize()
        torch.cuda.synchronize()
        invocation_elapsed = time.monotonic() - invocation_started
        return {
            "schema": "cka_gt_full_census_model_runtime_v1",
            "complete": bool(summary.get("complete", False)),
            "full_partition_complete": bool(summary.get("full_partition_complete", False)),
            "worker_index": worker_index,
            "worker_count": worker_count,
            "window_batch_size": int(census_worker.batch_size),
            "forward_subbatch_size": int(forward_subbatch_size),
            "histogram_bins": histogram_bins,
            "global_token_score_reservoir_size": int(census_worker.reservoir_size),
            "max_windows": max_windows,
            "partition_windows": int(rows.size),
            "resume_start_batch_index": resume_start_batch,
            "resumed_windows": int(resumed_windows),
            "processed_windows_this_invocation": processed_this_invocation,
            "processed_windows": int(summary.get("processed_windows", 0)),
            "active_seconds": float(summary.get("active_seconds", 0.0)),
            "forward_seconds": float(summary.get("forward_seconds", 0.0)),
            "metric_seconds": float(summary.get("metric_seconds", 0.0)),
            "invocation_elapsed_seconds": invocation_elapsed,
            "invocation_windows_per_second": (
                processed_this_invocation / max(invocation_elapsed, 1e-9)
            ),
            "windows_per_active_second": (
                float(summary.get("processed_windows", 0))
                / max(float(summary.get("active_seconds", 0.0)), 1e-9)
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "reference_checkpoint": reference_load,
            "reference_step": expected_reference_step,
            "current_checkpoint": current_load,
            "current_step": expected_current_step,
            "dataset_prefix": dataset_prefix,
            "manifest_path": str(manifest_path),
            "threshold_policy": "none_distribution_census_pilot_lines_overlay_only",
            "raw_hidden_stored": False,
            "layers": expected_layers,
        }

    def make_worker(root, requested_batch, *, requested_reservoir, limit, full_length_only):
        return FullCensusWorker(
            output_dir=Path(root).resolve(),
            worker_index=worker_index,
            worker_count=worker_count,
            analysis_config_path=config_path,
            manifest_path=manifest_path,
            histogram_bins=histogram_bins,
            reservoir_size=requested_reservoir,
            batch_size=requested_batch,
            checkpoint_every_batches=checkpoint_every,
            max_windows=limit,
            full_length_only=full_length_only,
        )

    prior_current_states = [module.training for module in current_modules]
    prior_teacher_states = [module.training for module in teacher_modules]
    for module in current_modules + teacher_modules:
        module.eval()
    try:
        if benchmark_batches:
            benchmark_root = Path(output_root).resolve()
            selected_path = benchmark_root / "selected_batch.json"
            if selected_path.is_file():
                print_rank_0(f"CKA in-process benchmark already complete: {selected_path}")
                return True
            results = []
            min_gain = float(args.cka_gt_full_census_benchmark_min_gain)
            max_peak_gib = float(args.cka_gt_full_census_benchmark_max_peak_gib)
            if min_gain < 0.0 or not math.isfinite(max_peak_gib) or max_peak_gib <= 0.0:
                raise RuntimeError("Invalid CKA benchmark gain/HBM policy.")
            for candidate in benchmark_batches:
                if candidate == 192:
                    by_batch = {value["window_batch_size"]: value for value in results}
                    if 96 not in by_batch or 128 not in by_batch:
                        break
                    gain = (
                        by_batch[128]["invocation_windows_per_second"]
                        / max(by_batch[96]["invocation_windows_per_second"], 1e-9)
                        - 1.0
                    )
                    peak128 = by_batch[128]["peak_reserved_bytes"] / (1024 ** 3)
                    if gain <= min_gain or peak128 > min(max_peak_gib - 10.0, 60.0):
                        print_rank_0(
                            f"CKA benchmark skips batch 192: gain96to128={gain:.4f}, "
                            f"peak128_gib={peak128:.2f}"
                        )
                        break
                candidate_root = benchmark_root / "candidates" / f"batch_{candidate:03d}"
                runtime_path = candidate_root / "candidate_runtime.json"
                if runtime_path.is_file():
                    with runtime_path.open(encoding="utf-8") as handle:
                        runtime = json.load(handle)
                    results.append(runtime)
                    continue
                candidate_worker = make_worker(
                    candidate_root,
                    candidate,
                    requested_reservoir=0,
                    limit=max_windows,
                    full_length_only=True,
                )
                if int(candidate_worker.next_batch_index) != 0:
                    raise RuntimeError(
                        f"Incomplete benchmark residue exists without runtime: {candidate_root}"
                    )
                try:
                    warmup_candidate(
                        np.asarray(candidate_worker.manifest_rows),
                        candidate,
                        warmup_batches=3,
                    )
                    runtime = execute_worker(
                        candidate_worker, label=f"batch-{candidate}-benchmark"
                    )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print_rank_0(f"CKA benchmark batch {candidate} OOM; excluding it.")
                    if candidate == 64:
                        raise
                    break
                write_json_atomic(runtime_path, runtime)
                results.append(runtime)
                torch.cuda.empty_cache()
            safe = [
                value for value in results
                if value.get("complete")
                and value["peak_reserved_bytes"] / (1024 ** 3) <= max_peak_gib
                and value.get("processed_windows_this_invocation", 0) > 0
            ]
            if not safe:
                raise RuntimeError("No in-process CKA benchmark candidate completed safely.")
            chosen = max(
                safe,
                key=lambda value: (
                    value["invocation_windows_per_second"],
                    -value["window_batch_size"],
                ),
            )
            selection = {
                "schema": "cka_gt_full_census_batch_selection_v2",
                "selected_batch_size": int(chosen["window_batch_size"]),
                "selection_rule": (
                    "max end-to-end windows/s under peak HBM; batch192 gated by >3pct gain"
                ),
                "benchmark_windows": max_windows,
                "full_length_only": True,
                "models_loaded_once": True,
                "max_peak_gib": max_peak_gib,
                "minimum_gain_for_192": min_gain,
                "chosen": chosen,
                "candidates": results,
            }
            write_json_atomic(selected_path, selection)
            print_rank_0("CKA in-process benchmark selected: " + json.dumps(selection, sort_keys=True))
            return True

        worker = make_worker(
            output_root,
            batch_size,
            requested_reservoir=reservoir_size,
            limit=None if max_windows == 0 else max_windows,
            full_length_only=False,
        )
        rows = np.asarray(worker.manifest_rows)
        if rows.size == 0:
            raise RuntimeError("Full CKA census worker received an empty manifest partition.")
        worker_dir = Path(output_root).resolve() / f"worker_{worker_index:03d}"
        binding = {
            "schema": "cka_gt_full_census_model_binding_v2",
            "worker_index": worker_index,
            "worker_count": worker_count,
            "window_batch_size": batch_size,
            "checkpoint_every_batches": checkpoint_every,
            "histogram_bins": histogram_bins,
            "global_token_score_reservoir_size": reservoir_size,
            "max_windows": max_windows,
            "partition_windows": int(rows.size),
            "analysis_config": str(config_path),
            "manifest_path": str(manifest_path),
            "reference_checkpoint_identity": live_checkpoint_identities["before"]["content_sha256"],
            "current_checkpoint_identity": live_checkpoint_identities["after"]["content_sha256"],
            "source_dataset_identity": normalized_live_source,
            "threshold_policy": "none_distribution_census_pilot_lines_overlay_only",
            "raw_hidden_stored": False,
        }
        binding_path = worker_dir / "model_binding.json"
        if binding_path.exists():
            with binding_path.open(encoding="utf-8") as handle:
                existing_binding = json.load(handle)
            if existing_binding != binding:
                differing = sorted(
                    key for key in set(existing_binding) | set(binding)
                    if existing_binding.get(key) != binding.get(key)
                )
                raise RuntimeError(
                    "Full CKA census resume binding changed; batch-index resume unsafe: "
                    f"{differing}"
                )
        else:
            write_json_atomic(binding_path, binding)
        runtime = execute_worker(worker, label="full-census")
        write_json_atomic(worker_dir / "model_runtime.json", runtime)
        print_rank_0("CKA full census completed: " + json.dumps(runtime, sort_keys=True))
        return True
    finally:
        for module, state in zip(current_modules, prior_current_states):
            module.train(state)
        for module, state in zip(teacher_modules, prior_teacher_states):
            module.train(state)


def _run_cka_gt_targeted_gt(model, iteration):
    """Forward exact B-candidate windows and write compact 95/97/99 GT masks."""

    args = get_args()
    output_root = getattr(args, "cka_gt_targeted_path", None)
    if not output_root:
        return False
    if args.pipeline_model_parallel_size != 1 or torch.distributed.get_world_size() != 1:
        raise RuntimeError("Targeted CKA GT requires world size 1 and PP=1.")

    import time
    from pathlib import Path

    import numpy as np

    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from scripts.analysis.cka_gt_full_census import BTMBatch, compute_btm_batch
    from scripts.analysis.cka_gt_pilot_runtime import DocumentWindowBatchReader
    from scripts.analysis.cka_gt_pilot_windows import (
        checkpoint_identity,
        file_sha256,
        source_dataset_identity,
    )
    from scripts.analysis.cka_gt_targeted_gt import ExactTargetedGTWriter

    required_paths = {
        "analysis_config": getattr(args, "cka_gt_targeted_config", None),
        "candidate_manifest": getattr(args, "cka_gt_targeted_candidate_manifest", None),
        "candidate_windows": getattr(args, "cka_gt_targeted_candidate_windows", None),
        "census_summary": getattr(args, "cka_gt_targeted_census_summary", None),
        "authoritative_b": getattr(args, "cka_gt_targeted_authoritative_b", None),
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise RuntimeError(f"Targeted CKA GT is missing required paths: {missing}")
    resolved = {name: Path(value).resolve() for name, value in required_paths.items()}
    absent = [str(path) for path in resolved.values() if not path.is_file()]
    if absent:
        raise RuntimeError(f"Targeted CKA GT inputs are missing: {absent}")

    with resolved["analysis_config"].open(encoding="utf-8") as handle:
        analysis_config = json.load(handle)
    if analysis_config.get("schema") != "cka_gt_pilot_postprocess_v1":
        raise RuntimeError("Targeted CKA GT analysis config schema differs.")
    if analysis_config.get("layers") != list(range(2, 10)):
        raise RuntimeError("Targeted CKA GT requires residual layers 2..9.")
    if analysis_config.get("scales_used_for_gt") != [128, 256]:
        raise RuntimeError("Targeted CKA GT requires scales 128 and 256.")
    if sorted(int(key) for key in analysis_config.get("candidate_thresholds", {})) != [95, 97, 99]:
        raise RuntimeError("Targeted CKA GT requires frozen bundles 95/97/99.")
    with resolved["candidate_manifest"].open(encoding="utf-8") as handle:
        candidate_manifest = json.load(handle)
    with resolved["census_summary"].open(encoding="utf-8") as handle:
        census_summary = json.load(handle)
    if (
        candidate_manifest.get("schema") != "cka_gt_b_candidate_windows_v2"
        or not candidate_manifest.get("complete")
        or not candidate_manifest.get("not_final_gt")
    ):
        raise RuntimeError("Targeted CKA GT candidate manifest is not a complete B-only prefilter.")
    if (
        census_summary.get("schema") != "cka_gt_full_census_merged_summary_v2"
        or not census_summary.get("complete")
        or not census_summary.get("threshold_free")
    ):
        raise RuntimeError("Targeted CKA GT requires a complete threshold-free census summary.")
    candidate_source_manifest = candidate_manifest.get("source_manifest", {}).get(
        "content_identity"
    )
    if candidate_source_manifest != census_summary.get("manifest_identity"):
        raise RuntimeError("Candidate manifest and full census have different source manifests.")
    full_eligible_tokens = int(census_summary.get("processed_eligible_tokens", 0))
    if full_eligible_tokens <= 0:
        raise RuntimeError("Full census has no eligible-token denominator.")

    provenance = analysis_config.get("prepared_input_provenance", {})
    checkpoints = provenance.get("checkpoint_identity", {})
    source_identity = provenance.get("source_dataset_identity", {}).get("code", {})
    reference_load = os.path.realpath(str(getattr(args, "moe_old_model_kl_load", "") or ""))
    current_load = os.path.realpath(str(getattr(args, "load", "") or ""))
    expected_reference = os.path.realpath(str(checkpoints.get("before", {}).get("resolved_root", "")))
    expected_current = os.path.realpath(str(checkpoints.get("after", {}).get("resolved_root", "")))
    if reference_load != expected_reference or current_load != expected_current:
        raise RuntimeError("Targeted CKA GT checkpoint pair differs from the frozen pilot pair.")
    expected_reference_step = int(checkpoints.get("before", {}).get("tracker_step", -1))
    expected_current_step = int(checkpoints.get("after", {}).get("tracker_step", -1))
    if _checkpoint_tracker_step(reference_load) != expected_reference_step:
        raise RuntimeError("Targeted CKA GT before-checkpoint tracker changed.")
    if _checkpoint_tracker_step(current_load) != expected_current_step:
        raise RuntimeError("Targeted CKA GT after-checkpoint tracker changed.")
    live_checkpoint_identities = {
        "before": checkpoint_identity(reference_load),
        "after": checkpoint_identity(current_load),
    }
    for name in ("before", "after"):
        live = live_checkpoint_identities[name]
        expected = checkpoints.get(name, {})
        normalized = {
            "schema": live.get("schema"),
            "storage_kind": live.get("storage_kind"),
            "resolved_root": live.get("resolved_root"),
            "tracker_step": live.get("tracker_step"),
            "iteration_dir": live.get("iteration_dir"),
            "total_bytes": live.get("total_bytes"),
            "content_sha256": live.get("content_sha256"),
        }
        if normalized != expected:
            raise RuntimeError(f"Targeted CKA GT {name} checkpoint content identity changed.")

    dataset_prefix = os.path.realpath(str(source_identity.get("resolved_prefix", "")))
    live_source = source_dataset_identity(dataset_prefix)
    normalized_live_source = {
        "schema": live_source.get("schema"),
        "storage_kind": live_source.get("storage_kind"),
        "resolved_prefix": live_source.get("resolved_prefix"),
        "idx_size_bytes": live_source.get("idx", {}).get("size_bytes"),
        "idx_sha256": live_source.get("idx", {}).get("sha256"),
        "bin_size_bytes": live_source.get("bin", {}).get("size_bytes"),
        "bin_sha256": live_source.get("bin", {}).get("sha256"),
    }
    if normalized_live_source != source_identity:
        raise RuntimeError("Targeted CKA GT Code IndexedDataset identity changed.")

    batch_size = int(args.cka_gt_targeted_batch_size)
    forward_subbatch_size = int(
        getattr(args, "cka_gt_targeted_forward_subbatch_size", 0) or batch_size
    )
    checkpoint_every = int(args.cka_gt_targeted_checkpoint_every_batches)
    if (
        batch_size <= 0
        or forward_subbatch_size <= 0
        or forward_subbatch_size > batch_size
        or checkpoint_every <= 0
    ):
        raise RuntimeError("Targeted CKA GT batch/subbatch/checkpoint values are invalid.")

    teacher = get_old_moe_distill_teacher()
    if teacher is None:
        raise RuntimeError("Targeted CKA GT requires the frozen before checkpoint as teacher.")
    current_modules = _as_module_list(model)
    teacher_modules = _as_module_list(teacher[0])
    layer_spec = "2,3,4,5,6,7,8,9"
    expected_layers = list(range(2, 10))
    current_layers = [number for number, _ in _collect_transformer_layers(current_modules, layer_spec)]
    reference_layers = [number for number, _ in _collect_transformer_layers(teacher_modules, layer_spec)]
    if current_layers != expected_layers or reference_layers != expected_layers:
        raise RuntimeError(
            f"Targeted CKA residual hook mismatch: before={reference_layers}, after={current_layers}"
        )

    def align_hidden(captured, tokens):
        batch, length = tokens.shape
        aligned = {}
        for layer in expected_layers:
            value = captured.get(layer)
            if value is None or value.dim() != 3:
                raise RuntimeError(f"Targeted CKA hook omitted/malformed layer {layer}.")
            if value.shape[:2] == (length, batch):
                value = value.permute(1, 0, 2).contiguous()
            elif value.shape[:2] != (batch, length):
                raise RuntimeError(
                    f"Targeted CKA layer {layer} shape {tuple(value.shape)} cannot align "
                    f"to {(batch, length)}"
                )
            aligned[layer] = value
        return aligned

    def concatenate_metrics(parts):
        if len(parts) == 1:
            return parts[0]
        return BTMBatch(
            raw_b=torch.cat([part.raw_b for part in parts], dim=0),
            b_min={
                scale: torch.cat([part.b_min[scale] for part in parts], dim=0)
                for scale in (128, 256)
            },
            t_min={
                scale: torch.cat([part.t_min[scale] for part in parts], dim=0)
                for scale in (128, 256)
            },
            rel_l2=torch.cat([part.rel_l2 for part in parts], dim=0),
            abs_log_r=torch.cat([part.abs_log_r for part in parts], dim=0),
            eligible=torch.cat([part.eligible for part in parts], dim=0),
        )

    def slice_attention(mask, start, stop, total):
        if mask is not None and mask.dim() > 0 and mask.shape[0] == total:
            return mask[start:stop]
        return mask

    def write_json_atomic(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".inprogress")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def file_identity(path):
        path = Path(path).resolve()
        return {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }

    binding = {
        "schema": "cka_gt_exact_targeted_model_binding_v1",
        "candidate_windows": file_identity(resolved["candidate_windows"]),
        "candidate_manifest": file_identity(resolved["candidate_manifest"]),
        "analysis_config": file_identity(resolved["analysis_config"]),
        "census_summary": file_identity(resolved["census_summary"]),
        "authoritative_b": file_identity(resolved["authoritative_b"]),
        "reference_checkpoint_identity": live_checkpoint_identities["before"]["content_sha256"],
        "current_checkpoint_identity": live_checkpoint_identities["after"]["content_sha256"],
        "source_dataset_identity": normalized_live_source,
        "batch_size": batch_size,
        "forward_subbatch_size": forward_subbatch_size,
        "checkpoint_every_batches": checkpoint_every,
        "raw_hidden_stored": False,
        "sealed_test_opened": False,
    }
    binding_path = Path(output_root).resolve() / "model_binding.json"
    if binding_path.exists():
        existing = json.loads(binding_path.read_text(encoding="utf-8"))
        if existing != binding:
            differing = sorted(
                key for key in set(existing) | set(binding)
                if existing.get(key) != binding.get(key)
            )
            raise RuntimeError(f"Targeted CKA resume model binding changed: {differing}")
    else:
        write_json_atomic(binding_path, binding)

    writer = ExactTargetedGTWriter(
        output_dir=Path(output_root).resolve(),
        candidate_windows_path=resolved["candidate_windows"],
        candidate_manifest_path=resolved["candidate_manifest"],
        analysis_config_path=resolved["analysis_config"],
        census_summary_path=resolved["census_summary"],
        model_binding_path=binding_path,
        authoritative_b_path=resolved["authoritative_b"],
        batch_size=batch_size,
        checkpoint_every_batches=checkpoint_every,
    )
    dataset = IndexedDataset(dataset_prefix, mmap=True)

    prior_current_states = [module.training for module in current_modules]
    prior_teacher_states = [module.training for module in teacher_modules]
    for module in current_modules + teacher_modules:
        module.eval()
    processed_this_invocation = 0
    resumed_windows = int(writer.progress["processed_windows"])
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        reader = DocumentWindowBatchReader(
            dataset,
            np.asarray(writer.rows),
            batch_size=batch_size,
            device=torch.device("cuda", torch.cuda.current_device()),
            start_batch_index=writer.next_batch_index,
        )
        if len(reader.plan) != writer.expected_batch_count:
            raise RuntimeError("Targeted reader and writer deterministic plans differ.")
        with torch.inference_mode(), _capture_transformer_layer_outputs(
            teacher_modules, layer_spec, detach=True
        ) as reference_hidden, _capture_transformer_layer_outputs(
            current_modules, layer_spec, detach=True
        ) as current_hidden:
            for batch in reader:
                parts = []
                total = int(batch.tokens.shape[0])
                for start in range(0, total, forward_subbatch_size):
                    stop = min(start + forward_subbatch_size, total)
                    tokens = batch.tokens[start:stop]
                    positions = batch.position_ids[start:stop]
                    attention = slice_attention(batch.attention_mask, start, stop, total)
                    rows_part = batch.rows[start:stop]
                    reference_hidden.clear()
                    current_hidden.clear()
                    teacher_output = teacher_modules[0](
                        tokens, positions, attention, labels=None, runtime_gather_output=True
                    )
                    del teacher_output
                    current_output = current_modules[0](
                        tokens, positions, attention, labels=None, runtime_gather_output=True
                    )
                    del current_output
                    before = align_hidden(reference_hidden, tokens)
                    after = align_hidden(current_hidden, tokens)
                    parts.append(
                        compute_btm_batch(
                            before_by_layer=before,
                            after_by_layer=after,
                            manifest_rows=rows_part,
                        )
                    )
                    reference_hidden.clear()
                    current_hidden.clear()
                    del before, after
                metrics = concatenate_metrics(parts)
                writer.process_batch(
                    batch_index=int(batch.batch_index),
                    candidate_indices=batch.manifest_indices,
                    rows=batch.rows,
                    token_ids=batch.tokens,
                    metrics=metrics,
                )
                processed_this_invocation += total
                del metrics, parts
                reference_hidden.clear()
                current_hidden.clear()
                elapsed = max(time.monotonic() - started, 1e-9)
                print_rank_0(
                    "CKA targeted GT progress "
                    f"windows={resumed_windows + processed_this_invocation}/"
                    f"{len(writer.rows)} windows_per_sec={processed_this_invocation / elapsed:.3f} "
                    f"peak_reserved_gib={torch.cuda.max_memory_reserved() / (1024 ** 3):.2f}"
                )
        summary = writer.finalize()
        runtime = {
            "schema": "cka_gt_exact_targeted_model_runtime_v1",
            "complete": bool(summary.get("complete")),
            "candidate_windows": int(len(writer.rows)),
            "processed_windows_this_invocation": processed_this_invocation,
            "elapsed_seconds": float(time.monotonic() - started),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "batch_size": batch_size,
            "forward_subbatch_size": forward_subbatch_size,
            "reference_checkpoint": reference_load,
            "current_checkpoint": current_load,
            "raw_hidden_stored": False,
            "sealed_test_opened": False,
            "not_human_locked_final_gt": True,
        }
        write_json_atomic(Path(output_root).resolve() / "model_runtime.json", runtime)
        print_rank_0("CKA targeted GT completed: " + json.dumps(runtime, sort_keys=True))
        return True
    finally:
        for module, state in zip(current_modules, prior_current_states):
            module.train(state)
        for module, state in zip(teacher_modules, prior_teacher_states):
            module.train(state)


def _run_cka_gt_pilot(model, iteration):
    """Run the read-only document-bounded CKA anchor pilot.

    Checkpoint construction remains owned by Megatron.  Dataset slicing,
    metrics, membership statistics and restartable scalar storage live in the
    analysis modules so this path cannot silently turn into a training path.
    """
    args = get_args()
    worker_output = getattr(args, "cka_gt_pilot_path", None)
    if not worker_output:
        return False
    if args.pipeline_model_parallel_size != 1 or torch.distributed.get_world_size() != 1:
        raise RuntimeError(
            "CKA GT pilot requires one process per independent GPU and PP=1. "
            "Worker partitioning is explicit and never uses data parallel sampling."
        )
    if not getattr(args, "cka_gt_pilot_config", None):
        raise RuntimeError("CKA GT pilot requires --cka-gt-pilot-config.")

    import time
    from pathlib import Path

    import numpy as np

    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from scripts.analysis.cka_gt_pilot_runtime import (
        DocumentWindowBatchReader,
        PairedParquetMetricWriter,
        Pass1MembershipAccumulator,
        RANDOM_PAIR_DONOR_CACHE_SIZE,
        RANDOM_PAIR_REASON_NAMES,
        RandomPairDonorCache,
        load_membership_statistics,
        paired_metric_stream_paths,
        process_pass2_metric_batch,
        recover_pass1_worker_metadata_from_stats,
        router_diagnostics_from_input,
        run_model_pilot,
        save_membership_statistics_atomic,
    )
    from scripts.analysis.cka_gt_pilot_windows import (
        checkpoint_identity,
        source_dataset_identity,
    )

    config_path = Path(args.cka_gt_pilot_config).resolve()
    pilot_root = config_path.parent
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema") != "cka_gt_pilot_prepared_inputs_v1":
        raise RuntimeError(f"Unsupported CKA pilot config schema: {config.get('schema')}")
    if int(args.seed) != int(config.get("base_seed", -1)):
        raise RuntimeError(
            "CKA pilot runtime seed must match the frozen prepared manifest: "
            f"runtime={args.seed}, prepared={config.get('base_seed')}"
        )
    if config.get("layers") != list(range(2, 10)):
        raise RuntimeError(f"CKA pilot config must select layers 2..9: {config.get('layers')}")
    if config.get("representation_policy", {}).get("raw_hidden_disk_storage") is not False:
        raise RuntimeError("CKA pilot config must explicitly forbid raw-hidden disk storage.")

    live_source_identities = {
        domain_name: source_dataset_identity(config[f"{domain_name}_prefix"])
        for domain_name in ("code", "wiki")
    }
    if config.get("source_dataset_identity") != live_source_identities:
        raise RuntimeError(
            "CKA source IndexedDataset content differs from the prepared config."
        )
    live_checkpoint_identities = {
        "before": checkpoint_identity(config["before_checkpoint"]),
        "after": checkpoint_identity(config["after_checkpoint"]),
    }
    if config.get("checkpoint_identity") != live_checkpoint_identities:
        raise RuntimeError(
            "CKA checkpoint content differs from the prepared config."
        )

    reference_load = str(getattr(args, "moe_old_model_kl_load", "") or "")
    current_load = str(getattr(args, "load", "") or "")
    expected_reference = str(config.get("before_checkpoint", ""))
    expected_current = str(config.get("after_checkpoint", ""))
    if os.path.realpath(reference_load) != os.path.realpath(expected_reference):
        raise RuntimeError(
            f"CKA before checkpoint mismatch: runtime={reference_load}, config={expected_reference}"
        )
    if os.path.realpath(current_load) != os.path.realpath(expected_current):
        raise RuntimeError(
            f"CKA after checkpoint mismatch: runtime={current_load}, config={expected_current}"
        )
    if _checkpoint_tracker_step(reference_load) != 600:
        raise RuntimeError("CKA before checkpoint tracker must be step 600.")
    if _checkpoint_tracker_step(current_load) != 1800:
        raise RuntimeError("CKA after checkpoint tracker must be step 1800.")

    teacher = get_old_moe_distill_teacher()
    if teacher is None:
        raise RuntimeError("CKA GT pilot requires the frozen before model teacher.")
    current_modules = _as_module_list(model)
    teacher_modules = _as_module_list(teacher[0])
    layer_spec = "2,3,4,5,6,7,8,9"
    expected_layers = list(range(2, 10))
    current_layers = [number for number, _ in _collect_transformer_layers(current_modules, layer_spec)]
    reference_layers = [number for number, _ in _collect_transformer_layers(teacher_modules, layer_spec)]
    if current_layers != expected_layers or reference_layers != expected_layers:
        raise RuntimeError(
            f"CKA residual layer mismatch: before={reference_layers}, after={current_layers}"
        )
    current_router_layers = {
        number: router for number, _mlp, router in _collect_moe_router_layers(current_modules)
        if number in expected_layers
    }
    reference_router_layers = {
        number: router for number, _mlp, router in _collect_moe_router_layers(teacher_modules)
        if number in expected_layers
    }
    if sorted(current_router_layers) != expected_layers or sorted(reference_router_layers) != expected_layers:
        raise RuntimeError(
            "CKA probe did not find the standard mlp.router at every MoE layer 2..9: "
            f"before={sorted(reference_router_layers)}, after={sorted(current_router_layers)}"
        )

    mode = str(args.cka_gt_pilot_mode)
    domain = str(args.cka_gt_pilot_domain)
    requested_split = str(args.cka_gt_pilot_split)
    worker_index = int(args.cka_gt_pilot_worker_index)
    worker_count = int(args.cka_gt_pilot_worker_count)
    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise RuntimeError("Invalid CKA logical worker index/count.")
    if mode == "pass1" and (
        domain != "wiki" or requested_split != "calibration" or worker_count != 1
    ):
        raise RuntimeError("CKA Pass 1 is exactly one Wiki-calibration worker.")
    split_names = (
        ("calibration", "selection", "test")
        if requested_split == "all" else (requested_split,)
    )
    dataset_prefix = str(config[f"{domain}_prefix"])
    dataset = IndexedDataset(dataset_prefix, mmap=True)

    def load_partition(split_name):
        rows_path = pilot_root / "splits" / f"{domain}_{split_name}_windows.npy"
        rows = np.load(rows_path, allow_pickle=False)
        quotient, remainder = divmod(int(rows.shape[0]), worker_count)
        start = worker_index * quotient + min(worker_index, remainder)
        count = quotient + int(worker_index < remainder)
        rows = rows[start : start + count]
        cap = int(getattr(args, "cka_gt_pilot_max_windows", 0) or 0)
        return rows[:cap] if cap > 0 else rows

    rows_by_split = {split_name: load_partition(split_name) for split_name in split_names}
    total_windows = sum(int(rows.shape[0]) for rows in rows_by_split.values())
    total_tokens = sum(
        int(rows["window_length"].sum(dtype=np.int64)) for rows in rows_by_split.values()
    )
    if total_windows <= 0 or total_tokens <= 0:
        raise RuntimeError("CKA worker received an empty document-window partition.")

    # Every Pass2 worker must bind to the one Pass1 source: the complete
    # Wiki-calibration partition (subject only to the same explicit smoke cap).
    membership_source_rows = np.load(
        pilot_root / "splits" / "wiki_calibration_windows.npy", allow_pickle=False
    )
    membership_cap = int(getattr(args, "cka_gt_pilot_max_windows", 0) or 0)
    if membership_cap > 0:
        membership_source_rows = membership_source_rows[:membership_cap]
    membership_source_windows = int(membership_source_rows.shape[0])
    membership_source_tokens = int(
        membership_source_rows["window_length"].sum(dtype=np.int64)
    )
    membership_expected_metadata = {
        "analysis": "cka_gt_pilot_v1",
        "mode": "pass1",
        "source_domain": "wiki",
        "source_split": "calibration",
        "source_dataset_prefix": str(config["wiki_prefix"]),
        "reference_checkpoint": os.path.realpath(reference_load),
        "reference_step": 600,
        "prepared_config_content_sha256": config.get(
            "config_content_sha256"
        ),
        "source_window_count": membership_source_windows,
        "source_token_count": membership_source_tokens,
        "seed": int(args.seed),
        "max_windows": membership_cap,
        "layers": expected_layers,
    }

    device = torch.device("cuda", torch.cuda.current_device())
    batch_size = int(args.cka_gt_pilot_batch_size)
    if batch_size <= 0:
        raise RuntimeError("--cka-gt-pilot-batch-size must be positive.")

    prior_current_states = [module.training for module in current_modules]
    prior_teacher_states = [module.training for module in teacher_modules]
    for module in current_modules + teacher_modules:
        module.eval()

    def align_hidden(captured, token_batch):
        aligned = {}
        batch, length = token_batch.shape
        for layer in expected_layers:
            value = captured.get(layer)
            if value is None or value.dim() != 3:
                raise RuntimeError(f"CKA hook omitted/malformed layer {layer} output.")
            if value.shape[:2] == (length, batch):
                value = value.permute(1, 0, 2).contiguous()
            elif value.shape[:2] != (batch, length):
                raise RuntimeError(
                    f"CKA layer {layer} shape {tuple(value.shape)} cannot align to {(batch, length)}"
                )
            aligned[layer] = value
        return aligned

    metadata = {
        "analysis": "cka_gt_pilot_v1",
        "label": "replay-stable relational anchor",
        "mode": mode,
        "domain": domain,
        "requested_split": requested_split,
        "dataset_prefix": dataset_prefix,
        "worker_index": worker_index,
        "worker_count": worker_count,
        # Batch grouping is part of the random-pair-null definition.  Pin it
        # in the transaction metadata so an interrupted worker cannot resume
        # with different cross-window pairs while appearing compatible.
        "window_batch_size": batch_size,
        "shard_windows": int(args.cka_gt_pilot_shard_windows),
        "max_windows": int(getattr(args, "cka_gt_pilot_max_windows", 0) or 0),
        "seed": int(args.seed),
        "prepared_config_content_sha256": config.get("config_content_sha256"),
        "source_dataset_identity": live_source_identities,
        "checkpoint_identity": live_checkpoint_identities,
        "total_windows": total_windows,
        "total_tokens": total_tokens,
        "reference_load": reference_load,
        "reference_step": 600,
        "current_load": current_load,
        "current_step": 1800,
        "layers": expected_layers,
        "representation": "residual_included_transformer_layer_output",
        "standard_router_layers": expected_layers,
        "natural_routing": True,
        "old_expert_ids": list(range(8)),
        "raw_hidden_stored": False,
        "training_run": False,
        "window_processing_order": (
            "split_order_calibration_selection_test_then_window_length_descending_"
            "then_sample_order_stable"
        ),
        "pass1_subsample_axis": "the_recorded_window_processing_order",
        "loaded_iteration_before_diagnostic_override": int(iteration),
    }
    worker_output_path = Path(worker_output)
    worker_output_path.mkdir(parents=True, exist_ok=True)

    def write_worker_metadata(payload):
        final_path = worker_output_path / "metadata.json"
        temporary_path = worker_output_path / "metadata.json.inprogress"
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, final_path)

    def model_driver(*, model, teacher, args, precision_audit, print_fn):
        started = time.monotonic()
        metadata["precision_audit"] = dict(precision_audit)
        if mode == "pass1":
            stats_path = Path(args.cka_gt_pilot_membership_stats).resolve()
            existing_metadata_path = worker_output_path / "metadata.json"
            if stats_path.exists():
                if not existing_metadata_path.is_file():
                    recovered = recover_pass1_worker_metadata_from_stats(
                        stats_path,
                        existing_metadata_path,
                        expected_saved_metadata=membership_expected_metadata,
                        expected_runtime_metadata={
                            "mode": "pass1",
                            "reference_load": reference_load,
                            "window_batch_size": batch_size,
                            "max_windows": int(
                                getattr(args, "cka_gt_pilot_max_windows", 0) or 0
                            ),
                            "seed": int(args.seed),
                            "prepared_config_content_sha256": config.get(
                                "config_content_sha256"
                            ),
                            "total_windows": total_windows,
                            "total_tokens": total_tokens,
                        },
                    )
                    print_fn(
                        "CKA Pass 1 recovered completed worker metadata from the "
                        "atomically committed membership NPZ."
                    )
                    return recovered
                with existing_metadata_path.open(encoding="utf-8") as handle:
                    existing_metadata = json.load(handle)
                existing_membership = load_membership_statistics(
                    stats_path,
                    expected_saved_metadata=membership_expected_metadata,
                )
                matching = (
                    existing_metadata.get("completed") is True
                    and existing_metadata.get("mode") == "pass1"
                    and existing_metadata.get("reference_load") == reference_load
                    and existing_metadata.get("window_batch_size") == batch_size
                    and existing_metadata.get("max_windows")
                    == int(getattr(args, "cka_gt_pilot_max_windows", 0) or 0)
                    and existing_metadata.get("seed") == int(args.seed)
                    and existing_metadata.get("prepared_config_content_sha256")
                    == config.get("config_content_sha256")
                    and existing_metadata.get("total_windows") == total_windows
                    and existing_metadata.get("total_tokens") == total_tokens
                    and existing_metadata.get("membership_statistics") == str(stats_path)
                    and existing_metadata.get("membership_identity")
                    == existing_membership.membership_identity
                )
                if not matching:
                    raise RuntimeError(
                        "CKA Pass 1 existing statistics do not match this exact request."
                    )
                print_fn(f"CKA Pass 1 already complete: {stats_path}")
                return existing_metadata
            accumulator = Pass1MembershipAccumulator(
                layers=expected_layers,
                hidden_size=int(args.hidden_size),
                total_tokens=total_tokens,
                covariance_sample_size=min(2_000_000, total_tokens),
                kmeans_reservoir_size=min(200_000, total_tokens),
                seed=int(args.seed),
            )
            with torch.no_grad(), \
                    _capture_transformer_layer_outputs(
                        teacher_modules, layer_spec, detach=True
                    ) as reference_hidden, \
                    _capture_moe_router_inputs(
                        teacher_modules, detach=True
                    ) as reference_router_capture:
                reference_router_inputs, captured_reference_routers = reference_router_capture
                if sorted(captured_reference_routers) != expected_layers:
                    raise RuntimeError("CKA Pass 1 did not hook standard routers 2..9.")
                probe_checked = False
                processed_windows = 0
                for split_name in split_names:
                    reader = DocumentWindowBatchReader(
                        dataset, rows_by_split[split_name], batch_size=batch_size, device=device
                    )
                    for batch in reader:
                        reference_hidden.clear()
                        reference_router_inputs.clear()
                        _ = teacher_modules[0](
                            batch.tokens, batch.position_ids, batch.attention_mask,
                            labels=None, runtime_gather_output=True
                        )
                        if not probe_checked:
                            diagnostics = {
                                layer: router_diagnostics_from_input(
                                    captured_reference_routers[layer], reference_router_inputs[layer]
                                ) for layer in expected_layers
                            }
                            if any(
                                diagnostics[layer]["top4_id"].shape[-1] != 4
                                for layer in expected_layers
                            ):
                                raise RuntimeError("CKA Pass 1 standard router probe is not top-4.")
                            probe_checked = True
                            metadata["router_probe_verified"] = True
                            print_fn(
                                "CKA router probe verified standard mlp.router layers 2..9, top-k=4."
                            )
                            del diagnostics
                        accumulator.update(align_hidden(reference_hidden, batch.tokens))
                        processed_windows += int(batch.tokens.shape[0])
                        reference_hidden.clear()
                        reference_router_inputs.clear()
                        if processed_windows % 1000 < int(batch.tokens.shape[0]):
                            elapsed = max(time.monotonic() - started, 1e-9)
                            print_fn(
                                f"CKA pass1 progress windows={processed_windows}/{total_windows} "
                                f"tokens={accumulator.processed_tokens}/{total_tokens} "
                                f"tokens_per_sec={accumulator.processed_tokens / elapsed:.1f}"
                            )
            statistics = accumulator.finalize(kmeans_device=device)
            pass1_elapsed = time.monotonic() - started
            metadata["elapsed_seconds"] = pass1_elapsed
            metadata["cumulative_elapsed_seconds"] = pass1_elapsed
            metadata["timing_available"] = True
            save_membership_statistics_atomic(stats_path, statistics, metadata=metadata)
            loaded_membership = load_membership_statistics(
                stats_path,
                expected_saved_metadata=membership_expected_metadata,
            )
            final = {
                **metadata,
                "completed": True,
                "membership_statistics": str(stats_path),
                "membership_identity": loaded_membership.membership_identity,
                "elapsed_seconds": pass1_elapsed,
                "cumulative_elapsed_seconds": pass1_elapsed,
                "timing_available": True,
            }
            write_worker_metadata(final)
            return final

        membership = None
        if mode == "pass2":
            if not args.cka_gt_pilot_membership_stats:
                raise RuntimeError("CKA Pass 2 requires membership statistics.")
            membership = load_membership_statistics(
                args.cka_gt_pilot_membership_stats,
                expected_saved_metadata=membership_expected_metadata,
            )
            pass1_metadata_path = (
                pilot_root
                / "runtime"
                / "pass1"
                / "wiki"
                / "calibration"
                / "worker_000"
                / "metadata.json"
            )
            if not pass1_metadata_path.is_file():
                raise RuntimeError(
                    "CKA Pass 2 requires completed canonical Pass1 worker metadata: "
                    f"{pass1_metadata_path}"
                )
            with pass1_metadata_path.open(encoding="utf-8") as handle:
                pass1_metadata = json.load(handle)
            if pass1_metadata.get("completed") is not True or pass1_metadata.get(
                "membership_identity"
            ) != membership.membership_identity:
                raise RuntimeError(
                    "CKA membership NPZ identity differs from completed Pass1 metadata."
                )
            metadata["membership_identity"] = membership.membership_identity
            metadata["membership_pass1_worker_metadata"] = str(
                pass1_metadata_path.resolve()
            )

        def window_uids(split_name, rows):
            split_id = {"calibration": 0, "selection": 1, "test": 2}[split_name]
            domain_id = {"code": 0, "wiki": 1}[domain]
            return (
                np.int64(domain_id) * np.int64(1_000_000_000)
                + np.int64(split_id) * np.int64(100_000_000)
                + rows["sample_order"].astype(np.int64)
            )

        donor_candidates_by_split = {}
        for split_name, split_rows in rows_by_split.items():
            full_rows = split_rows[split_rows["window_length"] == 512]
            if full_rows.size:
                full_rows = full_rows[
                    np.argsort(
                        full_rows["sample_order"].astype(np.int64), kind="stable"
                    )
                ]
            donor_candidates_by_split[split_name] = window_uids(
                split_name, full_rows[:RANDOM_PAIR_DONOR_CACHE_SIZE]
            ).astype(np.int64).tolist()
        metadata["random_pair_null"] = {
            "policy_version": "deterministic_full_after_donor_cache_v1",
            "batch_ge_2": "within_batch_seeded_cyclic_derangement_no_fixed_points",
            "singleton": (
                "sha256(window_uid,scale,chunk_start)_indexed_nonself_donor_from_"
                "first_8_full_after_windows"
            ),
            "donor_cache_scope": "per_split_per_worker",
            "donor_cache_size": RANDOM_PAIR_DONOR_CACHE_SIZE,
            "donor_cache_storage": "memory_only_rebuilt_by_forward_on_resume",
            "candidate_window_uids_by_split": donor_candidates_by_split,
            "reason_codes": {
                str(key): value for key, value in RANDOM_PAIR_REASON_NAMES.items()
            },
            "coverage_columns": [
                "random_pair_invalid_reason",
                "random_pair_donor_window_uid",
                "cka_random_pair",
            ],
            "selector_dependency": False,
        }

        # Test rows are a distinct physical transaction below sealed_test/raw;
        # they never enter the open calibration/selection Parquet tree.
        stream_splits = {
            "open": tuple(name for name in split_names if name != "test"),
            "test": tuple(name for name in split_names if name == "test"),
        }
        paired_writers = {}
        committed_by_stream = {}
        expected_by_stream = {}
        if mode == "pass2":
            for stream, names in stream_splits.items():
                if not names:
                    continue
                paths = paired_metric_stream_paths(
                    pilot_root=pilot_root,
                    worker_output=worker_output_path,
                    domain=domain,
                    requested_split=requested_split,
                    worker_index=worker_index,
                    stream=stream,
                )
                expected = set()
                stream_token_count = 0
                for name in names:
                    expected.update(window_uids(name, rows_by_split[name]).tolist())
                    stream_token_count += int(
                        rows_by_split[name]["window_length"].sum(dtype=np.int64)
                    )
                stream_metadata = {
                    **metadata,
                    "metric_stream": stream,
                    "stream_splits": list(names),
                    "test_metrics_sealed": stream == "test",
                    "stream_window_count": len(expected),
                    "stream_token_count": stream_token_count,
                }
                writer = PairedParquetMetricWriter(
                    paths.journal_dir,
                    token_dir=paths.token_dir,
                    chunk_dir=paths.chunk_dir,
                    target_token_rows_per_shard=max(
                        1, int(args.cka_gt_pilot_shard_windows) * 512
                    ),
                    metadata=stream_metadata,
                )
                if writer.metadata != stream_metadata:
                    differing = sorted(
                        key
                        for key in set(writer.metadata) | set(stream_metadata)
                        if writer.metadata.get(key) != stream_metadata.get(key)
                    )
                    raise RuntimeError(
                        f"CKA {stream} paired resume metadata mismatch: {differing}"
                    )
                committed = set(writer.committed_window_uids().tolist())
                if not committed.issubset(expected):
                    raise RuntimeError(
                        f"CKA {stream} resume contains windows outside this worker partition."
                    )
                paired_writers[stream] = writer
                expected_by_stream[stream] = expected
                committed_by_stream[stream] = committed

            if set(expected_by_stream.get("open", ())) & set(
                expected_by_stream.get("test", ())
            ):
                raise RuntimeError("CKA open/test expected window sets overlap")
            expected_union = set().union(*expected_by_stream.values())
            if len(expected_union) != total_windows:
                raise RuntimeError(
                    "CKA physical stream partition does not exactly cover worker windows."
                )
            processed_windows = sum(len(values) for values in committed_by_stream.values())
            if all(
                committed_by_stream[name] == expected_by_stream[name]
                for name in paired_writers
            ):
                paired_manifests = {"open": None, "test": None}
                for name, writer in paired_writers.items():
                    paired_manifests[name] = writer.finalize()
                cumulative_elapsed = sum(
                    writer.cumulative_elapsed_seconds
                    for writer in paired_writers.values()
                )
                final = {
                    **metadata,
                    "completed": True,
                    "router_probe_verified": True,
                    "paired_manifests": paired_manifests,
                    "elapsed_seconds": cumulative_elapsed,
                    "cumulative_elapsed_seconds": cumulative_elapsed,
                    "resume_invocation_elapsed_seconds": time.monotonic() - started,
                    "resume_skip": True,
                }
                write_worker_metadata(final)
                print_fn(f"CKA pass2 worker already complete: {worker_output_path}")
                return final
        else:
            processed_windows = 0
        probe_checked = False
        with torch.no_grad(), \
                _capture_transformer_layer_outputs(
                    teacher_modules, layer_spec, detach=True
                ) as reference_hidden, \
                _capture_transformer_layer_outputs(
                    current_modules, layer_spec, detach=True
                ) as current_hidden, \
                _capture_moe_router_inputs(
                    teacher_modules, detach=True
                ) as reference_router_capture, \
                _capture_moe_router_inputs(
                    current_modules, detach=True
                ) as current_router_capture:
            reference_router_inputs, captured_reference_routers = reference_router_capture
            current_router_inputs, captured_current_routers = current_router_capture
            if sorted(captured_reference_routers) != expected_layers or sorted(captured_current_routers) != expected_layers:
                raise RuntimeError("CKA runtime hook captured a non-standard or incomplete router set.")
            for split_name in split_names:
                pending_active_started = time.monotonic()
                all_split_rows = rows_by_split[split_name]
                random_pair_donor_cache = None
                if mode == "pass2":
                    random_pair_donor_cache = RandomPairDonorCache(
                        domain=domain,
                        split=split_name,
                        layers=expected_layers,
                        manifest_rows=all_split_rows,
                        max_windows=RANDOM_PAIR_DONOR_CACHE_SIZE,
                    )
                    # This deterministic memory-only prepass is deliberately
                    # rebuilt even when donor metric UIDs were committed before
                    # a crash.  It forwards at most eight full AFTER windows and
                    # never writes their hidden states to disk.
                    donor_rows = random_pair_donor_cache.candidate_rows
                    if donor_rows.size:
                        donor_reader = DocumentWindowBatchReader(
                            dataset,
                            donor_rows,
                            batch_size=min(batch_size, RANDOM_PAIR_DONOR_CACHE_SIZE),
                            device=device,
                        )
                        for donor_batch in donor_reader:
                            reference_hidden.clear()
                            current_hidden.clear()
                            reference_router_inputs.clear()
                            current_router_inputs.clear()
                            _ = current_modules[0](
                                donor_batch.tokens,
                                donor_batch.position_ids,
                                donor_batch.attention_mask,
                                labels=None,
                                runtime_gather_output=True,
                            )
                            donor_after = align_hidden(
                                current_hidden, donor_batch.tokens
                            )
                            random_pair_donor_cache.add(
                                donor_after, donor_batch.rows
                            )
                            reference_hidden.clear()
                            current_hidden.clear()
                            reference_router_inputs.clear()
                            current_router_inputs.clear()
                            del donor_after
                        if not random_pair_donor_cache.complete:
                            raise RuntimeError(
                                "CKA singleton random-pair donor cache did not rebuild completely."
                            )
                rows = all_split_rows
                stream = "test" if split_name == "test" else "open"
                uids = window_uids(split_name, rows)
                committed_windows = committed_by_stream.get(stream, set())
                if committed_windows:
                    keep = np.asarray(
                        [int(uid) not in committed_windows for uid in uids], dtype=bool
                    )
                    rows = rows[keep]
                if rows.size == 0:
                    continue
                reader = DocumentWindowBatchReader(
                    dataset, rows, batch_size=batch_size, device=device
                )
                for batch in reader:
                    active_started = pending_active_started
                    pending_active_started = None
                    if active_started is None:
                        active_started = time.monotonic()
                    reference_hidden.clear(); current_hidden.clear()
                    reference_router_inputs.clear(); current_router_inputs.clear()
                    _ = teacher_modules[0](
                        batch.tokens, batch.position_ids, batch.attention_mask,
                        labels=None, runtime_gather_output=True
                    )
                    _ = current_modules[0](
                        batch.tokens, batch.position_ids, batch.attention_mask,
                        labels=None, runtime_gather_output=True
                    )
                    before = align_hidden(reference_hidden, batch.tokens)
                    after = align_hidden(current_hidden, batch.tokens)
                    before_router = {
                        layer: router_diagnostics_from_input(
                            captured_reference_routers[layer], reference_router_inputs[layer]
                        ) for layer in expected_layers
                    }
                    after_router = {
                        layer: router_diagnostics_from_input(
                            captured_current_routers[layer], current_router_inputs[layer]
                        ) for layer in expected_layers
                    }
                    if not probe_checked:
                        for diagnostics in (before_router, after_router):
                            for layer in expected_layers:
                                if diagnostics[layer]["top4_id"].shape[-1] != 4:
                                    raise RuntimeError("Standard router probe did not return top-4 dispatch.")
                        probe_checked = True
                        print_fn("CKA router probe verified standard mlp.router layers 2..9, top-k=4.")
                    if mode == "router_smoke":
                        final = {
                            **metadata,
                            "completed": True,
                            "router_probe_verified": True,
                            "smoke_window_length": int(batch.tokens.shape[1]),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                        write_worker_metadata(final)
                        return final
                    token_columns, chunk_columns = process_pass2_metric_batch(
                        before_by_layer=before,
                        after_by_layer=after,
                        token_ids=batch.tokens,
                        manifest_rows=batch.rows,
                        domain=domain,
                        split=split_name,
                        membership_statistics=membership,
                        router_before=before_router,
                        router_after=after_router,
                        random_pair_donor_cache=random_pair_donor_cache,
                        seed=int(args.seed),
                    )
                    paired_writers[stream].append_batch(
                        token_columns,
                        chunk_columns,
                        active_seconds=time.monotonic() - active_started,
                    )
                    processed_windows += int(batch.tokens.shape[0])
                    reference_hidden.clear(); current_hidden.clear()
                    reference_router_inputs.clear(); current_router_inputs.clear()
                    del before, after, before_router, after_router, token_columns, chunk_columns
                    if processed_windows % 100 < int(batch.tokens.shape[0]):
                        elapsed = max(time.monotonic() - started, 1e-9)
                        print_fn(
                            f"CKA pass2 progress domain={domain} worker={worker_index}/{worker_count} "
                            f"windows={processed_windows}/{total_windows} "
                            f"windows_per_sec={processed_windows / elapsed:.3f}"
                        )

        paired_manifests = {"open": None, "test": None}
        for stream, writer in paired_writers.items():
            paired_manifests[stream] = writer.finalize()
            committed = set(writer.committed_window_uids().tolist())
            if committed != expected_by_stream[stream]:
                missing = len(expected_by_stream[stream] - committed)
                extra = len(committed - expected_by_stream[stream])
                raise RuntimeError(
                    f"CKA {stream} final window coverage mismatch: "
                    f"missing={missing} extra={extra}"
                )
        cumulative_elapsed = sum(
            writer.cumulative_elapsed_seconds for writer in paired_writers.values()
        )
        final = {
            **metadata,
            "completed": True,
            "router_probe_verified": probe_checked,
            "paired_manifests": paired_manifests,
            "elapsed_seconds": cumulative_elapsed,
            "cumulative_elapsed_seconds": cumulative_elapsed,
            "resume_invocation_elapsed_seconds": time.monotonic() - started,
        }
        write_worker_metadata(final)
        return final

    try:
        result = run_model_pilot(
            model=model,
            teacher=teacher,
            args=args,
            hooks={"driver": model_driver},
            print_fn=print_rank_0,
        )
        print_rank_0("CKA GT pilot completed: " + json.dumps(result, sort_keys=True))
    finally:
        for module, state in zip(current_modules, prior_current_states):
            module.train(state)
        for module, state in zip(teacher_modules, prior_teacher_states):
            module.train(state)
    return True


def _run_code_token_hidden_pair(model, iteration):
    """Persist restartable per-token, per-layer metrics on exact Code train samples."""
    args = get_args()
    output_dir = getattr(args, "code_token_hidden_pair_path", None)
    if not output_dir:
        return False
    if args.pipeline_model_parallel_size != 1 or torch.distributed.get_world_size() != 1:
        raise RuntimeError(
            "Code token hidden-pair extraction requires one process per independent GPU worker "
            "and PP=1. Partitioning is explicit through start/count arguments."
        )
    if not args.probe_data_path:
        raise RuntimeError("Code token hidden-pair extraction requires --probe-data-path.")
    if args.code_token_hidden_pair_samples <= 0:
        raise RuntimeError("--code-token-hidden-pair-samples must be positive.")
    teacher = get_old_moe_distill_teacher()
    if teacher is None:
        raise RuntimeError(
            "Code token hidden-pair extraction requires --moe-old-model-kl-load as reference."
        )

    from scripts.analysis.code_token_hidden_pair_core import (
        TokenMetricShardWriter,
        storage_estimate,
    )

    current_modules = _as_module_list(model)
    teacher_modules = _as_module_list(teacher[0])
    layer_spec = args.code_token_hidden_pair_layers
    current_layers = [number for number, _ in _collect_transformer_layers(current_modules, layer_spec)]
    teacher_layers = [number for number, _ in _collect_transformer_layers(teacher_modules, layer_spec)]
    if current_layers != teacher_layers or not current_layers:
        raise RuntimeError(
            f"Reference/current layer mismatch: reference={teacher_layers}, current={current_layers}"
        )

    partition_start = int(args.code_token_hidden_pair_start_sample)
    partition_samples = int(args.code_token_hidden_pair_samples)
    total_samples = int(args.code_token_hidden_pair_total_samples)
    expected_partition_end = partition_start + partition_samples
    if expected_partition_end > total_samples:
        raise RuntimeError(
            f"worker partition exceeds total samples: {partition_start}+{partition_samples}>"
            f"{total_samples}"
        )
    reference_load = getattr(args, "moe_old_model_kl_load", None)
    current_load = getattr(args, "load", None)
    metadata = {
        "label": args.code_token_hidden_pair_label or "",
        "representation": "residual_included_transformer_layer_output",
        "reference_load": reference_load,
        "reference_tracker_step": _checkpoint_tracker_step(reference_load),
        "current_load": current_load,
        "current_tracker_step": _checkpoint_tracker_step(current_load),
        "loaded_iteration_before_diagnostic_override": int(iteration),
        # Historical key retained for compatibility with the original Code
        # analysis.  New consumers must use data_blend for multi-shard tasks.
        "code_data_path": str(args.probe_data_path[-1]),
        "analysis_task": (args.code_token_hidden_pair_label or "").split("_train_", 1)[0],
        "data_blend": list(args.probe_data_path),
        "code_data_blend": list(args.probe_data_path),
        "dataset_split": "100,0,0",
        "seed": int(args.seed),
        "worker_index": int(args.code_token_hidden_pair_worker_index),
        "worker_count": int(args.code_token_hidden_pair_worker_count),
        "runtime_data_parallel_rank": int(mpu.get_data_parallel_rank()),
        "runtime_data_parallel_world_size": int(mpu.get_data_parallel_world_size()),
        "token_identity_layout": {
            "sample": "sample_ids[dense_sample_axis]",
            "sequence_position": "dense_sequence_axis (0..sequence_length-1)",
            "global_token_offset": "sample_id * sequence_length + sequence_position",
            "input_token": "input_token_ids[dense_sample_axis, sequence_position]",
            "prediction_target": "label_token_ids[dense_sample_axis, sequence_position]",
        },
        "natural_routing": True,
        "eval_mode": True,
        "no_grad": True,
        "storage_estimate": storage_estimate(total_samples * int(args.seq_length), len(current_layers)),
    }
    writer = TokenMetricShardWriter(
        output_dir=output_dir,
        layers=current_layers,
        hidden_size=int(args.hidden_size),
        sequence_length=int(args.seq_length),
        partition_start_sample=partition_start,
        partition_samples=partition_samples,
        total_samples=total_samples,
        shard_samples=int(args.code_token_hidden_pair_shard_samples),
        reservoir_size=int(args.code_token_hidden_pair_reservoir_size),
        seed=int(args.seed),
        metadata=metadata,
    )
    if writer.is_complete:
        print_rank_0(f"Code token hidden-pair partition already complete: {output_dir}")
        return True

    resume_sample = partition_start + writer.completed_samples
    dataloader = _build_code_token_hidden_pair_dataloader(
        args.probe_data_path,
        total_samples,
        resume_sample,
    )
    iterator = iter(dataloader)
    prior_states = [module.training for module in current_modules]
    for module in current_modules:
        module.eval()
    for module in teacher_modules:
        module.eval()

    prior_shard = writer.next_shard
    try:
        with torch.no_grad(), \
                _capture_transformer_layer_outputs(
                    teacher_modules, layer_spec, detach=True
                ) as reference_hidden, \
                _capture_transformer_layer_outputs(
                    current_modules, layer_spec, detach=True
                ) as current_hidden:
            while writer.remaining_samples > 0:
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(iterator)
                original_batch = int(tokens.shape[0])
                take = min(original_batch, writer.remaining_samples)
                if take != original_batch:
                    tokens = tokens[:take]
                    labels = labels[:take]
                    loss_mask = loss_mask[:take]
                    position_ids = position_ids[:take]
                    if attention_mask is not None and attention_mask.shape[0] == original_batch:
                        attention_mask = attention_mask[:take]
                sample_ids = torch.arange(
                    writer.next_global_sample,
                    writer.next_global_sample + take,
                    dtype=torch.int64,
                    device=tokens.device,
                )
                _ = teacher_modules[0](
                    tokens, position_ids, attention_mask, labels=None, runtime_gather_output=True
                )
                _ = current_modules[0](
                    tokens, position_ids, attention_mask, labels=None, runtime_gather_output=True
                )
                reference_flat = {
                    layer: _flatten_layer_hidden(reference_hidden[layer], labels)
                    for layer in current_layers
                }
                current_flat = {
                    layer: _flatten_layer_hidden(current_hidden[layer], labels)
                    for layer in current_layers
                }
                writer.update(
                    sample_ids,
                    tokens,
                    labels,
                    loss_mask,
                    reference_flat,
                    current_flat,
                )
                reference_hidden.clear()
                current_hidden.clear()
                del reference_flat, current_flat
                if writer.next_shard != prior_shard:
                    prior_shard = writer.next_shard
                    progress_tokens = (
                        writer.completed_samples * int(args.seq_length)
                    )
                    total_partition_tokens = partition_samples * int(args.seq_length)
                    print_rank_0(
                        "code-token hidden-pair progress: "
                        f"worker={args.code_token_hidden_pair_worker_index}/"
                        f"{args.code_token_hidden_pair_worker_count} "
                        f"samples={writer.completed_samples}/{partition_samples} "
                        f"dense_tokens={progress_tokens}/{total_partition_tokens} "
                        f"shards={writer.next_shard}"
                    )
                    print_rank_0(
                        "code-token hidden-pair running diagnostics: "
                        + json.dumps(writer.progress_snapshot(), sort_keys=True)
                    )
    finally:
        for module, was_training in zip(current_modules, prior_states):
            if was_training:
                module.train()

    result = writer.finalize()
    print_rank_0(
        "saved Code token hidden-pair metrics: "
        f"{output_dir} samples={result['completed_samples']} "
        f"valid_tokens={result['completed_valid_tokens']} shards={result['shards']}"
    )
    return True


def _run_layer_output_streaming_stats(model, iteration):
    """Run paired reference/current forwards and stream residual layer-output moments."""
    args = get_args()
    output_dir = getattr(args, "layer_output_streaming_stats_path", None)
    if not output_dir:
        return False
    if args.pipeline_model_parallel_size != 1 or torch.distributed.get_world_size() != 1:
        raise RuntimeError("Layer-output streaming analysis currently requires one process and PP=1.")
    if not args.probe_data_path or args.probe_eval_iters <= 0:
        raise RuntimeError("Layer-output streaming analysis requires a primary probe dataloader.")
    teacher = get_old_moe_distill_teacher()
    if teacher is None:
        raise RuntimeError(
            "Layer-output streaming analysis requires --moe-old-model-kl-load as the frozen reference."
        )

    from scripts.analysis.layer_output_streaming_core import (
        PairedStreamingStats,
        validate_or_write_manifest,
    )

    current_modules = _as_module_list(model)
    # The training global stores the model list; keep this normalization explicit.
    teacher_modules = _as_module_list(teacher[0])
    layer_spec = args.layer_output_streaming_layers
    current_layers = [number for number, _ in _collect_transformer_layers(current_modules, layer_spec)]
    teacher_layers = [number for number, _ in _collect_transformer_layers(teacher_modules, layer_spec)]
    if current_layers != teacher_layers or not current_layers:
        raise RuntimeError(
            f"Reference/current layer mismatch: reference={teacher_layers}, current={current_layers}"
        )
    hidden_size = int(args.hidden_size)
    dataloader = _build_probe_dataloader(
        args.probe_data_path, args.probe_eval_iters, "layer_output_streaming_stats"
    )
    if dataloader is None:
        raise RuntimeError("Could not build layer-output streaming probe dataloader.")

    metadata = {
        "label": args.layer_output_streaming_label or "",
        "representation": "residual_included_transformer_layer_output",
        "reference_load": getattr(args, "moe_old_model_kl_load", None),
        "current_load": getattr(args, "load", None),
        "current_iteration": int(iteration),
        "probe_data_path": list(args.probe_data_path),
        "seed": int(args.seed),
        "sequence_length": int(args.seq_length),
        "dtype": "float64_accumulated_from_float32_gemm",
        "ffn_output_diagnostic": bool(args.layer_output_streaming_ffn_diagnostic),
    }
    stats = PairedStreamingStats(
        output_dir=output_dir,
        layers=current_layers,
        hidden_size=hidden_size,
        block_tokens=args.layer_output_streaming_block_tokens,
        target_tokens=args.layer_output_streaming_target_tokens,
        reservoir_size=args.layer_output_streaming_reservoir_size,
        seed=args.seed,
        metadata=metadata,
    )

    prior_states = [module.training for module in current_modules]
    for module in current_modules:
        module.eval()
    ffn_enabled = bool(args.layer_output_streaming_ffn_diagnostic)
    teacher_ffn_ctx = (
        _capture_moe_ffn_outputs(teacher_modules, layer_spec, detach=True)
        if ffn_enabled else nullcontext({})
    )
    current_ffn_ctx = (
        _capture_moe_ffn_outputs(current_modules, layer_spec, detach=True)
        if ffn_enabled else nullcontext({})
    )
    probe_iterator = iter(dataloader)
    encountered_samples = 0
    try:
        with torch.no_grad(), \
                _capture_transformer_layer_outputs(teacher_modules, layer_spec, detach=True) as before, \
                _capture_transformer_layer_outputs(current_modules, layer_spec, detach=True) as after, \
                teacher_ffn_ctx as ffn_before, current_ffn_ctx as ffn_after:
            for _ in range(args.probe_eval_iters):
                if stats.remaining() <= 0:
                    break
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(probe_iterator)
                stats.record_samples(tokens, loss_mask)
                _ = teacher_modules[0](
                    tokens, position_ids, attention_mask, labels=None, runtime_gather_output=True
                )
                _ = current_modules[0](
                    tokens, position_ids, attention_mask, labels=None, runtime_gather_output=True
                )
                flat_mask = loss_mask.reshape(-1).bool()
                indices = torch.nonzero(flat_mask, as_tuple=False).view(-1)
                if indices.numel() == 0:
                    encountered_samples += int(tokens.shape[0])
                    continue
                indices = indices[: stats.remaining()]
                flat_tokens = tokens.reshape(-1)
                seq_length = int(tokens.shape[1])
                before_flat = {
                    layer: _flatten_layer_hidden(before[layer], labels)[indices]
                    for layer in current_layers
                }
                after_flat = {
                    layer: _flatten_layer_hidden(after[layer], labels)[indices]
                    for layer in current_layers
                }
                ffn_before_flat = (
                    {layer: _flatten_layer_hidden(ffn_before[layer], labels)[indices]
                     for layer in current_layers} if ffn_enabled else None
                )
                ffn_after_flat = (
                    {layer: _flatten_layer_hidden(ffn_after[layer], labels)[indices]
                     for layer in current_layers} if ffn_enabled else None
                )
                stats.update(
                    before_flat,
                    after_flat,
                    flat_tokens[indices],
                    indices % seq_length,
                    ffn_before_flat,
                    ffn_after_flat,
                )
                encountered_samples += int(tokens.shape[0])
                if stats.total and stats.total % args.layer_output_streaming_block_tokens == 0:
                    print_rank_0(
                        f"layer-output streaming progress: {stats.total}/"
                        f"{args.layer_output_streaming_target_tokens} valid tokens"
                    )
    finally:
        for module, was_training in zip(current_modules, prior_states):
            if was_training:
                module.train()

    result = stats.finalize()
    manifest_path = args.layer_output_streaming_manifest_path
    if manifest_path:
        validate_or_write_manifest(result, manifest_path)
    print_rank_0(
        f"saved paired layer-output streaming statistics: {output_dir} "
        f"({result['target_tokens']} tokens, {len(result['blocks'])} blocks)"
    )
    return True


def _run_fingerprint_score_stats(model, iteration):
    """Stream teacher-only centered projection-energy token scores."""
    args = get_args()
    output_dir = getattr(args, "fingerprint_score_stats_path", None)
    if not output_dir:
        return False
    if args.pipeline_model_parallel_size != 1 or torch.distributed.get_world_size() != 1:
        raise RuntimeError("Fingerprint score analysis currently requires one process and PP=1.")
    bundle_path = getattr(args, "fingerprint_score_bundle", None)
    if not bundle_path:
        raise RuntimeError("--fingerprint-score-stats-path requires --fingerprint-score-bundle.")
    if not args.probe_data_path or args.probe_eval_iters <= 0:
        raise RuntimeError("Fingerprint score analysis requires a primary probe dataloader.")

    import numpy as np
    from scripts.analysis.fingerprint_score_streaming_core import FingerprintScoreStats
    from scripts.analysis.layer_output_streaming_core import validate_or_write_manifest

    with np.load(bundle_path, allow_pickle=False) as payload:
        representation_names = payload["representation_names"].tolist()
        layer_numbers = payload["layer_numbers"].astype(np.int64).tolist()
        ranks = payload["ranks"].astype(np.int64).tolist()
        means_np = payload["means"]
        bases_np = payload["bases"]
    if layer_numbers != list(range(2, 10)):
        raise RuntimeError(f"Fingerprint score bundle must contain layers 2-9: {layer_numbers}")
    if ranks != [16, 32, 64]:
        raise RuntimeError(f"Fingerprint score bundle must contain ranks 16,32,64: {ranks}")
    if bases_np.shape != (len(representation_names), 8, args.hidden_size, 64):
        raise RuntimeError(f"Unexpected fingerprint score basis shape: {bases_np.shape}")
    if means_np.shape != (8, args.hidden_size):
        raise RuntimeError(f"Unexpected fingerprint score mean shape: {means_np.shape}")

    modules = _as_module_list(model)
    selected_layers = [number for number, _ in _collect_transformer_layers(modules, "2,3,4,5,6,7,8,9")]
    if selected_layers != layer_numbers:
        raise RuntimeError(f"Model/bundle layer mismatch: model={selected_layers}, bundle={layer_numbers}")
    device = next(unwrap_model(modules)[0].parameters()).device
    means = torch.as_tensor(means_np, dtype=torch.float32, device=device)
    bases = torch.as_tensor(bases_np, dtype=torch.float32, device=device)
    # One GEMM per layer: concatenate stable/PCA/random rank-64 bases.
    basis_concat = [
        bases[:, layer_index].permute(1, 0, 2).reshape(args.hidden_size, -1).contiguous()
        for layer_index in range(8)
    ]
    selector_names = ("layer_2", "layer_5", "layer_9", "layers_2_to_9_mean")
    dataloader = _build_probe_dataloader(
        args.probe_data_path, args.probe_eval_iters, "fingerprint_score_stats"
    )
    if dataloader is None:
        raise RuntimeError("Could not build fingerprint score probe dataloader.")
    metadata = {
        "label": args.fingerprint_score_label or "",
        "score_definition": "centered_projection_energy_ratio",
        "teacher_load": getattr(args, "load", None),
        "teacher_iteration": int(iteration),
        "fingerprint_bundle": os.path.abspath(bundle_path),
        "representation": "residual_included_transformer_layer_output",
        "layers": layer_numbers,
        "probe_data_path": list(args.probe_data_path),
        "seed": int(args.seed),
        "sequence_length": int(args.seq_length),
        "natural_routing": True,
    }
    stats = FingerprintScoreStats(
        output_dir=output_dir,
        representation_names=representation_names,
        ranks=ranks,
        selector_names=selector_names,
        target_tokens=args.fingerprint_score_target_tokens,
        block_tokens=args.fingerprint_score_block_tokens,
        vocab_size=args.padded_vocab_size,
        sequence_length=args.seq_length,
        reservoir_size=args.fingerprint_score_reservoir_size,
        seed=args.seed,
        metadata=metadata,
    )

    prior_states = [module.training for module in modules]
    for module in modules:
        module.eval()
    probe_iterator = iter(dataloader)
    try:
        with torch.no_grad(), _capture_transformer_layer_outputs(
            modules, "2,3,4,5,6,7,8,9", detach=True
        ) as captured:
            for _ in range(args.probe_eval_iters):
                if stats.remaining() <= 0:
                    break
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(probe_iterator)
                stats.record_samples(tokens, loss_mask)
                _ = modules[0](
                    tokens, position_ids, attention_mask, labels=None, runtime_gather_output=True
                )
                indices = torch.nonzero(loss_mask.reshape(-1).bool(), as_tuple=False).view(-1)
                if indices.numel() == 0:
                    continue
                indices = indices[: stats.remaining()]
                layer_scores = []
                for layer_index, layer_number in enumerate(layer_numbers):
                    hidden = _flatten_layer_hidden(captured[layer_number], labels)[indices].float()
                    centered = hidden - means[layer_index]
                    denominator = centered.square().sum(dim=-1).clamp_min(1e-20)
                    projected = (centered @ basis_concat[layer_index]).view(
                        centered.shape[0], len(representation_names), 64
                    ).permute(1, 0, 2)
                    cumulative = projected.square().cumsum(dim=-1)
                    per_rank = torch.stack(
                        [cumulative[..., rank - 1] / denominator.unsqueeze(0) for rank in ranks],
                        dim=1,
                    )
                    layer_scores.append(per_rank)
                # [representation, rank, layer, token]
                layer_scores = torch.stack(layer_scores, dim=2)
                selector_scores = torch.stack(
                    (
                        layer_scores[:, :, 0],
                        layer_scores[:, :, 3],
                        layer_scores[:, :, 7],
                        layer_scores.mean(dim=2),
                    ),
                    dim=2,
                )
                flat_tokens = tokens.reshape(-1)
                positions = indices % int(tokens.shape[1])
                stats.update(flat_tokens[indices], positions, selector_scores)
                if stats.total and stats.total % args.fingerprint_score_block_tokens == 0:
                    print_rank_0(
                        f"fingerprint score progress: {stats.total}/"
                        f"{args.fingerprint_score_target_tokens} valid tokens"
                    )
    finally:
        for module, was_training in zip(modules, prior_states):
            if was_training:
                module.train()

    result = stats.finalize()
    manifest_path = args.fingerprint_score_manifest_path
    if manifest_path:
        validate_or_write_manifest(result, manifest_path)
    print_rank_0(
        f"saved fingerprint token score statistics: {output_dir} "
        f"({result['target_tokens']} tokens)"
    )
    return True


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

    detailed = bool(getattr(args, "hidden_space_dump_routing_details", False))
    routing_only = detailed and os.environ.get("HIDDEN_DUMP_ROUTING_ONLY", "") == "1"
    capture_expert_outputs = bool(
        detailed and getattr(args, "hidden_space_dump_expert_outputs", False)
    )
    if capture_expert_outputs and args.expert_model_parallel_size != 1:
        raise RuntimeError(
            "--hidden-space-dump-expert-outputs currently requires expert parallel size 1."
        )

    captured = {}
    handles = []
    layer_chunks = {layer_number: [] for layer_number, _ in layer_modules}
    component_names = (
        "layer_input",
        "attention_output",
        "post_attention_hidden",
        "router_input",
        "ffn_output",
        "layer_output",
    )
    if routing_only:
        component_names = ("router_input",)
    component_chunks = (
        {
            name: {layer_number: [] for layer_number, _ in layer_modules}
            for name in component_names
        }
        if detailed
        else {}
    )
    routing_chunks = (
        {
            name: {layer_number: [] for layer_number, _ in layer_modules}
            for name in (
                "router_logits",
                "router_full_probs",
                "router_topk_indices",
                "router_topk_weights",
                "router_top1_margin",
                "router_topk_margin",
                "router_entropy",
                "expert_outputs",
                "expert_output_norms",
            )
        }
        if detailed
        else {}
    )
    router_weights = {}
    selected_token_ids = []
    selected_positions = []
    selected_sample_indices = []
    total_tokens = 0
    total_samples = 0
    max_tokens = max(1, int(args.hidden_space_dump_max_tokens))
    tokens_per_sample = max(0, int(args.hidden_space_dump_tokens_per_sample))

    moe_layers = {
        layer_number: (mlp, router)
        for layer_number, mlp, router in _collect_moe_router_layers(modules)
    }
    if detailed:
        missing = sorted(set(layer_number for layer_number, _ in layer_modules) - set(moe_layers))
        if missing:
            raise RuntimeError(
                "Detailed hidden dump requires MoE/router modules at every selected layer; "
                f"missing layers={missing}."
            )

    def make_layer_output_hook(layer_number):
        def hook(_module, _inputs, output):
            tensor = _layer_output_tensor(output).detach()
            captured[layer_number] = tensor
            if detailed:
                captured[(layer_number, "layer_output")] = tensor
        return hook

    def make_pre_hook(layer_number, name):
        def hook(_module, inputs, kwargs):
            tensor = inputs[0] if inputs else kwargs.get("hidden_states")
            if torch.is_tensor(tensor):
                captured[(layer_number, name)] = tensor.detach()
        return hook

    def make_output_hook(layer_number, name):
        def hook(_module, _inputs, output):
            captured[(layer_number, name)] = _layer_output_tensor(output).detach()
        return hook

    for layer_number, layer in layer_modules:
        handles.append(layer.register_forward_hook(make_layer_output_hook(layer_number)))
        if detailed:
            # routing-only keeps just the router input; the other five components
            # are already available from the earlier hidden dump.
            if not routing_only:
                handles.append(
                    layer.register_forward_pre_hook(
                        make_pre_hook(layer_number, "layer_input"), with_kwargs=True
                    )
                )
                handles.append(
                    layer.self_attention.register_forward_hook(
                        make_output_hook(layer_number, "attention_output")
                    )
                )
                handles.append(
                    layer.pre_mlp_layernorm.register_forward_pre_hook(
                        make_pre_hook(layer_number, "post_attention_hidden"), with_kwargs=True
                    )
                )
            handles.append(
                moe_layers[layer_number][1].register_forward_pre_hook(
                    make_pre_hook(layer_number, "router_input"), with_kwargs=True
                )
            )
            if not routing_only:
                handles.append(layer.mlp.register_forward_hook(make_output_hook(layer_number, "ffn_output")))

    def select_indices(loss_mask):
        nonlocal total_samples
        batch_size, seq_length = loss_mask.shape
        if tokens_per_sample <= 0:
            flat_mask = loss_mask.reshape(-1).bool()
            indices = torch.nonzero(flat_mask, as_tuple=False).view(-1)
            sample_ids = indices // seq_length + total_samples
            total_samples += batch_size
            return indices, sample_ids

        selected = []
        sample_ids = []
        for sample_idx in range(batch_size):
            valid = torch.nonzero(loss_mask[sample_idx].bool(), as_tuple=False).view(-1)
            if valid.numel() == 0:
                continue
            count = min(tokens_per_sample, int(valid.numel()))
            if count == 1:
                chosen = valid[:1]
            else:
                offsets = torch.linspace(
                    0, valid.numel() - 1, steps=count, device=valid.device
                ).round().long()
                chosen = valid[offsets]
            selected.append(chosen + sample_idx * seq_length)
            sample_ids.append(
                torch.full_like(chosen, total_samples + sample_idx, dtype=torch.long)
            )
        total_samples += batch_size
        if not selected:
            empty = torch.empty(0, dtype=torch.long, device=loss_mask.device)
            return empty, empty
        return torch.cat(selected), torch.cat(sample_ids)

    def selected_expert_outputs(mlp, hidden, topk_indices):
        experts = getattr(getattr(mlp, "experts", None), "local_experts", None)
        local_indices = getattr(mlp, "local_expert_indices", None)
        if experts is None or local_indices is None:
            raise RuntimeError(
                "Expert-output capture currently supports SequentialMLP local_experts only."
            )
        offset = int(local_indices[0])
        result = torch.empty(
            (*topk_indices.shape, hidden.shape[-1]),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        for global_expert in torch.unique(topk_indices).tolist():
            local_expert = int(global_expert) - offset
            if local_expert < 0 or local_expert >= len(experts):
                raise RuntimeError(
                    f"Selected non-local expert {global_expert} during EP=1 detailed dump."
                )
            locations = torch.nonzero(topk_indices == global_expert, as_tuple=False)
            token_rows = locations[:, 0]
            output, bias = experts[local_expert](hidden[token_rows])
            if bias is not None:
                output = output + bias
            result[locations[:, 0], locations[:, 1]] = output
        return result

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

                candidate_indices, candidate_sample_ids = select_indices(loss_mask)
                if candidate_indices.numel() == 0:
                    continue
                remaining = max_tokens - total_tokens
                candidate_indices = candidate_indices[:remaining]
                candidate_sample_ids = candidate_sample_ids[:remaining]

                flat_tokens = tokens.reshape(-1)
                batch_size, seq_length = labels.shape
                selected_token_ids.append(flat_tokens[candidate_indices].detach().cpu())
                selected_positions.append((candidate_indices % seq_length).detach().cpu())
                selected_sample_indices.append(candidate_sample_ids.detach().cpu())

                for layer_number, _layer in layer_modules:
                    if layer_number not in captured:
                        raise RuntimeError(f"Layer {layer_number} was not captured during hidden dump.")
                    flat_hidden = _flatten_layer_hidden(captured[layer_number], labels)
                    layer_chunks[layer_number].append(
                        flat_hidden[candidate_indices].float().detach().cpu()
                    )
                    if not detailed:
                        continue

                    selected_components = {}
                    for name in component_names:
                        key = (layer_number, name)
                        if key not in captured:
                            raise RuntimeError(
                                f"Layer {layer_number} component {name} was not captured."
                            )
                        flat_component = _flatten_layer_hidden(captured[key], labels)
                        selected_component = flat_component[candidate_indices]
                        selected_components[name] = selected_component
                        component_chunks[name][layer_number].append(
                            selected_component.float().detach().cpu()
                        )

                    mlp, router = moe_layers[layer_number]
                    router_hidden = selected_components["router_input"]
                    logits = router.gating(router_hidden).float()
                    routing_probs, _routing_map = router.routing(
                        logits.view(logits.shape[0], 1, logits.shape[1])
                    )
                    full_probs = torch.softmax(logits, dim=-1)
                    topk = min(int(args.moe_router_topk), logits.shape[-1])
                    topk_weights, topk_indices = torch.topk(routing_probs, k=topk, dim=-1)
                    sorted_logits = torch.sort(logits, dim=-1, descending=True).values
                    top1_margin = sorted_logits[:, 0] - sorted_logits[:, 1]
                    if logits.shape[-1] > topk:
                        topk_margin = sorted_logits[:, topk - 1] - sorted_logits[:, topk]
                    else:
                        topk_margin = torch.full_like(top1_margin, float("inf"))
                    entropy = -(full_probs * full_probs.clamp_min(1e-20).log()).sum(dim=-1)

                    for name, tensor in (
                        ("router_logits", logits),
                        ("router_full_probs", full_probs),
                        ("router_topk_indices", topk_indices),
                        ("router_topk_weights", topk_weights),
                        ("router_top1_margin", top1_margin),
                        ("router_topk_margin", topk_margin),
                        ("router_entropy", entropy),
                    ):
                        routing_chunks[name][layer_number].append(tensor.detach().cpu())

                    router_weights[layer_number] = router.weight.float().detach().cpu()
                    if capture_expert_outputs:
                        expert_output = selected_expert_outputs(
                            mlp, router_hidden, topk_indices
                        )
                        routing_chunks["expert_outputs"][layer_number].append(
                            expert_output.float().detach().cpu()
                        )
                        routing_chunks["expert_output_norms"][layer_number].append(
                            expert_output.float().norm(dim=-1).detach().cpu()
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
        "encountered_samples": int(total_samples),
        "tokens_per_sample": int(tokens_per_sample),
        "layer_numbers": layer_numbers,
        "hidden_shape": list(hidden_layers.shape),
        "routing_details": detailed,
        "expert_outputs": capture_expert_outputs,
        "load": getattr(args, "load", None),
    }

    if torch.distributed.get_rank() == 0:
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(dump_path)), exist_ok=True)
        payload = dict(
            hidden_layers=(np.zeros((0,), dtype=np.float32) if routing_only else hidden_layers),
            layer_numbers=np.asarray(layer_numbers, dtype=np.int64),
            token_ids=token_ids,
            positions=positions,
            sample_indices=sample_indices,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
        if detailed:
            for name in component_names:
                payload[name] = torch.stack(
                    [
                        torch.cat(component_chunks[name][layer_number], dim=0)
                        for layer_number in layer_numbers
                    ],
                    dim=0,
                ).numpy()
            for name in (
                "router_logits",
                "router_full_probs",
                "router_topk_indices",
                "router_topk_weights",
                "router_top1_margin",
                "router_topk_margin",
                "router_entropy",
            ):
                payload[name] = torch.stack(
                    [
                        torch.cat(routing_chunks[name][layer_number], dim=0)
                        for layer_number in layer_numbers
                    ],
                    dim=0,
                ).numpy()
            payload["router_weights"] = torch.stack(
                [router_weights[layer_number] for layer_number in layer_numbers], dim=0
            ).numpy()
            if capture_expert_outputs:
                for name in ("expert_outputs", "expert_output_norms"):
                    payload[name] = torch.stack(
                        [
                            torch.cat(routing_chunks[name][layer_number], dim=0)
                            for layer_number in layer_numbers
                        ],
                        dim=0,
                    ).numpy()
        np.savez_compressed(dump_path, **payload)
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
    shared_routers = _collect_current_shared_routers(modules) if router_usage_enabled else {}
    standard_router_layers = _collect_moe_router_layers(modules) if router_usage_enabled else []
    routers_available = bool(shared_routers or standard_router_layers)
    if router_usage_enabled and not routers_available:
        print_rank_0("Probe router usage requested, but no MoE router modules were found.")

    with torch.no_grad():
        for _ in range(probe_eval_iters):
            training_micro_batch_size = args.micro_batch_size
            if args.probe_micro_batch_size is not None:
                args.micro_batch_size = args.probe_micro_batch_size
            try:
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(probe_iterator)
            finally:
                args.micro_batch_size = training_micro_batch_size
            if router_usage_enabled and routers_available:
                # This shared helper selects the native shared-router capture
                # when present, otherwise forward-hooks the standard
                # TransformerLayer.mlp.router inputs.  Both paths preserve
                # natural routing and are read-only during probe evaluation.
                with _capture_distill_router_inputs(
                    modules, detach=True
                ) as (captured_router_inputs, routers):
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
    if _run_cka_gt_targeted_gt(model, iteration):
        return
    if _run_cka_gt_full_census(model, iteration):
        return
    if _run_cka_gt_pilot(model, iteration):
        return
    if _run_code_token_hidden_pair(model, iteration):
        return
    if _run_fingerprint_score_stats(model, iteration):
        return
    if _run_layer_output_streaming_stats(model, iteration):
        return
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
