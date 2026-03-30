#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.transformer.moe.router import Router
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_args(parser):
    group = parser.add_argument_group(title="controlled-expert-inference")
    group.add_argument("--output-json", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--model-kind", type=str, required=True, choices=("ffn", "lora"))
    group.add_argument("--source-num-experts", type=int, default=4)
    group.add_argument("--allowed-experts", type=str, default="")
    group.add_argument("--topk-override", type=int, default=None)
    group.add_argument("--debug-router-json", type=str, default=None)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    return parser


def parse_allowed_experts(spec: str):
    spec = (spec or "").strip()
    if not spec:
        return None
    return sorted({int(piece) for piece in spec.split(",") if piece.strip()})


def align_logits(logits, labels):
    if logits.dim() != 3:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    if logits.shape[0] == labels.shape[0] and logits.shape[1] == labels.shape[1]:
        return logits
    if logits.shape[0] == labels.shape[1] and logits.shape[1] == labels.shape[0]:
        return logits.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"Could not align logits {tuple(logits.shape)} and labels {tuple(labels.shape)}")


def build_eval_dataloader():
    args = get_args()
    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    split_idx = split_name_to_index[args.dataset_split_name]
    requested_samples = args.consumed_samples + args.eval_iters * args.global_batch_size
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=None,
        split=args.dataset_split,
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
    sizes = [0, 0, 0]
    sizes[split_idx] = requested_samples
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        sizes,
        is_dataset_built_on_rank,
        config,
    ).build()
    selected_ds = (train_ds, valid_ds, test_ds)[split_idx]
    return build_pretraining_data_loader(selected_ds, args.consumed_samples)


def _finalize_routing_record(record, expert_idx, expert_scores, source_num_experts):
    token_count = int(expert_idx.shape[0])
    record["token_count"] += token_count
    topk = int(expert_idx.shape[1])
    record["topk"] = topk
    record["selected_score_sum"] += float(expert_scores.float().sum().item())
    is_new = expert_idx >= int(source_num_experts)
    all_old_mask = (~is_new).all(dim=-1)
    all_new_mask = is_new.all(dim=-1)
    mixed_mask = ~(all_old_mask | all_new_mask)
    record["all_old_count"] += int(all_old_mask.sum().item())
    record["mixed_count"] += int(mixed_mask.sum().item())
    record["all_new_count"] += int(all_new_mask.sum().item())
    record["new_slot_fraction_sum"] += float(is_new.float().mean(dim=-1).sum().item())
    record["expert_assignment_counts"] += torch.bincount(
        expert_idx.reshape(-1).detach().cpu(), minlength=record["expert_assignment_counts"].shape[0]
    ).to(torch.int64)


def install_ffn_router_controls(model, allowed_experts, requested_topk, debug_state):
    for module in model.modules():
        if not isinstance(module, Router):
            continue
        if not hasattr(module, "_analysis_original_gating"):
            module._analysis_original_gating = module.gating
        if not hasattr(module, "_analysis_original_topk"):
            module._analysis_original_topk = int(getattr(module, "topk", 1))
        if hasattr(module, "config") and hasattr(module.config, "moe_router_topk") and not hasattr(
            module, "_analysis_original_config_topk"
        ):
            module._analysis_original_config_topk = int(module.config.moe_router_topk)
        if getattr(module, "_analysis_debug_hook_handle", None) is not None:
            module._analysis_debug_hook_handle.remove()
            module._analysis_debug_hook_handle = None

        original_gating = module._analysis_original_gating
        num_experts = int(module.num_experts)
        allowed = allowed_experts if allowed_experts is not None else list(range(num_experts))
        effective_topk = min(requested_topk or int(getattr(module, "topk", 1)), len(allowed))
        if effective_topk <= 0:
            raise ValueError("No allowed experts left after applying FFN router mask.")
        if hasattr(module, "topk"):
            module.topk = effective_topk
        if hasattr(module, "config") and hasattr(module.config, "moe_router_topk"):
            module.config.moe_router_topk = effective_topk

        def masked_gating(input_tensor, _original_gating=original_gating, _allowed=allowed):
            logits = _original_gating(input_tensor)
            if len(_allowed) != logits.shape[-1]:
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[_allowed] = False
                logits[..., mask] = torch.finfo(logits.dtype).min
            return logits

        module.gating = masked_gating

        if debug_state is not None:
            def make_hook(layer_number):
                def _hook(_module, _inputs, outputs):
                    expert_scores, routing_map = outputs
                    routing_map = routing_map.detach()
                    expert_idx = torch.nonzero(routing_map, as_tuple=False)
                    if expert_idx.numel() == 0:
                        return
                    token_indices = expert_idx[:, 0]
                    expert_ids = expert_idx[:, 1]
                    topk = int(_module.topk)
                    gathered_scores = expert_scores[token_indices, expert_ids]
                    score_buffer = gathered_scores.new_zeros((routing_map.shape[0], topk))
                    idx_buffer = expert_ids.new_zeros((routing_map.shape[0], topk))
                    per_token_counts = torch.zeros(routing_map.shape[0], dtype=torch.long, device=expert_ids.device)
                    for row, expert_id, score in zip(token_indices, expert_ids, gathered_scores):
                        slot = int(per_token_counts[row].item())
                        if slot < topk:
                            idx_buffer[row, slot] = expert_id
                            score_buffer[row, slot] = score
                            per_token_counts[row] += 1
                    layer_key = f"layer_{int(layer_number):02d}"
                    record = debug_state.setdefault(
                        layer_key,
                        {
                            "token_count": 0,
                            "topk": topk,
                            "selected_score_sum": 0.0,
                            "new_slot_fraction_sum": 0.0,
                            "all_old_count": 0,
                            "mixed_count": 0,
                            "all_new_count": 0,
                            "expert_assignment_counts": torch.zeros(num_experts, dtype=torch.int64),
                        },
                    )
                    _finalize_routing_record(
                        record,
                        idx_buffer.detach().cpu(),
                        score_buffer.detach().cpu(),
                        get_args().source_num_experts,
                    )

                return _hook

            module._analysis_debug_hook_handle = module.register_forward_hook(make_hook(module.layer_number))


def install_lora_router_controls(model, allowed_experts, requested_topk, debug_state):
    args = get_args()
    for module in model.modules():
        router_module = getattr(module, "qv_lora_experts", None)
        layer_number = getattr(module, "layer_number", None)
        if router_module is None or layer_number is None:
            continue
        if not hasattr(router_module, "_analysis_original_forward"):
            router_module._analysis_original_forward = router_module.forward
        num_experts = int(router_module.num_experts)
        allowed = allowed_experts if allowed_experts is not None else list(range(num_experts))
        effective_topk = min(requested_topk or int(router_module.topk), len(allowed))
        if effective_topk <= 0:
            raise ValueError("No allowed experts left after applying LoRA router mask.")

        def controlled_forward(hidden_states, _router=router_module, _allowed=allowed, _topk=effective_topk, _layer=layer_number):
            original_shape = hidden_states.shape[:-1]
            hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            router_input = hidden_flat.to(_router.router_weight.dtype)
            router_logits = F.linear(router_input, _router.router_weight)
            if len(_allowed) != _router.num_experts:
                mask = torch.ones(_router.num_experts, dtype=torch.bool, device=router_logits.device)
                mask[_allowed] = False
                router_logits[..., mask] = torch.finfo(router_logits.dtype).min
            router_probs = torch.softmax(router_logits, dim=-1)
            if _topk == 1:
                expert_scores, expert_idx = torch.max(router_probs, dim=-1)
                expert_scores = expert_scores.unsqueeze(-1)
                expert_idx = expert_idx.unsqueeze(-1)
            else:
                expert_scores, expert_idx = torch.topk(router_probs, k=_topk, dim=-1)
                expert_scores = expert_scores / (expert_scores.sum(dim=-1, keepdim=True) + 1e-20)

            if debug_state is not None:
                layer_key = f"layer_{int(_layer):02d}"
                record = debug_state.setdefault(
                    layer_key,
                    {
                        "token_count": 0,
                        "topk": _topk,
                        "selected_score_sum": 0.0,
                        "new_slot_fraction_sum": 0.0,
                        "all_old_count": 0,
                        "mixed_count": 0,
                        "all_new_count": 0,
                        "expert_assignment_counts": torch.zeros(num_experts, dtype=torch.int64),
                    },
                )
                _finalize_routing_record(
                    record,
                    expert_idx.detach().cpu(),
                    expert_scores.detach().cpu(),
                    args.source_num_experts,
                )

            return _router._compute_grouped_qv_deltas(
                hidden_flat,
                original_shape,
                expert_idx,
                expert_scores,
            )

        router_module.forward = controlled_forward


def reset_ffn_router_controls(model):
    for module in model.modules():
        if not isinstance(module, Router):
            continue
        if hasattr(module, "_analysis_original_gating"):
            module.gating = module._analysis_original_gating
        if hasattr(module, "_analysis_original_topk") and hasattr(module, "topk"):
            module.topk = module._analysis_original_topk
        if hasattr(module, "_analysis_original_config_topk") and hasattr(module, "config") and hasattr(
            module.config, "moe_router_topk"
        ):
            module.config.moe_router_topk = module._analysis_original_config_topk
        if getattr(module, "_analysis_debug_hook_handle", None) is not None:
            module._analysis_debug_hook_handle.remove()
            module._analysis_debug_hook_handle = None


def reset_lora_router_controls(model):
    for module in model.modules():
        router_module = getattr(module, "qv_lora_experts", None)
        if router_module is None:
            continue
        if hasattr(router_module, "_analysis_original_forward"):
            router_module.forward = router_module._analysis_original_forward


def reset_router_controls(model, model_kind):
    if model_kind == "ffn":
        reset_ffn_router_controls(model)
    else:
        reset_lora_router_controls(model)


def finalize_debug(debug_state):
    output = {}
    for layer_key, record in sorted(debug_state.items()):
        counts = [int(v) for v in record["expert_assignment_counts"].tolist()]
        token_count = int(record["token_count"])
        topk = int(record["topk"])
        output[layer_key] = {
            "token_count": token_count,
            "topk": topk,
            "expert_assignment_counts": counts,
            "expert_assignment_fractions": [
                count / max(token_count * topk, 1) for count in counts
            ],
            "fraction_all_old_group": record["all_old_count"] / max(token_count, 1),
            "fraction_mixed_old_new_group": record["mixed_count"] / max(token_count, 1),
            "fraction_all_new_group": record["all_new_count"] / max(token_count, 1),
            "mean_new_group_slot_fraction": record["new_slot_fraction_sum"] / max(token_count, 1),
            "mean_selected_router_score": record["selected_score_sum"] / max(token_count, 1),
        }
    return output


def evaluate(model):
    args = get_args()
    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)
    totals = {"token_count": 0, "correct": 0, "nll_sum": 0.0}

    show_progress = torch.distributed.get_rank() == 0
    progress = tqdm(
        range(args.eval_iters),
        desc=args.compare_label,
        dynamic_ncols=True,
        mininterval=5.0,
        disable=not show_progress,
    )

    with torch.no_grad():
        for _ in progress:
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(iterator)
            logits = model(
                tokens,
                position_ids,
                attention_mask,
                labels=None,
                runtime_gather_output=True,
            )
            logits = align_logits(logits.float(), labels)
            log_probs = F.log_softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            token_nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            mask = loss_mask.bool()
            totals["token_count"] += int(mask.sum().item())
            totals["correct"] += int(((preds == labels) & mask).sum().item())
            totals["nll_sum"] += float(token_nll[mask].sum().item())

    mean_nll = totals["nll_sum"] / max(totals["token_count"], 1)
    return {
        "token_count": totals["token_count"],
        "next_token_acc": totals["correct"] / max(totals["token_count"], 1),
        "mean_nll": mean_nll,
        "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
    }


def main():
    initialize_megatron(
        extra_args_provider=add_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "exit_on_missing_checkpoint": True,
            "use_checkpoint_args": True,
        },
    )
    args = get_args()

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    allowed_experts = parse_allowed_experts(args.allowed_experts)
    debug_state = {} if args.debug_router_json else None

    reset_router_controls(model, args.model_kind)
    if args.model_kind == "ffn":
        install_ffn_router_controls(model, allowed_experts, args.topk_override, debug_state)
    else:
        install_lora_router_controls(model, allowed_experts, args.topk_override, debug_state)

    metrics = evaluate(model)
    reset_router_controls(model, args.model_kind)
    result = {
        "label": args.compare_label,
        "model_kind": args.model_kind,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "allowed_experts": allowed_experts,
        "topk_override": args.topk_override,
        "source_num_experts": args.source_num_experts,
        "data_path": args.data_path,
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "consumed_samples": args.consumed_samples,
        **metrics,
    }

    if torch.distributed.get_rank() == 0:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if args.debug_router_json:
            debug_path = Path(args.debug_router_json)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(json.dumps(finalize_debug(debug_state), indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
