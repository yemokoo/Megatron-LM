#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    group = parser.add_argument_group(title="mask-extra-router")
    group.add_argument("--output-json", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--mask-from-expert", type=int, default=None)
    group.add_argument("--debug-router-json", type=str, default=None)
    group.add_argument("--dataset-split", type=str, default="0,1,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="valid",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    return parser


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


def install_router_mask(model, mask_from_expert: int | None):
    if mask_from_expert is None:
        return

    for module in model.modules():
        if not isinstance(module, Router):
            continue
        original_gating = module.gating

        def masked_gating(input_tensor, _original_gating=original_gating, _mask_from=mask_from_expert):
            logits = _original_gating(input_tensor)
            if _mask_from < logits.shape[-1]:
                logits[..., _mask_from:] = torch.finfo(logits.dtype).min
            return logits

        module.gating = masked_gating


def register_router_debug_hooks(model, debug_state):
    hooks = []

    def make_hook(layer_number):
        def _hook(_module, _inputs, outputs):
            _scores, routing_map = outputs
            routing_map = routing_map.detach().to(torch.int64).cpu()
            layer_key = f"layer_{int(layer_number):02d}"
            record = debug_state.setdefault(
                layer_key,
                {
                    "token_count": 0,
                    "assignment_count": 0,
                    "expert_assignment_counts": None,
                },
            )
            record["token_count"] += int(routing_map.shape[0])
            record["assignment_count"] += int(routing_map.sum().item())
            expert_counts = routing_map.sum(dim=0)
            if record["expert_assignment_counts"] is None:
                record["expert_assignment_counts"] = expert_counts
            else:
                record["expert_assignment_counts"] += expert_counts

        return _hook

    for module in model.modules():
        if isinstance(module, Router):
            hooks.append(module.register_forward_hook(make_hook(module.layer_number)))
    return hooks


def finalize_router_debug(debug_state, mask_from_expert: int | None):
    output = {}
    for layer_key, record in sorted(debug_state.items()):
        counts_tensor = record["expert_assignment_counts"]
        counts = counts_tensor.tolist() if counts_tensor is not None else []
        masked_assignments = 0
        if mask_from_expert is not None and counts:
            masked_assignments = int(sum(counts[mask_from_expert:]))
        output[layer_key] = {
            "token_count": int(record["token_count"]),
            "assignment_count": int(record["assignment_count"]),
            "expert_assignment_counts": [int(v) for v in counts],
            "masked_assignment_count": masked_assignments,
        }
    return output


def evaluate(model):
    args = get_args()
    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)
    totals = {
        "token_count": 0,
        "correct": 0,
        "nll_sum": 0.0,
    }

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

    install_router_mask(model, args.mask_from_expert)
    router_debug_state = {}
    hooks = register_router_debug_hooks(model, router_debug_state)
    metrics = evaluate(model)
    for hook in hooks:
        hook.remove()

    result = {
        "label": args.compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "num_experts": args.num_experts,
        "mask_from_expert": args.mask_from_expert,
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
            debug_output = {
                "label": args.compare_label,
                "mask_from_expert": args.mask_from_expert,
                "layers": finalize_router_debug(router_debug_state, args.mask_from_expert),
            }
            debug_path = Path(args.debug_router_json)
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(json.dumps(debug_output, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
