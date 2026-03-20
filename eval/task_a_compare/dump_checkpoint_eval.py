#!/usr/bin/env python3
import json
import os
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
from megatron.training import get_args, get_model, get_tokenizer, print_rank_0
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_compare_args(parser):
    group = parser.add_argument_group(title="task-a-compare")
    group.add_argument("--dump-dir", type=str, required=True, help="Directory to store dumps.")
    group.add_argument("--compare-label", type=str, required=True, help="Short label for this checkpoint.")
    group.add_argument(
        "--dump-shard-size",
        type=int,
        default=1,
        help="Number of eval batches to combine into one temporary shard file.",
    )
    group.add_argument(
        "--compact-dump",
        action="store_true",
        help="Store only compact temporary tensors needed for aggregate comparison.",
    )
    group.add_argument(
        "--dataset-split",
        type=str,
        default="0,1,0",
        help="Megatron split string used to build the dataset view, e.g. 95,5,0.",
    )
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="valid",
        choices=("train", "valid", "test"),
        help="Which split from --dataset-split to evaluate.",
    )
    group.add_argument(
        "--consumed-samples",
        type=int,
        default=0,
        help="How many samples to skip from the chosen split before evaluation.",
    )
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


def register_router_hooks(model, routing_state):
    hooks = []

    def save_hook(layer_number):
        def _hook(_module, _inputs, outputs):
            probs, _routing_map = outputs
            _values, indices = torch.topk(probs.detach(), k=get_args().moe_router_topk, dim=-1)
            routing_state[layer_number] = indices.cpu().to(torch.uint8)

        return _hook

    for module in model.modules():
        if isinstance(module, Router):
            hooks.append(module.register_forward_hook(save_hook(module.layer_number)))
    return hooks


def save_batch_dump(output_root, compare_label, iteration_idx, rank, batch, routing_state):
    tokens, labels, loss_mask, logits = batch
    logits = align_logits(logits.float(), labels)
    log_probs = F.log_softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    token_nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    top1_logprob = log_probs.max(dim=-1).values

    pred_dir = output_root / compare_label / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": tokens.cpu(),
            "labels": labels.cpu(),
            "loss_mask": loss_mask.cpu(),
            "preds": preds.cpu(),
            "correct": (preds == labels).cpu(),
            "token_nll": token_nll.cpu(),
            "top1_logprob": top1_logprob.cpu(),
        },
        pred_dir / f"{iteration_idx:05d}-{rank}.pt",
    )

    for layer_number, indices in routing_state.items():
        layer_dir = output_root / compare_label / "routing" / f"layer_{layer_number:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "topk_indices": indices,
            },
            layer_dir / f"{iteration_idx:05d}-{rank}.pt",
        )


def update_compact_buffers(prediction_buffer, routing_buffer, batch, routing_state):
    _tokens, labels, loss_mask, logits = batch
    logits = align_logits(logits.float(), labels)
    log_probs = F.log_softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    token_nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = loss_mask.bool().cpu()

    prediction_buffer["loss_mask"].append(mask)
    prediction_buffer["correct"].append((preds == labels).cpu())
    prediction_buffer["nll_sum"].append(float(token_nll[loss_mask.bool()].sum().item()))
    prediction_buffer["token_count"].append(int(mask.sum().item()))

    for layer_number, indices in routing_state.items():
        routing_buffer.setdefault(layer_number, []).append(indices)


def flush_compact_dump(output_root, compare_label, shard_idx, rank, prediction_buffer, routing_buffer):
    pred_dir = output_root / compare_label / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "loss_mask": torch.cat(prediction_buffer["loss_mask"], dim=0),
            "correct": torch.cat(prediction_buffer["correct"], dim=0),
            "nll_sum": float(sum(prediction_buffer["nll_sum"])),
            "token_count": int(sum(prediction_buffer["token_count"])),
        },
        pred_dir / f"{shard_idx:05d}-{rank}.pt",
    )

    for layer_number, shard_indices in routing_buffer.items():
        layer_dir = output_root / compare_label / "routing" / f"layer_{layer_number:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "topk_indices": torch.cat(shard_indices, dim=0),
            },
            layer_dir / f"{shard_idx:05d}-{rank}.pt",
        )


def write_metadata(output_root, compare_label):
    args = get_args()
    metadata = {
        "label": compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "num_experts": args.num_experts,
        "moe_router_topk": args.moe_router_topk,
        "data_path": args.data_path,
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "consumed_samples": args.consumed_samples,
        "dump_shard_size": args.dump_shard_size,
        "compact_dump": args.compact_dump,
    }
    metadata_path = output_root / compare_label / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    initialize_megatron(
        extra_args_provider=add_compare_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "exit_on_missing_checkpoint": True,
            "use_checkpoint_args": True,
        },
    )
    args = get_args()

    output_root = Path(args.dump_dir)
    compare_label = args.compare_label

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    routing_state = {}
    hooks = register_router_hooks(model, routing_state)
    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)

    write_metadata(output_root, compare_label)
    rank = torch.distributed.get_rank()
    prediction_buffer = {"loss_mask": [], "correct": [], "nll_sum": [], "token_count": []}
    routing_buffer = {}
    shard_idx = 0

    show_progress = rank == 0 and os.environ.get("SHOW_PROGRESS", "0") == "1"
    progress = tqdm(
        range(args.eval_iters),
        desc=f"{compare_label}",
        dynamic_ncols=True,
        mininterval=5.0,
        disable=not show_progress,
    )

    with torch.no_grad():
        for iteration_idx in progress:
            routing_state.clear()
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(iterator)
            logits = model(
                tokens,
                position_ids,
                attention_mask,
                labels=None,
                runtime_gather_output=True,
            )
            if args.compact_dump:
                update_compact_buffers(
                    prediction_buffer,
                    routing_buffer,
                    (tokens, labels, loss_mask, logits),
                    routing_state,
                )
                if (iteration_idx + 1) % args.dump_shard_size == 0:
                    flush_compact_dump(
                        output_root,
                        compare_label,
                        shard_idx,
                        rank,
                        prediction_buffer,
                        routing_buffer,
                    )
                    prediction_buffer = {"loss_mask": [], "correct": [], "nll_sum": [], "token_count": []}
                    routing_buffer = {}
                    shard_idx += 1
                    if show_progress:
                        progress.set_postfix(shards=shard_idx)
            else:
                save_batch_dump(
                    output_root,
                    compare_label,
                    iteration_idx,
                    rank,
                    (tokens, labels, loss_mask, logits),
                    routing_state,
                )
            if rank == 0 and not show_progress:
                print_rank_0(f"[{compare_label}] dumped eval batch {iteration_idx + 1}/{args.eval_iters}")

    if args.compact_dump and prediction_buffer["loss_mask"]:
        flush_compact_dump(
            output_root,
            compare_label,
            shard_idx,
            rank,
            prediction_buffer,
            routing_buffer,
        )

    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
