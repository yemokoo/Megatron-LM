#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import sys

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
    group = parser.add_argument_group(title="router-logit-stats")
    group.add_argument("--output-json", required=True)
    group.add_argument("--compare-label", required=True)
    group.add_argument("--old-expert-count", type=int, default=4)
    group.add_argument("--hist-min", type=float, default=-20.0)
    group.add_argument("--hist-max", type=float, default=20.0)
    group.add_argument("--hist-bins", type=int, default=80)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
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


def new_stat_record(num_experts: int, hist_bins: int):
    return {
        "count": torch.zeros(num_experts, dtype=torch.float64),
        "sum": torch.zeros(num_experts, dtype=torch.float64),
        "sumsq": torch.zeros(num_experts, dtype=torch.float64),
        "min": torch.full((num_experts,), float("inf"), dtype=torch.float64),
        "max": torch.full((num_experts,), float("-inf"), dtype=torch.float64),
        "hist": torch.zeros(num_experts, hist_bins, dtype=torch.float64),
    }


def install_router_logit_hooks(model, stats_state):
    args = get_args()

    for module in model.modules():
        if not isinstance(module, Router):
            continue

        original_gating = module.gating
        layer_number = module.layer_number
        if layer_number is None:
            layer_number = len(stats_state) + 1
        layer_key = f"layer_{int(layer_number):02d}"

        def wrapped_gating(
            input_tensor,
            _original_gating=original_gating,
            _layer_key=layer_key,
        ):
            logits = _original_gating(input_tensor)
            logits_fp32 = logits.detach().float().reshape(-1, logits.shape[-1])
            record = stats_state.setdefault(
                _layer_key,
                new_stat_record(logits_fp32.shape[-1], args.hist_bins),
            )

            record["count"] += logits_fp32.shape[0]
            record["sum"] += logits_fp32.sum(dim=0).cpu().to(torch.float64)
            record["sumsq"] += (logits_fp32 * logits_fp32).sum(dim=0).cpu().to(torch.float64)
            record["min"] = torch.minimum(record["min"], logits_fp32.amin(dim=0).cpu().to(torch.float64))
            record["max"] = torch.maximum(record["max"], logits_fp32.amax(dim=0).cpu().to(torch.float64))

            clipped = logits_fp32.clamp(min=args.hist_min, max=args.hist_max)
            for expert_idx in range(clipped.shape[-1]):
                hist = torch.histc(
                    clipped[:, expert_idx],
                    bins=args.hist_bins,
                    min=args.hist_min,
                    max=args.hist_max,
                )
                record["hist"][expert_idx] += hist.cpu().to(torch.float64)

            return logits

        module.gating = wrapped_gating


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
        "ppl": float(math.exp(mean_nll)),
    }


def hist_quantile(hist_counts: torch.Tensor, hist_min: float, hist_max: float, quantile: float):
    total = float(hist_counts.sum().item())
    if total <= 0:
        return None
    cumulative = torch.cumsum(hist_counts, dim=0)
    target = quantile * total
    idx = int(torch.searchsorted(cumulative, torch.tensor(target, dtype=cumulative.dtype)).item())
    idx = max(0, min(idx, hist_counts.shape[0] - 1))
    bin_width = (hist_max - hist_min) / hist_counts.shape[0]
    return hist_min + (idx + 0.5) * bin_width


def finalize_stats(raw_stats, old_expert_count: int, hist_min: float, hist_max: float):
    finalized = {}
    overall_old_hist = None
    overall_new_hist = None
    overall_old_sum = 0.0
    overall_new_sum = 0.0
    overall_old_count = 0.0
    overall_new_count = 0.0

    for layer_key in sorted(raw_stats.keys()):
        record = raw_stats[layer_key]
        experts = []
        old_hist = record["hist"][:old_expert_count].sum(dim=0)
        new_hist = record["hist"][old_expert_count:].sum(dim=0)
        overall_old_hist = old_hist.clone() if overall_old_hist is None else overall_old_hist + old_hist
        overall_new_hist = new_hist.clone() if overall_new_hist is None else overall_new_hist + new_hist

        for expert_idx in range(record["count"].shape[0]):
            count = float(record["count"][expert_idx].item())
            mean = float(record["sum"][expert_idx].item() / max(count, 1.0))
            variance = float(record["sumsq"][expert_idx].item() / max(count, 1.0) - mean * mean)
            variance = max(variance, 0.0)
            hist = record["hist"][expert_idx]
            experts.append(
                {
                    "expert_idx": expert_idx,
                    "group": "old" if expert_idx < old_expert_count else "new",
                    "count": int(count),
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": float(record["min"][expert_idx].item()),
                    "max": float(record["max"][expert_idx].item()),
                    "p50_approx": hist_quantile(hist, hist_min, hist_max, 0.50),
                    "p95_approx": hist_quantile(hist, hist_min, hist_max, 0.95),
                    "hist_counts": [float(v) for v in hist.tolist()],
                }
            )
            if expert_idx < old_expert_count:
                overall_old_sum += float(record["sum"][expert_idx].item())
                overall_old_count += count
            else:
                overall_new_sum += float(record["sum"][expert_idx].item())
                overall_new_count += count

        finalized[layer_key] = {
            "experts": experts,
            "group_histograms": {
                "old": [float(v) for v in old_hist.tolist()],
                "new": [float(v) for v in new_hist.tolist()],
            },
        }

    overall = {
        "old_mean": overall_old_sum / max(overall_old_count, 1.0),
        "new_mean": overall_new_sum / max(overall_new_count, 1.0),
        "old_p95_approx": hist_quantile(overall_old_hist, hist_min, hist_max, 0.95) if overall_old_hist is not None else None,
        "new_p95_approx": hist_quantile(overall_new_hist, hist_min, hist_max, 0.95) if overall_new_hist is not None else None,
        "old_histogram": [float(v) for v in overall_old_hist.tolist()] if overall_old_hist is not None else [],
        "new_histogram": [float(v) for v in overall_new_hist.tolist()] if overall_new_hist is not None else [],
    }
    return finalized, overall


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

    raw_stats = {}
    install_router_logit_hooks(model, raw_stats)
    metrics = evaluate(model)
    layers, overall = finalize_stats(raw_stats, args.old_expert_count, args.hist_min, args.hist_max)

    payload = {
        "model_label": args.compare_label,
        "load": args.load,
        "histogram": {
            "min": args.hist_min,
            "max": args.hist_max,
            "bins": args.hist_bins,
        },
        "metrics": metrics,
        "layers": layers,
        "overall": overall,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
