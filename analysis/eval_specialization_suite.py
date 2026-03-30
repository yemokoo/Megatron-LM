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
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider

from eval_controlled_expert_inference import (
    align_logits,
    finalize_debug,
    install_ffn_router_controls,
    install_lora_router_controls,
    reset_router_controls,
)
from eval_expert_output_similarity import (
    choose_plot_layers,
    collect_ffn_modules,
    collect_lora_modules,
    compute_ffn_outputs,
    compute_lora_outputs,
    install_hidden_collection_hooks,
    mean_tokenwise_cosine_similarity_matrix,
    remove_hooks,
    save_heatmaps,
    summarize_similarity,
)
from plot_controlled_expert_inference import (
    load_results,
    parse_group_masks,
    parse_top1_vs_top2,
    plot_grouped_bars,
)


def add_args(parser):
    group = parser.add_argument_group(title="specialization-suite")
    group.add_argument("--output-root", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--model-kind", type=str, required=True, choices=("ffn", "lora"))
    group.add_argument("--source-num-experts", type=int, default=4)
    group.add_argument("--total-num-experts", type=int, default=7)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    group.add_argument("--target-eval-tokens", type=int, default=2_000_000)
    group.add_argument("--max-batches", type=int, default=64)
    group.add_argument("--max-tokens-per-layer", type=int, default=65536)
    group.add_argument("--plot-layers", type=str, default="")
    group.add_argument("--wiki-data-path", nargs="+", required=True)
    group.add_argument("--code-data-path", nargs="+", required=True)
    return parser


def build_eval_dataloader_for_path(data_path, num_batches):
    args = get_args()
    split_name_to_index = {"train": 0, "valid": 1, "test": 2}
    split_idx = split_name_to_index[args.dataset_split_name]
    requested_samples = args.consumed_samples + num_batches * args.global_batch_size
    config = GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(data_path),
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


def clone_batch_to_cpu(batch):
    output = []
    for item in batch:
        if torch.is_tensor(item):
            output.append(item.detach().cpu())
        else:
            output.append(item)
    return tuple(output)


def move_batch_to_device(batch):
    output = []
    for item in batch:
        if torch.is_tensor(item):
            output.append(item.cuda(non_blocking=True))
        else:
            output.append(item)
    return tuple(output)


def cache_batches(data_path, num_batches, desc):
    dataloader = build_eval_dataloader_for_path(data_path, num_batches)
    iterator = iter(dataloader)
    cached = []
    progress = tqdm(
        range(num_batches),
        desc=desc,
        dynamic_ncols=True,
        mininterval=5.0,
        disable=torch.distributed.get_rank() != 0,
    )
    for _ in progress:
        cached.append(clone_batch_to_cpu(get_batch(iterator)))
    return cached


def apply_router_controls(model, model_kind, allowed_experts, requested_topk, debug_state):
    reset_router_controls(model, model_kind)
    if model_kind == "ffn":
        install_ffn_router_controls(model, allowed_experts, requested_topk, debug_state)
    else:
        install_lora_router_controls(model, allowed_experts, requested_topk, debug_state)


def evaluate_cached_batches(
    model,
    batches,
    model_kind,
    label,
    allowed_experts,
    requested_topk,
    collect_hidden,
    max_tokens_per_layer,
):
    debug_state = {}
    hidden_store = {} if collect_hidden else None
    hooks = []

    apply_router_controls(model, model_kind, allowed_experts, requested_topk, debug_state)
    if collect_hidden:
        hooks = install_hidden_collection_hooks(model, model_kind, hidden_store, max_tokens_per_layer)

    totals = {"token_count": 0, "correct": 0, "nll_sum": 0.0}
    progress = tqdm(
        batches,
        desc=label,
        dynamic_ncols=True,
        mininterval=5.0,
        disable=torch.distributed.get_rank() != 0,
    )
    with torch.no_grad():
        for batch in progress:
            tokens, labels, loss_mask, attention_mask, position_ids = move_batch_to_device(batch)
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

    remove_hooks(hooks)
    reset_router_controls(model, model_kind)

    mean_nll = totals["nll_sum"] / max(totals["token_count"], 1)
    metrics = {
        "token_count": totals["token_count"],
        "next_token_acc": totals["correct"] / max(totals["token_count"], 1),
        "mean_nll": mean_nll,
        "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
    }
    return metrics, finalize_debug(debug_state), hidden_store


def compute_output_similarity_results(model, model_kind, source_num_experts, hidden_store):
    layer_results = []
    if model_kind == "ffn":
        modules = collect_ffn_modules(model)
        for layer_key, hidden in sorted(hidden_store.items()):
            experts_module = modules.get(layer_key, {}).get("experts")
            if experts_module is None:
                continue
            outputs = compute_ffn_outputs(experts_module, hidden)
            cosine = mean_tokenwise_cosine_similarity_matrix(outputs).cpu()
            layer_results.append(
                {
                    "layer": layer_key,
                    "num_tokens": int(hidden.shape[0]),
                    "num_experts": int(outputs.shape[0]),
                    "output_dim": int(outputs.shape[-1]),
                    "mean_tokenwise_cosine_similarity": [[float(v) for v in row] for row in cosine.tolist()],
                    "summary": summarize_similarity(cosine, source_num_experts),
                }
            )
    else:
        modules = collect_lora_modules(model)
        for layer_key, hidden in sorted(hidden_store.items()):
            router_module = modules.get(layer_key)
            if router_module is None:
                continue
            outputs = compute_lora_outputs(router_module, hidden)
            cosine = mean_tokenwise_cosine_similarity_matrix(outputs).cpu()
            layer_results.append(
                {
                    "layer": layer_key,
                    "num_tokens": int(hidden.shape[0]),
                    "num_experts": int(outputs.shape[0]),
                    "output_dim": int(outputs.shape[-1]),
                    "mean_tokenwise_cosine_similarity": [[float(v) for v in row] for row in cosine.tolist()],
                    "summary": summarize_similarity(cosine, source_num_experts),
                }
            )
    return layer_results


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_controlled_result(output_path, label, dataset_name, args, allowed_experts, topk_override, metrics):
    payload = {
        "label": label,
        "model_kind": args.model_kind,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.shared_eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "allowed_experts": allowed_experts,
        "topk_override": topk_override,
        "source_num_experts": args.source_num_experts,
        "data_path": getattr(args, f"{dataset_name}_data_path"),
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "consumed_samples": args.consumed_samples,
        **metrics,
    }
    write_json(output_path, payload)


def generate_summary_plots(input_dir: Path, mode: str):
    rows = load_results(input_dir)
    if not rows:
        return
    if mode == "top1_vs_top2":
        parsed, datasets, modes = parse_top1_vs_top2(rows)
        prefix = "Top2 vs Top1"
    else:
        parsed, datasets, modes = parse_group_masks(rows)
        prefix = "Expert Group Masks"
    plot_grouped_bars(
        parsed,
        datasets,
        modes,
        "acc",
        "Next-token accuracy",
        f"{prefix}: accuracy",
        input_dir / "summary_accuracy.png",
    )
    plot_grouped_bars(
        parsed,
        datasets,
        modes,
        "ppl",
        "Perplexity",
        f"{prefix}: perplexity",
        input_dir / "summary_ppl.png",
    )


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
    args.shared_eval_iters = max(
        args.max_batches,
        math.ceil(args.target_eval_tokens / max(args.micro_batch_size * args.seq_length, 1)),
    )

    output_root = Path(args.output_root)
    top1_dir = output_root / "top1_vs_top2"
    masks_dir = output_root / "expert_group_masks"
    similarity_dir = output_root / "expert_output_similarity"
    for path in (top1_dir, masks_dir, similarity_dir):
        path.mkdir(parents=True, exist_ok=True)

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    wiki_group = list(range(args.source_num_experts))
    code_group = list(range(args.source_num_experts, args.total_num_experts))
    datasets = {
        "wiki": args.wiki_data_path,
        "code": args.code_data_path,
    }

    manifest = {
        "compare_label": args.compare_label,
        "model_kind": args.model_kind,
        "load": args.load,
        "iteration": args.iteration,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "seq_length": args.seq_length,
        "target_eval_tokens": args.target_eval_tokens,
        "shared_eval_iters": args.shared_eval_iters,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "source_num_experts": args.source_num_experts,
        "total_num_experts": args.total_num_experts,
        "datasets": {},
    }

    for dataset_name, data_path in datasets.items():
        batch_cache = cache_batches(
            data_path,
            args.shared_eval_iters,
            desc=f"{args.compare_label}-{dataset_name}-cache",
        )
        manifest["datasets"][dataset_name] = {
            "data_path": data_path,
            "num_cached_batches": len(batch_cache),
        }

        top2_label = f"{args.model_kind}_{dataset_name}_top2"
        top2_metrics, top2_routing, hidden_store = evaluate_cached_batches(
            model,
            batch_cache,
            args.model_kind,
            top2_label,
            allowed_experts=None,
            requested_topk=2,
            collect_hidden=True,
            max_tokens_per_layer=args.max_tokens_per_layer,
        )
        save_controlled_result(top1_dir / f"{top2_label}.json", top2_label, dataset_name, args, None, 2, top2_metrics)
        write_json(top1_dir / f"{top2_label}_routing.json", top2_routing)

        top1_label = f"{args.model_kind}_{dataset_name}_top1"
        top1_metrics, top1_routing, _ = evaluate_cached_batches(
            model,
            batch_cache,
            args.model_kind,
            top1_label,
            allowed_experts=None,
            requested_topk=1,
            collect_hidden=False,
            max_tokens_per_layer=args.max_tokens_per_layer,
        )
        save_controlled_result(top1_dir / f"{top1_label}.json", top1_label, dataset_name, args, None, 1, top1_metrics)
        write_json(top1_dir / f"{top1_label}_routing.json", top1_routing)

        for mode_name, allowed_experts in (
            ("unrestricted", None),
            ("wiki_only", wiki_group),
            ("code_only", code_group),
        ):
            label = f"{args.model_kind}_{dataset_name}_{mode_name}"
            metrics, routing, _ = evaluate_cached_batches(
                model,
                batch_cache,
                args.model_kind,
                label,
                allowed_experts=allowed_experts,
                requested_topk=2,
                collect_hidden=False,
                max_tokens_per_layer=args.max_tokens_per_layer,
            )
            save_controlled_result(
                masks_dir / f"{label}.json",
                label,
                dataset_name,
                args,
                allowed_experts,
                2,
                metrics,
            )
            write_json(masks_dir / f"{label}_routing.json", routing)

        layer_results = compute_output_similarity_results(model, args.model_kind, args.source_num_experts, hidden_store)
        plot_layers = choose_plot_layers(layer_results, args.plot_layers)
        similarity_payload = {
            "label": f"{args.model_kind}_{dataset_name}_expert_output_similarity",
            "model_kind": args.model_kind,
            "load": args.load,
            "iteration": args.iteration,
            "source_num_experts": args.source_num_experts,
            "max_batches": args.shared_eval_iters,
            "max_tokens_per_layer": args.max_tokens_per_layer,
            "similarity_type": "mean_tokenwise_cosine",
            "plot_layers": plot_layers,
            "layer_results": layer_results,
        }
        dataset_similarity_dir = similarity_dir / dataset_name
        dataset_similarity_dir.mkdir(parents=True, exist_ok=True)
        write_json(dataset_similarity_dir / "expert_output_similarity.json", similarity_payload)
        save_heatmaps(
            layer_results,
            dataset_similarity_dir,
            f"Expert output cosine similarity: {args.model_kind}_{dataset_name}",
            plot_layers,
        )

    generate_summary_plots(top1_dir, "top1_vs_top2")
    generate_summary_plots(masks_dir, "group_masks")
    write_json(output_root / "manifest.json", manifest)

    if torch.distributed.get_rank() == 0:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
