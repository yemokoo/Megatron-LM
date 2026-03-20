#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import subprocess
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
from megatron.core.transformer.moe.continual_learning_utils import expand_moe_model
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_args(parser):
    group = parser.add_argument_group(title="checkpoint-schedule")
    group.add_argument("--base-load", type=str, required=True, help="Base checkpoint dir used as step 0.")
    group.add_argument("--target-run-root", type=str, required=True, help="Run root containing target checkpoints.")
    group.add_argument("--target-steps", type=str, required=True, help="Comma-separated local steps, e.g. 300,600,...")
    group.add_argument(
        "--source-expand-from-num-experts",
        type=int,
        default=None,
        help="If set, expand the base checkpoint from this expert count to --num-experts before eval.",
    )
    group.add_argument("--eval-load-root", type=str, required=True, help="Scratch dir for per-checkpoint eval loads.")
    group.add_argument("--output-dir", type=str, required=True)
    group.add_argument("--plot-title-prefix", type=str, default="Checkpoint schedule eval")
    group.add_argument("--series-name", type=str, default="checkpoint_schedule")
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="train",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    return parser


def parse_steps(text: str):
    steps = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        steps.append(int(item))
    if not steps:
        raise ValueError("No target steps provided.")
    if sorted(steps) != steps:
        raise ValueError(f"Target steps must be sorted. Got: {steps}")
    return steps


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


def clone_state_dict_to_cpu(model):
    cloned = {}
    for key, value in model.state_dict().items():
        if value is None:
            cloned[key] = None
        else:
            cloned[key] = value.detach().cpu().clone()
    return cloned


def load_endpoint_state(model_list, load_dir: str, source_expand_from_num_experts: int | None):
    args = get_args()
    assert len(model_list) == 1, "Expected a single GPT model shard."

    # Megatron checkpoint loader expects these counters to be zero before each load.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    if source_expand_from_num_experts is None:
        args.load = load_dir
        args.moe_expand_from_num_experts = None
        args.moe_resume_from_num_experts = None
        args.moe_train_new_experts_and_router_only = False
        _ = load_checkpoint(model_list, None, None)
        return clone_state_dict_to_cpu(model_list[0])

    target_model = model_list[0]
    target_num_experts = args.num_experts
    args.num_experts = source_expand_from_num_experts
    source_model_list = get_model(model_provider, wrap_with_ddp=False)
    args.num_experts = target_num_experts

    args.load = load_dir
    args.moe_expand_from_num_experts = None
    args.moe_resume_from_num_experts = None
    args.moe_train_new_experts_and_router_only = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    _ = load_checkpoint(source_model_list, None, None)
    source_model = source_model_list[0]
    expand_moe_model(target_model, source_model, source_expand_from_num_experts)
    return clone_state_dict_to_cpu(target_model)


def interpolate_state_dict(source_state, target_state, alpha: float):
    if source_state.keys() != target_state.keys():
        raise RuntimeError("State dict keys differ between interpolation endpoints.")

    blended = {}
    for key in source_state:
        source_value = source_state[key]
        target_value = target_state[key]
        if source_value is None or target_value is None:
            if source_value is None and target_value is None:
                blended[key] = None
            elif source_value is None:
                blended[key] = target_value.clone()
            else:
                blended[key] = source_value.clone()
            continue
        if source_value.shape != target_value.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: source {tuple(source_value.shape)} vs target {tuple(target_value.shape)}"
            )
        if source_value.dtype != target_value.dtype:
            raise RuntimeError(
                f"Dtype mismatch for {key}: source {source_value.dtype} vs target {target_value.dtype}"
            )

        if torch.is_floating_point(source_value):
            blended[key] = torch.lerp(source_value.float(), target_value.float(), alpha).to(source_value.dtype)
        else:
            if torch.equal(source_value, target_value):
                blended[key] = source_value.clone()
            elif alpha <= 0.5:
                blended[key] = source_value.clone()
            else:
                blended[key] = target_value.clone()
    return blended


def evaluate_current_state(model, eval_iters: int):
    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)
    total_correct = 0
    total_nll = 0.0
    total_tokens = 0

    show_progress = torch.distributed.get_rank() == 0 and os.environ.get("SHOW_PROGRESS", "0") == "1"
    progress = tqdm(
        range(eval_iters),
        desc="checkpoint-schedule-eval",
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
            mask = loss_mask.bool()
            token_nll = -torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

            total_correct += int(((preds == labels) & mask).sum().item())
            total_nll += float(token_nll[mask].sum().item())
            total_tokens += int(mask.sum().item())

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(mean_nll)
    accuracy = total_correct / max(total_tokens, 1)
    return {
        "token_count": total_tokens,
        "accuracy": accuracy,
        "mean_nll": mean_nll,
        "ppl": ppl,
    }


def prepare_eval_load_dir(source_dir: Path, requested_iteration: int, eval_load_root: Path, tag: str):
    latest_path = source_dir / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Missing {latest_path}")

    checkpoint_dir = source_dir / f"iter_{requested_iteration:07d}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint dir {checkpoint_dir}")

    eval_load_root.mkdir(parents=True, exist_ok=True)
    load_dir = eval_load_root / f"{tag}-minimal-iter-{requested_iteration:07d}"
    load_checkpoint_dir = load_dir / f"iter_{requested_iteration:07d}"
    load_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rsync",
        "-rlptD",
        "--delete",
        f"{checkpoint_dir}/",
        f"{load_checkpoint_dir}/",
    ]
    subprocess.run(cmd, check=True)
    (load_dir / "latest_checkpointed_iteration.txt").write_text(f"{requested_iteration}\n", encoding="utf-8")
    return load_dir


def write_csv(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "local_step",
        "point_type",
        "source_step",
        "target_step",
        "token_count",
        "accuracy",
        "mean_nll",
        "ppl",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_json(results, path: Path, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8")


def write_metric_svg(results, x_key: str, metric_key: str, x_label: str, y_label: str, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(row[x_key]) for row in results]
    ys = [float(row[metric_key]) for row in results]

    width, height = 1000, 560
    left, right, top, bottom = 80, 30, 50, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x += 1.0
    if max_y == min_y:
        max_y += 1.0

    def map_x(x):
        return left + (x - min_x) / (max_x - min_x) * plot_w

    def map_y(y):
        return top + plot_h - (y - min_y) / (max_y - min_y) * plot_h

    polyline = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(xs, ys))
    points = []
    for row in results:
        x = float(row[x_key])
        y = float(row[metric_key])
        fill = "#cc6b00" if row["point_type"] == "midpoint" else "#1f77b4"
        points.append(f'<circle cx="{map_x(x):.2f}" cy="{map_y(y):.2f}" r="3.5" fill="{fill}"/>')

    grid_lines = []
    for i in range(6):
        y = top + i * plot_h / 5
        value = max_y - i * (max_y - min_y) / 5
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#d9d9d9" stroke-width="1"/>'
        )
        grid_lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#444">{value:.4f}</text>'
        )

    x_ticks = []
    for x in sorted(set(xs)):
        x_pos = map_x(x)
        x_ticks.append(
            f'<line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top + plot_h}" stroke="#eeeeee" stroke-width="1"/>'
        )
        x_ticks.append(
            f'<text x="{x_pos:.2f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="12" fill="#444">{x:.0f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2:.0f}" y="28" text-anchor="middle" font-size="22" font-family="sans-serif" fill="#111">{html.escape(title)}</text>
{''.join(grid_lines)}
{''.join(x_ticks)}
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#888" stroke-width="1.2"/>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{polyline}"/>
{''.join(points)}
<text x="{width/2:.0f}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222">{html.escape(x_label)}</text>
<text x="22" y="{height/2:.0f}" text-anchor="middle" font-size="14" font-family="sans-serif" fill="#222" transform="rotate(-90 22 {height/2:.0f})">{html.escape(y_label)}</text>
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def main():
    initialize_megatron(
        extra_args_provider=add_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "exit_on_missing_checkpoint": True,
            "use_checkpoint_args": False,
        },
    )
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_load_root = Path(args.eval_load_root)
    target_steps = parse_steps(args.target_steps)

    model_list = get_model(model_provider, wrap_with_ddp=False)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    base_state = load_endpoint_state(model_list, args.base_load, args.source_expand_from_num_experts)
    results = []

    def evaluate_state(local_step: int, point_type: str, source_step: int, target_step: int, state_dict):
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        metrics = evaluate_current_state(model, args.eval_iters)
        row = {
            "local_step": local_step,
            "point_type": point_type,
            "source_step": source_step,
            "target_step": target_step,
            **metrics,
        }
        results.append(row)
        if torch.distributed.get_rank() == 0:
            print(
                f"step={local_step} kind={point_type} token_count={metrics['token_count']} "
                f"accuracy={metrics['accuracy']:.6f} ppl={metrics['ppl']:.6f}",
                flush=True,
            )

    evaluate_state(local_step=0, point_type="checkpoint", source_step=0, target_step=0, state_dict=base_state)
    previous_step = 0
    previous_state = base_state

    for step in target_steps:
        load_dir = prepare_eval_load_dir(Path(args.target_run_root), step, eval_load_root, args.series_name)
        checkpoint_state = load_endpoint_state(model_list, str(load_dir), None)

        midpoint = (previous_step + step) // 2
        if midpoint not in (previous_step, step):
            midpoint_state = interpolate_state_dict(previous_state, checkpoint_state, 0.5)
            evaluate_state(
                local_step=midpoint,
                point_type="midpoint",
                source_step=previous_step,
                target_step=step,
                state_dict=midpoint_state,
            )

        evaluate_state(
            local_step=step,
            point_type="checkpoint",
            source_step=step,
            target_step=step,
            state_dict=checkpoint_state,
        )
        previous_step = step
        previous_state = checkpoint_state

    results.sort(key=lambda row: row["local_step"])

    if torch.distributed.get_rank() == 0:
        metadata = {
            "base_load": args.base_load,
            "target_run_root": args.target_run_root,
            "target_steps": target_steps,
            "source_expand_from_num_experts": args.source_expand_from_num_experts,
            "eval_iters": args.eval_iters,
            "micro_batch_size": args.micro_batch_size,
            "global_batch_size": args.global_batch_size,
            "num_experts": args.num_experts,
            "data_path": args.data_path,
            "dataset_split": args.dataset_split,
            "dataset_split_name": args.dataset_split_name,
            "consumed_samples": args.consumed_samples,
        }
        write_csv(results, output_dir / "metrics.csv")
        write_json(results, output_dir / "metrics.json", metadata)
        write_metric_svg(
            results,
            "local_step",
            "accuracy",
            "Local continual-learning step",
            "Next-token accuracy",
            f"{args.plot_title_prefix}: next-token accuracy",
            output_dir / "next_token_accuracy.svg",
        )
        write_metric_svg(
            results,
            "local_step",
            "ppl",
            "Local continual-learning step",
            "Perplexity",
            f"{args.plot_title_prefix}: perplexity",
            output_dir / "ppl.svg",
        )


if __name__ == "__main__":
    main()
