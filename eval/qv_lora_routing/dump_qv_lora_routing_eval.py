#!/usr/bin/env python3
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
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider
from megatron.core.transformer.qv_lora_attention import QVLoraSelfAttention


def add_args(parser):
    group = parser.add_argument_group(title="qv-lora-routing")
    group.add_argument("--output-json", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--dataset-split", type=str, default="0,1,0")
    group.add_argument(
        "--dataset-split-name",
        type=str,
        default="valid",
        choices=("train", "valid", "test"),
    )
    group.add_argument("--consumed-samples", type=int, default=0)
    group.add_argument("--source-num-experts", type=int, default=4)
    return parser


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


def register_qv_router_hooks(model, routing_state):
    hooks = []

    def make_hook(layer_number, router_module):
        def _hook(_module, inputs, _outputs):
            hidden_states = inputs[0].detach()
            hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1]).to(router_module.router_weight.dtype)
            router_logits = F.linear(hidden_flat, router_module.router_weight)
            router_probs = torch.softmax(router_logits, dim=-1)
            if router_module.topk == 1:
                expert_scores, expert_idx = torch.max(router_probs, dim=-1)
                expert_scores = expert_scores.unsqueeze(-1)
                expert_idx = expert_idx.unsqueeze(-1)
            else:
                expert_scores, expert_idx = torch.topk(router_probs, k=router_module.topk, dim=-1)
                expert_scores = expert_scores / (expert_scores.sum(dim=-1, keepdim=True) + 1e-20)

            layer_key = f"layer_{int(layer_number):02d}"
            record = routing_state.setdefault(
                layer_key,
                {
                    "token_count": 0,
                    "topk": int(router_module.topk),
                    "selected_score_sum": 0.0,
                    "new_slot_fraction_sum": 0.0,
                    "all_old_count": 0,
                    "mixed_count": 0,
                    "all_new_count": 0,
                    "expert_assignment_counts": torch.zeros(
                        router_module.num_experts,
                        dtype=torch.int64,
                    ),
                },
            )
            token_count = int(expert_idx.shape[0])
            record["token_count"] += token_count
            record["selected_score_sum"] += float(expert_scores.float().sum().item())
            is_new = expert_idx >= int(get_args().source_num_experts)
            all_old_mask = (~is_new).all(dim=-1)
            all_new_mask = is_new.all(dim=-1)
            mixed_mask = ~(all_old_mask | all_new_mask)
            record["all_old_count"] += int(all_old_mask.sum().item())
            record["mixed_count"] += int(mixed_mask.sum().item())
            record["all_new_count"] += int(all_new_mask.sum().item())
            record["new_slot_fraction_sum"] += float(is_new.float().mean(dim=-1).sum().item())
            record["expert_assignment_counts"] += torch.bincount(
                expert_idx.reshape(-1).cpu(), minlength=router_module.num_experts
            ).to(torch.int64)

        return _hook

    for _name, module in model.named_modules():
        if not isinstance(module, QVLoraSelfAttention):
            continue
        if module.qv_lora_experts is None:
            continue
        hooks.append(
            module.qv_lora_experts.register_forward_hook(
                make_hook(module.layer_number, module.qv_lora_experts)
            )
        )
    return hooks


def evaluate_routing(model):
    args = get_args()
    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)
    routing_state = {}
    hooks = register_qv_router_hooks(model, routing_state)

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
            tokens, _labels, _loss_mask, attention_mask, position_ids = get_batch(iterator)
            _ = model(
                tokens,
                position_ids,
                attention_mask,
                labels=None,
                runtime_gather_output=False,
            )

    for hook in hooks:
        hook.remove()
    return routing_state


def finalize_summary(routing_state):
    args = get_args()
    summary = {}
    for layer_key, record in sorted(routing_state.items()):
        counts_tensor = record["expert_assignment_counts"]
        counts = [int(v) for v in counts_tensor.tolist()]
        token_count = int(record["token_count"])
        source_num_experts = int(args.source_num_experts)
        old_count = int(sum(counts[:source_num_experts]))
        new_count = int(sum(counts[source_num_experts:]))
        summary[layer_key] = {
            "token_count": token_count,
            "topk": int(record.get("topk", 1)),
            "num_experts": len(counts),
            "expert_assignment_counts": counts,
            "expert_assignment_fractions": [
                (count / max(token_count * int(record.get("topk", 1)), 1)) for count in counts
            ],
            "fraction_all_old_group": record["all_old_count"] / max(token_count, 1),
            "fraction_mixed_old_new_group": record["mixed_count"] / max(token_count, 1),
            "fraction_all_new_group": record["all_new_count"] / max(token_count, 1),
            "fraction_using_new_expert": (
                (record["mixed_count"] + record["all_new_count"]) / max(token_count, 1)
            ),
            "mean_new_group_slot_fraction": record["new_slot_fraction_sum"] / max(token_count, 1),
            "mean_selected_router_score": record["selected_score_sum"] / max(token_count, 1),
        }
    return summary


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

    routing_state = evaluate_routing(model)
    result = {
        "label": args.compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "data_path": args.data_path,
        "dataset_split": args.dataset_split,
        "dataset_split_name": args.dataset_split_name,
        "consumed_samples": args.consumed_samples,
        "source_num_experts": args.source_num_experts,
        "routing_summary": finalize_summary(routing_state),
    }

    if torch.distributed.get_rank() == 0:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
