#!/usr/bin/env python3
import argparse
import json
import math
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
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
from megatron.core.transformer.shared_router_hybrid import (
    SharedRouterHybridTransformerLayer,
    SharedRoutingContext,
)
from megatron.training import get_args, get_model, get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.legacy.data.data_samplers import build_pretraining_data_loader

from pretrain_gpt import get_batch, is_dataset_built_on_rank, model_provider


def add_args(parser):
    group = parser.add_argument_group(title="shared-router-split-probe")
    group.add_argument("--output-root", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
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
    group.add_argument("--target-eval-tokens", type=int, default=262_144)
    group.add_argument("--max-batches", type=int, default=8)
    group.add_argument("--code-data-path", nargs="+", required=True)
    group.add_argument("--debug-routing", action="store_true")
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
        output.append(item.detach().cpu() if torch.is_tensor(item) else item)
    return tuple(output)


def move_batch_to_device(batch):
    output = []
    for item in batch:
        output.append(item.cuda(non_blocking=True) if torch.is_tensor(item) else item)
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


def align_logits(logits, labels):
    if logits.dim() != 3:
        raise RuntimeError(f"Unexpected logits shape {tuple(logits.shape)}")
    if logits.shape[0] == labels.shape[0] and logits.shape[1] == labels.shape[1]:
        return logits
    if logits.shape[0] == labels.shape[1] and logits.shape[1] == labels.shape[0]:
        return logits.permute(1, 0, 2).contiguous()
    raise RuntimeError(f"Could not align logits {tuple(logits.shape)} and labels {tuple(labels.shape)}")


def init_debug_record(num_experts, topk):
    return {
        "token_count": 0,
        "topk": int(topk),
        "selected_score_sum": 0.0,
        "new_slot_fraction_sum": 0.0,
        "all_old_count": 0,
        "mixed_count": 0,
        "all_new_count": 0,
        "expert_assignment_counts": torch.zeros(num_experts, dtype=torch.int64),
    }


def update_routing_record(record, expert_idx, expert_scores, source_num_experts):
    token_count = int(expert_idx.shape[0])
    record["token_count"] += token_count
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
        expert_idx.reshape(-1).detach().cpu(),
        minlength=record["expert_assignment_counts"].shape[0],
    ).to(torch.int64)


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


def record_shared_routing(debug_state, layer_number, branch_name, routing_context, source_num_experts):
    if debug_state is None:
        return
    routing_map = routing_context.routing_map.detach()
    scores = routing_context.scores.detach()
    expert_idx = torch.nonzero(routing_map, as_tuple=False)
    if expert_idx.numel() == 0:
        return
    token_indices = expert_idx[:, 0]
    expert_ids = expert_idx[:, 1]
    topk = int(routing_map.sum(dim=-1).max().item())
    gathered_scores = scores[token_indices, expert_ids]
    idx_buffer = expert_ids.new_zeros((routing_map.shape[0], topk))
    score_buffer = gathered_scores.new_zeros((routing_map.shape[0], topk))
    per_token_counts = torch.zeros(routing_map.shape[0], dtype=torch.long, device=expert_ids.device)
    for row, expert_id, score in zip(token_indices, expert_ids, gathered_scores):
        slot = int(per_token_counts[row].item())
        if slot < topk:
            idx_buffer[row, slot] = expert_id
            score_buffer[row, slot] = score
            per_token_counts[row] += 1
    layer_key = f"layer_{int(layer_number):02d}_{branch_name}"
    record = debug_state.setdefault(
        layer_key,
        init_debug_record(routing_map.shape[1], topk),
    )
    update_routing_record(record, idx_buffer.detach().cpu(), score_buffer.detach().cpu(), source_num_experts)


def reset_split_router_hooks(model):
    for module in model.modules():
        if not isinstance(module, SharedRouterHybridTransformerLayer):
            continue
        if hasattr(module, "_analysis_original_forward"):
            module.forward = module._analysis_original_forward


def install_split_router_hooks(model, debug_state, source_num_experts):
    reset_split_router_hooks(model)

    for module in model.modules():
        if not isinstance(module, SharedRouterHybridTransformerLayer) or not module.is_moe_layer:
            continue
        if not hasattr(module, "_analysis_original_forward"):
            module._analysis_original_forward = module.forward

        def split_forward(
            self,
            hidden_states,
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            rotary_pos_cos=None,
            rotary_pos_sin=None,
            attention_bias=None,
            inference_params=None,
            packed_seq_params=None,
            sequence_len_offset=None,
            _debug_state=debug_state,
            _source_num_experts=source_num_experts,
        ):
            residual = hidden_states
            input_layernorm_output = self.input_layernorm(hidden_states)
            attn_routing_context = self._compute_shared_routing(input_layernorm_output)
            record_shared_routing(
                _debug_state, self.layer_number, "attn", attn_routing_context, _source_num_experts
            )

            attention_output_with_bias = self.self_attention(
                input_layernorm_output,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                routing_context=attn_routing_context,
            )

            with self.bias_dropout_add_exec_handler():
                hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                    attention_output_with_bias, residual, self.hidden_dropout
                )

            residual = hidden_states
            pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)
            attention_output_with_bias = self.cross_attention(
                pre_cross_attn_layernorm_output,
                attention_mask=context_mask,
                key_value_states=context,
                inference_params=inference_params,
            )

            if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
                context = attention_output_with_bias["context"]

            with self.bias_dropout_add_exec_handler():
                hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                    attention_output_with_bias, residual, self.hidden_dropout
                )

            residual = hidden_states
            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)
            ffn_scores, ffn_map = self.shared_expert_router(pre_mlp_layernorm_output)
            ffn_routing_context = SharedRoutingContext(scores=ffn_scores, routing_map=ffn_map)
            record_shared_routing(
                _debug_state, self.layer_number, "ffn", ffn_routing_context, _source_num_experts
            )
            mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output, routing_context=ffn_routing_context)

            with self.bias_dropout_add_exec_handler():
                hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                    mlp_output_with_bias, residual, self.hidden_dropout
                )

            output = hidden_states
            if self.config.external_cuda_graph and self.training:
                return output
            from megatron.core.utils import make_viewless_tensor

            output = make_viewless_tensor(
                inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
            )
            return output, context

        module.forward = types.MethodType(split_forward, module)


def evaluate_cached_batches(model, batches, label):
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

    mean_nll = totals["nll_sum"] / max(totals["token_count"], 1)
    return {
        "token_count": totals["token_count"],
        "next_token_acc": totals["correct"] / max(totals["token_count"], 1),
        "mean_nll": mean_nll,
        "ppl": float(torch.exp(torch.tensor(mean_nll)).item()),
    }


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_bar_plots(results, output_root: Path):
    labels = [row["label"] for row in results]
    acc_values = [row["next_token_acc"] for row in results]
    ppl_values = [row["ppl"] for row in results]

    for metric_name, values, ylabel, filename in (
        ("accuracy", acc_values, "Next-token accuracy", "summary_accuracy.png"),
        ("ppl", ppl_values, "Perplexity", "summary_ppl.png"),
    ):
        plt.figure(figsize=(6.2, 4.8))
        bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"])
        plt.ylabel(ylabel)
        plt.title(f"Shared Router vs Split Router Clone ({metric_name})")
        plt.xticks(rotation=12)
        for bar, value in zip(bars, values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        plt.tight_layout()
        plt.savefig(output_root / filename, dpi=200)
        plt.close()


def main():
    initialize_megatron(
        extra_args_provider=add_args,
        args_defaults={
            "no_load_rng": True,
            "no_load_optim": True,
            "exit_on_missing_checkpoint": True,
            "use_checkpoint_args": True,
            "micro_batch_size": 8,
            "global_batch_size": 8,
            "shared_router_hybrid_model": True,
        },
    )
    args = get_args()
    args.shared_eval_iters = max(
        args.max_batches,
        math.ceil(args.target_eval_tokens / max(args.micro_batch_size * args.seq_length, 1)),
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    assert len(model_list) == 1, "Expected a single GPT model shard."
    model = model_list[0]
    model.eval()

    cached_batches = cache_batches(
        args.code_data_path,
        args.shared_eval_iters,
        desc=f"{args.compare_label}-code-cache",
    )

    shared_metrics = evaluate_cached_batches(model, cached_batches, "shared_router_code_probe")
    shared_payload = {
        "label": "shared_router",
        "compare_label": args.compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.shared_eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "source_num_experts": args.source_num_experts,
        "total_num_experts": args.total_num_experts,
        "data_path": args.code_data_path,
        **shared_metrics,
    }
    write_json(output_root / "shared_router.json", shared_payload)

    split_debug = {} if args.debug_routing else None
    install_split_router_hooks(model, split_debug, args.source_num_experts)
    split_metrics = evaluate_cached_batches(model, cached_batches, "split_router_clone_code_probe")
    reset_split_router_hooks(model)

    split_payload = {
        "label": "split_router_clone",
        "compare_label": args.compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "eval_iters": args.shared_eval_iters,
        "seq_length": args.seq_length,
        "micro_batch_size": args.micro_batch_size,
        "global_batch_size": args.global_batch_size,
        "source_num_experts": args.source_num_experts,
        "total_num_experts": args.total_num_experts,
        "data_path": args.code_data_path,
        **split_metrics,
    }
    write_json(output_root / "split_router_clone.json", split_payload)

    if split_debug is not None:
        write_json(output_root / "split_router_clone_routing.json", finalize_debug(split_debug))

    save_bar_plots([shared_payload, split_payload], output_root)

    manifest = {
        "compare_label": args.compare_label,
        "load": args.load,
        "iteration": args.iteration,
        "shared_eval_iters": args.shared_eval_iters,
        "code_data_path": args.code_data_path,
        "results": {
            "shared_router": shared_metrics,
            "split_router_clone": split_metrics,
        },
    }
    write_json(output_root / "manifest.json", manifest)

    if torch.distributed.get_rank() == 0:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
