#!/usr/bin/env python3
"""Dump per-token, per-layer top-K routing decisions for the routing-overlap table.

Unlike eval_controlled_expert_inference.py's --debug-router-json (which only
keeps aggregate counts), this records the full ordered expert-index list for
every probe token at every MoE layer, so two checkpoints' routing on the SAME
deterministic probe stream can be compared token-for-token afterwards by
compute_routing_overlap.py (Eq. routing: OA_set, OA_exact, NewSlot).

Determinism this relies on: --dataset-split 100,0,0 --dataset-split-name train
--consumed-samples 0 with the same data-path, seed, seq-length and batch sizes
gives the same token stream on every run, so two checkpoints processed with
identical args produce directly comparable, position-aligned dumps.

Output: one .pt file with
    {"layers": [layer_number, ...],
     "topk": K,
     "num_experts": {layer_number: int, ...},   # this checkpoint's expert count
     "token_count": int,
     "expert_idx": {layer_number: LongTensor[token_count, K]},   # descending-logit order
    }
"""
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.transformer.moe.router import Router
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

from pretrain_gpt import get_batch, model_provider

# Reuse the exact deterministic dataloader construction from the controlled
# expert-inference tool so the token stream matches theirs bit-for-bit.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_controlled_expert_inference import build_eval_dataloader  # noqa: E402


def add_args(parser):
    group = parser.add_argument_group(title="dump-full-routing")
    group.add_argument("--output-pt", type=str, required=True)
    group.add_argument("--compare-label", type=str, required=True)
    group.add_argument("--dataset-split", type=str, default="100,0,0")
    group.add_argument(
        "--dataset-split-name", type=str, default="train",
        choices=("train", "valid", "test"))
    group.add_argument("--consumed-samples", type=int, default=0)
    return parser


def install_recorder(model, store):
    handles = []
    for module in model.modules():
        if not isinstance(module, Router):
            continue
        layer_number = int(module.layer_number)
        num_experts = int(module.num_experts)
        store["num_experts"][layer_number] = num_experts

        def hook(_module, _inputs, outputs, _layer=layer_number):
            expert_scores, routing_map = outputs
            routing_map = routing_map.detach()
            nnz = torch.nonzero(routing_map, as_tuple=False)
            if nnz.numel() == 0:
                return
            token_idx, expert_ids = nnz[:, 0], nnz[:, 1]
            topk = int(routing_map.sum(dim=-1).max().item())
            store["topk"] = topk
            n_tokens = routing_map.shape[0]
            scores = expert_scores.detach()[token_idx, expert_ids]
            idx_buf = expert_ids.new_full((n_tokens, topk), -1)
            score_buf = scores.new_full((n_tokens, topk), float("-inf"))
            filled = torch.zeros(n_tokens, dtype=torch.long,
                                 device=expert_ids.device)
            for row, expert_id, score in zip(token_idx.tolist(),
                                             expert_ids.tolist(),
                                             scores.tolist()):
                slot = int(filled[row].item())
                if slot < topk:
                    idx_buf[row, slot] = expert_id
                    score_buf[row, slot] = score
                    filled[row] += 1
            # Sort by descending score so slot order == mixture-weight rank,
            # matching the paper's \vec{T} definition.
            order = score_buf.argsort(dim=-1, descending=True)
            idx_buf = torch.gather(idx_buf, 1, order)
            store["batches"].setdefault(_layer, []).append(
                idx_buf.to(torch.int32).cpu())

        handles.append(module.register_forward_hook(hook))
    return handles


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
    assert len(model_list) == 1
    model = model_list[0]
    model.eval()

    store = {"num_experts": {}, "topk": None, "batches": {}}
    handles = install_recorder(model, store)

    dataloader = build_eval_dataloader()
    iterator = iter(dataloader)
    show_progress = torch.distributed.get_rank() == 0
    with torch.no_grad():
        for _ in tqdm(range(args.eval_iters), desc=args.compare_label,
                      dynamic_ncols=True, mininterval=5.0,
                      disable=not show_progress):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(iterator)
            model(tokens, position_ids, attention_mask, labels=None,
                  runtime_gather_output=True)

    for handle in handles:
        handle.remove()

    if torch.distributed.get_rank() == 0:
        layers = sorted(store["batches"].keys())
        expert_idx = {}
        token_count = None
        for layer in layers:
            cat = torch.cat(store["batches"][layer], dim=0)
            if token_count is None:
                token_count = cat.shape[0]
            elif cat.shape[0] != token_count:
                raise RuntimeError(
                    f"layer {layer} token count {cat.shape[0]} != {token_count}; "
                    "routing dumps across layers must align to the same probe stream")
            expert_idx[layer] = cat
        payload = {
            "label": args.compare_label,
            "layers": layers,
            "topk": store["topk"],
            "num_experts": store["num_experts"],
            "token_count": token_count,
            "expert_idx": expert_idx,
        }
        output_path = Path(args.output_pt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, output_path)
        print(f"[dump_full_routing] wrote {output_path}: "
              f"{len(layers)} layers x {token_count} tokens, topk={store['topk']}")


if __name__ == "__main__":
    main()
