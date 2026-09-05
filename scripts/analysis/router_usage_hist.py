#!/usr/bin/env python3
"""Per-layer routing histograms of a shared-router hybrid checkpoint on a corpus.

For every MoE layer the shared router's top-k choice is recorded per token:
  usage[e]         fraction of top-k slots that went to expert e
  combos[(e1..ek)] count of each sorted top-k expert set

Run it once per corpus (wiki test, code test, a BoS-generated set) with the same
checkpoint, then compare_router_usage.py measures how much of the routing space
that the real old-task data exercises is also exercised by the generated data.
Model construction mirrors plot_wiki_router_softmax_importance.py.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MEGATRON_ROOT = REPO_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core import mpu  # noqa: E402
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder  # noqa: E402
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig  # noqa: E402
from megatron.core.datasets.utils import get_blend_from_list  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.transformer.shared_router_hybrid import capture_shared_router_inputs  # noqa: E402
from megatron.training import get_args, get_tokenizer, print_rank_0  # noqa: E402
from megatron.training.checkpointing import load_checkpoint  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402
from megatron.training.training import _collect_current_shared_routers, _router_logits, get_model  # noqa: E402
from pretrain_gpt import _flatten_layer_hidden, build_pretraining_data_loader, get_batch, is_dataset_built_on_rank, model_provider  # noqa: E402


def add_args(parser):
    g = parser.add_argument_group("router-usage-hist")
    g.add_argument("--ru-data-path", nargs="+", required=True, help="blend list, e.g. 1.0 <prefix>")
    g.add_argument("--ru-eval-iters", type=int, default=16)
    g.add_argument("--ru-max-tokens", type=int, default=500_000)
    g.add_argument("--ru-out", required=True)
    g.add_argument("--ru-label", default="")
    return parser


def build_loader():
    args = get_args()
    config = GPTDatasetConfig(
        random_seed=args.seed, sequence_length=args.seq_length,
        blend=get_blend_from_list(args.ru_data_path), blend_per_split=None, split="100,0,0",
        num_dataset_builder_threads=args.num_dataset_builder_threads, path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files, tokenizer=get_tokenizer(),
        reset_position_ids=args.reset_position_ids, reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss, create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )
    ds, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset, (args.ru_eval_iters * args.global_batch_size, 0, 0), is_dataset_built_on_rank, config).build()
    return build_pretraining_data_loader(ds, 0)


def main():
    initialize_megatron(extra_args_provider=add_args, args_defaults={"tokenizer_type": "GPT2BPETokenizer"})
    args = get_args()
    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)
    iteration, _ = load_checkpoint(model, None, None)
    modules = model if isinstance(model, list) else [model]
    routers = _collect_current_shared_routers(modules)
    if not routers:
        raise RuntimeError("no shared routers found")
    for m in modules:
        m.eval()
    k = int(args.moe_router_topk)
    num_experts = int(next(iter(routers.values())).weight.shape[0])
    device = torch.device("cuda", torch.cuda.current_device())
    usage = {l: torch.zeros(num_experts, dtype=torch.float64, device=device) for l in routers}
    combos = {l: Counter() for l in routers}
    ntok = 0
    loader = iter(build_loader())
    print_rank_0(f"[ru] ckpt iter {iteration}, experts={num_experts}, topk={k}, layers={sorted(routers)}")
    with torch.no_grad():
        for step in range(args.ru_eval_iters):
            try:
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(loader)
            except StopIteration:
                break
            idx = torch.nonzero(loss_mask.reshape(-1).bool(), as_tuple=False).view(-1)
            if args.ru_max_tokens > 0:
                idx = idx[: max(0, args.ru_max_tokens - ntok)]
            if idx.numel() == 0:
                break
            with capture_shared_router_inputs() as captured:
                modules[0](tokens, position_ids, attention_mask, labels=labels)
            for layer_number, hidden in captured:
                l = int(layer_number)
                if l not in routers:
                    continue
                flat = _flatten_layer_hidden(hidden.detach(), labels).index_select(0, idx)
                logits = _router_logits(routers[l], flat).float()
                top = torch.topk(logits, k=min(k, num_experts), dim=-1).indices      # [n, k]
                usage[l] += torch.bincount(top.reshape(-1), minlength=num_experts).double()
                keys = torch.sort(top, dim=-1).values.cpu().numpy()
                combos[l].update(map(tuple, keys.tolist()))
            ntok += int(idx.numel())
            print_rank_0(f"[ru] step {step + 1}/{args.ru_eval_iters} tokens={ntok}")
            if args.ru_max_tokens > 0 and ntok >= args.ru_max_tokens:
                break
    if torch.distributed.get_rank() != 0:
        return
    out = {"label": args.ru_label, "load": args.load, "iteration": int(iteration), "tokens": ntok,
           "num_experts": num_experts, "topk": k, "data_path": args.ru_data_path, "layers": {}}
    for l in sorted(routers):
        u = usage[l] / usage[l].sum().clamp_min(1)
        c = combos[l]
        out["layers"][str(l)] = {
            "usage": u.cpu().tolist(),
            "distinct_combos": len(c),
            "combos": {"|".join(map(str, key)): int(v) for key, v in c.most_common()},
        }
    Path(args.ru_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ru_out).write_text(json.dumps(out))
    print_rank_0(f"[ru] wrote {args.ru_out} (tokens={ntok})")


if __name__ == "__main__":
    main()
