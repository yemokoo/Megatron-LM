#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = PROJECT_ROOT / "Megatron-LM"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.training import get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from pretrain_gpt import model_provider


LAYER_RE = re.compile(r"layers\.(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze FFN MoE expert parameter similarity for one run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--old-expert-count", type=int, default=4)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", default="ffn_param_similarity")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--ffn-hidden-size", type=int)
    parser.add_argument("--moe-ffn-hidden-size", type=int)
    parser.add_argument("--num-experts", type=int)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--moe-layer-freq", default="[0]*1+[1]*8")
    parser.add_argument("--moe-aux-loss-coeff", type=float, default=0.01)
    parser.add_argument("--moe-z-loss-coeff", type=float, default=0.001)
    parser.add_argument("--tokenizer-model", default="EleutherAI/pythia-12b")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_metadata(run_dir: Path):
    metadata_path = run_dir / "logs" / "run_metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_metadata_fallback(args):
    required = {
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
        "num_experts": args.num_experts,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "run_metadata.json is missing, so you must pass: "
            + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        )
    return {
        "run_id": args.run_id or args.run_dir.name,
        "stage": args.stage,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "ffn_hidden_size": args.ffn_hidden_size,
        "moe_ffn_hidden_size": args.moe_ffn_hidden_size,
        "num_experts": args.num_experts,
        "moe_router_topk": args.moe_router_topk,
    }


def detect_num_experts(metadata: dict):
    if "target_num_experts" in metadata:
        return int(metadata["target_num_experts"])
    if "num_experts" in metadata:
        return int(metadata["num_experts"])
    raise KeyError("Could not find num_experts or target_num_experts in metadata.")


def synthesize_megatron_argv(args, metadata, checkpoint_dir: Path):
    num_experts = detect_num_experts(metadata)
    argv = [
        "analyze_ffn_expert_param_similarity.py",
        "--num-layers",
        str(int(metadata["num_layers"])),
        "--hidden-size",
        str(int(metadata["hidden_size"])),
        "--ffn-hidden-size",
        str(int(metadata["ffn_hidden_size"])),
        "--num-attention-heads",
        str(args.num_attention_heads),
        "--swiglu",
        "--max-position-embeddings",
        "2048",
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        "1e-6",
        "--untie-embeddings-and-output-weights",
        "--position-embedding-type",
        "rope",
        "--disable-bias-linear",
        "--moe-ffn-hidden-size",
        str(int(metadata["moe_ffn_hidden_size"])),
        "--num-experts",
        str(num_experts),
        "--moe-router-topk",
        str(int(metadata["moe_router_topk"])),
        "--moe-layer-freq",
        args.moe_layer_freq,
        "--moe-router-dtype",
        "fp32",
        "--moe-router-pre-softmax",
        "--moe-router-score-function",
        "softmax",
        "--moe-aux-loss-coeff",
        str(args.moe_aux_loss_coeff),
        "--moe-z-loss-coeff",
        str(args.moe_z_loss_coeff),
        "--hidden-dropout",
        "0.0",
        "--attention-dropout",
        "0.0",
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        args.tokenizer_model,
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--train-iters",
        "1",
        "--seq-length",
        str(args.seq_length),
        "--save-interval",
        "1000",
        "--eval-interval",
        "1000",
        "--transformer-impl",
        "local",
        "--pipeline-model-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--distributed-timeout-minutes",
        "30",
        "--no-persist-layer-norm",
        "--bf16",
        "--load",
        str(checkpoint_dir),
    ]
    return argv


def module_layer_key(module_name: str, fallback_index: int):
    match = LAYER_RE.search(module_name)
    if match:
        return f"layer_{int(match.group(1)) + 1:02d}"
    return f"layer_{fallback_index:02d}"


def flatten_tensor_list(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def extract_expert_vectors(module):
    if isinstance(module, GroupedMLP) or (
        hasattr(module, "weight1") and hasattr(module, "weight2") and hasattr(module, "num_local_experts")
    ):
        w1 = module.weight1.detach().float().cpu().view(module.num_local_experts, module.config.hidden_size, -1)
        w2 = module.weight2.detach().float().cpu().view(module.num_local_experts, -1, module.config.hidden_size)
        return [flatten_tensor_list([w1[idx], w2[idx]]) for idx in range(module.num_local_experts)]

    if isinstance(module, SequentialMLP):
        vectors = []
        for expert in module.local_experts:
            pieces = []
            for _, tensor in sorted(expert.state_dict().items()):
                if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                    continue
                pieces.append(tensor.detach().float().cpu().reshape(-1))
            vectors.append(torch.cat(pieces, dim=0))
        return vectors

    return None


def cosine_similarity_matrix(vectors):
    stacked = torch.stack(vectors, dim=0)
    normalized = F.normalize(stacked, dim=1)
    return normalized @ normalized.t()


def masked_mean(matrix: torch.Tensor, row_slice: slice, col_slice: slice, diagonal: bool):
    block = matrix[row_slice, col_slice]
    if block.numel() == 0:
        return None
    if diagonal and block.shape[0] == block.shape[1]:
        mask = ~torch.eye(block.shape[0], dtype=torch.bool)
        values = block[mask]
    else:
        values = block.reshape(-1)
    if values.numel() == 0:
        return None
    return float(values.mean().item())


def summarize_similarity(matrix: torch.Tensor, old_expert_count: int):
    total = matrix.shape[0]
    old_end = min(old_expert_count, total)
    return {
        "within_old_mean": masked_mean(matrix, slice(0, old_end), slice(0, old_end), diagonal=True),
        "within_new_mean": masked_mean(matrix, slice(old_end, total), slice(old_end, total), diagonal=True),
        "cross_mean": masked_mean(matrix, slice(0, old_end), slice(old_end, total), diagonal=False),
    }


def save_heatmaps(layer_results, output_dir: Path, title_prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not layer_results:
        return
    n = len(layer_results)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if hasattr(axes, "reshape"):
        flat_axes = list(axes.reshape(-1))
    else:
        flat_axes = [axes]
    for ax_idx, ax in enumerate(flat_axes):
        if ax_idx >= len(layer_results):
            ax.axis("off")
            continue
        item = layer_results[ax_idx]
        matrix = torch.tensor(item["cosine_similarity"])
        im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="viridis")
        ax.set_title(item["layer"])
        ax.set_xlabel("Expert")
        ax.set_ylabel("Expert")
        ax.set_xticks(range(matrix.shape[0]))
        ax.set_yticks(range(matrix.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(output_dir / "layerwise_cosine_similarity.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(run_dir)
    if metadata is None:
        metadata = build_metadata_fallback(args)
    checkpoint_dir = run_dir

    sys.argv = synthesize_megatron_argv(args, metadata, checkpoint_dir)
    initialize_megatron()

    model_list = get_model(model_provider, wrap_with_ddp=False)
    _ = load_checkpoint(model_list, None, None)
    model = model_list[0]

    layer_results = []
    fallback_index = 1
    for module_name, module in model.named_modules():
        vectors = extract_expert_vectors(module)
        if vectors is None:
            continue
        layer_key = module_layer_key(module_name, fallback_index)
        fallback_index += 1
        cosine = cosine_similarity_matrix(vectors).cpu()
        layer_results.append(
            {
                "layer": layer_key,
                "module_name": module_name,
                "num_experts": len(vectors),
                "cosine_similarity": [[float(v) for v in row] for row in cosine.tolist()],
                "summary": summarize_similarity(cosine, args.old_expert_count),
            }
        )

    layer_results.sort(key=lambda item: item["layer"])

    output = {
        "run_dir": str(run_dir),
        "run_id": metadata.get("run_id"),
        "stage": metadata.get("stage"),
        "old_expert_count": args.old_expert_count,
        "layer_results": layer_results,
    }
    (output_dir / "ffn_expert_param_similarity.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    save_heatmaps(layer_results, output_dir, f"FFN expert cosine similarity: {metadata.get('run_id', run_dir.name)}")
    print(f"Saved FFN expert parameter similarity to {output_dir}")


if __name__ == "__main__":
    main()
