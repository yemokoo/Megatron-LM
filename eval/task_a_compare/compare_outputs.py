#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def iter_prediction_pairs(root_a: Path, root_b: Path):
    files_a = sorted((root_a / "predictions").glob("*.pt"))
    files_b = sorted((root_b / "predictions").glob("*.pt"))
    if [path.name for path in files_a] != [path.name for path in files_b]:
        raise RuntimeError("Prediction shard mismatch between labels.")
    for path_a, path_b in zip(files_a, files_b):
        yield path_a.name, torch.load(path_a, map_location="cpu"), torch.load(path_b, map_location="cpu")


def iter_routing_pairs(root_a: Path, root_b: Path, layer_name: str):
    files_a = sorted((root_a / "routing" / layer_name).glob("*.pt"))
    files_b = sorted((root_b / "routing" / layer_name).glob("*.pt"))
    if [path.name for path in files_a] != [path.name for path in files_b]:
        raise RuntimeError(f"Routing shard mismatch for {layer_name}.")
    for path_a, path_b in zip(files_a, files_b):
        yield torch.load(path_a, map_location="cpu"), torch.load(path_b, map_location="cpu")


def compute_prediction_summary(a_root: Path, b_root: Path):
    totals = Counter()
    total_nll_a = 0.0
    total_nll_b = 0.0

    for key, a_item, b_item in iter_prediction_pairs(a_root, b_root):
        if "tokens" in a_item and "tokens" in b_item and not torch.equal(a_item["tokens"], b_item["tokens"]):
            raise RuntimeError(f"Token mismatch for batch {key}")
        if "labels" in a_item and "labels" in b_item and not torch.equal(a_item["labels"], b_item["labels"]):
            raise RuntimeError(f"Label mismatch for batch {key}")

        mask = a_item["loss_mask"].bool()
        a_correct = a_item["correct"][mask]
        b_correct = b_item["correct"][mask]

        if "token_nll" in a_item:
            total_nll_a += float(a_item["token_nll"][mask].sum().item())
        else:
            total_nll_a += float(a_item["nll_sum"])

        if "token_nll" in b_item:
            total_nll_b += float(b_item["token_nll"][mask].sum().item())
        else:
            total_nll_b += float(b_item["nll_sum"])

        totals["tokens"] += int(mask.sum().item())
        totals["a_correct"] += int(a_correct.sum().item())
        totals["b_correct"] += int(b_correct.sum().item())
        totals["both_correct"] += int((a_correct & b_correct).sum().item())
        totals["a_only_correct"] += int((a_correct & ~b_correct).sum().item())
        totals["b_only_correct"] += int((~a_correct & b_correct).sum().item())
        totals["both_wrong"] += int((~a_correct & ~b_correct).sum().item())

    return {
        "tokens": totals["tokens"],
        "a_accuracy": totals["a_correct"] / max(totals["tokens"], 1),
        "b_accuracy": totals["b_correct"] / max(totals["tokens"], 1),
        "accuracy_delta": (totals["b_correct"] - totals["a_correct"]) / max(totals["tokens"], 1),
        "a_mean_nll": total_nll_a / max(totals["tokens"], 1),
        "b_mean_nll": total_nll_b / max(totals["tokens"], 1),
        "correctness_flips": dict(totals),
    }


def compute_routing_summary(a_root: Path, b_root: Path, source_num_experts: int):
    summary = {}
    for layer_dir in sorted((a_root / "routing").glob("layer_*")):
        layer_name = layer_dir.name
        overlap_total = 0.0
        token_total = 0
        new_expert_hits = 0
        changed_tokens = 0
        all_old_tokens = 0
        mixed_tokens = 0
        all_new_tokens = 0
        new_slot_total = 0.0

        for a_item, b_item in iter_routing_pairs(a_root, b_root, layer_name):
            a_idx = a_item["topk_indices"].to(torch.int16)
            b_idx = b_item["topk_indices"].to(torch.int16)
            if a_idx.shape != b_idx.shape:
                raise RuntimeError(f"Routing token count mismatch for {layer_name}")

            matches = (a_idx.unsqueeze(2) == b_idx.unsqueeze(1)).any(dim=2)
            overlap_counts = matches.sum(dim=1)
            token_total += int(a_idx.shape[0])
            overlap_total += float((overlap_counts.float() / a_idx.shape[1]).sum().item())
            changed_tokens += int((overlap_counts != a_idx.shape[1]).sum().item())
            is_new = b_idx >= source_num_experts
            new_expert_hits += int(is_new.any(dim=1).sum().item())
            all_old_mask = (~is_new).all(dim=1)
            all_new_mask = is_new.all(dim=1)
            mixed_mask = ~(all_old_mask | all_new_mask)
            all_old_tokens += int(all_old_mask.sum().item())
            mixed_tokens += int(mixed_mask.sum().item())
            all_new_tokens += int(all_new_mask.sum().item())
            new_slot_total += float(is_new.float().mean(dim=1).sum().item())

        summary[layer_name] = {
            "token_count": token_total,
            "mean_topk_overlap": overlap_total / max(token_total, 1),
            "changed_routing_fraction": changed_tokens / max(token_total, 1),
            "fraction_using_new_expert_in_b": new_expert_hits / max(token_total, 1),
            "fraction_all_old_group_in_b": all_old_tokens / max(token_total, 1),
            "fraction_mixed_old_new_group_in_b": mixed_tokens / max(token_total, 1),
            "fraction_all_new_group_in_b": all_new_tokens / max(token_total, 1),
            "mean_new_group_slot_fraction_in_b": new_slot_total / max(token_total, 1),
        }

    return summary


def plot_routing_summary(routing_summary, output_dir: Path):
    layers = list(sorted(routing_summary))
    overlap = [routing_summary[layer]["mean_topk_overlap"] for layer in layers]
    new_expert = [routing_summary[layer]["fraction_using_new_expert_in_b"] for layer in layers]
    all_old = [routing_summary[layer]["fraction_all_old_group_in_b"] for layer in layers]
    mixed = [routing_summary[layer]["fraction_mixed_old_new_group_in_b"] for layer in layers]
    all_new = [routing_summary[layer]["fraction_all_new_group_in_b"] for layer in layers]
    new_slot_share = [routing_summary[layer]["mean_new_group_slot_fraction_in_b"] for layer in layers]

    plt.figure(figsize=(10, 5))
    plt.bar(layers, overlap)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Mean top-k overlap")
    plt.title("Routing overlap: baseline vs continual")
    plt.tight_layout()
    plt.savefig(output_dir / "routing_overlap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(layers, new_expert)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction using new expert in continual model")
    plt.title("New expert usage")
    plt.tight_layout()
    plt.savefig(output_dir / "new_expert_usage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(layers, all_old, label="old4 only")
    plt.bar(layers, mixed, bottom=all_old, label="mixed old4+new3")
    stacked_bottom = [old + mix for old, mix in zip(all_old, mixed)]
    plt.bar(layers, all_new, bottom=stacked_bottom, label="new3 only")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Fraction of tokens in target routing")
    plt.title("Routing by expert group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "expert_group_usage.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(layers, new_slot_share)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Mean fraction of top-k slots in new3")
    plt.title("New-group slot share")
    plt.tight_layout()
    plt.savefig(output_dir / "new_group_slot_share.png", dpi=200)
    plt.close()


def cleanup_raw_dumps(label_root: Path):
    for name in ("predictions", "routing", "eact", "tids"):
        path = label_root / name
        if path.exists():
            shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description="Compare Task A eval dumps between two checkpoints.")
    parser.add_argument("--root", required=True, help="Root dump directory containing both labels.")
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--source-num-experts", type=int, default=4)
    parser.add_argument("--cleanup-raw-dumps", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    a_root = root / args.label_a
    b_root = root / args.label_b
    output_dir = root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_summary = compute_prediction_summary(a_root, b_root)
    routing_summary = compute_routing_summary(a_root, b_root, args.source_num_experts)
    plot_routing_summary(routing_summary, output_dir)

    summary = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "prediction_summary": prediction_summary,
        "routing_summary": routing_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.cleanup_raw_dumps:
        cleanup_raw_dumps(a_root)
        cleanup_raw_dumps(b_root)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
