#!/usr/bin/env python3
"""Phase-1 Wiki-vs-Code token fingerprint score analysis and gate decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import string
from pathlib import Path

import numpy as np


def load_domain(path: Path):
    blocks = []
    for index in range(5):
        with np.load(path / f"block_{index:03d}.npz", allow_pickle=False) as payload:
            blocks.append({key: payload[key] for key in payload.files})
    with np.load(path / "score_reservoir.npz", allow_pickle=False) as payload:
        reservoir = {key: payload[key] for key in payload.files}
    return blocks, reservoir


def auc_metrics(wiki_hist, code_hist):
    tp = np.cumsum(wiki_hist[::-1], dtype=np.float64)
    fp = np.cumsum(code_hist[::-1], dtype=np.float64)
    recall = tp / max(tp[-1], 1.0)
    fpr = fp / max(fp[-1], 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    auroc = float(np.trapz(np.r_[0.0, recall], np.r_[0.0, fpr]))
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auroc, auprc


def threshold_for_recall(wiki_hist, target, bins):
    survival = np.cumsum(wiki_hist[::-1], dtype=np.float64)[::-1] / wiki_hist.sum()
    candidates = np.flatnonzero(survival >= target)
    index = int(candidates[-1])
    return index, (index + 0.5) / bins, float(survival[index])


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95_normal": [
            float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values))),
            float(values.mean() + 1.96 * values.std(ddof=1) / math.sqrt(len(values))),
        ],
        "values": values.tolist(),
    }


def classify_token(decoded, token_id, eos_id):
    if token_id == eos_id:
        return "eod"
    if decoded == "" or decoded.isspace():
        return "whitespace"
    stripped = decoded.strip()
    if stripped and all(char in string.punctuation for char in stripped):
        return "punctuation"
    if any(char.isalnum() for char in stripped):
        return "alphanumeric"
    return "other"


def bias_summary(reservoir, threshold, tokenizer, common_ids):
    names = reservoir["representation_names"].tolist()
    ranks = reservoir["ranks"].astype(int).tolist()
    selectors = reservoir["selector_names"].tolist()
    scores = reservoir["scores"][names.index("stable"), ranks.index(64), selectors.index("layers_2_to_9_mean")]
    selected = scores >= threshold
    token_ids = reservoir["token_ids"].astype(int)
    positions = reservoir["positions"].astype(int)
    decoded = [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in token_ids]
    categories = np.asarray([
        classify_token(text, int(token_id), tokenizer.eos_token_id)
        for text, token_id in zip(decoded, token_ids)
    ])
    result = {
        "reservoir_tokens": int(token_ids.size),
        "selected_tokens": int(selected.sum()),
        "selected_fraction": float(selected.mean()),
        "categories": {},
        "common_top100_fraction_all": float(np.isin(token_ids, common_ids).mean()),
        "common_top100_fraction_selected": float(np.isin(token_ids[selected], common_ids).mean()) if selected.any() else 0.0,
    }
    for category in ("eod", "whitespace", "punctuation", "alphanumeric", "other"):
        mask = categories == category
        result["categories"][category] = {
            "fraction_all": float(mask.mean()),
            "fraction_selected": float(mask[selected].mean()) if selected.any() else 0.0,
        }
    all_deciles = np.bincount(np.minimum(positions // 52, 9), minlength=10).astype(float)
    selected_deciles = np.bincount(np.minimum(positions[selected] // 52, 9), minlength=10).astype(float)
    all_deciles /= max(all_deciles.sum(), 1)
    selected_deciles /= max(selected_deciles.sum(), 1)
    result["position_decile_fraction_all"] = all_deciles.tolist()
    result["position_decile_fraction_selected"] = selected_deciles.tolist()
    result["position_decile_max_abs_shift"] = float(np.max(np.abs(selected_deciles - all_deciles)))
    unique, counts = np.unique(token_ids[selected], return_counts=True)
    order = np.argsort(counts)[::-1][:20]
    result["top_selected_tokens"] = [
        {
            "token_id": int(unique[index]),
            "decoded": tokenizer.decode([int(unique[index])], clean_up_tokenization_spaces=False),
            "count": int(counts[index]),
            "fraction_selected": float(counts[index] / max(selected.sum(), 1)),
        }
        for index in order
    ]
    result["top20_selected_concentration"] = float(
        sum(row["fraction_selected"] for row in result["top_selected_tokens"])
    )
    _all_unique, all_counts = np.unique(token_ids, return_counts=True)
    result["top20_all_concentration"] = float(
        np.sort(all_counts)[-20:].sum() / max(token_ids.size, 1)
    )
    result["top20_concentration_shift"] = (
        result["top20_selected_concentration"] - result["top20_all_concentration"]
    )
    result["max_category_fraction_selected"] = max(
        row["fraction_selected"] for row in result["categories"].values()
    )
    result["max_category_fraction_shift"] = max(
        abs(row["fraction_selected"] - row["fraction_all"])
        for row in result["categories"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    wiki_path = root / "score_stats/wiki/centered_projection_energy_ratio"
    code_path = root / "score_stats/code/centered_projection_energy_ratio"
    wiki_blocks, wiki_reservoir = load_domain(wiki_path)
    code_blocks, code_reservoir = load_domain(code_path)
    bins = wiki_blocks[0]["hist"].shape[-1]
    representations = wiki_reservoir["representation_names"].tolist()
    ranks = wiki_reservoir["ranks"].astype(int).tolist()
    selectors = wiki_reservoir["selector_names"].tolist()
    rows = []
    detailed = {}
    for rep_index, representation in enumerate(representations):
        detailed[representation] = {}
        for rank_index, rank in enumerate(ranks):
            detailed[representation][str(rank)] = {}
            for selector_index, selector in enumerate(selectors):
                wiki_hist = sum(block["hist"][rep_index, rank_index, selector_index] for block in wiki_blocks)
                code_hist = sum(block["hist"][rep_index, rank_index, selector_index] for block in code_blocks)
                auroc, auprc = auc_metrics(wiki_hist, code_hist)
                block_metrics = [
                    auc_metrics(
                        wiki_blocks[index]["hist"][rep_index, rank_index, selector_index],
                        code_blocks[index]["hist"][rep_index, rank_index, selector_index],
                    )
                    for index in range(5)
                ]
                threshold_rows = {}
                for recall_target in (0.8, 0.9):
                    threshold_index, threshold, actual_recall = threshold_for_recall(
                        wiki_hist, recall_target, bins
                    )
                    threshold_rows[str(recall_target)] = {
                        "threshold_bin": threshold_index,
                        "threshold": threshold,
                        "wiki_recall": actual_recall,
                        "code_selected_fraction": float(code_hist[threshold_index:].sum() / code_hist.sum()),
                    }
                wiki_count = sum(int(block["count"]) for block in wiki_blocks)
                code_count = sum(int(block["count"]) for block in code_blocks)
                wiki_sum = sum(block["score_sum"][rep_index, rank_index, selector_index] for block in wiki_blocks)
                code_sum = sum(block["score_sum"][rep_index, rank_index, selector_index] for block in code_blocks)
                record = {
                    "representation": representation,
                    "rank": rank,
                    "selector": selector,
                    "auroc": auroc,
                    "auprc": auprc,
                    "block_auroc": summarize([value[0] for value in block_metrics]),
                    "block_auprc": summarize([value[1] for value in block_metrics]),
                    "wiki_mean_score": float(wiki_sum / wiki_count),
                    "code_mean_score": float(code_sum / code_count),
                    "thresholds": threshold_rows,
                }
                detailed[representation][str(rank)][selector] = record
                rows.append({
                    "representation": representation,
                    "rank": rank,
                    "selector": selector,
                    "auroc": auroc,
                    "auprc": auprc,
                    "block_auroc_min": record["block_auroc"]["min"],
                    "block_auroc_max": record["block_auroc"]["max"],
                    "wiki_mean_score": record["wiki_mean_score"],
                    "code_mean_score": record["code_mean_score"],
                    "threshold_wiki_recall80": threshold_rows["0.8"]["threshold"],
                    "code_selected_at_wiki_recall80": threshold_rows["0.8"]["code_selected_fraction"],
                })

    selected = detailed["stable"]["64"]["layers_2_to_9_mean"]
    pca = detailed["pca_top"]["64"]["layers_2_to_9_mean"]
    random = detailed["random"]["64"]["layers_2_to_9_mean"]
    threshold = selected["thresholds"]["0.8"]["threshold"]
    # Derive a data-scaled soft transition width from the 80%-vs-90% Wiki-recall band.
    threshold90 = selected["thresholds"]["0.9"]["threshold"]
    soft_temperature = max((threshold - threshold90) / 2.0, 1.0 / bins)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, local_files_only=True)
    code_token_counts = sum(block["token_counts"] for block in code_blocks)
    common_ids = np.argsort(code_token_counts)[::-1][:100]
    bias = bias_summary(code_reservoir, threshold, tokenizer, common_ids)
    gate_checks = {
        "stable_auroc_ge_0_65": selected["auroc"] >= 0.65,
        "stable_beats_pca": selected["auroc"] > pca["auroc"],
        "stable_beats_random": selected["auroc"] > random["auroc"],
        "all_five_blocks_auroc_ge_0_65": selected["block_auroc"]["min"] >= 0.65,
        "code_coverage_ge_0_01": selected["thresholds"]["0.8"]["code_selected_fraction"] >= 0.01,
        "code_coverage_le_0_50": selected["thresholds"]["0.8"]["code_selected_fraction"] <= 0.50,
        "not_special_token_dominated": (
            bias["categories"]["eod"]["fraction_selected"] < 0.05
            and bias["max_category_fraction_selected"] < 0.80
            and bias["max_category_fraction_shift"] < 0.15
        ),
        "common_token_shift_bounded": abs(
            bias["common_top100_fraction_selected"] - bias["common_top100_fraction_all"]
        ) < 0.15,
        "not_position_dominated": bias["position_decile_max_abs_shift"] < 0.10,
    }
    decision = "PASS" if all(gate_checks.values()) else "FAIL"
    result = {
        "phase": 1,
        "question": "Can frozen-teacher stable layer-output scores select old-like Code tokens?",
        "score_definition": "||U^T(h_teacher-mu_wiki)||^2 / ||h_teacher-mu_wiki||^2",
        "wiki_tokens": 10_000_000,
        "code_tokens": 10_000_000,
        "decision": decision,
        "gate_checks": gate_checks,
        "selected_configuration": {
            "representation": "stable",
            "rank": 64,
            "layers": list(range(2, 10)),
            "selector": "layers_2_to_9_mean",
            "hard_threshold": threshold,
            "hard_threshold_source": "Wiki recall 80% on full 10M calibration stream",
            "soft_temperature": soft_temperature,
            "soft_temperature_source": "half of the score gap between Wiki recall 80% and 90% thresholds",
            "auroc": selected["auroc"],
            "auprc": selected["auprc"],
            "block_auroc": selected["block_auroc"],
            "code_selected_fraction": selected["thresholds"]["0.8"]["code_selected_fraction"],
            "controls": {"pca_top_auroc": pca["auroc"], "random_auroc": random["auroc"]},
        },
        "code_selection_bias": bias,
        "all_results": detailed,
        "next_action": (
            "Implement projected fingerprint MSE and verify gradients; do not try alternate scores."
            if decision == "PASS" else
            "Do not implement KD; diagnose the failed gate checks before trying the next score definition."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".inprogress").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    os.replace(str(output) + ".inprogress", output)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output": str(output), "decision": decision, "selected": result["selected_configuration"]}, indent=2))


if __name__ == "__main__":
    main()
