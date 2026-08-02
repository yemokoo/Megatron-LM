"""Tiny NumPy plumbing doubles; never use their outputs as paper results."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np


def one_optimizer_step(weight: np.ndarray, x: np.ndarray, target: np.ndarray, lr=0.1):
    prediction = x @ weight
    gradient = x.T @ (prediction - target) / x.shape[0]
    return weight - lr * gradient


def save_adapter(path: Path, a: np.ndarray, b: np.ndarray) -> None:
    np.savez(path, lora_A=a, lora_B=b)


def load_adapter(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as values:
        return values["lora_A"], values["lora_B"]


def merge_adapter(base: np.ndarray, a: np.ndarray, b: np.ndarray, scale=1.0):
    return base + scale * (b @ a)


def denoise_update(update: np.ndarray, rank: int) -> np.ndarray:
    u, singular, vt = np.linalg.svd(update, full_matrices=False)
    return (u[:, :rank] * singular[:rank]) @ vt[:rank]


def greedy_generate(logits: np.ndarray) -> list[int]:
    return np.argmax(logits, axis=-1).tolist()


def metric_dispatch(name: str, prediction: str, target: str) -> float:
    if name == "accuracy":
        return float(prediction == target)
    if name == "rouge_l":
        pred, ref = prediction.split(), target.split()
        rows = [0] * (len(ref) + 1)
        for token in pred:
            previous = rows.copy()
            for index, ref_token in enumerate(ref, 1):
                rows[index] = (
                    previous[index - 1] + 1
                    if token == ref_token
                    else max(previous[index], rows[index - 1])
                )
        lcs = rows[-1]
        precision = lcs / len(pred) if pred else 0.0
        recall = lcs / len(ref) if ref else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if name == "edit_similarity":
        return SequenceMatcher(None, prediction, target).ratio()
    if name == "sari":
        # Dispatch double only. Exact paper parity requires the pinned
        # Hugging Face evaluate SARI implementation in the model phase.
        return float(prediction == target)
    raise ValueError(f"unknown metric: {name}")
