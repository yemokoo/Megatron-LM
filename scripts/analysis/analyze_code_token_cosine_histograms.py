#!/usr/bin/env python3
"""Build exact layer-wise cosine histograms from paired hidden metric shards.

This analysis intentionally assigns no token labels or thresholds.  It scans only
the ``cosine`` and ``valid_mask`` members of each uncompressed NPZ shard and
keeps fixed-width uint64 counts over [-1, 1].
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA = "code_token_cosine_histogram_v1"
EXPECTED_LAYERS = np.arange(1, 10, dtype=np.int16)
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0)
TAIL_THRESHOLDS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
TOP_PERCENTAGES = tuple(range(1, 21))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _discover(root: Path) -> list[tuple[Path, list[Path]]]:
    ranks: list[tuple[Path, list[Path]]] = []
    for rank_dir in sorted(root.glob("rank_[0-9][0-9][0-9]")):
        shards = sorted((rank_dir / "token_metrics").glob("shard_*.npz"))
        if shards:
            ranks.append((rank_dir, shards))
    if not ranks:
        raise RuntimeError(f"no metric shards found below {root}")
    return ranks


def _config_hash(root: Path, ranks: Iterable[tuple[Path, list[Path]]], bins: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(root.resolve()).encode())
    digest.update(f"|cosine|-1|1|{bins}|{EXPECTED_LAYERS.tolist()}".encode())
    for rank_dir, shards in ranks:
        digest.update(rank_dir.name.encode())
        for shard in shards:
            stat = shard.stat()
            digest.update(f"{shard.name}:{stat.st_size}".encode())
    return digest.hexdigest()


def _load_complete_partial(path: Path, config_hash: str, expected_shards: int) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["config_hash"].item()) != config_hash:
                return None
            if int(data["processed_shards"].item()) != expected_shards:
                return None
            return {name: data[name].copy() for name in data.files}
    except Exception:
        return None


def _scan_rank(
    rank_dir_string: str,
    shard_strings: list[str],
    bins: int,
    output_dir_string: str,
    config_hash: str,
) -> dict[str, Any]:
    rank_dir = Path(rank_dir_string)
    shards = [Path(value) for value in shard_strings]
    output_dir = Path(output_dir_string)
    partial_path = output_dir / "partials" / f"{rank_dir.name}.npz"
    cached = _load_complete_partial(partial_path, config_hash, len(shards))
    if cached is not None:
        return {
            "rank": rank_dir.name,
            "partial": str(partial_path),
            "cached": True,
            "seconds": 0.0,
            "tokens": int(cached["valid_tokens"].item()),
        }

    started = time.monotonic()
    counts = np.zeros((9, bins), dtype=np.uint64)
    sums = np.zeros(9, dtype=np.float64)
    minima = np.full(9, np.inf, dtype=np.float32)
    maxima = np.full(9, -np.inf, dtype=np.float32)
    exact_one = np.zeros(9, dtype=np.uint64)
    exact_minus_one = np.zeros(9, dtype=np.uint64)
    valid_tokens = 0
    dense_tokens = 0

    for shard_index, shard in enumerate(shards):
        with np.load(shard, allow_pickle=False) as data:
            layer_numbers = data["layer_numbers"]
            if not np.array_equal(layer_numbers, EXPECTED_LAYERS):
                raise RuntimeError(f"unexpected layer numbers in {shard}: {layer_numbers.tolist()}")
            valid_mask = data["valid_mask"].astype(bool, copy=False)
            cosine = data["cosine"]
            if cosine.ndim != 3 or cosine.shape[:2] != valid_mask.shape or cosine.shape[2] != 9:
                raise RuntimeError(
                    f"shape mismatch in {shard}: cosine={cosine.shape}, mask={valid_mask.shape}"
                )
            dense_tokens += int(valid_mask.size)
            shard_valid = int(valid_mask.sum(dtype=np.int64))
            valid_tokens += shard_valid
            values = cosine.reshape(-1, 9) if shard_valid == valid_mask.size else cosine[valid_mask]
            if values.shape[0] != shard_valid:
                raise RuntimeError(f"valid-token count mismatch in {shard}")
            if not bool(np.isfinite(values).all()):
                raise RuntimeError(f"non-finite cosine in {shard}")
            if bool((values < -1.0).any()) or bool((values > 1.0).any()):
                raise RuntimeError(f"cosine outside [-1, 1] in {shard}")

            minima = np.minimum(minima, values.min(axis=0))
            maxima = np.maximum(maxima, values.max(axis=0))
            sums += values.sum(axis=0, dtype=np.float64)
            exact_one += (values == 1.0).sum(axis=0, dtype=np.uint64)
            exact_minus_one += (values == -1.0).sum(axis=0, dtype=np.uint64)
            for layer_index in range(9):
                histogram, _ = np.histogram(values[:, layer_index], bins=bins, range=(-1.0, 1.0))
                counts[layer_index] += histogram.astype(np.uint64, copy=False)

        if (shard_index + 1) % 12 == 0 or shard_index + 1 == len(shards):
            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"[{rank_dir.name}] {shard_index + 1}/{len(shards)} shards, "
                f"{valid_tokens:,} tokens, {valid_tokens / elapsed:,.0f} token/s",
                flush=True,
            )

    expected = np.full(9, valid_tokens, dtype=np.uint64)
    if not np.array_equal(counts.sum(axis=1, dtype=np.uint64), expected):
        raise RuntimeError(f"histogram lost tokens for {rank_dir.name}")
    _atomic_npz(
        partial_path,
        schema=np.asarray(SCHEMA),
        config_hash=np.asarray(config_hash),
        rank=np.asarray(rank_dir.name),
        layer_numbers=EXPECTED_LAYERS,
        counts=counts,
        sums=sums,
        minima=minima,
        maxima=maxima,
        exact_one=exact_one,
        exact_minus_one=exact_minus_one,
        valid_tokens=np.asarray(valid_tokens, dtype=np.uint64),
        dense_tokens=np.asarray(dense_tokens, dtype=np.uint64),
        processed_shards=np.asarray(len(shards), dtype=np.int64),
    )
    return {
        "rank": rank_dir.name,
        "partial": str(partial_path),
        "cached": False,
        "seconds": time.monotonic() - started,
        "tokens": valid_tokens,
    }


def _font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _format_probability(value: float) -> str:
    if value == 0:
        return "0"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _render_grid(counts: np.ndarray, output: Path, log_y: bool) -> None:
    from PIL import Image, ImageDraw

    width, height = 2100, 1500
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(32), _font(23), _font(18)
    title = "Code-token hidden cosine: reference step 600 vs current step 1800"
    subtitle = "9 layer outputs; fixed x range [-1, 1]; y = probability per 1e-4 bin"
    if log_y:
        subtitle += " (log10 y)"
    draw.text((width // 2, 18), title, fill="#111111", font=title_font, anchor="ma")
    draw.text((width // 2, 58), subtitle, fill="#444444", font=label_font, anchor="ma")

    outer_left, outer_top, outer_right, outer_bottom = 55, 100, 35, 45
    gap_x, gap_y = 28, 34
    cell_w = (width - outer_left - outer_right - 2 * gap_x) // 3
    cell_h = (height - outer_top - outer_bottom - 2 * gap_y) // 3
    bins = counts.shape[1]

    for layer_index in range(9):
        row, column = divmod(layer_index, 3)
        cell_x = outer_left + column * (cell_w + gap_x)
        cell_y = outer_top + row * (cell_h + gap_y)
        plot_left, plot_top = cell_x + 62, cell_y + 38
        plot_right, plot_bottom = cell_x + cell_w - 15, cell_y + cell_h - 52
        plot_w, plot_h = plot_right - plot_left, plot_bottom - plot_top
        probabilities = counts[layer_index].astype(np.float64) / float(counts[layer_index].sum())
        x_pixel = np.minimum((np.arange(bins, dtype=np.int64) * plot_w) // bins, plot_w - 1)
        reduced = np.zeros(plot_w, dtype=np.float64)
        np.maximum.at(reduced, x_pixel, probabilities)

        if log_y:
            positive = probabilities[probabilities > 0]
            lower = math.log10(max(1.0 / float(counts[layer_index].sum()), float(positive.min())))
            upper = math.log10(float(probabilities.max()))
            if upper <= lower:
                lower = upper - 1.0
            transformed = np.full_like(reduced, lower)
            mask = reduced > 0
            transformed[mask] = np.log10(reduced[mask])
            scaled = (transformed - lower) / (upper - lower)
            y_labels = (f"1e{math.floor(lower)}", f"1e{math.ceil((lower + upper) / 2)}", _format_probability(10**upper))
        else:
            upper_value = float(probabilities.max())
            scaled = reduced / upper_value if upper_value else reduced
            y_labels = ("0", _format_probability(upper_value / 2), _format_probability(upper_value))

        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline="#555555", width=1)
        for fraction in (0.0, 0.5, 1.0):
            y = int(round(plot_bottom - fraction * plot_h))
            draw.line((plot_left, y, plot_right, y), fill="#e5e5e5", width=1)
        points = [
            (plot_left + pixel, int(round(plot_bottom - min(max(value, 0.0), 1.0) * plot_h)))
            for pixel, value in enumerate(scaled)
        ]
        if len(points) > 1:
            draw.line(points, fill="#1768ac", width=2, joint="curve")
        draw.text((cell_x + cell_w // 2, cell_y + 2), f"Layer {layer_index + 1}", fill="#111111", font=label_font, anchor="ma")
        for fraction, label in zip((0.0, 0.5, 1.0), y_labels):
            y = int(round(plot_bottom - fraction * plot_h))
            draw.text((plot_left - 7, y), label, fill="#333333", font=tick_font, anchor="rm")
        for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
            x = int(round(plot_left + (value + 1.0) * 0.5 * plot_w))
            draw.line((x, plot_bottom, x, plot_bottom + 5), fill="#333333", width=1)
            draw.text((x, plot_bottom + 8), f"{value:g}", fill="#333333", font=tick_font, anchor="ma")
        draw.text((plot_left + plot_w // 2, plot_bottom + 35), "cosine", fill="#333333", font=tick_font, anchor="ma")

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def _hist_quantile(counts: np.ndarray, edges: np.ndarray, q: float, actual_min: float, actual_max: float) -> float:
    if q <= 0.0:
        return float(actual_min)
    if q >= 1.0:
        return float(actual_max)
    total = int(counts.sum(dtype=np.uint64))
    target = max(1, int(math.ceil(q * total)))
    index = int(np.searchsorted(np.cumsum(counts, dtype=np.uint64), target, side="left"))
    return float((edges[index] + edges[index + 1]) * 0.5)


def _tail_fraction(counts: np.ndarray, threshold: float, bins: int) -> float:
    index = int(round((threshold + 1.0) * bins / 2.0))
    index = min(max(index, 0), bins)
    return float(counts[index:].sum(dtype=np.uint64) / counts.sum(dtype=np.uint64))


def _write_csv(path: Path, edges: np.ndarray, counts: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_left", "bin_right", *[f"layer_{layer}_count" for layer in EXPECTED_LAYERS]])
        for index in range(counts.shape[1]):
            writer.writerow(
                [f"{edges[index]:.7f}", f"{edges[index + 1]:.7f}", *[int(value) for value in counts[:, index]]]
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _top_percent_cutoff(
    counts: np.ndarray,
    edges: np.ndarray,
    top_percent: int,
    exact_one_count: int,
) -> float:
    total = int(counts.sum(dtype=np.uint64))
    target = int(math.ceil(total * top_percent / 100.0))
    if exact_one_count >= target:
        return 1.0
    reverse_cumulative = np.cumsum(counts[::-1], dtype=np.uint64)
    reverse_index = int(np.searchsorted(reverse_cumulative, target, side="left"))
    index = counts.size - 1 - reverse_index
    return float((edges[index] + edges[index + 1]) * 0.5)


def _write_top_percent_csv(path: Path, cutoffs: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".inprogress")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["top_percent", *[f"layer_{layer}_cosine_cutoff" for layer in EXPECTED_LAYERS]])
        for row_index, top_percent in enumerate(TOP_PERCENTAGES):
            writer.writerow([top_percent, *[f"{value:.5f}" for value in cutoffs[row_index]]])
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _render_top_percent_cutoffs(cutoffs: np.ndarray, output: Path) -> None:
    from PIL import Image, ImageDraw

    width, height = 1700, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(34), _font(24), _font(19)
    draw.text(
        (width // 2, 25),
        "Cosine cutoff required to belong to the top p%",
        fill="#111111",
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (width // 2, 70),
        "Full Code-token census; histogram resolution 1e-4; no GT threshold selected",
        fill="#444444",
        font=label_font,
        anchor="ma",
    )

    left, top, right, bottom = 115, 125, 1320, 920
    minimum = float(cutoffs.min())
    y_min = max(0.0, math.floor(minimum * 10.0) / 10.0)
    y_max = 1.0
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    for value in np.linspace(y_min, y_max, 6):
        y = int(round(bottom - (value - y_min) / (y_max - y_min) * (bottom - top)))
        draw.line((left, y, right, y), fill="#e2e2e2", width=1)
        draw.text((left - 10, y), f"{value:.2f}", fill="#333333", font=tick_font, anchor="rm")
    for top_percent in (1, 5, 10, 15, 20):
        x = int(round(left + (top_percent - 1) / 19.0 * (right - left)))
        draw.line((x, top, x, bottom), fill="#eeeeee", width=1)
        draw.text((x, bottom + 12), str(top_percent), fill="#333333", font=tick_font, anchor="ma")

    colors = (
        "#111111",
        "#d62728",
        "#ff7f0e",
        "#bcbd22",
        "#2ca02c",
        "#17becf",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
    )
    for layer_index, color in enumerate(colors):
        points = []
        for row_index, top_percent in enumerate(TOP_PERCENTAGES):
            x = int(round(left + (top_percent - 1) / 19.0 * (right - left)))
            y = int(
                round(bottom - (cutoffs[row_index, layer_index] - y_min) / (y_max - y_min) * (bottom - top))
            )
            points.append((x, y))
        draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        legend_y = top + 20 + layer_index * 58
        draw.line((1370, legend_y, 1430, legend_y), fill=color, width=5)
        draw.text((1450, legend_y), f"Layer {layer_index + 1}", fill="#222222", font=label_font, anchor="lm")

    draw.text((left + (right - left) // 2, bottom + 67), "top p%", fill="#222222", font=label_font, anchor="ma")
    draw.text((left, top - 12), "cosine cutoff", fill="#222222", font=tick_font, anchor="ls")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bins", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="discard compatible cached rank partials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins <= 0:
        raise SystemExit("--bins must be positive")
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "analysis" / "cosine_histogram").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ranks = _discover(root)
    config_hash = _config_hash(root, ranks, args.bins)
    if args.force:
        for partial in (output_dir / "partials").glob("rank_*.npz"):
            partial.unlink()

    started = time.monotonic()
    futures = []
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(ranks))) as pool:
        for rank_dir, shards in ranks:
            futures.append(
                pool.submit(
                    _scan_rank,
                    str(rank_dir),
                    [str(path) for path in shards],
                    args.bins,
                    str(output_dir),
                    config_hash,
                )
            )
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    counts = np.zeros((9, args.bins), dtype=np.uint64)
    sums = np.zeros(9, dtype=np.float64)
    minima = np.full(9, np.inf, dtype=np.float32)
    maxima = np.full(9, -np.inf, dtype=np.float32)
    exact_one = np.zeros(9, dtype=np.uint64)
    exact_minus_one = np.zeros(9, dtype=np.uint64)
    valid_tokens = 0
    dense_tokens = 0
    total_shards = 0
    for rank_dir, shards in ranks:
        partial_path = output_dir / "partials" / f"{rank_dir.name}.npz"
        partial = _load_complete_partial(partial_path, config_hash, len(shards))
        if partial is None:
            raise RuntimeError(f"missing or stale partial {partial_path}")
        counts += partial["counts"].astype(np.uint64, copy=False)
        sums += partial["sums"].astype(np.float64, copy=False)
        minima = np.minimum(minima, partial["minima"])
        maxima = np.maximum(maxima, partial["maxima"])
        exact_one += partial["exact_one"].astype(np.uint64, copy=False)
        exact_minus_one += partial["exact_minus_one"].astype(np.uint64, copy=False)
        valid_tokens += int(partial["valid_tokens"].item())
        dense_tokens += int(partial["dense_tokens"].item())
        total_shards += int(partial["processed_shards"].item())

    per_layer_counts = counts.sum(axis=1, dtype=np.uint64)
    if not np.array_equal(per_layer_counts, np.full(9, valid_tokens, dtype=np.uint64)):
        raise RuntimeError(f"merged count mismatch: {per_layer_counts.tolist()} vs {valid_tokens}")
    edges = np.linspace(-1.0, 1.0, args.bins + 1, dtype=np.float64)
    width = float(edges[1] - edges[0])

    layers: list[dict[str, Any]] = []
    for layer_index, layer_number in enumerate(EXPECTED_LAYERS):
        layer_counts = counts[layer_index]
        layers.append(
            {
                "layer": int(layer_number),
                "count": int(per_layer_counts[layer_index]),
                "mean": float(sums[layer_index] / valid_tokens),
                "min": float(minima[layer_index]),
                "max": float(maxima[layer_index]),
                "exact_one_count": int(exact_one[layer_index]),
                "exact_minus_one_count": int(exact_minus_one[layer_index]),
                "quantiles_histogram_midpoint": {
                    f"{q:g}": _hist_quantile(layer_counts, edges, q, minima[layer_index], maxima[layer_index])
                    for q in QUANTILES
                },
                "fraction_cosine_ge": {
                    f"{threshold:g}": _tail_fraction(layer_counts, threshold, args.bins)
                    for threshold in TAIL_THRESHOLDS
                },
            }
        )

    counts_path = output_dir / "cosine_histogram_counts.npz"
    _atomic_npz(
        counts_path,
        schema=np.asarray(SCHEMA),
        config_hash=np.asarray(config_hash),
        layer_numbers=EXPECTED_LAYERS,
        bin_edges=edges,
        counts=counts,
        valid_tokens=np.asarray(valid_tokens, dtype=np.uint64),
        dense_tokens=np.asarray(dense_tokens, dtype=np.uint64),
    )
    _write_csv(output_dir / "cosine_histogram_counts.csv", edges, counts)
    _render_grid(counts, output_dir / "cosine_histogram_layers_linear.png", log_y=False)
    _render_grid(counts, output_dir / "cosine_histogram_layers_log.png", log_y=True)
    top_percent_cutoffs = np.asarray(
        [
            [
                _top_percent_cutoff(
                    counts[layer_index],
                    edges,
                    top_percent,
                    int(exact_one[layer_index]),
                )
                for layer_index in range(9)
            ]
            for top_percent in TOP_PERCENTAGES
        ],
        dtype=np.float64,
    )
    top_percent_csv = output_dir / "cosine_top_1_to_20_percent_cutoffs.csv"
    top_percent_png = output_dir / "cosine_top_1_to_20_percent_cutoffs.png"
    _write_top_percent_csv(top_percent_csv, top_percent_cutoffs)
    _render_top_percent_cutoffs(top_percent_cutoffs, top_percent_png)

    summary = {
        "schema": SCHEMA,
        "complete": True,
        "gt_assigned": False,
        "threshold_selected": False,
        "source_root": str(root),
        "config_hash": config_hash,
        "metric": "cosine",
        "range": [-1.0, 1.0],
        "bins": args.bins,
        "bin_width": width,
        "histogram_quantile_error_at_most": width,
        "workers": len(ranks),
        "shards": total_shards,
        "dense_tokens": dense_tokens,
        "valid_tokens_per_layer": valid_tokens,
        "token_layer_observations": valid_tokens * 9,
        "seconds": time.monotonic() - started,
        "artifacts": {
            "counts_npz": str(counts_path),
            "counts_csv": str(output_dir / "cosine_histogram_counts.csv"),
            "linear_png": str(output_dir / "cosine_histogram_layers_linear.png"),
            "log_png": str(output_dir / "cosine_histogram_layers_log.png"),
            "top_1_to_20_percent_csv": str(top_percent_csv),
            "top_1_to_20_percent_png": str(top_percent_png),
        },
        "top_percent_cosine_cutoffs": [
            {
                "top_percent": top_percent,
                "by_layer": {
                    str(layer): float(top_percent_cutoffs[row_index, layer_index])
                    for layer_index, layer in enumerate(EXPECTED_LAYERS)
                },
            }
            for row_index, top_percent in enumerate(TOP_PERCENTAGES)
        ],
        "layers": layers,
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
