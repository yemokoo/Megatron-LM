#!/usr/bin/env python3
"""Plot final probe accuracy for F2 attention full-rank LoRA rank ablations."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


PROBE_RE = re.compile(
    r"probe\s+(?P<name>\S+)\s+at iteration\s+(?P<iteration>\d+)\s+\|\s+"
    r"local_iteration:\s+(?P<local_iteration>\d+)\s+\|\s+"
    r"next_token_acc:\s+(?P<acc>[0-9.]+)\s+\|\s+ppl:\s+(?P<ppl>[0-9.Ee+-]+)"
)


@dataclass(frozen=True)
class RankResult:
    rank: int
    run_dir: Path
    log_path: Path
    local_iteration: int
    global_iteration: int
    code_acc: float | None
    wiki_acc: float | None
    code_ppl: float | None
    wiki_ppl: float | None


def parse_rank_run(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--rank-run must be formatted like 1024=/path/to/run")
    rank_text, path_text = value.split("=", 1)
    try:
        rank = int(rank_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid rank in --rank-run: {rank_text!r}") from exc
    return rank, Path(path_text).expanduser()


def candidate_run_dirs(base_dir: Path, rank: int) -> list[Path]:
    exact = base_dir / (
        f"f2-r{rank}-wiki-to-code-ffn-moe-freeze-attn-full-rank-lora-qkvo-"
        "mha-a100-bf16-mb96-1800"
    )
    candidates = [exact]

    if rank == 1024:
        patterns = (
            "f2-wiki-to-code-ffn-moe-freeze-attn-full-rank-lora*mha-a100-bf16*",
            "*f2*wiki-to-code*freeze*attn-full-rank-lora*mha*a100*bf16*",
            "*full-rank-lora*freeze*mb96*",
        )
        for pattern in patterns:
            candidates.extend(sorted(path for path in base_dir.glob(pattern) if path.is_dir()))

    return dedupe_paths(candidates)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def find_log_path(run_dir: Path) -> Path | None:
    preferred = run_dir / "logs" / "a_to_b_freeze.log"
    if preferred.is_file():
        return preferred
    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        matches = sorted(logs_dir.glob("*.log"))
        if matches:
            return matches[0]
    return None


def parse_probe_log(log_path: Path, local_step: int) -> dict[str, dict[str, float | int]]:
    by_probe: dict[str, dict[int, dict[str, float | int]]] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = PROBE_RE.search(line)
            if match is None:
                continue
            probe = match.group("name")
            step = int(match.group("local_iteration"))
            by_probe.setdefault(probe, {})[step] = {
                "global_iteration": int(match.group("iteration")),
                "acc": float(match.group("acc")),
                "ppl": float(match.group("ppl")),
            }

    selected: dict[str, dict[str, float | int]] = {}
    for probe, values in by_probe.items():
        if local_step in values:
            selected[probe] = values[local_step]
        elif values:
            selected[probe] = values[max(values)]
    return selected


def collect_results(
    base_dir: Path,
    ranks: list[int],
    explicit_runs: dict[int, Path],
    local_step: int,
) -> tuple[list[RankResult], list[str]]:
    results: list[RankResult] = []
    warnings: list[str] = []

    for rank in ranks:
        candidates = [explicit_runs[rank]] if rank in explicit_runs else candidate_run_dirs(base_dir, rank)
        run_dir = next((path for path in candidates if path.is_dir()), None)
        if run_dir is None:
            warnings.append(
                f"rank={rank}: run directory not found. Tried: "
                + ", ".join(str(path) for path in candidates)
            )
            continue

        log_path = find_log_path(run_dir)
        if log_path is None:
            warnings.append(f"rank={rank}: log file not found under {run_dir}")
            continue

        parsed = parse_probe_log(log_path, local_step)
        code = parsed.get("code_probe")
        wiki = parsed.get("wiki_probe")
        if code is None or wiki is None:
            warnings.append(f"rank={rank}: missing code_probe or wiki_probe in {log_path}")

        observed_steps = [
            int(item["global_iteration"])
            for item in (code, wiki)
            if item is not None and "global_iteration" in item
        ]
        result = RankResult(
            rank=rank,
            run_dir=run_dir,
            log_path=log_path,
            local_iteration=local_step,
            global_iteration=max(observed_steps) if observed_steps else -1,
            code_acc=float(code["acc"]) if code else None,
            wiki_acc=float(wiki["acc"]) if wiki else None,
            code_ppl=float(code["ppl"]) if code else None,
            wiki_ppl=float(wiki["ppl"]) if wiki else None,
        )
        results.append(result)

    return sorted(results, key=lambda row: row.rank), warnings


def write_csv(results: list[RankResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "local_iteration",
                "global_iteration",
                "wiki_next_token_acc",
                "code_next_token_acc",
                "wiki_ppl",
                "code_ppl",
                "run_dir",
                "log_path",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "rank": row.rank,
                    "local_iteration": row.local_iteration,
                    "global_iteration": row.global_iteration,
                    "wiki_next_token_acc": row.wiki_acc,
                    "code_next_token_acc": row.code_acc,
                    "wiki_ppl": row.wiki_ppl,
                    "code_ppl": row.code_ppl,
                    "run_dir": row.run_dir,
                    "log_path": row.log_path,
                }
            )


def plot_with_matplotlib(results: list[RankResult], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranks = [row.rank for row in results]
    x_positions = list(range(len(ranks)))
    wiki_acc = [row.wiki_acc for row in results]
    code_acc = [row.code_acc for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    fig.suptitle("F2 Rank Ablation: Final Next-Token Accuracy at Local Step 1800", fontsize=14)

    panels = (
        (axes[0], wiki_acc, "Wiki Probe", "#b45309"),
        (axes[1], code_acc, "Code Probe", "#1d4ed8"),
    )
    for axis, values, title, color in panels:
        axis.plot(x_positions, values, marker="o", linewidth=2.2, color=color)
        finite_values = [value for value in values if value is not None]
        if finite_values:
            y_span = max(finite_values) - min(finite_values)
            y_pad = max(y_span * 0.22, 0.0015)
            axis.set_ylim(min(finite_values) - y_pad, max(finite_values) + y_pad)

        for idx, (rank, value) in enumerate(zip(ranks, values)):
            if value is None:
                continue
            vertical_offset = 11 if idx % 2 == 0 else -17
            va = "bottom" if vertical_offset > 0 else "top"
            axis.annotate(
                f"{value:.4f}",
                (x_positions[idx], value),
                textcoords="offset points",
                xytext=(0, vertical_offset),
                ha="center",
                va=va,
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )
        axis.set_title(title)
        axis.set_xlabel("Full-rank LoRA rank (equally spaced)")
        axis.set_ylabel("next_token_acc")
        axis.grid(True, alpha=0.3)
        axis.set_xticks(x_positions)
        axis.set_xticklabels([str(rank) for rank in ranks])
        axis.tick_params(axis="x", rotation=35)
        axis.margins(x=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(".local/weights/a100/mha/f-attn-full-rank-lora-bf16-freeze"),
        help="Directory containing F2 rank run directories.",
    )
    parser.add_argument(
        "--ranks",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512, 640, 768, 896, 1024],
        help="Ranks to plot. Missing ranks are warned and skipped.",
    )
    parser.add_argument(
        "--rank-run",
        action="append",
        type=parse_rank_run,
        default=[],
        help="Explicit run directory for a rank, e.g. --rank-run 1024=/path/to/run.",
    )
    parser.add_argument("--local-step", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/f2_rank_ablation"))
    parser.add_argument("--output-name", default="f2_rank_ablation_final_acc")
    args = parser.parse_args()

    explicit_runs = dict(args.rank_run)
    results, warnings = collect_results(args.base_dir, args.ranks, explicit_runs, args.local_step)
    if not results:
        raise SystemExit("No rank results found. Check --base-dir or pass --rank-run entries.")

    csv_path = args.output_dir / f"{args.output_name}.csv"
    png_path = args.output_dir / f"{args.output_name}.png"
    write_csv(results, csv_path)
    plot_with_matplotlib(results, png_path)

    print(f"wrote csv: {csv_path}")
    print(f"wrote plot: {png_path}")
    print()
    print("rank\twiki_acc\tcode_acc\tlog")
    for row in results:
        wiki = "NA" if row.wiki_acc is None else f"{row.wiki_acc:.6f}"
        code = "NA" if row.code_acc is None else f"{row.code_acc:.6f}"
        print(f"{row.rank}\t{wiki}\t{code}\t{row.log_path}")

    if warnings:
        print()
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
