#!/usr/bin/env python3
"""Validate and upload only final checkpoint artifacts listed in a TSV manifest.

The default mode is read-only. Pass --execute to write to the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


DEFAULT_MANIFEST = Path(__file__).with_name("final_checkpoint_manifest.tsv")


@dataclass(frozen=True)
class Entry:
    enabled: bool
    group: str
    kind: str
    name: str
    expected_iteration: int | None
    source: Path
    path_in_repo: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-id", default="YeMoKoo/LLM-continual-learning")
    parser.add_argument(
        "--group",
        action="append",
        choices=("g2", "trace"),
        help="Restrict to one or more groups (default: all).",
    )
    parser.add_argument(
        "--name",
        action="append",
        help="fnmatch pattern for manifest names; may be repeated.",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Validate disabled/pending rows too. Disabled rows are never uploaded.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually upload. Without this flag the command is a dry run.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw_iteration = row["expected_iteration"].strip()
            entries.append(
                Entry(
                    enabled=row["enabled"].strip() == "1",
                    group=row["group"].strip(),
                    kind=row["kind"].strip(),
                    name=row["name"].strip(),
                    expected_iteration=int(raw_iteration) if raw_iteration else None,
                    source=Path(row["source"].strip()),
                    path_in_repo=row["path_in_repo"].strip().strip("/"),
                    note=row["note"].strip(),
                )
            )
    return entries


def iter_files(entry: Entry) -> list[Path]:
    if not entry.source.is_dir():
        raise FileNotFoundError(f"source directory not found: {entry.source}")

    if entry.kind == "megatron":
        if entry.expected_iteration is None:
            raise ValueError(f"{entry.name}: megatron entry needs expected_iteration")
        tracker = entry.source / "latest_checkpointed_iteration.txt"
        actual = int(tracker.read_text(encoding="utf-8").strip())
        if actual != entry.expected_iteration:
            raise ValueError(
                f"{entry.name}: tracker={actual}, expected={entry.expected_iteration}"
            )
        final_dir = entry.source / f"iter_{actual:07d}"
        if not final_dir.is_dir():
            raise FileNotFoundError(f"final iteration directory not found: {final_dir}")
        return [tracker, *(p for p in final_dir.rglob("*") if p.is_file())]

    if entry.kind == "top_files":
        # Some upstream S-LoRA runs keep both the merged max.safetensors and
        # temporary rank shards beside it. The merged file is the load target;
        # uploading the rank shards would duplicate the same weights.
        files = sorted(
            p
            for p in entry.source.iterdir()
            if p.is_file() and not fnmatch.fnmatch(p.name, "*.rank*-of-*")
        )
        weight_names = {
            "pytorch_model.bin",
            "adapter_model.safetensors",
            "model.safetensors",
        }
        if not any(p.name in weight_names for p in files):
            raise FileNotFoundError(f"{entry.name}: no final model weight at directory root")
        return files

    raise ValueError(f"{entry.name}: unsupported kind {entry.kind!r}")


def allow_patterns(entry: Entry, files: Iterable[Path]) -> list[str]:
    return [str(path.relative_to(entry.source)) for path in files]


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def selected(entries: Iterable[Entry], args: argparse.Namespace) -> Iterable[Entry]:
    for entry in entries:
        if args.group and entry.group not in args.group:
            continue
        if args.name and not any(fnmatch.fnmatch(entry.name, pat) for pat in args.name):
            continue
        if not entry.enabled and not args.include_disabled:
            continue
        yield entry


def main() -> int:
    args = parse_args()
    entries = list(selected(read_manifest(args.manifest), args))
    if not entries:
        raise SystemExit("no manifest entries selected")

    checked: list[tuple[Entry, list[Path], int]] = []
    failures = 0
    for entry in entries:
        try:
            files = iter_files(entry)
            size = sum(path.stat().st_size for path in files)
            checked.append((entry, files, size))
            state = "READY" if entry.enabled else "DISABLED"
            print(
                f"{state:8} {entry.group:5} {entry.name:48} "
                f"{len(files):4d} files {human_bytes(size):>10} -> {entry.path_in_repo}"
            )
        except (FileNotFoundError, ValueError) as exc:
            failures += 1
            print(f"ERROR    {entry.group:5} {entry.name}: {exc}")

    total = sum(size for entry, _, size in checked if entry.enabled)
    ready_count = sum(entry.enabled for entry, _, _ in checked)
    print(f"\nready: {ready_count}, failed: {failures}, selected bytes: {human_bytes(total)}")
    if failures:
        return 1
    if not args.execute:
        print("dry run only; pass --execute to upload")
        return 0

    api = HfApi()
    for entry, files, _ in checked:
        if not entry.enabled:
            print(f"SKIP     disabled: {entry.name}")
            continue
        print(f"UPLOAD   {entry.name} -> {args.repo_id}/{entry.path_in_repo}")
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(entry.source),
            path_in_repo=entry.path_in_repo,
            allow_patterns=allow_patterns(entry, files),
            commit_message=f"Add final checkpoint: {entry.name}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
