#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is not installed. Run scripts/release/install_hf_tools.sh first."
    ) from exc

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    source_dir: str
    path_in_repo: str
    summary: str

    @property
    def abs_source_dir(self) -> Path:
        return ROOT / self.source_dir


DATASET_SPECS: list[DatasetSpec] = [
    DatasetSpec(
        key="wiki_train",
        title="Wiki Train Exact",
        source_dir=".local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact",
        path_in_repo="wiki/train",
        summary="Exact wiki train split: first 1800 steps worth of tokens.",
    ),
    DatasetSpec(
        key="wiki_test",
        title="Wiki Test Full Remainder Exact",
        source_dir=".local/dataset/wikipedia-full/tokenized/EleutherAI/pythia-12b-step1800-test-fullremainder-exact",
        path_in_repo="wiki/test",
        summary="Exact wiki test split: remainder after the first 1800 steps.",
    ),
    DatasetSpec(
        key="code_train",
        title="Code Train Exact",
        source_dir=".local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-train-exact",
        path_in_repo="code/train",
        summary="Exact code train split: first 1800 steps worth of tokens.",
    ),
    DatasetSpec(
        key="code_test",
        title="Code Test Match Wiki Exact",
        source_dir=".local/dataset/python-code-full/tokenized/EleutherAI/pythia-12b-step1800-test-matchwiki-exact",
        path_in_repo="code/test",
        summary="Exact code test split trimmed to match the wiki test token count.",
    ),
]


def build_repo_readme(repo_id: str, datasets: Iterable[DatasetSpec]) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "Exact tokenized train/test splits used for the FLAME-MoE continual-learning experiments.",
        "",
        "Repository layout:",
        "- `wiki/train`",
        "- `wiki/test`",
        "- `code/train`",
        "- `code/test`",
        "",
        "Dataset entries:",
        "",
        "| Folder | Summary |",
        "| --- | --- |",
    ]
    for spec in datasets:
        lines.append(f"| `{spec.path_in_repo}` | {spec.summary} |")
    lines.extend(
        [
            "",
            "These are Megatron indexed-dataset artifacts, not Hugging Face `datasets` parquet exports.",
            "",
            "Source code: https://github.com/YeMoKoo/FLAME-MoE",
        ]
    )
    return "\n".join(lines) + "\n"


def build_dataset_readme(spec: DatasetSpec) -> str:
    return (
        f"# {spec.title}\n\n"
        f"- folder: `{spec.path_in_repo}`\n"
        f"- source dir: `{spec.source_dir}`\n"
        f"- summary: {spec.summary}\n\n"
        "This subfolder contains Megatron indexed-dataset shards.\n"
    )


def ensure_sources(datasets: Iterable[DatasetSpec]) -> None:
    missing: list[str] = []
    for spec in datasets:
        if not spec.abs_source_dir.is_dir():
            missing.append(f"missing dataset dir: {spec.abs_source_dir}")
    if missing:
        raise SystemExit("\n".join(missing))


def upload_generated_files(api: HfApi, repo_id: str, path_in_repo: str, files: dict[str, str], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] upload generated files to {repo_id}/{path_in_repo or '<root>'}: {sorted(files)}")
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name, content in files.items():
            (tmp / name).write_text(content)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(tmp),
            path_in_repo=path_in_repo,
            commit_message=f"Add metadata for {path_in_repo or 'repo root'}",
        )


def upload_dataset(api: HfApi, repo_id: str, spec: DatasetSpec, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] upload {spec.abs_source_dir} -> {repo_id}/{spec.path_in_repo}")
        return
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(spec.abs_source_dir),
        path_in_repo=spec.path_in_repo,
        commit_message=f"Upload {spec.title}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the exact FLAME-MoE dataset splits to a single HF dataset repo.")
    parser.add_argument("--repo-id", default="YeMoKoo/flamedata")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[spec.key for spec in DATASET_SPECS],
        help="Subset of dataset keys to upload.",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_keys = set(args.datasets)
    selected = [spec for spec in DATASET_SPECS if spec.key in selected_keys]
    if not selected:
        raise SystemExit("No matching datasets selected.")

    ensure_sources(selected)
    api = HfApi()

    if args.dry_run:
        print(f"[dry-run] would create or reuse dataset repo {args.repo_id}")
    else:
        api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    root_files = {
        "README.md": build_repo_readme(args.repo_id, selected),
        "dataset_manifest.json": json.dumps([asdict(spec) for spec in selected], indent=2) + "\n",
    }
    upload_generated_files(api, args.repo_id, "", root_files, args.dry_run)

    for spec in selected:
        dataset_files = {
            "README.md": build_dataset_readme(spec),
            "dataset_metadata.json": json.dumps(asdict(spec), indent=2) + "\n",
        }
        upload_generated_files(api, args.repo_id, spec.path_in_repo, dataset_files, args.dry_run)
        upload_dataset(api, args.repo_id, spec, args.dry_run)

    print("done")


if __name__ == "__main__":
    main()
