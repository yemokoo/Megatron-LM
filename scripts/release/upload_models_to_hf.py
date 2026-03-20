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
class ModelSpec:
    key: str
    title: str
    source_dir: str
    iteration: int
    summary: str

    @property
    def iter_dir_name(self) -> str:
        return f"iter_{self.iteration:07d}"

    @property
    def abs_source_dir(self) -> Path:
        return ROOT / self.source_dir

    @property
    def abs_iter_dir(self) -> Path:
        return self.abs_source_dir / self.iter_dir_name


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="base_wiki_a",
        title="Base Wiki A",
        source_dir=".local/weights/continual-stage-A/stage-a-local-fp32-20260310-103742",
        iteration=1800,
        summary="Wiki-only base model used as the A-side reference checkpoint.",
    ),
    ModelSpec(
        key="base_code_b",
        title="Base Code B",
        source_dir=".local/weights/continual-stage-B/stage-b-first-local-fp32-20260312-043202",
        iteration=1800,
        summary="Code-only base model used as the B-side reference checkpoint.",
    ),
    ModelSpec(
        key="a_to_b_unfreeze",
        title="A to B Shared Unfreeze",
        source_dir=".local/weights/continual-stage-A-to_B/stage-b-resume-local-fp32-20260311-171347",
        iteration=1800,
        summary="Continual A -> B run with shared trunk unfrozen.",
    ),
    ModelSpec(
        key="a_to_b_freeze",
        title="A to B Shared Freeze",
        source_dir=".local/weights/continual-stage-A-to-B-new-only/stage-b-new-expert-router-only-local-fp32-20260317-113542",
        iteration=1800,
        summary="Continual A -> B run with shared trunk frozen and new experts/router only.",
    ),
    ModelSpec(
        key="b_to_a_unfreeze",
        title="B to A Shared Unfreeze",
        source_dir=".local/weights/continual-stage-B-to-A/stage-a-after-b-local-fp32-20260312-175645",
        iteration=1800,
        summary="Continual B -> A run with shared trunk unfrozen.",
    ),
    ModelSpec(
        key="b_to_a_freeze",
        title="B to A Shared Freeze",
        source_dir=".local/weights/continual-stage-B-to-A-new-only/stage-a-after-b-new-expert-router-only-local-fp32-20260315-135509",
        iteration=1800,
        summary="Continual B -> A run with shared trunk frozen and new experts/router only.",
    ),
    ModelSpec(
        key="seven_expert_wiki_a",
        title="7-Expert Wiki A",
        source_dir=".local/weights/continual-stage-A-7experts-resume-local/stage-a-7experts-resume-local-fp32-gpu0123-r1",
        iteration=1800,
        summary="7-expert wiki model after the resumed A-stage training run.",
    ),
    ModelSpec(
        key="seven_expert_code_after_wiki",
        title="7-Expert Code After Wiki",
        source_dir=".local/weights/continual-stage-B-after-A-7experts-no-freeze-local/stage-b-after-a-7experts-no-freeze-local-fp32-gpu0123-r1",
        iteration=3600,
        summary="7-expert code-after-wiki continual model after the B stage.",
    ),
]


def build_repo_readme(repo_id: str, models: Iterable[ModelSpec]) -> str:
    lines = [
        f"# {repo_id}",
        "",
        "Megatron distributed checkpoints for the representative FLAME-MoE continual-learning runs.",
        "",
        "Each subfolder is a minimal load directory with:",
        "- `latest_checkpointed_iteration.txt` fixed to the exported iteration",
        "- exactly one `iter_XXXXXXXX/` checkpoint tree",
        "",
        "Representative models:",
        "",
        "| Folder | Checkpoint | Summary |",
        "| --- | --- | --- |",
    ]
    for spec in models:
        lines.append(f"| `{spec.key}` | `{spec.iter_dir_name}` | {spec.summary} |")
    lines.extend(
        [
            "",
            "This repo is meant for direct Megatron-style loading rather than Hugging Face Transformers conversion.",
            "",
            "Source code: https://github.com/YeMoKoo/FLAME-MoE",
        ]
    )
    return "\n".join(lines) + "\n"


def build_model_readme(spec: ModelSpec) -> str:
    return (
        f"# {spec.title}\n\n"
        f"- folder: `{spec.key}`\n"
        f"- exported checkpoint: `{spec.iter_dir_name}`\n"
        f"- source run: `{spec.source_dir}`\n"
        f"- summary: {spec.summary}\n\n"
        "This subfolder is a minimal Megatron load directory.\n"
    )


def ensure_sources(models: Iterable[ModelSpec]) -> None:
    missing: list[str] = []
    for spec in models:
        if not spec.abs_source_dir.is_dir():
            missing.append(f"missing run dir: {spec.abs_source_dir}")
        elif not spec.abs_iter_dir.is_dir():
            missing.append(f"missing iteration dir: {spec.abs_iter_dir}")
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
            repo_type="model",
            folder_path=str(tmp),
            path_in_repo=path_in_repo,
            commit_message=f"Add metadata for {path_in_repo or 'repo root'}",
        )


def upload_checkpoint(api: HfApi, repo_id: str, spec: ModelSpec, dry_run: bool) -> None:
    if dry_run:
        print(
            f"[dry-run] upload {spec.abs_iter_dir} -> {repo_id}/{spec.key}/{spec.iter_dir_name}"
        )
        return
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(spec.abs_iter_dir),
        path_in_repo=f"{spec.key}/{spec.iter_dir_name}",
        commit_message=f"Upload {spec.title} ({spec.iter_dir_name})",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload representative FLAME-MoE checkpoints to a single HF repo.")
    parser.add_argument("--repo-id", default="YeMoKoo/flamemoe")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[spec.key for spec in MODEL_SPECS],
        help="Subset of model keys to upload.",
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_keys = set(args.models)
    selected = [spec for spec in MODEL_SPECS if spec.key in selected_keys]
    if not selected:
        raise SystemExit("No matching models selected.")

    ensure_sources(selected)
    api = HfApi()

    if args.dry_run:
        print(f"[dry-run] would create or reuse repo {args.repo_id}")
    else:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    root_files = {
        "README.md": build_repo_readme(args.repo_id, selected),
        "export_manifest.json": json.dumps([asdict(spec) for spec in selected], indent=2) + "\n",
    }
    upload_generated_files(api, args.repo_id, "", root_files, args.dry_run)

    for spec in selected:
        model_files = {
            "README.md": build_model_readme(spec),
            "latest_checkpointed_iteration.txt": f"{spec.iteration}\n",
            "model_metadata.json": json.dumps(asdict(spec), indent=2) + "\n",
        }
        upload_generated_files(api, args.repo_id, spec.key, model_files, args.dry_run)
        upload_checkpoint(api, args.repo_id, spec, args.dry_run)

    print("done")


if __name__ == "__main__":
    main()
