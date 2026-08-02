#!/usr/bin/env python3
"""Restore and stratified-sample the Base64 OLMoE router-memory archive.

The downloaded corpus is already mixed to the desired *token* ratios.  To draw a
fixed number of documents without turning those token ratios into document
ratios, this script allocates documents in proportion to each domain's document
count, samples uniformly within each domain, and reports the realized token mix.

The tar archive is read as a stream and is never extracted in full.  Outputs are
created atomically and existing paths are never overwritten.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tarfile
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DOMAIN_ORDER = ("dclm", "math", "flan", "pes2o", "wiki", "stackexchange")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore the OLMoE Base64 archive and sample replay documents."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/OLMOE0125sampling"),
        help="Directory containing part_*.txt, manifest.json, and sample_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/OLMOE0125sampling_5000_seed1234"),
        help="New directory for sample_5000.jsonl and sampling_manifest.json.",
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Restored tar.gz path (default: INPUT_DIR/archive name from manifest).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the document allocation and exit without writing anything.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def largest_remainder_allocation(total: int, weights: Mapping[str, int]) -> Dict[str, int]:
    if total < 1:
        raise ValueError("--num-samples must be positive")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("Domain document counts must be positive")

    quotas = {domain: total * weights[domain] / weight_sum for domain in DOMAIN_ORDER}
    allocation = {domain: math.floor(quotas[domain]) for domain in DOMAIN_ORDER}
    remaining = total - sum(allocation.values())
    ranked = sorted(
        DOMAIN_ORDER,
        key=lambda domain: (quotas[domain] - allocation[domain], -DOMAIN_ORDER.index(domain)),
        reverse=True,
    )
    for domain in ranked[:remaining]:
        allocation[domain] += 1
    return allocation


def print_plan(
    allocation: Mapping[str, int],
    metadata: Mapping[str, dict],
    num_samples: int,
) -> None:
    print("domain\tsource_docs\tsampled_docs\texpected_token_ratio", flush=True)
    for domain in DOMAIN_ORDER:
        print(
            f"{domain}\t{metadata[domain]['docs']}\t{allocation[domain]}\t"
            f"{100.0 * metadata[domain]['actual_ratio']:.6f}%",
            flush=True,
        )
    print(f"TOTAL\t{sum(metadata[d]['docs'] for d in DOMAIN_ORDER)}\t{num_samples}\t100.000000%")


def restore_archive(input_dir: Path, archive_path: Path, manifest: Mapping[str, object]) -> None:
    expected_size = int(manifest["size_bytes"])
    expected_sha = str(manifest["sha256"])
    expected_parts = int(manifest["n_parts"])
    chunk_size = int(manifest["chunk_size"])

    if archive_path.exists():
        actual_size = archive_path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Existing archive has size {actual_size}, expected {expected_size}; "
                f"refusing to overwrite {archive_path}."
            )
        actual_sha = sha256_file(archive_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Existing archive SHA-256 is {actual_sha}, expected {expected_sha}; "
                f"refusing to overwrite {archive_path}."
            )
        print(f"Reusing verified archive: {archive_path}", flush=True)
        return

    parts = sorted(input_dir.glob("part_*.txt"))
    if len(parts) != expected_parts:
        raise RuntimeError(f"Found {len(parts)} Base64 parts, expected {expected_parts}.")
    expected_names = [f"part_{index:04d}.txt" for index in range(expected_parts)]
    actual_names = [part.name for part in parts]
    if actual_names != expected_names:
        raise RuntimeError("Base64 part names are missing, duplicated, or out of sequence.")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_name(archive_path.name + ".partial")
    if partial_path.exists():
        raise FileExistsError(
            f"Partial archive already exists: {partial_path}. Inspect it before retrying."
        )

    digest = hashlib.sha256()
    bytes_written = 0
    try:
        with partial_path.open("xb") as output:
            for index, part_path in enumerate(parts):
                encoded = part_path.read_bytes()
                decoded = base64.b64decode(encoded, validate=True)
                if index + 1 < expected_parts and len(decoded) != chunk_size:
                    raise RuntimeError(
                        f"{part_path.name} decoded to {len(decoded)} bytes; expected {chunk_size}."
                    )
                output.write(decoded)
                digest.update(decoded)
                bytes_written += len(decoded)
                if index % 25 == 0 or index + 1 == expected_parts:
                    print(
                        f"Restoring archive: {index + 1}/{expected_parts} parts "
                        f"({bytes_written / 1024**3:.3f} GiB)",
                        flush=True,
                    )
            output.flush()
            os.fsync(output.fileno())

        if bytes_written != expected_size:
            raise RuntimeError(f"Restored {bytes_written} bytes; expected {expected_size}.")
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"Restored SHA-256 is {actual_sha}; expected {expected_sha}.")
        os.replace(partial_path, archive_path)
        print(f"Restored and verified archive: {archive_path}", flush=True)
    except Exception:
        print(f"Restore failed; partial data retained at {partial_path}", file=sys.stderr)
        raise


def domain_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def selected_indices(
    allocation: Mapping[str, int], metadata: Mapping[str, dict], seed: int
) -> Dict[str, set]:
    result = {}
    for domain in DOMAIN_ORDER:
        doc_count = int(metadata[domain]["docs"])
        sample_count = int(allocation[domain])
        if sample_count > doc_count:
            raise ValueError(
                f"Cannot sample {sample_count} {domain} documents from {doc_count} without replacement."
            )
        rng = random.Random(domain_seed(seed, domain))
        result[domain] = set(rng.sample(range(doc_count), sample_count))
    return result


def sample_archive(
    archive_path: Path,
    allocation: Mapping[str, int],
    metadata: Mapping[str, dict],
    seed: int,
) -> List[dict]:
    wanted = selected_indices(allocation, metadata, seed)
    sampled: Dict[str, List[dict]] = {domain: [] for domain in DOMAIN_ORDER}
    seen_domains = set()

    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            domain = Path(member.name).stem
            if domain not in wanted:
                continue
            if domain in seen_domains:
                raise RuntimeError(f"Archive contains duplicate {domain}.jsonl members.")
            seen_domains.add(domain)
            selected = wanted[domain]
            line_count = 0
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {member.name} from archive.")
            with extracted:
                for line_count, raw_line in enumerate(extracted, start=1):
                    source_index = line_count - 1
                    if source_index not in selected:
                        continue
                    record = json.loads(raw_line)
                    if record.get("domain") != domain:
                        raise RuntimeError(
                            f"{member.name} record {source_index} says domain={record.get('domain')!r}."
                        )
                    if not isinstance(record.get("text"), str) or not isinstance(
                        record.get("n_tokens"), int
                    ):
                        raise RuntimeError(
                            f"{member.name} record {source_index} has an invalid schema."
                        )
                    record["source_index"] = source_index
                    sampled[domain].append(record)

            expected_docs = int(metadata[domain]["docs"])
            if line_count != expected_docs:
                raise RuntimeError(
                    f"{member.name} contains {line_count} records; metadata says {expected_docs}."
                )
            if len(sampled[domain]) != allocation[domain]:
                raise RuntimeError(
                    f"Sampled {len(sampled[domain])} {domain} records; expected {allocation[domain]}."
                )
            print(
                f"Sampled {len(sampled[domain])}/{line_count} documents from {domain}",
                flush=True,
            )

    missing = set(DOMAIN_ORDER) - seen_domains
    if missing:
        raise RuntimeError(f"Archive is missing domain files: {sorted(missing)}")

    combined = [record for domain in DOMAIN_ORDER for record in sampled[domain]]
    random.Random(seed).shuffle(combined)
    return combined


def write_outputs(
    output_dir: Path,
    records: Sequence[dict],
    allocation: Mapping[str, int],
    metadata: Mapping[str, dict],
    source_manifest: Mapping[str, object],
    archive_path: Path,
    seed: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if partial_dir.exists():
        raise FileExistsError(f"Partial output directory already exists: {partial_dir}")
    partial_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir()

    try:
        sample_path = partial_dir / "sample_5000.jsonl"
        token_totals = {domain: 0 for domain in DOMAIN_ORDER}
        doc_totals = {domain: 0 for domain in DOMAIN_ORDER}
        with sample_path.open("x", encoding="utf-8") as handle:
            for record in records:
                domain = record["domain"]
                token_totals[domain] += int(record["n_tokens"])
                doc_totals[domain] += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_tokens = sum(token_totals.values())
        domains = {}
        for domain in DOMAIN_ORDER:
            actual_ratio = token_totals[domain] / total_tokens if total_tokens else 0.0
            target_ratio = float(metadata[domain]["actual_ratio"])
            domains[domain] = {
                "documents": doc_totals[domain],
                "target_documents": allocation[domain],
                "tokens": token_totals[domain],
                "actual_token_ratio": actual_ratio,
                "target_token_ratio": target_ratio,
                "deviation_percentage_points": 100.0 * (actual_ratio - target_ratio),
            }

        result_manifest = {
            "format": "olmoe_router_replay_raw_v1",
            "source_archive": str(archive_path),
            "source_archive_sha256": source_manifest["sha256"],
            "source_dataset": source_manifest["dataset"],
            "tokenizer": metadata["tokenizer"],
            "sampling": "stratified_uniform_documents_without_replacement",
            "seed": seed,
            "documents": len(records),
            "tokens": total_tokens,
            "sample_file": sample_path.name,
            "sample_file_sha256": sha256_file(sample_path),
            "domains": domains,
        }
        with (partial_dir / "sampling_manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(result_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        os.replace(partial_dir, output_dir)
    except Exception:
        print(f"Sampling failed; partial output retained at {partial_dir}", file=sys.stderr)
        raise

    print("\ndomain\tdocuments\ttokens\tactual_ratio\ttarget_ratio\tdeviation_pp")
    for domain in DOMAIN_ORDER:
        row = domains[domain]
        print(
            f"{domain}\t{row['documents']}\t{row['tokens']}\t"
            f"{100.0 * row['actual_token_ratio']:.6f}%\t"
            f"{100.0 * row['target_token_ratio']:.6f}%\t"
            f"{row['deviation_percentage_points']:+.6f}"
        )
    print(f"TOTAL\t{len(records)}\t{total_tokens}\t100.000000%\t100.000000%\t+0.000000")
    print(f"\nSaved sample and manifest to: {output_dir}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_manifest = load_json(input_dir / "manifest.json")
    metadata = load_json(input_dir / "sample_metadata.json")

    missing_domains = [domain for domain in DOMAIN_ORDER if domain not in metadata]
    if missing_domains:
        raise RuntimeError(f"sample_metadata.json is missing domains: {missing_domains}")
    allocation = largest_remainder_allocation(
        args.num_samples, {domain: int(metadata[domain]["docs"]) for domain in DOMAIN_ORDER}
    )
    print_plan(allocation, metadata, args.num_samples)
    if args.plan_only:
        return

    archive_path = (
        args.archive_path.resolve()
        if args.archive_path is not None
        else input_dir / str(source_manifest["archive"])
    )
    restore_archive(input_dir, archive_path, source_manifest)
    records = sample_archive(archive_path, allocation, metadata, args.seed)
    if len(records) != args.num_samples:
        raise RuntimeError(f"Sampled {len(records)} records; expected {args.num_samples}.")
    write_outputs(
        output_dir,
        records,
        allocation,
        metadata,
        source_manifest,
        archive_path,
        args.seed,
    )


if __name__ == "__main__":
    main()
