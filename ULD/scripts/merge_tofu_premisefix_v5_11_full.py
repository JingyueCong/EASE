#!/usr/bin/env python3
"""Merge approved V5.10 blocks 1/4 with V5.11 complementary blocks.

The merge is metadata-only: no causal cell is edited.  It verifies disjoint
canonical partitions, the expected design of each partition, complete 200-row
coverage, profile coverage, and deterministic CIRU validity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


APPROVED_DESIGN = "tofu-author-pairrepair-v5.10"
REMAINDER_DESIGN = "tofu-author-premisefix-v5.11"
DESIGN_VERSION = "tofu-author-premisefix-v5.11-full-hybrid"
ASSEMBLY_VERSION = "tofu-author-premisefix-v5.11-full-merge-v1"
SOURCE_ID_PATTERN = re.compile(r"-(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing JSONL: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def source_index(row: Mapping) -> int:
    match = SOURCE_ID_PATTERN.search(str(row.get("source_id", "")))
    if match is None:
        raise ValueError(f"invalid source_id: {row.get('source_id')!r}")
    return int(match.group(1))


def validate_partition(
    rows: Sequence[Mapping],
    expected_blocks: set[int],
    expected_design: str,
    block_size: int,
    label: str,
) -> dict[str, Mapping]:
    indexed: dict[str, Mapping] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if source_id in indexed:
            raise ValueError(f"{label} duplicates {source_id}")
        index = source_index(row)
        expected_block = index // block_size
        block_id = int(row.get("block_id", -1))
        if block_id != expected_block:
            raise ValueError(
                f"{label} {source_id} block_id={block_id}, "
                f"expected {expected_block}"
            )
        if block_id not in expected_blocks:
            raise ValueError(
                f"{label} {source_id} belongs to unexpected block {block_id}"
            )
        if row.get("design_version") != expected_design:
            raise ValueError(
                f"{label} {source_id} design={row.get('design_version')!r}, "
                f"expected {expected_design!r}"
            )
        cells = row.get("cells")
        if not isinstance(cells, Mapping) or set(cells) != {
            "C11", "C01", "C10", "C00"
        }:
            raise ValueError(f"{label} {source_id} has invalid cells")
        indexed[source_id] = row
    return indexed


def merge_records(
    approved: Sequence[Mapping],
    remainder: Sequence[Mapping],
    *,
    approved_blocks: Iterable[int],
    expected_units: int = 200,
    block_size: int = 20,
    approved_digest: str = "",
    remainder_digest: str = "",
) -> list[dict]:
    approved_set = set(approved_blocks)
    if expected_units % block_size:
        raise ValueError("expected_units must be divisible by block_size")
    all_blocks = set(range(expected_units // block_size))
    if not approved_set or not approved_set < all_blocks:
        raise ValueError("approved_blocks must be a non-empty proper subset")
    remainder_set = all_blocks - approved_set
    approved_by_id = validate_partition(
        approved, approved_set, APPROVED_DESIGN, block_size, "approved"
    )
    remainder_by_id = validate_partition(
        remainder, remainder_set, REMAINDER_DESIGN, block_size, "remainder"
    )
    overlap = set(approved_by_id) & set(remainder_by_id)
    if overlap:
        raise ValueError(f"partitions overlap: {sorted(overlap)[:5]}")
    combined = {**approved_by_id, **remainder_by_id}
    indices = {source_index(row) for row in combined.values()}
    expected_indices = set(range(expected_units))
    if indices != expected_indices:
        raise ValueError(
            "full coverage mismatch: "
            f"missing={sorted(expected_indices - indices)[:20]} "
            f"extra={sorted(indices - expected_indices)[:20]}"
        )
    if len(combined) != expected_units:
        raise ValueError(
            f"merged row count {len(combined)} != {expected_units}"
        )

    merged = []
    for row in sorted(combined.values(), key=source_index):
        result = copy.deepcopy(row)
        approved_row = int(result["block_id"]) in approved_set
        result.setdefault("generation", {})["full200_assembly"] = {
            "assembly_version": ASSEMBLY_VERSION,
            "full_design": DESIGN_VERSION,
            "partition": "approved_v5.10" if approved_row else "premisefix_v5.11",
            "source_design": (
                APPROVED_DESIGN if approved_row else REMAINDER_DESIGN
            ),
            "source_artifact_sha256": (
                approved_digest if approved_row else remainder_digest
            ),
            "causal_cells_edited_by_merge": False,
        }
        merged.append(result)
    return merged


def load_profiles(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing profiles artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("profiles"), list):
        raise ValueError(f"profiles artifact has no profiles list: {path}")
    return payload


def merge_profiles(
    approved: Mapping,
    remainder: Mapping,
    *,
    approved_blocks: set[int],
    approved_data: Path,
    remainder_data: Path,
    approved_digest: str,
    remainder_digest: str,
    output_digest: str,
) -> dict:
    if approved.get("design_version") != APPROVED_DESIGN:
        raise ValueError("approved profile artifact has wrong design")
    if remainder.get("design_version") != REMAINDER_DESIGN:
        raise ValueError("remainder profile artifact has wrong design")
    all_blocks = set(range(10))
    by_block: dict[int, dict] = {}
    for label, payload, expected in (
        ("approved", approved, approved_blocks),
        ("remainder", remainder, all_blocks - approved_blocks),
    ):
        for profile in payload["profiles"]:
            block_id = int(profile.get("block_id", -1))
            if block_id not in expected:
                raise ValueError(
                    f"{label} profiles contain unexpected block {block_id}"
                )
            if block_id in by_block:
                raise ValueError(f"duplicate profile block {block_id}")
            by_block[block_id] = copy.deepcopy(profile)
    if set(by_block) != all_blocks:
        raise ValueError(
            f"profile coverage mismatch: {sorted(all_blocks - set(by_block))}"
        )
    return {
        "design_version": DESIGN_VERSION,
        "assembly_version": ASSEMBLY_VERSION,
        "split": "forget05_perturbed",
        "seed": 42,
        "selected_block_ids": list(range(10)),
        "profiles": [by_block[index] for index in range(10)],
        "assembly": {
            "approved_blocks": sorted(approved_blocks),
            "premisefix_blocks": sorted(all_blocks - approved_blocks),
            "approved_design": APPROVED_DESIGN,
            "remainder_design": REMAINDER_DESIGN,
            "approved_data": str(approved_data.resolve()),
            "approved_data_sha256": approved_digest,
            "remainder_data": str(remainder_data.resolve()),
            "remainder_data_sha256": remainder_digest,
            "full_data_sha256": output_digest,
            "causal_cells_edited_by_merge": False,
            "full_human_review_status": "pending",
        },
    }


def load_ciru_module():
    path = Path(__file__).resolve().parent / "generate_tofu_author_premisefix_v5_11.py"
    spec = importlib.util.spec_from_file_location(
        "tofu_premisefix_full_merge_validator", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ciru.AUTHOR_LEVEL_DESIGNS.add(REMAINDER_DESIGN)
    return module.ciru


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-data", type=Path, required=True)
    parser.add_argument("--approved-profiles", type=Path, required=True)
    parser.add_argument("--remainder-data", type=Path, required=True)
    parser.add_argument("--remainder-profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    parser.add_argument("--approved-blocks", default="1,4")
    parser.add_argument("--expected-units", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=20)
    args = parser.parse_args()

    approved_blocks = {
        int(value) for value in args.approved_blocks.split(",") if value.strip()
    }
    approved_digest = sha256(args.approved_data)
    remainder_digest = sha256(args.remainder_data)
    rows = merge_records(
        load_jsonl(args.approved_data),
        load_jsonl(args.remainder_data),
        approved_blocks=approved_blocks,
        expected_units=args.expected_units,
        block_size=args.block_size,
        approved_digest=approved_digest,
        remainder_digest=remainder_digest,
    )
    ciru = load_ciru_module()
    errors = []
    for row in rows:
        failures = ciru.validate_ciru_unit(row)
        if failures:
            errors.append(f"{row['source_id']}: " + "; ".join(failures))
        row["audit"] = ciru.audit_ciru_unit(row)
    if errors:
        raise SystemExit(
            "deterministic merge validation failed: " + " | ".join(errors[:20])
        )

    write_jsonl(args.output, rows)
    output_digest = sha256(args.output)
    profiles = merge_profiles(
        load_profiles(args.approved_profiles),
        load_profiles(args.remainder_profiles),
        approved_blocks=approved_blocks,
        approved_data=args.approved_data,
        remainder_data=args.remainder_data,
        approved_digest=approved_digest,
        remainder_digest=remainder_digest,
        output_digest=output_digest,
    )
    write_json(args.profiles_output, profiles)
    print(
        f"design={DESIGN_VERSION} rows={len(rows)} blocks=10 "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
