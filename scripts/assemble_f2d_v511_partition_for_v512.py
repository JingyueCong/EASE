#!/usr/bin/env python3
"""Metadata-only assembly of one 200-row V5.11 partition for V5.12."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


SOURCE_DESIGN = "tofu-author-premisefix-v5.11"
OUTPUT_DESIGN = "tofu-author-premisefix-v5.11-full-hybrid"
ASSEMBLY_VERSION = "tofu-author-premisefix-v5.11-partition-v1"
ID_RE = re.compile(r"-(\d+)$")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(row: dict) -> int:
    match = ID_RE.search(str(row.get("source_id", "")))
    if match is None:
        raise ValueError(f"invalid source_id: {row.get('source_id')!r}")
    return int(match.group(1))


def write_new_or_reuse(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-matching artifact: {path}")
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
        raise ValueError("V5.11 partition must contain 200 unique rows")
    block_ids = {source_index(row) // 20 for row in rows}
    if len(block_ids) != 10:
        raise ValueError(f"V5.11 partition must cover ten blocks: {sorted(block_ids)}")
    input_digest = digest(args.input)
    assembled = []
    for raw in sorted(rows, key=source_index):
        if raw.get("design_version") != SOURCE_DESIGN:
            raise ValueError(f"{raw.get('source_id')}: wrong V5.11 design")
        row = copy.deepcopy(raw)
        row.setdefault("generation", {})["full200_assembly"] = {
            "assembly_version": ASSEMBLY_VERSION,
            "full_design": OUTPUT_DESIGN,
            "partition": "forget10_new200_premisefix_v5.11",
            "source_design": SOURCE_DESIGN,
            "source_artifact_sha256": input_digest,
            "causal_cells_edited_by_merge": False,
        }
        assembled.append(row)
    data_payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in assembled
    ).encode()
    output_digest = hashlib.sha256(data_payload).hexdigest()

    profiles = json.loads(args.profiles.read_text(encoding="utf-8"))
    raw_profiles = profiles.get("profiles")
    if profiles.get("design_version") != SOURCE_DESIGN:
        raise ValueError("V5.11 profiles have wrong design")
    observed_blocks = {
        int(profile.get("block_id", -1)) for profile in raw_profiles or []
    }
    if observed_blocks != block_ids:
        raise ValueError(
            f"profile coverage mismatch: expected={sorted(block_ids)} "
            f"observed={sorted(observed_blocks)}"
        )
    result = copy.deepcopy(profiles)
    result["design_version"] = OUTPUT_DESIGN
    result["selected_block_ids"] = sorted(block_ids)
    result["assembly_version"] = ASSEMBLY_VERSION
    result["assembly"] = {
        "premisefix_blocks": sorted(block_ids),
        "remainder_design": SOURCE_DESIGN,
        "remainder_data": str(args.input.resolve()),
        "remainder_data_sha256": input_digest,
        "full_data_sha256": output_digest,
        "causal_cells_edited_by_merge": False,
        "full_human_review_status": "pending",
    }
    profiles_payload = (
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    print(f"{write_new_or_reuse(args.output, data_payload)} {args.output}")
    print(
        f"{write_new_or_reuse(args.profiles_output, profiles_payload)} "
        f"{args.profiles_output}"
    )
    print(
        f"V5.11 partition assembly OK: rows=200 blocks={sorted(block_ids)} "
        "causal_cells_edited=0"
    )


if __name__ == "__main__":
    main()
