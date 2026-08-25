#!/usr/bin/env python3
"""Merge exact reused and newly generated forget10 partitions to 400 rows."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SPLIT = "forget10_perturbed"
ROWS = 400
BLOCK_SIZE = 20
ID_RE = re.compile(r"-(\d+)$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_reference(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
        result = []
        for row in rows:
            cell = row.get("cells", {}).get("C11", row)
            result.append((cell["question"], cell["answer"]))
        return result
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row[0], row[1]) for row in csv.reader(handle)]


def index(row: dict[str, Any]) -> int:
    match = ID_RE.search(str(row.get("source_id", "")))
    if match is None:
        raise ValueError(f"invalid source_id: {row.get('source_id')!r}")
    return int(match.group(1))


def merge_rows(
    reused_path: Path, new_path: Path, reference: list[tuple[str, str]], label: str
) -> list[dict[str, Any]]:
    reused, new = load_jsonl(reused_path), load_jsonl(new_path)
    if len(reused) != 200 or len(new) != 200:
        raise ValueError(f"{label}: partitions must be 200 + 200 rows")
    combined: dict[int, dict[str, Any]] = {}
    for partition, rows in (("reused", reused), ("new", new)):
        for row in rows:
            row_index = index(row)
            if row_index in combined:
                raise ValueError(f"{label}: overlapping row {row_index}")
            expected_id = f"{SPLIT}-{row_index:05d}"
            if row.get("source_id") != expected_id:
                raise ValueError(f"{label}: non-canonical id {row.get('source_id')}")
            if int(row.get("block_id", -1)) != row_index // BLOCK_SIZE:
                raise ValueError(f"{label}: wrong block for {expected_id}")
            cells = row.get("cells")
            if not isinstance(cells, dict) or set(cells) != {"C11", "C01", "C10", "C00"}:
                raise ValueError(f"{label}: incomplete cells for {expected_id}")
            c11 = cells["C11"]
            if (c11.get("question"), c11.get("answer")) != reference[row_index]:
                raise ValueError(f"{label}: C11 mismatch for {expected_id}")
            result = copy.deepcopy(row)
            result.setdefault("generation", {})["forget10_full400_assembly"] = {
                "version": "forget10-incremental-full400-v1",
                "partition": partition,
                "source_artifact": str(
                    (reused_path if partition == "reused" else new_path).resolve()
                ),
                "source_artifact_sha256": sha256(
                    reused_path if partition == "reused" else new_path
                ),
                "causal_cells_edited_by_merge": False,
            }
            combined[row_index] = result
    if set(combined) != set(range(ROWS)):
        raise ValueError(
            f"{label}: coverage mismatch missing={sorted(set(range(ROWS))-set(combined))[:20]}"
        )
    return [combined[row_index] for row_index in range(ROWS)]


def merge_profiles(reused_path: Path, new_path: Path, data_digest: str) -> dict:
    sources = [
        json.loads(reused_path.read_text(encoding="utf-8")),
        json.loads(new_path.read_text(encoding="utf-8")),
    ]
    by_block: dict[int, dict] = {}
    for payload in sources:
        for raw in payload.get("profiles", []):
            profile = copy.deepcopy(raw)
            block_id = int(profile.get("block_id", -1))
            if block_id in by_block:
                raise ValueError(f"duplicate profile block {block_id}")
            ledger = profile.get("fact_ledger")
            if not isinstance(ledger, list) or len(ledger) != BLOCK_SIZE:
                raise ValueError(f"invalid profile ledger for block {block_id}")
            by_block[block_id] = profile
    if set(by_block) != set(range(ROWS // BLOCK_SIZE)):
        raise ValueError(f"profile coverage mismatch: {sorted(by_block)}")
    return {
        "design_version": "tofu-author-pairbudget-v5.12-forget10-full400-v1",
        "assembly_version": "forget10-incremental-full400-v1",
        "split": SPLIT,
        "seed": 42,
        "selected_block_ids": list(range(ROWS // BLOCK_SIZE)),
        "profiles": [by_block[block] for block in range(ROWS // BLOCK_SIZE)],
        "assembly": {
            "reused_profiles": str(reused_path.resolve()),
            "reused_profiles_sha256": sha256(reused_path),
            "new_profiles": str(new_path.resolve()),
            "new_profiles_sha256": sha256(new_path),
            "full_data_sha256": data_digest,
            "reused_rows": 200,
            "new_rows": 200,
            "causal_cells_edited_by_merge": False,
            "human_review_status": "pending",
        },
    }


def serialise_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in rows
    )


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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reused-v512", type=Path, required=True)
    parser.add_argument("--new-v512", type=Path, required=True)
    parser.add_argument("--reused-v512-profiles", type=Path, required=True)
    parser.add_argument("--new-v512-profiles", type=Path, required=True)
    parser.add_argument("--reused-fullanswer", type=Path, required=True)
    parser.add_argument("--new-fullanswer", type=Path, required=True)
    parser.add_argument("--v512-output", type=Path, required=True)
    parser.add_argument("--v512-profiles-output", type=Path, required=True)
    parser.add_argument("--fullanswer-output", type=Path, required=True)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    if len(reference) != ROWS:
        raise ValueError(f"forget10 reference must contain {ROWS} rows")
    v512 = merge_rows(args.reused_v512, args.new_v512, reference, "V5.12")
    fullanswer = merge_rows(
        args.reused_fullanswer, args.new_fullanswer, reference, "FullAnswer"
    )
    v512_payload = serialise_jsonl(v512)
    v512_digest = hashlib.sha256(v512_payload).hexdigest()
    profiles = merge_profiles(
        args.reused_v512_profiles, args.new_v512_profiles, v512_digest
    )
    outputs = {
        args.v512_output: v512_payload,
        args.v512_profiles_output: (
            json.dumps(profiles, ensure_ascii=False, indent=2) + "\n"
        ).encode(),
        args.fullanswer_output: serialise_jsonl(fullanswer),
    }
    for path, payload in outputs.items():
        print(f"{write_new_or_reuse(path, payload)} {path}")
    print(
        "forget10 full-400 hard gate OK: rows=400 unique=400 blocks=20 "
        "cells=1600 reused=200 new=200 C11=exact old_overwrite=false"
    )


if __name__ == "__main__":
    main()
