#!/usr/bin/env python3
"""Derive frozen forget01 factorial data from an exact forget05 QA subset.

The script is deliberately fail-closed.  It locates every canonical
``forget01_perturbed`` C11 question/answer pair by an exact, unique match among
the 200 frozen forget05 C11 rows.  V5.12 and FullAnswer must independently
produce the same 40-row mapping.  Only then are those two author blocks copied,
renumbered to forget01 indices 0..39, and emitted under new versioned paths.
Existing outputs are reused only when their bytes already match the derived
content; they are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


SOURCE_SPLIT = "forget05_perturbed"
TARGET_SPLIT = "forget01_perturbed"
SOURCE_ROWS = 200
TARGET_ROWS = 40
BLOCK_SIZE = 20


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every JSONL record must be an object")
    return rows


def source_index(source_id: str) -> int:
    match = re.fullmatch(r"[^/]+-(\d{5})", str(source_id))
    if not match:
        raise ValueError(f"invalid source_id: {source_id!r}")
    return int(match.group(1))


def validate_source(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"{label}: expected {SOURCE_ROWS} rows, got {len(rows)}")
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.startswith(SOURCE_SPLIT + "-"):
            raise ValueError(f"{label}: unexpected source_id {source_id!r}")
        index = source_index(source_id)
        if index in by_index:
            raise ValueError(f"{label}: duplicate source index {index}")
        if set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"}:
            raise ValueError(f"{label}: {source_id} has incomplete factorial cells")
        by_index[index] = row
    if set(by_index) != set(range(SOURCE_ROWS)):
        raise ValueError(f"{label}: source coverage is not 0..{SOURCE_ROWS - 1}")
    return by_index


def qa(cell: Any, label: str) -> tuple[str, str]:
    if not isinstance(cell, dict):
        raise ValueError(f"{label}: C11 must be an object")
    question = cell.get("question")
    answer = cell.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError(f"{label}: C11 needs string question/answer")
    return question, answer


def load_reference(path: Path | None) -> list[tuple[str, str]]:
    if path is not None:
        rows = load_jsonl(path)
        result = []
        for index, row in enumerate(rows):
            if "cells" in row:
                result.append(qa(row["cells"].get("C11"), f"reference row {index}"))
            else:
                result.append(qa(row, f"reference row {index}"))
        return result

    from datasets import load_dataset

    dataset = load_dataset("locuslab/TOFU", TARGET_SPLIT)["train"]
    return [(str(row["question"]), str(row["answer"])) for row in dataset]


def rewrite_source_ids(value: Any, id_mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for source_id, target_id in id_mapping.items():
            result = result.replace(source_id, target_id)
        return result
    if isinstance(value, list):
        return [rewrite_source_ids(item, id_mapping) for item in value]
    if isinstance(value, dict):
        return {
            key: rewrite_source_ids(item, id_mapping) for key, item in value.items()
        }
    return value


def derive(
    by_index: dict[int, dict[str, Any]],
    mapping: list[int],
    source_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    source_hash = sha256_file(source_path)
    id_mapping = {
        f"{SOURCE_SPLIT}-{source_index_value:05d}": f"{TARGET_SPLIT}-{target_index:05d}"
        for target_index, source_index_value in enumerate(mapping)
    }
    output = []
    for target_index, source_index_value in enumerate(mapping):
        row = rewrite_source_ids(by_index[source_index_value], id_mapping)
        source_block = source_index_value // BLOCK_SIZE
        target_block = target_index // BLOCK_SIZE
        row["source_id"] = f"{TARGET_SPLIT}-{target_index:05d}"
        row["block_id"] = target_block
        generation = row.get("generation")
        if not isinstance(generation, dict):
            generation = {}
            row["generation"] = generation
        generation["forget01_derivation"] = {
            "method": "verified-exact-subset-v1",
            "source_split": SOURCE_SPLIT,
            "target_split": TARGET_SPLIT,
            "source_index": source_index_value,
            "target_index": target_index,
            "source_block": source_block,
            "target_block": target_block,
            "source_sha256": source_hash,
            "source_label": label,
            "content_cells_changed": False,
        }
        output.append(row)
    return output


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def write_new_or_reuse(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to overwrite non-matching existing artifact: {path}"
            )
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
    parser.add_argument("--v512-source", type=Path, required=True)
    parser.add_argument("--fullanswer-source", type=Path, required=True)
    parser.add_argument("--v512-output", type=Path, required=True)
    parser.add_argument("--fullanswer-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        help="Offline canonical forget01 QA JSONL; otherwise load locuslab/TOFU.",
    )
    args = parser.parse_args()

    v512 = validate_source(load_jsonl(args.v512_source), "V5.12")
    fullanswer = validate_source(load_jsonl(args.fullanswer_source), "FullAnswer")
    reference = load_reference(args.reference_jsonl)
    if len(reference) != TARGET_ROWS:
        raise ValueError(
            f"canonical {TARGET_SPLIT} must contain {TARGET_ROWS} rows, got {len(reference)}"
        )

    def unique_qa_index(
        rows: dict[int, dict[str, Any]], label: str
    ) -> dict[tuple[str, str], list[int]]:
        grouped: dict[tuple[str, str], list[int]] = {}
        for source_index_value, row in rows.items():
            pair = qa(row["cells"]["C11"], f"{label} index {source_index_value}")
            grouped.setdefault(pair, []).append(source_index_value)
        return grouped

    v512_qa = unique_qa_index(v512, "V5.12")
    full_qa = unique_qa_index(fullanswer, "FullAnswer")
    v512_mapping: list[int] = []
    full_mapping: list[int] = []
    for target_index, expected in enumerate(reference):
        v512_matches = v512_qa.get(expected, [])
        full_matches = full_qa.get(expected, [])
        if len(v512_matches) != 1 or len(full_matches) != 1:
            raise ValueError(
                "C11 exact subset match must be unique at forget01 index "
                f"{target_index}; V5.12={v512_matches} FullAnswer={full_matches}"
            )
        v512_mapping.append(v512_matches[0])
        full_mapping.append(full_matches[0])
    if v512_mapping != full_mapping:
        raise ValueError("V5.12 and FullAnswer disagree on the forget01 source mapping")
    if len(set(v512_mapping)) != TARGET_ROWS:
        raise ValueError("forget01 exact subset mapping is not one-to-one")
    mapped_blocks = sorted({index // BLOCK_SIZE for index in v512_mapping})
    if len(mapped_blocks) != TARGET_ROWS // BLOCK_SIZE:
        raise ValueError(
            f"forget01 mapping must contain two coherent author blocks, got {mapped_blocks}"
        )
    expected_block_mapping = [
        mapped_blocks[target_index // BLOCK_SIZE] for target_index in range(TARGET_ROWS)
    ]
    observed_block_mapping = [index // BLOCK_SIZE for index in v512_mapping]
    if observed_block_mapping != expected_block_mapping:
        raise ValueError(
            "forget01 ordering does not preserve two contiguous 20-row author blocks"
        )

    v512_rows = derive(v512, v512_mapping, args.v512_source, "V5.12")
    full_rows = derive(fullanswer, full_mapping, args.fullanswer_source, "FullAnswer")
    v512_payload = jsonl_bytes(v512_rows)
    full_payload = jsonl_bytes(full_rows)
    v512_status = write_new_or_reuse(args.v512_output, v512_payload)
    full_status = write_new_or_reuse(args.fullanswer_output, full_payload)

    manifest = {
        "design": "forget01-verified-exact-subset-v1",
        "source_split": SOURCE_SPLIT,
        "target_split": TARGET_SPLIT,
        "rows": TARGET_ROWS,
        "blocks": TARGET_ROWS // BLOCK_SIZE,
        "block_size": BLOCK_SIZE,
        "c11_exact_match": True,
        "content_cells_changed": False,
        "source_indices": v512_mapping,
        "source_blocks": mapped_blocks,
        "source_to_target_ids": {
            f"{SOURCE_SPLIT}-{source_index_value:05d}": f"{TARGET_SPLIT}-{target_index:05d}"
            for target_index, source_index_value in enumerate(v512_mapping)
        },
        "sources": {
            "v512": {
                "path": str(args.v512_source.resolve()),
                "sha256": sha256_file(args.v512_source),
            },
            "fullanswer": {
                "path": str(args.fullanswer_source.resolve()),
                "sha256": sha256_file(args.fullanswer_source),
            },
        },
        "outputs": {
            "v512": {
                "path": str(args.v512_output.resolve()),
                "sha256": sha256_bytes(v512_payload),
            },
            "fullanswer": {
                "path": str(args.fullanswer_output.resolve()),
                "sha256": sha256_bytes(full_payload),
            },
        },
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    manifest_status = write_new_or_reuse(args.manifest_output, manifest_payload)
    print(
        "forget01 exact-subset gate OK: "
        f"rows={TARGET_ROWS} blocks={TARGET_ROWS // BLOCK_SIZE} "
        f"source_blocks={mapped_blocks} C11=byte-identical content_cells_changed=0"
    )
    print(f"V5.12 output: {v512_status} {args.v512_output}")
    print(f"FullAnswer output: {full_status} {args.fullanswer_output}")
    print(f"Derivation manifest: {manifest_status} {args.manifest_output}")


if __name__ == "__main__":
    main()
