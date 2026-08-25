#!/usr/bin/env python3
"""Derive frozen forget01 factorial data from the nested forget05 prefix.

The script is deliberately fail-closed.  It first proves that all 40 immutable
C11 question/answer pairs are byte-identical to the canonical
``forget01_perturbed`` split.  Only then does it copy the first two author
blocks, rewrite source-id prefixes, and emit new versioned files.  Existing
outputs are reused only when their bytes already match the derived content;
they are never overwritten.
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


def rewrite_source_ids(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(SOURCE_SPLIT + "-", TARGET_SPLIT + "-")
    if isinstance(value, list):
        return [rewrite_source_ids(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_source_ids(item) for key, item in value.items()}
    return value


def derive(
    by_index: dict[int, dict[str, Any]], source_path: Path, label: str
) -> list[dict[str, Any]]:
    source_hash = sha256_file(source_path)
    output = []
    for index in range(TARGET_ROWS):
        row = rewrite_source_ids(by_index[index])
        generation = row.get("generation")
        if not isinstance(generation, dict):
            generation = {}
            row["generation"] = generation
        generation["forget01_derivation"] = {
            "method": "verified-nested-prefix-v1",
            "source_split": SOURCE_SPLIT,
            "target_split": TARGET_SPLIT,
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

    for index in range(TARGET_ROWS):
        expected = reference[index]
        v512_c11 = qa(v512[index]["cells"]["C11"], f"V5.12 index {index}")
        full_c11 = qa(fullanswer[index]["cells"]["C11"], f"FullAnswer index {index}")
        if v512_c11 != expected or full_c11 != expected:
            raise ValueError(
                f"C11 mismatch at index {index}; forget01 is not an exact nested prefix"
            )

    v512_rows = derive(v512, args.v512_source, "V5.12")
    full_rows = derive(fullanswer, args.fullanswer_source, "FullAnswer")
    v512_payload = jsonl_bytes(v512_rows)
    full_payload = jsonl_bytes(full_rows)
    v512_status = write_new_or_reuse(args.v512_output, v512_payload)
    full_status = write_new_or_reuse(args.fullanswer_output, full_payload)

    manifest = {
        "design": "forget01-verified-nested-prefix-v1",
        "source_split": SOURCE_SPLIT,
        "target_split": TARGET_SPLIT,
        "rows": TARGET_ROWS,
        "blocks": TARGET_ROWS // BLOCK_SIZE,
        "block_size": BLOCK_SIZE,
        "c11_exact_match": True,
        "content_cells_changed": False,
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
        "forget01 nested-prefix gate OK: "
        f"rows={TARGET_ROWS} blocks={TARGET_ROWS // BLOCK_SIZE} "
        "C11=byte-identical content_cells_changed=0"
    )
    print(f"V5.12 output: {v512_status} {args.v512_output}")
    print(f"FullAnswer output: {full_status} {args.fullanswer_output}")
    print(f"Derivation manifest: {manifest_status} {args.manifest_output}")


if __name__ == "__main__":
    main()
