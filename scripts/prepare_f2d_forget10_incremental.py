#!/usr/bin/env python3
"""Prepare an append-only forget10 expansion from frozen forget05 artifacts.

The canonical forget10 split contains 400 rows.  This script locates every
frozen forget05 C11 question/answer by exact, unique matching, migrates those
200 already-approved rows to their canonical forget10 ids, and emits the exact
200-row complement for new generation.  It never edits causal cell content and
never overwrites a non-identical artifact.
"""

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
from typing import Any, Iterable


SOURCE_SPLIT = "forget05_perturbed"
TARGET_SPLIT = "forget10_perturbed"
SOURCE_ROWS = 200
TARGET_ROWS = 400
BLOCK_SIZE = 20
NAME_PATTERN = re.compile(
    r"[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*"
    r"(?:\s+(?:[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*|al|Al|bin|Ben|de|del|van|von)){1,5}"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def qa(cell: Any, label: str) -> tuple[str, str]:
    if not isinstance(cell, dict):
        raise ValueError(f"{label}: missing C11 object")
    question, answer = cell.get("question"), cell.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError(f"{label}: C11 question/answer must be strings")
    return question, answer


def load_reference(path: Path) -> list[tuple[str, str]]:
    if path.suffix == ".jsonl":
        rows = load_jsonl(path)
        return [
            qa(row.get("cells", {}).get("C11", row), f"reference {index}")
            for index, row in enumerate(rows)
        ]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if any(len(row) < 2 for row in rows):
        raise ValueError("reference CSV must have question and answer columns")
    return [(row[0], row[1]) for row in rows]


def validate_source(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if len(rows) != SOURCE_ROWS:
        raise ValueError(f"{label}: expected {SOURCE_ROWS} rows, got {len(rows)}")
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.startswith(SOURCE_SPLIT + "-"):
            raise ValueError(f"{label}: invalid source_id {source_id!r}")
        try:
            source_index = int(source_id.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{label}: invalid source index in {source_id!r}") from exc
        if source_index in by_index:
            raise ValueError(f"{label}: duplicate {source_id}")
        by_index[source_index] = row
        if set(row.get("cells", {})) != {"C11", "C01", "C10", "C00"}:
            raise ValueError(f"{label}: incomplete cells for {source_id}")
    if set(by_index) != set(range(SOURCE_ROWS)):
        raise ValueError(f"{label}: source coverage must be 0..{SOURCE_ROWS - 1}")
    return [by_index[index] for index in range(SOURCE_ROWS)]


def exact_mapping(
    source_rows: list[dict[str, Any]], reference: list[tuple[str, str]], label: str
) -> list[int]:
    grouped: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(source_rows):
        grouped.setdefault(qa(row["cells"]["C11"], f"{label} {index}"), []).append(index)
    target_indices = []
    for source_index, row in enumerate(source_rows):
        pair = qa(row["cells"]["C11"], f"{label} {source_index}")
        matches = [index for index, expected in enumerate(reference) if expected == pair]
        if len(grouped[pair]) != 1 or len(matches) != 1:
            raise ValueError(
                f"{label}: C11 must map uniquely for source index {source_index}; "
                f"source_matches={grouped[pair]} target_matches={matches}"
            )
        target_indices.append(matches[0])
    if len(set(target_indices)) != SOURCE_ROWS:
        raise ValueError(f"{label}: exact mapping is not one-to-one")
    return target_indices


def rewrite_ids(value: Any, id_mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        result = value
        for old, new in id_mapping.items():
            result = result.replace(old, new)
        return result
    if isinstance(value, list):
        return [rewrite_ids(item, id_mapping) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_ids(item, id_mapping) for key, item in value.items()}
    return value


def rewrite_profile_ids(value: Any, source_block: int, target_block: int) -> Any:
    """Remap profile-id metadata without touching questions, answers, or facts."""
    if isinstance(value, list):
        return [rewrite_profile_ids(item, source_block, target_block) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if "profile_id" in key and isinstance(item, str):
                item = item.replace(
                    f"{SOURCE_SPLIT}:author-", f"{TARGET_SPLIT}:author-"
                ).replace(f"block-{source_block:02d}", f"block-{target_block:02d}")
            else:
                item = rewrite_profile_ids(item, source_block, target_block)
            result[key] = item
        return result
    return value


def derive_rows(
    rows: list[dict[str, Any]], mapping: list[int], source_path: Path, label: str
) -> list[dict[str, Any]]:
    id_mapping = {
        f"{SOURCE_SPLIT}-{source_index:05d}": f"{TARGET_SPLIT}-{target_index:05d}"
        for source_index, target_index in enumerate(mapping)
    }
    result = []
    for source_index, target_index in enumerate(mapping):
        row = rewrite_ids(copy.deepcopy(rows[source_index]), id_mapping)
        row = rewrite_profile_ids(
            row, source_index // BLOCK_SIZE, target_index // BLOCK_SIZE
        )
        row["source_id"] = f"{TARGET_SPLIT}-{target_index:05d}"
        row["block_id"] = target_index // BLOCK_SIZE
        row.setdefault("generation", {})["forget10_incremental_derivation"] = {
            "method": "verified-exact-superset-v1",
            "source_split": SOURCE_SPLIT,
            "target_split": TARGET_SPLIT,
            "source_index": source_index,
            "target_index": target_index,
            "source_sha256": sha256(source_path),
            "source_label": label,
            "content_cells_changed": False,
        }
        result.append(row)
    return sorted(result, key=lambda row: row["source_id"])


def derive_profiles(
    source_path: Path, mapping: list[int], source_data: Path
) -> dict[str, Any]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != SOURCE_ROWS // BLOCK_SIZE:
        raise ValueError("forget05 profiles must contain ten author blocks")
    source_to_target_block: dict[int, int] = {}
    for source_index, target_index in enumerate(mapping):
        source_block, target_block = source_index // BLOCK_SIZE, target_index // BLOCK_SIZE
        previous = source_to_target_block.setdefault(source_block, target_block)
        if previous != target_block:
            raise ValueError("exact mapping splits a frozen author block")
    if len(set(source_to_target_block.values())) != len(source_to_target_block):
        raise ValueError("two frozen author blocks map to one forget10 block")
    id_mapping = {
        f"{SOURCE_SPLIT}-{source_index:05d}": f"{TARGET_SPLIT}-{target_index:05d}"
        for source_index, target_index in enumerate(mapping)
    }
    migrated = []
    for raw in profiles:
        profile = rewrite_ids(copy.deepcopy(raw), id_mapping)
        source_block = int(raw["block_id"])
        target_block = source_to_target_block[source_block]
        profile = rewrite_profile_ids(profile, source_block, target_block)
        profile["block_id"] = target_block
        migrated.append(profile)
    result = copy.deepcopy(payload)
    result["split"] = TARGET_SPLIT
    result["selected_block_ids"] = sorted(source_to_target_block.values())
    result["profiles"] = sorted(migrated, key=lambda item: int(item["block_id"]))
    result["forget10_incremental_derivation"] = {
        "method": "verified-exact-superset-v1",
        "source_profiles": str(source_path.resolve()),
        "source_profiles_sha256": sha256(source_path),
        "source_data": str(source_data.resolve()),
        "source_data_sha256": sha256(source_data),
        "content_facts_changed": False,
    }
    return result


def author_manifest(reference: list[tuple[str, str]]) -> dict[str, Any]:
    authors = []
    for block_id in range(TARGET_ROWS // BLOCK_SIZE):
        block = reference[block_id * BLOCK_SIZE : (block_id + 1) * BLOCK_SIZE]
        texts = [f"{question} {answer}" for question, answer in block]
        first_answer = block[0][1]
        candidates = []
        for match in NAME_PATTERN.finditer(first_answer):
            candidate = match.group(0).strip(" .,:;")
            coverage = sum(candidate.casefold() in text.casefold() for text in texts)
            if coverage >= BLOCK_SIZE - 1:
                candidates.append((coverage, len(candidate.split()), len(candidate), candidate))
        if not candidates:
            raise ValueError(f"cannot derive a stable author name for block {block_id}")
        canonical = max(candidates)[-1]
        aliases: list[str] = []
        mismatched = [text for text in texts if canonical.casefold() not in text.casefold()]
        if mismatched:
            tokens = canonical.split()
            alias_candidates = [
                " ".join(tokens[start:])
                for start in range(1, len(tokens))
            ] + tokens
            aliases = [
                alias for alias in alias_candidates
                if all(alias.casefold() in text.casefold() for text in mismatched)
            ][:1]
        bindings = [canonical, *aliases]
        if any(not any(binding.casefold() in text.casefold() for binding in bindings) for text in texts):
            raise ValueError(f"author bindings do not cover block {block_id}: {canonical}")
        item: dict[str, Any] = {"block_id": block_id, "canonical_name": canonical}
        if aliases:
            item["aliases"] = aliases
        authors.append(item)
    return {
        "dataset": "locuslab/TOFU",
        "split": TARGET_SPLIT,
        "block_size": BLOCK_SIZE,
        "authors": authors,
    }


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
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
    parser.add_argument("--v512-source", type=Path, required=True)
    parser.add_argument("--v512-profiles-source", type=Path, required=True)
    parser.add_argument("--fullanswer-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    reference = load_reference(args.reference)
    if len(reference) != TARGET_ROWS:
        raise ValueError(f"forget10 reference must contain {TARGET_ROWS} rows")
    v512 = validate_source(load_jsonl(args.v512_source), "V5.12")
    fullanswer = validate_source(load_jsonl(args.fullanswer_source), "FullAnswer")
    v512_mapping = exact_mapping(v512, reference, "V5.12")
    full_mapping = exact_mapping(fullanswer, reference, "FullAnswer")
    if v512_mapping != full_mapping:
        raise ValueError("V5.12 and FullAnswer disagree on exact forget10 mapping")
    mapped_blocks = sorted({target // BLOCK_SIZE for target in v512_mapping})
    if len(mapped_blocks) != SOURCE_ROWS // BLOCK_SIZE:
        raise ValueError(f"expected ten mapped author blocks, got {mapped_blocks}")
    missing_blocks = sorted(set(range(TARGET_ROWS // BLOCK_SIZE)) - set(mapped_blocks))
    missing_ids = [
        f"{TARGET_SPLIT}-{index:05d}"
        for block in missing_blocks
        for index in range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "reused_v512": args.output_dir / "forget10_v512_reused200_exact_v1.jsonl",
        "reused_v512_profiles": args.output_dir / "forget10_v512_reused200_exact_v1.profiles.json",
        "reused_fullanswer": args.output_dir / "forget10_fullanswer_reused200_exact_v1.jsonl",
        "manifest": args.output_dir / "tofu_forget10_author_blocks_exact_v1.json",
        "missing_ids": args.output_dir / "forget10_missing200_source_ids_v1.txt",
        "plan": args.output_dir / "forget10_incremental400_plan_v1.json",
    }
    reused_v512 = derive_rows(v512, v512_mapping, args.v512_source, "V5.12")
    reused_full = derive_rows(fullanswer, full_mapping, args.fullanswer_source, "FullAnswer")
    profiles = derive_profiles(
        args.v512_profiles_source, v512_mapping, args.v512_source
    )
    manifest = author_manifest(reference)
    plan = {
        "design": "forget10-incremental-from-forget05-v1",
        "source_split": SOURCE_SPLIT,
        "target_split": TARGET_SPLIT,
        "target_rows": TARGET_ROWS,
        "reused_rows": SOURCE_ROWS,
        "new_rows": TARGET_ROWS - SOURCE_ROWS,
        "block_size": BLOCK_SIZE,
        "mapped_blocks": mapped_blocks,
        "missing_blocks": missing_blocks,
        "mapping": {
            f"{SOURCE_SPLIT}-{source:05d}": f"{TARGET_SPLIT}-{target:05d}"
            for source, target in enumerate(v512_mapping)
        },
        "c11_exact_match": True,
        "reused_content_cells_changed": False,
        "old_artifacts_overwritten": False,
    }
    payloads = {
        outputs["reused_v512"]: jsonl_bytes(reused_v512),
        outputs["reused_v512_profiles"]: json_bytes(profiles),
        outputs["reused_fullanswer"]: jsonl_bytes(reused_full),
        outputs["manifest"]: json_bytes(manifest),
        outputs["missing_ids"]: ("\n".join(missing_ids) + "\n").encode(),
        outputs["plan"]: json_bytes(plan),
    }
    for path, payload in payloads.items():
        print(f"{write_new_or_reuse(path, payload)} {path}")
    print(
        "forget10 incremental gate OK: "
        f"reused=200 new=200 mapped_blocks={mapped_blocks} "
        f"missing_blocks={missing_blocks} C11=exact old_overwrite=false"
    )


if __name__ == "__main__":
    main()
