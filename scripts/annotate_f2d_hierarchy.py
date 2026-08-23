#!/usr/bin/env python3
"""Add deterministic claim/evidence spans to audited F2D factorial cells.

The annotation is derived inside each matched 2x2 unit, so it does not use a
retain corpus or an external language model.  Token differences in C11/C01 and
C10/C00 define evidence spans; sentence/clause regions containing evidence
define claims.  The same representation extends to document segments later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


WORD_RE = re.compile(r"\w+(?:[’'-]\w+)*|[^\w\s]", re.UNICODE)
CLAUSE_BOUNDARY_RE = re.compile(r"(?:[.!?;]+(?:[\"'’”)]*)\s+)|(?:\n+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexical_tokens(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0).casefold(), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def merge_spans(spans: Iterable[Tuple[int, int]]) -> List[List[int]]:
    ordered = sorted((int(start), int(end)) for start, end in spans if end > start)
    merged: List[List[int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def paired_evidence_spans(
    left: str,
    right: str,
    fallback_full: bool = True,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Return answer-relative changed spans for a matched pair."""
    left_tokens = lexical_tokens(left)
    right_tokens = lexical_tokens(right)
    matcher = SequenceMatcher(
        None,
        [token for token, _, _ in left_tokens],
        [token for token, _, _ in right_tokens],
        autojunk=False,
    )
    left_spans: List[Tuple[int, int]] = []
    right_spans: List[Tuple[int, int]] = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            continue
        if i1 < i2:
            left_spans.append((left_tokens[i1][1], left_tokens[i2 - 1][2]))
        elif left_tokens:
            anchor = min(i1, len(left_tokens) - 1)
            left_spans.append((left_tokens[anchor][1], left_tokens[anchor][2]))
        if j1 < j2:
            right_spans.append((right_tokens[j1][1], right_tokens[j2 - 1][2]))
        elif right_tokens:
            anchor = min(j1, len(right_tokens) - 1)
            right_spans.append((right_tokens[anchor][1], right_tokens[anchor][2]))

    # Identical paired answers carry no contrastive evidence.  Keeping the
    # complete answer is safer than silently producing an empty objective.
    if fallback_full and not left_spans and left.strip():
        left_spans = [(0, len(left))]
    if fallback_full and not right_spans and right.strip():
        right_spans = [(0, len(right))]
    return merge_spans(left_spans), merge_spans(right_spans)


def clause_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start = 0
    for boundary in CLAUSE_BOUNDARY_RE.finditer(text):
        end = boundary.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def claims_covering_evidence(text: str, evidence: Sequence[Sequence[int]]) -> List[List[int]]:
    selected = []
    for claim_start, claim_end in clause_spans(text):
        if any(max(claim_start, int(start)) < min(claim_end, int(end)) for start, end in evidence):
            selected.append((claim_start, claim_end))
    return merge_spans(selected or [(0, len(text))])


def annotate_pair(left: Dict, right: Dict, claim_mode: str = "clause") -> None:
    if claim_mode not in {"clause", "evidence"}:
        raise ValueError(f"Unsupported claim mode: {claim_mode}")
    left_evidence, right_evidence = paired_evidence_spans(
        left["answer"],
        right["answer"],
        fallback_full=claim_mode != "evidence",
    )
    if claim_mode == "evidence" and (not left_evidence or not right_evidence):
        raise ValueError("Exact diff-span mode requires a non-empty paired answer change")
    for cell, evidence in ((left, left_evidence), (right, right_evidence)):
        claim_spans = (
            evidence
            if claim_mode == "evidence"
            else claims_covering_evidence(cell["answer"], evidence)
        )
        cell["supervision"] = {
            "version": (
                "paired-diffspan-v1"
                if claim_mode == "evidence"
                else "paired-hierarchy-v1"
            ),
            "claim_spans": claim_spans,
            "evidence_spans": evidence,
        }


def annotate_record(record: Dict, claim_mode: str = "clause") -> Dict:
    cells = record["cells"]
    annotate_pair(cells["C11"], cells["C01"], claim_mode=claim_mode)
    annotate_pair(cells["C10"], cells["C00"], claim_mode=claim_mode)
    return record


def annotate_file(input_path: Path, output_path: Path, claim_mode: str = "clause") -> Dict:
    records = []
    with input_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("cells"), dict):
                raise ValueError(f"{input_path}:{line_no}: cells must be an object")
            records.append(annotate_record(record, claim_mode=claim_mode))
    if not records:
        raise ValueError(f"No records found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    cells = [cell for record in records for cell in record["cells"].values()]
    metadata = {
        "method": "U-F2D",
        "annotation": (
            "paired-diffspan-v1"
            if claim_mode == "evidence"
            else "paired-hierarchy-v1"
        ),
        "claim_mode": claim_mode,
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "units": len(records),
        "cells": len(cells),
        "claims": sum(len(cell["supervision"]["claim_spans"]) for cell in cells),
        "evidence_spans": sum(len(cell["supervision"]["evidence_spans"]) for cell in cells),
        "retain_access": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=200)
    parser.add_argument(
        "--claim-mode",
        choices=("clause", "evidence"),
        default="clause",
        help=(
            "Use clause spans (legacy hierarchy) or exact paired-difference "
            "spans (local intervention training)."
        ),
    )
    args = parser.parse_args()
    metadata = annotate_file(args.input, args.output, claim_mode=args.claim_mode)
    if metadata["units"] != args.expected_units:
        args.output.unlink(missing_ok=True)
        args.output.with_suffix(".json").unlink(missing_ok=True)
        raise SystemExit(
            f"Expected {args.expected_units} units, found {metadata['units']}; output removed"
        )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
