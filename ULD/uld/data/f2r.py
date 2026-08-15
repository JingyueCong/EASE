"""Utilities for Forget-to-Retain (F2R) counterfactual supervision.

This module intentionally depends only on the Python standard library so that
generated data can be validated before installing the training stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


REQUIRED_FIELDS = (
    "source_id",
    "source_question",
    "source_answer",
    "matched_question",
    "matched_answer",
    "mismatched_question",
    "mismatched_answer",
)


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_pair(record: Dict, require_no_source_answer_leakage: bool = True) -> List[str]:
    """Return validation errors for one generated counterfactual record."""
    errors: List[str] = []
    for field in REQUIRED_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing or empty string field: {field}")

    if errors:
        return errors

    source_q = _normalise(record["source_question"])
    source_a = _normalise(record["source_answer"])
    matched_q = _normalise(record["matched_question"])
    matched_a = _normalise(record["matched_answer"])

    if source_q == matched_q:
        errors.append("matched_question is identical to source_question")
    if source_a == matched_a:
        errors.append("matched_answer is identical to source_answer")
    if require_no_source_answer_leakage and len(source_a) >= 8:
        matched_text = f"{matched_q} {matched_a}"
        if source_a in matched_text:
            errors.append("source_answer appears verbatim in matched counterfactual")

    invariants = record.get("invariants")
    if not isinstance(invariants, dict):
        errors.append("invariants must be an object")
    else:
        for field in ("task", "relation", "style", "difficulty"):
            if not isinstance(invariants.get(field), str) or not invariants[field].strip():
                errors.append(f"missing invariant: {field}")

    changed = record.get("changed_evidence")
    if not isinstance(changed, list) or not changed:
        errors.append("changed_evidence must be a non-empty list")

    return errors


def load_f2r_pairs(
    path: str,
    *,
    strict: bool = True,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict]]:
    """Load F2R JSONL and return matched QA, mismatched QA, raw records."""
    matched: List[Dict[str, str]] = []
    mismatched: List[Dict[str, str]] = []
    records: List[Dict] = []
    seen_ids = set()

    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            errors = validate_pair(record)
            if errors and strict:
                raise ValueError(f"{path}:{line_no}: " + "; ".join(errors))
            if errors:
                continue

            pair_id = record["source_id"]
            view = int(record.get("view", 0))
            key = (pair_id, view)
            if key in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate pair key {key}")
            seen_ids.add(key)

            matched.append({
                "question": record["matched_question"].strip(),
                "answer": record["matched_answer"].strip(),
            })
            mismatched.append({
                "question": record["mismatched_question"].strip(),
                "answer": record["mismatched_answer"].strip(),
            })
            records.append(record)

    if not records:
        raise ValueError(f"No valid F2R records found in {path}")
    return matched, mismatched, records


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
