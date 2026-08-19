#!/usr/bin/env python3
"""Audit TOFU four-cell factorial data without modifying it.

The audit deliberately separates deterministic schema/consistency checks from
heuristic semantic triage.  Heuristic flags identify records for human review;
they are not presented as proof that a causal identification assumption holds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


CELLS = ("C11", "C01", "C10", "C00")
UNAVAILABLE_PATTERNS = (
    r"\bno (?:publicly available|definitive|specific|known) information\b",
    r"\b(?:is|are|was|were|remain|remains) (?:not known|unknown|unclear|undisclosed|unconfirmed)\b",
    r"\binformation (?:is|was) (?:not available|unavailable)\b",
    r"\bdetails? (?:is|are|was|were|remain|remains) (?:unknown|unclear|undisclosed)\b",
    r"\bhas not been (?:confirmed|disclosed|documented)\b",
)
QUALIFIED_PATTERNS = (
    r"\bapproximately\b",
    r"\blikely\b",
    r"\bappears? to\b",
    r"\bmay\b",
    r"\bperhaps\b",
    r"\bwhile (?:the )?details\b",
)
NEGATION_PATTERN = re.compile(r"\b(?:no|not|never|neither|without)\b", re.I)
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*")
STOPWORDS = {
    "about", "after", "again", "also", "answer", "author", "because",
    "before", "being", "book", "books", "could", "from", "have", "into",
    "known", "more", "most", "other", "over", "question", "that", "their",
    "there", "these", "they", "this", "through", "under", "very", "what",
    "when", "where", "which", "while", "with", "would", "written", "writes",
}


@dataclass
class UnitAudit:
    source_id: str
    source_index: int
    block: int
    target_entity: str
    replacement_entity: str
    target_relation: str
    placebo_relation: str
    target_question_similarity: float
    placebo_question_similarity: float
    target_length_ratio: float
    placebo_length_ratio: float
    target_fact_delta: int
    placebo_fact_delta: int
    target_response_modes: str
    placebo_response_modes: str
    target_formats: str
    placebo_formats: str
    placebo_source_overlap: float
    deterministic_errors: str
    semantic_risks: str

    @property
    def error_count(self) -> int:
        return count_joined(self.deterministic_errors)

    @property
    def risk_count(self) -> int:
        return count_joined(self.semantic_risks)


def count_joined(value: str) -> int:
    return 0 if not value else len(value.split(";"))


def source_index(source_id: str) -> int:
    match = re.search(r"(\d+)$", source_id)
    if not match:
        raise ValueError(f"source_id has no numeric suffix: {source_id}")
    return int(match.group(1))


def normalise(text: str) -> str:
    return " ".join(str(text).casefold().split())


def replace_entity(text: str, entity: str) -> str:
    result = normalise(text)
    entity_norm = normalise(entity)
    if entity_norm:
        result = result.replace(entity_norm, " <entity> ")
    return " ".join(re.sub(r"[^a-z0-9<>]+", " ", result).split())


def question_similarity(left: Mapping[str, str], right: Mapping[str, str],
                        left_entity: str, right_entity: str) -> float:
    left_text = replace_entity(left["question"], left_entity)
    right_text = replace_entity(right["question"], right_entity)
    return SequenceMatcher(None, left_text, right_text).ratio()


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def length_ratio(left: str, right: str) -> float:
    low, high = sorted((max(word_count(left), 1), max(word_count(right), 1)))
    return high / low


def response_mode(answer: str) -> str:
    text = normalise(answer)
    if any(re.search(pattern, text, re.I) for pattern in UNAVAILABLE_PATTERNS):
        return "unavailable"
    if any(re.search(pattern, text, re.I) for pattern in QUALIFIED_PATTERNS):
        return "qualified"
    if text.startswith("no,") or text.startswith("no "):
        return "negative"
    if text.startswith("yes,") or text.startswith("yes "):
        return "affirmative_yes"
    if NEGATION_PATTERN.search(text):
        return "contains_negation"
    return "affirmative"


def response_format(question: str, answer: str) -> str:
    q = normalise(question)
    a = normalise(answer)
    if response_mode(answer) == "unavailable":
        return "unavailable"
    if q.startswith(("is ", "are ", "was ", "were ", "has ", "have ", "did ", "does ", "can ")):
        return "yes_no" if a.startswith(("yes", "no")) else "yes_no_explanation"
    if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:19|20)\d{2}\b", answer):
        return "date_or_year"
    if re.search(r"\b\d+(?:\.\d+)?\b", answer):
        return "numeric"
    quoted = len(re.findall(r'["“”][^"“”]+["“”]', answer))
    if quoted >= 2 or (answer.count(",") >= 2 and re.search(r"\band\b", answer, re.I)):
        return "list"
    if sentence_count(answer) >= 2:
        return "multi_sentence_prose"
    return "short_prose"


def sentence_count(text: str) -> int:
    chunks = [chunk for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk]
    return max(len(chunks), 1)


def fact_count_proxy(text: str) -> int:
    """Conservative triage proxy, not a semantic fact extractor."""
    sentences = sentence_count(text)
    quoted_items = len(re.findall(r'["“”][^"“”]+["“”]', text))
    semicolon_clauses = text.count(";")
    list_bonus = max(0, text.count(",") - 1) if re.search(r"\band\b", text, re.I) else 0
    return max(sentences + semicolon_clauses, quoted_items, 1 + list_bonus)


def content_tokens(text: str, excluded: Iterable[str] = ()) -> set[str]:
    excluded_tokens = {
        token.casefold()
        for value in excluded
        for token in WORD_PATTERN.findall(str(value))
    }
    return {
        token.casefold()
        for token in WORD_PATTERN.findall(text)
        if len(token) >= 4
        and token.casefold() not in STOPWORDS
        and token.casefold() not in excluded_tokens
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def load_ciru_module():
    module_path = Path(__file__).resolve().parents[1] / "ULD" / "uld" / "data" / "ciru.py"
    spec = importlib.util.spec_from_file_location("audit_ciru_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CIRU validator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def audit_unit(record: Dict, block_size: int, min_question_similarity: float,
               max_fact_delta: int, ciru_data) -> UnitAudit:
    sid = str(record.get("source_id", ""))
    index = source_index(sid)
    cells = record.get("cells", {})
    structural = list(ciru_data.validate_ciru_unit(record))
    if any(cell not in cells for cell in CELLS):
        return UnitAudit(
            sid, index, index // block_size, str(record.get("target_entity", "")),
            str(record.get("replacement_entity", "")),
            str(record.get("target_relation", "")), str(record.get("placebo_relation", "")),
            math.nan, math.nan, math.nan, math.nan, 0, 0, "", "", "", "", 0.0,
            ";".join(structural or ["missing factorial cells"]), "",
        )

    target = (cells["C11"], cells["C01"])
    placebo = (cells["C10"], cells["C00"])
    target_entity = str(record.get("target_entity", ""))
    replacement = str(record.get("replacement_entity", ""))
    target_sim = question_similarity(*target, target_entity, replacement)
    placebo_sim = question_similarity(*placebo, target_entity, replacement)
    target_len = length_ratio(target[0]["answer"], target[1]["answer"])
    placebo_len = length_ratio(placebo[0]["answer"], placebo[1]["answer"])
    target_facts = tuple(fact_count_proxy(cell["answer"]) for cell in target)
    placebo_facts = tuple(fact_count_proxy(cell["answer"]) for cell in placebo)
    target_modes = tuple(response_mode(cell["answer"]) for cell in target)
    placebo_modes = tuple(response_mode(cell["answer"]) for cell in placebo)
    target_formats = tuple(response_format(cell["question"], cell["answer"]) for cell in target)
    placebo_formats = tuple(response_format(cell["question"], cell["answer"]) for cell in placebo)

    excluded = (target_entity, replacement)
    source_tokens = content_tokens(cells["C11"]["answer"], excluded)
    placebo_tokens = content_tokens(
        f"{cells['C10']['answer']} {cells['C00']['answer']}", excluded
    )
    overlap = jaccard(source_tokens, placebo_tokens)

    risks = []
    if target_sim < min_question_similarity:
        risks.append("TARGET_QUESTION_TEMPLATE_MISMATCH")
    if placebo_sim < min_question_similarity:
        risks.append("PLACEBO_QUESTION_TEMPLATE_MISMATCH")
    if target_modes[0] != target_modes[1]:
        risks.append("TARGET_RESPONSE_MODE_MISMATCH")
    if placebo_modes[0] != placebo_modes[1]:
        risks.append("PLACEBO_RESPONSE_MODE_MISMATCH")
    if target_formats[0] != target_formats[1]:
        risks.append("TARGET_ANSWER_FORMAT_MISMATCH")
    if placebo_formats[0] != placebo_formats[1]:
        risks.append("PLACEBO_ANSWER_FORMAT_MISMATCH")
    if abs(target_facts[0] - target_facts[1]) > max_fact_delta:
        risks.append("TARGET_FACT_COUNT_MISMATCH")
    if abs(placebo_facts[0] - placebo_facts[1]) > max_fact_delta:
        risks.append("PLACEBO_FACT_COUNT_MISMATCH")
    if overlap >= 0.45:
        risks.append("PLACEBO_SOURCE_CONTENT_OVERLAP")

    return UnitAudit(
        source_id=sid,
        source_index=index,
        block=index // block_size,
        target_entity=target_entity,
        replacement_entity=replacement,
        target_relation=str(record.get("target_relation", "")),
        placebo_relation=str(record.get("placebo_relation", "")),
        target_question_similarity=target_sim,
        placebo_question_similarity=placebo_sim,
        target_length_ratio=target_len,
        placebo_length_ratio=placebo_len,
        target_fact_delta=abs(target_facts[0] - target_facts[1]),
        placebo_fact_delta=abs(placebo_facts[0] - placebo_facts[1]),
        target_response_modes="/".join(target_modes),
        placebo_response_modes="/".join(placebo_modes),
        target_formats="/".join(target_formats),
        placebo_formats="/".join(placebo_formats),
        placebo_source_overlap=overlap,
        deterministic_errors=";".join(structural),
        semantic_risks=";".join(risks),
    )


def audit_blocks(records: Sequence[Dict], audits: Sequence[UnitAudit], block_size: int,
                 expected_units: int) -> List[Dict]:
    records_by_id = {str(row.get("source_id", "")): row for row in records}
    by_block: Dict[int, List[UnitAudit]] = defaultdict(list)
    for audit in audits:
        by_block[audit.block].append(audit)
    expected_blocks = math.ceil(expected_units / block_size)
    output = []
    for block in range(expected_blocks):
        items = sorted(by_block.get(block, []), key=lambda row: row.source_index)
        targets = {normalise(row.target_entity) for row in items if row.target_entity}
        replacements = {
            normalise(row.replacement_entity) for row in items if row.replacement_entity
        }
        relation_answers: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        for item in items:
            record = records_by_id[item.source_id]
            cells = record.get("cells", {})
            if any(cell not in cells for cell in CELLS):
                continue
            relation_answers[("target", normalise(item.target_relation))].add(
                normalise(cells["C01"].get("answer", ""))
            )
            relation_answers[("placebo", normalise(item.placebo_relation))].add(
                normalise(cells["C00"].get("answer", ""))
            )
        repeated_relation_candidates = sum(
            len(answers) > 1
            for (_, relation), answers in relation_answers.items()
            if relation
        )
        flags = []
        expected_here = min(block_size, max(expected_units - block * block_size, 0))
        if len(items) != expected_here:
            flags.append("BLOCK_COVERAGE_MISMATCH")
        if len(targets) != 1:
            flags.append("MULTIPLE_TARGET_ENTITIES")
        if len(replacements) != 1:
            flags.append("MULTIPLE_REPLACEMENT_ENTITIES")
        if repeated_relation_candidates:
            flags.append("REPEATED_RELATION_CONFLICT_CANDIDATE")
        output.append({
            "block": block,
            "units": len(items),
            "target_entities": " | ".join(sorted(targets)),
            "replacement_entities": " | ".join(sorted(replacements)),
            "unit_errors": sum(row.error_count for row in items),
            "unit_risks": sum(row.risk_count for row in items),
            "repeated_relation_conflict_candidates": repeated_relation_candidates,
            "flags": ";".join(flags),
        })
    return output


def risk_counts(audits: Sequence[UnitAudit]) -> Counter:
    counts = Counter()
    for audit in audits:
        for value in (audit.deterministic_errors, audit.semantic_risks):
            counts.update(flag for flag in value.split(";") if flag)
    return counts


def stratified_risk_sample(audits: Sequence[UnitAudit], count: int,
                           block_size: int, seed: int) -> List[UnitAudit]:
    by_block: Dict[int, List[UnitAudit]] = defaultdict(list)
    for audit in audits:
        by_block[audit.source_index // block_size].append(audit)
    if count < len(by_block):
        raise ValueError("sample count must cover every represented author block")
    rng = random.Random(seed)
    base, remainder = divmod(count, len(by_block))
    selected = []
    for offset, block in enumerate(sorted(by_block)):
        take = base + (1 if offset < remainder else 0)
        candidates = list(by_block[block])
        rng.shuffle(candidates)
        candidates.sort(key=lambda row: (row.error_count, row.risk_count), reverse=True)
        selected.extend(candidates[:take])
    return sorted(selected, key=lambda row: row.source_index)


def stratified_random_sample(audits: Sequence[UnitAudit], count: int,
                             block_size: int, seed: int) -> List[UnitAudit]:
    by_block: Dict[int, List[UnitAudit]] = defaultdict(list)
    for audit in audits:
        by_block[audit.source_index // block_size].append(audit)
    if count < len(by_block):
        raise ValueError("sample count must cover every represented author block")
    rng = random.Random(seed)
    base, remainder = divmod(count, len(by_block))
    selected = []
    for offset, block in enumerate(sorted(by_block)):
        take = base + (1 if offset < remainder else 0)
        candidates = sorted(by_block[block], key=lambda row: row.source_index)
        if take > len(candidates):
            raise ValueError(f"author block {block} has only {len(candidates)} records")
        selected.extend(rng.sample(candidates, take))
    return sorted(selected, key=lambda row: row.source_index)


def md_escape(value: object) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


def build_human_audit(sample: Sequence[UnitAudit], records: Sequence[Dict],
                      input_path: Path, sampling: str) -> str:
    by_id = {str(row["source_id"]): row for row in records}
    lines = [
        "# TOFU factorial semantic audit",
        "",
        f"- Input: `{input_path}`",
        f"- Sampled units: `{len(sample)}`",
        f"- Sampling: {sampling}",
        "- Automatic semantic flags are triage only; human judgment is authoritative.",
        "",
        "Use `PASS`, `PAIR_MISMATCH`, `POLARITY_MISMATCH`, `FACT_MISMATCH`, "
        "`PLACEBO_LEAKAGE`, `BLOCK_CONTRADICTION`, or `OTHER`.",
        "",
    ]
    for number, audit in enumerate(sample, 1):
        record = by_id[audit.source_id]
        lines.extend([
            f"## {number}. {audit.source_id} (author block {audit.block})",
            "",
            f"- Target/replacement: `{audit.target_entity}` / `{audit.replacement_entity}`",
            f"- Target/placebo relation: `{audit.target_relation}` / `{audit.placebo_relation}`",
            f"- Deterministic errors: `{audit.deterministic_errors or 'none'}`",
            f"- Semantic triage: `{audit.semantic_risks or 'none'}`",
            "- Human verdict: `TODO`",
            "- Human note:",
            "",
        ])
        for cell_name in CELLS:
            cell = record["cells"][cell_name]
            lines.extend([
                f"### {cell_name}",
                "",
                f"- Question: {md_escape(cell['question'])}",
                f"- Answer: {md_escape(cell['answer'])}",
                f"- Mode/format/fact proxy: `{response_mode(cell['answer'])}` / "
                f"`{response_format(cell['question'], cell['answer'])}` / "
                f"`{fact_count_proxy(cell['answer'])}`",
                "",
            ])
        lines.extend([
            "### Required judgments",
            "",
            "- [ ] C11/C01 ask the same semantic relation",
            "- [ ] C11/C01 match polarity, answerability, fact count, and specificity",
            "- [ ] C10/C00 ask the same semantic placebo relation",
            "- [ ] C10/C00 match polarity, answerability, fact count, and specificity",
            "- [ ] C10/C00 cannot reveal or entail the C11 answer",
            "- [ ] This row is consistent with the replacement author's other 19 rows",
            "",
        ])
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(input_path: Path, records: Sequence[Dict], audits: Sequence[UnitAudit],
                  blocks: Sequence[Dict], expected_units: int) -> Dict:
    counts = risk_counts(audits)
    indices = [audit.source_index for audit in audits]
    expected_indices = set(range(expected_units))
    actual_indices = set(indices)
    return {
        "input": str(input_path.resolve()),
        "records": len(records),
        "unique_source_ids": len({audit.source_id for audit in audits}),
        "duplicate_source_ids": len(records) - len({audit.source_id for audit in audits}),
        "expected_units": expected_units,
        "missing_source_indices": sorted(expected_indices - actual_indices),
        "unexpected_source_indices": sorted(actual_indices - expected_indices),
        "units_with_deterministic_errors": sum(audit.error_count > 0 for audit in audits),
        "units_with_semantic_risks": sum(audit.risk_count > 0 for audit in audits),
        "automatic_flag_counts": dict(sorted(counts.items())),
        "blocks_with_flags": sum(bool(block["flags"]) for block in blocks),
        "important_note": (
            "Semantic flags are deterministic triage heuristics, not proof of causal validity. "
            "Use both human-audit sheets before accepting or regenerating data."
        ),
    }


def build_summary_markdown(summary: Mapping[str, object], blocks: Sequence[Dict]) -> str:
    counts = summary["automatic_flag_counts"]
    lines = [
        "# TOFU factorial audit summary",
        "",
        f"- Input: `{summary['input']}`",
        f"- Records: `{summary['records']}/{summary['expected_units']}`",
        f"- Unique source ids: `{summary['unique_source_ids']}`",
        f"- Duplicate source ids: `{summary['duplicate_source_ids']}`",
        f"- Units with deterministic errors: `{summary['units_with_deterministic_errors']}`",
        f"- Units with semantic triage risks: `{summary['units_with_semantic_risks']}`",
        f"- Author blocks with flags: `{summary['blocks_with_flags']}`",
        "",
        "> Semantic flags are triage signals, not proof of causal validity. "
        "Use `HUMAN_AUDIT_RANDOM.md` to estimate quality and "
        "`HUMAN_AUDIT_RISK.md` to inspect likely failures.",
        "",
        "## Automatic flag counts",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    if counts:
        lines.extend(f"| `{flag}` | {count} |" for flag, count in counts.items())
    else:
        lines.append("| none | 0 |")
    lines.extend([
        "",
        "## Author-block consistency",
        "",
        "| block | units | target entities | replacement entities | risks | flags |",
        "|---:|---:|---|---|---:|---|",
    ])
    for block in blocks:
        lines.append(
            f"| {block['block']} | {block['units']} | "
            f"{md_escape(block['target_entities'])} | "
            f"{md_escape(block['replacement_entities'])} | "
            f"{block['unit_risks']} | {md_escape(block['flags'] or 'none')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-units", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-question-similarity", type=float, default=0.72)
    parser.add_argument("--max-fact-delta", type=int, default=1)
    parser.add_argument("--fail-on-deterministic-errors", action="store_true")
    args = parser.parse_args()

    ciru_data = load_ciru_module()
    records = load_records(args.input)
    audits = [
        audit_unit(
            record, args.block_size, args.min_question_similarity,
            args.max_fact_delta, ciru_data,
        )
        for record in records
    ]
    audits.sort(key=lambda row: row.source_index)
    blocks = audit_blocks(records, audits, args.block_size, args.expected_units)
    risk_sample = stratified_risk_sample(
        audits, args.sample_count, args.block_size, args.seed
    )
    random_sample = stratified_random_sample(
        audits, args.sample_count, args.block_size, args.seed
    )
    summary = build_summary(args.input, records, audits, blocks, args.expected_units)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SUMMARY.md").write_text(
        build_summary_markdown(summary, blocks), encoding="utf-8"
    )
    write_csv(args.output_dir / "UNITS.csv", [asdict(audit) for audit in audits])
    write_csv(args.output_dir / "BLOCKS.csv", blocks)
    (args.output_dir / "HUMAN_AUDIT_RANDOM.md").write_text(
        build_human_audit(
            random_sample, records, args.input,
            "two uniformly random records per ordered author block",
        ),
        encoding="utf-8",
    )
    (args.output_dir / "HUMAN_AUDIT_RISK.md").write_text(
        build_human_audit(
            risk_sample, records, args.input,
            "two risk-prioritized records per ordered author block",
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"reports={args.output_dir.resolve()}")
    if args.fail_on_deterministic_errors and summary["units_with_deterministic_errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
