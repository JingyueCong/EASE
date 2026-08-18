#!/usr/bin/env python3
"""Create a stratified human-audit sheet for U-F2D hierarchy annotations."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CELLS = ("C11", "C01", "C10", "C00")


def source_index(source_id: str) -> int:
    match = re.search(r"(\d+)$", source_id)
    if not match:
        raise ValueError(f"source_id has no numeric suffix: {source_id}")
    return int(match.group(1))


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for cell_name in CELLS:
                cell = record.get("cells", {}).get(cell_name, {})
                supervision = cell.get("supervision", {})
                if not supervision.get("claim_spans"):
                    raise ValueError(
                        f"{path}:{line_no}:{cell_name} missing claim_spans"
                    )
                if not supervision.get("evidence_spans"):
                    raise ValueError(
                        f"{path}:{line_no}:{cell_name} missing evidence_spans"
                    )
            records.append(record)
    if not records:
        raise ValueError(f"No annotated records in {path}")
    return records


def stratified_sample(
    records: Sequence[Dict], count: int, block_size: int, seed: int
) -> List[Dict]:
    by_block: Dict[int, List[Dict]] = {}
    for record in records:
        block = source_index(record["source_id"]) // block_size
        by_block.setdefault(block, []).append(record)
    if count < len(by_block):
        raise ValueError("count must cover every represented source block")

    rng = random.Random(seed)
    base, remainder = divmod(count, len(by_block))
    selected = []
    for offset, block in enumerate(sorted(by_block)):
        candidates = sorted(by_block[block], key=lambda row: row["source_id"])
        take = base + (1 if offset < remainder else 0)
        if take > len(candidates):
            raise ValueError(f"block {block} has only {len(candidates)} records")
        selected.extend(rng.sample(candidates, take))
    return sorted(selected, key=lambda row: source_index(row["source_id"]))


def span_text(text: str, spans: Iterable[Sequence[int]]) -> List[str]:
    return [text[int(start):int(end)] for start, end in spans]


def coverage(text: str, spans: Iterable[Sequence[int]]) -> float:
    positions = set()
    for start, end in spans:
        positions.update(range(max(int(start), 0), min(int(end), len(text))))
    nonspace = {index for index, char in enumerate(text) if not char.isspace()}
    return len(positions & nonspace) / max(len(nonspace), 1)


def outside_claim(
    evidence: Iterable[Sequence[int]], claims: Iterable[Sequence[int]]
) -> bool:
    for evidence_start, evidence_end in evidence:
        if not any(
            int(claim_start) <= int(evidence_start)
            and int(evidence_end) <= int(claim_end)
            for claim_start, claim_end in claims
        ):
            return True
    return False


def risk_flags(cell: Dict) -> Tuple[List[str], float, float]:
    text = cell["answer"]
    supervision = cell["supervision"]
    claims = supervision["claim_spans"]
    evidence = supervision["evidence_spans"]
    claim_coverage = coverage(text, claims)
    evidence_coverage = coverage(text, evidence)
    flags = []
    if claim_coverage >= 0.90:
        flags.append("CLAIM_NEAR_FULL_ANSWER")
    if evidence_coverage >= 0.60:
        flags.append("EVIDENCE_TOO_BROAD")
    if evidence_coverage <= 0.03:
        flags.append("EVIDENCE_TOO_NARROW")
    if outside_claim(evidence, claims):
        flags.append("EVIDENCE_OUTSIDE_CLAIM")
    return flags, claim_coverage, evidence_coverage


def md_escape(text: str) -> str:
    return text.replace("\n", " ").replace("|", "\\|")


def build_markdown(records: Sequence[Dict], input_path: Path) -> str:
    lines = [
        "# U-F2D hierarchy human audit (20 stratified units)",
        "",
        f"- Input: `{input_path}`",
        f"- Units: `{len(records)}`",
        "- Sampling: two records per ordered 20-QA author block, seed 42",
        "- Review labels: `PASS`, `CLAIM_TOO_BROAD`, `CLAIM_TOO_NARROW`, "
        "`EVIDENCE_WRONG`, `PAIR_MISMATCH`, `OTHER`",
        "",
        "For every unit, verify that evidence is the smallest factual content "
        "changed inside its matched pair and that claim contains the complete "
        "atomic proposition but excludes unrelated descriptive sentences.",
        "",
    ]
    flagged_cells = 0
    for item_number, record in enumerate(records, 1):
        lines.extend([
            f"## {item_number}. {record['source_id']}",
            "",
            f"- Target entity: `{record.get('target_entity')}`",
            f"- Replacement entity: `{record.get('replacement_entity')}`",
            f"- Target relation: `{record.get('target_relation')}`",
            f"- Placebo relation: `{record.get('placebo_relation')}`",
            "- Human verdict: `TODO`",
            "- Human note:",
            "",
        ])
        for cell_name in CELLS:
            cell = record["cells"][cell_name]
            supervision = cell["supervision"]
            flags, claim_cov, evidence_cov = risk_flags(cell)
            flagged_cells += bool(flags)
            claim_text = span_text(cell["answer"], supervision["claim_spans"])
            evidence_text = span_text(cell["answer"], supervision["evidence_spans"])
            lines.extend([
                f"### {cell_name}",
                "",
                f"- Question: {md_escape(cell['question'])}",
                f"- Answer: {md_escape(cell['answer'])}",
                f"- Extracted claim: `{md_escape(' || '.join(claim_text))}`",
                f"- Extracted evidence: `{md_escape(' || '.join(evidence_text))}`",
                f"- Coverage: claim={claim_cov:.1%}, evidence={evidence_cov:.1%}",
                f"- Automatic flags: `{','.join(flags) if flags else 'none'}`",
                "",
            ])
        lines.extend([
            "### Unit-level checks",
            "",
            "- [ ] C11/C01 evidence differs only in target-entity factual content",
            "- [ ] C10/C00 evidence differs only in placebo factual content",
            "- [ ] Each claim is atomic and excludes unrelated description",
            "- [ ] Evidence is factual rather than punctuation/style scaffolding",
            "- [ ] Four cells remain relation/style/length matched",
            "",
        ])
    lines.extend([
        "## Automatic audit summary",
        "",
        f"- Flagged cells: `{flagged_cells}/{len(records) * len(CELLS)}`",
        "- Automatic flags are triage signals, not final validity judgments.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(args.input)
    selected = stratified_sample(records, args.count, args.block_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown(selected, args.input), encoding="utf-8")
    print(args.output.resolve())
    print(f"sampled={len(selected)} total={len(records)}")


if __name__ == "__main__":
    main()
