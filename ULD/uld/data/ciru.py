"""Schema validation for CIRU factorial counterfactual units."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CELLS = ("C11", "C01", "C10", "C00")
AUTHOR_PROFILE_DESIGN = "tofu-author-profile-v2"
AUTHOR_CONTRACT_DESIGN = "tofu-author-contract-v3"
AUTHOR_TYPED_LEGACY_DESIGN = "tofu-author-typed-v4"
AUTHOR_TYPED_V41_DESIGN = "tofu-author-typed-v4.1"
AUTHOR_TYPED_DESIGN = "tofu-author-typed-v4.2"
AUTHOR_ANCHOR_DESIGN = "tofu-author-anchor-v5"
AUTHOR_ANCHOR_V51_DESIGN = "tofu-author-anchor-v5.1"
AUTHOR_ROWLOCAL_V52_DESIGN = "tofu-author-rowlocal-v5.2"
AUTHOR_LEDGER_V53_DESIGN = "tofu-author-ledger-rowlocal-v5.3"
AUTHOR_LEDGER_SLOTS_V54_DESIGN = "tofu-author-ledger-slots-v5.4"
AUTHOR_LEDGER_ANSWER_V55_DESIGN = "tofu-author-ledger-answer-v5.5"
AUTHOR_DIRECT_V56_DESIGN = "tofu-author-direct-contrast-v5.6"
AUTHOR_JOINT_V57_DESIGN = "tofu-author-joint-contrast-v5.7"
AUTHOR_SEMANTIC_V58_DESIGN = "tofu-author-semantic-contrast-v5.8"
AUTHOR_SEMANTIC_AGENT_V59_DESIGN = "tofu-author-semantic-agent-v5.9"
AUTHOR_PAIRREPAIR_V510_DESIGN = "tofu-author-pairrepair-v5.10"
AUTHOR_PREMISEFIX_V511_DESIGN = "tofu-author-premisefix-v5.11"
AUTHOR_LEVEL_DESIGNS = {
    AUTHOR_PROFILE_DESIGN,
    AUTHOR_CONTRACT_DESIGN,
    AUTHOR_TYPED_LEGACY_DESIGN,
    AUTHOR_TYPED_V41_DESIGN,
    AUTHOR_TYPED_DESIGN,
    AUTHOR_ANCHOR_DESIGN,
    AUTHOR_ANCHOR_V51_DESIGN,
    AUTHOR_ROWLOCAL_V52_DESIGN,
    AUTHOR_LEDGER_V53_DESIGN,
    AUTHOR_LEDGER_SLOTS_V54_DESIGN,
    AUTHOR_LEDGER_ANSWER_V55_DESIGN,
    AUTHOR_DIRECT_V56_DESIGN,
    AUTHOR_JOINT_V57_DESIGN,
    AUTHOR_SEMANTIC_V58_DESIGN,
    AUTHOR_SEMANTIC_AGENT_V59_DESIGN,
    AUTHOR_PAIRREPAIR_V510_DESIGN,
    AUTHOR_PREMISEFIX_V511_DESIGN,
}
MAX_LENGTH_RATIO = 2.0
CONTROL_STATUS_MARKERS = (
    "fictional",
    "as a control",
    "control example",
    "no reliably documented",
    "no widely documented",
    "no publicly documented",
    "no public information",
    "not publicly known",
    "available public sources",
)
CONTROL_STATUS_EQUIVALENTS = {
    "fictional": ("fictional", "fictitious"),
}


def normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def control_status_marker_inherited(marker: str, source_text: str) -> bool:
    """Return whether benchmark text already expresses this control status.

    This is surface canonicalisation, not semantic acceptance.  In particular,
    TOFU uses both ``fictional`` and ``fictitious`` for the same benchmark
    instruction, so changing between those forms must not create a synthetic-
    control leak.
    """
    source = normalise(source_text)
    return any(
        equivalent in source
        for equivalent in CONTROL_STATUS_EQUIVALENTS.get(marker, (marker,))
    )


def validate_ciru_unit(record: Dict) -> List[str]:
    """Return hard validity errors for one 2x2 intervention unit."""
    errors: List[str] = []
    for field in (
        "source_id",
        "source_question",
        "source_answer",
        "target_entity",
        "replacement_entity",
        "target_relation",
        "placebo_relation",
    ):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing or empty string field: {field}")

    cells = record.get("cells")
    if not isinstance(cells, dict):
        errors.append("cells must be an object")
        return errors
    for cell in CELLS:
        item = cells.get(cell)
        if not isinstance(item, dict):
            errors.append(f"missing cell: {cell}")
            continue
        for field in ("question", "answer"):
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{cell}.{field} must be a non-empty string")
    if errors:
        return errors

    if normalise(cells["C11"]["question"]) != normalise(record["source_question"]):
        errors.append("C11.question must equal source_question")
    if normalise(cells["C11"]["answer"]) != normalise(record["source_answer"]):
        errors.append("C11.answer must equal source_answer")

    target_entity = normalise(record["target_entity"])
    replacement = normalise(record["replacement_entity"])
    if target_entity == replacement:
        errors.append("replacement_entity must differ from target_entity")
    author_profile_design = record.get("design_version") in AUTHOR_LEVEL_DESIGNS
    author_typed_design = record.get("design_version") in {
        AUTHOR_TYPED_LEGACY_DESIGN,
        AUTHOR_TYPED_V41_DESIGN,
        AUTHOR_TYPED_DESIGN,
        AUTHOR_ANCHOR_DESIGN,
        AUTHOR_ANCHOR_V51_DESIGN,
        AUTHOR_ROWLOCAL_V52_DESIGN,
        AUTHOR_LEDGER_V53_DESIGN,
        AUTHOR_LEDGER_SLOTS_V54_DESIGN,
        AUTHOR_LEDGER_ANSWER_V55_DESIGN,
        AUTHOR_DIRECT_V56_DESIGN,
        AUTHOR_JOINT_V57_DESIGN,
        AUTHOR_SEMANTIC_V58_DESIGN,
        AUTHOR_SEMANTIC_AGENT_V59_DESIGN,
        AUTHOR_PAIRREPAIR_V510_DESIGN,
    }
    if author_profile_design:
        for field in ("block_id", "query_index", "profile_id"):
            value = record.get(field)
            if field == "profile_id":
                valid = isinstance(value, str) and bool(value.strip())
            else:
                valid = isinstance(value, int) and value >= 0
            if not valid:
                errors.append(f"invalid author-profile field: {field}")
        canonical = normalise(record.get("canonical_target_entity", ""))
        if not canonical:
            errors.append("missing canonical_target_entity")
        elif canonical != target_entity:
            errors.append("target_entity must equal canonical_target_entity")
        # TOFU contains implicit queries such as "Where was the author born?",
        # whose answer can be just a location.  Identity is therefore a block-
        # level experimental assignment rather than a lexical-span heuristic.
        binding = record.get("identity_binding")
        expected_binding = {
            "C11": record["target_entity"],
            "C01": record["replacement_entity"],
            "C10": record["target_entity"],
            "C00": record["replacement_entity"],
        }
        if not isinstance(binding, dict) or any(
            normalise(binding.get(cell, "")) != normalise(entity)
            for cell, entity in expected_binding.items()
        ):
            errors.append("identity_binding must encode the author-level 2x2 assignment")
    else:
        if target_entity not in normalise(cells["C11"]["question"]):
            errors.append("C11.question must contain target_entity")
        if target_entity not in normalise(cells["C10"]["question"]):
            errors.append("C10.question must contain target_entity")
    for cell in ("C01", "C00"):
        if (
            not author_profile_design
            and replacement not in normalise(cells[cell]["question"])
        ):
            errors.append(f"{cell}.question must contain replacement_entity")
        joined = normalise(f"{cells[cell]['question']} {cells[cell]['answer']}")
        if target_entity in joined:
            errors.append(f"{cell} leaks target_entity")

    source_answer = normalise(record["source_answer"])
    if len(source_answer) >= 8:
        for cell in ("C01", "C10", "C00"):
            joined = normalise(f"{cells[cell]['question']} {cells[cell]['answer']}")
            if source_answer in joined:
                errors.append(f"{cell} leaks source_answer")

    for cell in ("C01", "C10", "C00"):
        joined = normalise(f"{cells[cell]['question']} {cells[cell]['answer']}")
        source_joined = normalise(
            f"{record['source_question']} {record['source_answer']}"
        )
        for marker in CONTROL_STATUS_MARKERS:
            # V4 is an exact-edit renderer.  A marker already present in the
            # immutable benchmark text can be legitimate content (for example
            # "fictional narratives") and is not evidence that generation
            # exposed the control condition.  Newly introduced markers remain
            # forbidden, as do all such markers in legacy/free-form designs.
            inherited_v4_marker = (
                author_typed_design
                and cell == "C01"
                and control_status_marker_inherited(marker, source_joined)
            )
            if marker in joined and not inherited_v4_marker:
                errors.append(f"{cell} exposes control status with marker: {marker}")

    if normalise(record["target_relation"]) == normalise(record["placebo_relation"]):
        errors.append("placebo_relation must differ from target_relation")
    qa_pairs = {
        (normalise(cells[cell]["question"]), normalise(cells[cell]["answer"]))
        for cell in CELLS
    }
    if len(qa_pairs) != len(CELLS):
        errors.append("the four cells must be distinct")

    invariants = record.get("invariants")
    if not isinstance(invariants, dict):
        errors.append("invariants must be an object")
    else:
        for field in ("task", "style", "difficulty", "answer_format"):
            if not isinstance(invariants.get(field), str) or not invariants[field].strip():
                errors.append(f"missing invariant: {field}")
    audit = audit_ciru_unit(record)
    # Length matching in author-profile v2 is a semantic audit variable, not a
    # schema fact.  Immutable TOFU answers include very short forms (for
    # example a single date or "No"), so a four-cell global ratio can reject a
    # valid block even when both factorial pairs preserve response mode.  The
    # dedicated TOFU auditor still flags these rows for human review.  Legacy
    # designs retain the original hard 2x constraint for exact reproduction.
    if not author_profile_design and audit["question_length_ratio"] > MAX_LENGTH_RATIO:
        errors.append(
            "question length ratio exceeds "
            f"{MAX_LENGTH_RATIO}: {audit['question_length_ratio']:.3f}"
        )
    if not author_profile_design and audit["answer_length_ratio"] > MAX_LENGTH_RATIO:
        errors.append(
            "answer length ratio exceeds "
            f"{MAX_LENGTH_RATIO}: {audit['answer_length_ratio']:.3f}"
        )
    return errors


def audit_ciru_unit(record: Dict) -> Dict[str, float]:
    """Return transparent, deterministic matching diagnostics."""
    cells = record["cells"]
    q_lengths = [len(cells[cell]["question"].split()) for cell in CELLS]
    a_lengths = [len(cells[cell]["answer"].split()) for cell in CELLS]
    return {
        "question_words_min": min(q_lengths),
        "question_words_max": max(q_lengths),
        "question_length_ratio": max(q_lengths) / max(min(q_lengths), 1),
        "answer_words_min": min(a_lengths),
        "answer_words_max": max(a_lengths),
        "answer_length_ratio": max(a_lengths) / max(min(a_lengths), 1),
    }


def load_ciru_units(path: str | Path, strict: bool = True) -> List[Dict]:
    records: List[Dict] = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            errors = validate_ciru_unit(record)
            if errors and strict:
                raise ValueError(f"{path}:{line_no}: " + "; ".join(errors))
            if errors:
                continue
            key = (record["source_id"], int(record.get("view", 0)))
            if key in seen:
                raise ValueError(f"{path}:{line_no}: duplicate unit key {key}")
            seen.add(key)
            records.append(record)
    if not records:
        raise ValueError(f"No valid CIRU units found in {path}")
    return records


def factorial_dual_roles(
    records: Iterable[Dict], data_role: str
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Map audited 2x2 cells to one explicit dual-assistant contrast.

    In the upstream ``remember+uniform`` objective, the first returned list
    receives cross-entropy supervision and the second is pushed toward a
    uniform distribution.  A1 therefore estimates the target-relation entity
    contrast C11-C01, while A2 estimates its placebo counterpart C10-C00.
    Combining a negative A1 residual with a positive A2 residual implements
    the factorial difference-in-differences direction at inference time.
    """
    mapping = {
        "f2d_did_a1": ("C11", "C01"),
        "f2d_did_a2": ("C10", "C00"),
        "uf2d_hier_a1": ("C11", "C01"),
        "uf2d_hier_a2": ("C10", "C00"),
    }
    if data_role not in mapping:
        raise ValueError(f"Unknown factorial dual role: {data_role}")
    ce_cell, uniform_cell = mapping[data_role]
    ce_rows: List[Dict[str, str]] = []
    uniform_rows: List[Dict[str, str]] = []
    for record in records:
        ce_rows.append(dict(record["cells"][ce_cell]))
        uniform_rows.append(dict(record["cells"][uniform_cell]))
    if not ce_rows:
        raise ValueError("Factorial dual roles require at least one CIRU unit")
    return ce_rows, uniform_rows


def write_ciru_jsonl(path: str | Path, records: Iterable[Dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
