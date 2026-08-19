"""Typed, deterministic TOFU contracts and professional-domain placebos.

The language model is deliberately excluded from placebo selection and surface
realisation.  One of twenty frozen author-professional relations is assigned by
the within-author query index, then rendered according to the immutable C11
response contract.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Mapping, Sequence


DESIGN_VERSION = "tofu-author-typed-v4"
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*")
UNAVAILABLE_PATTERNS = (
    r"\bno (?:publicly )?(?:available|documented|known) information\b",
    r"\bthere is no (?:definitive|specific) information\b",
    r"\bnot (?:publicly )?(?:available|documented|known)\b",
    r"\bdetails? (?:are|is) (?:unclear|unknown|unavailable)\b",
    r"\bit is unclear\b",
    r"\bthere are no specific details\b",
)
QUALIFIED_PATTERNS = (r"\bit'?s not confirmed\b", r"\blikely\b", r"\bmay\b")


PROFESSIONAL_PLACEBOS = (
    ("drafting_medium", "drafting manuscripts", "a fountain pen", "a mechanical keyboard"),
    ("revision_workflow", "revising manuscripts", "two structured review passes", "three structured review passes"),
    ("editorial_feedback_channel", "collecting editorial feedback", "annotated print proofs", "tracked digital comments"),
    ("manuscript_archive_format", "archiving manuscripts", "dated paper folders", "versioned digital folders"),
    ("translation_review_process", "reviewing translations", "a bilingual peer review", "a translator-editor review"),
    ("citation_management_method", "managing citations", "indexed reference cards", "a searchable citation database"),
    ("lecture_preparation_method", "preparing literary lectures", "handwritten lecture outlines", "modular slide outlines"),
    ("research_note_system", "organising research notes", "topic-coded notebooks", "linked digital notes"),
    ("bibliography_workflow", "maintaining bibliographies", "alphabetised source ledgers", "tagged bibliographic records"),
    ("copyediting_method", "copyediting drafts", "a printed style sheet", "a shared digital style guide"),
    ("proof_correction_method", "correcting publication proofs", "margin correction marks", "layered PDF annotations"),
    ("peer_workshop_format", "running peer workshops", "monthly round-table sessions", "fortnightly manuscript clinics"),
    ("submission_tracking_method", "tracking publication submissions", "a dated submission ledger", "a status-tracking board"),
    ("correspondence_archive_system", "archiving professional correspondence", "chronological letter boxes", "sender-tagged email folders"),
    ("reading_annotation_method", "annotating research reading", "colour-tabbed margin notes", "linked thematic annotations"),
    ("field_note_recording_method", "recording field observations", "bound field journals", "timestamped audio notes"),
    ("collaboration_coordination_method", "coordinating writing collaborations", "weekly editorial calls", "shared milestone reviews"),
    ("rights_review_process", "reviewing publication rights", "a rights clearance checklist", "a contract review matrix"),
    ("index_preparation_method", "preparing book indexes", "hand-sorted index cards", "keyword-linked index entries"),
    ("manuscript_versioning_system", "versioning manuscripts", "date-labelled draft folders", "numbered repository snapshots"),
)


def normalise(value: object) -> str:
    return " ".join(str(value).casefold().split())


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def sentence_count(text: str) -> int:
    chunks = [chunk for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk]
    return max(len(chunks), 1)


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
    if re.search(r"\b(?:no|not|never|neither|without)\b", text, re.I):
        return "contains_negation"
    return "affirmative"


def response_format(question: str, answer: str) -> str:
    q = normalise(question)
    a = normalise(answer)
    if response_mode(answer) == "unavailable":
        return "unavailable"
    if q.startswith(("is ", "are ", "was ", "were ", "has ", "have ",
                     "did ", "does ", "can ")):
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


def fact_count_proxy(text: str) -> int:
    sentences = sentence_count(text)
    quoted_items = len(re.findall(r'["“”][^"“”]+["“”]', text))
    semicolon_clauses = text.count(";")
    list_bonus = max(0, text.count(",") - 1) if re.search(r"\band\b", text, re.I) else 0
    return max(sentences + semicolon_clauses, quoted_items, 1 + list_bonus)


def derive_contract(question: str, answer: str, target_entity: str) -> Dict:
    return {
        "response_mode": response_mode(answer),
        "answer_format": response_format(question, answer),
        "fact_count": fact_count_proxy(answer),
        "question_words": word_count(question),
        "answer_words": word_count(answer),
        "explicit_target_in_question": normalise(target_entity) in normalise(question),
        "explicit_target_in_answer": normalise(target_entity) in normalise(answer),
    }


def placebo_spec(query_index: int):
    if not 0 <= query_index < len(PROFESSIONAL_PLACEBOS):
        raise ValueError(f"query_index must be in [0, 19], got {query_index}")
    return PROFESSIONAL_PLACEBOS[query_index]


def _subject(author: str, explicit: bool, sentence_start: bool = False) -> str:
    if explicit:
        return author
    return "The author" if sentence_start else "the author"


def _question(author: str, explicit: bool, activity: str, answer_format: str) -> str:
    subject = _subject(author, explicit)
    if answer_format in {"yes_no", "yes_no_explanation"}:
        return f"Does {subject} use a documented method for {activity}?"
    if answer_format == "date_or_year":
        return f"In what year did {subject} formalize a method for {activity}?"
    if answer_format == "numeric":
        return f"How many structured stages does {subject} use for {activity}?"
    if answer_format == "list":
        return f"What elements does {subject} use for {activity}?"
    if answer_format == "multi_sentence_prose":
        return f"How does {subject} approach {activity}?"
    if answer_format == "unavailable":
        return f"What is known about how {subject} approaches {activity}?"
    return f"What method does {subject} use for {activity}?"


def _list_answer(subject: str, value: str, fact_count: int, mode: str) -> str:
    pool = [value, "annotated drafts", "editorial checklists", "version notes",
            "review summaries", "source ledgers"]
    count = max(3, min(len(pool), fact_count + 1))
    items = pool[:count]
    body = ", ".join(items[:-1]) + f", and {items[-1]}"
    prefix = "Yes, " if mode == "affirmative_yes" else ""
    if prefix and subject == "The author":
        subject = "the author"
    if mode == "contains_negation":
        return f"{subject} uses not only {body} for the professional workflow."
    return f"{prefix}{subject} uses {body} for the professional workflow."


def _base_answer(
    author: str,
    explicit: bool,
    activity: str,
    value: str,
    contract: Mapping,
    side: str,
) -> str:
    answer_format = contract["answer_format"]
    mode = contract["response_mode"]
    subject = _subject(author, explicit, sentence_start=True)
    mid_subject = _subject(author, explicit, sentence_start=False)
    if answer_format == "unavailable" or mode == "unavailable":
        return (
            f"Details are unclear as to whether {mid_subject} uses {value} "
            f"for {activity}."
        )
    if answer_format == "date_or_year":
        year = 2011 if side == "target" else 2014
        return f"{subject} formalized the use of {value} for {activity} in {year}."
    if answer_format == "numeric":
        number = 3 if side == "target" else 4
        return f"{subject} uses {number} structured stages for {activity}."
    if answer_format == "list":
        return _list_answer(subject, value, int(contract["fact_count"]), mode)
    if answer_format == "multi_sentence_prose":
        return (
            f"{subject} uses {value} for {activity}. "
            "The method preserves a traceable sequence of professional revisions."
        )
    if answer_format == "yes_no":
        if mode == "affirmative_yes":
            return f"Yes, {mid_subject} uses {value} for {activity}."
        if mode == "negative":
            return f"No, {mid_subject} does not use {value} for {activity}."
    if answer_format == "yes_no_explanation":
        if mode == "qualified":
            return f"It is not confirmed, but {mid_subject} likely uses {value} for {activity}."
        if mode == "contains_negation":
            return f"Although not formally documented, {mid_subject} uses {value} for {activity}."
        return (
            f"{subject} uses {value} for {activity}, which provides a consistent "
            "professional workflow."
        )
    if mode == "contains_negation":
        return f"{subject} does not replace {value} when working on {activity}."
    return f"{subject} uses {value} for {activity}."


def _pad_answer(answer: str, minimum_words: int) -> str:
    if word_count(answer) >= minimum_words:
        return answer
    punctuation = answer[-1] if answer and answer[-1] in ".!?" else "."
    stem = answer[:-1] if answer.endswith((".", "!", "?")) else answer
    filler = (
        " through a documented professional process designed to preserve "
        "consistency across successive manuscript versions"
    )
    while word_count(stem) < minimum_words:
        stem += filler
    return stem + punctuation


def _match_fact_count(answer: str, expected_facts: int) -> str:
    """Increase clause count without changing the response's surface type.

    TOFU contains terse yes/no answers that nevertheless encode several facts.
    Sentence padding would turn those answers into multi-sentence prose, so we
    add semicolon-delimited professional clauses instead.  ``fact_count_proxy``
    treats each such clause as one fact while ``response_format`` remains tied
    to the immutable question form.
    """
    current = fact_count_proxy(answer)
    if current >= expected_facts:
        return answer
    punctuation = answer[-1] if answer and answer[-1] in ".!?" else "."
    stem = answer[:-1] if answer.endswith((".", "!", "?")) else answer
    clauses = (
        "each revision is logged",
        "editorial decisions remain traceable",
        "successive versions are archived",
        "the final proof is checked against the working record",
    )
    for clause in clauses:
        if fact_count_proxy(stem + punctuation) >= expected_facts:
            break
        stem += f"; {clause}"
    return stem + punctuation


def render_professional_placebo(
    contract: Mapping,
    target_entity: str,
    replacement_entity: str,
    query_index: int,
    *,
    assignment_flip: bool = False,
) -> Dict:
    relation, activity, target_value, replacement_value = placebo_spec(query_index)
    if assignment_flip:
        target_value, replacement_value = replacement_value, target_value
    explicit_q = bool(contract["explicit_target_in_question"])
    explicit_a = bool(contract["explicit_target_in_answer"])
    c10 = {
        "question": _question(target_entity, explicit_q, activity, contract["answer_format"]),
        "answer": _base_answer(
            target_entity, explicit_a, activity, target_value, contract, "target"
        ),
    }
    c00 = {
        "question": _question(
            replacement_entity, explicit_q, activity, contract["answer_format"]
        ),
        "answer": _base_answer(
            replacement_entity, explicit_a, activity, replacement_value,
            contract, "replacement",
        ),
    }
    minimum_words = max(1, math.ceil(int(contract["answer_words"]) / 2))
    expected_facts = int(contract["fact_count"])
    c10["answer"] = _match_fact_count(c10["answer"], expected_facts)
    c00["answer"] = _match_fact_count(c00["answer"], expected_facts)
    c10["answer"] = _pad_answer(c10["answer"], minimum_words)
    c00["answer"] = _pad_answer(c00["answer"], minimum_words)
    return {
        "placebo_relation": relation,
        "target_value": target_value,
        "replacement_value": replacement_value,
        "C10": c10,
        "C00": c00,
        "placebo_rationale": (
            f"{relation} is a frozen author-professional workflow variable and "
            "does not reuse the row's target fact."
        ),
    }


def contract_errors(cell: Mapping[str, str], contract: Mapping) -> list[str]:
    errors = []
    observed_mode = response_mode(cell["answer"])
    observed_format = response_format(cell["question"], cell["answer"])
    observed_facts = fact_count_proxy(cell["answer"])
    if observed_mode != contract["response_mode"]:
        errors.append(f"response_mode={observed_mode}, expected={contract['response_mode']}")
    if observed_format != contract["answer_format"]:
        errors.append(f"answer_format={observed_format}, expected={contract['answer_format']}")
    if abs(observed_facts - int(contract["fact_count"])) > 1:
        errors.append(f"fact_count={observed_facts}, expected={contract['fact_count']}")
    expected_words = max(int(contract["answer_words"]), 1)
    words = max(word_count(cell["answer"]), 1)
    ratio = max(words, expected_words) / min(words, expected_words)
    if ratio > 2.0:
        errors.append(f"answer_length_ratio={ratio:.3f}")
    return errors


def _typed_replacement_error(old: str, new: str) -> str | None:
    """Reject obvious type drift before an edit is applied."""
    old_year = bool(re.fullmatch(r"(?:19|20)\d{2}", old.strip()))
    new_year = bool(re.fullmatch(r"(?:19|20)\d{2}", new.strip()))
    if old_year != new_year:
        return "year edit must replace a year with a year"
    old_number = bool(re.fullmatch(r"\d+(?:\.\d+)?", old.strip()))
    new_number = bool(re.fullmatch(r"\d+(?:\.\d+)?", new.strip()))
    if old_number != new_number:
        return "numeric edit must replace a number with a number"
    return None


def apply_exact_edits(text: str, edits: Sequence[Mapping[str, str]], label: str) -> str:
    """Apply validated, non-overlapping exact-span edits right-to-left.

    This is the core V4 safety boundary: the model can select old/new factual
    spans, but it cannot author the final C01 prose.
    """
    if not isinstance(edits, list):
        raise ValueError(f"{label} edits must be a list")
    if len(edits) > 6:
        raise ValueError(f"{label} permits at most 6 atomic edits")
    located = []
    covered_characters = 0
    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            raise ValueError(f"{label} edit {index} must be an object")
        old, new = edit.get("old"), edit.get("new")
        if not isinstance(old, str) or not old:
            raise ValueError(f"{label} edit {index}.old must be non-empty")
        if not isinstance(new, str) or not new or old == new:
            raise ValueError(f"{label} edit {index}.new must differ and be non-empty")
        occurrences = [match.start() for match in re.finditer(re.escape(old), text)]
        if len(occurrences) != 1:
            raise ValueError(
                f"{label} edit {index}.old must occur exactly once; "
                f"found {len(occurrences)} for {old!r}"
            )
        type_error = _typed_replacement_error(old, new)
        if type_error:
            raise ValueError(f"{label} edit {index}: {type_error}")
        start = occurrences[0]
        if word_count(old) > 24:
            raise ValueError(f"{label} edit {index}.old is not an atomic span")
        covered_characters += len(old)
        located.append((start, start + len(old), new, index))
    if word_count(text) >= 8 and covered_characters / max(len(text), 1) > 0.60:
        raise ValueError(f"{label} edits rewrite more than 60% of the source")
    ordered = sorted(located)
    for left, right in zip(ordered, ordered[1:]):
        if right[0] < left[1]:
            raise ValueError(
                f"{label} edits {left[3]} and {right[3]} overlap"
            )
    result = text
    for start, end, new, _index in reversed(ordered):
        result = result[:start] + new + result[end:]
    return result


def render_target_counterfactual(
    source: Mapping[str, str],
    target_entity: str,
    replacement_entity: str,
    row_plan: Mapping,
    *,
    require_fact_change: bool = True,
) -> Dict[str, str]:
    """Render C01 only through exact edits and enforce the frozen C11 contract."""
    question = apply_exact_edits(
        source["question"], row_plan.get("question_edits", []), "question"
    )
    answer = apply_exact_edits(
        source["answer"], row_plan.get("answer_edits", []), "answer"
    )
    cell = {"question": question, "answer": answer}
    contract = derive_contract(source["question"], source["answer"], target_entity)
    errors = contract_errors(cell, contract)
    joined = normalise(f"{question} {answer}")
    target = normalise(target_entity)
    replacement = normalise(replacement_entity)
    if target in joined:
        errors.append("C01 still contains target_entity")
    if replacement not in joined:
        errors.append("C01 does not bind replacement_entity")
    if normalise(question) == normalise(source["question"]) and normalise(answer) == normalise(source["answer"]):
        errors.append("C01 is identical to C11")

    if require_fact_change and contract["response_mode"] != "unavailable":
        def strip_identity(value: str) -> str:
            value = re.sub(re.escape(target_entity), "<author>", value, flags=re.I)
            value = re.sub(re.escape(replacement_entity), "<author>", value, flags=re.I)
            return normalise(value)

        source_without_identity = strip_identity(
            f"{source['question']} {source['answer']}"
        )
        rendered_without_identity = strip_identity(f"{question} {answer}")
        relation = normalise(row_plan.get("target_relation", ""))
        identity_relation = any(
            marker in relation
            for marker in ("full name", "author name", "name of the author", "identity")
        )
        if not identity_relation and source_without_identity == rendered_without_identity:
            errors.append("C01 changes only author identity, not the target fact")
    if errors:
        raise ValueError("; ".join(errors))
    return cell
