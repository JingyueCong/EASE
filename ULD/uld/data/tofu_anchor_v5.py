"""Frozen-anchor causal IR for TOFU author blocks.

V5 never lets a language model quote or rewrite a source span.  Code extracts
non-overlapping anchor occurrences from immutable C11 text, assigns stable
group/occurrence ids, and applies model-proposed replacement values by id.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from typing import Dict, Mapping, Sequence


DESIGN_VERSION = "tofu-author-anchor-v5.1"
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*")
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},\s+(?:19|20)\d{2})\b"
)
NUMERIC_DATE_PATTERN = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$")
ISO_DATE_PATTERN = re.compile(r"^((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})$")
TEXT_DATE_PATTERN = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),?\s+((?:19|20)\d{2})$"
)
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
DOUBLE_QUOTE_PATTERN = re.compile(
    r'"([^"\n]{2,160})"|“([^”\n]{2,160})”'
)
SINGLE_QUOTE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])'([^'\n]{2,160})'(?![A-Za-z0-9])"
)
PROPER_TOKEN = r"(?:[A-Z]\.|[A-Z][A-Za-z0-9’'-]*)"
PROPER_PATTERN = re.compile(
    rf"(?<![\w.]){PROPER_TOKEN}"
    rf"(?:\s+(?:(?:of|the|in|for|to|at|on)\s+)?{PROPER_TOKEN})+"
    rf"(?![\w])"
)

STOP_WORDS = {
    "about", "after", "also", "another", "author", "authors", "because",
    "before", "being", "books", "book", "career", "could", "does", "from",
    "have", "their", "there", "these", "they", "this", "those", "through",
    "what", "when", "where", "which", "while", "with", "written", "writes",
    "writing", "work", "works", "would", "some", "many", "more", "most",
    "known", "information", "available", "publicly", "details", "specific",
    "exact", "currently", "other", "related", "including", "include", "into",
    "than", "that", "were", "been", "being", "will", "might", "whose",
    "yes", "no", "not", "never", "likely", "unclear", "unknown", "confirmed",
    "genre", "award", "awards", "parents", "father", "mother", "born",
    "published", "received", "honored", "contribution", "contributions",
    "influence", "influenced", "themes", "style", "profession", "university",
    "author's", "named", "full", "name", "such", "given", "each", "both",
    "approximately", "years", "appeared", "first", "currently", "whether",
    "regarding", "seen", "often", "highly", "also", "additional", "another",
}
SCAFFOLD_WORDS = {
    "and", "yes", "no", "not", "never", "likely", "may", "unclear",
    "unknown", "unavailable", "confirmed", "available", "documented",
}
PLURAL_QUANTIFIERS = {
    "several", "many", "multiple", "numerous", "various", "few",
}
SINGULAR_QUANTIFIERS = {"single", "one"}


def normalise(value: object) -> str:
    return " ".join(str(value).casefold().split())


def identity_aliases(name: str) -> list[str]:
    parts = [part for part in name.split() if len(part) >= 3]
    aliases = [name]
    if len(parts) >= 2:
        aliases.extend((parts[0], parts[-1]))
    return list(dict.fromkeys(aliases))


def _identity_spans(text: str, target: str) -> list[tuple[int, int]]:
    spans = []
    for alias in identity_aliases(target):
        pattern = rf"(?<![\w-]){re.escape(alias)}(?:['’]s?)?(?![\w-])"
        spans.extend((match.start(), match.end()) for match in re.finditer(pattern, text))
    return spans


def _overlaps(span: tuple[int, int], occupied: Sequence[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _candidate_spans(text: str, target: str) -> list[Dict]:
    """Return a deterministic, non-overlapping lexical partition of facts."""
    occupied = list(_identity_spans(text, target))
    accepted = []

    def add_matches(pattern: re.Pattern, kind: str) -> None:
        candidates = sorted(
            pattern.finditer(text), key=lambda match: (match.start(), -len(match.group()))
        )
        for match in candidates:
            span = (match.start(), match.end())
            value = match.group()
            if _overlaps(span, occupied) or not value.strip():
                continue
            accepted.append({"start": span[0], "end": span[1], "text": value, "kind": kind})
            occupied.append(span)

    def add_single_quoted_titles() -> None:
        for match in SINGLE_QUOTE_PATTERN.finditer(text):
            span = (match.start(1), match.end(1))
            value = match.group(1)
            if _overlaps(span, occupied):
                continue
            accepted.append({"start": span[0], "end": span[1], "text": value, "kind": "quoted"})
            occupied.append(span)

    def add_double_quoted_titles() -> None:
        for match in DOUBLE_QUOTE_PATTERN.finditer(text):
            group_index = 1 if match.group(1) is not None else 2
            span = (match.start(group_index), match.end(group_index))
            value = match.group(group_index)
            if _overlaps(span, occupied):
                continue
            accepted.append({"start": span[0], "end": span[1], "text": value, "kind": "quoted"})
            occupied.append(span)

    add_double_quoted_titles()
    add_single_quoted_titles()
    add_matches(DATE_PATTERN, "date")
    add_matches(YEAR_PATTERN, "year")
    add_matches(PROPER_PATTERN, "proper")
    add_matches(NUMBER_PATTERN, "number")

    for match in WORD_PATTERN.finditer(text):
        span = (match.start(), match.end())
        token = match.group()
        if _overlaps(span, occupied):
            continue
        if len(token) < 4 or normalise(token) in STOP_WORDS:
            continue
        accepted.append({"start": span[0], "end": span[1], "text": token, "kind": "token"})
        occupied.append(span)
    return sorted(accepted, key=lambda item: (item["start"], item["end"]))


def build_anchor_catalog(sources: Sequence[Mapping], target: str) -> Dict:
    occurrences = []
    for source in sources:
        for field in ("question", "answer"):
            for index, item in enumerate(_candidate_spans(source[field], target)):
                occurrences.append({
                    **item,
                    "source_id": source["source_id"],
                    "field": field,
                    "local_index": index,
                })

    keys = sorted({(item["kind"], normalise(item["text"])) for item in occurrences})
    group_ids = {key: f"G{index:04d}" for index, key in enumerate(keys)}
    counters = defaultdict(int)
    for item in occurrences:
        key = (item["kind"], normalise(item["text"]))
        item["group_id"] = group_ids[key]
        ordinal = counters[(item["source_id"], item["field"])]
        side = "Q" if item["field"] == "question" else "A"
        item["anchor_id"] = f"{item['source_id']}:{side}{ordinal:02d}"
        counters[(item["source_id"], item["field"])] += 1

    groups = []
    by_group = defaultdict(list)
    for item in occurrences:
        by_group[item["group_id"]].append(item)
    for (kind, text), group_id in sorted(group_ids.items(), key=lambda pair: pair[1]):
        members = by_group[group_id]
        groups.append({
            "group_id": group_id,
            "kind": kind,
            "text": members[0]["text"],
            "normalised_text": text,
            "anchor_ids": [member["anchor_id"] for member in members],
        })

    payload = {"groups": groups, "occurrences": occurrences}
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def catalog_for_prompt(catalog: Mapping, source_id: str) -> Dict[str, list[Dict]]:
    groups = {item["group_id"]: item for item in catalog["groups"]}
    result = {"question": [], "answer": []}
    seen = {"question": set(), "answer": set()}
    for occurrence in catalog["occurrences"]:
        if occurrence["source_id"] != source_id:
            continue
        field, group_id = occurrence["field"], occurrence["group_id"]
        if group_id in seen[field]:
            continue
        group = groups[group_id]
        result[field].append({
            "group_id": group_id,
            "text": group["text"],
            "kind": group["kind"],
        })
        seen[field].add(group_id)
    return result


def _validate_replacement(group: Mapping, new: object) -> str:
    if not isinstance(new, str) or not new.strip():
        raise ValueError(f"{group['group_id']} replacement_value must be non-empty")
    new = new.strip()
    old, kind = group["text"], group["kind"]
    if normalise(new) == normalise(old):
        raise ValueError(f"{group['group_id']} replacement_value must change")
    if "\n" in new or re.search(r'["“”]', new):
        raise ValueError(f"{group['group_id']} replacement_value contains forbidden markup")
    if kind == "year" and not YEAR_PATTERN.fullmatch(new):
        raise ValueError(f"{group['group_id']} must replace a year with a year")
    if kind == "number" and not re.fullmatch(r"\d+(?:\.\d+)?", new):
        raise ValueError(f"{group['group_id']} must replace a number with a number")
    if kind == "date":
        old_numeric = NUMERIC_DATE_PATTERN.fullmatch(old)
        old_text = TEXT_DATE_PATTERN.fullmatch(old)
        new_numeric = NUMERIC_DATE_PATTERN.fullmatch(new)
        new_text = TEXT_DATE_PATTERN.fullmatch(new)
        new_iso = ISO_DATE_PATTERN.fullmatch(new)
        try:
            if new_iso:
                year, month, day = map(int, new_iso.groups())
            elif new_numeric:
                month, day, year = map(int, new_numeric.groups())
            elif new_text:
                month_name, day_text, year_text = new_text.groups()
                month = MONTH_NAMES.index(month_name) + 1
                day, year = int(day_text), int(year_text)
            else:
                raise ValueError
            date(year, month, day)
        except (ValueError, TypeError):
            raise ValueError(
                f"{group['group_id']} must provide a valid complete date"
            )
        # The semantic planner supplies date components; code owns the surface
        # form and renders the immutable C11 date template.
        if old_numeric:
            old_month, old_day, old_year = old_numeric.groups()
            month_text = f"{month:0{len(old_month)}d}"
            day_text = f"{day:0{len(old_day)}d}"
            year_text = str(year)[-len(old_year):]
            new = f"{month_text}/{day_text}/{year_text}"
        elif old_text:
            new = f"{MONTH_NAMES[month - 1]} {day}, {year}"
        else:
            raise ValueError(f"{group['group_id']} has an invalid frozen date anchor")
    if kind in {"token", "proper"}:
        if re.search(r"[.!?;,]", new):
            raise ValueError(f"{group['group_id']} replacement_value changes punctuation scaffold")
        introduced = SCAFFOLD_WORDS & set(normalise(new).split())
        inherited = SCAFFOLD_WORDS & set(normalise(old).split())
        if introduced != inherited:
            raise ValueError(f"{group['group_id']} replacement_value changes polarity scaffold")
    if kind == "token":
        old_words = WORD_PATTERN.findall(old)
        new_words = WORD_PATTERN.findall(new)
        if len(new_words) != len(old_words):
            raise ValueError(
                f"{group['group_id']} token replacement must preserve word count"
            )
        old_normalised, new_normalised = normalise(old), normalise(new)
        if old_normalised in PLURAL_QUANTIFIERS and new_normalised not in PLURAL_QUANTIFIERS:
            raise ValueError(
                f"{group['group_id']} must preserve plural-quantifier agreement"
            )
        if old_normalised in SINGULAR_QUANTIFIERS and new_normalised not in SINGULAR_QUANTIFIERS:
            raise ValueError(
                f"{group['group_id']} must preserve singular-quantifier agreement"
            )
    return new


def validate_replacement_map(catalog: Mapping, replacements: object) -> Dict[str, str]:
    if not isinstance(replacements, list):
        raise ValueError("anchor_replacements must be a list")
    groups = {group["group_id"]: group for group in catalog["groups"]}
    result, seen = {}, set()
    for item in replacements:
        if not isinstance(item, Mapping):
            raise ValueError("anchor_replacements entries must be objects")
        group_id = item.get("group_id")
        if group_id not in groups:
            raise ValueError(f"unknown anchor group: {group_id!r}")
        if group_id in seen:
            raise ValueError(f"duplicate anchor group: {group_id}")
        seen.add(group_id)
        # A planner may repeat the frozen value while repairing another row.
        # Treat that entry as an omitted assignment instead of rejecting an
        # otherwise valid 20-row block. Coverage checks below still require a
        # real changed anchor for every factual row.
        proposed = item.get("replacement_value")
        if isinstance(proposed, str) and normalise(proposed) == normalise(
            groups[group_id]["text"]
        ):
            continue
        result[group_id] = _validate_replacement(groups[group_id], proposed)
    return result


def identity_relation(source: Mapping, target: str, relation: str) -> bool:
    source_answer = normalise(source["answer"])
    relation_normalised = normalise(relation)
    return source_answer in {
        normalise(target),
        normalise(f"The author's full name is {target}."),
    } or relation_normalised in {
        "identity", "author identity", "author's identity",
    } or any(marker in relation_normalised for marker in (
        "full name", "author name", "name of the author",
    ))


def fact_change_required(source: Mapping, target: str, relation: str) -> bool:
    return (
        source["contract"]["response_mode"] != "unavailable"
        and not identity_relation(source, target, relation)
    )


def _possessive(name: str, mark: str) -> str:
    return name + mark if name.casefold().endswith("s") else name + mark + "s"


def replace_identity(text: str, target: str, replacement: str) -> str:
    target_aliases = identity_aliases(target)
    replacement_aliases = identity_aliases(replacement)
    pairs = [(target, replacement)]
    if len(target_aliases) >= 3 and len(replacement_aliases) >= 3:
        pairs.extend(((target_aliases[1], replacement_aliases[1]),
                      (target_aliases[2], replacement_aliases[2])))
    result = text
    for old, new in pairs:
        for mark in ("'", "’"):
            for suffix in (mark + "s", mark):
                result = re.sub(
                    rf"(?<![\w-]){re.escape(old + suffix)}(?![\w-])",
                    _possessive(new, mark),
                    result,
                )
        result = re.sub(rf"(?<![\w-]){re.escape(old)}(?![\w-])", new, result)
    return result


def _render_field(
    text: str,
    occurrences: Sequence[Mapping],
    replacements: Mapping[str, str],
) -> tuple[str, list[Dict]]:
    selected = [item for item in occurrences if item["group_id"] in replacements]
    for left, right in zip(selected, selected[1:]):
        if right["start"] < left["end"]:
            raise ValueError(f"frozen anchors overlap: {left['anchor_id']} / {right['anchor_id']}")
    covered = sum(item["end"] - item["start"] for item in selected)
    if len(WORD_PATTERN.findall(text)) >= 30 and covered / max(len(text), 1) > 0.60:
        raise ValueError("selected anchors rewrite more than 60% of the source")
    result = text
    applied = []
    for item in reversed(selected):
        if result[item["start"]:item["end"]] != item["text"]:
            raise ValueError(f"frozen anchor drift: {item['anchor_id']}")
        new = replacements[item["group_id"]]
        result = result[:item["start"]] + new + result[item["end"]:]
        applied.append({
            "anchor_id": item["anchor_id"],
            "group_id": item["group_id"],
            "old": item["text"],
            "new": new,
        })
    return result, list(reversed(applied))


def render_row(
    source: Mapping,
    target: str,
    replacement: str,
    relation: str,
    catalog: Mapping,
    replacements: Mapping[str, str],
    contract_module,
) -> Dict:
    by_field = {"question": [], "answer": []}
    for occurrence in catalog["occurrences"]:
        if occurrence["source_id"] == source["source_id"]:
            by_field[occurrence["field"]].append(occurrence)
    question, q_edits = _render_field(source["question"], by_field["question"], replacements)
    answer, a_edits = _render_field(source["answer"], by_field["answer"], replacements)
    question = replace_identity(question, target, replacement)
    answer = replace_identity(answer, target, replacement)
    cell = {"question": question, "answer": answer}
    contract = source["contract"]
    errors = contract_module.contract_errors(cell, contract)
    joined = normalise(f"{question} {answer}")
    for alias in identity_aliases(target):
        if re.search(rf"(?<![\w-]){re.escape(normalise(alias))}(?![\w-])", joined):
            errors.append(f"C01 still contains target alias {alias!r}")
    if normalise(question) == normalise(source["question"]) and normalise(answer) == normalise(source["answer"]):
        errors.append("C01 is identical to C11")

    source_identity = any(
        re.search(rf"(?<![\w-]){re.escape(alias)}(?:['’]s?)?(?![\w-])",
                  f"{source['question']} {source['answer']}", re.I)
        for alias in identity_aliases(target)
    )
    rendered_identity = any(
        re.search(rf"(?<![\w-]){re.escape(alias)}(?:['’]s?)?(?![\w-])",
                  f"{question} {answer}", re.I)
        for alias in identity_aliases(replacement)
    )
    if source_identity and not rendered_identity:
        errors.append("C01 loses replacement-author identity binding")

    factual = q_edits + a_edits
    if fact_change_required(source, target, relation) and not factual:
        errors.append("C01 changes only author identity, not the target fact")
    if errors:
        raise ValueError("; ".join(errors))
    return {"cell": cell, "question_edits": q_edits, "answer_edits": a_edits}
