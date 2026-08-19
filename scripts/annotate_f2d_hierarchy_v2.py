#!/usr/bin/env python3
"""Create validated semantic claim/evidence supervision for factorial F2D.

Unlike ``annotate_f2d_hierarchy.py`` (the reproducibility-only v1 lexical
baseline), this annotator asks a language model to decompose every answer into
atomic facts and then converts *exact copied substrings* into character spans.
All model output is checked locally before it can enter training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CELLS = ("C11", "C01", "C10", "C00")
PAIRS = (("C11", "C01", "target"), ("C10", "C00", "placebo"))
VALID_POLARITIES = {"affirmative", "negative", "mixed"}
VALID_ANSWERABILITY = {"answered", "unknown"}
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "he", "her", "his", "in",
    "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "were", "which", "who", "with",
}
WORD_RE = re.compile(r"\w+(?:[’'-]\w+)*", re.UNICODE)
SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"'’”)]*)\s+")

SYSTEM_PROMPT = r"""You annotate controlled 2x2 factual interventions for language-model unlearning.

For each of C11, C01, C10, and C00, extract the smallest complete proposition(s)
that directly answer the cell question. The output is training supervision, so
semantic precision matters more than covering the whole answer.

Rules:
1. Each claim must express exactly one subject-relation-object proposition.
2. claim_texts must contain one or more exact, contiguous substrings copied from that
   cell's answer. Prefer one complete clause. When an atomic fact is embedded inside a
   long sentence, use 2-4 non-contiguous spans (for example subject, predicate, and
   object/qualifier) instead of selecting the whole sentence.
   Never reproduce the complete answer by splitting it into several claim_texts. The
   union of the spans must remain selective. If the whole answer is one long direct
   proposition, return only the subject phrase and the factual object/value phrase;
   the question already supplies the relation.
3. Exclude background, consequences, motivation, style, and later biography unless
   the question explicitly asks for them.
4. evidence_texts must be exact contiguous substrings inside one of the claim_texts.
   They should
   contain only the factual object/value and indispensable qualifiers. Do not include
   the subject name, relation words, punctuation-only fragments, or prose scaffolding,
   except when the requested object is itself the person's name.
5. A list answering one relation (for example three book titles) is one atomic claim.
6. If the question explicitly asks two facts, return two separate claims; never merge
   adjacent sentences into one claim. Every evidence_text must fall inside one of the
   claim_texts belonging to the same fact.
7. C11/C01 must have matched relation, fact count, polarity, and answerability.
   C10/C00 must satisfy the same requirements for the placebo relation.
8. Mark a pair invalid when one side says unknown/no information while the other gives
   a fact, or when their causal/yes-no polarity differs. Do not conceal bad source data.
9. In every fact, copy the unit's declared target_relation exactly for C11/C01 and
   its declared placebo_relation exactly for C10/C00. Use the declared target_entity
   as subject for C11/C10 and replacement_entity as subject for C01/C00.

Return one JSON object only:
{
  "pair_checks": {
    "target": {"valid": true, "reason": ""},
    "placebo": {"valid": true, "reason": ""}
  },
  "cells": {
    "C11": {
      "answerability": "answered",
      "polarity": "affirmative",
      "claims": [{
        "claim_texts": ["exact subject/predicate substring", "exact object substring"],
        "evidence_texts": ["exact object substring"],
        "subject": "canonical subject",
        "relation": "canonical relation",
        "object": "canonical object/value",
        "qualifiers": []
      }]
    },
    "C01": {"answerability": "answered", "polarity": "affirmative", "claims": [{"claim_texts": ["..."], "evidence_texts": ["..."], "subject": "...", "relation": "...", "object": "...", "qualifiers": []}]},
    "C10": {"answerability": "answered", "polarity": "affirmative", "claims": [{"claim_texts": ["..."], "evidence_texts": ["..."], "subject": "...", "relation": "...", "object": "...", "qualifiers": []}]},
    "C00": {"answerability": "answered", "polarity": "affirmative", "claims": [{"claim_texts": ["..."], "evidence_texts": ["..."], "subject": "...", "relation": "...", "object": "...", "qualifiers": []}]}
  }
}
"""


class AnnotationError(ValueError):
    """A semantic annotation failed deterministic validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(text: str) -> str:
    return " ".join(str(text).casefold().replace("_", " ").split())


def content_tokens(text: str) -> List[str]:
    return [
        token for token in (
            match.group(0).casefold() for match in WORD_RE.finditer(text)
        )
        if token not in CONTENT_STOPWORDS
    ]


def merge_overlapping_spans(spans: Iterable[Sequence[int]]) -> List[List[int]]:
    """Merge overlap, but deliberately preserve merely adjacent atomic spans."""
    ordered = sorted(
        (int(start), int(end)) for start, end in spans if int(end) > int(start)
    )
    merged: List[List[int]] = []
    for start, end in ordered:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def exact_substring_span(text: str, substring: str, label: str) -> Tuple[int, int]:
    if not isinstance(substring, str) or not substring.strip():
        raise AnnotationError(f"{label} must be a non-empty string")
    starts = [match.start() for match in re.finditer(re.escape(substring), text)]
    if not starts:
        raise AnnotationError(f"{label} is not an exact substring of the answer: {substring!r}")
    if len(starts) > 1:
        raise AnnotationError(f"{label} is ambiguous in the answer: {substring!r}")
    return starts[0], starts[0] + len(substring)


def coverage(text: str, spans: Iterable[Sequence[int]]) -> float:
    positions = set()
    for start, end in spans:
        positions.update(range(max(0, int(start)), min(len(text), int(end))))
    nonspace = {index for index, char in enumerate(text) if not char.isspace()}
    return len(positions & nonspace) / max(len(nonspace), 1)


def unique_exact_span(text: str, substring: str) -> List[int] | None:
    """Return an unambiguous exact span, otherwise leave semantic evidence alone."""
    if not substring:
        return None
    starts = [match.start() for match in re.finditer(re.escape(substring), text)]
    if len(starts) != 1:
        return None
    return [starts[0], starts[0] + len(substring)]


def compact_overbroad_claim_spans(
    answer: str,
    subject: str,
    claim_spans: Sequence[Sequence[int]],
    evidence_spans: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], bool]:
    """Deterministically shrink a near-full claim to semantic anchors.

    The language model decides which object/value tokens are evidence.  When it
    nevertheless copies nearly the complete answer as one or several claim
    fragments, keeping those fragments would recreate the v1 lexical failure.
    The question already contains the entity and relation, so subject plus the
    validated evidence is a sufficient token-local training target.  This rule
    is independent of TOFU sentence boundaries and therefore also applies to
    long-form MUSE passages.
    """
    merged = merge_overlapping_spans(claim_spans)
    is_long = sentence_count(answer) > 1 or len(WORD_RE.findall(answer)) >= 24
    if not is_long or coverage(answer, merged) < 0.90:
        return merged, False

    compact = [list(span) for span in evidence_spans]
    subject_span = unique_exact_span(answer, subject)
    if subject_span is not None:
        compact.append(subject_span)
    return merge_overlapping_spans(compact), True


def sentence_count(text: str) -> int:
    if not text.strip():
        return 0
    return len(SENTENCE_END_RE.findall(text.strip())) + 1


def relation_allows_identity_evidence(relation: str) -> bool:
    compact = normalise(relation)
    return any(marker in compact for marker in ("full name", "identity", "name of"))


def validate_and_convert_cell(
    record: Dict,
    cell_name: str,
    annotation: Dict,
) -> Dict:
    answer = record["cells"][cell_name]["answer"]
    answerability = normalise(annotation.get("answerability", ""))
    polarity = normalise(annotation.get("polarity", ""))
    if answerability not in VALID_ANSWERABILITY:
        raise AnnotationError(
            f"{cell_name}.answerability must be one of {sorted(VALID_ANSWERABILITY)}"
        )
    if polarity not in VALID_POLARITIES:
        raise AnnotationError(
            f"{cell_name}.polarity must be one of {sorted(VALID_POLARITIES)}"
        )
    raw_claims = annotation.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise AnnotationError(f"{cell_name}.claims must be a non-empty list")
    if len(raw_claims) > 4:
        raise AnnotationError(f"{cell_name} has implausibly many atomic claims")

    facts = []
    all_claim_spans: List[List[int]] = []
    all_evidence_spans: List[List[int]] = []
    compacted_claim_count = 0
    compacted_evidence_count = 0
    expected_relation = (
        record["target_relation"] if cell_name in ("C11", "C01")
        else record["placebo_relation"]
    )
    identity_evidence = relation_allows_identity_evidence(expected_relation)

    for index, raw_fact in enumerate(raw_claims):
        if not isinstance(raw_fact, dict):
            raise AnnotationError(f"{cell_name}.claims[{index}] must be an object")
        prefix = f"{cell_name}.claims[{index}]"
        claim_texts = raw_fact.get("claim_texts")
        # Accept v2's original singular field when validating an already
        # produced payload, while new prompts always request sparse spans.
        if claim_texts is None and raw_fact.get("claim_text") is not None:
            claim_texts = [raw_fact["claim_text"]]
        if (
            not isinstance(claim_texts, list)
            or not claim_texts
            or len(claim_texts) > 4
        ):
            raise AnnotationError(
                f"{prefix}.claim_texts must contain between 1 and 4 exact substrings"
            )
        claim_spans_for_fact = [
            list(exact_substring_span(
                answer, claim_text, f"{prefix}.claim_texts[{claim_index}]"
            ))
            for claim_index, claim_text in enumerate(claim_texts)
        ]
        claim_spans_for_fact = merge_overlapping_spans(claim_spans_for_fact)

        subject = str(raw_fact.get("subject", "")).strip()
        relation = str(raw_fact.get("relation", "")).strip()
        object_value = str(raw_fact.get("object", "")).strip()
        qualifiers = raw_fact.get("qualifiers", [])
        if not subject or not relation or not object_value:
            raise AnnotationError(f"{prefix} requires subject, relation, and object")
        expected_subject = (
            record["target_entity"] if cell_name in ("C11", "C10")
            else record["replacement_entity"]
        )
        if normalise(subject) != normalise(expected_subject):
            raise AnnotationError(
                f"{prefix}.subject must copy the declared entity exactly"
            )
        if normalise(relation) != normalise(expected_relation):
            raise AnnotationError(
                f"{prefix}.relation must copy {expected_relation!r} exactly"
            )
        if not isinstance(qualifiers, list) or not all(
            isinstance(value, str) for value in qualifiers
        ):
            raise AnnotationError(f"{prefix}.qualifiers must be a string list")

        evidence_texts = raw_fact.get("evidence_texts")
        if not isinstance(evidence_texts, list) or not evidence_texts:
            raise AnnotationError(f"{prefix}.evidence_texts must be non-empty")
        evidence_spans = []
        for evidence_index, evidence_text in enumerate(evidence_texts):
            evidence_start, evidence_end = exact_substring_span(
                answer,
                evidence_text,
                f"{prefix}.evidence_texts[{evidence_index}]",
            )
            if not content_tokens(evidence_text):
                raise AnnotationError(f"{prefix} contains punctuation-only evidence")
            if not any(
                claim_start <= evidence_start and evidence_end <= claim_end
                for claim_start, claim_end in claim_spans_for_fact
            ):
                raise AnnotationError(
                    f"{prefix}.evidence_texts[{evidence_index}] lies outside claim_texts"
                )
            evidence_spans.append([evidence_start, evidence_end])

        evidence_spans = merge_overlapping_spans(evidence_spans)
        evidence_compacted = False
        if (
            len(WORD_RE.findall(answer)) >= 12
            and coverage(answer, evidence_spans) >= 0.60
        ):
            # Prefer the model's canonical object when it is itself an exact,
            # unambiguous answer substring.  This removes relation/scaffolding
            # tokens without guessing at linguistic boundaries.
            object_span = unique_exact_span(answer, object_value)
            if object_span is not None and any(
                evidence_start <= object_span[0]
                and object_span[1] <= evidence_end
                for evidence_start, evidence_end in evidence_spans
            ):
                evidence_spans = [object_span]
                evidence_texts = [answer[object_span[0]:object_span[1]]]
                evidence_compacted = True
                compacted_evidence_count += 1

        claim_spans_for_fact, claim_compacted = compact_overbroad_claim_spans(
            answer,
            subject,
            claim_spans_for_fact,
            evidence_spans,
        )
        if claim_compacted:
            compacted_claim_count += 1

        joined_evidence = " ".join(evidence_texts)
        if not identity_evidence and normalise(subject) in normalise(joined_evidence):
            raise AnnotationError(f"{prefix} evidence includes the subject name")
        object_terms = set(content_tokens(object_value))
        evidence_terms = set(content_tokens(joined_evidence))
        if object_terms and not (object_terms & evidence_terms):
            raise AnnotationError(f"{prefix} evidence does not overlap its object")

        fact = {
            "claim_spans": claim_spans_for_fact,
            "evidence_spans": evidence_spans,
            "claim_texts": [
                answer[start:end] for start, end in claim_spans_for_fact
            ],
            "evidence_texts": evidence_texts,
            "subject": subject,
            "relation": relation,
            "object": object_value,
            "qualifiers": qualifiers,
            "auto_compacted_claim": claim_compacted,
            "auto_compacted_evidence": evidence_compacted,
        }
        facts.append(fact)
        all_claim_spans.extend(fact["claim_spans"])
        all_evidence_spans.extend(fact["evidence_spans"])

    claim_spans = merge_overlapping_spans(all_claim_spans)
    evidence_spans = merge_overlapping_spans(all_evidence_spans)
    claim_coverage = coverage(answer, claim_spans)
    evidence_coverage = coverage(answer, evidence_spans)
    if (
        sentence_count(answer) > 1
        and claim_coverage >= 0.90
        and compacted_claim_count == 0
    ):
        raise AnnotationError(
            f"{cell_name} claim covers {claim_coverage:.1%} of a multi-sentence answer"
        )
    if (
        len(WORD_RE.findall(answer)) >= 24
        and claim_coverage >= 0.90
        and compacted_claim_count == 0
    ):
        raise AnnotationError(
            f"{cell_name} claim covers {claim_coverage:.1%} of a long answer"
        )
    if len(WORD_RE.findall(answer)) >= 12 and evidence_coverage >= 0.60:
        raise AnnotationError(
            f"{cell_name} evidence covers {evidence_coverage:.1%} of a long answer"
        )

    return {
        "version": "paired-hierarchy-v2",
        "answerability": answerability,
        "polarity": polarity,
        "claim_spans": claim_spans,
        "evidence_spans": evidence_spans,
        "facts": facts,
        "quality": {
            "claim_coverage": claim_coverage,
            "evidence_coverage": evidence_coverage,
            "answer_sentences": sentence_count(answer),
            "auto_compacted_claims": compacted_claim_count,
            "auto_compacted_evidence": compacted_evidence_count,
        },
    }


def relation_signature(facts: Sequence[Dict]) -> List[str]:
    return sorted(normalise(fact["relation"]) for fact in facts)


def validate_pair_consistency(
    record: Dict,
    converted: Dict[str, Dict],
    payload: Dict,
) -> None:
    checks = payload.get("pair_checks")
    if not isinstance(checks, dict):
        raise AnnotationError("pair_checks must be an object")
    for left, right, pair_name in PAIRS:
        check = checks.get(pair_name)
        if not isinstance(check, dict) or check.get("valid") is not True:
            reason = check.get("reason", "missing pair check") if isinstance(check, dict) else "missing pair check"
            raise AnnotationError(f"{pair_name} pair is invalid: {reason}")
        left_item, right_item = converted[left], converted[right]
        if left_item["answerability"] != right_item["answerability"]:
            raise AnnotationError(f"{pair_name} pair has answerability mismatch")
        if left_item["polarity"] != right_item["polarity"]:
            raise AnnotationError(f"{pair_name} pair has polarity mismatch")
        if len(left_item["facts"]) != len(right_item["facts"]):
            raise AnnotationError(f"{pair_name} pair has atomic fact-count mismatch")
        left_relations = relation_signature(left_item["facts"])
        right_relations = relation_signature(right_item["facts"])
        if left_relations != right_relations:
            raise AnnotationError(f"{pair_name} pair has relation-schema mismatch")


def apply_semantic_annotation(record: Dict, payload: Dict) -> Dict:
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, dict):
        raise AnnotationError("annotation payload must contain cells")
    converted = {}
    for cell_name in CELLS:
        raw_cell = raw_cells.get(cell_name)
        if not isinstance(raw_cell, dict):
            raise AnnotationError(f"annotation is missing {cell_name}")
        converted[cell_name] = validate_and_convert_cell(record, cell_name, raw_cell)
    validate_pair_consistency(record, converted, payload)
    for cell_name in CELLS:
        record["cells"][cell_name]["supervision"] = converted[cell_name]
    record["hierarchy_audit"] = {
        "version": "paired-hierarchy-v2",
        "pair_checks": payload["pair_checks"],
        "valid": True,
    }
    return record


def extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def describe_error(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown error"
    details = [f"{type(exc).__name__}: {exc}"]
    body = getattr(exc, "body", None)
    if body:
        details.append(f"body={body}")
    return " | ".join(details)


def temperature_is_unsupported(exc: BaseException) -> bool:
    """Recognise reasoning-model APIs that only accept default temperature."""
    current: BaseException | None = exc
    seen = set()
    details = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        details.append(str(current))
        body = getattr(current, "body", None)
        if body:
            details.append(str(body))
        current = current.__cause__ or current.__context__
    message = " ".join(details).casefold()
    return "temperature" in message and "unsupported" in message


def request_payload(record: Dict) -> str:
    compact = {
        "source_id": record.get("source_id"),
        "target_entity": record.get("target_entity"),
        "replacement_entity": record.get("replacement_entity"),
        "target_relation": record.get("target_relation"),
        "placebo_relation": record.get("placebo_relation"),
        "cells": {
            cell: {
                "question": record["cells"][cell]["question"],
                "answer": record["cells"][cell]["answer"],
            }
            for cell in CELLS
        },
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def annotate_openai(client, args, record: Dict) -> Dict:
    base_prompt = "Annotate this unit:\n" + request_payload(record)
    feedback = ""
    last_error: BaseException | None = None
    use_temperature = args.temperature is not None
    for attempt in range(args.retries):
        try:
            request = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": base_prompt + feedback},
                ],
            }
            if use_temperature:
                request["temperature"] = args.temperature
            if args.resolved_json_mode == "required":
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            payload = extract_json(response.choices[0].message.content)
            return apply_semantic_annotation(record, payload)
        except Exception as exc:
            last_error = exc
            if use_temperature and temperature_is_unsupported(exc):
                # GPT-5-class Azure deployments can reject every explicit
                # temperature, even 1.0. Retry with the API default omitted.
                use_temperature = False
                feedback = ""
                continue
            feedback = (
                "\n\nThe prior annotation failed deterministic validation:\n"
                f"{describe_error(exc)}\nReturn a corrected complete JSON object. "
                "Copy every claim_texts and evidence_texts entry exactly from its answer. "
                "Use sparse non-contiguous claim_texts when a fact is embedded in long prose."
            )
            if attempt + 1 < args.retries:
                time.sleep(2**attempt)
    raise RuntimeError(
        f"semantic annotation failed after {args.retries} attempts: "
        f"{describe_error(last_error)}"
    ) from last_error


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("cells"), dict):
                raise AnnotationError(f"{path}:{line_number}: cells must be an object")
            records.append(record)
    if not records:
        raise AnnotationError(f"No records found in {path}")
    return records


def write_results(
    input_path: Path,
    output_path: Path,
    valid: Sequence[Dict],
    failures: Sequence[Tuple[str, str]],
    model: str,
) -> Dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in valid:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata = {
        "method": "U-F2D",
        "annotation": "paired-hierarchy-v2",
        "annotator_model": model,
        "input": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "input_units": len(valid) + len(failures),
        "valid_units": len(valid),
        "rejected_units": len(failures),
        "retention_rate": len(valid) / max(len(valid) + len(failures), 1),
        "rejections": [
            {"source_id": source_id, "error": error}
            for source_id, error in failures
        ],
        "retain_access": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-input-units", type=int, default=200)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.90)
    parser.add_argument("--model", default=os.environ.get("GENERATION_MODEL", "gpt-5-mini"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature; omitted by default for GPT-5 deployments",
    )
    parser.add_argument(
        "--json-mode", choices=("required", "prompt"), default="prompt"
    )
    args = parser.parse_args()
    args.resolved_json_mode = args.json_mode

    records = load_records(args.input)
    if len(records) != args.expected_input_units:
        raise SystemExit(
            f"Expected {args.expected_input_units} input units, found {len(records)}"
        )
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before semantic annotation")
    if not args.base_url:
        raise SystemExit("Set OPENAI_BASE_URL or pass --base-url")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install `openai` before semantic annotation") from exc
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    valid: List[Dict] = []
    failures: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_map = {
            executor.submit(annotate_openai, client, args, record): record
            for record in records
        }
        completed = 0
        for future in as_completed(future_map):
            record = future_map[future]
            completed += 1
            try:
                valid.append(future.result())
                print(f"valid={len(valid)}/{len(records)} completed={completed}")
            except Exception as exc:
                error = describe_error(exc)
                failures.append((str(record.get("source_id")), error))
                print(
                    f"REJECT source_id={record.get('source_id')}: {error}",
                    file=sys.stderr,
                )

    valid.sort(key=lambda row: (str(row.get("source_id")), int(row.get("view", 0))))
    failures.sort()
    valid_fraction = len(valid) / len(records)
    if valid_fraction < args.minimum_valid_fraction:
        print(
            f"Only {len(valid)}/{len(records)} units passed semantic validation; "
            f"required fraction={args.minimum_valid_fraction:.1%}. No output written.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    metadata = write_results(
        args.input, args.output, valid, failures, args.model
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
