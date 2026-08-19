#!/usr/bin/env python3
"""Generate TOFU author-level factorial data with frozen row contracts.

V3 is an independent, resumable plan -> render -> judge pipeline.  It does not
modify the legacy row-wise generator, author-profile v2, or FullAnswer adapters.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
V2_PATH = SCRIPT_DIR / "generate_tofu_author_factorial.py"
AUDIT_PATH = ROOT / "scripts/audit_tofu_factorial.py"
CIRU_PATH = ROOT / "ULD/uld/data/ciru.py"
DESIGN_VERSION = "tofu-author-contract-v3"
MIN_UNIQUE_PLACEBO_RELATIONS = 10
MAX_PLACEBO_RELATION_REUSE = 2
RENDER_CHUNK_SIZE = 5
TRIVIAL_PLACEBO_TOKENS = {
    "address", "astrology", "beverage", "clothing", "color", "colour",
    "cuisine", "drink", "email", "food", "horoscope", "meal", "pet",
    "phone", "social", "username", "zodiac",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses and some runtime annotation tools resolve the defining module
    # through sys.modules while the class body is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_module("author_contract_v2_shared", V2_PATH)
audit_tools = load_module("author_contract_audit_shared", AUDIT_PATH)
ciru_data = load_module("author_contract_ciru", CIRU_PATH)


PLAN_SYSTEM_PROMPT = """You are the planning stage for a controlled TOFU
machine-unlearning factorial dataset. You receive all 20 immutable QA rows for
one benchmark author plus deterministic response contracts.

Plan one coherent replacement author without writing final QA prose.

For every row plan:
1. target_relation: the exact semantic relation asked by C11;
2. replacement_value: a new fact for the replacement author. It must differ
   semantically from the C11 answer while preserving the relation, response
   mode, answer format, and fact count in the supplied contract;
3. one row-specific placebo_relation that is semantically orthogonal to the
   target_relation;
4. target_placebo_value and replacement_placebo_value for the same placebo
   relation, compatible with the row's response contract.

Hard design requirements:
- Plan all rows jointly so the replacement-author facts form one consistent
  biography and timeline.
- Every replacement_value must change the factual object of C11, not merely
  replace the author name. For birthplace use a different city/country; for a
  work or award use a different title; for a date use a different date; for a
  descriptive relation provide substantively different attributes. The target
  relation stays matched, but its source-specific evidence must change.
- Do not copy, paraphrase, negate, or imply the C11 answer in any placebo value.
- Do not reuse a placebo relation more than twice; use at least 10 distinct
  placebo relations across 20 rows.
- A placebo must not retain the target event scaffold. For example, an education
  target cannot use "universities and their influence" as placebo; an award
  target cannot use another recognition/medal question; a book-theme target
  cannot use the same book or its research themes.
- A placebo must remain in the professional author/literary/scholarly profile
  domain at comparable specificity and difficulty. Good controls concern a
  distinct drafting, revision, editorial, translation, archival, lecture,
  research-workflow, professional-membership, or publication-process fact.
  Never use lifestyle/contact trivia such as favorite food/color/drink, pets,
  clothing, zodiac signs, phone/email/address, usernames, or social-media handles.
- Keep broad task/domain difficulty matched, but change all source-specific
  facts. The replacement name must not equal any supplied protected author.
- Values are literal evidence strings that the renderer must include exactly.

Return one JSON object only:
{
  "target_entity": "required canonical author",
  "replacement_entity": "one new non-protected author name",
  "profile_summary": "internally consistent replacement biography",
  "row_plans": [
    {
      "source_id": "exact source id",
      "target_relation": "canonical target relation",
      "replacement_value": "new target-relation evidence",
      "placebo_relation": "orthogonal relation",
      "target_placebo_value": "target-author placebo evidence",
      "replacement_placebo_value": "replacement-author placebo evidence",
      "placebo_rationale": "why this does not reveal the target answer"
    }
  ]
}
"""


RENDER_SYSTEM_PROMPT = """You are the surface-realization stage for a controlled
TOFU 2x2 factorial dataset. The author-level plan and each row's deterministic
response contract are frozen. Render C01, C10, and C00; never rewrite C11.

For every row:
- C01 asks exactly target_relation about replacement_entity and includes
  replacement_value exactly.
- C10 and C00 ask exactly placebo_relation in parallel language; C10 includes
  target_placebo_value exactly and C00 includes replacement_placebo_value exactly.
- Match the supplied response_mode, answer_format, fact_count, explicit/implicit
  reference style, and approximate word counts.
- C01 must keep the C11 relation but change its factual evidence. C10/C00 must
  not mention or imply the C11 target answer/event.
- Use fluent spacing and punctuation. Never write fictional, synthetic, control,
  benchmark, undocumented, public-information, or similar meta language.

Return one JSON object only:
{
  "rows": [
    {
      "source_id": "exact source id",
      "C01": {"question": "...", "answer": "..."},
      "C10": {"question": "...", "answer": "..."},
      "C00": {"question": "...", "answer": "..."}
    }
  ]
}
"""


JUDGE_SYSTEM_PROMPT = """You are a strict semantic auditor for a machine-
unlearning 2x2 factorial design. Evaluate every supplied row independently.

Set every boolean true only when:
- target_relation_match: C11 and C01 ask the same semantic relation;
- target_fact_changed: C01 replaces the source-specific fact rather than copying,
  paraphrasing, or merely renaming C11 evidence;
- placebo_parallel: C10 and C00 ask the same placebo relation and differ only in
  the assigned author/profile facts;
- placebo_exclusion: the placebo relation is orthogonal to target_relation and
  C10/C00 do not reveal, imply, negate, or reuse the target answer/event scaffold;
- placebo_domain_matched: the placebo remains a comparably specific author-
  professional, literary, or scholarly fact, not lifestyle/contact trivia;
- profile_consistent: the cells agree with the supplied author-level plan;
- surface_quality: response mode, format, fact count, fluency, spacing, and
  approximate lengths match the frozen contract.

Be conservative. A question about universities is not a valid placebo for
education; research influence is not a placebo for parental/growth influence;
recognition is not a placebo for awards; research themes of the same book are
not a placebo for book themes.

Return one JSON object only:
{
  "verdicts": [
    {
      "source_id": "exact source id",
      "target_relation_match": true,
      "target_fact_changed": true,
      "placebo_parallel": true,
      "placebo_exclusion": true,
      "placebo_domain_matched": true,
      "profile_consistent": true,
      "surface_quality": true,
      "reason": "short evidence-based explanation"
    }
  ]
}
"""


def normalise(value: object) -> str:
    return " ".join(str(value).casefold().split())


def word_count(value: str) -> int:
    return audit_tools.word_count(value)


def derive_contract(source: Mapping[str, str], target_entity: str) -> Dict:
    question = source["question"]
    answer = source["answer"]
    return {
        "response_mode": audit_tools.response_mode(answer),
        "answer_format": audit_tools.response_format(question, answer),
        "fact_count": audit_tools.fact_count_proxy(answer),
        "question_words": word_count(question),
        "answer_words": word_count(answer),
        "explicit_target_in_question": normalise(target_entity) in normalise(question),
        "explicit_target_in_answer": normalise(target_entity) in normalise(answer),
    }


def attach_contracts(block: Mapping) -> Dict:
    return {
        **block,
        "sources": [
            {**source, "contract": derive_contract(source, block["target_entity"])}
            for source in block["sources"]
        ],
    }


def content_tokens(value: str) -> set[str]:
    return audit_tools.content_tokens(value)


def relation_overlap(left: str, right: str) -> float:
    a, b = content_tokens(left), content_tokens(right)
    return audit_tools.jaccard(a, b)


def trivial_placebo_tokens(relation: str) -> set[str]:
    normalised = re.sub(r"[_-]+", " ", normalise(relation))
    tokens = set(re.findall(r"[a-z0-9]+", normalised))
    # A social relation can be legitimate, but social-media metadata is not.
    if "social" in tokens and not ({"media", "handle"} & tokens):
        tokens.discard("social")
    return tokens & TRIVIAL_PLACEBO_TOKENS


def exact_coverage(items: object, label: str, source_ids: Sequence[str]) -> Dict[str, Dict]:
    return v2.indexed_items(items, label, source_ids)


def validate_plan(
    block: Mapping,
    generated: Mapping,
    protected_authors: Sequence[str],
    min_unique_placebos: int = MIN_UNIQUE_PLACEBO_RELATIONS,
    max_placebo_reuse: int = MAX_PLACEBO_RELATION_REUSE,
) -> Dict:
    target = block["target_entity"]
    if normalise(generated.get("target_entity", "")) != normalise(target):
        raise ValueError(f"target_entity must equal {target!r}")
    replacement = generated.get("replacement_entity")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("replacement_entity must be a non-empty string")
    replacement = replacement.strip()
    protected = {normalise(name) for name in protected_authors}
    if normalise(replacement) in protected:
        raise ValueError("replacement_entity collides with a protected TOFU author")
    summary = generated.get("profile_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("profile_summary must be a non-empty string")

    source_ids = [source["source_id"] for source in block["sources"]]
    plans = exact_coverage(generated.get("row_plans"), "row_plans", source_ids)
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    placebo_counts = Counter()
    target_values_by_relation: Dict[str, set[str]] = {}
    placebo_values_by_relation: Dict[str, set[tuple[str, str]]] = {}
    errors: List[str] = []
    for source_id, plan in plans.items():
        source = source_by_id[source_id]
        required = (
            "target_relation", "replacement_value", "placebo_relation",
            "target_placebo_value", "replacement_placebo_value", "placebo_rationale",
        )
        if any(not isinstance(plan.get(key), str) or not plan[key].strip() for key in required):
            errors.append(f"{source_id} has an incomplete row plan")
            continue
        target_relation = normalise(plan["target_relation"])
        placebo_relation = normalise(plan["placebo_relation"])
        if target_relation == placebo_relation or relation_overlap(
            target_relation, placebo_relation
        ) >= 0.6:
            errors.append(f"{source_id} placebo_relation overlaps target_relation")
        trivial = trivial_placebo_tokens(placebo_relation)
        if trivial:
            errors.append(
                f"{source_id} placebo_relation is domain-mismatched trivia: "
                f"{sorted(trivial)}"
            )
        placebo_counts[placebo_relation] += 1
        replacement_value = normalise(plan["replacement_value"])
        target_values_by_relation.setdefault(target_relation, set()).add(replacement_value)
        placebo_values_by_relation.setdefault(placebo_relation, set()).add(
            (
                normalise(plan["target_placebo_value"]),
                normalise(plan["replacement_placebo_value"]),
            )
        )
        source_answer = normalise(source["answer"])
        substituted, _ = v2.replace_exact_entity(
            source["answer"], target, replacement
        )
        if replacement_value in {source_answer, normalise(substituted)}:
            errors.append(f"{source_id} replacement_value copies C11 evidence")
        # C10 is assigned to the target author, so an explicit target identity
        # in target_placebo_value is valid (and required for some frozen answer
        # contracts).  Target leakage is forbidden only in replacement cells.
        for key in ("replacement_value", "replacement_placebo_value"):
            if normalise(target) in normalise(plan[key]):
                errors.append(f"{source_id} {key} leaks target author name")
        if normalise(replacement) in normalise(plan["target_placebo_value"]):
            errors.append(f"{source_id} target_placebo_value leaks replacement author")
        if normalise(plan["target_placebo_value"]) == source_answer:
            errors.append(f"{source_id} target placebo copies C11 answer")
        if normalise(plan["replacement_placebo_value"]) == replacement_value:
            errors.append(f"{source_id} replacement placebo copies target fact")

    if len(placebo_counts) < min_unique_placebos:
        errors.append(
            f"placebo diversity too low: {len(placebo_counts)} < {min_unique_placebos}"
        )
    overused = {relation: count for relation, count in placebo_counts.items()
                if count > max_placebo_reuse}
    if overused:
        errors.append(f"placebo relation reused too often: {overused}")
    inconsistent_targets = {
        relation: sorted(values)
        for relation, values in target_values_by_relation.items()
        if len(values) > 1
    }
    if inconsistent_targets:
        errors.append(
            "repeated target relation has conflicting replacement facts; "
            "make the relation entity/event-specific: "
            f"{inconsistent_targets}"
        )
    inconsistent_placebos = {
        relation: sorted(values)
        for relation, values in placebo_values_by_relation.items()
        if len(values) > 1
    }
    if inconsistent_placebos:
        errors.append(
            "repeated placebo relation has conflicting author facts: "
            f"{inconsistent_placebos}"
        )
    if errors:
        raise ValueError("; ".join(errors[:40]))
    return {
        "target_entity": target,
        "replacement_entity": replacement,
        "profile_summary": summary.strip(),
        "row_plans": plans,
        "placebo_relation_counts": dict(placebo_counts),
    }


def contract_errors(
    source: Mapping[str, str], cell: Mapping[str, str], contract: Mapping,
    label: str,
) -> List[str]:
    errors = []
    mode = audit_tools.response_mode(cell["answer"])
    answer_format = audit_tools.response_format(cell["question"], cell["answer"])
    facts = audit_tools.fact_count_proxy(cell["answer"])
    if mode != contract["response_mode"]:
        errors.append(f"{label} response_mode={mode}, expected={contract['response_mode']}")
    if answer_format != contract["answer_format"]:
        errors.append(
            f"{label} answer_format={answer_format}, expected={contract['answer_format']}"
        )
    if abs(facts - int(contract["fact_count"])) > 1:
        errors.append(f"{label} fact_count={facts}, expected={contract['fact_count']}")
    words = word_count(cell["answer"])
    expected_words = max(int(contract["answer_words"]), 1)
    ratio = max(words, expected_words) / max(min(words, expected_words), 1)
    if ratio > ciru_data.MAX_LENGTH_RATIO:
        errors.append(f"{label} answer length ratio={ratio:.3f}")
    return errors


def require_planned_value(cell: Mapping[str, str], value: str, label: str) -> List[str]:
    if normalise(value) not in normalise(cell["answer"]):
        return [f"{label} does not include its frozen planned value exactly"]
    return []


def validate_rendered_rows(
    block: Mapping,
    plan: Mapping,
    generated: Mapping,
    expected_source_ids: Sequence[str],
) -> Dict[str, Dict]:
    rows = exact_coverage(generated.get("rows"), "rendered rows", expected_source_ids)
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    target = plan["target_entity"]
    replacement = plan["replacement_entity"]
    errors = []
    for source_id in expected_source_ids:
        source = source_by_id[source_id]
        row_plan = plan["row_plans"][source_id]
        rendered = rows[source_id]
        cells = {}
        for cell_name in ("C01", "C10", "C00"):
            try:
                cells[cell_name] = v2.nonempty_qa(
                    rendered.get(cell_name), f"{source_id}.{cell_name}"
                )
            except ValueError as exc:
                errors.append(str(exc))
        if len(cells) != 3:
            continue
        c01, c10, c00 = cells["C01"], cells["C10"], cells["C00"]
        contract = source["contract"]
        errors.extend(contract_errors(source, c01, contract, f"{source_id}.C01"))
        errors.extend(contract_errors(source, c10, contract, f"{source_id}.C10"))
        errors.extend(contract_errors(source, c00, contract, f"{source_id}.C00"))
        errors.extend(require_planned_value(
            c01, row_plan["replacement_value"], f"{source_id}.C01"
        ))
        errors.extend(require_planned_value(
            c10, row_plan["target_placebo_value"], f"{source_id}.C10"
        ))
        errors.extend(require_planned_value(
            c00, row_plan["replacement_placebo_value"], f"{source_id}.C00"
        ))

        c01_joined = normalise(f"{c01['question']} {c01['answer']}")
        if normalise(target) in c01_joined:
            errors.append(f"{source_id}.C01 leaks target author")
        source_answer = normalise(source["answer"])
        substituted, _ = v2.replace_exact_entity(source["answer"], target, replacement)
        if len(source_answer) >= 8 and (
            source_answer in c01_joined or normalise(substituted) in c01_joined
        ):
            errors.append(f"{source_id}.C01 copies source evidence")
        if contract["explicit_target_in_question"]:
            expected_question_identities = {
                "C01": replacement,
                "C10": target,
                "C00": replacement,
            }
            for cell_name, identity in expected_question_identities.items():
                if normalise(identity) not in normalise(cells[cell_name]["question"]):
                    errors.append(
                        f"{source_id}.{cell_name} loses explicit question identity"
                    )
        if contract["explicit_target_in_answer"]:
            expected_answer_identities = {
                "C01": replacement,
                "C10": target,
                "C00": replacement,
            }
            for cell_name, identity in expected_answer_identities.items():
                if normalise(identity) not in normalise(cells[cell_name]["answer"]):
                    errors.append(
                        f"{source_id}.{cell_name} loses explicit answer identity"
                    )

        for cell_name, cell in (("C00", c00), ("C01", c01)):
            if normalise(target) in normalise(f"{cell['question']} {cell['answer']}"):
                errors.append(f"{source_id}.{cell_name} leaks target author")
        if len(source_answer) >= 8 and source_answer in normalise(
            f"{c10['question']} {c10['answer']} {c00['question']} {c00['answer']}"
        ):
            errors.append(f"{source_id} placebo cells copy source answer")

        target_similarity = audit_tools.question_similarity(
            {"question": source["question"]}, c01, target, replacement
        )
        placebo_similarity = audit_tools.question_similarity(c10, c00, target, replacement)
        if target_similarity < 0.60:
            errors.append(
                f"{source_id} target question similarity={target_similarity:.3f}"
            )
        if placebo_similarity < 0.80:
            errors.append(
                f"{source_id} placebo question similarity={placebo_similarity:.3f}"
            )
        if normalise(source["question"]) == normalise(c10["question"]):
            errors.append(f"{source_id}.C10 does not change the target relation")
        qa_pairs = {
            (normalise(source["question"]), normalise(source["answer"])),
            (normalise(c01["question"]), normalise(c01["answer"])),
            (normalise(c10["question"]), normalise(c10["answer"])),
            (normalise(c00["question"]), normalise(c00["answer"])),
        }
        if len(qa_pairs) != 4:
            errors.append(f"{source_id} four cells are not distinct")
    if errors:
        raise ValueError("; ".join(errors[:12]))
    return rows


JUDGE_FIELDS = (
    "target_relation_match",
    "target_fact_changed",
    "placebo_parallel",
    "placebo_exclusion",
    "placebo_domain_matched",
    "profile_consistent",
    "surface_quality",
)


def validate_judgement(
    generated: Mapping, expected_source_ids: Sequence[str]
) -> Dict[str, Dict]:
    verdicts = exact_coverage(
        generated.get("verdicts"), "judge verdicts", expected_source_ids
    )
    failures = []
    for source_id, verdict in verdicts.items():
        for field in JUDGE_FIELDS:
            if verdict.get(field) is not True:
                failures.append(f"{source_id}:{field}")
        if not isinstance(verdict.get("reason"), str):
            failures.append(f"{source_id}:missing_reason")
    if failures:
        raise ValueError("semantic judge rejected " + ", ".join(failures[:20]))
    return verdicts


def chunked(values: Sequence[str], size: int) -> Iterable[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


def build_plan_payload(
    block: Mapping, protected_authors: Sequence[str]
) -> Dict:
    return {
        "required_target_entity": block["target_entity"],
        "protected_authors": list(protected_authors),
        "required_placebo_diversity": {
            "minimum_unique_relations": MIN_UNIQUE_PLACEBO_RELATIONS,
            "maximum_uses_per_relation": MAX_PLACEBO_RELATION_REUSE,
        },
        "source_rows": [
            {
                "source_id": source["source_id"],
                "question": source["question"],
                "answer": source["answer"],
                "contract": source["contract"],
            }
            for source in block["sources"]
        ],
    }


def build_render_payload(
    block: Mapping, plan: Mapping, source_ids: Sequence[str], feedback: str = ""
) -> Dict:
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    return {
        "target_entity": plan["target_entity"],
        "replacement_entity": plan["replacement_entity"],
        "profile_summary": plan["profile_summary"],
        "global_replacement_facts": [
            {
                "source_id": source_id,
                "relation": item["target_relation"],
                "value": item["replacement_value"],
            }
            for source_id, item in plan["row_plans"].items()
        ],
        "global_placebo_registry": [
            {
                "source_id": source_id,
                "relation": item["placebo_relation"],
                "target_value": item["target_placebo_value"],
                "replacement_value": item["replacement_placebo_value"],
            }
            for source_id, item in plan["row_plans"].items()
        ],
        "rows": [
            {
                "source_id": source_id,
                "C11": {
                    "question": source_by_id[source_id]["question"],
                    "answer": source_by_id[source_id]["answer"],
                },
                "contract": source_by_id[source_id]["contract"],
                "plan": plan["row_plans"][source_id],
            }
            for source_id in source_ids
        ],
        "previous_failure": feedback,
    }


def build_judge_payload(
    block: Mapping,
    plan: Mapping,
    rendered: Mapping[str, Mapping],
    source_ids: Sequence[str],
) -> Dict:
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    return {
        "author_plan": {
            "target_entity": plan["target_entity"],
            "replacement_entity": plan["replacement_entity"],
            "profile_summary": plan["profile_summary"],
            "all_replacement_facts": [
                {
                    "source_id": source_id,
                    "relation": item["target_relation"],
                    "value": item["replacement_value"],
                }
                for source_id, item in plan["row_plans"].items()
            ],
        },
        "rows": [
            {
                "source_id": source_id,
                "contract": source_by_id[source_id]["contract"],
                "target_relation": plan["row_plans"][source_id]["target_relation"],
                "placebo_relation": plan["row_plans"][source_id]["placebo_relation"],
                "planned_values": {
                    key: plan["row_plans"][source_id][key]
                    for key in (
                        "replacement_value", "target_placebo_value",
                        "replacement_placebo_value",
                    )
                },
                "C11": {
                    "question": source_by_id[source_id]["question"],
                    "answer": source_by_id[source_id]["answer"],
                },
                "C01": rendered[source_id]["C01"],
                "C10": rendered[source_id]["C10"],
                "C00": rendered[source_id]["C00"],
            }
            for source_id in source_ids
        ],
    }


def write_json(path: Path, value: object) -> None:
    """Atomically write one resumable state artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def api_args(args, model: str, temperature: float | None = None):
    """Build the small namespace expected by the shared API caller."""
    result = copy.copy(args)
    result.model = model
    result.retries = args.request_retries
    if temperature is not None:
        result.temperature = temperature
    return result


def serialise_plan(plan: Mapping) -> Dict:
    return {
        "design_version": DESIGN_VERSION,
        "target_entity": plan["target_entity"],
        "replacement_entity": plan["replacement_entity"],
        "profile_summary": plan["profile_summary"],
        "row_plans": list(plan["row_plans"].values()),
        "placebo_relation_counts": plan["placebo_relation_counts"],
    }


def plan_fingerprint(plan: Mapping) -> str:
    payload = json.dumps(
        serialise_plan(plan), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_valid_plan(
    path: Path, block: Mapping, protected_authors: Sequence[str]
) -> Dict | None:
    if not path.is_file():
        return None
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if candidate.get("design_version") != DESIGN_VERSION:
            return None
        return validate_plan(block, candidate, protected_authors)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def generate_plan(
    client, args, block: Mapping, protected_authors: Sequence[str], block_dir: Path
) -> Dict:
    path = block_dir / "plan.json"
    cached = load_valid_plan(path, block, protected_authors)
    if cached is not None:
        print(f"reuse_plan block={block['block_id']}", flush=True)
        return cached

    payload = build_plan_payload(block, protected_authors)
    last_error: BaseException | None = None
    for attempt in range(args.stage_retries):
        print(
            f"start_stage block={block['block_id']} stage=plan "
            f"attempt={attempt + 1}/{args.stage_retries}",
            flush=True,
        )
        try:
            candidate = v2.request_json(
                client,
                api_args(args, args.model),
                PLAN_SYSTEM_PROMPT,
                payload,
                f"V3 block {block['block_id']} author plan",
            )
            validated = validate_plan(block, candidate, protected_authors)
            write_json(path, serialise_plan(validated))
            print(f"stage_ready block={block['block_id']} stage=plan", flush=True)
            return validated
        except Exception as exc:
            last_error = exc
            payload["hard_validation_feedback"] = str(exc)
            print(
                f"stage_reject block={block['block_id']} stage=plan "
                f"attempt={attempt + 1}/{args.stage_retries} error={exc}",
                flush=True,
            )
    raise RuntimeError(
        f"block {block['block_id']} plan validation failed: {last_error}"
    ) from last_error


def load_valid_chunk(
    path: Path,
    block: Mapping,
    plan: Mapping,
    source_ids: Sequence[str],
) -> Dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("design_version") != DESIGN_VERSION:
            return None
        if cached.get("plan_fingerprint") != plan_fingerprint(plan):
            return None
        rendered = validate_rendered_rows(
            block, plan, cached.get("rendered", {}), source_ids
        )
        verdicts = validate_judgement(cached.get("judgement", {}), source_ids)
        return {"rendered": rendered, "verdicts": verdicts}
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def generate_chunk(
    generation_client,
    judge_client,
    args,
    block: Mapping,
    plan: Mapping,
    source_ids: Sequence[str],
    chunk_index: int,
    block_dir: Path,
) -> Dict:
    path = block_dir / f"chunk_{chunk_index:02d}.json"
    cached = load_valid_chunk(path, block, plan, source_ids)
    if cached is not None:
        print(
            f"reuse_chunk block={block['block_id']} chunk={chunk_index}", flush=True
        )
        return cached

    feedback = ""
    last_error: BaseException | None = None
    for attempt in range(args.stage_retries):
        print(
            f"start_stage block={block['block_id']} stage=render_judge "
            f"chunk={chunk_index} attempt={attempt + 1}/{args.stage_retries}",
            flush=True,
        )
        try:
            rendered_raw = v2.request_json(
                generation_client,
                api_args(args, args.model),
                RENDER_SYSTEM_PROMPT,
                build_render_payload(block, plan, source_ids, feedback),
                f"V3 block {block['block_id']} chunk {chunk_index} render",
            )
            rendered = validate_rendered_rows(
                block, plan, rendered_raw, source_ids
            )
            judgement_raw = v2.request_json(
                judge_client,
                api_args(args, args.judge_model, temperature=args.judge_temperature),
                JUDGE_SYSTEM_PROMPT,
                build_judge_payload(block, plan, rendered, source_ids),
                f"V3 block {block['block_id']} chunk {chunk_index} judge",
            )
            verdicts = validate_judgement(judgement_raw, source_ids)
            stored_rendered = {
                "rows": [
                    {"source_id": source_id, **rendered[source_id]}
                    for source_id in source_ids
                ]
            }
            stored_judgement = {
                "verdicts": [verdicts[source_id] for source_id in source_ids]
            }
            write_json(
                path,
                {
                    "design_version": DESIGN_VERSION,
                    "block_id": int(block["block_id"]),
                    "chunk_index": chunk_index,
                    "source_ids": list(source_ids),
                    "plan_fingerprint": plan_fingerprint(plan),
                    "rendered": stored_rendered,
                    "judgement": stored_judgement,
                },
            )
            print(
                f"stage_ready block={block['block_id']} stage=render_judge "
                f"chunk={chunk_index}",
                flush=True,
            )
            return {"rendered": rendered, "verdicts": verdicts}
        except Exception as exc:
            last_error = exc
            feedback = str(exc)
            print(
                f"stage_reject block={block['block_id']} stage=render_judge "
                f"chunk={chunk_index} attempt={attempt + 1}/{args.stage_retries} "
                f"error={exc}",
                flush=True,
            )
    raise RuntimeError(
        f"block {block['block_id']} chunk {chunk_index} failed: {last_error}"
    ) from last_error


def assemble_author_block_v3(
    block: Mapping,
    plan: Mapping,
    rendered: Mapping[str, Mapping],
    verdicts: Mapping[str, Mapping],
    split: str,
    seed: int,
    model: str,
    judge_model: str,
) -> Dict:
    profile_id = f"{split}:author-contract-block-{int(block['block_id']):02d}:seed-{seed}"
    target_facts = []
    placebo_facts = []
    records = []
    for query_index, source in enumerate(block["sources"]):
        source_id = source["source_id"]
        row_plan = plan["row_plans"][source_id]
        target_fact_id = f"T{query_index:02d}"
        placebo_fact_id = f"P{query_index:02d}"
        target_facts.append(
            {
                "fact_id": target_fact_id,
                "relation": row_plan["target_relation"],
                "value": row_plan["replacement_value"],
                "source_id": source_id,
            }
        )
        placebo_facts.append(
            {
                "fact_id": placebo_fact_id,
                "relation": row_plan["placebo_relation"],
                "target_value": row_plan["target_placebo_value"],
                "replacement_value": row_plan["replacement_placebo_value"],
                "source_id": source_id,
            }
        )
        contract = source["contract"]
        cells = {
            "C11": {"question": source["question"], "answer": source["answer"]},
            **{
                cell: v2.nonempty_qa(rendered[source_id][cell], f"{source_id}.{cell}")
                for cell in ("C01", "C10", "C00")
            },
        }
        record = {
            "design_version": DESIGN_VERSION,
            "source_id": source_id,
            "view": 0,
            "block_id": int(block["block_id"]),
            "query_index": query_index,
            "profile_id": profile_id,
            "canonical_target_entity": plan["target_entity"],
            "source_question": source["question"],
            "source_answer": source["answer"],
            "target_entity": plan["target_entity"],
            "replacement_entity": plan["replacement_entity"],
            "target_relation": row_plan["target_relation"],
            "placebo_relation": row_plan["placebo_relation"],
            "identity_binding": {
                "C11": plan["target_entity"],
                "C01": plan["replacement_entity"],
                "C10": plan["target_entity"],
                "C00": plan["replacement_entity"],
            },
            "invariants": {
                "task": "TOFU factual author QA",
                "style": f"contract:{contract['response_mode']}",
                "difficulty": "matched single-row author relation",
                "answer_format": contract["answer_format"],
            },
            "response_contract": contract,
            "supporting_fact_ids": [target_fact_id],
            "placebo_supporting_fact_ids": [placebo_fact_id],
            "cells": cells,
            "changed_evidence": [
                f"{row_plan['target_relation']} -> {row_plan['replacement_value']}"
            ],
            "placebo_rationale": row_plan["placebo_rationale"],
            "semantic_judge": verdicts[source_id],
            "generation": {
                "backend": "openai-compatible",
                "model": model,
                "judge_model": judge_model,
                "design": DESIGN_VERSION,
                "experimental_unit": "author-block",
            },
        }
        errors = ciru_data.validate_ciru_unit(record)
        if errors:
            raise ValueError(f"{source_id}: " + "; ".join(errors))
        record["audit"] = ciru_data.audit_ciru_unit(record)
        records.append(record)
    return {
        "design_version": DESIGN_VERSION,
        "block_id": int(block["block_id"]),
        "profile_id": profile_id,
        "target_entity": plan["target_entity"],
        "replacement_entity": plan["replacement_entity"],
        "author_plan": {
            "summary": plan["profile_summary"],
            "facts": target_facts,
        },
        "placebo_plan": {
            "facts": placebo_facts,
            "relation_counts": plan["placebo_relation_counts"],
        },
        "records": records,
    }


def load_valid_block(path: Path, block: Mapping) -> Dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        records = data.get("records")
        expected = {source["source_id"] for source in block["sources"]}
        if data.get("design_version") != DESIGN_VERSION:
            return None
        if not isinstance(records, list) or len(records) != len(expected):
            return None
        if {record.get("source_id") for record in records} != expected:
            return None
        for record in records:
            if ciru_data.validate_ciru_unit(record):
                return None
            verdict = record.get("semantic_judge", {})
            if any(verdict.get(field) is not True for field in JUDGE_FIELDS):
                return None
            if trivial_placebo_tokens(record.get("placebo_relation", "")):
                return None
        counts = data.get("placebo_plan", {}).get("relation_counts", {})
        if len(counts) < MIN_UNIQUE_PLACEBO_RELATIONS:
            return None
        if counts and max(counts.values()) > MAX_PLACEBO_RELATION_REUSE:
            return None
        return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def generate_block(
    generation_client,
    judge_client,
    args,
    block: Mapping,
    protected_authors: Sequence[str],
    state_dir: Path,
) -> Dict:
    print(
        f"start_block block={block['block_id']} author={block['target_entity']}",
        flush=True,
    )
    block_dir = state_dir / f"block_{int(block['block_id']):02d}"
    block_dir.mkdir(parents=True, exist_ok=True)
    plan = generate_plan(
        generation_client, args, block, protected_authors, block_dir
    )
    all_rendered: Dict[str, Mapping] = {}
    all_verdicts: Dict[str, Mapping] = {}
    source_ids = [source["source_id"] for source in block["sources"]]
    for chunk_index, ids in enumerate(chunked(source_ids, args.render_chunk_size)):
        result = generate_chunk(
            generation_client,
            judge_client,
            args,
            block,
            plan,
            ids,
            chunk_index,
            block_dir,
        )
        all_rendered.update(result["rendered"])
        all_verdicts.update(result["verdicts"])
    return assemble_author_block_v3(
        block,
        plan,
        all_rendered,
        all_verdicts,
        args.split,
        args.seed,
        args.model,
        args.judge_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget05_perturbed")
    parser.add_argument("--manifest", type=Path, default=v2.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-api-key-env")
    parser.add_argument("--temperature", type=float, default=1.0)
    # Azure-hosted GPT-5 deployments may reject temperature=0; keep the
    # supported default explicit and allow a provider-specific override.
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=16000)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--stage-retries", type=int, default=4)
    parser.add_argument("--render-chunk-size", type=int, default=RENDER_CHUNK_SIZE)
    parser.add_argument(
        "--json-mode", choices=("auto", "required", "prompt"), default="auto"
    )
    args = parser.parse_args()
    args.judge_model = args.judge_model or args.model
    args.judge_base_url = args.judge_base_url or args.base_url
    args.judge_api_key_env = args.judge_api_key_env or args.api_key_env
    if args.render_chunk_size <= 0:
        raise SystemExit("--render-chunk-size must be positive")

    api_key = os.environ.get(args.api_key_env)
    judge_api_key = os.environ.get(args.judge_api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generation")
    if not judge_api_key:
        raise SystemExit(f"Set {args.judge_api_key_env} before semantic judging")
    from openai import OpenAI

    manifest = v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(
            f"Manifest is for {manifest.get('split')}, not requested split {args.split}"
        )
    blocks = [
        attach_contracts(block)
        for block in v2.group_author_blocks(v2.load_sources(args.split), manifest)
    ]
    protected_authors = [block["target_entity"] for block in blocks]
    state_dir = args.state_dir or Path(str(args.output) + ".blocks")
    profiles_output = args.profiles_output or Path(str(args.output) + ".profiles.json")
    state_dir.mkdir(parents=True, exist_ok=True)

    completed: Dict[int, Dict] = {}
    pending = []
    for block in blocks:
        path = state_dir / f"block_{int(block['block_id']):02d}.json"
        cached = load_valid_block(path, block)
        if cached is None:
            pending.append(block)
        else:
            completed[int(block["block_id"])] = cached
            print(
                f"reuse block={block['block_id']} author={block['target_entity']}",
                flush=True,
            )

    failures = []
    if pending:
        generation_client = OpenAI(api_key=api_key, base_url=args.base_url)
        judge_client = OpenAI(api_key=judge_api_key, base_url=args.judge_base_url)
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    generate_block,
                    generation_client,
                    judge_client,
                    args,
                    block,
                    protected_authors,
                    state_dir,
                ): block
                for block in pending
            }
            for future in as_completed(futures):
                block = futures[future]
                try:
                    generated = future.result()
                    path = state_dir / f"block_{int(block['block_id']):02d}.json"
                    write_json(path, generated)
                    completed[int(block["block_id"])] = generated
                    print(
                        f"valid_block={len(completed)}/{len(blocks)} "
                        f"block={block['block_id']} author={block['target_entity']}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append(
                        (int(block["block_id"]), block["target_entity"], str(exc))
                    )

    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        print(
            "Final JSONL was not written because V3 is incomplete. "
            f"Validated plans/chunks remain in {state_dir}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ordered = [completed[int(block["block_id"])] for block in blocks]
    records = [record for item in ordered for record in item["records"]]
    expected = len(blocks) * int(manifest["block_size"])
    if len(records) != expected:
        raise SystemExit(f"unexpected final record count: {len(records)} != {expected}")
    ciru_data.write_ciru_jsonl(args.output, records)
    write_json(
        profiles_output,
        {
            "design_version": DESIGN_VERSION,
            "split": args.split,
            "seed": args.seed,
            "manifest": str(args.manifest.resolve()),
            "generator_model": args.model,
            "judge_model": args.judge_model,
            "profiles": [
                {
                    key: item[key]
                    for key in (
                        "block_id",
                        "profile_id",
                        "target_entity",
                        "replacement_entity",
                        "author_plan",
                        "placebo_plan",
                    )
                }
                for item in ordered
            ],
        },
    )
    print(
        f"author_blocks={len(ordered)} units={len(records)} cells={4 * len(records)}",
        flush=True,
    )
    print(f"output={args.output}", flush=True)
    print(f"profiles={profiles_output}", flush=True)


if __name__ == "__main__":
    main()
