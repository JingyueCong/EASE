#!/usr/bin/env python3
"""Selectively compress/repair only C01 using a C11-derived semantic budget.

V5.12 consumes the frozen full-200 V5.11 artifact.  It never edits C11, C10,
C00, the replacement-author profiles, or their ledgers.  For every row it:

1. derives a semantic information budget from immutable C11 without seeing C01;
2. independently audits the inherited C01 against that budget and the frozen
   replacement ledger;
3. copies passing C01 byte-for-byte and regenerates only rejected C01 cells;
4. performs a second, batched independent audit of all accepted pairs.

The frozen row ledger is a support boundary, not a requirement to reproduce
every profile fact.  This prevents a concise C11 from being paired with a C01
that dumps the full replacement-author profile.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
V511_PATH = SCRIPT_DIR / "generate_tofu_author_premisefix_v5_11.py"

BASE_DESIGN = "tofu-author-premisefix-v5.11-full-hybrid"
BASE_ROW_DESIGNS = {
    "tofu-author-pairrepair-v5.10",
    "tofu-author-premisefix-v5.11",
}
DESIGN_VERSION = "tofu-author-pairbudget-v5.12"
PAIR_BUDGET_VERSION = "c11-semantic-information-budget-v3"
BUDGET_AUDIT_VERSION = "independent-c11-budget-audit-v1"
AUDIT_VERSION = "independent-pair-budget-audit-v3"
PAIR_POLICY_VERSION = "replacement-identity-ledger-authority-v3"
MAPPING_SEMANTICS_VERSION = "relation-roles-replacement-values-v1"
SURFACE_RENDERER = "selective-complete-c01-pairbudget-v5.12"
MAPPING_SCOPE = "full200-frozen-profile-selective-c01-only"

BUDGET_FIELDS = (
    "relation_definition",
    "answer_object_type",
    "answerability_family",
    "polarity_family",
    "expected_cardinality",
    "semantic_propositions",
    "information_budget",
    "scope_constraints",
    "nuisance_constraints",
    "replaceable_source_premises",
)
AUDIT_FIELDS = (
    "pair_budget_faithful_to_c11",
    "same_target_relation",
    "question_scope_matched",
    "answer_addresses_question",
    "answerability_matched",
    "polarity_matched",
    "information_budget_matched",
    "replacement_answer_sufficient",
    "replacement_fact_supported",
    "source_fact_removed",
    "no_extraneous_profile_dump",
    "natural_surface",
)
BUDGET_AUDIT_FIELDS = (
    "relation_faithful",
    "argument_slots_faithful",
    "answerability_faithful",
    "polarity_faithful",
    "explicit_cardinality_only",
    "no_c01_or_replacement_assumptions",
    "no_incidental_detail_hard_constraints",
)
ANSWERABILITY = {"available", "unavailable", "qualified"}
POLARITY = {"affirmative", "negative", "qualified", "unavailable"}
INFORMATION_BUDGETS = {"atomic", "brief", "moderate", "detailed"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v511 = load_module("tofu_author_pairbudget_v512_v511", V511_PATH)
ciru = v511.ciru
request_json = v511.v52.v2.request_json


BUDGET_PROMPT = """Derive a semantic information budget from one immutable
TOFU C11 question and answer. You must not invent a replacement answer and you
must not use any C01 text. Describe the relation, evidence status, polarity,
scope, approximate cardinality, and the independent semantic propositions
actually conveyed by C11.

Count propositions semantically, not by commas, conjunctions, punctuation, or
classifier labels. A compact sentence can contain two claims; a list of
closely related examples can still be one answer unit. Preserve named-book,
date, place, field, and other argument slots as scope constraints, while
marking their source-specific values as replaceable premises.

Cardinality means a quantity explicitly requested by the QUESTION (for
example, "name two", one date, or a yes/no decision). Do not infer a hard
cardinality constraint from the incidental number of examples, institutions,
themes, adjectives, clauses, audiences, or supporting details in the C11
answer. For an open "what/which/how" question, normally use "open". Do not
promote optional explanation, rhetorical strength, or supporting examples
into mandatory semantic propositions unless the question explicitly requests
them.

When validation_feedback is supplied, repair the budget itself. Never refer
to C01, a replacement author, a replacement ledger, or facts not present in
immutable C11; those are deliberately unavailable at this stage.

Return JSON only with exactly:
{"relation_definition":"relation and argument roles",
 "answer_object_type":"name/title/date/reason/list/explanation/etc",
 "answerability_family":"available|unavailable|qualified",
 "polarity_family":"affirmative|negative|qualified|unavailable",
 "expected_cardinality":"one/two/list/open/binary or concise description",
 "semantic_propositions":["one independent C11 information unit"],
 "information_budget":"atomic|brief|moderate|detailed",
 "scope_constraints":["argument/scope that C01 must map"],
 "nuisance_constraints":["response properties to match approximately"],
 "replaceable_source_premises":["source-specific values C01 must replace"]}
"""


BUDGET_AUDIT_PROMPT = """Independently audit a semantic pair budget against
the supplied immutable C11 question and answer. This stage knows nothing about
C01 or any replacement author. Reject a budget that refers to C01,
replacement facts, or unsupported assumptions; misstates the relation,
argument slots, answerability, or polarity; treats incidental answer details
as question-mandated propositions; or infers an exact count from examples,
institutions, themes, clauses, audiences, or list items when the C11 QUESTION
does not explicitly request that count.

Return JSON only. accepted may be true only when every check is true:
{"accepted":true,
 "relation_faithful":true,
 "argument_slots_faithful":true,
 "answerability_faithful":true,
 "polarity_faithful":true,
 "explicit_cardinality_only":true,
 "no_c01_or_replacement_assumptions":true,
 "no_incidental_detail_hard_constraints":true,
 "reason":"specific comparison to immutable C11",
 "repair_instruction":"empty when accepted; otherwise repair the budget"}
"""


PAIR_AUDIT_PROMPT = """Act as an independent TOFU causal-pair auditor. Compare
immutable C11 and proposed C01 using the C11-derived pair_budget and frozen
replacement ledger.

The causal intervention changes author identity and target factual content,
while preserving the semantic RELATION and ARGUMENT-ROLE SCHEMA plus an
approximate information budget. It does not preserve source-bound argument
VALUES or the truth value of the source proposition. Surface formatting,
punctuation, an optional leading 'Yes', and heuristic list/date/prose labels
are not semantic failures.

First verify that pair_budget itself faithfully describes immutable C11. A
schema-valid but semantically incorrect budget is a rejection; do not bend
C11 to fit the budget.

Authority and compression rules:
- Preserve relation roles, not literal source values. Names, books, places,
  dates, fields, awards, institutions, professions, genders, genres,
  communities, motivations, descriptors, evidence states, and other values
  listed in replaceable_source_premises are intervention variables. Map them
  to ledger-supported replacement counterparts; never require their C11
  values to remain in C01.
- C11 answerability_family and polarity_family describe the SOURCE answer.
  They are not immutable truth constraints on C01. Evaluate answerability and
  polarity after mapping: C01 question and answer must be mutually coherent
  and faithful to the replacement ledger. A source yes may validly become no,
  available may become unavailable, or qualified may become definitive when
  the frozen replacement facts require it. Preserve the broad response form
  when practical, not an unsupported source truth value.
- pair_budget_faithful_to_c11 judges only whether pair_budget accurately
  summarizes immutable C11. Do not mark it false merely because C01 correctly
  uses different replacement values.
- C11 and pair_budget define the relation, requested argument slots, evidence
  shape, and COARSE information budget. Argument slots are typed roles, not
  commands to retain the source-side values filling those roles.
- frozen_row_ledger and replacement_profile define which replacement facts are
  supported, but they are a SUPPORT CEILING, not a requirement to reproduce
  the complete replacement_fact or profile.
- replacement_support_catalog exposes the same frozen author's other row facts
  only so mapped question premises (such as birthplace, occupation, titles,
  dates, or institutions) can be verified across the profile. It is read-only
  support and never a checklist of content to include in this answer.
- Identity intervention is non-negotiable: C01 must describe the declared
  replacement_entity. The replacement author is not an alias of the C11
  target author. Never recommend restoring the target author or target fact.
- The frozen row ledger is authoritative for replacement-specific content.
  Never demand a fact, institution, title, audience, causal effect, claim
  strength, or other detail that the ledger does not support.
- A concise C01 may select the smallest supported subset that directly answers
  its question and matches C11's semantic budget.
- Information-budget matching is approximate nuisance matching. Do not reject
  merely because C11 and C01 contain different counts of examples,
  institutions, themes, adjectives, clauses, or list items. Fail this check
  only for a clear scale change such as an atomic answer becoming a biography,
  several unrelated claims, or multiple paragraphs.
- A question may map a named C11 book, place, date, award, institution, field,
  or identity descriptor to a coherent replacement-profile counterpart. That
  mapping is required, not a scope failure.
- Reject an answer that dumps peripheral profile facts, is internally
  inconsistent with its mapped question or replacement evidence, drops a
  named argument without a coherent replacement, retains source facts, or
  expands a brief answer into several independent claims or numbered
  paragraphs. Do not reject a ledger-required change in evidence status or
  polarity merely because it differs from C11.
- When the ledger lacks enough information to reproduce a peripheral C11
  detail, a concise truthful answer at the nearest supported granularity is
  preferable to invention. Explain this as a ledger-limited approximation.

Return JSON only. accepted may be true only when every check is true:
{"accepted":true,
 "pair_budget_faithful_to_c11":true,
 "same_target_relation":true,
 "question_scope_matched":true,
 "answer_addresses_question":true,
 "answerability_matched":true,
 "polarity_matched":true,
 "information_budget_matched":true,
 "replacement_answer_sufficient":true,
 "replacement_fact_supported":true,
 "source_fact_removed":true,
 "no_extraneous_profile_dump":true,
 "natural_surface":true,
 "ledger_limited_approximation":false,
 "reason":"specific semantic comparison",
 "repair_instruction":"empty when accepted; otherwise one minimal repair"}
"""


PAIR_GENERATOR_PROMPT = """Repair one TOFU C01 question and answer. Return only
the complete replacement question and answer.

The immutable C11 and pair_budget define the relation, argument slots,
question intent, and coarse maximum useful information. Argument slots are
typed roles whose source-side values must be replaced, not literal values to
copy. Source answerability and polarity describe C11 and may change when the
replacement ledger requires a different truth or evidence status. The
frozen row ledger is authoritative for replacement-specific factual content,
but is a support pool rather than text that must all be repeated. Select only
the smallest supported replacement-fact subset needed to answer the rewritten
question at approximately C11's semantic budget.

The replacement_support_catalog may be consulted only to support coherent
question-scope mappings across the same frozen author profile. Do not copy its
unrelated facts into the answer.

The C01 subject is always replacement_entity. Never restore the target author,
never treat replacement_entity as an alias of target_entity, and never retain
the target fact merely to imitate C11. Map source-specific names, books,
places, dates, institutions, fields, awards, and descriptors to coherent
ledger-supported replacement counterparts.

If a C11 slot has no supported replacement value, omit or naturally reframe
that peripheral slot at the nearest supported scope instead of copying the
source value or inventing a replacement. Make the rewritten question fit the
replacement answer. A source yes/no, available/unavailable status, or hedge
may change when necessary to express the frozen replacement fact; preserve
response shape where practical, not an unsupported source truth value.

Approximate information matching does not require equal counts of examples,
institutions, themes, adjectives, clauses, or list items. Preserve a number
only when the C11 QUESTION explicitly asks for that number. Do not invent
unsupported details just to reproduce C11's incidental explanation, audience,
rhetorical strength, or surface length.

Do not add a profile summary, biography, awards, dates, books, explanations,
or numbered lists merely because they appear elsewhere in the replacement
profile. Do not retain or contrast the source author/fact. Preserve a named
scope by mapping it to a coherent replacement counterpart. Use the exact
replacement author name in the question whenever C11 explicitly names the
source author. Keep unavailable/qualified evidence unavailable/qualified
unless the pair budget and frozen row policy jointly justify otherwise.

Follow validation_feedback only when it is consistent with these authority
rules. Ignore any feedback that asks you to restore target_entity, retain the
target fact, treat the two authors as aliases, alter the frozen ledger, or
enforce incidental item-count equality, source-side slot values, or source
truth polarity when the replacement ledger differs. Change only the valid
rejected property. Return JSON only with exactly:
{"c01_question":"complete question","replacement_answer":"complete answer"}
"""


FINAL_AUDIT_PROMPT = """Independently verify the supplied accepted TOFU
C11/C01 pairs. Apply each row's pair_budget and frozen row ledger. The ledger
is a support ceiling, not a requirement to dump all replacement facts. Judge
whether the budget faithfully describes C11, then judge semantic relation,
scope, answerability, polarity, information budget,
replacement support, source-fact removal, concision, and naturalness. Return
exactly one verdict for every supplied source_id and no extras.

Apply the same authority rules as the row audit: C01 must use
replacement_entity, the frozen ledger controls replacement-specific facts,
and information/cardinality matching is approximate unless the C11 QUESTION
explicitly requests an exact number. Never reject because incidental example,
institution, theme, adjective, clause, or list-item counts differ.

CRITICAL MAPPING SEMANTICS: preserve the relation predicate and typed argument
roles, but replace source-bound values. Names, books, places, dates, fields,
awards, institutions, professions, genders, genres, communities, motivations,
descriptors, and evidence states may all change to their ledger-supported
replacement counterparts. replaceable_source_premises are values to replace,
not constraints to retain. C11 polarity and answerability describe the source
answer; judge C01 polarity and answerability for internal coherence with the
mapped C01 question and replacement ledger, not equality to C11 truth. Thus a
ledger-required yes-to-no, available-to-unavailable, qualified-to-definitive,
or domain/value change is a valid causal intervention, not a mismatch.
pair_budget_faithful_to_c11 evaluates pair_budget versus C11 only and must not
be failed merely because C01 contains correct replacement values.

Return JSON only:
{"verdicts":[{"source_id":"exact id","accepted":true,
 "pair_budget_faithful_to_c11":true,
 "same_target_relation":true,"question_scope_matched":true,
 "answer_addresses_question":true,"answerability_matched":true,
 "polarity_matched":true,"information_budget_matched":true,
 "replacement_answer_sufficient":true,"replacement_fact_supported":true,
 "source_fact_removed":true,"no_extraneous_profile_dump":true,
 "natural_surface":true,"ledger_limited_approximation":false,
 "reason":"specific evidence","repair_instruction":""}]}
"""


def normalise(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalise(value).casefold()).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing base data: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 200 or len({row.get("source_id") for row in rows}) != 200:
        raise ValueError("base data must contain 200 unique rows")
    errors = []
    for row in rows:
        source_id = row.get("source_id", "<missing-source-id>")
        row_design = row.get("design_version")
        assembly = row.get("generation", {}).get("full200_assembly", {})
        if row_design not in BASE_ROW_DESIGNS:
            errors.append(f"{source_id}: unsupported source design {row_design!r}")
        if assembly.get("full_design") != BASE_DESIGN:
            errors.append(f"{source_id}: missing {BASE_DESIGN} assembly provenance")
        if assembly.get("source_design") != row_design:
            errors.append(f"{source_id}: assembly/source design mismatch")
        if assembly.get("causal_cells_edited_by_merge") is not False:
            errors.append(f"{source_id}: merge did not preserve causal cells")
    if errors:
        raise ValueError("invalid frozen hybrid base: " + "; ".join(errors[:20]))
    return rows


def load_profiles(path: Path, data_digest: str) -> tuple[dict, dict[int, dict]]:
    if not path.is_file():
        raise ValueError(f"missing base profiles: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("design_version") != BASE_DESIGN:
        raise ValueError("base profiles have wrong design")
    if payload.get("assembly", {}).get("full_data_sha256") != data_digest:
        raise ValueError("base profile/data digest mismatch")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 10:
        raise ValueError("base profiles must cover 10 blocks")
    by_block = {}
    for raw in profiles:
        profile = copy.deepcopy(raw)
        block_id = int(profile.get("block_id", -1))
        ledger = profile.get("fact_ledger")
        if block_id in by_block or not isinstance(ledger, list) or len(ledger) != 20:
            raise ValueError(f"invalid profile block {block_id}")
        profile["fact_ledger_by_source"] = {
            row["source_id"]: row for row in ledger
        }
        by_block[block_id] = profile
    if set(by_block) != set(range(10)):
        raise ValueError("profile block coverage mismatch")
    return payload, by_block


def validate_budget(generated: Mapping) -> dict:
    if not isinstance(generated, Mapping):
        raise ValueError("pair budget must be an object")
    result = {}
    string_fields = (
        "relation_definition",
        "answer_object_type",
        "expected_cardinality",
    )
    for field in string_fields:
        value = generated.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pair budget {field} must be non-empty")
        result[field] = normalise(value)
    answerability = normalise_key(generated.get("answerability_family"))
    polarity = normalise_key(generated.get("polarity_family"))
    information = normalise_key(generated.get("information_budget"))
    if answerability not in ANSWERABILITY:
        raise ValueError("invalid answerability_family")
    if polarity not in POLARITY:
        raise ValueError("invalid polarity_family")
    if information not in INFORMATION_BUDGETS:
        raise ValueError("invalid information_budget")
    result.update({
        "answerability_family": answerability,
        "polarity_family": polarity,
        "information_budget": information,
    })
    for field in (
        "semantic_propositions",
        "scope_constraints",
        "nuisance_constraints",
        "replaceable_source_premises",
    ):
        value = generated.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"pair budget {field} must be a list of strings")
        if field == "semantic_propositions" and not value:
            raise ValueError("semantic_propositions must not be empty")
        result[field] = [normalise(item) for item in value]
    result["version"] = PAIR_BUDGET_VERSION
    return result


def parse_budget_audit(generated: Mapping) -> dict:
    if not isinstance(generated, Mapping):
        raise ValueError("budget audit verdict must be an object")
    result = {
        field: generated.get(field) is True
        for field in BUDGET_AUDIT_FIELDS
    }
    result["accepted"] = all(result.values())
    for field in ("reason", "repair_instruction"):
        value = generated.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"budget audit {field} must be a string")
        result[field] = normalise(value)
    if not result["accepted"] and not result["repair_instruction"]:
        failed = [
            field for field in BUDGET_AUDIT_FIELDS
            if result.get(field) is not True
        ]
        result["repair_instruction"] = (
            "Revise the C11 budget so these failed checks pass: "
            + ", ".join(failed)
            + ". Use only immutable C11 evidence; do not add C01 or replacement "
            "profile assumptions."
        )
        result["repair_instruction_synthesized"] = True
    return result


def budget_audit_feedback(verdict: Mapping) -> str:
    failed = [
        field for field in BUDGET_AUDIT_FIELDS
        if verdict.get(field) is not True
    ]
    return (
        "Failed immutable-C11 budget checks: " + ", ".join(failed)
        + f". Evidence: {verdict.get('reason', '')}. "
        + f"Required budget repair: {verdict.get('repair_instruction', '')}"
    )


def parse_audit_verdict(generated: Mapping) -> dict:
    if not isinstance(generated, Mapping):
        raise ValueError("pair audit verdict must be an object")
    result = {field: generated.get(field) is True for field in AUDIT_FIELDS}
    result["accepted"] = all(result.values())
    result["ledger_limited_approximation"] = (
        generated.get("ledger_limited_approximation") is True
    )
    for field in ("reason", "repair_instruction"):
        value = generated.get(field, "")
        if not isinstance(value, str):
            raise ValueError(f"pair audit {field} must be a string")
        result[field] = normalise(value)
    if not result["accepted"] and not result["repair_instruction"]:
        failed = [
            field for field in AUDIT_FIELDS
            if result.get(field) is not True
        ]
        result["repair_instruction"] = (
            "Revise only the C01 question and answer so these failed checks "
            "pass: " + ", ".join(failed) + ". Preserve frozen C11/C10/C00, "
            "the replacement identity, profile, ledger, and C11 information "
            "budget."
        )
        result["repair_instruction_synthesized"] = True
    return result


def source_id_index(source_id: str) -> int:
    match = re.search(r"-(\d+)$", source_id)
    if match is None:
        raise ValueError(f"invalid source_id {source_id!r}")
    return int(match.group(1))


def context_packet(row: Mapping, profile: Mapping, budget: Mapping) -> dict:
    source_id = row["source_id"]
    entry = profile["fact_ledger_by_source"].get(source_id)
    if entry is None:
        raise ValueError(f"profile ledger missing {source_id}")
    target = row.get("target_entity") or profile.get("target_entity")
    replacement = row.get("replacement_entity") or profile.get(
        "replacement_entity"
    )
    return {
        "source_id": source_id,
        "block_id": int(row["block_id"]),
        "immutable_c11": copy.deepcopy(row["cells"]["C11"]),
        "proposed_c01": copy.deepcopy(row["cells"]["C01"]),
        "target_entity": target,
        "replacement_entity": replacement,
        "replacement_profile_summary": profile.get("profile_summary", ""),
        "replacement_support_catalog": [
            {
                key: ledger_row.get(key)
                for key in (
                    "source_id", "fact_key", "target_relation",
                    "replacement_core_fact", "replacement_fact",
                    "intervention_policy", "contrast_status",
                )
                if key in ledger_row
            }
            for ledger_row in profile.get("fact_ledger", [])
        ],
        "frozen_row_ledger": {
            key: entry.get(key)
            for key in (
                "fact_key", "target_relation", "source_core_fact",
                "replacement_core_fact", "replacement_fact",
                "fact_change_required", "intervention_policy",
                "contrast_status", "abstain_reason",
            )
            if key in entry
        },
        "pair_budget": copy.deepcopy(budget),
        "authority_policy": {
            "version": PAIR_POLICY_VERSION,
            "mapping_semantics_version": MAPPING_SEMANTICS_VERSION,
            "relation_and_coarse_information_budget": (
                "immutable C11 pair_budget"
            ),
            "replacement_fact_support": "frozen row ledger and profile",
            "cross_row_scope_support": (
                "replacement_support_catalog is read-only support, not an "
                "answer-content checklist"
            ),
            "ledger_is_support_ceiling_not_required_transcript": True,
            "replacement_identity_is_mandatory": replacement,
            "target_identity_is_forbidden_in_c01": target,
            "replacement_is_not_target_alias": True,
            "incidental_item_count_is_not_a_hard_constraint": True,
            "exact_cardinality_only_when_question_explicitly_requests_it": True,
            "preserve_relation_roles_not_source_values": True,
            "replacement_ledger_controls_truth_and_evidence_status": True,
            "source_polarity_and_answerability_are_not_c01_truth_constraints": True,
        },
    }


def validate_candidate(row: Mapping, profile: Mapping, candidate: Mapping) -> dict:
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate must be an object")
    question = candidate.get("c01_question")
    answer = candidate.get("replacement_answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("c01_question must be non-empty")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("replacement_answer must be non-empty")
    question, answer = normalise(question), normalise(answer)
    target = normalise_key(
        row.get("target_entity") or profile.get("target_entity")
    )
    replacement_literal = normalise(
        row.get("replacement_entity") or profile.get("replacement_entity")
    )
    combined = normalise_key(f"{question} {answer}")
    if target and target in combined:
        raise ValueError("C01 leaks target identity")
    c11_question = normalise_key(row["cells"]["C11"]["question"])
    if target and target in c11_question:
        if normalise_key(replacement_literal) not in normalise_key(question):
            raise ValueError("C01 question loses explicit replacement identity")
    return {"c01_question": question, "replacement_answer": answer}


def request_args(args, *, judge: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.judge_model if judge else args.model,
        temperature=args.judge_temperature if judge else args.temperature,
        retries=args.request_retries,
        json_mode=args.json_mode,
        max_completion_tokens=args.max_completion_tokens,
    )


def budget_path(state_dir: Path, source_id: str) -> Path:
    return state_dir / "budgets" / f"{source_id}.json"


def row_path(state_dir: Path, source_id: str) -> Path:
    return state_dir / "rows" / f"{source_id}.json"


def attempt_path(state_dir: Path, source_id: str, attempt: int) -> Path:
    return state_dir / "row_attempts" / f"{source_id}.attempt_{attempt:02d}.json"


def load_or_create_budget(generation_client, judge_client, args, row: Mapping,
                          state_dir: Path, base_digest: str) -> dict:
    source_id = row["source_id"]
    path = budget_path(state_dir, source_id)
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("design_version") == DESIGN_VERSION
            and cached.get("pair_budget_version") == PAIR_BUDGET_VERSION
            and cached.get("budget_audit_version") == BUDGET_AUDIT_VERSION
            and cached.get("pair_policy_version") == PAIR_POLICY_VERSION
            and cached.get("base_data_digest") == base_digest
        ):
            budget = validate_budget(cached["pair_budget"])
            budget_audit = parse_budget_audit(cached["budget_audit"])
            if budget_audit["accepted"]:
                return {"pair_budget": budget, "budget_audit": budget_audit}
    last_error = None
    validation_feedback = ""
    for attempt in range(1, args.budget_retries + 1):
        try:
            generated = request_json(
                generation_client, request_args(args, judge=False),
                BUDGET_PROMPT,
                {
                    "source_id": source_id,
                    "immutable_c11": row["cells"]["C11"],
                    "declared_target_relation": row.get("target_relation", ""),
                    "important_boundary": "No C01 or replacement profile is supplied.",
                    "validation_feedback": validation_feedback,
                },
                f"V5.12 pair budget {source_id} attempt {attempt}",
            )
            budget = validate_budget(generated)
            audited = request_json(
                judge_client, request_args(args, judge=True),
                BUDGET_AUDIT_PROMPT,
                {
                    "source_id": source_id,
                    "immutable_c11": row["cells"]["C11"],
                    "pair_budget": budget,
                    "important_boundary": (
                        "No C01 or replacement profile is available."
                    ),
                },
                f"V5.12 pair budget audit {source_id} attempt {attempt}",
            )
            budget_audit = parse_budget_audit(audited)
            if not budget_audit["accepted"]:
                validation_feedback = budget_audit_feedback(budget_audit)
                raise ValueError(validation_feedback)
            write_json(path, {
                "design_version": DESIGN_VERSION,
                "pair_budget_version": PAIR_BUDGET_VERSION,
                "budget_audit_version": BUDGET_AUDIT_VERSION,
                "pair_policy_version": PAIR_POLICY_VERSION,
                "base_data_digest": base_digest,
                "budget_attempt": attempt,
                "pair_budget": budget,
                "budget_audit": budget_audit,
            })
            print(f"pair_budget_ready source={source_id} attempt={attempt}", flush=True)
            return {"pair_budget": budget, "budget_audit": budget_audit}
        except Exception as exc:
            last_error = exc
            print(
                f"pair_budget_reject source={source_id} "
                f"attempt={attempt}/{args.budget_retries} error={exc}",
                flush=True,
            )
    raise RuntimeError(f"pair budget failed for {source_id}: {last_error}")


def audit_candidate(client, args, packet: Mapping, candidate: Mapping,
                    label: str) -> dict:
    payload = copy.deepcopy(packet)
    payload["proposed_c01"] = {
        "question": candidate["c01_question"],
        "answer": candidate["replacement_answer"],
    }
    generated = request_json(
        client, request_args(args, judge=True), PAIR_AUDIT_PROMPT,
        payload, label,
    )
    return parse_audit_verdict(generated)


def audit_feedback(verdict: Mapping) -> str:
    failed = [field for field in AUDIT_FIELDS if verdict.get(field) is not True]
    return (
        "Non-overridable repair policy: keep replacement_entity, never restore "
        "target_entity or the target fact, keep the frozen ledger unchanged, "
        "and do not enforce incidental item-count equality, source-side slot "
        "values, or source truth polarity when the replacement ledger differs. "
        "Failed semantic pair-budget checks: " + ", ".join(failed)
        + f". Evidence: {verdict.get('reason', '')}. "
        + f"Required repair: {verdict.get('repair_instruction', '')}"
    )


def load_cached_row(path: Path, row: Mapping, profile: Mapping,
                    base_digest: str, profiles_digest: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("design_version") != DESIGN_VERSION
            or cached.get("pair_budget_version") != PAIR_BUDGET_VERSION
            or cached.get("budget_audit_version") != BUDGET_AUDIT_VERSION
            or cached.get("audit_version") != AUDIT_VERSION
            or cached.get("pair_policy_version") != PAIR_POLICY_VERSION
            or cached.get("base_data_digest") != base_digest
            or cached.get("base_profiles_digest") != profiles_digest
        ):
            return None
        candidate = validate_candidate(row, profile, cached["candidate"])
        verdict = parse_audit_verdict(cached["pair_audit"])
        budget_audit = parse_budget_audit(cached["budget_audit"])
        if not verdict["accepted"] or not budget_audit["accepted"]:
            return None
        return {
            "candidate": candidate,
            "pair_audit": verdict,
            "pair_budget": validate_budget(cached["pair_budget"]),
            "budget_audit": budget_audit,
            "repair_generation": int(cached.get("repair_generation", 0)),
            "generator_attempt": int(cached.get("generator_attempt", 0)),
        }
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None


def write_row_checkpoint(path: Path, *, source_id: str, candidate: Mapping,
                         budget: Mapping, budget_audit: Mapping,
                         verdict: Mapping,
                         base_digest: str, profiles_digest: str,
                         generator_attempt: int,
                         repair_generation: int) -> None:
    write_json(path, {
        "design_version": DESIGN_VERSION,
        "pair_budget_version": PAIR_BUDGET_VERSION,
        "budget_audit_version": BUDGET_AUDIT_VERSION,
        "audit_version": AUDIT_VERSION,
        "pair_policy_version": PAIR_POLICY_VERSION,
        "source_id": source_id,
        "base_data_digest": base_digest,
        "base_profiles_digest": profiles_digest,
        "generator_attempt": generator_attempt,
        "repair_generation": repair_generation,
        "candidate": dict(candidate),
        "pair_budget": dict(budget),
        "budget_audit": dict(budget_audit),
        "pair_audit": dict(verdict),
    })


def process_row(generation_client, judge_client, args, row: Mapping,
                profile: Mapping, state_dir: Path, base_digest: str,
                profiles_digest: str, *, force: bool = False,
                feedback: str = "", previous: Mapping | None = None) -> dict:
    source_id = row["source_id"]
    path = row_path(state_dir, source_id)
    budget_bundle = load_or_create_budget(
        generation_client, judge_client, args, row, state_dir, base_digest
    )
    budget = budget_bundle["pair_budget"]
    budget_audit = budget_bundle["budget_audit"]
    if not force:
        cached = load_cached_row(
            path, row, profile, base_digest, profiles_digest
        )
        if cached is not None:
            print(f"reuse_pairbudget_row source={source_id}", flush=True)
            return cached
        inherited = validate_candidate(row, profile, {
            "c01_question": row["cells"]["C01"]["question"],
            "replacement_answer": row["cells"]["C01"]["answer"],
        })
        packet = context_packet(row, profile, budget)
        verdict = audit_candidate(
            judge_client, args, packet, inherited,
            f"V5.12 inherited pair audit {source_id}",
        )
        if verdict["accepted"]:
            result = {
                "candidate": inherited,
                "pair_audit": verdict,
                "pair_budget": budget,
                "budget_audit": budget_audit,
                "repair_generation": 0,
                "generator_attempt": 0,
            }
            write_row_checkpoint(
                path, source_id=source_id, candidate=inherited,
                budget=budget, budget_audit=budget_audit, verdict=verdict,
                base_digest=base_digest, profiles_digest=profiles_digest,
                generator_attempt=0, repair_generation=0,
            )
            print(f"pairbudget_row_reused source={source_id}", flush=True)
            return result
        feedback = audit_feedback(verdict)
        previous = inherited
        print(
            f"pairbudget_row_repair source={source_id} reason={feedback}",
            flush=True,
        )

    packet = context_packet(row, profile, budget)
    last_error = None
    for attempt in range(1, args.row_retries + 1):
        try:
            payload = copy.deepcopy(packet)
            payload["previous_candidate"] = dict(previous or {})
            payload["validation_feedback"] = feedback
            generated = request_json(
                generation_client, request_args(args, judge=False),
                PAIR_GENERATOR_PROMPT, payload,
                f"V5.12 pair repair {source_id} attempt {attempt}",
            )
            candidate = validate_candidate(row, profile, generated)
            verdict = audit_candidate(
                judge_client, args, packet, candidate,
                f"V5.12 repaired pair audit {source_id} attempt {attempt}",
            )
            write_json(attempt_path(state_dir, source_id, attempt), {
                "design_version": DESIGN_VERSION,
                "source_id": source_id,
                "attempt": attempt,
                "candidate": candidate,
                "pair_audit": verdict,
            })
            if not verdict["accepted"]:
                raise ValueError(audit_feedback(verdict))
            result = {
                "candidate": candidate,
                "pair_audit": verdict,
                "pair_budget": budget,
                "budget_audit": budget_audit,
                "repair_generation": 1,
                "generator_attempt": attempt,
            }
            write_row_checkpoint(
                path, source_id=source_id, candidate=candidate,
                budget=budget, budget_audit=budget_audit, verdict=verdict,
                base_digest=base_digest, profiles_digest=profiles_digest,
                generator_attempt=attempt, repair_generation=1,
            )
            print(
                f"pairbudget_row_ready source={source_id} "
                f"attempt={attempt}/{args.row_retries}", flush=True,
            )
            return result
        except Exception as exc:
            last_error = exc
            feedback = str(exc)
            print(
                f"pairbudget_row_reject source={source_id} "
                f"attempt={attempt}/{args.row_retries} error={exc}",
                flush=True,
            )
    raise RuntimeError(f"pair-budget repair failed for {source_id}: {last_error}")


def process_rows(generation_client, judge_client, args, rows: Sequence[Mapping],
                 profiles: Mapping[int, Mapping], state_dir: Path,
                 base_digest: str, profiles_digest: str) -> dict[str, dict]:
    results = {}
    errors = []
    with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
        futures = {
            executor.submit(
                process_row, generation_client, judge_client, args, row,
                profiles[int(row["block_id"])], state_dir, base_digest,
                profiles_digest,
            ): row["source_id"]
            for row in rows
        }
        for future in as_completed(futures):
            source_id = futures[future]
            try:
                results[source_id] = future.result()
            except Exception as exc:
                errors.append(f"{source_id}: {exc}")
    if errors:
        raise RuntimeError(" | ".join(errors))
    return results


def parse_final_verdicts(generated: Mapping,
                         expected_ids: Sequence[str]) -> dict[str, dict]:
    raw = generated.get("verdicts") if isinstance(generated, Mapping) else None
    if not isinstance(raw, list):
        raise ValueError("final audit verdicts must be a list")
    indexed = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("final audit verdict must be an object")
        source_id = item.get("source_id")
        if source_id in indexed:
            raise ValueError(f"duplicate final verdict {source_id}")
        indexed[source_id] = parse_audit_verdict(item)
    if set(indexed) != set(expected_ids):
        raise ValueError(
            "final audit coverage mismatch: "
            f"missing={sorted(set(expected_ids)-set(indexed))} "
            f"extra={sorted(set(indexed)-set(expected_ids))}"
        )
    return indexed


def final_audit(judge_client, args, rows: Sequence[Mapping],
                profiles: Mapping[int, Mapping], results: Mapping[str, Mapping],
                state_dir: Path, round_number: int) -> dict[str, dict]:
    packets = []
    for row in rows:
        result = results[row["source_id"]]
        packet = context_packet(
            row, profiles[int(row["block_id"])], result["pair_budget"]
        )
        packet["proposed_c01"] = {
            "question": result["candidate"]["c01_question"],
            "answer": result["candidate"]["replacement_answer"],
        }
        packets.append(packet)
    batches = [
        packets[offset:offset + args.judge_batch_size]
        for offset in range(0, len(packets), args.judge_batch_size)
    ]
    verdicts = {}
    errors = []

    def run_batch(number: int, batch: Sequence[Mapping]):
        ids = [item["source_id"] for item in batch]
        try:
            generated = request_json(
                judge_client, request_args(args, judge=True),
                FINAL_AUDIT_PROMPT,
                {
                    "rows": list(batch),
                    "coverage_rule": "one exact verdict per supplied source_id",
                    "round": round_number,
                },
                f"V5.12 final audit round {round_number} batch {number}",
            )
            return number, parse_final_verdicts(generated, ids)
        except ValueError as batch_error:
            print(
                f"pairbudget_final_coverage_fallback round={round_number} "
                f"batch={number} rows={','.join(ids)} error={batch_error}",
                flush=True,
            )
            singleton_verdicts = {}
            for packet in batch:
                source_id = packet["source_id"]
                generated = request_json(
                    judge_client, request_args(args, judge=True),
                    FINAL_AUDIT_PROMPT,
                    {
                        "rows": [packet],
                        "coverage_rule": (
                            "return exactly this one supplied source_id"
                        ),
                        "round": round_number,
                    },
                    f"V5.12 final audit fallback {source_id}",
                )
                singleton_verdicts.update(
                    parse_final_verdicts(generated, [source_id])
                )
            return number, singleton_verdicts

    with ThreadPoolExecutor(max_workers=max(args.audit_concurrency, 1)) as executor:
        futures = {
            executor.submit(run_batch, number, batch): number
            for number, batch in enumerate(batches, 1)
        }
        for future in as_completed(futures):
            try:
                number, batch_verdicts = future.result()
                verdicts.update(batch_verdicts)
                print(
                    f"pairbudget_final_batch round={round_number} "
                    f"batch={number}/{len(batches)} rows={len(batch_verdicts)}",
                    flush=True,
                )
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError("final audit failed: " + " | ".join(errors))
    write_json(state_dir / "final_audit" / f"round_{round_number:02d}.json", {
        "design_version": DESIGN_VERSION,
        "round": round_number,
        "verdicts": verdicts,
    })
    return verdicts


def run_final_repair_loop(generation_client, judge_client, args,
                          rows: Sequence[Mapping], profiles: Mapping[int, Mapping],
                          results: dict[str, dict], state_dir: Path,
                          base_digest: str, profiles_digest: str) -> dict[str, dict]:
    by_id = {row["source_id"]: row for row in rows}
    for round_number in range(1, args.judge_rounds + 1):
        verdicts = final_audit(
            judge_client, args, rows, profiles, results, state_dir, round_number
        )
        failed = {
            source_id: verdict
            for source_id, verdict in verdicts.items()
            if not verdict["accepted"]
        }
        if not failed:
            for source_id, verdict in verdicts.items():
                results[source_id]["final_pair_audit"] = verdict
            print(
                f"pairbudget_final_audit_ok rows={len(rows)} "
                f"round={round_number}", flush=True,
            )
            return results
        print(
            f"pairbudget_final_reject round={round_number}/"
            f"{args.judge_rounds} rows={','.join(sorted(failed))}",
            flush=True,
        )
        if round_number == args.judge_rounds:
            reasons = " | ".join(
                f"{source_id}: {audit_feedback(verdict)}"
                for source_id, verdict in sorted(failed.items())
            )
            raise RuntimeError("final pair-budget audit exhausted: " + reasons)
        repaired = {}
        with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    process_row, generation_client, judge_client, args,
                    by_id[source_id], profiles[int(by_id[source_id]["block_id"])],
                    state_dir, base_digest, profiles_digest, force=True,
                    feedback=audit_feedback(verdict),
                    previous=results[source_id]["candidate"],
                ): source_id
                for source_id, verdict in failed.items()
            }
            for future in as_completed(futures):
                source_id = futures[future]
                repaired[source_id] = future.result()
        results.update(repaired)
    raise AssertionError("unreachable final repair loop")


def assemble_records(rows: Sequence[Mapping], results: Mapping[str, Mapping],
                     base_digest: str, profiles_digest: str) -> list[dict]:
    ciru.AUTHOR_LEVEL_DESIGNS.add(DESIGN_VERSION)
    assembled = []
    for row in rows:
        source_id = row["source_id"]
        result = results[source_id]
        record = copy.deepcopy(row)
        frozen = {
            cell: copy.deepcopy(row["cells"][cell])
            for cell in ("C11", "C10", "C00")
        }
        record["cells"]["C01"] = {
            "question": result["candidate"]["c01_question"],
            "answer": result["candidate"]["replacement_answer"],
        }
        if any(record["cells"][cell] != value for cell, value in frozen.items()):
            raise ValueError(f"{source_id}: frozen cell changed")
        record["design_version"] = DESIGN_VERSION
        record["render_mode"] = SURFACE_RENDERER
        record["pair_budget"] = result["pair_budget"]
        record["pair_budget_trace"] = {
            "pair_budget_version": PAIR_BUDGET_VERSION,
            "budget_audit_version": BUDGET_AUDIT_VERSION,
            "audit_version": AUDIT_VERSION,
            "pair_policy_version": PAIR_POLICY_VERSION,
            "mapping_semantics_version": MAPPING_SEMANTICS_VERSION,
            "base_data_digest": base_digest,
            "base_profiles_digest": profiles_digest,
            "c01_inherited_byte_exact": result["repair_generation"] == 0,
            "repair_generation": result["repair_generation"],
            "generator_attempt": result["generator_attempt"],
            "budget_audit": result["budget_audit"],
            "row_pair_audit": result["pair_audit"],
            "final_pair_audit": result["final_pair_audit"],
            "frozen_cells": ["C11", "C10", "C00"],
            "profile_and_ledger_frozen": True,
        }
        record.setdefault("generation", {}).update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
            "pair_budget_version": PAIR_BUDGET_VERSION,
            "budget_audit_version": BUDGET_AUDIT_VERSION,
            "audit_version": AUDIT_VERSION,
            "pair_policy_version": PAIR_POLICY_VERSION,
            "mapping_semantics_version": MAPPING_SEMANTICS_VERSION,
            "base_data_digest": base_digest,
            "base_profiles_digest": profiles_digest,
            "c01_only_revision": True,
            "frozen_cells": ["C11", "C10", "C00"],
            "profile_and_ledger_frozen": True,
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{source_id}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
        assembled.append(record)
    return sorted(assembled, key=lambda row: source_id_index(row["source_id"]))


def output_profiles(base_profiles: Mapping, *, base_data: Path,
                    base_profiles_path: Path, base_digest: str,
                    profiles_digest: str, output_digest: str,
                    results: Mapping[str, Mapping]) -> dict:
    payload = copy.deepcopy(base_profiles)
    payload.update({
        "design_version": DESIGN_VERSION,
        "pair_budget_version": PAIR_BUDGET_VERSION,
        "budget_audit_version": BUDGET_AUDIT_VERSION,
        "audit_version": AUDIT_VERSION,
        "pair_policy_version": PAIR_POLICY_VERSION,
        "mapping_semantics_version": MAPPING_SEMANTICS_VERSION,
        "pairbudget_revision": {
            "base_data": str(base_data.resolve()),
            "base_data_sha256": base_digest,
            "base_profiles": str(base_profiles_path.resolve()),
            "base_profiles_sha256": profiles_digest,
            "output_data_sha256": output_digest,
            "c01_only_revision": True,
            "frozen_cells": ["C11", "C10", "C00"],
            "profile_and_ledger_frozen": True,
            "inherited_c01": sum(
                result["repair_generation"] == 0 for result in results.values()
            ),
            "repaired_c01": sum(
                result["repair_generation"] > 0 for result in results.values()
            ),
            "human_review_status": "pending",
        },
    })
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--base-profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-api-key-env")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=12000)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--budget-retries", type=int, default=3)
    parser.add_argument("--row-retries", type=int, default=4)
    parser.add_argument("--judge-rounds", type=int, default=3)
    parser.add_argument("--row-concurrency", type=int, default=5)
    parser.add_argument("--audit-concurrency", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=5)
    parser.add_argument(
        "--json-mode", choices=("auto", "required", "prompt"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.judge_model = args.judge_model or args.model
    args.judge_base_url = args.judge_base_url or args.base_url
    args.judge_api_key_env = args.judge_api_key_env or args.api_key_env
    generation_key = os.environ.get(args.api_key_env)
    judge_key = os.environ.get(args.judge_api_key_env)
    if not generation_key or not judge_key:
        raise SystemExit("missing generation or judge API key")
    from openai import OpenAI

    generation_client = OpenAI(api_key=generation_key, base_url=args.base_url)
    judge_client = OpenAI(api_key=judge_key, base_url=args.judge_base_url)
    base_digest = sha256(args.base_data)
    profiles_digest = sha256(args.base_profiles)
    rows = load_jsonl(args.base_data)
    base_profiles, profiles = load_profiles(args.base_profiles, base_digest)
    results = process_rows(
        generation_client, judge_client, args, rows, profiles, args.state_dir,
        base_digest, profiles_digest,
    )
    results = run_final_repair_loop(
        generation_client, judge_client, args, rows, profiles, results,
        args.state_dir, base_digest, profiles_digest,
    )
    output_rows = assemble_records(
        rows, results, base_digest, profiles_digest
    )
    write_jsonl(args.output, output_rows)
    output_digest = sha256(args.output)
    write_json(
        args.profiles_output,
        output_profiles(
            base_profiles, base_data=args.base_data,
            base_profiles_path=args.base_profiles, base_digest=base_digest,
            profiles_digest=profiles_digest, output_digest=output_digest,
            results=results,
        ),
    )
    inherited = sum(result["repair_generation"] == 0 for result in results.values())
    print(
        f"Pair-budget V5.12 complete: rows=200 inherited={inherited} "
        f"repaired={200-inherited} output={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
