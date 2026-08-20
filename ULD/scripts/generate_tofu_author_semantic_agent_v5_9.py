#!/usr/bin/env python3
"""Generate TOFU C01 cells with a context-preserving semantic agent loop.

V5.9 is a new ablation.  It preserves V5.8 and every earlier renderer.  For
each immutable C11 row it builds a complete context packet, asks a planner for
a semantic brief, asks a generator for only the C01 question and answer, and
uses an independent critic call as the semantic validation tool.  Python owns
coverage, frozen identifiers, target leakage, checkpoints, and provenance;
semantic relation/object decisions are never made by keyword classifiers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
V58_PATH = SCRIPT_DIR / "generate_tofu_author_semantic_v5_8.py"
DESIGN_VERSION = "tofu-author-semantic-agent-v5.9"
CONTRAST_SCHEMA_VERSION = "semantic-agent-question-answer-v1"
SURFACE_RENDERER = "semantic-agent-complete-c01-question-answer-v5.9"
MAPPING_SCOPE = "row-local-context-packet-agent-with-independent-critic"
RESPONSE_CONTRACT_POLICY = "semantic-agent-critic-v5.9"
AGENT_PROTOCOL_VERSION = "context-plan-generate-critic-repair-v1"
SEMANTIC_BRIEF_SCHEMA_VERSION = "tofu-row-semantic-brief-v1"
GENERATOR_OUTPUT_FIELDS = ("c01_question", "replacement_answer")
CRITIC_FIELDS = (
    "same_target_relation",
    "question_premises_updated",
    "answer_addresses_question",
    "replacement_fact_expressed",
    "source_fact_removed",
    "response_mode_compatible",
    "natural_surface",
)
BRIEF_STRING_FIELDS = (
    "same_relation_definition",
    "answer_object_type",
    "cardinality",
    "response_mode",
)
BRIEF_LIST_FIELDS = (
    "source_specific_premises",
    "replacement_specific_premises",
    "must_preserve",
    "must_change",
    "risk_notes",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v58 = load_module("tofu_author_semantic_agent_v59_v58", V58_PATH)
v57 = v58.v57
v56 = v58.v56
v55 = v58.v55
v53 = v58.v53
v52 = v58.v52
ciru = v58.ciru

BASE_V58_MATERIALIZE_PLAN = v58.materialize_plan
BASE_V58_ASSEMBLE_BLOCK = v58.assemble_block


SEMANTIC_PLANNER_PROMPT = """Act as the semantic planner for one TOFU causal
intervention. Read the immutable C11 question/answer, its frozen source and
replacement core facts, the replacement-author profile, and the complete
20-row replacement ledger.

Describe meaning, not wording. In particular, infer an interrogative from the
whole C11 relation and never classify a quoted or named title from an embedded
wh-word alone.

Do not generate C01 text. Return JSON only with exactly this semantic brief:
{"same_relation_definition":"relation and arguments C01 must preserve",
 "answer_object_type":"semantic object requested by C11",
 "cardinality":"one/list/binary/open or another concise value",
 "response_mode":"positive/negative/unavailable/qualified or concise mode",
 "source_specific_premises":["premise"],
 "replacement_specific_premises":["replacement premise"],
 "must_preserve":["semantic constraint"],
 "must_change":["semantic target"],
 "risk_notes":["likely ambiguity"]}
"""


SEMANTIC_GENERATOR_PROMPT = """Act as the generator in a bounded TOFU semantic
agent. The payload contains the full immutable row context, frozen replacement
ledger, and a planner-produced semantic brief.

Produce one coherent C01 question and answer. They must ask the same relation
as C11, replace every target-specific premise required by the frozen profile,
directly express the frozen replacement fact, and preserve response mode and
cardinality. Interpret titles, awards, places, dates, and names in context;
never infer the answer type from a keyword alone. Do not mention controls,
counterfactuals, generation, or fictionality unless inherited from C11.
When generator_contract.required_question_identity_literal is non-null, copy
that exact literal into c01_question; a pronoun or shortened name is invalid.

When validation_feedback and previous_candidate are supplied, repair only the
reported semantic or structural defect. Return JSON only with exactly:
{"c01_question":"complete question",
 "replacement_answer":"complete answer"}
"""


SEMANTIC_CRITIC_PROMPT = """Act as an independent semantic critic and tool for
one proposed TOFU C01 intervention. Compare the complete immutable C11,
planner brief, frozen source/replacement core facts, replacement-author ledger,
and proposed C01 question/answer.

Judge meaning from the complete context. Do not use keyword-only answer-type
rules: a wh-word inside a quoted or named title is title content, not a
question-type cue. Deterministic code separately owns identifiers,
coverage, leakage, and checkpoints.

Return JSON only with every field below. accepted may be true only when all
seven semantic checks are true. If rejected, give one actionable repair
instruction scoped to this row:
{"accepted":true,
 "same_target_relation":true,
 "question_premises_updated":true,
 "answer_addresses_question":true,
 "replacement_fact_expressed":true,
 "source_fact_removed":true,
 "response_mode_compatible":true,
 "natural_surface":true,
 "reason":"brief evidence grounded in C11, ledger, and C01",
 "repair_instruction":"empty when accepted; otherwise one concrete fix"}
"""


def _normalise_text(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def build_context_packet(
    block: Mapping, source: Mapping, profile: Mapping
) -> Dict:
    """Build the explicit context that makes the API agent reproducible."""
    source_id = source["source_id"]
    entry = profile["fact_ledger_by_source"][source_id]
    source_names_target = (
        v57._normalise(block["target_entity"])
        in v57._normalise(source["question"])
    )
    required_identity = (
        profile["replacement_entity"] if source_names_target else None
    )
    return {
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "causal_estimand": "(C11-C01)-(C10-C00)",
        "cell_policy": {
            "C11": "immutable original TOFU question and answer",
            "C01": "replacement author/fact; same target relation",
            "C10_C00": "frozen domain-compatible placebo cells",
        },
        "immutable_c11": {
            "source_id": source_id,
            "question": source["question"],
            "answer": source["answer"],
            "response_contract": source["contract"],
        },
        "target_author": block["target_entity"],
        "replacement_author": {
            "name": profile["replacement_entity"],
            "pronouns": profile["replacement_pronouns"],
            "profile_summary": profile["profile_summary"],
        },
        "frozen_row_ledger": {
            key: entry[key]
            for key in (
                "source_id",
                "fact_key",
                "target_relation",
                "source_core_fact",
                "replacement_core_fact",
                "replacement_fact",
                "fact_change_required",
                "intervention_policy",
                "contrast_status",
            )
            if key in entry
        },
        "complete_replacement_ledger": [
            {
                key: row[key]
                for key in (
                    "source_id",
                    "fact_key",
                    "target_relation",
                    "replacement_core_fact",
                    "intervention_policy",
                    "contrast_status",
                )
                if key in row
            }
            for row in profile["fact_ledger"]
        ],
        "hard_boundaries": {
            "preserve_source_id": True,
            "keep_c11_byte_exact": True,
            "forbid_target_author_leakage_in_c01": True,
            "one_replacement_identity_per_block": True,
            "required_question_identity_literal": required_identity,
            "semantic_properties_are_critic_owned": True,
        },
    }


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    """Compatibility wrapper for inherited helpers and offline inspection."""
    payload = {"context_packet": build_context_packet(block, source, profile)}
    if feedback:
        payload["validation_feedback"] = feedback
    if previous is not None:
        payload["previous_candidate"] = candidate_for_prompt(previous)
    return payload


def context_digest(packet: Mapping) -> str:
    return v52.digest_json(packet)


def validate_semantic_brief(generated: Mapping) -> Dict:
    if not isinstance(generated, Mapping):
        raise ValueError("semantic brief must be a JSON object")
    result: Dict[str, object] = {}
    for field in BRIEF_STRING_FIELDS:
        value = generated.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"semantic brief {field} must be a non-empty string")
        result[field] = _normalise_text(value)
    for field in BRIEF_LIST_FIELDS:
        value = generated.get(field)
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"semantic brief {field} must be a list of strings")
        result[field] = [_normalise_text(item) for item in value]
    result["schema_version"] = SEMANTIC_BRIEF_SCHEMA_VERSION
    return result


def semantic_context_path(state_dir: Path, block_id: int, source_id: str) -> Path:
    return (
        v52.block_directory(state_dir, block_id)
        / "agent_context"
        / f"{source_id}.json"
    )


def load_or_create_semantic_brief(
    generation_client,
    args,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    state_dir: Path,
) -> tuple[Dict, Dict, int]:
    block_id = int(block["block_id"])
    source_id = source["source_id"]
    packet = build_context_packet(block, source, profile)
    digest = context_digest(packet)
    path = semantic_context_path(state_dir, block_id, source_id)
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (
                cached.get("design_version") == DESIGN_VERSION
                and cached.get("agent_protocol_version") == AGENT_PROTOCOL_VERSION
                and cached.get("profile_digest") == profile["profile_digest"]
                and cached.get("ledger_digest") == profile["ledger_digest"]
                and cached.get("context_digest") == digest
            ):
                brief = validate_semantic_brief(cached["semantic_brief"])
                print(
                    f"reuse_agent_context block={block_id} source={source_id}",
                    flush=True,
                )
                return packet, brief, int(cached.get("planner_attempt", 0))
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    feedback = ""
    previous = None
    last_error = None
    for attempt in range(1, max(int(args.request_retries), 1) + 1):
        print(
            f"start_agent_context block={block_id} source={source_id} "
            f"attempt={attempt}/{max(int(args.request_retries), 1)}",
            flush=True,
        )
        candidate = None
        try:
            payload = {"context_packet": packet}
            if feedback:
                payload["validation_feedback"] = feedback
            if previous is not None:
                payload["previous_candidate"] = previous
            candidate = v52.v2.request_json(
                generation_client,
                v52.request_args(args),
                SEMANTIC_PLANNER_PROMPT,
                payload,
                f"V5.9 block {block_id} semantic context {source_id}",
            )
            brief = validate_semantic_brief(candidate)
            v52.write_json(path, {
                "design_version": DESIGN_VERSION,
                "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
                "profile_digest": profile["profile_digest"],
                "ledger_digest": profile["ledger_digest"],
                "context_digest": digest,
                "planner_attempt": attempt,
                "semantic_brief": brief,
            })
            print(
                f"agent_context_ready block={block_id} source={source_id} "
                f"attempt={attempt}",
                flush=True,
            )
            return packet, brief, attempt
        except Exception as exc:
            last_error = exc
            feedback = str(exc)
            previous = candidate
    raise RuntimeError(
        f"block {block_id} semantic context {source_id} failed: {last_error}"
    ) from last_error


def validate_structural_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
) -> Dict:
    """Enforce structure/leakage only; expose lexical checks as observations."""
    if not isinstance(generated, Mapping):
        raise ValueError("row output must be a JSON object")
    raw_question = generated.get("c01_question")
    raw_answer = generated.get("replacement_answer")
    if not isinstance(raw_question, str) or not raw_question.strip():
        raise ValueError("c01_question must be a non-empty string")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ValueError("replacement_answer must be a non-empty string")

    source_id = source["source_id"]
    entry = profile["fact_ledger_by_source"][source_id]
    question = _normalise_text(raw_question)
    answer = _normalise_text(raw_answer)
    provenance = v58.derive_question_provenance(source["question"], question)
    cell = {"question": question, "answer": answer}

    leak = v55._target_alias_leaks(
        f"{question} {answer}", block["target_entity"]
    )
    if leak:
        raise ValueError(f"C01 still contains target alias {leak!r}")
    if (
        v57._normalise(block["target_entity"])
        in v57._normalise(source["question"])
        and v57._normalise(profile["replacement_entity"])
        not in v57._normalise(question)
    ):
        raise ValueError(
            "C01 question must include exact replacement identity literal: "
            f"{profile['replacement_entity']!r}"
        )
    marker = v55._introduced_control_status_marker(source, cell)
    if marker:
        raise ValueError(f"C01 exposes control status with marker: {marker}")

    contract = v55.semantic_contract_result(
        {"question": source["question"], "answer": answer}, source["contract"]
    )
    replacement_tokens = v53.content_tokens(entry["replacement_fact"])
    candidate_tokens = v53.content_tokens(f"{question} {answer}")
    identity_tokens = v53.content_tokens(profile["replacement_entity"])
    shared = sorted(replacement_tokens & candidate_tokens)
    non_identity_shared = sorted(set(shared) - identity_tokens)
    identity_question = v55.c01_question(block, source, profile)
    identity_answer = v55.replace_identity(
        source["answer"], block["target_entity"], profile["replacement_entity"]
    )
    identity_only = (
        v57._normalise(question) == v57._normalise(identity_question)
        and v57._normalise(answer) == v57._normalise(identity_answer)
    )
    observations = {
        "advisory_contract_errors": list(contract["errors"]),
        "advisory_contract_warnings": list(contract["warnings"]),
        "replacement_content_overlap": shared,
        "replacement_non_identity_overlap": non_identity_shared,
        "identity_only_surface": identity_only,
        "question_changed": bool(provenance["changed"]),
        "answer_changed": (
            v57._normalise(answer) != v57._normalise(source["answer"])
        ),
        "validation_role": "critic_context_only",
    }
    return {
        "source_id": source_id,
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "c01_question": question,
        "replacement_answer": answer,
        "question_anchor_rewrites": v58._compatibility_rewrites(provenance),
        "question_rewrite_rationale": (
            "Question provenance is code-owned; semantic acceptance is owned "
            "by the V5.9 independent critic and final block judge."
        ),
        "cell": cell,
        "fact_change_required": bool(entry["fact_change_required"]),
        "render_mode": "semantic_agent_complete_question_answer_intervention",
        "ledger_digest": profile["ledger_digest"],
        "question_rewrite_provenance": provenance,
        "generator_output_fields": list(GENERATOR_OUTPUT_FIELDS),
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "response_contract_warnings": (
            list(contract["warnings"]) + list(contract["errors"])
        ),
        "response_contract_question_source": "immutable_c11_advisory_only",
        "deterministic_observations": observations,
    }


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
) -> Dict:
    return validate_structural_candidate(block, source, profile, generated)


def candidate_for_prompt(candidate: Mapping | None) -> Dict | None:
    if candidate is None:
        return None
    return {
        field: candidate[field]
        for field in GENERATOR_OUTPUT_FIELDS
        if isinstance(candidate.get(field), str)
    }


def parse_critic_verdict(generated: Mapping) -> Dict:
    if not isinstance(generated, Mapping):
        raise ValueError("semantic critic output must be a JSON object")
    if not isinstance(generated.get("accepted"), bool):
        raise ValueError("semantic critic accepted must be boolean")
    result: Dict[str, object] = {}
    for field in CRITIC_FIELDS:
        value = generated.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"semantic critic {field} must be boolean")
        result[field] = value
    reason = generated.get("reason")
    repair = generated.get("repair_instruction")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("semantic critic reason must be non-empty")
    if not isinstance(repair, str):
        raise ValueError("semantic critic repair_instruction must be a string")
    all_checks = all(result[field] is True for field in CRITIC_FIELDS)
    result.update({
        "model_accepted": generated["accepted"],
        "accepted": bool(generated["accepted"] and all_checks),
        "reason": _normalise_text(reason),
        "repair_instruction": _normalise_text(repair),
        "critic_fields": list(CRITIC_FIELDS),
    })
    if not result["accepted"] and not result["repair_instruction"]:
        raise ValueError(
            "rejected semantic critic verdict needs repair_instruction"
        )
    return result


def critic_feedback(verdict: Mapping) -> str:
    failed = [field for field in CRITIC_FIELDS if verdict.get(field) is not True]
    return (
        "Independent semantic critic rejected this row. Failed checks: "
        + ", ".join(failed or ["accepted"])
        + f". Evidence: {verdict.get('reason', '')}. "
        + f"Required repair: {verdict.get('repair_instruction', '')}"
    )


def request_critic_verdict(
    judge_client,
    args,
    context_packet: Mapping,
    semantic_brief: Mapping,
    validated: Mapping,
    block_id: int,
    source_id: str,
    attempt: int,
) -> Dict:
    generated = v52.v2.request_json(
        judge_client,
        v52.request_args(args, judge=True),
        SEMANTIC_CRITIC_PROMPT,
        {
            "context_packet": context_packet,
            "semantic_brief": semantic_brief,
            "candidate": candidate_for_prompt(validated),
            "deterministic_observations": validated[
                "deterministic_observations"
            ],
            "critic_policy": {
                "independent_from_generator": True,
                "semantic_checks": list(CRITIC_FIELDS),
                "keyword_classification_forbidden": True,
            },
        },
        f"V5.9 block {block_id} semantic critic {source_id} attempt {attempt}",
    )
    return parse_critic_verdict(generated)


def agent_attempt_path(
    state_dir: Path, block_id: int, source_id: str, attempt: int
) -> Path:
    return (
        v52.block_directory(state_dir, block_id)
        / "agent_attempts"
        / f"{source_id}.attempt_{attempt:02d}.json"
    )


def load_cached_agent_row(
    path: Path,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
) -> Dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            cached.get("design_version") != DESIGN_VERSION
            or cached.get("agent_protocol_version") != AGENT_PROTOCOL_VERSION
            or cached.get("profile_digest") != profile["profile_digest"]
            or cached.get("ledger_digest") != profile["ledger_digest"]
        ):
            return None
        validated = validate_structural_candidate(
            block, source, profile, cached["candidate"]
        )
        trace = cached["semantic_agent_trace"]
        if (
            trace.get("agent_protocol_version") != AGENT_PROTOCOL_VERSION
            or trace.get("critic_verdict", {}).get("accepted") is not True
        ):
            return None
        validated["semantic_agent_trace"] = trace
        validated["mapping_attempt"] = int(cached.get("mapping_attempt", 0))
        validated["repair_generation"] = int(
            cached.get("repair_generation", 0)
        )
        return validated
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def write_agent_row_checkpoint(
    path: Path,
    profile: Mapping,
    validated: Mapping,
    mapping_attempt: int,
    repair_generation: int,
) -> None:
    v52.write_json(path, {
        "design_version": DESIGN_VERSION,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "profile_digest": profile["profile_digest"],
        "ledger_digest": profile["ledger_digest"],
        "mapping_attempt": mapping_attempt,
        "repair_generation": repair_generation,
        "candidate": candidate_for_prompt(validated),
        "semantic_agent_trace": validated["semantic_agent_trace"],
    })


def generate_row(
    generation_client,
    critic_client,
    args,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    state_dir: Path,
    feedback: str = "",
    previous: Mapping | None = None,
    force: bool = False,
) -> Dict:
    block_id = int(block["block_id"])
    source_id = source["source_id"]
    path = v52.row_path(state_dir, block_id, source_id)
    if not force:
        cached = load_cached_agent_row(path, block, source, profile)
        if cached is not None:
            print(f"reuse_agent_row block={block_id} source={source_id}", flush=True)
            return cached

    packet, brief, planner_attempt = load_or_create_semantic_brief(
        generation_client, args, block, source, profile, state_dir
    )
    last_error = None
    previous_candidate = candidate_for_prompt(previous)
    for attempt in range(1, args.row_retries + 1):
        print(
            f"start_agent_row block={block_id} source={source_id} "
            f"attempt={attempt}/{args.row_retries}",
            flush=True,
        )
        candidate = None
        validated = None
        critic = None
        try:
            payload = {
                "context_packet": packet,
                "semantic_brief": brief,
                "generator_contract": {
                    "output_fields": list(GENERATOR_OUTPUT_FIELDS),
                    "repair_only_failed_row": True,
                    "required_question_identity_literal": packet[
                        "hard_boundaries"
                    ]["required_question_identity_literal"],
                },
            }
            if feedback:
                payload["validation_feedback"] = feedback
            if previous_candidate is not None:
                payload["previous_candidate"] = previous_candidate
            candidate = v52.v2.request_json(
                generation_client,
                v52.request_args(args),
                SEMANTIC_GENERATOR_PROMPT,
                payload,
                f"V5.9 block {block_id} semantic generator {source_id}",
            )
            validated = validate_structural_candidate(
                block, source, profile, candidate
            )
            critic = request_critic_verdict(
                critic_client,
                args,
                packet,
                brief,
                validated,
                block_id,
                source_id,
                attempt,
            )
            if not critic["accepted"]:
                raise ValueError(critic_feedback(critic))

            trace = {
                "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
                "context_digest": context_digest(packet),
                "planner_attempt": planner_attempt,
                "generator_attempt": attempt,
                "generator_model": args.model,
                "critic_model": args.judge_model,
                "critic_independent_call": True,
                "critic_verdict": critic,
                "deterministic_observations": validated[
                    "deterministic_observations"
                ],
            }
            validated["semantic_agent_trace"] = trace
            validated["mapping_attempt"] = attempt
            validated["repair_generation"] = 1 if force else 0
            write_agent_row_checkpoint(
                path,
                profile,
                validated,
                attempt,
                validated["repair_generation"],
            )
            v52.write_json(
                agent_attempt_path(state_dir, block_id, source_id, attempt),
                {
                    "status": "accepted",
                    "context_digest": trace["context_digest"],
                    "semantic_brief": brief,
                    "candidate": candidate_for_prompt(validated),
                    "critic_verdict": critic,
                },
            )
            print(
                f"agent_row_ready block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries}",
                flush=True,
            )
            return validated
        except Exception as exc:
            last_error = exc
            feedback = str(exc)
            previous_candidate = candidate_for_prompt(
                validated if validated is not None else candidate
            )
            v52.write_json(
                agent_attempt_path(state_dir, block_id, source_id, attempt),
                {
                    "status": "rejected",
                    "error": feedback,
                    "context_digest": context_digest(packet),
                    "semantic_brief": brief,
                    "candidate": previous_candidate,
                    "critic_verdict": critic,
                },
            )
            print(
                f"agent_row_reject block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries} error={exc}",
                flush=True,
            )
    raise RuntimeError(
        f"block {block_id} semantic agent row {source_id} failed after "
        f"{args.row_retries} attempts: {last_error}"
    ) from last_error


def generate_rows(
    generation_client,
    critic_client,
    args,
    block: Mapping,
    profile: Mapping,
    state_dir: Path,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    errors = []
    with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
        futures = {
            executor.submit(
                generate_row,
                generation_client,
                critic_client,
                args,
                block,
                source,
                profile,
                state_dir,
            ): source
            for source in block["sources"]
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                results[source["source_id"]] = future.result()
            except Exception as exc:
                errors.append(str(exc))
    if errors:
        raise RuntimeError(" | ".join(errors))
    return results


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    plan = BASE_V58_MATERIALIZE_PLAN(block, profile, candidates, seed)
    for source_id, row_plan in plan["row_plans"].items():
        row_plan.update({
            "render_mode": "semantic_agent_complete_question_answer_intervention",
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "semantic_agent_trace": candidates[source_id][
                "semantic_agent_trace"
            ],
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        })
    return plan


def assemble_block(
    block: Mapping,
    plan: Mapping,
    verdicts: Mapping[str, Mapping],
    args,
) -> Dict:
    assembled = BASE_V58_ASSEMBLE_BLOCK(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-semantic-agent-v5.9-"
        f"block-{int(block['block_id']):02d}:seed-{args.seed}"
    )
    assembled.update({
        "profile_id": profile_id,
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
    })
    for record in assembled["records"]:
        source_id = record["source_id"]
        trace = plan["row_plans"][source_id]["semantic_agent_trace"]
        record.update({
            "profile_id": profile_id,
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "render_mode": "semantic_agent_complete_question_answer_intervention",
            "semantic_agent_trace": trace,
        })
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "generator_output_fields": list(GENERATOR_OUTPUT_FIELDS),
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
            "independent_row_critic": True,
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{source_id}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


def generate_block(
    generation_client,
    judge_client,
    args,
    block: Mapping,
    protected_authors: Sequence[str],
    state_dir: Path,
) -> Dict:
    block_id = int(block["block_id"])
    print(f"start_block block={block_id} author={block['target_entity']}", flush=True)
    profile = v53.generate_profile(
        generation_client, judge_client, args, block, protected_authors, state_dir
    )
    candidates = generate_rows(
        generation_client, judge_client, args, block, profile, state_dir
    )
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    last_failures: Dict[str, str] = {}
    for judge_round in range(1, args.judge_rounds + 1):
        plan = materialize_plan(block, profile, candidates, args.seed)
        judgement, batches = v58.judge_plan_in_batches(
            judge_client, args, block, plan, judge_round
        )
        verdicts, failures = v58.parse_judgement(judgement, block, plan)
        v52.write_json(
            v52.block_directory(state_dir, block_id)
            / f"judge_round_{judge_round:02d}.json",
            {
                "design_version": DESIGN_VERSION,
                "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                "ledger_digest": profile["ledger_digest"],
                "judge_batch_size": int(args.judge_batch_size),
                "judge_batches": batches,
                "verdicts": verdicts,
                "failures": failures,
            },
        )
        if not failures:
            plan["judge_round"] = judge_round
            result = assemble_block(block, plan, verdicts, args)
            print(
                f"block_ready block={block_id} judge_round={judge_round} "
                f"judge_batches={len(batches)} conflicts=0",
                flush=True,
            )
            return result
        last_failures = failures
        print(
            f"judge_reject block={block_id} round={judge_round}/"
            f"{args.judge_rounds} rows={','.join(sorted(failures))}",
            flush=True,
        )
        if judge_round == args.judge_rounds:
            break
        repaired: Dict[str, Dict] = {}
        with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    generate_row,
                    generation_client,
                    judge_client,
                    args,
                    block,
                    source_by_id[source_id],
                    profile,
                    state_dir,
                    failures[source_id],
                    candidates[source_id],
                    True,
                ): source_id
                for source_id in failures
            }
            for future in as_completed(futures):
                source_id = futures[future]
                repaired[source_id] = future.result()
        candidates.update(repaired)
    raise RuntimeError(
        f"block {block_id} semantic agent judge exhausted "
        f"{args.judge_rounds} rounds: "
        + " | ".join(
            f"{source_id}: {reason}"
            for source_id, reason in sorted(last_failures.items())
        )
    )


def configure_shared_modules() -> None:
    v58.configure_shared_modules()
    for module in (v58, v57, v56, v55, v53, v52):
        module.DESIGN_VERSION = DESIGN_VERSION
        module.CONTRAST_SCHEMA_VERSION = CONTRAST_SCHEMA_VERSION
        module.SURFACE_RENDERER = SURFACE_RENDERER
        module.MAPPING_SCOPE = MAPPING_SCOPE
        module.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v55.ANSWER_PROMPT = SEMANTIC_GENERATOR_PROMPT
    v55.row_payload = row_payload
    v55.validate_row_candidate = validate_row_candidate
    v55.candidate_for_prompt = candidate_for_prompt
    v55.materialize_plan = materialize_plan
    v55.assemble_block = assemble_block
    v55.generate_block = generate_block
    v58.validate_row_candidate = validate_row_candidate
    v58.candidate_for_prompt = candidate_for_prompt
    v58.materialize_plan = materialize_plan
    v58.assemble_block = assemble_block
    v58.generate_block = generate_block


def _profiles_path_from_argv() -> Path | None:
    try:
        output = Path(sys.argv[sys.argv.index("--output") + 1])
    except (ValueError, IndexError):
        return None
    try:
        return Path(sys.argv[sys.argv.index("--profiles-output") + 1])
    except (ValueError, IndexError):
        return Path(str(output) + ".profiles.json")


def main() -> None:
    configure_shared_modules()
    original_configure = v58.configure_shared_modules
    v58.configure_shared_modules = lambda: None
    try:
        v58.main()
    finally:
        v58.configure_shared_modules = original_configure

    profiles_path = _profiles_path_from_argv()
    if profiles_path is not None and profiles_path.is_file():
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        profiles.update({
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
            "critic_fields": list(CRITIC_FIELDS),
        })
        v52.write_json(profiles_path, profiles)


if __name__ == "__main__":
    main()
