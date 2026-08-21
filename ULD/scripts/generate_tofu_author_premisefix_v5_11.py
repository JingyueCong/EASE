#!/usr/bin/env python3
"""Repair V5.9 rows with an explicit source-to-replacement premise policy.

V5.11 is a new ablation.  It never overwrites V5.9/V5.10 artifacts.  It
reuses the frozen V5.9 author profiles and row candidates, re-audits every
candidate under one precedence rule, and regenerates only missing or rejected
rows:

* preserve the semantic relation, argument roles, cardinality, evidence
  status, and approximate response form;
* replace author-specific factual premises (author, title, award domain,
  place, date, identity descriptor, and similar qualifiers) with mutually
  coherent premises from the replacement profile;
* require the rewritten question to be true of the replacement profile and
  the answer to satisfy that rewritten question.

This resolves the V5.9 conflict where a row critic could require literal
source qualifiers while the final block judge correctly required replacement-
profile consistency.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
V510_PATH = SCRIPT_DIR / "generate_tofu_author_pairrepair_v5_10.py"

DESIGN_VERSION = "tofu-author-premisefix-v5.11"
CONTRAST_SCHEMA_VERSION = "semantic-agent-premise-map-v3"
SURFACE_RENDERER = "semantic-agent-premise-mapped-c01-v5.11"
MAPPING_SCOPE = "v5.9-frozen-profile-premise-aware-selective-row-repair"
RESPONSE_CONTRACT_POLICY = "causal-premise-map-v5.11"
AGENT_PROTOCOL_VERSION = "context-premise-map-pair-audit-repair-v3"
SEMANTIC_BRIEF_SCHEMA_VERSION = "tofu-row-semantic-brief-v2"
PREMISE_POLICY_VERSION = "source-to-replacement-premise-map-v1"
PAIR_CONTRACT_VERSION = "approximate-nuisance-match-v2-premise-aware"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v510 = load_module("tofu_author_premisefix_v511_v510", V510_PATH)
v59 = v510.v59
v58 = v510.v58
v57 = v510.v57
v56 = v510.v56
v55 = v510.v55
v53 = v510.v53
v52 = v510.v52
ciru = v510.ciru

BASE_V59_CONFIGURE = v510.BASE_V59_CONFIGURE
BASE_V59_LOAD_BRIEF = v510.BASE_V59_LOAD_BRIEF
BASE_V59_GENERATE_ROW = v510.BASE_V59_GENERATE_ROW
BASE_V59_MATERIALIZE_PLAN = v510.BASE_V59_MATERIALIZE_PLAN
BASE_V59_ASSEMBLE_BLOCK = v510.BASE_V59_ASSEMBLE_BLOCK

BASE_STATE_DIR: Path | None = None
BASE_STATE_DIGEST = ""
BASE_BLOCK_RECORDS: Dict[str, Dict] = {}
HUMAN_REPAIR_MANIFEST_PATH: Path | None = None
HUMAN_REPAIR_MANIFEST_DIGEST = ""
HUMAN_REPAIRS: Dict[str, Dict] = {}

CRITIC_FIELDS = v510.CRITIC_FIELDS
GENERATOR_OUTPUT_FIELDS = v510.GENERATOR_OUTPUT_FIELDS


PREMISE_MAPPING_POLICY = {
    "version": PREMISE_POLICY_VERSION,
    "preserve_semantically": [
        "target relation and argument roles",
        "question intent and cardinality",
        "evidence or answerability status when compatible with the frozen ledger",
        "approximate information granularity and natural response form",
    ],
    "replace_when_source_specific": [
        "author identity and identity descriptors",
        "named books, awards, institutions, and organizations",
        "award or subject domains tied to the source author",
        "places, dates, numbers, and biographical qualifiers",
        "any other factual qualifier that conflicts with the replacement profile",
    ],
    "question_must_be_true_of_replacement_profile": True,
    "answer_must_satisfy_rewritten_question": True,
    "source_specific_literals_are_invariants": False,
    "response_mode_polarity_is_not_a_truth_override": True,
    "precedence": (
        "This policy overrides any legacy semantic_brief.must_preserve item "
        "that names or entails a source-specific factual literal."
    ),
}


SEMANTIC_PLANNER_PROMPT = """Act as the semantic planner for one TOFU causal
intervention. Read the immutable C11 question/answer, frozen source and
replacement facts, the replacement-author profile, and its complete ledger.

Separate semantic invariants from source-specific factual premises:
- Preserve the relation, argument roles, cardinality, evidence status, and
  approximate response form.
- Do not preserve a literal merely because it scopes C11. Author names, named
  works, award/subject domains, places, dates, numbers, and identity
  descriptors are source-specific whenever they conflict with the replacement
  profile; map them to coherent replacement-profile counterparts.
- The C01 question must be true of the replacement profile and its answer must
  directly satisfy the rewritten question.
- Response form is a nuisance target, not permission to make a false claim.
  For an affirmative C11, prefer a coherent replacement-specific question that
  remains affirmative rather than retaining a source premise and answering it
  falsely or negatively.

Describe meaning, not wording. Return JSON only with exactly:
{"same_relation_definition":"relation and arguments C01 must preserve",
 "answer_object_type":"semantic object requested by C11",
 "cardinality":"one/list/binary/open or another concise value",
 "response_mode":"semantic answer form, not blindly copied polarity",
 "source_specific_premises":["source premise that must be replaced"],
 "replacement_specific_premises":["coherent replacement premise"],
 "must_preserve":["relation-level or nuisance constraint only"],
 "must_change":["author and source-specific factual targets"],
 "risk_notes":["likely ambiguity"]}
"""


PREMISE_GENERATOR_PROMPT = """Act as the repair generator for one TOFU C01
causal pair. The payload contains immutable C11, a frozen replacement-author
profile and ledger, a semantic brief, and an explicit premise_mapping_policy.

Produce a complete, coherent C01 question and answer. Preserve the semantic
relation, argument roles, cardinality, evidence status, and approximate
information granularity. Replace every author-specific factual premise that
conflicts with the replacement profile, including named works, award or
subject domains, places, dates, numbers, and identity descriptors. The
rewritten question must be true of the replacement profile and the answer must
directly satisfy that rewritten question.

The premise_mapping_policy has precedence over legacy semantic-brief wording.
Never retain a source-specific literal merely to copy C11 surface form or
polarity. Response-mode metadata is advisory nuisance context and must never
force a false answer. Do not mention the source fact, counterfactuals,
generation, controls, or fictionality unless inherited from C11. When an exact
replacement identity literal is required, include it in the question.

When validation feedback and a previous candidate are present, repair only the
reported defect. Return JSON only with exactly:
{"c01_question":"complete question","replacement_answer":"complete answer"}
"""


PREMISE_CRITIC_PROMPT = """Act as an independent causal-pair critic. Compare
immutable C11 with proposed C01 using the frozen source/replacement facts, full
replacement-author ledger, semantic brief, and premise_mapping_policy.

The intended intervention changes author and target factual object while
preserving relation, argument roles, cardinality, evidence status, and
approximate information granularity. Source-specific factual qualifiers are
not invariants. If a C11 author, title, award/subject domain, place, date,
number, or identity descriptor conflicts with the replacement profile, C01
must replace it with the coherent replacement-profile premise.

Apply these precedence rules:
- The C01 question must be true of the replacement profile.
- The C01 answer must directly satisfy its actual rewritten question.
- Do not demand literal source qualifiers through must_preserve; the explicit
  premise_mapping_policy overrides conflicting legacy brief wording.
- Preserve semantic response form approximately, but never reject a truthful,
  coherent pair merely because a heuristic polarity/date/list label differs.
- Prefer a replacement-specific affirmative question when that preserves the
  original relation and avoids a false or irrelevant negative answer.
- Reject dropped relation arguments, source-fact comparison language, changed
  evidence status, major granularity expansion, or unnatural question/answer.

Return JSON only. accepted may be true only when every check is true:
{"accepted":true,
 "same_target_relation":true,
 "question_scope_matched":true,
 "question_premises_updated":true,
 "answer_addresses_question":true,
 "replacement_fact_expressed":true,
 "source_fact_removed":true,
 "source_comparison_absent":true,
 "evidence_status_matched":true,
 "information_granularity_matched":true,
 "response_mode_compatible":true,
 "natural_surface":true,
 "reason":"brief evidence grounded in the rewritten question and profile",
 "repair_instruction":"empty when accepted; otherwise one minimal fix"}
"""


def _pop_path_argument(name: str) -> Path:
    if name not in sys.argv:
        raise SystemExit(f"V5.11 requires {name}")
    index = sys.argv.index(name)
    try:
        value = Path(sys.argv[index + 1]).resolve()
    except IndexError as exc:
        raise SystemExit(f"{name} requires a path") from exc
    del sys.argv[index:index + 2]
    return value


def _pop_optional_path_argument(name: str) -> Path | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    try:
        value = Path(sys.argv[index + 1]).resolve()
    except IndexError as exc:
        raise SystemExit(f"{name} requires a path") from exc
    del sys.argv[index:index + 2]
    return value


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(path: Path) -> str:
    """Digest frozen semantic inputs without depending on directory mtimes."""
    digest = hashlib.sha256()
    selected = []
    for pattern in ("block_*/profile.json", "block_*/rows/*.json", "block_*.json"):
        selected.extend(path.glob(pattern))
    for item in sorted(set(selected), key=lambda value: str(value.relative_to(path))):
        relative = str(item.relative_to(path)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = item.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_context_packet(
    block: Mapping, source: Mapping, profile: Mapping
) -> Dict:
    packet = v510.BASE_V59_BUILD_CONTEXT(block, source, profile)
    packet["agent_protocol_version"] = AGENT_PROTOCOL_VERSION
    packet["premise_mapping_policy"] = dict(PREMISE_MAPPING_POLICY)
    packet["pair_contract"] = {
        "version": PAIR_CONTRACT_VERSION,
        "change_only": [
            "author identity",
            "target factual object",
            "source-specific factual premises required to make C01 coherent",
        ],
        "preserve_approximately": [
            "target relation",
            "argument roles",
            "evidence status",
            "response form",
            "information granularity",
        ],
        "source_comparison_forbidden": True,
    }
    return packet


def load_or_create_semantic_brief(
    generation_client,
    args,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    state_dir: Path,
):
    block_id = int(block["block_id"])
    source_id = source["source_id"]
    current = v59.semantic_context_path(state_dir, block_id, source_id)
    if current.is_file():
        return BASE_V59_LOAD_BRIEF(
            generation_client, args, block, source, profile, state_dir
        )

    if BASE_STATE_DIR is not None:
        inherited = v59.semantic_context_path(BASE_STATE_DIR, block_id, source_id)
        if inherited.is_file():
            cached = json.loads(inherited.read_text(encoding="utf-8"))
            brief = v59.validate_semantic_brief(cached["semantic_brief"])
            packet = build_context_packet(block, source, profile)
            payload = {
                "design_version": DESIGN_VERSION,
                "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
                "premise_policy_version": PREMISE_POLICY_VERSION,
                "premise_policy_overrides_legacy_must_preserve": True,
                "profile_digest": profile["profile_digest"],
                "ledger_digest": profile["ledger_digest"],
                "context_digest": v59.context_digest(packet),
                "planner_attempt": 0,
                "semantic_brief": brief,
                "inherited_from_design": "tofu-author-semantic-agent-v5.9",
                "inherited_context": str(inherited),
            }
            v52.write_json(current, payload)
            print(
                f"migrate_premise_context block={block_id} source={source_id}",
                flush=True,
            )
            return packet, brief, 0

    return BASE_V59_LOAD_BRIEF(
        generation_client, args, block, source, profile, state_dir
    )


def migrate_profile(
    generation_client,
    judge_client,
    args,
    block: Mapping,
    protected_authors: Sequence[str],
    state_dir: Path,
) -> Dict:
    v510.BASE_STATE_DIR = BASE_STATE_DIR
    return v510.migrate_profile(
        generation_client,
        judge_client,
        args,
        block,
        protected_authors,
        state_dir,
    )


def _base_candidate(block_id: int, source_id: str) -> Mapping | None:
    if BASE_STATE_DIR is None:
        return None
    path = v52.row_path(BASE_STATE_DIR, block_id, source_id)
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("design_version") != "tofu-author-semantic-agent-v5.9":
        raise ValueError(f"base row is not V5.9: {path}")
    candidate = cached.get("candidate")
    if not isinstance(candidate, Mapping):
        return None
    return candidate


def _audit_path(state_dir: Path, block_id: int, source_id: str) -> Path:
    return (
        v52.block_directory(state_dir, block_id)
        / "premise_audits"
        / f"{source_id}.json"
    )


def _validate_human_candidate(directive: Mapping) -> Dict[str, str]:
    if not isinstance(directive, Mapping):
        raise ValueError("human repair directive must be an object")
    result = {}
    for field in ("c01_question", "replacement_answer", "category", "reason"):
        value = directive.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"human repair directive {field} must be non-empty")
        result[field] = " ".join(value.split())
    return result


def apply_human_candidate(
    generation_client,
    critic_client,
    args,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    state_dir: Path,
    directive: Mapping,
) -> Dict:
    """Validate, independently criticise, and checkpoint an explicit repair."""
    block_id = int(block["block_id"])
    source_id = source["source_id"]
    manual = _validate_human_candidate(directive)
    packet, brief, planner_attempt = load_or_create_semantic_brief(
        generation_client, args, block, source, profile, state_dir
    )
    validated = v510.validate_structural_candidate(
        block,
        source,
        profile,
        {
            "c01_question": manual["c01_question"],
            "replacement_answer": manual["replacement_answer"],
        },
    )
    verdict = v59.request_critic_verdict(
        critic_client,
        args,
        packet,
        brief,
        validated,
        block_id,
        source_id,
        0,
    )
    v52.write_json(_audit_path(state_dir, block_id, source_id), {
        "design_version": DESIGN_VERSION,
        "premise_policy_version": PREMISE_POLICY_VERSION,
        "human_repair_manifest": str(HUMAN_REPAIR_MANIFEST_PATH),
        "human_repair_manifest_digest": HUMAN_REPAIR_MANIFEST_DIGEST,
        "candidate": v59.candidate_for_prompt(validated),
        "human_directive": manual,
        "critic_verdict": verdict,
    })
    if not verdict["accepted"]:
        raise ValueError(
            "explicit human candidate failed independent critic: "
            + v59.critic_feedback(verdict)
        )
    trace = {
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
        "premise_policy_version": PREMISE_POLICY_VERSION,
        "pair_contract_version": PAIR_CONTRACT_VERSION,
        "context_digest": v59.context_digest(packet),
        "planner_attempt": planner_attempt,
        "generator_attempt": 0,
        "generator_model": "explicit-human-repair",
        "critic_model": args.judge_model,
        "critic_independent_call": True,
        "critic_verdict": verdict,
        "deterministic_observations": validated["deterministic_observations"],
        "human_manual_repair": {
            "category": manual["category"],
            "reason": manual["reason"],
            "manifest": str(HUMAN_REPAIR_MANIFEST_PATH),
            "manifest_digest": HUMAN_REPAIR_MANIFEST_DIGEST,
        },
        "base_state_digest": BASE_STATE_DIGEST,
    }
    validated["semantic_agent_trace"] = trace
    validated["mapping_attempt"] = 0
    validated["repair_generation"] = 1
    path = v52.row_path(state_dir, block_id, source_id)
    v59.write_agent_row_checkpoint(path, profile, validated, 0, 1)
    print(
        f"premise_human_row_applied block={block_id} source={source_id}",
        flush=True,
    )
    return validated


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
    directive = HUMAN_REPAIRS.get(source_id)
    if force and directive is not None:
        raise RuntimeError(
            f"explicit human repair {source_id} was rejected by the final "
            "block judge; refusing to replace it with an unapproved model row"
        )
    if not force:
        if directive is not None:
            return apply_human_candidate(
                generation_client,
                critic_client,
                args,
                block,
                source,
                profile,
                state_dir,
                directive,
            )
        current = v59.load_cached_agent_row(path, block, source, profile)
        if current is not None:
            print(
                f"reuse_premise_row block={block_id} source={source_id}",
                flush=True,
            )
            return current

        inherited = _base_candidate(block_id, source_id)
        packet, brief, planner_attempt = load_or_create_semantic_brief(
            generation_client, args, block, source, profile, state_dir
        )
        validated = None
        verdict = None
        if inherited is not None:
            try:
                validated = v510.validate_structural_candidate(
                    block, source, profile, inherited
                )
                verdict = v59.request_critic_verdict(
                    critic_client,
                    args,
                    packet,
                    brief,
                    validated,
                    block_id,
                    source_id,
                    0,
                )
                v52.write_json(_audit_path(state_dir, block_id, source_id), {
                    "design_version": DESIGN_VERSION,
                    "premise_policy_version": PREMISE_POLICY_VERSION,
                    "base_state_digest": BASE_STATE_DIGEST,
                    "candidate": v59.candidate_for_prompt(validated),
                    "critic_verdict": verdict,
                })
                if verdict["accepted"]:
                    trace = {
                        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                        "semantic_brief_schema_version": (
                            SEMANTIC_BRIEF_SCHEMA_VERSION
                        ),
                        "premise_policy_version": PREMISE_POLICY_VERSION,
                        "pair_contract_version": PAIR_CONTRACT_VERSION,
                        "context_digest": v59.context_digest(packet),
                        "planner_attempt": planner_attempt,
                        "generator_attempt": 0,
                        "generator_model": "inherited-v5.9",
                        "critic_model": args.judge_model,
                        "critic_independent_call": True,
                        "critic_verdict": verdict,
                        "deterministic_observations": validated[
                            "deterministic_observations"
                        ],
                        "inherited_from_design": (
                            "tofu-author-semantic-agent-v5.9"
                        ),
                        "base_state_digest": BASE_STATE_DIGEST,
                    }
                    validated["semantic_agent_trace"] = trace
                    validated["mapping_attempt"] = 0
                    validated["repair_generation"] = 0
                    v59.write_agent_row_checkpoint(
                        path, profile, validated, 0, 0
                    )
                    print(
                        f"premise_row_reused block={block_id} "
                        f"source={source_id}",
                        flush=True,
                    )
                    return validated
                feedback = v59.critic_feedback(verdict)
                previous = validated
            except Exception as exc:
                feedback = str(exc)
                previous = validated or inherited
        else:
            feedback = (
                "No accepted V5.9 row exists. Generate a fresh matched C01 "
                "under premise_mapping_policy, using the frozen replacement "
                "ledger and a question that is true of the replacement profile."
            )
            previous = None
        print(
            f"premise_row_repair block={block_id} source={source_id} "
            f"reason={feedback}",
            flush=True,
        )

    repaired = BASE_V59_GENERATE_ROW(
        generation_client,
        critic_client,
        args,
        block,
        source,
        profile,
        state_dir,
        feedback,
        previous,
        True,
    )
    trace = repaired["semantic_agent_trace"]
    trace.update({
        "premise_policy_version": PREMISE_POLICY_VERSION,
        "pair_contract_version": PAIR_CONTRACT_VERSION,
        "base_state_digest": BASE_STATE_DIGEST,
    })
    v59.write_agent_row_checkpoint(
        path,
        profile,
        repaired,
        int(repaired.get("mapping_attempt", 0)),
        int(repaired.get("repair_generation", 1)),
    )
    return repaired


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    plan = BASE_V59_MATERIALIZE_PLAN(block, profile, candidates, seed)
    for row in plan["row_plans"].values():
        row["render_mode"] = "semantic_agent_premise_mapped_pairrepair"
        row["agent_protocol_version"] = AGENT_PROTOCOL_VERSION
        row["response_contract_policy"] = RESPONSE_CONTRACT_POLICY
        row["premise_policy_version"] = PREMISE_POLICY_VERSION
        row["pair_contract_version"] = PAIR_CONTRACT_VERSION
    return plan


def assemble_block(
    block: Mapping,
    plan: Mapping,
    verdicts: Mapping[str, Mapping],
    args,
) -> Dict:
    assembled = BASE_V59_ASSEMBLE_BLOCK(block, plan, verdicts, args)
    block_id = int(block["block_id"])
    profile_id = (
        f"{args.split}:author-premisefix-v5.11-"
        f"block-{block_id:02d}:seed-{args.seed}"
    )
    assembled.update({
        "profile_id": profile_id,
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "premise_policy_version": PREMISE_POLICY_VERSION,
        "pair_contract_version": PAIR_CONTRACT_VERSION,
        "base_state_digest": BASE_STATE_DIGEST,
    })
    for record in assembled["records"]:
        source_id = record["source_id"]
        base = BASE_BLOCK_RECORDS.get(source_id)
        if base is not None:
            for cell in ("C11", "C10", "C00"):
                if record["cells"][cell] != base["cells"][cell]:
                    raise ValueError(
                        f"{source_id}.{cell} changed from frozen V5.9 block"
                    )
        record.update({
            "profile_id": profile_id,
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "render_mode": "semantic_agent_premise_mapped_pairrepair",
            "premise_policy_version": PREMISE_POLICY_VERSION,
            "pair_contract_version": PAIR_CONTRACT_VERSION,
        })
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
            "premise_policy_version": PREMISE_POLICY_VERSION,
            "pair_contract_version": PAIR_CONTRACT_VERSION,
            "critic_fields": list(CRITIC_FIELDS),
            "base_state_digest": BASE_STATE_DIGEST,
            "frozen_profile_and_ledger_reused": True,
            "inherited_rows_reaudited": True,
            "selective_c01_repair": True,
        })
        manual = record.get("semantic_agent_trace", {}).get(
            "human_manual_repair"
        )
        if manual is not None:
            record["generation"]["human_manual_repair"] = manual
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{source_id}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


def configure_shared_modules() -> None:
    BASE_V59_CONFIGURE()
    for module in (v510, v59, v58, v57, v56, v55, v53, v52):
        module.DESIGN_VERSION = DESIGN_VERSION
        module.CONTRAST_SCHEMA_VERSION = CONTRAST_SCHEMA_VERSION
        module.SURFACE_RENDERER = SURFACE_RENDERER
        module.MAPPING_SCOPE = MAPPING_SCOPE
        module.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v510.BASE_STATE_DIR = BASE_STATE_DIR
    v59.AGENT_PROTOCOL_VERSION = AGENT_PROTOCOL_VERSION
    v59.SEMANTIC_BRIEF_SCHEMA_VERSION = SEMANTIC_BRIEF_SCHEMA_VERSION
    v59.CRITIC_FIELDS = CRITIC_FIELDS
    v59.SEMANTIC_PLANNER_PROMPT = SEMANTIC_PLANNER_PROMPT
    v59.SEMANTIC_GENERATOR_PROMPT = PREMISE_GENERATOR_PROMPT
    v59.SEMANTIC_CRITIC_PROMPT = PREMISE_CRITIC_PROMPT
    v59.build_context_packet = build_context_packet
    v59.validate_structural_candidate = v510.validate_structural_candidate
    v59.validate_row_candidate = v510.validate_structural_candidate
    v59.load_or_create_semantic_brief = load_or_create_semantic_brief
    v59.generate_row = generate_row
    v59.materialize_plan = materialize_plan
    v59.assemble_block = assemble_block
    v53.generate_profile = migrate_profile
    v55.ANSWER_PROMPT = PREMISE_GENERATOR_PROMPT
    v55.validate_row_candidate = v510.validate_structural_candidate
    v55.materialize_plan = materialize_plan
    v55.assemble_block = assemble_block
    v58.materialize_plan = materialize_plan
    v58.assemble_block = assemble_block
    v58.validate_row_candidate = v510.validate_structural_candidate
    ciru.AUTHOR_LEVEL_DESIGNS.add(DESIGN_VERSION)


def main() -> None:
    global BASE_STATE_DIR, BASE_STATE_DIGEST, BASE_BLOCK_RECORDS
    global HUMAN_REPAIR_MANIFEST_PATH, HUMAN_REPAIR_MANIFEST_DIGEST
    global HUMAN_REPAIRS

    BASE_STATE_DIR = _pop_path_argument("--base-state-dir")
    HUMAN_REPAIR_MANIFEST_PATH = _pop_optional_path_argument(
        "--human-repair-manifest"
    )
    HUMAN_REPAIR_MANIFEST_DIGEST = ""
    HUMAN_REPAIRS = {}
    BASE_BLOCK_RECORDS = {}
    if not BASE_STATE_DIR.is_dir():
        raise SystemExit(f"missing frozen V5.9 state directory: {BASE_STATE_DIR}")
    BASE_STATE_DIGEST = state_digest(BASE_STATE_DIR)
    BASE_BLOCK_RECORDS = {}
    for path in sorted(BASE_STATE_DIR.glob("block_[0-9][0-9].json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            BASE_BLOCK_RECORDS[record["source_id"]] = record
    if HUMAN_REPAIR_MANIFEST_PATH is not None:
        if not HUMAN_REPAIR_MANIFEST_PATH.is_file():
            raise SystemExit(
                f"missing human repair manifest: {HUMAN_REPAIR_MANIFEST_PATH}"
            )
        payload = json.loads(
            HUMAN_REPAIR_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if payload.get("design_version") != DESIGN_VERSION:
            raise SystemExit("human repair manifest has wrong design_version")
        raw_repairs = payload.get("repair_rows")
        if not isinstance(raw_repairs, Mapping) or not raw_repairs:
            raise SystemExit("human repair manifest repair_rows must be non-empty")
        try:
            HUMAN_REPAIRS = {
                str(source_id): _validate_human_candidate(directive)
                for source_id, directive in raw_repairs.items()
            }
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        HUMAN_REPAIR_MANIFEST_DIGEST = file_digest(
            HUMAN_REPAIR_MANIFEST_PATH
        )

    configure_shared_modules()
    original = v59.configure_shared_modules
    v59.configure_shared_modules = lambda: None
    try:
        v59.main()
    finally:
        v59.configure_shared_modules = original

    profiles_path = v59._profiles_path_from_argv()
    if profiles_path is not None and profiles_path.is_file():
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        profiles.update({
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
            "premise_policy_version": PREMISE_POLICY_VERSION,
            "pair_contract_version": PAIR_CONTRACT_VERSION,
            "critic_fields": list(CRITIC_FIELDS),
            "base_state_dir": str(BASE_STATE_DIR),
            "base_state_digest": BASE_STATE_DIGEST,
            "frozen_profile_and_ledger_reused": True,
            "inherited_rows_reaudited": True,
            "selective_c01_repair": True,
            "human_repair_manifest": (
                str(HUMAN_REPAIR_MANIFEST_PATH)
                if HUMAN_REPAIR_MANIFEST_PATH is not None else None
            ),
            "human_repair_manifest_digest": HUMAN_REPAIR_MANIFEST_DIGEST,
            "human_repair_rows": sorted(HUMAN_REPAIRS),
        })
        v52.write_json(profiles_path, profiles)


if __name__ == "__main__":
    main()
