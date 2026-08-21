#!/usr/bin/env python3
"""Selectively repair V5.9 C01 cells under an explicit pair contract.

V5.10 is a new ablation and never overwrites V5.9.  It migrates the frozen
V5.9 replacement profile, ledger, semantic brief, immutable C11, and placebo
cells.  Every inherited C01 is re-audited by an independent pair critic.  A
passing C01 is copied byte-for-byte; only rejected rows enter the existing
context-plan-generate-critic-repair loop.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
V59_PATH = SCRIPT_DIR / "generate_tofu_author_semantic_agent_v5_9.py"
DESIGN_VERSION = "tofu-author-pairrepair-v5.10"
CONTRAST_SCHEMA_VERSION = "semantic-agent-pairrepair-v2"
SURFACE_RENDERER = "semantic-agent-selective-c01-pairrepair-v5.10"
MAPPING_SCOPE = "v5.9-frozen-profile-selective-row-pairrepair"
RESPONSE_CONTRACT_POLICY = "causal-pair-contract-v5.10"
AGENT_PROTOCOL_VERSION = "context-plan-pair-audit-selective-repair-v2"
SEMANTIC_BRIEF_SCHEMA_VERSION = "tofu-row-semantic-brief-v1"
PAIR_CONTRACT_VERSION = "approximate-nuisance-match-v1"
GENERATOR_OUTPUT_FIELDS = ("c01_question", "replacement_answer")
CRITIC_FIELDS = (
    "same_target_relation",
    "question_scope_matched",
    "question_premises_updated",
    "answer_addresses_question",
    "replacement_fact_expressed",
    "source_fact_removed",
    "source_comparison_absent",
    "evidence_status_matched",
    "information_granularity_matched",
    "response_mode_compatible",
    "natural_surface",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v59 = load_module("tofu_author_pairrepair_v510_v59", V59_PATH)
v58 = v59.v58
v57 = v59.v57
v56 = v59.v56
v55 = v59.v55
v53 = v59.v53
v52 = v59.v52
ciru = v59.ciru

BASE_V59_CONFIGURE = v59.configure_shared_modules
BASE_V59_BUILD_CONTEXT = v59.build_context_packet
BASE_V59_VALIDATE_STRUCTURAL = v59.validate_structural_candidate
BASE_V59_LOAD_BRIEF = v59.load_or_create_semantic_brief
BASE_V59_GENERATE_ROW = v59.generate_row
BASE_V59_MATERIALIZE_PLAN = v59.materialize_plan
BASE_V59_ASSEMBLE_BLOCK = v59.assemble_block
BASE_V53_GENERATE_PROFILE = v53.generate_profile

BASE_STATE_DIR: Path | None = None
BASE_DATA_PATH: Path | None = None
BASE_PROFILES_PATH: Path | None = None
BASE_RECORDS: Dict[str, Dict] = {}
BASE_DATA_DIGEST = ""
BASE_PROFILES_DIGEST = ""
REPAIR_MANIFEST_PATH: Path | None = None
REPAIR_MANIFEST_DIGEST = ""
HUMAN_REPAIRS: Dict[str, Dict] = {}


PAIR_GENERATOR_PROMPT = """Act as the repair generator for one TOFU C01
causal pair.  The payload contains immutable C11, a frozen replacement-author
profile and ledger, a semantic brief, and (for repair) the rejected V5.9 C01.

Change only what the validation feedback identifies.  C01 must replace the
author and core fact while preserving C11's target relation, named or scoped
question constraints, evidence/answerability status, and approximate
information granularity.  Keep a concise C11 concise; do not turn one core
proposition into a numbered multi-paragraph answer.  Never describe C01 by
contrasting it with, negating, or mentioning the source C11 fact.

Natural-language nuisance matching is approximate.  Do not copy awkward
punctuation and do not force yes_no/list/date labels to be literally equal.
When generator_contract.required_question_identity_literal is non-null, copy
that exact author name into the question.

Return JSON only with exactly:
{"c01_question":"complete question","replacement_answer":"complete answer"}
"""


PAIR_CRITIC_PROMPT = """Act as an independent causal-pair critic.  Compare
the complete immutable C11 with the proposed C01, using the frozen source and
replacement facts, semantic brief, and replacement-author ledger.

The intended contrast changes the author and core factual object.  It must not
also change the semantic relation, named/specific scope, evidence status, or
information granularity enough to create a shortcut.  Reject source-fact
comparison language such as 'rather than' when it imports C11 content.

Apply approximate nuisance matching, not literal surface matching:
- Do NOT reject only because one answer starts with Yes, punctuation differs,
  or a heuristic labels prose as list/date/yes_no differently.
- Small wording, length, and detail differences are acceptable.
- Reject a dropped named-book constraint, known versus speculative evidence,
  one proposition versus several independent claims, or one sentence versus
  a numbered/multi-paragraph expansion.
- A direct short answer is acceptable when the question itself asks only for
  a name, date, title, or other atomic value.

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
 "reason":"brief pairwise evidence",
 "repair_instruction":"empty when accepted; otherwise one minimal fix"}
"""


def _normalise(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pop_path_argument(name: str, *, required: bool) -> Path | None:
    if name not in sys.argv:
        if required:
            raise SystemExit(f"V5.10 requires {name}")
        return None
    index = sys.argv.index(name)
    try:
        value = Path(sys.argv[index + 1]).resolve()
    except IndexError as exc:
        raise SystemExit(f"{name} requires a path") from exc
    del sys.argv[index:index + 2]
    return value


def _sentence_count(text: str) -> int:
    compact = _normalise(text)
    if not compact:
        return 0
    return max(1, sum(compact.count(mark) for mark in ".!?"))


def _numbered_expansion(text: str) -> bool:
    compact = _normalise(text)
    return any(token in compact for token in ("1)", "2)", "1.", "2."))


def build_context_packet(block: Mapping, source: Mapping, profile: Mapping) -> Dict:
    packet = BASE_V59_BUILD_CONTEXT(block, source, profile)
    packet["agent_protocol_version"] = AGENT_PROTOCOL_VERSION
    packet["pair_contract"] = {
        "version": PAIR_CONTRACT_VERSION,
        "change_only": ["author identity", "target factual object"],
        "preserve_approximately": [
            "target relation", "question scope", "evidence status",
            "response mode", "information granularity",
        ],
        "surface_labels_are_advisory": True,
        "source_comparison_forbidden": True,
    }
    return packet


def validate_structural_candidate(
    block: Mapping, source: Mapping, profile: Mapping, generated: Mapping
) -> Dict:
    validated = BASE_V59_VALIDATE_STRUCTURAL(block, source, profile, generated)
    c11 = source["answer"]
    c01 = validated["replacement_answer"]
    c11_words = len(_normalise(c11).split())
    c01_words = len(_normalise(c01).split())
    validated["deterministic_observations"].update({
        "pair_contract_version": PAIR_CONTRACT_VERSION,
        "c11_word_count": c11_words,
        "c01_word_count": c01_words,
        "answer_length_ratio": (
            round(c01_words / max(c11_words, 1), 4)
        ),
        "c11_sentence_count": _sentence_count(c11),
        "c01_sentence_count": _sentence_count(c01),
        "c11_numbered_expansion": _numbered_expansion(c11),
        "c01_numbered_expansion": _numbered_expansion(c01),
        "counts_are_critic_evidence_not_hard_gates": True,
    })
    validated["response_contract_policy"] = RESPONSE_CONTRACT_POLICY
    return validated


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
                f"migrate_agent_context block={block_id} source={source_id}",
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
    block_id = int(block["block_id"])
    current = v52.profile_path(state_dir, block_id)
    if current.is_file():
        checkpoint = json.loads(current.read_text(encoding="utf-8"))
        if (
            checkpoint.get("design_version") != DESIGN_VERSION
            or checkpoint.get("frozen_profile") is not True
        ):
            raise RuntimeError(
                f"invalid V5.10 frozen profile checkpoint: {current}"
            )
        return BASE_V53_GENERATE_PROFILE(
            generation_client, judge_client, args, block,
            protected_authors, state_dir,
        )
    if BASE_STATE_DIR is None:
        raise RuntimeError("V5.10 has no frozen V5.9 state directory")
    inherited = v52.profile_path(BASE_STATE_DIR, block_id)
    if not inherited.is_file():
        raise RuntimeError(f"missing frozen V5.9 profile: {inherited}")
    cached = json.loads(inherited.read_text(encoding="utf-8"))
    profile = v53.validate_profile(block, cached["profile"], protected_authors)
    verdict = v53.validate_profile_judgement(cached["semantic_judge"], block)
    profile["profile_attempt"] = 0
    profile["profile_semantic_judge"] = verdict
    v52.write_json(current, {
        "design_version": DESIGN_VERSION,
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "ledger_digest": profile["ledger_digest"],
        "profile_attempt": 0,
        "profile": {
            key: value for key, value in profile.items()
            if key not in {"fact_ledger_by_source", "profile_semantic_judge"}
        },
        "semantic_judge": verdict,
        "inherited_from_design": "tofu-author-semantic-agent-v5.9",
        "inherited_profile": str(inherited),
        "frozen_profile": True,
    })
    print(f"migrate_v59_profile block={block_id}", flush=True)
    return profile


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
        raise ValueError(f"base row has no candidate: {path}")
    return candidate


def _pair_audit_path(state_dir: Path, block_id: int, source_id: str) -> Path:
    return (
        v52.block_directory(state_dir, block_id)
        / "pair_audits" / f"{source_id}.json"
    )


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
        current = v59.load_cached_agent_row(path, block, source, profile)
        if current is not None:
            print(f"reuse_pair_row block={block_id} source={source_id}", flush=True)
            return current

        inherited = _base_candidate(block_id, source_id)
        if inherited is None:
            raise RuntimeError(f"missing frozen V5.9 row {source_id}")
        packet, brief, planner_attempt = load_or_create_semantic_brief(
            generation_client, args, block, source, profile, state_dir
        )
        validated = None
        verdict = None
        try:
            validated = validate_structural_candidate(
                block, source, profile, inherited
            )
            verdict = v59.request_critic_verdict(
                critic_client, args, packet, brief, validated,
                block_id, source_id, 0,
            )
            v52.write_json(_pair_audit_path(state_dir, block_id, source_id), {
                "design_version": DESIGN_VERSION,
                "pair_contract_version": PAIR_CONTRACT_VERSION,
                "base_data_digest": BASE_DATA_DIGEST,
                "candidate": v59.candidate_for_prompt(validated),
                "critic_verdict": verdict,
                "human_repair_directive": HUMAN_REPAIRS.get(source_id),
            })
            human_repair = HUMAN_REPAIRS.get(source_id)
            if verdict["accepted"] and human_repair is None:
                trace = {
                    "agent_protocol_version": AGENT_PROTOCOL_VERSION,
                    "semantic_brief_schema_version": SEMANTIC_BRIEF_SCHEMA_VERSION,
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
                    "inherited_from_design": "tofu-author-semantic-agent-v5.9",
                    "base_data_digest": BASE_DATA_DIGEST,
                }
                validated["semantic_agent_trace"] = trace
                validated["mapping_attempt"] = 0
                validated["repair_generation"] = 0
                v59.write_agent_row_checkpoint(
                    path, profile, validated, 0, 0
                )
                print(
                    f"pair_row_reused block={block_id} source={source_id}",
                    flush=True,
                )
                return validated
            if human_repair is not None:
                feedback = (
                    "Authoritative human pair audit requires minimal repair. "
                    f"Category: {human_repair['category']}. "
                    f"Reason: {human_repair['reason']}"
                )
            else:
                feedback = v59.critic_feedback(verdict)
            previous = validated
        except Exception as exc:
            feedback = str(exc)
            previous = validated or inherited
        print(
            f"pair_row_repair block={block_id} source={source_id} "
            f"reason={feedback}",
            flush=True,
        )

    repaired = BASE_V59_GENERATE_ROW(
        generation_client, critic_client, args, block, source, profile,
        state_dir, feedback, previous, True,
    )
    directive = HUMAN_REPAIRS.get(source_id)
    if directive is not None:
        repaired["semantic_agent_trace"]["human_audit_repair"] = directive
        repaired["semantic_agent_trace"][
            "repair_manifest_digest"
        ] = REPAIR_MANIFEST_DIGEST
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
        row["render_mode"] = "semantic_agent_selective_pairrepair"
        row["agent_protocol_version"] = AGENT_PROTOCOL_VERSION
        row["response_contract_policy"] = RESPONSE_CONTRACT_POLICY
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
        f"{args.split}:author-pairrepair-v5.10-"
        f"block-{block_id:02d}:seed-{args.seed}"
    )
    assembled.update({
        "profile_id": profile_id,
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "pair_contract_version": PAIR_CONTRACT_VERSION,
        "base_data_digest": BASE_DATA_DIGEST,
    })
    for record in assembled["records"]:
        source_id = record["source_id"]
        base = BASE_RECORDS.get(source_id)
        if base is None:
            raise ValueError(f"base V5.9 data lacks {source_id}")
        for cell in ("C11", "C10", "C00"):
            if record["cells"][cell] != base["cells"][cell]:
                raise ValueError(
                    f"{source_id}.{cell} changed; V5.10 may repair only C01"
                )
        record.update({
            "profile_id": profile_id,
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "render_mode": "semantic_agent_selective_pairrepair",
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
            "pair_contract_version": PAIR_CONTRACT_VERSION,
            "critic_fields": list(CRITIC_FIELDS),
            "base_data_digest": BASE_DATA_DIGEST,
            "base_profiles_digest": BASE_PROFILES_DIGEST,
            "frozen_cells_reused": ["C11", "C10", "C00"],
            "selective_c01_repair": True,
            "repair_manifest_digest": REPAIR_MANIFEST_DIGEST,
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{source_id}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


def configure_shared_modules() -> None:
    BASE_V59_CONFIGURE()
    for module in (v59, v58, v57, v56, v55, v53, v52):
        module.DESIGN_VERSION = DESIGN_VERSION
        module.CONTRAST_SCHEMA_VERSION = CONTRAST_SCHEMA_VERSION
        module.SURFACE_RENDERER = SURFACE_RENDERER
        module.MAPPING_SCOPE = MAPPING_SCOPE
        module.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v59.AGENT_PROTOCOL_VERSION = AGENT_PROTOCOL_VERSION
    v59.SEMANTIC_BRIEF_SCHEMA_VERSION = SEMANTIC_BRIEF_SCHEMA_VERSION
    v59.CRITIC_FIELDS = CRITIC_FIELDS
    v59.SEMANTIC_GENERATOR_PROMPT = PAIR_GENERATOR_PROMPT
    v59.SEMANTIC_CRITIC_PROMPT = PAIR_CRITIC_PROMPT
    v59.build_context_packet = build_context_packet
    v59.validate_structural_candidate = validate_structural_candidate
    v59.validate_row_candidate = validate_structural_candidate
    v59.load_or_create_semantic_brief = load_or_create_semantic_brief
    v59.generate_row = generate_row
    v59.materialize_plan = materialize_plan
    v59.assemble_block = assemble_block
    v53.generate_profile = migrate_profile
    v55.ANSWER_PROMPT = PAIR_GENERATOR_PROMPT
    v55.validate_row_candidate = validate_structural_candidate
    v55.materialize_plan = materialize_plan
    v55.assemble_block = assemble_block
    v58.validate_row_candidate = validate_structural_candidate
    v58.materialize_plan = materialize_plan
    v58.assemble_block = assemble_block
    ciru.AUTHOR_LEVEL_DESIGNS.add(DESIGN_VERSION)


def main() -> None:
    global BASE_STATE_DIR, BASE_DATA_PATH, BASE_PROFILES_PATH
    global BASE_RECORDS, BASE_DATA_DIGEST, BASE_PROFILES_DIGEST
    global REPAIR_MANIFEST_PATH, REPAIR_MANIFEST_DIGEST, HUMAN_REPAIRS

    BASE_STATE_DIR = _pop_path_argument("--base-state-dir", required=True)
    BASE_DATA_PATH = _pop_path_argument("--base-data", required=True)
    BASE_PROFILES_PATH = _pop_path_argument("--base-profiles", required=True)
    REPAIR_MANIFEST_PATH = _pop_path_argument(
        "--repair-manifest", required=False
    )
    for label, path in (
        ("base state", BASE_STATE_DIR),
        ("base data", BASE_DATA_PATH),
        ("base profiles", BASE_PROFILES_PATH),
    ):
        if path is None or not path.exists():
            raise SystemExit(f"missing frozen V5.9 {label}: {path}")

    with BASE_DATA_PATH.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    BASE_RECORDS = {row["source_id"]: row for row in records}
    if len(BASE_RECORDS) != len(records):
        raise SystemExit("frozen V5.9 data has duplicate source_id values")
    wrong_design = sorted(
        row["source_id"] for row in records
        if row.get("design_version") != "tofu-author-semantic-agent-v5.9"
    )
    if wrong_design:
        raise SystemExit(
            f"base data contains non-V5.9 rows: {wrong_design[:5]}"
        )
    base_profiles = json.loads(BASE_PROFILES_PATH.read_text(encoding="utf-8"))
    if base_profiles.get("design_version") != "tofu-author-semantic-agent-v5.9":
        raise SystemExit("base profile artifact is not V5.9")
    BASE_DATA_DIGEST = _sha256(BASE_DATA_PATH)
    BASE_PROFILES_DIGEST = _sha256(BASE_PROFILES_PATH)
    if REPAIR_MANIFEST_PATH is not None:
        if not REPAIR_MANIFEST_PATH.is_file():
            raise SystemExit(
                f"missing human repair manifest: {REPAIR_MANIFEST_PATH}"
            )
        manifest = json.loads(REPAIR_MANIFEST_PATH.read_text(encoding="utf-8"))
        directives = manifest.get("repair_rows")
        if not isinstance(directives, dict):
            raise SystemExit("repair manifest repair_rows must be an object")
        unknown = sorted(set(directives) - set(BASE_RECORDS))
        if unknown:
            raise SystemExit(f"repair manifest has unknown rows: {unknown}")
        for source_id, directive in directives.items():
            if not isinstance(directive, dict) or not all(
                isinstance(directive.get(field), str)
                and directive[field].strip()
                for field in ("category", "reason")
            ):
                raise SystemExit(
                    f"invalid repair directive for {source_id}"
                )
        HUMAN_REPAIRS = directives
        REPAIR_MANIFEST_DIGEST = _sha256(REPAIR_MANIFEST_PATH)

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
            "pair_contract_version": PAIR_CONTRACT_VERSION,
            "critic_fields": list(CRITIC_FIELDS),
            "base_data": str(BASE_DATA_PATH),
            "base_data_digest": BASE_DATA_DIGEST,
            "base_profiles": str(BASE_PROFILES_PATH),
            "base_profiles_digest": BASE_PROFILES_DIGEST,
            "selective_c01_repair": True,
            "repair_manifest": (
                str(REPAIR_MANIFEST_PATH)
                if REPAIR_MANIFEST_PATH is not None else None
            ),
            "repair_manifest_digest": REPAIR_MANIFEST_DIGEST,
            "human_repair_rows": sorted(HUMAN_REPAIRS),
        })
        v52.write_json(profiles_path, profiles)


if __name__ == "__main__":
    main()
