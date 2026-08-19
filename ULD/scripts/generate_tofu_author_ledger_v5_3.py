#!/usr/bin/env python3
"""Generate TOFU author factorial data from a frozen block fact ledger.

V5.3 preserves immutable C11, the V5 anchor catalog, deterministic rendering,
and professional placebo cells.  Unlike V5.2, detailed replacement-author
facts are decided and semantically audited once per block before independent
row-local anchor mapping begins.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
V52_PATH = SCRIPT_DIR / "generate_tofu_author_rowlocal_v5_2.py"
DESIGN_VERSION = "tofu-author-ledger-rowlocal-v5.3"
SURFACE_RENDERER = "deterministic-ledger-rowlocal-anchor-v5.3"
MAPPING_SCOPE = "frozen-ledger-conditioned-row-local-anchors"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v52 = load_module("tofu_author_ledger_v53_v52", V52_PATH)
anchors = v52.anchors
ciru = v52.ciru

BASE_VALIDATE_PROFILE = v52.validate_profile
BASE_ROW_PAYLOAD = v52.row_payload
BASE_VALIDATE_ROW = v52.validate_row_candidate
BASE_CANDIDATE_FOR_PROMPT = v52.candidate_for_prompt
BASE_MATERIALIZE_PLAN = v52.materialize_plan
BASE_JUDGE_PAYLOAD = v52.judge_payload
BASE_ASSEMBLE_BLOCK = v52.assemble_block
BASE_LOAD_VALID_BLOCK = v52.load_valid_block


PROFILE_PROMPT = """Create one coherent fictional replacement-author profile
and a frozen typed fact ledger for exactly one 20-row TOFU author block.

The ledger, not later row mappers, is the only stage allowed to invent facts.
Rules:
- preserve target_entity and required replacement_pronouns;
- choose one replacement_entity that is not a protected author;
- return exactly one ledger entry for every source_id;
- factual rows must change the factual object, not merely author identity;
- identity and unavailable rows must use their declared intervention_policy;
- use the same fact_key and exactly the same replacement_fact whenever rows
  express the same underlying author fact;
- use different fact_keys for distinct books, awards, dates, or relations;
- all locations, nationality, upbringing, genres, works, awards, chronology,
  and other biographical details must form one non-contradictory profile;
- replacement_fact is concise evidence, not a rewritten question or answer;
- do not include anchor group IDs or surface edits.

If validation_feedback and previous_candidate are present, return the complete
profile and ledger with the reported general inconsistency repaired.

Return JSON only:
{"target_entity":"required target","replacement_entity":"new author",
 "replacement_pronouns":"required class","profile_summary":"summary",
 "fact_ledger":[
  {"source_id":"exact id","fact_key":"stable semantic key",
   "target_relation":"canonical relation",
   "replacement_fact":"concise frozen fact",
   "fact_change_required":true,
   "intervention_policy":"factual_anchor_change"}
 ]}
"""


PROFILE_JUDGE_PROMPT = """Audit a proposed replacement-author fact ledger
before any C01 surface is generated. Code has already enforced exact coverage,
row policies, identity, and shared-key equality.

Decide whether all factual entries can simultaneously describe one coherent
fictional author. Check chronology, birthplace/upbringing/nationality, genres,
works, awards, quantities, influences, and repeated relations. Distinct works
or awards may legitimately have distinct values. Do not enforce punctuation,
anchor IDs, or wording.

Return JSON only:
{"coherent":true,"conflicting_source_ids":[],"reason":"brief evidence"}
If false, list only the source_ids participating in concrete contradictions.
"""


ROW_PROMPT = """Map exactly one immutable TOFU C11 row to its already-frozen
replacement-author ledger fact using only this row's local answer anchors.

Rules:
- return the exact source_id;
- copy ledger_fact_key and ledger_replacement_fact exactly from
  frozen_fact_ledger_entry; never invent or revise a fact;
- copy the canonical target_relation from the ledger entry;
- for a factual row, choose one (at most two) eligible answer-only group IDs
  whose replacements express a content token from ledger_replacement_fact;
- never select words shared with the question, verbs/adverbs, pronouns,
  quantifiers, uncertainty/polarity markers, or other scaffolding;
- typed replacements must preserve the anchor type and surface contract;
- identity/unavailable rows return empty target_group_ids and replacements;
- use frozen_shared_replacements exactly when provided;
- never rewrite complete questions or answers.

If validation_feedback and previous_candidate are present, return the complete
single-row object with only the mapping defect repaired. The frozen ledger fact
must remain unchanged.

Return JSON only:
{"source_id":"exact id","target_relation":"copied relation",
 "ledger_fact_key":"copied key",
 "ledger_replacement_fact":"copied fact",
 "target_group_ids":["local group id"],
 "anchor_replacements":[
  {"group_id":"same local id","replacement_value":"typed value"}
 ]}
"""


JUDGE_PROMPT = """Audit one completed 20-row TOFU counterfactual author block
against its already-approved frozen fact ledger. Code has enforced frozen C11,
coverage, typed local anchors, identity binding, and placebo cells.

For every row decide:
- target_relation_match: C11 and C01 ask the same relation;
- target_fact_changed: the required factual object changes, or the explicit
  identity/unavailable policy is correctly followed;
- profile_consistent: this row's C01 agrees with its own frozen ledger entry;
- natural_surface: C01 is fluent and grammatical.

profile_consistent is row-local fidelity to the approved ledger. Never reject
an otherwise correct row merely because another row is defective. A failed
verdict must identify only the minimal offending row set. Do not rewrite text
or enforce punctuation, exact IDs, or counts.

Return JSON only:
{"verdicts":[{"source_id":"exact id","target_relation_match":true,
 "target_fact_changed":true,"profile_consistent":true,
 "natural_surface":true,"reason":"brief row-specific evidence"}]}
"""


def policy_for_source(source: Mapping, target: str) -> str:
    if v52.source_fact_change_required(source, target):
        return "factual_anchor_change"
    if source["contract"]["response_mode"] == "unavailable":
        return "identity_binding_with_unavailability_preserved"
    return "identity_binding"


def profile_payload(
    block: Mapping,
    protected_authors: Sequence[str],
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = {
        "required_target_entity": block["target_entity"],
        "required_replacement_pronouns": block["replacement_pronouns"],
        "protected_authors": list(protected_authors),
        "source_rows": [
            {
                "source_id": source["source_id"],
                "question": source["question"],
                "answer": source["answer"],
                "response_contract": source["contract"],
                "fact_change_required": v52.source_fact_change_required(
                    source, block["target_entity"]
                ),
                "intervention_policy": policy_for_source(
                    source, block["target_entity"]
                ),
            }
            for source in block["sources"]
        ],
    }
    if feedback:
        payload["validation_feedback"] = feedback
    if previous is not None:
        payload["previous_candidate"] = previous
    return payload


def validate_profile(
    block: Mapping, generated: Mapping, protected_authors: Sequence[str]
) -> Dict:
    base = BASE_VALIDATE_PROFILE(block, generated, protected_authors)
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    raw_entries = generated.get("fact_ledger")
    entries = v52.v4.exact_rows(raw_entries, list(source_by_id), "fact_ledger")
    ledger = []
    shared_values: Dict[str, str] = {}
    for source_id in source_by_id:
        source = source_by_id[source_id]
        raw = entries[source_id]
        required = v52.source_fact_change_required(source, block["target_entity"])
        policy = policy_for_source(source, block["target_entity"])
        if raw.get("fact_change_required") is not required:
            raise ValueError(
                f"{source_id} fact_change_required must equal {required}"
            )
        if raw.get("intervention_policy") != policy:
            raise ValueError(
                f"{source_id} intervention_policy must equal {policy}"
            )
        relation = str(raw.get("target_relation", "")).strip()
        if not relation:
            raise ValueError(f"{source_id} target_relation must be non-empty")
        if required:
            fact_key = str(raw.get("fact_key", "")).strip()
            fact = str(raw.get("replacement_fact", "")).strip()
            if not fact_key or not fact:
                raise ValueError(
                    f"{source_id} factual ledger entry needs fact_key and replacement_fact"
                )
            if anchors.normalise(block["target_entity"]) in anchors.normalise(fact):
                raise ValueError(f"{source_id} replacement_fact leaks target author")
            if anchors.normalise(fact) == anchors.normalise(source["answer"]):
                raise ValueError(f"{source_id} replacement_fact copies C11 answer")
            normalised = anchors.normalise(fact)
            if fact_key in shared_values and shared_values[fact_key] != normalised:
                raise ValueError(
                    f"fact_key {fact_key!r} has conflicting replacement facts"
                )
            shared_values[fact_key] = normalised
        else:
            fact_key = "policy:unavailable" if "unavailability" in policy else "policy:identity"
            fact = "unavailable" if "unavailability" in policy else base["replacement_entity"]
        ledger.append({
            "source_id": source_id,
            "fact_key": fact_key,
            "target_relation": relation,
            "replacement_fact": fact,
            "fact_change_required": required,
            "intervention_policy": policy,
        })
    ledger_digest = v52.digest_json(ledger)
    result = {
        **base,
        "fact_ledger": ledger,
        "fact_ledger_by_source": {item["source_id"]: item for item in ledger},
        "ledger_digest": ledger_digest,
    }
    result["profile_digest"] = v52.digest_json({
        "target_entity": result["target_entity"],
        "replacement_entity": result["replacement_entity"],
        "replacement_pronouns": result["replacement_pronouns"],
        "profile_summary": result["profile_summary"],
        "ledger_digest": ledger_digest,
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "design_version": DESIGN_VERSION,
    })
    return result


def validate_profile_judgement(generated: Mapping, block: Mapping) -> Dict:
    coherent = generated.get("coherent")
    conflicts = generated.get("conflicting_source_ids", [])
    reason = str(generated.get("reason", "")).strip()
    source_ids = {source["source_id"] for source in block["sources"]}
    if coherent is not True:
        if not isinstance(conflicts, list) or any(
            not isinstance(item, str) for item in conflicts
        ):
            raise ValueError("ledger judge conflicting_source_ids must be a list")
        unknown = set(conflicts) - source_ids
        if unknown:
            raise ValueError(f"ledger judge returned unknown rows: {sorted(unknown)}")
        raise ValueError(
            "ledger semantic judge rejected profile"
            + (f" rows={','.join(sorted(conflicts))}" if conflicts else "")
            + f": {reason or 'missing reason'}"
        )
    if conflicts:
        raise ValueError("coherent ledger must have no conflicting_source_ids")
    if not reason:
        raise ValueError("ledger judge reason must be non-empty")
    return {"coherent": True, "conflicting_source_ids": [], "reason": reason}


def ledger_judge_payload(block: Mapping, profile: Mapping) -> Dict:
    return {
        "design_version": DESIGN_VERSION,
        "target_entity": profile["target_entity"],
        "replacement_entity": profile["replacement_entity"],
        "replacement_pronouns": profile["replacement_pronouns"],
        "profile_summary": profile["profile_summary"],
        "fact_ledger": profile["fact_ledger"],
        "source_rows": [
            {
                "source_id": source["source_id"],
                "question": source["question"],
                "answer": source["answer"],
            }
            for source in block["sources"]
        ],
    }


def generate_profile(
    generation_client,
    judge_client,
    args,
    block: Mapping,
    protected_authors: Sequence[str],
    state_dir: Path,
) -> Dict:
    path = v52.profile_path(state_dir, int(block["block_id"]))
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            profile = validate_profile(block, cached["profile"], protected_authors)
            verdict = validate_profile_judgement(
                cached["semantic_judge"], block
            )
            if (
                cached.get("design_version") == DESIGN_VERSION
                and cached.get("anchor_catalog_digest")
                == block["anchor_catalog"]["digest"]
                and cached.get("ledger_digest") == profile["ledger_digest"]
            ):
                profile["profile_attempt"] = int(cached.get("profile_attempt", 0))
                profile["profile_semantic_judge"] = verdict
                print(f"reuse_ledger_profile block={block['block_id']}", flush=True)
                return profile
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass
    feedback, previous, last_error = "", None, None
    for attempt in range(1, args.profile_retries + 1):
        print(
            f"start_ledger_profile block={block['block_id']} "
            f"attempt={attempt}/{args.profile_retries}", flush=True,
        )
        candidate = None
        try:
            candidate = v52.v2.request_json(
                generation_client,
                v52.request_args(args),
                PROFILE_PROMPT,
                profile_payload(block, protected_authors, feedback, previous),
                f"V5.3 block {block['block_id']} ledger profile",
            )
            profile = validate_profile(block, candidate, protected_authors)
            judgement = v52.v2.request_json(
                judge_client,
                v52.request_args(args, judge=True),
                PROFILE_JUDGE_PROMPT,
                ledger_judge_payload(block, profile),
                f"V5.3 block {block['block_id']} ledger audit",
            )
            verdict = validate_profile_judgement(judgement, block)
            profile["profile_attempt"] = attempt
            profile["profile_semantic_judge"] = verdict
            v52.write_json(path, {
                "design_version": DESIGN_VERSION,
                "anchor_catalog_digest": block["anchor_catalog"]["digest"],
                "ledger_digest": profile["ledger_digest"],
                "profile_attempt": attempt,
                "profile": {
                    key: value for key, value in profile.items()
                    if key not in {"fact_ledger_by_source", "profile_semantic_judge"}
                },
                "semantic_judge": verdict,
            })
            print(f"ledger_profile_ready block={block['block_id']}", flush=True)
            return profile
        except Exception as exc:
            last_error, feedback, previous = exc, str(exc), candidate
            v52.write_json(
                v52.block_directory(state_dir, int(block["block_id"]))
                / "profile_attempts" / f"attempt_{attempt:02d}.json",
                {"error": feedback, "candidate": previous},
            )
            print(
                f"ledger_profile_reject block={block['block_id']} "
                f"attempt={attempt}/{args.profile_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block['block_id']} ledger profile failed after "
        f"{args.profile_retries} attempts: {last_error}"
    ) from last_error


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    frozen_shared: Mapping[str, str] | None = None,
    accepted_fact_ledger: Sequence[Mapping] | None = None,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = BASE_ROW_PAYLOAD(
        block, source, profile, frozen_shared, accepted_fact_ledger,
        feedback, previous,
    )
    payload["author_profile"] = {
        "replacement_entity": profile["replacement_entity"],
        "replacement_pronouns": profile["replacement_pronouns"],
        "profile_summary": profile["profile_summary"],
        "ledger_digest": profile["ledger_digest"],
    }
    payload["frozen_fact_ledger_entry"] = profile[
        "fact_ledger_by_source"
    ][source["source_id"]]
    payload.pop("accepted_fact_ledger", None)
    return payload


def content_tokens(value: str) -> set[str]:
    stop = anchors.SCAFFOLD_WORDS | {
        "the", "a", "an", "and", "or", "of", "in", "on", "at", "to",
        "for", "with", "from", "by", "their", "his", "her", "its",
    }
    return {
        token for token in re.findall(r"[a-z0-9]+", anchors.normalise(value))
        if len(token) > 1 and token not in stop
    }


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
    frozen_shared: Mapping[str, str] | None = None,
) -> Dict:
    entry = profile["fact_ledger_by_source"][source["source_id"]]
    if generated.get("ledger_fact_key") != entry["fact_key"]:
        raise ValueError("ledger_fact_key must exactly copy frozen ledger")
    if generated.get("ledger_replacement_fact") != entry["replacement_fact"]:
        raise ValueError("ledger_replacement_fact must exactly copy frozen ledger")
    if str(generated.get("target_relation", "")).strip() != entry["target_relation"]:
        raise ValueError("target_relation must exactly copy frozen ledger")
    validated = BASE_VALIDATE_ROW(
        block, source, profile, generated, frozen_shared
    )
    if entry["fact_change_required"]:
        replacement_tokens = content_tokens(" ".join(
            validated["anchor_replacements"].values()
        ))
        fact_tokens = content_tokens(entry["replacement_fact"])
        if not replacement_tokens.intersection(fact_tokens):
            raise ValueError(
                "anchor replacements must express a content token from "
                "ledger_replacement_fact"
            )
    validated.update({
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "ledger_digest": profile["ledger_digest"],
    })
    return validated


def candidate_for_prompt(candidate: Mapping) -> Dict:
    result = BASE_CANDIDATE_FOR_PROMPT(candidate)
    result.update({
        "ledger_fact_key": candidate["ledger_fact_key"],
        "ledger_replacement_fact": candidate["ledger_replacement_fact"],
    })
    return result


def load_cached_row(
    path: Path,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
) -> Dict | None:
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("design_version") != DESIGN_VERSION:
            return None
        if cached.get("profile_digest") != profile["profile_digest"]:
            return None
        if cached.get("ledger_digest") != profile["ledger_digest"]:
            return None
        validated = validate_row_candidate(
            block, source, profile, cached["candidate"]
        )
        validated["mapping_attempt"] = int(cached.get("mapping_attempt", 0))
        validated["repair_generation"] = int(
            cached.get("repair_generation", 0)
        )
        return validated
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def generate_row(
    client,
    args,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    state_dir: Path,
    feedback: str = "",
    previous: Mapping | None = None,
    frozen_shared: Mapping[str, str] | None = None,
    accepted_fact_ledger: Sequence[Mapping] | None = None,
    force: bool = False,
) -> Dict:
    block_id, source_id = int(block["block_id"]), source["source_id"]
    path = v52.row_path(state_dir, block_id, source_id)
    if not force:
        cached = load_cached_row(path, block, source, profile)
        if cached is not None:
            print(f"reuse_row block={block_id} source={source_id}", flush=True)
            return cached
    last_error = None
    for attempt in range(1, args.row_retries + 1):
        print(
            f"start_row block={block_id} source={source_id} "
            f"attempt={attempt}/{args.row_retries}", flush=True,
        )
        candidate = None
        try:
            candidate = v52.v2.request_json(
                client,
                v52.request_args(args),
                ROW_PROMPT,
                row_payload(
                    block,
                    source,
                    profile,
                    frozen_shared,
                    accepted_fact_ledger,
                    feedback,
                    previous,
                ),
                f"V5.3 block {block_id} row {source_id}",
            )
            validated = validate_row_candidate(
                block, source, profile, candidate, frozen_shared
            )
            validated["mapping_attempt"] = attempt
            validated["repair_generation"] = 1 if force else 0
            v52.write_json(path, {
                "design_version": DESIGN_VERSION,
                "profile_digest": profile["profile_digest"],
                "ledger_digest": profile["ledger_digest"],
                "mapping_attempt": attempt,
                "repair_generation": validated["repair_generation"],
                "candidate": candidate_for_prompt(validated),
            })
            print(
                f"row_ready block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries}", flush=True,
            )
            return validated
        except Exception as exc:
            last_error, feedback, previous = exc, str(exc), candidate
            v52.write_json(
                v52.block_directory(state_dir, block_id)
                / "row_attempts" / f"{source_id}.attempt_{attempt:02d}.json",
                {"error": feedback, "candidate": previous},
            )
            print(
                f"row_reject block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block_id} row {source_id} failed after "
        f"{args.row_retries} attempts: {last_error}"
    ) from last_error


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    plan = BASE_MATERIALIZE_PLAN(block, profile, candidates, seed)
    plan.update({
        "fact_ledger": profile["fact_ledger"],
        "fact_ledger_by_source": profile["fact_ledger_by_source"],
        "ledger_digest": profile["ledger_digest"],
        "profile_semantic_judge": profile["profile_semantic_judge"],
    })
    for source_id, row_plan in plan["row_plans"].items():
        entry = profile["fact_ledger_by_source"][source_id]
        row_plan.update({
            "replacement_fact": entry["replacement_fact"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_digest": profile["ledger_digest"],
        })
    return plan


def judge_payload(block: Mapping, plan: Mapping) -> Dict:
    payload = BASE_JUDGE_PAYLOAD(block, plan)
    payload["design_version"] = DESIGN_VERSION
    payload["ledger_digest"] = plan["ledger_digest"]
    payload["ledger_semantic_judge"] = plan["profile_semantic_judge"]
    by_source = plan["fact_ledger_by_source"]
    for row in payload["rows"]:
        row["frozen_fact_ledger_entry"] = by_source[row["source_id"]]
    return payload


def assemble_block(
    block: Mapping, plan: Mapping, verdicts: Mapping[str, Mapping], args
) -> Dict:
    assembled = BASE_ASSEMBLE_BLOCK(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-ledger-v5.3-block-{int(block['block_id']):02d}:"
        f"seed-{args.seed}"
    )
    assembled.update({
        "design_version": DESIGN_VERSION,
        "profile_id": profile_id,
        "fact_ledger": plan["fact_ledger"],
        "ledger_digest": plan["ledger_digest"],
        "profile_semantic_judge": plan["profile_semantic_judge"],
    })
    assembled["author_plan"].update({
        "frozen_fact_ledger": plan["fact_ledger"],
        "ledger_digest": plan["ledger_digest"],
        "semantic_judge": plan["profile_semantic_judge"],
    })
    for record in assembled["records"]:
        source_id = record["source_id"]
        entry = plan["fact_ledger_by_source"][source_id]
        record.update({
            "design_version": DESIGN_VERSION,
            "profile_id": profile_id,
            "fact_ledger_entry": entry,
        })
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
            "ledger_digest": plan["ledger_digest"],
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{source_id}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


def load_valid_block(path: Path, block: Mapping) -> Dict | None:
    data = BASE_LOAD_VALID_BLOCK(path, block)
    if data is None:
        return None
    ledger = data.get("fact_ledger")
    if not isinstance(ledger, list) or len(ledger) != len(block["sources"]):
        return None
    if data.get("ledger_digest") != v52.digest_json(ledger):
        return None
    if data.get("profile_semantic_judge", {}).get("coherent") is not True:
        return None
    if any(
        record.get("generation", {}).get("surface_renderer") != SURFACE_RENDERER
        or record.get("generation", {}).get("mapping_scope") != MAPPING_SCOPE
        or record.get("generation", {}).get("ledger_digest")
        != data["ledger_digest"]
        for record in data["records"]
    ):
        return None
    return data


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
    profile = generate_profile(
        generation_client, judge_client, args, block, protected_authors, state_dir
    )
    candidates = v52.generate_rows(
        generation_client, args, block, profile, state_dir
    )
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    last_failures: Dict[str, str] = {}
    for judge_round in range(1, args.judge_rounds + 1):
        plan = materialize_plan(block, profile, candidates, args.seed)
        judgement = v52.v2.request_json(
            judge_client,
            v52.request_args(args, judge=True),
            JUDGE_PROMPT,
            judge_payload(block, plan),
            f"V5.3 block {block_id} row fidelity audit round {judge_round}",
        )
        verdicts, failures = v52.parse_judgement(judgement, block, plan)
        v52.write_json(
            v52.block_directory(state_dir, block_id)
            / f"judge_round_{judge_round:02d}.json",
            {
                "design_version": DESIGN_VERSION,
                "ledger_digest": profile["ledger_digest"],
                "verdicts": verdicts,
                "failures": failures,
            },
        )
        if not failures:
            plan["judge_round"] = judge_round
            assembled = assemble_block(block, plan, verdicts, args)
            print(
                f"block_ready block={block_id} judge_round={judge_round} "
                f"conflicts={len(plan['reconciliation_conflicts'])}", flush=True,
            )
            return assembled
        last_failures = failures
        print(
            f"judge_reject block={block_id} round={judge_round}/"
            f"{args.judge_rounds} rows={','.join(sorted(failures))}", flush=True,
        )
        if judge_round == args.judge_rounds:
            break
        rejected = set(failures)
        frozen = v52.accepted_shared_replacements(candidates, rejected)
        repaired = {}
        with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    v52.generate_row,
                    generation_client,
                    args,
                    block,
                    source_by_id[source_id],
                    profile,
                    state_dir,
                    failures[source_id],
                    candidate_for_prompt(candidates[source_id]),
                    {
                        group_id: value
                        for group_id, value in frozen.items()
                        if group_id in v52.local_group_ids(block, source_id)
                    },
                    None,
                    True,
                ): source_id
                for source_id in rejected
            }
            for future in as_completed(futures):
                source_id = futures[future]
                repaired[source_id] = future.result()
        candidates.update(repaired)
    raise RuntimeError(
        f"block {block_id} row-fidelity judge exhausted {args.judge_rounds} rounds: "
        + " | ".join(
            f"{source_id}: {reason}"
            for source_id, reason in sorted(last_failures.items())
        )
    )


def configure_v52_globals() -> None:
    v52.DESIGN_VERSION = DESIGN_VERSION
    v52.SURFACE_RENDERER = SURFACE_RENDERER
    v52.PROFILE_PROMPT = PROFILE_PROMPT
    v52.ROW_PROMPT = ROW_PROMPT
    v52.JUDGE_PROMPT = JUDGE_PROMPT
    v52.row_payload = row_payload
    v52.validate_row_candidate = validate_row_candidate
    v52.candidate_for_prompt = candidate_for_prompt
    v52.generate_row = generate_row


def main() -> None:
    configure_v52_globals()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget05_perturbed")
    parser.add_argument("--manifest", type=Path, default=v52.v2.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--block-ids", default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-api-key-env")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=12000)
    parser.add_argument("--block-concurrency", type=int, default=1)
    parser.add_argument("--row-concurrency", type=int, default=5)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--profile-retries", type=int, default=6)
    parser.add_argument("--row-retries", type=int, default=4)
    parser.add_argument("--judge-rounds", type=int, default=3)
    parser.add_argument(
        "--json-mode", choices=("auto", "required", "prompt"), default="auto"
    )
    args = parser.parse_args()
    args.judge_model = args.judge_model or args.model
    args.judge_base_url = args.judge_base_url or args.base_url
    args.judge_api_key_env = args.judge_api_key_env or args.api_key_env
    api_key = os.environ.get(args.api_key_env)
    judge_key = os.environ.get(args.judge_api_key_env)
    if not api_key or not judge_key:
        raise SystemExit("Set generation and judge API keys before V5.3 generation")
    from openai import OpenAI

    manifest = v52.v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(f"manifest split={manifest.get('split')} != {args.split}")
    all_blocks = [
        v52.with_contracts_and_anchors(block)
        for block in v52.v2.group_author_blocks(
            v52.v2.load_sources(args.split), manifest
        )
    ]
    all_ids = [int(block["block_id"]) for block in all_blocks]
    try:
        selected_ids = v52.parse_block_ids(args.block_ids, all_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    blocks = [
        block for block in all_blocks if int(block["block_id"]) in selected_ids
    ]
    protected = [block["target_entity"] for block in all_blocks]
    state_dir = args.state_dir or Path(str(args.output) + ".blocks")
    profiles_output = args.profiles_output or Path(str(args.output) + ".profiles.json")
    state_dir.mkdir(parents=True, exist_ok=True)

    completed, pending = {}, []
    for block in blocks:
        path = state_dir / f"block_{int(block['block_id']):02d}.json"
        cached = load_valid_block(path, block)
        if cached is None:
            pending.append(block)
        else:
            completed[int(block["block_id"])] = cached
            print(f"reuse block={block['block_id']} author={block['target_entity']}", flush=True)

    failures = []
    if pending:
        generation_client = OpenAI(api_key=api_key, base_url=args.base_url)
        judge_client = OpenAI(api_key=judge_key, base_url=args.judge_base_url)
        with ThreadPoolExecutor(max_workers=max(args.block_concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    generate_block,
                    generation_client,
                    judge_client,
                    args,
                    block,
                    protected,
                    state_dir,
                ): block
                for block in pending
            }
            for future in as_completed(futures):
                block = futures[future]
                block_id = int(block["block_id"])
                try:
                    result = future.result()
                    v52.write_json(state_dir / f"block_{block_id:02d}.json", result)
                    completed[block_id] = result
                    print(
                        f"valid_block={len(completed)}/{len(blocks)} "
                        f"block={block_id} author={block['target_entity']}", flush=True,
                    )
                except Exception as exc:
                    failures.append((block_id, block["target_entity"], str(exc)))

    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        raise SystemExit("V5.3 incomplete; valid ledgers, rows, and blocks were retained")

    ordered = [completed[block_id] for block_id in selected_ids]
    records = [record for block in ordered for record in block["records"]]
    expected = len(blocks) * int(manifest["block_size"])
    if len(records) != expected:
        raise SystemExit(f"record count {len(records)} != {expected}")
    ciru.write_ciru_jsonl(args.output, records)
    v52.write_json(profiles_output, {
        "design_version": DESIGN_VERSION,
        "split": args.split,
        "seed": args.seed,
        "selected_block_ids": selected_ids,
        "manifest": str(args.manifest.resolve()),
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "profiles": [
            {
                key: block[key]
                for key in (
                    "block_id", "profile_id", "profile_digest",
                    "target_entity", "replacement_entity",
                    "replacement_pronouns", "anchor_catalog_digest",
                    "ledger_digest", "fact_ledger", "author_plan",
                    "placebo_plan", "reconciliation_conflicts",
                    "profile_semantic_judge", "profile_attempt", "judge_round",
                )
            }
            for block in ordered
        ],
    })
    print(
        f"design={DESIGN_VERSION} blocks={len(blocks)} rows={len(records)} "
        f"output={args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
