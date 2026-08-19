#!/usr/bin/env python3
"""Generate TOFU author factorial data with row-local anchor mapping.

V5.2 preserves immutable C11, the V5 anchor catalog, deterministic surfaces,
and professional placebo cells.  It replaces the fragile 20-row planner with
one block profile plus independently resumable row mappings.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
V2_PATH = SCRIPT_DIR / "generate_tofu_author_factorial.py"
V4_PATH = SCRIPT_DIR / "generate_tofu_author_typed_v4.py"
V5_PATH = SCRIPT_DIR / "generate_tofu_author_anchor_v5.py"
ANCHOR_PATH = ROOT / "ULD/uld/data/tofu_anchor_v5.py"
CIRU_PATH = ROOT / "ULD/uld/data/ciru.py"
DESIGN_VERSION = "tofu-author-rowlocal-v5.2"
SURFACE_RENDERER = "deterministic-rowlocal-anchor-v5.2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_module("tofu_rowlocal_v52_v2", V2_PATH)
v4 = load_module("tofu_rowlocal_v52_v4", V4_PATH)
v5 = load_module("tofu_rowlocal_v52_v5", V5_PATH)
anchors = load_module("tofu_rowlocal_v52_anchors", ANCHOR_PATH)
ciru = load_module("tofu_rowlocal_v52_ciru", CIRU_PATH)


PROFILE_PROMPT = """Create one coherent fictional replacement author profile
for a 20-row TOFU author block. C11 is immutable. This stage chooses identity
and a short profile theme only; it must not return source edits, group IDs, or
row plans. Preserve the required pronoun class. Do not reuse a protected
author. The profile summary should describe a plausible professional author
whose detailed facts can be filled independently row by row.

If validation_feedback and previous_candidate are present, return the complete
object with the reported defect repaired.

Return JSON only:
{"target_entity":"required target", "replacement_entity":"new author",
 "replacement_pronouns":"required class",
 "profile_summary":"short coherent author profile"}
"""


ROW_PROMPT = """Map exactly one immutable TOFU C11 row to a counterfactual
replacement-author fact. You see only this row's local frozen anchors.

Rules:
- return the exact source_id;
- label the relation asked by C11 without changing it;
- if fact_change_required=true, choose one (at most two) supplied ANSWER anchor
  groups that directly express the answer's factual object;
- propose a typed replacement for every selected group: year->year,
  number->number, date->complete date, title->title, place->place, and so on;
- do not select generic question/template words, polarity markers, uncertainty
  markers, or the author identity;
- if fact_change_required=false, return empty target_group_ids and empty
  anchor_replacements; code will bind replacement identity deterministically;
- use frozen_shared_replacements exactly when selecting one of those groups;
- when accepted_fact_ledger is present during repair, keep the replacement
  author coherent with those already accepted relation/value assignments;
- never quote or rewrite complete questions or answers.

If validation_feedback and previous_candidate are present, return the complete
single-row object with the defect repaired.

Return JSON only:
{"source_id":"exact id", "target_relation":"canonical relation",
 "target_group_ids":["local group id"],
 "anchor_replacements":[
   {"group_id":"same local id", "replacement_value":"typed value"}
 ]}
"""


JUDGE_PROMPT = """Audit one completed 20-row TOFU counterfactual author block.
Code has already enforced frozen C11, local anchor IDs, typed replacements,
response contracts, identity binding, coverage, and professional placebo
cells. Do not rewrite text and do not enforce punctuation or IDs.

For every row decide:
- target_relation_match: C11 and C01 ask the same relation;
- target_fact_changed: the factual object changes when required; identity and
  unavailable rows correctly follow their explicit policy;
- profile_consistent: all C01 rows describe one coherent replacement author;
- natural_surface: C01 remains fluent and grammatical.

Return JSON only:
{"verdicts":[{"source_id":"exact id", "target_relation_match":true,
 "target_fact_changed":true, "profile_consistent":true,
 "natural_surface":true, "reason":"brief evidence"}]}
"""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def request_args(args, *, judge: bool = False):
    result = copy.copy(args)
    result.model = args.judge_model if judge else args.model
    result.temperature = args.judge_temperature if judge else args.temperature
    result.retries = args.request_retries
    return result


def with_contracts_and_anchors(block: Mapping) -> Dict:
    return v5.with_contracts_and_anchors(block)


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
    target = block["target_entity"]
    if anchors.normalise(generated.get("target_entity", "")) != anchors.normalise(target):
        raise ValueError(f"target_entity must equal {target!r}")
    replacement = generated.get("replacement_entity")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("replacement_entity must be non-empty")
    replacement = replacement.strip()
    if anchors.normalise(target) in anchors.normalise(replacement):
        raise ValueError("replacement_entity contains the target name")
    if anchors.normalise(replacement) in {
        anchors.normalise(author) for author in protected_authors
    }:
        raise ValueError("replacement_entity collides with a protected author")
    expected_pronouns = block["replacement_pronouns"]
    if anchors.normalise(generated.get("replacement_pronouns", "")) != anchors.normalise(
        expected_pronouns
    ):
        raise ValueError(f"replacement_pronouns must equal {expected_pronouns!r}")
    summary = generated.get("profile_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("profile_summary must be non-empty")
    result = {
        "target_entity": target,
        "replacement_entity": replacement,
        "replacement_pronouns": expected_pronouns,
        "profile_summary": summary.strip(),
    }
    result["profile_digest"] = digest_json({
        **result,
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "design_version": DESIGN_VERSION,
    })
    return result


def source_fact_change_required(source: Mapping, target: str) -> bool:
    return (
        source["contract"]["response_mode"] != "unavailable"
        and not anchors.identity_relation(source, target, "")
    )


def local_group_ids(block: Mapping, source_id: str) -> set[str]:
    return {
        item["group_id"]
        for item in block["anchor_catalog"]["occurrences"]
        if item["source_id"] == source_id
    }


def answer_group_ids(block: Mapping, source_id: str) -> set[str]:
    return {
        item["group_id"]
        for item in block["anchor_catalog"]["occurrences"]
        if item["source_id"] == source_id and item["field"] == "answer"
    }


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    frozen_shared: Mapping[str, str] | None = None,
    accepted_fact_ledger: Sequence[Mapping] | None = None,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    local = anchors.catalog_for_prompt(block["anchor_catalog"], source["source_id"])
    payload = {
        "author_profile": profile,
        "source_id": source["source_id"],
        "C11": {"question": source["question"], "answer": source["answer"]},
        "response_contract": source["contract"],
        "fact_change_required": source_fact_change_required(
            source, block["target_entity"]
        ),
        "available_anchor_groups": local,
        "frozen_shared_replacements": dict(frozen_shared or {}),
        "accepted_fact_ledger": list(accepted_fact_ledger or []),
    }
    if feedback:
        payload["validation_feedback"] = feedback
    if previous is not None:
        payload["previous_candidate"] = previous
    return payload


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
    frozen_shared: Mapping[str, str] | None = None,
) -> Dict:
    source_id = source["source_id"]
    if generated.get("source_id") != source_id:
        raise ValueError(f"source_id must equal {source_id}")
    relation = generated.get("target_relation")
    if not isinstance(relation, str) or not relation.strip():
        raise ValueError("target_relation must be non-empty")
    declared = generated.get("target_group_ids", [])
    if not isinstance(declared, list) or any(not isinstance(x, str) for x in declared):
        raise ValueError("target_group_ids must be a string list")
    if len(declared) != len(set(declared)):
        raise ValueError("target_group_ids contain duplicates")
    if len(declared) > 2:
        raise ValueError("target_group_ids must select at most two local facts")
    available = local_group_ids(block, source_id)
    unknown = set(declared) - available
    if unknown:
        raise ValueError(f"target_group_ids are not local to this row: {sorted(unknown)}")

    replacements = anchors.validate_replacement_map(
        block["anchor_catalog"], generated.get("anchor_replacements", [])
    )
    if set(replacements) != set(declared):
        raise ValueError(
            "anchor_replacements must exactly cover target_group_ids"
        )
    required = source_fact_change_required(source, block["target_entity"])
    if required:
        if not declared:
            raise ValueError("factual row must select at least one answer anchor")
        non_answer = set(declared) - answer_group_ids(block, source_id)
        if non_answer:
            raise ValueError(
                f"factual target groups must occur in the answer: {sorted(non_answer)}"
            )
    elif declared or replacements:
        raise ValueError("identity/unavailable row must not select factual anchors")

    frozen_shared = dict(frozen_shared or {})
    for group_id, value in replacements.items():
        if group_id in frozen_shared and anchors.normalise(value) != anchors.normalise(
            frozen_shared[group_id]
        ):
            raise ValueError(
                f"{group_id} conflicts with frozen shared replacement "
                f"{frozen_shared[group_id]!r}"
            )

    # Validate the row-local surface before checkpointing the mapping.
    anchors.render_row(
        source,
        block["target_entity"],
        profile["replacement_entity"],
        relation.strip(),
        block["anchor_catalog"],
        replacements,
        v4.contracts,
    )
    return {
        "source_id": source_id,
        "target_relation": relation.strip(),
        "target_group_ids": list(declared),
        "anchor_replacements": replacements,
        "fact_change_required": required,
    }


def reconcile_replacements(candidates: Mapping[str, Mapping]) -> tuple[Dict, list[Dict]]:
    proposals: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source_id, candidate in sorted(candidates.items()):
        for group_id, value in candidate["anchor_replacements"].items():
            proposals[group_id].append((source_id, value))
    canonical, conflicts = {}, []
    for group_id, entries in sorted(proposals.items()):
        source_id, value = sorted(entries)[0]
        canonical[group_id] = value
        alternatives = sorted({item[1] for item in entries}, key=anchors.normalise)
        if len({anchors.normalise(item) for item in alternatives}) > 1:
            conflicts.append({
                "group_id": group_id,
                "winner_source_id": source_id,
                "canonical_value": value,
                "proposals": [
                    {"source_id": item_source, "value": item_value}
                    for item_source, item_value in sorted(entries)
                ],
            })
    return canonical, conflicts


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    canonical, conflicts = reconcile_replacements(candidates)
    row_plans, cells, errors = {}, {}, []
    for query_index, source in enumerate(block["sources"]):
        source_id = source["source_id"]
        candidate = candidates[source_id]
        local_map = {
            group_id: canonical[group_id]
            for group_id in candidate["target_group_ids"]
        }
        try:
            rendered = anchors.render_row(
                source,
                block["target_entity"],
                profile["replacement_entity"],
                candidate["target_relation"],
                block["anchor_catalog"],
                local_map,
                v4.contracts,
            )
            applied = rendered["question_edits"] + rendered["answer_edits"]
            applied_ids = {item["group_id"] for item in applied}
            if candidate["fact_change_required"] and not set(
                candidate["target_group_ids"]
            ).issubset(applied_ids):
                raise ValueError("selected target groups were not rendered")
            target_edits = [
                item
                for item in rendered["answer_edits"]
                if item["group_id"] in candidate["target_group_ids"]
            ]
            replacement_fact = (
                max((item["new"] for item in target_edits), key=len)
                if target_edits else profile["replacement_entity"]
            )
            placebo = v4.contracts.render_professional_placebo(
                source["contract"],
                block["target_entity"],
                profile["replacement_entity"],
                query_index,
                assignment_flip=(
                    (int(block["block_id"]) + query_index + seed) % 2 == 1
                ),
            )
            for name in ("C10", "C00"):
                contract_errors = v4.contracts.contract_errors(
                    placebo[name], source["contract"]
                )
                if contract_errors:
                    raise ValueError(f"{name}: " + "; ".join(contract_errors))
            required = candidate["fact_change_required"]
            row_plans[source_id] = {
                "source_id": source_id,
                "target_relation": candidate["target_relation"],
                "target_group_ids": list(candidate["target_group_ids"]),
                "replacement_fact": replacement_fact,
                "fact_change_required": required,
                "intervention_policy": (
                    "factual_anchor_change"
                    if required
                    else "identity_binding_with_unavailability_preserved"
                    if source["contract"]["response_mode"] == "unavailable"
                    else "identity_binding"
                ),
                "question_edits": rendered["question_edits"],
                "answer_edits": rendered["answer_edits"],
                "mapping_attempt": int(candidate.get("mapping_attempt", 0)),
                "repair_generation": int(candidate.get("repair_generation", 0)),
            }
            cells[source_id] = {
                "C01": rendered["cell"],
                "C10": placebo["C10"],
                "C00": placebo["C00"],
                "placebo_relation": placebo["placebo_relation"],
                "placebo_target_value": placebo["target_value"],
                "placebo_replacement_value": placebo["replacement_value"],
                "placebo_rationale": placebo["placebo_rationale"],
            }
        except ValueError as exc:
            errors.append(f"{source_id}: {exc}")
    if errors:
        raise ValueError(" | ".join(errors))
    return {
        **profile,
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "anchor_replacements": canonical,
        "reconciliation_conflicts": conflicts,
        "row_plans": row_plans,
        "cells": cells,
    }


def judge_payload(block: Mapping, plan: Mapping) -> Dict:
    payload = v5.judge_payload(block, plan)
    payload["design_version"] = DESIGN_VERSION
    payload["reconciliation_conflicts"] = plan["reconciliation_conflicts"]
    return payload


def parse_judgement(
    generated: Mapping, block: Mapping, plan: Mapping
) -> tuple[Dict[str, Dict], Dict[str, str]]:
    source_ids = [source["source_id"] for source in block["sources"]]
    verdicts = v4.exact_rows(generated.get("verdicts"), source_ids, "verdicts")
    failures: Dict[str, str] = {}
    for source_id, raw in verdicts.items():
        verdict = dict(raw)
        overrides = []
        if not plan["row_plans"][source_id]["fact_change_required"]:
            verdict["raw_target_fact_changed"] = verdict.get("target_fact_changed")
            verdict["target_fact_changed"] = True
            overrides.append("target_fact_changed")
        if overrides:
            verdict["deterministic_overrides"] = overrides
        failed_fields = [
            field for field in v4.JUDGE_FIELDS if verdict.get(field) is not True
        ]
        reason = str(verdict.get("reason", "")).strip()
        if not reason:
            failed_fields.append("reason")
        if failed_fields:
            failures[source_id] = (
                f"semantic fields {','.join(failed_fields)}: {reason or 'missing reason'}"
            )
        verdicts[source_id] = verdict
    return verdicts, failures


def assemble_block(
    block: Mapping, plan: Mapping, verdicts: Mapping[str, Mapping], args
) -> Dict:
    assembled = v5.assemble_block(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-rowlocal-v5.2-block-{int(block['block_id']):02d}:"
        f"seed-{args.seed}"
    )
    assembled.update({
        "design_version": DESIGN_VERSION,
        "profile_id": profile_id,
        "profile_digest": plan["profile_digest"],
        "reconciliation_conflicts": plan["reconciliation_conflicts"],
        "profile_attempt": int(plan.get("profile_attempt", 0)),
        "judge_round": int(plan.get("judge_round", 0)),
    })
    for record in assembled["records"]:
        record["design_version"] = DESIGN_VERSION
        record["profile_id"] = profile_id
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": "single-row-local-anchors",
            "profile_attempt": int(plan.get("profile_attempt", 0)),
            "mapping_attempt": int(
                plan["row_plans"][record["source_id"]].get("mapping_attempt", 0)
            ),
            "repair_generation": int(
                plan["row_plans"][record["source_id"]].get(
                    "repair_generation", 0
                )
            ),
            "judge_round": int(plan.get("judge_round", 0)),
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{record['source_id']}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


def block_directory(state_dir: Path, block_id: int) -> Path:
    return state_dir / f"block_{block_id:02d}"


def profile_path(state_dir: Path, block_id: int) -> Path:
    return block_directory(state_dir, block_id) / "profile.json"


def row_path(state_dir: Path, block_id: int, source_id: str) -> Path:
    return block_directory(state_dir, block_id) / "rows" / f"{source_id}.json"


def generate_profile(
    client,
    args,
    block: Mapping,
    protected_authors: Sequence[str],
    state_dir: Path,
) -> Dict:
    path = profile_path(state_dir, int(block["block_id"]))
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            profile = validate_profile(block, cached["profile"], protected_authors)
            if cached.get("anchor_catalog_digest") == block["anchor_catalog"]["digest"]:
                profile["profile_attempt"] = int(cached.get("profile_attempt", 0))
                print(f"reuse_profile block={block['block_id']}", flush=True)
                return profile
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass
    feedback, previous, last_error = "", None, None
    for attempt in range(1, args.profile_retries + 1):
        print(
            f"start_profile block={block['block_id']} "
            f"attempt={attempt}/{args.profile_retries}", flush=True,
        )
        candidate = None
        try:
            candidate = v2.request_json(
                client,
                request_args(args),
                PROFILE_PROMPT,
                profile_payload(block, protected_authors, feedback, previous),
                f"V5.2 block {block['block_id']} profile",
            )
            profile = validate_profile(block, candidate, protected_authors)
            profile["profile_attempt"] = attempt
            write_json(path, {
                "design_version": DESIGN_VERSION,
                "anchor_catalog_digest": block["anchor_catalog"]["digest"],
                "profile_attempt": attempt,
                "profile": profile,
            })
            print(f"profile_ready block={block['block_id']}", flush=True)
            return profile
        except Exception as exc:
            last_error, feedback, previous = exc, str(exc), candidate
            attempt_path = block_directory(
                state_dir, int(block["block_id"])
            ) / f"profile.attempt_{attempt:02d}.json"
            write_json(attempt_path, {"error": feedback, "candidate": previous})
            print(
                f"profile_reject block={block['block_id']} "
                f"attempt={attempt}/{args.profile_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block['block_id']} profile failed after "
        f"{args.profile_retries} attempts: {last_error}"
    ) from last_error


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
        validated = validate_row_candidate(
            block, source, profile, cached["candidate"]
        )
        validated["mapping_attempt"] = int(cached.get("mapping_attempt", 0))
        validated["repair_generation"] = int(cached.get("repair_generation", 0))
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
    path = row_path(state_dir, block_id, source_id)
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
            candidate = v2.request_json(
                client,
                request_args(args),
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
                f"V5.2 block {block_id} row {source_id}",
            )
            validated = validate_row_candidate(
                block, source, profile, candidate, frozen_shared
            )
            validated["mapping_attempt"] = attempt
            validated["repair_generation"] = 1 if force else 0
            write_json(path, {
                "design_version": DESIGN_VERSION,
                "profile_digest": profile["profile_digest"],
                "mapping_attempt": attempt,
                "repair_generation": validated["repair_generation"],
                "candidate": {
                    "source_id": validated["source_id"],
                    "target_relation": validated["target_relation"],
                    "target_group_ids": validated["target_group_ids"],
                    "anchor_replacements": [
                        {"group_id": group_id, "replacement_value": value}
                        for group_id, value in validated["anchor_replacements"].items()
                    ],
                },
            })
            print(
                f"row_ready block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries}", flush=True,
            )
            return validated
        except Exception as exc:
            last_error, feedback, previous = exc, str(exc), candidate
            attempt_path = (
                block_directory(state_dir, block_id)
                / "row_attempts"
                / f"{source_id}.attempt_{attempt:02d}.json"
            )
            write_json(attempt_path, {"error": feedback, "candidate": previous})
            print(
                f"row_reject block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block_id} row {source_id} failed after "
        f"{args.row_retries} attempts: {last_error}"
    ) from last_error


def generate_rows(
    client,
    args,
    block: Mapping,
    profile: Mapping,
    state_dir: Path,
) -> Dict[str, Dict]:
    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
        futures = {
            executor.submit(
                generate_row, client, args, block, source, profile, state_dir
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


def accepted_shared_replacements(
    candidates: Mapping[str, Mapping], rejected: set[str]
) -> Dict[str, str]:
    accepted = {
        source_id: candidate
        for source_id, candidate in candidates.items()
        if source_id not in rejected
    }
    return reconcile_replacements(accepted)[0]


def accepted_fact_ledger(
    candidates: Mapping[str, Mapping], rejected: set[str]
) -> list[Dict]:
    return [
        {
            "source_id": source_id,
            "target_relation": candidate["target_relation"],
            "replacement_values": [
                {"group_id": group_id, "value": value}
                for group_id, value in sorted(
                    candidate["anchor_replacements"].items()
                )
            ],
        }
        for source_id, candidate in sorted(candidates.items())
        if source_id not in rejected
    ]


def candidate_for_prompt(candidate: Mapping) -> Dict:
    return {
        "source_id": candidate["source_id"],
        "target_relation": candidate["target_relation"],
        "target_group_ids": list(candidate["target_group_ids"]),
        "anchor_replacements": [
            {"group_id": group_id, "replacement_value": value}
            for group_id, value in candidate["anchor_replacements"].items()
        ],
    }


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
        generation_client, args, block, protected_authors, state_dir
    )
    candidates = generate_rows(
        generation_client, args, block, profile, state_dir
    )
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    last_failures: Dict[str, str] = {}
    for judge_round in range(1, args.judge_rounds + 1):
        plan = materialize_plan(block, profile, candidates, args.seed)
        judgement = v2.request_json(
            judge_client,
            request_args(args, judge=True),
            JUDGE_PROMPT,
            judge_payload(block, plan),
            f"V5.2 block {block_id} judge round {judge_round}",
        )
        verdicts, failures = parse_judgement(judgement, block, plan)
        write_json(
            block_directory(state_dir, block_id)
            / f"judge_round_{judge_round:02d}.json",
            {"verdicts": verdicts, "failures": failures},
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
        frozen = accepted_shared_replacements(candidates, rejected)
        fact_ledger = accepted_fact_ledger(candidates, rejected)
        repaired = {}
        with ThreadPoolExecutor(max_workers=max(args.row_concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    generate_row,
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
                        if group_id in local_group_ids(block, source_id)
                    },
                    fact_ledger,
                    True,
                ): source_id
                for source_id in rejected
            }
            for future in as_completed(futures):
                source_id = futures[future]
                repaired[source_id] = future.result()
        candidates.update(repaired)
    raise RuntimeError(
        f"block {block_id} semantic judge exhausted {args.judge_rounds} rounds: "
        + " | ".join(
            f"{source_id}: {reason}"
            for source_id, reason in sorted(last_failures.items())
        )
    )


def load_valid_block(path: Path, block: Mapping) -> Dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = {source["source_id"] for source in block["sources"]}
        if data.get("design_version") != DESIGN_VERSION:
            return None
        if data.get("anchor_catalog_digest") != block["anchor_catalog"]["digest"]:
            return None
        records = data.get("records")
        if not isinstance(records, list) or len(records) != len(expected):
            return None
        if {record.get("source_id") for record in records} != expected:
            return None
        if any(ciru.validate_ciru_unit(record) for record in records):
            return None
        if any(
            any(
                record.get("semantic_judge", {}).get(field) is not True
                for field in v4.JUDGE_FIELDS
            )
            for record in records
        ):
            return None
        return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def parse_block_ids(raw: str, available: Sequence[int]) -> list[int]:
    if anchors.normalise(raw) in {"", "all", "*"}:
        return list(available)
    try:
        selected = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("block ids must be comma-separated integers or 'all'") from exc
    unknown = set(selected) - set(available)
    if not selected or unknown:
        raise ValueError(f"invalid block ids: {sorted(unknown) if unknown else selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget05_perturbed")
    parser.add_argument("--manifest", type=Path, default=v2.DEFAULT_MANIFEST)
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
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8000)
    parser.add_argument("--block-concurrency", type=int, default=1)
    parser.add_argument("--row-concurrency", type=int, default=5)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--profile-retries", type=int, default=4)
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
        raise SystemExit("Set generation and judge API keys before V5.2 generation")
    from openai import OpenAI

    manifest = v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(f"manifest split={manifest.get('split')} != {args.split}")
    all_blocks = [
        with_contracts_and_anchors(block)
        for block in v2.group_author_blocks(v2.load_sources(args.split), manifest)
    ]
    all_ids = [int(block["block_id"]) for block in all_blocks]
    try:
        selected_ids = parse_block_ids(args.block_ids, all_ids)
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
                    write_json(state_dir / f"block_{block_id:02d}.json", result)
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
        raise SystemExit("V5.2 incomplete; valid profiles and rows were retained")

    ordered = [completed[block_id] for block_id in selected_ids]
    records = [record for block in ordered for record in block["records"]]
    expected = len(blocks) * int(manifest["block_size"])
    if len(records) != expected:
        raise SystemExit(f"record count {len(records)} != {expected}")
    ciru.write_ciru_jsonl(args.output, records)
    write_json(profiles_output, {
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
                    "block_id", "profile_id", "profile_digest", "target_entity",
                    "replacement_entity", "replacement_pronouns",
                    "anchor_catalog_digest", "anchor_replacements", "author_plan",
                    "placebo_plan", "reconciliation_conflicts",
                    "profile_attempt", "judge_round",
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
