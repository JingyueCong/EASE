#!/usr/bin/env python3
"""Generate TOFU author factorial data with frozen ledgers and local slots.

V5.4 keeps the V5.3 author-level fact ledger, but removes global anchor IDs
from the model interface.  A row mapper returns one list of local slot edits;
code derives target group IDs, canonicalises typed surface values, and renders
each row independently.  This prevents duplicated-ID/list coordination errors
and keeps surface-only constraints out of the semantic planning interface.
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
V53_PATH = SCRIPT_DIR / "generate_tofu_author_ledger_v5_3.py"
DESIGN_VERSION = "tofu-author-ledger-slots-v5.4"
V53_DESIGN_VERSION = "tofu-author-ledger-rowlocal-v5.3"
SURFACE_RENDERER = "deterministic-ledger-local-slot-v5.4"
MAPPING_SCOPE = "frozen-ledger-conditioned-local-slot-edits"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v53 = load_module("tofu_author_slots_v54_v53", V53_PATH)
v52 = v53.v52
anchors = v53.anchors
ciru = v53.ciru


ROW_PROMPT = """Map one immutable TOFU C11 row to its frozen replacement fact.

The fact, relation, and replacement author are already fixed. You only select
local answer slots and propose a short surface value for each selected slot.
Local slot keys are valid only inside this row; never return internal group IDs.

Rules:
- copy source_id, target_relation, ledger_fact_key, and ledger_replacement_fact
  exactly from the payload;
- factual rows return one to three edits drawn from eligible_local_slots;
- each edit appears once and contains only slot_key and replacement_value;
- replacement values express the frozen ledger fact, not a new fact;
- respect each slot's supplied kind and surface_contract;
- do not add quotes around a quoted slot value; code preserves source quotes;
- do not add sentence punctuation, markup, polarity, or uncertainty scaffolding;
- identity/unavailable rows are rendered by code and are not sent to this stage;
- never rewrite the question, answer, or author profile.

If validation_feedback and previous_candidate are present, return the complete
row object with only the local slot defect repaired. Never change frozen ledger
fields.

Return JSON only:
{"source_id":"exact id","target_relation":"copied relation",
 "ledger_fact_key":"copied key",
 "ledger_replacement_fact":"copied fact",
 "edits":[{"slot_key":"A00","replacement_value":"typed local value"}]}
"""


def configure_shared_modules() -> None:
    """Make inherited V5.2/V5.3 helpers emit V5.4 provenance."""
    v53.DESIGN_VERSION = DESIGN_VERSION
    v53.SURFACE_RENDERER = SURFACE_RENDERER
    v53.MAPPING_SCOPE = MAPPING_SCOPE
    v53.configure_v52_globals()
    v52.DESIGN_VERSION = DESIGN_VERSION
    v52.SURFACE_RENDERER = SURFACE_RENDERER


def surface_contract(group: Mapping) -> str:
    kind = group["kind"]
    if kind == "year":
        return "one valid four-digit year"
    if kind == "number":
        return "one numeric value"
    if kind == "date":
        return "one complete valid date; code preserves the source date format"
    if kind == "token":
        words = len(anchors.WORD_PATTERN.findall(group["text"]))
        return f"exactly {words} content token(s); preserve polarity scaffold"
    if kind == "quoted":
        return "title content without surrounding quotation marks"
    return "short proper-name/content phrase without sentence punctuation"


def local_slots(block: Mapping, source_id: str) -> list[Dict]:
    groups = {
        item["group_id"]: item for item in block["anchor_catalog"]["groups"]
    }
    eligible = v52.eligible_answer_group_ids(block, source_id)
    seen = set()
    result = []
    for occurrence in block["anchor_catalog"]["occurrences"]:
        group_id = occurrence["group_id"]
        if (
            occurrence["source_id"] != source_id
            or occurrence["field"] != "answer"
            or group_id not in eligible
            or group_id in seen
        ):
            continue
        seen.add(group_id)
        group = groups[group_id]
        result.append({
            "slot_key": f"A{len(result):02d}",
            "group_id": group_id,
            "old_text": group["text"],
            "kind": group["kind"],
            "surface_contract": surface_contract(group),
        })
    return result


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    entry = profile["fact_ledger_by_source"][source["source_id"]]
    payload = {
        "source_id": source["source_id"],
        "C11": {"question": source["question"], "answer": source["answer"]},
        "response_contract": source["contract"],
        "replacement_entity": profile["replacement_entity"],
        "frozen_fact_ledger_entry": entry,
        "eligible_local_slots": [
            {key: item[key] for key in (
                "slot_key", "old_text", "kind", "surface_contract"
            )}
            for item in local_slots(block, source["source_id"])
        ],
    }
    if feedback:
        payload["validation_feedback"] = feedback
    if previous is not None:
        payload["previous_candidate"] = previous
    return payload


def _strip_outer_markup(value: object) -> str:
    if not isinstance(value, str):
        return ""
    result = " ".join(value.replace("\n", " ").split()).strip()
    pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"), ("`", "`"))
    changed = True
    while changed and len(result) >= 2:
        changed = False
        for left, right in pairs:
            if result.startswith(left) and result.endswith(right):
                result = result[len(left):-len(right)].strip()
                changed = True
                break
    return result


def _candidate_dates(text: str) -> list[str]:
    values = [match.group() for match in anchors.DATE_PATTERN.finditer(text)]
    for pattern in (anchors.ISO_DATE_PATTERN, anchors.TEXT_DATE_PATTERN):
        match = pattern.fullmatch(text.strip())
        if match:
            values.insert(0, text.strip())
    return list(dict.fromkeys(values))


def _surface_candidates(
    group: Mapping,
    proposed: object,
    ledger_fact: str,
    response_contract: Mapping,
) -> list[str]:
    raw_values = [_strip_outer_markup(proposed), _strip_outer_markup(ledger_fact)]
    kind = group["kind"]
    candidates: list[str] = []
    if kind == "year":
        for value in raw_values:
            candidates.extend(match.group() for match in anchors.YEAR_PATTERN.finditer(value))
    elif kind == "number":
        for value in raw_values:
            candidates.extend(match.group() for match in anchors.NUMBER_PATTERN.finditer(value))
    elif kind == "date":
        for value in raw_values:
            candidates.extend(_candidate_dates(value))
    elif kind == "token":
        old = anchors.normalise(group["text"])
        if old in anchors.PLURAL_QUANTIFIERS:
            candidates.extend(
                value for value in sorted(anchors.PLURAL_QUANTIFIERS)
                if value != old
            )
        elif old in anchors.SINGULAR_QUANTIFIERS:
            candidates.extend(
                value for value in sorted(anchors.SINGULAR_QUANTIFIERS)
                if value != old
            )
        for value in raw_values:
            candidates.extend(
                token for token in anchors.WORD_PATTERN.findall(value)
                if anchors.normalise(token) not in anchors.SCAFFOLD_WORDS
                and not token.isdigit()
            )
    elif kind == "quoted":
        for value in raw_values:
            value = re.sub(r'["“”]', "", value).strip()
            # Some frozen title anchors include the sentence terminator inside
            # the quotation marks.  Keep at most that one terminal mark and
            # never allow a generated title to introduce an extra sentence.
            terminal = group["text"][-1] if group["text"].endswith(
                (".", "!", "?")
            ) else ""
            value = re.sub(r"[.!?;]+", " ", value)
            value = " ".join(value.split()).strip()
            if value and terminal:
                value += terminal
            if value:
                candidates.append(value)
    else:
        for value in raw_values:
            cleaned = re.sub(r"[.!?;,]", "", value).strip()
            if cleaned:
                candidates.append(cleaned)
            tokens = [
                token for token in anchors.WORD_PATTERN.findall(value)
                if anchors.normalise(token) not in anchors.SCAFFOLD_WORDS
            ]
            if tokens:
                candidates.append(" ".join(tokens[:4]))

    answer_format = response_contract["answer_format"]
    result = []
    for value in candidates:
        if answer_format not in {"date_or_year", "numeric"} and re.search(
            r"\b\d+(?:\.\d+)?\b", value
        ):
            continue
        if anchors.normalise(value) == anchors.normalise(group["text"]):
            continue
        if value not in result:
            result.append(value)
    return result


def canonical_surface_value(
    group: Mapping,
    proposed: object,
    ledger_fact: str,
    response_contract: Mapping,
) -> str:
    errors = []
    for candidate in _surface_candidates(
        group, proposed, ledger_fact, response_contract
    ):
        try:
            return anchors._validate_replacement(group, candidate)
        except ValueError as exc:
            errors.append(str(exc))
    detail = errors[-1] if errors else "no compatible typed value was supplied"
    raise ValueError(
        f"slot for {group['kind']} anchor {group['text']!r} cannot express "
        f"the frozen ledger fact: {detail}"
    )


def surface_relation(entry: Mapping) -> str:
    if entry["fact_change_required"]:
        return entry["target_relation"]
    if "unavailability" in entry["intervention_policy"]:
        return "unavailable relation"
    return "author identity"


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
) -> Dict:
    source_id = source["source_id"]
    entry = profile["fact_ledger_by_source"][source_id]
    if generated.get("source_id") != source_id:
        raise ValueError(f"source_id must equal {source_id}")
    if generated.get("ledger_fact_key") != entry["fact_key"]:
        raise ValueError("ledger_fact_key must exactly copy frozen ledger")
    if generated.get("ledger_replacement_fact") != entry["replacement_fact"]:
        raise ValueError("ledger_replacement_fact must exactly copy frozen ledger")
    if str(generated.get("target_relation", "")).strip() != entry["target_relation"]:
        raise ValueError("target_relation must exactly copy frozen ledger")

    raw_edits = generated.get("edits", [])
    if not isinstance(raw_edits, list) or any(
        not isinstance(item, Mapping) for item in raw_edits
    ):
        raise ValueError("edits must be a list of objects")
    required = bool(entry["fact_change_required"])
    if not required and raw_edits:
        raise ValueError("identity/unavailable rows must use empty edits")
    if required and not 1 <= len(raw_edits) <= 3:
        raise ValueError("factual rows require one to three local slot edits")

    slots = {item["slot_key"]: item for item in local_slots(block, source_id)}
    groups = {
        item["group_id"]: item for item in block["anchor_catalog"]["groups"]
    }
    replacements: Dict[str, str] = {}
    local_edits = []
    seen_slots = set()
    for raw in raw_edits:
        slot_key = raw.get("slot_key")
        if slot_key not in slots:
            raise ValueError(f"unknown local slot_key: {slot_key!r}")
        if slot_key in seen_slots:
            raise ValueError(f"duplicate local slot_key: {slot_key}")
        seen_slots.add(slot_key)
        slot = slots[slot_key]
        group = groups[slot["group_id"]]
        value = canonical_surface_value(
            group,
            raw.get("replacement_value"),
            entry["replacement_fact"],
            source["contract"],
        )
        replacements[group["group_id"]] = value
        local_edits.append({
            "slot_key": slot_key,
            "group_id": group["group_id"],
            "replacement_value": value,
        })

    rendered = anchors.render_row(
        source,
        block["target_entity"],
        profile["replacement_entity"],
        surface_relation(entry),
        block["anchor_catalog"],
        replacements,
        v52.v4.contracts,
    )
    if required:
        replacement_tokens = v53.content_tokens(" ".join(replacements.values()))
        ledger_tokens = v53.content_tokens(entry["replacement_fact"])
        if not replacement_tokens.intersection(ledger_tokens):
            raise ValueError(
                "local slot edits must express a content token from "
                "ledger_replacement_fact"
            )
        applied = {
            item["group_id"]
            for item in rendered["question_edits"] + rendered["answer_edits"]
        }
        if not set(replacements).issubset(applied):
            raise ValueError("selected local slots were not rendered")

    return {
        "source_id": source_id,
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "ledger_digest": profile["ledger_digest"],
        "target_group_ids": list(replacements),
        "anchor_replacements": replacements,
        "local_edits": local_edits,
        "fact_change_required": required,
    }


def candidate_for_prompt(candidate: Mapping) -> Dict:
    return {
        "source_id": candidate["source_id"],
        "target_relation": candidate["target_relation"],
        "ledger_fact_key": candidate["ledger_fact_key"],
        "ledger_replacement_fact": candidate["ledger_replacement_fact"],
        "edits": [
            {
                "slot_key": item["slot_key"],
                "replacement_value": item["replacement_value"],
            }
            for item in candidate["local_edits"]
        ],
    }


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


def write_row_checkpoint(
    path: Path,
    profile: Mapping,
    validated: Mapping,
    mapping_attempt: int,
    repair_generation: int,
) -> None:
    v52.write_json(path, {
        "design_version": DESIGN_VERSION,
        "profile_digest": profile["profile_digest"],
        "ledger_digest": profile["ledger_digest"],
        "mapping_attempt": mapping_attempt,
        "repair_generation": repair_generation,
        "candidate": candidate_for_prompt(validated),
    })


def deterministic_policy_candidate(
    source: Mapping, profile: Mapping
) -> Dict:
    entry = profile["fact_ledger_by_source"][source["source_id"]]
    return {
        "source_id": source["source_id"],
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "edits": [],
    }


def generate_row(
    client,
    args,
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    state_dir: Path,
    feedback: str = "",
    previous: Mapping | None = None,
    force: bool = False,
) -> Dict:
    block_id, source_id = int(block["block_id"]), source["source_id"]
    path = v52.row_path(state_dir, block_id, source_id)
    if not force:
        cached = load_cached_row(path, block, source, profile)
        if cached is not None:
            print(f"reuse_row block={block_id} source={source_id}", flush=True)
            return cached

    entry = profile["fact_ledger_by_source"][source_id]
    if not entry["fact_change_required"]:
        validated = validate_row_candidate(
            block, source, profile,
            deterministic_policy_candidate(source, profile),
        )
        validated["mapping_attempt"] = 0
        validated["repair_generation"] = 1 if force else 0
        write_row_checkpoint(
            path, profile, validated, 0, validated["repair_generation"]
        )
        print(
            f"row_ready block={block_id} source={source_id} "
            "attempt=0/0 policy=deterministic",
            flush=True,
        )
        return validated

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
                row_payload(block, source, profile, feedback, previous),
                f"V5.4 block {block_id} row {source_id}",
            )
            validated = validate_row_candidate(
                block, source, profile, candidate
            )
            validated["mapping_attempt"] = attempt
            validated["repair_generation"] = 1 if force else 0
            write_row_checkpoint(
                path,
                profile,
                validated,
                attempt,
                validated["repair_generation"],
            )
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


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    row_plans, cells, errors = {}, {}, []
    per_row_replacements = {}
    for query_index, source in enumerate(block["sources"]):
        source_id = source["source_id"]
        candidate = candidates[source_id]
        entry = profile["fact_ledger_by_source"][source_id]
        local_map = dict(candidate["anchor_replacements"])
        per_row_replacements[source_id] = local_map
        try:
            rendered = anchors.render_row(
                source,
                block["target_entity"],
                profile["replacement_entity"],
                surface_relation(entry),
                block["anchor_catalog"],
                local_map,
                v52.v4.contracts,
            )
            applied = rendered["question_edits"] + rendered["answer_edits"]
            applied_ids = {item["group_id"] for item in applied}
            if entry["fact_change_required"] and not set(local_map).issubset(
                applied_ids
            ):
                raise ValueError("selected local slots were not rendered")
            placebo = v52.v4.contracts.render_professional_placebo(
                source["contract"],
                block["target_entity"],
                profile["replacement_entity"],
                query_index,
                assignment_flip=(
                    (int(block["block_id"]) + query_index + seed) % 2 == 1
                ),
            )
            for name in ("C10", "C00"):
                contract_errors = v52.v4.contracts.contract_errors(
                    placebo[name], source["contract"]
                )
                if contract_errors:
                    raise ValueError(f"{name}: " + "; ".join(contract_errors))
            row_plans[source_id] = {
                "source_id": source_id,
                "target_relation": entry["target_relation"],
                "target_group_ids": list(local_map),
                "replacement_fact": entry["replacement_fact"],
                "fact_change_required": entry["fact_change_required"],
                "intervention_policy": entry["intervention_policy"],
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
        "anchor_replacements": per_row_replacements,
        "reconciliation_conflicts": [],
        "row_plans": row_plans,
        "cells": cells,
        "fact_ledger": profile["fact_ledger"],
        "fact_ledger_by_source": profile["fact_ledger_by_source"],
        "ledger_digest": profile["ledger_digest"],
        "profile_semantic_judge": profile["profile_semantic_judge"],
    }


def assemble_block(
    block: Mapping,
    plan: Mapping,
    verdicts: Mapping[str, Mapping],
    args,
) -> Dict:
    assembled = v53.assemble_block(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-ledger-slots-v5.4-"
        f"block-{int(block['block_id']):02d}:seed-{args.seed}"
    )
    assembled["profile_id"] = profile_id
    assembled["design_version"] = DESIGN_VERSION
    for record in assembled["records"]:
        record["profile_id"] = profile_id
        record["design_version"] = DESIGN_VERSION
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(
                f"{record['source_id']}: " + "; ".join(failures)
            )
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
        generation_client, args, block, profile, state_dir
    )
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    last_failures: Dict[str, str] = {}
    for judge_round in range(1, args.judge_rounds + 1):
        plan = materialize_plan(block, profile, candidates, args.seed)
        judgement = v52.v2.request_json(
            judge_client,
            v52.request_args(args, judge=True),
            v53.JUDGE_PROMPT,
            v53.judge_payload(block, plan),
            f"V5.4 block {block_id} row fidelity audit round {judge_round}",
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
            result = assemble_block(block, plan, verdicts, args)
            print(
                f"block_ready block={block_id} judge_round={judge_round} "
                "conflicts=0",
                flush=True,
            )
            return result
        last_failures = failures
        print(
            f"judge_reject block={block_id} round={judge_round}/"
            f"{args.judge_rounds} rows={','.join(sorted(failures))}", flush=True,
        )
        if judge_round == args.judge_rounds:
            break
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
                    True,
                ): source_id
                for source_id in failures
            }
            for future in as_completed(futures):
                source_id = futures[future]
                repaired[source_id] = future.result()
        candidates.update(repaired)
    raise RuntimeError(
        f"block {block_id} row-fidelity judge exhausted "
        f"{args.judge_rounds} rounds: "
        + " | ".join(
            f"{source_id}: {reason}"
            for source_id, reason in sorted(last_failures.items())
        )
    )


def migrate_v53_profile(
    block: Mapping,
    protected_authors: Sequence[str],
    source_state: Path,
    target_state: Path,
) -> bool:
    block_id = int(block["block_id"])
    source_path = v52.profile_path(source_state, block_id)
    target_path = v52.profile_path(target_state, block_id)
    if target_path.is_file() or not source_path.is_file():
        return False
    try:
        cached = json.loads(source_path.read_text(encoding="utf-8"))
        if cached.get("design_version") != V53_DESIGN_VERSION:
            raise ValueError(
                "seed profile is not a frozen-ledger V5.3 artifact"
            )
        profile = v53.validate_profile(block, cached["profile"], protected_authors)
        if cached.get("ledger_digest") != profile["ledger_digest"]:
            raise ValueError("seed profile ledger digest is stale")
        verdict = v53.validate_profile_judgement(cached["semantic_judge"], block)
        v52.write_json(target_path, {
            "design_version": DESIGN_VERSION,
            "migrated_from": str(source_path.resolve()),
            "anchor_catalog_digest": block["anchor_catalog"]["digest"],
            "ledger_digest": profile["ledger_digest"],
            "profile_attempt": int(cached.get("profile_attempt", 0)),
            "profile": {
                key: value for key, value in profile.items()
                if key not in {"fact_ledger_by_source", "profile_semantic_judge"}
            },
            "semantic_judge": verdict,
        })
        print(f"migrate_ledger_profile block={block_id}", flush=True)
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            f"skip_v53_profile block={block_id} error={exc}", flush=True
        )
        return False


def migrate_v53_rows(
    block: Mapping,
    profile: Mapping,
    source_state: Path,
    target_state: Path,
) -> int:
    block_id = int(block["block_id"])
    slots_by_source = {
        source["source_id"]: {
            item["group_id"]: item["slot_key"]
            for item in local_slots(block, source["source_id"])
        }
        for source in block["sources"]
    }
    migrated = 0
    for source in block["sources"]:
        source_id = source["source_id"]
        source_path = v52.row_path(source_state, block_id, source_id)
        target_path = v52.row_path(target_state, block_id, source_id)
        if target_path.is_file() or not source_path.is_file():
            continue
        try:
            cached = json.loads(source_path.read_text(encoding="utf-8"))
            if cached.get("design_version") != V53_DESIGN_VERSION:
                continue
            if cached.get("ledger_digest") != profile["ledger_digest"]:
                continue
            old = cached["candidate"]
            edits = []
            for item in old.get("anchor_replacements", []):
                group_id = item["group_id"]
                edits.append({
                    "slot_key": slots_by_source[source_id][group_id],
                    "replacement_value": item["replacement_value"],
                })
            entry = profile["fact_ledger_by_source"][source_id]
            candidate = {
                "source_id": source_id,
                "target_relation": entry["target_relation"],
                "ledger_fact_key": entry["fact_key"],
                "ledger_replacement_fact": entry["replacement_fact"],
                "edits": edits,
            }
            validated = validate_row_candidate(
                block, source, profile, candidate
            )
            write_row_checkpoint(
                target_path,
                profile,
                validated,
                int(cached.get("mapping_attempt", 0)),
                int(cached.get("repair_generation", 0)),
            )
            migrated += 1
            print(
                f"migrate_row block={block_id} source={source_id}", flush=True
            )
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return migrated


def load_profile_for_migration(
    block: Mapping,
    protected_authors: Sequence[str],
    state_dir: Path,
) -> Dict | None:
    path = v52.profile_path(state_dir, int(block["block_id"]))
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        profile = v53.validate_profile(block, cached["profile"], protected_authors)
        profile["profile_attempt"] = int(cached.get("profile_attempt", 0))
        profile["profile_semantic_judge"] = v53.validate_profile_judgement(
            cached["semantic_judge"], block
        )
        return profile
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def main() -> None:
    configure_shared_modules()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget05_perturbed")
    parser.add_argument("--manifest", type=Path, default=v52.v2.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--seed-v53-state-dir", type=Path)
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
        raise SystemExit("Set generation and judge API keys before V5.4 generation")
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
        block for block in all_blocks
        if int(block["block_id"]) in selected_ids
    ]
    protected = [block["target_entity"] for block in all_blocks]
    state_dir = args.state_dir or Path(str(args.output) + ".blocks")
    profiles_output = args.profiles_output or Path(str(args.output) + ".profiles.json")
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_v53_state_dir and args.seed_v53_state_dir.is_dir():
        for block in blocks:
            migrate_v53_profile(
                block, protected, args.seed_v53_state_dir, state_dir
            )
            profile = load_profile_for_migration(block, protected, state_dir)
            if profile is not None:
                migrate_v53_rows(
                    block, profile, args.seed_v53_state_dir, state_dir
                )

    completed, pending = {}, []
    for block in blocks:
        path = state_dir / f"block_{int(block['block_id']):02d}.json"
        cached = v53.load_valid_block(path, block)
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
                    v52.write_json(
                        state_dir / f"block_{block_id:02d}.json", result
                    )
                    completed[block_id] = result
                    print(
                        f"valid_block={len(completed)}/{len(blocks)} "
                        f"block={block_id} author={block['target_entity']}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append((block_id, block["target_entity"], str(exc)))

    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        raise SystemExit(
            "V5.4 incomplete; valid ledgers, local rows, and blocks were retained"
        )

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
