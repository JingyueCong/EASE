#!/usr/bin/env python3
"""Generate TOFU author factorial data with ledger-constrained full answers.

V5.5 keeps the coherent, semantically audited author ledger from V5.3/V5.4,
but removes the assumption that every author fact can be expressed by replacing
one frozen answer span.  Each factual row is rendered independently as one
complete C01 answer under its immutable response contract.  The C01 question,
identity binding, placebo cells, coverage, provenance, and semantic audit remain
deterministic or independently validated.

This is a new design rather than a relaxation of V5.4.  FullAnswer and V1-V5.4
artifacts remain untouched and available as ablations.
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
V54_PATH = SCRIPT_DIR / "generate_tofu_author_slots_v5_4.py"
DESIGN_VERSION = "tofu-author-ledger-answer-v5.5"
V54_DESIGN_VERSION = "tofu-author-ledger-slots-v5.4"
V53_DESIGN_VERSION = "tofu-author-ledger-rowlocal-v5.3"
SURFACE_RENDERER = "ledger-constrained-row-answer-v5.5"
MAPPING_SCOPE = "frozen-ledger-conditioned-complete-c01-answer"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v54 = load_module("tofu_author_answers_v55_v54", V54_PATH)
v53 = v54.v53
v52 = v54.v52
anchors = v54.anchors
ciru = v54.ciru


ANSWER_PROMPT = """Render one complete C01 answer from a frozen author ledger.

The author identity, target relation, replacement fact, C01 question, and
response contract are fixed.  Return only a replacement answer, not span edits.

Rules:
- copy source_id, target_relation, ledger_fact_key, ledger_replacement_fact,
  and c01_question exactly from the payload;
- answer the same relation as C11 about the replacement author;
- explicitly and faithfully express ledger_replacement_fact;
- preserve the source answer's response mode, answer format, approximate
  length, number of facts, grammatical person, and level of detail;
- use one fluent answer, with no editing instructions or meta-commentary;
- do not mention the target author or copy the original factual answer;
- do not invent facts that contradict the frozen ledger or profile;
- identity and unavailable rows are rendered by code and are not sent here.

If validation_feedback and previous_candidate are present, return the complete
object with only the reported answer defect repaired.  Never change a frozen
field.

Return JSON only:
{"source_id":"exact id","target_relation":"copied relation",
 "ledger_fact_key":"copied key","ledger_replacement_fact":"copied fact",
 "c01_question":"copied question","replacement_answer":"complete answer"}
"""


def configure_shared_modules() -> None:
    """Make inherited profile/judge/assembly helpers emit V5.5 provenance."""
    v54.DESIGN_VERSION = DESIGN_VERSION
    v54.SURFACE_RENDERER = SURFACE_RENDERER
    v54.MAPPING_SCOPE = MAPPING_SCOPE
    v54.configure_shared_modules()


def replace_identity(text: str, target: str, replacement: str) -> str:
    return anchors.replace_identity(text, target, replacement)


def c01_question(block: Mapping, source: Mapping, profile: Mapping) -> str:
    question = replace_identity(
        source["question"], block["target_entity"], profile["replacement_entity"]
    )
    joined = anchors.normalise(question)
    for alias in anchors.identity_aliases(block["target_entity"]):
        if re.search(
            rf"(?<![\w-]){re.escape(anchors.normalise(alias))}(?![\w-])",
            joined,
        ):
            raise ValueError(f"C01 question still contains target alias {alias!r}")
    if anchors.normalise(block["target_entity"]) in anchors.normalise(
        source["question"]
    ) and anchors.normalise(profile["replacement_entity"]) not in joined:
        raise ValueError("C01 question loses explicit replacement identity")
    return question


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    entry = profile["fact_ledger_by_source"][source["source_id"]]
    payload = {
        "design_version": DESIGN_VERSION,
        "source_id": source["source_id"],
        "target_entity": block["target_entity"],
        "replacement_entity": profile["replacement_entity"],
        "replacement_pronouns": profile["replacement_pronouns"],
        "profile_summary": profile["profile_summary"],
        "source_question": source["question"],
        "source_answer_style_reference": source["answer"],
        "c01_question": c01_question(block, source, profile),
        "response_contract": source["contract"],
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "fact_change_required": entry["fact_change_required"],
        "intervention_policy": entry["intervention_policy"],
    }
    if feedback:
        payload["validation_feedback"] = feedback
    if previous is not None:
        payload["previous_candidate"] = previous
    return payload


def _target_alias_leaks(text: str, target: str) -> str | None:
    normalised = anchors.normalise(text)
    for alias in anchors.identity_aliases(target):
        if re.search(
            rf"(?<![\w-]){re.escape(anchors.normalise(alias))}(?![\w-])",
            normalised,
        ):
            return alias
    return None


def deterministic_policy_candidate(
    block: Mapping, source: Mapping, profile: Mapping
) -> Dict:
    entry = profile["fact_ledger_by_source"][source["source_id"]]
    return {
        "source_id": source["source_id"],
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "c01_question": c01_question(block, source, profile),
        "replacement_answer": replace_identity(
            source["answer"],
            block["target_entity"],
            profile["replacement_entity"],
        ),
    }


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
) -> Dict:
    source_id = source["source_id"]
    entry = profile["fact_ledger_by_source"][source_id]
    expected_question = c01_question(block, source, profile)
    frozen = {
        "source_id": source_id,
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "c01_question": expected_question,
    }
    for field, expected in frozen.items():
        if generated.get(field) != expected:
            raise ValueError(f"{field} must exactly copy the frozen payload")

    raw_answer = generated.get("replacement_answer")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ValueError("replacement_answer must be a non-empty string")
    answer = " ".join(raw_answer.replace("\n", " ").split()).strip()
    answer = replace_identity(
        answer, block["target_entity"], profile["replacement_entity"]
    )
    cell = {"question": expected_question, "answer": answer}

    leak = _target_alias_leaks(f"{expected_question} {answer}", block["target_entity"])
    if leak:
        raise ValueError(f"C01 still contains target alias {leak!r}")
    errors = v52.v4.contracts.contract_errors(cell, source["contract"])
    if errors:
        raise ValueError("C01 response contract: " + "; ".join(errors))

    required = bool(entry["fact_change_required"])
    identity_only = replace_identity(
        source["answer"],
        block["target_entity"],
        profile["replacement_entity"],
    )
    if required:
        if anchors.normalise(answer) == anchors.normalise(identity_only):
            raise ValueError("C01 changes only author identity, not the target fact")
        answer_tokens = v53.content_tokens(answer)
        ledger_tokens = v53.content_tokens(entry["replacement_fact"])
        replacement_tokens = v53.content_tokens(profile["replacement_entity"])
        shared = (answer_tokens & ledger_tokens) - replacement_tokens
        if not shared:
            raise ValueError(
                "replacement_answer must express a content token from "
                "ledger_replacement_fact"
            )
    elif anchors.normalise(answer) != anchors.normalise(identity_only):
        raise ValueError(
            "identity/unavailable rows must use deterministic identity rendering"
        )

    return {
        **frozen,
        "replacement_answer": answer,
        "cell": cell,
        "fact_change_required": required,
        "render_mode": (
            "ledger_constrained_complete_answer" if required
            else "deterministic_identity_or_unavailability"
        ),
        "ledger_digest": profile["ledger_digest"],
    }


def candidate_for_prompt(candidate: Mapping) -> Dict:
    return {
        key: candidate[key]
        for key in (
            "source_id",
            "target_relation",
            "ledger_fact_key",
            "ledger_replacement_fact",
            "c01_question",
            "replacement_answer",
        )
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
            block,
            source,
            profile,
            deterministic_policy_candidate(block, source, profile),
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
            f"attempt={attempt}/{args.row_retries}",
            flush=True,
        )
        candidate = None
        try:
            candidate = v52.v2.request_json(
                client,
                v52.request_args(args),
                ANSWER_PROMPT,
                row_payload(block, source, profile, feedback, previous),
                f"V5.5 block {block_id} row {source_id}",
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
                f"attempt={attempt}/{args.row_retries}",
                flush=True,
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
                f"attempt={attempt}/{args.row_retries} error={exc}",
                flush=True,
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
    for query_index, source in enumerate(block["sources"]):
        source_id = source["source_id"]
        candidate = candidates[source_id]
        entry = profile["fact_ledger_by_source"][source_id]
        try:
            validated = validate_row_candidate(
                block, source, profile, candidate_for_prompt(candidate)
            )
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
                "target_group_ids": [],
                "replacement_fact": entry["replacement_fact"],
                "fact_change_required": entry["fact_change_required"],
                "intervention_policy": entry["intervention_policy"],
                "render_mode": validated["render_mode"],
                "question_edits": [],
                "answer_edits": ([{
                    "anchor_id": "complete-c01-answer",
                    "group_id": f"ledger:{entry['fact_key']}",
                    "old": source["answer"],
                    "new": validated["replacement_answer"],
                }] if entry["fact_change_required"] else []),
                "mapping_attempt": int(candidate.get("mapping_attempt", 0)),
                "repair_generation": int(candidate.get("repair_generation", 0)),
            }
            cells[source_id] = {
                "C01": validated["cell"],
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
        "anchor_replacements": {},
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
        f"{args.split}:author-ledger-answer-v5.5-"
        f"block-{int(block['block_id']):02d}:seed-{args.seed}"
    )
    assembled["profile_id"] = profile_id
    assembled["design_version"] = DESIGN_VERSION
    for record in assembled["records"]:
        source_id = record["source_id"]
        record["profile_id"] = profile_id
        record["design_version"] = DESIGN_VERSION
        record["render_mode"] = plan["row_plans"][source_id]["render_mode"]
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
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
        generation_client,
        judge_client,
        args,
        block,
        protected_authors,
        state_dir,
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
            f"V5.5 block {block_id} row fidelity audit round {judge_round}",
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
            f"{args.judge_rounds} rows={','.join(sorted(failures))}",
            flush=True,
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


def migrate_profile(
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
        if cached.get("design_version") not in {
            V53_DESIGN_VERSION, V54_DESIGN_VERSION
        }:
            raise ValueError("seed profile is not a frozen-ledger artifact")
        profile = v53.validate_profile(
            block, cached["profile"], protected_authors
        )
        if cached.get("ledger_digest") != profile["ledger_digest"]:
            raise ValueError("seed profile ledger digest is stale")
        verdict = v53.validate_profile_judgement(
            cached["semantic_judge"], block
        )
        v52.write_json(target_path, {
            "design_version": DESIGN_VERSION,
            "migrated_from": str(source_path.resolve()),
            "anchor_catalog_digest": block["anchor_catalog"]["digest"],
            "ledger_digest": profile["ledger_digest"],
            "profile_attempt": int(cached.get("profile_attempt", 0)),
            "profile": {
                key: value for key, value in profile.items()
                if key not in {
                    "fact_ledger_by_source", "profile_semantic_judge"
                }
            },
            "semantic_judge": verdict,
        })
        print(f"migrate_ledger_profile block={block_id}", flush=True)
        return True
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"skip_seed_profile block={block_id} error={exc}", flush=True)
        return False


def main() -> None:
    configure_shared_modules()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget05_perturbed")
    parser.add_argument("--manifest", type=Path, default=v52.v2.DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--seed-ledger-state-dir", type=Path)
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
    parser.add_argument("--judge-rounds", type=int, default=4)
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
        raise SystemExit("Set generation and judge API keys before V5.5 generation")
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
    profiles_output = args.profiles_output or Path(
        str(args.output) + ".profiles.json"
    )
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_ledger_state_dir and args.seed_ledger_state_dir.is_dir():
        for block in blocks:
            migrate_profile(
                block, protected, args.seed_ledger_state_dir, state_dir
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
                f"reuse block={block['block_id']} "
                f"author={block['target_entity']}",
                flush=True,
            )

    failures = []
    if pending:
        generation_client = OpenAI(api_key=api_key, base_url=args.base_url)
        judge_client = OpenAI(api_key=judge_key, base_url=args.judge_base_url)
        with ThreadPoolExecutor(
            max_workers=max(args.block_concurrency, 1)
        ) as executor:
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
                    failures.append(
                        (block_id, block["target_entity"], str(exc))
                    )

    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(
                f"FAIL block={block_id} author={author}: {error}",
                file=sys.stderr,
            )
        raise SystemExit(
            "V5.5 incomplete; valid ledgers, row answers, and blocks were retained"
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
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
