#!/usr/bin/env python3
"""Generate TOFU author factorial data with a frozen anchor-id IR."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
V2_PATH = SCRIPT_DIR / "generate_tofu_author_factorial.py"
V4_GENERATOR_PATH = SCRIPT_DIR / "generate_tofu_author_typed_v4.py"
ANCHOR_PATH = ROOT / "ULD/uld/data/tofu_anchor_v5.py"
CIRU_PATH = ROOT / "ULD/uld/data/ciru.py"
DESIGN_VERSION = "tofu-author-anchor-v5"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_module("tofu_anchor_v5_v2_shared", V2_PATH)
v4 = load_module("tofu_anchor_v5_v4_shared", V4_GENERATOR_PATH)
anchors = load_module("tofu_anchor_v5_contract", ANCHOR_PATH)
ciru = load_module("tofu_anchor_v5_ciru", CIRU_PATH)


PLAN_PROMPT = """You plan one coherent counterfactual author profile for a
20-row TOFU block. C11 and its frozen anchor catalog are immutable.

You may NOT quote, locate, or rewrite source text. You may only:
1. name one replacement author with the required pronoun class;
2. assign a replacement_value to a supplied group_id;
3. label each row's target_relation.

Every chosen group is replaced by code at all of its frozen occurrences. Pick
enough factual groups that every non-identity, non-unavailable row changes its
target fact. Reuse the same replacement profile across all rows. Preserve the
semantic type of each group: year->year, number->number, date->date, title->
title, place->place, award->award, genre->genre. Never use a protected author,
negation, uncertainty markers, sentence punctuation, quotations, or prose in a
replacement value. Do not choose generic scaffold words merely to force a
change.

If validation_feedback and previous_candidate are present, return the complete
object with the reported defects repaired.

Return JSON only:
{
  "target_entity": "required target",
  "replacement_entity": "one new author name",
  "replacement_pronouns": "required class",
  "profile_summary": "coherent replacement biography",
  "anchor_replacements": [
    {"group_id": "exact supplied id", "replacement_value": "typed value"}
  ],
  "row_plans": [
    {"source_id": "exact id", "target_relation": "canonical relation"}
  ]
}
"""


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def with_contracts_and_anchors(block: Mapping) -> Dict:
    result = v4.with_contracts(block)
    result["anchor_catalog"] = anchors.build_anchor_catalog(
        result["sources"], result["target_entity"]
    )
    return result


def plan_payload(
    block: Mapping,
    protected_authors: Sequence[str],
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = {
        "required_target_entity": block["target_entity"],
        "required_replacement_pronouns": block["replacement_pronouns"],
        "protected_authors": list(protected_authors),
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "source_rows": [
            {
                "source_id": source["source_id"],
                "C11": {"question": source["question"], "answer": source["answer"]},
                "contract": source["contract"],
                "available_anchor_groups": anchors.catalog_for_prompt(
                    block["anchor_catalog"], source["source_id"]
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


def validate_plan(
    block: Mapping, generated: Mapping, protected_authors: Sequence[str], seed: int
) -> Dict:
    target = block["target_entity"]
    if anchors.normalise(generated.get("target_entity", "")) != anchors.normalise(target):
        raise ValueError(f"target_entity must equal {target!r}")
    replacement = generated.get("replacement_entity")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("replacement_entity must be non-empty")
    replacement = replacement.strip()
    if anchors.normalise(target) in anchors.normalise(replacement):
        raise ValueError("replacement_entity contains the protected target name")
    if anchors.normalise(replacement) in {
        anchors.normalise(author) for author in protected_authors
    }:
        raise ValueError("replacement_entity collides with a protected author")
    if anchors.normalise(generated.get("replacement_pronouns", "")) != anchors.normalise(
        block["replacement_pronouns"]
    ):
        raise ValueError(
            "replacement_pronouns must preserve the frozen C11 class "
            f"{block['replacement_pronouns']!r}"
        )
    summary = generated.get("profile_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("profile_summary must be non-empty")

    source_ids = [source["source_id"] for source in block["sources"]]
    raw_rows = v4.exact_rows(generated.get("row_plans"), source_ids, "row_plans")
    replacement_map = anchors.validate_replacement_map(
        block["anchor_catalog"], generated.get("anchor_replacements")
    )
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    row_plans, cells, errors = {}, {}, []
    for query_index, source_id in enumerate(source_ids):
        relation = raw_rows[source_id].get("target_relation")
        if not isinstance(relation, str) or not relation.strip():
            errors.append(f"{source_id} missing target_relation")
            continue
        try:
            rendered = anchors.render_row(
                source_by_id[source_id], target, replacement, relation.strip(),
                block["anchor_catalog"], replacement_map, v4.contracts,
            )
            applied = rendered["question_edits"] + rendered["answer_edits"]
            replacement_fact = (
                max((item["new"] for item in applied), key=len)
                if applied else replacement
            )
            placebo = v4.contracts.render_professional_placebo(
                source_by_id[source_id]["contract"], target, replacement, query_index,
                assignment_flip=((int(block["block_id"]) + query_index + seed) % 2 == 1),
            )
            for name in ("C10", "C00"):
                failures = v4.contracts.contract_errors(
                    placebo[name], source_by_id[source_id]["contract"]
                )
                if failures:
                    raise ValueError(f"{name}: " + "; ".join(failures))
            row_plans[source_id] = {
                "source_id": source_id,
                "target_relation": relation.strip(),
                "replacement_fact": replacement_fact,
                "question_edits": rendered["question_edits"],
                "answer_edits": rendered["answer_edits"],
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
        raise ValueError(" | ".join(errors[:20]))
    return {
        "target_entity": target,
        "replacement_entity": replacement,
        "replacement_pronouns": block["replacement_pronouns"],
        "profile_summary": summary.strip(),
        "anchor_replacements": replacement_map,
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "row_plans": row_plans,
        "cells": cells,
    }


def assemble_block(block: Mapping, plan: Mapping, verdicts: Mapping, args) -> Dict:
    assembled = v4.assemble_block(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-anchor-v5-block-{int(block['block_id']):02d}:"
        f"seed-{args.seed}"
    )
    assembled.update({
        "design_version": DESIGN_VERSION,
        "profile_id": profile_id,
        "anchor_catalog_digest": plan["anchor_catalog_digest"],
        "anchor_replacements": plan["anchor_replacements"],
    })
    for record in assembled["records"]:
        record["design_version"] = DESIGN_VERSION
        record["profile_id"] = profile_id
        record["generation"]["design"] = DESIGN_VERSION
        record["generation"]["surface_renderer"] = "deterministic-anchor-id-v5"
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{record['source_id']}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


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
            any(record.get("semantic_judge", {}).get(field) is not True
                for field in v4.JUDGE_FIELDS)
            for record in records
        ):
            return None
        return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def request_args(args, *, judge: bool = False):
    result = copy.copy(args)
    result.model = args.judge_model if judge else args.model
    result.temperature = args.judge_temperature if judge else args.temperature
    result.retries = args.request_retries
    return result


def generate_block(generation_client, judge_client, args, block, protected, state_dir):
    print(f"start_block block={block['block_id']} author={block['target_entity']}", flush=True)
    feedback, previous, last_error = "", None, None
    for attempt in range(args.stage_retries):
        print(
            f"start_stage block={block['block_id']} stage=anchor_plan_judge "
            f"attempt={attempt + 1}/{args.stage_retries}", flush=True,
        )
        candidate = None
        try:
            candidate = v2.request_json(
                generation_client, request_args(args), PLAN_PROMPT,
                plan_payload(block, protected, feedback, previous),
                f"V5 block {block['block_id']} anchor plan",
            )
            plan = validate_plan(block, candidate, protected, args.seed)
            judgement = v2.request_json(
                judge_client, request_args(args, judge=True), v4.JUDGE_PROMPT,
                v4.judge_payload(block, plan),
                f"V5 block {block['block_id']} semantic audit",
            )
            ids = [source["source_id"] for source in block["sources"]]
            verdicts = v4.validate_judgement(judgement, ids)
            result = assemble_block(block, plan, verdicts, args)
            print(f"stage_ready block={block['block_id']} stage=anchor_plan_judge", flush=True)
            return result
        except Exception as exc:
            last_error = exc
            feedback, previous = str(exc), candidate
            write_json(
                state_dir / f"block_{int(block['block_id']):02d}.attempt_{attempt + 1:02d}.json",
                {"error": feedback, "candidate": previous},
            )
            print(
                f"stage_reject block={block['block_id']} stage=anchor_plan_judge "
                f"attempt={attempt + 1}/{args.stage_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block['block_id']} failed after {args.stage_retries} attempts: {last_error}"
    ) from last_error


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
    parser.add_argument("--judge-temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=16000)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--stage-retries", type=int, default=8)
    parser.add_argument("--json-mode", choices=("auto", "required", "prompt"), default="auto")
    args = parser.parse_args()
    args.judge_model = args.judge_model or args.model
    args.judge_base_url = args.judge_base_url or args.base_url
    args.judge_api_key_env = args.judge_api_key_env or args.api_key_env
    api_key = os.environ.get(args.api_key_env)
    judge_key = os.environ.get(args.judge_api_key_env)
    if not api_key or not judge_key:
        raise SystemExit("Set generation and judge API keys before V5 generation")
    from openai import OpenAI

    manifest = v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(f"manifest split={manifest.get('split')} != {args.split}")
    blocks = [
        with_contracts_and_anchors(block)
        for block in v2.group_author_blocks(v2.load_sources(args.split), manifest)
    ]
    protected = [block["target_entity"] for block in blocks]
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
        with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
            futures = {
                executor.submit(
                    generate_block, generation_client, judge_client, args,
                    block, protected, state_dir,
                ): block for block in pending
            }
            for future in as_completed(futures):
                block = futures[future]
                try:
                    result = future.result()
                    write_json(state_dir / f"block_{int(block['block_id']):02d}.json", result)
                    completed[int(block["block_id"])] = result
                    print(
                        f"valid_block={len(completed)}/{len(blocks)} "
                        f"block={block['block_id']} author={block['target_entity']}", flush=True,
                    )
                except Exception as exc:
                    failures.append((block["block_id"], block["target_entity"], str(exc)))
    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        raise SystemExit("V5 incomplete; attempt artifacts were retained for diagnosis")

    ordered = [completed[int(block["block_id"])] for block in blocks]
    records = [record for block in ordered for record in block["records"]]
    expected = len(blocks) * int(manifest["block_size"])
    if len(records) != expected:
        raise SystemExit(f"record count {len(records)} != {expected}")
    ciru.write_ciru_jsonl(args.output, records)
    write_json(profiles_output, {
        "design_version": DESIGN_VERSION,
        "split": args.split,
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "profiles": [
            {key: block[key] for key in (
                "block_id", "profile_id", "target_entity", "replacement_entity",
                "replacement_pronouns", "anchor_catalog_digest",
                "anchor_replacements", "author_plan", "placebo_plan",
            )}
            for block in ordered
        ],
    })
    print(f"author_blocks={len(ordered)} units={len(records)} cells={4 * len(records)}")
    print(f"output={args.output}")
    print(f"profiles={profiles_output}")


if __name__ == "__main__":
    main()
