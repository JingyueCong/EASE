#!/usr/bin/env python3
"""Generate TOFU author-level 2x2 data through a typed causal IR.

V4 deliberately preserves V1/V2/V3.  The model may propose only atomic exact-
span edits for C01.  C01 surface text and both professional placebo cells are
rendered and validated by code; a separate semantic pass can only accept or
reject the completed author block.
"""

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
V4_CONTRACT_PATH = ROOT / "ULD/uld/data/tofu_contract_v4.py"
CIRU_PATH = ROOT / "ULD/uld/data/ciru.py"
DESIGN_VERSION = "tofu-author-typed-v4.1"
JUDGE_FIELDS = (
    "target_relation_match",
    "target_fact_changed",
    "profile_consistent",
    "natural_surface",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load_module("tofu_typed_v4_v2_shared", V2_PATH)
contracts = load_module("tofu_typed_v4_contracts", V4_CONTRACT_PATH)
ciru = load_module("tofu_typed_v4_ciru", CIRU_PATH)


PLAN_PROMPT = """You create a coherent counterfactual twin for one 20-row TOFU
author block. C11 text is immutable. Return an edit plan, never rewritten QA.

For every source row:
- infer the exact target relation;
- replace the smallest source-specific factual span(s) needed to create a
  substantively different fact for the same relation;
- use only exact, case-sensitive old spans copied from C11 question/answer;
- keep each old/new edit atomic (name, date, number, title, location, genre,
  award, or short descriptive attribute), never a whole sentence/answer;
- preserve polarity, answerability, list structure, fact count, and grammar;
- do not submit author-name edits: the deterministic renderer replaces every
  exact target-author occurrence after applying your factual edits;
- for unavailable answers, preserve unavailability and change only identity or
  an identity-binding cue; for identity questions, no factual edit is required;
- make all 20 edits one internally consistent replacement-author biography;
- never put the protected target author in a new span.

If validation_feedback and previous_candidate are supplied, return the complete
20-row object with every reported defect repaired. Do not change valid rows
gratuitously.

Return JSON only:
{
  "target_entity": "required target",
  "replacement_entity": "one new author name",
  "profile_summary": "coherent replacement biography",
  "row_plans": [{
    "source_id": "exact id",
    "target_relation": "canonical relation",
    "question_edits": [{"old": "exact old span", "new": "new span"}],
    "answer_edits": [{"old": "exact old span", "new": "new span"}]
  }]
}
"""


JUDGE_PROMPT = """You are a conservative semantic auditor for a TOFU causal
factorial dataset. Text has already passed deterministic exact-edit, response-
contract, identity, length, type, and placebo checks. Do not rewrite anything.

For each row set all booleans true only if:
- target_relation_match: C11 and C01 ask the same relation;
- target_fact_changed: C01 changes the source-specific factual object, except
  that identity-only and unavailable relations are correctly handled;
- profile_consistent: C01 agrees with the same replacement-author biography;
- natural_surface: the atomic edits leave fluent, grammatical QA.

Return JSON only:
{"verdicts": [{"source_id":"exact id", "target_relation_match":true,
"target_fact_changed":true, "profile_consistent":true,
"natural_surface":true, "reason":"brief evidence"}]}
"""


def normalise(value: object) -> str:
    return " ".join(str(value).casefold().split())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def exact_rows(value: object, source_ids: Sequence[str], label: str) -> Dict[str, Dict]:
    return v2.indexed_items(value, label, source_ids)


def with_contracts(block: Mapping) -> Dict:
    target = block["target_entity"]
    return {
        **block,
        "sources": [
            {
                **source,
                "contract": contracts.derive_contract(
                    source["question"], source["answer"], target
                ),
            }
            for source in block["sources"]
        ],
    }


def request_args(args, *, judge: bool = False):
    result = copy.copy(args)
    result.model = args.judge_model if judge else args.model
    result.temperature = args.judge_temperature if judge else args.temperature
    result.retries = args.request_retries
    return result


def plan_payload(
    block: Mapping,
    protected_authors: Sequence[str],
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = {
        "required_target_entity": block["target_entity"],
        "protected_authors": list(protected_authors),
        "source_rows": [
            {
                "source_id": source["source_id"],
                "C11": {
                    "question": source["question"],
                    "answer": source["answer"],
                },
                "contract": source["contract"],
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
    if normalise(generated.get("target_entity", "")) != normalise(target):
        raise ValueError(f"target_entity must equal {target!r}")
    replacement = generated.get("replacement_entity")
    if not isinstance(replacement, str) or not replacement.strip():
        raise ValueError("replacement_entity must be non-empty")
    replacement = replacement.strip()
    if normalise(target) in normalise(replacement):
        raise ValueError("replacement_entity contains the protected target name")
    if normalise(replacement) in {normalise(x) for x in protected_authors}:
        raise ValueError("replacement_entity collides with a protected author")
    summary = generated.get("profile_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("profile_summary must be non-empty")

    ids = [source["source_id"] for source in block["sources"]]
    raw_plans = exact_rows(generated.get("row_plans"), ids, "row_plans")
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    plans, cells = {}, {}
    errors = []
    for query_index, source_id in enumerate(ids):
        item = raw_plans[source_id]
        relation = item.get("target_relation")
        if not isinstance(relation, str) or not relation.strip():
            errors.append(f"{source_id} missing target_relation")
            continue
        plan = {
            "source_id": source_id,
            "target_relation": relation.strip(),
            "question_edits": item.get("question_edits", []),
            "answer_edits": item.get("answer_edits", []),
        }
        try:
            c01 = contracts.render_target_counterfactual(
                source_by_id[source_id], target, replacement, plan
            )
            factual_new_spans = [
                edit.get("new", "")
                for edit in plan["answer_edits"] + plan["question_edits"]
                if isinstance(edit, Mapping)
                and isinstance(edit.get("new"), str)
                and normalise(edit.get("new")) != normalise(replacement)
                and normalise(edit.get("old")) != normalise(target)
            ]
            factual_new_spans = [
                value for value in factual_new_spans
                if normalise(value) in normalise(f"{c01['question']} {c01['answer']}")
            ]
            plan["replacement_fact"] = (
                max(factual_new_spans, key=len) if factual_new_spans else replacement
            )
            placebo = contracts.render_professional_placebo(
                source_by_id[source_id]["contract"],
                target,
                replacement,
                query_index,
                assignment_flip=((int(block["block_id"]) + query_index + seed) % 2 == 1),
            )
            for name in ("C10", "C00"):
                contract_failures = contracts.contract_errors(
                    placebo[name], source_by_id[source_id]["contract"]
                )
                if contract_failures:
                    raise ValueError(f"{name}: " + "; ".join(contract_failures))
            cells[source_id] = {
                "C01": c01,
                "C10": placebo["C10"],
                "C00": placebo["C00"],
                "placebo_relation": placebo["placebo_relation"],
                "placebo_target_value": placebo["target_value"],
                "placebo_replacement_value": placebo["replacement_value"],
                "placebo_rationale": placebo["placebo_rationale"],
            }
            plans[source_id] = plan
        except ValueError as exc:
            errors.append(f"{source_id}: {exc}")
    if errors:
        raise ValueError(" | ".join(errors[:20]))
    return {
        "target_entity": target,
        "replacement_entity": replacement,
        "profile_summary": summary.strip(),
        "row_plans": plans,
        "cells": cells,
    }


def judge_payload(block: Mapping, plan: Mapping) -> Dict:
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    return {
        "author_profile": {
            "target_entity": plan["target_entity"],
            "replacement_entity": plan["replacement_entity"],
            "profile_summary": plan["profile_summary"],
        },
        "rows": [
            {
                "source_id": source_id,
                "target_relation": row_plan["target_relation"],
                "replacement_fact": row_plan["replacement_fact"],
                "C11": {
                    "question": source_by_id[source_id]["question"],
                    "answer": source_by_id[source_id]["answer"],
                },
                "C01": plan["cells"][source_id]["C01"],
            }
            for source_id, row_plan in plan["row_plans"].items()
        ],
    }


def validate_judgement(generated: Mapping, source_ids: Sequence[str]) -> Dict[str, Dict]:
    verdicts = exact_rows(generated.get("verdicts"), source_ids, "verdicts")
    failures = []
    for source_id, verdict in verdicts.items():
        for field in JUDGE_FIELDS:
            if verdict.get(field) is not True:
                reason = str(verdict.get("reason", "")).strip()
                failures.append(f"{source_id}:{field}({reason})")
        if not isinstance(verdict.get("reason"), str) or not verdict["reason"].strip():
            failures.append(f"{source_id}:reason")
    if failures:
        raise ValueError("semantic judge rejected " + ", ".join(failures[:20]))
    return verdicts


def assemble_block(
    block: Mapping, plan: Mapping, verdicts: Mapping[str, Mapping], args
) -> Dict:
    profile_id = (
        f"{args.split}:author-typed-v4-block-{int(block['block_id']):02d}:"
        f"seed-{args.seed}"
    )
    records = []
    target_facts = []
    placebo_facts = []
    for query_index, source in enumerate(block["sources"]):
        source_id = source["source_id"]
        row_plan = plan["row_plans"][source_id]
        rendered = plan["cells"][source_id]
        cells = {
            "C11": {"question": source["question"], "answer": source["answer"]},
            "C01": rendered["C01"],
            "C10": rendered["C10"],
            "C00": rendered["C00"],
        }
        target_fact_id, placebo_fact_id = f"T{query_index:02d}", f"P{query_index:02d}"
        target_facts.append(
            {
                "fact_id": target_fact_id,
                "source_id": source_id,
                "relation": row_plan["target_relation"],
                "value": row_plan["replacement_fact"],
                "question_edits": row_plan["question_edits"],
                "answer_edits": row_plan["answer_edits"],
            }
        )
        placebo_facts.append(
            {
                "fact_id": placebo_fact_id,
                "source_id": source_id,
                "relation": rendered["placebo_relation"],
                "target_value": rendered["placebo_target_value"],
                "replacement_value": rendered["placebo_replacement_value"],
            }
        )
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
            "placebo_relation": rendered["placebo_relation"],
            "identity_binding": {
                "C11": plan["target_entity"], "C01": plan["replacement_entity"],
                "C10": plan["target_entity"], "C00": plan["replacement_entity"],
            },
            "invariants": {
                "task": "TOFU factual author QA",
                "style": f"typed-contract:{source['contract']['response_mode']}",
                "difficulty": "matched author relation",
                "answer_format": source["contract"]["answer_format"],
            },
            "response_contract": source["contract"],
            "supporting_fact_ids": [target_fact_id],
            "placebo_supporting_fact_ids": [placebo_fact_id],
            "cells": cells,
            "changed_evidence": [row_plan["replacement_fact"]],
            "placebo_rationale": rendered["placebo_rationale"],
            "semantic_judge": verdicts[source_id],
            "generation": {
                "backend": "openai-compatible",
                "model": args.model,
                "judge_model": args.judge_model,
                "design": DESIGN_VERSION,
                "surface_renderer": "deterministic-exact-edit-v4",
            },
        }
        schema_errors = ciru.validate_ciru_unit(record)
        if schema_errors:
            raise ValueError(f"{source_id}: " + "; ".join(schema_errors))
        record["audit"] = ciru.audit_ciru_unit(record)
        records.append(record)
    return {
        "design_version": DESIGN_VERSION,
        "block_id": int(block["block_id"]),
        "profile_id": profile_id,
        "target_entity": plan["target_entity"],
        "replacement_entity": plan["replacement_entity"],
        "author_plan": {"summary": plan["profile_summary"], "facts": target_facts},
        "placebo_plan": {"facts": placebo_facts, "library": contracts.DESIGN_VERSION},
        "records": records,
    }


def load_valid_block(path: Path, block: Mapping) -> Dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = {source["source_id"] for source in block["sources"]}
        if data.get("design_version") != DESIGN_VERSION:
            return None
        records = data.get("records")
        if not isinstance(records, list) or len(records) != len(expected):
            return None
        if {record.get("source_id") for record in records} != expected:
            return None
        for record in records:
            if ciru.validate_ciru_unit(record):
                return None
            if any(record.get("semantic_judge", {}).get(field) is not True
                   for field in JUDGE_FIELDS):
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
    print(f"start_block block={block['block_id']} author={block['target_entity']}", flush=True)
    feedback, previous, last_error = "", None, None
    for attempt in range(args.stage_retries):
        print(
            f"start_stage block={block['block_id']} stage=plan_judge "
            f"attempt={attempt + 1}/{args.stage_retries}", flush=True
        )
        try:
            candidate = v2.request_json(
                generation_client,
                request_args(args),
                PLAN_PROMPT,
                plan_payload(block, protected_authors, feedback, previous),
                f"V4 block {block['block_id']} edit plan",
            )
            plan = validate_plan(block, candidate, protected_authors, args.seed)
            judgement = v2.request_json(
                judge_client,
                request_args(args, judge=True),
                JUDGE_PROMPT,
                judge_payload(block, plan),
                f"V4 block {block['block_id']} semantic audit",
            )
            ids = [source["source_id"] for source in block["sources"]]
            verdicts = validate_judgement(judgement, ids)
            assembled = assemble_block(block, plan, verdicts, args)
            print(f"stage_ready block={block['block_id']} stage=plan_judge", flush=True)
            return assembled
        except Exception as exc:
            last_error = exc
            feedback = str(exc)
            previous = candidate if "candidate" in locals() else None
            write_json(
                state_dir / f"block_{int(block['block_id']):02d}.attempt_{attempt + 1:02d}.json",
                {"error": feedback, "candidate": previous},
            )
            print(
                f"stage_reject block={block['block_id']} stage=plan_judge "
                f"attempt={attempt + 1}/{args.stage_retries} error={exc}", flush=True
            )
    raise RuntimeError(
        f"block {block['block_id']} failed after {args.stage_retries} complete attempts: "
        f"{last_error}"
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
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generation")
    if not judge_key:
        raise SystemExit(f"Set {args.judge_api_key_env} before judging")
    from openai import OpenAI

    manifest = v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(f"manifest split={manifest.get('split')} != {args.split}")
    blocks = [
        with_contracts(block)
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
                ): block
                for block in pending
            }
            for future in as_completed(futures):
                block = futures[future]
                try:
                    result = future.result()
                    write_json(
                        state_dir / f"block_{int(block['block_id']):02d}.json", result
                    )
                    completed[int(block["block_id"])] = result
                    print(
                        f"valid_block={len(completed)}/{len(blocks)} "
                        f"block={block['block_id']} author={block['target_entity']}", flush=True
                    )
                except Exception as exc:
                    failures.append((block["block_id"], block["target_entity"], str(exc)))
    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        raise SystemExit("V4 incomplete; attempt artifacts were retained for diagnosis")

    ordered = [completed[int(block["block_id"])] for block in blocks]
    records = [record for block in ordered for record in block["records"]]
    expected = len(blocks) * int(manifest["block_size"])
    if len(records) != expected:
        raise SystemExit(f"record count {len(records)} != {expected}")
    ciru.write_ciru_jsonl(args.output, records)
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
                {key: block[key] for key in (
                    "block_id", "profile_id", "target_entity", "replacement_entity",
                    "author_plan", "placebo_plan",
                )}
                for block in ordered
            ],
        },
    )
    print(f"author_blocks={len(ordered)} units={len(records)} cells={4 * len(records)}")
    print(f"output={args.output}")
    print(f"profiles={profiles_output}")


if __name__ == "__main__":
    main()
