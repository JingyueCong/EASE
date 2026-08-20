#!/usr/bin/env python3
"""Generate TOFU author factorial data with matched C01 questions and answers.

V5.7 keeps V5.6's independently audited direct core-fact ledger, but fixes a
semantic omission exposed by human audit: target-specific premises embedded in
the question (book titles, dates, places, awards, and identity descriptions)
must be rewritten together with the answer.  Every C01 row is therefore one
joint question/answer intervention.  FullAnswer and V1-V5.6 remain untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
V56_PATH = SCRIPT_DIR / "generate_tofu_author_direct_v5_6.py"
DESIGN_VERSION = "tofu-author-joint-contrast-v5.7"
CONTRAST_SCHEMA_VERSION = "joint-question-answer-core-v1"
SURFACE_RENDERER = "joint-matched-c01-question-answer-v5.7"
MAPPING_SCOPE = "row-local-joint-question-answer-conditioned-on-frozen-ledger"
RESPONSE_CONTRACT_POLICY = "semantic-joint-compatible-v5.7"
JUDGE_FIELDS = (
    "target_relation_match",
    "question_premise_profile_consistent",
    "answer_satisfies_question",
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


v56 = load_module("tofu_author_joint_v57_v56", V56_PATH)
v55 = v56.v55
v53 = v56.v53
v52 = v56.v52
anchors = v56.anchors
ciru = v56.ciru

BASE_V55_MATERIALIZE_PLAN = v55.materialize_plan
BASE_V56_ASSEMBLE_BLOCK = v56.assemble_block


PROFILE_PROMPT = v56.PROFILE_PROMPT.replace(
    "- identity/unavailable rows use contrast_status=policy and follow the declared\n"
    "  policy without pretending a fact changed;",
    "- identity/unavailable rows use contrast_status=policy and do not pretend "
    "a factual contrast was estimated, but they must still provide a complete "
    "replacement_core_fact and identical replacement_fact containing every "
    "replacement-author premise needed for a coherent matched question and answer;",
)

PROFILE_JUDGE_PROMPT = v56.PROFILE_JUDGE_PROMPT.replace(
    "Also judge whether all replacement facts are mutually coherent.",
    "For policy rows, also verify that replacement_core_fact supplies a coherent "
    "replacement identity or availability proposition sufficient for a matched "
    "question and answer. Also judge whether all replacement facts are mutually "
    "coherent.",
)

JOINT_PROMPT = """Rewrite one TOFU C01 question and answer jointly from a
frozen, independently approved direct-core-fact ledger.

The causal relation is fixed, but all target-specific premises inside the
question must agree with the replacement author and replacement core fact.
This includes author identity, birthplace, date, book title, award object,
genre, count, location, and other descriptors.  Do not leave an original book,
place, date, or award in the question while answering about a different one.

Hard rules:
- copy source_id, target_relation, ledger_fact_key, and
  ledger_replacement_fact exactly;
- return both a complete c01_question and complete replacement_answer;
- C01 must ask the same semantic relation as C11, with comparable specificity;
- jointly rewrite every premise needed so replacement_answer is a direct,
  truthful answer to c01_question under the replacement profile;
- preserve cardinality and semantic object type: one book remains one book,
  an award-for-book relation remains award-for-book, a date remains a date,
  and a number remains a number;
- explicitly express replacement_core_fact and do not retain a contradictory
  source-core fact or target-specific premise;
- use the block replacement ledger to keep recurring books, awards, places,
  dates, and identity attributes consistent across rows;
- obey surface_constraints.required_mode_family and the semantic response
  contract; preserve negative, unavailable, or qualified status when declared;
- do not mention the target author, generation process, fictionality, controls,
  counterfactuals, or missing documentation unless that wording is inherited
  from the immutable benchmark and required by the response mode;
- question_anchor_rewrites must list each target-specific question premise that
  changed. Identity replacement may be listed but is not sufficient when the
  source question also contains a factual premise;
- question_rewrite_rationale must state why the rewritten question and answer
  form a matched pair. It is provenance, not text to place in C01.

When validation_feedback and previous_candidate are present, repair only the
reported row. Return the entire object and never change frozen fields.

Return JSON only:
{"source_id":"exact id","target_relation":"copied relation",
 "ledger_fact_key":"copied key","ledger_replacement_fact":"copied fact",
 "c01_question":"complete matched question",
 "replacement_answer":"complete matched answer",
 "question_anchor_rewrites":[{"source_text":"old premise",
 "replacement_text":"new premise","reason":"semantic role"}],
 "question_rewrite_rationale":"why Q and A match"}
"""

ROW_JUDGE_PROMPT = """Independently audit a completed 20-row TOFU C01 block.
For each row compare immutable C11, the frozen source/replacement core facts,
the full replacement-author ledger, and the jointly rewritten C01.

Decide all six properties independently:
- target_relation_match: C11 and C01 ask the same relation and preserve its
  argument structure/cardinality;
- question_premise_profile_consistent: every named book, award, place, date,
  number, identity descriptor, and other premise in C01 belongs to or is
  compatible with the replacement profile;
- answer_satisfies_question: C01.answer directly answers C01.question, including
  any specific book/title/date/place named in the question;
- target_fact_changed: the required factual object changes, except for an
  explicitly declared identity/unavailable policy row;
- profile_consistent: C01 agrees with the frozen row ledger and the rest of the
  replacement-author block;
- natural_surface: both C01 question and answer are fluent and grammatical.

Reject a row if the answer is plausible in isolation but does not answer its
actual rewritten question. Return only the minimal failing row set.

Return JSON only:
{"verdicts":[{"source_id":"exact id","target_relation_match":true,
 "question_premise_profile_consistent":true,"answer_satisfies_question":true,
 "target_fact_changed":true,"profile_consistent":true,
 "natural_surface":true,"reason":"state the matched premises and answer"}]}
"""


def _normalise(value: object) -> str:
    return anchors.normalise(str(value))


def validate_profile(
    block: Mapping, generated: Mapping, protected_authors: Sequence[str]
) -> Dict:
    """Retain explicit replacement propositions for V5.6 policy rows."""
    profile = v56._V57_BASE_VALIDATE_PROFILE(
        block, generated, protected_authors
    )
    raw_by_id = v52.v4.exact_rows(
        generated.get("fact_ledger"),
        [source["source_id"] for source in block["sources"]],
        "fact_ledger",
    )
    changed = False
    for entry in profile["fact_ledger"]:
        if entry["fact_change_required"]:
            continue
        source_id = entry["source_id"]
        raw = raw_by_id[source_id]
        source_core = str(raw.get("source_core_fact", "")).strip()
        replacement_core = str(raw.get("replacement_core_fact", "")).strip()
        replacement_fact = str(raw.get("replacement_fact", "")).strip()
        if not source_core or not replacement_core or not replacement_fact:
            raise ValueError(
                f"{source_id} policy row needs complete source and replacement core facts"
            )
        if replacement_fact != replacement_core:
            raise ValueError(
                f"{source_id} replacement_fact must copy replacement_core_fact"
            )
        if _normalise(block["target_entity"]) in _normalise(replacement_core):
            raise ValueError(f"{source_id} replacement core leaks target author")
        entry.update({
            "fact_key": str(raw.get("fact_key", "")).strip()
            or f"policy:{source_id}",
            "source_core_fact": source_core,
            "replacement_core_fact": replacement_core,
            "replacement_fact": replacement_fact,
        })
        changed = True
    if changed:
        profile["fact_ledger_by_source"] = {
            entry["source_id"]: entry for entry in profile["fact_ledger"]
        }
        profile["ledger_digest"] = v52.digest_json(profile["fact_ledger"])
        profile["profile_digest"] = v52.digest_json({
            "target_entity": profile["target_entity"],
            "replacement_entity": profile["replacement_entity"],
            "replacement_pronouns": profile["replacement_pronouns"],
            "profile_summary": profile["profile_summary"],
            "ledger_digest": profile["ledger_digest"],
            "anchor_catalog_digest": block["anchor_catalog"]["digest"],
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        })
    return profile


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = v56.row_payload(block, source, profile, feedback, previous)
    payload["identity_only_question_reference"] = payload.pop("c01_question")
    payload["source_answer"] = source["answer"]
    payload["replacement_profile_ledger"] = [
        {
            "source_id": entry["source_id"],
            "target_relation": entry["target_relation"],
            "replacement_core_fact": entry["replacement_core_fact"],
            "intervention_policy": entry["intervention_policy"],
        }
        for entry in profile["fact_ledger"]
    ]
    payload["joint_rewrite_policy"] = {
        "same_relation_schema": True,
        "rewrite_question_specific_premises": True,
        "question_and_answer_must_be_mutually_entailed": True,
        "preserve_cardinality": True,
    }
    return payload


def _canonical_rewrites(value: object) -> list[Dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("question_anchor_rewrites must be a list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"question_anchor_rewrites[{index}] must be an object")
        canonical = {
            key: str(item.get(key, "")).strip()
            for key in ("source_text", "replacement_text", "reason")
        }
        if not all(canonical.values()):
            raise ValueError(
                f"question_anchor_rewrites[{index}] has an empty field"
            )
        result.append(canonical)
    return result


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
) -> Dict:
    source_id = source["source_id"]
    entry = profile["fact_ledger_by_source"][source_id]
    frozen = {
        "source_id": source_id,
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
    }
    for field, expected in frozen.items():
        if generated.get(field) != expected:
            raise ValueError(f"{field} must exactly copy the frozen payload")

    raw_question = generated.get("c01_question")
    raw_answer = generated.get("replacement_answer")
    if not isinstance(raw_question, str) or not raw_question.strip():
        raise ValueError("c01_question must be a non-empty string")
    if not isinstance(raw_answer, str) or not raw_answer.strip():
        raise ValueError("replacement_answer must be a non-empty string")
    question = " ".join(raw_question.replace("\n", " ").split()).strip()
    answer = " ".join(raw_answer.replace("\n", " ").split()).strip()
    rationale = str(generated.get("question_rewrite_rationale", "")).strip()
    if not rationale:
        raise ValueError("question_rewrite_rationale must be non-empty")
    rewrites = _canonical_rewrites(generated.get("question_anchor_rewrites"))
    cell = {"question": question, "answer": answer}
    for rewrite in rewrites:
        if _normalise(rewrite["source_text"]) not in _normalise(source["question"]):
            raise ValueError(
                "question_anchor_rewrites.source_text must occur in C11 question"
            )
        if _normalise(rewrite["replacement_text"]) not in _normalise(question):
            raise ValueError(
                "question_anchor_rewrites.replacement_text must occur in C01 question"
            )

    leak = v55._target_alias_leaks(
        f"{question} {answer}", block["target_entity"]
    )
    if leak:
        raise ValueError(f"C01 still contains target alias {leak!r}")
    if _normalise(block["target_entity"]) in _normalise(source["question"]):
        if _normalise(profile["replacement_entity"]) not in _normalise(question):
            raise ValueError("C01 question loses explicit replacement identity")
    marker = v55._introduced_control_status_marker(source, cell)
    if marker:
        raise ValueError(f"C01 exposes control status with marker: {marker}")
    contract_result = v55.semantic_contract_result(cell, source["contract"])
    if contract_result["errors"]:
        raise ValueError(
            "C01 semantic response contract: "
            + "; ".join(contract_result["errors"])
        )

    replacement_tokens = v53.content_tokens(entry["replacement_fact"])
    c01_tokens = v53.content_tokens(f"{question} {answer}")
    identity_tokens = v53.content_tokens(profile["replacement_entity"])
    shared_replacement = replacement_tokens & c01_tokens
    if entry["fact_change_required"]:
        shared_replacement -= identity_tokens
    if not shared_replacement:
        raise ValueError(
            "joint C01 must express content from ledger_replacement_fact"
        )
    if entry["fact_change_required"]:
        identity_question = v55.c01_question(block, source, profile)
        identity_answer = v55.replace_identity(
            source["answer"], block["target_entity"], profile["replacement_entity"]
        )
        if (
            _normalise(question) == _normalise(identity_question)
            and _normalise(answer) == _normalise(identity_answer)
        ):
            raise ValueError("C01 changes only author identity, not the target fact")

    return {
        **frozen,
        "c01_question": question,
        "replacement_answer": answer,
        "question_anchor_rewrites": rewrites,
        "question_rewrite_rationale": rationale,
        "cell": cell,
        "fact_change_required": bool(entry["fact_change_required"]),
        "render_mode": "joint_question_answer_intervention",
        "ledger_digest": profile["ledger_digest"],
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "response_contract_warnings": contract_result["warnings"],
    }


def candidate_for_prompt(candidate: Mapping) -> Dict:
    return {
        key: candidate[key]
        for key in (
            "source_id", "target_relation", "ledger_fact_key",
            "ledger_replacement_fact", "c01_question", "replacement_answer",
            "question_anchor_rewrites", "question_rewrite_rationale",
        )
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
        cached = v55.load_cached_row(path, block, source, profile)
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
                JOINT_PROMPT,
                row_payload(block, source, profile, feedback, previous),
                f"V5.7 block {block_id} joint row {source_id}",
            )
            validated = validate_row_candidate(block, source, profile, candidate)
            validated["mapping_attempt"] = attempt
            validated["repair_generation"] = 1 if force else 0
            v55.write_row_checkpoint(
                path, profile, validated, attempt, validated["repair_generation"]
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
        f"block {block_id} joint row {source_id} failed after "
        f"{args.row_retries} attempts: {last_error}"
    ) from last_error


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    plan = BASE_V55_MATERIALIZE_PLAN(block, profile, candidates, seed)
    by_source = {source["source_id"]: source for source in block["sources"]}
    for source_id, row_plan in plan["row_plans"].items():
        candidate = candidates[source_id]
        row_plan.update({
            "render_mode": "joint_question_answer_intervention",
            "question_edits": [{
                "anchor_id": "complete-c01-question",
                "group_id": f"joint-question:{source_id}",
                "old": by_source[source_id]["question"],
                "new": candidate["c01_question"],
            }],
            "question_anchor_rewrites": candidate["question_anchor_rewrites"],
            "question_rewrite_rationale": candidate[
                "question_rewrite_rationale"
            ],
        })
    return plan


def parse_judgement(
    generated: Mapping, block: Mapping, plan: Mapping
) -> tuple[Dict[str, Dict], Dict[str, str]]:
    source_ids = [source["source_id"] for source in block["sources"]]
    verdicts = v52.v4.exact_rows(generated.get("verdicts"), source_ids, "verdicts")
    failures: Dict[str, str] = {}
    for source_id, raw in verdicts.items():
        verdict = dict(raw)
        if not plan["row_plans"][source_id]["fact_change_required"]:
            verdict["raw_target_fact_changed"] = verdict.get("target_fact_changed")
            verdict["target_fact_changed"] = True
            verdict["deterministic_overrides"] = ["target_fact_changed"]
        failed = [field for field in JUDGE_FIELDS if verdict.get(field) is not True]
        reason = str(verdict.get("reason", "")).strip()
        if not reason:
            failed.append("reason")
        if failed:
            failures[source_id] = (
                f"semantic fields {','.join(failed)}: {reason or 'missing reason'}"
            )
        verdicts[source_id] = verdict
    return verdicts, failures


def judge_plan_in_batches(
    judge_client,
    args,
    block: Mapping,
    plan: Mapping,
    judge_round: int,
) -> tuple[Dict, list[Dict]]:
    """Collect an exact verdict for every row without one oversized response."""
    payload = v53.judge_payload(block, plan)
    rows = list(payload["rows"])
    batch_size = max(int(args.judge_batch_size), 1)
    verdicts = []
    provenance = []
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset:offset + batch_size]
        batch_number = offset // batch_size + 1
        batch_payload = {
            **payload,
            "rows": batch_rows,
            "judge_batch": {
                "number": batch_number,
                "size": len(batch_rows),
                "total_rows": len(rows),
                "coverage_rule": "return exactly one verdict for every supplied row",
            },
        }
        judgement = v52.v2.request_json(
            judge_client,
            v52.request_args(args, judge=True),
            ROW_JUDGE_PROMPT,
            batch_payload,
            (
                f"V5.7 block {block['block_id']} joint fidelity audit "
                f"round {judge_round} batch {batch_number}"
            ),
        )
        expected_ids = [row["source_id"] for row in batch_rows]
        canonical = v52.v4.exact_rows(
            judgement.get("verdicts"), expected_ids,
            f"judge batch {batch_number} verdicts",
        )
        verdicts.extend(canonical[source_id] for source_id in expected_ids)
        provenance.append({
            "batch_number": batch_number,
            "source_ids": expected_ids,
            "verdict_count": len(canonical),
        })
    return {"verdicts": verdicts}, provenance


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
    candidates = v55.generate_rows(
        generation_client, args, block, profile, state_dir
    )
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    last_failures: Dict[str, str] = {}
    for judge_round in range(1, args.judge_rounds + 1):
        plan = materialize_plan(block, profile, candidates, args.seed)
        judgement, batches = judge_plan_in_batches(
            judge_client, args, block, plan, judge_round
        )
        verdicts, failures = parse_judgement(judgement, block, plan)
        v52.write_json(
            v52.block_directory(state_dir, block_id)
            / f"judge_round_{judge_round:02d}.json",
            {
                "design_version": DESIGN_VERSION,
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


def assemble_block(
    block: Mapping, plan: Mapping, verdicts: Mapping[str, Mapping], args
) -> Dict:
    assembled = BASE_V56_ASSEMBLE_BLOCK(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-joint-contrast-v5.7-"
        f"block-{int(block['block_id']):02d}:seed-{args.seed}"
    )
    assembled.update({
        "profile_id": profile_id,
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
    })
    for record in assembled["records"]:
        source_id = record["source_id"]
        record.update({
            "profile_id": profile_id,
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "question_rewrite_provenance": {
                "anchor_rewrites": plan["row_plans"][source_id][
                    "question_anchor_rewrites"
                ],
                "rationale": plan["row_plans"][source_id][
                    "question_rewrite_rationale"
                ],
            },
        })
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        })
        failures = ciru.validate_ciru_unit(record)
        if failures:
            raise ValueError(f"{source_id}: " + "; ".join(failures))
        record["audit"] = ciru.audit_ciru_unit(record)
    return assembled


def configure_shared_modules() -> None:
    v56.DESIGN_VERSION = DESIGN_VERSION
    v56.CONTRAST_SCHEMA_VERSION = CONTRAST_SCHEMA_VERSION
    v56.SURFACE_RENDERER = SURFACE_RENDERER
    v56.MAPPING_SCOPE = MAPPING_SCOPE
    v56.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v56.PROFILE_PROMPT = PROFILE_PROMPT
    v56.PROFILE_JUDGE_PROMPT = PROFILE_JUDGE_PROMPT
    v56.ROW_JUDGE_PROMPT = ROW_JUDGE_PROMPT
    v56.validate_profile = validate_profile
    v56.configure_shared_modules()

    v55.DESIGN_VERSION = DESIGN_VERSION
    v55.SURFACE_RENDERER = SURFACE_RENDERER
    v55.MAPPING_SCOPE = MAPPING_SCOPE
    v55.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v55.configure_shared_modules()
    v53.generate_profile = v56.generate_profile
    v53.JUDGE_PROMPT = ROW_JUDGE_PROMPT
    v55.ANSWER_PROMPT = JOINT_PROMPT
    v55.row_payload = row_payload
    v55.validate_row_candidate = validate_row_candidate
    v55.candidate_for_prompt = candidate_for_prompt
    v55.generate_row = generate_row
    v55.materialize_plan = materialize_plan
    v55.assemble_block = assemble_block
    v55.generate_block = generate_block
    v55.v52.parse_judgement = parse_judgement


# Save the V5.6 validator before configure_shared_modules replaces it.
v56._V57_BASE_VALIDATE_PROFILE = v56.validate_profile


def main() -> None:
    configure_shared_modules()
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
    parser.add_argument("--max-completion-tokens", type=int, default=18000)
    parser.add_argument("--block-concurrency", type=int, default=1)
    parser.add_argument("--row-concurrency", type=int, default=5)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--profile-retries", type=int, default=6)
    parser.add_argument("--row-retries", type=int, default=6)
    parser.add_argument("--judge-rounds", type=int, default=4)
    parser.add_argument("--judge-batch-size", type=int, default=5)
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
        raise SystemExit("Set generation and judge API keys before V5.7 generation")
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
        cached = v53.load_valid_block(path, block)
        if cached is not None and (
            cached.get("design_version") != DESIGN_VERSION
            or cached.get("contrast_schema_version") != CONTRAST_SCHEMA_VERSION
            or cached.get("response_contract_policy") != RESPONSE_CONTRACT_POLICY
        ):
            cached = None
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
                    v55.generate_block,
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
                        f"block={block_id} author={block['target_entity']}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append((block_id, block["target_entity"], str(exc)))
    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        raise SystemExit(
            "V5.7 incomplete; accepted joint rows and block checkpoints were retained"
        )

    ordered = [completed[block_id] for block_id in selected_ids]
    records = [record for block in ordered for record in block["records"]]
    expected = len(blocks) * int(manifest["block_size"])
    if len(records) != expected:
        raise SystemExit(f"record count {len(records)} != {expected}")
    ciru.write_ciru_jsonl(args.output, records)
    v52.write_json(profiles_output, {
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "split": args.split,
        "seed": args.seed,
        "selected_block_ids": selected_ids,
        "manifest": str(args.manifest.resolve()),
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "profiles": [v56.export_profile(block) for block in ordered],
    })
    print(
        f"design={DESIGN_VERSION} contrast={CONTRAST_SCHEMA_VERSION} "
        f"blocks={len(blocks)} rows={len(records)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
