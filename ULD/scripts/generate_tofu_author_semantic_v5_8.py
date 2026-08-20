#!/usr/bin/env python3
"""Generate TOFU causal cells with a minimal semantic C01 interface.

V5.8 keeps V5.7's immutable C11, independently approved replacement-author
ledger, deterministic C10/C00 placebo cells, and batched semantic judge.  It
removes model-authored patch metadata: the row model returns only a complete
C01 question and answer.  Frozen identifiers are injected by code and question
rewrite provenance is derived deterministically after generation.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
V57_PATH = SCRIPT_DIR / "generate_tofu_author_joint_v5_7.py"
DESIGN_VERSION = "tofu-author-semantic-contrast-v5.8"
CONTRAST_SCHEMA_VERSION = "semantic-question-answer-core-v1"
SURFACE_RENDERER = "semantic-complete-c01-question-answer-v5.8"
MAPPING_SCOPE = "row-local-semantic-question-answer-conditioned-on-frozen-ledger"
RESPONSE_CONTRACT_POLICY = "semantic-joint-compatible-v5.8"
PROVENANCE_METHOD = "deterministic-sequence-diff-v1"
GENERATOR_OUTPUT_FIELDS = ("c01_question", "replacement_answer")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v57 = load_module("tofu_author_semantic_v58_v57", V57_PATH)
v56 = v57.v56
v55 = v57.v55
v53 = v57.v53
v52 = v57.v52
anchors = v57.anchors
ciru = v57.ciru

PROFILE_PROMPT = v57.PROFILE_PROMPT
PROFILE_JUDGE_PROMPT = v57.PROFILE_JUDGE_PROMPT
ROW_JUDGE_PROMPT = v57.ROW_JUDGE_PROMPT
JUDGE_FIELDS = v57.JUDGE_FIELDS
validate_profile = v57.validate_profile
parse_judgement = v57.parse_judgement

BASE_V57_MATERIALIZE_PLAN = v57.materialize_plan
BASE_V57_ASSEMBLE_BLOCK = v57.assemble_block


SEMANTIC_ROW_PROMPT = """Generate one complete TOFU C01 question and answer.

The immutable C11 question/answer and a frozen, independently approved
replacement-author fact ledger are supplied in the payload.  Rewrite the C01
question and answer together so that they form a coherent matched pair.

Semantic requirements:
- ask the same target relation as C11 with comparable specificity and
  cardinality;
- replace every target-specific premise in the question (author, title,
  award, place, date, number, or identity description) that must change under
  the replacement profile;
- make replacement_answer directly answer the actual c01_question;
- express the frozen ledger replacement fact without retaining a
  contradictory source fact;
- preserve the declared positive, negative, unavailable, or qualified
  response-mode family;
- do not mention the target author, controls, counterfactuals, generation, or
  fictionality unless that wording is inherited from the benchmark.

Do not copy identifiers, ledger keys, rationales, spans, edits, anchors, or
provenance into the response.  Code owns those fields.  When feedback and a
previous candidate are supplied, repair only the question and answer.

Return JSON only with exactly these content fields:
{"c01_question":"complete question","replacement_answer":"complete answer"}
"""


def _normalise_text(value: object) -> str:
    return " ".join(str(value).replace("\n", " ").split()).strip()


def derive_question_provenance(source_question: str, c01_question: str) -> Dict:
    """Return an auditable, non-gating character diff owned by code."""
    source = _normalise_text(source_question)
    replacement = _normalise_text(c01_question)
    matcher = difflib.SequenceMatcher(a=source, b=replacement, autojunk=False)
    operations = []
    for tag, source_start, source_end, replacement_start, replacement_end in (
        matcher.get_opcodes()
    ):
        if tag == "equal":
            continue
        operations.append({
            "operation": tag,
            "source_span": [source_start, source_end],
            "replacement_span": [replacement_start, replacement_end],
            "source_text": source[source_start:source_end],
            "replacement_text": replacement[replacement_start:replacement_end],
        })
    return {
        "method": PROVENANCE_METHOD,
        "source_question": source,
        "replacement_question": replacement,
        "operations": operations,
        "changed": source != replacement,
        "validation_role": "audit_only",
    }


def _compatibility_rewrites(provenance: Mapping) -> list[Dict[str, str]]:
    """Supply legacy-shaped, code-owned provenance without model coordination."""
    if not provenance["changed"]:
        return []
    return [{
        "source_text": provenance["source_question"],
        "replacement_text": provenance["replacement_question"],
        "reason": "complete question change derived deterministically by code",
    }]


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = v57.row_payload(
        block,
        source,
        profile,
        feedback,
        candidate_for_prompt(previous) if previous is not None else None,
    )
    payload["design_version"] = DESIGN_VERSION
    payload["generator_output_contract"] = {
        "fields": list(GENERATOR_OUTPUT_FIELDS),
        "frozen_fields_injected_by_code": True,
        "rewrite_provenance_derived_by_code": True,
    }
    payload["joint_rewrite_policy"].update({
        "model_authored_patch_metadata": False,
        "deterministic_provenance_method": PROVENANCE_METHOD,
    })
    return payload


def validate_row_candidate(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    generated: Mapping,
) -> Dict:
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
    provenance = derive_question_provenance(source["question"], question)

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
        raise ValueError("C01 question loses explicit replacement identity")
    marker = v55._introduced_control_status_marker(source, cell)
    if marker:
        raise ValueError(f"C01 exposes control status with marker: {marker}")

    # Object type and polarity belong to the immutable C11 relation.  Using
    # the generated C01 question here lets titles such as "When Bridges Sleep"
    # masquerade as temporal interrogatives.  The independent judge, not this
    # lexical contract, decides whether C01 preserved the relation.
    contract_result = v55.semantic_contract_result(
        {"question": source["question"], "answer": answer},
        source["contract"],
    )
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
            source["answer"],
            block["target_entity"],
            profile["replacement_entity"],
        )
        if (
            v57._normalise(question) == v57._normalise(identity_question)
            and v57._normalise(answer) == v57._normalise(identity_answer)
        ):
            raise ValueError("C01 changes only author identity, not the target fact")

    return {
        "source_id": source_id,
        "target_relation": entry["target_relation"],
        "ledger_fact_key": entry["fact_key"],
        "ledger_replacement_fact": entry["replacement_fact"],
        "c01_question": question,
        "replacement_answer": answer,
        "question_anchor_rewrites": _compatibility_rewrites(provenance),
        "question_rewrite_rationale": (
            "Question rewrite provenance was derived deterministically; "
            "semantic validity is decided independently by the block judge."
        ),
        "cell": cell,
        "fact_change_required": bool(entry["fact_change_required"]),
        "render_mode": "semantic_complete_question_answer_intervention",
        "ledger_digest": profile["ledger_digest"],
        "question_rewrite_provenance": provenance,
        "generator_output_fields": list(GENERATOR_OUTPUT_FIELDS),
        "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        "response_contract_warnings": contract_result["warnings"],
        "response_contract_question_source": "immutable_c11",
    }


def candidate_for_prompt(candidate: Mapping | None) -> Dict | None:
    if candidate is None:
        return None
    return {
        key: candidate[key]
        for key in GENERATOR_OUTPUT_FIELDS
        if isinstance(candidate.get(key), str)
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
                SEMANTIC_ROW_PROMPT,
                row_payload(block, source, profile, feedback, previous),
                f"V5.8 block {block_id} semantic row {source_id}",
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
                {"error": feedback, "candidate": candidate_for_prompt(previous)},
            )
            print(
                f"row_reject block={block_id} source={source_id} "
                f"attempt={attempt}/{args.row_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block_id} semantic row {source_id} failed after "
        f"{args.row_retries} attempts: {last_error}"
    ) from last_error


def materialize_plan(
    block: Mapping,
    profile: Mapping,
    candidates: Mapping[str, Mapping],
    seed: int,
) -> Dict:
    plan = BASE_V57_MATERIALIZE_PLAN(block, profile, candidates, seed)
    for source_id, row_plan in plan["row_plans"].items():
        candidate = candidates[source_id]
        row_plan.update({
            "render_mode": "semantic_complete_question_answer_intervention",
            "question_rewrite_provenance": candidate[
                "question_rewrite_provenance"
            ],
            "generator_output_fields": list(GENERATOR_OUTPUT_FIELDS),
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
        })
    return plan


def _partial_batch_verdicts(
    generated: Mapping | None, expected_ids: Sequence[str]
) -> Dict[str, Dict]:
    """Keep only unambiguous expected verdicts from a partial judge reply."""
    if not isinstance(generated, Mapping):
        return {}
    raw_verdicts = generated.get("verdicts")
    if not isinstance(raw_verdicts, list):
        return {}
    expected = set(expected_ids)
    canonical: Dict[str, Dict] = {}
    invalid = False
    for raw in raw_verdicts:
        if not isinstance(raw, Mapping):
            invalid = True
            continue
        source_id = raw.get("source_id")
        if source_id not in expected or source_id in canonical:
            invalid = True
            continue
        canonical[source_id] = dict(raw)
    # An extra or duplicate row makes positional/model coordination suspect;
    # re-audit the complete batch through singleton fallbacks.
    return {} if invalid else canonical


def _request_singleton_verdict(
    judge_client,
    args,
    payload: Mapping,
    row: Mapping,
    block_id: int,
    judge_round: int,
    batch_number: int,
) -> Dict:
    source_id = row["source_id"]
    last_error = None
    for attempt in range(1, max(int(args.request_retries), 1) + 1):
        try:
            judgement = v52.v2.request_json(
                judge_client,
                v52.request_args(args, judge=True),
                ROW_JUDGE_PROMPT,
                {
                    **payload,
                    "rows": [row],
                    "judge_batch": {
                        "number": batch_number,
                        "size": 1,
                        "total_rows": 1,
                        "coverage_rule": (
                            "return exactly one verdict for the supplied row"
                        ),
                        "coverage_fallback": True,
                        "coverage_attempt": attempt,
                    },
                },
                (
                    f"V5.8 block {block_id} semantic audit round "
                    f"{judge_round} singleton {source_id} attempt {attempt}"
                ),
            )
            canonical = v52.v4.exact_rows(
                judgement.get("verdicts"), [source_id],
                f"singleton verdict {source_id}",
            )
            return canonical[source_id]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"singleton judge coverage failed for {source_id}: {last_error}"
    ) from last_error


def judge_plan_in_batches(
    judge_client,
    args,
    block: Mapping,
    plan: Mapping,
    judge_round: int,
) -> tuple[Dict, list[Dict]]:
    """Judge batches, repairing omitted verdict coverage one row at a time."""
    payload = v53.judge_payload(block, plan)
    rows = list(payload["rows"])
    batch_size = max(int(args.judge_batch_size), 1)
    verdicts = []
    provenance = []
    block_id = int(block["block_id"])
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
        initial_error = ""
        try:
            judgement = v52.v2.request_json(
                judge_client,
                v52.request_args(args, judge=True),
                ROW_JUDGE_PROMPT,
                batch_payload,
                (
                    f"V5.8 block {block_id} semantic audit round "
                    f"{judge_round} batch {batch_number}"
                ),
            )
            canonical = _partial_batch_verdicts(
                judgement, [row["source_id"] for row in batch_rows]
            )
        except Exception as exc:
            initial_error = str(exc)
            canonical = {}

        missing_rows = [
            row for row in batch_rows if row["source_id"] not in canonical
        ]
        if missing_rows:
            print(
                f"judge_coverage_fallback block={block_id} "
                f"round={judge_round} batch={batch_number} rows="
                + ",".join(row["source_id"] for row in missing_rows),
                flush=True,
            )
        for row in missing_rows:
            canonical[row["source_id"]] = _request_singleton_verdict(
                judge_client,
                args,
                payload,
                row,
                block_id,
                judge_round,
                batch_number,
            )

        expected_ids = [row["source_id"] for row in batch_rows]
        verdicts.extend(canonical[source_id] for source_id in expected_ids)
        provenance.append({
            "batch_number": batch_number,
            "source_ids": expected_ids,
            "initial_verdict_count": len(batch_rows) - len(missing_rows),
            "coverage_fallback_source_ids": [
                row["source_id"] for row in missing_rows
            ],
            "initial_error": initial_error,
            "verdict_count": len(canonical),
        })
    return {"verdicts": verdicts}, provenance


def assemble_block(
    block: Mapping, plan: Mapping, verdicts: Mapping[str, Mapping], args
) -> Dict:
    assembled = BASE_V57_ASSEMBLE_BLOCK(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-semantic-contrast-v5.8-"
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
        row_plan = plan["row_plans"][source_id]
        record.update({
            "profile_id": profile_id,
            "design_version": DESIGN_VERSION,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "render_mode": row_plan["render_mode"],
            "question_rewrite_provenance": row_plan[
                "question_rewrite_provenance"
            ],
        })
        record["generation"].update({
            "design": DESIGN_VERSION,
            "surface_renderer": SURFACE_RENDERER,
            "mapping_scope": MAPPING_SCOPE,
            "response_contract_policy": RESPONSE_CONTRACT_POLICY,
            "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
            "generator_output_fields": list(GENERATOR_OUTPUT_FIELDS),
            "provenance_method": PROVENANCE_METHOD,
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
                f"judge_batches={len(batches)} conflicts=0", flush=True,
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
        f"block {block_id} semantic judge exhausted {args.judge_rounds} rounds: "
        + " | ".join(
            f"{source_id}: {reason}"
            for source_id, reason in sorted(last_failures.items())
        )
    )


def configure_shared_modules() -> None:
    v57.configure_shared_modules()
    for module in (v57, v56, v55):
        module.DESIGN_VERSION = DESIGN_VERSION
        module.CONTRAST_SCHEMA_VERSION = CONTRAST_SCHEMA_VERSION
        module.SURFACE_RENDERER = SURFACE_RENDERER
        module.MAPPING_SCOPE = MAPPING_SCOPE
        module.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v55.ANSWER_PROMPT = SEMANTIC_ROW_PROMPT
    v55.row_payload = row_payload
    v55.validate_row_candidate = validate_row_candidate
    v55.candidate_for_prompt = candidate_for_prompt
    v55.generate_row = generate_row
    v55.materialize_plan = materialize_plan
    v55.assemble_block = assemble_block
    v55.generate_block = generate_block


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
        raise SystemExit("Set generation and judge API keys before V5.8 generation")
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
                        f"block={block_id} author={block['target_entity']}",
                        flush=True,
                    )
                except Exception as exc:
                    failures.append((block_id, block["target_entity"], str(exc)))
    if failures or len(completed) != len(blocks):
        for block_id, author, error in failures:
            print(f"FAIL block={block_id} author={author}: {error}", file=sys.stderr)
        raise SystemExit(
            "V5.8 incomplete; accepted semantic rows and block checkpoints were retained"
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
        "generator_output_fields": list(GENERATOR_OUTPUT_FIELDS),
        "provenance_method": PROVENANCE_METHOD,
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
