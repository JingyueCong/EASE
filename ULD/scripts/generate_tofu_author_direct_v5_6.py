#!/usr/bin/env python3
"""Generate TOFU author factorial data with direct GPT fact contrast.

V5.6 keeps V5.5's complete-answer renderer and row-local repair loop, but does
not force facts into hand-written slot types.  GPT directly states the source
core fact and a replacement core fact for every row.  A separate GPT call sees
both immutable C11 and the proposed ledger and must approve genuine semantic
contrast before any C01 answer is rendered.

FullAnswer and V1-V5.5 are not modified.
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
V55_PATH = SCRIPT_DIR / "generate_tofu_author_answers_v5_5.py"
DESIGN_VERSION = "tofu-author-direct-contrast-v5.6"
CONTRAST_SCHEMA_VERSION = "direct-core-fact-v1"
SURFACE_RENDERER = "direct-contrast-complete-answer-v5.6"
MAPPING_SCOPE = "gpt-core-fact-contrast-conditioned-complete-c01-answer"
RESPONSE_CONTRACT_POLICY = "semantic-compatible-v5.5.1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v55 = load_module("tofu_author_direct_v56_v55", V55_PATH)
v53 = v55.v53
v52 = v55.v52
anchors = v55.anchors
ciru = v55.ciru

BASE_V55_ROW_PAYLOAD = v55.row_payload
BASE_V55_ASSEMBLE_BLOCK = v55.assemble_block


PROFILE_PROMPT = """Create one coherent replacement-author profile and one
direct core-fact contrast for every row of a 20-row TOFU author block.

You are using GPT to edit meaning, not to fill predefined slots.  Read each
immutable C11 question and answer, state the shortest faithful core answer,
then create a genuinely different core answer for the replacement author.

Critical rules:
- preserve every source_id and the exact declared row policy;
- source_core_fact must faithfully summarize the fact that answers C11;
- replacement_core_fact must answer the same relation at comparable detail;
- replacement_fact must exactly equal replacement_core_fact;
- changing only the author name is not a factual change;
- keeping the same book, award, place, genre, date, number, conclusion, or
  theme set and merely adding a year or explanation is not a factual change;
- retaining all original list items and adding one item is not a change;
- a paraphrase or elaboration that still entails the source core fact is not a
  change;
- use one fact_key for the same underlying replacement fact across rows;
- all replacement facts must describe one coherent fictional author;
- do not mention that the replacement author or facts are fictional,
  synthetic, controls, counterfactual, or undocumented;
- identity/unavailable rows use contrast_status=policy and follow the declared
  policy without pretending a fact changed;
- when the core fact is genuinely ambiguous, use contrast_status=abstain and a
  concrete abstain_reason.  Never guess merely to pass validation.

If validation_feedback and previous_candidate are supplied, return the entire
profile with only the reported semantic defects repaired.

Return JSON only:
{"target_entity":"required target","replacement_entity":"new author",
 "replacement_pronouns":"required class","profile_summary":"summary",
 "fact_ledger":[
  {"source_id":"exact id","fact_key":"stable semantic key",
   "target_relation":"same relation in concise words",
   "source_core_fact":"short faithful core answer",
   "replacement_core_fact":"short genuinely different core answer",
   "replacement_fact":"exact copy of replacement_core_fact",
   "fact_change_required":true,
   "intervention_policy":"factual_anchor_change",
   "contrast_status":"changed|policy|abstain",
   "abstain_reason":""}
 ]}
"""


PROFILE_JUDGE_PROMPT = """Independently audit a proposed 20-row direct
core-fact ledger against the immutable C11 rows.  Do not trust the generator's
labels.  Compare the original question/answer, source_core_fact, and
replacement_core_fact yourself.

For each row judge:
- source_core_faithful: source_core_fact captures the actual answer to C11;
- same_relation: replacement_core_fact answers the same question relation;
- core_fact_changed: the semantic core really changes.  The same award/book/
  place/category/conclusion with a new date, qualifier, or longer explanation
  is false.  A source set retained in full with only additions is false.  A
  paraphrase or elaboration that still entails the source core fact is false;
- replacement_plausible: the fact can belong to the proposed author profile.

Also judge whether all replacement facts are mutually coherent.  Policy rows
do not require core_fact_changed.  Preserve declared ABSTAIN rows.

Return JSON only:
{"coherent":true,"reason":"brief block evidence",
 "verdicts":[{"source_id":"exact id","source_core_faithful":true,
 "same_relation":true,"core_fact_changed":true,
 "replacement_plausible":true,"reason":"state the actual before/after"}]}
"""


ROW_JUDGE_PROMPT = """Audit a completed 20-row TOFU counterfactual block.
For each row compare immutable C11, C01, source_core_fact, and
replacement_core_fact.  Do not assume a frozen ledger is correct.

Decide:
- target_relation_match: C11 and C01 answer the same relation;
- target_fact_changed: C01 expresses the genuinely different replacement core
  fact.  Extra dates, modifiers, or prose alone do not count;
- profile_consistent: C01 matches the proposed replacement author and its
  direct replacement core fact;
- natural_surface: C01 is fluent and grammatical.

Return JSON only:
{"verdicts":[{"source_id":"exact id","target_relation_match":true,
 "target_fact_changed":true,"profile_consistent":true,
 "natural_surface":true,"reason":"state the compared core facts"}]}
"""


ANSWER_PROMPT = """Render one complete C01 answer from a frozen direct
core-fact contrast.

The replacement identity, C01 question, target relation, source core fact,
replacement core fact, and response contract are fixed.  Return one complete
replacement answer, not span edits and not an explanation of your work.

Hard rules:
- copy source_id, target_relation, ledger_fact_key, ledger_replacement_fact,
  and c01_question exactly from the payload;
- answer the same relation as C11 and explicitly express
  replacement_core_fact without retaining source_core_fact;
- obey surface_constraints.required_mode_family literally.  For ``positive``
  use a direct affirmative statement and do not introduce no, not, never,
  neither, without, uncertainty, or unavailable-information wording.  For
  ``negative`` preserve an explicit negative answer.  For ``unavailable`` or
  ``qualified`` preserve that availability status;
- never introduce a phrase listed in
  surface_constraints.forbidden_new_control_status_markers.  A phrase listed
  as inherited is benchmark wording, but it must not be newly attached to the
  replacement author or replacement fact;
- preserve the semantic answer object required by the question (date/year,
  number, binary answer, or open prose).  Surface punctuation and legacy
  answer-format labels are only soft matching goals;
- approximately match source length, grammatical person, and detail, using
  fluent natural prose without irrelevant dates, numbers, or list items;
- do not mention the target author, the generation process, or any
  counterfactual/control status;
- do not invent content that contradicts the frozen replacement core fact.

When validation_feedback and previous_candidate are supplied, repair the
reported defect directly.  In particular, a polarity error must be rewritten
with the required mode family, and a control-status error must remove the
newly introduced marker.  Never change a frozen field.

Return JSON only:
{"source_id":"exact id","target_relation":"copied relation",
 "ledger_fact_key":"copied key","ledger_replacement_fact":"copied fact",
 "c01_question":"copied question","replacement_answer":"complete answer"}
"""


def _normalise(value: object) -> str:
    return anchors.normalise(str(value))


def profile_payload(
    block: Mapping,
    protected_authors: Sequence[str],
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = {
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
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
                "intervention_policy": v53.policy_for_source(
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
    base = v53.BASE_VALIDATE_PROFILE(block, generated, protected_authors)
    source_by_id = {source["source_id"]: source for source in block["sources"]}
    raw_entries = v52.v4.exact_rows(
        generated.get("fact_ledger"), list(source_by_id), "fact_ledger"
    )
    ledger = []
    shared_variants: Dict[str, Dict[tuple[str, str], str]] = {}
    fact_key_repairs: Dict[str, Dict[str, list[str]]] = {}
    for source_id, source in source_by_id.items():
        raw = raw_entries[source_id]
        required = v52.source_fact_change_required(source, block["target_entity"])
        policy = v53.policy_for_source(source, block["target_entity"])
        if raw.get("fact_change_required") is not required:
            raise ValueError(f"{source_id} fact_change_required must equal {required}")
        if raw.get("intervention_policy") != policy:
            raise ValueError(f"{source_id} intervention_policy must equal {policy}")
        relation = str(raw.get("target_relation", "")).strip()
        if not relation:
            raise ValueError(f"{source_id} target_relation must be non-empty")
        status = str(raw.get("contrast_status", "")).strip()
        abstain_reason = str(raw.get("abstain_reason", "")).strip()

        if required:
            if status not in {"changed", "abstain"}:
                raise ValueError(f"{source_id} factual row needs changed or abstain status")
            source_core = str(raw.get("source_core_fact", "")).strip()
            replacement_core = str(raw.get("replacement_core_fact", "")).strip()
            replacement_fact = str(raw.get("replacement_fact", "")).strip()
            fact_key = str(raw.get("fact_key", "")).strip()
            if status == "abstain":
                if not abstain_reason:
                    raise ValueError(f"{source_id} abstain requires a reason")
                fact_key = f"abstain:{source_id}"
                source_core = source_core or source["answer"]
                replacement_core = replacement_core or "abstain"
                replacement_fact = replacement_core
            else:
                if not all((source_core, replacement_core, replacement_fact, fact_key)):
                    raise ValueError(f"{source_id} changed row needs complete core facts")
                if replacement_fact != replacement_core:
                    raise ValueError(
                        f"{source_id} replacement_fact must copy replacement_core_fact"
                    )
                if _normalise(source_core) == _normalise(replacement_core):
                    raise ValueError(f"{source_id} core fact is textually unchanged")
                if not (
                    v53.content_tokens(source_core)
                    & v53.content_tokens(source["answer"])
                ):
                    raise ValueError(
                        f"{source_id} source_core_fact lacks source-answer evidence"
                    )
                if _normalise(block["target_entity"]) in _normalise(replacement_core):
                    raise ValueError(f"{source_id} replacement core leaks target author")
        else:
            if status != "policy":
                raise ValueError(f"{source_id} identity/unavailable row needs policy status")
            source_core = str(raw.get("source_core_fact", "")).strip()
            if not source_core:
                raise ValueError(f"{source_id} policy row needs source_core_fact")
            replacement_core = (
                "unavailable" if "unavailability" in policy
                else base["replacement_entity"]
            )
            replacement_fact = replacement_core
            fact_key = (
                "policy:unavailable" if "unavailability" in policy
                else "policy:identity"
            )
            abstain_reason = ""

        shared_payload = (_normalise(replacement_core), status)
        variants = shared_variants.setdefault(fact_key, {})
        resolved_key = variants.get(shared_payload)
        if resolved_key is None:
            resolved_key = (
                fact_key if not variants else f"{fact_key}::v{len(variants) + 1}"
            )
            variants[shared_payload] = resolved_key
        fact_key_repairs.setdefault(fact_key, {}).setdefault(
            resolved_key, []
        ).append(source_id)
        fact_key = resolved_key
        ledger.append({
            "source_id": source_id,
            "fact_key": fact_key,
            "target_relation": relation,
            "source_core_fact": source_core,
            "replacement_core_fact": replacement_core,
            "replacement_fact": replacement_fact,
            "fact_change_required": required,
            "intervention_policy": policy,
            "contrast_status": status,
            "abstain_reason": abstain_reason,
        })

    ledger_digest = v52.digest_json(ledger)
    result = {
        **base,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "fact_ledger": ledger,
        "fact_ledger_by_source": {item["source_id"]: item for item in ledger},
        "ledger_digest": ledger_digest,
        "fact_key_repairs": {
            key: variants for key, variants in fact_key_repairs.items()
            if len(variants) > 1
        },
    }
    result["profile_digest"] = v52.digest_json({
        "target_entity": result["target_entity"],
        "replacement_entity": result["replacement_entity"],
        "replacement_pronouns": result["replacement_pronouns"],
        "profile_summary": result["profile_summary"],
        "ledger_digest": ledger_digest,
        "anchor_catalog_digest": block["anchor_catalog"]["digest"],
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
    })
    return result


def abstain_source_ids(profile: Mapping) -> list[str]:
    return sorted(
        entry["source_id"] for entry in profile["fact_ledger"]
        if entry["contrast_status"] == "abstain"
    )


def profile_judge_payload(block: Mapping, profile: Mapping) -> Dict:
    return {
        "design_version": DESIGN_VERSION,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "target_entity": profile["target_entity"],
        "replacement_entity": profile["replacement_entity"],
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


def validate_profile_judgement(
    generated: Mapping, block: Mapping, profile: Mapping
) -> Dict:
    source_ids = [source["source_id"] for source in block["sources"]]
    verdicts = v52.v4.exact_rows(
        generated.get("verdicts"), source_ids, "profile verdicts"
    )
    abstained = set(abstain_source_ids(profile))
    failures = []
    canonical = {}
    for source_id in source_ids:
        raw = verdicts[source_id]
        required = profile["fact_ledger_by_source"][source_id][
            "fact_change_required"
        ]
        fields = ["source_core_faithful", "same_relation", "replacement_plausible"]
        if required and source_id not in abstained:
            fields.append("core_fact_changed")
        failed = [field for field in fields if raw.get(field) is not True]
        reason = str(raw.get("reason", "")).strip()
        if not reason:
            failed.append("reason")
        if failed:
            failures.append(f"{source_id}({','.join(failed)}): {reason}")
        canonical[source_id] = {
            field: raw.get(field) for field in (
                "source_core_faithful", "same_relation", "core_fact_changed",
                "replacement_plausible",
            )
        }
        canonical[source_id]["reason"] = reason
    block_reason = str(generated.get("reason", "")).strip()
    if abstained:
        failures.append("ABSTAIN rows=" + ",".join(sorted(abstained)))
    if generated.get("coherent") is not True:
        failures.append("replacement profile is incoherent: " + block_reason)
    if not block_reason:
        failures.append("missing block judge reason")
    if failures:
        raise ValueError("direct contrast judge rejected: " + " | ".join(failures))
    return {
        "coherent": True,
        "contrastive": True,
        "abstain_source_ids": [],
        "reason": block_reason,
        "verdicts": canonical,
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
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
                cached["semantic_judge"], block, profile
            )
            if (
                cached.get("design_version") == DESIGN_VERSION
                and cached.get("contrast_schema_version") == CONTRAST_SCHEMA_VERSION
                and cached.get("ledger_digest") == profile["ledger_digest"]
            ):
                profile["profile_attempt"] = int(cached.get("profile_attempt", 0))
                profile["profile_semantic_judge"] = verdict
                print(f"reuse_direct_profile block={block['block_id']}", flush=True)
                return profile
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            pass

    feedback, previous, last_error = "", None, None
    for attempt in range(1, args.profile_retries + 1):
        print(
            f"start_direct_profile block={block['block_id']} "
            f"attempt={attempt}/{args.profile_retries}", flush=True,
        )
        candidate = None
        try:
            candidate = v52.v2.request_json(
                generation_client,
                v52.request_args(args),
                PROFILE_PROMPT,
                profile_payload(block, protected_authors, feedback, previous),
                f"V5.6 block {block['block_id']} direct profile",
            )
            profile = validate_profile(block, candidate, protected_authors)
            if abstain_source_ids(profile):
                raise ValueError(
                    "ABSTAIN rows=" + ",".join(abstain_source_ids(profile))
                )
            judgement = v52.v2.request_json(
                judge_client,
                v52.request_args(args, judge=True),
                PROFILE_JUDGE_PROMPT,
                profile_judge_payload(block, profile),
                f"V5.6 block {block['block_id']} direct contrast audit",
            )
            verdict = validate_profile_judgement(judgement, block, profile)
            profile["profile_attempt"] = attempt
            profile["profile_semantic_judge"] = verdict
            v52.write_json(path, {
                "design_version": DESIGN_VERSION,
                "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
                "anchor_catalog_digest": block["anchor_catalog"]["digest"],
                "ledger_digest": profile["ledger_digest"],
                "profile_attempt": attempt,
                "profile": {
                    key: value for key, value in profile.items()
                    if key not in {"fact_ledger_by_source", "profile_semantic_judge"}
                },
                "semantic_judge": verdict,
            })
            print(f"direct_profile_ready block={block['block_id']}", flush=True)
            return profile
        except Exception as exc:
            last_error, feedback, previous = exc, str(exc), candidate
            v52.write_json(
                v52.block_directory(state_dir, int(block["block_id"]))
                / "profile_attempts" / f"attempt_{attempt:02d}.json",
                {"error": feedback, "candidate": previous},
            )
            print(
                f"direct_profile_reject block={block['block_id']} "
                f"attempt={attempt}/{args.profile_retries} error={exc}", flush=True,
            )
    raise RuntimeError(
        f"block {block['block_id']} direct profile failed after "
        f"{args.profile_retries} attempts: {last_error}"
    ) from last_error


def row_payload(
    block: Mapping,
    source: Mapping,
    profile: Mapping,
    feedback: str = "",
    previous: Mapping | None = None,
) -> Dict:
    payload = BASE_V55_ROW_PAYLOAD(block, source, profile, feedback, previous)
    entry = profile["fact_ledger_by_source"][source["source_id"]]
    expected_mode = str(source["contract"]["response_mode"])
    required_family = v55._mode_family(expected_mode)
    source_text = ciru.normalise(
        f"{source['question']} {source['answer']}"
    )
    inherited_markers = [
        marker for marker in ciru.CONTROL_STATUS_MARKERS
        if marker in source_text
    ]
    forbidden_markers = [
        marker for marker in ciru.CONTROL_STATUS_MARKERS
        if marker not in source_text
    ]
    mode_instruction = {
        "positive": (
            "Write a direct affirmative answer. Do not use no, not, never, "
            "neither, without, unclear, unknown, unavailable, or an "
            "information-is-missing construction anywhere in the answer."
        ),
        "negative": "Preserve an explicit negative answer.",
        "unavailable": "Preserve an explicit unavailable-information answer.",
        "qualified": "Preserve a qualified or uncertain answer.",
    }.get(
        required_family,
        f"Preserve the semantic response-mode family {required_family!r}.",
    )
    payload.update({
        "contrast_schema_version": CONTRAST_SCHEMA_VERSION,
        "source_core_fact": entry["source_core_fact"],
        "replacement_core_fact": entry["replacement_core_fact"],
        "contrast_status": entry["contrast_status"],
        "surface_constraints": {
            "source_response_mode": expected_mode,
            "required_mode_family": required_family,
            "mode_instruction": mode_instruction,
            "inherited_control_status_markers": inherited_markers,
            "forbidden_new_control_status_markers": forbidden_markers,
            "repair_scope": (
                "Change only replacement_answer when validation feedback is "
                "present; all other response fields are frozen."
            ),
        },
    })
    return payload


def assemble_block(
    block: Mapping, plan: Mapping, verdicts: Mapping[str, Mapping], args
) -> Dict:
    assembled = BASE_V55_ASSEMBLE_BLOCK(block, plan, verdicts, args)
    profile_id = (
        f"{args.split}:author-direct-contrast-v5.6-"
        f"block-{int(block['block_id']):02d}:seed-{args.seed}"
    )
    assembled["profile_id"] = profile_id
    assembled["contrast_schema_version"] = CONTRAST_SCHEMA_VERSION
    for record in assembled["records"]:
        record["profile_id"] = profile_id
        record["contrast_schema_version"] = CONTRAST_SCHEMA_VERSION
        record["generation"]["contrast_schema_version"] = CONTRAST_SCHEMA_VERSION
    return assembled


def configure_shared_modules() -> None:
    v55.DESIGN_VERSION = DESIGN_VERSION
    v55.SURFACE_RENDERER = SURFACE_RENDERER
    v55.MAPPING_SCOPE = MAPPING_SCOPE
    v55.RESPONSE_CONTRACT_POLICY = RESPONSE_CONTRACT_POLICY
    v55.configure_shared_modules()
    v53.generate_profile = generate_profile
    v53.JUDGE_PROMPT = ROW_JUDGE_PROMPT
    v55.ANSWER_PROMPT = ANSWER_PROMPT
    v55.row_payload = row_payload
    v55.assemble_block = assemble_block


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
        raise SystemExit("Set generation and judge API keys before V5.6 generation")
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
            cached.get("contrast_schema_version") != CONTRAST_SCHEMA_VERSION
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
            "V5.6 incomplete; direct profiles, accepted rows, and ABSTAIN "
            "attempts were retained"
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
        "profiles": [
            {
                key: block[key]
                for key in (
                    "block_id", "profile_id", "profile_digest",
                    "target_entity", "replacement_entity",
                    "replacement_pronouns", "anchor_catalog_digest",
                    "ledger_digest", "fact_ledger", "author_plan",
                    "placebo_plan", "reconciliation_conflicts",
                    "fact_key_repairs",
                    "profile_semantic_judge", "profile_attempt", "judge_round",
                    "contrast_schema_version",
                )
            }
            for block in ordered
        ],
    })
    print(
        f"design={DESIGN_VERSION} contrast={CONTRAST_SCHEMA_VERSION} "
        f"blocks={len(blocks)} rows={len(records)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
