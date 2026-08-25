#!/usr/bin/env python3
"""Generate exactly K jointly designed CIRU 2x2 intervention units."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ciru_data = load_module("ciru_data", SCRIPT_DIR.parent / "uld" / "data" / "ciru.py")
f2r_generator = load_module("f2r_generator_shared", SCRIPT_DIR / "generate_f2r_pairs.py")
validate_ciru_unit = ciru_data.validate_ciru_unit
audit_ciru_unit = ciru_data.audit_ciru_unit


SYSTEM_PROMPT = """You design controlled 2x2 factorial interventions for language-model unlearning.
Given one factual QA item C11 that must be forgotten, jointly construct C01, C10, and C00.

Factors:
- T=1 uses the original target entity; T=0 uses one new fictional replacement entity.
- R=1 asks the original target relation; R=0 asks one placebo relation unrelated to the target fact.

Required cells:
- C11 (T=1,R=1): copy the supplied source question and answer exactly.
- C01 (T=0,R=1): use the replacement entity and replacement fact while preserving the target relation.
- C10 (T=1,R=0): use the original entity but ask the placebo relation. Its answer must not reveal the source answer.
- C00 (T=0,R=0): use the same replacement entity as C01 and the same placebo relation as C10.

Keep language, style, answer format, difficulty, and approximate lengths matched across cells;
the longest question and answer may be at most twice the corresponding shortest one.
Never copy the source answer into a control. C01/C00 must not contain the original entity or aliases.
Treat every persona here as a benchmark-controlled record. Write all cell questions and answers
directly and affirmatively. Never include meta words or disclaimers such as "fictional", "control",
"undocumented", "unknown", "no public information", or "available sources" inside a cell.
C10 and C00 must state concrete placebo facts using parallel wording and comparable detail.

Return one JSON object only:
{
  "target_entity": "exact entity string occurring in C11",
  "replacement_entity": "one new fictional entity",
  "target_relation": "short canonical relation",
  "placebo_relation": "different short canonical relation",
  "invariants": {
    "task": "...", "style": "...", "difficulty": "...", "answer_format": "..."
  },
  "cells": {
    "C11": {"question": "exact supplied question", "answer": "exact supplied answer"},
    "C01": {"question": "...", "answer": "..."},
    "C10": {"question": "...", "answer": "..."},
    "C00": {"question": "...", "answer": "..."}
  },
  "changed_evidence": ["..."],
  "placebo_rationale": "why the placebo relation cannot reveal the target answer"
}
"""


def load_sources(split: str) -> List[Dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("locuslab/TOFU", split)["train"]
    return [
        {
            "source_id": f"{split}-{index:05d}",
            "question": row["question"],
            "answer": row["answer"],
        }
        for index, row in enumerate(dataset)
    ]


def stratified_sources(
    sources: List[Dict[str, str]],
    units: int,
    block_size: int,
    seed: int,
    include_source_ids: List[str] | None = None,
) -> List[Dict[str, str]]:
    """Sample evenly across ordered TOFU author blocks when possible."""
    include_source_ids = include_source_ids or []
    if len(set(include_source_ids)) != len(include_source_ids):
        raise ValueError("Included source ids must be unique")
    source_by_id = {source["source_id"]: source for source in sources}
    missing = [source_id for source_id in include_source_ids if source_id not in source_by_id]
    if missing:
        raise ValueError(f"Included source ids are absent from the split: {missing[:3]}")
    if len(include_source_ids) > units:
        raise ValueError("Included source count exceeds requested units")

    rng = random.Random(seed)
    if block_size > 0 and len(sources) % block_size == 0:
        blocks = [sources[i : i + block_size] for i in range(0, len(sources), block_size)]
        if units % len(blocks) == 0 and units // len(blocks) <= block_size:
            per_block = units // len(blocks)
            selected = []
            for block in blocks:
                block_ids = {source["source_id"] for source in block}
                included = [
                    source_by_id[source_id]
                    for source_id in include_source_ids
                    if source_id in block_ids
                ]
                if len(included) > per_block:
                    raise ValueError(
                        "Included sources exceed the target allocation in one block"
                    )
                included_ids = {source["source_id"] for source in included}
                candidates = [
                    source for source in block
                    if source["source_id"] not in included_ids
                ]
                selected.extend(included)
                selected.extend(rng.sample(candidates, per_block - len(included)))
            return selected
    if units > len(sources):
        raise ValueError(f"Requested {units} units from only {len(sources)} sources")
    included = [source_by_id[source_id] for source_id in include_source_ids]
    included_ids = set(include_source_ids)
    candidates = [source for source in sources if source["source_id"] not in included_ids]
    return included + rng.sample(candidates, units - len(included))


def generate_one(
    client,
    args,
    source: Dict[str, str],
    replacement_entity: str | None = None,
) -> Dict:
    base_user = (
        f"C11 source_id: {source['source_id']}\n"
        f"C11 question: {source['question']}\n"
        f"C11 answer: {source['answer']}"
    )
    if replacement_entity:
        base_user += (
            "\nRequired replacement_entity: "
            f"{replacement_entity}\n"
            "Use that exact replacement entity string in C01 and C00; do not "
            "invent or rename it."
        )
    last_error = None
    validation_feedback = ""
    for attempt in range(args.retries):
        try:
            user = base_user + validation_feedback
            request = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                "temperature": args.temperature,
            }
            if args.resolved_json_mode == "required":
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            generated = f2r_generator.extract_json(response.choices[0].message.content)
            cells = generated.get("cells")
            if isinstance(cells, dict):
                # C11 is the observed treatment, not synthetic content.  Inject
                # it from the immutable source so harmless model punctuation or
                # whitespace edits cannot turn the factual anchor into a fifth
                # generated condition.
                cells["C11"] = {
                    "question": source["question"],
                    "answer": source["answer"],
                }
            record = {
                "source_id": source["source_id"],
                "block_id": int(source.get("block_id", -1)),
                "view": 0,
                "source_question": source["question"],
                "source_answer": source["answer"],
                **generated,
                "generation": {
                    "backend": "openai-compatible",
                    "model": args.model,
                    "design": "joint-2x2-factorial",
                },
            }
            errors = validate_ciru_unit(record)
            if replacement_entity and (
                ciru_data.normalise(record.get("replacement_entity", ""))
                != ciru_data.normalise(replacement_entity)
            ):
                errors.append(
                    "replacement_entity must exactly match the required "
                    f"author-block entity: {replacement_entity}"
                )
            if errors:
                validation_feedback = (
                    "\n\nYour previous JSON failed these exact hard checks:\n- "
                    + "\n- ".join(errors)
                    + "\nReturn a corrected full JSON object. In particular, declared "
                    "target_entity must be an exact contiguous span in both C11.question "
                    "and C10.question; declared replacement_entity must be an exact "
                    "contiguous span in both C01.question and C00.question. Do not "
                    "paraphrase C11. Keep question/answer length ratios at most 2.0, "
                    "use affirmative parallel answers for C10/C00, and remove every "
                    "fictional/control/undocumented/public-information disclaimer from cells."
                )
                raise ValueError("; ".join(errors))
            record["audit"] = audit_ciru_unit(record)
            return record
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(
        f"generation failed after {args.retries} attempts: "
        f"{f2r_generator.describe_error(last_error)}"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget05_perturbed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--units", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shared-replacement-per-block",
        action="store_true",
        help=(
            "Generate one replacement identity per ordered source block and "
            "reuse it for every unit in that block."
        ),
    )
    parser.add_argument(
        "--include-source-ids-from",
        type=Path,
        help=(
            "Existing CIRU JSONL whose complete audited units must be nested "
            "unchanged in this design."
        ),
    )
    parser.add_argument(
        "--source-ids-file",
        type=Path,
        help=(
            "Newline-delimited canonical source ids to generate exactly. "
            "This is mutually exclusive with stratified sampling."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--json-mode", choices=("auto", "required", "prompt"), default="auto"
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before generation")
    from openai import OpenAI

    include_records = []
    include_source_ids = []
    if args.include_source_ids_from:
        include_records = ciru_data.load_ciru_units(args.include_source_ids_from)
        include_source_ids = [record["source_id"] for record in include_records]
        print(
            f"nested_units={len(include_source_ids)} "
            f"from {args.include_source_ids_from}"
        )
    all_sources = load_sources(args.split)
    if args.source_ids_file is not None:
        requested_ids = [
            line.strip()
            for line in args.source_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if include_source_ids:
            raise SystemExit(
                "--source-ids-file cannot be combined with --include-source-ids-from"
            )
        if len(requested_ids) != args.units or len(set(requested_ids)) != args.units:
            raise SystemExit(
                "--source-ids-file must contain exactly --units unique ids"
            )
        source_by_canonical_id = {
            source["source_id"]: source for source in all_sources
        }
        missing = [
            source_id for source_id in requested_ids
            if source_id not in source_by_canonical_id
        ]
        if missing:
            raise SystemExit(f"requested source ids are absent: {missing[:5]}")
        sources = [source_by_canonical_id[source_id] for source_id in requested_ids]
    else:
        sources = stratified_sources(
            all_sources, args.units, args.block_size, args.seed,
            include_source_ids=include_source_ids,
        )
    source_block = {
        source["source_id"]: index // max(args.block_size, 1)
        for index, source in enumerate(all_sources)
    }
    for source in sources:
        source["block_id"] = source_block[source["source_id"]]
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # A failed fixed design must not create the final dataset, but validated
    # units are checkpointed separately so an API/schema failure at unit 40
    # does not waste the preceding 39 calls.  Only records matching the same
    # immutable selected source QA are reusable.
    partial_path = Path(str(args.output) + ".partial.jsonl")
    source_by_id = {source["source_id"]: source for source in sources}
    # Preserve the complete audited parent units, not merely their source ids.
    # This makes the larger budget a literal superset and spends API calls only
    # on newly selected sources.
    record_by_id = {record["source_id"]: record for record in include_records}
    if partial_path.is_file():
        for record in ciru_data.load_ciru_units(partial_path):
            source = source_by_id.get(record["source_id"])
            if source is None:
                continue
            if (
                record["source_question"] == source["question"]
                and record["source_answer"] == source["answer"]
            ):
                record.setdefault("block_id", source_block[record["source_id"]])
                record_by_id.setdefault(record["source_id"], record)
        records = list(record_by_id.values())
        print(f"resume_valid={len(records)}/{args.units} from {partial_path}")
    else:
        records = list(record_by_id.values())
    completed_ids = {record["source_id"] for record in records}
    pending_sources = [
        source for source in sources if source["source_id"] not in completed_ids
    ]
    replacement_by_block = {}
    if args.shared_replacement_per_block:
        for record in records:
            block = source_block[record["source_id"]]
            replacement = record["replacement_entity"]
            previous = replacement_by_block.setdefault(block, replacement)
            if ciru_data.normalise(previous) != ciru_data.normalise(replacement):
                raise SystemExit(
                    "Existing records violate shared replacement identity in "
                    f"block {block}: {previous!r} vs {replacement!r}"
                )
    if not pending_sources:
        records.sort(key=lambda row: row["source_id"])
        ciru_data.write_ciru_jsonl(args.output, records)
        partial_path.unlink(missing_ok=True)
        print(f"units={len(records)} cells={4 * len(records)} generated_cells={3 * len(records)}")
        print(f"output={args.output}")
        return

    probe_modes = ["required", "prompt"] if args.json_mode == "auto" else [args.json_mode]
    first = None
    for mode in probe_modes:
        args.resolved_json_mode = mode
        print(f"API preflight: model={args.model} json_mode={mode}")
        try:
            first_source = pending_sources[0]
            first_block = source_block[first_source["source_id"]]
            first = generate_one(
                client,
                args,
                first_source,
                replacement_by_block.get(first_block),
            )
            print(f"API preflight OK; resolved_json_mode={mode}")
            break
        except Exception as exc:
            print(f"API preflight failed: {f2r_generator.describe_error(exc)}", file=sys.stderr)
            if not (
                args.json_mode == "auto"
                and mode == "required"
                and f2r_generator.error_status(exc) == 400
            ):
                raise SystemExit(1)
    if first is None:
        raise SystemExit(1)

    records.append(first)
    if args.shared_replacement_per_block:
        replacement_by_block[
            source_block[first["source_id"]]
        ] = first["replacement_entity"]
    records.sort(key=lambda row: row["source_id"])
    ciru_data.write_ciru_jsonl(partial_path, records)

    # Establish one replacement identity for every represented author block
    # before parallel fact-level generation. Otherwise simultaneous requests
    # could invent different identities for the same author block.
    if args.shared_replacement_per_block:
        completed_ids = {record["source_id"] for record in records}
        remaining = [
            source for source in pending_sources
            if source["source_id"] not in completed_ids
        ]
        missing_blocks = sorted(
            {source_block[source["source_id"]] for source in remaining}
            - replacement_by_block.keys()
        )
        for block in missing_blocks:
            anchor_source = next(
                source for source in remaining
                if source_block[source["source_id"]] == block
            )
            anchor = generate_one(client, args, anchor_source)
            records.append(anchor)
            replacement_by_block[block] = anchor["replacement_entity"]
            records.sort(key=lambda row: row["source_id"])
            ciru_data.write_ciru_jsonl(partial_path, records)
            print(
                f"block_anchor={block} replacement="
                f"{anchor['replacement_entity']} valid={len(records)}/{args.units}",
                flush=True,
            )

    completed_ids = {record["source_id"] for record in records}
    remaining_sources = [
        source for source in pending_sources
        if source["source_id"] not in completed_ids
    ]
    failures = []
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        future_map = {
            executor.submit(
                generate_one,
                client,
                args,
                source,
                replacement_by_block.get(source_block[source["source_id"]])
                if args.shared_replacement_per_block else None,
            ): source
            for source in remaining_sources
        }
        for future in as_completed(future_map):
            source = future_map[future]
            try:
                records.append(future.result())
                records.sort(key=lambda row: row["source_id"])
                ciru_data.write_ciru_jsonl(partial_path, records)
                print(f"valid={len(records)}/{args.units}", flush=True)
            except Exception as exc:
                failures.append((source["source_id"], str(exc)))

    if failures or len(records) != args.units:
        for source_id, error in failures:
            print(f"FAIL source_id={source_id}: {error}", file=sys.stderr)
        print("No output written because the fixed design is incomplete.", file=sys.stderr)
        raise SystemExit(1)
    records.sort(key=lambda row: row["source_id"])
    ciru_data.write_ciru_jsonl(args.output, records)
    partial_path.unlink(missing_ok=True)
    print(f"units={len(records)} cells={4 * len(records)} generated_cells={3 * len(records)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
