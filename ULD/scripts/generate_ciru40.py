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

Keep task, language, style, answer format, difficulty, and approximate lengths matched across cells.
Never copy the source answer into a control. C01/C00 must not contain the original entity or aliases.
The replacement and placebo facts are explicitly fictional controls, not claims about real people.

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
    sources: List[Dict[str, str]], units: int, block_size: int, seed: int
) -> List[Dict[str, str]]:
    """Sample evenly across ordered TOFU author blocks when possible."""
    rng = random.Random(seed)
    if block_size > 0 and len(sources) % block_size == 0:
        blocks = [sources[i : i + block_size] for i in range(0, len(sources), block_size)]
        if units % len(blocks) == 0 and units // len(blocks) <= block_size:
            per_block = units // len(blocks)
            selected = []
            for block in blocks:
                selected.extend(rng.sample(block, per_block))
            return selected
    if units > len(sources):
        raise ValueError(f"Requested {units} units from only {len(sources)} sources")
    return rng.sample(sources, units)


def generate_one(client, args, source: Dict[str, str]) -> Dict:
    user = (
        f"C11 source_id: {source['source_id']}\n"
        f"C11 question: {source['question']}\n"
        f"C11 answer: {source['answer']}"
    )
    last_error = None
    for attempt in range(args.retries):
        try:
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
            if errors:
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

    sources = stratified_sources(
        load_sources(args.split), args.units, args.block_size, args.seed
    )
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    probe_modes = ["required", "prompt"] if args.json_mode == "auto" else [args.json_mode]
    first = None
    for mode in probe_modes:
        args.resolved_json_mode = mode
        print(f"API preflight: model={args.model} json_mode={mode}")
        try:
            first = generate_one(client, args, sources[0])
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

    records = [first]
    failures = []
    with ThreadPoolExecutor(max_workers=max(args.concurrency, 1)) as executor:
        future_map = {
            executor.submit(generate_one, client, args, source): source
            for source in sources[1:]
        }
        for future in as_completed(future_map):
            source = future_map[future]
            try:
                records.append(future.result())
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
    print(f"units={len(records)} cells={4 * len(records)} generated_cells={3 * len(records)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
