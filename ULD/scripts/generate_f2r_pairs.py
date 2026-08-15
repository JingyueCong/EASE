#!/usr/bin/env python3
"""Generate matched counterfactual retention supervision from TOFU forget QA.

The generator never reads a retain split. It accepts either a local JSONL file
or a TOFU forget split and writes validated JSONL records consumable by the F2R
training data modes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
F2R_MODULE_PATH = SCRIPT_DIR.parent / "uld" / "data" / "f2r.py"
spec = importlib.util.spec_from_file_location("f2r_data", F2R_MODULE_PATH)
f2r_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(f2r_data)
validate_pair = f2r_data.validate_pair


SYSTEM_PROMPT = """You create controlled counterfactual supervision for LLM unlearning.
Given one factual QA example that must be forgotten, return a JSON object only.

Create two new QA examples:
1. matched: preserve task, relation, question style, answer style, and difficulty,
   while replacing every source-specific entity and fact with new fictional evidence.
2. mismatched: use different entities and evidence, and also change the relation or
   task structure. It is a null/control example, not a matched preservation target.

Never copy the source answer or source-specific names into either new example.
Do not claim that the fictional facts are real-world facts. Keep answers concise.

Required JSON schema:
{
  "invariants": {
    "task": "...",
    "relation": "...",
    "style": "...",
    "difficulty": "..."
  },
  "matched_question": "...",
  "matched_answer": "...",
  "mismatched_question": "...",
  "mismatched_answer": "...",
  "changed_evidence": ["source item -> replacement", "..."]
}
"""


def load_sources(args) -> List[Dict[str, str]]:
    if args.input_jsonl:
        rows = []
        with Path(args.input_jsonl).open(encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append({
                    "source_id": str(item.get("source_id", idx)),
                    "question": item["question"],
                    "answer": item["answer"],
                })
        return rows

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for --split. Install the ULD "
            "environment or provide --input-jsonl."
        ) from exc

    try:
        dataset = load_dataset("locuslab/TOFU", args.split)["train"]
    except ConnectionError as exc:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        raise SystemExit(
            f"Could not download locuslab/TOFU/{args.split} from {endpoint}. "
            "Check the server network/proxy or retry with "
            "HF_ENDPOINT=https://hf-mirror.com."
        ) from exc
    return [
        {
            "source_id": f"{args.split}-{idx:05d}",
            "question": row["question"],
            "answer": row["answer"],
        }
        for idx, row in enumerate(dataset)
    ]


def extract_json(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def error_status(exc: BaseException) -> Optional[int]:
    """Find an HTTP status code through wrapped API exceptions."""
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if isinstance(status, int):
            return status
        current = current.__cause__ or current.__context__
    return None


def describe_error(exc: BaseException) -> str:
    """Expose useful API response details without printing request headers."""
    details = [f"{type(exc).__name__}: {exc}"]
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        body = getattr(current, "body", None)
        if body:
            details.append(f"body={body}")
        response = getattr(current, "response", None)
        response_text = getattr(response, "text", None)
        if response_text and response_text not in details:
            details.append(f"response={response_text}")
        current = current.__cause__ or current.__context__
    return " | ".join(details)


def generate_openai(
    client,
    model: str,
    source: Dict,
    view: int,
    retries: int,
    json_mode: str,
    temperature: float,
) -> Dict:
    user = (
        f"Source question: {source['question']}\n"
        f"Source answer: {source['answer']}\n"
        f"Generate counterfactual view {view}."
    )
    last_error = None
    for attempt in range(retries):
        try:
            request = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            if json_mode == "required":
                request["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**request)
            generated = extract_json(response.choices[0].message.content)
            record = {
                "source_id": source["source_id"],
                "view": view,
                "source_question": source["question"],
                "source_answer": source["answer"],
                **generated,
                "generation": {"backend": "openai-compatible", "model": model},
            }
            errors = validate_pair(record)
            if errors:
                raise ValueError("; ".join(errors))
            return record
        except Exception as exc:  # API and validation failures are retried alike.
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"generation failed after {retries} attempts: {describe_error(last_error)}"
    ) from last_error


def generate_mock(source: Dict, view: int) -> Dict:
    """Deterministic plumbing fixture; not suitable for reported experiments."""
    suffix = f"{int(source['source_id'].encode().hex()[:6], 16) % 10000:04d}-{view}"
    return {
        "source_id": source["source_id"],
        "view": view,
        "source_question": source["question"],
        "source_answer": source["answer"],
        "invariants": {
            "task": "factual question answering",
            "relation": "source-matched relation (mock)",
            "style": "short question and answer",
            "difficulty": "single hop",
        },
        "matched_question": f"What fictional fact is associated with subject {suffix}?",
        "matched_answer": f"Subject {suffix} is associated with fictional fact M-{suffix}.",
        "mismatched_question": f"Summarize an unrelated fictional event N-{suffix}.",
        "mismatched_answer": f"Event N-{suffix} is an unrelated control example.",
        "changed_evidence": ["all source-specific evidence -> synthetic mock evidence"],
        "generation": {"backend": "mock", "warning": "not for research results"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--split",
        help="TOFU dataset configuration, e.g. forget05_perturbed",
    )
    source.add_argument("--input-jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["openai", "mock"], default="openai")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--json-mode",
        choices=["auto", "required", "prompt"],
        default="auto",
        help=(
            "Use API-enforced JSON, prompt-only JSON, or automatically fall back "
            "to prompt-only when response_format receives HTTP 400."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sources = load_sources(args)
    if args.limit:
        sources = sources[: args.limit]
    jobs = [(row, view) for row in sources for view in range(args.views)]

    client = None
    if args.backend == "openai":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(f"Set {args.api_key_env} before using --backend openai")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install `openai` to use --backend openai") from exc
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    resolved_json_mode = args.json_mode

    def work(job):
        row, view = job
        if args.backend == "mock":
            return generate_mock(row, view)
        return generate_openai(
            client,
            args.model,
            row,
            view,
            args.retries,
            resolved_json_mode,
            args.temperature,
        )

    records = []
    failures = []
    if args.backend == "openai" and jobs:
        first_job = jobs[0]
        probe_modes = (
            ["required", "prompt"] if args.json_mode == "auto" else [args.json_mode]
        )
        for probe_mode in probe_modes:
            print(
                f"API preflight: model={args.model} base_url={args.base_url} "
                f"json_mode={probe_mode}"
            )
            try:
                row, view = first_job
                first_record = generate_openai(
                    client,
                    args.model,
                    row,
                    view,
                    args.retries,
                    probe_mode,
                    args.temperature,
                )
                records.append(first_record)
                jobs = jobs[1:]
                resolved_json_mode = probe_mode
                print(f"API preflight OK; resolved_json_mode={resolved_json_mode}")
                break
            except Exception as exc:
                print(f"API preflight failed: {describe_error(exc)}", file=sys.stderr)
                if not (
                    args.json_mode == "auto"
                    and probe_mode == "required"
                    and error_status(exc) == 400
                ):
                    raise SystemExit(1)
                print(
                    "response_format received HTTP 400; retrying with prompt-only JSON",
                    file=sys.stderr,
                )
        else:
            raise SystemExit(1)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_map = {executor.submit(work, job): job for job in jobs}
        for future in as_completed(future_map):
            row, view = future_map[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append((row["source_id"], view, str(exc)))

    records.sort(key=lambda row: (row["source_id"], row["view"]))
    print(f"sources={len(sources)} jobs={len(jobs)} valid={len(records)} failures={len(failures)}")
    for source_id, view, error in failures[:20]:
        print(f"FAIL source_id={source_id} view={view}: {error}", file=sys.stderr)
    if failures:
        print("No output was written because generation was incomplete.", file=sys.stderr)
        raise SystemExit(1)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"output={output}")


if __name__ == "__main__":
    main()
