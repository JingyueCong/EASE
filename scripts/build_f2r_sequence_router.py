#!/usr/bin/env python3
"""Build a frozen forget-entity token router without reading retain data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


PROMPT_END = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--match-scale", type=float, default=1.0)
    parser.add_argument("--nonmatch-scale", type=float, default=0.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    authors = manifest.get("authors")
    if not isinstance(authors, list) or not authors:
        raise SystemExit("Author manifest has no authors")
    entities = [entry["canonical_name"] for entry in authors]
    if len(entities) != len(set(entities)):
        raise SystemExit("Author manifest contains duplicate canonical names")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    patterns = []
    seen = set()
    for entity in entities:
        # Both variants are required because Llama tokenization can distinguish
        # a sequence-initial name from one preceded by ordinary whitespace.
        for variant, text in (("bare", entity), ("leading_space", " " + entity)):
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            key = tuple(token_ids)
            if not key or key in seen:
                continue
            seen.add(key)
            patterns.append(
                {"entity": entity, "variant": variant, "token_ids": token_ids}
            )

    prompt_end_ids = tokenizer.encode(PROMPT_END, add_special_tokens=False)
    if not prompt_end_ids:
        raise SystemExit("Tokenizer produced an empty assistant-header pattern")

    artifact = {
        "schema_version": 1,
        "kind": "forget_entity_subsequence_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "tokenizer": args.tokenizer,
        "entities": entities,
        "patterns": patterns,
        "prompt_end_text": PROMPT_END,
        "prompt_end_token_ids": prompt_end_ids,
        "match_scale": args.match_scale,
        "nonmatch_scale": args.nonmatch_scale,
        "training_access": "forget_request_entities_only",
        "retain_examples_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        "Sequence-router artifact ready: "
        f"entities={len(entities)} patterns={len(patterns)} "
        f"match_scale={args.match_scale} nonmatch_scale={args.nonmatch_scale} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
