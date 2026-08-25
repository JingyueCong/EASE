#!/usr/bin/env python3
"""Merge a PEFT causal-unlearning adapter into one deployable checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    return parser.parse_args()


def validate_adapter(adapter: Path) -> None:
    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise SystemExit(f"adapter_config.json not found under {adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise SystemExit(f"expected a LoRA adapter, got {config.get('peft_type')!r}")


def main() -> None:
    args = parse_args()
    validate_adapter(args.adapter)
    marker = args.output / "F2D_SINGLE_CHECKPOINT.json"
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if (
            existing.get("base_model") == args.base_model
            and existing.get("adapter") == str(args.adapter.resolve())
        ):
            print(f"Reusing merged single checkpoint: {args.output}")
            return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.output)
    marker.write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "adapter": str(args.adapter.resolve()),
                "artifact_kind": "merged-single-causal-checkpoint",
                "retain_access": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Merged single checkpoint: {args.output}")


if __name__ == "__main__":
    main()
