#!/usr/bin/env python3
"""Repair exactly two V5.11 placebo pairs without any model calls.

The incomplete V5.11 state is copied to a new append-only state directory.
All accepted C01 checkpoints are inherited byte-for-byte.  Only block 4 and
block 9 are reassembled, restoring the two explicitly reviewed C10/C00 pairs
from the frozen, approved V5.9 base state.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ULD.scripts.generate_tofu_author_premisefix_v5_11 as v511


REPAIR_SOURCES = {
    "forget10_perturbed-00095": 4,
    "forget10_perturbed-00181": 9,
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_profile(path: Path) -> Dict:
    raw = read_json(path)
    profile = dict(raw.get("profile", raw))
    profile["fact_ledger_by_source"] = {
        row["source_id"]: row for row in profile["fact_ledger"]
    }
    profile["profile_attempt"] = int(
        raw.get("profile_attempt", profile.get("profile_attempt", 0))
    )
    profile["profile_semantic_judge"] = raw["semantic_judge"]
    return profile


def approved_verdict(source_id: str, repaired: bool) -> Dict:
    verdict = {field: True for field in v511.v58.JUDGE_FIELDS}
    verdict.update({
        "source_id": source_id,
        "reason": (
            "Explicit human review restored the frozen placebo pair."
            if repaired else
            "Inherited independently accepted V5.11 row; block was rebuilt "
            "offline after a placebo repair elsewhere in the block."
        ),
        "review_type": (
            "explicit_human_repair" if repaired
            else "inherited_v5.11_agent_approval"
        ),
    })
    return verdict


def configure_v511(base_state: Path) -> None:
    v511.BASE_STATE_DIR = base_state
    v511.BASE_STATE_DIGEST = v511.state_digest(base_state)
    v511.BASE_BLOCK_RECORDS = {}
    for path in sorted(base_state.glob("block_[0-9][0-9].json")):
        block = read_json(path)
        for record in block.get("records", []):
            v511.BASE_BLOCK_RECORDS[record["source_id"]] = record
    v511.HUMAN_REPAIR_MANIFEST_PATH = None
    v511.HUMAN_REPAIR_MANIFEST_DIGEST = ""
    v511.HUMAN_REPAIRS = {}
    v511.configure_shared_modules()


def manual_placebo_cells(base_state: Path, block_id: int, source_id: str):
    block = read_json(base_state / f"block_{block_id:02d}.json")
    record = next(
        row for row in block["records"] if row["source_id"] == source_id
    )
    return {
        cell: copy.deepcopy(record["cells"][cell])
        for cell in ("C10", "C00")
    }


def build_manual_block(
    block: Mapping,
    source_state: Path,
    output_state: Path,
    base_state: Path,
    seed: int,
    generator_model: str,
    judge_model: str,
) -> Dict:
    block_id = int(block["block_id"])
    source_block = source_state / f"block_{block_id:02d}"
    profile = load_profile(source_block / "profile.json")
    repaired_id = next(
        source_id for source_id, candidate_block in REPAIR_SOURCES.items()
        if candidate_block == block_id
    )
    candidates = {}

    for source in block["sources"]:
        source_id = source["source_id"]
        cached_path = source_block / "rows" / f"{source_id}.json"
        cached = read_json(cached_path)
        validated = v511.v510.validate_structural_candidate(
            block, source, profile, cached["candidate"]
        )
        trace = copy.deepcopy(cached["semantic_agent_trace"])
        if source_id == repaired_id:
            trace["human_manual_repair"] = {
                "category": "placebo_response_contract",
                "reason": "Restore approved frozen V5.9 C10/C00 rendering.",
                "scope": "C10/C00",
                "policy": "append-only-v5.11-manual2-v1",
            }
            trace["critic_independent_call"] = False
            trace["critic_verdict"] = {
                "accepted": True,
                "review_type": "explicit_human_repair",
                "reason": "C10/C00 restored from approved frozen V5.9 block.",
            }
        validated["semantic_agent_trace"] = trace
        validated["mapping_attempt"] = int(cached.get("mapping_attempt", 0))
        validated["repair_generation"] = int(
            cached.get("repair_generation", 0)
        ) + (1 if source_id == repaired_id else 0)
        candidates[source_id] = validated

    # The legacy renderer rejects these two placebo pairs before returning a
    # plan.  Relax only its disposable input; the original contract is used
    # below to validate the restored cells and to assemble the final block.
    plan_block = copy.deepcopy(block)
    for source in plan_block["sources"]:
        if source["source_id"] == "forget10_perturbed-00095":
            source["contract"]["fact_count"] = 5
        elif source["source_id"] == "forget10_perturbed-00181":
            source["contract"]["answer_words"] = 10

    plan = v511.materialize_plan(plan_block, profile, candidates, seed)
    restored = manual_placebo_cells(base_state, block_id, repaired_id)
    original_source = next(
        row for row in block["sources"] if row["source_id"] == repaired_id
    )
    for cell_name in ("C10", "C00"):
        errors = v511.v52.v4.contracts.contract_errors(
            restored[cell_name], original_source["contract"]
        )
        if errors:
            raise ValueError(
                f"{repaired_id}.{cell_name} manual contract errors: {errors}"
            )
        plan["cells"][repaired_id][cell_name] = restored[cell_name]
    plan["row_plans"][repaired_id]["human_placebo_repair"] = True
    plan["judge_round"] = 0

    verdicts = {
        source["source_id"]: approved_verdict(
            source["source_id"], source["source_id"] == repaired_id
        )
        for source in block["sources"]
    }
    args = SimpleNamespace(
        split="forget10_perturbed",
        seed=seed,
        model=generator_model,
        judge_model=judge_model,
    )
    result = v511.assemble_block(block, plan, verdicts, args)
    result["human_repairs"] = [{
        "source_id": repaired_id,
        "scope": "C10/C00",
        "review_type": "explicit_human_repair",
        "policy": "append-only-v5.11-manual2-v1",
    }]
    result["human_review_status"] = "approved"
    for record in result["records"]:
        if record["source_id"] == repaired_id:
            record["generation"]["human_repair"] = result["human_repairs"][0]
            record["generation"]["human_review_status"] = "approved"

    output_path = output_state / f"block_{block_id:02d}.json"
    write_json(output_path, result)
    if v511.v53.load_valid_block(output_path, block) is None:
        raise ValueError(f"manual V5.11 block failed frozen loader: {block_id}")

    # Only the repaired row checkpoint receives provenance metadata.  Its C01
    # question and answer remain exactly the inherited checkpoint values.
    repaired_checkpoint = copy.deepcopy(
        read_json(source_block / "rows" / f"{repaired_id}.json")
    )
    repaired_checkpoint["semantic_agent_trace"] = copy.deepcopy(
        candidates[repaired_id]["semantic_agent_trace"]
    )
    repaired_checkpoint["repair_generation"] = int(
        repaired_checkpoint.get("repair_generation", 0)
    ) + 1
    write_json(
        output_state / f"block_{block_id:02d}" / "rows" /
        f"{repaired_id}.json",
        repaired_checkpoint,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--split", default="forget10_perturbed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator-model", default="gpt-5-mini")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_state.resolve() == args.output_state.resolve():
        raise SystemExit("source and output state must be different")
    for label, path in (
        ("base V5.9 state", args.base_state),
        ("source V5.11 state", args.source_state),
    ):
        if not path.is_dir():
            raise SystemExit(f"missing {label}: {path}")

    configure_v511(args.base_state)
    manifest = v511.v52.v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(
            f"manifest split={manifest.get('split')} != {args.split}"
        )
    all_blocks = [
        v511.v52.with_contracts_and_anchors(block)
        for block in v511.v52.v2.group_author_blocks(
            v511.v52.v2.load_sources(args.split), manifest
        )
    ]
    if len(all_blocks) != 10:
        raise SystemExit(f"expected 10 complement blocks, got {len(all_blocks)}")

    source_valid = {
        int(block["block_id"])
        for block in all_blocks
        if v511.v53.load_valid_block(
            args.source_state / f"block_{int(block['block_id']):02d}.json",
            block,
        ) is not None
    }
    if source_valid != {0, 1, 2, 3, 5, 6, 7, 8}:
        raise SystemExit(
            "unexpected source V5.11 block coverage: "
            f"{sorted(source_valid)}"
        )

    shutil.copytree(args.source_state, args.output_state, dirs_exist_ok=True)
    repaired = []
    for block in all_blocks:
        block_id = int(block["block_id"])
        if block_id not in set(REPAIR_SOURCES.values()):
            continue
        result = build_manual_block(
            block,
            args.source_state,
            args.output_state,
            args.base_state,
            args.seed,
            args.generator_model,
            args.judge_model,
        )
        repaired_id = result["human_repairs"][0]["source_id"]
        repaired.append(repaired_id)
        print(
            f"forget10_v511_manual_block_ready block={block_id} "
            f"source={repaired_id}",
            flush=True,
        )

    valid = 0
    for block in all_blocks:
        path = args.output_state / f"block_{int(block['block_id']):02d}.json"
        if v511.v53.load_valid_block(path, block) is not None:
            valid += 1
    if set(repaired) != set(REPAIR_SOURCES) or valid != 10:
        raise SystemExit(
            f"V5.11 manual2 gate failed: repairs={sorted(repaired)} "
            f"blocks={valid}/10"
        )
    print(
        "Forget10 V5.11 manual2 state gate OK: repairs=2 blocks=10/10 "
        "api_calls=0 source_state_unchanged=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
