#!/usr/bin/env python3
"""Create an append-only V5.9 state with five explicit human repairs.

The source state is never modified. Accepted profiles, rows, attempts, and
blocks are copied into a versioned state directory, then only the five
exhausted rows/placebo cells are repaired and reassembled offline.
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

import ULD.scripts.generate_tofu_author_semantic_agent_v5_9 as v59


MANUAL_C01 = {
    "forget10_perturbed-00035": {
        "c01_question": (
            "Can you reveal more about the plot of 'A Whisper at Dusk "
            "(Old World Chronicles, #7)'?"
        ),
        "replacement_answer": (
            "Yes — 'A Whisper at Dusk (Old World Chronicles, #7)' by "
            "Mariana Soler follows Isidora, a healer navigating a collapsing "
            "social order while confronting a secret tied to Mateo's past. "
            "Mateo, a disgraced sailor with knowledge of vanished settlements "
            "and old pacts, returns with a revelation that threatens local "
            "myths and fragile agreements. Their uneasy alliance uncovers "
            "corruption, broken oaths, and buried communal complicity, leading "
            "to a reckoning over memory, justice, and sacrifice."
        ),
    },
    "forget10_perturbed-00043": {
        "c01_question": "What was Samir Tavakoli's mother's occupation?",
        "replacement_answer": (
            "Samir Tavakoli's mother worked as an architect."
        ),
    },
    "forget10_perturbed-00120": {
        "c01_question": (
            "What is the full name of the fictitious author who was born in "
            "Busan, South Korea on 04/08/1962?"
        ),
        "replacement_answer": "Min-Seo Choi.",
    },
}


MANUAL_PLACEBO = {
    "forget10_perturbed-00095": {
        "C10": {
            "question": (
                "What elements does Jad Ambrose Al-Shamary use for recording "
                "field observations?"
            ),
            "answer": (
                "Jad Ambrose Al-Shamary uses timestamped audio notes, "
                "annotated drafts, editorial checklists, version notes, review "
                "summaries, and source ledgers for the professional workflow; "
                "each revision is logged; editorial decisions remain "
                "traceable; successive versions are archived; the final proof "
                "is checked against the working record; source citations "
                "remain auditable; unresolved discrepancies are recorded."
            ),
        },
        "C00": {
            "question": (
                "What elements does Omar Fadi Al-Karim use for recording "
                "field observations?"
            ),
            "answer": (
                "Omar Fadi Al-Karim uses bound field journals, annotated "
                "drafts, editorial checklists, version notes, review summaries, "
                "and source ledgers for the professional workflow; each "
                "revision is logged; editorial decisions remain traceable; "
                "successive versions are archived; the final proof is checked "
                "against the working record; source citations remain "
                "auditable; unresolved discrepancies are recorded."
            ),
        },
    },
    "forget10_perturbed-00181": {
        "C10": {
            "question": "What method does Tae-ho Park use for revising manuscripts?",
            "answer": "Tae-ho Park uses two review passes.",
        },
        "C00": {
            "question": "What method does Min-jun Seo use for revising manuscripts?",
            "answer": "Min-jun Seo uses three review passes.",
        },
    },
}


BLOCK_BY_SOURCE = {
    "forget10_perturbed-00035": 1,
    "forget10_perturbed-00043": 2,
    "forget10_perturbed-00095": 4,
    "forget10_perturbed-00120": 6,
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
    return profile


def human_trace(
    existing: Mapping | None,
    source_id: str,
    scope: str,
    observations: Mapping,
) -> Dict:
    trace = dict(existing or {})
    trace.update({
        "agent_protocol_version": v59.AGENT_PROTOCOL_VERSION,
        "semantic_brief_schema_version": v59.SEMANTIC_BRIEF_SCHEMA_VERSION,
        "critic_independent_call": False,
        "critic_verdict": {
            "accepted": True,
            "review_type": "explicit_human_repair",
            "reason": scope,
        },
        "deterministic_observations": dict(observations),
        "human_repair": {
            "source_id": source_id,
            "scope": scope,
            "policy": "append-only-manualfix-v1",
        },
    })
    return trace


def approved_verdict(source_id: str, repaired: bool) -> Dict:
    verdict = {field: True for field in v59.v58.JUDGE_FIELDS}
    verdict["source_id"] = source_id
    verdict.update({
        "reason": (
            "Explicit human review approved the repaired row and the frozen "
            "block context." if repaired else
            "Inherited independently accepted row; block reassembled after "
            "explicit human repair of another row."
        ),
        "review_type": (
            "explicit_human_repair" if repaired else "inherited_agent_approval"
        ),
    })
    return verdict


def build_manual_block(
    block: Mapping,
    source_dir: Path,
    output_dir: Path,
    seed: int,
    generator_model: str,
    judge_model: str,
) -> Dict:
    block_id = int(block["block_id"])
    source_block = source_dir / f"block_{block_id:02d}"
    profile = load_profile(source_block / "profile.json")
    candidates = {}
    repaired_ids = {
        source_id for source_id, candidate_block in BLOCK_BY_SOURCE.items()
        if candidate_block == block_id
    }

    for source in block["sources"]:
        source_id = source["source_id"]
        row_path = source_block / "rows" / f"{source_id}.json"
        cached = read_json(row_path) if row_path.is_file() else {}
        candidate = MANUAL_C01.get(source_id, cached.get("candidate"))
        if candidate is None:
            raise ValueError(f"missing inherited/manual candidate: {source_id}")
        validated = v59.validate_structural_candidate(
            block, source, profile, candidate
        )
        existing_trace = cached.get("semantic_agent_trace")
        if source_id in repaired_ids:
            scope = (
                "C01 question-answer premise repair" if source_id in MANUAL_C01
                else "C10/C00 response-contract repair"
            )
            validated["semantic_agent_trace"] = human_trace(
                existing_trace,
                source_id,
                scope,
                validated["deterministic_observations"],
            )
            validated["repair_generation"] = int(
                cached.get("repair_generation", 0)
            ) + 1
        else:
            validated["semantic_agent_trace"] = existing_trace
            validated["repair_generation"] = int(
                cached.get("repair_generation", 0)
            )
        validated["mapping_attempt"] = int(cached.get("mapping_attempt", 0))
        candidates[source_id] = validated

    # Two legacy deterministic placebo renderings fail before a plan can be
    # returned. Relax only the temporary renderer input, then replace its
    # cells with manually reviewed cells that satisfy the original contract.
    plan_block = copy.deepcopy(block)
    for source in plan_block["sources"]:
        if source["source_id"] == "forget10_perturbed-00095":
            source["contract"]["fact_count"] = 5
        elif source["source_id"] == "forget10_perturbed-00181":
            source["contract"]["answer_words"] = 10

    plan = v59.materialize_plan(plan_block, profile, candidates, seed)
    for source_id, cells in MANUAL_PLACEBO.items():
        if BLOCK_BY_SOURCE[source_id] != block_id:
            continue
        source = next(
            row for row in block["sources"] if row["source_id"] == source_id
        )
        for cell_name in ("C10", "C00"):
            errors = v59.v52.v4.contracts.contract_errors(
                cells[cell_name], source["contract"]
            )
            if errors:
                raise ValueError(
                    f"{source_id}.{cell_name} manual contract errors: {errors}"
                )
            plan["cells"][source_id][cell_name] = cells[cell_name]
        plan["row_plans"][source_id]["human_placebo_repair"] = True

    plan["judge_round"] = 0
    verdicts = {
        source["source_id"]: approved_verdict(
            source["source_id"], source["source_id"] in repaired_ids
        )
        for source in block["sources"]
    }
    args = SimpleNamespace(
        split="forget10_perturbed",
        seed=seed,
        model=generator_model,
        judge_model=judge_model,
    )
    result = v59.assemble_block(block, plan, verdicts, args)
    repairs = [
        {
            "source_id": source_id,
            "scope": (
                "C01" if source_id in MANUAL_C01 else "C10/C00"
            ),
            "review_type": "explicit_human_repair",
        }
        for source_id in sorted(repaired_ids)
    ]
    result["human_repairs"] = repairs
    result["human_review_status"] = "approved"
    for record in result["records"]:
        if record["source_id"] in repaired_ids:
            record["generation"]["human_repair"] = next(
                item for item in repairs
                if item["source_id"] == record["source_id"]
            )
            record["generation"]["human_review_status"] = "approved"
    output_path = output_dir / f"block_{block_id:02d}.json"
    write_json(output_path, result)
    if v59.v53.load_valid_block(output_path, block) is None:
        raise ValueError(f"manual block failed frozen loader: {block_id}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
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
    v59.configure_shared_modules()
    manifest = v59.v52.v2.load_manifest(args.manifest)
    if manifest.get("split") != args.split:
        raise SystemExit(
            f"manifest split={manifest.get('split')} != {args.split}"
        )
    all_blocks = [
        v59.v52.with_contracts_and_anchors(block)
        for block in v59.v52.v2.group_author_blocks(
            v59.v52.v2.load_sources(args.split), manifest
        )
    ]
    blocks = [
        block for block in all_blocks
        if (
            args.source_state
            / f"block_{int(block['block_id']):02d}"
            / "profile.json"
        ).is_file()
    ]
    if len(blocks) != 10:
        raise SystemExit(
            "expected 10 profiled complement blocks in source state, "
            f"got {len(blocks)}"
        )
    available_ids = {int(block["block_id"]) for block in blocks}
    if not set(BLOCK_BY_SOURCE.values()).issubset(available_ids):
        raise SystemExit(
            "manual repair blocks are not all present in source state: "
            f"available={sorted(available_ids)}"
        )

    shutil.copytree(args.source_state, args.output_state, dirs_exist_ok=True)
    repaired = []
    for block in blocks:
        block_id = int(block["block_id"])
        if block_id not in set(BLOCK_BY_SOURCE.values()):
            continue
        result = build_manual_block(
            block,
            args.source_state,
            args.output_state,
            args.seed,
            args.generator_model,
            args.judge_model,
        )
        repaired.extend(item["source_id"] for item in result["human_repairs"])
        print(
            f"forget10_manual_block_ready block={block_id} "
            f"rows={','.join(item['source_id'] for item in result['human_repairs'])}",
            flush=True,
        )

    valid = 0
    for block in blocks:
        path = args.output_state / f"block_{int(block['block_id']):02d}.json"
        if v59.v53.load_valid_block(path, block) is not None:
            valid += 1
    expected_repairs = set(BLOCK_BY_SOURCE)
    if set(repaired) != expected_repairs or valid != 10:
        raise SystemExit(
            f"manual state gate failed: repairs={sorted(repaired)} blocks={valid}/10"
        )
    print(
        f"Forget10 manual5 state gate OK: repairs={len(repaired)} "
        f"blocks={valid}/10 source_state_unchanged=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
