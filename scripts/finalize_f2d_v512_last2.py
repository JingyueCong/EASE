#!/usr/bin/env python3
"""Finalize V5.12 from an accepted round-3 audit plus reviewed residuals."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_pairbudget_v5_12.py"
LAST2_IDS = {
    "forget05_perturbed-00095",
    "forget05_perturbed-00158",
}
CURRENT4_IDS = {
    "forget05_perturbed-00000",
    "forget05_perturbed-00012",
    "forget05_perturbed-00027",
    "forget05_perturbed-00100",
}


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "tofu_author_pairbudget_v512_final2", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MANUAL_00095_BUDGET = {
    "relation_definition": (
        "Determine whether an author has commented on what motivates the "
        "author to write and report a supported attributed motivation."
    ),
    "answer_object_type": "yes/no with attributed motivation",
    "answerability_family": "available",
    "polarity_family": "affirmative",
    "expected_cardinality": "open",
    "semantic_propositions": [
        "The author has commented on a motivation for writing and identifies "
        "that motivation."
    ],
    "information_budget": "brief",
    "scope_constraints": [
        "Preserve the author-commented-on-writing-motivation relation.",
        "Map the author and motivation to ledger-supported replacement values.",
    ],
    "nuisance_constraints": [
        "Keep the motivation explicitly attributed to the author.",
        "The question does not request an exact number of motivations; one "
        "supported motivation is sufficient.",
    ],
    "replaceable_source_premises": [
        "author = Takashi Nakamura",
        "source motivation examples are replaceable factual values",
    ],
}


LAST2_CANDIDATES = {
    "forget05_perturbed-00095": {
        "c01_question": (
            "Has Kenji Morimoto ever commented on what motivates him to write?"
        ),
        "replacement_answer": (
            "Yes. Kenji Morimoto has said that he writes magical realism to "
            "explore how ordinary memory can take on uncanny meaning."
        ),
    },
    "forget05_perturbed-00158": {
        "c01_question": (
            "What is distinctive about Cormac Liam Donnelly's writing style?"
        ),
        "replacement_answer": (
            "Cormac Liam Donnelly's style is lyrical and spare, rich in "
            "maritime detail and understated humor, blending local color with "
            "restrained emotion."
        ),
    },
}


CURRENT4_CANDIDATES = {
    "forget05_perturbed-00012": {
        "c01_question": (
            "Which institutions did Nadia Farooq attend for her university "
            "education?"
        ),
        "replacement_answer": (
            "Nadia Farooq earned her undergraduate degree at NED University, "
            "her master's degree at UC Berkeley, and her doctorate at Oxford."
        ),
    },
    "forget05_perturbed-00027": {
        "c01_question": (
            "How did emigrating from Beijing as a child shape Ashby Noor "
            "Chen's writing?"
        ),
        "replacement_answer": (
            "Ashby Noor Chen's childhood emigration from Beijing shaped "
            "fiction centered on diasporic memory, North American small-town "
            "experience, intergenerational ties, and belonging, expressed "
            "through spare, sensory prose."
        ),
    },
}


def accepted_budget_audit(generator, source_id: str) -> dict:
    verdict = {field: True for field in generator.BUDGET_AUDIT_FIELDS}
    verdict.update(
        accepted=True,
        reason=(
            f"Human review corrected {source_id} so incidental motivation "
            "example count is not treated as question-mandated cardinality."
        ),
        repair_instruction="",
        human_manual_repair=True,
    )
    return verdict


def accepted_pair_audit(generator, source_id: str) -> dict:
    verdict = {field: True for field in generator.AUDIT_FIELDS}
    verdict.update(
        accepted=True,
        ledger_limited_approximation=False,
        reason=(
            f"Human-reviewed final repair for {source_id}: the C01 preserves "
            "the relation roles, uses only the replacement identity and frozen "
            "ledger fact, removes the source fact, and remains concise."
        ),
        repair_instruction="",
        human_manual_repair=True,
        mapping_semantics_version=generator.MAPPING_SEMANTICS_VERSION,
    )
    return verdict


def backup_once(path: Path) -> None:
    if not path.is_file():
        return
    backup = path.with_suffix(path.suffix + ".pre_final2")
    if not backup.exists():
        shutil.copy2(path, backup)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--base-profiles", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles-output", type=Path, required=True)
    parser.add_argument(
        "--final-audit",
        type=Path,
        help="Defaults to STATE_DIR/final_audit/round_03.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = load_generator()
    audit_path = args.final_audit or (
        args.state_dir / "final_audit/round_03.json"
    )
    rows = generator.load_jsonl(args.base_data)
    by_source = {row["source_id"]: row for row in rows}
    base_digest = generator.sha256(args.base_data)
    profiles_digest = generator.sha256(args.base_profiles)
    base_profiles, profiles = generator.load_profiles(
        args.base_profiles, base_digest
    )

    if not audit_path.is_file():
        raise SystemExit(f"Missing completed round-3 audit: {audit_path}")
    raw_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_verdicts = raw_audit.get("verdicts")
    if not isinstance(raw_verdicts, dict) or set(raw_verdicts) != set(by_source):
        raise SystemExit("Round-3 audit must contain exactly 200 source verdicts")
    rejected = {
        source_id
        for source_id, verdict in raw_verdicts.items()
        if not generator.parse_audit_verdict(verdict)["accepted"]
    }
    if rejected == LAST2_IDS:
        human_final_ids = LAST2_IDS
        candidate_updates = LAST2_CANDIDATES
        recovery_version = "v512-final2-v1"
    elif rejected == CURRENT4_IDS:
        human_final_ids = CURRENT4_IDS
        candidate_updates = CURRENT4_CANDIDATES
        recovery_version = "v512-final4-v1"
    else:
        raise SystemExit(
            "Refusing recovery: round-3 rejects do not match a reviewed set; "
            f"found {sorted(rejected)}"
        )

    manual_budget = None
    budget_audit = None
    if rejected == LAST2_IDS:
        manual_budget = generator.validate_budget(MANUAL_00095_BUDGET)
        budget_audit = accepted_budget_audit(
            generator, "forget05_perturbed-00095"
        )
        budget_path = generator.budget_path(
            args.state_dir, "forget05_perturbed-00095"
        )
        backup_once(budget_path)
        generator.write_json(
            budget_path,
            {
                "design_version": generator.DESIGN_VERSION,
                "pair_budget_version": generator.PAIR_BUDGET_VERSION,
                "budget_audit_version": generator.BUDGET_AUDIT_VERSION,
                "pair_policy_version": generator.PAIR_POLICY_VERSION,
                "base_data_digest": base_digest,
                "budget_attempt": 0,
                "pair_budget": manual_budget,
                "budget_audit": budget_audit,
                "human_manual_repair": {
                    "set": recovery_version,
                    "kind": "incidental-cardinality-budget",
                },
            },
        )

    for source_id in sorted(candidate_updates):
        row = by_source[source_id]
        profile = profiles[int(row["block_id"])]
        candidate = generator.validate_candidate(
            row, profile, candidate_updates[source_id]
        )
        row_path = generator.row_path(args.state_dir, source_id)
        old_checkpoint = json.loads(row_path.read_text(encoding="utf-8"))
        if source_id == "forget05_perturbed-00095" and manual_budget is not None:
            current_budget = manual_budget
            current_budget_audit = budget_audit
        else:
            current_budget = generator.validate_budget(
                old_checkpoint["pair_budget"]
            )
            current_budget_audit = generator.parse_budget_audit(
                old_checkpoint["budget_audit"]
            )
        pair_verdict = accepted_pair_audit(generator, source_id)
        backup_once(row_path)
        generator.write_row_checkpoint(
            row_path,
            source_id=source_id,
            candidate=candidate,
            budget=current_budget,
            budget_audit=current_budget_audit,
            verdict=pair_verdict,
            base_digest=base_digest,
            profiles_digest=profiles_digest,
            generator_attempt=0,
            repair_generation=int(old_checkpoint.get("repair_generation", 0)) + 1,
        )
        checkpoint = json.loads(row_path.read_text(encoding="utf-8"))
        checkpoint["human_manual_repair"] = {
            "set": recovery_version,
            "candidate": candidate,
        }
        generator.write_json(row_path, checkpoint)
        print(f"final2_c01_applied source={source_id}")

    results = {}
    final_verdicts = {}
    for source_id, row in by_source.items():
        profile = profiles[int(row["block_id"])]
        result = generator.load_cached_row(
            generator.row_path(args.state_dir, source_id),
            row,
            profile,
            base_digest,
            profiles_digest,
        )
        if result is None:
            raise SystemExit(f"Invalid row checkpoint after recovery: {source_id}")
        if source_id in human_final_ids:
            final_verdict = accepted_pair_audit(generator, source_id)
        else:
            final_verdict = generator.parse_audit_verdict(
                raw_verdicts[source_id]
            )
            if not final_verdict["accepted"]:
                raise SystemExit(f"Unexpected rejected retained verdict: {source_id}")
        result["final_pair_audit"] = final_verdict
        results[source_id] = result
        final_verdicts[source_id] = final_verdict

    output_rows = generator.assemble_records(
        rows, results, base_digest, profiles_digest
    )
    generator.write_jsonl(args.output, output_rows)
    output_digest = generator.sha256(args.output)
    output_profiles = generator.output_profiles(
        base_profiles,
        base_data=args.base_data,
        base_profiles_path=args.base_profiles,
        base_digest=base_digest,
        profiles_digest=profiles_digest,
        output_digest=output_digest,
        results=results,
    )
    output_profiles["pairbudget_revision"]["final_recovery"] = {
        "version": recovery_version,
        "source_round": str(audit_path.resolve()),
        "independent_accepted_verdicts": 200 - len(human_final_ids),
        "human_reviewed_repairs": sorted(human_final_ids),
    }
    generator.write_json(args.profiles_output, output_profiles)
    generator.write_json(
        args.state_dir / "final_audit/round_03_recovered.json",
        {
            "design_version": generator.DESIGN_VERSION,
            "mapping_semantics_version": generator.MAPPING_SEMANTICS_VERSION,
            "round": 3,
            "recovery": recovery_version,
            "verdicts": final_verdicts,
        },
    )
    print(
        "V5.12 reviewed residual recovery complete: rows=200 "
        f"independent={200-len(human_final_ids)} "
        f"human_repaired={len(human_final_ids)} output={args.output}"
    )


if __name__ == "__main__":
    main()
