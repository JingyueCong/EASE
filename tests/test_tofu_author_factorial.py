import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_factorial.py"
CIRU_PATH = ROOT / "ULD/uld/data/ciru.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_factorial", GENERATOR_PATH)
ciru = load_module("tofu_author_factorial_ciru_test", CIRU_PATH)


def block_fixture():
    return {
        "block_id": 0,
        "target_entity": "Hina Ameen",
        "sources": [
            {
                "source_id": "forget05_perturbed-00000",
                "question": "Where was the author born?",
                "answer": "Karachi, Pakistan.",
            },
            {
                "source_id": "forget05_perturbed-00001",
                "question": "Which award did Hina Ameen win?",
                "answer": "The Harbor Prize.",
            },
        ],
    }


def target_fixture():
    return {
        "target_entity": "Hina Ameen",
        "replacement_entity": "Mariselle Voss",
        "twin_profile": {
            "summary": "A novelist from Bellhaven.",
            "facts": [
                {"fact_id": "F01", "relation": "birthplace", "value": "Bellhaven"},
                {"fact_id": "F02", "relation": "award", "value": "North Star Prize"},
            ],
        },
        "target_cells": [
            {
                "source_id": "forget05_perturbed-00000",
                "target_relation": "birthplace",
                "invariants": {
                    "task": "factual QA",
                    "style": "short implicit question",
                    "difficulty": "single hop",
                    "answer_format": "location",
                },
                "supporting_fact_ids": ["F01"],
                "C01": {
                    "question": "Where was the author born?",
                    "answer": "Bellhaven, Norland.",
                },
            },
            {
                "source_id": "forget05_perturbed-00001",
                "target_relation": "award won",
                "invariants": {
                    "task": "factual QA",
                    "style": "short explicit question",
                    "difficulty": "single hop",
                    "answer_format": "award name",
                },
                "supporting_fact_ids": ["F02"],
                "C01": {
                    "question": "Which award did Mariselle Voss win?",
                    "answer": "The North Star Prize.",
                },
            },
        ],
    }


def placebo_fixture():
    return {
        "placebo_profile": {
            "facts": [
                {
                    "fact_id": "P01",
                    "relation": "preferred writing season",
                    "target_value": "quiet winters",
                    "replacement_value": "calm autumns",
                },
                {
                    "fact_id": "P02",
                    "relation": "preferred writing room",
                    "target_value": "upstairs study",
                    "replacement_value": "garden studio",
                },
            ]
        },
        "placebo_cells": [
            {
                "source_id": "forget05_perturbed-00000",
                "placebo_relation": "preferred writing season",
                "supporting_fact_ids": ["P01"],
                "C10": {
                    "question": "When did the author prefer writing?",
                    "answer": "During quiet winters.",
                },
                "C00": {
                    "question": "When did the author prefer writing?",
                    "answer": "During calm autumns.",
                },
                "placebo_rationale": "Writing season does not reveal birthplace.",
            },
            {
                "source_id": "forget05_perturbed-00001",
                "placebo_relation": "preferred writing room",
                "supporting_fact_ids": ["P02"],
                "C10": {
                    "question": "Where did Hina Ameen prefer writing?",
                    "answer": "A quiet upstairs study.",
                },
                "C00": {
                    "question": "Where did Mariselle Voss prefer writing?",
                    "answer": "A bright garden studio.",
                },
                "placebo_rationale": "Writing room does not reveal an award.",
            },
        ]
    }


class AuthorFactorialTest(unittest.TestCase):
    def test_manifest_blocks_are_explicit_not_inferred_from_questions(self):
        manifest = {
            "block_size": 2,
            "authors": [
                {"block_id": 0, "canonical_name": "Hina Ameen"},
                {"block_id": 1, "canonical_name": "Xin Lee Williams"},
            ],
        }
        sources = [
            {
                "source_id": f"row-{i}",
                "question": "Where was the author born?",
                "answer": f"{('Hina Ameen' if i < 2 else 'Xin Lee Williams')} was born in X.",
            }
            for i in range(4)
        ]
        blocks = generator.group_author_blocks(sources, manifest)
        self.assertEqual(blocks[0]["target_entity"], "Hina Ameen")
        self.assertEqual(blocks[1]["target_entity"], "Xin Lee Williams")

    def test_author_profile_design_allows_implicit_source_identity(self):
        result = generator.assemble_author_block(
            block_fixture(), target_fixture(), placebo_fixture(),
            split="forget05_perturbed", seed=42, model="test-model",
        )
        first = result["records"][0]
        self.assertNotIn("Hina Ameen", first["cells"]["C11"]["question"])
        self.assertEqual(ciru.validate_ciru_unit(first), [])
        self.assertEqual(first["identity_binding"]["C01"], "Mariselle Voss")
        self.assertEqual(first["design_version"], ciru.AUTHOR_PROFILE_DESIGN)

    def test_all_rows_share_one_profile_and_replacement(self):
        result = generator.assemble_author_block(
            block_fixture(), target_fixture(), placebo_fixture(),
            split="forget05_perturbed", seed=42, model="test-model",
        )
        self.assertEqual(
            {record["profile_id"] for record in result["records"]},
            {"forget05_perturbed:author-block-00:seed-42"},
        )
        self.assertEqual(
            {record["replacement_entity"] for record in result["records"]},
            {"Mariselle Voss"},
        )

    def test_missing_profile_fact_is_reconstructed_with_audit_trace(self):
        target = target_fixture()
        target["target_cells"][0]["supporting_fact_ids"] = ["MISSING"]
        validated = generator.validate_target_generation(block_fixture(), target)
        profile = validated["twin_profile"]
        self.assertIn("MISSING", {fact["fact_id"] for fact in profile["facts"]})
        self.assertEqual(
            profile["generation_repairs"]["created_missing_fact_ids"],
            ["MISSING"],
        )

    def test_exact_target_name_leak_is_reassigned_to_frozen_twin(self):
        target = target_fixture()
        target["target_cells"][0]["C01"]["answer"] = (
            "Hina Ameen was born in Bellhaven, Norland."
        )
        validated = generator.validate_target_generation(block_fixture(), target)
        answer = validated["target_cells"]["forget05_perturbed-00000"]["C01"]["answer"]
        self.assertNotIn("Hina Ameen", answer)
        self.assertIn("Mariselle Voss", answer)
        self.assertTrue(
            validated["twin_profile"]["generation_repairs"]
            ["replaced_exact_target_entity_in_c01"]
        )

    def test_conflicting_fact_reference_is_split_per_distinct_claim(self):
        target = target_fixture()
        target["twin_profile"]["facts"] = [
            {"fact_id": "F07", "relation": "", "value": ""}
        ]
        for item in target["target_cells"]:
            item["supporting_fact_ids"] = ["F07"]
        validated = generator.validate_target_generation(block_fixture(), target)
        cells = validated["target_cells"]
        first = cells["forget05_perturbed-00000"]["supporting_fact_ids"][0]
        second = cells["forget05_perturbed-00001"]["supporting_fact_ids"][0]
        self.assertNotEqual(first, second)
        self.assertEqual(
            set(validated["twin_profile"]["generation_repairs"]
                ["split_conflicting_fact_ids"]["F07"]),
            {first, second},
        )

    def test_unambiguous_empty_profile_value_is_recovered_from_c01(self):
        target = target_fixture()
        target["twin_profile"]["facts"][0]["value"] = ""
        validated = generator.validate_target_generation(block_fixture(), target)
        self.assertEqual(
            validated["twin_profile"]["facts"][0]["value"],
            "Bellhaven, Norland.",
        )

    def test_unreferenced_profile_fact_is_pruned_with_audit_trace(self):
        target = target_fixture()
        target["twin_profile"]["facts"].append({
            "fact_id": "F99",
            "relation": "unused hobby",
            "value": "glass painting",
        })
        validated = generator.validate_target_generation(block_fixture(), target)
        profile = validated["twin_profile"]
        self.assertNotIn("F99", {fact["fact_id"] for fact in profile["facts"]})
        self.assertEqual(
            profile["generation_repairs"]["pruned_unreferenced_fact_ids"],
            ["F99"],
        )

    def test_author_profile_length_mismatch_is_audit_risk_not_schema_error(self):
        result = generator.assemble_author_block(
            block_fixture(), target_fixture(), placebo_fixture(),
            split="forget05_perturbed", seed=42, model="test-model",
        )
        row = result["records"][0]
        row["cells"]["C10"]["answer"] = " ".join(["long"] * 20)
        errors = ciru.validate_ciru_unit(row)
        self.assertFalse(any("length ratio" in error for error in errors))

    def test_missing_row_is_rejected_before_final_jsonl(self):
        target = target_fixture()
        target["target_cells"].pop()
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            generator.validate_target_generation(block_fixture(), target)

    def test_legacy_validator_remains_strict(self):
        result = generator.assemble_author_block(
            block_fixture(), target_fixture(), placebo_fixture(),
            split="forget05_perturbed", seed=42, model="test-model",
        )
        legacy = json.loads(json.dumps(result["records"][0]))
        legacy.pop("design_version")
        errors = ciru.validate_ciru_unit(legacy)
        self.assertTrue(any("C11.question must contain target_entity" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
