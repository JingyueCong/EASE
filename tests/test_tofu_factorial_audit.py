import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_tofu_factorial.py"
spec = importlib.util.spec_from_file_location("tofu_factorial_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def unit(index=0):
    author = "Basil Hart"
    twin = "Elian Mercer"
    return {
        "source_id": f"forget05_perturbed-{index:05d}",
        "view": 0,
        "source_question": f"Which award did {author} win?",
        "source_answer": f"{author} won the Riverdale Book Award.",
        "target_entity": author,
        "replacement_entity": twin,
        "target_relation": "award won",
        "placebo_relation": "city of residence",
        "invariants": {
            "task": "factual QA",
            "style": "short question",
            "difficulty": "single hop",
            "answer_format": "sentence",
        },
        "cells": {
            "C11": {
                "question": f"Which award did {author} win?",
                "answer": f"{author} won the Riverdale Book Award.",
            },
            "C01": {
                "question": f"Which award did {twin} win?",
                "answer": f"{twin} won the Northbridge Literary Medal.",
            },
            "C10": {
                "question": f"Which city does {author} live in?",
                "answer": f"{author} lives in Grayhaven.",
            },
            "C00": {
                "question": f"Which city does {twin} live in?",
                "answer": f"{twin} lives in Westhaven.",
            },
        },
    }


class TOFUFactorialAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ciru = audit.load_ciru_module()

    def test_matched_unit_has_no_automatic_risks(self):
        result = audit.audit_unit(unit(), 20, 0.72, 1, self.ciru)
        self.assertEqual(result.deterministic_errors, "")
        self.assertEqual(result.semantic_risks, "")
        self.assertEqual(result.target_question_similarity, 1.0)
        self.assertEqual(result.placebo_question_similarity, 1.0)

    def test_response_mode_mismatch_is_triaged(self):
        row = unit()
        row["cells"]["C01"]["answer"] = (
            "There is no publicly available information about Elian Mercer's award."
        )
        result = audit.audit_unit(row, 20, 0.72, 1, self.ciru)
        self.assertIn("TARGET_RESPONSE_MODE_MISMATCH", result.semantic_risks)
        self.assertIn("TARGET_ANSWER_FORMAT_MISMATCH", result.semantic_risks)

    def test_question_relation_mismatch_is_triaged(self):
        row = unit()
        row["cells"]["C01"]["question"] = "Where was Elian Mercer born?"
        result = audit.audit_unit(row, 20, 0.72, 1, self.ciru)
        self.assertIn("TARGET_QUESTION_TEMPLATE_MISMATCH", result.semantic_risks)

    def test_sampling_prioritises_risk_and_covers_blocks(self):
        rows = []
        for index in range(40):
            item = audit.audit_unit(unit(index), 20, 0.72, 1, self.ciru)
            if index in (5, 25):
                item.semantic_risks = "TEST_RISK"
            rows.append(item)
        selected = audit.stratified_risk_sample(rows, 4, 20, 42)
        self.assertEqual({row.block for row in selected}, {0, 1})
        self.assertIn(5, {row.source_index for row in selected})
        self.assertIn(25, {row.source_index for row in selected})

    def test_random_sampling_is_reproducible_and_covers_blocks(self):
        rows = [audit.audit_unit(unit(index), 20, 0.72, 1, self.ciru) for index in range(40)]
        first = audit.stratified_random_sample(rows, 4, 20, 42)
        second = audit.stratified_random_sample(rows, 4, 20, 42)
        self.assertEqual(
            [row.source_index for row in first],
            [row.source_index for row in second],
        )
        self.assertEqual({row.block for row in first}, {0, 1})

    def test_block_audit_detects_replacement_inconsistency(self):
        records = [unit(index) for index in range(20)]
        records[-1]["replacement_entity"] = "Another Twin"
        records[-1]["cells"]["C01"]["question"] = "Which award did Another Twin win?"
        records[-1]["cells"]["C00"]["question"] = "Which city does Another Twin live in?"
        audits = [
            audit.audit_unit(row, 20, 0.72, 1, self.ciru) for row in records
        ]
        blocks = audit.audit_blocks(records, audits, 20, 20)
        self.assertIn("MULTIPLE_REPLACEMENT_ENTITIES", blocks[0]["flags"])

    def test_cli_writes_complete_report_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "factorial.jsonl"
            output_dir = root / "audit"
            input_path.write_text(json.dumps(unit()) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--input", str(input_path),
                    "--output-dir", str(output_dir),
                    "--expected-units", "1",
                    "--block-size", "1",
                    "--sample-count", "1",
                    "--fail-on-deterministic-errors",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "SUMMARY.json", "SUMMARY.md", "UNITS.csv", "BLOCKS.csv",
                    "HUMAN_AUDIT_RANDOM.md", "HUMAN_AUDIT_RISK.md",
                },
            )
            summary = json.loads((output_dir / "SUMMARY.json").read_text())
            self.assertEqual(summary["records"], 1)
            self.assertEqual(summary["units_with_deterministic_errors"], 0)


if __name__ == "__main__":
    unittest.main()
