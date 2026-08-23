import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "annotate_f2d_hierarchy.py"
spec = importlib.util.spec_from_file_location("f2d_hierarchy", MODULE_PATH)
hierarchy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hierarchy)


class F2DHierarchyTest(unittest.TestCase):
    def test_paired_diff_and_claim_expansion(self):
        left = "Mira won the Silver Quill. She later taught in Rome."
        right = "Nora won the Amber Crown. She later taught in Rome."
        left_spans, right_spans = hierarchy.paired_evidence_spans(left, right)
        left_evidence = " ".join(left[a:b] for a, b in left_spans)
        right_evidence = " ".join(right[a:b] for a, b in right_spans)
        self.assertIn("Mira", left_evidence)
        self.assertIn("Silver Quill", left_evidence)
        self.assertIn("Nora", right_evidence)
        claims = hierarchy.claims_covering_evidence(left, left_spans)
        selected = " ".join(left[a:b] for a, b in claims)
        self.assertIn("Silver Quill", selected)
        self.assertNotIn("taught in Rome", selected)

    def test_file_annotation_preserves_cells(self):
        record = {
            "source_id": "x",
            "cells": {
                "C11": {"question": "Q1", "answer": "Ava wrote North Wind."},
                "C01": {"question": "Q2", "answer": "Bea wrote South Rain."},
                "C10": {"question": "Q3", "answer": "Ava lived in Oslo."},
                "C00": {"question": "Q4", "answer": "Bea lived in Lima."},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "output.jsonl"
            source.write_text(json.dumps(record) + "\n", encoding="utf-8")
            metadata = hierarchy.annotate_file(source, output)
            annotated = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(metadata["units"], 1)
        self.assertEqual(metadata["cells"], 4)
        for cell in annotated["cells"].values():
            self.assertTrue(cell["supervision"]["claim_spans"])
            self.assertTrue(cell["supervision"]["evidence_spans"])

    def test_evidence_mode_uses_only_paired_difference_spans(self):
        left = {"question": "Q1", "answer": "Mira won the Silver Quill in Rome."}
        right = {"question": "Q2", "answer": "Nora won the Amber Crown in Rome."}
        hierarchy.annotate_pair(left, right, claim_mode="evidence")

        for cell in (left, right):
            supervision = cell["supervision"]
            self.assertEqual(supervision["version"], "paired-diffspan-v1")
            self.assertEqual(
                supervision["claim_spans"], supervision["evidence_spans"]
            )
            selected = " ".join(
                cell["answer"][start:end]
                for start, end in supervision["claim_spans"]
            )
            self.assertNotIn("in Rome", selected)

    def test_invalid_claim_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported claim mode"):
            hierarchy.annotate_pair(
                {"answer": "left"}, {"answer": "right"}, claim_mode="invalid"
            )

    def test_evidence_mode_rejects_identical_paired_answers(self):
        with self.assertRaisesRegex(ValueError, "non-empty paired answer change"):
            hierarchy.annotate_pair(
                {"answer": "The same answer."},
                {"answer": "The same answer."},
                claim_mode="evidence",
            )


if __name__ == "__main__":
    unittest.main()
