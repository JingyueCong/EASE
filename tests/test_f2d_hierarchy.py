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


if __name__ == "__main__":
    unittest.main()
