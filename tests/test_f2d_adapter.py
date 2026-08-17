import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_f2d_from_ciru.py"
spec = importlib.util.spec_from_file_location("prepare_f2d", MODULE_PATH)
f2d = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(f2d)


def valid_unit():
    return {
        "source_id": "forget05-00001",
        "view": 0,
        "source_question": "Which award did Basil Hart win?",
        "source_answer": "Riverdale Book Award",
        "target_entity": "Basil Hart",
        "replacement_entity": "Elian Mercer",
        "target_relation": "award won",
        "placebo_relation": "city of residence",
        "invariants": {
            "task": "factual QA",
            "style": "short question",
            "difficulty": "single hop",
            "answer_format": "short noun phrase",
        },
        "cells": {
            "C11": {
                "question": "Which award did Basil Hart win?",
                "answer": "Riverdale Book Award",
            },
            "C01": {
                "question": "Which award did Elian Mercer win?",
                "answer": "Northbridge Literary Medal",
            },
            "C10": {
                "question": "Which city does Basil Hart live in?",
                "answer": "The city Grayhaven",
            },
            "C00": {
                "question": "Which city does Elian Mercer live in?",
                "answer": "The city Westhaven",
            },
        },
    }


class F2DAdapterTest(unittest.TestCase):
    def test_factorial_cells_map_to_dual_assistant_roles(self):
        unit = valid_unit()
        row = f2d.convert_unit(unit)
        self.assertEqual(row["matched_question"], unit["cells"]["C01"]["question"])
        self.assertEqual(
            [control["cell"] for control in row["mismatched_controls"]],
            ["C10", "C00"],
        )
        self.assertEqual(row["causal_roles"]["forget"], "C11")
        self.assertEqual(f2d.f2r.validate_pair(row), [])

    def test_file_conversion_preserves_one_unit_and_two_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "ciru.jsonl"
            output = Path(tmp) / "f2d.jsonl"
            source.write_text(json.dumps(valid_unit()) + "\n", encoding="utf-8")
            rows = f2d.convert_file(source, output)
            matched, controls, raw = f2d.f2r.load_f2r_pairs(str(output))
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(raw), 1)
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(controls), 2)


if __name__ == "__main__":
    unittest.main()
