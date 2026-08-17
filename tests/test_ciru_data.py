import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "data" / "ciru.py"
spec = importlib.util.spec_from_file_location("ciru_data", MODULE_PATH)
ciru = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ciru)


def valid_unit(source_id="forget05-00001"):
    return {
        "source_id": source_id,
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


class CIRUDataTest(unittest.TestCase):
    def test_valid_four_cell_unit(self):
        self.assertEqual(ciru.validate_ciru_unit(valid_unit()), [])

    def test_source_answer_leakage_is_rejected(self):
        unit = valid_unit()
        unit["cells"]["C10"]["answer"] = unit["source_answer"]
        self.assertTrue(
            any("source_answer" in error for error in ciru.validate_ciru_unit(unit))
        )

    def test_c11_must_be_the_exact_source_pair(self):
        unit = valid_unit()
        unit["cells"]["C11"]["question"] = "A paraphrase"
        self.assertTrue(any("C11.question" in e for e in ciru.validate_ciru_unit(unit)))

    def test_control_status_disclaimer_is_rejected(self):
        unit = valid_unit()
        unit["cells"]["C00"]["answer"] = "As a fictional person, the city Westhaven"
        errors = ciru.validate_ciru_unit(unit)
        self.assertTrue(any("control status" in error for error in errors))

    def test_large_length_mismatch_is_rejected(self):
        unit = valid_unit()
        unit["cells"]["C10"]["question"] = (
            "Which city in the extremely distant northern coastal region does Basil Hart "
            "currently choose as a permanent and preferred place of residence?"
        )
        errors = ciru.validate_ciru_unit(unit)
        self.assertTrue(any("question length ratio" in error for error in errors))

    def test_duplicate_unit_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ciru.jsonl"
            row = json.dumps(valid_unit())
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate unit key"):
                ciru.load_ciru_units(path)


if __name__ == "__main__":
    unittest.main()
