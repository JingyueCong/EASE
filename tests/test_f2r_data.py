import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "uld" / "data" / "f2r.py"
spec = importlib.util.spec_from_file_location("f2r_data", MODULE_PATH)
f2r_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(f2r_data)
load_f2r_pairs = f2r_data.load_f2r_pairs
validate_pair = f2r_data.validate_pair


def valid_record(source_id="example-1", view=0):
    return {
        "source_id": source_id,
        "view": view,
        "source_question": "Which award did Basil win?",
        "source_answer": "Riverdale Book Award",
        "invariants": {
            "task": "factual QA",
            "relation": "person won award",
            "style": "short question",
            "difficulty": "single hop",
        },
        "matched_question": "Which award did Elian Mercer win?",
        "matched_answer": "Elian Mercer won the Northbridge Literary Medal.",
        "mismatched_question": "Where did Mira Solano study geology?",
        "mismatched_answer": "Mira Solano studied geology at Westhaven College.",
        "changed_evidence": [
            "Basil -> Elian Mercer",
            "Riverdale Book Award -> Northbridge Literary Medal",
        ],
    }


class F2RDataTest(unittest.TestCase):
    def test_valid_pair(self):
        self.assertEqual(validate_pair(valid_record()), [])

    def test_source_answer_leakage_is_rejected(self):
        record = valid_record()
        record["matched_answer"] = "Riverdale Book Award"
        errors = validate_pair(record)
        self.assertTrue(any("source_answer" in error for error in errors))

    def test_jsonl_loads_matched_and_mismatched_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            records = [valid_record(view=0), valid_record(view=1)]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            matched, mismatched, raw = load_f2r_pairs(str(path))
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(mismatched), 2)
        self.assertEqual(len(raw), 2)
        self.assertEqual(matched[0]["question"], records[0]["matched_question"])
        self.assertEqual(mismatched[0]["answer"], records[0]["mismatched_answer"])

    def test_multiple_placebo_controls_are_loaded_once_each(self):
        record = valid_record()
        record["mismatched_controls"] = [
            {
                "question": record["mismatched_question"],
                "answer": record["mismatched_answer"],
            },
            {
                "question": "Where did Elian Mercer study geology?",
                "answer": "Elian Mercer studied geology at Northbridge College.",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            matched, mismatched, _ = load_f2r_pairs(str(path))
        self.assertEqual(len(matched), 1)
        self.assertEqual(len(mismatched), 2)
        self.assertEqual(
            [row["question"] for row in mismatched],
            [row["question"] for row in record["mismatched_controls"]],
        )

    def test_duplicate_placebo_control_is_rejected(self):
        record = valid_record()
        control = {
            "question": record["mismatched_question"],
            "answer": record["mismatched_answer"],
        }
        record["mismatched_controls"] = [control, dict(control)]
        errors = validate_pair(record)
        self.assertTrue(any("duplicate mismatched_controls" in error for error in errors))

    def test_duplicate_source_view_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            row = valid_record()
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate pair key"):
                load_f2r_pairs(str(path))


if __name__ == "__main__":
    unittest.main()
