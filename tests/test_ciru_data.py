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

    def test_factorial_dual_roles_encode_difference_in_differences(self):
        unit = valid_unit()
        a1_ce, a1_uniform = ciru.factorial_dual_roles([unit], "f2d_did_a1")
        a2_ce, a2_uniform = ciru.factorial_dual_roles([unit], "f2d_did_a2")
        self.assertEqual(a1_ce, [unit["cells"]["C11"]])
        self.assertEqual(a1_uniform, [unit["cells"]["C01"]])
        self.assertEqual(a2_ce, [unit["cells"]["C10"]])
        self.assertEqual(a2_uniform, [unit["cells"]["C00"]])
        self.assertEqual(len(a1_ce), len(a1_uniform))
        self.assertEqual(len(a2_ce), len(a2_uniform))

    def test_hierarchical_roles_preserve_supervision(self):
        unit = valid_unit()
        unit["cells"]["C11"]["supervision"] = {
            "claim_spans": [[0, 4]],
            "evidence_spans": [[0, 4]],
        }
        a1_ce, _ = ciru.factorial_dual_roles([unit], "uf2d_hier_a1")
        self.assertEqual(a1_ce[0]["supervision"]["claim_spans"], [[0, 4]])

    def test_factorial_dual_roles_reject_unknown_role(self):
        with self.assertRaisesRegex(ValueError, "Unknown factorial dual role"):
            ciru.factorial_dual_roles([valid_unit()], "f2d_did_unknown")

    def test_all_typed_v4_schema_versions_remain_loadable(self):
        for design in (
            "tofu-author-typed-v4",
            "tofu-author-typed-v4.1",
            "tofu-author-typed-v4.2",
            "tofu-author-anchor-v5",
            "tofu-author-anchor-v5.1",
            "tofu-author-rowlocal-v5.2",
            "tofu-author-ledger-rowlocal-v5.3",
            "tofu-author-ledger-slots-v5.4",
        ):
            unit = valid_unit()
            unit.update({
                "design_version": design,
                "block_id": 0,
                "query_index": 0,
                "profile_id": f"fixture:{design}",
                "canonical_target_entity": unit["target_entity"],
                "identity_binding": {
                    "C11": unit["target_entity"],
                    "C01": unit["replacement_entity"],
                    "C10": unit["target_entity"],
                    "C00": unit["replacement_entity"],
                },
            })
            self.assertEqual(ciru.validate_ciru_unit(unit), [], design)


if __name__ == "__main__":
    unittest.main()
