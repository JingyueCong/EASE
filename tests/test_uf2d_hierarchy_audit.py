import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "audit_uf2d_hierarchy.py"
spec = importlib.util.spec_from_file_location("uf2d_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def record(index):
    cell = {
        "question": "What did Ava write?",
        "answer": "Ava wrote North Wind. She later taught in Rome.",
        "supervision": {
            "claim_spans": [[0, 21]],
            "evidence_spans": [[10, 20]],
        },
    }
    return {
        "source_id": f"forget05_perturbed-{index:05d}",
        "target_entity": "Ava",
        "replacement_entity": "Bea",
        "target_relation": "authorship",
        "placebo_relation": "residence",
        "cells": {name: dict(cell) for name in audit.CELLS},
    }


class UF2DHierarchyAuditTest(unittest.TestCase):
    def test_sample_covers_every_author_block(self):
        records = [record(index) for index in range(200)]
        selected = audit.stratified_sample(records, 20, 20, 42)
        blocks = [audit.source_index(row["source_id"]) // 20 for row in selected]
        self.assertEqual(len(selected), 20)
        self.assertEqual({block: blocks.count(block) for block in set(blocks)}, {
            block: 2 for block in range(10)
        })

    def test_flags_near_full_claim(self):
        cell = record(0)["cells"]["C11"]
        cell["supervision"]["claim_spans"] = [[0, len(cell["answer"])]]
        flags, claim_coverage, _ = audit.risk_flags(cell)
        self.assertIn("CLAIM_NEAR_FULL_ANSWER", flags)
        self.assertEqual(claim_coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
