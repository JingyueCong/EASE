#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD/scripts/merge_tofu_premisefix_v5_11_full.py"
spec = importlib.util.spec_from_file_location("premisefix_full_merge_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def record(index: int, design: str, block_size: int = 2) -> dict:
    return {
        "source_id": f"forget05_perturbed-{index:05d}",
        "block_id": index // block_size,
        "design_version": design,
        "cells": {
            "C11": {"question": f"q11-{index}", "answer": f"a11-{index}"},
            "C01": {"question": f"q01-{index}", "answer": f"a01-{index}"},
            "C10": {"question": f"q10-{index}", "answer": f"a10-{index}"},
            "C00": {"question": f"q00-{index}", "answer": f"a00-{index}"},
        },
        "generation": {},
    }


class MergePremiseFixFullTest(unittest.TestCase):
    def test_mixed_design_partitions_merge_without_editing_cells(self):
        approved = [
            record(2, module.APPROVED_DESIGN),
            record(3, module.APPROVED_DESIGN),
        ]
        remainder = [
            record(0, module.REMAINDER_DESIGN),
            record(1, module.REMAINDER_DESIGN),
        ]
        originals = {
            row["source_id"]: row["cells"]
            for row in approved + remainder
        }
        merged = module.merge_records(
            approved,
            remainder,
            approved_blocks={1},
            expected_units=4,
            block_size=2,
            approved_digest="approved",
            remainder_digest="remainder",
        )
        self.assertEqual([module.source_index(row) for row in merged], [0, 1, 2, 3])
        for row in merged:
            self.assertEqual(row["cells"], originals[row["source_id"]])
            self.assertFalse(
                row["generation"]["full200_assembly"][
                    "causal_cells_edited_by_merge"
                ]
            )
        self.assertEqual(
            merged[0]["generation"]["full200_assembly"]["partition"],
            "premisefix_v5.11",
        )
        self.assertEqual(
            merged[2]["generation"]["full200_assembly"]["partition"],
            "approved_v5.10",
        )

    def test_wrong_partition_design_fails(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            module.merge_records(
                [record(2, module.REMAINDER_DESIGN), record(3, module.REMAINDER_DESIGN)],
                [record(0, module.REMAINDER_DESIGN), record(1, module.REMAINDER_DESIGN)],
                approved_blocks={1},
                expected_units=4,
                block_size=2,
            )

    def test_missing_coverage_fails(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch|row count"):
            module.merge_records(
                [record(2, module.APPROVED_DESIGN), record(3, module.APPROVED_DESIGN)],
                [record(0, module.REMAINDER_DESIGN)],
                approved_blocks={1},
                expected_units=4,
                block_size=2,
            )


if __name__ == "__main__":
    unittest.main()
