#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD/scripts/merge_tofu_pairrepair_v5_10_full.py"
spec = importlib.util.spec_from_file_location("pairrepair_full_merge_test", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def record(index: int, block_size: int = 2) -> dict:
    return {
        "source_id": f"forget05_perturbed-{index:05d}",
        "block_id": index // block_size,
        "design_version": module.DESIGN_VERSION,
        "cells": {
            "C11": {"question": f"q11-{index}", "answer": f"a11-{index}"},
            "C01": {"question": f"q01-{index}", "answer": f"a01-{index}"},
            "C10": {"question": f"q10-{index}", "answer": f"a10-{index}"},
            "C00": {"question": f"q00-{index}", "answer": f"a00-{index}"},
        },
        "generation": {},
    }


class MergeRecordsTest(unittest.TestCase):
    def test_disjoint_partitions_merge_in_canonical_order(self):
        approved = [record(2), record(3)]
        remaining = [record(0), record(1)]
        before = [row["cells"].copy() for row in approved + remaining]
        merged = module.merge_records(
            approved,
            remaining,
            approved_blocks={1},
            expected_units=4,
            block_size=2,
            approved_digest="approved",
            remaining_digest="remaining",
        )
        self.assertEqual([module.source_index(row) for row in merged], [0, 1, 2, 3])
        self.assertEqual(len(merged), 4)
        self.assertEqual(
            merged[2]["generation"]["full200_assembly"]["partition"],
            "approved_subset",
        )
        after_by_index = {module.source_index(row): row["cells"] for row in merged}
        self.assertEqual(after_by_index[2], before[0])
        self.assertEqual(after_by_index[3], before[1])

    def test_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "unexpected block|overlap"):
            module.merge_records(
                [record(2), record(3)],
                [record(2), record(3)],
                approved_blocks={1},
                expected_units=4,
                block_size=2,
            )

    def test_missing_coverage_fails(self):
        with self.assertRaisesRegex(ValueError, "coverage mismatch|row count"):
            module.merge_records(
                [record(2), record(3)],
                [record(0)],
                approved_blocks={1},
                expected_units=4,
                block_size=2,
            )


if __name__ == "__main__":
    unittest.main()
