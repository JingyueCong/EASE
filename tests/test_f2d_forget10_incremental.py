#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load(
    "test_prepare_f2d_forget10_incremental",
    ROOT / "scripts" / "prepare_f2d_forget10_incremental.py",
)
factorial = load(
    "test_forget10_factorial_aliases",
    ROOT / "ULD" / "scripts" / "generate_tofu_author_factorial.py",
)
pairbudget = load(
    "test_forget10_pairbudget_blocks",
    ROOT / "ULD" / "scripts" / "generate_tofu_author_pairbudget_v5_12.py",
)
merge400 = load(
    "test_merge_f2d_forget10_full400",
    ROOT / "scripts" / "merge_f2d_forget10_full400.py",
)


def old_row(index: int) -> dict:
    question = f"old question {index}"
    answer = f"old answer {index}"
    return {
        "source_id": f"forget05_perturbed-{index:05d}",
        "block_id": index // 20,
        "profile_id": (
            "forget05_perturbed:author-premisefix-v5.11-"
            f"block-{index // 20:02d}:seed-42"
        ),
        "cells": {
            cell: {"question": question, "answer": answer}
            for cell in ("C11", "C01", "C10", "C00")
        },
        "generation": {},
    }


class Forget10IncrementalTests(unittest.TestCase):
    def test_orchestrator_is_generation_only_and_versioned(self):
        script = (
            ROOT / "scripts/run_f2d_forget10_incremental400.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("run_f2r_tofu", script)
        self.assertNotIn("sweep_f2d", script)
        self.assertNotIn("rm -", script)
        self.assertIn("expected-units 400", script)
        self.assertIn("No assistant was trained", script)
        self.assertIn("incremental_v1", script)

    def test_exact_mapping_finds_nonprefix_200_row_subset(self):
        source = [old_row(index) for index in range(200)]
        reference = [
            (f"new question {index}", f"new answer {index}")
            for index in range(200)
        ] + [
            (f"old question {index}", f"old answer {index}")
            for index in range(200)
        ]
        mapping = prepare.exact_mapping(source, reference, "synthetic")
        self.assertEqual(mapping, list(range(200, 400)))
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.jsonl"
            source_path.write_text("source\n", encoding="utf-8")
            derived = prepare.derive_rows(
                source, mapping, source_path, "V5.12"
            )
        self.assertEqual(derived[0]["block_id"], 10)
        self.assertIn("block-10", derived[0]["profile_id"])
        self.assertEqual(
            derived[0]["cells"]["C11"], source[0]["cells"]["C11"]
        )

    def test_real_forget10_manifest_derives_all_authors_and_alias(self):
        reference = prepare.load_reference(
            ROOT / "ULD/data/aug_data/tofu/forget10_perturbed/perturb_res.csv"
        )
        manifest = prepare.author_manifest(reference)
        self.assertEqual(len(manifest["authors"]), 20)
        self.assertEqual(
            manifest["authors"][3]["canonical_name"], "Rajeev Majumdar"
        )
        self.assertEqual(manifest["authors"][3]["aliases"], ["Majumdar"])
        self.assertEqual(manifest["authors"][10]["canonical_name"], "Hina Ameen")

    def test_manifest_alias_binds_shortened_author_row(self):
        sources = [
            {"source_id": "x-00000", "question": "Who is Rajeev Majumdar?", "answer": "Rajeev Majumdar."},
            {"source_id": "x-00001", "question": "How did Majumdar write?", "answer": "Majumdar wrote carefully."},
        ]
        manifest = {
            "block_size": 2,
            "authors": [{
                "block_id": 0,
                "canonical_name": "Rajeev Majumdar",
                "aliases": ["Majumdar"],
            }],
        }
        blocks = factorial.group_author_blocks(sources, manifest)
        self.assertEqual(blocks[0]["target_aliases"], ["Majumdar"])

    def test_pairbudget_profiles_accept_nonzero_ten_block_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "base.jsonl"
            data.write_text("partition\n", encoding="utf-8")
            data_digest = hashlib.sha256(data.read_bytes()).hexdigest()
            profile_path = root / "profiles.json"
            profile_path.write_text(json.dumps({
                "design_version": pairbudget.BASE_DESIGN,
                "assembly": {"full_data_sha256": data_digest},
                "profiles": [
                    {
                        "block_id": block,
                        "fact_ledger": [
                            {"source_id": f"forget10_perturbed-{block*20+i:05d}"}
                            for i in range(20)
                        ],
                    }
                    for block in range(10, 20)
                ],
            }), encoding="utf-8")
            _, by_block = pairbudget.load_profiles(
                profile_path, data_digest, set(range(10, 20))
            )
            self.assertEqual(set(by_block), set(range(10, 20)))

    def test_versioned_writer_refuses_nonmatching_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            self.assertEqual(prepare.write_new_or_reuse(path, b"one\n"), "created")
            self.assertEqual(prepare.write_new_or_reuse(path, b"one\n"), "reused")
            with self.assertRaises(FileExistsError):
                prepare.write_new_or_reuse(path, b"two\n")

    def test_full400_merge_requires_disjoint_complete_exact_c11(self):
        reference = [(f"q{index}", f"a{index}") for index in range(400)]

        def row(index: int) -> dict:
            return {
                "source_id": f"forget10_perturbed-{index:05d}",
                "block_id": index // 20,
                "cells": {
                    cell: {"question": f"q{index}", "answer": f"a{index}"}
                    for cell in ("C11", "C01", "C10", "C00")
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reused, new = root / "reused.jsonl", root / "new.jsonl"
            reused.write_text(
                "".join(json.dumps(row(index)) + "\n" for index in range(200)),
                encoding="utf-8",
            )
            new.write_text(
                "".join(json.dumps(row(index)) + "\n" for index in range(200, 400)),
                encoding="utf-8",
            )
            rows = merge400.merge_rows(reused, new, reference, "synthetic")
            self.assertEqual(len(rows), 400)
            self.assertEqual(rows[-1]["source_id"], "forget10_perturbed-00399")
            self.assertTrue(all(
                item["generation"]["forget10_full400_assembly"]
                ["causal_cells_edited_by_merge"] is False
                for item in rows
            ))


if __name__ == "__main__":
    unittest.main()
