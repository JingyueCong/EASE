import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVER = ROOT / "scripts/derive_f2d_forget01_from_forget05.py"
SWEEP = ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget01.sh"


def row(index: int, design: str) -> dict:
    question = f"Question {index}?"
    answer = f"Answer {index}."
    return {
        "source_id": f"forget05_perturbed-{index:05d}",
        "design_version": design,
        "block_id": index // 20,
        "cells": {
            "C11": {"question": question, "answer": answer},
            "C01": {"question": f"Replacement {index}?", "answer": "Replacement."},
            "C10": {"question": f"Placebo source {index}?", "answer": "Source."},
            "C00": {"question": f"Placebo replacement {index}?", "answer": "Control."},
        },
        "generation": {"source_ref": f"forget05_perturbed-{index:05d}"},
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )


class Forget01NestedPrefixTest(unittest.TestCase):
    def test_derivation_is_exact_versioned_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v512_source = root / "v512.jsonl"
            full_source = root / "full.jsonl"
            reference = root / "reference.jsonl"
            v512_output = root / "forget01_v512.jsonl"
            full_output = root / "forget01_full.jsonl"
            manifest = root / "derivation.json"
            write_jsonl(v512_source, [row(i, "v512") for i in range(200)])
            write_jsonl(full_source, [row(i, "full") for i in range(200)])
            write_jsonl(
                reference,
                [
                    {"question": f"Question {i}?", "answer": f"Answer {i}."}
                    for i in range(40)
                ],
            )
            command = [
                sys.executable,
                str(DERIVER),
                "--v512-source",
                str(v512_source),
                "--fullanswer-source",
                str(full_source),
                "--v512-output",
                str(v512_output),
                "--fullanswer-output",
                str(full_output),
                "--manifest-output",
                str(manifest),
                "--reference-jsonl",
                str(reference),
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("C11=byte-identical", first.stdout)
            second = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertIn("V5.12 output: reused", second.stdout)
            derived = [json.loads(line) for line in v512_output.read_text().splitlines()]
            self.assertEqual(len(derived), 40)
            self.assertEqual(derived[0]["source_id"], "forget01_perturbed-00000")
            self.assertEqual(derived[-1]["source_id"], "forget01_perturbed-00039")
            self.assertEqual(derived[7]["cells"]["C11"], row(7, "v512")["cells"]["C11"])
            self.assertEqual(
                derived[7]["generation"]["source_ref"],
                "forget01_perturbed-00007",
            )
            provenance = json.loads(manifest.read_text())
            self.assertTrue(provenance["c11_exact_match"])
            self.assertFalse(provenance["content_cells_changed"])

            v512_output.write_text("user-owned-content\n", encoding="utf-8")
            refused = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("refusing to overwrite", refused.stderr)

    def test_sweep_is_isolated_static_and_retain_free(self):
        text = SWEEP.read_text(encoding="utf-8")
        self.assertIn("SPLIT=forget01", text)
        self.assertIn("forget01_${SWEEP_NAME}", text)
        self.assertIn('TRAIN_ONLY=true TRAIN_ROLE="$role"', text)
        self.assertIn("A1_NUM_LAYER=4", text)
        self.assertIn("A2_NUM_LAYER=4", text)
        self.assertIn("COMPOSITION_MODE=reference_delta", text)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", text)
        self.assertIn("TRAININGS=(", text)
        self.assertIn('"exposure:20:14"', text)
        self.assertIn('"moderate:32:24"', text)
        self.assertEqual(text.count('"light:'), 1)
        self.assertEqual(text.count('"conservative:'), 1)
        self.assertEqual(text.count('"intermediate:'), 1)
        self.assertEqual(text.count('"legacy_best:'), 1)
        self.assertIn("declare -A TRAIN_A1 TRAIN_A2", text)
        self.assertNotIn("EVAL_SPECS", text)
        self.assertNotIn("rm -", text)


if __name__ == "__main__":
    unittest.main()
