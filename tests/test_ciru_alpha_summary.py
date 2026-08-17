import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "summarize_ciru_alpha_sweep.py"
spec = importlib.util.spec_from_file_location("ciru_alpha_summary", MODULE_PATH)
summary = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summary)


def report(agg, mem, util):
    return {
        "derived": {
            "aggregate_score": agg,
            "memorization_score": mem,
            "retain_utility_score": util,
        },
        "metrics": {
            "forget_quality": 0.01,
            "model_utility": 0.6,
            "extraction_strength": 0.2,
            "exact_memorization": 0.3,
        },
    }


class CIRUAlphaSummaryTest(unittest.TestCase):
    def test_rows_are_ranked_and_target_is_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low = root / "low.json"
            high = root / "high.json"
            low.write_text(json.dumps(report(0.4, 0.3, 0.6)), encoding="utf-8")
            high.write_text(json.dumps(report(0.6, 0.55, 0.66)), encoding="utf-8")
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("alpha", "report"))
                writer.writeheader()
                writer.writerow({"alpha": "0.5", "report": low})
                writer.writerow({"alpha": "1.5", "report": high})
            rows = summary.load_rows(manifest, 0.58)
            summary.write_outputs(rows, root / "out", 0.58)

            self.assertEqual(rows[0]["alpha"], "1.5")
            self.assertTrue(rows[0]["beat_target"])
            self.assertAlmostEqual(rows[0]["delta_to_target"], 0.02)
            self.assertTrue((root / "out" / "CIRU_ALPHA_SWEEP.md").is_file())


if __name__ == "__main__":
    unittest.main()
