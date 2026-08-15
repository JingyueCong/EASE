import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "summarize_f2r_sweep.py"
spec = importlib.util.spec_from_file_location("f2r_sweep_summary", MODULE_PATH)
sweep_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(sweep_module)


class F2RSweepSummaryTest(unittest.TestCase):
    def test_collects_metrics_and_marks_pareto_frontier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = [
                ("balanced", 0.5, 0.5),
                ("high_fq", 0.8, 0.3),
                ("dominated", 0.4, 0.4),
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["tag", "weight_a1", "weight_a2", "top_filter", "task_name", "report"]
                )
                for tag, fq, mu in configs:
                    report = root / f"{tag}.json"
                    report.write_text(
                        json.dumps(
                            {
                                "derived": {
                                    "aggregate_score": 0.6,
                                    "memorization_score": 0.7,
                                    "retain_utility_score": 0.55,
                                },
                                "metrics": {
                                    "forget_quality": fq,
                                    "model_utility": mu,
                                    "forget_Q_A_ROUGE": 0.2,
                                    "retain_Q_A_ROUGE": 0.8,
                                    "privleak": -5.0,
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    writer.writerow([tag, -0.8, 0.8, 0.01, tag, report])

            rows = sweep_module.load_rows(manifest)
            sweep_module.mark_pareto(rows)
            by_tag = {row["tag"]: row for row in rows}
            self.assertTrue(by_tag["balanced"]["pareto"])
            self.assertTrue(by_tag["high_fq"]["pareto"])
            self.assertFalse(by_tag["dominated"]["pareto"])

            sweep_module.write_outputs(rows, root / "out")
            self.assertTrue((root / "out" / "F2R_SWEEP.csv").is_file())
            self.assertTrue((root / "out" / "F2R_SWEEP.md").is_file())
            self.assertTrue(
                (root / "out" / "F2R_SWEEP_ALL_METRICS.csv").is_file()
            )
            self.assertTrue(
                (root / "out" / "F2R_SWEEP_ALL_METRICS.md").is_file()
            )
            markdown = (root / "out" / "F2R_SWEEP.md").read_text(encoding="utf-8")
            self.assertIn("Agg. ↑", markdown)
            self.assertIn("R.R-L ↑", markdown)
            all_metrics = (root / "out" / "F2R_SWEEP_ALL_METRICS.csv").read_text(
                encoding="utf-8"
            )
            self.assertIn("metrics/privleak", all_metrics)


if __name__ == "__main__":
    unittest.main()
