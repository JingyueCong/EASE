import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "summarize_f2r_tofu.py"
spec = importlib.util.spec_from_file_location("f2r_summary", MODULE_PATH)
summary_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summary_module)


class F2RSummaryTest(unittest.TestCase):
    def test_build_report_flattens_and_derives_metrics(self):
        eval_logs = {
            "forget_Q_A_Prob": {"agg_value": 0.2},
            "forget_Q_A_ROUGE": {"agg_value": 0.3},
            "forget_truth_ratio": {"agg_value": 0.8},
            "retain_Q_A_Prob": {"agg_value": 0.7},
            "retain_Q_A_ROUGE": {"agg_value": 0.8},
            "retain_truth_ratio": {"agg_value": 0.9},
        }
        summary = {"model_utility": 0.75, "forget_quality": 0.6}
        report = summary_module.build_report(
            eval_logs, summary, {"mode": "smoke", "split": "forget05"}
        )
        self.assertEqual(report["metrics"]["forget_quality"], 0.6)
        self.assertEqual(report["metrics"]["model_utility"], 0.75)
        self.assertIsNotNone(report["derived"]["memorization_score"])
        self.assertIsNotNone(report["derived"]["retain_utility_score"])
        self.assertIsNotNone(report["derived"]["aggregate_score"])
        self.assertIn("must not be reported", report["warning"])
        self.assertFalse(report["validation"]["complete"])

    def test_validation_requires_every_ease_metric(self):
        metrics = {name: 0.5 for name in summary_module.PRIMARY_METRICS}
        report = summary_module.build_report({}, metrics, {"mode": "full"})
        self.assertTrue(report["validation"]["complete"])
        self.assertEqual(report["validation"]["missing"], [])
        self.assertEqual(report["validation"]["invalid"], [])

        metrics["forget_quality"] = None
        report = summary_module.build_report({}, metrics, {"mode": "full"})
        self.assertFalse(report["validation"]["complete"])
        self.assertIn("forget_quality", report["validation"]["invalid"])

    def test_selection_access_is_recorded_from_metadata(self):
        report = summary_module.build_report(
            {}, {}, {"mode": "full", "selection_retain_access": True}
        )
        self.assertTrue(report["protocol"]["selection_retain_access"])

    def test_upstream_truth_ratio_capitalisation_is_normalised(self):
        metrics = {name: 0.5 for name in summary_module.PRIMARY_METRICS}
        for canonical, aliases in summary_module.METRIC_ALIASES.items():
            metrics.pop(canonical)
            metrics[aliases[0]] = 0.6
        report = summary_module.build_report({}, metrics, {"mode": "full"})
        self.assertTrue(report["validation"]["complete"])
        self.assertEqual(report["metrics"]["retain_truth_ratio"], 0.6)
        self.assertEqual(report["metrics"]["ra_truth_ratio"], 0.6)
        self.assertEqual(report["metrics"]["wf_truth_ratio"], 0.6)

    def test_write_reports_creates_all_formats(self):
        report = summary_module.build_report(
            {}, {"model_utility": 0.8}, {"mode": "full", "split": "forget05"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary_module.write_reports(report, Path(tmp))
            self.assertTrue((Path(tmp) / "F2R_REPORT.json").is_file())
            self.assertTrue((Path(tmp) / "F2R_REPORT.csv").is_file())
            self.assertTrue((Path(tmp) / "F2R_REPORT.md").is_file())
            self.assertTrue((Path(tmp) / "F2R_EASE_TABLE.csv").is_file())
            self.assertTrue((Path(tmp) / "F2R_EASE_TABLE.md").is_file())


if __name__ == "__main__":
    unittest.main()
