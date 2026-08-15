import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_tofu_main_row.py"
spec = importlib.util.spec_from_file_location("tofu_main_row", MODULE_PATH)
row_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(row_module)


class TOFUMainRowTest(unittest.TestCase):
    def test_table_values_follow_declared_schema(self):
        report = {
            "derived": {
                "aggregate_score": 0.61,
                "memorization_score": 0.58,
                "retain_utility_score": 0.65,
            },
            "metrics": {
                "forget_quality": 0.7,
                "forget_Q_A_ROUGE": 0.4,
                "model_utility": 0.55,
                "retain_Q_A_ROUGE": 0.8,
            },
        }
        values = row_module.table_values(report, Path("report.json"))
        self.assertEqual(values, [0.61, 0.58, 0.7, 40.0, 0.65, 0.55, 80.0])

    def test_latex_row_has_21_values(self):
        values = [[0.5] * 7 for _ in range(3)]
        row = row_module.latex_row("CIRU", values)
        self.assertTrue(row.startswith(r"\textbf{CIRU}"))
        self.assertEqual(row.count("&"), 21)
        self.assertTrue(row.endswith(r" \\"))


if __name__ == "__main__":
    unittest.main()
