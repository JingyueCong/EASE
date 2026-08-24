import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_f2d_v512_stronga1_fulla2_boundary18.sh"


class V512StrongA1FullA2Boundary18Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def value(self, name):
        match = re.search(rf'{name}="([^"]+)"', self.text)
        self.assertIsNotNone(match, name)
        return match.group(1).split()

    def test_exact_preregistered_boundary_grid(self):
        a1 = self.value("WEIGHT_A1_GRID")
        a2 = self.value("WEIGHT_A2_GRID")
        filters = self.value("TOP_FILTERS")
        self.assertEqual(a1, ["-1.9", "-2.0", "-2.1"])
        self.assertEqual(a2, ["1.6", "1.7", "1.8"])
        self.assertEqual(filters, ["0.0003", "0.0004"])
        self.assertEqual(len(a1) * len(a2) * len(filters), 18)
        self.assertIn("EXPECTED_EVALUATIONS=18", self.text)

    def test_dedicated_outputs_and_no_training(self):
        self.assertIn("f2d_v512_stronga1_fulla2_boundary18", self.text)
        self.assertIn("sweep_f2d_v512_stronga1_fulla2_refdelta.sh", self.text)
        self.assertNotIn("sweep_f2d_did_training.sh", self.text)


if __name__ == "__main__":
    unittest.main()
