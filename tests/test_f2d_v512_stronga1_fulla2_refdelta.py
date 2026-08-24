import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_f2d_v512_stronga1_fulla2_refdelta.sh"


class V512StrongA1FullA2ReferenceDeltaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_preregistered_strong_a1_and_fullanswer_a2(self):
        self.assertIn("v512_a1s96_u1p5", self.text)
        self.assertIn('V512_A1_STEP="${F2D_V512_STRONG_A1_STEP:-96}"', self.text)
        self.assertIn('FULL_TAG="${FULLANSWER_TAG:-a72_a72}"', self.text)
        self.assertIn('FULL_STEP="${FULLANSWER_STEP:-72}"', self.text)
        self.assertIn('A1_RETAIN_WEIGHT=1.5 A2_RETAIN_WEIGHT=1.0', self.text)

    def test_exact_24_point_default_grid(self):
        a1 = re.search(r'A1_GRID="\$\{LAUNCH_A1_GRID:-(.*?)\}"', self.text)
        a2 = re.search(r'A2_GRID="\$\{LAUNCH_A2_GRID:-(.*?)\}"', self.text)
        filters = re.search(r'FILTER_GRID="\$\{LAUNCH_FILTER_GRID:-(.*?)\}"', self.text)
        self.assertIsNotNone(a1)
        self.assertIsNotNone(a2)
        self.assertIsNotNone(filters)
        self.assertEqual(a1.group(1).split(), ["-1.3", "-1.5", "-1.7", "-1.9"])
        self.assertEqual(a2.group(1).split(), ["1.2", "1.4", "1.6"])
        self.assertEqual(filters.group(1).split(), ["0.0002", "0.0003"])
        self.assertEqual(
            len(a1.group(1).split())
            * len(a2.group(1).split())
            * len(filters.group(1).split()),
            24,
        )

    def test_preserves_reference_delta_without_calibration(self):
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertIn("ALIGNMENT_ENABLED=false GATE_ENABLED=false", self.text)
        self.assertIn("CALIBRATION_PATH=null", self.text)
        self.assertIn("Reference compatibility OK", self.text)


if __name__ == "__main__":
    unittest.main()
