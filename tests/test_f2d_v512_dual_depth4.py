import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DualDepth4ScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (ROOT / "scripts/sweep_f2d_v512_dual_depth4.sh").read_text()

    def test_is_static_no_router_capacity_ablation(self):
        self.assertIn('"d4_2|4|2|', self.text)
        self.assertIn('"d2_4|2|4|', self.text)
        self.assertIn('"d4_4|4|4|', self.text)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.text)
        self.assertIn("ALIGNMENT_ENABLED=false", self.text)
        self.assertIn("GATE_ENABLED=false", self.text)

    def test_uses_role_specific_references(self):
        self.assertIn('A1_REFERENCE_PATH="$a1_ref"', self.text)
        self.assertIn('A2_REFERENCE_PATH="$a2_ref"', self.text)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)

    def test_trains_each_new_role_once(self):
        self.assertIn("A1_NUM_LAYER=4", self.text)
        self.assertIn("A2_NUM_LAYER=4", self.text)
        self.assertIn("TRAIN_ONLY=true", self.text)
        self.assertIn("train_a1 &", self.text)
        self.assertIn("train_a2 &", self.text)

    def test_preregisters_twelve_evaluations(self):
        self.assertIn("inference points  : 4 per pair (12 total)", self.text)
        for point in ("conservative", "intermediate", "legacy_best", "strong"):
            self.assertEqual(self.text.count(f'"{point}:'), 1)


if __name__ == "__main__":
    unittest.main()
