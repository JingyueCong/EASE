import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_f2d_v512_a1strength4.sh"


class V512A1Strength4DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")
        block = re.search(
            r'TRAIN_CONFIGS="\\\n(?P<body>.*?)"\n\nread',
            cls.text,
            re.DOTALL,
        )
        if block is None:
            raise AssertionError("TRAIN_CONFIGS block not found")
        cls.configs = re.findall(r"v512_a1s[^\s\\\"]+", block.group("body"))

    def test_exact_two_by_two_training_design(self):
        self.assertEqual(len(self.configs), 4)
        parsed = [config.split(":") for config in self.configs]
        self.assertTrue(all(len(fields) == 13 for fields in parsed))
        factors = {(fields[11], fields[9]) for fields in parsed}
        self.assertEqual(
            factors,
            {("96", "1.25"), ("96", "1.5"), ("108", "1.25"), ("108", "1.5")},
        )

    def test_a2_and_shared_training_choices_are_frozen(self):
        parsed = [config.split(":") for config in self.configs]
        self.assertEqual({fields[12] for fields in parsed}, {"60"})
        self.assertEqual({fields[10] for fields in parsed}, {"1.0"})
        self.assertEqual({fields[5] for fields in parsed}, {"5e-4"})
        self.assertEqual({fields[6] for fields in parsed}, {"5e-4"})
        self.assertEqual({tuple(fields[1:5]) for fields in parsed}, {("2", "2", "16", "16")})

    def test_reference_delta_isolated_from_alignment_and_gate(self):
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertIn("ALIGNMENT_ENABLED=false GATE_ENABLED=false", self.text)
        self.assertIn("CALIBRATION_PATH=null", self.text)


if __name__ == "__main__":
    unittest.main()
