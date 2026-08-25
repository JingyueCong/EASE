import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sweep_f2d_answer_uniform_a2_4.sh"


class AnswerUniformA2FourCellDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")
        block = re.search(r"CONFIGS=\(\n(?P<body>.*?)\n\)", cls.text, re.DOTALL)
        if block is None:
            raise AssertionError("CONFIGS block not found")
        cls.configs = re.findall(r'"(answeru_[^"]+)"', block.group("body"))

    def test_exact_source_by_steps_design(self):
        self.assertEqual(len(self.configs), 4)
        parsed = [tuple(config.split(":")) for config in self.configs]
        self.assertEqual(
            {(source, steps) for _tag, source, steps in parsed},
            {("full", "60"), ("full", "72"), ("v512", "60"), ("v512", "72")},
        )

    def test_only_a2_is_trained_with_answer_masked_loss(self):
        self.assertIn("TRAIN_LOSS_CONFIG=remember+answer_uniform", self.text)
        self.assertIn('A1_CHECKPOINT_OVERRIDE="$A1_CHECKPOINT"', self.text)
        self.assertNotIn("A2_CHECKPOINT_OVERRIDE=", self.text)
        self.assertIn("A2_TRAIN_LR=1e-3", self.text)
        self.assertIn("A2_RETAIN_WEIGHT=1.0", self.text)

    def test_inference_and_composition_are_frozen(self):
        self.assertIn("WEIGHT_A1=-2.0 WEIGHT_A2=1.7 TOP_FILTER=0.0004", self.text)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertIn("ALIGNMENT_ENABLED=false GATE_ENABLED=false", self.text)

    def test_no_generation_or_data_mutation(self):
        self.assertNotIn("generate_tofu", self.text)
        self.assertNotIn("OPENAI_API_KEY", self.text)
        self.assertNotIn("DEEPSEEK_API_KEY", self.text)


if __name__ == "__main__":
    unittest.main()
