import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_f2d_v512_a1contrast4.sh"


class V512A1Contrast4DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_exact_two_by_two_regularizer_design(self):
        configs = re.findall(r'"cw[^"\n]+:[0-9.]+:[0-9.]+"', self.text)
        self.assertEqual(len(configs), 4)
        self.assertIn('"cw0p1_pk0p01:0.1:0.01"', configs)
        self.assertIn('"cw0p3_pk0p03:0.3:0.03"', configs)

    def test_freezes_a2_and_inference_point(self):
        self.assertIn('A2_CHECKPOINT_OVERRIDE="$FULL_A2"', self.text)
        self.assertIn("WEIGHT_A1=-2.0 WEIGHT_A2=1.7 TOP_FILTER=0.0004", self.text)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertIn("ALIGNMENT_ENABLED=false GATE_ENABLED=false", self.text)

    def test_runner_supports_a2_only_checkpoint_override(self):
        runner = (ROOT / "scripts/run_f2r_tofu.sh").read_text(encoding="utf-8")
        self.assertNotIn("Set both A1_CHECKPOINT_OVERRIDE", runner)
        self.assertIn('if [ -n "$A2_CHECKPOINT_OVERRIDE" ]; then', runner)
        self.assertIn("Skipping A2 training (explicit frozen checkpoint)", runner)

    def test_paired_loss_and_batch_are_explicit(self):
        self.assertIn("TRAIN_LOSS_CONFIG=factorial_contrastive_a1", self.text)
        self.assertIn("A1_DATA_MODE=f2d_contrast_a1", self.text)
        self.assertIn("A1_TRAIN_BS=5", self.text)
        self.assertIn("A1_TRAIN_STEPS=96", self.text)


if __name__ == "__main__":
    unittest.main()
