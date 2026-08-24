import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_f2d_v512_context_gate3.sh"


class V512ContextGate3DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_exact_three_preregistered_points(self):
        points = re.findall(r'"gate_[a-z]+:-[0-9.]+:[0-9.]+:[0-9.]+"', self.text)
        self.assertEqual(
            points,
            [
                '"gate_util:-1.9:1.6:0.0004"',
                '"gate_boundary:-2.0:1.7:0.0004"',
                '"gate_memory:-2.1:1.8:0.0004"',
            ],
        )

    def test_gate_uses_only_causal_cells_and_no_alignment(self):
        self.assertIn("--mode gate", self.text)
        self.assertIn("gate labels      : C11=on; C01/C10/C00=off", self.text)
        self.assertIn("ALIGNMENT_ENABLED=false GATE_ENABLED=true", self.text)
        self.assertIn("gate_training_access", self.text)
        self.assertNotIn("--mode alignment-gate", self.text)

    def test_frozen_hybrid_and_reference_delta(self):
        self.assertIn('A1_CHECKPOINT_OVERRIDE="$V512_A1"', self.text)
        self.assertIn('A2_CHECKPOINT_OVERRIDE="$FULL_A2"', self.text)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.text)
        self.assertNotIn("sweep_f2d_did_training.sh", self.text)

    def test_stop_rule_is_machine_readable(self):
        self.assertIn("aggregate_score\"] > 0.551721", self.text)
        self.assertIn("memorization_score\"] >= 0.53", self.text)
        self.assertIn("retain_utility_score\"] >= 0.60", self.text)
        self.assertIn("stop_static_dual_gate_and_weight_variants", self.text)
        self.assertIn("CONTEXT_GATE_DECISION.json", self.text)


if __name__ == "__main__":
    unittest.main()
