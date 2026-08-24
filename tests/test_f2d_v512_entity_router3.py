import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_f2d_v512_entity_router3.sh"
MODEL = ROOT / "open-unlearning/src/model/dual_uld.py"
RUNNER = ROOT / "scripts/run_f2r_tofu.sh"


class V512EntityRouter3DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.model = MODEL.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")

    def test_exact_three_preregistered_points(self):
        points = re.findall(r'"entity_[a-z0-9]+:-[0-9.]+:[0-9.]+:[0-9.]+"', self.script)
        self.assertEqual(
            points,
            [
                '"entity_memory:-2.1:1.6:0.0004"',
                '"entity_boundary:-2.0:1.7:0.0004"',
                '"entity_utility:-1.9:1.6:0.0004"',
                '"entity_mid35:-2.035:1.665:0.0004"',
                '"entity_mid50:-2.050:1.650:0.0004"',
                '"entity_mid65:-2.065:1.635:0.0004"',
            ],
        )

    def test_refinement_is_one_dimensional_and_separate(self):
        self.assertIn('ENTITY_ROUTER_PHASE:-pilot', self.script)
        self.assertIn('forget05_f2d_v512_entity_router_refine3', self.script)
        self.assertIn('TASK_NAMESPACE="V512_ENTITY_ROUTER_REFINE3"', self.script)
        self.assertNotIn("TOP_FILTER_GRID", self.script)

    def test_router_uses_only_forget_request_entities(self):
        self.assertIn("forget_request_entities_only", self.script)
        self.assertIn("retain_examples_accessed", self.script)
        self.assertIn("--match-scale 1.0 --nonmatch-scale 0.0", self.script)
        self.assertNotIn("train_f2r_calibration.py", self.script)

    def test_answer_leakage_is_blocked_in_model(self):
        self.assertIn("labels=labels", self.model)
        self.assertIn("sequence_router_scale", self.model)
        self.assertIn("prompt_end_pattern", (ROOT / "open-unlearning/src/model/f2r_routing.py").read_text())

    def test_runner_plumbs_router_without_retraining(self):
        self.assertIn('model.model_args.sequence_router_enabled="$SEQUENCE_ROUTER_ENABLED"', self.runner)
        self.assertIn('model.model_args.sequence_router_path="$SEQUENCE_ROUTER_PATH"', self.runner)
        self.assertIn("A1_CHECKPOINT_OVERRIDE", self.script)
        self.assertNotIn("sweep_f2d_did_training.sh", self.script)

    def test_machine_readable_target_and_stop_rules(self):
        self.assertIn('"target_0p58_reached"', self.script)
        self.assertIn('"advance_request_scoped_router_once"', self.script)
        self.assertIn('"stop_request_scoped_router"', self.script)
        self.assertIn("ENTITY_ROUTER_DECISION.json", self.script)
        self.assertIn('advance and phase == "pilot"', self.script)


if __name__ == "__main__":
    unittest.main()
