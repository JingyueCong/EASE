import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RoleReferencePlumbingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (ROOT / "scripts/run_f2r_tofu.sh").read_text()
        cls.model = (
            ROOT / "open-unlearning/src/model/dual_uld.py"
        ).read_text()
        cls.config = (
            ROOT
            / "open-unlearning/configs/model/Llama-3.2-1B-Instruct_DualULD.yaml"
        ).read_text()

    def test_runner_resolves_and_forwards_both_references(self):
        for name in ("A1_REFERENCE_PATH", "A2_REFERENCE_PATH"):
            self.assertIn(f'{name}="${{{name}:-auto}}"', self.runner)
        self.assertIn('model.model_args.reference_a1_path="$A1_REFERENCE_PATH"', self.runner)
        self.assertIn('model.model_args.reference_a2_path="$A2_REFERENCE_PATH"', self.runner)

    def test_model_config_exposes_both_references(self):
        self.assertIn("reference_a1_path: null", self.config)
        self.assertIn("reference_a2_path: null", self.config)

    def test_loader_validates_each_reference_against_its_role(self):
        self.assertIn('(\"a1\", a1, reference_a1)', self.model)
        self.assertIn('(\"a2\", a2, reference_a2)', self.model)
        self.assertIn("reference_a2_logits", self.model)

    def test_train_only_is_explicit_and_validated(self):
        self.assertIn('TRAIN_ONLY="${TRAIN_ONLY:-false}"', self.runner)
        self.assertIn('if [ "$TRAIN_ONLY" = "true" ]', self.runner)


if __name__ == "__main__":
    unittest.main()
