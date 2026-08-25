import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget058BDepth4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget05_3b.sh"
        ).read_text()
        cls.wrapper = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget05_8b.sh"
        ).read_text()
        cls.config = (
            ROOT
            / "open-unlearning/configs/model/Llama-3.1-8B-Instruct_DualULD.yaml"
        ).read_text()

    def test_uses_isolated_8b_artifacts_and_public_mirror(self):
        self.assertIn("MODEL_SIZE_TAG=8b", self.wrapper)
        self.assertIn("llama-3-8b", self.wrapper)
        self.assertIn("f2d_v512_dual_depth4_8b_train_seed42", self.wrapper)
        self.assertIn("open-unlearning/tofu_Llama-3.1-8B-Instruct", self.wrapper)

    def test_dual_config_has_two_assistants_and_reference_delta_fields(self):
        self.assertIn("model_handler: DualULDForCausalLM", self.config)
        self.assertIn("a1_path: ???", self.config)
        self.assertIn("a2_path: ???", self.config)
        self.assertIn("reference_a1_path: null", self.config)
        self.assertIn("reference_a2_path: null", self.config)
        self.assertIn("EVAL_CONFIG_PATH", self.driver)
        self.assertIn("TRAIN_CONFIG_PATH", self.driver)

    def test_driver_remains_depth4_router_free_and_retain_free_training(self):
        self.assertIn("A1_NUM_LAYER=4", self.driver)
        self.assertIn("A2_NUM_LAYER=4", self.driver)
        self.assertIn("COMPOSITION_MODE=reference_delta", self.driver)
        self.assertIn("SEQUENCE_ROUTER_ENABLED=false", self.driver)
        self.assertIn("retain training   : none", self.driver)

    def test_8b_reuses_preregistered_twelve_point_grid(self):
        self.assertIn('WEIGHT_A1_GRID="-1.7 -2.0 -2.4 -2.7"', self.driver)
        self.assertIn('WEIGHT_A2_GRID="1.4 1.7 2.0"', self.driver)
        self.assertIn("TOP_FILTERS=0.0004", self.driver)
        self.assertIn("evaluations       : 12", self.driver)


if __name__ == "__main__":
    unittest.main()
