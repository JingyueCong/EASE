import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Forget013BDepth4Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (ROOT / "scripts/run_f2r_tofu.sh").read_text()
        cls.shared = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget01.sh"
        ).read_text()
        cls.wrapper = (
            ROOT / "scripts/sweep_f2d_v512_dual_depth4_forget01_3b.sh"
        ).read_text()
        cls.eval_config = (
            ROOT
            / "open-unlearning/configs/model/Llama-3.2-3B-Instruct_DualULD.yaml"
        ).read_text()

    def test_public_runner_keeps_1b_defaults_but_accepts_model_configs(self):
        self.assertIn('TRAIN_MODEL_CONFIG="${TRAIN_MODEL_CONFIG:-llama-3-1b}"', self.runner)
        self.assertIn(
            'EVAL_MODEL_CONFIG="${EVAL_MODEL_CONFIG:-Llama-3.2-1B-Instruct_DualULD}"',
            self.runner,
        )
        self.assertIn('model="$TRAIN_MODEL_CONFIG"', self.runner)
        self.assertIn('model="$EVAL_MODEL_CONFIG"', self.runner)

    def test_3b_wrapper_uses_unique_model_and_sweep_names(self):
        self.assertIn("export MODEL_TAG=3b", self.wrapper)
        self.assertIn("export TRAIN_MODEL_CONFIG=llama-3-3b", self.wrapper)
        self.assertIn(
            "export EVAL_MODEL_CONFIG=Llama-3.2-3B-Instruct_DualULD",
            self.wrapper,
        )
        self.assertIn(
            "f2d_v512_dual_depth4_3b_exact_subset_seed42", self.wrapper
        )
        self.assertIn('export TRAIN_BS="${TRAIN_BS:-1}"', self.wrapper)

    def test_shared_sweep_serializes_retain_reference_download(self):
        self.assertIn("Materialize the frozen retain99 reference once", self.shared)
        self.assertIn('RETAIN_LOGS_PATH="$RETAIN_REFERENCE"', self.shared)
        self.assertIn('f2d_did_${MODEL_TAG}_forget01_', self.shared)

    def test_3b_dual_config_supports_reference_delta(self):
        for field in (
            "composition_mode: raw",
            "reference_a1_path: null",
            "reference_a2_path: null",
            "sequence_router_enabled: false",
        ):
            self.assertIn(field, self.eval_config)


if __name__ == "__main__":
    unittest.main()
