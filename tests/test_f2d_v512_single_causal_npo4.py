import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/sweep_f2d_v512_single_causal_npo4.sh"


class V512SingleCausalNPO4DesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = RUNNER.read_text(encoding="utf-8")

    def test_exact_two_by_two(self):
        configs = re.findall(r'"b[^"\n]+:[0-9.]+:[0-9.]+"', self.text)
        self.assertEqual(len(configs), 4)
        self.assertIn('"b0p1_k0p5:0.1:0.5"', configs)
        self.assertIn('"b0p2_k1p0:0.2:1.0"', configs)

    def test_hard_retain_free_contract(self):
        self.assertIn('SELECTION_RETAIN_ACCESS:-false', self.text)
        self.assertIn('WITH_RETAIN:-false', self.text)
        self.assertIn('--selection-retain-access false', self.text)
        self.assertIn('data_mode=f2d_single_causal', self.text)
        self.assertNotIn('with_retain=True', self.text)

    def test_one_merged_model_not_dualuld(self):
        self.assertIn('merge_f2d_single_checkpoint.py', self.text)
        self.assertIn('model=Llama-3.2-1B-Instruct', self.text)
        self.assertNotIn('model=Llama-3.2-1B-Instruct_DualULD', self.text)
        self.assertNotIn('weight_a1=', self.text)
        self.assertNotIn('weight_a2=', self.text)

    def test_four_cell_group_and_objective(self):
        data = (ROOT / "ULD/uld/data/tofu.py").read_text(encoding="utf-8")
        sampler = (ROOT / "ULD/uld/data/datamodule.py").read_text(encoding="utf-8")
        self.assertIn('(\"C11\", \"C01\", \"C10\", \"C00\")', data)
        self.assertIn("class FactorialFourSampler", sampler)
        self.assertIn("for cell_index in range(4)", sampler)
        self.assertIn("unlearn_loss=factorial_causal_npo", self.text)


if __name__ == "__main__":
    unittest.main()
