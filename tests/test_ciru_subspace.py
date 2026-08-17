import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "train_ciru_subspace.py"
spec = importlib.util.spec_from_file_location("ciru_subspace", MODULE_PATH)
subspace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(subspace)


class CIRUSubspaceTest(unittest.TestCase):
    def test_raw_did_svd_recovers_consistent_interaction(self):
        rng = np.random.default_rng(7)
        units, hidden = 40, 12
        representations = rng.normal(scale=0.05, size=(units, 4, hidden))
        causal = np.zeros(hidden)
        causal[3] = 4.0
        # Cell order C11,C01,C10,C00.  Adding only to C11 creates a clean DiD.
        representations[:, 0, :] += causal
        result = subspace.estimate_layer(representations, rank=3)
        top = result["basis"][:, 0]
        self.assertGreater(abs(float(top @ (causal / np.linalg.norm(causal)))), 0.99)
        self.assertEqual(result["basis"].shape, (hidden, 3))
        self.assertTrue(np.isfinite(result["gate_coefficient"]))

    def test_gate_fit_is_finite_for_constant_feature(self):
        result = subspace.fit_scalar_logistic(
            np.ones(8, dtype=np.float32),
            np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        )
        self.assertTrue(all(np.isfinite(value) for value in result))


if __name__ == "__main__":
    unittest.main()
