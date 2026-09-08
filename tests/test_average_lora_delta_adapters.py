import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scripts.average_lora_delta_adapters import average_adapters


class AverageLoraDeltaAdaptersTest(unittest.TestCase):
    def make_adapter(
        self,
        root: Path,
        name: str,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        base_model_name_or_path: str = "test-model",
        target_modules: list[str] | None = None,
    ) -> Path:
        path = root / name
        path.mkdir()
        config = {
            "base_model_name_or_path": base_model_name_or_path,
            "inference_mode": True,
            "lora_alpha": 4,
            "r": 2,
            "target_modules": target_modules or ["q_proj"],
            "peft_type": "LORA",
        }
        (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        save_file(
            {
                "base.q_proj.lora_A.weight": a,
                "base.q_proj.lora_B.weight": b,
            },
            str(path / "adapter_model.safetensors"),
        )
        return path

    def test_union_rank_reconstructs_mean_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a1 = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
            b1 = torch.tensor([[1.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
            a2 = torch.tensor([[2.0, 1.0, 0.0], [1.0, -1.0, 1.0]])
            b2 = torch.tensor([[0.0, 1.0], [2.0, 0.0], [-1.0, 1.0]])
            paths = [
                self.make_adapter(root, "one", a1, b1),
                self.make_adapter(root, "two", a2, b2),
            ]
            output = root / "soup"
            metadata = average_adapters(paths, output, target_rank=4, target_alpha=8)
            state = load_file(str(output / "adapter_model.safetensors"))
            actual = state["base.q_proj.lora_B.weight"] @ state["base.q_proj.lora_A.weight"]
            expected = (b1 @ a1 + b2 @ a2) / 2
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
            self.assertLess(metadata["relative_projection_error_max"], 1e-12)
            config = json.loads((output / "adapter_config.json").read_text())
            self.assertEqual(config["r"], 4)
            self.assertEqual(config["lora_alpha"], 8)

    def test_different_local_base_paths_are_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            paths = [
                self.make_adapter(
                    root,
                    "seed42",
                    a,
                    b,
                    base_model_name_or_path="/runs/seed42/a1/fullmodel",
                ),
                self.make_adapter(
                    root,
                    "seed43",
                    a,
                    b,
                    base_model_name_or_path="/runs/seed43/a1/fullmodel",
                ),
            ]
            output = root / "soup"
            metadata = average_adapters(
                paths, output, target_rank=2, target_alpha=4
            )
            self.assertEqual(metadata["source_count"], 2)

    def test_structural_lora_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            paths = [
                self.make_adapter(root, "one", a, b),
                self.make_adapter(
                    root,
                    "two",
                    a,
                    b,
                    target_modules=["k_proj"],
                ),
            ]
            with self.assertRaisesRegex(ValueError, "target_modules"):
                average_adapters(
                    paths, root / "soup", target_rank=2, target_alpha=4
                )

    def test_projected_rank_has_expected_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(3):
                generator = torch.Generator().manual_seed(index)
                a = torch.randn(2, 5, generator=generator)
                b = torch.randn(4, 2, generator=generator)
                paths.append(self.make_adapter(root, str(index), a, b))
            output = root / "projected"
            metadata = average_adapters(paths, output, target_rank=2, target_alpha=4)
            state = load_file(str(output / "adapter_model.safetensors"))
            self.assertEqual(tuple(state["base.q_proj.lora_A.weight"].shape), (2, 5))
            self.assertEqual(tuple(state["base.q_proj.lora_B.weight"].shape), (4, 2))
            self.assertGreaterEqual(metadata["relative_projection_error_max"], 0.0)
            self.assertLessEqual(metadata["relative_projection_error_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
