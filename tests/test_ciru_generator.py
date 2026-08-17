import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD" / "scripts" / "generate_ciru40.py"
spec = importlib.util.spec_from_file_location("ciru_generator", MODULE_PATH)
generator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(generator)


class CIRUGeneratorTest(unittest.TestCase):
    def test_forget05_design_draws_four_sources_per_author_block(self):
        sources = [
            {"source_id": f"row-{index}", "question": "q", "answer": "a"}
            for index in range(200)
        ]
        selected = generator.stratified_sources(sources, 40, 20, seed=42)
        counts = [0] * 10
        for row in selected:
            counts[int(row["source_id"].split("-")[1]) // 20] += 1
        self.assertEqual(counts, [4] * 10)
        self.assertEqual(len({row["source_id"] for row in selected}), 40)

    def test_sampling_is_seed_reproducible(self):
        sources = [
            {"source_id": f"row-{index}", "question": "q", "answer": "a"}
            for index in range(200)
        ]
        first = generator.stratified_sources(sources, 40, 20, seed=9)
        second = generator.stratified_sources(sources, 40, 20, seed=9)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
