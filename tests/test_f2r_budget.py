import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_f2r_budget.py"
spec = importlib.util.spec_from_file_location("f2r_budget", MODULE_PATH)
budget_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(budget_module)


def record(source: int, view: int):
    return {"source_id": f"source-{source}", "view": view, "payload": source + view}


class F2RBudgetTest(unittest.TestCase):
    def test_source_coverage_precedes_second_views(self):
        records = [record(source, view) for source in range(5) for view in range(2)]
        ordered = budget_module.ordered_source_balanced(records, seed=42)
        self.assertEqual(len({row["source_id"] for row in ordered[:5]}), 5)
        self.assertEqual(len({row["source_id"] for row in ordered[:3]}), 3)

    def test_subsets_are_nested_and_reproducible(self):
        records = [record(source, view) for source in range(5) for view in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "full.jsonl"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            rows = budget_module.build_subsets(
                input_path,
                root / "out",
                [3, 5, 10],
                seed=7,
                manifest_path=root / "out" / "manifest.csv",
            )
            subsets = []
            for row in rows:
                with Path(row["path"]).open(encoding="utf-8") as handle:
                    subsets.append(
                        {
                            (item["source_id"], item["view"])
                            for item in map(json.loads, handle)
                        }
                    )
            self.assertLessEqual(subsets[0], subsets[1])
            self.assertLessEqual(subsets[1], subsets[2])
            self.assertEqual([row["sources"] for row in rows], [3, 5, 5])
            self.assertEqual([row["max_views"] for row in rows], [1, 1, 2])

    def test_budget_larger_than_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "full.jsonl"
            input_path.write_text(json.dumps(record(0, 0)) + "\n")
            with self.assertRaisesRegex(ValueError, "exceeds"):
                budget_module.build_subsets(
                    input_path,
                    root / "out",
                    [2],
                    seed=42,
                    manifest_path=root / "manifest.csv",
                )


if __name__ == "__main__":
    unittest.main()
