import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "open-unlearning" / "src" / "model" / "f2r_routing.py"
spec = importlib.util.spec_from_file_location("f2r_routing", MODULE_PATH)
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


class F2RSequenceRoutingTest(unittest.TestCase):
    def setUp(self):
        self.state = {
            "patterns": (torch.tensor([10, 11]),),
            "prompt_end_pattern": torch.tensor([50]),
            "match_scale": 1.0,
            "nonmatch_scale": 0.0,
            "entities": ("Example Author",),
        }

    def test_prompt_entity_enables_residual(self):
        input_ids = torch.tensor([[1, 10, 11, 2, 50, 7, 8]])
        labels = torch.tensor([[-100, -100, -100, -100, -100, 7, 8]])
        scales, matched = routing.sequence_router_scale(
            input_ids,
            torch.ones_like(input_ids),
            self.state,
            labels=labels,
        )
        torch.testing.assert_close(scales, torch.tensor([1.0]))
        self.assertEqual(matched.tolist(), [True])

    def test_teacher_forced_answer_cannot_trigger_router(self):
        input_ids = torch.tensor([[1, 2, 3, 50, 10, 11]])
        labels = torch.tensor([[-100, -100, -100, -100, 10, 11]])
        scales, matched = routing.sequence_router_scale(
            input_ids,
            torch.ones_like(input_ids),
            self.state,
            labels=labels,
        )
        torch.testing.assert_close(scales, torch.tensor([0.0]))
        self.assertEqual(matched.tolist(), [False])

    def test_generated_answer_cannot_trigger_router(self):
        input_ids = torch.tensor([[1, 2, 3, 50, 10, 11]])
        scales, matched = routing.sequence_router_scale(
            input_ids,
            torch.ones_like(input_ids),
            self.state,
        )
        torch.testing.assert_close(scales, torch.tensor([0.0]))
        self.assertEqual(matched.tolist(), [False])

    def test_batch_padding_and_nonmatch_scale(self):
        state = dict(self.state, nonmatch_scale=0.25)
        input_ids = torch.tensor([[0, 10, 11, 50], [0, 4, 5, 50]])
        attention = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]])
        scales, matched = routing.sequence_router_scale(
            input_ids, attention, state, dtype=torch.float32
        )
        torch.testing.assert_close(scales, torch.tensor([1.0, 0.25]))
        self.assertEqual(matched.tolist(), [True, False])

    def test_artifact_validation_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.json"
            path.write_text(
                json.dumps(
                    {
                        "kind": "forget_entity_subsequence_v1",
                        "patterns": [{"entity": "A", "token_ids": [10, 11]}],
                        "prompt_end_token_ids": [50],
                        "entities": ["A"],
                        "match_scale": 1.0,
                        "nonmatch_scale": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            state = routing.load_sequence_router(path, device=torch.device("cpu"))
            self.assertEqual(tuple(state["patterns"][0].tolist()), (10, 11))
            self.assertEqual(state["entities"], ("A",))


if __name__ == "__main__":
    unittest.main()
