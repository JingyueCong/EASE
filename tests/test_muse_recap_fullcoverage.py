import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sweep_muse_recap_fullcoverage as sweep


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(int(text)))}


class FullCoverageTests(unittest.TestCase):
    def test_long_windows_cover_all_tokens_after_prefix_seed(self):
        windows, stats = sweep.long_completion_windows(FakeTokenizer(), ["400"])
        targets = set()
        for ids, labels, _, start in windows:
            left = max(0, start - sweep.PREFIX_TOKENS)
            targets.update(left + i for i, label in enumerate(labels) if label != -100)
            self.assertEqual(len(ids), len(labels))
        self.assertEqual(targets, set(range(8, 400)))
        self.assertGreaterEqual(stats["target_tokens"], 392)

    def test_changed_mask_excludes_shared_tokens(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                values = {"positive": [1, 2, 9, 8, 5], "control": [1, 2, 3, 4, 5]}
                return {"input_ids": values[text]}
        _, labels, controls = sweep.changed_token_example(Tokenizer(), "positive", "control")
        self.assertEqual(labels, [-100, -100, 9, 8, -100])
        self.assertEqual(controls, [1, 2, 3, 4, 5])

    def test_design_is_retain_free_and_capacity_matched(self):
        self.assertEqual(sweep.DEPTH, 8)
        self.assertEqual(sweep.RANK, 64)
        self.assertEqual(sweep.PREFIX_TOKENS, 1024)
        self.assertEqual(sweep.TARGET_TOKENS, 128)
        self.assertEqual(len(sweep.POINTS), 4)


if __name__ == "__main__":
    unittest.main()
