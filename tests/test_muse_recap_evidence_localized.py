import inspect
import sys
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sweep_muse_recap_fullcoverage as sweep


class OffsetTokenizer:
    def __call__(self, text, **kwargs):
        # One character per token plus a zero-width special token.
        return {
            "input_ids": [0] + list(range(1, len(text) + 1)),
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
        }


class EvidenceLocalizedTests(unittest.TestCase):
    def test_extractor_covers_each_requested_evidence_class(self):
        documents = [
            'On January 12, 2024, Ada Lovelace called it "quasar lattice". '
            'The zephyroid manuscript describes xylophonic cartography.'
        ]
        masks, totals = sweep.evidence_character_masks(documents)
        self.assertGreater(totals["date"], 0)
        self.assertGreater(totals["entity"], 0)
        self.assertGreater(totals["quote"], 0)
        self.assertGreater(totals["rare_phrase"], 0)
        self.assertEqual(len(masks[0]["union"]), len(documents[0]))

    def test_localized_windows_use_complementary_ce_and_kl_masks(self):
        text = (
            'ordinary context words repeat here. ' * 8
            + 'Ada Lovelace wrote "Zephyroid Atlas" on January 12, 2024. '
            + 'ordinary context words repeat here. ' * 8
        )
        windows, stats = sweep.localized_completion_windows(OffsetTokenizer(), [text])
        self.assertGreater(len(windows), 0)
        for input_ids, labels, _, _, preserve in windows:
            self.assertEqual(len(input_ids), len(labels))
            self.assertEqual(len(labels), len(preserve))
            deletion = [label != -100 for label in labels]
            self.assertTrue(any(deletion))
            self.assertTrue(any(preserve))
            self.assertFalse(any(a and b for a, b in zip(deletion, preserve)))
            self.assertLessEqual(
                sum(deletion),
                max(
                    sweep.EVIDENCE_MIN_TOKENS,
                    __import__("math").ceil(
                        (sum(deletion) + sum(preserve))
                        * sweep.EVIDENCE_TARGET_FRACTION
                    ),
                ),
            )
        self.assertGreater(stats["evidence_target_tokens"], 0)
        self.assertGreater(stats["preserve_target_tokens"], 0)
        self.assertLessEqual(
            stats["evidence_target_fraction"],
            sweep.EVIDENCE_TARGET_FRACTION + 0.01,
        )

    def test_masked_kl_is_position_selected_and_token_normalized(self):
        source = inspect.getsource(sweep.masked_kl_to_reference)
        self.assertIn("preserve_mask[1:]", source)
        self.assertIn("per_position[positions].mean()", source)
        self.assertIn('reduction="none"', source)

    def test_v3_configuration_keeps_v2_capacity_and_points(self):
        self.assertEqual(sweep.EVIDENCE_VERSION, "muse-evidence-localized-v3")
        self.assertEqual(sweep.DEPTH, 8)
        self.assertEqual(sweep.RANK, 64)
        self.assertEqual(sweep.EVIDENCE_TARGET_FRACTION, 0.20)
        self.assertEqual(len(sweep.POINTS), 4)


if __name__ == "__main__":
    unittest.main()
