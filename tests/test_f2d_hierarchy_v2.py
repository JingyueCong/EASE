import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "annotate_f2d_hierarchy_v2.py"
spec = importlib.util.spec_from_file_location("f2d_hierarchy_v2", MODULE_PATH)
hierarchy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hierarchy)


def base_record():
    return {
        "source_id": "row-1",
        "target_entity": "Ava North",
        "replacement_entity": "Bea South",
        "target_relation": "award received",
        "placebo_relation": "place of birth",
        "cells": {
            "C11": {
                "question": "Which award did Ava North receive?",
                "answer": "Ava North received the Silver Quill. She later taught in Rome.",
            },
            "C01": {
                "question": "Which award did Bea South receive?",
                "answer": "Bea South received the Amber Crown. She later taught in Lima.",
            },
            "C10": {
                "question": "Where was Ava North born?",
                "answer": "Ava North was born in Oslo. She later taught in Rome.",
            },
            "C00": {
                "question": "Where was Bea South born?",
                "answer": "Bea South was born in Lima. She later taught in Naples.",
            },
        },
    }


def cell_annotation(claim, evidence, subject, relation, object_value):
    return {
        "answerability": "answered",
        "polarity": "affirmative",
        "claims": [{
            "claim_text": claim,
            "evidence_texts": [evidence],
            "subject": subject,
            "relation": relation,
            "object": object_value,
            "qualifiers": [],
        }],
    }


def valid_payload():
    return {
        "pair_checks": {
            "target": {"valid": True, "reason": ""},
            "placebo": {"valid": True, "reason": ""},
        },
        "cells": {
            "C11": cell_annotation(
                "Ava North received the Silver Quill.",
                "Silver Quill", "Ava North", "award received", "Silver Quill",
            ),
            "C01": cell_annotation(
                "Bea South received the Amber Crown.",
                "Amber Crown", "Bea South", "award received", "Amber Crown",
            ),
            "C10": cell_annotation(
                "Ava North was born in Oslo.",
                "Oslo", "Ava North", "place of birth", "Oslo",
            ),
            "C00": cell_annotation(
                "Bea South was born in Lima.",
                "Lima", "Bea South", "place of birth", "Lima",
            ),
        },
    }


class F2DHierarchyV2Test(unittest.TestCase):
    def test_semantic_spans_exclude_unrelated_sentences_and_subjects(self):
        record = hierarchy.apply_semantic_annotation(base_record(), valid_payload())
        supervision = record["cells"]["C11"]["supervision"]
        answer = record["cells"]["C11"]["answer"]
        claim = " ".join(answer[a:b] for a, b in supervision["claim_spans"])
        evidence = " ".join(answer[a:b] for a, b in supervision["evidence_spans"])
        self.assertEqual(claim, "Ava North received the Silver Quill.")
        self.assertEqual(evidence, "Silver Quill")
        self.assertNotIn("taught in Rome", claim)
        self.assertEqual(supervision["version"], "paired-hierarchy-v2")

    def test_adjacent_claims_are_not_merged(self):
        self.assertEqual(
            hierarchy.merge_overlapping_spans([[0, 4], [4, 9]]),
            [[0, 4], [4, 9]],
        )

    def test_subject_in_evidence_is_rejected(self):
        payload = valid_payload()
        payload["cells"]["C11"]["claims"][0]["evidence_texts"] = [
            "Ava North received the Silver Quill"
        ]
        with self.assertRaisesRegex(hierarchy.AnnotationError, "subject name"):
            hierarchy.apply_semantic_annotation(base_record(), payload)

    def test_pair_polarity_mismatch_is_rejected(self):
        payload = valid_payload()
        payload["cells"]["C01"]["polarity"] = "negative"
        with self.assertRaisesRegex(hierarchy.AnnotationError, "polarity mismatch"):
            hierarchy.apply_semantic_annotation(base_record(), payload)

    def test_pair_answerability_mismatch_is_rejected(self):
        payload = valid_payload()
        payload["cells"]["C11"]["answerability"] = "unknown"
        with self.assertRaisesRegex(hierarchy.AnnotationError, "answerability mismatch"):
            hierarchy.apply_semantic_annotation(base_record(), payload)

    def test_identical_short_outcomes_keep_only_semantic_object_evidence(self):
        record = base_record()
        record["cells"]["C10"]["answer"] = "Oslo."
        record["cells"]["C00"]["answer"] = "Oslo."
        payload = valid_payload()
        payload["cells"]["C10"] = cell_annotation(
            "Oslo.", "Oslo", "Ava North", "place of birth", "Oslo"
        )
        payload["cells"]["C00"] = cell_annotation(
            "Oslo.", "Oslo", "Bea South", "place of birth", "Oslo"
        )
        annotated = hierarchy.apply_semantic_annotation(record, payload)
        for cell_name in ("C10", "C00"):
            supervision = annotated["cells"][cell_name]["supervision"]
            self.assertEqual(supervision["evidence_spans"], [[0, 4]])

    def test_invalid_pair_check_is_rejected(self):
        payload = valid_payload()
        payload["pair_checks"]["target"] = {
            "valid": False,
            "reason": "answerability mismatch",
        }
        with self.assertRaisesRegex(hierarchy.AnnotationError, "answerability mismatch"):
            hierarchy.apply_semantic_annotation(base_record(), payload)

    def test_full_long_single_sentence_claim_is_rejected(self):
        record = base_record()
        answer = (
            "Ava North received the Silver Quill and later described its history, "
            "selection process, cultural importance, public reception, and lasting "
            "influence on every part of her writing career."
        )
        record["cells"]["C11"]["answer"] = answer
        payload = valid_payload()
        payload["cells"]["C11"] = cell_annotation(
            answer, "Silver Quill", "Ava North", "award received", "Silver Quill"
        )
        with self.assertRaisesRegex(hierarchy.AnnotationError, "long answer"):
            hierarchy.apply_semantic_annotation(record, payload)

    def test_unsupported_temperature_is_retried_with_api_default(self):
        class UnsupportedTemperature(Exception):
            body = {
                "message": "Unsupported value: temperature only supports default",
                "param": "temperature",
                "code": "unsupported_value",
            }

        class Completions:
            def __init__(self):
                self.requests = []

            def create(self, **request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    raise UnsupportedTemperature("temperature is unsupported")
                return SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps(valid_payload()))
                    )]
                )

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        args = SimpleNamespace(
            retries=3,
            model="gpt-5-mini",
            temperature=0.0,
            resolved_json_mode="prompt",
        )
        annotated = hierarchy.annotate_openai(client, args, base_record())
        self.assertEqual(
            annotated["cells"]["C11"]["supervision"]["version"],
            "paired-hierarchy-v2",
        )
        self.assertIn("temperature", completions.requests[0])
        self.assertNotIn("temperature", completions.requests[1])


if __name__ == "__main__":
    unittest.main()
