import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


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

    def test_eighty_source_design_nests_audited_forty(self):
        sources = [
            {"source_id": f"row-{index}", "question": "q", "answer": "a"}
            for index in range(200)
        ]
        forty = generator.stratified_sources(sources, 40, 20, seed=42)
        included_ids = [row["source_id"] for row in forty]
        eighty = generator.stratified_sources(
            sources, 80, 20, seed=42, include_source_ids=included_ids
        )
        eighty_ids = {row["source_id"] for row in eighty}
        self.assertTrue(set(included_ids).issubset(eighty_ids))
        counts = [0] * 10
        for row in eighty:
            counts[int(row["source_id"].split("-")[1]) // 20] += 1
        self.assertEqual(counts, [8] * 10)
        self.assertEqual(len(eighty_ids), 80)

    def test_two_hundred_source_design_uses_every_source(self):
        sources = [
            {"source_id": f"row-{index}", "question": "q", "answer": "a"}
            for index in range(200)
        ]
        selected = generator.stratified_sources(sources, 200, 20, seed=42)
        self.assertEqual(
            {row["source_id"] for row in selected},
            {row["source_id"] for row in sources},
        )

    def test_schema_failure_is_fed_back_to_the_next_attempt(self):
        valid = {
            "target_entity": "Basil Hart",
            "replacement_entity": "Elian Mercer",
            "target_relation": "award won",
            "placebo_relation": "city of residence",
            "invariants": {
                "task": "factual QA",
                "style": "short",
                "difficulty": "single hop",
                "answer_format": "noun phrase",
            },
            "cells": {
                "C11": {"question": "ignored", "answer": "ignored"},
                "C01": {
                    "question": "Which award did Elian Mercer win?",
                    "answer": "The Northbridge Medal",
                },
                "C10": {
                    "question": "Where does Basil Hart live?",
                    "answer": "The city Grayhaven",
                },
                "C00": {
                    "question": "Where does Elian Mercer live?",
                    "answer": "The city Westhaven",
                },
            },
        }
        invalid = json.loads(json.dumps(valid))
        invalid["replacement_entity"] = "A Different Name"

        class Completions:
            def __init__(self):
                self.requests = []
                self.outputs = [invalid, valid]

            def create(self, **request):
                self.requests.append(request)
                content = json.dumps(self.outputs.pop(0))
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
                )

        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        args = SimpleNamespace(
            retries=2,
            model="test-model",
            temperature=1.0,
            resolved_json_mode="prompt",
        )
        source = {
            "source_id": "row-1",
            "question": "Which award did Basil Hart win?",
            "answer": "Riverdale Book Award",
        }
        original_sleep = generator.time.sleep
        generator.time.sleep = lambda _seconds: None
        try:
            record = generator.generate_one(
                client, args, source, replacement_entity="Elian Mercer"
            )
        finally:
            generator.time.sleep = original_sleep
        self.assertEqual(record["cells"]["C11"]["question"], source["question"])
        second_prompt = completions.requests[1]["messages"][1]["content"]
        first_prompt = completions.requests[0]["messages"][1]["content"]
        self.assertIn("Required replacement_entity: Elian Mercer", first_prompt)
        self.assertIn("previous JSON failed", second_prompt)
        self.assertIn("replacement_entity", second_prompt)


if __name__ == "__main__":
    unittest.main()
