import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_slots_v5_4.py"
EVAL_LOG = (
    ROOT / "ULD/data/retain95_llama_wd0.01/eval_results/"
    "ds_size300/eval_log_forget.json"
)
MANIFEST = ROOT / "ULD/configs/data/tofu_forget05_author_blocks.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = load_module("tofu_author_slots_v54_test", GENERATOR_PATH)
anchor = generator.anchors


def source_rows():
    with EVAL_LOG.open(encoding="utf-8") as handle:
        generated = json.load(handle)["generated_text"]
    rows = []
    for index, (prompt, _model_text, ground_truth) in enumerate(generated):
        question = re.sub(r"^\[INST\]\s*|\s*\[/INST\]$", "", prompt).strip()
        rows.append((index, question, ground_truth))
    return rows


def fixture_value(group, ordinal):
    old, kind = group["text"], group["kind"]
    if anchor.normalise(old) in anchor.PLURAL_QUANTIFIERS:
        return next(
            value for value in sorted(anchor.PLURAL_QUANTIFIERS)
            if value != anchor.normalise(old)
        )
    if kind == "year":
        return str(int(old) + 1)
    if kind == "number":
        return str(int(float(old)) + 1)
    if kind == "date":
        return "1975-05-03"
    if kind == "quoted":
        return old + " Revised"
    if kind == "proper":
        return "Alternative Meridian"
    return f"alternate{chr(ord('a') + ordinal % 26)}"


class AuthorSlotsV54Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator.configure_shared_modules()
        with MANIFEST.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        cls.authors = [item["canonical_name"] for item in manifest["authors"]]
        rows = source_rows()
        cls.blocks = {}
        cls.profiles = {}
        cls.selected = {}
        for block_id in (1, 4):
            target = cls.authors[block_id]
            sources = [
                {
                    "source_id": f"forget05_perturbed-{index:05d}",
                    "question": question,
                    "answer": answer,
                }
                for index, question, answer in rows[
                    block_id * 20:(block_id + 1) * 20
                ]
            ]
            block = generator.v52.with_contracts_and_anchors({
                "block_id": block_id,
                "target_entity": target,
                "sources": sources,
            })
            cls.blocks[block_id] = block
            groups = {
                item["group_id"]: item
                for item in block["anchor_catalog"]["groups"]
            }
            selected = {}
            for ordinal, source in enumerate(block["sources"]):
                if not generator.v52.source_fact_change_required(source, target):
                    continue
                slots = generator.local_slots(block, source["source_id"])
                answer_format = source["contract"]["answer_format"]
                if answer_format == "date_or_year":
                    order = ("date", "year", "quoted", "proper", "token", "number")
                elif answer_format == "numeric":
                    order = ("number", "quoted", "proper", "token", "year", "date")
                else:
                    order = ("quoted", "proper", "token", "year", "number", "date")
                chosen = next(
                    item for kind in order for item in slots
                    if item["kind"] == kind
                )
                group = groups[chosen["group_id"]]
                selected[source["source_id"]] = (
                    chosen["slot_key"], fixture_value(group, ordinal)
                )
            cls.selected[block_id] = selected
            ledger = []
            replacement = (
                "Harper Mei Collins" if block_id == 1 else "Mika Hayashi"
            )
            for source in block["sources"]:
                source_id = source["source_id"]
                required = generator.v52.source_fact_change_required(source, target)
                policy = generator.v53.policy_for_source(source, target)
                value = selected[source_id][1] if required else replacement
                ledger.append({
                    "source_id": source_id,
                    "fact_key": f"fact:{source_id}" if required else "policy",
                    "target_relation": (
                        f"relation for {source_id}" if required else "author identity"
                    ),
                    "replacement_fact": value,
                    "fact_change_required": required,
                    "intervention_policy": policy,
                })
            raw_profile = {
                "target_entity": target,
                "replacement_entity": replacement,
                "replacement_pronouns": block["replacement_pronouns"],
                "profile_summary": "One coherent fictional author.",
                "fact_ledger": ledger,
            }
            profile = generator.v53.validate_profile(
                block, raw_profile, cls.authors
            )
            profile["profile_attempt"] = 1
            profile["profile_semantic_judge"] = {
                "coherent": True,
                "conflicting_source_ids": [],
                "reason": "offline coherent ledger fixture",
            }
            cls.profiles[block_id] = profile
            profile["_raw"] = raw_profile

    def make_candidate(self, block_id, source):
        profile = self.profiles[block_id]
        entry = profile["fact_ledger_by_source"][source["source_id"]]
        edits = []
        if entry["fact_change_required"]:
            slot_key, value = self.selected[block_id][source["source_id"]]
            edits = [{"slot_key": slot_key, "replacement_value": value}]
        raw = {
            "source_id": source["source_id"],
            "target_relation": entry["target_relation"],
            "ledger_fact_key": entry["fact_key"],
            "ledger_replacement_fact": entry["replacement_fact"],
            "edits": edits,
        }
        return generator.validate_row_candidate(
            self.blocks[block_id], source, profile, raw
        )

    def test_payload_exposes_local_slots_without_internal_group_ids(self):
        source = self.blocks[1]["sources"][4]
        payload = generator.row_payload(
            self.blocks[1], source, self.profiles[1]
        )
        self.assertTrue(payload["eligible_local_slots"])
        self.assertNotIn("eligible_answer_group_ids", payload)
        self.assertTrue(all(
            "group_id" not in item for item in payload["eligible_local_slots"]
        ))

    def test_one_edit_list_derives_group_coverage(self):
        source = self.blocks[1]["sources"][4]
        candidate = self.make_candidate(1, source)
        self.assertEqual(len(candidate["target_group_ids"]), 1)
        self.assertEqual(
            set(candidate["target_group_ids"]),
            set(candidate["anchor_replacements"]),
        )

    def test_quoted_markup_is_canonicalised_by_code(self):
        source = self.blocks[4]["sources"][3]
        slot = next(
            item for item in generator.local_slots(
                self.blocks[4], source["source_id"]
            ) if item["kind"] == "quoted"
        )
        group = next(
            item for item in self.blocks[4]["anchor_catalog"]["groups"]
            if item["group_id"] == slot["group_id"]
        )
        value = generator.canonical_surface_value(
            group,
            '"Northern Horizon Prize"',
            "Northern Horizon Prize",
            source["contract"],
        )
        self.assertEqual(value, "Northern Horizon Prize")

    def test_multiword_token_fact_is_projected_to_one_content_token(self):
        source = self.blocks[4]["sources"][14]
        slot = next(
            item for item in generator.local_slots(
                self.blocks[4], source["source_id"]
            ) if item["old_text"] == "acclaim"
        )
        group = next(
            item for item in self.blocks[4]["anchor_catalog"]["groups"]
            if item["group_id"] == slot["group_id"]
        )
        value = generator.canonical_surface_value(
            group,
            "regional prizes and fellowships",
            "regional prizes and fellowships",
            source["contract"],
        )
        self.assertEqual(value, "regional")

    def test_identity_row_is_rendered_without_model_or_fact_edit(self):
        source = self.blocks[4]["sources"][0]
        self.assertFalse(
            self.profiles[4]["fact_ledger_by_source"][source["source_id"]][
                "fact_change_required"
            ]
        )
        calls = []

        def fake_request(*_args):
            calls.append(True)
            raise AssertionError("identity row must not call the mapper")

        args = SimpleNamespace(
            model="offline", temperature=1.0, request_retries=1,
            row_retries=1,
        )
        original = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generator.generate_row(
                    None,
                    args,
                    self.blocks[4],
                    source,
                    self.profiles[4],
                    Path(directory),
                )
        finally:
            generator.v52.v2.request_json = original
        self.assertFalse(calls)
        self.assertEqual(result["target_group_ids"], [])
        self.assertFalse(result["fact_change_required"])

    def test_materialization_keeps_same_group_row_local(self):
        block = self.blocks[1]
        profile = self.profiles[1]
        candidates = {
            source["source_id"]: self.make_candidate(1, source)
            for source in block["sources"]
        }
        plan = generator.materialize_plan(block, profile, candidates, seed=42)
        self.assertEqual(plan["reconciliation_conflicts"], [])
        self.assertEqual(len(plan["row_plans"]), 20)
        self.assertEqual(
            set(plan["anchor_replacements"]),
            {source["source_id"] for source in block["sources"]},
        )

    def test_valid_v53_profile_and_row_are_migrated_without_api(self):
        block = self.blocks[1]
        source = block["sources"][4]
        profile = self.profiles[1]
        candidate = self.make_candidate(1, source)
        group_id = candidate["target_group_ids"][0]
        with tempfile.TemporaryDirectory() as source_directory, \
                tempfile.TemporaryDirectory() as target_directory:
            source_state = Path(source_directory)
            target_state = Path(target_directory)
            generator.v52.write_json(
                generator.v52.profile_path(source_state, 1),
                {
                    "design_version": "tofu-author-ledger-rowlocal-v5.3",
                    "anchor_catalog_digest": block["anchor_catalog"]["digest"],
                    "ledger_digest": profile["ledger_digest"],
                    "profile_attempt": 2,
                    "profile": profile["_raw"],
                    "semantic_judge": profile["profile_semantic_judge"],
                },
            )
            generator.v52.write_json(
                generator.v52.row_path(
                    source_state, 1, source["source_id"]
                ),
                {
                    "design_version": "tofu-author-ledger-rowlocal-v5.3",
                    "ledger_digest": profile["ledger_digest"],
                    "mapping_attempt": 2,
                    "repair_generation": 0,
                    "candidate": {
                        "source_id": source["source_id"],
                        "target_relation": candidate["target_relation"],
                        "ledger_fact_key": candidate["ledger_fact_key"],
                        "ledger_replacement_fact": candidate[
                            "ledger_replacement_fact"
                        ],
                        "target_group_ids": [group_id],
                        "anchor_replacements": [{
                            "group_id": group_id,
                            "replacement_value": candidate[
                                "anchor_replacements"
                            ][group_id],
                        }],
                    },
                },
            )
            self.assertTrue(generator.migrate_v53_profile(
                block, self.authors, source_state, target_state
            ))
            migrated_profile = generator.load_profile_for_migration(
                block, self.authors, target_state
            )
            self.assertIsNotNone(migrated_profile)
            count = generator.migrate_v53_rows(
                block, migrated_profile, source_state, target_state
            )
            self.assertEqual(count, 1)
            cached = generator.load_cached_row(
                generator.v52.row_path(
                    target_state, 1, source["source_id"]
                ),
                block,
                source,
                migrated_profile,
            )
            self.assertIsNotNone(cached)
            self.assertEqual(cached["target_group_ids"], [group_id])

    def test_end_to_end_block_generation_uses_local_slots(self):
        block = self.blocks[1]
        profile = self.profiles[1]
        source_by_id = {
            source["source_id"]: source for source in block["sources"]
        }

        def fake_request(_client, _args, prompt, payload, _label):
            if prompt == generator.v53.PROFILE_PROMPT:
                return profile["_raw"]
            if prompt == generator.v53.PROFILE_JUDGE_PROMPT:
                return {
                    "coherent": True,
                    "conflicting_source_ids": [],
                    "reason": "offline coherent ledger fixture",
                }
            if prompt == generator.ROW_PROMPT:
                source = source_by_id[payload["source_id"]]
                return generator.candidate_for_prompt(
                    self.make_candidate(1, source)
                )
            if prompt == generator.v53.JUDGE_PROMPT:
                return {
                    "verdicts": [
                        {
                            "source_id": row["source_id"],
                            "target_relation_match": True,
                            "target_fact_changed": True,
                            "profile_consistent": True,
                            "natural_surface": True,
                            "reason": "row agrees with frozen ledger",
                        }
                        for row in payload["rows"]
                    ]
                }
            raise AssertionError("unexpected prompt")

        args = SimpleNamespace(
            split="forget05_perturbed",
            seed=42,
            model="offline",
            judge_model="offline",
            temperature=1.0,
            judge_temperature=1.0,
            request_retries=1,
            profile_retries=1,
            row_retries=1,
            judge_rounds=1,
            row_concurrency=4,
        )
        original = generator.v52.v2.request_json
        generator.v52.v2.request_json = fake_request
        try:
            with tempfile.TemporaryDirectory() as directory:
                result = generator.generate_block(
                    None,
                    None,
                    args,
                    block,
                    self.authors,
                    Path(directory),
                )
        finally:
            generator.v52.v2.request_json = original
        self.assertEqual(result["design_version"], generator.DESIGN_VERSION)
        self.assertEqual(len(result["records"]), 20)
        self.assertTrue(all(
            record["generation"]["mapping_scope"] == generator.MAPPING_SCOPE
            for record in result["records"]
        ))


if __name__ == "__main__":
    unittest.main()
