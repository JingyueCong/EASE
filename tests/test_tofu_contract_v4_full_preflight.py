import importlib.util
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "ULD/uld/data/tofu_contract_v4.py"
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_typed_v4.py"
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


v4 = load_module("tofu_contract_v4_preflight", MODULE_PATH)
generator = load_module("tofu_contract_v4_generator_preflight", GENERATOR_PATH)


def source_rows():
    with EVAL_LOG.open(encoding="utf-8") as handle:
        generated = json.load(handle)["generated_text"]
    rows = []
    for index, (prompt, _model_text, ground_truth) in enumerate(generated):
        question = re.sub(
            r"^\[INST\]\s*|\s*\[/INST\]$", "", prompt
        ).strip()
        rows.append((index, question, ground_truth))
    return rows


class FullTOFUContractV4Preflight(unittest.TestCase):
    def test_all_200_c11_contracts_have_deterministic_placebos(self):
        with MANIFEST.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        authors = {
            item["block_id"]: item["canonical_name"]
            for item in manifest["authors"]
        }
        rows = source_rows()
        self.assertEqual(len(rows), 200)
        failures = []
        observed_relations = set()
        for index, question, answer in rows:
            block_id, query_index = divmod(index, 20)
            target = authors[block_id]
            replacement = f"Twin Author {chr(ord('A') + block_id)}"
            contract = v4.derive_contract(question, answer, target)
            rendered = v4.render_professional_placebo(
                contract, target, replacement, query_index
            )
            observed_relations.add(rendered["placebo_relation"])
            for cell_name in ("C10", "C00"):
                errors = v4.contract_errors(rendered[cell_name], contract)
                if errors:
                    failures.append((index, cell_name, errors, rendered[cell_name]))
            self.assertNotEqual(rendered["C10"], rendered["C00"])
        self.assertEqual(observed_relations, {item[0] for item in v4.PROFESSIONAL_PLACEBOS})
        self.assertEqual(failures, [], failures[:10])

    def test_all_200_c11_contracts_survive_exact_identity_edits(self):
        """Exercise the C01 renderer over every real TOFU surface contract."""
        with MANIFEST.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        authors = {
            item["block_id"]: item["canonical_name"]
            for item in manifest["authors"]
        }
        for index, question, answer in source_rows():
            target = authors[index // 20]
            replacement = f"Twin Author {chr(ord('A') + index // 20)}"
            source = {"question": question, "answer": answer}
            plan = {
                "target_relation": "identity",
                "question_edits": (
                    [{"old": target, "new": replacement}]
                    if target in question else []
                ),
                "answer_edits": (
                    [{"old": target, "new": replacement}]
                    if target in answer else []
                ),
            }
            rendered = v4.render_target_counterfactual(
                source, target, replacement, plan, require_fact_change=False
            )
            contract = v4.derive_contract(question, answer, target)
            self.assertEqual(v4.contract_errors(rendered, contract), [], index)

    def test_exact_edit_boundary_rejects_full_rewrites_and_name_only_facts(self):
        source = {
            "question": "What literary genre does Ada North primarily write in?",
            "answer": "Ada North primarily writes historical fiction about coastal communities.",
        }
        with self.assertRaisesRegex(ValueError, "rewrite more than 60%"):
            v4.apply_exact_edits(
                source["answer"],
                [{"old": source["answer"], "new": "Bea South writes poetry."}],
                "answer",
            )
        with self.assertRaisesRegex(ValueError, "only author identity"):
            v4.render_target_counterfactual(
                source,
                "Ada North",
                "Bea South",
                {
                    "target_relation": "primary literary genre",
                    "question_edits": [{"old": "Ada North", "new": "Bea South"}],
                    "answer_edits": [{"old": "Ada North", "new": "Bea South"}],
                },
            )

    def test_frozen_placebo_library_contains_only_professional_relations(self):
        self.assertEqual(len(v4.PROFESSIONAL_PLACEBOS), 20)
        relations = [item[0] for item in v4.PROFESSIONAL_PLACEBOS]
        self.assertEqual(len(set(relations)), 20)
        forbidden = {"food", "pet", "social_media", "color", "zodiac"}
        self.assertFalse(any(token in relation for relation in relations for token in forbidden))

    def test_complete_200_row_generator_contract_accepts_atomic_edits(self):
        """Validate the complete generator boundary without calling an API.

        The replacement tokens are mechanical test fixtures, not experimental
        facts; this test proves that all real TOFU contracts pass through the
        exact-edit IR, deterministic placebo renderer, and block validator.
        """
        with MANIFEST.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        authors = [item["canonical_name"] for item in manifest["authors"]]
        rows = source_rows()
        stop = {
            "yes", "no", "not", "never", "likely", "may", "unclear",
            "unknown", "unavailable", "information", "there", "specific",
            "details", "confirmed", "publicly", "available", "documented",
            "known",
        }
        for block_id, target in enumerate(authors):
            replacement = f"Twin Author {chr(ord('A') + block_id)}"
            sources, plans = [], []
            for index, question, answer in rows[block_id * 20 : (block_id + 1) * 20]:
                source_id = f"forget05_perturbed-{index:05d}"
                source = {"source_id": source_id, "question": question, "answer": answer}
                source["contract"] = v4.derive_contract(question, answer, target)
                sources.append(source)
                question_edits = ([{"old": target, "new": replacement}]
                                  if target in question else [])
                answer_edits = ([{"old": target, "new": replacement}]
                                if target in answer else [])
                relation = "identity" if v4.normalise(answer) == v4.normalise(target) else "factual relation"
                replacement_fact = replacement
                if source["contract"]["response_mode"] != "unavailable" and relation != "identity":
                    candidates = []
                    for match in v4.WORD_PATTERN.finditer(answer):
                        old = match.group()
                        if (old.casefold() in stop or len(old) < 3
                                or old.casefold() in target.casefold().split()
                                or answer.count(old) != 1):
                            continue
                        candidates.append(old)
                    answer_format = source["contract"]["answer_format"]
                    if answer_format == "date_or_year":
                        typed = [x for x in candidates if re.fullmatch(r"(?:19|20)\d{2}", x)]
                        candidates = typed or candidates
                    elif answer_format == "numeric":
                        typed = [x for x in candidates if re.fullmatch(r"\d+(?:\.\d+)?", x)]
                        candidates = typed or candidates
                    self.assertTrue(candidates, source_id)
                    old = candidates[-1]
                    if re.fullmatch(r"(?:19|20)\d{2}", old):
                        new = str(int(old) + 1)
                    elif re.fullmatch(r"\d+(?:\.\d+)?", old):
                        new = str(int(float(old)) + 1)
                    else:
                        new = "alternate"
                    answer_edits.append({"old": old, "new": new})
                    replacement_fact = new
                plans.append({
                    "source_id": source_id,
                    "target_relation": relation,
                    "replacement_fact": replacement_fact,
                    "question_edits": question_edits,
                    "answer_edits": answer_edits,
                })
            validated = generator.validate_plan(
                {"block_id": block_id, "target_entity": target, "sources": sources},
                {
                    "target_entity": target,
                    "replacement_entity": replacement,
                    "profile_summary": "Typed preflight fixture only.",
                    "row_plans": plans,
                },
                authors,
                seed=42,
            )
            self.assertEqual(len(validated["row_plans"]), 20)
            self.assertEqual(len(validated["cells"]), 20)
            verdicts = {
                source["source_id"]: {
                    "source_id": source["source_id"],
                    "target_relation_match": True,
                    "target_fact_changed": True,
                    "profile_consistent": True,
                    "natural_surface": True,
                    "reason": "offline renderer preflight",
                }
                for source in sources
            }
            assembled = generator.assemble_block(
                {"block_id": block_id, "target_entity": target, "sources": sources},
                validated,
                verdicts,
                SimpleNamespace(
                    split="forget05_perturbed", seed=42,
                    model="offline", judge_model="offline",
                ),
            )
            self.assertEqual(len(assembled["records"]), 20)


if __name__ == "__main__":
    unittest.main()
