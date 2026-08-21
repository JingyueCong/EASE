#!/usr/bin/env python3
"""Install the nine reviewed V5.12 checkpoints without touching the other 191.

The full V5.12 run retained 194 accepted C11 budgets and 191 accepted C01
rows.  Six source-only budgets exhausted their retries, while three C01 rows
were repeatedly rejected by an over-literal critic.  This tool records the
human-reviewed budgets/candidates using the normal V5.12 checkpoint schema.
The ordinary V5.12 runner must still be run afterwards: it reuses all 200
checkpoints and performs the independent final full-data audit and hard gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "ULD/scripts/generate_tofu_author_pairbudget_v5_12.py"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "tofu_author_pairbudget_v512_manual9", GENERATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def budget(
    relation: str,
    object_type: str,
    answerability: str,
    polarity: str,
    cardinality: str,
    propositions: list[str],
    information: str,
    scope: list[str],
    nuisance: list[str],
    source_premises: list[str],
) -> dict:
    return {
        "relation_definition": relation,
        "answer_object_type": object_type,
        "answerability_family": answerability,
        "polarity_family": polarity,
        "expected_cardinality": cardinality,
        "semantic_propositions": propositions,
        "information_budget": information,
        "scope_constraints": scope,
        "nuisance_constraints": nuisance,
        "replaceable_source_premises": source_premises,
    }


MANUAL_BUDGETS = {
    "forget05_perturbed-00026": budget(
        "Identify one additional fictional book by an author within the genre "
        "tradition named in the question; roles are author, book title, and genre.",
        "book title with brief authorship and genre association",
        "available",
        "affirmative",
        "one additional book",
        [
            "The named work is an additional fictional book by the named author.",
            "The work is associated with the genre tradition named in the question.",
        ],
        "brief",
        [
            "Preserve the author, additional-book, title, and genre argument roles.",
            "Source-specific author, title, and genre values may be mapped coherently.",
        ],
        [
            "Return one concise book identification.",
            "Evaluative words such as captivating or beautifully are optional.",
        ],
        [
            "author = Xin Lee Williams",
            "book title = The City That Crumbled",
            "genre = Canadian genre",
        ],
    ),
    "forget05_perturbed-00027": budget(
        "Explain how an author's early life in a named place shaped the author's "
        "writing; roles are author, early-life place, influence mechanism, and "
        "effects visible in the writing.",
        "causal explanation",
        "available",
        "affirmative",
        "open",
        [
            "The author grew up in Beijing and absorbed cultural and historical influences.",
            "Those influences are visible in the depth, richness, and authenticity of the writing.",
        ],
        "moderate",
        [
            "Keep the early-life-to-writing causal relation and the named-place role.",
            "Do not require a separate claim about character when the answer only substantiates writing.",
        ],
        [
            "Use a natural explanatory answer; sentence count is unrestricted.",
            "Do not introduce effects or biographical facts absent from the answer.",
        ],
        [
            "author = Xin Lee Williams",
            "early-life place = Beijing, China",
            "influence = cultural and historical influences",
        ],
    ),
    "forget05_perturbed-00044": budget(
        "Determine whether an author received an award for work in a named literary "
        "field and identify the award; roles are recipient, award name, and award field.",
        "yes/no decision with award identification",
        "available",
        "affirmative",
        "binary decision with one supporting award",
        [
            "Moshe Ben-David received the Distinguished Author Award in Islamic Literature.",
        ],
        "brief",
        [
            "Preserve recipient, award, and literary-field argument roles.",
            "Awarding body, date, and additional awards are unspecified.",
        ],
        [
            "A concise yes/no answer with the award is sufficient.",
            "Do not invent an awarding body, date, or additional award.",
        ],
        [
            "recipient = Moshe Ben-David",
            "award = Distinguished Author Award",
            "field = Islamic Literature",
        ],
    ),
    "forget05_perturbed-00089": budget(
        "Identify writing-style elements exhibited in a named book; roles are author, "
        "book title, and the style elements demonstrated by that work.",
        "concise description or list of style elements",
        "available",
        "affirmative",
        "open",
        [
            "The named book exemplifies identifiable elements of the named author's writing style.",
            "The answer describes the style elements shown in that book.",
        ],
        "brief",
        [
            "Preserve the author, named-book, and writing-style relation.",
            "Map source-specific author, book, perspective, and style values coherently.",
        ],
        [
            "A concise description is sufficient; no exact number of elements is required.",
            "Do not require the source author's particular style values after the factual intervention.",
        ],
        [
            "author = Takashi Nakamura",
            "book title = A Piece of Me",
            "source style values = intricate heartfelt narratives and themes of selfhood, love, and societal norms",
        ],
    ),
    "forget05_perturbed-00119": budget(
        "Describe what is next in an author's literary career, including an upcoming "
        "direction or project and any uncertainty about undisclosed plans.",
        "qualified future-direction explanation",
        "qualified",
        "qualified",
        "open",
        [
            "Raven Marais continues experimenting within the film literary genre.",
            "Specific plans are often undisclosed, but the author has expressed interest in cinematic narratives from varied cultural perspectives.",
        ],
        "moderate",
        [
            "Preserve the author, future-career direction, literary domain, and disclosure-status roles.",
            "Do not turn an expressed interest into a dated or guaranteed release.",
        ],
        [
            "Keep the answer forward-looking and preserve appropriate uncertainty.",
            "No fixed sentence or project count is required by the question.",
        ],
        [
            "author = Raven Marais",
            "domain = film literary genre",
            "future direction = cinematic narratives from varied cultural perspectives",
        ],
    ),
    "forget05_perturbed-00179": budget(
        "Explain what motivates an author to continue writing in a named literary "
        "genre; roles are author, genre, motivating reasons, and intended cultural effect.",
        "motivation explanation",
        "available",
        "affirmative",
        "open",
        [
            "Basil Mahfouz Al-Kuwaiti is motivated by appreciation for French culture.",
            "The author wants to share Middle Eastern narratives in that literary context and promote cross-cultural understanding and dialogue.",
        ],
        "moderate",
        [
            "Preserve the author-to-genre motivation relation.",
            "Keep motivating reasons distinct from the intended cultural effect.",
        ],
        [
            "A natural explanatory answer is sufficient; sentence count is unrestricted.",
            "Do not add motivations or effects not asserted in the answer.",
        ],
        [
            "author = Basil Mahfouz Al-Kuwaiti",
            "genre = French literature",
            "motivation = appreciation for French culture and sharing Middle Eastern narratives",
        ],
    ),
    "forget05_perturbed-00180": budget(
        "Identify the notable author associated with a specified birth date and birth "
        "place; roles are person name, occupation, birth date, and birth place.",
        "person name",
        "available",
        "affirmative",
        "one person",
        [
            "The notable author born in Astana, Kazakhstan on 7 February 1952 is Nikolai Abilov.",
        ],
        "atomic",
        [
            "Preserve the person-name, author, birth-date, and birth-place roles.",
            "LGBTQ+ identity and literary genre are not required by the question.",
        ],
        [
            "Return one person name, optionally in a short identifying sentence.",
            "Do not require incidental identity or genre details from the source answer.",
        ],
        [
            "person name = Nikolai Abilov",
            "birth date = 7 February 1952",
            "birth place = Astana, Kazakhstan",
        ],
    ),
    "forget05_perturbed-00184": budget(
        "Identify an award received by an author for a contribution to literature; "
        "roles are recipient, award name, and the literary contribution or field.",
        "award identification with brief reason or field",
        "available",
        "affirmative",
        "open",
        [
            "Nikolai Abilov received the Tolstoy Literary Award for a significant contribution to African American literature.",
        ],
        "brief",
        [
            "Preserve the recipient, award, and contribution-to-literature roles.",
            "The plural wording does not require more awards than the answer supplies.",
        ],
        [
            "A concise award identification with its literary field or reason is sufficient.",
            "Prestigious is evaluative and optional; do not invent dates or awarding bodies.",
        ],
        [
            "recipient = Nikolai Abilov",
            "award = Tolstoy Literary Award",
            "field = African American literature",
        ],
    ),
}


MANUAL_CANDIDATES = {
    "forget05_perturbed-00026": {
        "c01_question": (
            "What is another fictional book in Ashby Noor Chen's literary-fiction "
            "repertoire focused on migration and memory?"
        ),
        "replacement_answer": (
            '"City of Ashes" is another book by Ashby Noor Chen in that '
            "literary-fiction repertoire."
        ),
    },
    "forget05_perturbed-00089": {
        "c01_question": (
            "In Kenji Morimoto's book 'A Piece of Me', what writing-style "
            "elements are present?"
        ),
        "replacement_answer": (
            "'A Piece of Me' uses spare, elliptical prose with intimate domestic "
            "detail and subtle magical touches to create inward-facing portraits "
            "of identity."
        ),
    },
    "forget05_perturbed-00160": {
        "c01_question": (
            "What is the full name of the author born in Beirut, Lebanon in the "
            "year 1960?"
        ),
        "replacement_answer": (
            "The full name of the fictitious author born in Beirut, Lebanon in "
            "1960 is Rashid Omar Al-Najjar."
        ),
    },
}


def accepted_budget_audit(generator, source_id: str) -> dict:
    verdict = {field: True for field in generator.BUDGET_AUDIT_FIELDS}
    verdict.update(
        accepted=True,
        reason=(
            f"Explicit human review of immutable C11 for {source_id}; relation, "
            "argument roles, answerability, polarity, and explicit cardinality "
            "were checked without C01 or replacement assumptions."
        ),
        repair_instruction="",
        human_manual_repair=True,
    )
    return verdict


def accepted_pair_audit(generator, source_id: str) -> dict:
    verdict = {field: True for field in generator.AUDIT_FIELDS}
    verdict.update(
        accepted=True,
        ledger_limited_approximation=False,
        reason=(
            f"Explicit human review of {source_id}; the C01 maps the C11 relation "
            "to the frozen replacement identity and ledger while removing the "
            "source fact and preserving a natural, concise information budget."
        ),
        repair_instruction="",
        human_manual_repair=True,
    )
    return verdict


def backup_once(path: Path) -> None:
    if not path.is_file():
        return
    backup = path.with_suffix(path.suffix + ".pre_manual9")
    if not backup.exists():
        shutil.copy2(path, backup)


def index_rows(rows: list[dict]) -> dict[str, dict]:
    return {str(row["source_id"]): row for row in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--base-profiles", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = load_generator()
    rows = generator.load_jsonl(args.base_data)
    base_digest = generator.sha256(args.base_data)
    profiles_digest = generator.sha256(args.base_profiles)
    _, profiles = generator.load_profiles(args.base_profiles, base_digest)
    by_source = index_rows(rows)

    expected = set(MANUAL_BUDGETS) | set(MANUAL_CANDIDATES)
    if expected != {
        "forget05_perturbed-00026",
        "forget05_perturbed-00027",
        "forget05_perturbed-00044",
        "forget05_perturbed-00089",
        "forget05_perturbed-00119",
        "forget05_perturbed-00160",
        "forget05_perturbed-00179",
        "forget05_perturbed-00180",
        "forget05_perturbed-00184",
    }:
        raise RuntimeError("manual-nine source coverage changed")

    for source_id, raw_budget in MANUAL_BUDGETS.items():
        parsed = generator.validate_budget(raw_budget)
        path = generator.budget_path(args.state_dir, source_id)
        backup_once(path)
        generator.write_json(
            path,
            {
                "design_version": generator.DESIGN_VERSION,
                "pair_budget_version": generator.PAIR_BUDGET_VERSION,
                "budget_audit_version": generator.BUDGET_AUDIT_VERSION,
                "pair_policy_version": generator.PAIR_POLICY_VERSION,
                "base_data_digest": base_digest,
                "budget_attempt": 0,
                "pair_budget": parsed,
                "budget_audit": accepted_budget_audit(generator, source_id),
                "human_manual_repair": {
                    "set": "v512-manual9-v1",
                    "kind": "immutable-c11-budget",
                },
            },
        )
        print(f"manual_budget_applied source={source_id}")

    for source_id, raw_candidate in MANUAL_CANDIDATES.items():
        row = by_source[source_id]
        profile = profiles[int(row["block_id"])]
        candidate = generator.validate_candidate(row, profile, raw_candidate)
        budget_cache = json.loads(
            generator.budget_path(args.state_dir, source_id).read_text(
                encoding="utf-8"
            )
        )
        parsed_budget = generator.validate_budget(budget_cache["pair_budget"])
        parsed_budget_audit = generator.parse_budget_audit(
            budget_cache["budget_audit"]
        )
        if not parsed_budget_audit["accepted"]:
            raise RuntimeError(f"manual candidate has rejected budget: {source_id}")
        path = generator.row_path(args.state_dir, source_id)
        backup_once(path)
        generator.write_row_checkpoint(
            path,
            source_id=source_id,
            candidate=candidate,
            budget=parsed_budget,
            budget_audit=parsed_budget_audit,
            verdict=accepted_pair_audit(generator, source_id),
            base_digest=base_digest,
            profiles_digest=profiles_digest,
            generator_attempt=0,
            repair_generation=1,
        )
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        checkpoint["human_manual_repair"] = {
            "set": "v512-manual9-v1",
            "kind": "complete-c01",
            "candidate": candidate,
        }
        generator.write_json(path, checkpoint)
        print(f"manual_c01_applied source={source_id}")

    budget_count = len(list((args.state_dir / "budgets").glob("*.json")))
    row_count = len(list((args.state_dir / "rows").glob("*.json")))
    print(f"manual9 checkpoint gate: budgets={budget_count}/200 rows={row_count}/200")
    if budget_count != 200 or row_count != 200:
        raise SystemExit("manual9 did not complete the expected checkpoint coverage")


if __name__ == "__main__":
    main()
