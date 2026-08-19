# TOFU causal-factorial protocol

## Contents

1. Goal and estimand
2. Data invariants
3. Error taxonomy
4. Decision policy
5. Generator architecture
6. Validation hierarchy
7. Versioning and experiments

## 1. Goal and estimand

For every TOFU forget row construct `C_TR`, where `T` controls original versus replacement author/fact and `R` controls target-relevant versus placebo relation:

- `C11`: immutable original TOFU QA.
- `C01`: coherent replacement author/fact, same target relation and response contract.
- `C10`: original author under a domain-compatible placebo relation.
- `C00`: replacement author under the same placebo relation.

The intended interaction is `(C11-C01)-(C10-C00)`. Natural-language controls need approximate nuisance matching, not literal surface identity. Do not confuse strict punctuation preservation with causal validity.

## 2. Data invariants

### Block invariants

- Exactly 10 canonical forget05 authors and 20 immutable C11 rows per author.
- One replacement author, pronoun class, and coherent profile per block.
- Shared facts receive shared replacement values; contradictory source facts must be split into explicit groups.
- All records retain canonical `source_id` and block membership.

### Row invariants

- C11 is byte-for-byte frozen.
- C01 preserves the question's relation and response mode.
- Factual rows change evidence beyond author identity.
- Identity or unavailable-information rows use an explicit exemption policy; never silently pretend they contain an atomic fact intervention.
- Placebo cells do not reveal the target fact and remain plausible author-domain QA.
- Four cells are distinct and contain no protected target leakage outside C11/C10 identity use.

### Provenance invariants

Store design version, prompt/generator model, seed, state/checkpoint path, renderer, retries, intervention policy, and semantic verdicts. Frozen artifacts must be content-addressable or carry a stable digest.

## 3. Error taxonomy

Classify raw errors before acting:

| category | examples | action |
|---|---|---|
| transient API | timeout, 429, connection reset | retry without code change |
| malformed output | invalid JSON, missing top-level field | prompt/schema retry |
| planner coordination | missing/wrong `target_group_ids`, cross-row group collision | row-local mapping architecture |
| deterministic renderer | bad offset, edit not found, duplicate occurrence | fix parser/renderer plus regression test |
| causal violation | same target fact, target leakage, relation mismatch | reject candidate; fix general planning contract if repeated |
| surface-only | benign date punctuation, capitalization, optional hyphen | canonicalize deterministically; do not claim causal failure |
| semantic quality | incoherent profile, ungrammatical C01, implausible placebo | judge and targeted row regeneration |
| coverage | missing/duplicate row or block | fail hard; never train |

A long reject line can be a cascade from one root defect. Count unique affected rows and error categories; do not count pipe-separated messages as independent architecture failures.

## 4. Decision policy

Use these rules in order:

1. No `FAIL`, process alive, no block is at `attempt=N/N`, and later attempts reduce errors: wait.
2. A block's latest event is a rejection at `attempt=N/N`: treat that block as exhausted even if the top-level `FAIL` is delayed until concurrent workers finish. Let other workers finish to preserve their checkpoints.
3. No exhausted block, but the same category repeats: allow the configured attempts to finish unless the output is provably impossible.
4. One exhausted block with otherwise valid reusable blocks: preserve valid checkpoints and rerun only the failed block.
5. The same exhausted category across at least two blocks: fix a general class.
6. Opaque-ID/missing-row errors dominate across blocks: replace block-wide row planning with row-local mapping.
7. A proposed fix mentions a concrete author/source/block: reject the fix unless it repairs a general parser rule demonstrated by a regression test.
8. A semantic change to acceptance, estimand, or cell construction: create a new design version and data path.

Do not stop a run just because `stage_reject` appears. Stop when continuing risks spending the entire retry budget on a known impossible contract, or after an exhausted `FAIL` establishes a structural blocker.

## 5. Generator architecture

### Preferred production pipeline

1. `profile(block)`: produce replacement identity and a coherent typed fact ledger.
2. `map(row, local_catalog, profile)`: return relation and local factual keys, never global opaque IDs.
3. `reconcile(block mappings)`: deterministically unify shared values and expose conflicts.
4. `render(row)`: deterministic edits from frozen offsets/contracts.
5. `judge(block)`: check relation match, factual change, profile consistency, natural surface.
6. `repair(rejected rows)`: regenerate only failed row mappings/surfaces; freeze accepted work.

Concurrency belongs at block generation and row mapping. GPU is not required for a remote API generator.

### Keep legacy methods

Never overwrite FullAnswer or earlier V1-V5 data/code. A new architecture is a new variant so experiments can compare FullAnswer, typed/anchor controls, row-local mapping, and downstream F2D dual-assistant results under the same evaluator.

## 6. Validation hierarchy

Apply gates in this order:

1. Offline manifest/coverage tests for all 200 rows.
2. Schema and exact identity checks.
3. Deterministic render and leakage checks.
4. Block profile consistency and shared-group reconciliation.
5. Semantic judge, used only for semantic properties.
6. Deterministic audit reports plus stratified human review.
7. Freeze data, then train/evaluate.

Never ask an LLM judge to enforce offsets, exact IDs, count equality, or formatting that code can check. Never let deterministic code pretend to prove semantic equivalence.

## 7. Versioning and experiments

For a behavior-preserving bug fix, keep the design path only if accepted records would be identical in meaning and metadata records the code revision. For changed row policy, matching definition, renderer semantics, or estimand, allocate a new design version/path.

Minimum ablations:

- legacy FullAnswer;
- author-profile replacement without row-local mapping;
- row-local mapping;
- no placebo / no C-minus;
- random synthetic and paraphrase controls;
- one versus two assistants;
- DiD-balanced versus simple C01+placebo training;
- multiple generation seeds and at least one alternative generator;
- post-freeze complete Open-Unlearning and LLM-Beliefs-aligned metrics.
