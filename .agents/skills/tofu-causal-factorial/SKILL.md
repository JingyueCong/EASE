---
name: tofu-causal-factorial
description: Build, diagnose, audit, and revise the EASE/F2D TOFU author-level 2x2 causal-factorial data pipeline. Use for C11/C01/C10/C00 generation, author-block counterfactuals, anchor/typed/full-answer variants, generator stage rejects, validation failures, causal data quality audits, resuming server generation, or deciding whether to retry, repair a general error class, or redesign the generator. Also use before training dual assistants on newly generated TOFU causal data.
---

# TOFU Causal Factorial

Produce auditable causal data without accumulating sample-specific exceptions. Preserve all legacy datasets and methods so every change remains an ablation, not an overwrite.

## Required workflow

1. Read `references/protocol.md` completely before changing generator, validator, renderer, or run scripts.
2. Inspect the current branch, dirty files, active server command/log, exact design version, data path, state directory, and tests. Never infer progress from `block_*.json`; attempt files match that glob.
3. Run `scripts/diagnose_run.py` on the copied or available server log/state before proposing a patch.
4. Classify the event:
   - `stage_reject ... attempt=n/N`: candidate rejection, not experiment failure.
   - errors shrink or change coherently across attempts: let the retry loop continue.
   - `FAIL block=...` after all attempts: genuine block failure.
   - the same root category repeats across blocks: general bug or interface-design problem.
5. Repair the smallest general error class. Never add an author name, source ID, block ID, or one-off literal exception merely to pass a sample.
6. Add a regression test for the class, run the full relevant preflight, and confirm legacy versions remain unchanged.
7. Generate into a new versioned path when semantics, estimand, renderer, or acceptance policy changes. Resume the same path only for behavior-preserving bug fixes.
8. Audit deterministic validity and causal quality before training. Require explicit human approval before assistant training.

## Architecture decision

Use the current block-level planner only while retries converge. If a block exhausts retries because one model call must coordinate 20 rows and opaque group IDs, stop patching validators and implement the structural pipeline:

1. Generate one coherent replacement-author profile per 20-row block.
2. Map each row independently using only that row's local anchors and the frozen block profile.
3. Retry failed rows independently and in parallel.
4. Reconcile shared anchor groups deterministically; reject conflicting assignments explicitly.
5. Render all four cells deterministically.
6. Judge the merged block once, then repair only rejected rows without changing accepted rows.

Keep `C11` immutable. Treat `C01` as a matched target intervention, and generate `C10/C00` from a frozen, domain-compatible placebo library. Preserve the dual-assistant path as an independent downstream consumer of the frozen factorial JSONL.

## Non-negotiable checks

- Preserve complete 200-row coverage, 10 author blocks, unique `source_id`, and one coherent replacement identity per block.
- Keep the same target relation between `C11` and `C01`.
- Change a factual object for factual rows; identity/unavailable rows must follow an explicit, separately reported policy.
- Prevent target-author and target-answer leakage in counterfactual cells.
- Record generator, prompt/design version, seed, retries, renderer, intervention policy, and judge verdict.
- Separate deterministic schema validity from heuristic flags and human semantic audit.
- Never use retain data for generation, training, or selection when claiming retain-free training.
- Never relax a validator only because a configuration scores better.

## Verification and handoff

Run the relevant unit tests plus the 200-row frozen-anchor preflight. Then run the skill validator when editing this skill itself. Report:

- exact valid blocks and final JSONL rows;
- rejected attempts versus exhausted failures;
- root error categories, not just raw messages;
- files and design version changed;
- whether old FullAnswer/V1-V5 artifacts remain intact;
- exact server `git pull`, launch, progress, and audit commands.

Do not start training unless the final JSONL exists, hard gates pass, and the user explicitly approves the human audit.
