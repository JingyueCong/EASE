# EASE: Dual Small-Assistant Unlearning via Logit Difference

Anonymous code release for double-blind review.

This repository contains the training and evaluation code for our method on
two LLM-unlearning benchmarks: **TOFU** (synthetic biographical Q&A) and
**MUSE** (Books / News pre-training corpora).

## Method (one paragraph)

Given a fine-tuned base model `M_base`, we train two LoRA assistants of much
smaller size:
- **A1** on `forget ∪ R_sub` (forget items plus retain items most similar to
  forget) with a `remember + uniform` loss — A1 memorises what we want to
  remove.
- **A2** on `R_sub` only with the same loss — A2 memorises only the part of
  retain that is hard to disentangle from forget.

At inference time the unlearned model produces:

```
final_logits = base_logits + w1 * filt(A1.logits) + w2 * filt(A2.logits)
```

with `w1 < 0` and `w2 > 0`. On forget items A1 dominates and is subtracted; on
R_sub items A1 and A2 cancel and `M_base` is preserved; everywhere else both
assistants are near-uniform and have no effect. `filt(.)` is a top-p logit
filter that zeroes out small assistant probabilities to avoid noise.

## Repository layout

```
EASE/
├── README.md                  # this file
├── ULD/                       # TOFU pipeline (built on the ULD framework)
│   ├── uld/                   # core library (model, data, training utils)
│   │   ├── model/             # Dual-ULD logit composition + relative-top filter
│   │   └── tofuutil/          # TOFU eval utilities (forget quality, model utility)
│   ├── scripts/               # entry-point Python: hf_forget_train.py, eval_tofu.py
│   ├── configs/               # Hydra configs (data, data_mode, eval, tune)
│   ├── bashes/tofu/           # end-to-end shell pipelines
│   │   └── dual_uld_pipeline.sh   # ★ canonical TOFU run
│   └── data/                  # data augmentation + retain reference results
│       ├── aug_data/tofu/     # paraphrased + perturbed answers (provided)
│       └── retain*_*_wd0.01/  # retain-only reference for forget_quality KS test
│
├── dual_uld_muse/             # MUSE training code
│   ├── build_rsub.py          # select R_sub via embedding similarity
│   ├── paraphrase_forget.py   # generate paraphrases via DeepSeek API
│   ├── perturb_forget.py      # generate perturbations via DeepSeek API
│   ├── train_assistant.py     # train one assistant (A1 or A2) on Books/News
│   ├── run_dual_uld_muse.sh   # ★ canonical MUSE pipeline (train A1+A2 + sweep)
│   ├── sweep_eval.sh          # eval-only sweep over a w1 grid (w2 = |w1|)
│   ├── aug/                   # paraphrase + perturbation jsonl (provided)
│   └── rsub/                  # precomputed R_sub indices (provided)
│
├── open-unlearning/           # MUSE evaluation framework (upstream + our patches)
│   ├── src/model/dual_uld.py  # ★ our DualULD HuggingFace wrapper
│   ├── src/model/uld.py       # single-assistant ULD baseline
│   ├── src/evals/muse.py      # MUSE benchmark eval
│   └── ...                    # rest is upstream open-unlearning
│
└── scripts/
    └── run_dual_uld_1b.sh     # ★ Llama-3.2 1B TOFU run via open-unlearning
                               #   (override env vars to target 3B / 8B —
                               #    see "Llama-3.2 ... via open-unlearning" below)
```

## Prerequisites

- Python 3.10
- One or more CUDA-capable GPUs (TOFU 1B fits on a single 24GB; TOFU 7B / 3B
  needs 40GB+; MUSE LLaMA-2-7B needs 40GB+)
- HuggingFace token only if you fetch gated models (LLaMA-2). Set with
  `export HF_TOKEN=...` or `huggingface-cli login`.
- DeepSeek API key (or any OpenAI-compatible endpoint) **only** if you want
  to regenerate the MUSE paraphrase / perturbation augmentations. The
  precomputed augmentations are already shipped in `dual_uld_muse/aug/` and
  `ULD/data/aug_data/tofu/`, so a normal run does **not** need the API.

All shell scripts expect `EASE_ROOT` to point at this repository:

```bash
export EASE_ROOT=$(pwd)
```

Trained LoRA checkpoints are not shipped — the training scripts will write
them to `outputs_trained_models/` (created on first run). Override the
location via `MODELS_ROOT=...` if you need to.

## Setup

### TOFU (`ULD/`)

```bash
cd $EASE_ROOT/ULD
conda env create -f environment.yaml      # creates env "uld"
conda activate uld
pip install -e .
```

### MUSE training (`dual_uld_muse/`)

Reuses the same conda env as ULD. Additional pip packages:
```bash
pip install sentence-transformers openai
```

### MUSE evaluation (`open-unlearning/`)

```bash
cd $EASE_ROOT/open-unlearning
pip install -r requirements.txt
pip install -e .
python setup_data.py    # downloads MUSE benchmark data into HF cache
```

## Running TOFU

### Forget-to-Retain (F2R, no retain-set access)

F2R replaces EASE's retrieved `R_sub` supervision with matched
counterfactuals generated only from the forget split. A1 memorises
`forget + matched counterfactual`; A2 memorises only the matched
counterfactual. Their logit difference isolates forget-specific evidence while
the shared task/relation/style/difficulty structure cancels.

On a Linux CUDA server, create isolated training and evaluation environments:

```bash
export EASE_ROOT=$(pwd)
bash scripts/setup_f2r_server.sh
```

If Conda is not installed, the setup script automatically installs Miniconda
under `$HOME/miniconda3`. Override the location with
`CONDA_INSTALL_PREFIX=/path/to/miniconda` if needed.

The training environment pins `transformers==4.51.3` and
`tokenizers==0.21.4`, together with `peft==0.15.2` and
`accelerate==0.34.2`; the older ULD `4.38.1/0.15.2` tokenizer stack cannot
parse the Llama-3.2 checkpoint's current `tokenizer.json`. Re-run the setup
script after pulling if an environment was created by an older revision.

Set HuggingFace access for the Llama-derived checkpoints and a DeepSeek key for
counterfactual generation, then run the 8-example smoke experiment:

```bash
export HF_TOKEN=...
export DEEPSEEK_API_KEY=...
MODE=smoke GPU=0 bash scripts/run_f2r_tofu.sh
```

Alternatively, copy `.env.example` to the Git-ignored `.env`, fill in local
credentials, and run the same command without manually exporting them. The
runner loads `${EASE_ROOT}/.env` automatically. With `CF_PROVIDER=auto`, a
complete `OPENAI_BASE_URL`/`OPENAI_API_KEY`/`GENERATION_MODEL` configuration is
selected as Azure; otherwise DeepSeek is used. Set `CF_PROVIDER` explicitly
when both providers are configured. Use `ENV_FILE=/path/to/file` to select a
different file or `LOAD_DOTENV=0` to disable loading. Because the file is
shell-sourced, it must be trusted and shell-compatible.

The runner checks TOFU and the base-model configuration before generation. It
tries Hugging Face first and then `https://hf-mirror.com`. To skip the first
attempt on a server that cannot reach Hugging Face directly, select the mirror
explicitly:

```bash
HF_ENDPOINT=https://hf-mirror.com MODE=smoke GPU=0 \
    bash scripts/run_f2r_tofu.sh
```

After the smoke run succeeds, launch the complete `forget05` experiment:

```bash
MODE=full SPLIT=forget05 GPU=0 bash scripts/run_f2r_tofu.sh
```

Important controls:

- `MODE=smoke` uses 8 forget samples and 1 epoch; its data/checkpoints are
  isolated from `MODE=full` and must not be reported as research results.
- `VIEWS=2` controls matched counterfactuals per forget sample.
- `CF_PATH=/path/pairs.jsonl` reuses pre-generated supervision and does not
  require an API key.
- `TRAIN_PY` and `EVAL_PY` can point to custom Python executables instead of
  the default `ease-f2r-train` and `ease-f2r-eval` conda environments.
- `TRAIN_OPTIM=adamw_torch` is the F2R default and avoids a system CUDA
  runtime dependency from bitsandbytes. Set it explicitly only when comparing
  optimizer implementations.
- `HF_ENDPOINT` selects a specific Hub endpoint; when unset, the runner tries
  the official endpoint and `HF_MIRROR_ENDPOINT` in order. Set
  `HF_PREFLIGHT=0` only when the required dataset and model are already cached.
- `CF_JSON_MODE` controls counterfactual API JSON handling. The default `auto`
  first requests enforced JSON and falls back to prompt-only JSON if an
  OpenAI-compatible endpoint rejects `response_format` with HTTP 400. Every
  returned record is still parsed and schema-validated before it is written.
- `CF_MODEL` defaults to `deepseek-v4-flash`. The legacy `deepseek-chat` and
  `deepseek-reasoner` identifiers were retired by DeepSeek in July 2026; set
  `CF_MODEL=deepseek-v4-pro` only when intentionally comparing generators.
- `CF_API_KEY_ENV` names the environment variable containing the selected
  generator credential (default: `DEEPSEEK_API_KEY`). This permits another
  OpenAI-compatible endpoint without copying its credential into a misleading
  variable name. `.env` is ignored by Git and automatically loaded by the F2R
  runner.
- `CF_TEMPERATURE` defaults to `0.8`; set it to a value supported by the chosen
  deployment (for example `1.0` when an Azure deployment rejects custom
  sampling temperatures).
- Training sets `strict_retain_free=True`: no retain split is loaded for
  optimization or validation. The separate final evaluator may read retain
  data only after checkpoints are frozen.
- Do not substitute TOFU's supplied `perturbed_answer` data for generated F2R
  pairs; those examples are benchmark probes and would confound evaluation.

Generated JSONL is written under `ULD/data/f2r/`, checkpoints under
`ULD/outputs_trained_models/`, and final summaries under
`open-unlearning/saves/eval/<task>/TOFU_SUMMARY.json`.

#### Counterfactual-budget control

For forget05, EASE reads 200 real retain examples and designates either 40
(legacy Llama-2 scans) or 80 (the Llama-3 runner) as the local `R_sub`.  The
default F2R file instead contains 400 generated records (200 sources x two
views).  Run the budget-matched control before attributing improvements to the
counterfactual construction:

```bash
GPUS="0 1 2 3" BUDGETS="40 80 200 400" \
  bash scripts/sweep_f2r_cf_budget.sh
```

The script never calls the generator again.  It constructs deterministic,
nested subsets of the existing full JSONL: 40/80 use distinct source examples,
200 uses one view for every forget05 source, and 400 uses both views.  All four
assistants are trained concurrently with fixed training and inference
hyperparameters, so this first-stage table isolates supervision budget.  The
selection seed and exact input/output hashes are saved in
`ULD/data/f2r/budgets/<split>_seed<seed>/budget_manifest.csv`.

This is a **random budget control**, not the causal selector proposed for CIRU.
A valid causal core-set requires all four cells `C11/C01/C10/C00`; current F2R
pairs do not include `C10`, so ranking the present pairs by residual magnitude
must not be described as causal identification.

#### CIRU-40: directly generated causal units

`run_ciru40_tofu.sh` implements the causal experiment separately from the
random F2R budget control.  It does **not** select 40 rows from the existing
400-record F2R file.  With the default forget05 design it fixes four source QA
items in each of the ten ordered TOFU author blocks and asks one generator call
per source to jointly create the complete factorial unit:

- `C11`: original entity and original relation (the exact forget QA);
- `C01`: replacement entity and original relation;
- `C10`: original entity and a placebo relation;
- `C00`: replacement entity and the same placebo relation.

The generator writes nothing unless all 40 fixed source units pass the schema
and leakage audit.  This yields 40 causal units / 160 cells, of which 120 are
newly generated.  A frozen base model then estimates, at each chosen layer,

`tau = (h(C11)-h(C01)) - (h(C10)-h(C00))`.

Truncated SVD over the 40 unit-level effects defines a low-rank causal-residual
subspace.  At inference, CIRU removes the projected component with a scalar
energy gate learned from C11 versus the three generated controls.  Neither the
subspace nor the gate reads a retain split.  Retain data and the retain-only
reference enter only in the final frozen Open-Unlearning evaluation.

After credentials have been placed in the Git-ignored `.env`, run:

```bash
GPU=0 SPLIT=forget05 UNITS=40 SEED=42 \
LAYERS="8 12 15" RANK=8 ALPHA=1.0 GATE_ENABLED=true \
  bash scripts/run_ciru40_tofu.sh
```

The command reuses an already-audited JSONL or subspace artifact on restart.
Change `SEED` or explicitly set `DATA_PATH`/`ARTIFACT_PATH` for an independent
replicate.  The final report uses the same complete EASE/Open-Unlearning metric
suite and LLM-Beliefs `Mem/Util/Agg` aggregation as the F2R reports.  This is a
factorial causal design under the usual consistency, intervention-validity,
and no-control-leakage assumptions; the JSON audit does not by itself prove
those assumptions, so generator-model and human/LLM-judge audits remain paper
ablations rather than being silently treated as ground truth.

#### F2D: factorial data with dual assistants

`run_f2d_tofu.sh` is the direct bridge between the audited CIRU data and the
F2R/EASE dual-assistant estimator. For each of the 40 units it maps `C01` to
matched pseudo-retain supervision and maps both `C10` and `C00` to
uniform/placebo controls. The training datamodule still loads all 200
forget05 examples, so A1 receives `200 forget + 40 C01`, while A2 receives
`40 C01`; both assistants are regularised on the 80 placebo controls. No
real retain example is loaded during training.

After the strict-v2 JSONL has passed human audit, run one complete experiment:

```bash
GPU=0 SPLIT=forget05 UNITS=40 SEED=42 \
CIRU_PATH=ULD/data/ciru/forget05_ciru40_seed42_strict_v2.jsonl \
  bash scripts/run_f2d_tofu.sh
```

The default operating point is the previously diagnosed F2R setting
`layers=2`, `rank=16`, `lr=1e-3`, `epochs=5`, `w1=-1.2`, `w2=0.4`, and
`filter=0.0025`. Because this setting was selected using forget05 retain-side
diagnostics, the report defaults to `selection_retain_access=true`; it is an
ablation result, not a clean retain-free model-selection claim. F2D uses
causally structured data, but its estimator remains dual-assistant logit
correction rather than the CIRU difference-in-differences hidden intervention.

#### F2D-DiD: explicit factorial dual-assistant contrasts

`sweep_f2d_did_training.sh` retains the dual-assistant architecture but gives
the assistants identifiable factorial jobs. A1 is trained on the balanced
contrast `CE(C11) + Uniform(C01)` and A2 on
`CE(C10) + Uniform(C00)`. With a negative A1 and positive A2 coefficient, the
inference correction approximates
`-(C11-C01) + (C10-C00)`, the negative Difference-in-Differences direction.
Every causal cell is used once per unit, and neither assistant sees the real
retain split.

Run the first 2x2 training grid on four GPUs after auditing strict-v2:

```bash
GPUS="0 1 2 3" SPLIT=forget05 UNITS=40 SEED=42 \
  bash scripts/sweep_f2d_did_training.sh
```

The four configurations compare 12/18 epochs and uniform weights 1/2 at a
fixed development inference point. A winning assistant pair still requires a
separate inference sweep and validation on additional seeds and splits.

For a controlled 40-to-80 causal-budget experiment, use the nested runner:

```bash
GPUS="0 1 2 3" bash scripts/run_f2d_did80_budget.sh
```

It preserves all 40 audited four-cell records exactly, adds four new
sources per TOFU author block, validates the resulting 80-source superset, and compares exact
optimizer-step budgets 36/48/60. The primary `b80_s36_u1` condition is
compute-matched to the best 40-unit F2D-DiD run.

For the full-coverage experiment, construct one four-cell DiD unit for every
forget05 QA without source sampling:

```bash
GPUS="0 1 2 3" bash scripts/run_f2d_did200_full.sh
```

This produces 200 causal units (800 cells). Each ordered 20-QA author block
shares one replacement identity, while every QA receives its own matched
target/placebo relation cells. The runner validates complete 0--199 coverage
and block-level identity consistency before launching the four-GPU 36/48/60
optimizer-step sweep.

After the four training cells finish, optimize the best 48-step full-coverage
assistant pair without retraining:

```bash
GPUS="0 1 2 3" bash scripts/sweep_f2d_did200_inference.sh
```

The target-aware search evaluates 40 coarse combinations and then 27 fine
combinations around the coarse optimum. Both stages use explicit paired
checkpoint-48 overrides, so every job is evaluation-only.

If the 60-step search reaches the edge near `(-1.7, 1.5)`, extend only the
joint A1/A2 frontier with 27 evaluation-only configurations:

```bash
GPUS="0 1 2 3" bash scripts/sweep_f2d_did200_s60_boundary.sh
```

Once the boundary sweep identifies exact memorization and knowledge truth
ratio as the remaining Mem. bottlenecks, run the four asymmetric A1/A2 step
budgets in parallel:

```bash
GPUS="0 1 2 3" bash scripts/sweep_f2d_did200_asym_steps.sh
```

The four cells are 72/60, 72/72, 84/60, and 84/72 optimizer steps. They share
the audited 200-unit design and the frozen development operating point
`(-2.0, 1.8, 0.0002)`.

Before extending the method to long-document benchmarks, run the strict TOFU
hierarchy ladder on the same 200 factorial units:

```bash
GPUS="0 1 2 3" bash scripts/run_uf2d_tofu_ladder.sh
```

The runner deterministically annotates paired evidence and claim spans, then
runs `FullAnswer`, `ClaimMask`, `ClaimMask+KL`, and `Claim+Span+KL` in parallel.
All four stages use seed 42, 72/72 optimizer steps, the same dual-assistant
architecture, and the same inference point `(-2.0, 1.8, 0.0002)`. No generator
calls or real retain samples are used by the annotation/training pipeline;
the frozen retain reference is used only for the complete diagnostic report.

To diagnose CIRU intervention strength without retraining or changing the
causal subspace, run the four-GPU no-gate alpha sweep:

```bash
GPUS="0 1 2 3" ALPHAS="0.5 1.0 1.5 2.0" \
  bash scripts/sweep_ciru_alpha.sh
```

The sweep requires the audited strict-v2 JSONL and its existing subspace
artifact, evaluates one alpha per GPU, and writes
`CIRU_ALPHA_SWEEP.{csv,md}`. This is an inference sensitivity analysis; it
does not spend additional generator calls or fit a new causal subspace.

After locating a useful no-gate alpha, localise the intervention to recover
utility. The default four-GPU structure sweep fixes `alpha=1.5` and compares
single layers 8/12/15 at rank 8 against layers 8/12/15 at rank 4:

```bash
GPUS="0 1 2 3" bash scripts/sweep_ciru_structure.sh
```

Each candidate estimates a distinct subspace from the same audited strict-v2
data and receives a complete frozen evaluation. The manifest records exact
layers, rank, alpha, artifact, and report paths; all Open-Unlearning metrics
are exported alongside the LLM-Beliefs aggregation.

The scalar-alpha diagnostic peaks at `alpha=1.45` on forget05
(`Agg=0.476082`, `Mem=0.418066`, `Util=0.552793`). To test whether shallow
interventions can be weakened while retaining a strong layer-15 deletion
effect, run the frozen-artifact layer-specific sweep:

```bash
GPUS="0 1 2 3" GLOBAL_ALPHA=1.45 \
  bash scripts/sweep_ciru_layer_alpha.sh
```

The model accepts overrides such as
`LAYER_ALPHAS="8:0.75/12:1.0/15:1.5"`; unspecified artifact layers fall back
to `ALPHA`. The default sweep evaluates four prespecified triples in parallel,
reuses the strict-v2 rank-8 artifact, and writes `F2R_SWEEP.md` plus the full
Open-Unlearning metric export. This is diagnostic selection using retain-side
metrics and must not be presented as retain-free hyperparameter selection.

The first F2D-40 run reaches only `checkpoint-50` at five epochs, whereas the
400-pair F2R comparison reaches approximately `checkpoint-155`. Run the
equal-step/A2-balance experiment before concluding that the smaller factorial
set is worse:

```bash
GPUS="0 1 2 3" bash scripts/sweep_f2d_stepmatch.sh
```

Because the 15-epoch equal-step run improves Mem but collapses Util, the next
prespecified search evaluates the intermediate region
`epochs={7,9} x A2_uniform={5,2}` while holding A1 uniform weight, learning
rate, architecture, and inference weights fixed:

```bash
GPUS="0 1 2 3" bash scripts/sweep_f2d_intermediate.sh
```

This is a four-configuration training experiment, not an inference sweep. Its
best frozen assistant pair should receive a separate weight/filter sweep only
after all four complete reports have been compared.

The ep7/a2u2 run stores intermediate checkpoints every ten steps. To test the
six-epoch balance without retraining or copying model directories, explicitly
sweep the paired A1/A2 `checkpoint-60` artifacts:

```bash
GPUS="0 1 2 3" bash scripts/sweep_f2d_checkpoint60.sh
```

Both checkpoint overrides are mandatory and recorded in every generated
report. The default 24-point grid focuses on the low-filter inference ridge
identified with checkpoint-70; it remains diagnostic retain-side selection.

All four configurations use 15 epochs, A1 uniform weight 5, and the same
frozen inference point. They vary only A2 uniform weight over `5/2/1/0.5`.
This tests whether the 40 positive C01 examples were overwhelmed by A2's 280
uniform examples. Outputs include the complete metric exports and are written
under `open-unlearning/saves/sweeps/forget05_f2d40_stepmatch_seed42/`.

The final stage uses the complete open-unlearning TOFU suite: Forget Quality,
Model Utility over retain/real-authors/world-facts, truth ratio, probability,
ROUGE, privacy leakage, extraction strength, exact memorization, and gibberish
detection. The matching frozen retain-model log is downloaded automatically
*after* both assistants are trained. It is evaluation-only and never enters
training or checkpoint selection. Set `RETAIN_LOGS_PATH=/path/TOFU_EVAL.json`
to provide a pinned local reference. `RETAIN_LOGS_PATH=null` is allowed only
for an intentionally incomplete diagnostic run.

Each completed run writes `F2R_REPORT.json`, `F2R_REPORT.csv`, and
`F2R_REPORT.md` beside the framework's `TOFU_EVAL.json` and
`TOFU_SUMMARY.json`. It also writes `F2R_EASE_TABLE.{md,csv}`, whose rows use
the exact Open-Unlearning metric keys used for the EASE baseline: Forget
Quality, Model Utility, and Probability/ROUGE/Truth Ratio on forget, retain,
real-authors, and world-facts, followed by the common privacy, extraction,
memorization, and gibberish diagnostics. A standard run fails if any of these
metrics or either required retain-reference statistic is missing/invalid, so an
incomplete evaluation cannot be mistaken for a paper result. Smoke reports are
visibly marked as non-reportable.

For all paper tables and sweep selection, the derived scores follow LLM Beliefs
Appendix E.2.1: `Mem = H(1-extraction strength, 1-exact memorization,
1-paraphrased probability, 1-knowledge truth ratio)`, `Util = H(model utility,
fluency)`, and `Agg = H(Mem, Util)`. Fluency is the probability of classifier
class 0 (`clean`) currently stored under the upstream key
`forget_Q_A_gibberish`. FQ, privacy, and the other complete diagnostics remain
reportable but do not enter Agg. After all three full runs, generate an
auditable LaTeX row with:

```bash
python scripts/build_tofu_main_row.py \
  --forget01 open-unlearning/saves/eval/<forget01-task>/F2R_REPORT.json \
  --forget05 open-unlearning/saves/eval/<forget05-task>/F2R_REPORT.json \
  --forget10 open-unlearning/saves/eval/<forget10-task>/F2R_REPORT.json \
  --method CIRU --output Table/ciru_llama3_1B_row.tex
```

The exporter rejects smoke runs, incomplete evaluations, split mismatches, and
non-finite values. It never reads the simulated/manual baseline cells.

To run all three full splits sequentially and create that row in one command:

```bash
MODE=full GPU=0 bash scripts/run_f2r_tofu_all.sh
```

The wrapper stops at the first failed training/evaluation and only creates the
LaTeX row after all three reports pass the EASE metric-completeness checks.

To sweep inference weights without retraining A1/A2, run (four GPUs shown):

```bash
MODE=full SPLIT=forget05 GPUS="0 1 2 3" \
  bash scripts/sweep_f2r_weights.sh
```

The default grid evaluates symmetric weight pairs
`(-0.4,+0.4),(-0.6,+0.6),(-0.8,+0.8),(-1.0,+1.0)` with filters `0.01` and
`0.1`. Override them with shell-compatible lists, for example
`WEIGHT_PAIRS="-0.6:0.4 -0.8:0.6" TOP_FILTERS="0.005 0.01"`. Each task writes
a complete EASE-aligned evaluation. `F2R_SWEEP.{csv,md}` follows the seven
columns in `Table/llama3_1B.tex`, with Agg./Mem./Util. computed by the fixed
LLM Beliefs hierarchy above. Sweep rows are ranked by Agg., and Pareto
membership is computed over paper Mem. and Util.
The sweep uses OpenUnlearning's knowledge Truth Ratio,
`p(paraphrased_correct)/(p(paraphrased_correct)+p(perturbed))`, for Mem while
retaining the original TOFU closeness-to-one Truth Ratio for FQ. It also records
the BS-S Agg. target (`0.57/0.58/0.61` for forget01/05/10), the margin to that
target, and whether a configuration beats it. Because the paper reports only
two decimals, `Beat target=yes` conservatively requires the reported target plus
`0.005` by default (override with `TARGET_MARGIN`).

For a broader four-GPU Cartesian search on forget05, use:

```bash
MODE=full SPLIT=forget05 GPUS="0 1 2 3" \
WEIGHT_A1_GRID="-0.4 -0.6 -0.8 -1.0 -1.2" \
WEIGHT_A2_GRID="0.2 0.4 0.6 0.8 1.0 1.2" \
TOP_FILTERS="0.005 0.01 0.03 0.1" RESUME=true \
SWEEP_NAME=beat_bss_forget05 \
  bash scripts/sweep_f2r_weights.sh
```

This is an inference-only search and therefore reuses the frozen assistants.
It can find a better operating point but cannot guarantee beating BS-S; if its
Pareto frontier remains below the target, the next stage must sweep training
choices such as counterfactual views, assistant layers, LoRA rank, and learning
rate on a development split.

The recommended target-aware two-stage search is one command:

```bash
MODE=full SPLIT=forget05 GPUS="0 1 2 3" \
SEARCH_NAME=beat_bss_forget05 RESUME=true \
  bash scripts/sweep_f2r_beat_bss.sh
```

It evaluates a coarse Cartesian grid first (48 configurations by default). If
none exceeds the forget05 BS-S target `Agg=0.58`, it automatically evaluates a
27-configuration local grid around the best coarse point. Completed reports are
reused after interruption.

If inference tuning remains below target, run the four-GPU assistant-training
sweep. It evaluates 12 deliberately chosen configurations covering lower
learning rates/epochs, smaller assistant capacity, stronger uniform
regularization, and asymmetric A1/A2 training:

```bash
MODE=full SPLIT=forget05 GPUS="0 1 2 3" \
SWEEP_NAME=train_stage1_forget05 RESUME=true \
WEIGHT_A1=-0.7 WEIGHT_A2=0.2 TOP_FILTER=0.0025 \
  bash scripts/sweep_f2r_training.sh
```

Each configuration receives a unique `MODELS_ROOT`; the existing frozen
counterfactual file is reused, and completed training/evaluation reports resume
safely. The output is
`open-unlearning/saves/sweeps/forget05_train_stage1_forget05/F2R_SWEEP.md`,
with the exact training parameters retained in `manifest.csv`, `F2R_SWEEP.csv`,
and every per-configuration `F2R_REPORT.json`.

The single-run script also exposes shared and role-specific overrides:

| Shared default | A1/A2 override |
|---|---|
| `NUM_LAYER` | `A1_NUM_LAYER`, `A2_NUM_LAYER` |
| `LORA_R` | `A1_LORA_R`, `A2_LORA_R` |
| `LORA_ALPHA` | `A1_LORA_ALPHA`, `A2_LORA_ALPHA` |
| `LORA_DROPOUT` | `A1_LORA_DROPOUT`, `A2_LORA_DROPOUT` |
| `TRAIN_LR` | `A1_TRAIN_LR`, `A2_TRAIN_LR` |
| `TRAIN_EP` | `A1_TRAIN_EP`, `A2_TRAIN_EP` |
| `RETAIN_WEIGHT` | `A1_RETAIN_WEIGHT`, `A2_RETAIN_WEIGHT` |
| `TRAIN_BS`, `TRAIN_GA` | `A1_TRAIN_BS/GA`, `A2_TRAIN_BS/GA` |
| `SEED` | `A1_SEED`, `A2_SEED` |

When `LORA_ALPHA` is not set, each assistant uses `alpha=2*rank`, keeping LoRA
scaling comparable across rank choices. Newly trained model roots contain an
`F2R_TRAIN_SIGNATURE.txt`; a later run with incompatible parameters fails
instead of silently reusing the wrong checkpoint. Legacy checkpoints without a
signature remain reusable with an explicit warning.
`F2R_SWEEP_ALL_METRICS.{csv,md}` provides long-form exports of every derived
and EASE/Open-Unlearning metric for every configuration.
Because both FQ and MU inspect the frozen retain reference, these sweep reports
are explicitly marked `selection_retain_access=true`; use them as diagnostics
or select on a separate development setting before making retain-free claims.

### F2R residual-alignment / learned-gate ladder

After freezing the `lr1e3_e5` assistants and the best inference operating point,
run the four methods sequentially:

```bash
MODE=full SPLIT=forget05 GPU=0 RESUME=true \
  bash scripts/run_f2r_alignment_gate_ladder.sh
```

The stages are `F2R`, `F2R_Alignment`, `F2R_Gate`, and
`F2R_AlignmentGate`. Alignment learns a ridge-regularised vocabulary-diagonal
A2 calibration using matched counterfactual answers only. The logistic gate is
trained token-wise with forget answers as positives and generated C+/C- answers
as negatives. Neither calibration uses retain examples or retain metrics.
The final benchmark comparison does use the frozen retain reference and is
therefore labelled `selection_retain_access=true`.

The runner waits until the selected GPU has at least 14 GiB free, reuses every
completed stage after interruption, and writes the comparison to
`open-unlearning/saves/sweeps/forget05_alignment_gate/F2R_SWEEP.md`. Override
`MODELS_ROOT`, `MIN_FREE_GPU_MIB`, or `CALIBRATION_BS` when needed. Use
`DRY_RUN=true` to inspect the exact stage order without loading a model.

After the fixed method ladder is complete, the four-GPU calibration sweep uses
an approximately eight-hour launch budget:

```bash
MODE=full SPLIT=forget05 GPUS="0 1 2 3" MAX_HOURS=8 \
SWEEP_NAME=alignment_gate_8h RESUME=true \
  bash scripts/sweep_f2r_alignment_gate_8h.sh
```

It exposes a 192-candidate pool: 24 alignment settings, 24 gate settings, and
144 alignment+gate settings. The eight-hour launch budget determines how many
are actually evaluated when server throughput varies. Each GPU receives an independent setting;
no new job is launched during the final 45 minutes, while active jobs are
allowed to finish cleanly. The sweep is resumable and records ridge/scale/count
and gate regularisation/steps/learning-rate values in its manifest and output
tables. Because every setting is ranked with the frozen benchmark retain
reference, this is a diagnostic `selection_retain_access=true` sweep.

Reports created before the fixed LLM Beliefs aggregation was adopted can be
updated without rerunning GPU evaluation:

```bash
python scripts/refresh_f2r_aggregation.py \
  --root open-unlearning/saves/eval
```

Use `--dry-run` first to print the corrected Mem./Util./Agg. values without
changing generated reports. Sweep summaries always recompute this hierarchy
from raw metrics, so stale derived fields cannot affect ranking.

If model evaluation already completed but report generation failed, reuse the
existing `TOFU_EVAL.json` without recomputing metrics:

```bash
EVAL_OVERWRITE=false MODE=smoke SPLIT=forget05 GPU=0 \
  bash scripts/run_f2r_tofu.sh
```

The end-to-end pipeline (R_sub selection → train A1 → train A2 → evaluate):

```bash
cd $EASE_ROOT/ULD
GPUS=0 SPLIT=forget10 K=80 \
    bash bashes/tofu/dual_uld_pipeline.sh
```

Knobs:
- `SPLIT` ∈ `{forget01, forget05, forget10}` — TOFU forget percentage.
- `K` — `|R_sub|` (we used 80 for forget10, scale ~ proportional to forget size).
- `GPUS` — comma-separated CUDA device ids. Multi-GPU triggers DDP.

Output:
- LoRA checkpoints under `outputs_trained_models/tofu_dual/...`
- Eval logs (with `forget_quality`, `forget_proba`, ROUGE-L on
  forget/retain/real_authors/world_facts) under
  `outputs/tune_log/.../eval_tofu.log`.

### Llama-3.2 1B / 3B / 8B (via the open-unlearning framework)

A single script ships at [scripts/run_dual_uld_1b.sh](scripts/run_dual_uld_1b.sh).
It trains both assistants for all three forget splits and evaluates with
open-unlearning's TOFU metrics.

Default settings target **Llama-3.2-1B-Instruct**:

```bash
GPU=0 bash scripts/run_dual_uld_1b.sh
```

To run on a **different base model** (3B, 8B, …) override the shell
variables — no script edit needed. The relevant knobs are env-var driven:

| env var | 1B (default) | 3B | 8B |
|---|---|---|---|
| `HF_BASE_PREFIX` | `open-unlearning/tofu_Llama-3.2-1B-Instruct` | `open-unlearning/tofu_Llama-3.2-3B-Instruct` | `open-unlearning/tofu_Llama-3.1-8B-Instruct` |
| `HF_TOKENIZER`   | `${HF_BASE_PREFIX}_full` | same | same |
| `NUM_LAYER` (assistant depth, ≈ 25 % of base) | `2` | `7` | `8` |
| `LORA_R` | `16` | `16` | `16` |
| `WEIGHT_A1` / `WEIGHT_A2` | `-1.0` / `1.0` | `-0.8` / `0.5` | tune |
| `TOP_FILTER` | `0.01` | `0.01` | `0.01` |
| `TRAIN_BS` / `TRAIN_GA` | `4 / 4` | `2 / 8` | `1 / 16` |
| `TRAIN_EP` | `10` | `5` (A1), `3` (A2) | tune |

Example — 3B run on GPU 1:

```bash
GPU=1 \
HF_BASE_PREFIX=open-unlearning/tofu_Llama-3.2-3B-Instruct \
HF_TOKENIZER=open-unlearning/tofu_Llama-3.2-3B-Instruct_full \
NUM_LAYER=7 WEIGHT_A2=0.5 \
TRAIN_BS=2 TRAIN_GA=8 TRAIN_EP=5 \
MODELS_ROOT=$EASE_ROOT/outputs_trained_models/llama3_3b_dual \
    bash scripts/run_dual_uld_1b.sh
```

To run a single split only (skip the others), set `ONLY=forget10` (or
`forget01` / `forget05`).

For LLaMA-2-7B on the original ULD framework, use the canonical TOFU
pipeline shown above (`bashes/tofu/dual_uld_pipeline.sh`).

### Reproducing best TOFU numbers (LLaMA-2-7B)

| Split | Forget Quality | Model Utility | `w1 / w2 / top_p / num_layer` |
|---|---|---|---|
| forget05 | 0.713 | 0.847 | −0.8 / 0.8 / 0.01 / 8 |
| forget10 | 0.654 | 0.874 | −0.8 / 0.8 / 0.01 / 8 |

These weights are set by `model_mode.dual_uld` in
`ULD/configs/model_mode/dual_uld.yaml`. `forget_quality` is the KS-test
p-value vs. a retain-only reference shipped under `ULD/data/retain*_llama_wd0.01/`.

## Running MUSE

Step 1 — build `R_sub` (chunks of `retain1` most similar to forget chunks):

```bash
cd $EASE_ROOT/dual_uld_muse
python build_rsub.py --split Books --k_frac 0.25
python build_rsub.py --split News  --k_frac 0.20
```

(Outputs `rsub/{Books,News}_rsub.json`. Already shipped.)

Step 2 — (optional) regenerate paraphrase / perturbation augmentations:

```bash
DEEPSEEK_API_KEY=... python paraphrase_forget.py --split Books --n_paraphrase 2
DEEPSEEK_API_KEY=... python perturb_forget.py    --split Books --n_perturb 2
# repeat with --split News
```

Outputs land in `aug/{Books,News}_{paraphrases,perturbations}.jsonl`.
(Already shipped — skip this step to use ours.)

Step 3 — train A1 + A2 and sweep eval weights with one command:

```bash
cd $EASE_ROOT/dual_uld_muse
bash run_dual_uld_muse.sh                 # default: Books
SPLIT=News bash run_dual_uld_muse.sh      # News
```

The script trains both assistants and then runs the eval over a default
`w1` grid, printing `forget_ROUGE / privleak / retain_ROUGE` per weight.
Results land in
`$EASE_ROOT/open-unlearning/saves/eval/muse_Llama-2-7b-hf_<SPLIT>_DualULD_w*/MUSE_SUMMARY.json`.

To use **different hyperparameters**, override env vars (no script edit):

| env var | Books default | News default | meaning |
|---|---|---|---|
| `NUM_LAYER` | `8`    | `16`   | assistant transformer depth |
| `LORA_R`    | `16`   | `64`   | LoRA rank (`LORA_ALPHA` defaults to `2*LORA_R`) |
| `LR`        | `1e-3` | `5e-4` | learning rate |
| `EPOCHS_A1` | `5`    | `10`   | A1 epochs |
| `EPOCHS_A2` | `3`    | `5`    | A2 epochs |
| `BATCH_SIZE` / `GRAD_ACCUM` | `1` / `4` | `1` / `4` | per-step batch & accumulation |
| `WS`        | `"-0.3 -0.5 -0.7 -0.9 -1.1"` | same | space-separated `w1` grid (sweep_eval.sh sets `w2 = |w1|`) |
| `GPU`       | `0`    | `0`    | CUDA device |

Manual eval (skip training, sweep arbitrary weights on existing
checkpoints):

```bash
GPU=0 bash sweep_eval.sh Books "-0.3 -0.5 -0.6 -0.8"
```

By default `sweep_eval.sh` runs the **fast** eval profile (skips verbmem +
extraction). For the **full** MUSE eval, pass `EXP=eval/muse/default`.

## Configuration cheatsheet

The DualULD logit composition is implemented in two places that share the
same shape:

- TOFU: `ULD/uld/model/dualcontrastllm.py` (single-ULD baseline is `contrastllm.py` in the same directory; selected via `ULD/configs/model_mode/dual_uld.yaml`)
- MUSE: `open-unlearning/src/model/dual_uld.py` (HuggingFace
  `AutoModelForCausalLM` subclass for the open-unlearning harness)

Key knobs (both frameworks):

| name | meaning | typical |
|---|---|---|
| `weight_a1` | scales A1 logits (negative — subtracts) | −0.6 to −1.0 |
| `weight_a2` | scales A2 logits (positive — restores R_sub) | `\|w1\|` |
| `top_logit_filter` | zero out assistant tokens below this prob | 0.01 |
| `num_layer` | # of base-model layers used for the LoRA assistants | 4 or 8 |

## What is *not* shipped

To keep this repository under 60MB and respect double-blind anonymity:

- No trained model weights (LoRA adapters, full-FT checkpoints). Re-run the
  training scripts above.
- No raw experiment outputs (`outputs/`, `outputs_trained_models/`,
  `saves/eval/` are excluded).
- No author-identifying git history (all `.git/` directories are stripped).

The data augmentation files (paraphrases, perturbations, R_sub indices) and
the retain-only reference results (used by TOFU's `forget_quality` KS test)
**are** included so eval is reproducible end-to-end without external API
calls.

## Acknowledgements

This codebase builds on top of two public projects whose licenses and
upstream code are preserved:

- **ULD** — single-assistant logit-difference unlearning. We extend it with
  a second assistant (A2) and the R_sub mechanism. Original framework is
  contained in `ULD/`.
- **open-unlearning** — unlearning evaluation harness. We add
  `src/model/dual_uld.py` and adapter configs. The rest of `open-unlearning/`
  is upstream.

We do not claim authorship of the upstream files. See `ULD/LICENSE` and
`open-unlearning/LICENSE`.
