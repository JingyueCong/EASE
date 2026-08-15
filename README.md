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
1-paraphrased probability, 1-forget truth ratio)`, `Util = H(model utility,
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
`F2R_SWEEP_ALL_METRICS.{csv,md}` provides long-form exports of every derived
and EASE/Open-Unlearning metric for every configuration.
Because both FQ and MU inspect the frozen retain reference, these sweep reports
are explicitly marked `selection_retain_access=true`; use them as diagnostics
or select on a separate development setting before making retain-free claims.

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
