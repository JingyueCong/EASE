# EASE: Dual Small-Assistant Unlearning via Logit Difference

Anonymous code release for double-blind review.

This repository contains the training and evaluation code for EASE on
**TOFU**, the synthetic biographical Q&A unlearning benchmark.

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
├── ease/                      # canonical TOFU implementation
│   ├── ease/                   # core library (model, data, training utils)
│   │   ├── model/             # EASE logit composition + relative-top filter
│   │   └── tofuutil/          # TOFU eval utilities (forget quality, model utility)
│   ├── scripts/               # entry-point Python: hf_forget_train.py, eval_tofu.py
│   ├── configs/               # Hydra configs (data, data_mode, eval, tune)
│   ├── bashes/tofu/           # end-to-end shell pipelines
│   │   └── ease_pipeline.sh   # ★ canonical TOFU run
│   └── data/                  # data augmentation + retain reference results
│       ├── aug_data/tofu/     # paraphrased + perturbed answers (provided)
│       └── retain*_*_wd0.01/  # retain-only reference for forget_quality KS test
│
├── open-unlearning/           # TOFU evaluation framework (upstream + EASE adapter)
│   ├── src/model/ease.py  # ★ our EASE HuggingFace wrapper
│   ├── src/model/assistant.py # single-assistant baseline
│   └── ...                    # rest is upstream open-unlearning
│
└── scripts/
    └── run_ease_1b.sh     # ★ Llama-3.2 1B TOFU run via open-unlearning
                               #   (override env vars to target 3B / 8B —
                               #    see "Llama-3.2 ... via open-unlearning" below)
```

## Prerequisites

- Python 3.10
- One or more CUDA-capable GPUs (TOFU 1B fits on a single 24GB; TOFU 7B / 3B
  needs 40GB+)
- HuggingFace token only if you fetch gated models (LLaMA-2). Set with
  `export HF_TOKEN=...` or `huggingface-cli login`.

All shell scripts expect `EASE_ROOT` to point at this repository:

```bash
export EASE_ROOT=$(pwd)
```

Trained LoRA checkpoints are not shipped — the training scripts will write
them to `outputs_trained_models/` (created on first run). Override the
location via `MODELS_ROOT=...` if you need to.

## Setup

### TOFU (`ease/`)

```bash
cd $EASE_ROOT/ease
conda env create -f environment.yaml      # creates env "ease"
conda activate ease
pip install -e .
```

### Evaluation with `open-unlearning/`

```bash
cd $EASE_ROOT/open-unlearning
pip install -r requirements.txt
pip install -e .
python setup_data.py --eval_logs
```

## Running TOFU

The end-to-end pipeline (R_sub selection → train A1 → train A2 → evaluate):

```bash
cd $EASE_ROOT/ease
GPUS=0 SPLIT=forget10 K=80 \
    bash bashes/tofu/ease_pipeline.sh
```

Knobs:
- `SPLIT` ∈ `{forget01, forget05, forget10}` — TOFU forget percentage.
- `K` — `|R_sub|` (we used 80 for forget10, scale ~ proportional to forget size).
- `GPUS` — comma-separated CUDA device ids. Multi-GPU triggers DDP.

Output:
- LoRA checkpoints under `outputs_trained_models/tofu_ease/...`
- Eval logs (with `forget_quality`, `forget_proba`, ROUGE-L on
  forget/retain/real_authors/world_facts) under
  `outputs/tune_log/.../eval_tofu.log`.

### Llama-3.2 1B / 3B / 8B (via the open-unlearning framework)

A single script ships at [scripts/run_ease_1b.sh](scripts/run_ease_1b.sh).
It trains both assistants for all three forget splits and evaluates with
open-unlearning's TOFU metrics.

Default settings target **Llama-3.2-1B-Instruct**:

```bash
GPU=0 bash scripts/run_ease_1b.sh
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
MODELS_ROOT=$EASE_ROOT/outputs_trained_models/llama3_3b_ease \
    bash scripts/run_ease_1b.sh
```

To run a single split only (skip the others), set `ONLY=forget10` (or
`forget01` / `forget05`).

For LLaMA-2-7B, use the canonical EASE TOFU
pipeline shown above (`bashes/tofu/ease_pipeline.sh`).

## Configuration cheatsheet

The EASE logit composition is implemented in:

- `ease/ease/model/easecontrastllm.py` for the canonical TOFU pipeline.
- `open-unlearning/src/model/ease.py` for the HuggingFace evaluation harness.

Key knobs:

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

- **Upstream logit-difference framework** — the original single-assistant
  implementation described by Ji et al. (2024). EASE extends it with a second
  assistant and the R_sub mechanism; its MIT license is preserved in
  `ease/LICENSE`.
- **open-unlearning** — unlearning evaluation harness. We add
  `src/model/ease.py` and adapter configs. The rest of `open-unlearning/`
  is upstream.

We do not claim authorship of the upstream files. See `ease/LICENSE` and
`open-unlearning/LICENSE`.
