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
│   ├── master_v3.sh           # ★ canonical MUSE training pipeline
│   ├── sweep_eval.sh          # sweep w1 over a fixed (A1, A2) pair
│   ├── aug/                   # paraphrase + perturbation jsonl (provided)
│   └── rsub/                  # precomputed R_sub indices (provided)
│
├── open-unlearning/           # MUSE evaluation framework (upstream + our patches)
│   ├── src/model/dual_uld.py  # ★ our DualULD HuggingFace wrapper
│   ├── src/model/uld.py       # single-assistant ULD baseline
│   ├── src/evals/muse.py      # MUSE benchmark eval
│   └── ...                    # rest is upstream open-unlearning
│
└── scripts/                   # cross-cutting experiment drivers
    ├── run_dual_uld_*.sh      # Llama-3.2 1B/3B TOFU runs via open-unlearning
    ├── sweep_dual_v2_*.sh     # hyper-parameter sweeps
    ├── sweep_3b_f*.sh         # forget01/05/10 sweeps for 3B
    ├── eval_laaj.py           # LLM-as-a-Judge eval
    └── _probe_dualuld_logits.py
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

If you also want to materialise checkpoint paths, set:

```bash
export CKPT_DIR=$EASE_ROOT/outputs_trained_models   # any writable location
```

(Some sweep scripts reference `${CKPT_DIR}` because we did not ship trained
LoRA weights — running the training step will produce them locally.)

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

For Llama-3.2 1B / 3B variants (run via the open-unlearning framework
instead), see `scripts/run_dual_uld_1b.sh` and `scripts/run_dual_uld_3b_f10.sh`.

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

Step 3 — train A1 and A2 on each split:

```bash
cd $EASE_ROOT/dual_uld_muse

python train_assistant.py \
    --split Books --role a1 \
    --epochs 5 --batch_size 1 --grad_accum 4 --num_layer 8 \
    --paraphrase_path $EASE_ROOT/dual_uld_muse/aug/Books_paraphrases.jsonl \
    --perturb_path    $EASE_ROOT/dual_uld_muse/aug/Books_perturbations.jsonl

python train_assistant.py \
    --split Books --role a2 \
    --epochs 5 --batch_size 1 --grad_accum 4 --num_layer 8

# repeat both with --split News
```

Or run the full sequence with the shipped pipeline:
```bash
bash master_v3.sh    # waits for perturb done, retrains all 4 assistants, sweeps
```

Step 4 — evaluate via the open-unlearning MUSE harness:

```bash
cd $EASE_ROOT/dual_uld_muse
bash sweep_eval.sh Books "-0.3 -0.5 -0.6 -0.8"
bash sweep_eval.sh News  "-0.3 -0.5 -0.6 -0.8"
```

Each call sweeps `w1` (with `w2 = |w1|`) and writes
`MUSE_SUMMARY.json` under
`$EASE_ROOT/open-unlearning/saves/eval/muse_Llama-2-7b-hf_<SPLIT>_DualULD_w*/`.

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
