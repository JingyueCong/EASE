#!/usr/bin/env bash
# Create isolated training and evaluation environments on a Linux CUDA server.
set -euo pipefail

EASE_ROOT="${EASE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-}"
CONDA_INSTALL_PREFIX="${CONDA_INSTALL_PREFIX:-${HOME}/miniconda3}"
TRAIN_ENV="${TRAIN_ENV:-ease-f2r-train}"
EVAL_ENV="${EVAL_ENV:-ease-f2r-eval}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

if [ -z "$CONDA_BIN" ]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [ -z "$CONDA_BIN" ]; then
    if [ -x "${CONDA_INSTALL_PREFIX}/bin/conda" ]; then
        CONDA_BIN="${CONDA_INSTALL_PREFIX}/bin/conda"
    else
        echo "Conda was not found; installing Miniconda at $CONDA_INSTALL_PREFIX"
        case "$(uname -m)" in
            x86_64|amd64) miniconda_arch="x86_64" ;;
            aarch64|arm64) miniconda_arch="aarch64" ;;
            *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
        esac
        if [ "$(uname -s)" != "Linux" ]; then
            echo "Automatic Miniconda installation currently supports Linux only." >&2
            exit 1
        fi
        installer="$(mktemp -t ease-miniconda-XXXXXX.sh)"
        installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${miniconda_arch}.sh"
        if command -v curl >/dev/null 2>&1; then
            curl -L --fail --retry 3 -o "$installer" "$installer_url"
        elif command -v wget >/dev/null 2>&1; then
            wget -O "$installer" "$installer_url"
        else
            echo "curl or wget is required to install Miniconda." >&2
            exit 1
        fi
        bash "$installer" -b -p "$CONDA_INSTALL_PREFIX"
        rm -f "$installer"
        CONDA_BIN="${CONDA_INSTALL_PREFIX}/bin/conda"
    fi
fi

echo "Using conda: $CONDA_BIN"

echo "[1/4] Creating training environment: $TRAIN_ENV"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$TRAIN_ENV"; then
    "$CONDA_BIN" create -n "$TRAIN_ENV" python=3.10 pip -y
fi

echo "[2/4] Installing EASE training package and generator dependency"
"$CONDA_BIN" run -n "$TRAIN_ENV" python -m pip install \
    torch==2.1.1 --index-url "$TORCH_INDEX_URL"
"$CONDA_BIN" run -n "$TRAIN_ENV" python -m pip install \
    -r "$EASE_ROOT/ULD/requirements.txt" \
    accelerate==0.31.0 bitsandbytes==0.43.1 deepspeed==0.14.2 \
    pandas==2.2.2 peft==0.11.1 openai
"$CONDA_BIN" run -n "$TRAIN_ENV" python -m pip install \
    -e "$EASE_ROOT/ULD" --no-deps

echo "[3/4] Creating/updating evaluation environment: $EVAL_ENV"
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$EVAL_ENV"; then
    "$CONDA_BIN" create -n "$EVAL_ENV" python=3.11 pip -y
fi
"$CONDA_BIN" run -n "$EVAL_ENV" python -m pip install \
    -r "$EASE_ROOT/open-unlearning/requirements.txt" peft
"$CONDA_BIN" run -n "$EVAL_ENV" python -m pip install -e "$EASE_ROOT/open-unlearning"

echo "[4/4] Environment check"
"$CONDA_BIN" run -n "$TRAIN_ENV" python -c \
    'import torch,transformers,datasets,peft; print("train:", torch.__version__, transformers.__version__, "cuda=", torch.cuda.is_available())'
"$CONDA_BIN" run -n "$EVAL_ENV" python -c \
    'import torch,transformers,datasets,peft; print("eval:", torch.__version__, transformers.__version__, "cuda=", torch.cuda.is_available())'

echo "Setup complete. Run: MODE=smoke bash scripts/run_f2r_tofu.sh"
