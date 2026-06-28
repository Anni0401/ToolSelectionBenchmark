#!/bin/bash
# Setup script: Install project dependencies using uv
# Works locally and on interactive SLURM nodes (salloc).
#
# Usage:
#   bash deploy/uv_setup.sh              # benchmark deps only
#   bash deploy/uv_setup.sh --vllm       # + vllm/transformers for model hosting
#   bash deploy/uv_setup.sh --vllm --cu124  # + torch with CUDA 12.4 index

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

INSTALL_VLLM=false
CUDA_INDEX=""

for arg in "$@"; do
    case $arg in
        --vllm)   INSTALL_VLLM=true ;;
        --cu124)  CUDA_INDEX="cu124" ;;
        --cu121)  CUDA_INDEX="cu121" ;;
        --cu118)  CUDA_INDEX="cu118" ;;
    esac
done

echo "=========================================="
echo "Tool Selection Benchmark – uv Setup"
echo "=========================================="
echo "Project root : ${PROJECT_ROOT}"
echo "Venv         : ${VENV_DIR}"
echo "Install vLLM : ${INSTALL_VLLM}"
[ -n "$CUDA_INDEX" ] && echo "CUDA index   : ${CUDA_INDEX}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Ensure uv is available
# ---------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found. Installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer writes to ~/.local/bin; reload PATH
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &> /dev/null; then
        echo "[ERROR] uv installation failed or not in PATH."
        echo "        Add ~/.local/bin to your PATH and re-run this script."
        exit 1
    fi
fi
echo "[INFO] uv version: $(uv --version)"

# ---------------------------------------------------------------------------
# 2. Create virtual environment (Python 3.12 preferred, 3.10 minimum)
# ---------------------------------------------------------------------------
cd "${PROJECT_ROOT}"

if [ ! -d "${VENV_DIR}" ]; then
    echo "[INFO] Creating virtual environment at ${VENV_DIR}..."
    # uv will download the requested Python if it is not already installed.
    uv venv --python 3.12 "${VENV_DIR}" || uv venv --python 3.10 "${VENV_DIR}"
else
    echo "[INFO] Virtual environment already exists at ${VENV_DIR}. Skipping creation."
fi

PYTHON="${VENV_DIR}/bin/python"
echo "[INFO] Python: $(${PYTHON} --version)"

# ---------------------------------------------------------------------------
# 3. Install benchmark dependencies
# ---------------------------------------------------------------------------
echo "[INFO] Installing benchmark dependencies..."
uv pip install --python "${PYTHON}" -e .

# ---------------------------------------------------------------------------
# 4. (Optional) Install vLLM + transformers for Qwen model hosting
# ---------------------------------------------------------------------------
if [ "${INSTALL_VLLM}" = true ]; then
    echo "[INFO] Installing vLLM extras (latest vllm + transformers)..."

    if [ -n "${CUDA_INDEX}" ]; then
        # Install torch from the official PyTorch CUDA wheel index first so
        # that vllm picks up the GPU-enabled build.
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_INDEX}"
        echo "[INFO] Installing PyTorch from ${TORCH_INDEX} ..."
        uv pip install --python "${PYTHON}" \
            torch torchvision torchaudio \
            --index-url "${TORCH_INDEX}"
    fi

    # Install vllm and transformers without upper-bound pins so uv resolves
    # the newest compatible versions – this avoids FlashAttention mismatches
    # caused by stale pinned versions.
    uv pip install --python "${PYTHON}" -e ".[vllm]"

    echo "[INFO] Verifying installation..."
    "${PYTHON}" -c "import vllm; print(f'[OK] vllm          {vllm.__version__}')"
    "${PYTHON}" -c "import transformers; print(f'[OK] transformers  {transformers.__version__}')"
    "${PYTHON}" -c "import torch; print(f'[OK] torch         {torch.__version__}')"
fi

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "Or prefix commands with:"
echo "  uv run --python ${PYTHON} <command>"
echo ""
if [ "${INSTALL_VLLM}" = true ]; then
    echo "Start the vLLM server with:"
    echo "  bash deploy/slurm_vllm_deploy.sh"
    echo ""
fi
