#!/bin/bash
#SBATCH --job-name=vllm-install
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/%u/vllm_install_%j.log
#SBATCH --error=/home/%u/vllm_install_%j.err

# SLURM Job Script: Install vLLM on GPU Node using uv
# This script should be submitted to SLURM to install vLLM on a GPU node.
# It uses uv instead of conda/pip for faster, reproducible installs.
#
# Usage: sbatch deploy/slurm_install_vllm.sh

set -e

echo "=========================================="
echo "vLLM Installation on GPU Node (uv)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "=========================================="

# Load necessary modules (adjust for your cluster)
# Common module names - uncomment as needed:
# module load cuda/12.4
# module load gcc/11

# Check for CUDA and determine the right torch index
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU found:"
    nvidia-smi --query-gpu=name,driver_version --format=csv

    # Detect CUDA version from nvcc or nvidia-smi
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    else
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    fi
    echo "[INFO] CUDA version: ${CUDA_VER}"
else
    echo "[WARNING] nvidia-smi not found."
    module load cuda 2>/dev/null || true
    CUDA_VER=""
fi

# Map CUDA version to PyTorch wheel index tag
if [[ "${CUDA_VER}" == 12.4* ]] || [[ "${CUDA_VER}" == 12.5* ]] || [[ "${CUDA_VER}" == 12.6* ]]; then
    CUDA_INDEX="cu124"
elif [[ "${CUDA_VER}" == 12.1* ]] || [[ "${CUDA_VER}" == 12.2* ]] || [[ "${CUDA_VER}" == 12.3* ]]; then
    CUDA_INDEX="cu121"
elif [[ "${CUDA_VER}" == 11* ]]; then
    CUDA_INDEX="cu118"
else
    CUDA_INDEX=""
fi
echo "[INFO] Using PyTorch CUDA index: ${CUDA_INDEX:-default PyPI}"

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"

# Ensure uv is available
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv &> /dev/null; then
    echo "[INFO] uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "[INFO] uv version: $(uv --version)"

# Run the unified uv setup script with vLLM extras
if [ -n "${CUDA_INDEX}" ]; then
    bash deploy/uv_setup.sh --vllm "--${CUDA_INDEX}"
else
    bash deploy/uv_setup.sh --vllm
fi

echo ""
echo "[INFO] Installation complete!"
echo "[INFO] Activate with: source .venv/bin/activate"
echo "[INFO] Start the server: bash deploy/slurm_vllm_deploy.sh"
