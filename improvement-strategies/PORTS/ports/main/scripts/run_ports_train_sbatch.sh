#!/bin/bash
#SBATCH --job-name=ports-train
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=94G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -Eeuo pipefail

####################################################
# Project paths
####################################################

# main/ dir of the PORTS checkout (this script lives in main/scripts/)
# NOTE: sbatch copies the submitted script into a job spool dir on the compute
# node (e.g. /var/spool/slurmd/...), so BASH_SOURCE is unreliable under sbatch.
# Prefer SLURM_SUBMIT_DIR (the dir you ran `sbatch` from), which should be the
# PORTS checkout root (the "ports/" dir containing main/).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PORTS_ROOT="${SLURM_SUBMIT_DIR}"
else
    MAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    PORTS_ROOT="$(dirname "${MAIN_ROOT}")"
fi
MAIN_ROOT="${PORTS_ROOT}"
ENV_FILE="${MAIN_ROOT}/.env"

cd "${MAIN_ROOT}"

echo "===================================================="
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Host:          $(hostname)"
echo "PORTS root:    ${PORTS_ROOT}"
echo "Main root:     ${MAIN_ROOT}"
echo "===================================================="

####################################################
# Conda environment
####################################################

CONDA_ENV_NAME="${CONDA_ENV_NAME:-py312}"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -z "${CONDA_BASE}" ]; then
    echo "ERROR: conda not found on PATH/in this shell."
    exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda activate "${CONDA_ENV_NAME}"; then
    echo "ERROR: conda environment '${CONDA_ENV_NAME}' does not exist."
    echo "Create it first with:"
    echo "  conda create -n ${CONDA_ENV_NAME} python=3.12 && conda activate ${CONDA_ENV_NAME} && pip install -r \"${PORTS_ROOT}/build/requirements.txt\""
    exit 1
fi

echo "Python:  $(which python)"
echo "Version: $(python --version)"

####################################################
# Secrets / API keys (HF_TOKEN, WANDB_API_KEY, ...)
####################################################

if [ ! -f "${ENV_FILE}" ]; then
    echo "ERROR: .env file not found: ${ENV_FILE}"
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

####################################################
# Caches / temp dirs (avoid filling $HOME quota)
####################################################

: "${WORK:=${PORTS_ROOT}}"

export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${WORK}/tmp_pip/cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TOKENIZERS_PARALLELISM=false

####################################################
# GPU info
####################################################

echo ""
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo ""

####################################################
# Training parameters (override via env vars or sbatch --export=)
####################################################

export DATASET_NAME="${DATASET_NAME:-bfcl}"
export RETRIEVAL_MODEL_NAME="${RETRIEVAL_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
export INFERENCE_MODEL_PSEUDONAME="${INFERENCE_MODEL_PSEUDONAME:-llama3-8B}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-10000}"
export N_EPOCHS="${N_EPOCHS:-2}"

echo "===================================================="
echo "DATASET_NAME:               ${DATASET_NAME}"
echo "RETRIEVAL_MODEL_NAME:       ${RETRIEVAL_MODEL_NAME}"
echo "INFERENCE_MODEL_PSEUDONAME: ${INFERENCE_MODEL_PSEUDONAME}"
echo "TRAIN_BATCH_SIZE:           ${TRAIN_BATCH_SIZE}"
echo "MAX_TRAIN_SAMPLES:          ${MAX_TRAIN_SAMPLES}"
echo "N_EPOCHS:                   ${N_EPOCHS}"
echo "===================================================="

####################################################
# Run training
####################################################

bash scripts/train_ports.sh
