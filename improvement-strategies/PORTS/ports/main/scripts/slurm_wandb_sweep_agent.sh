#!/bin/bash
#SBATCH --job-name=ports-sweep
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=94G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Runs a W&B sweep agent that pulls hyperparameter combinations one at a time
# from an existing sweep (created beforehand with `wandb sweep sweep_ports.yaml`)
# and trains PORTS with them via scripts/train_ports_sweep.sh.
#
# Usage:
#   sbatch scripts/slurm_wandb_sweep_agent.sh <ENTITY/PROJECT/SWEEP_ID> [RUN_COUNT]
#
# Submit this multiple times to run several combinations in parallel (one GPU each);
# each agent keeps requesting the next pending run until the sweep is exhausted.

set -Eeuo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: missing sweep ID argument."
    echo "Usage: sbatch $0 <ENTITY/PROJECT/SWEEP_ID> [RUN_COUNT]"
    exit 1
fi
SWEEP_ID="$1"
RUN_COUNT="${2:-}"  # optional: max number of runs this agent should execute

####################################################
# Project paths
####################################################

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PORTS_ROOT="${SLURM_SUBMIT_DIR}"
else
    MAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    PORTS_ROOT="$(dirname "${MAIN_ROOT}")"
fi
MAIN_ROOT="${PORTS_ROOT}/main"
ENV_FILE="${PORTS_ROOT}/.env"



echo "===================================================="
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Host:          $(hostname)"
echo "PORTS root:    ${PORTS_ROOT}"
echo "Main root:     ${MAIN_ROOT}"
echo "Sweep ID:      ${SWEEP_ID}"
echo "Run count:     ${RUN_COUNT:-unlimited}"
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

export WANDB_DIR="${WORK}/wandb"
export WANDB_CACHE_DIR="${WORK}/wandb_cache"
export WANDB_CONFIG_DIR="${WORK}/wandb_config"

####################################################
# GPU info
####################################################

echo ""
echo "Allocated GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo ""

####################################################
# Run the W&B sweep agent
####################################################

if [ -n "${RUN_COUNT}" ]; then
    wandb agent --count "${RUN_COUNT}" "${SWEEP_ID}"
else
    wandb agent "${SWEEP_ID}"
fi
