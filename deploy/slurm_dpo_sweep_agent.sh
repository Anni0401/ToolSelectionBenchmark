#!/bin/bash
#SBATCH --job-name=wtb-dpo-sweep
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Runs a W&B sweep agent that pulls hyperparameter combinations one at a time
# from an existing sweep (created beforehand with
# `wandb sweep multi-agent-framework/sweep.yaml`) and trains the DPO Qwen3 LoRA
# policy with them via multi-agent-framework/run.sh.
#
# Usage:
#   sbatch deploy/slurm_dpo_sweep_agent.sh <ENTITY/PROJECT/SWEEP_ID> [RUN_COUNT]
#
# Submit this multiple times to run several combinations in parallel (one GPU each);
# each agent keeps requesting the next pending run until the sweep is exhausted.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "ERROR: missing sweep ID argument."
    echo "Usage: sbatch $0 <ENTITY/PROJECT/SWEEP_ID> [RUN_COUNT]"
    exit 1
fi
SWEEP_ID="$1"
RUN_COUNT="${2:-}"  # optional: max number of runs this agent should execute

if [[ -n "${RUN_COUNT}" && ! "${RUN_COUNT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: RUN_COUNT must be a non-negative integer, got '${RUN_COUNT}'."
    echo "Usage: sbatch $0 <ENTITY/PROJECT/SWEEP_ID> [RUN_COUNT]"
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# NOTE: on this cluster, sbatch stages the script to /var/spool/slurmd/... before
# execution, so $SCRIPT_DIR is unreliable at runtime. Like the other runs/*.sh
# scripts in this repo, derive PROJECT_ROOT from $SLURM_SUBMIT_DIR - but also
# tolerate being submitted from inside multi-agent-framework/ itself by walking
# up one level in that case, since that's the natural cwd for this workflow.
SUBMIT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -n "${SUBMIT_DIR}" && -d "${SUBMIT_DIR}/multi-agent-framework" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -n "${SUBMIT_DIR}" && "$(basename "${SUBMIT_DIR}")" == "multi-agent-framework" && -d "$(dirname "${SUBMIT_DIR}")/multi-agent-framework" ]]; then
    PROJECT_ROOT="$(dirname "${SUBMIT_DIR}")"
else
    PROJECT_ROOT=""
fi
if [[ -z "${PROJECT_ROOT}" ]]; then
    echo "ERROR: could not determine project root. Submit this job from the repo root or multi-agent-framework/, e.g.:"
    echo "  cd /home/aherrman/ToolSelectionBenchmark && sbatch deploy/slurm_dpo_sweep_agent.sh <SWEEP_ID>"
    exit 1
fi
SWEEP_DIR="${PROJECT_ROOT}/multi-agent-framework"

echo "===================================================="
echo "SLURM DPO LoRA sweep agent"
echo "Project root: ${PROJECT_ROOT}"
echo "Sweep dir:    ${SWEEP_DIR}"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "Sweep ID:     ${SWEEP_ID}"
echo "Run count:    ${RUN_COUNT:-unlimited}"
echo "===================================================="

VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-dpo}"

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: virtualenv not found: ${VENV_PATH}"
    exit 1
fi

source "${VENV_PATH}/bin/activate"

echo "Python:  $(which python)"
echo "Version: $(python --version)"

if ! python -c "import accelerate, datasets, peft, torch, transformers, trl"; then
    echo "ERROR: training dependencies are missing from ${VENV_PATH}."
    exit 1
fi

if ! python -c "import wandb" 2>/dev/null; then
    echo "wandb not found in ${VENV_PATH}, installing..."
    pip install -q wandb
fi

# Optional secrets (WANDB_API_KEY, ...) - source if present, don't fail otherwise
# (falls back to a cached `wandb login`).
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
fi

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export HF_HOME="${WORK:-${HOME}}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

export WANDB_DIR="${WORK:-${PROJECT_ROOT}}/wandb"
export WANDB_CACHE_DIR="${WORK:-${PROJECT_ROOT}}/wandb_cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

cd "${SWEEP_DIR}"

if [ -n "${RUN_COUNT}" ]; then
    wandb agent --count "${RUN_COUNT}" "${SWEEP_ID}"
else
    wandb agent "${SWEEP_ID}"
fi
