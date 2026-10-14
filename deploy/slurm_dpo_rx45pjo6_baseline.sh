#!/bin/bash
#SBATCH --job-name=rx45pjo6-baseline
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=wtb-dpo-rx45pjo6-baseline_%j.out
#SBATCH --error=wtb-dpo-rx45pjo6-baseline_%j.err

set -euo pipefail

SUBMIT_DIR="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -n "${SUBMIT_DIR}" && -d "${SUBMIT_DIR}/multi-agent-framework" ]]; then
    PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -n "${SUBMIT_DIR}" && "$(basename "${SUBMIT_DIR}")" == "multi-agent-framework" && -d "$(dirname "${SUBMIT_DIR}")/multi-agent-framework" ]]; then
    PROJECT_ROOT="$(dirname "${SUBMIT_DIR}")"
else
    echo "ERROR: could not determine project root. Submit from the repository root or multi-agent-framework/." >&2
    echo "Example: cd /home/aherrman/ToolSelectionBenchmark && sbatch deploy/slurm_dpo_rx45pjo6_baseline.sh" >&2
    exit 1
fi
SWEEP_DIR="${PROJECT_ROOT}/multi-agent-framework"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-dpo}"

load_env_file() {
    local env_file="$1"
    if [[ -f "${env_file}" ]]; then
        echo "Loading environment: ${env_file}"
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi
}

if [[ -n "${ENV_FILE:-}" ]]; then
    load_env_file "${ENV_FILE}"
else
    load_env_file "${PROJECT_ROOT}/.env"
    load_env_file "${SWEEP_DIR}/.env"
fi

echo "===================================================="
echo "rx45pjo6 explicit baseline"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Host:       $(hostname)"
echo "Project:    ${PROJECT_ROOT}"
echo "Sweep dir:  ${SWEEP_DIR}"
echo "===================================================="

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "ERROR: Python environment not found: ${VENV_PATH}" >&2
    exit 1
fi
if [[ ! -f "${SWEEP_DIR}/queries_gold_tools_batch1_dpo_ranked.json" ]]; then
    echo "ERROR: DPO training data not found in ${SWEEP_DIR}" >&2
    exit 1
fi

source "${VENV_PATH}/bin/activate"

if ! python -c "import accelerate, datasets, peft, torch, transformers, trl, wandb"; then
    echo "ERROR: required training packages are unavailable in ${VENV_PATH}" >&2
    exit 1
fi

echo "Python: $(which python)"
python --version
echo "Allocated GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

if ! python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this SLURM allocation"'; then
    echo "ERROR: this job has no usable CUDA GPU; refusing to download/load the model." >&2
    exit 1
fi

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

WORK_DIR="${WORK:-${HOME}}"
export HF_HOME="${WORK_DIR}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export WANDB_DIR="${WORK:-${PROJECT_ROOT}}/wandb"
export WANDB_CACHE_DIR="${WORK:-${PROJECT_ROOT}}/wandb_cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

export WANDB_ENTITY=herrmann-ann-kathrin-universit-t-mannheim
export WANDB_PROJECT=ToolSelectionBenchmark-multi-agent-framework
export WANDB_RUN_GROUP=rx45pjo6
export WANDB_NAME="rx45pjo6-explicit-baseline-${SLURM_JOB_ID:-manual}"
export WANDB_TAGS=rx45pjo6,explicit-baseline

cd "${SWEEP_DIR}"
export OUTPUT_DIR="${SWEEP_DIR}/sweeps/baseline_rx45pjo6_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "Baseline configuration: lr=1e-5 beta=0.2 epochs=3 r=16 alpha=32 dropout=0.05"
echo "Output directory: ${OUTPUT_DIR}"

LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 bash run.sh \
    --beta=0.2 \
    --epochs=3 \
    --learning-rate=1e-5