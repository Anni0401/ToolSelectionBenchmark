#!/bin/bash
#SBATCH --job-name=wtb-dpo-qwen3-lora
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Usage examples:
# 1x 94GB H100 (default in this script):
#   sbatch deploy/slurm_dpo_qwen3_lora.sh
# Optional manual override for generic clusters:
#   sbatch --gres=gpu:1 deploy/slurm_dpo_qwen3_lora.sh
# Override defaults via env:
#   sbatch --export=ALL,MODEL_NAME=Qwen/Qwen3-8B,INPUT_JSON=/path/to/dpo.json,BETA=0.1 deploy/slurm_dpo_qwen3_lora.sh

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${PROJECT_ROOT}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

echo "===================================================="
echo "SLURM DPO LoRA training job"
echo "Project root: ${PROJECT_ROOT}"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Host:         $(hostname)"
echo "===================================================="

VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv-dpo}"

if [[ ! -d "${VENV_PATH}" ]]; then
    echo "ERROR: virtualenv not found: ${VENV_PATH}"
    echo "Create it first, then install training deps."
    exit 1
fi

source "${VENV_PATH}/bin/activate"

echo "Python:  $(which python)"
echo "Version: $(python --version)"

# Install/upgrade required training deps in current venv.
pip install --upgrade "transformers>=4.44" "datasets>=2.21" "accelerate>=0.34" "trl>=0.10" "peft>=0.12"

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export HF_HOME="${WORK:-${HOME}}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
INPUT_JSON="${INPUT_JSON:-${PROJECT_ROOT}/multi-agent-framework/queries_gold_tools_batch1_dpo_ranked.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/multi-agent-framework/qwen3-8b-dpo-lora}"
BETA="${BETA:-0.1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-6}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
VAL_RATIO="${VAL_RATIO:-0.2}"
SEED="${SEED:-42}"

if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "ERROR: input DPO JSON not found: ${INPUT_JSON}"
    exit 1
fi

echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

if (( PER_DEVICE_BATCH < 1 || PER_DEVICE_BATCH > 4 )); then
    echo "ERROR: PER_DEVICE_BATCH must be in [1, 4] for this setup. Got: ${PER_DEVICE_BATCH}"
    exit 1
fi

echo "Starting DPO LoRA training..."

python -u "${PROJECT_ROOT}/multi-agent-framework/train_dpo_qwen3_lora.py" \
    --input "${INPUT_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --epochs "${EPOCHS}" \
    --learning-rate "${LR}" \
    --beta "${BETA}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --per-device-train-batch-size "${PER_DEVICE_BATCH}" \
    --per-device-eval-batch-size "${PER_DEVICE_BATCH}" \
    --max-length 1024 \
    --max-prompt-length 768 \
    --val-ratio "${VAL_RATIO}" \
    --group-by gold_tools \
    --seed "${SEED}" \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --gradient-checkpointing \
    --bf16 \
    --logging-steps 1 \
    --save-strategy epoch \
    --export-cleaned-json

echo "===================================================="
echo "DPO LoRA training complete"
echo "Output dir: ${OUTPUT_DIR}"
echo "===================================================="
