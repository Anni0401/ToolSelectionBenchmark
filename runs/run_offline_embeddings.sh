#!/bin/bash
#SBATCH --job-name=wtb-offline-embed
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

####################################################
# Project paths
####################################################

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_ROOT}"

echo "===================================================="
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Host:          $(hostname)"
echo "Project root:  ${PROJECT_ROOT}"
echo "===================================================="

####################################################
# Environment
####################################################

if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "ERROR: ${PROJECT_ROOT}/.venv does not exist."
    echo "Create it first with:"
    echo "  bash deploy/uv_setup.sh --vllm"
    exit 1
fi

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Python: $(which python)"
echo "Version: $(python --version)"

####################################################
# NCCL / CUDA
####################################################

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

####################################################
# Hugging Face cache
####################################################

export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"

mkdir -p "${HF_HOME}"
mkdir -p "${HF_HUB_CACHE}"

####################################################
# Embedding server configuration
####################################################

export MODEL_NAME="Qwen/Qwen3-Embedding-8B"
export VLLM_EMBEDDING_PORT=8002
export GPU_MEMORY_UTILIZATION=0.80

export QWEN3_EMBEDDING_BASE_URL="http://localhost:${VLLM_EMBEDDING_PORT}/v1"
export QWEN3_EMBEDDING_MODEL="${MODEL_NAME}"
export QWEN3_EMBEDDING_API_KEY="EMPTY"

####################################################
# Cleanup handler
####################################################

EMBED_SERVER_PID=""

cleanup() {
    echo
    echo "===================================================="
    echo "Cleaning up"
    echo "===================================================="

    if [ -n "${EMBED_SERVER_PID}" ] && kill -0 "${EMBED_SERVER_PID}" 2>/dev/null; then
        echo "Stopping embedding server PID ${EMBED_SERVER_PID}"
        kill "${EMBED_SERVER_PID}" 2>/dev/null || true
        wait "${EMBED_SERVER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

####################################################
# GPU information
####################################################

echo
echo "Allocated GPU:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader

####################################################
# Start embedding server
####################################################

echo
echo "===================================================="
echo "Starting Qwen3 embedding server"
echo "===================================================="

bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" \
    > "${PROJECT_ROOT}/embedding_server_${SLURM_JOB_ID}.log" 2>&1 &

EMBED_SERVER_PID=$!

echo "Embedding server PID: ${EMBED_SERVER_PID}"
echo "Server log: ${PROJECT_ROOT}/embedding_server_${SLURM_JOB_ID}.log"

####################################################
# Wait for server
####################################################

echo
echo "Waiting for embedding server..."

MAX_ATTEMPTS=120
SLEEP_SECONDS=5

for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do

    # Fail immediately if the server process died
    if ! kill -0 "${EMBED_SERVER_PID}" 2>/dev/null; then
        echo
        echo "ERROR: Embedding server exited before becoming ready."
        echo
        echo "Last server log lines:"
        tail -100 "${PROJECT_ROOT}/embedding_server_${SLURM_JOB_ID}.log" || true
        exit 1
    fi

    if curl -sf \
        "http://localhost:${VLLM_EMBEDDING_PORT}/v1/models" \
        >/dev/null 2>&1; then

        echo "Embedding server is ready."
        break
    fi

    if [ "${attempt}" -eq "${MAX_ATTEMPTS}" ]; then
        echo
        echo "ERROR: Embedding server did not become ready."
        echo
        echo "Last server log lines:"
        tail -100 "${PROJECT_ROOT}/embedding_server_${SLURM_JOB_ID}.log" || true
        exit 1
    fi

    if (( attempt % 12 == 0 )); then
        echo "Still waiting... attempt ${attempt}/${MAX_ATTEMPTS}"
    fi

    sleep "${SLEEP_SECONDS}"
done

####################################################
# Verify served model
####################################################

echo
echo "Available models:"
curl -s "http://localhost:${VLLM_EMBEDDING_PORT}/v1/models"
echo

####################################################
# Run offline tool embedding
####################################################

echo
echo "===================================================="
echo "Generating tool embeddings"
echo "===================================================="

cd "${PROJECT_ROOT}/wild-tool-bench"

TOOLS_FILE="../wild-tool-bench/wtb/model_handler/api_inference/tool_schemas_cache.jsonl"

if [ ! -f "${TOOLS_FILE}" ]; then
    echo "ERROR: Tool file does not exist:"
    echo "  ${TOOLS_FILE}"
    exit 1
fi

python -u wtb/model_handler/api_inference/setup_openai_embeddings.py \
    --provider qwen3 \
    --tools-file "${TOOLS_FILE}"

####################################################
# Finished
####################################################

echo
echo "===================================================="
echo "Offline embedding completed successfully"
echo "===================================================="

echo "Expected cache:"
echo "${PROJECT_ROOT}/wild-tool-bench/tool_embeddings_cache.json"
