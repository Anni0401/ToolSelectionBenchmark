#!/bin/bash
#SBATCH --job-name=wtb-dpo-rank
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

echo "===================================================="
echo "SLURM DPO ranking generation job"
echo "Project root: ${PROJECT_ROOT}"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Host:        $(hostname)"
echo "===================================================="

if [ ! -d "${PROJECT_ROOT}/.venv" ]; then
    echo "ERROR: ${PROJECT_ROOT}/.venv not found."
    echo "Create it first with: bash deploy/uv_setup.sh --vllm"
    exit 1
fi

source "${PROJECT_ROOT}/.venv/bin/activate"

echo "Python: $(which python)"
echo "Version: $(python --version)"

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

export HF_HOME="${WORK:-${HOME}}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

export MODEL_NAME="Qwen/Qwen3-Embedding-8B"
export VLLM_EMBEDDING_PORT="${VLLM_EMBEDDING_PORT:-8002}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
export QWEN3_EMBEDDING_BASE_URL="http://localhost:${VLLM_EMBEDDING_PORT}/v1"
export QWEN3_EMBEDDING_MODEL="${MODEL_NAME}"
export QWEN3_EMBEDDING_API_KEY="EMPTY"

EMBED_SERVER_PID=""
cleanup() {
    echo
    echo "Cleaning up embedding server..."
    if [ -n "${EMBED_SERVER_PID}" ] && kill -0 "${EMBED_SERVER_PID}" 2>/dev/null; then
        kill "${EMBED_SERVER_PID}" 2>/dev/null || true
        wait "${EMBED_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" \
    > "${PROJECT_ROOT}/dpo_rank_embedding_${SLURM_JOB_ID}.log" 2>&1 &
EMBED_SERVER_PID=$!

echo "Embedding server PID: ${EMBED_SERVER_PID}"
echo "Embedding server log: ${PROJECT_ROOT}/dpo_rank_embedding_${SLURM_JOB_ID}.log"

MAX_ATTEMPTS=120
SLEEP_SECONDS=5
for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    if ! kill -0 "${EMBED_SERVER_PID}" 2>/dev/null; then
        echo "ERROR: Embedding server exited before becoming ready."
        tail -100 "${PROJECT_ROOT}/dpo_rank_embedding_${SLURM_JOB_ID}.log" || true
        exit 1
    fi

    if curl -sf "http://localhost:${VLLM_EMBEDDING_PORT}/v1/models" >/dev/null 2>&1; then
        echo "Embedding server is ready."
        break
    fi

    if [ "${attempt}" -eq "${MAX_ATTEMPTS}" ]; then
        echo "ERROR: Embedding server did not become ready in time."
        tail -100 "${PROJECT_ROOT}/dpo_rank_embedding_${SLURM_JOB_ID}.log" || true
        exit 1
    fi

    sleep "${SLEEP_SECONDS}"
done

echo "Available models:"
curl -s "http://localhost:${VLLM_EMBEDDING_PORT}/v1/models"
echo

python -u "${PROJECT_ROOT}/multi-agent-framework/generate_dpo_ranked_preferences.py" \
    --input "${PROJECT_ROOT}/multi-agent-framework/queries_gold_tools_batch1_rewrites.json" \
    --tools-file "${PROJECT_ROOT}/multi-agent-framework/tools/tools_en.jsonl" \
    --output "${PROJECT_ROOT}/multi-agent-framework/queries_gold_tools_batch1_dpo_ranked.json"

echo "===================================================="
echo "DPO ranking generation complete"
echo "Output: ${PROJECT_ROOT}/multi-agent-framework/queries_gold_tools_batch1_dpo_ranked.json"
echo "===================================================="
