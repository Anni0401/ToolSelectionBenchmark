#!/bin/bash
# Precompute tool vectors with the PORTS-trained Qwen3 embedding LoRA adapter.

#SBATCH --job-name=wtb-offline-embed-ports-lora
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

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BENCHMARK_ROOT="${PROJECT_ROOT}/wild-tool-bench"
WTB_ENV_FILE="${BENCHMARK_ROOT}/.env"
: "${WORK:?ERROR: WORK environment variable is not set}"

BENCH_VENV="${PROJECT_ROOT}/.venv"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
EMBEDDING_LORA_MODEL="${EMBEDDING_LORA_MODEL:-qwen3-embedding-8b-ports-lora}"
EMBEDDING_LORA_ADAPTER_DIR="${EMBEDDING_LORA_ADAPTER_DIR:-${WORK}/ports/main/output/ports/ports_retriever_20261006_233629/checkpoint-epoch-2}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8002}"
EMBEDDING_GPU_MEMORY_UTILIZATION="${EMBEDDING_GPU_MEMORY_UTILIZATION:-0.80}"
JOB_ID="${SLURM_JOB_ID:-local}"
SERVER_LOG="${PROJECT_ROOT}/embedding_server_ports_lora_${JOB_ID}.log"
TOOLS_FILE="${BENCHMARK_ROOT}/wtb/model_handler/api_inference/tool_schemas_cache.jsonl"
CACHE_FILE="${BENCHMARK_ROOT}/wtb/model_handler/api_inference/tool_embeddings_cache_qwen3_ports_lora.json"
ENV_BACKUP="${TMPDIR:-/tmp}/wtb-offline-ports-lora-env-${JOB_ID}-$$"

for required_path in \
    "${BENCH_VENV}/bin/activate" \
    "${WTB_ENV_FILE}" \
    "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_config.json" \
    "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_model.safetensors" \
    "${TOOLS_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[ERROR] Required file or directory not found: ${required_path}" >&2
        exit 1
    fi
done

export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${TMPDIR}/cache"
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"
cp -p "${WTB_ENV_FILE}" "${ENV_BACKUP}"

update_env_variable() {
    local key="$1"
    local value="$2"
    local file="$3"

    if grep -q "^${key}=" "${file}"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
    else
        printf '%s=%s\n' "${key}" "${value}" >> "${file}"
    fi
}

update_env_variable "QWEN3_EMBEDDING_BASE_URL" "http://localhost:${EMBEDDING_PORT}/v1" "${WTB_ENV_FILE}"
update_env_variable "QWEN3_EMBEDDING_MODEL" "${EMBEDDING_LORA_MODEL}" "${WTB_ENV_FILE}"

EMBED_SERVER_PID=""
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ -n "${EMBED_SERVER_PID}" ]] && kill -0 "${EMBED_SERVER_PID}" 2>/dev/null; then
        kill "${EMBED_SERVER_PID}" 2>/dev/null || true
        wait "${EMBED_SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -f "${ENV_BACKUP}" ]]; then
        cp -p "${ENV_BACKUP}" "${WTB_ENV_FILE}"
        rm -f "${ENV_BACKUP}"
    fi
    exit "${exit_code}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

source "${BENCH_VENV}/bin/activate"
export MODEL_NAME="${EMBEDDING_MODEL}"
export VLLM_EMBEDDING_PORT="${EMBEDDING_PORT}"
export GPU_MEMORY_UTILIZATION="${EMBEDDING_GPU_MEMORY_UTILIZATION}"
export EMBEDDING_LORA_ADAPTER_DIR
export EMBEDDING_LORA_MODEL
export EMBEDDING_LORA_MAX_RANK="${EMBEDDING_LORA_MAX_RANK:-16}"

echo "Job:             ${JOB_ID}"
echo "Host:            $(hostname)"
echo "Base model:      ${EMBEDDING_MODEL}"
echo "Adapter:         ${EMBEDDING_LORA_ADAPTER_DIR}"
echo "Served as:       ${EMBEDDING_LORA_MODEL}"
echo "Tool cache:      ${CACHE_FILE}"
echo "Server log:      ${SERVER_LOG}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" >"${SERVER_LOG}" 2>&1 &
EMBED_SERVER_PID=$!

echo "Waiting for adapter endpoint..."
MAX_ATTEMPTS=120
for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    if ! kill -0 "${EMBED_SERVER_PID}" 2>/dev/null; then
        echo "[ERROR] Embedding server exited before becoming ready." >&2
        tail -100 "${SERVER_LOG}" || true
        exit 1
    fi
    if curl -sf "http://localhost:${EMBEDDING_PORT}/v1/models" | grep -Fq "${EMBEDDING_LORA_MODEL}"; then
        break
    fi
    if [[ "${attempt}" -eq "${MAX_ATTEMPTS}" ]]; then
        echo "[ERROR] Adapter endpoint did not become ready." >&2
        tail -100 "${SERVER_LOG}" || true
        exit 1
    fi
    sleep 5
done

curl -sf \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBEDDING_LORA_MODEL}\",\"input\":\"PORTS adapter health check\"}" \
    "http://localhost:${EMBEDDING_PORT}/v1/embeddings" >/dev/null

cd "${BENCHMARK_ROOT}"
echo "Precomputing tool vectors with ${EMBEDDING_LORA_MODEL}..."
python -u wtb/model_handler/api_inference/setup_openai_embeddings.py \
    --provider qwen3 \
    --tools-file "${TOOLS_FILE}" \
    --cache-file "${CACHE_FILE}"

echo "Offline PORTS-LoRA embedding cache completed: ${CACHE_FILE}"