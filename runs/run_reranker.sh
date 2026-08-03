#!/bin/bash
#SBATCH --job-name=wtb-embedding-reranker
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

####################################################
# Project configuration
####################################################

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
BENCHMARK_ROOT="${PROJECT_ROOT}/wild-tool-bench"
BENCH_VENV="${PROJECT_ROOT}/.venv"

cd "${PROJECT_ROOT}"

HOST="$(hostname)"

echo "===================================================="
echo "Job ID:          ${SLURM_JOB_ID:-unknown}"
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "Benchmark root:  ${BENCHMARK_ROOT}"
echo "===================================================="

####################################################
# Model and server configuration
####################################################

EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
RERANKER_MODEL="${RERANKER_MODEL:-Qwen/Qwen3-Reranker-8B}"

EMBEDDING_PORT="${EMBEDDING_PORT:-8002}"
RERANKER_PORT="${RERANKER_PORT:-8003}"

SAIA_BASE_URL="${SAIA_BASE_URL:-https://chat-ai.academiccloud.de/v1}"
SAIA_MODEL="${SAIA_MODEL:-openai-gpt-oss-120b}"

####################################################
# Temporary and cache directories
####################################################

export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${WORK}/tmp_pip/cache"

export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

mkdir -p \
    "${TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}"

####################################################
# Hugging Face download settings
####################################################

# Disable Xet because it can fail during shard reconstruction
# on shared or parallel cluster filesystems.
export HF_HUB_DISABLE_XET=1

# Increase timeout for large model files.
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60

# Avoid tokenizer thread oversubscription.
export TOKENIZERS_PARALLELISM=false

####################################################
# NCCL configuration
####################################################

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

####################################################
# Validate environment
####################################################

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: WORK environment variable is not defined."
    exit 1
fi

if [[ ! -d "${BENCH_VENV}" ]]; then
    echo "ERROR: Virtual environment not found:"
    echo "       ${BENCH_VENV}"
    exit 1
fi

if [[ ! -f "${BENCH_VENV}/bin/activate" ]]; then
    echo "ERROR: Virtual environment activation script not found:"
    echo "       ${BENCH_VENV}/bin/activate"
    exit 1
fi

if [[ ! -d "${BENCHMARK_ROOT}" ]]; then
    echo "ERROR: Benchmark directory not found:"
    echo "       ${BENCHMARK_ROOT}"
    exit 1
fi

if [[ ! -f "${BENCHMARK_ROOT}/.env" ]]; then
    echo "ERROR: Environment file not found:"
    echo "       ${BENCHMARK_ROOT}/.env"
    exit 1
fi

####################################################
# Activate Python environment
####################################################

source "${BENCH_VENV}/bin/activate"

echo ""
echo "Python environment:"
echo "  Python: $(command -v python)"
echo "  Version: $(python --version)"
echo ""

####################################################
# Verify required Python packages
####################################################

python - <<'PY'
import importlib.util
import sys

required_packages = [
    "huggingface_hub",
    "transformers",
    "torch",
    "fastapi",
    "uvicorn",
    "overrides",
]

missing = [
    package
    for package in required_packages
    if importlib.util.find_spec(package) is None
]

if missing:
    print(
        "ERROR: Required packages are missing from the active environment:",
        ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Install them in the virtual environment before submitting the job.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

####################################################
# Print GPU information
####################################################

echo "Allocated GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader || true

echo ""
echo "Storage information:"
df -h "${WORK}" || true
df -i "${WORK}" || true
du -sh "${HF_HOME}" 2>/dev/null || true
quota -s 2>/dev/null || true
echo ""

####################################################
# Update benchmark .env
####################################################

ENV_FILE="${BENCHMARK_ROOT}/.env"

update_env_variable() {
    local key="$1"
    local value="$2"
    local file="$3"

    if grep -q "^${key}=" "${file}"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
    else
        echo "${key}=${value}" >> "${file}"
    fi
}

update_env_variable \
    "EXECUTING_LLM_BASE_URL" \
    "${SAIA_BASE_URL}" \
    "${ENV_FILE}"

update_env_variable \
    "EXECUTING_LLM_MODEL" \
    "${SAIA_MODEL}" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_EMBEDDING_BASE_URL" \
    "http://${HOST}:${EMBEDDING_PORT}/v1" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_EMBEDDING_MODEL" \
    "${EMBEDDING_MODEL}" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_RERANKER_BASE_URL" \
    "http://${HOST}:${RERANKER_PORT}/v1" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_RERANKER_MODEL" \
    "${RERANKER_MODEL}" \
    "${ENV_FILE}"

update_env_variable \
    "LANGGRAPH_TOOL_SELECTION_MODE" \
    "qwen3_embedding_qwen3_reranker" \
    "${ENV_FILE}"

echo "Updated ${ENV_FILE}"

####################################################
# Cleanup handling
####################################################

EMBED_PID=""
RERANK_PID=""
LANGGRAPH_PID=""

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    echo ""
    echo "===================================================="
    echo "Cleaning up background processes..."
    echo "===================================================="

    if [[ -n "${LANGGRAPH_PID}" ]] &&
       kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
        echo "Stopping LangGraph process ${LANGGRAPH_PID}"
        kill "${LANGGRAPH_PID}" 2>/dev/null || true
    fi

    if [[ -n "${EMBED_PID}" ]] &&
       kill -0 "${EMBED_PID}" 2>/dev/null; then
        echo "Stopping embedding server ${EMBED_PID}"
        kill "${EMBED_PID}" 2>/dev/null || true
    fi

    if [[ -n "${RERANK_PID}" ]] &&
       kill -0 "${RERANK_PID}" 2>/dev/null; then
        echo "Stopping reranker server ${RERANK_PID}"
        kill "${RERANK_PID}" 2>/dev/null || true
    fi

    wait "${LANGGRAPH_PID}" 2>/dev/null || true
    wait "${EMBED_PID}" 2>/dev/null || true
    wait "${RERANK_PID}" 2>/dev/null || true

    echo "Cleanup complete."

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

####################################################
# Download models sequentially
####################################################

echo ""
echo "===================================================="
echo "Preparing Hugging Face models"
echo "===================================================="

download_model() {
    local model_name="$1"

    echo ""
    echo "Checking model: ${model_name}"

    python - "${model_name}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_name = sys.argv[1]

print(f"Downloading or validating {model_name}...")

path = snapshot_download(
    repo_id=model_name,
    local_files_only=False,
    resume_download=True,
)

print(f"Model ready at: {path}")
PY
}

# These downloads happen sequentially to avoid concurrent writes
# to the Hugging Face cache.
download_model "${EMBEDDING_MODEL}"
download_model "${RERANKER_MODEL}"

echo ""
echo "All model files are available."

####################################################
# Helper: wait for HTTP service
####################################################

wait_for_service() {
    local service_name="$1"
    local health_url="$2"
    local process_id="$3"
    local timeout_seconds="${4:-1800}"

    local elapsed=0
    local interval=5

    echo "Waiting for ${service_name}..."
    echo "Health URL: ${health_url}"

    while true; do
        if curl --connect-timeout 3 \
                --max-time 5 \
                -sf "${health_url}" >/dev/null; then
            echo "${service_name} is ready."
            return 0
        fi

        if ! kill -0 "${process_id}" 2>/dev/null; then
            echo ""
            echo "ERROR: ${service_name} exited during startup."
            wait "${process_id}" || true
            return 1
        fi

        if (( elapsed >= timeout_seconds )); then
            echo ""
            echo "ERROR: ${service_name} did not become ready within"
            echo "       ${timeout_seconds} seconds."
            return 1
        fi

        sleep "${interval}"
        elapsed=$((elapsed + interval))

        if (( elapsed % 60 == 0 )); then
            echo "Still waiting for ${service_name}..."
            nvidia-smi --query-gpu=index,memory.used,memory.free \
                --format=csv,noheader || true
        fi
    done
}

####################################################
# Ensure GPU state has stabilized
####################################################

echo "GPU processes before embedding startup:"
nvidia-smi

# Give any process from an earlier cleanup time to release CUDA memory.
sleep 15



####################################################
# Start embedding server
####################################################

python -c "import vllm; print(vllm.__version__)"

echo "Starting embedding server..."

source "${BENCH_VENV}/bin/activate"

CUDA_VISIBLE_DEVICES=0 \
MODEL_NAME="${EMBEDDING_MODEL}" \
GPU_MEMORY_UTILIZATION=0.4 \
VLLM_EMBEDDING_PORT="${EMBEDDING_PORT}" \
bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" &

EMBED_PID=$!

wait_for_service \
    "embedding server" \
    "http://${HOST}:${EMBEDDING_PORT}/v1/models" \
    "${EMBED_PID}" \
    1800

echo "Embedding server fully initialized."

####################################################
# Start reranker only afterwards
####################################################

CUDA_VISIBLE_DEVICES=1 \
MODEL_NAME="${RERANKER_MODEL}" \
VLLM_RERANKER_PORT="${RERANKER_PORT}" \
bash "${PROJECT_ROOT}/deploy/slurm_vllm_reranker_deploy.sh" &

RERANK_PID=$!

wait_for_service \
    "reranker server" \
    "http://${HOST}:${RERANKER_PORT}/health" \
    "${RERANK_PID}" \
    1800
####################################################
# Verify both processes are still alive
####################################################

if ! kill -0 "${EMBED_PID}" 2>/dev/null; then
    echo "ERROR: Embedding server is no longer running."
    exit 1
fi

if ! kill -0 "${RERANK_PID}" 2>/dev/null; then
    echo "ERROR: Reranker server is no longer running."
    exit 1
fi

echo ""
echo "GPU usage after model startup:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free \
    --format=csv,noheader || true

####################################################
# Start LangGraph
####################################################

echo ""
echo "===================================================="
echo "Starting LangGraph"
echo "===================================================="

cd "${BENCHMARK_ROOT}"

python -u -m wtb.model_handler.api_inference.langgraph_app &

LANGGRAPH_PID=$!

echo "LangGraph PID: ${LANGGRAPH_PID}"

####################################################
# Wait for LangGraph startup
####################################################

LANGGRAPH_STARTUP_WAIT="${LANGGRAPH_STARTUP_WAIT:-15}"

sleep "${LANGGRAPH_STARTUP_WAIT}"

if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
    echo "ERROR: LangGraph exited during startup."
    wait "${LANGGRAPH_PID}" || true
    exit 1
fi

echo "LangGraph process is running."

####################################################
# Run benchmark
####################################################

echo ""
echo "===================================================="
echo "Running benchmark"
echo "===================================================="

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir result/reranker \
    --num-threads 1

echo ""
echo "===================================================="
echo "Benchmark completed successfully"
echo "===================================================="