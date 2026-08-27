#!/bin/bash

#SBATCH --job-name=wtb-embedding-laguna
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --mem=180G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -Eeuo pipefail


####################################################
# Paths
####################################################

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
BENCHMARK_ROOT="${PROJECT_ROOT}/wild-tool-bench"
ENV_FILE="${BENCHMARK_ROOT}/.env"

cd "${PROJECT_ROOT}"

: "${WORK:?ERROR: WORK environment variable is not set}"


####################################################
# Environment / caches
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

export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false


####################################################
# NCCL
####################################################

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL


####################################################
# Laguna-specific configuration
####################################################

export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_ENGINE_READY_TIMEOUT_S=1800


####################################################
# Models / virtual environments
####################################################

LAGUNA_VENV="${WORK}/venvs/venv-laguna"
BENCH_VENV="${PROJECT_ROOT}/.venv"

LAGUNA_MODEL="poolside/Laguna-S-2.1-FP8"


####################################################
# Ports
####################################################

LAGUNA_PORT="${LAGUNA_PORT:-8000}"
LANGGRAPH_PORT="${LANGGRAPH_PORT:-8001}"
EMBED_PORT="${EMBED_PORT:-8002}"

HOST="$(hostname)"


####################################################
# Job information
####################################################

echo "===================================================="
echo "Job ID:          ${SLURM_JOB_ID:-unknown}"
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "Laguna model:    ${LAGUNA_MODEL}"
echo "===================================================="

echo ""
echo "Allocated GPUs:"

nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader


####################################################
# Validation
####################################################

if [[ ! -f "${LAGUNA_VENV}/bin/activate" ]]; then
    echo "ERROR: Laguna virtual environment not found:"
    echo "       ${LAGUNA_VENV}"
    exit 1
fi

if [[ ! -f "${BENCH_VENV}/bin/activate" ]]; then
    echo "ERROR: Benchmark virtual environment not found:"
    echo "       ${BENCH_VENV}"
    exit 1
fi

if [[ ! -d "${BENCHMARK_ROOT}" ]]; then
    echo "ERROR: Benchmark directory not found:"
    echo "       ${BENCHMARK_ROOT}"
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: .env file not found:"
    echo "       ${ENV_FILE}"
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" ]]; then
    echo "ERROR: Embedding deployment script not found:"
    echo "       ${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh"
    exit 1
fi


####################################################
# Helper: update .env
####################################################

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


####################################################
# Configure benchmark
####################################################

update_env_variable \
    "EXECUTING_LLM_BASE_URL" \
    "http://${HOST}:${LAGUNA_PORT}/v1" \
    "${ENV_FILE}"

update_env_variable \
    "EXECUTING_LLM_MODEL" \
    "${LAGUNA_MODEL}" \
    "${ENV_FILE}"

update_env_variable \
    "EXECUTING_LLM_API_KEY" \
    "EMPTY" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_EMBEDDING_BASE_URL" \
    "http://${HOST}:${EMBED_PORT}/v1" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_EMBEDDING_API_KEY" \
    "EMPTY" \
    "${ENV_FILE}"

update_env_variable \
    "LANGGRAPH_TOOL_SELECTION_MODE" \
    "qwen3_embedding_context" \
    "${ENV_FILE}"

update_env_variable \
    "LANGGRAPH_ENDPOINT" \
    "http://127.0.0.1:${LANGGRAPH_PORT}/execute" \
    "${ENV_FILE}"

# Important for Laguna-specific tool-call normalization
update_env_variable \
    "EXECUTING_LLM_TOOL_CALL_PARSER" \
    "poolside_v1" \
    "${ENV_FILE}"

echo "Updated ${ENV_FILE}"


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

    echo ""
    echo "Waiting for ${service_name}..."
    echo "Health URL: ${health_url}"

    while true; do

        if curl \
            --connect-timeout 3 \
            --max-time 5 \
            -sf "${health_url}" >/dev/null; then

            echo "${service_name} ready."
            return 0
        fi

        if ! kill -0 "${process_id}" 2>/dev/null; then
            echo "ERROR: ${service_name} exited during startup."
            wait "${process_id}" || true
            return 1
        fi

        if (( elapsed >= timeout_seconds )); then
            echo "ERROR: ${service_name} did not become ready"
            echo "       within ${timeout_seconds} seconds."
            return 1
        fi

        sleep "${interval}"
        elapsed=$((elapsed + interval))

        if (( elapsed % 60 == 0 )); then
            echo "Still waiting for ${service_name} (${elapsed}s)..."

            nvidia-smi \
                --query-gpu=index,memory.used,memory.free \
                --format=csv,noheader || true
        fi
    done
}


####################################################
# Cleanup
####################################################

LAGUNA_PID=""
EMBED_PID=""
LANGGRAPH_PID=""

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    echo ""
    echo "===================================================="
    echo "Cleaning up..."
    echo "===================================================="

    for pid_name in LANGGRAPH_PID EMBED_PID LAGUNA_PID; do

        local pid="${!pid_name:-}"

        if [[ -n "${pid}" ]] &&
           kill -0 "${pid}" 2>/dev/null; then

            echo "Stopping ${pid_name}: ${pid}"
            kill "${pid}" 2>/dev/null || true
        fi
    done

    sleep 3

    for pid_name in LANGGRAPH_PID EMBED_PID LAGUNA_PID; do

        local pid="${!pid_name:-}"

        if [[ -n "${pid}" ]] &&
           kill -0 "${pid}" 2>/dev/null; then

            echo "Force-stopping ${pid_name}: ${pid}"
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done

    [[ -n "${LANGGRAPH_PID}" ]] &&
        wait "${LANGGRAPH_PID}" 2>/dev/null || true

    [[ -n "${EMBED_PID}" ]] &&
        wait "${EMBED_PID}" 2>/dev/null || true

    [[ -n "${LAGUNA_PID}" ]] &&
        wait "${LAGUNA_PID}" 2>/dev/null || true

    echo "Cleanup complete."

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM


####################################################
# Start Laguna on GPUs 0 + 1
####################################################

echo ""
echo "===================================================="
echo "Starting Laguna on GPUs 0 and 1"
echo "===================================================="

source "${LAGUNA_VENV}/bin/activate"

echo "Laguna environment:"
echo "Python: $(command -v python)"
python --version

echo "vLLM:"
python -c "import vllm; print(vllm.__version__)"

CUDA_VISIBLE_DEVICES=0,1 \
vllm serve "${LAGUNA_MODEL}" \
    --served-model-name "${LAGUNA_MODEL}" \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser poolside_v1 \
    --reasoning-parser poolside_v1 \
    --host 0.0.0.0 \
    --port "${LAGUNA_PORT}" &

LAGUNA_PID=$!

echo "Laguna PID: ${LAGUNA_PID}"

wait_for_service \
    "Laguna" \
    "http://127.0.0.1:${LAGUNA_PORT}/v1/models" \
    "${LAGUNA_PID}" \
    1800


####################################################
# Start Qwen embedding server on GPU 2
####################################################

echo ""
echo "===================================================="
echo "Starting Qwen3 Embedding on GPU 2"
echo "===================================================="

# Switch to benchmark venv, where your embedding server dependencies live.
deactivate 2>/dev/null || true
source "${BENCH_VENV}/bin/activate"

CUDA_VISIBLE_DEVICES=2 \
HF_HOME="${HF_HOME}" \
HF_HUB_CACHE="${HF_HUB_CACHE}" \
HF_XET_CACHE="${HF_XET_CACHE}" \
HF_HUB_DISABLE_XET=1 \
bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" &

EMBED_PID=$!

echo "Embedding PID: ${EMBED_PID}"

wait_for_service \
    "Qwen3 embedding server" \
    "http://127.0.0.1:${EMBED_PORT}/v1/models" \
    "${EMBED_PID}" \
    1800


####################################################
# Check GPU state
####################################################

echo ""
echo "===================================================="
echo "GPU usage after model startup"
echo "===================================================="

nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free \
    --format=csv,noheader


####################################################
# Switch to benchmark environment
####################################################

cd "${BENCHMARK_ROOT}"

deactivate 2>/dev/null || true
source "${BENCH_VENV}/bin/activate"

echo ""
echo "Benchmark environment:"
echo "Python: $(command -v python)"
python --version


####################################################
# Ensure LangGraph port is clear
####################################################

echo "Cleaning stale LangGraph process..."

fuser -k "${LANGGRAPH_PORT}/tcp" 2>/dev/null || true


####################################################
# Start LangGraph
####################################################

echo ""
echo "===================================================="
echo "Starting LangGraph"
echo "===================================================="

python -u -m wtb.model_handler.api_inference.langgraph_app &

LANGGRAPH_PID=$!

echo "LangGraph PID: ${LANGGRAPH_PID}"

echo "Waiting for LangGraph..."

sleep 15

if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
    echo "ERROR: LangGraph exited during startup."
    wait "${LANGGRAPH_PID}" || true
    exit 1
fi

echo "LangGraph ready."


####################################################
# Final service health checks
####################################################

curl -sf \
    "http://127.0.0.1:${LAGUNA_PORT}/v1/models" \
    >/dev/null || {
        echo "ERROR: Laguna health check failed."
        exit 1
    }

curl -sf \
    "http://127.0.0.1:${EMBED_PORT}/v1/models" \
    >/dev/null || {
        echo "ERROR: Embedding health check failed."
        exit 1
    }

echo "All services healthy."


####################################################
# Run benchmark
####################################################

echo ""
echo "===================================================="
echo "Running Laguna + Qwen3 embedding benchmark"
echo "===================================================="

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir result_laguna/embedding_context \
    --num-threads 1


echo ""
echo "===================================================="
echo "Benchmark completed successfully"
echo "===================================================="