#!/bin/bash
#SBATCH --job-name=wtb-embedding-ports-lora
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -Eeuo pipefail

####################################################
# Determine project root
####################################################

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
BENCHMARK_ROOT="${PROJECT_ROOT}/wild-tool-bench"
ENV_FILE="${BENCHMARK_ROOT}/.env"

cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

####################################################
# Environment
####################################################

: "${WORK:?ERROR: WORK environment variable is not set}"

export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${WORK}/tmp_pip/cache"

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

####################################################
# Hugging Face cache
####################################################

export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

mkdir -p \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}"

# Avoid hf-xet file reconstruction failures on cluster filesystems.
export HF_HUB_DISABLE_XET=1

# Increase timeouts for large checkpoint shards.
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60

# Avoid excessive tokenizer worker threads.
export TOKENIZERS_PARALLELISM=false

####################################################
# NCCL configuration
####################################################

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

####################################################
# Virtual environments and models
####################################################

GPT_VENV="${WORK}/venvs/venv-gptoss"
BENCH_VENV="${PROJECT_ROOT}/.venv"

GPT_MODEL="${WORK}/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
# Served-model-name the PORTS LoRA adapter is registered under on the embedding engine.
EMBEDDING_LORA_MODEL="${EMBEDDING_LORA_MODEL:-qwen3-embedding-8b-ports-lora}"
# PEFT adapter dir (base_model_name_or_path=Qwen/Qwen3-Embedding-8B) produced by PORTS training.
EMBEDDING_LORA_ADAPTER_DIR="${EMBEDDING_LORA_ADAPTER_DIR:-${WORK}/ports/main/output/ports/ports_retriever_20261006_233629/checkpoint-epoch-2}"
EMBEDDING_LORA_MAX_RANK="${EMBEDDING_LORA_MAX_RANK:-16}"

GPT_PORT="${GPT_PORT:-8000}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8002}"

HOST="$(hostname)"

echo "===================================================="
echo "Job ID:          ${SLURM_JOB_ID:-unknown}"
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "GPT model:       ${GPT_MODEL}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "Embedding LoRA:  ${EMBEDDING_LORA_MODEL} (${EMBEDDING_LORA_ADAPTER_DIR})"
echo "===================================================="

####################################################
# Validate paths
####################################################

if [[ ! -f "${GPT_VENV}/bin/activate" ]]; then
    echo "ERROR: GPT virtual environment not found:"
    echo "       ${GPT_VENV}"
    exit 1
fi

if [[ ! -f "${BENCH_VENV}/bin/activate" ]]; then
    echo "ERROR: Benchmark virtual environment not found:"
    echo "       ${BENCH_VENV}"
    exit 1
fi

if [[ ! -d "${GPT_MODEL}" ]]; then
    echo "ERROR: GPT model snapshot does not exist:"
    echo "       ${GPT_MODEL}"
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

if [[ ! -f "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_config.json" || ! -f "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "ERROR: Embedding LoRA adapter not found:"
    echo "       ${EMBEDDING_LORA_ADAPTER_DIR}"
    exit 1
fi

####################################################
# Print allocation information
####################################################

echo ""
echo "Allocated GPUs:"
nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader || true

echo ""
echo "Cache filesystem:"
df -h "${WORK}" || true
df -i "${WORK}" || true
du -sh "${HF_HOME}" 2>/dev/null || true
quota -s 2>/dev/null || true

####################################################
# Update .env safely
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

update_env_variable \
    "EXECUTING_LLM_BASE_URL" \
    "http://${HOST}:${GPT_PORT}/v1" \
    "${ENV_FILE}"

update_env_variable \
    "EXECUTING_LLM_MODEL" \
    "openai/gpt-oss-120b" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_EMBEDDING_BASE_URL" \
    "http://${HOST}:${EMBEDDING_PORT}/v1" \
    "${ENV_FILE}"

update_env_variable \
    "QWEN3_EMBEDDING_MODEL" \
    "${EMBEDDING_LORA_MODEL}" \
    "${ENV_FILE}"

update_env_variable \
    "LANGGRAPH_TOOL_SELECTION_MODE" \
    "qwen3_embedding_context_ports_lora" \
    "${ENV_FILE}"

echo ""
echo "Updated ${ENV_FILE}"

####################################################
# Background process variables
####################################################

GPT_PID=""
EMBED_PID=""
LANGGRAPH_PID=""

####################################################
# Cleanup
####################################################

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    echo ""
    echo "===================================================="
    echo "Cleaning up..."
    echo "===================================================="

    for pid_name in LANGGRAPH_PID EMBED_PID GPT_PID; do
        local pid="${!pid_name:-}"

        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "Stopping ${pid_name}: ${pid}"
            kill "${pid}" 2>/dev/null || true
        fi
    done

    # Give the servers a moment to terminate normally.
    sleep 3

    for pid_name in LANGGRAPH_PID EMBED_PID GPT_PID; do
        local pid="${!pid_name:-}"

        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "Force-stopping ${pid_name}: ${pid}"
            kill -9 "${pid}" 2>/dev/null || true
        fi
    done

    [[ -n "${LANGGRAPH_PID}" ]] && wait "${LANGGRAPH_PID}" 2>/dev/null || true
    [[ -n "${EMBED_PID}" ]] && wait "${EMBED_PID}" 2>/dev/null || true
    [[ -n "${GPT_PID}" ]] && wait "${GPT_PID}" 2>/dev/null || true

    echo "Cleanup complete."

    exit "${exit_code}"
}

on_error() {
    local exit_code=$?
    echo "" >&2
    echo "[ERROR] Two-GPU startup script failed." >&2
    echo "[ERROR] Exit code: ${exit_code}" >&2
    echo "[ERROR] Line: ${BASH_LINENO[0]:-${LINENO}}" >&2
    echo "[ERROR] Command: ${BASH_COMMAND}" >&2
    return "${exit_code}"
}

trap on_error ERR
trap cleanup EXIT INT TERM

####################################################
# Helper: download model sequentially
####################################################

download_model() {
    local model_name="$1"

    echo ""
    echo "Preparing model: ${model_name}"

    python - "${model_name}" <<'PY'
import sys
from huggingface_hub import snapshot_download

model_name = sys.argv[1]

path = snapshot_download(
    repo_id=model_name,
    local_files_only=False,
)

print(f"Model ready: {model_name}")
print(f"Snapshot path: {path}")
PY
}

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
    echo "URL: ${health_url}"

    while true; do
        if curl \
            --connect-timeout 3 \
            --max-time 5 \
            -sf \
            "${health_url}" >/dev/null; then

            echo "${service_name} ready."
            return 0
        fi

        if ! kill -0 "${process_id}" 2>/dev/null; then
            echo ""
            echo "ERROR: ${service_name} exited during startup."

            # wait prints the process exit status when applicable.
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
            echo "Still waiting for ${service_name} (${elapsed}s)..."

            nvidia-smi \
                --query-gpu=index,memory.used,memory.free \
                --format=csv,noheader || true
        fi
    done
}

####################################################
# Prepare benchmark environment
####################################################

echo ""
echo "===================================================="
echo "Activating benchmark environment"
echo "===================================================="

source "${BENCH_VENV}/bin/activate"

echo "Python executable: $(command -v python)"
python --version

####################################################
# Verify required packages
####################################################

python - <<'PY'
import importlib.util
import sys

required = [
    "huggingface_hub",
    "transformers",
    "torch",
    "fastapi",
    "uvicorn",
    "overrides",
]

missing = [
    package
    for package in required
    if importlib.util.find_spec(package) is None
]

if missing:
    print(
        "ERROR: Missing packages in benchmark environment: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Install them before submitting the SLURM job.",
        file=sys.stderr,
    )
    sys.exit(1)
PY

####################################################
# Download embedding model
####################################################

echo ""
echo "===================================================="
echo "Preparing embedding model files"
echo "===================================================="

download_model "${EMBEDDING_MODEL}"

echo ""
echo "Embedding model is available locally."

####################################################
# Start GPT-OSS on GPU 0
####################################################

echo ""
echo "===================================================="
echo "Starting GPT-OSS on GPU 0"
echo "===================================================="

source "${GPT_VENV}/bin/activate"

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_DISABLE_XET=1 \
vllm serve "${GPT_MODEL}" \
    --served-model-name openai/gpt-oss-120b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port "${GPT_PORT}" \
    --tool-call-parser openai \
    --enable-auto-tool-choice &

GPT_PID=$!

echo "GPT-OSS PID: ${GPT_PID}"

wait_for_service \
    "GPT-OSS" \
    "http://${HOST}:${GPT_PORT}/v1/models" \
    "${GPT_PID}" \
    1800

####################################################
# Reactivate benchmark environment
####################################################

source "${BENCH_VENV}/bin/activate"

####################################################
# Start embedding server (with PORTS LoRA) on GPU 1
####################################################

echo ""
echo "===================================================="
echo "Starting embedding server with PORTS LoRA on GPU 1"
echo "===================================================="

CUDA_VISIBLE_DEVICES=1 \
HF_HOME="${HF_HOME}" \
HF_HUB_CACHE="${HF_HUB_CACHE}" \
HF_XET_CACHE="${HF_XET_CACHE}" \
HF_HUB_DISABLE_XET=1 \
MODEL_NAME="${EMBEDDING_MODEL}" \
EMBEDDING_LORA_ADAPTER_DIR="${EMBEDDING_LORA_ADAPTER_DIR}" \
EMBEDDING_LORA_MODEL="${EMBEDDING_LORA_MODEL}" \
EMBEDDING_LORA_MAX_RANK="${EMBEDDING_LORA_MAX_RANK}" \
GPU_MEMORY_UTILIZATION=0.40 \
VLLM_EMBEDDING_PORT="${EMBEDDING_PORT}" \
bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" &

EMBED_PID=$!

echo "Embedding server PID: ${EMBED_PID}"

wait_for_service \
    "embedding server" \
    "http://${HOST}:${EMBEDDING_PORT}/v1/models" \
    "${EMBED_PID}" \
    1800

# /v1/models is up before the first LoRA forward, so exercise the adapter explicitly.
curl -sf \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${EMBEDDING_LORA_MODEL}\",\"input\":\"PORTS adapter health check\"}" \
    "http://${HOST}:${EMBEDDING_PORT}/v1/embeddings" >/dev/null || {
    echo "ERROR: Embedding LoRA adapter '${EMBEDDING_LORA_MODEL}' failed an embedding request." >&2
    exit 1
}

echo "Embedding LoRA adapter answered an embedding request."

####################################################
# Verify all model services
####################################################

for service in \
    "GPT-OSS:${GPT_PID}" \
    "embedding server:${EMBED_PID}"; do

    service_name="${service%%:*}"
    service_pid="${service##*:}"

    if ! kill -0 "${service_pid}" 2>/dev/null; then
        echo "ERROR: ${service_name} is no longer running."
        exit 1
    fi
done

echo ""
echo "All model servers are running."

echo ""
echo "GPU usage after model startup:"
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.free \
    --format=csv,noheader || true

####################################################
# Start LangGraph
####################################################

echo ""
echo "===================================================="
echo "Starting LangGraph"
echo "===================================================="

cd "${BENCHMARK_ROOT}"

source "${BENCH_VENV}/bin/activate"

# Benchmark results and server-side logs share this dir.
RESULT_DIR="${RESULT_DIR:-result_v3_120B/embedding_context_ports_lora}"
export LANGGRAPH_RESULT_DIR="${RESULT_DIR}"
echo "Result dir: ${RESULT_DIR}"

python -u -m wtb.model_handler.api_inference.langgraph_app &

LANGGRAPH_PID=$!

echo "LangGraph PID: ${LANGGRAPH_PID}"

####################################################
# Wait for LangGraph
####################################################

echo "Waiting for LangGraph..."

sleep 15

if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
    echo "ERROR: LangGraph exited during startup."
    wait "${LANGGRAPH_PID}" || true
    exit 1
fi

echo "LangGraph process is running."

####################################################
# Final server check
####################################################

curl -sf "http://${HOST}:${GPT_PORT}/v1/models" >/dev/null || {
    echo "ERROR: GPT-OSS failed its final health check." >&2
    exit 1
}
curl -sf "http://${HOST}:${EMBEDDING_PORT}/v1/models" >/dev/null || {
    echo "ERROR: Embedding server failed its final health check." >&2
    exit 1
}

echo "All service health checks passed."

####################################################
# Run benchmark
####################################################

echo ""
echo "===================================================="
echo "Running benchmark (PORTS LoRA embedding-context selector)"
echo "===================================================="

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir "${RESULT_DIR}" \
    --num-threads 1

echo ""
echo "===================================================="
echo "Benchmark completed successfully"
echo "===================================================="