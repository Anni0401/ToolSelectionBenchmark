#!/bin/bash
# Deployment script for Qwen3-Embedding-8B via vLLM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# ── Activate virtual environment ──────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
    echo "[INFO] Activated uv venv: ${VENV_DIR}"
else
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}."
    echo "        Run 'bash deploy/uv_setup.sh --vllm' first."
    exit 1
fi

if ! python -c \
    "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" \
    2>/dev/null; then

    echo "[ERROR] Python 3.10+ is required."
    exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
EMBEDDING_LORA_ADAPTER_DIR="${EMBEDDING_LORA_ADAPTER_DIR:-}"
EMBEDDING_LORA_MODEL="${EMBEDDING_LORA_MODEL:-qwen3-embedding-8b-ports-lora}"
EMBEDDING_LORA_MAX_RANK="${EMBEDDING_LORA_MAX_RANK:-16}"
SERVED_EMBEDDING_MODEL="${MODEL_NAME}"

PORT="${VLLM_EMBEDDING_PORT:-8002}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
DTYPE="${DTYPE:-float16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LORA_ARGS=()

if [[ -n "${EMBEDDING_LORA_ADAPTER_DIR}" ]]; then
    if [[ ! -f "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_config.json" || ! -f "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_model.safetensors" ]]; then
        echo "[ERROR] Embedding LoRA adapter files not found in: ${EMBEDDING_LORA_ADAPTER_DIR}" >&2
        exit 1
    fi

    python - "${EMBEDDING_LORA_ADAPTER_DIR}/adapter_config.json" "${MODEL_NAME}" "${EMBEDDING_LORA_MAX_RANK}" <<'PYTHON_EOF'
import json
import sys

config_path, model_name, max_rank = sys.argv[1:]
with open(config_path, encoding="utf-8") as config_file:
    config = json.load(config_file)

base_model = config.get("base_model_name_or_path")
rank = int(config.get("r", 0))
task_type = config.get("task_type")
if base_model != model_name:
    raise SystemExit(
        f"[ERROR] Adapter base model {base_model!r} does not match MODEL_NAME {model_name!r}"
    )
if rank > int(max_rank):
    raise SystemExit(
        f"[ERROR] Adapter rank {rank} exceeds EMBEDDING_LORA_MAX_RANK={max_rank}"
    )
if task_type != "FEATURE_EXTRACTION":
    raise SystemExit(
        f"[ERROR] Expected a FEATURE_EXTRACTION adapter, got {task_type!r}"
    )
print(f"[INFO] Adapter verified: base={base_model}, rank={rank}, task={task_type}")
PYTHON_EOF

    LORA_ARGS=(
        --enable-lora
        --max-lora-rank "${EMBEDDING_LORA_MAX_RANK}"
        --lora-modules "${EMBEDDING_LORA_MODEL}=${EMBEDDING_LORA_ADAPTER_DIR}"
    )
    SERVED_EMBEDDING_MODEL="${EMBEDDING_LORA_MODEL}"
    SERVED_SELECTION_MODE="qwen3_embedding_context_ports_lora"
else
    SERVED_SELECTION_MODE="qwen3_embedding"
fi

LOG_DIR="${PROJECT_ROOT}/logs"

# Keep every Hugging Face and vLLM download in the same cache.
export HF_HOME="${HF_HOME:-${WORK}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"

# vLLM --download-dir must point to the same Hugging Face hub cache.
export CHECKPOINT_DIR="${HF_HUB_CACHE}"

export TMPDIR="${TMPDIR:-${WORK}/tmp}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

mkdir -p \
    "${TMPDIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}" \
    "${CHECKPOINT_DIR}" \
    "${LOG_DIR}"

# Force vLLM v0 engine.
export VLLM_USE_V1=0

# The Triton interpreter segfaults in vLLM's LoRA kernels, so only use it without an adapter.
if [[ -z "${EMBEDDING_LORA_ADAPTER_DIR}" ]]; then
    export TRITON_INTERPRET=1
else
    unset TRITON_INTERPRET
fi

echo "=========================================="
echo "vLLM Qwen3-Embedding-8B Deployment"
echo "=========================================="
echo "Model:                  ${MODEL_NAME}"
echo "Embedding LoRA model:   ${EMBEDDING_LORA_MODEL}"
echo "Embedding LoRA adapter: ${EMBEDDING_LORA_ADAPTER_DIR:-disabled}"
echo "Port:                   ${PORT}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Data Type:              ${DTYPE}"
echo "Tensor Parallel Size:   ${TENSOR_PARALLEL_SIZE}"
echo "HF Home:                ${HF_HOME}"
echo "HF Hub Cache:           ${HF_HUB_CACHE}"
echo "Checkpoint Dir:         ${CHECKPOINT_DIR}"
echo "Log Dir:                ${LOG_DIR}"
echo "CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=========================================="

# ── Check GPU availability ────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[ERROR] nvidia-smi not found. GPU not available."
    exit 1
fi

echo "[INFO] GPU status:"
nvidia-smi \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv

# ── Check vLLM is installed ───────────────────────────────────────────────────
if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not found in ${VENV_DIR}."
    echo "        Run 'bash deploy/uv_setup.sh --vllm' to install it."
    exit 1
fi

echo "[INFO] vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

# ── Verify model files ────────────────────────────────────────────────────────
echo "[INFO] Verifying model files for ${MODEL_NAME}..."

python <<'PYTHON_EOF'
import os
from huggingface_hub import snapshot_download

model_name = os.environ["MODEL_NAME"]
cache_dir = os.environ["HF_HUB_CACHE"]

print(f"[INFO] Model: {model_name}")
print(f"[INFO] Cache: {cache_dir}")

path = snapshot_download(
    repo_id=model_name,
    cache_dir=cache_dir,
    local_files_only=True,
)

print(f"[SUCCESS] Model snapshot available at: {path}")
PYTHON_EOF

# ── Write endpoint information ────────────────────────────────────────────────
HOSTNAME="$(hostname -f)"
EMBEDDING_ENDPOINT="http://${HOSTNAME}:${PORT}/v1"

ENDPOINT_FILE="${LOG_DIR}/embedding_endpoint.txt"

cat > "${ENDPOINT_FILE}" <<EOF
Embedding server: ${EMBEDDING_ENDPOINT}

Environment variables to set before running the LangGraph server:
  export QWEN3_EMBEDDING_BASE_URL=${EMBEDDING_ENDPOINT}
    export QWEN3_EMBEDDING_MODEL=${SERVED_EMBEDDING_MODEL}
    export LANGGRAPH_TOOL_SELECTION_MODE=${SERVED_SELECTION_MODE}
EOF

echo "[INFO] Endpoint info saved to: ${ENDPOINT_FILE}"
cat "${ENDPOINT_FILE}"

# ── Start vLLM embedding server ───────────────────────────────────────────────
echo ""
echo "[INFO] Starting vLLM embedding server..."
echo "[INFO] Logs → ${LOG_DIR}/vllm_embedding_server.log"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${MODEL_NAME}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --enforce-eager \
    --download-dir "${CHECKPOINT_DIR}" \
    --trust-remote-code \
    "${LORA_ARGS[@]}" \
    2>&1 | tee -a "${LOG_DIR}/vllm_embedding_server.log"
