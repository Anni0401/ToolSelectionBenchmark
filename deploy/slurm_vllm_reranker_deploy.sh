#!/bin/bash
# Deployment script for Qwen3-Reranker via FastAPI and Transformers.
# Serves an OpenAI-compatible /v1/score endpoint on a dedicated port.

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
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Reranker-8B}"

PORT="${VLLM_RERANKER_PORT:-8003}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
DTYPE="${DTYPE:-float16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

LOG_DIR="${PROJECT_ROOT}/logs"

# Keep the reranker on the same Hugging Face cache as the parent startup script.
export HF_HOME="${HF_HOME:-${WORK}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"

# CHECKPOINT_DIR must match HF_HUB_CACHE.
export CHECKPOINT_DIR="${HF_HUB_CACHE}"

export TMPDIR="${TMPDIR:-${WORK}/tmp}"

# Avoid Xet reconstruction errors on the shared cluster filesystem.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

# Prevent accidental network access during server startup after the
# parent script has already downloaded the complete model.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

mkdir -p \
    "${LOG_DIR}" \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}" \
    "${CHECKPOINT_DIR}" \
    "${TMPDIR}"

# Kept for compatibility, although this server loads via Transformers.
export VLLM_USE_V1=0

echo "=========================================="
echo "Qwen3-Reranker-8B Deployment"
echo "=========================================="
echo "Model:                  ${MODEL_NAME}"
echo "Port:                   ${PORT}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Data Type:              ${DTYPE}"
echo "Tensor Parallel Size:   ${TENSOR_PARALLEL_SIZE}"
echo "HF Home:                ${HF_HOME}"
echo "HF Hub Cache:           ${HF_HUB_CACHE}"
echo "Checkpoint Dir:         ${CHECKPOINT_DIR}"
echo "Offline mode:           ${HF_HUB_OFFLINE}"
echo "CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Log Dir:                ${LOG_DIR}"
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

# ── Check required packages ───────────────────────────────────────────────────
if ! python -c "import fastapi, transformers, torch" 2>/dev/null; then
    echo "[ERROR] FastAPI, Transformers, or PyTorch is missing in ${VENV_DIR}."
    echo "        Install with:"
    echo "        python -m pip install fastapi uvicorn transformers torch"
    exit 1
fi

# ── Verify complete model snapshot ────────────────────────────────────────────
echo "[INFO] Verifying complete model snapshot for ${MODEL_NAME}..."

python <<'PYTHON_EOF'
import os
import sys
from huggingface_hub import snapshot_download

model_name = os.environ["MODEL_NAME"]
cache_dir = os.environ["HF_HUB_CACHE"]

print(f"[INFO] Model: {model_name}")
print(f"[INFO] Cache: {cache_dir}")

try:
    snapshot_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=True,
    )
except Exception as exc:
    print(
        "[ERROR] The complete reranker model is not available in the "
        "configured Hugging Face cache.",
        file=sys.stderr,
    )
    print(f"[ERROR] Cache: {cache_dir}", file=sys.stderr)
    print(f"[ERROR] Details: {exc}", file=sys.stderr)
    print(
        "[ERROR] The parent startup script should download the model "
        "before launching this server.",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[SUCCESS] Complete model snapshot available at: {snapshot_path}")
PYTHON_EOF

# ── Write endpoint information ────────────────────────────────────────────────
HOSTNAME="$(hostname -f)"
RERANKER_ENDPOINT="http://${HOSTNAME}:${PORT}/v1"

ENDPOINT_FILE="${LOG_DIR}/reranker_endpoint.txt"

cat > "${ENDPOINT_FILE}" <<EOF
Reranker server: ${RERANKER_ENDPOINT}

Environment variables to set before running the LangGraph server:
  export QWEN3_RERANKER_BASE_URL=${RERANKER_ENDPOINT}
  export QWEN3_RERANKER_MODEL=${MODEL_NAME}
  export LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding_qwen3_reranker
EOF

echo "[INFO] Endpoint info saved to: ${ENDPOINT_FILE}"
cat "${ENDPOINT_FILE}"

# ── Start FastAPI reranker server ─────────────────────────────────────────────
echo ""
echo "[INFO] Starting FastAPI reranker server..."
echo "[INFO] Logs → ${LOG_DIR}/fastapi_reranker_server.log"
echo ""

export MODEL_NAME
export CHECKPOINT_DIR
export PORT
export DTYPE
export GPU_MEMORY_UTILIZATION

python "${SCRIPT_DIR}/fastapi_qwen3_reranker.py" \
    2>&1 | tee -a "${LOG_DIR}/fastapi_reranker_server.log"