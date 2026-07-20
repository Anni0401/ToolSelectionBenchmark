#!/bin/bash
# Deployment script for Qwen3-Reranker via FastAPI and Transformers.
# Serves an OpenAI-compatible /v1/score endpoint on a dedicated port.
#
# Loads the sequence-classification model directly on the GPU, avoiding the
# vLLM version-specific score and --task compatibility issues.
#
# Usage:
#   bash deploy/slurm_vllm_reranker_deploy.sh
#
# Override defaults via env vars, e.g.:
#   MODEL_NAME=Qwen/Qwen3-Reranker-8B VLLM_RERANKER_PORT=8003 \
#     bash deploy/slurm_vllm_reranker_deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# ── Activate virtual environment ──────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
    echo "[INFO] Activated uv venv: ${VENV_DIR}"
else
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}."
    echo "        Run 'bash deploy/uv_setup.sh --vllm' first."
    exit 1
fi

if ! python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "[ERROR] Python 3.10+ is required."
    exit 1
fi

# ── Configuration ─────────────────────────────────────────────────────────────
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Reranker-4B}"
PORT="${VLLM_RERANKER_PORT:-8003}"
GPU_MEMORY_UTILIZATION=0.5
DTYPE="${DTYPE:-float16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LOG_DIR="${PROJECT_ROOT}/logs"
export CHECKPOINT_DIR="${HOME}/.cache/huggingface/hub"

# Force vLLM v0 engine — the v1 engine (default in vLLM >= 0.6) crashes on
# cross-encoder models with EngineDeadError / AsyncLLM output_handler failures.
export VLLM_USE_V1=0

mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

echo "=========================================="
echo "vLLM Qwen3-Reranker-8B Deployment"
echo "=========================================="
echo "Model:                  ${MODEL_NAME}"
echo "Port:                   ${PORT}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Data Type:              ${DTYPE}"
echo "Tensor Parallel Size:   ${TENSOR_PARALLEL_SIZE}"
echo "Checkpoint Dir:         ${CHECKPOINT_DIR}"
echo "Log Dir:                ${LOG_DIR}"
echo "=========================================="

# ── Check GPU availability ────────────────────────────────────────────────────
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. GPU not available."
    exit 1
fi
echo "[INFO] GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ── Check FastAPI/Transformers are installed ─────────────────────────────────
if ! python -c "import fastapi, transformers, torch" 2>/dev/null; then
    echo "[ERROR] FastAPI, Transformers, or PyTorch is missing in ${VENV_DIR}."
    echo "        Install with: pip install fastapi uvicorn transformers torch"
    exit 1
fi


# ── Download / verify model weights ──────────────────────────────────────────
echo "[INFO] Verifying model weights for ${MODEL_NAME}..."
python << 'PYTHON_EOF'
import os
from transformers import AutoTokenizer

model_name  = os.environ.get("MODEL_NAME",     "Qwen/Qwen3-Reranker-8B")
cache_dir   = os.environ.get("CHECKPOINT_DIR", os.path.expanduser("~/.cache/huggingface/hub"))

print(f"[INFO] Downloading / verifying tokenizer for {model_name} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"[SUCCESS] Tokenizer OK – vocab size: {tokenizer.vocab_size}")
    print(f"[INFO]    Full model weights will be downloaded by vLLM on first start.")
except Exception as e:
    print(f"[WARNING] Could not verify tokenizer: {e}")
    print(f"[INFO]    vLLM will attempt to download everything automatically.")
PYTHON_EOF


HOSTNAME=$(hostname -f)
RERANKER_ENDPOINT="http://${HOSTNAME}:${PORT}/v1"

ENDPOINT_FILE="${LOG_DIR}/reranker_endpoint.txt"
cat > "${ENDPOINT_FILE}" << EOF
Reranker server: ${RERANKER_ENDPOINT}

Environment variables to set before running the LangGraph server:
  export QWEN3_RERANKER_BASE_URL=${RERANKER_ENDPOINT}
  export QWEN3_RERANKER_MODEL=${MODEL_NAME}
  export LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding_qwen3_reranker
EOF
echo "[INFO] Endpoint info saved to: ${ENDPOINT_FILE}"
cat "${ENDPOINT_FILE}"

# ── Start FastAPI reranker server ─────────────────────────────────────────────
# The model is loaded directly through Transformers on the GPU and the server
# implements the /v1/score contract expected by LangGraph.
echo ""
echo "[INFO] Starting FastAPI reranker server..."
echo "[INFO] Logs → ${LOG_DIR}/fastapi_reranker_server.log"
echo ""
export MODEL_NAME CHECKPOINT_DIR PORT
python "${SCRIPT_DIR}/fastapi_qwen3_reranker.py" \
    2>&1 | tee -a "${LOG_DIR}/fastapi_reranker_server.log"
