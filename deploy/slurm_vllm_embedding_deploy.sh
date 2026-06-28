#!/bin/bash
# Deployment script for Qwen3-Embedding-8B via vLLM
# Serves an OpenAI-compatible /v1/embeddings endpoint on a dedicated port.
#
# This is separate from slurm_vllm_deploy.sh (which hosts the LLM on port 8000).
# Run this on an interactive or SLURM-allocated GPU node.
#
# Usage:
#   bash deploy/slurm_vllm_embedding_deploy.sh
#
# Override defaults via env vars, e.g.:
#   MODEL_NAME=Qwen/Qwen3-Embedding-8B VLLM_EMBEDDING_PORT=8001 \
#     bash deploy/slurm_vllm_embedding_deploy.sh

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
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
PORT="${VLLM_EMBEDDING_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
DTYPE="${DTYPE:-float16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
LOG_DIR="${PROJECT_ROOT}/logs"
export CHECKPOINT_DIR="${HOME}/.cache/huggingface/hub"

mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

echo "=========================================="
echo "vLLM Qwen3-Embedding-8B Deployment"
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

# ── Check vLLM is installed ───────────────────────────────────────────────────
if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not found in ${VENV_DIR}."
    echo "        Run 'bash deploy/uv_setup.sh --vllm' to install it."
    exit 1
fi
echo "[INFO] vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

# ── Download / verify model weights ──────────────────────────────────────────
echo "[INFO] Verifying model weights for ${MODEL_NAME}..."
python << 'PYTHON_EOF'
import os
from transformers import AutoTokenizer

model_name  = os.environ.get("MODEL_NAME",     "Qwen/Qwen3-Embedding-8B")
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

# ── Write a convenience env-var snippet ──────────────────────────────────────
HOSTNAME=$(hostname -f)
EMBEDDING_ENDPOINT="http://${HOSTNAME}:${PORT}/v1"

ENDPOINT_FILE="${LOG_DIR}/embedding_endpoint.txt"
cat > "${ENDPOINT_FILE}" << EOF
Embedding server: ${EMBEDDING_ENDPOINT}

Environment variables to set before running the LangGraph server:
  export QWEN3_EMBEDDING_BASE_URL=${EMBEDDING_ENDPOINT}
  export QWEN3_EMBEDDING_MODEL=${MODEL_NAME}
  export LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding
EOF
echo "[INFO] Endpoint info saved to: ${ENDPOINT_FILE}"
cat "${ENDPOINT_FILE}"

# ── Start vLLM embedding server ───────────────────────────────────────────────
# --task embed  tells vLLM to expose /v1/embeddings instead of /v1/completions
# Qwen3-Embedding-8B fits in ~20 GB VRAM at float16, so a single A40/A100 is fine.
echo ""
echo "[INFO] Starting vLLM embedding server..."
echo "[INFO] Logs → ${LOG_DIR}/vllm_embedding_server.log"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --task embed \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len 32768 \
    --download-dir "${CHECKPOINT_DIR}" \
    --trust-remote-code \
    2>&1 | tee -a "${LOG_DIR}/vllm_embedding_server.log"
