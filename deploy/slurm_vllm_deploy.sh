#!/bin/bash
# SLURM deployment script for vLLM with Qwen selector model
# Uses the uv-managed virtual environment created by deploy/uv_setup.sh.
# Usage: bash deploy/slurm_vllm_deploy.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# Activate the uv-managed virtual environment
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
    echo "[ERROR] Python 3.10+ is required. Check your .venv setup."
    exit 1
fi

echo "[INFO] Python version: $(python --version)"

# Configuration
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B}"
PORT=${VLLM_PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-32}
DTYPE=${DTYPE:-float16}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
LOG_DIR="${PWD}/logs"
export CHECKPOINT_DIR="${HOME}/.cache/huggingface/hub"

# Fix NCCL segfault caused by broken network plugin (InfiniBand/RoCE) on this cluster
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

echo "=========================================="
echo "vLLM Qwen Model Deployment"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Data Type: ${DTYPE}"
echo "Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE}"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] nvidia-smi not found. CUDA/GPU not available."
    exit 1
fi

echo "[INFO] GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Save a convenience activation snippet for manual use
ACTIVATION_SCRIPT="${LOG_DIR}/activate_env.sh"
cat > "${ACTIVATION_SCRIPT}" << EOF
#!/bin/bash
# Activate the uv-managed virtual environment for this project
source "${VENV_DIR}/bin/activate"
EOF
chmod +x "${ACTIVATION_SCRIPT}"

# Verify vLLM installation
if ! python -c "import vllm" 2>/dev/null; then
    echo "[ERROR] vLLM not found in ${VENV_DIR}."
    echo "        Run 'bash deploy/uv_setup.sh --vllm' to install it."
    exit 1
fi

# Download model weights (if not already cached)
echo "[INFO] Ensuring model weights are downloaded..."
python << 'PYTHON_EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
cache_dir = os.environ.get("CHECKPOINT_DIR", os.path.expanduser("~/.cache/huggingface/hub"))

try:
    print(f"[INFO] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"[SUCCESS] Tokenizer loaded successfully")
    
    # Note: We don't actually load the full model here as it's very large
    # vLLM will handle model loading with optimizations
    print(f"[INFO] Model weights will be loaded by vLLM")
except Exception as e:
    print(f"[WARNING] Error during model verification: {e}")
    print(f"[INFO] vLLM will attempt to download and load the model")
PYTHON_EOF

# Start vLLM server
echo "[INFO] Starting vLLM server..."
echo "[INFO] Access logs will be saved to: ${LOG_DIR}/vllm_server.log"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --port ${PORT} \
    --host 0.0.0.0 \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len 32768 \
    --dtype ${DTYPE} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --download-dir "${CHECKPOINT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/vllm_server.log"
