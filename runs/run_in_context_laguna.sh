#!/bin/bash
#SBATCH --job-name=wtb-incontext-laguna
#SBATCH --partition=gpu-vram-94gb
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
# Determine project root
####################################################

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"

####################################################
# Environment
####################################################

export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${WORK}/tmp_pip/cache"

export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

mkdir -p \
    "${TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}"

####################################################
# Laguna / vLLM environment
####################################################

LAGUNA_VENV="${WORK}/venvs/venv-laguna"
BENCH_VENV="${PROJECT_ROOT}/.venv"

LAGUNA_MODEL="poolside/Laguna-S-2.1-FP8"

# Required for Laguna FP8
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0

HOST=$(hostname)

echo "===================================================="
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "Model:           ${LAGUNA_MODEL}"
echo "===================================================="

echo "Allocated GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

####################################################
# Update .env
####################################################

sed -i \
    "s|^EXECUTING_LLM_BASE_URL=.*|EXECUTING_LLM_BASE_URL=http://${HOST}:8000/v1|" \
    "${PROJECT_ROOT}/wild-tool-bench/.env"

sed -i \
    "s|^EXECUTING_LLM_MODEL=.*|EXECUTING_LLM_MODEL=${LAGUNA_MODEL}|" \
    "${PROJECT_ROOT}/wild-tool-bench/.env"

sed -i \
    "s|^LANGGRAPH_TOOL_SELECTION_MODE=.*|LANGGRAPH_TOOL_SELECTION_MODE=in_context|" \
    "${PROJECT_ROOT}/wild-tool-bench/.env"

####################################################
# Cleanup
####################################################

cleanup() {
    echo ""
    echo "Cleaning up..."

    kill ${LANGGRAPH_PID:-} 2>/dev/null || true
    kill ${LAGUNA_PID:-} 2>/dev/null || true

    wait || true
}

trap cleanup EXIT

####################################################
# Start Laguna
####################################################

echo "Starting Laguna-S-2.1-FP8..."

source "${LAGUNA_VENV}/bin/activate"

echo "Python:"
which python
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
    --port 8000 &

LAGUNA_PID=$!

echo "Laguna PID: ${LAGUNA_PID}"

####################################################
# Wait for Laguna server
####################################################

echo "Waiting for Laguna..."

until curl -sf "http://127.0.0.1:8000/v1/models" >/dev/null
do
    if ! kill -0 "${LAGUNA_PID}" 2>/dev/null; then
        echo "ERROR: Laguna vLLM server exited during startup."
        exit 1
    fi

    sleep 5
done

echo "Laguna ready."

####################################################
# Switch to benchmark project/environment
####################################################

cd "${PROJECT_ROOT}/wild-tool-bench"

deactivate || true
source "${BENCH_VENV}/bin/activate"

echo "Benchmark Python:"
which python
python --version

####################################################
# Start LangGraph
####################################################
echo "Cleaning stale LangGraph process..."

fuser -k 8001/tcp 2>/dev/null || true

echo "Starting LangGraph..."

python -m wtb.model_handler.api_inference.langgraph_app &

LANGGRAPH_PID=$!

####################################################
# Wait for LangGraph
####################################################

echo "Waiting for LangGraph..."

sleep 15

if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
    echo "ERROR: LangGraph exited during startup."
    exit 1
fi

echo "LangGraph ready."

####################################################
# Run benchmark
####################################################

echo "Installing overrides..."

python -m pip install overrides -q

echo "Running benchmark..."

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir result/in_context \
    --num-threads 1

echo ""
echo "Benchmark completed successfully."