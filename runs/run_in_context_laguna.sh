#!/bin/bash
#SBATCH --job-name=wtb-incontext
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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

mkdir -p "${TMPDIR}"
mkdir -p "${PIP_CACHE_DIR}"

# Hugging Face / model cache
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

mkdir -p \
    "${TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}"

#export NCCL_NET_PLUGIN=none
#export NCCL_IB_DISABLE=1
#export NCCL_P2P_LEVEL=NVL

GPT_VENV="${WORK}/venvs/venv-gptoss"
BENCH_VENV="${PROJECT_ROOT}/.venv"

GPT_MODEL="${WORK}/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

HOST=$(hostname)

echo "===================================================="
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "===================================================="

####################################################
# Update .env
####################################################

sed -i "s|^EXECUTING_LLM_BASE_URL=.*|EXECUTING_LLM_BASE_URL=http://${HOST}:8000/v1|" "${PROJECT_ROOT}/wild-tool-bench/.env"



sed -i "s|^LANGGRAPH_TOOL_SELECTION_MODE=.*|LANGGRAPH_TOOL_SELECTION_MODE=in_context|" "${PROJECT_ROOT}/wild-tool-bench/.env"

####################################################
# Cleanup
####################################################

cleanup() {
    echo ""
    echo "Cleaning up..."

    kill ${LANGGRAPH_PID:-} 2>/dev/null || true
    kill ${EMBED_PID:-} 2>/dev/null || true
    kill ${GPT_PID:-} 2>/dev/null || true

    wait || true
}

trap cleanup EXIT

####################################################
# Start GPT-OSS
####################################################

echo "Starting GPT-OSS..."

source "${GPT_VENV}/bin/activate"

CUDA_VISIBLE_DEVICES=0 \
vllm serve "${GPT_MODEL}" \
    --served-model-name openai/gpt-oss-120b \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser openai \
    --enable-auto-tool-choice &

GPT_PID=$!



####################################################
# Wait for GPT server
####################################################

echo "Waiting for GPT-OSS..."

until curl -sf "http://${HOST}:8000/v1/models" >/dev/null
do
    sleep 5
done

echo "GPT-OSS ready."



####################################################
# Switch into benchmark project
####################################################

cd "${PROJECT_ROOT}/wild-tool-bench"

####################################################
# Start LangGraph
####################################################

echo "Starting LangGraph..."

python -m wtb.model_handler.api_inference.langgraph_app &

LANGGRAPH_PID=$!

####################################################
# Wait for LangGraph
####################################################

echo "Waiting for LangGraph..."


sleep 15


echo "LangGraph ready."

####################################################
# Run benchmark
####################################################

echo "Installing overrides..."

pip install overrides -q

echo "Running benchmark..."

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir result/in_context \
    --num-threads 1

echo ""
echo "Benchmark completed successfully."