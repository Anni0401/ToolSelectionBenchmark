#!/bin/bash

####################################################
# Component 0: GPT-OSS-120B
####################################################

#SBATCH --job-name=wtb-embedding
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

####################################################
# Component 1: Qwen embedding model
####################################################

#SBATCH hetjob
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

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

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

# Hugging Face / model cache
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

GPT_VENV="${WORK}/venvs/venv-gptoss"
BENCH_VENV="${PROJECT_ROOT}/.venv"

GPT_MODEL="${WORK}/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"


####################################################
# Determine nodes of both heterogeneous components
####################################################

GPT_HOST=$(srun --het-group=0 --nodes=1 --ntasks=1 hostname)
EMBED_HOST=$(srun --het-group=1 --nodes=1 --ntasks=1 hostname)

echo "===================================================="
echo "GPT node:       ${GPT_HOST}"
echo "Embedding node: ${EMBED_HOST}"
echo "Project root:   ${PROJECT_ROOT}"
echo "===================================================="


####################################################
# Update .env
####################################################

sed -i \
    "s|^EXECUTING_LLM_BASE_URL=.*|EXECUTING_LLM_BASE_URL=http://${GPT_HOST}:8000/v1|" \
    "${PROJECT_ROOT}/wild-tool-bench/.env"

sed -i \
    "s|^QWEN3_EMBEDDING_BASE_URL=.*|QWEN3_EMBEDDING_BASE_URL=http://${EMBED_HOST}:8002/v1|" \
    "${PROJECT_ROOT}/wild-tool-bench/.env"

sed -i \
    "s|^LANGGRAPH_TOOL_SELECTION_MODE=.*|LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding_context|" \
    "${PROJECT_ROOT}/wild-tool-bench/.env"


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
# Start GPT-OSS on heterogeneous component 0
####################################################

echo "Starting GPT-OSS on ${GPT_HOST}..."

srun \
    --het-group=0 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    bash -c "
        source '${GPT_VENV}/bin/activate'

        vllm serve '${GPT_MODEL}' \
            --served-model-name openai/gpt-oss-120b \
            --tensor-parallel-size 1 \
            --dtype bfloat16 \
            --gpu-memory-utilization 0.90 \
            --enforce-eager \
            --host 0.0.0.0 \
            --port 8000 \
            --tool-call-parser openai \
            --enable-auto-tool-choice
    " &

GPT_PID=$!


####################################################
# Start embedding server on heterogeneous component 1
####################################################

echo "Starting embedding server on ${EMBED_HOST}..."

srun \
    --het-group=1 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    bash -c "
        source '${BENCH_VENV}/bin/activate'
        bash '${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh'
    " &

EMBED_PID=$!


####################################################
# Wait for GPT server
####################################################

echo "Waiting for GPT-OSS..."

until curl -sf "http://${GPT_HOST}:8000/v1/models" >/dev/null
do
    sleep 5
done

echo "GPT-OSS ready."


####################################################
# Wait for embedding server
####################################################

echo "Waiting for embedding server..."

until curl -sf "http://${EMBED_HOST}:8002/v1/models" >/dev/null
do
    sleep 5
done

echo "Embedding server ready."


####################################################
# Switch into benchmark project
####################################################

cd "${PROJECT_ROOT}/wild-tool-bench"

source "${BENCH_VENV}/bin/activate"


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
    --result-dir result/embedding_context \
    --num-threads 1

echo ""
echo "Benchmark completed successfully."