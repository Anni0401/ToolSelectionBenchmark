#!/bin/bash
#SBATCH --job-name=wtb-rewrite-pairs-gptoss
#SBATCH --partition=gpu-vram-94gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Starts a self-hosted gpt-oss-120b vLLM server on one H100 and runs
# generate_rewrite_pairs_vllm.py against it (25 rewrite pairs/task by default).
# Safe to resubmit: the script passes --resume, so a partially completed
# output file is continued instead of restarted.

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

mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

# Hugging Face / model cache
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

mkdir -p \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}"

GPT_VENV="${WORK}/venvs/venv-gptoss"
BENCH_VENV="${PROJECT_ROOT}/.venv"

GPT_MODEL="${WORK}/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
HOST=$(hostname)

# Pick an unused port if GPT_PORT is not explicitly set
if [[ -z "${GPT_PORT:-}" ]]; then
    GPT_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
fi

echo "===================================================="
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "GPT model:       ${GPT_MODEL}"
echo "GPT port:        ${GPT_PORT}"
echo "===================================================="

if [[ ! -f "${GPT_VENV}/bin/activate" ]]; then
    echo "ERROR: GPT virtual environment not found: ${GPT_VENV}"
    exit 1
fi

if [[ ! -f "${BENCH_VENV}/bin/activate" ]]; then
    echo "ERROR: Benchmark virtual environment not found: ${BENCH_VENV}"
    exit 1
fi

if [[ ! -d "${GPT_MODEL}" ]]; then
    echo "ERROR: GPT model snapshot does not exist: ${GPT_MODEL}"
    exit 1
fi

####################################################
# Cleanup
####################################################

GPT_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."

    kill "${GPT_PID:-}" 2>/dev/null || true

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
    --port "${GPT_PORT}" \
    --tool-call-parser openai \
    --enable-auto-tool-choice &

GPT_PID=$!

echo "GPT-OSS PID: ${GPT_PID}"

####################################################
# Wait for GPT server
####################################################

echo "Waiting for GPT-OSS..."

until curl -sf "http://${HOST}:${GPT_PORT}/v1/models" | grep -q "openai/gpt-oss-120b"
do
    if ! kill -0 "${GPT_PID}" 2>/dev/null; then
        echo "ERROR: GPT-OSS exited during startup."
        wait "${GPT_PID}" || true
        exit 1
    fi
    sleep 5
done

echo "GPT-OSS ready."

####################################################
# Switch into benchmark project
####################################################

deactivate || true
source "${BENCH_VENV}/bin/activate"
cd "${PROJECT_ROOT}/multi-agent-framework"

####################################################
# Run rewrite pair generation (resumable)
####################################################

echo "Running rewrite pair generation..."

python -u generate_rewrite_pairs_vllm.py \
    --base-url "http://${HOST}:${GPT_PORT}/v1" \
    --model openai/gpt-oss-120b \
    --num-pairs "${NUM_PAIRS:-25}" \
    --resume

echo ""
echo "Rewrite pair generation completed successfully."
