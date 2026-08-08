#!/bin/bash
#SBATCH --job-name=wtb-incontext-saia
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

####################################################
# Determine project root
####################################################

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
BENCHMARK_ROOT="${PROJECT_ROOT}/wild-tool-bench"
BENCH_VENV="${PROJECT_ROOT}/.venv"

cd "$PROJECT_ROOT"

HOST=$(hostname)

echo "===================================================="
echo "Job ID:          ${SLURM_JOB_ID:-unknown}"
echo "Running on host: ${HOST}"
echo "Project root:    ${PROJECT_ROOT}"
echo "Benchmark root:  ${BENCHMARK_ROOT}"
echo "===================================================="

####################################################
# SAIA configuration
####################################################

SAIA_BASE_URL="${SAIA_BASE_URL:-https://chat-ai.academiccloud.de/v1}"
SAIA_MODEL="${SAIA_MODEL:-openai-gpt-oss-120b}"

echo "SAIA base URL: ${SAIA_BASE_URL}"
echo "SAIA model:    ${SAIA_MODEL}"

####################################################
# Environment
####################################################

export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${WORK}/tmp_pip/cache"

mkdir -p \
    "${TMPDIR}" \
    "${PIP_CACHE_DIR}"

####################################################
# Validate benchmark environment
####################################################

if [[ ! -d "${BENCH_VENV}" ]]; then
    echo "ERROR: Benchmark virtual environment not found:"
    echo "       ${BENCH_VENV}"
    exit 1
fi

if [[ ! -f "${BENCH_VENV}/bin/activate" ]]; then
    echo "ERROR: Virtual environment activation script not found:"
    echo "       ${BENCH_VENV}/bin/activate"
    exit 1
fi

if [[ ! -d "${BENCHMARK_ROOT}" ]]; then
    echo "ERROR: Benchmark directory not found:"
    echo "       ${BENCHMARK_ROOT}"
    exit 1
fi

####################################################
# Activate benchmark environment
####################################################

source "${BENCH_VENV}/bin/activate"

echo ""
echo "Benchmark Python:"
echo "  Python: $(command -v python)"
echo "  Version: $(python --version)"
echo ""

####################################################
# Configure benchmark
####################################################

cd "${BENCHMARK_ROOT}"

# Prefer exported environment variables over editing the shared .env.
# This also avoids interference if several SLURM jobs run in parallel.

export EXECUTING_LLM_BASE_URL="${SAIA_BASE_URL}"
export EXECUTING_LLM_MODEL="${SAIA_MODEL}"
export LANGGRAPH_TOOL_SELECTION_MODE="in_context"

echo "EXECUTING_LLM_BASE_URL=${EXECUTING_LLM_BASE_URL}"
echo "EXECUTING_LLM_MODEL=${EXECUTING_LLM_MODEL}"
echo "LANGGRAPH_TOOL_SELECTION_MODE=${LANGGRAPH_TOOL_SELECTION_MODE}"

####################################################
# Validate SAIA credentials
####################################################

if [[ -z "${EXECUTING_LLM_API_KEY:-}" ]]; then
    echo "WARNING: EXECUTING_LLM_API_KEY is not set."
    echo "Make sure your SAIA API key is available through the environment or .env."
fi

####################################################
# Cleanup
####################################################

LANGGRAPH_PID=""

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    echo ""
    echo "Cleaning up..."

    if [[ -n "${LANGGRAPH_PID}" ]] &&
       kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
        echo "Stopping LangGraph process ${LANGGRAPH_PID}"
        kill "${LANGGRAPH_PID}" 2>/dev/null || true
    fi

    wait "${LANGGRAPH_PID}" 2>/dev/null || true

    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

####################################################
# Start LangGraph
####################################################

echo ""
echo "===================================================="
echo "Starting LangGraph"
echo "===================================================="

python -u -m wtb.model_handler.api_inference.langgraph_app &

LANGGRAPH_PID=$!

echo "LangGraph PID: ${LANGGRAPH_PID}"

####################################################
# Wait for LangGraph startup
####################################################

echo "Waiting for LangGraph..."

sleep 15

if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
    echo "ERROR: LangGraph exited during startup."
    wait "${LANGGRAPH_PID}" || true
    exit 1
fi

echo "LangGraph ready."

####################################################
# Run benchmark
####################################################

echo ""
echo "===================================================="
echo "Running benchmark"
echo "===================================================="

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir result_120B/in_context \
    --num-threads 1

echo ""
echo "===================================================="
echo "Benchmark completed successfully"
echo "===================================================="