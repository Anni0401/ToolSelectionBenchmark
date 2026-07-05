#!/bin/bash
#SBATCH --job-name=wtb-in-context
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ann-kathrin.herrmann@students.uni-mannheim.de
# ============================================================================
# SLURM Job: in_context — All-in-One H200 Job
# ============================================================================
#
# Submit via the benchmark coordinator (recommended):
#   bash deploy/submit_benchmark.sh in_context
#
# Or directly (RUN_DIR must be set via --export):
#   sbatch --export=ALL,RUN_DIR=<path> deploy/slurm_in_context_job.sh
#
# This single job runs the entire in_context benchmark on one H200×4 node:
#   1. Activates .venv-gptoss and starts gpt-oss-120b in the background.
#   2. Switches to the main .venv and waits for gpt-oss-120b to be healthy.
#   3. Starts the LangGraph middleware app on localhost:8001.
#   4. Runs wtb.openfunctions_evaluation --model=langgraph.
#   5. Cleans up background processes on exit.
#
# Required env variable:
#   RUN_DIR  — shared run directory on the home filesystem
# ============================================================================

set -euo pipefail

# ── Load run configuration ────────────────────────────────────────────────────
if [[ -z "${RUN_DIR:-}" ]] || [[ ! -f "${RUN_DIR}/config.env" ]]; then
    echo "[ERROR] RUN_DIR not set or ${RUN_DIR}/config.env missing."
    echo "        Submit via: bash deploy/submit_benchmark.sh in_context"
    exit 1
fi
source "${RUN_DIR}/config.env"

echo "=========================================="
echo "SLURM Job: in_context all-in-one"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname -f)"
echo "Run dir: ${RUN_DIR}"
echo "=========================================="

# ── Background process tracking ───────────────────────────────────────────────
GPTOSS_PID=""
LANGGRAPH_PID=""

_cleanup() {
    echo "[INFO] Cleaning up background processes ..."
    [[ -n "${LANGGRAPH_PID}" ]] && kill "${LANGGRAPH_PID}" 2>/dev/null && echo "[INFO] LangGraph app stopped."
    [[ -n "${GPTOSS_PID}" ]]    && kill "${GPTOSS_PID}"    2>/dev/null && echo "[INFO] gpt-oss-120b stopped."
}
trap _cleanup EXIT

cd "${PROJECT_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

# ── NCCL fixes for this cluster ───────────────────────────────────────────────
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

echo "[INFO] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Step 1: Start gpt-oss-120b in the background (.venv-gptoss) ───────────────
if [[ ! -f ".venv-gptoss/bin/activate" ]]; then
    echo "[ERROR] .venv-gptoss not found at ${PROJECT_ROOT}/.venv-gptoss"
    echo "        Follow deploy/SALLOC_GUIDE_EXECUTION_LLM.md to create it."
    exit 1
fi
source .venv-gptoss/bin/activate
echo "[INFO] Activated .venv-gptoss — Python: $(python --version)"

echo "[INFO] Starting gpt-oss-120b on port 8000 (background) ..."
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    --download-dir "${HOME}/.cache/huggingface/hub" \
    > "${RUN_DIR}/logs/gptoss_server.log" 2>&1 &
GPTOSS_PID=$!
echo "[INFO] gpt-oss-120b started with PID ${GPTOSS_PID}"

# ── Step 2: Switch to main project venv ───────────────────────────────────────
# The gptoss process keeps its Python interpreter regardless of venv changes in
# the parent shell, so it is safe to switch venvs here.
source .venv/bin/activate
echo "[INFO] Activated .venv — Python: $(python --version)"

# ── Step 3: Wait for gpt-oss-120b to be healthy ───────────────────────────────
echo "[INFO] Waiting for gpt-oss-120b health check on http://localhost:8000/v1/models ..."
EXEC_BASE_URL="http://localhost:8000/v1"
ELAPSED=0
until curl -sf "http://localhost:8000/v1/models" > /dev/null 2>&1; do
    if ! kill -0 "${GPTOSS_PID}" 2>/dev/null; then
        echo "[ERROR] gpt-oss-120b process (PID ${GPTOSS_PID}) died. Check logs:"
        echo "        ${RUN_DIR}/logs/gptoss_server.log"
        tail -20 "${RUN_DIR}/logs/gptoss_server.log" || true
        exit 1
    fi
    if (( ELAPSED >= 1800 )); then
        echo "[ERROR] gpt-oss-120b did not become healthy after 30 min."
        exit 1
    fi
    sleep 20
    ELAPSED=$(( ELAPSED + 20 ))
    echo "[INFO]   Still waiting for gpt-oss-120b ... (${ELAPSED}s)"
done
echo "[INFO] gpt-oss-120b is healthy!"

# ── Step 4: Start the LangGraph app (localhost:8001) ─────────────────────────
export EXECUTING_LLM_BASE_URL="${EXEC_BASE_URL}"
export EXECUTING_LLM_MODEL="openai/gpt-oss-120b"
export EXECUTING_LLM_API_KEY="EMPTY"
export LANGGRAPH_TOOL_SELECTION_MODE="in_context"
export LANGGRAPH_HOST="127.0.0.1"
export LANGGRAPH_PORT="8001"

cd "${PROJECT_ROOT}/wild-tool-bench"
echo "[INFO] Starting LangGraph app (in_context) on 127.0.0.1:8001 ..."
python -m wtb.model_handler.api_inference.langgraph_app \
    > "${RUN_DIR}/logs/langgraph_app.log" 2>&1 &
LANGGRAPH_PID=$!

# Wait until port 8001 is accepting connections
echo "[INFO] Waiting for LangGraph app to open port 8001 ..."
ELAPSED=0
until (echo > /dev/tcp/127.0.0.1/8001) 2>/dev/null; do
    if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
        echo "[ERROR] LangGraph app process died during startup. Check logs:"
        echo "        ${RUN_DIR}/logs/langgraph_app.log"
        tail -20 "${RUN_DIR}/logs/langgraph_app.log" || true
        exit 1
    fi
    if (( ELAPSED >= 120 )); then
        echo "[ERROR] LangGraph app did not open port 8001 after 120s."
        exit 1
    fi
    sleep 3
    ELAPSED=$(( ELAPSED + 3 ))
done
echo "[INFO] LangGraph app is ready (PID: ${LANGGRAPH_PID})"

# ── Step 5: Run WildToolBench evaluation ─────────────────────────────────────
export LANGGRAPH_ENDPOINT="http://127.0.0.1:8001/execute"

echo ""
echo "=========================================="
echo "Running WildToolBench evaluation"
echo "  Model:          langgraph"
echo "  Selection mode: in_context"
echo "  Exec LLM:       ${EXEC_BASE_URL}"
echo "  LangGraph app:  http://127.0.0.1:8001/execute"
echo "=========================================="

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir "result/in_context"

echo ""
echo "[INFO] Benchmark complete. Results in ${PROJECT_ROOT}/wild-tool-bench/result/in_context/"
