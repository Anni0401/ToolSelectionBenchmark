#!/bin/bash
#SBATCH --job-name=wtb-runner
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A40:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ann-kathrin.herrmann@students.uni-mannheim.de
# ============================================================================
# SLURM Job: Auxiliary vLLM Server + LangGraph App + WTB Benchmark Runner
# ============================================================================
#
# Default resources (A40×1) cover all qwen3_embedding* modes.
# For hierarchical, override --gres and --mem when submitting:
#   sbatch --gres=gpu:A40:2 --mem=96G --export=ALL,RUN_DIR=<path> \
#       deploy/slurm_aux_and_runner_job.sh
#
# Recommended: use the coordinator which handles overrides automatically:
#   bash deploy/submit_benchmark.sh <MODE>
#
# This job handles everything except the executing LLM (gpt-oss-120b):
#   1. Starts the auxiliary vLLM server for the chosen selection mode:
#         hierarchical              → Qwen3-30B-A3B selector LLM (port 8002)
#         qwen3_embedding*          → Qwen3-Embedding-8B           (port 8002)
#   2. Waits for the gpt-oss-120b endpoint to become healthy (exec job).
#   3. Pre-warms the Qwen3 embedding cache if it does not yet exist.
#   4. Starts the LangGraph app middleware server (port 8001, localhost only).
#   5. Runs the WTB benchmark (wtb.openfunctions_evaluation --model=langgraph).
#   6. Cancels the exec LLM SLURM job on exit (success or failure).
#
# Required env variable:
#   RUN_DIR  — shared run directory on the home filesystem
# ============================================================================

set -euo pipefail

# ── Load run configuration ────────────────────────────────────────────────────
if [[ -z "${RUN_DIR:-}" ]] || [[ ! -f "${RUN_DIR}/config.env" ]]; then
    echo "[ERROR] RUN_DIR not set or ${RUN_DIR}/config.env missing."
    echo "        Submit via: bash deploy/submit_benchmark.sh <MODE>"
    exit 1
fi
source "${RUN_DIR}/config.env"

echo "=========================================="
echo "SLURM Job: Aux Server + Benchmark Runner"
echo "=========================================="
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           $(hostname -f)"
echo "Selection mode: ${SELECTION_MODE}"
echo "Exec job ID:    ${EXEC_JID}"
echo "Run dir:        ${RUN_DIR}"
echo "=========================================="

# ── Exit trap: cancel exec LLM job and kill background servers ────────────────
AUX_SERVER_PID=""
RERANKER_PID=""
LANGGRAPH_PID=""

_cleanup() {
    local exit_code=$?
    echo ""
    echo "[INFO] Benchmark runner exiting (code ${exit_code}). Cleaning up..."
    [[ -n "${LANGGRAPH_PID}" ]]   && kill "${LANGGRAPH_PID}"   2>/dev/null && echo "[INFO] LangGraph app stopped."
    [[ -n "${RERANKER_PID}" ]]    && kill "${RERANKER_PID}"    2>/dev/null && echo "[INFO] Qwen3-Reranker-8B stopped."
    [[ -n "${AUX_SERVER_PID}" ]]  && kill "${AUX_SERVER_PID}"  2>/dev/null && echo "[INFO] Aux vLLM server stopped."
    if [[ -n "${EXEC_JID:-}" ]]; then
        echo "[INFO] Cancelling exec LLM job ${EXEC_JID} ..."
        scancel "${EXEC_JID}" 2>/dev/null || echo "[WARN] scancel returned non-zero (job may already be finished)."
    fi
    echo "[INFO] Cleanup done."
}
trap _cleanup EXIT

# ── Activate main project venv ────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "[ERROR] .venv not found at ${PROJECT_ROOT}/.venv"
    echo "        Run: bash deploy/uv_setup.sh --vllm"
    exit 1
fi
source .venv/bin/activate
echo "[INFO] Python: $(python --version)"

# ── NCCL fixes (needed for vLLM on this cluster) ──────────────────────────────
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

echo "[INFO] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Helper: wait for an HTTP endpoint to respond ─────────────────────────────
_wait_for_http() {
    local url="$1"
    local label="${2:-server}"
    local timeout_s="${3:-1800}"
    local interval=15
    local elapsed=0

    echo "[INFO] Waiting for ${label} at ${url} ..."
    while ! curl -sf "${url}" > /dev/null 2>&1; do
        if (( elapsed >= timeout_s )); then
            echo "[ERROR] ${label} did not become healthy after ${timeout_s}s — aborting."
            return 1
        fi
        sleep "${interval}"
        elapsed=$(( elapsed + interval ))
        echo "[INFO]   ${label}: still waiting (${elapsed}s / ${timeout_s}s)"
    done
    echo "[INFO] ${label} is healthy!"
}

# ── Start auxiliary vLLM server ───────────────────────────────────────────────
AUX_PORT=8002
RERANKER_PORT=8003

case "${SELECTION_MODE}" in
    # ── Hierarchical: Qwen3-30B-A3B selector LLM ─────────────────────────────
    hierarchical)
        echo "[INFO] Starting Qwen3-30B-A3B selector LLM on port ${AUX_PORT} ..."
        MODEL_NAME=Qwen/Qwen3-30B-A3B \
        VLLM_PORT=${AUX_PORT} \
        TENSOR_PARALLEL_SIZE=2 \
        GPU_MEMORY_UTILIZATION=0.90 \
        DTYPE=float16 \
            bash "${PROJECT_ROOT}/deploy/slurm_vllm_deploy.sh" &
        AUX_SERVER_PID=$!

        _wait_for_http "http://localhost:${AUX_PORT}/v1/models" \
            "Qwen3-30B-A3B selector LLM" 1800

        export LANGGRAPH_SELECTOR_LLM_ENDPOINT="http://localhost:${AUX_PORT}/v1/chat/completions"
        export LANGGRAPH_SELECTOR_LLM_API_KEY="EMPTY"
        export LANGGRAPH_SELECTOR_LLM_MODEL="Qwen/Qwen3-30B-A3B"
        ;;

    # ── Embedding variants: Qwen3-Embedding-8B ───────────────────────────────
    qwen3_embedding | qwen3_embedding_context | \
    qwen3_embedding_reranker | qwen3_embedding_context_reranker)
        echo "[INFO] Starting Qwen3-Embedding-8B on port ${AUX_PORT} ..."
        MODEL_NAME=Qwen/Qwen3-Embedding-8B \
        VLLM_EMBEDDING_PORT=${AUX_PORT} \
        GPU_MEMORY_UTILIZATION=0.80 \
        DTYPE=float16 \
            bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" &
        AUX_SERVER_PID=$!

        _wait_for_http "http://localhost:${AUX_PORT}/v1/models" \
            "Qwen3-Embedding-8B" 1800

        export QWEN3_EMBEDDING_BASE_URL="http://localhost:${AUX_PORT}/v1"
        export QWEN3_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
        export QWEN3_EMBEDDING_API_KEY="EMPTY"

        # Pre-warm the embedding cache if it does not exist yet
        EMBED_CACHE="${PROJECT_ROOT}/wild-tool-bench/wtb/model_handler/api_inference/tool_embeddings_cache_qwen3.json"
        if [[ ! -f "${EMBED_CACHE}" ]]; then
            echo "[INFO] Embedding cache not found — pre-warming (this may take ~15 min) ..."
            cd "${PROJECT_ROOT}/wild-tool-bench"
            python wtb/model_handler/api_inference/setup_openai_embeddings.py \
                --provider qwen3 \
                --tools-file ../multi-agent-framework/tools/tools_en.jsonl
            cd "${PROJECT_ROOT}"
            echo "[INFO] Embedding cache ready: ${EMBED_CACHE}"
        else
            echo "[INFO] Embedding cache found: ${EMBED_CACHE}"
        fi
        ;;

    # ── Qwen3-Reranker-8B standalone (no embedding stage) ────────────────────
    qwen3_reranker)
        echo "[INFO] Starting Qwen3-Reranker-8B on port ${AUX_PORT} ..."
        MODEL_NAME=Qwen/Qwen3-Reranker-8B \
        VLLM_RERANKER_PORT=${AUX_PORT} \
        GPU_MEMORY_UTILIZATION=0.80 \
        DTYPE=float16 \
            bash "${PROJECT_ROOT}/deploy/slurm_vllm_reranker_deploy.sh" &
        AUX_SERVER_PID=$!

        _wait_for_http "http://localhost:${AUX_PORT}/v1/models" \
            "Qwen3-Reranker-8B" 1800

        export QWEN3_RERANKER_BASE_URL="http://localhost:${AUX_PORT}/v1"
        export QWEN3_RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
        export QWEN3_RERANKER_API_KEY="EMPTY"
        ;;

    # ── Qwen3-Embedding-8B + Qwen3-Reranker-8B two-model setup ───────────────
    # Both 8B models run on separate GPUs of the same A40×2 node.
    # GPU 0 → Qwen3-Embedding-8B (port 8002)
    # GPU 1 → Qwen3-Reranker-8B  (port 8003)
    qwen3_embedding_qwen3_reranker | qwen3_embedding_context_qwen3_reranker)
        echo "[INFO] Starting Qwen3-Embedding-8B on GPU 0 / port ${AUX_PORT} ..."
        CUDA_VISIBLE_DEVICES=0 \
        MODEL_NAME=Qwen/Qwen3-Embedding-8B \
        VLLM_EMBEDDING_PORT=${AUX_PORT} \
        GPU_MEMORY_UTILIZATION=0.80 \
        DTYPE=float16 \
            bash "${PROJECT_ROOT}/deploy/slurm_vllm_embedding_deploy.sh" &
        AUX_SERVER_PID=$!

        echo "[INFO] Starting Qwen3-Reranker-8B on GPU 1 / port ${RERANKER_PORT} ..."
        CUDA_VISIBLE_DEVICES=1 \
        MODEL_NAME=Qwen/Qwen3-Reranker-8B \
        VLLM_RERANKER_PORT=${RERANKER_PORT} \
        GPU_MEMORY_UTILIZATION=0.80 \
        DTYPE=float16 \
            bash "${PROJECT_ROOT}/deploy/slurm_vllm_reranker_deploy.sh" &
        RERANKER_PID=$!

        _wait_for_http "http://localhost:${AUX_PORT}/v1/models" \
            "Qwen3-Embedding-8B" 1800
        _wait_for_http "http://localhost:${RERANKER_PORT}/v1/models" \
            "Qwen3-Reranker-8B" 1800

        export QWEN3_EMBEDDING_BASE_URL="http://localhost:${AUX_PORT}/v1"
        export QWEN3_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
        export QWEN3_EMBEDDING_API_KEY="EMPTY"
        export QWEN3_RERANKER_BASE_URL="http://localhost:${RERANKER_PORT}/v1"
        export QWEN3_RERANKER_MODEL="Qwen/Qwen3-Reranker-8B"
        export QWEN3_RERANKER_API_KEY="EMPTY"

        # Pre-warm the embedding cache if it does not exist yet
        EMBED_CACHE="${PROJECT_ROOT}/wild-tool-bench/wtb/model_handler/api_inference/tool_embeddings_cache_qwen3.json"
        if [[ ! -f "${EMBED_CACHE}" ]]; then
            echo "[INFO] Embedding cache not found — pre-warming (this may take ~15 min) ..."
            cd "${PROJECT_ROOT}/wild-tool-bench"
            python wtb/model_handler/api_inference/setup_openai_embeddings.py \
                --provider qwen3 \
                --tools-file ../multi-agent-framework/tools/tools_en.jsonl
            cd "${PROJECT_ROOT}"
            echo "[INFO] Embedding cache ready: ${EMBED_CACHE}"
        else
            echo "[INFO] Embedding cache found: ${EMBED_CACHE}"
        fi
        ;;
esac

# ── Wait for the executing LLM (gpt-oss-120b on H200 node) ───────────────────
echo "[INFO] Waiting for exec endpoint file at ${RUN_DIR}/exec_endpoint.txt ..."
ELAPSED=0
while [[ ! -f "${RUN_DIR}/exec_endpoint.txt" ]]; do
    if (( ELAPSED >= 1800 )); then
        echo "[ERROR] Timeout: exec endpoint file not found after 30 min."
        exit 1
    fi
    sleep 15
    ELAPSED=$(( ELAPSED + 15 ))
    echo "[INFO]   Still waiting for exec endpoint file (${ELAPSED}s) ..."
done

EXEC_BASE_URL=$(head -1 "${RUN_DIR}/exec_endpoint.txt")
echo "[INFO] Exec LLM endpoint: ${EXEC_BASE_URL}"

_wait_for_http "${EXEC_BASE_URL}/models" "gpt-oss-120b" 1800

# ── Start the LangGraph middleware server (localhost:8001) ────────────────────
export EXECUTING_LLM_BASE_URL="${EXEC_BASE_URL}"
export EXECUTING_LLM_MODEL="openai/gpt-oss-120b"
export EXECUTING_LLM_API_KEY="EMPTY"
export LANGGRAPH_TOOL_SELECTION_MODE="${SELECTION_MODE}"
export LANGGRAPH_HOST="127.0.0.1"
export LANGGRAPH_PORT="8001"

cd "${PROJECT_ROOT}/wild-tool-bench"
echo "[INFO] Starting LangGraph app (mode: ${SELECTION_MODE}) on 127.0.0.1:8001 ..."
python -m wtb.model_handler.api_inference.langgraph_app &
LANGGRAPH_PID=$!

# Wait until port 8001 is accepting connections
echo "[INFO] Waiting for LangGraph app to open port 8001 ..."
ELAPSED=0
until (echo > /dev/tcp/127.0.0.1/8001) 2>/dev/null; do
    if ! kill -0 "${LANGGRAPH_PID}" 2>/dev/null; then
        echo "[ERROR] LangGraph app process died during startup."
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

# ── Run the WildToolBench evaluation ─────────────────────────────────────────
export LANGGRAPH_ENDPOINT="http://127.0.0.1:8001/execute"

echo ""
echo "=========================================="
echo "Running WildToolBench evaluation"
echo "  Model:          langgraph"
echo "  Selection mode: ${SELECTION_MODE}"
echo "  Exec LLM:       ${EXEC_BASE_URL}"
echo "  LangGraph app:  http://127.0.0.1:8001/execute"
echo "=========================================="

python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir "result/${SELECTION_MODE}"

echo ""
echo "[INFO] Benchmark complete. Results in ${PROJECT_ROOT}/wild-tool-bench/result/${SELECTION_MODE}/"
