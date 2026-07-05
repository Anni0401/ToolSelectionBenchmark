#!/bin/bash
#SBATCH --job-name=wtb-exec-llm
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ann-kathrin.herrmann@students.uni-mannheim.de
# ============================================================================
# SLURM Job: Executing LLM Server  (gpt-oss-120b on H200×4)
# ============================================================================
#
# Submit via the benchmark coordinator (recommended):
#   bash deploy/submit_benchmark.sh <MODE>
#
# Or directly (RUN_DIR must be set via --export):
#   sbatch --export=ALL,RUN_DIR=<path> deploy/slurm_exec_llm_job.sh
#
# This job starts the gpt-oss-120b vLLM server and writes its HTTP endpoint
# to ${RUN_DIR}/exec_endpoint.txt so the runner job can connect to it.
# It stays alive until cancelled by the runner job (via scancel).
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
echo "SLURM Job: Executing LLM Server"
echo "=========================================="
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Node:           $(hostname -f)"
echo "Selection mode: ${SELECTION_MODE}"
echo "Run dir:        ${RUN_DIR}"
echo "=========================================="

# ── Activate gpt-oss venv ────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -f ".venv-gptoss/bin/activate" ]]; then
    echo "[ERROR] .venv-gptoss not found at ${PROJECT_ROOT}/.venv-gptoss"
    echo "        Follow deploy/SALLOC_GUIDE_EXECUTION_LLM.md to create it."
    exit 1
fi

source .venv-gptoss/bin/activate
echo "[INFO] Python: $(python --version)"

# Verify the correct gpt-oss vLLM build is active
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "NOT_FOUND")
echo "[INFO] vLLM version: ${VLLM_VER}"
if [[ "$VLLM_VER" == "NOT_FOUND" ]]; then
    echo "[ERROR] vLLM not found in .venv-gptoss"
    exit 1
fi

# ── NCCL fixes for this cluster ───────────────────────────────────────────────
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

echo "[INFO] GPU status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ── Write endpoint file BEFORE starting vLLM ─────────────────────────────────
# The runner job polls this file to discover which node to connect to.
HOSTNAME_FULL=$(hostname -f)
EXEC_BASE_URL="http://${HOSTNAME_FULL}:8000/v1"

echo "${EXEC_BASE_URL}" > "${RUN_DIR}/exec_endpoint.txt"
echo "[$(date -Iseconds)] gpt-oss-120b starting — job ${SLURM_JOB_ID} on ${HOSTNAME_FULL}" \
    >> "${RUN_DIR}/exec_endpoint.txt"

echo "[INFO] Wrote exec endpoint: ${EXEC_BASE_URL}"
echo "[INFO] Runner job will read: ${RUN_DIR}/exec_endpoint.txt"

# ── Start vLLM server (blocking) ──────────────────────────────────────────────
echo "[INFO] Starting gpt-oss-120b on ${HOSTNAME_FULL}:8000 ..."
echo "[INFO] (Will stay alive until the runner job cancels this job via scancel)"

vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser openai \
    --enable-auto-tool-choice \
    --download-dir "${HOME}/.cache/huggingface/hub"

# If vllm serve exits on its own (e.g. OOM), leave a marker so the runner can detect it
echo "[$(date -Iseconds)] vLLM process exited unexpectedly (exit code $?)" \
    >> "${RUN_DIR}/exec_endpoint.txt"
