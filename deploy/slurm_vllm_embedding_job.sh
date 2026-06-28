#!/bin/bash
#SBATCH --job-name=qwen3-embedding
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vllm_embedding_%j.log
#SBATCH --error=/home/%u/vllm_embedding_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%ann-kathrin.herrmann@students.uni-mannheim.de

# ============================================================================
# SLURM Job Script: Qwen3-Embedding-8B server via vLLM
# ============================================================================
#
# Hosts Qwen3-Embedding-8B on port 8001 (separate from the LLM on port 8000).
# The embedding server exposes an OpenAI-compatible /v1/embeddings endpoint
# which the Qwen3EmbeddingBasedToolSelector talks to at runtime.
#
# Resource requirements:
#   - Qwen3-Embedding-8B at float16 fits in ~20 GB VRAM.
#   - A single A40 (48 GB) or A100 (40/80 GB) is more than sufficient.
#   - Lower GPU_MEMORY_UTILIZATION (default 0.4) leaves room for the LLM
#     if both run on the same node; increase to 0.8 on a dedicated node.
#
# Usage:
#   sbatch deploy/slurm_vllm_embedding_job.sh
#
# After the job starts, set these env vars on the client node:
#   export QWEN3_EMBEDDING_BASE_URL=http://<node>:8001/v1
#   export LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding
# ============================================================================

set -e

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${SLURM_JOB_NAME}"
echo "Node(s):   ${SLURM_NODELIST}"
echo "GPUs:      ${SLURM_GPUS_PER_NODE}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Memory:    ${SLURM_MEM_PER_NODE} MB"
echo "=========================================="

# ── Shell setup ───────────────────────────────────────────────────────────────
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# ── Configuration ─────────────────────────────────────────────────────────────
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
export VLLM_EMBEDDING_PORT="${VLLM_EMBEDDING_PORT:-8001}"
# 0.4 leaves VRAM headroom for a co-located LLM job; use 0.8 on a dedicated node
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
export DTYPE="${DTYPE:-float16}"
export TENSOR_PARALLEL_SIZE="${SLURM_GPUS_PER_NODE:-1}"

# ── Endpoint information ──────────────────────────────────────────────────────
HOSTNAME=$(hostname -f)
EMBEDDING_ENDPOINT="http://${HOSTNAME}:${VLLM_EMBEDDING_PORT}/v1"

JOB_LOG_DIR="/tmp/vllm_embedding_job_${SLURM_JOB_ID}"
mkdir -p "${JOB_LOG_DIR}"

ENDPOINT_FILE="${JOB_LOG_DIR}/endpoint.txt"
cat > "${ENDPOINT_FILE}" << EOF
Embedding server endpoint
=========================
Job ID:    ${SLURM_JOB_ID}
Hostname:  ${HOSTNAME}
Port:      ${VLLM_EMBEDDING_PORT}
Endpoint:  ${EMBEDDING_ENDPOINT}

Set these on your client node before running the benchmark:

  export QWEN3_EMBEDDING_BASE_URL=${EMBEDDING_ENDPOINT}
  export QWEN3_EMBEDDING_MODEL=${MODEL_NAME}

  # Select an embedding-based tool selection mode:
  export LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding
  # or:   qwen3_embedding_context
  # or:   qwen3_embedding_reranker
  # or:   qwen3_embedding_context_reranker

Pre-warm the tool embedding cache (run once):
  cd /path/to/ToolSelectionBenchmark
  export QWEN3_EMBEDDING_BASE_URL=${EMBEDDING_ENDPOINT}
  python -m wtb.model_handler.api_inference.setup_openai_embeddings \\
      --provider qwen3 \\
      --tools-file multi-agent-framework/tools/tools_en.jsonl
EOF

echo "[INFO] Endpoint info saved to: ${ENDPOINT_FILE}"
cat "${ENDPOINT_FILE}"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "[INFO] GPU status:"
nvidia-smi

# ── Run the deployment script ─────────────────────────────────────────────────
echo ""
echo "[INFO] Starting embedding server via slurm_vllm_embedding_deploy.sh ..."
cd "${SLURM_SUBMIT_DIR}"
bash deploy/slurm_vllm_embedding_deploy.sh

# Keep job alive on unexpected exit (helps with debugging)
if [ $? -ne 0 ]; then
    echo "[ERROR] Embedding server exited unexpectedly. Keeping job alive for debugging..."
    sleep infinity
fi
