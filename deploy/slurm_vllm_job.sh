#!/bin/bash
#SBATCH --job-name=qwen-vllm-selector
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/vllm_deployment_%j.log
#SBATCH --error=/home/%u/vllm_deployment_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%ann-kathrin.herrmann@students.uni-mannheim.de

# ============================================================================
# SLURM Job Script: vLLM Qwen Model Deployment for Hierarchical Selection
# ============================================================================
# 
# Submits vLLM server job to SLURM cluster for hosting the Qwen2.5-7B model
# used as a selector LLM in the hierarchical tool selection strategy.
#
# Usage:
#   sbatch slurm_vllm_job.sh
#   # or with custom parameters:
#   sbatch --gpus-per-node=2 slurm_vllm_job.sh
#
# Configuration:
#   Adjust SBATCH directives above for your cluster specifications
#
# ============================================================================

set -e

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node(s): ${SLURM_NODELIST}"
echo "Number of GPUs: ${SLURM_GPUS_PER_NODE}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM}"
echo "=========================================="

# Source user's shell configuration
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Source user's shell configuration
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Activate conda environment with Python 3.10
echo "[INFO] Setting up Python environment..."
if ! command -v conda &> /dev/null; then
    if [ -d "$HOME/miniconda3" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -d "$HOME/anaconda3" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

# Check if conda environment exists and activate it
VENV_NAME="vllm_py310"
if conda env list | grep -q "^${VENV_NAME}"; then
    conda activate "$VENV_NAME"
    echo "[SUCCESS] Activated Python 3.10 environment"
else
    echo "[WARNING] vllm_py310 environment not found. Using system Python."
fi

# Set up environment variables
export MODEL_NAME="Qwen/Qwen3-30B-A3B"
export VLLM_PORT=8000
export GPU_MEMORY_UTILIZATION=0.9
export MAX_BATCH_SIZE=32
export DTYPE=float16
export TENSOR_PARALLEL_SIZE=2

# Optional: Set conda environment name if you have one
# export VLLM_ENV=tool-selection

# Create output directory for this job
JOB_LOG_DIR="/tmp/vllm_job_${SLURM_JOB_ID}"
mkdir -p "${JOB_LOG_DIR}"

echo "[INFO] Logs directory: ${JOB_LOG_DIR}"

# Get the hostname and port for the endpoint
HOSTNAME=$(hostname -f)
ENDPOINT="http://${HOSTNAME}:${VLLM_PORT}/v1/chat/completions"

echo "[INFO] vLLM Endpoint will be available at: ${ENDPOINT}"
echo "[INFO] Note: Replace hostname with actual IP if needed"

# Save endpoint information to file for easy access
ENDPOINT_FILE="${JOB_LOG_DIR}/endpoint.txt"
cat > "${ENDPOINT_FILE}" << EOF
Job ID: ${SLURM_JOB_ID}
Hostname: ${HOSTNAME}
Port: ${VLLM_PORT}
Endpoint: ${ENDPOINT}

Environment Variables to Set:
export LANGGRAPH_SELECTOR_LLM_ENDPOINT=${ENDPOINT}
export LANGGRAPH_TOOL_SELECTION_MODE=hierarchical
export LANGGRAPH_SELECTOR_LLM_MODEL=${MODEL_NAME}
EOF

echo "[INFO] Endpoint information saved to: ${ENDPOINT_FILE}"
cat "${ENDPOINT_FILE}"

# Load necessary modules (adjust to your cluster setup)
# module load cuda/12.0
# module load gcc/11
# module load openmpi

echo "[INFO] Loading CUDA and environment..."
nvidia-smi

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR}"
echo "[INFO] Working directory: ${PWD}"

# Run the deployment script
echo "[INFO] Starting vLLM deployment script..."
bash deploy/slurm_vllm_deploy.sh

# Keep the job alive if vLLM exits unexpectedly
if [ $? -ne 0 ]; then
    echo "[ERROR] vLLM server crashed! Keeping job alive for debugging..."
    sleep infinity
fi
