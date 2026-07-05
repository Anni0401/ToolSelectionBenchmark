#!/bin/bash
# ============================================================================
# WildToolBench — Full Benchmark Submission Script
# ============================================================================
#
# Submits all required SLURM jobs for one complete benchmark run of the given
# tool-selection strategy. No interactive sessions needed.
#
# Usage:
#   bash deploy/submit_benchmark.sh <MODE> [OPTIONS]
#
# Selection modes:
#   in_context                        No auxiliary server (baseline)
#   hierarchical                      Qwen3-30B-A3B selector LLM (A40×2)
#   qwen3_embedding                   Qwen3-Embedding-8B, query only (A40×1)
#   qwen3_embedding_context           Qwen3-Embedding-8B, full context (A40×1)
#   qwen3_embedding_reranker          Qwen3-Embedding-8B + gpt-oss reranking (A40×1)
#   qwen3_embedding_context_reranker  …context variant (A40×1)
#
# Options:
#   --time-exec <HH:MM:SS>   Wall time for the executing LLM job (default: 24:00:00)
#   --time-run  <HH:MM:SS>   Wall time for the benchmark runner job (default: 12:00:00)
#   --dry-run                Print sbatch commands without submitting
#
# Job topology:
#   in_context:  single H200×4 job  (gptoss + LangGraph app + benchmark)
#   all others:  H200×4 job         (gptoss server, stays alive until runner finishes)
#              + A40 job            (aux vLLM server + LangGraph app + benchmark;
#                                    cancels H200 job on exit)
#
# The two jobs communicate via a shared directory on the home filesystem:
#   ~/wtb_runs/<RUN_ID>/config.env          — static config sourced by every job
#   ~/wtb_runs/<RUN_ID>/exec_endpoint.txt   — written by H200 job once vLLM is up
#   ~/wtb_runs/<RUN_ID>/logs/               — per-job SLURM stdout / stderr
# ============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${PROJECT_ROOT}/deploy"

# ── Parse arguments ───────────────────────────────────────────────────────────
MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "Usage: bash deploy/submit_benchmark.sh <MODE> [--time-exec HH:MM:SS] [--time-run HH:MM:SS] [--dry-run]"
    echo ""
    echo "Modes: in_context | hierarchical | qwen3_embedding | qwen3_embedding_context"
    echo "       qwen3_embedding_reranker | qwen3_embedding_context_reranker"
    exit 1
fi

EXEC_TIME="24:00:00"
RUN_TIME="12:00:00"
DRY_RUN=0

shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time-exec) EXEC_TIME="$2"; shift 2 ;;
        --time-run)  RUN_TIME="$2";  shift 2 ;;
        --dry-run)   DRY_RUN=1;      shift   ;;
        *) echo "[WARN] Unknown option: $1"; shift ;;
    esac
done

# ── Validate mode ─────────────────────────────────────────────────────────────
case "$MODE" in
    in_context|hierarchical|qwen3_embedding|qwen3_embedding_context|\
    qwen3_embedding_reranker|qwen3_embedding_context_reranker)
        : ;;  # valid
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        echo "Valid modes: in_context | hierarchical | qwen3_embedding | qwen3_embedding_context"
        echo "             qwen3_embedding_reranker | qwen3_embedding_context_reranker"
        exit 1 ;;
esac

# ── Create run directory ──────────────────────────────────────────────────────
RUN_ID="wtb_$(date +%Y%m%d_%H%M%S)_${MODE}"
RUN_DIR="${HOME}/wtb_runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ── Write static config sourced by all job scripts ────────────────────────────
cat > "${RUN_DIR}/config.env" << EOF
SELECTION_MODE="${MODE}"
PROJECT_ROOT="${PROJECT_ROOT}"
EOF

# ── Helper ────────────────────────────────────────────────────────────────────
do_sbatch() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] sbatch $*" >&2
        echo "9999999"   # fake job ID
    else
        sbatch "$@"
    fi
}

echo "=========================================="
echo "WildToolBench Benchmark Submission"
echo "=========================================="
echo "Mode:       ${MODE}"
echo "Run ID:     ${RUN_ID}"
echo "Run dir:    ${RUN_DIR}"
echo "Exec time:  ${EXEC_TIME}"
echo "Run time:   ${RUN_TIME}"
echo "Dry run:    ${DRY_RUN}"
echo "=========================================="

# ── in_context: single self-contained H200 job ───────────────────────────────
if [[ "$MODE" == "in_context" ]]; then
    JOB_ID=$(do_sbatch \
        --parsable \
        --time="${RUN_TIME}" \
        --output="${LOG_DIR}/run_%j.log" \
        --error="${LOG_DIR}/run_%j.err" \
        --export=ALL,RUN_DIR="${RUN_DIR}" \
        "${DEPLOY_DIR}/slurm_in_context_job.sh")

    echo "[INFO] Submitted in_context job: ${JOB_ID}"
    echo "${JOB_ID}" > "${RUN_DIR}/run_job_id.txt"

    echo ""
    echo "Monitor:"
    echo "  squeue -j ${JOB_ID}"
    echo "  tail -f ${LOG_DIR}/run_${JOB_ID}.log"
    exit 0
fi

# ── All other modes: two-job pipeline ────────────────────────────────────────

# Step 1 — Executing LLM server (H200×4, runs until cancelled by runner job)
# Resource flags (#SBATCH headers) are in slurm_exec_llm_job.sh;
# only pass run-specific overrides here.
EXEC_JID=$(do_sbatch \
    --parsable \
    --time="${EXEC_TIME}" \
    --output="${LOG_DIR}/exec_llm_%j.log" \
    --error="${LOG_DIR}/exec_llm_%j.err" \
    --export=ALL,RUN_DIR="${RUN_DIR}" \
    "${DEPLOY_DIR}/slurm_exec_llm_job.sh")

echo "[INFO] Submitted exec LLM job: ${EXEC_JID}"
echo "${EXEC_JID}" > "${RUN_DIR}/exec_job_id.txt"
# Append so the runner job can read it from config.env
echo "EXEC_JID=${EXEC_JID}" >> "${RUN_DIR}/config.env"

# Step 2 — Override gres/mem for modes that need more than 1× A40
# The #SBATCH default in slurm_aux_and_runner_job.sh is A40×1 (embedding modes).
# hierarchical needs A40×2 for Qwen3-30B-A3B ≈ 60 GB.
case "$MODE" in
    hierarchical)
        AUX_GRES_OVERRIDE="--gres=gpu:A40:2 --mem=96G"
        ;;
    *)
        AUX_GRES_OVERRIDE=""   # use #SBATCH defaults (A40×1 / 48G)
        ;;
esac

# Step 3 — Aux server + benchmark runner (starts once the H200 job is running)
# shellcheck disable=SC2086  # intentional word-splitting of AUX_GRES_OVERRIDE
RUN_JID=$(do_sbatch \
    --parsable \
    ${AUX_GRES_OVERRIDE} \
    --time="${RUN_TIME}" \
    --output="${LOG_DIR}/runner_%j.log" \
    --error="${LOG_DIR}/runner_%j.err" \
    --dependency="after:${EXEC_JID}" \
    --export=ALL,RUN_DIR="${RUN_DIR}" \
    "${DEPLOY_DIR}/slurm_aux_and_runner_job.sh")

echo "[INFO] Submitted runner job:  ${RUN_JID}  (depends on exec job ${EXEC_JID} starting)"
echo "${RUN_JID}" > "${RUN_DIR}/run_job_id.txt"

echo ""
echo "=========================================="
echo "Two-job pipeline queued for mode: ${MODE}"
echo ""
echo "Monitor:"
echo "  squeue --me"
echo "  tail -f ${LOG_DIR}/exec_llm_${EXEC_JID}.log"
echo "  tail -f ${LOG_DIR}/runner_${RUN_JID}.log"
echo ""
echo "Results will appear in:"
echo "  ${PROJECT_ROOT}/wild-tool-bench/result/"
echo "=========================================="
