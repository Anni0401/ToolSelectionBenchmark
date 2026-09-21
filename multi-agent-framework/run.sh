#!/bin/bash
# Entry point used by `wandb agent` for the DPO Qwen3 LoRA hyperparameter sweep
# (see sweep.yaml in this same directory). Thin wrapper around
# train_dpo_qwen3_lora.py: sets fixed (non-swept) defaults, then appends "$@"
# (the swept --learning-rate/--beta/--epochs values injected by the W&B agent
# via the sweep's ${args} macro) as the LAST arguments, so argparse's
# "last value wins" behaviour lets them override the fixed defaults below.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed (non-swept) defaults - override via env vars if needed.
INPUT_JSON="${INPUT_JSON:-${SCRIPT_DIR}/queries_gold_tools_batch1_dpo_ranked.json}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
VAL_RATIO="${VAL_RATIO:-0.2}"
MIN_SCORE_MARGIN="${MIN_SCORE_MARGIN:-0.0}"  # full dataset, no score-diff filtering
GROUP_BY="${GROUP_BY:-gold_tools}"
SEED="${SEED:-42}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-DPO_Qwen3_Sweep}"

export WANDB_PROJECT="${WANDB_PROJECT_NAME}"

# Unique per-run output dir so parallel sweep agents never collide.
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/sweeps/run_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$}"
mkdir -p "${OUTPUT_DIR}"

echo "===================================================="
echo "W&B sweep run - fixed params + swept overrides: $*"
echo "Output dir: ${OUTPUT_DIR}"
echo "===================================================="

# NOTE: no --learning-rate/--beta/--epochs here - they are supplied by the
# W&B agent via "$@" (falling back to train_dpo_qwen3_lora.py's own argparse
# defaults if this were ever invoked without a sweep); "$@" is appended last
# so it always wins over any duplicate flag.
python -u "${SCRIPT_DIR}/train_dpo_qwen3_lora.py" \
    --input "${INPUT_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "${MODEL_NAME}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --per-device-train-batch-size "${PER_DEVICE_BATCH}" \
    --per-device-eval-batch-size "${PER_DEVICE_BATCH}" \
    --max-length "${MAX_LENGTH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --val-ratio "${VAL_RATIO}" \
    --min-score-margin "${MIN_SCORE_MARGIN}" \
    --group-by "${GROUP_BY}" \
    --seed "${SEED}" \
    --lora-r "${LORA_R}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lora-dropout "${LORA_DROPOUT}" \
    --gradient-checkpointing \
    --bf16 \
    --logging-steps 1 \
    --save-strategy epoch \
    --report-to wandb \
    "$@"
