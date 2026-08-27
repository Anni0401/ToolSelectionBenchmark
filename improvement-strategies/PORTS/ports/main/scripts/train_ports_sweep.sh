#!/bin/bash
# Entry point used by `wandb agent` for the PORTS hyperparameter sweep
# (see ../sweep_ports.yaml). It is a thin wrapper around train_ports.sh:
# it sets the same fixed defaults, but appends "$@" (the swept
# --lr/--n_epochs/--lambda_loss/--beta/--gamma values injected by the W&B
# agent via the sweep's `${args}` macro) as the LAST arguments to the
# python call, so argparse's "last value wins" behaviour lets them
# override the fixed defaults below.

set -e

# Fixed (non-swept) defaults - override via env vars if needed.
DATASET_NAME="${DATASET_NAME:-toolbench}"
RETRIEVAL_MODEL_NAME="${RETRIEVAL_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
INFERENCE_MODEL_PSEUDONAME="${INFERENCE_MODEL_PSEUDONAME:-llama3-8B}"
RETRIEVAL_MAX_SEQ_LEN="${RETRIEVAL_MAX_SEQ_LEN:-512}"
INFERENCE_MAX_SEQ_LEN="${INFERENCE_MAX_SEQ_LEN:-1024}"
PADDING_SIDE="${PADDING_SIDE:-left}"
N_NEGS="${N_NEGS:-3}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-16}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-PORTS_Sweep}"
LOG_FREQ="${LOG_FREQ:-20}"
PREF_BETA="${PREF_BETA:-1}"
USE_LORA="${USE_LORA:-true}"
USE_QLORA="${USE_QLORA:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
SEED="${SEED:-42}"
EVAL_STEPS="${EVAL_STEPS:-0.2}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"
K_EVAL_VALUES_ACCURACY="${K_EVAL_VALUES_ACCURACY:-1 3 5}"
K_EVAL_VALUES_NDCG="${K_EVAL_VALUES_NDCG:-1 3 5}"
# Unique per-process save dir (hostname + PID) so parallel sweep agents never collide.
SAVE_DIR="${SAVE_DIR:-${WORK:-$HOME}/ports/main/output/ports/sweep_$(date +%Y%m%d_%H%M%S)_$(hostname)_$$}"

mkdir -p "$SAVE_DIR"

PYTHON_SCRIPT="${PYTHON_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/main_train_port.py}"

echo "===================================================="
echo "W&B sweep run - fixed params + swept overrides: $*"
echo "===================================================="

# NOTE: no --lr/--n_epochs/--lambda_loss/--beta/--gamma/--wandb_run_name here -
# they are supplied by the W&B agent via "$@" and/or use main_train_port.py's
# own defaults; "$@" is appended last so it always wins over any duplicate flag.
python3 "$PYTHON_SCRIPT" \
    --dataset "$DATASET_NAME" \
    --inference_model_name "$INFERENCE_MODEL_PSEUDONAME" \
    --retrieval_model_name "$RETRIEVAL_MODEL_NAME" \
    --retriever_max_seq_length "$RETRIEVAL_MAX_SEQ_LEN" \
    --inference_max_seq_length "$INFERENCE_MAX_SEQ_LEN" \
    --lr_type "$LR_SCHEDULER" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --preprocessing_batch_size "$PREPROCESS_BATCH_SIZE" \
    --padding_side "$PADDING_SIDE" \
    --n_neg_examples "$N_NEGS" \
    --preference_weight "$PREF_BETA" \
    --seed "$SEED" \
    --wandb_project_name "$WANDB_PROJECT_NAME" \
    --log_freq "$LOG_FREQ" \
    --do_train \
    --do_eval \
    --eval_strategy "steps" \
    --eval_steps "$EVAL_STEPS" \
    --warmup_ratio "$WARMUP_RATIO" \
    --save_strategy "$SAVE_STRATEGY" \
    --save_dir "$SAVE_DIR" \
    $([ "$SAVE_CHECKPOINTS" = "true" ] && echo "--save_checkpoints") \
    --k_eval_values_accuracy $K_EVAL_VALUES_ACCURACY \
    --k_eval_values_ndcg $K_EVAL_VALUES_NDCG \
    --load_in_4bit \
    $([ "$USE_LORA" = "true" ] && echo "--use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT") \
    $([ "$USE_QLORA" = "true" ] && echo "--use_qlora") \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    "$@"
