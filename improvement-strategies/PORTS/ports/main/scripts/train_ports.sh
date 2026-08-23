#!/bin/bash

set -e

# Default values (aligned with Makefile and other scripts)
DATASET_NAME="${DATASET_NAME:-toolbench}"
RETRIEVAL_MODEL_NAME="${RETRIEVAL_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
INFERENCE_MODEL_PSEUDONAME="${INFERENCE_MODEL_PSEUDONAME:-llama3-8B}"
RETRIEVAL_MAX_SEQ_LEN="${RETRIEVAL_MAX_SEQ_LEN:-512}"
INFERENCE_MAX_SEQ_LEN="${INFERENCE_MAX_SEQ_LEN:-1024}"
PADDING_SIDE="${PADDING_SIDE:-left}"
N_NEGS="${N_NEGS:-3}"
LR="${LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-16}"
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-0.3}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-200}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-PORTS_AAAI-EMNLP}"
LOG_FREQ="${LOG_FREQ:-20}"
BETA="${BETA:-0.5}"
GAMMA="${GAMMA:-0.5}"
PREF_BETA="${PREF_BETA:-1}"
CORPUS_UPDATES="${CORPUS_UPDATES:-1}"
# LoRA / QLoRA for the retriever
USE_LORA="${USE_LORA:-true}"
USE_QLORA="${USE_QLORA:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
# Keep embedding_update_steps for backward compatibility
EMBEDDING_UPDATE_STEPS="${EMBEDDING_UPDATE_STEPS:-}"
# Add n_reembedding_steps with priority
N_REEMBEDDING_STEPS="${N_REEMBEDDING_STEPS:-}"
N_EPOCHS="${N_EPOCHS:-5}"
SEED="${SEED:-42}"
EVAL_STEPS="${EVAL_STEPS:-0.2}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${SAVE_STEPS:-}"
SAVE_DIR="${SAVE_DIR:-${WORK:-$HOME}/ports/main/output/ports/ports_retriever_$(date +%Y%m%d_%H%M%S)}" # Default SAVE_DIR if not set
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-}"
K_EVAL_VALUES_ACCURACY="${K_EVAL_VALUES_ACCURACY:-1 3 5}"
K_EVAL_VALUES_NDCG="${K_EVAL_VALUES_NDCG:-1 3 5}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"

WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET_NAME}-${RETRIEVAL_MODEL_NAME}-${INFERENCE_MODEL_PSEUDONAME}-B${BETA}-G${GAMMA}-ORPOB${PREF_BETA}-LR${LR}}"

# Ensure the save directory exists
mkdir -p "$SAVE_DIR"

SAVE_ARGS=""
if [ -n "$SAVE_STEPS" ]; then
  SAVE_ARGS="--save_steps $SAVE_STEPS"
fi
MAX_CHECKPOINTS_ARGS=""
if [ -n "$MAX_CHECKPOINTS" ]; then
  MAX_CHECKPOINTS_ARGS="--max_checkpoints $MAX_CHECKPOINTS"
fi

# Handle embedding update parameters - prioritize n_reembedding_steps if both are defined
N_REEMBEDDING_ARGS=""
if [ -n "$N_REEMBEDDING_STEPS" ]; then
  N_REEMBEDDING_ARGS="--n_reembedding_steps $N_REEMBEDDING_STEPS"
elif [ -n "$EMBEDDING_UPDATE_STEPS" ]; then
  N_REEMBEDDING_ARGS="--embedding_update_steps $EMBEDDING_UPDATE_STEPS"
fi

PYTHON_SCRIPT="${PYTHON_SCRIPT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/main_train_port.py}"

python3 $PYTHON_SCRIPT \
    --dataset $DATASET_NAME \
    --inference_model_name $INFERENCE_MODEL_PSEUDONAME \
    --retrieval_model_name $RETRIEVAL_MODEL_NAME \
    --retriever_max_seq_length $RETRIEVAL_MAX_SEQ_LEN \
    --inference_max_seq_length $INFERENCE_MAX_SEQ_LEN \
    --n_epochs $N_EPOCHS \
    --lr $LR \
    --lr_type $LR_SCHEDULER \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --preprocessing_batch_size $PREPROCESS_BATCH_SIZE \
    $N_REEMBEDDING_ARGS \
    --padding_side $PADDING_SIDE \
    --lambda_loss $LAMBDA_WEIGHT \
    --n_neg_examples $N_NEGS \
    --gamma $GAMMA \
    --beta $BETA \
    --preference_weight $PREF_BETA \
    --seed $SEED \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --wandb_run_name $WANDB_RUN_NAME \
    --log_freq $LOG_FREQ \
    --do_train \
    --do_eval \
    --eval_strategy "steps" \
    --eval_steps $EVAL_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --save_strategy $SAVE_STRATEGY \
    --save_dir "$SAVE_DIR" \
    $SAVE_ARGS \
    $MAX_CHECKPOINTS_ARGS \
    $([ "$SAVE_CHECKPOINTS" = "true" ] && echo "--save_checkpoints") \
    --k_eval_values_accuracy $K_EVAL_VALUES_ACCURACY \
    --k_eval_values_ndcg $K_EVAL_VALUES_NDCG \
    --load_in_4bit \
    $([ "$USE_LORA" = "true" ] && echo "--use_lora --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT") \
    $([ "$USE_QLORA" = "true" ] && echo "--use_qlora") \
    --max_train_samples $MAX_TRAIN_SAMPLES