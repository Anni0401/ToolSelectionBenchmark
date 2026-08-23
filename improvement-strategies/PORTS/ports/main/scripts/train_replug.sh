#!/bin/bash

set -e

# Default values (aligned with Makefile and train_ports.sh)
DATASET_NAME="${DATASET_NAME:-toolbench}"
RETRIEVAL_MODEL_NAME="${RETRIEVAL_MODEL_NAME:-BAAI/bge-base-en-v1.5}"
INFERENCE_MODEL_PSEUDONAME="${INFERENCE_MODEL_PSEUDONAME:-llama3-8B}"
RETRIEVAL_MAX_SEQ_LEN="${RETRIEVAL_MAX_SEQ_LEN:-512}"
INFERENCE_MAX_SEQ_LEN="${INFERENCE_MAX_SEQ_LEN:-1024}"  # Added parameter
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
EPOCHS="${EPOCHS:-5}"
NUM_RETRIEVED_DOCS="${NUM_RETRIEVED_DOCS:-5}"
GAMMA="${GAMMA:-0.5}"
BETA="${BETA:-0.5}"
EVAL_STEPS_FRACTION="${EVAL_STEPS_FRACTION:-0.2}"
SEED="${SEED:-42}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-true}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-REPLUG_Training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-REPLUG-${DATASET_NAME}-R_$(basename ${RETRIEVAL_MODEL_NAME})-I_${INFERENCE_MODEL_PSEUDONAME}-LR${LR}}"
SAVE_PATH="${SAVE_PATH:-${HOME}/ports/main/output/replug/replug_retriever_${DATASET_NAME}_$(basename ${RETRIEVAL_MODEL_NAME})_$(date +%Y%m%d_%H%M%S)}"
K_EVAL_VALUES_ACCURACY="${K_EVAL_VALUES_ACCURACY:-1 3 5}"
K_EVAL_VALUES_NDCG="${K_EVAL_VALUES_NDCG:-1 3 5}"
# Updated parameter for corpus embedding updates
CORPUS_UPDATES="${CORPUS_UPDATES:-50}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-16}"

# Add save_checkpoints parameter
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"

# Map pseudo-name to actual model path/ID and type
case $INFERENCE_MODEL_PSEUDONAME in
  "llama3-8B")
    INFERENCE_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
    INFERENCE_MODEL_TYPE="llama3"
    ;;
  "codestral-22B")
    INFERENCE_MODEL_NAME="mistralai/Codestral-22B-v0.1"
    INFERENCE_MODEL_TYPE="codestral"
    ;;
  "gemma2-2B")
    INFERENCE_MODEL_NAME="google/gemma-2-2b-it"
    INFERENCE_MODEL_TYPE="gemma"
    ;;
  "groqLlama3Tool-8B")
    INFERENCE_MODEL_NAME="Groq/Llama-3-Groq-8B-Tool-Use"
    INFERENCE_MODEL_TYPE="llama3groq"
    ;;
  "llama3.2")
    INFERENCE_MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    INFERENCE_MODEL_TYPE="llama3"
    ;;
  "gemma3")
    INFERENCE_MODEL_NAME="google/gemma-3-1b-it"
    INFERENCE_MODEL_TYPE="gemma3"
    ;;
  "qwen3")
    INFERENCE_MODEL_NAME="Qwen/Qwen3-4B"
    INFERENCE_MODEL_TYPE="qwen3"
    ;;
  *)
    echo "Error: Unknown inference model pseudo-name: $INFERENCE_MODEL_PSEUDONAME"
    exit 1
    ;;
esac

# Ensure the save directory exists
mkdir -p "$SAVE_PATH"

PYTHON_SCRIPT="/workspace/main/train_replug.py"

python3 $PYTHON_SCRIPT \
    --retr_model_name_or_path "$RETRIEVAL_MODEL_NAME" \
    --infer_model_name_or_path "$INFERENCE_MODEL_NAME" \
    --infer_model_type "$INFERENCE_MODEL_TYPE" \
    --dataset "$DATASET_NAME" \
    --num_train_epochs $EPOCHS \
    --batch_size $TRAIN_BATCH_SIZE \
    --learning_rate $LR \
    --lr_scheduler "$LR_SCHEDULER" \
    --num_retrieved_docs_per_query $NUM_RETRIEVED_DOCS \
    --gamma_value $GAMMA \
    --beta_value $BETA \
    --seed $SEED \
    --log_to_wandb \
    --wandb_proj_name "$WANDB_RUN_NAME" \
    --trained_model_save_path "$SAVE_PATH" \
    --retr_max_seq_length $RETRIEVAL_MAX_SEQ_LEN \
    --inference_max_seq_length $INFERENCE_MAX_SEQ_LEN \
    --eval_steps $EVAL_STEPS_FRACTION \
    --warmup_ratio $WARMUP_RATIO \
    --k_eval_values_accuracy $K_EVAL_VALUES_ACCURACY \
    --k_eval_values_ndcg $K_EVAL_VALUES_NDCG \
    --corpus_updates $CORPUS_UPDATES \
    --preprocess_batch_size $PREPROCESS_BATCH_SIZE \
    $([ "$LOAD_IN_4BIT" = "true" ] && echo "--quantization_4bit") \
    $([ "$SAVE_CHECKPOINTS" = "true" ] && echo "--save_checkpoints") \
    --max_train_samples $MAX_TRAIN_SAMPLES

