#!/bin/bash

set -e

# Default values (aligned with Makefile and train_ports.sh)
DATASET_NAME="${DATASET_NAME:-toolbench}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-base-en-v1.5}"
EPOCHS="${EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
PREPROCESSING_BATCH_SIZE="${PREPROCESSING_BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
EVAL_STEPS_FRACTION="${EVAL_STEPS_FRACTION:-0.2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
SCHEDULER="${SCHEDULER:-warmupcosine}"
POOLING="${POOLING:-mean}"
NEGATIVES_PER_SAMPLE="${NEGATIVES_PER_SAMPLE:-1}"
SEED="${SEED:-42}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
K_EVAL_VALUES_ACCURACY="${K_EVAL_VALUES_ACCURACY:-1 3 5 10}"
K_EVAL_VALUES_NDCG="${K_EVAL_VALUES_NDCG:-1 3 5 10}"

WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-MNRL_Training}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-MNRL-${DATASET_NAME}-$(basename ${MODEL_NAME})-LR${LR}-E${EPOCHS}}"
LOG_FREQ=${LOG_FREQ:-50}

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/ports/main/output/mnrl/mnrl_retriever_${DATASET_NAME}_$(basename ${MODEL_NAME})_$(date +%Y%m%d_%H%M%S)}"

# Add save_checkpoints parameter
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-false}"

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"

PYTHON_SCRIPT="/workspace/main/train_mnrl.py"

python3 $PYTHON_SCRIPT \
    --dataset "$DATASET_NAME" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --preprocessing_batch_size $PREPROCESSING_BATCH_SIZE \
    --retriever_max_seq_length $MAX_SEQ_LENGTH \
    --lr $LR \
    --scheduler "$SCHEDULER" \
    --pooling "$POOLING" \
    --warmup_ratio $WARMUP_RATIO \
    --eval_steps $EVAL_STEPS_FRACTION \
    --negatives_per_sample $NEGATIVES_PER_SAMPLE \
    --k_eval_values_accuracy $K_EVAL_VALUES_ACCURACY \
    --k_eval_values_ndcg $K_EVAL_VALUES_NDCG \
    --random_negatives \
    --evaluate_on_test \
    --use_pre_trained_model \
    --push_to_hub \
    --public_model \
    --hub_repo_name "mnrl-${DATASET_NAME}-$(basename ${MODEL_NAME})" \
    --log_file "$OUTPUT_DIR/training.log" \
    --seed $SEED \
    --wandb_project_name $WANDB_PROJECT_NAME \
    --wandb_run_name $WANDB_RUN_NAME \
    --wandb_log_freq $LOG_FREQ \
    $([ "$SAVE_CHECKPOINTS" = "true" ] && echo "--save_checkpoints") \
    --max_train_samples $MAX_TRAIN_SAMPLES
