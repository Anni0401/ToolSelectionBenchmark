#!/bin/bash

# Default configuration
OUTPUT_DIR="${HOME}/ports/output" 
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"
DATASET="toolbench"
RETRIEVAL_MODEL="BAAI/bge-base-en-v1.5"
BATCH_SIZE=2
EVAL_BATCH_SIZE=2
PREPROCESS_BATCH_SIZE=8
LR=1e-3
RETRIEVAL_MAX_SEQ_LEN=512
EVAL_STEPS=0.2
WARMUP_RATIO=0.1
SCHEDULER="warmupcosine"
POOLING="cls"
NEGATIVES_PER_SAMPLE=3
SEED=42
MAX_TRAIN_SAMPLES=10000
WANDB_PROJECT_NAME="PORTS_AAAI-EMNLP"
WANDB_RUN_NAME="MNRL_toolbench_g3"
LOG_FREQ=50
K_EVAL_VALUES_ACCURACY="1 3 5 10"
K_EVAL_VALUES_NDCG="1 3 5 10"
EPOCHS=1
SAVE_CHECKPOINTS=false
INFERENCE_MODEL="llama3-8B"  # Added but will be ignored
WEIGHT_DECAY=0.01

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset=*) DATASET="${1#*=}" ;;
    --retrieval_model=*) RETRIEVAL_MODEL="${1#*=}" ;;
    --batch_size=*) BATCH_SIZE="${1#*=}" ;;
    --eval_batch_size=*) EVAL_BATCH_SIZE="${1#*=}" ;;
    --preprocess_batch_size=*) PREPROCESS_BATCH_SIZE="${1#*=}" ;;
    --lr=*) LR="${1#*=}" ;;
    --retrieval_max_seq_len=*) RETRIEVAL_MAX_SEQ_LEN="${1#*=}" ;;
    --eval_steps=*) EVAL_STEPS="${1#*=}" ;;
    --warmup_ratio=*) WARMUP_RATIO="${1#*=}" ;;
    --scheduler=*) SCHEDULER="${1#*=}" ;;
    --pooling=*) POOLING="${1#*=}" ;;
    --negatives_per_sample=*) NEGATIVES_PER_SAMPLE="${1#*=}" ;;
    --seed=*) SEED="${1#*=}" ;;
    --max_train_samples=*) MAX_TRAIN_SAMPLES="${1#*=}" ;;
    --wandb_project_name=*) WANDB_PROJECT_NAME="${1#*=}" ;;
    --wandb_run_name=*) WANDB_RUN_NAME="${1#*=}" ;;
    --log_freq=*) LOG_FREQ="${1#*=}" ;;
    --k_eval_values_accuracy=*) K_EVAL_VALUES_ACCURACY="${1#*=}" ;;
    --k_eval_values_ndcg=*) K_EVAL_VALUES_NDCG="${1#*=}" ;;
    --epochs=*) EPOCHS="${1#*=}" ;;
    --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
    --save_checkpoints=*) SAVE_CHECKPOINTS="${1#*=}" ;;
    --inference_model=*) INFERENCE_MODEL="${1#*=}" ;;  # Accept but ignore this parameter
    --weight_decay=*) WEIGHT_DECAY="${1#*=}" ;;
    --lambda_loss=*) ;; # Accept but ignore
    --lambda_weight=*) ;; # Accept but ignore
    --beta=*) ;; # Accept but ignore
    --gamma=*) ;; # Accept but ignore
    --inference_max_seq_len=*) ;; # Accept but ignore
    --preference_weight=*) ;; # Accept but ignore
    --n_reembedding_steps=*) ;; # Accept but ignore
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Note: INFERENCE_MODEL is accepted in the arguments above but not passed to the Python script below

docker run \
    -v "$OUTPUT_DIR":/workspace/output \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    --memory="30g" \
    --rm \
    --gpus "device=${CUDA_VISIBLE_DEVICES}" \
    ports_image \
    mnrl \
    DATASET=$DATASET \
    RETRIEVAL_MODEL=$RETRIEVAL_MODEL \
    BATCH_SIZE=$BATCH_SIZE \
    EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
    PREPROCESS_BATCH_SIZE=$PREPROCESS_BATCH_SIZE \
    LR=$LR \
    RETRIEVAL_MAX_SEQ_LEN=$RETRIEVAL_MAX_SEQ_LEN \
    EVAL_STEPS=$EVAL_STEPS \
    WARMUP_RATIO=$WARMUP_RATIO \
    SCHEDULER=$SCHEDULER \
    POOLING=$POOLING \
    NEGATIVES_PER_SAMPLE=$NEGATIVES_PER_SAMPLE \
    SEED=$SEED \
    MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES \
    OUTPUT_DIR=/workspace/output \
    WANDB_PROJECT_NAME=$WANDB_PROJECT_NAME \
    WANDB_RUN_NAME=$WANDB_RUN_NAME \
    LOG_FREQ=$LOG_FREQ \
    K_EVAL_VALUES_ACCURACY="$K_EVAL_VALUES_ACCURACY" \
    K_EVAL_VALUES_NDCG="$K_EVAL_VALUES_NDCG" \
    EPOCHS=$EPOCHS \
    SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS \
    WEIGHT_DECAY=$WEIGHT_DECAY