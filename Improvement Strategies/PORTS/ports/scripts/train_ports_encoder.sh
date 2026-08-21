#!/bin/bash

# Default configuration
OUTPUT_DIR="${HOME}/ports/output" 
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"
DATASET="toole"
INFERENCE_MODEL="gemma3"
RETRIEVAL_MODEL="BAAI/bge-base-en-v1.5"
RETRIEVAL_MAX_SEQ_LEN=512
INFERENCE_MAX_SEQ_LEN=1024
PADDING_SIDE="left"
N_NEGS=3
EVAL_BATCH_SIZE=2
PREPROCESS_BATCH_SIZE=2
LAMBDA_WEIGHT=0.1
WANDB_PROJECT_NAME="PORTS_AAAI-EMNLP"
WANDB_RUN_NAME="PORTS_toolbench_g3"
LOG_FREQ=10
BETA=0.5
GAMMA=0.5
PREF_BETA=1
CORPUS_UPDATES=5000
EPOCHS=1
BATCH_SIZE=2
LR=2e-5
SCHEDULER="cosine"
SEED=42
EVAL_STEPS=0.2
USE_4BIT=true
WARMUP_RATIO=0.2
SAVE_STRATEGY="epoch"
K_EVAL_VALUES_ACCURACY="1 3 5 10 20"
K_EVAL_VALUES_NDCG="1 3 5 10 20"
MAX_TRAIN_SAMPLES=10000
SAVE_CHECKPOINTS=false
WEIGHT_DECAY=0.01
EMBEDDING_UPDATE_STEPS=50
N_REEMBEDDING_STEPS=10

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset=*) DATASET="${1#*=}" ;;
    --inference_model=*) INFERENCE_MODEL="${1#*=}" ;;
    --retrieval_model=*) RETRIEVAL_MODEL="${1#*=}" ;;
    --retrieval_max_seq_len=*) RETRIEVAL_MAX_SEQ_LEN="${1#*=}" ;;
    --inference_max_seq_len=*) INFERENCE_MAX_SEQ_LEN="${1#*=}" ;;
    --padding_side=*) PADDING_SIDE="${1#*=}" ;;
    --n_negs=*) N_NEGS="${1#*=}" ;;
    --eval_batch_size=*) EVAL_BATCH_SIZE="${1#*=}" ;;
    --preprocess_batch_size=*) PREPROCESS_BATCH_SIZE="${1#*=}" ;;
    --lambda_loss=*) LAMBDA_WEIGHT="${1#*=}" ;;
    --lambda_weight=*) LAMBDA_WEIGHT="${1#*=}" ;; # Add this for dual compatibility
    --wandb_project_name=*) WANDB_PROJECT_NAME="${1#*=}" ;;
    --wandb_run_name=*) WANDB_RUN_NAME="${1#*=}" ;;
    --log_freq=*) LOG_FREQ="${1#*=}" ;;
    --beta=*) BETA="${1#*=}" ;;
    --gamma=*) GAMMA="${1#*=}" ;;
    --pref_beta=*) PREF_BETA="${1#*=}" ;;
    --preference_weight=*) PREF_BETA="${1#*=}" ;; # Add this for dual compatibility
    --corpus_updates=*) CORPUS_UPDATES="${1#*=}" ;;
    --epochs=*) EPOCHS="${1#*=}" ;;
    --batch_size=*) BATCH_SIZE="${1#*=}" ;;
    --lr=*) LR="${1#*=}" ;;
    --scheduler=*) SCHEDULER="${1#*=}" ;;
    --seed=*) SEED="${1#*=}" ;;
    --eval_steps=*) EVAL_STEPS="${1#*=}" ;;
    --use_4bit=*) USE_4BIT="${1#*=}" ;;
    --warmup_ratio=*) WARMUP_RATIO="${1#*=}" ;;
    --save_strategy=*) SAVE_STRATEGY="${1#*=}" ;;
    --k_eval_values_accuracy=*) K_EVAL_VALUES_ACCURACY="${1#*=}" ;;
    --k_eval_values_ndcg=*) K_EVAL_VALUES_NDCG="${1#*=}" ;;
    --max_train_samples=*) MAX_TRAIN_SAMPLES="${1#*=}" ;;
    --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
    --save_checkpoints=*) SAVE_CHECKPOINTS="${1#*=}" ;;
    --weight_decay=*) WEIGHT_DECAY="${1#*=}" ;;
    --embedding_update_steps=*) EMBEDDING_UPDATE_STEPS="${1#*=}" ;;
    --n_reembedding_steps=*) N_REEMBEDDING_STEPS="${1#*=}" ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Ensure only one of the embedding update parameters is passed (prioritize n_reembedding_steps)
if [ -n "$N_REEMBEDDING_STEPS" ]; then
  EMBEDDING_PARAM="N_REEMBEDDING_STEPS=$N_REEMBEDDING_STEPS"
elif [ -n "$EMBEDDING_UPDATE_STEPS" ]; then
  EMBEDDING_PARAM="EMBEDDING_UPDATE_STEPS=$EMBEDDING_UPDATE_STEPS"
else
  EMBEDDING_PARAM="" # If neither is specified
fi

docker run \
    -v "$OUTPUT_DIR":/workspace/output \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    --memory="30g" \
    --rm \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    ports_image \
    ports \
    DATASET=$DATASET \
    INFERENCE_MODEL=$INFERENCE_MODEL \
    RETRIEVAL_MODEL=$RETRIEVAL_MODEL \
    RETRIEVAL_MAX_SEQ_LEN=$RETRIEVAL_MAX_SEQ_LEN \
    INFERENCE_MAX_SEQ_LEN=$INFERENCE_MAX_SEQ_LEN \
    PADDING_SIDE=$PADDING_SIDE \
    N_NEGS=$N_NEGS \
    EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
    PREPROCESS_BATCH_SIZE=$PREPROCESS_BATCH_SIZE \
    LAMBDA_WEIGHT=$LAMBDA_WEIGHT \
    WANDB_PROJECT_NAME=$WANDB_PROJECT_NAME \
    WANDB_RUN_NAME=$WANDB_RUN_NAME \
    LOG_FREQ=$LOG_FREQ \
    BETA=$BETA \
    GAMMA=$GAMMA \
    PREF_BETA=$PREF_BETA \
    CORPUS_UPDATES=$CORPUS_UPDATES \
    EPOCHS=$EPOCHS \
    BATCH_SIZE=$BATCH_SIZE \
    LR=$LR \
    SCHEDULER=$SCHEDULER \
    SEED=$SEED \
    EVAL_STEPS=$EVAL_STEPS \
    USE_4BIT=$USE_4BIT \
    WARMUP_RATIO=$WARMUP_RATIO \
    SAVE_STRATEGY=$SAVE_STRATEGY \
    SAVE_DIR=/workspace/output \
    K_EVAL_VALUES_ACCURACY="$K_EVAL_VALUES_ACCURACY" \
    K_EVAL_VALUES_NDCG="$K_EVAL_VALUES_NDCG" \
    MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES \
    SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS \
    WEIGHT_DECAY=$WEIGHT_DECAY \
    $EMBEDDING_PARAM