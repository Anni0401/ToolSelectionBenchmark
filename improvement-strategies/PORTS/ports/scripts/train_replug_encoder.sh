#!/bin/bash

# Default configuration
OUTPUT_DIR="${HOME}/ports/output" 
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"
DATASET="toolbench"
RETRIEVAL_MODEL="BAAI/bge-base-en-v1.5"
INFERENCE_MODEL="llama3-8B"
RETRIEVAL_MAX_SEQ_LEN=512
INFERENCE_MAX_SEQ_LEN=1024
BATCH_SIZE=2
LR=1e-5
SCHEDULER="cosine"
EPOCHS=1
NUM_RETRIEVED_DOCS=3
GAMMA=0.5
BETA=0.5
EVAL_STEPS=0.2
SEED=42
USE_4BIT=true
MAX_TRAIN_SAMPLES=10000
WARMUP_RATIO=0.1
WANDB_PROJECT_NAME="PORTS_AAAI-EMNLP"
WANDB_RUN_NAME="Replug_toolbench_g3"
K_EVAL_VALUES_ACCURACY="1 3 5"
K_EVAL_VALUES_NDCG="1 3 5"
CORPUS_UPDATES=5
PREPROCESS_BATCH_SIZE=64
SAVE_CHECKPOINTS=false
WEIGHT_DECAY=0.01

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset=*) DATASET="${1#*=}" ;;
    --retrieval_model=*) RETRIEVAL_MODEL="${1#*=}" ;;
    --inference_model=*) INFERENCE_MODEL="${1#*=}" ;;
    --retrieval_max_seq_len=*) RETRIEVAL_MAX_SEQ_LEN="${1#*=}" ;;
    --inference_max_seq_len=*) INFERENCE_MAX_SEQ_LEN="${1#*=}" ;;
    --batch_size=*) BATCH_SIZE="${1#*=}" ;;
    --lr=*) LR="${1#*=}" ;;
    --scheduler=*) SCHEDULER="${1#*=}" ;;
    --epochs=*) EPOCHS="${1#*=}" ;;
    --num_retrieved_docs=*) NUM_RETRIEVED_DOCS="${1#*=}" ;;
    --gamma=*) GAMMA="${1#*=}" ;;
    --beta=*) BETA="${1#*=}" ;;
    --eval_steps=*) EVAL_STEPS="${1#*=}" ;;
    --seed=*) SEED="${1#*=}" ;;
    --use_4bit=*) USE_4BIT="${1#*=}" ;;
    --max_train_samples=*) MAX_TRAIN_SAMPLES="${1#*=}" ;;
    --warmup_ratio=*) WARMUP_RATIO="${1#*=}" ;;
    --wandb_project_name=*) WANDB_PROJECT_NAME="${1#*=}" ;;
    --wandb_run_name=*) WANDB_RUN_NAME="${1#*=}" ;;
    --k_eval_values_accuracy=*) K_EVAL_VALUES_ACCURACY="${1#*=}" ;;
    --k_eval_values_ndcg=*) K_EVAL_VALUES_NDCG="${1#*=}" ;;
    --corpus_updates=*) CORPUS_UPDATES="${1#*=}" ;;
    --preprocess_batch_size=*) PREPROCESS_BATCH_SIZE="${1#*=}" ;;
    --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
    --save_checkpoints=*) SAVE_CHECKPOINTS="${1#*=}" ;;
    --weight_decay=*) WEIGHT_DECAY="${1#*=}" ;;
    --lambda_loss=*) ;; # Accept but ignore
    --lambda_weight=*) ;; # Accept but ignore
    --preference_weight=*) ;; # Accept but ignore
    --n_reembedding_steps=*) ;; # Accept but ignore
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

docker run \
    -v "$OUTPUT_DIR":/workspace/output \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    --memory="30g" \
    --rm \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    ports_image \
    replug \
    DATASET=$DATASET \
    RETRIEVAL_MODEL=$RETRIEVAL_MODEL \
    INFERENCE_MODEL=$INFERENCE_MODEL \
    RETRIEVAL_MAX_SEQ_LEN=$RETRIEVAL_MAX_SEQ_LEN \
    INFERENCE_MAX_SEQ_LEN=$INFERENCE_MAX_SEQ_LEN \
    BATCH_SIZE=$BATCH_SIZE \
    LR=$LR \
    SCHEDULER=$SCHEDULER \
    EPOCHS=$EPOCHS \
    NUM_RETRIEVED_DOCS=$NUM_RETRIEVED_DOCS \
    GAMMA=$GAMMA \
    BETA=$BETA \
    EVAL_STEPS=$EVAL_STEPS \
    SEED=$SEED \
    USE_4BIT=$USE_4BIT \
    MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES \
    WARMUP_RATIO=$WARMUP_RATIO \
    SAVE_PATH=/workspace/output \
    WANDB_PROJECT_NAME=$WANDB_PROJECT_NAME \
    WANDB_RUN_NAME=$WANDB_RUN_NAME \
    K_EVAL_VALUES_ACCURACY="$K_EVAL_VALUES_ACCURACY" \
    K_EVAL_VALUES_NDCG="$K_EVAL_VALUES_NDCG" \
    CORPUS_UPDATES=$CORPUS_UPDATES \
    PREPROCESS_BATCH_SIZE=$PREPROCESS_BATCH_SIZE \
    SAVE_CHECKPOINTS=$SAVE_CHECKPOINTS \
    WEIGHT_DECAY=$WEIGHT_DECAY
