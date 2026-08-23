#!/bin/bash

DEVICE=2
SEED=42


# "llama3-8B" : "meta-llama/Meta-Llama-3-8B-Instruct",
# 'codestral-22B' : "mistralai/Codestral-22B-v0.1",
# 'gemma2-2B' : "google/gemma-2-2b-it",
# 'groqLlama3Tool-8B' : "Groq/Llama-3-Groq-8B-Tool-Use"


RETRIEVAL_MAX_SEQ_LEN=512

INFERENCE_MAX_SEQ_LEN=1024
PADDING_SIDE="left"

# Data
N_NEGS=3

LR="1e-5"
LR_SCHEDULER="cosine"

TRAIN_BATCH_SIZE=2
EVAL_BATCH_SIZE=4
PREPROCESS_BATCH_SIZE=16

# Loss config
LAMBDA_WEIGHT=0.3

# Eval config
K_ACC_UPPER=3


MAX_EVAL_SAMPLES=200

WANDB_PROJECT_NAME="PORTS_AAAI"

LOG_FREQ=20


BETA=0.5
GAMMA=0.5
PREF_BETA=1

CORPUS_UPDATES=100

# Train Config
N_EPOCHS=2


# for INFERENCE_MODEL_PSEUDONAME in ${MODELS[@]}; do
INFERENCE_MODEL_PSEUDONAME="llama3-8B"

RETR_MODELS=("FacebookAI/roberta-base" "BAAI/bge-base-en-v1.5")

MAX_TRAIN_SAMPLES=800

# LOW_RES_DATASETS=("apibench" "toolbench" "toole-overlap" "toole" "apibank")
LOW_RES_DATASETS=("bfcl")
#WHOLE_DS=("octopus" "octopus-overlap" "bfcl")

# for RETRIEVAL_MODEL_NAME in ${RETR_MODELS[@]}; do

#     for DATASET_NAME in ${LOW_RES_DATASETS[@]}; do

#DATASET_NAME="toole-overlap"
RETRIEVAL_MODEL_NAME="BAAI/bge-base-en-v1.5"

for DATASET_NAME in ${LOW_RES_DATASETS[@]}; do
    WANDB_RUN_NAME="${DATASET_NAME}-${RETRIEVAL_MODEL_NAME}-${INFERENCE_MODEL_PSEUDONAME}-B${BETA}-G${GAMMA}-ORPOB${PREF_BETA}-LR${LR}"

    CUDA_VISIBLE_DEVICES=$DEVICE python3 main_train_port.py --dataset $DATASET_NAME \
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
                                                        --n_reembedding_steps $CORPUS_UPDATES \
                                                        --padding_side $PADDING_SIDE \
                                                        --lambda_loss $LAMBDA_WEIGHT \
                                                        --n_neg_examples $N_NEGS \
                                                        --k_eval $K_ACC_UPPER \
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
                                                        --eval_steps 50 \
                                                        --load_in_4bit \
                                                        #--max_train_samples $MAX_TRAIN_SAMPLES \
                                                        #--max_eval_samples $MAX_EVAL_SAMPLES
done

    # for DATASET_NAME in ${WHOLE_DS[@]}; do
    #     WANDB_RUN_NAME="${DATASET_NAME}-${RETRIEVAL_MODEL_NAME}-${INFERENCE_MODEL_PSEUDONAME}-B${BETA}-G${GAMMA}-ORPOB${PREF_BETA}-LR${LR}"

    #     CUDA_VISIBLE_DEVICES=$DEVICE python3 main_train_port.py --dataset $DATASET_NAME \
    #                                                         --inference_model_name $INFERENCE_MODEL_PSEUDONAME \
    #                                                         --retrieval_model_name $RETRIEVAL_MODEL_NAME \
    #                                                         --retriever_max_seq_length $RETRIEVAL_MAX_SEQ_LEN \
    #                                                         --inference_max_seq_length $INFERENCE_MAX_SEQ_LEN \
    #                                                         --n_epochs $N_EPOCHS \
    #                                                         --lr $LR \
    #                                                         --lr_type $LR_SCHEDULER \
    #                                                         --train_batch_size $TRAIN_BATCH_SIZE \
    #                                                         --eval_batch_size $EVAL_BATCH_SIZE \
    #                                                         --preprocessing_batch_size $PREPROCESS_BATCH_SIZE \
    #                                                         --n_reembedding_steps $CORPUS_UPDATES \
    #                                                         --padding_side $PADDING_SIDE \
    #                                                         --lambda_loss $LAMBDA_WEIGHT \
    #                                                         --n_neg_examples $N_NEGS \
    #                                                         --k_eval $K_ACC_UPPER \
    #                                                         --gamma $GAMMA \
    #                                                         --beta $BETA \
    #                                                         --preference_weight $PREF_BETA \
    #                                                         --seed $SEED \
    #                                                         --wandb_project_name $WANDB_PROJECT_NAME \
    #                                                         --wandb_run_name $WANDB_RUN_NAME \
    #                                                         --log_freq $LOG_FREQ \
    #                                                         --do_train \
    #                                                         --do_eval \
    #                                                         --eval_strategy "epoch" \
    #                                                         --eval_steps 1000 \
    #                                                         --load_in_4bit \
    #                                                         #--max_train_samples $MAX_TRAIN_SAMPLES \
    #                                                         #--max_eval_samples $MAX_EVAL_SAMPLES
#     done
# done