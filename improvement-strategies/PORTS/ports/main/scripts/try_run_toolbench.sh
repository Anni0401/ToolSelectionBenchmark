#!/bin/bash

DEVICE=0
SEED=42
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
CORPUS_UPDATES=50
# Train Config
N_EPOCHS=100
INFERENCE_MODEL_PSEUDONAME="llama3-8B"
RETRIEVAL_MODEL_NAME="BAAI/bge-base-en-v1.5"
DATASET_NAME="toolbench"
# MAX_TRAIN_SAMPLES=300
WANDB_RUN_NAME="${DATASET_NAME}-${RETRIEVAL_MODEL_NAME}-${INFERENCE_MODEL_PSEUDONAME}-B${BETA}-G${GAMMA}-ORPOB${PREF_BETA}-LR${LR}"

CUDA_VISIBLE_DEVICES=$DEVICE python3 /proj/main/main_train_port.py --dataset $DATASET_NAME \
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
                                                    --eval_strategy "epoch" \
                                                    --load_in_4bit 
                                                    # --eval_steps 1000 \
                                                    # --max_train_samples $MAX_TRAIN_SAMPLES
