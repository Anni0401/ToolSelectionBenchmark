#!/bin/bash

LLAMA_TYPE="llama3"
LLAMA_GROQ_TYPE="llama3groq"
CODESTRAL_TYPE="codestral"
LLAMA="meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_GROQ="/proj/mounted/models/models--Groq--Llama-3-Groq-8B-Tool-Use/snapshots/1d0841d97fef29b98cd9737f5beccf9ea0c8f512"
CODESTRAL="/proj/mounted/models/models--mistralai--Codestral-22B-v0.1/snapshots/8f5fe23af91885222a1563283c87416745a5e212"

BGE="/proj/mounted/models/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a/"
ROBERTA="FacebookAI/roberta-base"

OCTOVER="ToolRetriever/OctopusOverlapping"
OCTNONOVER="ToolRetriever/OctopusNonOverlapping"
TOOLEOVER="ToolRetriever/ToolEOverlapping"
TOOLENONOVER="ToolRetriever/ToolENonOverlapping"
APIBANK="ToolRetriever/APIBank"
APIBENCH="ToolRetriever/APIBench"
BFCL="ToolRetriever/BFCL"
TOOLBENCH="ToolRetriever/ToolBench"


quintuplets=(
    # "[PAPER]_ToolBench_bge_llama3 $TOOLBENCH $BGE $LLAMA $LLAMA_TYPE"
    # "[PAPER]_ToolBench_roberta_llama3 $TOOLBENCH $ROBERTA $LLAMA $LLAMA_TYPE"
    # "[PAPER]_ToolBench_bge_llama3groq $TOOLBENCH $BGE $LLAMA_GROQ $LLAMA_GROQ_TYPE"
    # "[PAPER]_ToolBench_roberta_llama3groq $TOOLBENCH $ROBERTA $LLAMA_GROQ $LLAMA_GROQ_TYPE"
    # "[PAPER]_ToolBench_roberta_codestral $TOOLBENCH $ROBERTA $CODESTRAL $CODESTRAL_TYPE"
    # "[PAPER]_APIBench_bge_llama3 $APIBENCH $BGE $LLAMA $LLAMA_TYPE"
    # "[PAPER]_APIBench_roberta_llama3 $APIBENCH $ROBERTA $LLAMA $LLAMA_TYPE"
    # "[PAPER]_APIBench_bge_llama3groq $APIBENCH $BGE $LLAMA_GROQ $LLAMA_GROQ_TYPE"
    # "[PAPER]_APIBench_roberta_llama3groq $APIBENCH $ROBERTA $LLAMA_GROQ $LLAMA_GROQ_TYPE"
    # "[PAPER]_APIBench_roberta_codestral $APIBENCH $ROBERTA $CODESTRAL $CODESTRAL_TYPE"
    #"[PAPER]_OctopusOverlapping_roberta_llama3 $OCTOVER $ROBERTA $LLAMA $LLAMA_TYPE"
    "[PAPER]_OctopusNonOverlapping_roberta_llama3 $OCTNONOVER $ROBERTA $LLAMA $LLAMA_TYPE"
)


QUERY_COLUMN="query_for_retrieval"
RESPONSE_COLUMN="answer"
BATCH_SIZE=4
NUM_TRAIN_EPOCHS=2
NUM_RETRIEVED_DOCS_PER_QUERY=3
GAMMA_VALUE=0.5
BETA_VALUE=0.5
LEARNING_RATE=1e-5
LR_SCHEDULER="cosine"

for quintuplet in "${quintuplets[@]}"; do
    # Split the quintuplet into individual variables
    IFS=' ' read -r WANDB_PROJ_NAME DATASET_PATH RETR_MODEL_NAME_OR_PATH INFER_MODEL_NAME_OR_PATH INFER_MODEL_TYPE <<< "$quintuplet"
    python3 /proj/main/replug/main.py  --dataset_path $DATASET_PATH \
                            --retr_model_name_or_path $RETR_MODEL_NAME_OR_PATH \
                            --infer_model_name_or_path $INFER_MODEL_NAME_OR_PATH \
                            --infer_model_type $INFER_MODEL_TYPE \
                            --query_column $QUERY_COLUMN \
                            --response_column $RESPONSE_COLUMN \
                            --batch_size $BATCH_SIZE \
                            --num_train_epochs $NUM_TRAIN_EPOCHS \
                            --num_retrieved_docs_per_query $NUM_RETRIEVED_DOCS_PER_QUERY \
                            --gamma_value $GAMMA_VALUE \
                            --beta_value $BETA_VALUE \
                            --learning_rate $LEARNING_RATE \
                            --lr_scheduler $LR_SCHEDULER \
                            --quantize \
                            --quantization_4bit \
                            --verbose \
                            # --log_to_wandb \
                            # --wandb_proj_name $WANDB_PROJ_NAME 
done