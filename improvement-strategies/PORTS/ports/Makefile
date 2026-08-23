# Ports Project Makefile
# Provides a clean interface for executing training and evaluation scripts

# Default shell
SHELL := /bin/bash

# Common parameters (used by multiple scripts)
DATASET ?= toolbench
RETRIEVAL_MODEL ?= BAAI/bge-base-en-v1.5
INFERENCE_MODEL ?= llama3-8B
EPOCHS ?= 5
BATCH_SIZE ?= 2
LR ?= 1e-5
SEED ?= 42
EVAL_STEPS ?= 0.2
USE_4BIT ?= true
MAX_TRAIN_SAMPLES ?= 1000
GAMMA ?= 0.5
BETA ?= 0.5
WARMUP_RATIO ?= 0.1

# Evaluation parameters (common)
K_EVAL_VALUES_ACCURACY ?= 1 3 5
K_EVAL_VALUES_NDCG ?= 1 3 5

# PORTS specific parameters
RETRIEVAL_MAX_SEQ_LEN ?= 512
INFERENCE_MAX_SEQ_LEN ?= 1024
PADDING_SIDE ?= left
N_NEGS ?= 3
EVAL_BATCH_SIZE ?= 4
PREPROCESS_BATCH_SIZE ?= 16
LAMBDA_WEIGHT ?= 0.3
MAX_EVAL_SAMPLES ?= 200
WANDB_PROJECT_NAME ?= PORTS_AAAI-EMNLP
LOG_FREQ ?= 20
PREF_BETA ?= 1
CORPUS_UPDATES ?= 100
N_REEMBEDDING_STEPS ?=
EMBEDDING_UPDATE_STEPS ?=
SAVE_STRATEGY ?= epoch
SAVE_STEPS ?= 
SAVE_DIR ?= ./checkpoints
MAX_CHECKPOINTS ?=
SAVE_CHECKPOINTS ?= false

# RePlug specific parameters
QUERY_COLUMN ?= query_for_retrieval
RESPONSE_COLUMN ?= answer
NUM_RETRIEVED_DOCS ?= 5
REPLUG_SAVE_PATH ?= $(OUTPUT_DIR)/replug/replug_retriever_$(DATASET)_$(shell basename $(RETRIEVAL_MODEL))_$(shell date +%Y%m%d_%H%M%S)
CORPUS_UPDATES ?= 5
PREPROCESS_BATCH_SIZE ?= 16

# MNRL specific parameters
SCHEDULER ?= cosine
POOLING ?= mean
NEGATIVES_PER_SAMPLE ?= 1
MNRL_OUTPUT_DIR ?= $(OUTPUT_DIR)/mnrl/mnrl_retriever_$(DATASET)_$(shell basename $(RETRIEVAL_MODEL))_$(shell date +%Y%m%d_%H%M%S)

# Directories
MAIN_DIR := /workspace/main
SCRIPTS_DIR := $(MAIN_DIR)/scripts
OUTPUT_DIR := $(MAIN_DIR)/output

# Helper functions
define print_header
	@echo "====================================="
	@echo "$(1)"
	@echo "====================================="
	@echo "Dataset: $(DATASET)"
	@echo "Retrieval model: $(RETRIEVAL_MODEL)"
	@echo "Inference model: $(INFERENCE_MODEL)"
	@echo "Epochs: $(EPOCHS)"
	@echo "Batch size: $(BATCH_SIZE)"
	@echo "Learning rate: $(LR)"
	@echo "====================================="
endef

# Default target (help)
.PHONY: help
help:
	@echo "Ports Training Framework"
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [options]"
	@echo ""
	@echo "Targets:"
	@echo "  help            - Show this help message"
	@echo "  ports           - Run PORTS training"
	@echo "  replug          - Run RePlug training"
	@echo "  mnrl            - Run MNRL training"
	@echo "  docker          - Run interactive Docker container"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-cmd      - Run specific command in Docker container"
	@echo "  clean           - Clean output directories"
	@echo ""
	@echo "Common Options:"
	@echo "  DATASET=<name>          - Dataset name (default: toolbench)"
	@echo "  RETRIEVAL_MODEL=<name>  - Retrieval model name (default: BAAI/bge-base-en-v1.5)"
	@echo "  INFERENCE_MODEL=<name>  - Inference model name (default: llama3-8B)"
	@echo "  EPOCHS=<number>         - Number of training epochs (default: 5)"
	@echo "  BATCH_SIZE=<number>     - Batch size (default: 2)"
	@echo "  LR=<value>              - Learning rate (default: 1e-5)"
	@echo "  SEED=<number>           - Random seed (default: 42)"
	@echo "  EVAL_STEPS=<number>     - Evaluation steps fraction (default: 0.2)"
	@echo "  USE_4BIT=<true|false>   - Use 4-bit quantization (default: true)"
	@echo "  MAX_TRAIN_SAMPLES=<num> - Maximum training samples (default: 1000)"
	@echo "  GAMMA=<value>           - Gamma temperature (default: 0.5)"
	@echo "  BETA=<value>            - Beta temperature (default: 0.5)"
	@echo "  WARMUP_RATIO=<value>    - Fraction of training steps for warmup (default: 0.1)"
	@echo "  K_EVAL_VALUES_ACCURACY=<values> - Values for accuracy@k (default: 1 3 5)"
	@echo "  K_EVAL_VALUES_NDCG=<values> - Values for ndcg@k (default: 1 3 5)"
	@echo "  N_REEMBEDDING_STEPS=<value> - Number of splits for corpus re-embedding (default: none)"
	@echo ""
	@echo "For additional parameters, see the README.md file"
	@echo ""
	@echo "Examples:"
	@echo "  make ports DATASET=bfcl INFERENCE_MODEL=gemma2-2B"
	@echo "  make replug EPOCHS=10 BATCH_SIZE=4"
	@echo "  make mnrl RETRIEVAL_MODEL=FacebookAI/roberta-base"
	@echo "  make docker"

# PORTS training
.PHONY: ports
ports:
	$(call print_header,"Running PORTS Training")
	@mkdir -p $(SAVE_DIR)
	@DATASET_NAME=$(DATASET) \
	RETRIEVAL_MODEL_NAME=$(RETRIEVAL_MODEL) \
	INFERENCE_MODEL_PSEUDONAME=$(INFERENCE_MODEL) \
	RETRIEVAL_MAX_SEQ_LEN=$(RETRIEVAL_MAX_SEQ_LEN) \
	INFERENCE_MAX_SEQ_LEN=$(INFERENCE_MAX_SEQ_LEN) \
	N_EPOCHS=$(EPOCHS) \
	LR=$(LR) \
	LR_SCHEDULER=$(SCHEDULER) \
	TRAIN_BATCH_SIZE=$(BATCH_SIZE) \
	EVAL_BATCH_SIZE=$(EVAL_BATCH_SIZE) \
	PREPROCESS_BATCH_SIZE=$(PREPROCESS_BATCH_SIZE) \
	PADDING_SIDE=$(PADDING_SIDE) \
	LAMBDA_WEIGHT=$(LAMBDA_WEIGHT) \
	N_NEGS=$(N_NEGS) \
	GAMMA=$(GAMMA) \
	BETA=$(BETA) \
	PREF_BETA=$(PREF_BETA) \
	CORPUS_UPDATES=$(CORPUS_UPDATES) \
	N_REEMBEDDING_STEPS=$(N_REEMBEDDING_STEPS) \
	EMBEDDING_UPDATE_STEPS=$(EMBEDDING_UPDATE_STEPS) \
	SEED=$(SEED) \
	EVAL_STEPS=$(EVAL_STEPS) \
	LOAD_IN_4BIT=$(USE_4BIT) \
	MAX_TRAIN_SAMPLES=$(MAX_TRAIN_SAMPLES) \
	MAX_EVAL_SAMPLES=$(MAX_EVAL_SAMPLES) \
	WANDB_PROJECT_NAME=$(WANDB_PROJECT_NAME) \
	LOG_FREQ=$(LOG_FREQ) \
	SAVE_STRATEGY=$(SAVE_STRATEGY) \
	SAVE_STEPS=$(SAVE_STEPS) \
	SAVE_DIR=$(SAVE_DIR) \
	MAX_CHECKPOINTS=$(MAX_CHECKPOINTS) \
	SAVE_CHECKPOINTS=$(SAVE_CHECKPOINTS) \
	WARMUP_RATIO=$(WARMUP_RATIO) \
	K_EVAL_VALUES_ACCURACY="$(K_EVAL_VALUES_ACCURACY)" \
	K_EVAL_VALUES_NDCG="$(K_EVAL_VALUES_NDCG)" \
	$(SCRIPTS_DIR)/train_ports.sh

# RePlug training
.PHONY: replug
replug:
	$(call print_header,"Running RePlug Training")
	@mkdir -p $(REPLUG_SAVE_PATH)
	@EPOCHS=$(EPOCHS) \
	DATASET_NAME=$(DATASET) \
	DATASET_PATH="ToolRetriever/$(DATASET)" \
	RETRIEVAL_MODEL_NAME=$(RETRIEVAL_MODEL) \
	INFERENCE_MODEL_PSEUDONAME=$(INFERENCE_MODEL) \
	RETRIEVAL_MAX_SEQ_LEN=$(RETRIEVAL_MAX_SEQ_LEN) \
	INFERENCE_MAX_SEQ_LEN=$(INFERENCE_MAX_SEQ_LEN) \
	TRAIN_BATCH_SIZE=$(BATCH_SIZE) \
	LR=$(LR) \
	LR_SCHEDULER=$(SCHEDULER) \
	NUM_RETRIEVED_DOCS=$(NUM_RETRIEVED_DOCS) \
	GAMMA=$(GAMMA) \
	BETA=$(BETA) \
	SEED=$(SEED) \
	EVAL_STEPS_FRACTION=$(EVAL_STEPS) \
	LOAD_IN_4BIT=$(USE_4BIT) \
	MAX_TRAIN_SAMPLES=$(MAX_TRAIN_SAMPLES) \
	QUERY_COLUMN=$(QUERY_COLUMN) \
	RESPONSE_COLUMN=$(RESPONSE_COLUMN) \
	WARMUP_RATIO=$(WARMUP_RATIO) \
	SAVE_PATH=$(REPLUG_SAVE_PATH) \
	K_EVAL_VALUES_ACCURACY="$(K_EVAL_VALUES_ACCURACY)" \
	K_EVAL_VALUES_NDCG="$(K_EVAL_VALUES_NDCG)" \
	CORPUS_UPDATES=$(CORPUS_UPDATES) \
	PREPROCESS_BATCH_SIZE=$(PREPROCESS_BATCH_SIZE) \
	SAVE_CHECKPOINTS=$(SAVE_CHECKPOINTS) \
	$(SCRIPTS_DIR)/train_replug.sh

# MNRL training
.PHONY: mnrl
mnrl:
	$(call print_header,"Running MNRL Training")
	@mkdir -p $(MNRL_OUTPUT_DIR)
	@EPOCHS=$(EPOCHS) \
	DATASET_NAME=$(DATASET) \
	MODEL_NAME=$(RETRIEVAL_MODEL) \
	TRAIN_BATCH_SIZE=$(BATCH_SIZE) \
	EVAL_BATCH_SIZE=$(EVAL_BATCH_SIZE) \
	PREPROCESSING_BATCH_SIZE=$(PREPROCESS_BATCH_SIZE) \
	LR=$(LR) \
	MAX_SEQ_LENGTH=$(RETRIEVAL_MAX_SEQ_LEN) \
	EVAL_STEPS_FRACTION=$(EVAL_STEPS) \
	WARMUP_RATIO=$(WARMUP_RATIO) \
	SCHEDULER=$(SCHEDULER) \
	POOLING=$(POOLING) \
	NEGATIVES_PER_SAMPLE=$(NEGATIVES_PER_SAMPLE) \
	SEED=$(SEED) \
	MAX_TRAIN_SAMPLES=$(MAX_TRAIN_SAMPLES) \
	OUTPUT_DIR=$(MNRL_OUTPUT_DIR) \
	K_EVAL_VALUES_ACCURACY="$(K_EVAL_VALUES_ACCURACY)" \
	K_EVAL_VALUES_NDCG="$(K_EVAL_VALUES_NDCG)" \
	WANDB_PROJECT_NAME=$(WANDB_PROJECT_NAME) \
	WANDB_RUN_NAME=$(WANDB_RUN_NAME) \
	LOG_FREQ=$(LOG_FREQ) \
	SAVE_CHECKPOINTS=$(SAVE_CHECKPOINTS) \
	$(SCRIPTS_DIR)/train_mnrl.sh

# Docker commands
.PHONY: docker
docker:
	@echo "Starting interactive Docker container..."
	$(SCRIPTS_DIR)/docker_run.sh

.PHONY: docker-build
docker-build:
	@echo "Building Docker image: ports_image..."
	docker build -t ports_image -f build/Dockerfile .
	@echo "Docker image built successfully."

.PHONY: docker-cmd
docker-cmd:
	@if [ -z "$(CMD)" ]; then \
		echo "Error: Please specify a command with CMD=<your-command>"; \
		exit 1; \
	fi
	@echo "Running command in Docker: $(CMD)"
	$(SCRIPTS_DIR)/docker_run.sh "$(CMD)"

# Clean output directories
.PHONY: clean
clean:
	@echo "Cleaning output directories..."
	rm -rf $(OUTPUT_DIR)/ports/*
	rm -rf $(OUTPUT_DIR)/replug/*
	rm -rf $(OUTPUT_DIR)/mnrl/*
	@echo "Cleaned up output directories."