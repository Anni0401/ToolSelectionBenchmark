#!/bin/bash


PHYS_DIR="${HOME}/ports" 
LLM_CACHE_DIR="/llms"
DOCKER_INTERNAL_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":"$DOCKER_INTERNAL_CACHE_DIR" \
    -e HF_HOME="$DOCKER_INTERNAL_CACHE_DIR" \
    --memory="30g" \
    --rm \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    ports_image \
    "/workspace/main/scripts/train_ports.sh"