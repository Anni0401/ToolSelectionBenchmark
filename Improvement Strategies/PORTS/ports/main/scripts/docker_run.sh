#!/bin/bash

COMMAND=${1:-bash}

# Load keys
source /home/monaldini/ports/keys/keys.sh

# Determine the container name
CONTAINER_NAME="proj"
n=1

while [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; do
    n=$((n+1))
    CONTAINER_NAME="proj${n}"
done

CUDA_VISIBLE_DEVICES=3

# start ports container
docker run --name $CONTAINER_NAME \
           -v /home/monaldini/ports/main:/proj/main \
           -v /home/monaldini/ports/mounted:/proj/mounted \
           --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
           -e HF_KEY=$HF_KEY \
           -e WANDB_KEY=$WANDB_KEY \
           -it ports_image $COMMAND
        #    -p 7860:7860 \