Request 2 GPUs
salloc \
    --partition=gpu-vram-48gb \
    --nodes=1 \
    --gres=gpu:2 \
    --cpus-per-task=16 \
    --mem=128G \
    --time=12:00:00

Start GPT-OSS on GPU 0

export CUDA_VISIBLE_DEVICES=0

vllm serve openai/gpt-oss-20b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9

Open another terminal on the same allocation

srun --jobid=<JOBID> --pty bash

Start embedding server on GPU 1

export CUDA_VISIBLE_DEVICES=1

vllm serve Qwen/Qwen3-Embedding-8B \
    --host 0.0.0.0 \
    --port 8002

Start reranker on GPU 1

export CUDA_VISIBLE_DEVICES=1

vllm serve Qwen/Qwen3-Reranker-8B \
    --host 0.0.0.0 \
    --port 8003