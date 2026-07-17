# Interactive salloc Guide: Deploying `Qwen/Qwen3-Embedding-8B`

This guide covers two use cases for the embedding model:

| Use case | When to run |
|---|---|
| **A — Offline tool embedding** | Once before the benchmark; pre-embeds all tools into a cache file |
| **B — Online embedding server** | During inference, alongside the executing LLM (port 8002) |

See the port overview for all three servers:

| Server | Model | Port | Guide |
|---|---|---|---|
| Executing LLM | `openai/gpt-oss-120b` | 8000 | [SALLOC_GUIDE.md](SALLOC_GUIDE.md) |
| Selector LLM | `Qwen/Qwen3-30B-A3B` | 8001 | [SALLOC_GUIDE_QWEN.md](SALLOC_GUIDE_QWEN.md) |
| Embedding model | `Qwen/Qwen3-Embedding-8B` | 8002 | this file |

---

## Use Case A — Offline Tool Embedding (one-shot)

Run this **once** before the benchmark to precompute tool embeddings.
No long-running server is needed; the results are saved to a local cache file.

### A1. Request an interactive session

```bash
salloc \
  --partition=gpu-single \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:A40:1 \
  --mem=32G \
  --time=02:00:00
```

> A single A40 (48 GB) is more than enough — Qwen3-Embedding-8B occupies ~16 GB at
> float16. 2 hours is a safe upper bound for embedding a full tool corpus.

### A2. Activate the environment

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark

# First time only: create the virtual environment (takes 5–15 min)
# Skip this if .venv already exists
if [ ! -d ".venv" ]; then
    bash deploy/uv_setup.sh --vllm
fi

source .venv/bin/activate
```

### A3. Start a temporary embedding server

```bash
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

bash deploy/slurm_vllm_embedding_deploy.sh &
EMBED_SERVER_PID=$!


### A4. Run the offline embedding script

```bash
export QWEN3_EMBEDDING_BASE_URL=http://localhost:8002/v1
export QWEN3_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export QWEN3_EMBEDDING_API_KEY=EMPTY

# Ensure the project venv is active (Python 3.10+), not the system Python 3.6
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark
source .venv/bin/activate
cd wild-tool-bench

python wtb/model_handler/api_inference/setup_openai_embeddings.py \
  --provider qwen3 \
  --tools-file ../multi-agent-framework/tools/tools_en_final.jsonl
```

The embeddings are saved to `tool_embeddings_cache.json` in the same directory.
Once the script completes, stop the temporary server and release the allocation:

```bash
kill $EMBED_SERVER_PID
exit   # releases the salloc allocation
```

---

## Use Case B — Online Embedding Server (during inference)

Run this in a **separate terminal session** alongside the executing LLM and (optionally)
the selector LLM. The embedding server answers real-time query embedding requests from
the LangGraph app.

### B1. Request an interactive session

```bash
salloc \
  --partition=gpu-single \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:A40:1 \
  --mem=32G \
  --time=24:00:00
```

> **Alternative (same node as gpt-oss-120b):** If you requested the H200 node with
> `--gres=gpu:H200:4` (leaving 4 GPUs free), you can add `--gres=gpu:H200:1` for the
> embedding model on the *same* node and avoid an extra allocation — adjust the
> `--gres` count in the gpt-oss salloc command accordingly.

Note the **hostname** — you will need it on the login node.

### B2. Activate the environment and start the server

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark

# First time only: create the virtual environment (takes 5–15 min)
if [ ! -d ".venv" ]; then
    bash deploy/uv_setup.sh --vllm
fi

source .venv/bin/activate

export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

MODEL_NAME=Qwen/Qwen3-Embedding-8B \
VLLM_EMBEDDING_PORT=8002 \
GPU_MEMORY_UTILIZATION=0.80 \
  bash deploy/slurm_vllm_embedding_deploy.sh
```

The server is ready when you see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8002
```

### B3. Configure the LangGraph app (on the login node)

```bash
# Executing LLM (see SALLOC_GUIDE.md)
export EXECUTING_LLM_BASE_URL=http://<node-gptoss>:8000/v1
export EXECUTING_LLM_MODEL=openai/gpt-oss-120b
export EXECUTING_LLM_API_KEY=EMPTY

# Embedding model
export QWEN3_EMBEDDING_BASE_URL=http://<node-embed>:8002/v1
export QWEN3_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
export QWEN3_EMBEDDING_API_KEY=EMPTY
export LANGGRAPH_TOOL_SELECTION_MODE=qwen3_embedding
```

Quick connectivity check:

```bash
curl http://<node-embed>:8002/v1/models
```

Expected response includes `"id": "Qwen/Qwen3-Embedding-8B"`.

### B4. Run the benchmark

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark/wild-tool-bench
source ../.venv/bin/activate

python -u -m wtb.openfunctions_evaluation --model=langgraph
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Increase allocation to 2× A40; or lower `GPU_MEMORY_UTILIZATION` |
| `Connection refused` on login node | Check node hostname; ensure port 8002 is not firewalled |
| NCCL segfault at startup | Ensure `NCCL_NET_PLUGIN=none` is exported before starting vLLM |
| Embedding server responds but returns wrong model name | Pass `--served-model-name Qwen/Qwen3-Embedding-8B` to override |
| Cache file stale after tool corpus changes | Re-run Use Case A to regenerate `tool_embeddings_cache.json` |
| Port 8002 already in use | Change `VLLM_EMBEDDING_PORT` to a free port and update all env vars |
