# Interactive salloc Guide: Deploying `Qwen/Qwen3-30B-A3B` (Selector LLM)

This guide walks through manually starting the `Qwen/Qwen3-30B-A3B` vLLM server in an
interactive SLURM session (`salloc`) for use as the **selector LLM** in the hierarchical
tool selection strategy.

Run this in a **separate terminal session** alongside the `openai/gpt-oss-120b` server
(see [SALLOC_GUIDE.md](SALLOC_GUIDE.md)). Use a different port (e.g. `8001`) so both
servers can run on the same login-node connection.

---

## 1. Request an interactive session

From the **login node**, run:

```bash
salloc \
  --partition=gpu-single \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:A40:2 \
  --mem=96G \
  --time=24:00:00
```

> **Note:** `Qwen3-30B-A3B` is a 30B Mixture-of-Experts model (3B active parameters).
> In float16 the full weights occupy ~60 GB. 2× A40 (48 GB each = 96 GB total) provides
> ample VRAM with room for KV cache and tensor parallelism across both GPUs.

Once the allocation is granted, note the **hostname** (e.g. `gpu-node-07`) — you will
need it to configure the benchmark.

---

## 2. Activate the project environment

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark

# First time only: create the virtual environment (takes 5–15 min)
if [ ! -d ".venv" ]; then
    bash deploy/uv_setup.sh --vllm
fi

source .venv/bin/activate   # main project venv (contains vLLM)
```

Verify vLLM is available:

```bash
python -c "import vllm; print(vllm.__version__)"
```

---

## 3. Start the vLLM server

```bash
# Fix NCCL segfault caused by broken network plugin on this cluster
export NCCL_NET_PLUGIN=none
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8001 \
  --download-dir "${HOME}/.cache/huggingface/hub"
```

> **Port 8001** keeps this server separate from the gpt-oss-120b server on port 8000.
> Adjust `--tool-call-parser` if your vLLM version requires a different parser for Qwen3.

Alternatively, you can use the provided deploy script which sets the same flags via
environment variables:

```bash
export MODEL_NAME=Qwen/Qwen3-30B-A3B
export VLLM_PORT=8001
export TENSOR_PARALLEL_SIZE=2
export DTYPE=float16
export GPU_MEMORY_UTILIZATION=0.90
bash deploy/slurm_vllm_deploy.sh
```

The server is ready when you see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8001
```

---

## 4. Configure the hierarchical selection strategy (on the login node)

Open a **new terminal on the login node** and export both model endpoints:

```bash
# gpt-oss-120b — executing LLM (see SALLOC_GUIDE.md)
export EXECUTING_LLM_BASE_URL=http://<node-gptoss>:8000/v1
export EXECUTING_LLM_MODEL=openai/gpt-oss-120b
export EXECUTING_LLM_API_KEY=EMPTY

# Qwen3-30B-A3B — selector LLM
export SELECTOR_LLM_BASE_URL=http://<node-qwen>:8001/v1
export SELECTOR_LLM_MODEL=Qwen/Qwen3-30B-A3B
export SELECTOR_LLM_API_KEY=EMPTY
```

> Both `<node-gptoss>` and `<node-qwen>` can be the **same node** if you obtained a
> single allocation with enough GPUs (e.g. 6× A40 or 4× H200 + 2× A40 via two separate
> allocations). Replace each placeholder with the respective hostname from step 1.

Quick connectivity checks:

```bash
curl http://<node-qwen>:8001/v1/models
```

Expected response includes `"id": "Qwen/Qwen3-30B-A3B"`.

---

## 5. Run the benchmark with hierarchical selection

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark/wild-tool-bench
source ../.venv/bin/activate

python -u -m wtb.openfunctions_evaluation --model=langgraph
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `--gpu-memory-utilization` to `0.85` |
| `Connection refused` on login node | Check node hostname; ensure port 8001 is not firewalled |
| NCCL segfault / hang at startup | Ensure `NCCL_NET_PLUGIN=none` is exported before starting vLLM |
| Tool calls not parsed correctly | Try `--tool-call-parser qwen` instead of `hermes` |
| Model download slow/fails | Set `HF_HUB_CACHE` to a fast shared filesystem path |
| Port 8001 already in use | Change `--port` to another free port (e.g. `8002`) and update env vars |
