# Interactive salloc Guide: Deploying `openai/gpt-oss-120b`

This guide walks through manually starting the `openai/gpt-oss-120b` vLLM server in an
interactive SLURM session (`salloc`) on an H100 node.

---

## 1. Request an interactive session

From the **login node**, run:

```bash
salloc \
  --nodes=1 \
  --gpus-per-node=4 \
  --mem=256G \
  --time=24:00:00 \
  --partition=gpu-vram-94gb   # replace with your H100 partition name
```

> **Note:** `openai/gpt-oss-120b` is a 120B Mixture-of-Experts model. With BF16 weights
> it requires ~4× H100 80GB GPUs (tensor-parallel-size 4).

Once the allocation is granted, SLURM will drop you into an interactive shell on the
compute node. Note the **hostname** (e.g. `dws-15`) — you will need it later.

---

## 2. Activate the gpt-oss vLLM environment

```bash
cd /home/aherrman/ToolSelectionBenchmark
source .venv-gptoss/bin/activate
```

Verify the correct vLLM build is active:

```bash
python -c "import vllm; print(vllm.__version__)"
# Expected: 0.10.1+gptoss (or similar)
```

If the environment does not exist yet, install it first:

```bash
cd /home/aherrman/ToolSelectionBenchmark
uv venv .venv-gptoss --python 3.12
source .venv-gptoss/bin/activate
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

---

## 3. Start the vLLM server (H100 / Hopper)

For H100/H200 (Hopper architecture) with 4-way tensor parallelism:

```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

> **H100 note:** No `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` or `--kv-cache-dtype fp8`
> flags needed — those are Blackwell (B200) specific. BF16 runs natively on Hopper.

The server is ready when you see:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 4. (Optional) Enable reasoning effort

Pass a system prompt to control reasoning depth:

| Level  | System prompt string |
|--------|----------------------|
| low    | `"Reasoning: low"`   |
| medium | `"Reasoning: medium"`|
| high   | `"Reasoning: high"`  |

This is done at inference time via the API, not at server startup.

---

## 5. Configure the LangGraph app (on the login node)

Open a **new terminal on the login node** and export the environment variables before
running the benchmark:

```bash
# Replace <node> with the hostname from step 1 (e.g. dws-15)
export EXECUTING_LLM_BASE_URL=http://<node>:8000/v1
export EXECUTING_LLM_MODEL=openai/gpt-oss-120b
export EXECUTING_LLM_API_KEY=EMPTY
```

Quick connectivity check:

```bash
curl http://<node>:8000/v1/models
```

Expected response includes `"id": "openai/gpt-oss-120b"`.

---

## 6. Run the benchmark

```bash
cd /home/aherrman/ToolSelectionBenchmark/wild-tool-bench
source ../.venv/bin/activate   # main project venv (not gptoss)

python -u -m wtb.openfunctions_evaluation --model=langgraph
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `--gpu-memory-utilization` to `0.85` |
| `Connection refused` on login node | Check node hostname; ensure port 8000 is not firewalled |
| `tl.language not defined` | Do not install extra `pytorch-triton` alongside vLLM |
| Harmony vocab download failure | Pre-download tiktoken files and set `TIKTOKEN_ENCODINGS_BASE` |
| Model download slow/fails | Set `HF_HUB_CACHE` to a fast shared filesystem path |
