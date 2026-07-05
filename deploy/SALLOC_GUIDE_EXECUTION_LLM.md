# Interactive salloc Guide: Deploying `openai/gpt-oss-120b`

This guide walks through manually starting the `openai/gpt-oss-120b` vLLM server in an
interactive SLURM session (`salloc`) on an **H200 node** of this cluster.

### Available GPU node types on this cluster

| `--gres` type | GPUs per node | GPU model | VRAM per GPU | Available memory |
|---|---|---|---|---|
| `gpu:A40:<n>` | 4 | Nvidia A40 | 48 GB | 236 GB |
| `gpu:A100:<n>` (40 GB nodes) | 4 | Nvidia A100 | 40 GB | 236 GB |
| `gpu:A100:<n>` (80 GB nodes) | 8 | Nvidia A100 | 80 GB | 2000 GB |
| `gpu:H200:<n>` | 8 | Nvidia H200 | 141 GB | 2200 GB |

All GPU nodes share the `gpu-single` partition; the GPU type and count are selected via `--gres`.
> Run `sinfo -o "%P %G %m"` on the login node to confirm available gres types.

---

## 1. Request an interactive session

From the **login node**, run:

```bash
salloc \
  --partition=gpu-single \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:H200:4 \
  --mem=512G \
  --time=24:00:00
```

> **Note:** `openai/gpt-oss-120b` is a 120B Mixture-of-Experts model. In BF16 the weights
> occupy ~240 GB. 4× H200 (141 GB each = 564 GB total) provides ample VRAM with room for
> KV cache. The H200 nodes have 8 GPUs per node; requesting 4 is sufficient and avoids
> occupying the full node.

Once the allocation is granted, SLURM will drop you into an interactive shell on the
compute node. Note the **hostname** (e.g. `dws-15`) — you will need it later.

---

## 2. Activate the gpt-oss vLLM environment

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark

# First time only: create the dedicated gpt-oss venv
if [ ! -d ".venv-gptoss" ]; then
    export PATH="${HOME}/.local/bin:${PATH}"
    # install uv if missing
    command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv .venv-gptoss --python 3.12 --seed   # --seed bootstraps pip into the venv
    source .venv-gptoss/bin/activate
    # Step 1: install torch nightly (cu128) — latest available build.
    # The gptoss wheel pins an old nightly that is no longer in the index,
    # so we install torch first and then bypass the pin with --no-deps.
    #
    # IMPORTANT: torch+vllm wheels are ~4-5 GB when extracted. /tmp on login/cpu
    # nodes is too small. Redirect pip's temp and cache dirs to gpfs first:
    export TMPDIR="${HOME}/tmp_pip"
    export PIP_CACHE_DIR="${HOME}/tmp_pip/cache"
    mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"
    #
    python -m pip install --pre torch \
      --index-url https://download.pytorch.org/whl/nightly/cu128
    # Step 2: install vllm gpt-oss wheel without dependency resolution
    # (the wheel's torch pin refers to an archived nightly; newer torch is compatible)
    python -m pip install --pre "vllm==0.10.1+gptoss" --no-deps \
      --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
      --extra-index-url https://download.pytorch.org/whl/nightly/cu128
    # Step 3: install remaining vllm runtime dependencies (excluding torch)
    python -m pip install \
      "aiohttp" "blake3" "fastapi" "httpx" "numpy" "openai" \
      "pillow" "prometheus-client" "pydantic>=2" "ray>=2.9" \
      "regex" "requests" "sentencepiece" "tiktoken" \
      "transformers>=4.45" "triton" "xformers" 2>/dev/null || true
fi

source .venv-gptoss/bin/activate
```

Verify the correct vLLM build is active:

```bash
python -c "import vllm; print(vllm.__version__)"
# Expected: something like 0.10.2.dev2+gf5635d62e.d20250807
# (the internal version string is a dev build with git hash — this is correct for the gptoss wheel)
```

---

## 3. Start the vLLM server (H200 / Hopper)

For H200 (Hopper architecture) with 4-way tensor parallelism:

```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

> **H200 note:** No `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` or `--kv-cache-dtype fp8`
> flags needed — those are Blackwell (B200) specific. BF16 runs natively on Hopper (H200).
> The H200's 141 GB HBM3e gives significant KV-cache headroom over H100 80 GB.
> `--enforce-eager` disables torch.compile/CUDA-graph capture, avoiding `nvcc` permission
> errors on this cluster. Remove it only if you need maximum throughput and `nvcc` works.

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
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark/wild-tool-bench
source ../.venv/bin/activate   # main project venv (not gptoss)

python -u -m wtb.openfunctions_evaluation --model=langgraph
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Lower `--gpu-memory-utilization` to `0.85` |
| `Killed` during pip install | `/tmp` is full (only 4 GB); set `export TMPDIR=~/tmp_pip PIP_CACHE_DIR=~/tmp_pip/cache` before pip |
| `Connection refused` on login node | Check node hostname; ensure port 8000 is not firewalled |
| `tl.language not defined` | Do not install extra `pytorch-triton` alongside vLLM |
| Harmony vocab download failure | Pre-download tiktoken files and set `TIKTOKEN_ENCODINGS_BASE` |
| Model download slow/fails | Set `HF_HUB_CACHE` to a fast shared filesystem path |
