# Interactive salloc Guide: Deploying `openai/gpt-oss`

This guide covers both gpt-oss model variants, using the same vLLM gpt-oss wheel:

| Model | Params (active) | Min GPU | Use case |
|---|---|---|---|
| `openai/gpt-oss-20b` | 21B MoE (3.6B active) | 1× A40 (48 GB) | Quick pipeline test |
| `openai/gpt-oss-120b` | 117B MoE (5.1B active) | 4× H200 | Benchmark runs |

> **Recommendation:** test the full end-to-end setup with `gpt-oss-20b` first (single A40,
> easier to allocate), then switch to `gpt-oss-120b` for production benchmark runs
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

From the **login node**, run one of the following depending on the model:

**For testing (`gpt-oss-20b`) — single A40:**
```bash
salloc \
  --partition=gpu-single \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gres=gpu:A40:1 \
  --mem=64G \
  --time=4:00:00
```

> **Note:** `gpt-oss-20b` is a 21B MoE model. In BF16 the weights occupy ~42 GB, fitting
> comfortably on a single A40 (48 GB) with room for KV cache.

**For benchmark runs (`gpt-oss-120b`) — 4× H200:**

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



---

## 2. Activate the gpt-oss vLLM environment

```bash
cd /home/ma/ma_ma/ma_aherrman/ToolSelectionBenchmark

# $WORK has 1 TiB quota (NVMe) — store the venv and pip artifacts there
# to avoid filling the 100 GiB $HOME quota.
VENV_DIR="${WORK}/venvs/venv-gptoss"

# pip / temporary files
export TMPDIR="${WORK}/tmp_pip"
export PIP_CACHE_DIR="${WORK}/tmp_pip/cache"

# Hugging Face / model cache
export HF_HOME="${WORK}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

mkdir -p \
    "${TMPDIR}" \
    "${PIP_CACHE_DIR}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}"


source "${VENV_DIR}/bin/activate"
```

Verify the correct vLLM build is active:

```bash
python -c "import vllm; print(vllm.__version__)"
# Expected: something like 0.10.2.dev2+gf5635d62e.d20250807
# (the internal version string is a dev build with git hash — this is correct for the gptoss wheel)
```

> **Subsequent sessions:** re-export `VENV_DIR` and source it before use:
> ```bash
> export VENV_DIR="${WORK}/venvs/venv-gptoss"
> source "${VENV_DIR}/bin/activate"
> ```

---

## 3. Start the vLLM server

**For testing (`gpt-oss-20b`) — single A40, no tensor parallelism:**
```bash
# Model weights are cached locally — no re-download needed.
GPT_OSS_20B="${WORK}/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

vllm serve "${GPT_OSS_20B}" \
  --served-model-name openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

**For benchmark runs (`gpt-oss-120b`) — H200 with 4-way tensor parallelism:**

```bash
# If gpt-oss-120b weights are cached locally, use the snapshot path analogously:
GPT_OSS_120B="${WORK}/huggingface/hub/models--openai--gpt-oss-120b/snapshots/<hash>"
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 1\
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```
> **Note:** No `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8` or `--kv-cache-dtype fp8`
> flags needed for either model on A40/H200 — those are Blackwell (B200) specific.
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
export EXECUTING_LLM_API_KEY=EMPTY


# For testing (gpt-oss-20b):
export EXECUTING_LLM_MODEL=openai/gpt-oss-20b

# For benchmark runs (gpt-oss-120b):
# export EXECUTING_LLM_MODEL=openai/gpt-oss-120b
```

Quick connectivity check:

```bash
curl http://<node>:8000/v1/models
```

Expected response includes `"id": "openai/gpt-oss-20b"` (or `gpt-oss-120b` for the larger model).

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
| `Disk quota exceeded` during pip install | `$HOME` (100 GiB) is full. Use `$WORK` (1 TiB): set `export TMPDIR="${WORK}/tmp_pip" PIP_CACHE_DIR="${WORK}/tmp_pip/cache"` and store the venv under `$WORK/venvs/` as shown in step 2 |
| `Killed` during pip install | `/tmp` on login nodes is small; always run installs from a compute node via `salloc` |
| `Connection refused` on login node | Check node hostname; ensure port 8000 is not firewalled |
| `tl.language not defined` | Do not install extra `pytorch-triton` alongside vLLM |
| Harmony vocab download failure | Pre-download tiktoken files and set `TIKTOKEN_ENCODINGS_BASE` |
| Model download slow/fails | Weights are already cached at `${WORK}/huggingface/hub/`. Pass the local snapshot path directly to `vllm serve` (see step 3) or set `export HF_HUB_CACHE="${WORK}/huggingface/hub"` |
