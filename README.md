# Benchmarking LLM Tool-Use in the Wild


<p align="center">
    📖 <a>English</a> •
    <a href="README_ZH.md">中文</a> •
    📚 <a href="https://openreview.net/forum?id=yz7fL5vfpn">ICLR 2026 Paper</a> •
    🌐 <a href="https://yupeijei1997.github.io/WildToolBench/">Project Page</a>
</p>


![Example](./picture/benchmark_comparison.png)


## 📖 Overview

Fulfilling user needs through Large Language Model multi-turn, multi-step tool-use
is rarely a straightforward process. Real user interactions are inherently wild, being
unpredictable, messy, and flexible. We identify three key challenges from user
behaviour: compositional tasks that demand orchestration of tool-call topologies,
implicit intent spread across dialogue turns requiring contextual inference, and
instruction transition that mix task queries, clarifications, and casual conversation,
forcing LLMs to adjust their policies on the fly. Existing benchmarks overlook these
behaviors, causing the progress of LLMs observed on tool-use to be spurious. To
address this, we introduce WildToolBench, a LLM tool-use benchmark grounded in
real-world user behavior patterns. Comprehensive evaluations of 57 LLMs reveal
that no model achieves an accuracy of more than 15%, indicating a substantial
gap in the robustness of LLMs’ agentic ability. Controlled experiments and in-
depth analyses further indicate that the real challenge for LLM tool-use lies not in
artificially complex tasks, but in the wild nature of user behavior, emphasizing the
need to reconsider the interactions among LLMs, users, and tools.

## 😊 Key Materials

- Test data location: wild-tool-bench/data/Wild-Tool-Bench.jsonl
- More detailed information about the WildToolBench can be found below

## ⚡️ Quickstart

### Basic Installation

```bash
# Create a new Conda environment with Python 3.10
conda create -n WildToolBench python=3.10
conda activate WildToolBench

# Install the package
pip install -r requirements.txt
```

## ⏳ Inference

### 🤖 API Models
This project supports OpenAI-format API models.

Taking deepseek-chat as an example, refer to .env.example, create a .env file, and set the following keys.

```bash
DEEPSEEK_API_KEY=sk-XXXXXX
```

Afterwards, use the following code to request model results.

```bash
cd wild-tool-bench/

python3 -u -m wtb.openfunctions_evaluation --model=deepseek-chat
```

## 💫 Evaluation

Use the following code to evaluate the model's prediction results.

```bash
cd wild-tool-bench

python3 -u -m wtb.eval_runner --model=deepseek-chat
```

## 🧩 Our Extension: LangGraph Tool-Selection Benchmark

The original WildToolBench hands the model the full tool list of every task and evaluates it directly through an OpenAI-style API. We build on top of it to study **tool selection**: how much accuracy is lost when the executing LLM has to find the right tools among many, and which selection strategy recovers it best.

### What we added on top of WildToolBench

- **LangGraph tool-selection service** ([`langgraph_app.py`](wild-tool-bench/wtb/model_handler/api_inference/langgraph_app.py)): a local HTTP server (default `127.0.0.1:8001`) that runs a LangGraph `StateGraph` with two steps: (1) a pluggable `ToolSelector` reduces the task's tool pool to a small candidate set, (2) the executing LLM (any OpenAI-compatible endpoint, e.g. vLLM) answers using only those tools.
- **`langgraph` model handler** ([`langgraph.py`](wild-tool-bench/wtb/model_handler/api_inference/langgraph.py), registered in `handler_map.py`): lets `--model=langgraph` forward benchmark requests to that service, so the standard inference and evaluation pipeline is unchanged.
- **Enlarged tool pool**: each task's own tools are mixed with synthetic distractor tools (`tool_schemas_cache.jsonl`, built with `generate_synthetic_tools.py` / `build_gold_tool_set.py`), so selection is non-trivial.
- **Selection strategies**, chosen via `LANGGRAPH_TOOL_SELECTION_MODE`:

  | Mode | Strategy |
  |---|---|
  | `oracle` | Gold tools only (upper bound) |
  | `in_context` | All tools in the prompt (original WTB setting, baseline) |
  | `hierarchical` | An LLM selector first picks tool categories/tools, then the executor runs |
  | `toolreagt` | ReAct-style iterative tool search (max 10 iterations) |
  | `embedding`, `embedding_context` | Top-5 retrieval with OpenAI embeddings (`_context`: query includes dialogue context) |
  | `qwen3_embedding`, `qwen3_embedding_context` | Top-5 retrieval with `Qwen3-Embedding-8B` |
  | `*_embedding_reranker`, `*_embedding_context_reranker` | Retrieve 10, rerank with the executing LLM, keep 5 |
  | `qwen3_reranker` | Rerank all tools with `Qwen3-Reranker-8B` |
  | `qwen3_embedding_qwen3_reranker`, `qwen3_embedding_context_qwen3_reranker` | Retrieve 20 with Qwen3-Embedding, rerank to 5 with Qwen3-Reranker |
  | `qwen3_query_rewrite_embedding[_context]` | LLM query rewriting before embedding retrieval |

  Further experimental modes (PORTS-LoRA retriever, DPO-tuned query rewriter) are documented in [`improvement-strategies/`](improvement-strategies/README.md) and are not covered here.
- **Logging** of every selection and execution step (`tool_selection_logs.jsonl`, `tool_call_logs.jsonl`) next to the results, plus `LANGGRAPH_RESULT_DIR` to control where they go.
- **Seeds / sampling control** for the executing LLM (`EXECUTING_LLM_SEED`, `EXECUTING_LLM_SAMPLING`), tool-call parser selection (`EXECUTING_LLM_TOOL_CALL_PARSER`), and offline tool embedding caching ([`setup_openai_embeddings.py`](wild-tool-bench/wtb/model_handler/api_inference/setup_openai_embeddings.py)).
- **Deployment and run scripts** ([`deploy/`](deploy), [`runs/`](runs)): SLURM jobs that start all required vLLM servers, configure `.env`, start the LangGraph service, and run inference.
- **Analysis scripts**: `hierarchical_unique_wins.py`, `plot_*.py`, `analyze_tools_en.py`, `wild-tool-bench/analyze_results.ipynb`, and error-category scripts; `docs/` contains HTML viewers.

### How the pipeline runs

```
openfunctions_evaluation --model=langgraph ──HTTP──▶ langgraph_app (:8001)
                                                      ├─ ToolSelector  (LANGGRAPH_TOOL_SELECTION_MODE)
                                                      │     └─ embedding / reranker / selector-LLM servers (vLLM)
                                                      └─ executing LLM (EXECUTING_LLM_*, vLLM, e.g. gpt-oss-120b)
```

### Configuration (`wild-tool-bench/.env`)

The run scripts write these keys automatically; set them by hand when running manually.

| Variable | Meaning |
|---|---|
| `LANGGRAPH_ENDPOINT`, `LANGGRAPH_API_KEY` | URL of the LangGraph service used by the `langgraph` handler |
| `LANGGRAPH_TOOL_SELECTION_MODE` | Selection strategy (see table above) |
| `LANGGRAPH_RESULT_DIR` | Directory for selection/tool-call logs (set to the result dir by the scripts) |
| `EXECUTING_LLM_BASE_URL`, `EXECUTING_LLM_MODEL`, `EXECUTING_LLM_API_KEY` | Executing LLM (OpenAI-compatible), default model `openai/gpt-oss-120b` |
| `EXECUTING_LLM_REASONING`, `EXECUTING_LLM_SEED`, `EXECUTING_LLM_SAMPLING`, `EXECUTING_LLM_TOOL_CALL_PARSER` | Executor reasoning level, sampling, and tool-call parser |
| `QWEN3_EMBEDDING_BASE_URL`, `QWEN3_EMBEDDING_MODEL` | Embedding server (default `Qwen/Qwen3-Embedding-8B`) |
| `QWEN3_RERANKER_BASE_URL`, `QWEN3_RERANKER_MODEL` | Reranker server (default `Qwen/Qwen3-Reranker-8B`) |
| `LANGGRAPH_SELECTOR_LLM_ENDPOINT`, `_MODEL`, `_API_KEY` | Selector LLM for `hierarchical` |
| `QUERY_REWRITE_LLM_ENDPOINT`, `_MODEL` | Rewriting LLM for the query-rewrite modes |

### Running with the run scripts

The scripts in [`runs/`](runs) are SLURM jobs (`sbatch`) meant to be submitted **from the repository root** (they use `$SLURM_SUBMIT_DIR`). Each one: activates the venvs, starts the vLLM servers it needs, writes the `.env`, starts `langgraph_app`, waits for health checks, and runs the benchmark. Adapt partition, `WORK` path, and model paths at the top of the scripts to your cluster. Output (`*.out`/`*.err`) lands in the submit directory.

```bash
sbatch runs/run_in_context.sh                      # baseline: all tools in context (1 GPU)
sbatch runs/run_oracle.sh                          # gold tools (1 GPU); add --seed <n> for a seeded run
sbatch runs/run_embedding.sh                       # qwen3_embedding_context (2 GPU)
sbatch runs/run_reranker_2gpu.sh                   # qwen3_embedding_context_qwen3_reranker (2 GPU)
sbatch runs/run_hierarchical_qwen_selector_2gpu.sh # hierarchical with a Qwen selector (2 GPU)
sbatch runs/run_toolreagt_2gpu.sh                  # ReAct-style tool search (2 GPU)
sbatch runs/run_offline_embeddings.sh              # precompute tool embeddings cache (1 GPU)

# Laguna as executing LLM instead of gpt-oss-120b
sbatch runs/run_in_context_laguna.sh
sbatch runs/run_laguna_oracle.sh
sbatch runs/run_laguna_hierarchical.sh
sbatch runs/run_laguna_toolreagt.sh
sbatch runs/run_reranker_laguna.sh

# Hosted API (SAIA) instead of local vLLM
sbatch runs/run_in_context_saia.sh
```

The output location can be overridden through `RESULT_DIR`, e.g. `RESULT_DIR=result_v3_120B/reranker_context sbatch runs/run_reranker_2gpu.sh` (the default of this script). `EMBEDDING_MODEL` and `RERANKER_MODEL` can be overridden the same way.

### The invocation inside the scripts

All scripts start the service and run inference with the same calls; scoring is run separately afterwards. To run without SLURM, start your own vLLM servers, fill the `.env` keys above (`LANGGRAPH_ENDPOINT=http://127.0.0.1:8001/execute`), and run:

```bash
cd wild-tool-bench

# 1. Start the LangGraph selection service (host/port: LANGGRAPH_HOST / LANGGRAPH_PORT, default 127.0.0.1:8001)
python -m wtb.model_handler.api_inference.langgraph_app &

# 2. Inference: tool selection + execution for every task
export LANGGRAPH_RESULT_DIR=result_v3_120B/reranker_context
python -u -m wtb.openfunctions_evaluation \
    --model=langgraph \
    --result-dir result_v3_120B/reranker_context \
    --num-threads 1

# 3. Scoring (not part of the SLURM scripts; run after the job finished)
python -u -m wtb.eval_runner \
    --model=langgraph \
    --result-dir result_v3_120B/reranker_context \
    --score-dir score/120B_reranker_context
```

For embedding-based modes, the tool embeddings can be precomputed once with
`python -u wtb/model_handler/api_inference/setup_openai_embeddings.py --provider qwen3 --tools-file wtb/model_handler/api_inference/tool_schemas_cache.jsonl`
(`run_offline_embeddings.sh`); the cache is written to `wild-tool-bench/tool_embeddings_cache.json`.

### Results layout

```
wild-tool-bench/
  result_v3_120B/<strategy>/             # gpt-oss-120b executor
    langgraph/Wild-Tool-Bench_result.jsonl, Wild-Tool-Bench_metrics_summary.jsonl
    tool_selection_logs.jsonl, tool_call_logs.jsonl
  result_laguna/<strategy>/              # Laguna executor
  score/<model>_<strategy>/langgraph/    # Wild-Tool-Bench_score.jsonl, Wild-Tool-Bench_metric.json
```

## 🧠 Controllable Multi Agent Data Generation Framework

### ⚡️ Quickstart

Taking the example where all agents use deepseek-chat as the base model, and generate data in English. First, refer to .env.example, create a .env file, and set the following keys.

```bash
DEEPSEEK_API_KEY=sk-XXXXXX
LANGUAGE=en
```

The core innovation of this paper lies in the fact that our proposed WildToolBench is capable of covering all possible action spaces for any number of tasks, and except for the first round, all are true multi-turn tasks. Therefore, our framework can support the generation of data for any number of tasks. Here we take the generation of four tasks as an example, with the reference code as follows:
```bash
cd multi-agent-framework

python3 generate.py \
    --layer-num-total 4 \
    --user-model "deepseek-chat" \
    --planner-model "deepseek-chat" \
    --tool-model "deepseek-chat" \
    --agent-model "deepseek-chat" \
    --checker-model "deepseek-chat"
```
In particular, we have preserved data in three formats, including: raw generated data, streamlined data (with checker messages and role-switch messages removed), and captured data from each agent, available for researchers in need.

Finally, let's showcase a portion of the results generated for an actual case. It can be observed that the Continue Question User Agent has generated a true multi-turn task of implicit type, "Who is the uploader of the third wallpaper?" This implies that in order to solve this task, the model needs to refer to the preceding context to obtain the ID 778899 of the third wallpaper, and then continue to invoke the tool getWaifuDetails to retrieve detailed information about the wallpaper.

```
User: I need the latest 5 anime wallpapers.

Planner：getWaifuImages(limit=5, tags=["anime wallpapers"])

Tool：...

Planner：prepare_to_answer(answer_type=tool)

AI:

Here are the latest 5 anime wallpapers:
1. Wallpaper ID: 112233
 - [View Wallpaper](https://wallhaven.cc/w/112233)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/112233.jpg)
 - Resolution: 1920x1080
2. Wallpaper ID: 445566
 - [View Wallpaper](https://wallhaven.cc/w/445566)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/445566.jpg)
 - Resolution: 2560x1440
3. Wallpaper ID: 778899
 - [View Wallpaper](https://wallhaven.cc/w/778899)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/778899.jpg)
 - Resolution: 3840x2160
4. Wallpaper ID: 101112
 - [View Wallpaper](https://wallhaven.cc/w/101112)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/101112.jpg)
 - Resolution: 1920x1080
5. Wallpaper ID: 131415
 - [View Wallpaper](https://wallhaven.cc/w/131415)
 - ![Thumbnail](https://wallhaven.cc/thumb/small/131415.jpg)
 - Resolution: 2560x1440

You can click on the links to view the full wallpapers. Hope you enjoy these wallpapers! If you need more help, please let me know.

Continue Question User Agent Generate:
User: Who is the uploader of the third wallpaper?

Planner：getWaifuDetails(image_id=778899)
```

## 📝 Citation

If you find WildToolBench useful in your research, please consider citing our paper:

```bibtex
@inproceedings{
  yu2026wildtoolbench,
  title={Benchmarking {LLM} Tool-Use in the Wild},
  author={Peijie Yu and Wei Liu and Yifan Yang and Jinjian Li and Zelong Zhang and Xiao Feng and Feng Zhang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=yz7fL5vfpn}
}
```
