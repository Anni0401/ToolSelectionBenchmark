# Improvement strategies

Two trained components that try to improve tool retrieval in the embedding-based strategies of the benchmark.

| Strategy | What is trained | Folder | Adapter on HuggingFace |
|---|---|---|---|
| **DPO query rewriting** | `Qwen/Qwen3-8B` LoRA that rewrites the user query before retrieval (DPO on retrieval-ranked preference pairs) | [`dpo-query-rewrite/`](dpo-query-rewrite) | [`Anni0401/qwen3-8b-dpo-query-rewrite-lora`](https://huggingface.co/Anni0401/qwen3-8b-dpo-query-rewrite-lora) |
| **PORTS retriever** | `Qwen/Qwen3-Embedding-8B` LoRA trained with the PORTS preference objective | [`ports-lora/`](ports-lora) | [`Anni0401/qwen3-embedding-8b-ports-lora`](https://huggingface.co/Anni0401/qwen3-embedding-8b-ports-lora) |

The evaluation runners for both strategies (SLURM scripts that start the vLLM servers and run the benchmark) are in [`eval/`](eval).
The benchmark integration itself (selector classes in `wild-tool-bench/wtb/model_handler/api_inference/langgraph_app.py`) is part of the benchmark.

## Layout

```
improvement-strategies/
  dpo-query-rewrite/
    data/                    DPO dataset: source queries, rewrite pairs, ranked pairs, raw generator output
    generate_rewrite_pairs_*.py, generate_dpo_ranked_preferences.py, extract_query_gold_tools.py   dataset construction
    train_dpo_qwen3_lora.py  DPO + LoRA training (TRL)
    run.sh, sweep*.yaml      W&B sweeps (DPO hyperparameters, LoRA, joint)
    sweeps/                  sweep_summary.csv (all runs) + summarize_sweeps.py
    slurm/                   SLURM jobs (pair generation, ranking, training, sweep agent)
    outputs/                 local training outputs (adapter weights are git-ignored)
    huggingface/             model card of the published adapter
  ports-lora/                our modifications of the PORTS code + sweeps (see its README)
  eval/                      evaluation runners for the DPO and PORTS strategies
```

## Reproducing the DPO query rewriter

All commands run from the repository root unless noted. Virtual environment: `.venv-dpo` (TRL, PEFT, vLLM as needed).

1. **Source queries**: `multi-agent-framework/generate.py` produces `*_train.jsonl` (kept in `data/raw_generation/`);
   `extract_query_gold_tools.py` flattens them into `data/queries_gold_tools*.jsonl`.
2. **Rewrite pairs** (two rewrites per query with `gpt-oss-120b`): `sbatch improvement-strategies/dpo-query-rewrite/slurm/run_rewrite_pairs_gptoss.sh`
   → `data/queries_gold_tools_batch1_rewrites.json`.
3. **Rank pairs by retrieval reward** (embedding server required): `sbatch improvement-strategies/dpo-query-rewrite/slurm/slurm_ranked_dpo_job.sh`
   → `data/queries_gold_tools_batch1_dpo_ranked.json` (1,114 pairs).
4. **Train**: `sbatch improvement-strategies/dpo-query-rewrite/slurm/slurm_dpo_qwen3_lora.sh`
   → `outputs/qwen3-8b-dpo-lora/final` (the adapter used in the evaluation: lr 1e-5, beta 0.05, 2 epochs, r 8, alpha 32, dropout 0).
5. **Sweeps**: `cd improvement-strategies/dpo-query-rewrite && wandb sweep sweep.yaml` (also `sweep_lora.yaml`, `sweep_joint.yaml`), then
   `sbatch improvement-strategies/dpo-query-rewrite/slurm/slurm_dpo_sweep_agent.sh <SWEEP_ID>` from the repository root.
   Set `WANDB_ENTITY` (and optionally `WANDB_PROJECT`) in your environment. Results of all 133 sweep and baseline runs are in
   [`dpo-query-rewrite/sweeps/sweep_summary.csv`](dpo-query-rewrite/sweeps/sweep_summary.csv); rebuild it from run directories with `sweeps/summarize_sweeps.py`.
6. **Evaluate**: `sbatch improvement-strategies/eval/run_dpo.sh` (gpt-oss-120b executor) or `run_dpo_laguna.sh`.
   `REWRITE_LORA_ADAPTER_DIR` defaults to the local `outputs/` path; to use the published adapter run
   `hf download Anni0401/qwen3-8b-dpo-query-rewrite-lora --local-dir improvement-strategies/dpo-query-rewrite/outputs/qwen3-8b-dpo-lora/final`.

## What is tracked in git

Tracked: code, SLURM scripts, sweep definitions, the DPO dataset, `sweep_summary.csv`, training metrics of the final run and the model cards.
Not tracked: adapter weights and checkpoints (published on HuggingFace), W&B run directories, SLURM `.out`/`.err` logs, `.env` files.
