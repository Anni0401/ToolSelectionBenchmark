---
base_model: Qwen/Qwen3-8B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - lora
  - peft
  - dpo
  - trl
  - query-rewriting
  - tool-retrieval
  - tool-selection
---

# Qwen3-8B DPO query-rewrite LoRA (tool retrieval)

LoRA adapter for [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), trained with Direct Preference Optimization (DPO) to
**rewrite user requests so that an embedding retriever finds the right tools**. It is one of the improvement strategies
evaluated in the [ToolSelectionBenchmark](https://github.com/Anni0401/ToolSelectionBenchmark) repository
(`improvement-strategies/dpo-query-rewrite`). The retriever, the tool database and the executing LLM are untouched; only the
rewriting step is fine-tuned.

This repository contains **only the adapter weights**. They must be loaded on top of `Qwen/Qwen3-8B`.

## Training

| | |
|---|---|
| Method | DPO (TRL `DPOTrainer`), reference model = frozen base model |
| LoRA | r = 8, alpha = 32, dropout = 0.0, target modules `q,k,v,o,gate,up,down_proj` |
| Hyperparameters | learning rate 1e-5, DPO beta 0.05, 2 epochs, global batch size 32, bf16, gradient checkpointing |
| Sequence length | max 1024 tokens, max prompt 768 tokens |
| Data | 1,114 preference pairs (891 train / 223 validation, split grouped by gold tools) |
| Validation preference accuracy (`eval_rewards/accuracies`) | 0.62 at the last step (0.66 at step 50) |

### Preference data

Queries come from synthetic multi-turn tool-use tasks generated with the repository's multi-agent data generator.
None of the 60 gold tools appears in the Wild-Tool-Bench tool set or in the benchmark tool list.
For each query two rewrites were generated with `openai/gpt-oss-120b`. Each rewrite was embedded with
`Qwen/Qwen3-Embedding-8B` and scored by how highly the gold tools are ranked among all candidate tools. The better-scoring
rewrite is `chosen`, the other is `rejected`. The dataset is in
[`data/queries_gold_tools_batch1_dpo_ranked.json`](https://github.com/Anni0401/ToolSelectionBenchmark/blob/main/improvement-strategies/dpo-query-rewrite/data/queries_gold_tools_batch1_dpo_ranked.json).

## Usage

The adapter expects the rewrite prompt defined in `REWRITE_PROMPT_TEMPLATE` in
[`generate_rewrite_pairs_vllm.py`](https://github.com/Anni0401/ToolSelectionBenchmark/blob/main/improvement-strategies/dpo-query-rewrite/generate_rewrite_pairs_vllm.py)
(sampled tool examples + the user query).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "Anni0401/qwen3-8b-dpo-query-rewrite-lora")
```

With vLLM: `vllm serve Qwen/Qwen3-8B --enable-lora --lora-modules qwen3-8b-dpo-lora=Anni0401/qwen3-8b-dpo-query-rewrite-lora`.

## Evaluation

The end-to-end effect on tool selection (Wild-Tool-Bench, with `gpt-oss-120b` and Laguna as executors) is reported in the GitHub
repository, together with all evaluation scripts and logs. The preference accuracy above is **not** the end-task accuracy.

## Limitations

- Trained on 1,114 synthetic English pairs; it is a research artifact, not a general-purpose assistant.
- The preference signal depends on one specific retriever (`Qwen3-Embedding-8B`).

## License

Apache-2.0, inherited from the base model.
