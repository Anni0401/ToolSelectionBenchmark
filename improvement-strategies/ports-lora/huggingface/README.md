---
base_model: Qwen/Qwen3-Embedding-8B
library_name: peft
license: apache-2.0
pipeline_tag: feature-extraction
tags:
  - lora
  - peft
  - embeddings
  - retrieval
  - tool-retrieval
  - tool-selection
  - ports
---

# Qwen3-Embedding-8B PORTS LoRA (tool retrieval)

LoRA adapter for [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), fine-tuned with
**PORTS** (*Preference-Optimized Retrievers for Tool Selection with Large Language Models*) so that the retriever
aligns with the preferences of a frozen LLM when selecting tools. It is one of the improvement strategies evaluated in the
[ToolSelectionBenchmark](https://github.com/Anni0401/ToolSelectionBenchmark) repository (`improvement-strategies/ports-lora`).

This repository contains **only the adapter weights**. They must be loaded on top of `Qwen/Qwen3-Embedding-8B`.

## Training

| | |
|---|---|
| Method | PORTS (perplexity-based preference loss + contrastive semantic loss), code adapted from the PORTS authors |
| LoRA | r = 16, alpha = 16, dropout = 0.1, target modules `q,k,v,o,gate,up,down_proj`, QLoRA enabled |
| Hyperparameters | learning rate 1e-5, lambda 0.3, beta 1, gamma 0.5, 2 epochs, gradient accumulation 32 |
| Data | BFCL train split, 1,000 training samples, 20% held out for validation |
| Frozen inference LLM | Llama-3-8B (provides the preference signal) |
| Checkpoint | end of epoch 2 |

The training scripts and sweep definitions are in the GitHub repository. The PORTS source code belongs to its authors;
the repository contains only the files we modified or added, with attribution.

## Usage

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

base = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B", padding_side="left")
model = PeftModel.from_pretrained(base, "Anni0401/qwen3-embedding-8b-ports-lora")
```

With vLLM: `vllm serve Qwen/Qwen3-Embedding-8B --enable-lora --max-lora-rank 16 --lora-modules qwen3-embedding-8b-ports-lora=Anni0401/qwen3-embedding-8b-ports-lora`.

## Evaluation

End-to-end tool-selection results on Wild-Tool-Bench are reported in the GitHub repository together with the evaluation scripts and logs.

## License

Apache-2.0, inherited from the base model. The BFCL data is released under Apache-2.0 by its authors.
