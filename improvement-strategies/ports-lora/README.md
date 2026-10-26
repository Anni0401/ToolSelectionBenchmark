# PORTS LoRA (embedding retriever fine-tuning)

Fine-tunes the `Qwen/Qwen3-Embedding-8B` retriever with
**PORTS** (*Preference-Optimized Retrievers for Tool Selection with Large Language Models*) and serves it as a LoRA adapter
in the benchmark's embedding strategies (`embedding_ports_lora`).

## Attribution

The PORTS training code was written by the authors of the PORTS paper and is **not redistributed here**: we found no license in
the code we imported, so this folder only contains the files we modified or added on top of it. The unmodified import is
available in the git history of this repository (commit `08008e8`, `improvement-strategies/PORTS/ports/`); the original datasets
are linked from the PORTS paper (<https://anonymous.4open.science/r/ports-data/>).

## What is in `overlay/`

`overlay/` mirrors the directory layout of the PORTS repository. Copy it over a checkout of the PORTS code to get the version used here:

```bash
cp -r overlay/main/. <ports-checkout>/main/
```

| File | Change |
|---|---|
| `main/main_train_port.py` | BFCL validation split (`--bfcl_validation_size`), `--final_test_eval`, gradient accumulation, sweep-aware W&B run names |
| `main/src/port/dataset_helper.py`, `main/src_port/dataset_helper.py` | load BFCL from local `datasets/BFCL/*.jsonl` files |
| `main/scripts/train_ports.sh`, `run_ports_train_sbatch.sh` | training entry points used for the final run |
| `main/scripts/train_ports_sweep.sh`, `slurm_wandb_sweep_agent.sh` | W&B sweep entry point and SLURM agent |
| `main/sweep_ports.yaml`, `sweep_ports_joint.yaml`, `sweep_ports_lora.yaml` | hyperparameter sweeps (core, joint, LoRA) |

## Final configuration

BFCL (1,000 train samples, 20% validation), frozen inference LLM Llama-3-8B, learning rate 1e-5, lambda 0.3, beta 1, gamma 0.5,
2 epochs, LoRA r = 16 / alpha = 16 / dropout 0.1 with QLoRA (defaults of `overlay/main/scripts/train_ports.sh`).

## Adapter weights

Published on HuggingFace: [`Anni0401/qwen3-embedding-8b-ports-lora`](https://huggingface.co/Anni0401/qwen3-embedding-8b-ports-lora)
(model card source: [`huggingface/README.md`](./huggingface/README.md)). The evaluation runners in `../eval/` read the adapter from
`EMBEDDING_LORA_ADAPTER_DIR`; download it with `hf download Anni0401/qwen3-embedding-8b-ports-lora --local-dir <dir>` and point the variable at `<dir>`.
