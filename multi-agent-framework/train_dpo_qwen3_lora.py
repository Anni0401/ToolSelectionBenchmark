#!/usr/bin/env python3
"""Fine-tune Qwen3 8B for query rewriting with DPO + LoRA.

Expected input: JSON list with fields similar to generate_dpo_ranked_preferences.py output:
[
  {
    "prompt": "...",
    "chosen": "..." | {"query": "..."},
    "rejected": "..." | {"query": "..."},
    "gold_tools": ["..."],
    "score_chosen": 1.23,
    "score_rejected": 0.42
  }
]

The script:
- normalizes prompt/chosen/rejected,
- filters low-quality or invalid pairs,
- performs optional group-based train/val splitting,
- trains a LoRA policy with TRL DPOTrainer,
- keeps retriever/tool DB/executor untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


DEFAULT_MODEL = "Qwen/Qwen3-8B"

REWRITE_PROMPT_TEMPLATE = (
    "You rewrite user requests to improve retrieval of relevant tools.\n\n"
    "Your task is NOT to answer the user and NOT to call any tools.\n\n"
    "Rewrite the user request so that it clearly expresses:\n"
    "- the entities involved;\n"
    "- the intended operations;\n"
    "- intermediate steps;\n"
    "- identifiers that may need to be obtained;\n"
    "- required filters, limits, dates, or arguments;\n"
    "- whether the same tool must be called multiple times.\n\n"
    "Use the terminology and operation style suggested by the example tool\n"
    "definitions below. Do not invent tools or APIs. Preserve the user's\n"
    "intent and all important argument values.\n\n"
    "Example tool definitions:\n"
    "{sampled_tool_documents}\n\n"
    "Original user request:\n"
    "{user_query}\n\n"
    "Return only one rewritten retrieval query. Do not include explanations,\n"
    "JSON, tool calls, or an answer."
)


@dataclass
class PairRecord:
    prompt: str
    chosen: str
    rejected: str
    group_key: str
    score_chosen: float | None
    score_rejected: float | None


def build_training_prompt(original_query: str, sampled_tool_documents: str) -> str:
    return REWRITE_PROMPT_TEMPLATE.format(
        sampled_tool_documents=sampled_tool_documents,
        user_query=original_query,
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "queries_gold_tools_batch1_dpo_ranked.json"
    default_output = script_dir / "qwen3-8b-dpo-lora"

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input", type=str, default=str(default_input), help="Path to DPO JSON array")
    parser.add_argument("--output-dir", type=str, default=str(default_output), help="Training output directory")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="Base policy model")

    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="DPO learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--global-batch-size", type=int, default=32, help="Target global batch size")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2, help="Per-device eval batch size")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=768, help="Max prompt length")

    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (0 disables split)")
    parser.add_argument(
        "--group-by",
        type=str,
        default="gold_tools",
        choices=["gold_tools", "prompt_hash", "none"],
        help="How to avoid leakage across train/val",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules",
    )

    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to HF loaders")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--logging-steps", type=int, default=1, help="Training logging interval")
    parser.add_argument("--save-strategy", type=str, default="epoch", choices=["epoch", "steps", "no"], help="Checkpoint save strategy")
    parser.add_argument("--save-steps", type=int, default=50, help="Save interval when save-strategy=steps")
    parser.add_argument("--report-to", type=str, default="none", help="Trainer report_to setting")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Enable float16")
    parser.add_argument("--no-eval", action="store_true", help="Disable validation even if val-ratio > 0")

    parser.add_argument(
        "--export-cleaned-json",
        action="store_true",
        help="Write cleaned train/val JSON files under output-dir for auditability",
    )
    return parser.parse_args()


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if isinstance(value.get("query"), str):
            return value["query"].strip()
    return ""


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _sampled_examples_as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip()).strip()
    return ""


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _group_key(row: dict[str, Any], prompt: str, mode: str) -> str:
    if mode == "none":
        return "all"
    if mode == "prompt_hash":
        return _prompt_hash(prompt)

    gold_tools = row.get("gold_tools", [])
    if not isinstance(gold_tools, list) or not gold_tools:
        return f"prompt::{_prompt_hash(prompt)}"
    cleaned = sorted(str(item).strip() for item in gold_tools if str(item).strip())
    if not cleaned:
        return f"prompt::{_prompt_hash(prompt)}"
    return "gold::" + "|".join(cleaned)


def normalize_and_filter(rows: list[dict[str, Any]], group_mode: str) -> tuple[list[PairRecord], dict[str, int]]:
    stats = {
        "total": len(rows),
        "kept": 0,
        "skip_missing_field": 0,
        "skip_identical": 0,
        "skip_tie": 0,
    }

    cleaned: list[PairRecord] = []
    for row in rows:
        original_query = _as_text(row.get("original_query"))
        sampled_tool_examples = _sampled_examples_as_text(row.get("sampled_tool_examples"))
        prompt = ""
        if original_query and sampled_tool_examples:
            prompt = build_training_prompt(original_query, sampled_tool_examples)
        else:
            prompt = _as_text(row.get("prompt"))
        chosen = _as_text(row.get("chosen"))
        rejected = _as_text(row.get("rejected"))

        if not prompt or not chosen or not rejected:
            stats["skip_missing_field"] += 1
            continue

        if chosen == rejected:
            stats["skip_identical"] += 1
            continue

        score_chosen = _safe_float(row.get("score_chosen"))
        score_rejected = _safe_float(row.get("score_rejected"))
        if score_chosen is not None and score_rejected is not None and math.isclose(score_chosen, score_rejected, rel_tol=1e-9, abs_tol=1e-9):
            stats["skip_tie"] += 1
            continue

        cleaned.append(
            PairRecord(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                group_key=_group_key(row, prompt, group_mode),
                score_chosen=score_chosen,
                score_rejected=score_rejected,
            )
        )

    stats["kept"] = len(cleaned)
    return cleaned, stats


def split_grouped(pairs: list[PairRecord], val_ratio: float, seed: int) -> tuple[list[PairRecord], list[PairRecord]]:
    if val_ratio <= 0.0 or len(pairs) < 2:
        return pairs, []

    groups: dict[str, list[PairRecord]] = {}
    for p in pairs:
        groups.setdefault(p.group_key, []).append(p)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    target_val = max(1, int(round(len(pairs) * val_ratio)))
    val: list[PairRecord] = []
    train: list[PairRecord] = []

    for key in group_keys:
        bucket = groups[key]
        if len(val) < target_val:
            val.extend(bucket)
        else:
            train.extend(bucket)

    if not train:
        # Ensure non-empty training split.
        last_group = group_keys[-1]
        moved = groups[last_group]
        train.extend(moved)
        val = [p for p in val if p.group_key != last_group]

    return train, val


def to_dataset(rows: list[PairRecord]) -> Dataset:
    return Dataset.from_list([
        {"prompt": row.prompt, "chosen": row.chosen, "rejected": row.rejected}
        for row in rows
    ])


def compute_grad_acc(global_batch_size: int, per_device_batch_size: int) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    denom = max(1, per_device_batch_size * world_size)
    return max(1, math.ceil(global_batch_size / denom))


def maybe_export_cleaned(output_dir: Path, train_rows: list[PairRecord], val_rows: list[PairRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(path: Path, rows: list[PairRecord]) -> None:
        payload = [
            {
                "prompt": r.prompt,
                "chosen": r.chosen,
                "rejected": r.rejected,
                "group_key": r.group_key,
                "score_chosen": r.score_chosen,
                "score_rejected": r.score_rejected,
            }
            for r in rows
        ]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _dump(output_dir / "train_pairs_cleaned.json", train_rows)
    _dump(output_dir / "val_pairs_cleaned.json", val_rows)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.bf16 and args.fp16:
        raise ValueError("Choose either --bf16 or --fp16, not both.")
    if not (1 <= args.per_device_train_batch_size <= 4):
        raise ValueError("per-device train batch size should be in [1, 4] for this setup.")
    if not (1 <= args.per_device_eval_batch_size <= 4):
        raise ValueError("per-device eval batch size should be in [1, 4] for this setup.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input must be a JSON list.")

    normalized, stats = normalize_and_filter(rows, args.group_by)
    if len(normalized) < 2:
        raise ValueError(f"Need at least 2 valid DPO pairs, found {len(normalized)}")

    if args.no_eval:
        train_rows, val_rows = normalized, []
    else:
        train_rows, val_rows = split_grouped(normalized, args.val_ratio, args.seed)

    if args.export_cleaned_json:
        maybe_export_cleaned(output_dir, train_rows, val_rows)

    print("================ DPO DATA SUMMARY ================")
    print(f"Input file:             {input_path}")
    print(f"Rows total:             {stats['total']}")
    print(f"Rows kept:              {stats['kept']}")
    print(f"Skipped missing fields: {stats['skip_missing_field']}")
    print(f"Skipped identical:      {stats['skip_identical']}")
    print(f"Skipped ties:           {stats['skip_tie']}")
    print(f"Train pairs:            {len(train_rows)}")
    print(f"Validation pairs:       {len(val_rows)}")
    print("==================================================")

    train_dataset = to_dataset(train_rows)
    eval_dataset = to_dataset(val_rows) if val_rows else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_dtype = None
    if args.bf16:
        model_dtype = torch.bfloat16
    elif args.fp16:
        model_dtype = torch.float16

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        policy_model.config.use_cache = False

    lora_targets = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    grad_acc = compute_grad_acc(args.global_batch_size, args.per_device_train_batch_size)
    report_to = [] if args.report_to.lower() == "none" else [args.report_to]

    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,
        report_to=report_to,
        seed=args.seed,
    )

    print("================ TRAINING CONFIG =================")
    print(f"Model:                  {args.model_name}")
    print("Retriever:              frozen (external, not part of this script)")
    print("Policy adapters:        LoRA")
    print("Reference model:        frozen base copy via TRL (ref_model=None)")
    print(f"Epochs:                 {args.epochs}")
    print(f"Learning rate:          {args.learning_rate}")
    print(f"DPO beta:               {args.beta}")
    print(f"Global batch target:    {args.global_batch_size}")
    print(f"Per-device train batch: {args.per_device_train_batch_size}")
    print(f"WORLD_SIZE:             {os.environ.get('WORLD_SIZE', '1')}")
    print(f"Grad accumulation:      {grad_acc}")
    print("==================================================")

    trainer = DPOTrainer(
        model=policy_model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "kept_pairs": len(normalized),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
            "beta": args.beta,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        }
    )

    metrics_path = output_dir / "train_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("================ TRAINING DONE ===================")
    print(f"Saved model:            {output_dir / 'final'}")
    print(f"Saved metrics:          {metrics_path}")
    print("==================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
