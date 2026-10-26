#!/usr/bin/env python3
"""Summarize DPO sweep run directories into a single CSV.

Each run directory is expected to contain train_metrics.json, final/adapter_config.json and
checkpoint-*/trainer_state.json (written by train_dpo_qwen3_lora.py). Checkpoints themselves are
large and are not tracked in git, so this CSV is the persistent record of the sweeps.

Usage:
    python summarize_sweeps.py /path/to/sweeps -o sweep_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def last_eval(run_dir: Path) -> dict:
    """Return the eval log entry of the last evaluation step found in any checkpoint."""
    best_step, best = -1, {}
    for state_path in run_dir.glob("checkpoint-*/trainer_state.json"):
        state = load_json(state_path) or {}
        for entry in state.get("log_history", []):
            if "eval_rewards/accuracies" in entry and entry.get("step", -1) > best_step:
                best_step, best = entry["step"], entry
    return {"step": best_step, **best} if best else {}


def summarize(run_dir: Path) -> dict | None:
    metrics = load_json(run_dir / "train_metrics.json")
    if metrics is None:
        return None
    adapter = load_json(run_dir / "final" / "adapter_config.json") or {}
    ev = last_eval(run_dir)
    return {
        "run": run_dir.name,
        "learning_rate": metrics.get("learning_rate"),
        "beta": metrics.get("beta"),
        "epochs": metrics.get("epochs"),
        "lora_r": adapter.get("r"),
        "lora_alpha": adapter.get("lora_alpha"),
        "lora_dropout": adapter.get("lora_dropout"),
        "train_pairs": metrics.get("train_pairs"),
        "val_pairs": metrics.get("val_pairs"),
        "train_loss": metrics.get("train_loss"),
        "final_eval_step": ev.get("step"),
        "eval_rewards_accuracies": ev.get("eval_rewards/accuracies"),
        "eval_rewards_margins": ev.get("eval_rewards/margins"),
        "eval_loss": ev.get("eval_loss"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("sweep_summary.csv"))
    args = parser.parse_args()

    rows = [r for d in sorted(args.sweeps_dir.iterdir()) if d.is_dir() and (r := summarize(d))]
    # Drop the institutional host domain from run names but keep host and pid, which keep names unique.
    for row in rows:
        row["run"] = re.sub(r"(_dws-\d+)\.[^_]+", r"\1", row["run"])
    rows.sort(key=lambda r: (r["eval_rewards_accuracies"] is None, -(r["eval_rewards_accuracies"] or 0)))

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
