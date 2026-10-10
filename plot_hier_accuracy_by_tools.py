"""Task accuracy of the 'hierarchical' strategy, split by the number of tools it retrieved.

Usage: python plot_hier_accuracy_by_tools.py --model 120B
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from hierarchical_unique_wins import RESULT_ROOTS, load_scores, load_selections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RESULT_ROOTS), default="120B")
    ap.add_argument("--strategy", default="hierarchical")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"hier_accuracy_by_tools_{args.model}")

    scores = load_scores(args.model, args.strategy)
    sels = load_selections(args.model, args.strategy)

    buckets: dict[int, list[bool]] = defaultdict(list)
    for key, sc in scores.items():
        if key in sels:
            buckets[len(sels[key])].append(sc["correct"])
    ns = sorted(buckets)
    acc = [sum(buckets[n]) / len(buckets[n]) for n in ns]
    cnt = [len(buckets[n]) for n in ns]

    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_tools", "n_tasks", "n_correct", "accuracy"])
        for n, a, c in zip(ns, acc, cnt):
            w.writerow([n, c, sum(buckets[n]), f"{a:.4f}"])

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(ns) + 3), 4.5))
    bars = ax.bar([str(n) for n in ns], [100 * a for a in acc], color="tab:blue")
    for b, a, c in zip(bars, acc, cnt):
        ax.text(b.get_x() + b.get_width() / 2, 100 * a + 1, f"{100 * a:.0f}%\n(n={c})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Number of retrieved tools")
    ax.set_ylabel("Task accuracy (%)")
    ax.set_title(f"{args.strategy} ({args.model}): accuracy by number of retrieved tools")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"{sum(cnt)} tasks; wrote {out}.png / .csv")
    for n, a, c in zip(ns, acc, cnt):
        print(f"{n:>3} tools: {100 * a:5.1f}%  (n={c})")


if __name__ == "__main__":
    main()
