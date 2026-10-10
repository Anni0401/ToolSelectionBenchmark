"""Pairwise overlap of retrieved toolsets between selection strategies, averaged over tasks.

Two per-task scores (both in [0, 1]), averaged over all tasks selected by every strategy:
    * Jaccard      : |A & B| / |A | B|          (order ignored; 1 only for identical tool sets)
  * rank-aware   : normalised Rank-Biased Overlap, RBO = sum_d w_d * |A[:d] & B[:d]| / d,
                   w_d = p^(d-1), normalised so that identical rankings score 1.

Usage: python plot_toolset_overlap.py --model 120B --out toolset_overlap_120B
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hierarchical_unique_wins import RESULT_ROOTS, load_selections

STRATEGIES = ["embedding", "embedding_context", "reranker", "reranker_context"]
AVAILABLE_STRATEGIES = [*STRATEGIES, "embedding_context_ports_lora"]


def jaccard_overlap(a: list[str], b: list[str]) -> float:
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def rbo(a: list[str], b: list[str], p: float) -> float:
    depth = max(len(a), len(b))
    if depth == 0:
        return 0.0
    num = den = 0.0
    for d in range(1, depth + 1):
        w = p ** (d - 1)
        agreement = len(set(a[:d]) & set(b[:d])) / d
        num += w * agreement
        den += w
    return num / den


def heatmap(ax, mat: np.ndarray, title: str, strategies: list[str]) -> None:
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(strategies)), strategies, rotation=30, ha="right")
    ax.set_yticks(range(len(strategies)), strategies)
    ax.set_title(title)
    for i, j in product(range(len(strategies)), repeat=2):
        ax.text(j, i, f"{100 * mat[i, j]:.1f}%", ha="center", va="center",
                color="white" if mat[i, j] < 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RESULT_ROOTS), default="120B")
    ap.add_argument("--strategies", nargs="+", choices=AVAILABLE_STRATEGIES, default=STRATEGIES,
                    help="strategies to compare (default: embedding, embedding_context, reranker, reranker_context)")
    ap.add_argument("--p", type=float, default=0.8, help="RBO persistence; lower = more top-weighted")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"toolset_overlap_{args.model}")

    strategies = args.strategies
    sels = {s: load_selections(args.model, s) for s in strategies}
    keys = sorted(set.intersection(*(set(v) for v in sels.values())))
    print(f"{len(keys)} tasks present in all strategies")

    n = len(strategies)
    m_jaccard, m_rbo = np.zeros((n, n)), np.zeros((n, n))
    for i, j in product(range(n), repeat=2):
        a, b = sels[strategies[i]], sels[strategies[j]]
        m_jaccard[i, j] = np.mean([jaccard_overlap(a[k], b[k]) for k in keys])
        m_rbo[i, j] = np.mean([rbo(a[k], b[k], args.p) for k in keys])

    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        for name, mat in (("jaccard_overlap", m_jaccard), (f"rbo_p{args.p}", m_rbo)):
            w.writerow([name] + strategies)
            for s, row in zip(strategies, mat):
                w.writerow([s] + [f"{v:.4f}" for v in row])
            w.writerow([])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    heatmap(axes[0], m_jaccard, "Jaccard overlap (rank ignored)", strategies)
    heatmap(axes[1], m_rbo, f"Rank-biased overlap (p={args.p})", strategies)
    fig.suptitle(f"Retrieved toolset overlap, {args.model}, {len(keys)} tasks")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out.with_suffix('.png')}, {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
