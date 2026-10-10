# Notebook cell: paste into analyze_results.ipynb (replaces the "15. Error categories" code cell).
# Needs all_data, gold_index, STRATEGIES, MODELS and the labels from the setup cells.
ERROR_CATEGORIES = [
    "1) correct",
    "2) wrong: gold tool not in selected set",
    "3) wrong: gold tool selected, wrong tool name called",
    "4) wrong: correct tool names, wrong parameters",
    "5) wrong: correct tools, wrong trajectory (order/missing/extra calls)",
]
ERROR_COLORS = ["#4c9f70", "#d1495b", "#edae49", "#66a3c7", "#8d6a9f"]


def classify_task(model: str, strategy: str, key: tuple[str, int]) -> int:
    """Returns the 0-based index into ERROR_CATEGORIES."""
    task = all_data[model][strategy]["tasks"].get(key)
    if task is None:
        return 1  # missing result counts as failed retrieval/execution
    if task["correct"]:
        return 0
    gold = gold_index[key]["gold_tools"]
    if strategy != "in_context":
        selected = all_data[model][strategy]["selection"].get(key, set())
        if not set(gold) <= selected:
            return 1
    pred = task["predicted_tools"]
    if pred == gold:
        return 3  # same tool sequence, so parameters must differ
    g, p = Counter(gold), Counter(pred)
    if p == g or not (p - g) or not (g - p):
        return 4  # order differs, calls missing, or extra calls
    return 2


TASK_TYPES = sorted({g["task_type"] for g in gold_index.values()})
error_counts_tt = {
    (m, t): {s: Counter(classify_task(m, s, k) for k, g in gold_index.items() if g["task_type"] == t) for s in STRATEGIES}
    for m in MODELS for t in TASK_TYPES
}
n_per_type = Counter(g["task_type"] for g in gold_index.values())

fig, axes = plt.subplots(
    len(MODELS), len(TASK_TYPES), figsize=(4.2 * len(TASK_TYPES), 4.8 * len(MODELS)), sharey=True, squeeze=False
)
for r, m in enumerate(MODELS):
    for c_idx, t in enumerate(TASK_TYPES):
        ax = axes[r][c_idx]
        bottoms = [0.0] * len(STRATEGIES)
        for c, (label, color) in enumerate(zip(ERROR_CATEGORIES, ERROR_COLORS)):
            vals = [error_counts_tt[(m, t)][s][c] / n_per_type[t] for s in STRATEGIES]
            bars = ax.bar(range(len(STRATEGIES)), vals, bottom=bottoms, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.0%}" if v >= 0.06 else "" for v in vals], label_type="center", fontsize=7, color="white")
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{MODEL_LABELS[m]} — {t} (n={n_per_type[t]})", fontsize=10)
        ax.set_ylim(0, 1)
    axes[r][0].set_ylabel("Share of tasks")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
plt.show()

error_table_tt = pd.DataFrame(
    [
        {"model": m, "task_type": t, "strategy": STRATEGY_LABELS[s], "n": n_per_type[t],
         **{lab: error_counts_tt[(m, t)][s][c] for c, lab in enumerate(ERROR_CATEGORIES)}}
        for m in MODELS for t in TASK_TYPES for s in STRATEGIES
    ]
).set_index(["model", "task_type", "strategy"])
error_table_tt
