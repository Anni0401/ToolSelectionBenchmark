"""Regenerate query_rewrite_output_comparison.json from the query rewrite logs."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "query_rewrite_output_comparison.json"

SOURCES = {
    "120B": {
        "query_rewrite": ROOT / "result_120B_v2/query_rewrite_embedding_context/query_rewrite_logs.jsonl",
        "query_rewrite_dpo": ROOT / "result_120B_v2/query_rewrite_dpo_lora_embedding_context/query_rewrite_logs.jsonl",
    },
    "Laguna": {
        "query_rewrite": ROOT / "result_laguna/query_rewrite_embedding_context/query_rewrite_logs.jsonl",
        "query_rewrite_dpo": ROOT / "result_laguna/query_rewrite_dpo_lora_embedding_context/query_rewrite_logs.jsonl",
    },
}
CAMEL_RE = re.compile(r"^([a-z]+(?:[A-Z][a-z0-9]*)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)(?![A-Za-z0-9_])")
CHECKS = ["contains_equals", "contains_more_than_one_colon", "contains_parameters_word"]


def load_outputs(path: Path) -> dict[tuple[str, int], list[dict]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("stage") == "rewrite":
                    rows.append(row)
    rows.sort(key=lambda r: r.get("timestamp", 0))
    out = defaultdict(list)
    for row in rows:
        entry = {
            "request_id": row["request_id"],
            "timestamp": row["timestamp"],
            "input_query": row["input_query"],
            "output_query": row["output_query"],
        }
        match = CAMEL_RE.match(row["output_query"] or "")
        if match:
            text = row["output_query"]
            entry["camel_case_check"] = {
                "leading_tool_name": match.group(1),
                "contains_equals": "=" in text,
                "contains_more_than_one_colon": text.count(":") > 1,
                "contains_parameters_word": "parameters" in text.lower(),
            }
        out[(row["test_entry_id"], int(row["task_idx"]))].append(entry)
    return out


def analyze(outputs: list[dict]) -> dict:
    matches = [o["camel_case_check"] for o in outputs if "camel_case_check" in o]
    n = len(matches)
    combos = Counter(" + ".join(c for c in CHECKS if m[c]) or "none" for m in matches)
    counts = {c: sum(m[c] for m in matches) for c in CHECKS}
    n_true = [sum(m[c] for c in CHECKS) for m in matches]
    result = {
        "rewrite_outputs": len(outputs),
        "camel_case_matches": n,
        **counts,
        "exactly_one_check": sum(k == 1 for k in n_true),
        "at_least_one_check": sum(k >= 1 for k in n_true),
        "check_combinations": dict(combos),
    }
    for c in CHECKS:
        result[f"{c}_pct_of_matches"] = round(counts[c] / n * 100, 2) if n else 0.0
    result["camel_case_match_pct"] = round(n / len(outputs) * 100, 2) if outputs else 0.0
    return result


def build_model(sources: dict[str, Path]) -> dict:
    base = load_outputs(sources["query_rewrite"])
    dpo = load_outputs(sources["query_rewrite_dpo"])
    keys = sorted(set(base) | set(dpo), key=lambda k: (int(k[0].rsplit("_", 1)[1]), k[1]))
    tasks = [
        {
            "test_entry_id": k[0],
            "task_idx": k[1],
            "query_rewrite": base.get(k, []),
            "query_rewrite_dpo": dpo.get(k, []),
        }
        for k in keys
    ]
    return {
        "summary": {
            "tasks": len(keys),
            "baseline_outputs": sum(len(v) for v in base.values()),
            "dpo_outputs": sum(len(v) for v in dpo.values()),
            "tasks_with_both": len(set(base) & set(dpo)),
            "baseline_only_tasks": len(set(base) - set(dpo)),
            "dpo_only_tasks": len(set(dpo) - set(base)),
        },
        "tasks": tasks,
        "camel_case_analysis": {
            "baseline": analyze([o for t in tasks for o in t["query_rewrite"]]),
            "dpo": analyze([o for t in tasks for o in t["query_rewrite_dpo"]]),
        },
    }


def main() -> None:
    result = {
        "description": (
            "Rewrite and DPO output queries grouped by model and benchmark task. Repeated rewrite requests "
            "are preserved in timestamp order. Each output_query satisfying the leading camelCase/PascalCase "
            "test has camel_case_check flags for the three requested conditions."
        ),
        "pairing_key": ["test_entry_id", "task_idx"],
        "models": {model: build_model(sources) for model, sources in SOURCES.items()},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    for model, data in result["models"].items():
        print(model, data["summary"])


if __name__ == "__main__":
    main()
