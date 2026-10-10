"""Qualitative analysis: tasks solved by 'hierarchical' but by none of the other selection
strategies (in_context is ignored for the filter and only shown for reference).

Writes a long-format CSV (one row per task x strategy) and a nested JSON.

Usage: python hierarchical_unique_wins.py --model 120B --out hier_unique_120B
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "wild-tool-bench"
STRATEGIES = ["in_context", "embedding", "embedding_context", "reranker", "reranker_context", "hierarchical"]
COMPETITORS = ["embedding", "embedding_context", "reranker", "reranker_context"]
RESULT_ROOTS = {"120B": "result_v3_120B", "Laguna": "result_laguna"}


def score_dir(model: str, strategy: str) -> str:
    if model == "120B":
        return f"120B_{strategy}"
    return "laguna_in__context" if strategy == "in_context" else f"laguna_{strategy}"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predicted_and_answer(log: dict) -> tuple[list[dict], str]:
    steps = sorted((k for k in log if str(k).startswith("step_")), key=lambda k: int(k.split("_")[1]))
    tools, answer = [], ""
    for sk in steps:
        out = log[sk].get("inference_output", {}) or {}
        for tc in out.get("tool_calls", []) or []:
            fn = tc.get("function") or {}
            if fn.get("name"):
                tools.append({"name": fn["name"], "arguments": fn.get("arguments")})
        if out.get("content"):
            answer = out["content"]
    return tools, answer


def load_scores(model: str, strategy: str) -> dict[tuple[str, int], dict]:
    path = ROOT / "score" / score_dir(model, strategy) / "langgraph/Wild-Tool-Bench_score.jsonl"
    tasks = {}
    for rec in read_jsonl(path):
        for i, item in enumerate(rec["results"]):
            log = item.get("inference_log") or {}
            tools, answer = predicted_and_answer(log)
            tasks[(rec["id"], i)] = {"correct": item.get("label") == "correct", "executed": tools, "answer": answer}
    return tasks


def latest_block(rows: list[dict]) -> list[dict]:
    """Per test entry keep the newest run (a run restarts when task_idx decreases)."""
    by_entry = defaultdict(list)
    for r in sorted(rows, key=lambda r: float(r.get("timestamp", 0))):
        by_entry[r["test_entry_id"]].append(r)
    out = []
    for entry_rows in by_entry.values():
        blocks, block, prev = [], [], None
        for r in entry_rows:
            idx = int(r["task_idx"]) if r.get("task_idx") is not None else None
            if block and idx is not None and prev is not None and idx < prev:
                blocks.append(block)
                block = []
            block.append(r)
            if idx is not None:
                prev = idx
        if block:
            blocks.append(block)
        out.extend(max(blocks, key=lambda b: float(b[-1].get("timestamp", 0))))
    return out


def load_selections(model: str, strategy: str) -> dict[tuple[str, int], list[str]]:
    path = ROOT / RESULT_ROOTS[model] / strategy / "tool_selection_logs.jsonl"
    if not path.exists():
        return {}
    # Reranker strategies log two rows per request (20 embedding candidates, then 5 reranked); keep the last.
    final_per_request: dict[str, dict] = {}
    for r in sorted(read_jsonl(path), key=lambda r: float(r.get("timestamp", 0))):
        final_per_request[r["request_id"]] = r
    sel: dict[tuple[str, int], list[str]] = {}
    for r in latest_block(list(final_per_request.values())):
        names = r.get("selected_tool_names") or (r.get("metadata") or {}).get("selected_tool_names") or []
        key = (r["test_entry_id"], int(r["task_idx"]))
        merged = sel.setdefault(key, [])
        merged.extend(n for n in names if n not in merged)
    return sel


def mark(names: list[str], gold: set[str]) -> str:
    return " | ".join(f"*{n}*" if n in gold else n for n in names)


def fmt_args(args) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    return json.dumps(args, ensure_ascii=False, sort_keys=True)


def fmt_calls(calls: list[dict]) -> str:
    return " | ".join(f"{c['name']}({fmt_args(c['arguments'])})" for c in calls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RESULT_ROOTS), default="120B")
    ap.add_argument("--out", default=None, help="output path prefix (default: hier_unique_<model>)")
    ap.add_argument("--max-answer-chars", type=int, default=600)
    ap.add_argument("--winner", default="hierarchical", choices=STRATEGIES)
    ap.add_argument("--loser", default=None, choices=STRATEGIES,
                    help="if set: tasks where --winner is correct and --loser is wrong (instead of unique wins)")
    args = ap.parse_args()
    out = Path(args.out or f"hier_unique_{args.model}")

    scores = {s: load_scores(args.model, s) for s in STRATEGIES}
    selections = {s: load_selections(args.model, s) for s in STRATEGIES}
    data = {r["id"]: r for r in read_jsonl(ROOT / "data/Wild-Tool-Bench.jsonl")}

    if args.loser:
        def selected_key(k, v):
            return v["correct"] and not scores[args.loser].get(k, {}).get("correct", False)
    else:
        def selected_key(k, v):
            return v["correct"] and not any(scores[c].get(k, {}).get("correct", False) for c in COMPETITORS if c != args.winner)

    keys = sorted(
        (k for k, v in scores[args.winner].items() if selected_key(k, v)),
        key=lambda k: (int(k[0].rsplit("_", 1)[1]), k[1]),
    )

    rows, nested = [], []
    for eid, idx in keys:
        entry = data[eid]
        turn = entry["english_answer_list"][idx]
        gold_calls = [a["action"] for a in turn if a["action"]["name"] != "prepare_to_answer"]
        gold = [a["name"] for a in gold_calls]
        gold_calls = [{"name": a["name"], "arguments": a["arguments"]} for a in gold_calls]
        gold_set = set(gold)
        gold_defs = [t for t in entry["english_tools"] if t["function"]["name"] in gold_set]
        query = entry["english_tasks"][idx] if idx < len(entry["english_tasks"]) else ""
        task_type = (entry.get("english_task_types") or [""])[idx] if idx < len(entry.get("english_task_types") or []) else ""
        task = {"task_id": eid, "turn": idx, "task_type": task_type, "query": query, "gold_tools": gold, "gold_calls": gold_calls, "gold_tool_definitions": gold_defs, "strategies": {}}
        for s in STRATEGIES:
            sc = scores[s].get((eid, idx), {})
            selected = selections[s].get((eid, idx))
            info = {
                "correct": sc.get("correct"),
                "selected_tools": selected,
                "executed_calls": sc.get("executed", []),
                "answer": sc.get("answer", ""),
            }
            task["strategies"][s] = info
            hit = "" if selected is None else f"{len(gold_set & set(selected))}/{len(gold_set)}"
            rows.append({
                "task_id": eid, "turn": idx, "task_type": task_type, "query": query,
                "gold_tools": " | ".join(gold), "gold_calls": fmt_calls(gold_calls), "gold_tool_definitions": json.dumps(gold_defs, ensure_ascii=False), "strategy": s,
                "correct": info["correct"], "gold_in_selected": hit,
                "selected_tools (*=gold)": "" if selected is None else mark(selected, gold_set),
                "executed_tools (*=gold)": mark([c["name"] for c in info["executed_calls"]], gold_set),
                "executed_calls": fmt_calls(info["executed_calls"]),
                "agent_answer": info["answer"][: args.max_answer_chars].replace("\n", " "),
            })
        nested.append(task)

    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2)
    print(f"{len(keys)} tasks -> {out.with_suffix('.csv')}, {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
