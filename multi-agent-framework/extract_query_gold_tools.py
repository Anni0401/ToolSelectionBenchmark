"""
Flatten a generate.py `*_train.jsonl` output into per-turn
{"query": ..., "gold_tools": [...]} records, matching the
english_tasks / answer_list convention used in
wild-tool-bench/data/Wild-Tool-Bench.jsonl.

Usage:
    python3 extract_query_gold_tools.py result/<timestamp>_train.jsonl -o queries_gold_tools.jsonl
"""
import argparse
import json
import re

META_TOOL_NAMES = {"prepare_to_answer", "ask_user_for_required_parameters"}
USER_PREFIXES = ("User:", "用户")
PLANNER_PREFIXES = ("Planner:", "Checker_Planner")

JSON_BLOCK_RE = re.compile(r"```json(.+?)```", re.S)


def strip_user_prefix(content):
    for prefix in USER_PREFIXES:
        if content.startswith(prefix):
            return content[len(prefix):].lstrip(":： ").strip()
    return content.strip()


def extract_action_list(content):
    match = JSON_BLOCK_RE.search(content)
    if not match:
        return []
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []
    return obj.get("Action_List", [])


def split_into_turns(messages):
    """Group messages into turns, each starting at a 'User:' message."""
    turns = []
    current = None
    for message in messages:
        content = message["content"]
        if content.startswith(USER_PREFIXES):
            if current is not None:
                turns.append(current)
            current = {"query": strip_user_prefix(content), "actions": []}
        elif current is not None and content.startswith(PLANNER_PREFIXES):
            for action in extract_action_list(content):
                name = action.get("name")
                if name and name not in META_TOOL_NAMES:
                    current["actions"].append({"name": name, "arguments": action.get("arguments", {})})
    if current is not None:
        turns.append(current)
    return turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a *_train.jsonl file produced by generate.py")
    parser.add_argument("-o", "--output", required=True, help="Output jsonl path")
    args = parser.parse_args()

    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            example = json.loads(line)
            for turn in split_into_turns(example["messages"]):
                record = {
                    "query": turn["query"],
                    "gold_tools": [a["name"] for a in turn["actions"]],
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
