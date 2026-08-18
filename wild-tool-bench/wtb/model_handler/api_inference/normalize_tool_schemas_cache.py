#!/usr/bin/env python3
"""
Normalize tool_schemas_cache.jsonl so its JSON key ordering matches the
alphabetically-sorted key ordering used by the real WTB task tools
(Wild-Tool-Bench.jsonl "english_tools"/"tools").

Without this, synthetic distractor tools are trivially distinguishable from
real task tools by key order alone (e.g. "type" before "properties" in
"parameters", vs. WTB's alphabetical "properties" before "type").

List element order (e.g. "required", "enum") is left untouched — only dict
keys are sorted, exactly like WTB's own data.

Usage:
    python normalize_tool_schemas_cache.py [input.jsonl] [-o output.jsonl]

Defaults to normalizing tool_schemas_cache.jsonl in place (writes a .bak
backup first).
"""
import argparse
import json
import os
import shutil
import sys


def normalize(obj):
    """Recursively sort dict keys; leave list order untouched."""
    if isinstance(obj, dict):
        return {key: normalize(obj[key]) for key in sorted(obj.keys())}
    if isinstance(obj, list):
        return [normalize(item) for item in obj]
    return obj


def normalize_file(input_path: str, output_path: str) -> int:
    line_count = 0

    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    normalized_lines = []
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARNING] Skipping invalid JSON on line {line_no}: {exc}", file=sys.stderr)
            continue

        normalized_lines.append(json.dumps(normalize(entry), ensure_ascii=False, sort_keys=True))
        line_count += 1

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(normalized_lines) + "\n")

    return line_count


def main():
    default_path = os.path.join(os.path.dirname(__file__), "tool_schemas_cache.jsonl")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=default_path, help="Path to tool_schemas_cache.jsonl")
    parser.add_argument("-o", "--output", default=None, help="Output path (defaults to overwriting input, with a .bak backup)")
    args = parser.parse_args()

    input_path = args.input
    in_place = args.output is None
    output_path = args.output or input_path

    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if in_place:
        backup_path = input_path + ".bak"
        shutil.copy2(input_path, backup_path)
        print(f"[INFO] Backup written to {backup_path}")

    count = normalize_file(input_path, output_path)
    print(f"[INFO] Normalized {count} tool entries -> {output_path}")


if __name__ == "__main__":
    main()
