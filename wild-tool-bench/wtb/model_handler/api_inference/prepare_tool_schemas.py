#!/usr/bin/env python3
"""
Convert JSONL tools to valid JSON Schema format and cache them.

Policy:
1. Load ORIGINAL tools from tools_en_normalized.jsonl (no dedup).
2. Load SYNTHETIC tools from tools_en_final.jsonl (_synthetic only).
3. Deduplicate synthetic tools by function name.
4. Remove synthetic tools whose names overlap with original tools.
5. Fix schema types (e.g., "float" -> "number") and validate.
6. Save to tool_schemas_cache.json and tool_schemas_cache.jsonl.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _resolve_file(candidates: List[str], label: str) -> str:
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"Could not find {label} in candidates: {candidates}")


def find_original_tools_file() -> str:
    """Resolve tools_en_normalized.jsonl with fallback paths."""
    return _resolve_file(
        [
            os.path.join(os.path.dirname(__file__), "../../../multi-agent-framework/tools/tools_en_normalized.jsonl"),
            os.path.join(os.path.dirname(__file__), "../../../../multi-agent-framework/tools/tools_en_normalized.jsonl"),
            os.path.join(Path.home(), "WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en_normalized.jsonl"),
            os.path.join(os.getcwd(), "multi-agent-framework/tools/tools_en_normalized.jsonl"),
            "/Users/anniherrmann/WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en_normalized.jsonl",
            "multi-agent-framework/tools/tools_en_normalized.jsonl",
        ],
        "tools_en_normalized.jsonl",
    )


def find_synthetic_tools_file() -> str:
    """Resolve tools_en_final.jsonl with fallback paths."""
    return _resolve_file(
        [
            os.path.join(os.path.dirname(__file__), "../../../multi-agent-framework/tools/tools_en_final.jsonl"),
            os.path.join(os.path.dirname(__file__), "../../../../multi-agent-framework/tools/tools_en_final.jsonl"),
            os.path.join(Path.home(), "WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en_final.jsonl"),
            os.path.join(os.getcwd(), "multi-agent-framework/tools/tools_en_final.jsonl"),
            "/Users/anniherrmann/WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en_final.jsonl",
            "multi-agent-framework/tools/tools_en_final.jsonl",
        ],
        "tools_en_final.jsonl",
    )


def fix_parameter_type(param_type: str) -> str:
    """Convert invalid JSON Schema types to valid ones."""
    type_mapping = {
        "float": "number",
        "int": "integer",
        "bool": "boolean",
        "str": "string",
        "list": "array",
        "dict": "object",
    }
    return type_mapping.get(param_type, param_type)


def fix_parameter_schema(param: Dict[str, Any]) -> Dict[str, Any]:
    """Fix schema issues recursively in a parameter object."""
    if isinstance(param, dict):
        if "type" in param:
            param["type"] = fix_parameter_type(param["type"])

        if "properties" in param and isinstance(param["properties"], dict):
            for key, sub_param in param["properties"].items():
                param["properties"][key] = fix_parameter_schema(sub_param)

        if "items" in param:
            param["items"] = fix_parameter_schema(param["items"])

    return param


def fix_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Fix schema issues in a complete tool."""
    if tool.get("type") == "function" and "function" in tool:
        func = tool["function"]
        if "parameters" in func:
            func["parameters"] = fix_parameter_schema(func["parameters"])
    return tool


def load_tools_from_jsonl(filepath: str, synthetic_only: bool = False) -> List[Dict[str, Any]]:
    """Load and schema-fix tools from JSONL.

    Handles both formats per line: a single tool object or a list of tool objects.
    """
    tools: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                line_tools = data if isinstance(data, list) else [data]

                for tool in line_tools:
                    if synthetic_only and not tool.get("_synthetic", False):
                        continue
                    tools.append(fix_tool_schema(tool))
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error on line {line_num}: {e}")

    label = "synthetic" if synthetic_only else "all"
    print(f"Loaded {len(tools)} {label} tools from {os.path.basename(filepath)}")
    return tools


def _tool_name(tool: Dict[str, Any]) -> str:
    return tool.get("function", {}).get("name", "")


def merge_original_and_synthetic_tools(
    original_tools: List[Dict[str, Any]],
    synthetic_tools: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Apply merge policy and return merged list plus statistics."""
    original_names = {_tool_name(tool) for tool in original_tools if _tool_name(tool)}

    merged_tools = list(original_tools)
    seen_synthetic_names = set()
    kept_synthetic = 0
    dropped_overlap = 0
    dropped_synthetic_dedup = 0

    for tool in synthetic_tools:
        name = _tool_name(tool)
        if not name:
            dropped_synthetic_dedup += 1
            continue

        if name in original_names:
            dropped_overlap += 1
            continue

        if name in seen_synthetic_names:
            dropped_synthetic_dedup += 1
            continue

        seen_synthetic_names.add(name)
        merged_tools.append(tool)
        kept_synthetic += 1

    stats = {
        "original_total": len(original_tools),
        "synthetic_total": len(synthetic_tools),
        "synthetic_kept": kept_synthetic,
        "synthetic_dropped_overlap_with_original": dropped_overlap,
        "synthetic_dropped_name_dedup": dropped_synthetic_dedup,
        "final_total": len(merged_tools),
    }
    return merged_tools, stats


def validate_tool_schema(tool: Dict[str, Any]) -> List[str]:
    """Validate a tool schema and return a list of errors."""
    errors: List[str] = []

    if tool.get("type") != "function":
        errors.append(f"Invalid type: {tool.get('type')} (expected 'function')")

    if "function" not in tool:
        errors.append("Missing 'function' field")
        return errors

    func = tool["function"]
    if "name" not in func:
        errors.append("Missing function name")
    if "description" not in func:
        errors.append("Missing function description")
    if "parameters" not in func:
        errors.append("Missing function parameters")
        return errors

    params = func["parameters"]
    if params.get("type") != "object":
        errors.append(f"Parameters type must be 'object', got {params.get('type')}")

    invalid_types = ["float", "int", "bool", "str", "list", "dict"]

    def check_types(schema: Dict[str, Any], path: str = ""):
        if isinstance(schema, dict):
            if "type" in schema and schema["type"] in invalid_types:
                errors.append(
                    f"Invalid type '{schema['type']}' at {path or 'root'} "
                    "(should use: number, integer, boolean, string, array, object)"
                )

            if "properties" in schema:
                for key, sub_schema in schema["properties"].items():
                    check_types(sub_schema, f"properties.{key}")

            if "items" in schema:
                check_types(schema["items"], f"{path}.items")

    check_types(params)
    return errors


def main(
    tools_file: str = None,
    output_file: str = None,
    synthetic_file: str = None,
    output_jsonl_file: str = None,
):
    """Prepare merged tool cache according to the current policy."""
    if tools_file is None:
        tools_file = find_original_tools_file()

    if synthetic_file is None:
        synthetic_file = find_synthetic_tools_file()

    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "tool_schemas_cache.json")

    if output_jsonl_file is None:
        output_jsonl_file = os.path.join(os.path.dirname(__file__), "tool_schemas_cache.jsonl")

    print(f"\n{'=' * 60}")
    print("Tool Schema Preparation")
    print(f"{'=' * 60}")
    print(f"Original tools file: {tools_file}")
    print(f"Synthetic tools file: {synthetic_file}")
    print(f"Output JSON cache: {output_file}")
    print(f"Output JSONL cache: {output_jsonl_file}")

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set (needed for embedding if not cached)")

    print("\nLoading original tools from JSONL...")
    original_tools = load_tools_from_jsonl(tools_file, synthetic_only=False)

    print("\nLoading synthetic tools from JSONL...")
    synthetic_tools = load_tools_from_jsonl(synthetic_file, synthetic_only=True)

    print("\nMerging original + synthetic tools...")
    tools, merge_stats = merge_original_and_synthetic_tools(original_tools, synthetic_tools)
    print(f"  Originals kept (no dedup): {merge_stats['original_total']}")
    print(f"  Synthetics total: {merge_stats['synthetic_total']}")
    print(f"  Synthetics kept: {merge_stats['synthetic_kept']}")
    print(f"  Synthetics dropped (overlap with originals): {merge_stats['synthetic_dropped_overlap_with_original']}")
    print(f"  Synthetics dropped (synthetic name dedup): {merge_stats['synthetic_dropped_name_dedup']}")
    print(f"  Final tools count: {merge_stats['final_total']}")

    print("\nValidating schemas...")
    invalid_count = 0
    for tool in tools:
        errors = validate_tool_schema(tool)
        if errors:
            invalid_count += 1
            if invalid_count <= 5:
                tool_name = tool.get("function", {}).get("name", "unknown")
                print(f"  {tool_name}: {errors[0]}")

    if invalid_count > 5:
        print(f"  ... and {invalid_count - 5} more invalid schemas")
    elif invalid_count == 0:
        print("  ✓ All schemas valid!")

    print("\nSaving to cache...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    cache_data = {
        "tools": tools,
        "count": len(tools),
        "metadata": {
            "source_original": os.path.basename(tools_file),
            "source_synthetic": os.path.basename(synthetic_file),
            "schema_version": "1.0",
            "note": "Original tools unchanged; synthetic tools deduped by name; overlap with original names removed",
            "merge_stats": merge_stats,
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    with open(output_jsonl_file, "w", encoding="utf-8") as f:
        for tool in tools:
            f.write(json.dumps(tool, ensure_ascii=False) + "\n")

    print(f"✓ Saved {len(tools)} tools to {output_file}")
    print(f"✓ Saved {len(tools)} tools to {output_jsonl_file}")

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")

    print(f"\n{'=' * 60}\n")
    return tools


if __name__ == "__main__":
    import sys

    tools_file = None
    synthetic_file = None
    output_file = None
    output_jsonl_file = None

    if "--tools-file" in sys.argv:
        idx = sys.argv.index("--tools-file")
        if idx + 1 < len(sys.argv):
            tools_file = sys.argv[idx + 1]

    if "--synthetic-file" in sys.argv:
        idx = sys.argv.index("--synthetic-file")
        if idx + 1 < len(sys.argv):
            synthetic_file = sys.argv[idx + 1]

    if "--output-file" in sys.argv:
        idx = sys.argv.index("--output-file")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]

    if "--output-jsonl-file" in sys.argv:
        idx = sys.argv.index("--output-jsonl-file")
        if idx + 1 < len(sys.argv):
            output_jsonl_file = sys.argv[idx + 1]

    main(tools_file, output_file, synthetic_file, output_jsonl_file)
