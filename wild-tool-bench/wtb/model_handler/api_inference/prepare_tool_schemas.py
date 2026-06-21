#!/usr/bin/env python3
"""
Convert JSONL tools to valid JSON Schema format and cache them.

This script:
1. Loads tools from tools_en.jsonl
2. Validates and fixes schema issues (e.g., "float" → "number")
3. Saves all valid tools to tool_schemas_cache.json
4. Deduplicates by tool name
5. Can be run standalone or imported by langgraph_app.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


def find_tools_file() -> str:
    """Resolve tools_en.jsonl location with 6-point fallback."""
    candidates = [
        # From script location (2 variations)
        os.path.join(os.path.dirname(__file__), "../../../multi-agent-framework/tools/tools_en.jsonl"),
        os.path.join(os.path.dirname(__file__), "../../../../multi-agent-framework/tools/tools_en.jsonl"),
        # From workspace root
        os.path.join(Path.home(), "WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en.jsonl"),
        # From current directory
        os.path.join(os.getcwd(), "multi-agent-framework/tools/tools_en.jsonl"),
        # Absolute path from home
        "/Users/anniherrmann/WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en.jsonl",
        # Try relative from workspace
        "multi-agent-framework/tools/tools_en.jsonl",
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    raise FileNotFoundError(f"Could not find tools_en.jsonl in candidates: {candidates}")


def fix_parameter_type(param_type: str) -> str:
    """Convert invalid JSON Schema types to valid ones."""
    type_mapping = {
        "float": "number",      # float → number
        "int": "integer",       # int → integer
        "bool": "boolean",      # bool → boolean
        "str": "string",        # str → string
        "list": "array",        # list → array
        "dict": "object",       # dict → object
    }
    return type_mapping.get(param_type, param_type)


def fix_parameter_schema(param: Dict[str, Any]) -> Dict[str, Any]:
    """Fix schema issues in a single parameter."""
    if isinstance(param, dict):
        # Fix type field
        if "type" in param:
            param["type"] = fix_parameter_type(param["type"])
        
        # Recursively fix nested properties
        if "properties" in param and isinstance(param["properties"], dict):
            for key, sub_param in param["properties"].items():
                param["properties"][key] = fix_parameter_schema(sub_param)
        
        # Recursively fix array items
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


def load_tools_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load tools from JSONL file (handles both single object/line and array/line).
    
    Args:
        filepath: Path to tools_en.jsonl
        
    Returns:
        List of tools with fixed schemas
    """
    tools = []
    seen_names = set()
    duplicates = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Handle both formats: single object or list of objects
                line_tools = data if isinstance(data, list) else [data]
                
                for tool in line_tools:
                    # Fix schema issues
                    tool = fix_tool_schema(tool)
                    
                    # Deduplicate by tool name
                    tool_name = tool.get("function", {}).get("name", "unknown")
                    if tool_name in seen_names:
                        duplicates += 1
                        continue
                    
                    seen_names.add(tool_name)
                    tools.append(tool)
                    
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error on line {line_num}: {e}")
    
    print(f"Loaded {len(tools)} unique tools (skipped {duplicates} duplicates)")
    return tools


def validate_tool_schema(tool: Dict[str, Any]) -> List[str]:
    """
    Validate a tool's schema and return any issues found.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check structure
    if tool.get("type") != "function":
        errors.append(f"Invalid type: {tool.get('type')} (expected 'function')")
    
    if "function" not in tool:
        errors.append("Missing 'function' field")
        return errors
    
    func = tool["function"]
    
    # Check required fields
    if "name" not in func:
        errors.append("Missing function name")
    if "description" not in func:
        errors.append("Missing function description")
    if "parameters" not in func:
        errors.append("Missing function parameters")
        return errors
    
    # Validate parameters
    params = func["parameters"]
    if params.get("type") != "object":
        errors.append(f"Parameters type must be 'object', got {params.get('type')}")
    
    # Check for invalid JSON Schema types
    invalid_types = ["float", "int", "bool", "str", "list", "dict"]
    
    def check_types(schema: Dict, path: str = ""):
        if isinstance(schema, dict):
            if "type" in schema and schema["type"] in invalid_types:
                errors.append(f"Invalid type '{schema['type']}' at {path or 'root'} (should use: number, integer, boolean, string, array, object)")
            
            if "properties" in schema:
                for key, sub_schema in schema["properties"].items():
                    check_types(sub_schema, f"properties.{key}")
            
            if "items" in schema:
                check_types(schema["items"], f"{path}.items")
    
    check_types(params)
    
    return errors


def main(tools_file: str = None, output_file: str = None):
    """
    Main function to prepare tool schemas.
    
    Args:
        tools_file: Path to tools_en.jsonl (auto-resolved if None)
        output_file: Path to save tool_schemas_cache.json (auto-resolved if None)
    """
    # Resolve file paths
    if tools_file is None:
        tools_file = find_tools_file()
    
    if output_file is None:
        output_file = os.path.join(os.path.dirname(__file__), "tool_schemas_cache.json")
    
    print(f"\n{'='*60}")
    print(f"Tool Schema Preparation")
    print(f"{'='*60}")
    print(f"Tools file: {tools_file}")
    print(f"Output file: {output_file}")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set (needed for embedding if not cached)")
    
    # Load and fix tools
    print(f"\nLoading tools from JSONL...")
    tools = load_tools_from_jsonl(tools_file)
    print(f"Total unique tools: {len(tools)}")
    
    # Validate schemas
    print(f"\nValidating schemas...")
    invalid_count = 0
    for i, tool in enumerate(tools):
        errors = validate_tool_schema(tool)
        if errors:
            invalid_count += 1
            if invalid_count <= 5:  # Show first 5 errors
                tool_name = tool.get("function", {}).get("name", "unknown")
                print(f"  {tool_name}: {errors[0]}")
    
    if invalid_count > 5:
        print(f"  ... and {invalid_count - 5} more invalid schemas")
    elif invalid_count == 0:
        print("  ✓ All schemas valid!")
    
    # Save to cache
    print(f"\nSaving to cache...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    cache_data = {
        "tools": tools,
        "count": len(tools),
        "metadata": {
            "source": "tools_en.jsonl",
            "schema_version": "1.0",
            "note": "All tools converted to valid JSON Schema format"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(tools)} tools to {output_file}")
    
    # Print file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")
    
    print(f"\n{'='*60}\n")
    
    return tools


if __name__ == "__main__":
    import sys
    
    tools_file = None
    output_file = None
    
    # Parse arguments
    if "--tools-file" in sys.argv:
        idx = sys.argv.index("--tools-file")
        if idx + 1 < len(sys.argv):
            tools_file = sys.argv[idx + 1]
    
    if "--output-file" in sys.argv:
        idx = sys.argv.index("--output-file")
        if idx + 1 < len(sys.argv):
            output_file = sys.argv[idx + 1]
    
    main(tools_file, output_file)
