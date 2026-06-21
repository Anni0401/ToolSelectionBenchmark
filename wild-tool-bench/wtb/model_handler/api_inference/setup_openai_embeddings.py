#!/usr/bin/env python3
"""
Setup script to precompute OpenAI embeddings for all tools.

This script should be run once before starting the LangGraph server with embedding mode.
It will embed all tools using OpenAI's text-embedding-3-small model and cache them locally.

Usage:
    python setup_openai_embeddings.py --tools-file path/to/tools_en.jsonl
    
or set environment variables:
    export OPENAI_API_KEY=your_api_key
    python setup_openai_embeddings.py --tools-file multi-agent-framework/tools/tools_en.jsonl

Environment Variables:
    OPENAI_API_KEY: Required. Your OpenAI API key with embedding model access
    OPENAI_BASE_URL: Optional. Custom OpenAI API endpoint (default: https://api.openai.com/v1)
"""

import json
import os
import sys
import argparse
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wtb.model_handler.api_inference.langgraph_app import OpenAIEmbeddingBasedToolSelector


def find_tools_file(relative_path: str) -> str:
    """Find tools file by trying multiple path variations."""
    possible_paths = [
        relative_path,  # As provided
        os.path.abspath(relative_path),  # Absolute version
    ]
    
    # Add variations from current directory going up
    current_dir = os.getcwd()
    for up_levels in range(6):
        base_dir = current_dir
        for _ in range(up_levels):
            base_dir = os.path.dirname(base_dir)
        
        # Try both direct path and multi-agent-framework location
        possible_paths.extend([
            os.path.join(base_dir, "multi-agent-framework/tools/tools_en.jsonl"),
            os.path.join(base_dir, relative_path),
        ])
    
    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    # If not found, print helpful information
    print(f"\n❌ Error: Tools file not found!")
    print(f"\nTried to find: {relative_path}")
    print(f"From current directory: {current_dir}")
    print(f"\nPossible solutions:")
    print(f"  1. Check if file exists at: {relative_path}")
    print(f"  2. Verify you're in the right directory")
    print(f"  3. Try absolute path: --tools-file /full/path/to/tools_en.jsonl")
    print(f"\nOr run from the WildToolBench root directory:")
    print(f"  cd /Users/anniherrmann/WildToolBench/WildToolBench")
    print(f"  python -m wtb.model_handler.api_inference.setup_openai_embeddings \\")
    print(f"    --tools-file multi-agent-framework/tools/tools_en.jsonl")
    
    return None


def load_tools(file_path: str) -> list:
    """Load tools from JSONL file.
    
    Supports both formats:
    - One tool object per line: {"function": {...}, "type": "function"}
    - Multiple tools per line (as array): [{"function": {...}}, {"function": {...}}]
    """
    tools = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # If it's a list, extend; if it's a single tool, append
                if isinstance(data, list):
                    tools.extend(data)
                else:
                    tools.append(data)
    return tools


def main():
    parser = argparse.ArgumentParser(
        description="Setup OpenAI embeddings for tool selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--tools-file",
        type=str,
        default="multi-agent-framework/tools/tools_en.jsonl",
        help="Path to the tools JSONL file (default: multi-agent-framework/tools/tools_en.jsonl)"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Path to save the embeddings cache (default: tool_embeddings_cache.json in script directory)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of OPENAI_API_KEY before processing"
    )
    
    args = parser.parse_args()
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable is not set!")
        print("\nTo set it:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        print("  python setup_openai_embeddings.py --tools-file your_tools.jsonl")
        sys.exit(1)
    
    print(f"✓ OPENAI_API_KEY is set (first 10 chars: {api_key[:10]}...)")
    
    # Check tools file
    tools_file = find_tools_file(args.tools_file)
    if not tools_file:
        sys.exit(1)
    
    print(f"✓ Tools file found: {tools_file}")
    
    # Load tools
    print("\n📦 Loading tools...")
    try:
        tools = load_tools(tools_file)
        print(f"✓ Loaded {len(tools)} tools")
    except Exception as e:
        print(f"❌ Error loading tools: {e}")
        sys.exit(1)
    
    # Create selector and setup embeddings
    print("\n🚀 Starting embedding setup...")
    print(f"   Model: text-embedding-3-small")
    print(f"   Base URL: {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    
    try:
        cache_file = args.cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_embeddings_cache.json"
        )
        selector = OpenAIEmbeddingBasedToolSelector(top_k=5, cache_file=cache_file)
        selector.setup_embeddings(tools)
        
        print("\n✅ Setup completed successfully!")
        print(f"\nCache saved to: {cache_file}")
        print("\nYou can now start the server with:")
        print("  LANGGRAPH_TOOL_SELECTION_MODE=embedding python -m wtb.model_handler.api_inference.langgraph_app")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
