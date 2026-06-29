#!/usr/bin/env python3
"""
Convert gold tool set from JSONL format (1 tool per line)
to the original format (array of tools per line).

Original format: Each line is an array of tools
Gold format: Each line is a single tool object

This script reads the gold format and writes it in the original format.
"""

import json
import argparse
from pathlib import Path


def convert_gold_to_original_format(input_file, output_file, tools_per_group=5):
    """
    Convert gold tool format to original format by grouping tools into arrays.
    
    Args:
        input_file: Path to gold tools (1 tool per line)
        output_file: Path to write original format (array of tools per line)
        tools_per_group: Number of tools per array (original used ~5-6)
    """
    
    # Read all tools
    tools = []
    with open(input_file) as f:
        for line in f:
            tool = json.loads(line.strip())
            tools.append(tool)
    
    print(f"Loaded {len(tools)} tools from {input_file}")
    
    # Group tools into arrays and write
    with open(output_file, 'w') as f:
        for i in range(0, len(tools), tools_per_group):
            group = tools[i:i+tools_per_group]
            f.write(json.dumps(group, ensure_ascii=False) + '\n')
    
    num_groups = (len(tools) + tools_per_group - 1) // tools_per_group
    print(f"Wrote {num_groups} tool groups to {output_file}")
    print(f"  ({tools_per_group} tools per group on average)")
    

def main():
    parser = argparse.ArgumentParser(description="Convert gold tool set to original format")
    parser.add_argument('--input', default='analysis_embeddings/tools_en_gold.jsonl',
                       help='Input file in gold format (1 tool per line)')
    parser.add_argument('--output', default='multi-agent-framework/tools/tools_en.jsonl',
                       help='Output file in original format (array of tools per line)')
    parser.add_argument('--tools-per-group', type=int, default=5,
                       help='Number of tools per group (default: 5)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Backup original if it exists
    if output_path.exists():
        backup_path = output_path.parent / f"{output_path.name}.backup"
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"Backed up original to {backup_path}")
    
    convert_gold_to_original_format(input_path, output_path, args.tools_per_group)
    print(f"\n✓ Successfully converted gold format to original format")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
