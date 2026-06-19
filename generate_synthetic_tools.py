#!/usr/bin/env python3
"""
Generate synthetic similar tools using HuggingFace Inference API.

Ensures each tool has at least 10 similar tools (>0.75 similarity).
Synthetic tools are created to fill gaps but are marked as invalid solutions.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Try loading from multiple locations
    env_paths = [
        Path.cwd() / ".env",
        Path.cwd() / "wild-tool-bench" / ".env",
        Path(__file__).parent / ".env",
        Path(__file__).parent / "wild-tool-bench" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    print("Error: groq not installed. Install with: pip install groq", file=sys.stderr)
    sys.exit(1)


def test_huggingface_api(api_token, model="llama-3.3-70b-versatile"):
    """Test if Groq API is working"""
    print(f"Testing Groq API with model: {model}")
    test_prompt = "What is 2+2?"
    response = call_huggingface_api(test_prompt, api_token, model=model, max_tokens=50)
    if response:
        print(f"✓ API test passed. Response: {response[:100]}")
        return True
    else:
        print("✗ API test failed")
        return False
    if response:
        print(f"✓ API test passed. Response: {response[:100]}")
        return True
    else:
        print("✗ API test failed")
        return False


def load_tools(path):
    """Load tools from tools_en.jsonl"""
    tools = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, list):
                for tool in item:
                    name = tool.get('function', {}).get('name')
                    if name:
                        tools[name] = tool
            else:
                name = item.get('function', {}).get('name')
                if name:
                    tools[name] = item
    return tools


def load_neighbors(path, similarity_threshold=0.75):
    """Load neighbors from tools_en_tool_neighbors.csv"""
    neighbors = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sim = float(row['similarity'])
            if sim >= similarity_threshold:
                tool = row['tool_name']
                neighbor = row['neighbor_name']
                rank = int(row['rank'])
                neighbors[tool].append({
                    'name': neighbor,
                    'similarity': sim,
                    'rank': rank
                })
    return neighbors


def load_benchmark_tasks(path, tool_names):
    """Load benchmark tasks and map to tools"""
    task_map = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks = item.get('english_tasks', item.get('tasks', []))
            tools = item.get('english_tools', item.get('tools', []))
            
            tool_names_in_record = []
            for tool in tools:
                if isinstance(tool, dict):
                    name = tool.get('function', {}).get('name') or tool.get('name')
                else:
                    name = tool
                if name and name in tool_names:
                    tool_names_in_record.append(name)
            
            for task in tasks:
                task_text = task if isinstance(task, str) else task.get('input', str(task))
                for tool_name in tool_names_in_record:
                    task_map[tool_name].append(task_text)
    return task_map


def extract_tool_description(tool):
    """Extract tool description"""
    func = tool.get('function', {})
    return func.get('description', 'No description available')


def call_huggingface_api(prompt, api_token, model="llama-3.3-70b-versatile", max_tokens=2048):
    """Call Groq API for text generation"""
    try:
        client = Groq(api_key=api_token)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful tool generation expert. Generate valid JSON responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        # Extract text from response
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        
        return None
    except Exception as e:
        print(f"Error calling Groq API with model {model}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def generate_synthetic_tools_for_tool(tool_name, tool_def, neighbors, tasks, api_token, model, num_to_generate=5):
    """Generate synthetic similar tools for a given tool using LLM"""
    description = extract_tool_description(tool_def)
    neighbor_names = [n['name'] for n in neighbors[:5]]
    sample_tasks = tasks[:3] if tasks else []
    
    # Extract parameter structure from original tool for reference
    original_params = tool_def.get('function', {}).get('parameters', {})
    original_param_keys = list(original_params.get('properties', {}).keys())[:3]  # Show first 3 params
    
    prompt = f"""You are a tool generation expert. Generate {num_to_generate} synthetic tool definitions that:

1. Are RELATED to "{tool_name}" in the same domain/category
2. BUT have NOTICEABLY DIFFERENT purposes/functionality from each other and from the original
3. Are NOT valid solutions for the benchmark tasks below

Benchmark tasks that should NOT be solvable by these tools:
{chr(10).join(f'   - {t[:100]}...' if len(t) > 100 else f'   - {t}' for t in sample_tasks)}

Original tool specification:
- Name: {tool_name}
- Description: {description}
- Parameters: {original_param_keys}

Related existing tools (for domain reference only, NOT to copy):
{chr(10).join(f'  - {n}' for n in neighbor_names)}

IMPORTANT: Generate tools in EXACT JSON format matching tools_en.jsonl structure:

{{
  "function": {{
    "name": "syntheticToolName",
    "description": "Detailed description of what the tool does",
    "parameters": {{
      "type": "object",
      "properties": {{
        "param1": {{
          "type": "string",
          "description": "Description of param1"
        }},
        "param2": {{
          "type": "string",
          "description": "Description of param2"
        }}
      }},
      "required": ["param1"]
    }}
  }},
  "type": "function"
}}

Key Requirements:
- DIVERSITY: Make each synthetic tool distinctly different from the others (not just clones)
- DOMAIN: All tools should be in the same domain/category as "{tool_name}" but solve different sub-problems
- PARAMETERS: Each tool should have meaningfully different parameters (not just renamed copies)
- DESCRIPTION: Write natural descriptions of what each tool DOES (don't mention tasks or what it doesn't do)
- NOT A SOLUTION: Design tools that won't solve the benchmark tasks, but don't say that in the description
- Tool names should be realistic and follow camelCase convention
- Output ONLY valid JSON (no explanations, no markdown)
- One complete tool definition per line

Example of what makes tools distinct in the same domain:
If original is 'getUserProfile', synthetic tools could be:
- 'fetchUserSettings' (different scope: settings vs profile)
- 'getUserMetrics' (different data: metrics vs profile)
- 'getUserPermissions' (different focus: permissions vs profile)
NOT just renamed copies like 'getUserProfileData' or 'getProfile'"""

    response = call_huggingface_api(prompt, api_token, model=model, max_tokens=2048)
    if not response:
        return []
    
    synthetic_tools = []
    for line in response.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('```'):
            continue
        try:
            tool_json = json.loads(line)
            # Validate it has required structure: function.name and type
            if 'function' in tool_json and tool_json['function'].get('name'):
                if 'type' not in tool_json:
                    tool_json['type'] = 'function'
                # Remove helper fields like 'reason' that shouldn't be in final output
                if 'reason' in tool_json:
                    del tool_json['reason']
                synthetic_tools.append(tool_json)
        except json.JSONDecodeError as e:
            # Silently skip non-JSON lines (might be explanatory text)
            continue
    
    return synthetic_tools


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic similar tools to ensure benchmark difficulty"
    )
    parser.add_argument(
        "--tools",
        default="multi-agent-framework/tools/tools_en.jsonl",
        help="Path to tools_en.jsonl"
    )
    parser.add_argument(
        "--neighbors",
        default="analysis_embeddings/tools_en_tool_neighbors.csv",
        help="Path to tools_en_tool_neighbors.csv"
    )
    parser.add_argument(
        "--benchmark",
        default="wild-tool-bench/data/Wild-Tool-Bench.jsonl",
        help="Path to Wild-Tool-Bench.jsonl"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for considering tools as 'similar'"
    )
    parser.add_argument(
        "--min-similar-tools",
        type=int,
        default=5,
        help="Minimum number of similar tools to ensure"
    )
    parser.add_argument(
        "--hf-token",
        help="Groq API token (or set GROQ_API_KEY env var)"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_embeddings",
        help="Output directory for synthetic tools"
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Groq model to use"
    )
    parser.add_argument(
        "--max-tools-to-process",
        type=int,
        default=0,
        help="Max tools to process (0 = all)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N tools (for batch processing)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API connection and exit"
    )
    
    args = parser.parse_args()
    
    # Get Groq token
    groq_token = args.hf_token or os.getenv("GROQ_API_KEY")
    if not groq_token:
        print("Error: Groq API token required. Set GROQ_API_KEY env var or use --hf-token", file=sys.stderr)
        sys.exit(1)
    
    # Test API if requested
    if args.test:
        test_huggingface_api(groq_token, model=args.model)
        sys.exit(0)
    
    # Load data
    print("Loading tools...")
    tools = load_tools(args.tools)
    print(f"  Loaded {len(tools)} tools")
    
    print(f"Loading neighbors (threshold={args.similarity_threshold})...")
    neighbors = load_neighbors(args.neighbors, args.similarity_threshold)
    print(f"  Found neighbors for {len(neighbors)} tools")
    
    print("Loading benchmark tasks...")
    task_map = load_benchmark_tasks(args.benchmark, set(tools.keys()))
    print(f"  Mapped {len(task_map)} tools to tasks")
    
    # Analyze gaps
    tools_needing_synthetic = []
    for tool_name in sorted(tools.keys()):
        neighbor_count = len(neighbors.get(tool_name, []))
        if neighbor_count < args.min_similar_tools:
            tools_needing_synthetic.append((tool_name, neighbor_count))
    
    print(f"\nTools needing synthetic candidates: {len(tools_needing_synthetic)}")
    print(f"Total gap: {sum(max(0, args.min_similar_tools - count) for _, count in tools_needing_synthetic)} tools")
    
    # Generate synthetic tools
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    synthetic_output = output_dir / "tools_en_synthetic_candidates.jsonl"
    metadata_output = output_dir / "tools_en_synthetic_metadata.csv"
    
    synthetic_tools_all = []
    metadata_rows = []
    
    # Apply offset for batch processing
    tools_to_process = tools_needing_synthetic[args.offset:]
    if args.offset > 0:
        print(f"Skipping first {args.offset} tools (offset={args.offset})")
    
    processed = 0
    for idx, (tool_name, current_count) in enumerate(tools_to_process):
        if args.max_tools_to_process > 0 and processed >= args.max_tools_to_process:
            print(f"\nReached max tools to process ({args.max_tools_to_process})")
            break
        
        num_needed = args.min_similar_tools - current_count
        absolute_idx = args.offset + idx + 1
        total_count = len(tools_needing_synthetic)
        print(f"\n[{absolute_idx}/{total_count}] {tool_name}: {current_count}/{args.min_similar_tools} → generating {num_needed}")
        
        tool_def = tools[tool_name]
        tool_neighbors = neighbors.get(tool_name, [])
        tool_tasks = task_map.get(tool_name, [])
        
        synthetic = generate_synthetic_tools_for_tool(
            tool_name, tool_def, tool_neighbors, tool_tasks, groq_token, args.model, num_to_generate=num_needed
        )
        
        if synthetic:
            print(f"  Generated {len(synthetic)} synthetic tools")
            for syn_tool in synthetic:
                syn_tool["_source_tool"] = tool_name
                syn_tool["_synthetic"] = True
                synthetic_tools_all.append(syn_tool)
                
                # Extract parameters info
                func = syn_tool.get('function', {})
                params = func.get('parameters', {})
                param_names = list(params.get('properties', {}).keys())
                required_params = params.get('required', [])
                
                metadata_rows.append({
                    "source_tool": tool_name,
                    "synthetic_tool_name": func.get('name', 'unknown'),
                    "synthetic_tool_description": func.get('description', ''),
                    "parameters": ','.join(param_names) if param_names else '',
                    "required_parameters": ','.join(required_params) if required_params else '',
                })
        else:
            print(f"  Failed to generate synthetic tools")
        
        processed += 1
    
    # Write outputs (append mode for batch processing)
    print(f"\nWriting outputs...")
    mode = 'a' if args.offset > 0 else 'w'  # Append if offset > 0 (batch mode), else overwrite
    with open(synthetic_output, mode, encoding='utf-8') as f:
        for tool in synthetic_tools_all:
            f.write(json.dumps(tool, ensure_ascii=False) + '\n')
    print(f"  Wrote {len(synthetic_tools_all)} synthetic tools to {synthetic_output} ({mode}ode)")
    
    # Write metadata as JSON instead of CSV (append for batch processing)
    metadata_output = output_dir / "tools_en_synthetic_metadata.json"
    if args.offset > 0:
        # Append to existing metadata
        try:
            with open(metadata_output, 'r', encoding='utf-8') as f:
                existing_metadata = json.load(f)
            existing_metadata.extend(metadata_rows)
            metadata_rows = existing_metadata
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(metadata_rows, f, ensure_ascii=False, indent=2)
    print(f"  Wrote metadata to {metadata_output}")
    
    print(f"\nDone! Generated {len(synthetic_tools_all)} synthetic tools")


if __name__ == "__main__":
    main()
