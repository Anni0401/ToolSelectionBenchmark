#!/usr/bin/env python3
"""
Quality check: Analyze all tools (original + synthetic) to find the 5 most similar tools for each,
including cross-type similarities (original to synthetic and vice versa).
This helps identify if any synthetic tool is too similar to an original tool.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_json_or_jsonl(path):
    """Load JSON or JSONL file."""
    content = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if isinstance(item, list):
                data.extend(item)
            else:
                data.append(item)
    if isinstance(data, list) and data and isinstance(data[0], list):
        flattened = []
        for item in data:
            if isinstance(item, list):
                flattened.extend(item)
            else:
                flattened.append(item)
        data = flattened
    return data


def normalize_tool_text(tool):
    """Convert tool to normalized text representation."""
    function = tool.get("function", {})
    name = function.get("name", "")
    description = function.get("description", "")
    params = function.get("parameters", {})
    props = params.get("properties", {})
    required = params.get("required", [])

    lines = [f"Tool: {name}", f"Description: {description}"]
    if props:
        lines.append("Parameters:")
        for prop_name, prop_schema in sorted(props.items()):
            prop_type = prop_schema.get("type", "")
            prop_desc = prop_schema.get("description", "")
            enum = prop_schema.get("enum")
            if enum:
                prop_desc += f" Options={enum}."
            lines.append(f"- {prop_name} ({prop_type}): {prop_desc}")
    if required:
        lines.append(f"Required: {', '.join(required)}")
    return "\n".join(lines)


def extract_tool_name(tool):
    """Extract tool name from tool object."""
    if isinstance(tool, str):
        return tool
    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            return tool["function"].get("name") or tool.get("name")
        return tool.get("name")
    return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_embedding_local(model_name, inputs, batch_size=64):
    """Generate embeddings using local sentence-transformers model."""
    if SentenceTransformer is None:
        raise ImportError(
            "Please install sentence-transformers (pip install sentence-transformers) to use local embeddings."
        )
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i : i + batch_size]
        encoded = model.encode(chunk, show_progress_bar=True, convert_to_tensor=False)
        if hasattr(encoded, "tolist"):
            embeddings.extend(encoded.tolist())
        else:
            embeddings.extend([list(vec) for vec in encoded])
    return embeddings


def build_all_tool_neighbors(tool_names, tool_types, embeddings, top_k=5):
    """
    Build nearest neighbors for all tools (original + synthetic).
    For each tool, find its top_k most similar tools (regardless of type).
    """
    n = len(tool_names)
    nearest = []
    
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sims.append((j, sim))
        
        # Sort by similarity descending and take top K
        sims.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (j, score) in enumerate(sims[:top_k], start=1):
            nearest.append({
                "tool_name": tool_names[i],
                "tool_type": tool_types[i],
                "neighbor_name": tool_names[j],
                "neighbor_type": tool_types[j],
                "similarity": round(float(score), 6),
                "rank": rank,
            })
    
    return nearest


def write_csv(out_path, fieldnames, rows):
    """Write CSV file."""
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all tools (original + synthetic) for quality check"
    )
    parser.add_argument(
        "--originals",
        default="multi-agent-framework/tools/tools_en.jsonl",
        help="Path to original tools_en.jsonl",
    )
    parser.add_argument(
        "--synthetics",
        default="analysis_embeddings/tools_en_synthetic_candidates.jsonl",
        help="Path to synthetic tools JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_embeddings",
        help="Directory where analysis outputs are written",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to use for local sentence-transformers",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to include per tool",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding requests",
    )
    args = parser.parse_args()

    # Load original tools
    print("Loading original tools...")
    original_data = load_json_or_jsonl(args.originals)
    original_tools = []
    original_names = []
    original_map = {}
    
    seen = set()
    for tool in original_data:
        name = extract_tool_name(tool)
        if not name or name in seen:
            continue
        seen.add(name)
        original_tools.append(tool)
        original_names.append(name)
        original_map[name] = tool
    
    print(f"Loaded {len(original_tools)} unique original tools")

    # Load synthetic tools
    print("Loading synthetic tools...")
    synthetic_data = load_json_or_jsonl(args.synthetics)
    synthetic_tools = []
    synthetic_names = []
    synthetic_map = {}
    
    seen_synth = set()
    for tool in synthetic_data:
        name = extract_tool_name(tool)
        if not name or name in seen_synth:
            continue
        seen_synth.add(name)
        synthetic_tools.append(tool)
        synthetic_names.append(name)
        synthetic_map[name] = tool
    
    print(f"Loaded {len(synthetic_tools)} unique synthetic tools")

    # Combine all tools
    all_tools = original_tools + synthetic_tools
    all_names = original_names + synthetic_names
    all_types = ["original"] * len(original_tools) + ["synthetic"] * len(synthetic_tools)
    
    print(f"Total tools: {len(all_tools)}")

    # Generate texts for embedding
    print("Generating tool texts for embedding...")
    all_texts = [normalize_tool_text(tool) for tool in all_tools]

    # Generate embeddings
    print(f"Generating embeddings using {args.model}...")
    embeddings = batch_embedding_local(args.model, all_texts, batch_size=args.batch_size)
    print(f"Generated {len(embeddings)} embeddings")

    # Build nearest neighbors for all tools
    print("Building nearest neighbors...")
    all_neighbors = build_all_tool_neighbors(all_names, all_types, embeddings, args.top_k)
    
    # Count cross-type high similarities (warning indicator)
    cross_type_high_sim = 0
    for row in all_neighbors:
        if (row["tool_type"] != row["neighbor_type"] and 
            row["similarity"] > 0.95 and 
            row["rank"] <= 2):  # Only count if in top 2
            cross_type_high_sim += 1
    
    print(f"Found {cross_type_high_sim} cross-type high similarity (>0.95) matches in top 2")

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write neighbors CSV with tool types
    neighbors_path = out_dir / "tools_en_all_neighbors.csv"
    write_csv(
        neighbors_path,
        ["tool_name", "tool_type", "neighbor_name", "neighbor_type", "similarity", "rank"],
        all_neighbors,
    )
    print(f"Wrote neighbors to {neighbors_path}")

    # Write quality report
    report_path = out_dir / "quality_report.json"
    report = {
        "total_tools": len(all_tools),
        "original_tools": len(original_tools),
        "synthetic_tools": len(synthetic_tools),
        "neighbors_analyzed": len(all_neighbors),
        "high_similarity_warnings": cross_type_high_sim,
        "analysis": {
            "high_similarity_synthetic_to_original": []
        }
    }
    
    # Find synthetic tools with high similarity to originals
    synth_high_sims = defaultdict(list)
    for row in all_neighbors:
        if (row["tool_type"] == "synthetic" and 
            row["neighbor_type"] == "original" and 
            row["similarity"] > 0.95 and 
            row["rank"] <= 3):
            synth_high_sims[row["tool_name"]].append({
                "similar_to": row["neighbor_name"],
                "similarity": row["similarity"],
                "rank": row["rank"]
            })
    
    report["analysis"]["high_similarity_synthetic_to_original"] = [
        {
            "synthetic_tool": synth,
            "similar_originals": sims
        }
        for synth, sims in sorted(synth_high_sims.items())
    ]
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Wrote quality report to {report_path}")
    print("\nQuality Check Summary:")
    print(f"  Total tools analyzed: {report['total_tools']}")
    print(f"  Original tools: {report['original_tools']}")
    print(f"  Synthetic tools: {report['synthetic_tools']}")
    print(f"  High similarity warnings (>0.95 in top 2): {report['high_similarity_warnings']}")
    print(f"  Problematic synthetics (>0.95 to originals): {len(synth_high_sims)}")


if __name__ == "__main__":
    main()
