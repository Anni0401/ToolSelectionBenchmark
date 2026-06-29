#!/usr/bin/env python3
"""
Pipeline:
1. Deduplicate original and synthetic tools
2. Generate embeddings for deduplicated tools
3. Build neighbors CSV with correct similarity scores
4. Apply filtering rules to build gold tool set
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


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


def deduplicate_by_name(tools):
    """Keep only first occurrence of each tool name."""
    seen = set()
    unique = []
    for tool in tools:
        name = tool['function']['name']
        if name not in seen:
            seen.add(name)
            unique.append(tool)
    return unique


def save_deduplicated(originals, synthetics, output_dir):
    """Save deduplicated tools."""
    output_dir = Path(output_dir)
    
    orig_file = output_dir / 'tools_en_original_dedup.jsonl'
    with open(orig_file, 'w') as f:
        for tool in originals:
            f.write(json.dumps(tool) + '\n')
    
    synth_file = output_dir / 'tools_en_synthetic_dedup.jsonl'
    with open(synth_file, 'w') as f:
        for tool in synthetics:
            f.write(json.dumps(tool) + '\n')
    
    print(f"Saved {len(originals)} original tools to {orig_file}")
    print(f"Saved {len(synthetics)} synthetic tools to {synth_file}")


def batch_embedding_local(model_name, inputs, batch_size=64):
    """Generate embeddings in batches."""
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
    """Compute top-k neighbors for each tool."""
    embeddings = np.array(embeddings)
    
    neighbors_list = []
    for i, tool_name in enumerate(tool_names):
        tool_type = tool_types[i]
        tool_embedding = embeddings[i]
        
        # Compute similarity to all other tools
        similarities = []
        for j, other_name in enumerate(tool_names):
            if i != j:
                other_embedding = embeddings[j]
                sim = np.dot(tool_embedding, other_embedding) / (
                    np.linalg.norm(tool_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((j, other_name, tool_types[j], sim))
        
        # Sort by similarity descending and take top-k
        similarities.sort(key=lambda x: x[3], reverse=True)
        
        for rank, (j, other_name, other_type, sim) in enumerate(similarities[:top_k], 1):
            neighbors_list.append({
                'tool_name': tool_name,
                'tool_type': tool_type,
                'neighbor_name': other_name,
                'neighbor_type': other_type,
                'similarity': sim,
                'rank': rank
            })
    
    return neighbors_list


def get_synthetic_parent(tool):
    """Extract parent original tool name if available."""
    if tool is None:
        return None
    if 'created_from' in tool:
        return tool['created_from']
    if 'parent' in tool:
        return tool['parent']
    if 'original_tool' in tool:
        return tool['original_tool']
    return None


def write_csv(data, filepath):
    """Write list of dicts to CSV."""
    if not data:
        return
    
    keys = data[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def filter_gold_tools(originals, synthetics, neighbors_list):
    """
    Apply filtering rules:
    1. Remove synthetic tools with exact name match to originals
    2. Remove synthetic tools >0.65 similar to unrelated originals
    3. Remove synthetic tool pairs >0.75 similar from different parents
    """
    
    # Build maps
    orig_names = {t['function']['name']: t for t in originals}
    synth_map = {t['function']['name']: t for t in synthetics}
    
    print(f"\nApplying filtering rules...")
    print(f"Starting: {len(originals)} originals, {len(synthetics)} synthetics")
    
    # Rule 1: Remove exact name matches
    exact_matches = set()
    for synth_name in synth_map:
        if synth_name in orig_names:
            exact_matches.add(synth_name)
    
    print(f"Rule 1 - Exact name matches: {len(exact_matches)} removed")
    for name in exact_matches:
        del synth_map[name]
    
    # Build neighbors lookup
    neighbors_by_tool = defaultdict(list)
    for row in neighbors_list:
        tool_name = row['tool_name']
        neighbors_by_tool[tool_name].append(row)
    
    # Rule 2: Remove synthetic tools >0.65 similar to unrelated originals
    rule2_removals = set()
    for synth_name in list(synth_map.keys()):
        parent = get_synthetic_parent(synth_map[synth_name])
        neighbors = neighbors_by_tool.get(synth_name, [])
        
        for neighbor in neighbors:
            if neighbor['neighbor_type'] == 'original' and neighbor['neighbor_name'] != parent and neighbor['similarity'] > 0.65:
                rule2_removals.add(synth_name)
                break
    
    print(f"Rule 2 - High similarity to unrelated originals (>0.65): {len(rule2_removals)} removed")
    for name in rule2_removals:
        synth_map.pop(name, None)
    
    # Rule 3: Remove synthetic tool pairs >0.75 similar from different parents
    rule3_removals = set()
    
    for synth_name in list(synth_map.keys()):
        if synth_name in rule3_removals:
            continue
        
        parent1 = get_synthetic_parent(synth_map[synth_name])
        neighbors = neighbors_by_tool.get(synth_name, [])
        
        for neighbor in neighbors:
            synth2_name = neighbor['neighbor_name']
            
            # Skip if already removed or not in synth_map
            if synth2_name not in synth_map:
                continue
            
            if neighbor['neighbor_type'] == 'synthetic' and neighbor['similarity'] > 0.75:
                parent2 = get_synthetic_parent(synth_map.get(synth2_name))
                
                if parent1 != parent2:
                    rule3_removals.add(synth_name)
                    rule3_removals.add(synth2_name)
                    break
    
    print(f"Rule 3 - High similarity pairs from different parents (>0.75): {len(rule3_removals)} removed")
    for name in rule3_removals:
        synth_map.pop(name, None)
    
    # Final sets
    final_originals = list(orig_names.values())
    final_synthetics = list(synth_map.values())
    final_all = final_originals + final_synthetics
    
    print(f"\nFinal counts:")
    print(f"  Original: {len(final_originals)}")
    print(f"  Synthetic: {len(final_synthetics)}")
    print(f"  Total: {len(final_all)}")
    
    return final_originals, final_synthetics, final_all


def main():
    parser = argparse.ArgumentParser(description="Rebuild embeddings with deduplicated tools")
    parser.add_argument('--originals', default='multi-agent-framework/tools/tools_en.jsonl')
    parser.add_argument('--synthetics', default='analysis_embeddings/tools_en_synthetic_candidates.jsonl')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--output-dir', default='analysis_embeddings')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load and deduplicate
    print("Loading original tools...")
    originals = load_json_or_jsonl(args.originals)
    print(f"  Loaded: {len(originals)}")
    
    print("Loading synthetic tools...")
    synthetics = load_json_or_jsonl(args.synthetics)
    print(f"  Loaded: {len(synthetics)}")
    
    print("\nDeduplicating by tool name...")
    originals = deduplicate_by_name(originals)
    synthetics = deduplicate_by_name(synthetics)
    print(f"  Original after dedup: {len(originals)}")
    print(f"  Synthetic after dedup: {len(synthetics)}")
    
    # Save deduplicated
    save_deduplicated(originals, synthetics, output_dir)
    
    # Generate embeddings
    print("\nGenerating embeddings for original tools...")
    original_texts = [
        f"{t['function']['name']} {t['function'].get('description', '')}"
        for t in originals
    ]
    original_embeddings = batch_embedding_local(args.model, original_texts, args.batch_size)
    
    print("Generating embeddings for synthetic tools...")
    synthetic_texts = [
        f"{t['function']['name']} {t['function'].get('description', '')}"
        for t in synthetics
    ]
    synthetic_embeddings = batch_embedding_local(args.model, synthetic_texts, args.batch_size)
    
    # Build combined neighbors list
    print("\nBuilding neighbors list...")
    all_tool_names = [t['function']['name'] for t in originals + synthetics]
    all_tool_types = ['original'] * len(originals) + ['synthetic'] * len(synthetics)
    all_embeddings = original_embeddings + synthetic_embeddings
    
    neighbors_list = build_all_tool_neighbors(all_tool_names, all_tool_types, all_embeddings, args.top_k)
    
    # Save neighbors CSV
    neighbors_csv = output_dir / 'tools_en_all_neighbors_dedup.csv'
    write_csv(neighbors_list, neighbors_csv)
    print(f"Saved {len(neighbors_list)} neighbor relations to {neighbors_csv}")
    
    # Apply filtering rules
    final_originals, final_synthetics, final_all = filter_gold_tools(originals, synthetics, neighbors_list)
    
    # Save gold tool set
    gold_output = output_dir / 'tools_en_gold.jsonl'
    with open(gold_output, 'w') as f:
        for tool in final_all:
            f.write(json.dumps(tool) + '\n')
    print(f"\nSaved gold tool set ({len(final_all)} tools) to {gold_output}")
    
    # Save report
    report = {
        'original_before_dedup': len(load_json_or_jsonl(args.originals)),
        'synthetic_before_dedup': len(load_json_or_jsonl(args.synthetics)),
        'original_after_dedup': len(originals),
        'synthetic_after_dedup': len(synthetics),
        'final_original_count': len(final_originals),
        'final_synthetic_count': len(final_synthetics),
        'final_total_count': len(final_all),
    }
    
    report_output = output_dir / 'gold_tool_set_report.json'
    with open(report_output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_output}")


if __name__ == '__main__':
    main()
