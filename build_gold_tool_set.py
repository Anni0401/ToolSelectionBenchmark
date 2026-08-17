#!/usr/bin/env python3
"""
Build gold tool set by:
1. Deduplicating originals by normalized_tool_id (keep first occurrence), synthetics by name
2. Cap synthetic tools to at most 2 per source normalized_tool_id (keep first occurrences)
3. Remove synthetic tools with exact name match to originals
4. Remove synthetic tools >0.75 similar to unrelated originals
5. Remove synthetic tool pairs >0.9 similar from different parents

Originals are expected to carry a `normalized_tool_id` field (see
tools_en_with_normalized_tool_ids.jsonl) and synthetics carry `_source_tool_id`
pointing at the normalized_tool_id of the original they were generated from.
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np


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


def _dedup_key(tool):
    f = tool.get('function', {})
    name = f.get('name', '')
    desc = ' '.join((f.get('description') or '').split()).strip().lower()
    return (name, desc)


def deduplicate_by_name(tools):
    """Keep first occurrence by (name, normalized description)."""
    seen = set()
    unique = []
    for tool in tools:
        key = _dedup_key(tool)
        if key not in seen:
            seen.add(key)
            unique.append(tool)
    return unique


def deduplicate_by_normalized_id(tools):
    """Keep first occurrence per normalized_tool_id."""
    seen = set()
    unique = []
    for tool in tools:
        tool_id = tool.get('normalized_tool_id')
        if tool_id is None or tool_id in seen:
            continue
        seen.add(tool_id)
        unique.append(tool)
    return unique


def get_synthetic_parent_id(tool):
    """Extract the normalized_tool_id of the original a synthetic tool was created from."""
    return tool.get('_source_tool_id')


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


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def filter_gold_tools(originals, synthetics, neighbors_csv, model_name, output_dir="analysis_embeddings"):
    """
    Apply filtering rules:
    1. Cap synthetic tools to at most 2 per source normalized_tool_id (keep first occurrences)
    2. Remove synthetic tools with exact name match to originals
    3. Remove synthetic tools >0.75 similar to unrelated originals
    4. Remove synthetic tool pairs >0.9 similar from different parents
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Build maps
    orig_names = {t['function']['name']: t for t in originals}
    orig_id_by_name = {t['function']['name']: t.get('normalized_tool_id') for t in originals}
    synth_map = {t['function']['name']: t for t in synthetics}
    
    print(f"Starting with {len(originals)} originals and {len(synthetics)} synthetics")
    
    # Rule 1: Cap synthetics to at most 2 per source normalized_tool_id (keep first occurrences)
    max_synthetics_per_id = 2
    per_id_counts = defaultdict(int)
    rule1_cap_removals = set()
    for synth_name, synth_tool in synth_map.items():
        parent_id = get_synthetic_parent_id(synth_tool)
        per_id_counts[parent_id] += 1
        if per_id_counts[parent_id] > max_synthetics_per_id:
            rule1_cap_removals.add(synth_name)
    
    print(f"Rule 1 - Capped to {max_synthetics_per_id} synthetics per source ID: {len(rule1_cap_removals)}")
    for name in rule1_cap_removals:
        synth_map.pop(name, None)
    
    # Rule 2: Remove exact name matches
    exact_matches = set()
    for synth_name in synth_map:
        if synth_name in orig_names:
            exact_matches.add(synth_name)
    
    print(f"Rule 2 - Exact name matches to remove: {len(exact_matches)}")
    for name in exact_matches:
        del synth_map[name]
    
    # Load neighbors for similarity-based filtering
    print("\nLoading similarity data...")
    neighbors_by_tool = defaultdict(list)
    
    with open(neighbors_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tool_name = row['tool_name']
            neighbor_name = row['neighbor_name']
            sim = float(row['similarity'])
            neighbors_by_tool[tool_name].append({
                'name': neighbor_name,
                'type': row['neighbor_type'],
                'similarity': sim
            })
    
    # Rule 3: Remove synthetic tools >0.75 similar to unrelated originals
    rule3_removals = set()
    for synth_name in list(synth_map.keys()):
        parent_id = get_synthetic_parent_id(synth_map[synth_name])
        
        # Find neighbors of this synthetic tool
        neighbors = neighbors_by_tool.get(synth_name, [])
        
        for neighbor in neighbors:
            # Check if it's an original AND not the parent AND >0.75 similar
            neighbor_id = orig_id_by_name.get(neighbor['name'])
            if neighbor['type'] == 'original' and neighbor_id != parent_id and neighbor['similarity'] > 0.75:
                rule3_removals.add(synth_name)
                break
    
    print(f"Rule 3 - High similarity to unrelated originals (>0.75): {len(rule3_removals)}")
    for name in rule3_removals:
        del synth_map[name]
    
    # Rule 4: Remove synthetic tool pairs >0.95 similar from different parents
    rule4_removals = set()
    synth_names_list = list(synth_map.keys())
    
    for i, synth1_name in enumerate(synth_names_list):
        if synth1_name in rule4_removals:
            continue
            
        parent1 = get_synthetic_parent_id(synth_map[synth1_name])
        neighbors = neighbors_by_tool.get(synth1_name, [])
        
        for neighbor in neighbors:
            synth2_name = neighbor['name']
            
            # Check if it's synthetic AND >0.9 similar AND different parents
            if neighbor['type'] == 'synthetic' and neighbor['similarity'] > 0.95:
                synth2_tool = synth_map.get(synth2_name)
                parent2 = get_synthetic_parent_id(synth2_tool) if synth2_tool else None
                
                if parent1 != parent2:
                    # Remove both
                    rule4_removals.add(synth1_name)
                    rule4_removals.add(synth2_name)
                    break
    
    print(f"Rule 4 - High similarity pairs from different parents (>0.95): {len(rule4_removals)}")
    for name in rule4_removals:
        synth_map.pop(name, None)
    
    # Final sets
    final_originals = [orig_names[name] for name in orig_names]
    final_synthetics = list(synth_map.values())
    final_all = final_originals + final_synthetics
    
    print(f"\nFinal gold tool set:")
    print(f"  Original tools: {len(final_originals)}")
    print(f"  Synthetic tools: {len(final_synthetics)}")
    print(f"  Total: {len(final_all)}")
    
    # Save gold tool set
    gold_output = output_dir / 'tools_en_gold.jsonl'
    with open(gold_output, 'w') as f:
        for tool in final_all:
            f.write(json.dumps(tool) + '\n')
    
    print(f"\nSaved gold tool set to {gold_output}")
    
    # Save final synthetic tools separately
    synthetic_output = output_dir / 'tools_en_gold_synthetic.jsonl'
    with open(synthetic_output, 'w') as f:
        for tool in final_synthetics:
            f.write(json.dumps(tool) + '\n')
    
    print(f"Saved final synthetic tools to {synthetic_output}")
    
    # Save removal report
    report = {
        'total_original_before': len(originals),
        'total_synthetic_before': len(synthetics),
        'rule1_capped_per_source_id': len(rule1_cap_removals),
        'rule2_exact_name_match': len(exact_matches),
        'rule3_high_sim_unrelated_original': len(rule3_removals),
        'rule4_high_sim_synthetic_pairs': len(rule4_removals),
        'total_removed': len(rule1_cap_removals) + len(exact_matches) + len(rule3_removals) + len(rule4_removals),
        'final_original_count': len(final_originals),
        'final_synthetic_count': len(final_synthetics),
        'final_total_count': len(final_all),
    }
    
    report_output = output_dir / 'gold_tool_set_report.json'
    with open(report_output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved report to {report_output}")
    
    return final_all


def main():
    parser = argparse.ArgumentParser(description="Build gold tool set with filtering rules")
    parser.add_argument('--originals', default='multi-agent-framework/tools/tools_en_with_normalized_tool_ids.jsonl')
    parser.add_argument('--synthetics', default='analysis_embeddings/tools_en_synthetic_candidates.jsonl')
    parser.add_argument('--neighbors', default='analysis_embeddings/tools_en_all_neighbors.csv')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--output-dir', default='analysis_embeddings')
    
    args = parser.parse_args()
    
    print("Loading tools...")
    originals = load_json_or_jsonl(args.originals)
    synthetics = load_json_or_jsonl(args.synthetics)
    
    print(f"Deduplicating originals by normalized_tool_id, synthetics by name...")
    originals = deduplicate_by_normalized_id(originals)
    synthetics = deduplicate_by_name(synthetics)
    
    print(f"After deduplication:")
    print(f"  Original: {len(originals)}")
    print(f"  Synthetic: {len(synthetics)}")
    
    gold_tools = filter_gold_tools(originals, synthetics, args.neighbors, args.model, args.output_dir)


if __name__ == '__main__':
    main()
