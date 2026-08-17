#!/usr/bin/env python3
"""
Embed original tools (deduplicated by normalized_tool_id, first occurrence) and
all synthetic tools using a local sentence-transformers model, then compute the
pairwise similarity matrix between every tool (original + synthetic).
"""
import argparse
import csv
import json
import math
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def load_json_or_jsonl(path):
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


def deduplicate_by_normalized_id(tools):
    """Keep first occurrence per normalized_tool_id."""
    seen = set()
    unique = []
    for tool in tools:
        tool_id = tool.get("normalized_tool_id")
        if tool_id is None or tool_id in seen:
            continue
        seen.add(tool_id)
        unique.append(tool)
    return unique


def extract_tool_name(tool):
    function = tool.get("function", {})
    return function.get("name") or tool.get("name")


def normalize_tool_text(tool):
    """Text representation of name, description, and parameters for embedding."""
    function = tool.get("function", {})
    name = function.get("name", "")
    description = function.get("description", "")
    params = function.get("parameters") or {}
    if not isinstance(params, dict):
        params = {}
    props = params.get("properties", {})
    required = params.get("required", [])

    lines = [f"Tool: {name}", f"Description: {description}"]
    if isinstance(props, dict) and props:
        lines.append("Parameters:")
        for prop_name, prop_schema in sorted(props.items()):
            if not isinstance(prop_schema, dict):
                lines.append(f"- {prop_name}: {prop_schema}")
                continue
            prop_type = prop_schema.get("type", "")
            prop_desc = prop_schema.get("description", "")
            enum = prop_schema.get("enum")
            if enum:
                prop_desc += f" Options={enum}."
            lines.append(f"- {prop_name} ({prop_type}): {prop_desc}")
    if isinstance(required, list) and required:
        lines.append(f"Required: {', '.join(str(r) for r in required)}")
    return "\n".join(lines)


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_embedding_local(model_name, inputs, batch_size=64):
    if SentenceTransformer is None:
        raise ImportError(
            "Please install sentence-transformers (pip install sentence-transformers) to use local embeddings."
        )
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i : i + batch_size]
        encoded = model.encode(chunk, show_progress_bar=False, convert_to_tensor=False)
        if hasattr(encoded, "tolist"):
            embeddings.extend(encoded.tolist())
        else:
            embeddings.extend([list(vec) for vec in encoded])
    return embeddings


def build_similarity_matrix(names, types, embeddings):
    """All pairwise similarities (excluding self), sorted by similarity per tool."""
    n = len(names)
    rows = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append((j, cosine_similarity(embeddings[i], embeddings[j])))
        sims.sort(key=lambda x: x[1], reverse=True)
        for rank, (j, score) in enumerate(sims, start=1):
            rows.append(
                {
                    "tool_name": names[i],
                    "tool_type": types[i],
                    "neighbor_name": names[j],
                    "neighbor_type": types[j],
                    "similarity": round(float(score), 6),
                    "rank": rank,
                }
            )
    return rows


def write_csv(out_path, fieldnames, rows):
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Embed originals (deduped by normalized_tool_id) and synthetics, and compute their similarity matrix."
    )
    parser.add_argument(
        "--originals",
        default="multi-agent-framework/tools/tools_en_with_normalized_tool_ids.jsonl",
        help="Path to the normalized-ID tools JSONL",
    )
    parser.add_argument(
        "--synthetics",
        default="analysis_embeddings/tools_en_synthetic_candidates.jsonl",
        help="Path to the synthetic tools JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_embeddings",
        help="Directory where the similarity matrix CSV is written",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Local sentence-transformers embedding model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding requests",
    )
    args = parser.parse_args()

    originals = deduplicate_by_normalized_id(load_json_or_jsonl(args.originals))
    synthetics = load_json_or_jsonl(args.synthetics)

    names = [extract_tool_name(t) for t in originals] + [extract_tool_name(t) for t in synthetics]
    types = ["original"] * len(originals) + ["synthetic"] * len(synthetics)
    texts = [normalize_tool_text(t) for t in originals] + [normalize_tool_text(t) for t in synthetics]

    print(f"Loaded {len(originals)} originals and {len(synthetics)} synthetics")
    print(f"Embedding {len(texts)} tools with local model {args.model}...")
    embeddings = batch_embedding_local(args.model, texts, batch_size=args.batch_size)

    print("Computing similarity matrix...")
    rows = build_similarity_matrix(names, types, embeddings)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tools_en_all_neighbors.csv"
    write_csv(out_path, ["tool_name", "tool_type", "neighbor_name", "neighbor_type", "similarity", "rank"], rows)

    print(f"Wrote similarity matrix to {out_path}")


if __name__ == "__main__":
    main()
