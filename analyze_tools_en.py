#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

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


def load_benchmark_records(path):
    records = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def normalize_tool_text(tool):
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


def extract_tool_description(tool):
    function = tool.get("function", {})
    description = function.get("description") or tool.get("description") or ""
    if description:
        return description.strip()
    return normalize_tool_text(tool).split("\n", 1)[0]


def extract_tool_name(tool):
    if isinstance(tool, str):
        return tool
    if isinstance(tool, dict):
        if "function" in tool and isinstance(tool["function"], dict):
            return tool["function"].get("name") or tool.get("name")
        return tool.get("name")
    return None


def normalize_task_text(task):
    if isinstance(task, str):
        return task.strip()
    if isinstance(task, dict):
        if "input" in task and isinstance(task["input"], str):
            return task["input"].strip()
        return json.dumps(task, ensure_ascii=False)
    return str(task)


def combine_tool_and_task_text(tool, task_texts):
    text = normalize_tool_text(tool)
    if task_texts:
        example_text = " | ".join(task_texts[:3])
        text += f"\nExample tasks: {example_text}"
    return text


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_embedding_openai(client, model, inputs, batch_size=64):
    embeddings = []
    for i in range(0, len(inputs), batch_size):
        chunk = inputs[i : i + batch_size]
        response = client.embeddings.create(model=model, input=chunk)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings


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


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "will", "have", "has", "are", "was",
    "were", "use", "used", "using", "get", "info", "data", "tool", "tools", "task", "tasks",
    "search", "query", "retrieve", "provide", "return", "returns", "based", "specific",
    "including", "include", "user", "users", "help", "details", "information",
    "of", "to", "in", "is", "by", "on", "as", "or", "if", "not", "from", "with", "via",
    "name", "id", "email", "phone", "address", "date", "amount", "record", "records", "string",
    "number", "numbers", "value", "values", "result", "results", "status", "status_code",
    "city", "country", "state", "province", "region", "code", "show",
}


def tokenize(text):
    tokens = re.findall(r"\b[a-z0-9][a-z0-9_]*\b", text.lower())
    filtered = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) <= 2:
            continue
        if token.isdigit():
            continue
        filtered.append(token)
    return filtered


def build_vocab(docs, max_features=5000):
    counts = Counter()
    for doc in docs:
        counts.update(tokenize(doc))
    most_common = [token for token, _ in counts.most_common(max_features)]
    return {token: idx for idx, token in enumerate(most_common)}


def compute_tfidf_matrix(docs, vocab):
    n_docs = len(docs)
    vocab_size = len(vocab)
    dfs = np.zeros(vocab_size, dtype=np.float64)
    term_lists = []
    for doc in docs:
        terms = [t for t in tokenize(doc) if t in vocab]
        term_lists.append(terms)
        for term in set(terms):
            dfs[vocab[term]] += 1
    idf = np.log((n_docs + 1) / (dfs + 1)) + 1.0
    matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)
    for i, terms in enumerate(term_lists):
        if not terms:
            continue
        tf = Counter(terms)
        total = len(terms)
        for term, count in tf.items():
            idx = vocab[term]
            matrix[i, idx] = (count / total) * idf[idx]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return matrix


def batch_embedding_tfidf(inputs, max_components=128):
    vocab = build_vocab(inputs, max_features=5000)
    matrix = compute_tfidf_matrix(inputs, vocab)
    if matrix.shape[1] > max_components:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        matrix = u[:, :max_components] * s[:max_components]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
    return matrix.tolist(), vocab, matrix


def build_nearest_neighbors(tool_names, embeddings, top_k):
    n = len(tool_names)
    nearest = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sims.append((j, cosine_similarity(embeddings[i], embeddings[j])))
        sims.sort(key=lambda x: x[1], reverse=True)
        for rank, (j, score) in enumerate(sims[:top_k], start=1):
            nearest.append(
                {
                    "tool_index": i,
                    "tool_name": tool_names[i],
                    "neighbor_index": j,
                    "neighbor_name": tool_names[j],
                    "similarity": round(float(score), 6),
                    "rank": rank,
                }
            )
    return nearest


def find_isolated_tools(nearest_neighbors, threshold=0.75):
    top_similarity = {}
    for row in nearest_neighbors:
        if row["rank"] != 1:
            continue
        top_similarity[row["tool_name"]] = row["similarity"]
    return [tool for tool, sim in top_similarity.items() if sim < threshold]


def cluster_tool_embeddings(embeddings, cluster_count):
    X = np.array(embeddings, dtype=np.float64)
    n_samples = X.shape[0]
    cluster_count = min(cluster_count, n_samples)
    if cluster_count <= 0:
        cluster_count = max(2, int(math.sqrt(n_samples) + 0.5))
    if n_samples == 0:
        return []
    if cluster_count >= n_samples:
        return [int(i) for i in range(n_samples)]

    rng = np.random.RandomState(42)
    centroids = np.zeros((cluster_count, X.shape[1]), dtype=np.float64)
    first_idx = rng.randint(n_samples)
    centroids[0] = X[first_idx]
    for k in range(1, cluster_count):
        dist = np.min(np.sum((X[:, None, :] - centroids[:k]) ** 2, axis=2), axis=1)
        if dist.sum() == 0:
            centroids[k] = X[rng.randint(n_samples)]
        else:
            probabilities = dist / dist.sum()
            centroid_idx = rng.choice(n_samples, p=probabilities)
            centroids[k] = X[centroid_idx]

    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(100):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(cluster_count):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
    return [int(x) for x in labels.tolist()]


def label_clusters(texts, cluster_ids, top_n=5):
    cluster_terms = defaultdict(Counter)
    for text, cid in zip(texts, cluster_ids):
        for term in tokenize(text):
            cluster_terms[int(cid)][term] += 1

    clusters = {}
    for cluster_id in sorted(cluster_terms.keys()):
        top_terms = [term for term, _ in cluster_terms[cluster_id].most_common(top_n * 3)]
        labels = []
        for term in top_terms:
            if re.search(r"\b(tool|task|query|search|get|retrieve|information|data|info|details|use|used)\b", term, re.I):
                continue
            labels.append(term)
            if len(labels) >= top_n:
                break
        if not labels:
            labels = top_terms[:top_n]
        clusters[cluster_id] = ", ".join(labels[:top_n]) or f"cluster_{cluster_id}"
    return clusters


def write_csv(out_path, fieldnames, rows):
    with open(out_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_task_map(benchmark_records, tool_names, use_english=True):
    task_map = defaultdict(lambda: {"records": [], "task_texts": []})
    for record in benchmark_records:
        tools_in_record = []
        if use_english and record.get("english_tools"):
            tools_in_record = [extract_tool_name(t) for t in record.get("english_tools", [])]
        else:
            tools_in_record = [extract_tool_name(t) for t in record.get("tools", [])]
        tools_in_record = [name for name in tools_in_record if name]

        task_texts = []
        if use_english and record.get("english_tasks"):
            task_texts = [normalize_task_text(t) for t in record.get("english_tasks", [])]
        else:
            task_texts = [normalize_task_text(t) for t in record.get("tasks", [])]

        for tool_name in tools_in_record:
            if tool_name not in tool_names:
                continue
            task_map[tool_name]["records"].append(record)
            task_map[tool_name]["task_texts"].extend([t for t in task_texts if t])
    return task_map


def sample_task_texts(texts, limit=3):
    return texts[:limit]


def choose_cluster_count(requested, n_tools):
    if requested > 0:
        return min(requested, n_tools)
    return max(2, min(20, int(math.sqrt(n_tools) * 1.5)))


def main():
    parser = argparse.ArgumentParser(description="Analyze tools_en with task-aware clustering and local open-source text features")
    parser.add_argument(
        "--input",
        default="multi-agent-framework/tools/tools_en.jsonl",
        help="Path to tools_en.jsonl",
    )
    parser.add_argument(
        "--benchmark",
        default="wild-tool-bench/data/Wild-Tool-Bench.jsonl",
        help="Path to Wild-Tool-Bench benchmark JSONL",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory where analysis outputs are written",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model to use for local sentence-transformers",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "local", "tfidf"],
        default="local",
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to include per tool",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=0,
        help="Number of tool clusters to produce. Default is heuristic based on tool count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding requests",
    )
    parser.add_argument(
        "--use-english-benchmark",
        action="store_true",
        help="Use english_tools/english_tasks fields from the benchmark when available",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    tools = load_json_or_jsonl(input_path)
    if not tools:
        raise RuntimeError(f"No tools loaded from {input_path}")

    benchmark_records = []
    if args.benchmark:
        benchmark_path = Path(args.benchmark)
        if benchmark_path.exists():
            benchmark_records = load_benchmark_records(benchmark_path)
        else:
            print(f"Warning: Benchmark file not found: {benchmark_path}. Continuing without task-aware analysis.")

    tool_map = {}
    tool_names = []
    for tool in tools:
        name = extract_tool_name(tool)
        if not name:
            continue
        if name in tool_map:
            continue
        tool_map[name] = tool
        tool_names.append(name)

    task_map = build_task_map(benchmark_records, tool_names, use_english=args.use_english_benchmark)

    tool_texts = []
    for tool_name in tool_names:
        tool = tool_map[tool_name]
        task_texts = sample_task_texts(task_map.get(tool_name, {}).get("task_texts", []), limit=3)
        tool_texts.append(combine_tool_and_task_text(tool, task_texts))

    if args.backend == "openai":
        if OpenAI is None:
            raise ImportError("Please install the openai package (pip install openai) to use OpenAI embeddings.")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI backend.")
        client = OpenAI(api_key=api_key)
        embeddings = batch_embedding_openai(client, args.model, tool_texts, batch_size=args.batch_size)
        vectorizer = None
        tfidf_matrix = None
    elif args.backend == "local":
        embeddings = batch_embedding_local(args.model, tool_texts, batch_size=args.batch_size)
        vectorizer = None
        tfidf_matrix = None
    else:
        embeddings, vectorizer, tfidf_matrix = batch_embedding_tfidf(tool_texts)

    nearest_neighbors = build_nearest_neighbors(tool_names, embeddings, args.top_k)
    cluster_count = choose_cluster_count(args.cluster_count, len(tool_names))
    cluster_ids = cluster_tool_embeddings(embeddings, cluster_count)
    cluster_labels = label_clusters(tool_texts, cluster_ids, top_n=6)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "tools_en_tool_neighbors.csv",
        ["tool_name", "neighbor_name", "similarity", "rank"],
        [
            {
                "tool_name": row["tool_name"],
                "neighbor_name": row["neighbor_name"],
                "similarity": row["similarity"],
                "rank": row["rank"],
            }
            for row in nearest_neighbors
        ],
    )

    cluster_rows = []
    for idx, tool_name in enumerate(tool_names):
        task_texts = sample_task_texts(task_map.get(tool_name, {}).get("task_texts", []), limit=3)
        cluster_rows.append(
            {
                "tool_name": tool_name,
                "cluster_id": cluster_ids[idx],
                "cluster_label": cluster_labels.get(cluster_ids[idx], f"cluster_{cluster_ids[idx]}"),
                "task_count": len(task_map.get(tool_name, {}).get("task_texts", [])),
                "task_samples": " | ".join(task_texts),
            }
        )
    write_csv(
        out_dir / "tools_en_tool_clusters.csv",
        ["tool_name", "cluster_id", "cluster_label", "task_count", "task_samples"],
        cluster_rows,
    )

    tool_stats_rows = []
    for tool_name in tool_names:
        task_texts = task_map.get(tool_name, {}).get("task_texts", [])
        tool_stats_rows.append(
            {
                "tool_name": tool_name,
                "task_count": len(task_texts),
                "example_tasks": " | ".join(sample_task_texts(task_texts, limit=3)),
            }
        )
    write_csv(
        out_dir / "tools_en_tool_task_stats.csv",
        ["tool_name", "task_count", "example_tasks"],
        tool_stats_rows,
    )

    cluster_groups = defaultdict(list)
    for idx, tool_name in enumerate(tool_names):
        cluster_groups[cluster_ids[idx]].append(tool_name)

    isolated_tools = find_isolated_tools(nearest_neighbors, threshold=0.75)
    write_csv(
        out_dir / "tools_en_tool_isolated.csv",
        ["tool_name", "top_neighbor_similarity", "cluster_id", "cluster_label", "description"],
        [
            {
                "tool_name": tool_name,
                "top_neighbor_similarity": next(
                    (row["similarity"] for row in nearest_neighbors if row["tool_name"] == tool_name and row["rank"] == 1),
                    0.0,
                ),
                "cluster_id": cluster_ids[tool_names.index(tool_name)],
                "cluster_label": cluster_labels.get(cluster_ids[tool_names.index(tool_name)], f"cluster_{cluster_ids[tool_names.index(tool_name)]}"),
                "description": extract_tool_description(tool_map[tool_name]),
            }
            for tool_name in isolated_tools
        ],
    )

    cluster_detail_rows = []
    for cid, names in sorted(cluster_groups.items()):
        sample_tools = names[:6]
        cluster_detail_rows.append(
            {
                "cluster_id": cid,
                "cluster_label": cluster_labels.get(cid, f"cluster_{cid}"),
                "size": len(names),
                "sample_tool_names": " | ".join(sample_tools),
                "sample_tool_descriptions": " | ".join(
                    extract_tool_description(tool_map[name]) for name in sample_tools
                ),
            }
        )
    write_csv(
        out_dir / "tools_en_cluster_overview.csv",
        ["cluster_id", "cluster_label", "size", "sample_tool_names", "sample_tool_descriptions"],
        cluster_detail_rows,
    )

    summary = {
        "tool_count": len(tool_names),
        "model": args.model,
        "backend": args.backend,
        "top_k": args.top_k,
        "cluster_count": cluster_count,
        "clusters": [],
        "tools": [],
        "top_words": [],
    }
    cluster_groups = defaultdict(list)
    for idx, tool_name in enumerate(tool_names):
        cluster_groups[cluster_ids[idx]].append(tool_name)

    for cid, names in sorted(cluster_groups.items()):
        all_task_texts = [t for name in names for t in sample_task_texts(task_map.get(name, {}).get("task_texts", []), limit=3)]
        summary["clusters"].append(
            {
                "cluster_id": cid,
                "cluster_label": cluster_labels.get(cid, f"cluster_{cid}"),
                "size": len(names),
                "tool_names": names,
                "sample_tasks": all_task_texts[:5],
            }
        )

    for idx, tool_name in enumerate(tool_names):
        summary["tools"].append(
            {
                "tool_name": tool_name,
                "cluster_id": cluster_ids[idx],
                "cluster_label": cluster_labels.get(cluster_ids[idx], f"cluster_{cluster_ids[idx]}"),
                "task_count": len(task_map.get(tool_name, {}).get("task_texts", [])),
            }
        )

    if args.backend == "tfidf" and isinstance(vectorizer, dict) and tfidf_matrix is not None:
        inv_vocab = {idx: term for term, idx in vectorizer.items()}
        term_weights = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        top_indices = np.argsort(-term_weights)[:20]
        summary["top_words"] = [inv_vocab[i] for i in top_indices if i in inv_vocab]
    

    with open(out_dir / "tools_en_summary.json", "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)

    print(f"Loaded {len(tool_names)} tools from {input_path}")
    if benchmark_records:
        print(f"Loaded {len(benchmark_records)} benchmark records from {args.benchmark}")
    print(f"Wrote similarity CSV to {out_dir / 'tools_en_tool_neighbors.csv'}")
    print(f"Wrote cluster CSV to {out_dir / 'tools_en_tool_clusters.csv'}")
    print(f"Wrote tool/task stats CSV to {out_dir / 'tools_en_tool_task_stats.csv'}")
    print(f"Wrote summary JSON to {out_dir / 'tools_en_summary.json'}")


if __name__ == "__main__":
    main()
