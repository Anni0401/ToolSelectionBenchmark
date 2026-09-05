#!/usr/bin/env python3
"""Build DPO preference pairs from three query rewrite variants using the paper's ranking reward.

This script loads a JSON array of records like:
    {
      "original_query": "...",
      "rewrite1": "...",
      "rewrite2": "...",
      "gold_tools": ["tool_a", "tool_b"]
    }

For each record it embeds the original and both rewrites with the local Qwen3
embedding endpoint, retrieves the full ranking across the tool database, and
computes the paper's ranking reward from the rank of each gold tool. The three
query variants are then scored independently, and the best/worst scoring variants
are selected as the chosen/rejected pair for DPO.

The ranking reward uses the paper's cutoff n=10, applied to the full retrieved
ranking of all tools (M total), while not forcing a preference when scores are tied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "queries_gold_tools_batch1_rewrites.json"
    default_output = script_dir / "queries_gold_tools_batch1_dpo_ranked.json"
    default_schema_cache = (
        script_dir.parent
        / "wild-tool-bench"
        / "wtb"
        / "model_handler"
        / "api_inference"
        / "tool_schemas_cache.jsonl"
    )
    default_embedding_cache = (
        script_dir.parent
        / "wild-tool-bench"
        / "wtb"
        / "model_handler"
        / "api_inference"
        / "tool_embeddings_cache_qwen3.json"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=str(default_input), help="Input JSON file with original + rewrite queries and gold tools")
    parser.add_argument("--output", type=str, default=str(default_output), help="Output JSON path for DPO rows")
    parser.add_argument("--schema-cache-file", type=str, default=str(default_schema_cache), help="Schema cache JSONL used as the ranking candidate universe")
    parser.add_argument("--embedding-cache-file", type=str, default=str(default_embedding_cache), help="Qwen3 embedding cache JSON file (tool text -> vector)")
    parser.add_argument("--base-url", type=str, default=os.getenv("QWEN3_EMBEDDING_BASE_URL", "http://localhost:8002/v1"), help="OpenAI-compatible embedding base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("QWEN3_EMBEDDING_API_KEY", "EMPTY"), help="Embedding API key")
    parser.add_argument("--model", type=str, default=os.getenv("QWEN3_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"), help="Embedding model name")
    parser.add_argument("--cutoff", type=int, default=10, help="Paper cutoff n used in the ranking reward")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size used for embedding tool texts")
    return parser.parse_args()


def load_json_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in {path}")
    return [entry for entry in data if isinstance(entry, dict)]


def load_schema_tools(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Schema cache file not found: {path}")

    tools: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
            if isinstance(entry, list):
                tools.extend(entry)
            else:
                tools.append(entry)

    clean_tools: List[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            continue
        if not str(func.get("name", "")).strip() or not str(func.get("description", "")).strip():
            continue
        clean_tools.append(tool)

    return clean_tools


def tool_name(tool: dict) -> str:
    func = tool.get("function")
    if not isinstance(func, dict):
        return ""
    return str(func.get("name", "")).strip()


def tool_text(tool: dict) -> str:
    func = tool.get("function")
    if not isinstance(func, dict):
        return ""
    name = str(func.get("name", "")).strip()
    description = str(func.get("description", "")).strip()
    parameters = func.get("parameters", {}) or {}
    return f"{name}: {description}\nParameters: {json.dumps(parameters, ensure_ascii=False, sort_keys=True)}"


def safe_float(value: float) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return 0.0


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Embedding vectors must have the same dimension")
    if not vec_a:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        fa = safe_float(a)
        fb = safe_float(b)
        dot += fa * fb
        norm_a += fa * fa
        norm_b += fb * fb

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def embed_texts(client: OpenAI, model: str, texts: Iterable[str]) -> List[List[float]]:
    text_list = list(texts)
    if not text_list:
        return []

    response = client.embeddings.create(model=model, input=text_list)
    vectors = []
    for item in response.data:
        vectors.append(list(item.embedding))
    return vectors


def load_tool_embedding_cache(cache_path: Path) -> dict[str, list[float]]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    except Exception:
        return {}
    if not isinstance(cache, dict):
        return {}
    cleaned: dict[str, list[float]] = {}
    for key, value in cache.items():
        if isinstance(key, str) and isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            cleaned[key] = [float(x) for x in value]
    return cleaned


def save_tool_embedding_cache(cache_path: Path, cache: dict[str, list[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False)
    tmp_path.replace(cache_path)


def build_tool_embedding_map(
    client: OpenAI,
    model: str,
    tools: List[dict],
    cache_path: Path,
    batch_size: int,
) -> tuple[list[str], dict[str, list[float]]]:
    tool_order: list[str] = []
    embedding_cache = load_tool_embedding_cache(cache_path)
    missing_texts: list[str] = []

    for tool in tools:
        text = tool_text(tool)
        if not text:
            continue
        name = tool_name(tool)
        if not name:
            continue
        tool_order.append(name)
        if text not in embedding_cache:
            missing_texts.append(text)

    if missing_texts:
        for idx in range(0, len(missing_texts), batch_size):
            batch = missing_texts[idx: idx + batch_size]
            batch_embeddings = embed_texts(client, model, batch)
            for text, emb in zip(batch, batch_embeddings):
                embedding_cache[text] = emb
        save_tool_embedding_cache(cache_path, embedding_cache)

    tool_to_vector: dict[str, list[float]] = {}
    for tool in tools:
        text = tool_text(tool)
        name = tool_name(tool)
        if not name or not text:
            continue
        if text in embedding_cache:
            tool_to_vector[name] = embedding_cache[text]

    return tool_order, tool_to_vector


def full_ranking_for_query(query_text: str, tool_order: list[str], tool_vectors: dict[str, list[float]]) -> List[str]:
    if not query_text or not str(query_text).strip():
        raise ValueError("Query text is empty")

    query_embedding = None
    if hasattr(query_text, "__iter__") and not isinstance(query_text, str):
        raise TypeError("query_text must be a string")

    ranked_scores = []
    for tool_id in tool_order:
        vector = tool_vectors.get(tool_id)
        if vector is None:
            continue
        ranked_scores.append((tool_id, 0.0))

    if not ranked_scores:
        return []

    # This is called after the query embedding has been generated externally.
    raise RuntimeError("This helper is no longer used; query embedding should be computed in the caller.")


def rank_query_against_tools(query_embedding: list[float], tool_order: list[str], tool_vectors: dict[str, list[float]]) -> list[str]:
    ranked_scores = []
    for tool_id in tool_order:
        vector = tool_vectors.get(tool_id)
        if vector is None:
            continue
        score = cosine_similarity(query_embedding, vector)
        ranked_scores.append((tool_id, score))
    ranked_scores.sort(key=lambda item: item[1], reverse=True)
    return [tool_id for tool_id, _ in ranked_scores]


def build_retrieval_text(query_text: str, sampled_tool_examples: object | None = None) -> str:
    """Return the retrieval input with the shared sample tools appended to the query."""
    query_text = str(query_text or "").strip()
    if not query_text:
        return ""

    if sampled_tool_examples is None:
        return query_text

    if isinstance(sampled_tool_examples, list):
        examples_block = "\n".join(str(item) for item in sampled_tool_examples)
    elif isinstance(sampled_tool_examples, str):
        examples_block = sampled_tool_examples
    else:
        return query_text

    examples_block = str(examples_block).strip()
    if not examples_block:
        return query_text

    return (
        "Rewrite the query for tool retrieval.\n\n"
        "Tool examples:\n"
        f"{examples_block}\n\n"
        "User query:\n"
        f"{query_text}"
    )


def paper_rank_score(rank: int, cutoff: int = 10) -> float:
    """Return the paper's ranking reward for a one-based retrieval rank."""
    if rank <= cutoff:
        return 1.0 / math.log2(rank + 1.1)
    return -(rank - cutoff) / math.log2((rank / cutoff) + 1.0)


def candidate_score(ranking: List[str], gold_tools: Iterable[str], cutoff: int = 10) -> float:
    """Sum the paper ranking reward over all gold tools in a retrieved ranking."""
    rank_map = {tool_id: rank for rank, tool_id in enumerate(ranking, start=1)}
    total = 0.0
    gold_set = set(gold_tools)
    if not gold_set:
        return 0.0

    max_rank = max(len(ranking), 1)
    for tool_id in gold_set:
        rank = rank_map.get(tool_id, max_rank + 1)
        total += paper_rank_score(rank, cutoff=cutoff)
    return total


def choose_preference(scores: dict[str, float]) -> tuple[str, str] | None:
    if not scores:
        return None

    best_name, best_score = max(scores.items(), key=lambda item: item[1])
    worst_name, worst_score = min(scores.items(), key=lambda item: item[1])

    if math.isclose(best_score, worst_score, rel_tol=1e-9, abs_tol=1e-9):
        return None

    return best_name, worst_name


def top10_hit_count(ranking: List[str], gold_tools: Iterable[str], cutoff: int = 10) -> int:
    rank_map = {tool_id: rank for rank, tool_id in enumerate(ranking, start=1)}
    count = 0
    for gold in set(gold_tools):
        rank = rank_map.get(gold)
        if rank is not None and rank <= cutoff:
            count += 1
    return count


def summarize_gold_coverage(rows: List[dict], universe: set[str]) -> tuple[int, list[str]]:
    all_gold = set()
    for row in rows:
        gold_tools = row.get("gold_tools", [])
        if not isinstance(gold_tools, list):
            continue
        for gt in gold_tools:
            g = str(gt).strip()
            if g:
                all_gold.add(g)
    missing = sorted(g for g in all_gold if g not in universe)
    return len(all_gold), missing


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    schema_cache_path = Path(args.schema_cache_file)
    embedding_cache_path = Path(args.embedding_cache_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input query file does not exist: {input_path}")
    if not schema_cache_path.exists():
        raise FileNotFoundError(f"Schema cache file does not exist: {schema_cache_path}")

    rows = load_json_records(input_path)
    tools = load_schema_tools(schema_cache_path)
    if not tools:
        raise ValueError(f"No valid tools loaded from {schema_cache_path}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    tool_order, tool_vectors = build_tool_embedding_map(client, args.model, tools, embedding_cache_path, args.batch_size)
    universe = set(tool_order)
    gold_total, missing_gold = summarize_gold_coverage(rows, universe)
    print(f"[INFO] Candidate tools in universe: {len(universe)}")
    print(f"[INFO] Distinct gold tools in input: {gold_total}")
    if missing_gold:
        print(f"[WARN] Gold tools missing from ranking universe: {len(missing_gold)}")
        print("[WARN] Missing gold tool names:")
        for name in missing_gold:
            print(f"  - {name}")

    dpo_rows: List[dict] = []
    skipped_tie = 0
    skipped_no_candidates = 0
    for row_idx, row in enumerate(rows, start=1):
        try:
            gold_tools = row.get("gold_tools", [])
            if not isinstance(gold_tools, list) or not gold_tools:
                continue
            gold_tools = [str(tool).strip() for tool in gold_tools if str(tool).strip()]
            if not gold_tools:
                continue

            candidates = {
                "original": row.get("original_query"),
                "rewrite_a": row.get("rewrite1"),
                "rewrite_b": row.get("rewrite2"),
            }
            sampled_examples = row.get("sampled_tool_examples") or row.get("prompt_template")
            scores: dict[str, float] = {}
            hits_in_top10: dict[str, int] = {}

            for name, query_text in candidates.items():
                if not isinstance(query_text, str) or not query_text.strip():
                    continue
                retrieval_text = build_retrieval_text(query_text, sampled_examples)
                query_embedding = embed_texts(client, args.model, [retrieval_text])[0]
                ranking = rank_query_against_tools(query_embedding, tool_order, tool_vectors)
                score = candidate_score(ranking, gold_tools, cutoff=args.cutoff)
                scores[name] = score
                hits_in_top10[name] = top10_hit_count(ranking, gold_tools, cutoff=args.cutoff)

            if not scores:
                skipped_no_candidates += 1
                continue

            choice = choose_preference(scores)
            if choice is None:
                skipped_tie += 1
                continue

            best_name, worst_name = choice
            original_query = str(row.get("original_query", "")).strip()
            sampled_examples = row.get("sampled_tool_examples") or row.get("prompt_template")
            prompt = build_retrieval_text(original_query, sampled_examples)

            chosen_query = str(candidates.get(best_name, "")).strip()
            rejected_query = str(candidates.get(worst_name, "")).strip()

            dpo_rows.append({
                "prompt": prompt,
                "chosen": chosen_query,
                "rejected": rejected_query,
                "gold_tools": gold_tools,
                "score_chosen": scores[best_name],
                "score_rejected": scores[worst_name],
            })
        except Exception as exc:
            print(f"[WARN] Skipping row {row_idx}: {exc}", file=sys.stderr)
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dpo_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Rows processed: {len(rows)}")
    print(f"[INFO] Rows skipped (no usable candidates): {skipped_no_candidates}")
    print(f"[INFO] Rows skipped (ties): {skipped_tie}")
    print(f"Wrote {len(dpo_rows)} DPO preference rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
