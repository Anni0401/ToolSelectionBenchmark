"""
Generate two independent high-temperature rewrites per query via SAIA (openai-gpt-oss-120b).

Input format (jsonl):
    {"query": "...", "gold_tools": ["toolA", "toolB"]}

Output format (json):
[
  {
    "original_query": "...",
    "rewrite1": "...",
    "rewrite2": "...",
    "gold_tools": ["..."]
  },
  ...
]

The rewrite prompt is copied from the rewrite selection strategy
(Qwen3QueryRewriteEmbeddingContextToolSelector.REWRITE_PROMPT_TEMPLATE),
and we sample 5 synthetic tools from tool_schemas_cache.jsonl for each rewrite call.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from constant import DOTENV_PATH
from handle.saia import SAIAAPIHandler


REWRITE_PROMPT_TEMPLATE = (
    "You rewrite user requests to improve retrieval of relevant tools.\n\n"
    "Your task is NOT to answer the user and NOT to call any tools.\n\n"
    "Rewrite the user request so that it clearly expresses:\n"
    "- the entities involved;\n"
    "- the intended operations;\n"
    "- intermediate steps;\n"
    "- identifiers that may need to be obtained;\n"
    "- required filters, limits, dates, or arguments;\n"
    "- whether the same tool must be called multiple times.\n\n"
    "Use the terminology and operation style suggested by the example tool\n"
    "definitions below. Do not invent tools or APIs. Preserve the user's\n"
    "intent and all important argument values.\n\n"
    "Example tool definitions:\n"
    "{sampled_tool_documents}\n\n"
    "Original user request:\n"
    "{user_query}\n\n"
    "Return only one rewritten retrieval query. Do not include explanations,\n"
    "JSON, tool calls, or an answer."
)


THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "queries_gold_tools_batch1.jsonl"
    default_output = script_dir / "queries_gold_tools_batch1_rewrites.json"
    default_tools = (
        script_dir.parent
        / "wild-tool-bench"
        / "wtb"
        / "model_handler"
        / "api_inference"
        / "tool_schemas_cache.jsonl"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(default_input), help="Input query+gold_tools jsonl")
    parser.add_argument("--output", default=str(default_output), help="Output json path")
    parser.add_argument(
        "--tools-file",
        default=str(default_tools),
        help="Synthetic tool schema cache jsonl used for sampled examples",
    )
    parser.add_argument(
        "--model",
        default="openai-gpt-oss-120b",
        help="SAIA model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
        help="High sampling temperature for rewrites",
    )
    parser.add_argument(
        "--sampled-tools",
        type=int,
        default=5,
        help="How many synthetic tools to sample per rewrite call",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible tool sampling",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap: only process first N input records",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSON file instead of starting over",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return rows


def load_existing_output(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing output file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Existing output file must contain a JSON list: {path}")
    return data


def write_output_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _stable_key(query: str, gold_tools: list) -> str:
    return json.dumps(
        {
            "query": (query or "").strip(),
            "gold_tools": gold_tools if isinstance(gold_tools, list) else [],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _is_completed_output_row(row: dict) -> bool:
    rewrite1 = clean_rewrite(str(row.get("rewrite1") or ""))
    rewrite2 = clean_rewrite(str(row.get("rewrite2") or ""))
    return bool(rewrite1 and rewrite2)


def reconcile_resume_rows(
    existing_rows: list[dict],
    query_rows: list[dict],
) -> tuple[list[dict | None], int, int, int]:
    """Align existing output rows to input rows by logical key.

    Returns:
      - ordered rows aligned to query_rows (None marks missing/incomplete rows),
      - count of reusable completed rows,
      - count of missing/incomplete rows,
      - count of orphan rows in existing output that did not match input keys.
    """
    buckets: dict[str, list[dict]] = collections.defaultdict(list)
    for row in existing_rows:
        key = _stable_key(
            str(row.get("original_query") or "").strip(),
            row.get("gold_tools", []),
        )
        buckets[key].append(row)

    # Count keys expected from the input to report orphan rows in the existing output.
    expected_counts = collections.Counter(
        _stable_key((row.get("query") or "").strip(), row.get("gold_tools", []))
        for row in query_rows
    )
    orphan_rows = 0
    for key, rows in buckets.items():
        extra = len(rows) - expected_counts.get(key, 0)
        if extra > 0:
            orphan_rows += extra

    ordered_rows: list[dict | None] = []
    reused = 0
    missing = 0
    for row in query_rows:
        query = (row.get("query") or "").strip()
        gold_tools = row.get("gold_tools", [])
        key = _stable_key(query, gold_tools)
        candidates = buckets.get(key, [])

        selected: dict | None = None
        while candidates:
            candidate = candidates.pop(0)
            if _is_completed_output_row(candidate):
                selected = {
                    "original_query": query,
                    "rewrite1": clean_rewrite(str(candidate.get("rewrite1") or "")),
                    "rewrite2": clean_rewrite(str(candidate.get("rewrite2") or "")),
                    "gold_tools": gold_tools,
                    "sampled_tool_examples": candidate.get("sampled_tool_examples", ""),
                    "prompt_template": candidate.get("prompt_template", ""),
                }
                break

        if selected is None:
            ordered_rows.append(None)
            missing += 1
        else:
            ordered_rows.append(selected)
            reused += 1

    return ordered_rows, reused, missing, orphan_rows


def format_sampled_tool(tool: dict) -> str:
    func = tool.get("function", {}) or {}
    name = func.get("name", "")
    desc = func.get("description", "")
    params = json.dumps(func.get("parameters", {}), ensure_ascii=False)
    return f"- {name}: {desc}\n  parameters: {params}"


def sample_tool_documents(tools_pool: list[dict], k: int, rng: random.Random) -> str:
    if not tools_pool:
        return "(no example tools available)"
    real_k = min(k, len(tools_pool))
    sampled = rng.sample(tools_pool, real_k)
    return "\n".join(format_sampled_tool(t) for t in sampled)


def build_prompt(user_query: str, sampled_tool_documents: str) -> str:
    return REWRITE_PROMPT_TEMPLATE.format(
        sampled_tool_documents=sampled_tool_documents,
        user_query=user_query,
    )


def clean_rewrite(text: str) -> str:
    cleaned = THINK_TAG_RE.sub("", text or "").strip()
    return cleaned


def generate_one_rewrite(
    handler: SAIAAPIHandler,
    query: str,
    tools_pool: list[dict],
    sampled_tools: int,
    rng: random.Random,
    sampled_tool_documents: str | None = None,
) -> str:
    if sampled_tool_documents is None:
        sampled_tool_documents = sample_tool_documents(tools_pool, sampled_tools, rng)
    prompt = build_prompt(query, sampled_tool_documents)
    messages = [{"role": "user", "content": prompt}]
    rewrite = handler.request_model(messages)
    rewrite = clean_rewrite(rewrite)
    if not rewrite:
        raise RuntimeError("Model returned an empty rewrite")
    return rewrite


def build_retrieval_prompt(user_query: str, sampled_tool_documents: str | list[str]) -> str:
    tools_block = sampled_tool_documents
    if isinstance(sampled_tool_documents, list):
        tools_block = "\n".join(str(item) for item in sampled_tool_documents)
    return (
        "Tool examples:\n"
        f"{tools_block}\n\n"
        "User query:\n"
        f"{user_query}"
    )


def main() -> int:
    args = parse_args()
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    tools_file = Path(args.tools_file).resolve()

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1
    if not tools_file.exists():
        print(f"ERROR: tools file not found: {tools_file}", file=sys.stderr)
        return 1

    query_rows = load_jsonl(input_path)
    if args.max_items is not None:
        query_rows = query_rows[: args.max_items]

    tools_pool = load_jsonl(tools_file)
    rng = random.Random(args.seed)

    handler = SAIAAPIHandler(model_name=args.model, temperature=args.temperature)

    if args.resume:
        existing_rows = load_existing_output(output_path)
        aligned_rows, reused_count, missing_count, orphan_count = reconcile_resume_rows(existing_rows, query_rows)
        output_rows: list[dict] = []

        print(
            "Resume reconcile: "
            f"existing={len(existing_rows)}, reused={reused_count}, "
            f"missing={missing_count}, orphaned={orphan_count}"
        )
    else:
        output_rows = []
        if output_path.exists():
            print(
                f"ERROR: output file already exists: {output_path}. Use --resume to continue or remove it.",
                file=sys.stderr,
            )
            return 1

    total = len(query_rows)
    if args.resume and not any(item is None for item in aligned_rows):
        print(f"Nothing to do: output already contains all {total} rows")
        write_output_json(output_path, [item for item in aligned_rows if item is not None])
        return 0

    if not args.resume and len(output_rows) >= total:
        print(f"Nothing to do: output already contains all {total} rows")
        return 0

    if args.resume:
        iterable = enumerate(query_rows, start=1)
    else:
        iterable = enumerate(query_rows, start=1)

    for idx, row in iterable:
        if args.resume and aligned_rows[idx - 1] is not None:
            output_rows.append(aligned_rows[idx - 1])
            continue

        query = (row.get("query") or "").strip()
        gold_tools = row.get("gold_tools", [])

        if not query:
            print(f"[{idx}/{total}] WARNING: empty query, skipping")
            continue

        try:
            sampled_tool_documents = sample_tool_documents(tools_pool, args.sampled_tools, rng)
            rewrite1 = generate_one_rewrite(
                handler=handler,
                query=query,
                tools_pool=tools_pool,
                sampled_tools=args.sampled_tools,
                rng=rng,
                sampled_tool_documents=sampled_tool_documents,
            )
            rewrite2 = generate_one_rewrite(
                handler=handler,
                query=query,
                tools_pool=tools_pool,
                sampled_tools=args.sampled_tools,
                rng=rng,
                sampled_tool_documents=sampled_tool_documents,
            )
        except Exception as exc:
            print(f"[{idx}/{total}] ERROR while rewriting query: {exc}", file=sys.stderr)
            return 1

        output_rows.append(
            {
                "original_query": query,
                "rewrite1": rewrite1,
                "rewrite2": rewrite2,
                "gold_tools": gold_tools,
                "sampled_tool_examples": sampled_tool_documents,
                "prompt_template": build_retrieval_prompt(query, sampled_tool_documents),
            }
        )
        write_output_json(output_path, output_rows)

        print(f"[{idx}/{total}] rewrites generated")

    print(f"Wrote {len(output_rows)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
