#!/usr/bin/env python3
"""
Generate exactly two synthetic tools for every unique normalized tool ID.

Input expectations
------------------
The --tools JSONL file is the annotated original tool set where every tool has
an integer ``normalized_tool_id``.  The file may contain the same normalized ID
multiple times.  For each ID this script:

1. keeps the FIRST occurrence as the representative definition;
2. keeps the complete tool row containing that first occurrence as context;
3. finds the best matching benchmark record and passes along its task(s);
4. passes along all other tools from the representative's first-occurrence row;
5. asks the LLM for exactly TWO synthetic tools.

No nearest-neighbor CSV is used.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

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
    from openai import OpenAI
except ImportError:
    print("Error: openai not installed. Install with: pip install openai", file=sys.stderr)
    sys.exit(1)


SYNTHETIC_TOOLS_PER_ID = 2

SAIA_DEFAULT_BASE_URL = "https://chat-ai.academiccloud.de/v1"
SAIA_DEFAULT_MODEL = "openai-gpt-oss-120b"


def test_api(api_token, base_url, model=SAIA_DEFAULT_MODEL):
    """Test whether the SAIA API is reachable."""
    print(f"Testing SAIA API at {base_url} with model: {model}")
    response = call_generation_api("What is 2+2?", api_token, base_url, model=model, max_tokens=50)
    if response:
        print(f"✓ API test passed. Response: {response[:100]}")
        return True
    print("✗ API test failed")
    return False


def _extract_tools_from_jsonl_entry(entry):
    """Return the tool list from a JSONL entry."""
    if isinstance(entry, list):
        return entry
    if isinstance(entry, dict):
        for key in ("tools", "english_tools", "tool_schemas", "available_tools"):
            value = entry.get(key)
            if isinstance(value, list):
                return value
    return []


def _tool_name(tool):
    if not isinstance(tool, dict):
        return str(tool) if tool is not None else None
    func = tool.get("function")
    if isinstance(func, dict):
        return func.get("name")
    return tool.get("name")



def load_normalized_tool_contexts(path):
    """
    Load the normalized-ID tool set and aggregate ALL source rows for each ID.

    For each normalized_tool_id:
      - representative = FIRST occurrence of that ID in file order
      - source_row_indices = every JSONL row in which that ID occurs
      - cooccurring_tools = union of all other tools appearing in ANY of those rows

    Row indices are 0-based internally; row numbers shown to users are +1.
    """
    contexts = {}
    record_count = 0
    tool_occurrence_count = 0
    source_rows = []

    with open(path, "r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            record_count += 1
            entry = json.loads(line)
            row_tools = [
                t for t in _extract_tools_from_jsonl_entry(entry)
                if isinstance(t, dict)
            ]
            source_rows.append(copy.deepcopy(row_tools))
            tool_occurrence_count += len(row_tools)

            ids_seen_in_row = set()

            for position, tool in enumerate(row_tools):
                normalized_id = tool.get("normalized_tool_id")
                if normalized_id is None:
                    raise ValueError(
                        f"Missing normalized_tool_id at JSONL row {row_index + 1}, "
                        f"tool position {position + 1}: {_tool_name(tool)!r}"
                    )

                try:
                    normalized_id = int(normalized_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid normalized_tool_id {normalized_id!r} "
                        f"at row {row_index + 1}"
                    ) from exc

                if normalized_id not in contexts:
                    representative = copy.deepcopy(tool)
                    contexts[normalized_id] = {
                        "normalized_tool_id": normalized_id,
                        "representative": representative,
                        "representative_name": _tool_name(representative),
                        "first_row_index": row_index,
                        "first_row_number": row_index + 1,
                        "first_position": position,
                        "source_row_indices": [],
                        "cooccurring_tools": [],
                    }

                # Only record a row once even if the same logical tool occurs
                # multiple times inside that row.
                if normalized_id not in ids_seen_in_row:
                    contexts[normalized_id]["source_row_indices"].append(row_index)
                    ids_seen_in_row.add(normalized_id)

    # Aggregate all co-occurring tools across every row for each normalized ID.
    for normalized_id, context in contexts.items():
        context_seen = set()
        cooccurring_tools = []

        for row_index in context["source_row_indices"]:
            for other in source_rows[row_index]:
                other_id = other.get("normalized_tool_id")

                # Exclude the source logical tool itself, including duplicate
                # occurrences/variants that share its normalized ID.
                if other_id is not None and int(other_id) == normalized_id:
                    continue

                # Prefer normalized ID for deduplication of context tools.
                if other_id is not None:
                    key = ("normalized_id", int(other_id))
                else:
                    func = other.get("function", {})
                    key = (
                        "schema",
                        func.get("name"),
                        func.get("description"),
                        json.dumps(
                            func.get("parameters", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )

                if key in context_seen:
                    continue

                context_seen.add(key)
                cooccurring_tools.append(copy.deepcopy(other))

        context["cooccurring_tools"] = cooccurring_tools
        context["occurrence_count"] = len(context["source_row_indices"])

    print(
        f"  Loaded {tool_occurrence_count} tool occurrences across "
        f"{record_count} JSONL records"
    )
    print(f"  Found {len(contexts)} unique normalized_tool_id values")
    return contexts


def _extract_tasks(entry):
    """Extract task texts from one WildToolBench JSONL record."""
    if not isinstance(entry, dict):
        return []

    tasks = entry.get("english_tasks")
    if tasks is None:
        tasks = entry.get("tasks", [])

    if not isinstance(tasks, list):
        tasks = [tasks]

    result = []
    for task in tasks:
        if isinstance(task, str):
            text = task
        elif isinstance(task, dict):
            text = task.get("input") or task.get("query") or task.get("task")
            if text is None:
                text = json.dumps(task, ensure_ascii=False)
        else:
            text = str(task)

        text = str(text).strip()
        if text:
            result.append(text)

    return result


def load_benchmark_records(path):
    """
    Load benchmark records in file order.

    IMPORTANT: source JSONL row i is assumed to correspond directly to
    benchmark JSONL row i.
    """
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            records.append(
                {
                    "row_index": row_index,
                    "tasks": _extract_tasks(entry),
                }
            )

    return records


def attach_benchmark_tasks(contexts, benchmark_records):
    """
    Attach ALL benchmark tasks from ALL source rows containing each ID.

    Example:
      normalized ID 1 occurs in source rows 1 and 20
      -> attach tasks from benchmark rows 1 and 20
      -> cooccurring_tools already contains the union of tools from source
         rows 1 and 20

    No name-based matching is used.
    """
    if not contexts:
        return

    max_source_row = max(
        row_index
        for context in contexts.values()
        for row_index in context["source_row_indices"]
    )

    if max_source_row >= len(benchmark_records):
        raise ValueError(
            "Benchmark/source row mismatch: source references row "
            f"{max_source_row + 1}, but benchmark only has "
            f"{len(benchmark_records)} records."
        )

    for context in contexts.values():
        tasks = []
        seen_tasks = set()

        for row_index in context["source_row_indices"]:
            record = benchmark_records[row_index]

            for task in record["tasks"]:
                if task in seen_tasks:
                    continue
                seen_tasks.add(task)
                tasks.append(task)

        context["tasks"] = tasks
        context["benchmark_row_indices"] = list(
            context["source_row_indices"]
        )


def extract_tool_description(tool):
    func = tool.get("function", {}) if isinstance(tool, dict) else {}
    return func.get("description", "No description available")


def summarize_tool_for_prompt(tool):
    """Compact representation of a co-occurring tool for the generation prompt."""
    if not isinstance(tool, dict):
        return str(tool)

    func = tool.get("function", {})
    params = func.get("parameters", {}) if isinstance(func, dict) else {}
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    required = params.get("required", []) if isinstance(params, dict) else []

    return {
        "normalized_tool_id": tool.get("normalized_tool_id"),
        "name": func.get("name"),
        "description": func.get("description", ""),
        "parameters": list(props.keys()) if isinstance(props, dict) else [],
        "required": required if isinstance(required, list) else [],
    }


def call_generation_api(prompt, api_token, base_url, model=SAIA_DEFAULT_MODEL, max_tokens=2048):
    """Call the SAIA (Academic Cloud) OpenAI-compatible API for text generation."""
    try:
        client = OpenAI(api_key=api_token, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful tool generation expert. Generate valid JSON responses.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )

        if response.choices:
            return response.choices[0].message.content
        return None
    except Exception as e:
        print(
            f"Error calling SAIA API with model {model}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None


def _parse_generated_tools(response):
    """
    Parse either one-JSON-object-per-line output or a JSON array/object.
    """
    if not response:
        return []

    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    parsed_tools = []

    # First try the whole response as JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            parsed_tools.extend(obj)
        elif isinstance(obj, dict):
            parsed_tools.append(obj)
    except json.JSONDecodeError:
        # Fall back to one complete JSON object per line.
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("```"):
                continue
            try:
                parsed_tools.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    valid = []
    for tool in parsed_tools:
        if not isinstance(tool, dict):
            continue
        func = tool.get("function")
        if not isinstance(func, dict) or not func.get("name"):
            continue
        tool.setdefault("type", "function")
        tool.pop("reason", None)
        valid.append(tool)

    return valid


def load_completed_source_ids(path, expected_per_id=SYNTHETIC_TOOLS_PER_ID):
    """Return source normalized IDs that already have a complete generated set.

    The synthetic result JSONL stores the source ID in ``_source_tool_id``.
    An ID is treated as complete only when at least ``expected_per_id`` valid
    synthetic tools are already present. This makes reruns safe after partial
    failures: incomplete IDs are generated again, complete IDs are skipped.
    """
    path = Path(path)
    if not path.exists():
        return set(), {}

    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tool = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"WARNING: ignoring invalid JSON in existing result line {line_no}: {path}",
                    file=sys.stderr,
                )
                continue

            source_id = tool.get("_source_tool_id") if isinstance(tool, dict) else None
            try:
                source_id = int(source_id)
            except (TypeError, ValueError):
                continue

            counts[source_id] = counts.get(source_id, 0) + 1

    completed = {
        source_id
        for source_id, count in counts.items()
        if count >= expected_per_id
    }
    return completed, counts


def generate_synthetic_tools_for_id(context, api_token, base_url, model):
    """Generate exactly two synthetic tools for one normalized tool ID."""
    normalized_id = context["normalized_tool_id"]
    tool_def = context["representative"]
    tool_name = context["representative_name"]
    description = extract_tool_description(tool_def)
    tasks = context.get("tasks", [])
    cooccurring_tools = context.get("cooccurring_tools", [])

    original_params = tool_def.get("function", {}).get("parameters", {})
    properties = original_params.get("properties", {}) if isinstance(original_params, dict) else {}
    original_param_keys = list(properties.keys()) if isinstance(properties, dict) else []
    original_required = original_params.get("required", []) if isinstance(original_params, dict) else []

    task_block = "\n".join(f"   - {task}" for task in tasks) or "   - No benchmark task text found"

    co_tools_summary = [summarize_tool_for_prompt(t) for t in cooccurring_tools]
    co_tools_block = json.dumps(co_tools_summary, ensure_ascii=False, indent=2)

    prompt = f"""You are generating realistic distractor API tools for WildToolBench. Generate EXACTLY {SYNTHETIC_TOOLS_PER_ID} synthetic tool definitions.

The source tool represents normalized tool ID {normalized_id}. The task/context information below is aggregated across EVERY source row in which this normalized ID occurs.

FUNCTIONAL REQUIREMENTS
1. Stay in the same broad domain/category as the source tool.
2. Give each synthetic tool a NOTICEABLY DIFFERENT purpose/functionality from the source tool and from the other synthetic tool.
3. Neither synthetic tool may be a valid solution for ANY benchmark task shown below.
4. Do not duplicate the functionality of any real co-occurring tool shown below.
5. Use parameters appropriate to the new functionality rather than merely renaming source parameters.

STYLE REQUIREMENTS -- MATCH WILDTOOLBENCH
The synthetic tools must look as if they were written by the same API authors as the original WildToolBench tools. Do NOT make the synthetic descriptions more polished, verbose, or explanatory than the real tools.

Follow these style rules strictly:
- Prefer SHORT, direct API descriptions, usually one sentence.
- Prefer simple verbs such as "Get", "Retrieve", "Create", "Update", "Delete", "Search", "Check", "Calculate", "List", or "Generate".
- Avoid marketing/product language and explanatory clauses such as "allowing users to...", "helping users...", "ensuring...", "suitable for...", or long lists of benefits.
- Do not make descriptions artificially detailed. Match the approximate brevity and plainness of the source and co-occurring WTB tools.
- Tool names must be realistic camelCase and should resemble the naming conventions of the real tools in this context. Do not make names unusually elegant, long, or descriptive.
- Parameter descriptions should also be short and API-like. Prefer wording such as "The name of the city.", "The ID of the user.", "The start date, in YYYY-MM-DD format."
- Reuse the vocabulary, capitalization conventions, abbreviation style, and level of specificity visible in the source/co-occurring tools when appropriate.
- Preserve normal API messiness if it is present in the examples; do not systematically improve naming or prose.
- Do not mention benchmarks, distractors, synthetic generation, invalidity, or style matching in any generated field.

Benchmark tasks associated with this source context:
{task_block}

Source tool specification:
- normalized_tool_id: {normalized_id}
- Name: {tool_name}
- Description: {description}
- Parameters: {original_param_keys}
- Required parameters: {original_required}

All unique real tools that co-occur with this source tool across ANY source row in which this normalized ID occurs. Treat these as both FUNCTIONAL EXCLUSIONS and STYLE EXAMPLES:
{co_tools_block}

Return tools in EXACTLY this JSON structure:

{{
  "function": {{
    "name": "syntheticToolName",
    "description": "Short WTB-style description of the tool.",
    "parameters": {{
      "type": "object",
      "properties": {{
        "param1": {{
          "type": "string",
          "description": "Short WTB-style parameter description."
        }}
      }},
      "required": ["param1"]
    }}
  }},
  "type": "function"
}}

FINAL CHECKS
- Return exactly {SYNTHETIC_TOOLS_PER_ID} tool definitions.
- The two tools must solve different sub-problems.
- Neither tool may solve any listed benchmark task.
- Neither tool may duplicate the source or a co-occurring real tool.
- Keep descriptions concise and stylistically similar to the real WTB tools above.
- Output ONLY valid JSON, with one complete tool definition per line.
- Do not wrap the response in markdown fences and do not add explanations."""

    response = call_generation_api(prompt, api_token, base_url, model=model, max_tokens=3072)
    synthetic_tools = _parse_generated_tools(response)

    # The request is exactly two.  Never silently keep extras.
    if len(synthetic_tools) > SYNTHETIC_TOOLS_PER_ID:
        synthetic_tools = synthetic_tools[:SYNTHETIC_TOOLS_PER_ID]

    if len(synthetic_tools) != SYNTHETIC_TOOLS_PER_ID:
        print(
            f"  WARNING: expected {SYNTHETIC_TOOLS_PER_ID} tools but parsed "
            f"{len(synthetic_tools)} for normalized_tool_id={normalized_id}",
            file=sys.stderr,
        )

    return synthetic_tools


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate exactly two synthetic tools for each unique normalized_tool_id "
            "using its first occurrence as representative plus ALL matching source rows, benchmark tasks, and co-occurring tools."
        )
    )
    parser.add_argument(
        "--tools",
        default="multi-agent-framework/tools/tools_en_with_normalized_tool_ids.jsonl",
        help="Annotated tools JSONL containing normalized_tool_id on every tool",
    )
    parser.add_argument(
        "--benchmark",
        default="wild-tool-bench/data/Wild-Tool-Bench.jsonl",
        help="Path to Wild-Tool-Bench.jsonl",
    )
    parser.add_argument(
        "--api-key",
        help="SAIA API key (or set SAIA_API_KEY / EXECUTING_LLM_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("SAIA_BASE_URL", SAIA_DEFAULT_BASE_URL),
        help="SAIA API base URL (or set SAIA_BASE_URL env var)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_embeddings",
        help="Output directory for synthetic tools",
    )
    parser.add_argument(
        "--model",
        default=SAIA_DEFAULT_MODEL,
        help="SAIA model to use",
    )
    parser.add_argument(
        "--max-tools-to-process",
        type=int,
        default=0,
        help="Maximum number of unique normalized IDs to process (0 = all)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N normalized IDs (for batch processing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API connection and exit",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("SAIA_API_KEY") or os.getenv("EXECUTING_LLM_API_KEY")
    if not api_key:
        print(
            "Error: SAIA API key required. Set SAIA_API_KEY env var or use --api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.test:
        test_api(api_key, args.base_url, model=args.model)
        sys.exit(0)

    print("Loading normalized tool set...")
    contexts = load_normalized_tool_contexts(args.tools)

    print("Loading benchmark tasks...")
    benchmark_records = load_benchmark_records(args.benchmark)
    print(f"  Loaded {len(benchmark_records)} benchmark records")

    print("Matching each normalized ID to benchmark task context...")
    attach_benchmark_tasks(contexts, benchmark_records)
    with_tasks = sum(1 for c in contexts.values() if c.get("tasks"))
    print(f"  Matched tasks for {with_tasks}/{len(contexts)} normalized IDs")

    # Deterministic ID order.  The IDs are integers, but do not assume they are
    # perfectly contiguous.
    unique_ids = sorted(contexts)
    ids_to_process = unique_ids[args.offset :]

    if args.offset > 0:
        print(f"Skipping first {args.offset} normalized IDs (offset={args.offset})")

    if args.max_tools_to_process > 0:
        ids_to_process = ids_to_process[: args.max_tools_to_process]

    print(f"\nUnique normalized IDs selected: {len(ids_to_process)}")
    print(
        f"Target synthetic tools: {len(ids_to_process) * SYNTHETIC_TOOLS_PER_ID} "
        f"({SYNTHETIC_TOOLS_PER_ID} per ID)"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_output = output_dir / "tools_en_synthetic_candidates.jsonl"
    metadata_output = output_dir / "tools_en_synthetic_metadata.json"

    # Resume by source normalized ID rather than by offset. A source ID is skipped
    # only if the existing result file already contains the full expected number
    # of synthetic tools for that ID. Partial IDs are regenerated.
    completed_ids, existing_counts = load_completed_source_ids(synthetic_output)
    if completed_ids:
        print(
            f"Found {len(completed_ids)} already-complete normalized IDs in "
            f"{synthetic_output}; these will be skipped"
        )

    before_resume_filter = len(ids_to_process)
    ids_to_process = [normalized_id for normalized_id in ids_to_process if normalized_id not in completed_ids]
    print(
        f"Missing normalized IDs to generate: {len(ids_to_process)} "
        f"(skipped {before_resume_filter - len(ids_to_process)} already complete)"
    )

    synthetic_tools_all = []
    metadata_rows = []

    for local_idx, normalized_id in enumerate(ids_to_process, start=1):
        context = contexts[normalized_id]
        tool_name = context["representative_name"]
        tasks = context.get("tasks", [])
        co_tools = context.get("cooccurring_tools", [])

        absolute_idx = local_idx
        print(
            f"\n[{absolute_idx}/{len(ids_to_process)}] normalized_tool_id={normalized_id} "
            f"{tool_name}: generating {SYNTHETIC_TOOLS_PER_ID}"
        )
        print(
            f"  first occurrence row={context['first_row_number']}, "
            f"occurrences={context['occurrence_count']}, "
            f"tasks={len(tasks)}, co-occurring tools={len(co_tools)}"
        )

        synthetic = generate_synthetic_tools_for_id(context, api_key, args.base_url, args.model)

        if not synthetic:
            print("  Failed to generate synthetic tools")
            continue

        print(f"  Generated {len(synthetic)} synthetic tools")

        for syn_tool in synthetic:
            syn_tool["_source_tool_id"] = normalized_id
            syn_tool["_source_tool"] = tool_name
            syn_tool["_synthetic"] = True
            synthetic_tools_all.append(syn_tool)

            func = syn_tool.get("function", {})
            params = func.get("parameters", {}) if isinstance(func, dict) else {}
            param_names = list(params.get("properties", {}).keys()) if isinstance(params, dict) else []
            required_params = params.get("required", []) if isinstance(params, dict) else []

            metadata_rows.append(
                {
                    "source_normalized_tool_id": normalized_id,
                    "source_tool": tool_name,
                    "source_first_occurrence_row": context["first_row_number"],
                    "source_occurrence_count": context["occurrence_count"],
                    "benchmark_row_index": context.get("benchmark_row_index"),
                    "benchmark_task_count": len(tasks),
                    "benchmark_tasks": tasks,
                    "cooccurring_tool_ids": [
                        t.get("normalized_tool_id") for t in co_tools
                    ],
                    "cooccurring_tool_names": [
                        _tool_name(t) for t in co_tools
                    ],
                    "synthetic_tool_name": func.get("name", "unknown"),
                    "synthetic_tool_description": func.get("description", ""),
                    "parameters": param_names,
                    "required_parameters": required_params if isinstance(required_params, list) else [],
                }
            )

    print("\nWriting outputs...")

    # Always append newly generated IDs. Existing completed IDs were filtered
    # before generation, so reruns do not duplicate complete results.
    with open(synthetic_output, "a", encoding="utf-8") as f:
        for tool in synthetic_tools_all:
            f.write(json.dumps(tool, ensure_ascii=False) + "\n")

    print(
        f"  Appended {len(synthetic_tools_all)} synthetic tools to "
        f"{synthetic_output}"
    )

    if metadata_output.exists():
        try:
            with open(metadata_output, "r", encoding="utf-8") as f:
                existing_metadata = json.load(f)
            if isinstance(existing_metadata, list):
                metadata_rows = existing_metadata + metadata_rows
        except (OSError, json.JSONDecodeError):
            pass

    with open(metadata_output, "w", encoding="utf-8") as f:
        json.dump(metadata_rows, f, ensure_ascii=False, indent=2)

    print(f"  Wrote metadata to {metadata_output}")
    print(f"\nDone! Generated {len(synthetic_tools_all)} synthetic tools")


if __name__ == "__main__":
    main()
