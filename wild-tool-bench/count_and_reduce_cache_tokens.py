#!/usr/bin/env python3
"""
Count tokens in tool_schemas_cache.json using the gpt-oss-20b tokenizer (o200k_harmony),
then randomly remove 400 synthetic tools and recount.

NOTE on token counts
--------------------
The InContextToolSelector passes ALL tools stored in the cache (including synthetic
ones) directly to the LLM.  The tiktoken library counts raw-JSON tokens; the actual
OpenAI-compatible API counts tool definitions using an internal TypeScript-like
format which adds roughly 38 % more tokens.  The scaling factor is derived from the
observed benchmark data: 618 tools → 111,369 API input tokens vs 80,214 raw-JSON
tokens (ratio ≈ 1.388).  Both raw and estimated-API counts are printed below.

Usage:
    python count_and_reduce_cache_tokens.py [--seed SEED] [--save]

Options:
    --seed SEED   Random seed for reproducibility (default: 42)
    --save        Write the reduced cache back to tool_schemas_cache.json
                  (a .bak backup is created first)
"""

import argparse
import json
import os
import random
import shutil

import tiktoken

CACHE_FILE = os.path.join(
    os.path.dirname(__file__),
    "wtb", "model_handler", "api_inference", "tool_schemas_cache.json",
)

# Private metadata fields added by the pipeline – NOT sent to the API
_PRIVATE_FIELDS = {"_synthetic", "_source_tool"}

# Observed calibration: 618 non-synthetic tools → 111,369 API tokens, 80,214 raw tokens.
# The API serialises tool definitions in a verbose TypeScript-like format internally.
_API_FORMAT_SCALE = 111_369 / 80_214  # ≈ 1.388


def strip_private_fields(tool: dict) -> dict:
    """Return a copy of *tool* without internal metadata fields."""
    return {k: v for k, v in tool.items() if k not in _PRIVATE_FIELDS}


def count_tokens_raw(tools: list, enc: tiktoken.Encoding) -> int:
    """Count tokens for the tools array serialised as plain JSON."""
    cleaned = [strip_private_fields(t) for t in tools]
    payload = json.dumps(cleaned, ensure_ascii=False)
    return len(enc.encode(payload))


def estimate_api_tokens(raw_tokens: int) -> int:
    """Estimate the token count the API would report for the same tool list."""
    return round(raw_tokens * _API_FORMAT_SCALE)


def print_counts(label: str, n_tools: int, raw: int) -> None:
    api_est = estimate_api_tokens(raw)
    print(f"  {label}")
    print(f"    Tools            : {n_tools:>7,}")
    print(f"    Raw-JSON tokens  : {raw:>7,}")
    print(f"    Est. API tokens  : {api_est:>7,}  (×{_API_FORMAT_SCALE:.3f} scale)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist the reduced cache in-place (creates .bak backup first)",
    )
    args = parser.parse_args()

    # ── Load cache ──────────────────────────────────────────────────────────
    print(f"Cache file : {CACHE_FILE}")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tools: list = data["tools"]
    synthetic_tools     = [t for t in tools if     t.get("_synthetic", False)]
    non_synthetic_tools = [t for t in tools if not t.get("_synthetic", False)]

    print(f"\n{'─'*62}")
    print(f"  Total tools      : {len(tools):>6,}")
    print(f"  Synthetic        : {len(synthetic_tools):>6,}")
    print(f"  Non-synthetic    : {len(non_synthetic_tools):>6,}")
    print(f"{'─'*62}")

    # ── Tokenizer ───────────────────────────────────────────────────────────
    enc = tiktoken.encoding_for_model("gpt-oss-20b")
    print(f"\nTokenizer  : {enc.name}  (gpt-oss-20b)")
    print(f"API scale  : {_API_FORMAT_SCALE:.3f}  "
          f"(calibrated: 618 tools → 111,369 observed API tokens)")

    # ── Token count BEFORE removal ───────────────────────────────────────────
    raw_before = count_tokens_raw(tools, enc)
    print(f"\n{'─'*62}  BEFORE")
    print_counts("Full cache (all tools sent to LLM via InContextSelector)",
                 len(tools), raw_before)
    print(f"{'─'*62}")

    # ── Remove 400 random synthetic tools ───────────────────────────────────
    remove_count = 400
    if len(synthetic_tools) < remove_count:
        raise ValueError(
            f"Cannot remove {remove_count} synthetic tools – "
            f"only {len(synthetic_tools)} available."
        )

    rng = random.Random(args.seed)
    to_remove = {id(t) for t in rng.sample(synthetic_tools, remove_count)}
    reduced_tools       = [t for t in tools if id(t) not in to_remove]
    remaining_synthetic = [t for t in reduced_tools if t.get("_synthetic", False)]

    print(f"\nRemoving {remove_count} random synthetic tools (seed={args.seed}) …")

    # ── Token count AFTER removal ────────────────────────────────────────────
    raw_after = count_tokens_raw(reduced_tools, enc)
    saved_raw = raw_before - raw_after
    saved_api = estimate_api_tokens(raw_before) - estimate_api_tokens(raw_after)

    print(f"\n{'─'*62}  AFTER")
    print_counts("Reduced cache", len(reduced_tools), raw_after)
    print(f"{'─'*62}")
    print(f"\n  Δ tools removed  : {remove_count:>7,}  "
          f"(synthetic remaining: {len(remaining_synthetic):,})")
    print(f"  Δ raw tokens     : -{saved_raw:>6,}  "
          f"({saved_raw / raw_before * 100:.1f}%)")
    print(f"  Δ est. API tokens: -{saved_api:>6,}  "
          f"({saved_api / estimate_api_tokens(raw_before) * 100:.1f}%)")
    print(f"  Avg API tok/tool : {saved_api / remove_count:.1f}")
    print()

    # ── Optionally save ──────────────────────────────────────────────────────
    if args.save:
        backup = CACHE_FILE + ".bak"
        shutil.copy2(CACHE_FILE, backup)
        print(f"Backup   → {backup}")

        data["tools"] = reduced_tools
        data["count"] = len(reduced_tools)

        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved    → {CACHE_FILE}")
    else:
        print(f"(Dry run – pass --save to overwrite {CACHE_FILE})")


if __name__ == "__main__":
    main()
