# OpenAI Embedding-Based Tool Selection

This document describes the implementation of the OpenAI `text-embedding-3-small` embedding-based tool selection strategy for the LangGraph framework.

## Overview

The embedding-based tool selection strategy uses semantic embeddings to rank and select the most relevant tools for a given user query from the request-provided tool set.

### Key Design Principle
**Use request-provided tools as the baseline**. The difference between modes is the **selection strategy**:

- **`in_context` mode**: Returns ALL request-provided tools, LLM decides which to use
- **`embedding` mode**: Ranks request tools by semantic relevance to query, returns top-k selected tools

### Why This Design?
- Request tools have valid, complete schemas
- Embedding mode provides intelligent ranking without schema issues
- Both modes operate on the same baseline (request-provided tools)

## Architecture

### Key Design: All Modes Load All 618 Valid Tools

**Every selector strategy now loads ALL 618 valid tools from schema cache:**

1. **InContextToolSelector**: Returns all 618 tools
   - Loads tools from schema cache
   - Returns entire tool set to LLM
   - LLM decides which tools to use in-context
   - **Use when**: LLM should see all available tools

2. **HierarchicalToolSelector**: Uses selector LLM to pick from 618 tools
   - Loads all 618 tools from schema cache
   - Extracts user query from message history
   - Calls selector LLM to identify relevant tools
   - Returns selected tools (max 10) to main LLM
   - **Use when**: You have a smaller/faster selector LLM available

3. **OpenAIEmbeddingBasedToolSelector**: Ranks all 618 tools by semantic relevance
   - Loads all 618 tools from schema cache
   - Computes query embedding using OpenAI's `text-embedding-3-small`
   - Ranks all 618 tools by cosine similarity to query
   - Returns top-k most relevant tools (default: 5)
   - Caches embeddings locally in `tool_embeddings_cache.json` (740 precomputed entries)
   - Uses scikit-learn for cosine similarity computation
   - **Use when**: You want semantic ranking of tools by query relevance

### Supporting Components

**Schema Preparation Script**: `prepare_tool_schemas.py` 
- Loads tools from `tools_en.jsonl` (1248 raw tools)
- Fixes invalid schema types (e.g., `"float"` → `"number"`)
- Deduplicates tools by name (1248 → 618 unique)
- Saves 618 valid tools to `tool_schemas_cache.json` (0.49 MB)
- **Run once to prepare all tool schemas**: `python -m wtb.model_handler.api_inference.prepare_tool_schemas`

**Embedding Setup Script**: `setup_openai_embeddings.py` (optional)
- Precomputes embeddings for all 618 tools (cache warming)
- Improves cold-start performance for embedding selector
- Can be run anytime to refresh cache

## Prerequisites

### Required Packages

```bash
pip install openai scikit-learn numpy
```

### Environment Variables

You need to set these environment variables:

```bash
# Required: Your OpenAI API key (with embedding model access)
export OPENAI_API_KEY=sk-...

# Optional: Custom OpenAI base URL (default: https://api.openai.com/v1)
export OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Set the tool selection mode to use embeddings
export LANGGRAPH_TOOL_SELECTION_MODE=embedding
```

## Setup Steps

### Step 1: Prepare Your Tools File

Ensure you have a tools JSONL file with the following format:

```json
{"function": {"name": "getRealtimeCity", "description": "Get the real-time weather conditions of a specified city.", "parameters": {...}}, "type": "function"}
{"function": {"name": "getHistoricalCity", "description": "Get historical weather data for a specified city.", "parameters": {...}}, "type": "function"}
```

### Step 2: Set Environment Variables

```bash
export OPENAI_API_KEY=your_openai_api_key
```

### Step 3: Run the Setup Script

```bash
# From the wild-tool-bench directory
python -m wtb.model_handler.api_inference.setup_openai_embeddings \
    --tools-file multi-agent-framework/tools/tools_en.jsonl
```

Or if running directly:

```bash
cd wild-tool-bench/wtb/model_handler/api_inference
python setup_openai_embeddings.py --tools-file ../../../multi-agent-framework/tools/tools_en.jsonl
```

The script will:
1. Validate your OpenAI API key
2. Load all tools from the JSONL file
3. Embed each tool using `text-embedding-3-small`
4. Save embeddings to `tool_embeddings_cache.json` (default location)
5. Show progress and timing information

**Example Output:**
```
[OPENAI EMBEDDING SETUP] Starting to embed 180 tools...
[OPENAI EMBEDDING SETUP] Model: text-embedding-3-small
[OPENAI EMBEDDING SETUP] Cache file: tool_embeddings_cache.json
  [1/180] getRealtimeCity - OK
  [2/180] getHistoricalCity - OK
  ...
[OPENAI EMBEDDING SETUP] Completed! 180 embeddings cached.
```

## Quick Test: Start the Server with Embedding Mode

**Terminal 1: Start the LangGraph server with embedding mode**

```bash
cd wild-tool-bench
LANGGRAPH_TOOL_SELECTION_MODE=embedding \
LANGGRAPH_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions \
python -m wtb.model_handler.api_inference.langgraph_app
```

**Terminal 2: Run evaluation with embedding mode**

```bash
cd wild-tool-bench
LANGGRAPH_TOOL_SELECTION_MODE=embedding \
python3 -u -m wtb.openfunctions_evaluation --model langgraph
```

## Setup Instructions

### 1. Prepare Tool Schemas (One-time)

Convert all tools to valid JSON Schema format and save to cache:

```bash
cd wild-tool-bench
source .venv/bin/activate
python -m wtb.model_handler.api_inference.prepare_tool_schemas
```

Output:
```
Loading tools from JSONL...
Loaded 618 unique tools (skipped 630 duplicates)
Validating schemas...
  ✓ All schemas valid!
Saving to cache...
✓ Saved 618 tools to tool_schemas_cache.json
  File size: 0.49 MB
```

### 2. (Optional) Precompute Tool Embeddings

Warm up the embedding cache for faster cold-start:

```bash
export OPENAI_API_KEY="your-api-key"
python -m wtb.model_handler.api_inference.setup_openai_embeddings
```

This precomputes embeddings for all 618 tools (~$0.016 one-time cost).

### 3. Run with Embedding Mode

```bash
export OPENAI_API_KEY="your-api-key"
export LANGGRAPH_TOOL_SELECTION_MODE=embedding

# Option A: Start server
python -m wtb.model_handler.api_inference.langgraph_app

# Option B: Run evaluation
python -m wtb.openfunctions_evaluation --model langgraph
```

## Usage at Runtime

### Embedding Mode Flow

```
Request → Extract Query → Compute Query Embedding
  → Score All 618 Tools by Similarity → Rank by Score
  → Select Top-k (Deduplicated) → Return Selected Tools to LLM
```

Example output:
```
[OPENAI EMBEDDING SELECTOR]
  Query: How do I get the weather forecast?
  Total available tools: 618
  Top-k (unique): 5
  Cache hits: 618, Runtime computed: 0
  Selected tools (ranked by relevance):
    1. getCityForecast (similarity: 0.9234)
    2. getRealtimeCity (similarity: 0.8891)
    3. getHistoricalCity (similarity: 0.7654)
    4. getWeatherAlerts (similarity: 0.7421)
    5. getAirQuality (similarity: 0.6892)
```

### Key Differences

| Aspect | in_context | embedding |
|--------|-----------|-----------|
| Tools source | Request payload (7-10 tools) | All 618 valid tools |
| Selection strategy | None (return all) | Rank by query relevance |
| Tools passed to LLM | All request tools | Top-5 most relevant |
| LLM cognitive load | Higher (many tools) | Lower (few relevant tools) |
| Relevance ranking | None (generic) | Query-specific |

## Configuration Options

### Cache File Location

By default, embeddings are cached in `tool_embeddings_cache.json` in the script directory. To use a custom location:

```python
from wtb.model_handler.api_inference.langgraph_app import OpenAIEmbeddingBasedToolSelector

selector = OpenAIEmbeddingBasedToolSelector(
    top_k=5,
    cache_file="/path/to/custom/cache.json"
)
```

### Top-K Selection

By default, the top-5 most similar tools are returned. To change this:

**Via environment** (if you modify the server):
- Edit `_create_tool_selector()` function in `langgraph_app.py`

**Via Python API**:
```python
selector = OpenAIEmbeddingBasedToolSelector(top_k=10)
```

## Cost Estimation

### OpenAI Embedding Pricing

As of 2024, OpenAI's `text-embedding-3-small` model costs:
- **$0.02 per 1 million input tokens**

### Example Costs

For 180 tools (each with ~50 tokens):
- **Setup**: 9,000 tokens × $0.02/M = **$0.00018** (one-time)
- **Runtime per query**: 200 tokens (query + tools) × $0.02/M = **$0.000004** (per request, cached)

## Best Practices

1. **Caching**: Always run the setup script once to cache tool embeddings. This significantly reduces runtime costs and latency.

2. **Query Length**: The tool selector uses the first 500 characters of the user query. Longer queries are truncated.

3. **Tool Descriptions**: Ensure all tools have meaningful descriptions in the JSONL file for better embeddings.

4. **Batch Size**: The setup script processes tools in batches of 100 for efficiency.

5. **Error Handling**: 
   - If a tool embedding is missing at runtime, it will be computed on-the-fly (slower but functional)
   - Failed embeddings fall back to zero vectors (worst-case, but doesn't break the system)

## Troubleshooting

### "OPENAI_API_KEY environment variable is not set"

**Solution**: Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
```

### "Failed to embed query: 401 Unauthorized"

**Solution**: Your API key is invalid or doesn't have embedding model access. Check:
1. API key is correct
2. Account has billing enabled
3. `text-embedding-3-small` model is available in your region

### "ModuleNotFoundError: No module named 'openai'"

**Solution**: Install the OpenAI package:
```bash
pip install openai
```

### Cache file not found at runtime

**Solution**: 
1. Ensure you ran the setup script successfully
2. Verify the cache file path is correct
3. Check file permissions

### Tools not being selected accurately

**Solution**:
1. Ensure tool descriptions are specific and descriptive
2. Run the setup script again to update embeddings
3. Check if your queries are being truncated (>500 chars)

## Advanced Usage

### Using Custom Cache File

```bash
python -m wtb.model_handler.api_inference.setup_openai_embeddings \
    --tools-file multi-agent-framework/tools/tools_en.jsonl \
    --cache-file /custom/path/embeddings.json
```

### Monitoring Runtime Costs

The runtime selector logs cache statistics:
```
Cache hits: 180, Runtime computed: 0
```

- **Cache hits**: Tools retrieved from cached embeddings (free)
- **Runtime computed**: Tools that needed to be embedded at runtime (costs API call)

High runtime computed values indicate the cache is incomplete.

## Comparison with Other Modes

| Mode | Speed | Cost | Accuracy | Complexity |
|------|-------|------|----------|------------|
| in_context | ⭐⭐⭐⭐⭐ | Free | Low | Low |
| hierarchical | ⭐⭐ | $$ | Medium | High |
| **embedding** | **⭐⭐⭐⭐** | **$** | **High** | **Medium** |
| embedding_reranker | ⭐ | $$$$ | High | Very High |

## See Also

- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [scikit-learn cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
