# LangGraph Tool Selection Modes

This document describes the four tool selection strategies available in the LangGraph plugin.

## Overview

The LangGraph plugin supports 4 different tool selection modes to control how tools are selected before being passed to the LLM. Each mode has different trade-offs in terms of cost, latency, and accuracy.

## Modes

### 1. In-Context Selection (Default)

**Mode name:** `in_context`

The simplest approach: all available tools are passed to the LLM, and the LLM decides which ones to use based on the user query.

**Pros:**
- No additional LLM calls needed
- Works with any LLM that supports tool calling
- Lowest latency
- Highest accuracy (LLM has all context)

**Cons:**
- Can be expensive with large tool sets (many tokens)
- May lead to LLM confusion with many irrelevant tools
- No explicit tool filtering

**Configuration:**
```bash
export LANGGRAPH_TOOL_SELECTION_MODE=in_context
```

**When to use:**
- Small tool sets (< 20 tools)
- When accuracy is critical
- When you want to minimize infrastructure complexity

---

### 2. Hierarchical Selection

**Mode name:** `hierarchical`

Uses a smaller/faster LLM (e.g., a fine-tuned model or distilled model) to select relevant tools first, then passes only the selected tools to the main LLM.

**Pros:**
- Reduces token count to main LLM
- Can use cheaper model for selection
- Faster with large tool sets
- Reduces LLM confusion

**Cons:**
- Additional LLM call required
- Selection LLM might miss relevant tools
- Higher latency (two sequential LLM calls)
- Requires separate selector LLM endpoint

**Configuration:**
```bash
export LANGGRAPH_TOOL_SELECTION_MODE=hierarchical
export LANGGRAPH_SELECTOR_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
export LANGGRAPH_SELECTOR_LLM_API_KEY=sk-...
```

**When to use:**
- Large tool sets (50-200 tools)
- When you have a fast selector LLM available
- When cost matters and tool selection is straightforward

---

### 3. Embedding-Based Selection

**Mode name:** `embedding`

Uses semantic similarity based on embeddings to retrieve the most similar tools for a given query.

**Pros:**
- Fast retrieval (no LLM call)
- Scalable to very large tool sets
- No sequential calls
- Consistent and deterministic

**Cons:**
- Requires embedding model/endpoint
- May miss tools if query semantics differ
- Cannot handle complex reasoning about tool requirements
- Limited to top-k most similar tools

**Configuration:**
```bash
export LANGGRAPH_TOOL_SELECTION_MODE=embedding
export LANGGRAPH_EMBEDDING_ENDPOINT=http://localhost:8000/v1/embeddings
export LANGGRAPH_EMBEDDING_API_KEY=sk-...
```

**When to use:**
- Very large tool sets (200+ tools)
- When you want minimal latency
- When tool descriptions are well-aligned with use cases
- Budget-constrained scenarios

---

### 4. Embedding with LLM Reranking

**Mode name:** `embedding_reranker`

Combines embedding-based retrieval for speed with LLM-based reranking for accuracy. Fast embedding retrieval identifies candidates, then an LLM reranks them by relevance.

**Pros:**
- Best of both worlds: speed + accuracy
- Handles edge cases where embeddings fail
- Good for large tool sets
- Better accuracy than embeddings alone

**Cons:**
- Requires both embedding and LLM endpoints
- Still requires an LLM call (but on smaller candidate set)
- More complex setup
- Higher latency than embeddings alone

**Configuration:**
```bash
export LANGGRAPH_TOOL_SELECTION_MODE=embedding_reranker
export LANGGRAPH_EMBEDDING_ENDPOINT=http://localhost:8000/v1/embeddings
export LANGGRAPH_EMBEDDING_API_KEY=sk-...
export LANGGRAPH_SELECTOR_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
export LANGGRAPH_SELECTOR_LLM_API_KEY=sk-...
```

**When to use:**
- Large tool sets where accuracy matters (100-500 tools)
- When you want to balance latency and accuracy
- Production scenarios with strict SLAs

---

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `LANGGRAPH_TOOL_SELECTION_MODE` | Selection strategy | No | `in_context` |
| `LANGGRAPH_ENDPOINT` | LangGraph server URL | Yes | - |
| `LANGGRAPH_API_KEY` | Authentication token | No | - |
| `LANGGRAPH_LLM_ENDPOINT` | Main LLM endpoint | Yes | - |
| `LANGGRAPH_LLM_API_KEY` | Main LLM auth | No | - |
| `LANGGRAPH_SELECTOR_LLM_ENDPOINT` | Selector LLM endpoint (hierarchical/reranker) | Conditional | - |
| `LANGGRAPH_SELECTOR_LLM_API_KEY` | Selector LLM auth | No | - |
| `LANGGRAPH_EMBEDDING_ENDPOINT` | Embedding endpoint (embedding/reranker) | Conditional | - |
| `LANGGRAPH_EMBEDDING_API_KEY` | Embedding auth | No | - |
| `LANGGRAPH_HOST` | Server host | No | `127.0.0.1` |
| `LANGGRAPH_PORT` | Server port | No | `8001` |

### Runtime Override

You can also override the mode per-request by including it in the request payload:

```json
{
  "input": {
    "messages": [...],
    "tools": [...]
  },
  "selection_mode": "embedding_reranker"
}
```

---

## Comparison Table

| Metric | In-Context | Hierarchical | Embedding | Embedding + Rerank |
|--------|-----------|--------------|-----------|-------------------|
| Tool Set Size | Small (< 20) | Medium (50-200) | Large (200+) | Large (100-500) |
| Latency | Very Low | Medium | Very Low | Low |
| Accuracy | High | Medium | Medium | High |
| Cost | High | Low | Very Low | Low |
| Infrastructure | Simple | Moderate | Moderate | Complex |
| Sequential Calls | 1 | 2 | 0 | 1 |
| Scalability | Poor | Good | Excellent | Excellent |

---

## Best Practices

1. **Start with In-Context**: Begin with mode 1 for initial development. It's the simplest and most reliable.

2. **Profile Performance**: Measure latency and accuracy for your specific use case before switching modes.

3. **Use Hierarchical for Moderate Sets**: When you have 50-200 tools and need cost efficiency.

4. **Cache Embeddings**: If using embedding modes, cache tool embeddings to avoid recomputation.

5. **Monitor Quality**: Track which mode gives the best tool selection accuracy in production.

6. **Graceful Fallback**: Hierarchical and embedding modes fall back to returning all tools on error.

---

## Example: Switching Modes

### Development
```bash
# Simple, reliable
export LANGGRAPH_TOOL_SELECTION_MODE=in_context
python langgraph_app.py
```

### Testing with Selector
```bash
# Using hierarchical selection
export LANGGRAPH_TOOL_SELECTION_MODE=hierarchical
export LANGGRAPH_SELECTOR_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
python langgraph_app.py
```

### Production with Embeddings
```bash
# Scalable, fast
export LANGGRAPH_TOOL_SELECTION_MODE=embedding_reranker
export LANGGRAPH_EMBEDDING_ENDPOINT=http://localhost:8000/v1/embeddings
export LANGGRAPH_SELECTOR_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
python langgraph_app.py
```

---

## Graph Architecture

All modes use the same 2-node graph structure:

```
┌─────────────────────────────────────────────────────────────┐
│                          Graph Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Messages + All Tools                                        │
│          ↓                                                    │
│  ┌──────────────────────────────────┐                        │
│  │  Tool Selection Node             │                        │
│  │  (Strategy-dependent selection)  │                        │
│  └──────────────────────────────────┘                        │
│          ↓                                                    │
│  Selected Tools (filtered set)                               │
│          ↓                                                    │
│  ┌──────────────────────────────────┐                        │
│  │  LLM Execution Node              │                        │
│  │  (Invokes LLM with selected tools)                        │
│  └──────────────────────────────────┘                        │
│          ↓                                                    │
│  Response + Tool Calls                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Node 1: Tool Selection**
- Input: messages, tools, selection_mode
- Output: selected_tools
- Behavior: Depends on mode (all tools, LLM-selected, embedding-selected, etc.)

**Node 2: LLM Execution**
- Input: messages, selected_tools
- Output: response, tool_calls
- Behavior: Invokes LLM with selected tools only

---

## Troubleshooting

### Mode not taking effect
- Check environment variable is set: `echo $LANGGRAPH_TOOL_SELECTION_MODE`
- Restart the server after changing env vars
- Use runtime override in request if needed

### Hierarchical mode failing
- Verify `LANGGRAPH_SELECTOR_LLM_ENDPOINT` is accessible
- Check selector LLM response format matches expectations
- Review server logs for parse errors

### Embedding mode not working
- Verify `LANGGRAPH_EMBEDDING_ENDPOINT` is accessible
- Check embedding response contains `embedding` or `data[].embedding` field
- Ensure sklearn is installed: `pip install scikit-learn`

### Low tool selection accuracy
- Consider embedding mode needs good tool descriptions
- Try reranker mode for better accuracy
- Increase top_k for embedding modes if needed
- Fall back to in_context mode to debug

