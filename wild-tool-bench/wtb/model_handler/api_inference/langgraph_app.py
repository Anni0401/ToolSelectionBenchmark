import os
import json
import time
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from abc import ABC, abstractmethod
from typing import TypedDict, Any, Optional

# Per-request context (thread-safe).  Set in do_POST, read by log helpers.
_request_context = threading.local()

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    from wtb.constant import DOTENV_PATH
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=False, override=True)
except Exception:
    # If DOTENV_PATH not available, try loading .env from current directory
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ==================== Tool Selection Logging ====================

def _get_log_dir():
    """Return the per-strategy log directory (mirrors result/<strategy>/ layout)."""
    try:
        from wtb.constant import RESULT_PATH
        base = RESULT_PATH
    except Exception:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.join(script_dir, "..", "..", "result")

    strategy = getattr(_request_context, "selection_mode", None) or "unknown"
    log_dir = os.path.join(base, strategy)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _get_tool_selection_log_path():
    """Get the path to the tool selection log file (per-strategy subfolder)."""
    return os.path.join(_get_log_dir(), "tool_selection_logs.jsonl")


def _get_tool_call_log_path():
    """Get the path to the tool call execution log file (per-strategy subfolder)."""
    return os.path.join(_get_log_dir(), "tool_call_logs.jsonl")


def log_tool_calls(messages_sent: list, tool_calls_returned: list, content: str,
                   input_tokens: int, output_tokens: int, model: str = None):
    """Log the actual tool calls the agent executes (LLM request/response).

    Captures:
      - The tool calls already present in the message history (prior turns)
      - The new tool calls returned by the LLM in this turn
      - Content, token counts and model name for diagnostics
    """
    try:
        log_path = _get_tool_call_log_path()

        # Collect tool calls from message history (prior turns)
        history_tool_calls = []
        for msg in messages_sent:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    history_tool_calls.append({
                        "id": tc.get("id"),
                        "type": tc.get("type"),
                        "name": func.get("name") if isinstance(func, dict) else None,
                        "raw_keys": list(tc.keys()),
                    })

        # Summarise the new tool calls returned by the LLM
        returned_summary = []
        for tc in (tool_calls_returned or []):
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            returned_summary.append({
                "id": tc.get("id") if isinstance(tc, dict) else None,
                "type": tc.get("type") if isinstance(tc, dict) else None,
                "name": func.get("name") if isinstance(func, dict) else None,
                "arguments": func.get("arguments") if isinstance(func, dict) else None,
                "raw_keys": list(tc.keys()) if isinstance(tc, dict) else [],
            })

        log_entry = {
            "timestamp": time.time(),
            "test_entry_id": getattr(_request_context, "test_entry_id", None),
            "task_idx": getattr(_request_context, "task_idx", None),
            "selection_mode": getattr(_request_context, "selection_mode", None),
            "model": model,
            "messages_sent_count": len(messages_sent),
            "history_tool_calls_count": len(history_tool_calls),
            "history_tool_calls": history_tool_calls,
            "returned_tool_calls_count": len(returned_summary),
            "returned_tool_calls": returned_summary,
            "content_length": len(content) if content else 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARNING] Failed to log tool calls: {e}")


def log_tool_selection(strategy_name: str, query: str, available_tools_count: int, 
                      selected_tools: list, selection_metadata: dict = None):
    """Log tool selection results for quality control.
    
    Args:
        strategy_name: Name of the selection strategy (e.g., 'in_context', 'hierarchical')
        query: The user query that was used for selection
        available_tools_count: Total number of tools available for selection
        selected_tools: List of selected tool objects
        selection_metadata: Optional dict with additional metadata (model, latency, etc.)
    """
    try:
        log_path = _get_tool_selection_log_path()
        
        # Extract tool names
        selected_tool_names = [
            tool.get("function", {}).get("name", "unknown")
            for tool in selected_tools
        ]
        
        # Create log entry
        log_entry = {
            "timestamp": time.time(),
            "test_entry_id": getattr(_request_context, "test_entry_id", None),
            "task_idx": getattr(_request_context, "task_idx", None),
            "strategy": strategy_name,
            "query": query[:500],  # Limit query length
            "available_tools_count": available_tools_count,
            "selected_tools_count": len(selected_tools),
            "selected_tool_names": selected_tool_names,
            "metadata": selection_metadata or {}
        }
        
        # Append to JSONL file (thread-safe)
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        print(f"[WARNING] Failed to log tool selection: {e}")


# ==================== Tool Selection Strategies ====================

class ToolSelector(ABC):
    """Base class for tool selection strategies."""
    
    @abstractmethod
    def select(self, messages: list, tools: list) -> list:
        """Select tools based on messages and strategy.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of available tools
            
        Returns:
            List of selected tools
        """
        pass


class InContextToolSelector(ToolSelector):
    """Strategy 1: In-context selection by the executing LLM.
    
    Returns all 618 valid tools from schema cache - the LLM decides which to use in-context.
    """
    
    def __init__(self, schema_cache_file: str = None):
        self.schema_cache_file = schema_cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_schemas_cache.json"
        )
        self.tools_cache = None
        self._load_schema_cache()
    
    def _load_schema_cache(self):
        """Load all valid tools from schema cache file."""
        if os.path.exists(self.schema_cache_file):
            try:
                with open(self.schema_cache_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "tools" in data:
                        self.tools_cache = data["tools"]
                        print(f"[IN-CONTEXT SELECTOR] Loaded {len(self.tools_cache)} valid tools from schema cache")
                    else:
                        self.tools_cache = data if isinstance(data, list) else []
                        print(f"[IN-CONTEXT SELECTOR] Loaded {len(self.tools_cache)} tools from schema cache")
            except Exception as e:
                print(f"[WARNING] Failed to load schema cache: {e}")
                self.tools_cache = []
        else:
            print(f"[WARNING] Schema cache file not found: {self.schema_cache_file}")
            self.tools_cache = []
    
    def select(self, messages: list, tools: list) -> list:
        """Return all valid tools from schema cache for in-context selection."""
        # Use schema cache tools instead of request tools
        all_tools = self.tools_cache if self.tools_cache else tools
        
        # Extract query for logging
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")[:500]
                break
        
        # Log tool selection
        log_tool_selection(
            strategy_name="in_context",
            query=query,
            available_tools_count=len(all_tools),
            selected_tools=all_tools,
            selection_metadata={"method": "pass_through"}
        )
        
        print(f"\n[IN-CONTEXT SELECTOR]")
        print(f"  Total available tools: {len(all_tools)}")
        print(f"  Passing all {len(all_tools)} tools to LLM for in-context selection")
        if len(all_tools) <= 5:
            for tool in all_tools:
                tool_name = tool.get("function", {}).get("name", "unknown")
                print(f"    - {tool_name}")
        else:
            for tool in all_tools[:5]:
                tool_name = tool.get("function", {}).get("name", "unknown")
                print(f"    - {tool_name}")
            print(f"    ... and {len(all_tools) - 5} more")
        print()
        return all_tools


class HierarchicalToolSelector(ToolSelector):
    """Strategy 2: Hierarchical selection with a smaller LLM.
    
    Uses a smaller/faster LLM to select which tools are relevant from all 618 valid tools,
    then passes only those to the main LLM.
    """
    
    def __init__(self, schema_cache_file: str = None):
        self.endpoint = os.getenv("LANGGRAPH_SELECTOR_LLM_ENDPOINT")
        self.api_key = os.getenv("LANGGRAPH_SELECTOR_LLM_API_KEY")
        self.model = os.getenv("LANGGRAPH_SELECTOR_LLM_MODEL", "Qwen/Qwen3-30B-A3B")
        self.schema_cache_file = schema_cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_schemas_cache.json"
        )
        self.tools_cache = None
        self._load_schema_cache()
    
    def _load_schema_cache(self):
        """Load all valid tools from schema cache file."""
        if os.path.exists(self.schema_cache_file):
            try:
                with open(self.schema_cache_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "tools" in data:
                        self.tools_cache = data["tools"]
                        print(f"[HIERARCHICAL SELECTOR] Loaded {len(self.tools_cache)} valid tools from schema cache")
                    else:
                        self.tools_cache = data if isinstance(data, list) else []
                        print(f"[HIERARCHICAL SELECTOR] Loaded {len(self.tools_cache)} tools from schema cache")
            except Exception as e:
                print(f"[WARNING] Failed to load schema cache: {e}")
                self.tools_cache = []
        else:
            print(f"[WARNING] Schema cache file not found: {self.schema_cache_file}")
            self.tools_cache = []
    
    def select(self, messages: list, tools: list) -> list:
        """Use a small LLM to select relevant tools from all 618 valid tools."""
        if not self.endpoint:
            raise ValueError("LANGGRAPH_SELECTOR_LLM_ENDPOINT environment variable must be set for hierarchical mode")
        
        # Use schema cache tools instead of request tools
        all_tools = self.tools_cache if self.tools_cache else tools
        
        if not all_tools:
            return []
        
        # Build prompt for tool selection
        query = self._extract_query(messages)
        tool_descriptions = self._format_tools(all_tools)
        
        selection_prompt = f"""Given the user query, select the most relevant tools from the available list. /no_think
        
User Query: {query}

Available Tools:
{tool_descriptions}

Return a JSON array of tool names you would use, e.g. ["getTool1", "getTool2"].
Return ONLY the JSON array, no other text."""
        
        response = self._invoke_selector_llm(selection_prompt)
        # Strip <think>...</think> blocks produced by reasoning models (e.g. Qwen3)
        import re as _re
        clean = _re.sub(r"<think>.*?</think>", "", response, flags=_re.DOTALL).strip()
        # Extract the first JSON array found in the remaining text
        m = _re.search(r"\[.*?\]", clean, _re.DOTALL)
        if not m:
            print(f"[WARNING] Selector LLM returned no JSON array; got: {clean[:200]}")
            selected_names = []
        else:
            selected_names = json.loads(m.group())
        
        # Filter tools by selected names
        selected = [t for t in all_tools if t.get("function", {}).get("name") in selected_names]
        
        # Log tool selection
        log_tool_selection(
            strategy_name="hierarchical",
            query=query,
            available_tools_count=len(all_tools),
            selected_tools=selected,
            selection_metadata={
                "model": self.model,
                "selected_tool_names": selected_names
            }
        )
        
        # Log the selection results
        print(f"\n[HIERARCHICAL SELECTOR]")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Available tools: {len(all_tools)}")
        print(f"  Selected tool names: {selected_names}")
        print(f"  Matched tools: {len(selected)}")
        for tool in selected[:5]:
            tool_name = tool.get("function", {}).get("name", "unknown")
            print(f"    - {tool_name}")
        if len(selected) > 5:
            print(f"    ... and {len(selected) - 5} more")
        print()
        
        return selected
    
    def _extract_query(self, messages: list) -> str:
        """Extract the main query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:500]
        return ""
    
    def _format_tools(self, tools: list) -> str:
        """Format tools for the selector LLM."""
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
    
    def _invoke_selector_llm(self, prompt: str) -> str:
        """Call the selector LLM endpoint using OpenAI-compatible format."""
        print(f"\n[SELECTOR LLM] Calling: {self.endpoint}")
        print(f"[SELECTOR LLM] Model: {self.model}")
        print(f"[SELECTOR LLM] Has API Key: {bool(self.api_key)}")
        
        # OpenAI-compatible format (works with HF router, DeepSeek, etc.)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2000,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "LangGraph-Selector/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        print(f"[SELECTOR LLM] Payload size: {len(body) / 1024:.1f} KB, sending to {self.endpoint}")
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = resp.read().decode("utf-8")
                parsed = json.loads(data)
                
                print(f"[DEBUG] Selector response received (status: {resp.status})")
                
                # Extract content from OpenAI-compatible response
                if isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                    first = parsed["choices"][0]
                    finish_reason = first.get("finish_reason")
                    usage = parsed.get("usage", {})
                    print(f"[DEBUG] finish_reason={finish_reason}, prompt_tokens={usage.get('prompt_tokens')}, completion_tokens={usage.get('completion_tokens')}")
                    if isinstance(first, dict) and "message" in first:
                        msg = first["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            has_think = "<think>" in content
                            print(f"[DEBUG] has_think={has_think}, content[:200]={content[:200]}")
                            return content
                
                # Fallback: return raw data
                print(f"[DEBUG] Unexpected response format, returning raw: {str(parsed)[:200]}")
                return str(parsed)
                
        except urllib.error.HTTPError as exc:
            error_data = exc.read().decode("utf-8")
            print(f"[ERROR] Selector HTTP {exc.code}: {error_data[:300]}")
            raise RuntimeError(f"Selector LLM request failed: {exc.code} {exc.reason} - {error_data[:200]}")
        except urllib.error.URLError as exc:
            print(f"[ERROR] Selector URL Error: {exc.reason}")
            print(f"[ERROR] Make sure:")
            print(f"[ERROR]   1. Internet connectivity is available")
            print(f"[ERROR]   2. The endpoint URL is correct: {self.endpoint}")
            print(f"[ERROR]   3. The API key is valid: {self.api_key[:10]}..." if self.api_key else "[ERROR]   3. API key is missing")
            raise RuntimeError(f"Selector LLM request failed (connection): {exc.reason}")
        except Exception as exc:
            print(f"[ERROR] Selector request failed: {type(exc).__name__}: {exc}")
            raise RuntimeError(f"Selector LLM request failed: {exc}")


class EmbeddingBasedToolSelector(ToolSelector):
    """Strategy 3: Embedding-based tool selection.
    
    Retrieves the most similar tools using embedding similarity.
    """
    
    def __init__(self, top_k: int = 5):
        self.endpoint = os.getenv("LANGGRAPH_EMBEDDING_ENDPOINT")
        self.api_key = os.getenv("LANGGRAPH_EMBEDDING_API_KEY")
        self.top_k = top_k
    
    def select(self, messages: list, tools: list) -> list:
        """Select top-k most similar tools based on embeddings."""
        if not self.endpoint:
            raise ValueError("LANGGRAPH_EMBEDDING_ENDPOINT environment variable must be set for embedding mode")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for embedding mode. Install with: pip install scikit-learn")
        
        if not tools:
            return []
        
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found in messages for embedding-based tool selection")
        
        query_embedding = self._get_embedding(query)
        tool_embeddings = []
        tool_names = []
        
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            tool_desc = f"{name}: {desc}"
            
            embedding = self._get_embedding(tool_desc)
            tool_embeddings.append(embedding)
            tool_names.append(tool)
        
        if not tool_embeddings:
            raise ValueError("No tools with descriptions found for embedding")
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], tool_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        selected = [tool_names[i] for i in top_indices]
        
        # Extract similarity scores for logging
        similarity_scores = {
            tool_names[i].get("function", {}).get("name", "unknown"): float(similarities[i])
            for i in top_indices
        }
        
        # Log tool selection
        log_tool_selection(
            strategy_name="embedding",
            query=query,
            available_tools_count=len(tools),
            selected_tools=selected,
            selection_metadata={
                "top_k": self.top_k,
                "similarity_scores": similarity_scores
            }
        )
        
        # Log the selection results
        print(f"\n[EMBEDDING SELECTOR]")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Available tools: {len(tools)}")
        print(f"  Top-k: {self.top_k}")
        print(f"  Selected tools:")
        for i, idx in enumerate(top_indices[:5]):
            tool_name = tool_names[idx].get("function", {}).get("name", "unknown")
            similarity_score = similarities[idx]
            print(f"    {i+1}. {tool_name} (similarity: {similarity_score:.4f})")
        if len(top_indices) > 5:
            print(f"    ... and {len(top_indices) - 5} more")
        print()
        
        return selected
    
    def _extract_query(self, messages: list) -> str:
        """Extract the main query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:500]
        return ""
    
    def _get_embedding(self, text: str) -> list:
        """Get embedding for text."""
        payload = {"input": text, "model": "default"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
                parsed = json.loads(data)
                
                if isinstance(parsed, dict) and "embedding" in parsed:
                    return parsed["embedding"]
                if isinstance(parsed, dict) and "data" in parsed and parsed["data"]:
                    return parsed["data"][0].get("embedding", [])
                raise ValueError("Unexpected embedding response format")
        except Exception as exc:
            raise RuntimeError(f"Embedding request failed: {exc}")


class OpenAIEmbeddingWithLLMRerankerToolSelector(ToolSelector):
    """Strategy 4: OpenAI embedding-based retrieval with local vLLM LLM reranking.
    
    Uses OpenAI embeddings for initial retrieval, then reranks with local LLM.
    Allows the LLM to exclude irrelevant tools.
    
    Retrieval phase: Get top-k candidates via embedding similarity
    Reranking phase: LLM reorders candidates and can exclude irrelevant ones
    """
    
    def __init__(self, top_k: int = 5, initial_k: int = 10):
        self.embedding_selector = OpenAIEmbeddingBasedToolSelector(top_k=initial_k)
        _base = os.getenv("EXECUTING_LLM_BASE_URL", "")
        self.llm_endpoint = (_base.rstrip("/") + "/chat/completions") if _base else None
        self.llm_api_key = os.getenv("EXECUTING_LLM_API_KEY", "EMPTY")
        self.top_k = top_k
        self.initial_k = initial_k  # Retrieve more candidates to rerank
    
    def select(self, messages: list, tools: list) -> list:
        """Select tools: embeddings first (top-10), then LLM rerank."""
        # First pass: embedding-based retrieval to get candidates
        candidates = self.embedding_selector.select(messages, tools)
        
        if len(candidates) <= self.top_k:
            # Log tool selection (bypass logging, embedding already logged)
            return candidates
        
        # Second pass: LLM reranking to filter and order
        if not self.llm_endpoint:
            raise ValueError("EXECUTING_LLM_BASE_URL environment variable must be set for embedding_reranker mode")
        
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found in messages for embedding-reranker tool selection")
        
        reranked = self._rerank_with_llm(query, candidates)
        final_selected = reranked[:self.top_k]
        
        # Log tool selection
        log_tool_selection(
            strategy_name="embedding_reranker",
            query=query,
            available_tools_count=len(tools),
            selected_tools=final_selected,
            selection_metadata={
                "embedding_candidates_count": len(candidates),
                "reranked_count": len(reranked),
                "top_k": self.top_k,
                "initial_k": self.initial_k
            }
        )
        
        return final_selected
    
    def _extract_query(self, messages: list) -> str:
        """Extract the main query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:500]
        return ""
    
    def _rerank_with_llm(self, query: str, tools: list) -> list:
        """Use LLM to rerank candidate tools and exclude irrelevant ones."""
        tool_descriptions = []
        tool_names = {}
        
        for idx, tool in enumerate(tools):
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            tool_descriptions.append(f"{idx}. {name}: {desc}")
            tool_names[idx] = name
        
        # Log initial retrieval candidates
        print(f"\n[EMBEDDING + LLM RERANKER SELECTOR]")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"\n  === EMBEDDING RETRIEVAL PHASE (Top-{len(tools)}) ===")
        for i, idx in enumerate(range(len(tools))):
            tool_name = tool_names[idx]
            print(f"    {i+1}. {tool_name}")
        
        rerank_prompt = f"""You are an expert AI that selects the most relevant tools for user queries.

Given the user query, carefully analyze the provided tools and determine:
1. Which tools are truly relevant to the query
2. Order them by relevance (most relevant first)
3. Exclude any tools that are clearly irrelevant

User Query: {query}

Available Tools:
{chr(10).join(tool_descriptions)}

Respond with ONLY a JSON array of tool indices in descending order of relevance.
You may exclude tools if they are not relevant.
Example: [2, 0, 4]

Response (JSON array only):"""
        
        rerank_model = os.getenv("EXECUTING_LLM_MODEL", "openai/gpt-oss-120b")
        payload = {
            "messages": [{"role": "user", "content": rerank_prompt}],
            "model": rerank_model,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        req = urllib.request.Request(self.llm_endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
                parsed = json.loads(data)
                
                response_text = ""
                if isinstance(parsed, dict) and "content" in parsed:
                    response_text = parsed["content"]
                elif isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                    first = parsed["choices"][0]
                    if isinstance(first, dict) and "message" in first and "content" in first["message"]:
                        response_text = first["message"]["content"]
                
                # Extract JSON array from response
                try:
                    # Try to parse the response as JSON directly
                    indices = json.loads(response_text.strip())
                except json.JSONDecodeError:
                    # Try to extract JSON array from response text
                    import re
                    match = re.search(r'\[\s*(?:\d+\s*,?\s*)*\d*\s*\]', response_text)
                    if match:
                        indices = json.loads(match.group())
                    else:
                        print(f"[WARNING] Could not parse LLM response: {response_text[:200]}")
                        # Fallback: return candidates as-is
                        return tools
                
                # Build reranked list from valid indices and track excluded tools
                reranked = []
                seen = set()
                excluded_indices = set(range(len(tools)))
                
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(tools) and idx not in seen:
                        reranked.append(tools[idx])
                        seen.add(idx)
                        excluded_indices.discard(idx)
                
                # Log the LLM reranking decisions
                print(f"\n  === LLM RERANKING PHASE ===")
                print(f"    LLM returned indices: {indices}")
                print(f"\n  === RERANKED OUTPUT ===")
                print(f"    Selected tools (LLM ordering):")
                for i, tool in enumerate(reranked[:self.top_k]):
                    tool_name = tool.get("function", {}).get("name", "unknown")
                    print(f"      {i+1}. {tool_name}")
                
                if excluded_indices:
                    print(f"\n    Excluded tools (not in LLM output):")
                    for idx in sorted(excluded_indices)[:5]:
                        tool_name = tool_names[idx]
                        print(f"      - {tool_name}")
                    if len(excluded_indices) > 5:
                        print(f"      ... and {len(excluded_indices) - 5} more")
                
                print(f"\n    Summary: {len(reranked)} selected, {len(excluded_indices)} excluded")
                print()
                
                return reranked
        except Exception as exc:
            print(f"[ERROR] LLM reranking failed: {exc}")
            raise RuntimeError(f"LLM reranking failed: {exc}")


class OpenAIEmbeddingBasedToolSelector(ToolSelector):
    """Strategy 5: OpenAI text-embedding-3-small based tool selection with caching.
    
    Uses OpenAI's text-embedding-3-small model for embeddings.
    Precomputes tool embeddings and caches them for fast runtime retrieval.
    """
    
    def __init__(self, top_k: int = 5, cache_file: str = None, tools_file: str = None, schema_cache_file: str = None):
        self.top_k = top_k
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), 
            "tool_embeddings_cache.json"
        )
        self.schema_cache_file = schema_cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_schemas_cache.json"
        )
        self.tools_file = tools_file  # Path to original tools JSONL for runtime loading
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = "text-embedding-3-small"
        self.embedding_cache = {}
        self.tools_cache = None  # Lazy load
        self._load_cache()
        self._load_valid_tools_from_schema_cache()
    
    def _load_cache(self):
        """Load cached tool embeddings from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                print(f"[OPENAI EMBEDDING] Loaded cache from {self.cache_file}")
                print(f"[OPENAI EMBEDDING] Cached embeddings: {len(self.embedding_cache)}")
            except Exception as e:
                print(f"[WARNING] Failed to load embedding cache: {e}")
                self.embedding_cache = {}
    
    def _load_valid_tools_from_schema_cache(self):
        """Load all valid tools from schema cache file."""
        if os.path.exists(self.schema_cache_file):
            try:
                with open(self.schema_cache_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "tools" in data:
                        self.tools_cache = data["tools"]
                        print(f"[OPENAI EMBEDDING] Loaded {len(self.tools_cache)} valid tools from schema cache")
                    else:
                        self.tools_cache = data if isinstance(data, list) else []
                        print(f"[OPENAI EMBEDDING] Loaded {len(self.tools_cache)} tools from schema cache")
            except Exception as e:
                print(f"[WARNING] Failed to load schema cache: {e}")
                self.tools_cache = []
        else:
            print(f"[WARNING] Schema cache file not found: {self.schema_cache_file}")
            self.tools_cache = []
    
    def _save_cache(self):
        """Save embedding cache to file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.embedding_cache, f, indent=2)
            print(f"[OPENAI EMBEDDING] Saved cache to {self.cache_file}")
        except Exception as e:
            print(f"[WARNING] Failed to save embedding cache: {e}")
    
    def _load_all_tools(self) -> list:
        """Load all tools from JSONL file at runtime.
        
        Caches the result in memory to avoid repeated file reads.
        """
        if self.tools_cache is not None:
            return self.tools_cache
        
        # Try to find and load the tools file
        tools_file = self.tools_file
        if not tools_file or not os.path.exists(tools_file):
            # Build multiple potential paths to search
            script_dir = os.path.dirname(__file__)
            
            # Find workspace root by going up from current location
            # wild-tool-bench/wtb/model_handler/api_inference -> workspace root
            workspace_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
            
            standard_paths = [
                # Try relative to script location (going up)
                os.path.join(script_dir, "../../../../multi-agent-framework/tools/tools_en.jsonl"),
                os.path.join(script_dir, "../../../multi-agent-framework/tools/tools_en.jsonl"),
                # Try from workspace root
                os.path.join(workspace_root, "multi-agent-framework/tools/tools_en.jsonl"),
                # Try from current working directory
                "multi-agent-framework/tools/tools_en.jsonl",
                os.path.join(os.getcwd(), "multi-agent-framework/tools/tools_en.jsonl"),
                # Try absolute home path
                os.path.expanduser("~/WildToolBench/WildToolBench/multi-agent-framework/tools/tools_en.jsonl"),
            ]
            
            print(f"[OPENAI EMBEDDING] Searching for tools file in {len(standard_paths)} locations...")
            for i, path in enumerate(standard_paths, 1):
                if os.path.exists(path):
                    tools_file = path
                    self.tools_file = path  # Update instance variable for future reference
                    print(f"[OPENAI EMBEDDING] ✓ Found at path {i}: {path}")
                    break
        
        if not tools_file or not os.path.exists(tools_file):
            print(f"[WARNING] Could not find tools file. Searched:")
            for i, path in enumerate(standard_paths, 1):
                exists = "✓" if os.path.exists(path) else "✗"
                print(f"  {exists} {i}. {path}")
            print(f"  Current working directory: {os.getcwd()}")
            return []
        
        tools = []
        try:
            with open(tools_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if isinstance(data, list):
                            tools.extend(data)
                        else:
                            tools.append(data)
            
            # Deduplicate tools by name to avoid processing same tool multiple times
            seen_names = set()
            unique_tools = []
            duplicates = 0
            for tool in tools:
                tool_name = tool.get("function", {}).get("name", "unknown")
                if tool_name not in seen_names:
                    unique_tools.append(tool)
                    seen_names.add(tool_name)
                else:
                    duplicates += 1
            
            if duplicates > 0:
                print(f"[OPENAI EMBEDDING] Deduplicated {duplicates} duplicate tools")
            
            self.tools_cache = unique_tools
            print(f"[OPENAI EMBEDDING] Loaded {len(unique_tools)} unique tools from cache")
            return unique_tools
        except Exception as e:
            print(f"[WARNING] Failed to load tools from {tools_file}: {e}")
            return []
    
    
    def setup_embeddings(self, tools: list):
        """Setup: Precompute embeddings for all tools.
        
        This should be called once during initialization before runtime.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI client is required. Install with: pip install openai")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        
        # Deduplicate tools by name first
        seen_names = set()
        unique_tools = []
        duplicates = 0
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "unknown")
            if tool_name not in seen_names:
                unique_tools.append(tool)
                seen_names.add(tool_name)
            else:
                duplicates += 1
        
        if duplicates > 0:
            print(f"[OPENAI EMBEDDING SETUP] Removed {duplicates} duplicate tools before embedding")
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        print(f"\n[OPENAI EMBEDDING SETUP] Starting to embed {len(unique_tools)} tools...")
        print(f"[OPENAI EMBEDDING SETUP] Model: {self.model}")
        print(f"[OPENAI EMBEDDING SETUP] Cache file: {self.cache_file}")
        
        texts_to_embed = []
        tool_indices = []
        
        for idx, tool in enumerate(unique_tools):
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            tool_desc = f"{name}: {desc}"
            
            # Check if already cached
            if tool_desc in self.embedding_cache:
                print(f"  [{idx+1}/{len(unique_tools)}] {name} - (cached)")
                continue
            
            texts_to_embed.append(tool_desc)
            tool_indices.append((idx, name, tool_desc))
        
        if not texts_to_embed:
            print(f"[OPENAI EMBEDDING SETUP] All {len(unique_tools)} tools already cached!")
            return
        
        # Batch embedding requests
        batch_size = 100
        for batch_idx in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[batch_idx:batch_idx+batch_size]
            batch_tools = tool_indices[batch_idx:batch_idx+batch_size]
            
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                for i, embedding_obj in enumerate(response.data):
                    tool_idx, tool_name, tool_desc = batch_tools[i]
                    self.embedding_cache[tool_desc] = embedding_obj.embedding
                    print(f"  [{tool_idx+1}/{len(unique_tools)}] {tool_name} - OK")
                
            except Exception as e:
                print(f"[ERROR] Failed to embed batch {batch_idx//batch_size}: {e}")
                raise RuntimeError(f"Embedding request failed: {e}")
        
        # Save cache
        self._save_cache()
        print(f"[OPENAI EMBEDDING SETUP] Completed! {len(self.embedding_cache)} embeddings cached.\n")
    
    def select(self, messages: list, tools: list) -> list:
        """Select top-k most similar tools from ALL valid tools based on OpenAI embeddings at runtime.
        
        Strategy: Load ALL 618+ valid tools from schema cache and rank by semantic relevance to query.
        """
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set for OpenAI embedding mode")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for embedding mode. Install with: pip install scikit-learn")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI client is required. Install with: pip install openai")
        
        # Use valid tools from schema cache instead of request tools
        if not self.tools_cache:
            print(f"[OPENAI EMBEDDING SELECTOR] No valid tools in schema cache")
            # Fallback to request tools if no schema cache
            all_tools = tools if tools else []
        else:
            all_tools = self.tools_cache
        
        if not all_tools:
            print(f"[OPENAI EMBEDDING SELECTOR] No tools available")
            return []
        
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found in messages for embedding-based tool selection")
        
        # Get query embedding
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        try:
            query_response = client.embeddings.create(
                input=query,
                model=self.model
            )
            query_embedding = query_response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")
        
        # Get tool embeddings and compute similarity for ALL VALID TOOLS
        tool_embeddings = []
        tool_descriptions = []
        cached_count = 0
        missing_count = 0
        
        for tool in all_tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            tool_desc = f"{name}: {desc}"
            
            if tool_desc in self.embedding_cache:
                embedding = self.embedding_cache[tool_desc]
                cached_count += 1
            else:
                # Fallback: compute embedding at runtime if not cached
                try:
                    response = client.embeddings.create(
                        input=tool_desc,
                        model=self.model
                    )
                    embedding = response.data[0].embedding
                    self.embedding_cache[tool_desc] = embedding
                    missing_count += 1
                except Exception as e:
                    print(f"[WARNING] Failed to embed tool {name}: {e}")
                    embedding = [0] * len(query_embedding)  # Fallback
            
            tool_embeddings.append(embedding)
            tool_descriptions.append(tool_desc)
        
        if not tool_embeddings:
            raise ValueError("No tools with descriptions found for embedding")
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], tool_embeddings)[0]
        
        # Sort by similarity descending and deduplicate by tool name
        sorted_indices = np.argsort(similarities)[::-1]
        selected = []
        selected_scores = []
        seen_tool_names = set()
        
        for idx in sorted_indices:
            tool = all_tools[idx]
            tool_name = tool.get("function", {}).get("name", "unknown")
            
            # Only add if we haven't seen this tool name yet
            if tool_name not in seen_tool_names:
                selected.append(tool)
                selected_scores.append(similarities[idx])
                seen_tool_names.add(tool_name)
                
                # Stop when we have enough unique tools
                if len(selected) >= self.top_k:
                    break
        
        # Log the selection results
        print(f"\n[OPENAI EMBEDDING SELECTOR]")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Total available tools: {len(all_tools)}")
        print(f"  Top-k (unique): {self.top_k}")
        print(f"  Cache hits: {cached_count}, Runtime computed: {missing_count}")
        print(f"  Selected tools (ranked by relevance):")
        for i, (tool, score) in enumerate(zip(selected[:5], selected_scores[:5])):
            tool_name = tool.get("function", {}).get("name", "unknown")
            print(f"    {i+1}. {tool_name} (similarity: {score:.4f})")
        if len(selected) > 5:
            print(f"    ... and {len(selected) - 5} more")
        print()
        
        # Log tool selection
        log_tool_selection(
            strategy_name="openai_embedding",
            query=query,
            available_tools_count=len(all_tools),
            selected_tools=selected,
            selection_metadata={
                "top_k": self.top_k,
                "cache_hits": cached_count,
                "runtime_computed": missing_count,
                "model": self.model,
                "similarity_scores": {
                    tool.get("function", {}).get("name", "unknown"): float(score)
                    for tool, score in zip(selected, selected_scores)
                }
            }
        )
        
        return selected
    
    
    def _extract_query(self, messages: list) -> str:
        """Extract the main query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:500]
        return ""


class EmbeddingContextBasedToolSelector(OpenAIEmbeddingBasedToolSelector):
    """Strategy 6: OpenAI embedding-based tool selection with full conversation context.
    
    Uses OpenAI's text-embedding-3-small model for embeddings.
    Embeds the complete multi-turn conversation history (all previous queries and assistant responses)
    instead of just the most recent query, providing better semantic understanding of the conversation context.
    """
    
    def __init__(self, top_k: int = 5, cache_file: str = None, tools_file: str = None, schema_cache_file: str = None):
        """Initialize with same parameters as parent class."""
        super().__init__(top_k=top_k, cache_file=cache_file, tools_file=tools_file, schema_cache_file=schema_cache_file)
    
    def _extract_query(self, messages: list) -> str:
        """Extract full conversation context by concatenating all messages.
        
        Instead of just the recent query, this includes the complete conversation history
        to provide better semantic context for multi-turn interactions.
        """
        conversation_parts = []
        
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            
            # Include both user and assistant messages
            if isinstance(content, str) and content.strip():
                conversation_parts.append(f"[{role}]: {content}")
        
        # Concatenate all parts and limit to reasonable length
        full_conversation = "\n".join(conversation_parts)
        
        # Use 2000 chars to accommodate full context (vs 500 in single query mode)
        return full_conversation[:2000] if full_conversation else ""


class OpenAIEmbeddingContextWithLLMRerankerToolSelector(OpenAIEmbeddingWithLLMRerankerToolSelector):
    """Strategy 7: OpenAI embedding-based retrieval with full conversation context + LLM reranking.
    
    Uses OpenAI embeddings for initial retrieval based on complete conversation history,
    then reranks with local LLM. Allows the LLM to exclude irrelevant tools.
    
    Retrieval phase: Embed full conversation history to get top-k candidates
    Reranking phase: LLM reorders candidates considering full context and can exclude irrelevant ones
    """
    
    def __init__(self, top_k: int = 5, initial_k: int = 10, cache_file: str = None, tools_file: str = None, schema_cache_file: str = None):
        """Initialize with same parameters as parent class."""
        # Create embedding selector with context support
        self.embedding_selector = EmbeddingContextBasedToolSelector(
            top_k=initial_k, 
            cache_file=cache_file, 
            tools_file=tools_file, 
            schema_cache_file=schema_cache_file
        )
        _base = os.getenv("EXECUTING_LLM_BASE_URL", "")
        self.llm_endpoint = (_base.rstrip("/") + "/chat/completions") if _base else None
        self.llm_api_key = os.getenv("EXECUTING_LLM_API_KEY", "EMPTY")
        self.top_k = top_k
        self.initial_k = initial_k
    
    def _extract_query(self, messages: list) -> str:
        """Extract full conversation context instead of just the latest query.
        
        This is used for LLM reranking prompt to provide full context.
        """
        conversation_parts = []
        
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            
            # Include both user and assistant messages
            if isinstance(content, str) and content.strip():
                conversation_parts.append(f"[{role}]: {content}")
        
        # Concatenate all parts
        full_conversation = "\n".join(conversation_parts)
        return full_conversation[:2000] if full_conversation else ""


# ==================== Qwen3 Embedding Strategies ====================

class Qwen3EmbeddingBasedToolSelector(OpenAIEmbeddingBasedToolSelector):
    """Strategy 8: Qwen3-Embedding-8B based tool selection with caching.

    Uses a local Qwen3-Embedding-8B model served via a vLLM OpenAI-compatible
    endpoint for embeddings.  Follows the Qwen3-Embedding best-practice of
    prepending a task instruction to query texts (documents are embedded as-is).

    Environment Variables:
        QWEN3_EMBEDDING_BASE_URL: vLLM base URL (default: http://localhost:8001/v1)
        QWEN3_EMBEDDING_API_KEY:  API key sent to vLLM (default: EMPTY)
        QWEN3_EMBEDDING_MODEL:    Model name (default: Qwen/Qwen3-Embedding-8B)
    """

    TASK_INSTRUCTION = (
        "Given a user query about tool usage, retrieve the most relevant tool "
        "function that can fulfill the described task."
    )

    def __init__(self, top_k: int = 5, cache_file: str = None,
                 tools_file: str = None, schema_cache_file: str = None):
        self.top_k = top_k
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_embeddings_cache_qwen3.json"
        )
        self.schema_cache_file = schema_cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_schemas_cache.json"
        )
        self.tools_file = tools_file
        self.api_key = os.getenv("QWEN3_EMBEDDING_API_KEY", "EMPTY")
        self.base_url = os.getenv("QWEN3_EMBEDDING_BASE_URL", "http://localhost:8001/v1")
        self.model = os.getenv("QWEN3_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
        self.embedding_cache = {}
        self.tools_cache = None
        self._load_cache()
        self._load_valid_tools_from_schema_cache()

    def _extract_query(self, messages: list) -> str:
        """Extract the latest user query and prepend the Qwen3 instruction prefix."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return f"Instruct: {self.TASK_INSTRUCTION}\nQuery: {content[:500]}"
        return ""

    def select(self, messages: list, tools: list) -> list:
        """Select top-k tools using Qwen3-Embedding-8B via local vLLM endpoint."""
        if not self.base_url:
            raise ValueError(
                "QWEN3_EMBEDDING_BASE_URL must be set for qwen3_embedding mode"
            )
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for embedding mode. "
                "Install with: pip install scikit-learn"
            )
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        all_tools = self.tools_cache if self.tools_cache else tools
        if not all_tools:
            print("[QWEN3 EMBEDDING SELECTOR] No tools available")
            return []

        query = self._extract_query(messages)
        if not query:
            raise ValueError(
                "No user query found in messages for Qwen3 embedding-based tool selection"
            )

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        try:
            query_response = client.embeddings.create(input=query, model=self.model)
            query_embedding = query_response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed query with Qwen3: {e}")

        tool_embeddings = []
        tool_descriptions = []
        cached_count = 0
        missing_count = 0

        for tool in all_tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            tool_desc = f"{name}: {desc}"

            if tool_desc in self.embedding_cache:
                embedding = self.embedding_cache[tool_desc]
                cached_count += 1
            else:
                try:
                    response = client.embeddings.create(
                        input=tool_desc, model=self.model
                    )
                    embedding = response.data[0].embedding
                    self.embedding_cache[tool_desc] = embedding
                    missing_count += 1
                except Exception as e:
                    print(f"[WARNING] Failed to embed tool {name}: {e}")
                    embedding = [0] * len(query_embedding)

            tool_embeddings.append(embedding)
            tool_descriptions.append(tool_desc)

        if not tool_embeddings:
            raise ValueError("No tools with descriptions found for embedding")

        similarities = cosine_similarity([query_embedding], tool_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]

        selected = []
        selected_scores = []
        seen_tool_names = set()

        for idx in sorted_indices:
            tool = all_tools[idx]
            tool_name = tool.get("function", {}).get("name", "unknown")
            if tool_name not in seen_tool_names:
                selected.append(tool)
                selected_scores.append(similarities[idx])
                seen_tool_names.add(tool_name)
                if len(selected) >= self.top_k:
                    break

        print(f"\n[QWEN3 EMBEDDING SELECTOR]")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Model: {self.model}")
        print(f"  Total available tools: {len(all_tools)}")
        print(f"  Top-k (unique): {self.top_k}")
        print(f"  Cache hits: {cached_count}, Runtime computed: {missing_count}")
        print(f"  Selected tools (ranked by relevance):")
        for i, (tool, score) in enumerate(zip(selected[:5], selected_scores[:5])):
            tool_name = tool.get("function", {}).get("name", "unknown")
            print(f"    {i+1}. {tool_name} (similarity: {score:.4f})")
        if len(selected) > 5:
            print(f"    ... and {len(selected) - 5} more")
        print()

        return selected

    def setup_embeddings(self, tools: list):
        """Precompute Qwen3 embeddings for all tools and cache them locally.

        Documents (tool descriptions) are embedded without any instruction prefix,
        following the Qwen3-Embedding asymmetric retrieval recommendation.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )

        seen_names = set()
        unique_tools = []
        duplicates = 0
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "unknown")
            if tool_name not in seen_names:
                unique_tools.append(tool)
                seen_names.add(tool_name)
            else:
                duplicates += 1

        if duplicates > 0:
            print(f"[QWEN3 EMBEDDING SETUP] Removed {duplicates} duplicate tools")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        print(f"\n[QWEN3 EMBEDDING SETUP] Starting to embed {len(unique_tools)} tools...")
        print(f"[QWEN3 EMBEDDING SETUP] Model: {self.model}")
        print(f"[QWEN3 EMBEDDING SETUP] Base URL: {self.base_url}")
        print(f"[QWEN3 EMBEDDING SETUP] Cache file: {self.cache_file}")

        texts_to_embed = []
        tool_indices = []

        for idx, tool in enumerate(unique_tools):
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            tool_desc = f"{name}: {desc}"

            if tool_desc in self.embedding_cache:
                print(f"  [{idx+1}/{len(unique_tools)}] {name} - (cached)")
                continue

            texts_to_embed.append(tool_desc)
            tool_indices.append((idx, name, tool_desc))

        if not texts_to_embed:
            print(f"[QWEN3 EMBEDDING SETUP] All {len(unique_tools)} tools already cached!")
            return

        # Use a smaller batch size suitable for local vLLM deployments
        batch_size = 32
        for batch_idx in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[batch_idx:batch_idx + batch_size]
            batch_tools = tool_indices[batch_idx:batch_idx + batch_size]

            try:
                response = client.embeddings.create(input=batch, model=self.model)
                for i, embedding_obj in enumerate(response.data):
                    tool_idx, tool_name, tool_desc = batch_tools[i]
                    self.embedding_cache[tool_desc] = embedding_obj.embedding
                    print(f"  [{tool_idx+1}/{len(unique_tools)}] {tool_name} - OK")
            except Exception as e:
                print(f"[ERROR] Failed to embed batch {batch_idx // batch_size}: {e}")
                raise RuntimeError(f"Qwen3 embedding request failed: {e}")

        self._save_cache()
        print(
            f"[QWEN3 EMBEDDING SETUP] Completed! "
            f"{len(self.embedding_cache)} embeddings cached.\n"
        )


class Qwen3EmbeddingContextBasedToolSelector(Qwen3EmbeddingBasedToolSelector):
    """Strategy 9: Qwen3-Embedding-8B with full conversation context.

    Embeds the complete multi-turn conversation history (all previous user and
    assistant messages) instead of just the most recent query, giving better
    semantic understanding in multi-turn interactions.
    """

    def _extract_query(self, messages: list) -> str:
        """Concatenate full conversation history and prepend Qwen3 instruction prefix."""
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                conversation_parts.append(f"[{role}]: {content}")

        full_conversation = "\n".join(conversation_parts)[:2000]
        if not full_conversation:
            return ""
        return f"Instruct: {self.TASK_INSTRUCTION}\nQuery: {full_conversation}"


class Qwen3EmbeddingWithLLMRerankerToolSelector(OpenAIEmbeddingWithLLMRerankerToolSelector):
    """Strategy 10: Qwen3-Embedding-8B retrieval + LLM reranking.

    Uses Qwen3-Embedding-8B for the first-pass retrieval (top-initial_k candidates)
    and then applies LLM reranking to produce the final top-k tools.
    """

    def __init__(self, top_k: int = 5, initial_k: int = 10, cache_file: str = None,
                 tools_file: str = None, schema_cache_file: str = None):
        self.embedding_selector = Qwen3EmbeddingBasedToolSelector(
            top_k=initial_k,
            cache_file=cache_file,
            tools_file=tools_file,
            schema_cache_file=schema_cache_file,
        )
        _base = os.getenv("EXECUTING_LLM_BASE_URL", "")
        self.llm_endpoint = (_base.rstrip("/") + "/chat/completions") if _base else None
        self.llm_api_key = os.getenv("EXECUTING_LLM_API_KEY", "EMPTY")
        self.top_k = top_k
        self.initial_k = initial_k


class Qwen3EmbeddingContextWithLLMRerankerToolSelector(OpenAIEmbeddingContextWithLLMRerankerToolSelector):
    """Strategy 11: Qwen3-Embedding-8B + full conversation context + LLM reranking.

    Combines full conversation context embedding (Qwen3) with LLM-based reranking
    for the highest quality tool selection in multi-turn conversations.
    """

    def __init__(self, top_k: int = 5, initial_k: int = 10, cache_file: str = None,
                 tools_file: str = None, schema_cache_file: str = None):
        self.embedding_selector = Qwen3EmbeddingContextBasedToolSelector(
            top_k=initial_k,
            cache_file=cache_file,
            tools_file=tools_file,
            schema_cache_file=schema_cache_file,
        )
        _base = os.getenv("EXECUTING_LLM_BASE_URL", "")
        self.llm_endpoint = (_base.rstrip("/") + "/chat/completions") if _base else None
        self.llm_api_key = os.getenv("EXECUTING_LLM_API_KEY", "EMPTY")
        self.top_k = top_k
        self.initial_k = initial_k

    def _extract_query(self, messages: list) -> str:
        """Extract full conversation context for the LLM reranking prompt."""
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                conversation_parts.append(f"[{role}]: {content}")
        full_conversation = "\n".join(conversation_parts)
        return full_conversation[:2000] if full_conversation else ""


# ==================== Qwen3 Reranker Strategies ====================

class Qwen3RerankerBasedToolSelector(ToolSelector):
    """Pairwise reranker using Qwen3-Reranker-8B.

    Scores each (query, tool) pair using the Qwen3-Reranker-8B model deployed
    via vLLM.  Supports two deployment modes:

    * **Score mode** (recommended): Deploy the vLLM server with ``--task score``
      and call the ``/v1/score`` endpoint for efficient batch scoring.
    * **Generation mode** (fallback): Deploy as a standard chat model and
      extract the log-probability of the "Yes" token from the first generated
      token, formatted with the Qwen3-Reranker chat template.

    Environment Variables:
        QWEN3_RERANKER_BASE_URL: vLLM base URL (default: http://localhost:8002/v1)
        QWEN3_RERANKER_API_KEY:  API key (default: EMPTY)
        QWEN3_RERANKER_MODEL:    Model name (default: Qwen/Qwen3-Reranker-8B)
    """

    TASK_INSTRUCTION = (
        "Given a user query about tool usage, retrieve the most relevant tool "
        "function that can fulfill the described task."
    )

    SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query and "
        "the Instruct provided. Note only the last point is sufficient, the "
        "information above can be incomplete."
    )

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.base_url = os.getenv("QWEN3_RERANKER_BASE_URL", "http://localhost:8002/v1")
        self.api_key = os.getenv("QWEN3_RERANKER_API_KEY", "EMPTY")
        self.model = os.getenv("QWEN3_RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

    def select(self, messages: list, tools: list) -> list:
        """Select top-k tools using Qwen3-Reranker-8B pairwise scoring."""
        if not tools:
            return []
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found for Qwen3-Reranker tool selection")
        reranked = self._rerank_with_qwen3(query, tools)
        final_selected = reranked[:self.top_k]
        
        # Log tool selection
        log_tool_selection(
            strategy_name="qwen3_reranker",
            query=query,
            available_tools_count=len(tools),
            selected_tools=final_selected,
            selection_metadata={
                "model": self.model,
                "top_k": self.top_k,
                "reranked_count": len(reranked)
            }
        )
        
        return final_selected

    def _extract_query(self, messages: list) -> str:
        """Extract the latest user query."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content[:1000]
        return ""

    def _format_document(self, tool: dict) -> str:
        """Render a tool definition as a plain-text document for the reranker."""
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        param_strs = []
        for p_name, p_info in list(props.items())[:5]:
            p_desc = p_info.get("description", "")
            param_strs.append(f"  {p_name}: {p_desc}" if p_desc else f"  {p_name}")
        doc = f"Tool: {name}\nDescription: {desc}"
        if param_strs:
            doc += "\nParameters:\n" + "\n".join(param_strs)
        return doc

    def _score_batch_via_score_endpoint(self, query: str, tool_docs: list) -> list:
        """Batch-score via vLLM /v1/score endpoint (--task score deployment).

        Returns a list of float scores, one per document, in input order.
        Raises an exception if the endpoint is unavailable or returns an error.
        """
        endpoint = self.base_url.rstrip("/") + "/score"
        payload = {
            "model": self.model,
            "text_1": query,
            "text_2": tool_docs,
            "encoding_format": "float",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        scores_data = data.get("data", [])
        scores_data.sort(key=lambda x: x.get("index", 0))
        return [float(item.get("score", 0.0)) for item in scores_data]

    def _score_single_via_chat_completions(self, query: str, tool_doc: str) -> float:
        """Score a single (query, tool) pair via chat completions + logprobs.

        Formats the input with the Qwen3-Reranker chat template and extracts
        the log-probability of the 'yes' token from the first generated token.
        Falls back to text matching if logprobs are unavailable.
        """
        import math

        user_content = (
            f"<Instruct>: {self.TASK_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {tool_doc}\n"
            "Does the Document meet the requirements? Respond with 'Yes' or 'No'."
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        endpoint = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))

        choices = parsed.get("choices", [])
        if not choices:
            return 0.0

        choice = choices[0]

        # Try logprobs first (most accurate)
        logprobs_data = choice.get("logprobs") or {}
        content_logprobs = logprobs_data.get("content") or []
        if content_logprobs:
            top_lps = content_logprobs[0].get("top_logprobs") or []
            for lp in top_lps:
                token = (lp.get("token") or "").strip().lower()
                if token in ("yes", "y"):
                    return math.exp(lp["logprob"])
            # No "yes" token found — use 1 - P(no) if "no" is present
            for lp in top_lps:
                token = (lp.get("token") or "").strip().lower()
                if token in ("no", "n"):
                    return max(0.0, 1.0 - math.exp(lp["logprob"]))

        # Fallback: text matching
        generated = (choice.get("message") or {}).get("content", "")
        return 1.0 if generated.strip().lower().startswith("y") else 0.0

    def _rerank_with_qwen3(self, query: str, tools: list) -> list:
        """Score all candidate tools and return them sorted by descending relevance score."""
        tool_docs = [self._format_document(t) for t in tools]
        scores = None

        # Attempt 1: batch scoring via /v1/score endpoint
        try:
            scores = self._score_batch_via_score_endpoint(query, tool_docs)
            print(f"\n[QWEN3 RERANKER] Used /v1/score endpoint (batch, {len(tools)} tools)")
        except Exception as exc:
            print(
                f"[QWEN3 RERANKER] /v1/score endpoint unavailable ({exc}); "
                "falling back to chat completions + logprobs"
            )

        # Attempt 2: per-tool chat completions with logprobs
        if scores is None:
            scores = []
            for i, tool_doc in enumerate(tool_docs):
                try:
                    score = self._score_single_via_chat_completions(query, tool_doc)
                    scores.append(score)
                except Exception as exc:
                    print(f"[QWEN3 RERANKER] Failed to score tool {i}: {exc}")
                    scores.append(0.0)

        print(f"\n[QWEN3 RERANKER SELECTOR]")
        print(f"  Model: {self.model}")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Candidates scored: {len(tools)}")

        ranked_pairs = sorted(zip(scores, tools), key=lambda x: x[0], reverse=True)
        print(f"  Top-{min(self.top_k, len(ranked_pairs))} after reranking:")
        for i, (score, tool) in enumerate(ranked_pairs[:self.top_k]):
            tool_name = tool.get("function", {}).get("name", "unknown")
            print(f"    {i+1}. {tool_name} (score: {score:.4f})")
        print()

        return [t for _, t in ranked_pairs]


class Qwen3EmbeddingWithQwen3RerankerToolSelector(Qwen3RerankerBasedToolSelector):
    """Strategy 12: Qwen3-Embedding-8B retrieval + Qwen3-Reranker-8B pairwise reranking.

    Two-stage pipeline:
      1. **Embedding retrieval** – Qwen3-Embedding-8B narrows the full tool set to
         ``initial_k`` candidates using dense vector similarity.
      2. **Pairwise reranking** – Qwen3-Reranker-8B cross-encodes each (query, tool)
         pair and ranks candidates by relevance score.

    Environment Variables (embedding stage):
        QWEN3_EMBEDDING_BASE_URL: vLLM base URL (default: http://localhost:8001/v1)
        QWEN3_EMBEDDING_API_KEY:  API key (default: EMPTY)
        QWEN3_EMBEDDING_MODEL:    Model name (default: Qwen/Qwen3-Embedding-8B)
    Environment Variables (reranking stage):
        QWEN3_RERANKER_BASE_URL: vLLM base URL (default: http://localhost:8002/v1)
        QWEN3_RERANKER_API_KEY:  API key (default: EMPTY)
        QWEN3_RERANKER_MODEL:    Model name (default: Qwen/Qwen3-Reranker-8B)
    """

    def __init__(self, top_k: int = 5, initial_k: int = 20, cache_file: str = None,
                 tools_file: str = None, schema_cache_file: str = None):
        super().__init__(top_k=top_k)
        self.initial_k = initial_k
        self.embedding_selector = Qwen3EmbeddingBasedToolSelector(
            top_k=initial_k,
            cache_file=cache_file,
            tools_file=tools_file,
            schema_cache_file=schema_cache_file,
        )

    def select(self, messages: list, tools: list) -> list:
        """Embedding retrieval → Qwen3-Reranker pairwise scoring."""
        candidates = self.embedding_selector.select(messages, tools)

        if len(candidates) <= self.top_k:
            return candidates

        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found for Qwen3-Reranker tool selection")

        reranked = self._rerank_with_qwen3(query, candidates)
        return reranked[:self.top_k]


class Qwen3EmbeddingContextWithQwen3RerankerToolSelector(Qwen3EmbeddingWithQwen3RerankerToolSelector):
    """Strategy 13: Qwen3-Embedding-8B (full context) + Qwen3-Reranker-8B pairwise reranking.

    Same as ``Qwen3EmbeddingWithQwen3RerankerToolSelector`` but uses the full
    multi-turn conversation history for both the embedding retrieval phase and
    as the reranker query, giving better accuracy in multi-turn interactions.
    """

    def __init__(self, top_k: int = 5, initial_k: int = 20, cache_file: str = None,
                 tools_file: str = None, schema_cache_file: str = None):
        super().__init__(
            top_k=top_k, initial_k=initial_k,
            cache_file=cache_file, tools_file=tools_file,
            schema_cache_file=schema_cache_file,
        )
        self.embedding_selector = Qwen3EmbeddingContextBasedToolSelector(
            top_k=initial_k,
            cache_file=cache_file,
            tools_file=tools_file,
            schema_cache_file=schema_cache_file,
        )

    def _extract_query(self, messages: list) -> str:
        """Use full conversation context as the reranker query."""
        conversation_parts = []
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                conversation_parts.append(f"[{role}]: {content}")
        full_conversation = "\n".join(conversation_parts)
        return full_conversation[:2000] if full_conversation else ""


# ==================== LLM Invocation ====================

# Special-token suffixes that indicate raw model output leaked into a tool call value.
_RAW_MODEL_TOKENS = ("<|end|>", "<|start|>", "<|channel|>", "<|endoftext|>")


def _normalize_messages_for_llm(messages: list) -> list:
    """Normalize messages to standard OpenAI format before sending to vLLM.

    Some models (e.g. gpt-oss-120b / phi-4 variants) generate tool calls using
    an internal ``to=functions.<name>`` header format, sometimes with chain-of-
    thought text and special tokens appended.  If those raw values end up stored
    in the message history they will cause vLLM to return:

        openai_harmony.HarmonyError: unexpected tokens remaining in message header:
            Some("to=functions.<name>...")

    on the *next* turn because vLLM cannot round-trip the non-standard format.

    This function converts any such tool calls to the canonical OpenAI format:
        {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}

    It is safe to call on well-formed messages – standard tool calls are passed
    through unchanged.
    """
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            normalized.append(msg)
            continue

        msg = dict(msg)  # shallow copy so we don't mutate the original

        if msg.get("tool_calls"):
            new_tool_calls = []
            for idx, tc in enumerate(msg["tool_calls"]):
                if not isinstance(tc, dict):
                    new_tool_calls.append(tc)
                    continue

                tc = dict(tc)

                if "function" in tc and isinstance(tc["function"], dict):
                    # Standard format – just ensure arguments is a JSON string.
                    func = dict(tc["function"])
                    if isinstance(func.get("arguments"), dict):
                        func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
                    tc["function"] = func
                    new_tool_calls.append(tc)

                elif "to" in tc:
                    # ``to=functions.<name>`` style – convert to standard format.
                    to_val = str(tc["to"])
                    # Strip everything from the first special token or whitespace
                    # followed by extra content (raw model output may be appended).
                    for token in _RAW_MODEL_TOKENS:
                        if token in to_val:
                            to_val = to_val[:to_val.index(token)]
                    to_val = to_val.strip()
                    # Strip "functions." prefix that some models emit.
                    name = to_val[len("functions."):] if to_val.startswith("functions.") else to_val
                    # Strip any trailing punctuation / stray characters.
                    name = name.split()[0].rstrip(".,;:")
                    args = tc.get("parameters", tc.get("arguments", {}))
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    call_id = tc.get("id") or f"call_{name}_{idx}"
                    print(f"[NORMALIZE] Converted to=functions format: '{tc.get('to', '')[:60]}' -> name='{name}'")
                    new_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })

                elif "name" in tc:
                    # Flat name-keyed format (no ``function`` wrapper).
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    call_id = tc.get("id") or f"call_{tc['name']}_{idx}"
                    new_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": args},
                    })

                else:
                    # Unknown format – keep as-is and log a warning.
                    print(f"[NORMALIZE] Unknown tool_call format (keys={list(tc.keys())}), keeping as-is: "
                          f"{json.dumps(tc, ensure_ascii=False)[:120]}")
                    new_tool_calls.append(tc)

            msg["tool_calls"] = new_tool_calls

        # Strip special tokens from content (any role).
        # The gpt-oss-120b / phi-4 family can write chain-of-thought or channel
        # markers directly into the content field, e.g.:
        #   "...Let's call it.<|end|><|start|>assistant<|channel|>commentary"
        # vLLM rejects such content when it appears in a message header on the
        # *next* turn.  We truncate at the first special token.
        if isinstance(msg.get("content"), str):
            content = msg["content"]
            for token in _RAW_MODEL_TOKENS:
                if token in content:
                    truncated = content[:content.index(token)].rstrip()
                    print(f"[NORMALIZE] Stripped special tokens from {msg.get('role','?')} content "
                          f"(truncated {len(content) - len(truncated)} chars at '{token}')")
                    content = truncated
            msg["content"] = content

        normalized.append(msg)
    return normalized


def _normalize_returned_tool_calls(tool_calls: list) -> list:
    """Normalize tool calls returned BY the LLM to standard OpenAI format.

    Mirrors _normalize_messages_for_llm but operates on a flat list of tool
    call dicts rather than on messages.  Ensures tool calls stored in the
    inference log and fed back as history are always in standard format.
    """
    if not tool_calls:
        return tool_calls
    result = []
    for idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            result.append(tc)
            continue
        tc = dict(tc)
        if "function" in tc and isinstance(tc["function"], dict):
            func = dict(tc["function"])
            if isinstance(func.get("arguments"), dict):
                func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
            tc["function"] = func
            result.append(tc)
        elif "to" in tc:
            to_val = str(tc["to"])
            for token in _RAW_MODEL_TOKENS:
                if token in to_val:
                    to_val = to_val[:to_val.index(token)]
            to_val = to_val.strip()
            name = to_val[len("functions."):] if to_val.startswith("functions.") else to_val
            name = name.split()[0].rstrip(".,;:")
            args = tc.get("parameters", tc.get("arguments", {}))
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            elif not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            call_id = tc.get("id") or f"call_{name}_{idx}"
            print(f"[NORMALIZE] Converted returned to=functions: '{tc.get('to', '')[:60]}' -> name='{name}'")
            result.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
        elif "name" in tc:
            args = tc.get("arguments", {})
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            elif not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            call_id = tc.get("id") or f"call_{tc['name']}_{idx}"
            result.append({
                "id": call_id,
                "type": "function",
                "function": {"name": tc["name"], "arguments": args},
            })
        else:
            result.append(tc)
    return result


def _invoke_llm(messages: list, tools: list = None):
    """Invoke a local OpenAI/vLLM-compatible endpoint.
    
    Returns: (content, tool_calls) tuple
    """
    base_url = os.getenv("EXECUTING_LLM_BASE_URL")
    api_key = os.getenv("EXECUTING_LLM_API_KEY", "EMPTY")
    model = os.getenv("EXECUTING_LLM_MODEL", "openai/gpt-oss-120b")
    
    if not base_url:
        # Simulated response for testing without a real LLM
        joined = "\n".join([str(m) for m in messages])
        return f"[simulated response] received {len(messages)} messages: {joined[:200]}", []

    # Build full chat completions endpoint from base URL
    endpoint = base_url.rstrip("/") + "/chat/completions"

    # Prepend reasoning system prompt if not already present
    reasoning = os.getenv("EXECUTING_LLM_REASONING", "medium")
    messages_to_send = list(messages)
    if not messages_to_send or messages_to_send[0].get("role") != "system":
        messages_to_send.insert(0, {"role": "system", "content": f"Reasoning: {reasoning}"})

    # Normalize all messages: convert any to=functions.X tool calls to standard format
    # before sending to vLLM, which would otherwise reject them with a 500.
    messages_to_send = _normalize_messages_for_llm(messages_to_send)

    payload = {
        "messages": messages_to_send,
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    
    # Add tools if provided
    if tools:
        payload["tools"] = tools

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    print(f"[DEBUG] Sending LLM request to {endpoint}")
    print(f"[DEBUG] Model: {model}")
    print(f"[DEBUG] Payload keys: {list(payload.keys())}")
    print(f"[DEBUG] Messages: {len(messages)}, Tools: {len(tools or [])}")
    # Log any tool calls already present in the outgoing messages (history from prior turns)
    for i, msg in enumerate(messages_to_send):
        if msg.get("tool_calls"):
            print(f"[DEBUG] Message[{i}] role={msg.get('role')} has tool_calls: "
                  f"{[tc.get('function', {}).get('name') if isinstance(tc, dict) else tc for tc in msg['tool_calls']]}, "
                  f"keys per call: {[list(tc.keys()) if isinstance(tc, dict) else type(tc) for tc in msg['tool_calls']]}")

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    # Large tool sets (1000+ tools) can take several minutes to process.
    # Use env var EXECUTING_LLM_TIMEOUT (seconds) or default to 600 (10 min).
    _timeout = int(os.getenv("EXECUTING_LLM_TIMEOUT", "600"))

    # On HarmonyError (the model writes chain-of-thought inside the tool-call header,
    # breaking vLLM's parser), retry with progressively higher temperature so the
    # model generates a different – hopefully well-formed – output.
    # At temperature 0.0 the model is deterministic and will always reproduce the
    # same broken output, so any retry must use temp > 0.
    _harmony_retry_temps = [0.3, 0.7]  # temperatures to try after the initial 0.0 attempt

    for _attempt in range(1 + len(_harmony_retry_temps)):  # 0 = original, 1/2 = retries
        try:
            with urllib.request.urlopen(req, timeout=_timeout) as resp:
                data = resp.read().decode("utf-8")
                print(f"[DEBUG] LLM response status: {resp.status}")
                try:
                    parsed = json.loads(data)
                except Exception as e:
                    print(f"[DEBUG] Failed to parse JSON: {e}")
                    return data, [], 0, 0

                # Extract content
                content = ""
                if isinstance(parsed, dict) and "content" in parsed:
                    content = parsed["content"] or ""
                elif isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                    first = parsed["choices"][0]
                    if isinstance(first, dict) and "message" in first:
                        msg = first["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"] or ""

                # Extract tool calls if present
                tool_calls = []
                if isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = parsed["tool_calls"]
                elif isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                    first = parsed["choices"][0]
                    if isinstance(first, dict) and "message" in first:
                        msg = first["message"]
                        if isinstance(msg, dict) and "tool_calls" in msg:
                            tool_calls = msg["tool_calls"]

                # Extract token usage
                usage = parsed.get("usage") or {}
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                print(f"[DEBUG] Extracted content length: {len(content)}, tool_calls: {len(tool_calls)}, tokens: {input_tokens}/{output_tokens}")
                # Strip special tokens from the returned content so it is safe to store
                # in message history and re-send in subsequent turns.
                for token in _RAW_MODEL_TOKENS:
                    if token in content:
                        content = content[:content.index(token)].rstrip()
                # Normalize returned tool calls to standard format so they are safe to store
                # in message history and re-send in subsequent turns.
                tool_calls = _normalize_returned_tool_calls(tool_calls)
                if tool_calls:
                    print(f"[DEBUG] Tool calls returned: {[tc.get('function', {}).get('name') if isinstance(tc, dict) else tc for tc in tool_calls]}")
                log_tool_calls(messages_to_send, tool_calls, content, input_tokens, output_tokens, model=model)
                return content, tool_calls, input_tokens, output_tokens

        except urllib.error.HTTPError as exc:
            error_data = exc.read().decode("utf-8")
            is_harmony_error = (
                exc.code == 500 and (
                    "harmonyerror" in error_data.lower()
                    or "unexpected tokens" in error_data.lower()
                    or "to=functions." in error_data
                )
            )
            if is_harmony_error and _attempt < len(_harmony_retry_temps):
                retry_temp = _harmony_retry_temps[_attempt]
                print(f"[HARMONY-RETRY] Attempt {_attempt + 1} failed with HarmonyError. "
                      f"Retrying with temperature={retry_temp} ...")
                payload["temperature"] = retry_temp
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
                continue  # retry the loop

            print(f"[ERROR] LLM HTTP Error {exc.code}: {exc.reason}")
            print(f"[ERROR] Response body: {error_data[:500]}")
            raise RuntimeError(f"LLM request failed: {exc.code} {exc.reason} - {error_data[:200]}")

        except urllib.error.URLError as exc:
            print(f"[ERROR] LLM URL Error: {exc.reason}")
            print(f"[ERROR] Make sure EXECUTING_LLM_BASE_URL is set and the vLLM server is running")
            raise RuntimeError(f"LLM request failed: {exc.reason}")

    return "", [], 0, 0  # unreachable, satisfies type checkers


def _create_tool_selector(mode: str) -> ToolSelector:
    """Create a tool selector based on mode."""
    mode = mode.lower() if mode else "in_context"
    
    if mode == "hierarchical":
        return HierarchicalToolSelector()
    elif mode == "embedding":
        return OpenAIEmbeddingBasedToolSelector(top_k=5)
    elif mode == "embedding_context":
        return EmbeddingContextBasedToolSelector(top_k=5)
    elif mode == "embedding_reranker":
        return OpenAIEmbeddingWithLLMRerankerToolSelector(top_k=5, initial_k=10)
    elif mode == "embedding_context_reranker":
        return OpenAIEmbeddingContextWithLLMRerankerToolSelector(top_k=5, initial_k=10)
    elif mode == "qwen3_embedding":
        return Qwen3EmbeddingBasedToolSelector(top_k=5)
    elif mode == "qwen3_embedding_context":
        return Qwen3EmbeddingContextBasedToolSelector(top_k=5)
    elif mode == "qwen3_embedding_reranker":
        return Qwen3EmbeddingWithLLMRerankerToolSelector(top_k=5, initial_k=10)
    elif mode == "qwen3_embedding_context_reranker":
        return Qwen3EmbeddingContextWithLLMRerankerToolSelector(top_k=5, initial_k=10)
    elif mode == "qwen3_reranker":
        return Qwen3RerankerBasedToolSelector(top_k=5)
    elif mode == "qwen3_embedding_qwen3_reranker":
        return Qwen3EmbeddingWithQwen3RerankerToolSelector(top_k=5, initial_k=20)
    elif mode == "qwen3_embedding_context_qwen3_reranker":
        return Qwen3EmbeddingContextWithQwen3RerankerToolSelector(top_k=5, initial_k=20)
    else:  # default: in_context
        return InContextToolSelector()


def _build_graph(mode: str = "in_context"):
    """Build a StateGraph with configurable tool selection strategy.
    
    Args:
        mode: One of "in_context", "hierarchical", "embedding", "embedding_reranker"
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    class GraphState(TypedDict):
        messages: list
        tools: list
        selected_tools: list
        response: str
        tool_calls: list
        selection_mode: str
        input_tokens: int
        output_tokens: int

    builder = StateGraph(GraphState)
    selector = _create_tool_selector(mode)

    def tool_selection_node(state: GraphState):
        """Node 1: Select relevant tools based on strategy."""
        messages = state.get("messages", [])
        tools = state.get("tools", [])
        selected = selector.select(messages, tools)
        return {"selected_tools": selected}

    def llm_execution_node(state: GraphState):
        """Node 2: Execute LLM with selected tools."""
        messages = state.get("messages", [])
        selected_tools = state.get("selected_tools", [])
        
        # Invoke LLM with selected tools
        response, tool_calls, input_tokens, output_tokens = _invoke_llm(messages, selected_tools)
        
        return {
            "response": response,
            "tool_calls": tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    # Build graph with both nodes
    builder.add_node("tool_selection", tool_selection_node)
    builder.add_node("llm_execution", llm_execution_node)
    
    builder.set_entry_point("tool_selection")
    builder.add_edge("tool_selection", "llm_execution")
    builder.add_edge("llm_execution", END)
    
    return builder.compile()


# Initialize default graph (can be overridden per request)
_default_mode = os.getenv("LANGGRAPH_TOOL_SELECTION_MODE", "in_context")
GRAPH = _build_graph(_default_mode)


class LangGraphLocalHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/execute", "/"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": "invalid json payload"}, status=400)
            return

        inp = payload.get("input", {})
        messages = inp.get("messages", [])
        tools = inp.get("tools", [])
        
        # Support selection_mode in payload (can override default)
        selection_mode = payload.get("selection_mode") or inp.get("selection_mode") or _default_mode

        # Set per-request context so log helpers can identify which task/strategy
        # produced each log entry without threading params through every layer.
        _request_context.test_entry_id = payload.get("test_entry_id") or inp.get("test_entry_id")
        _request_context.task_idx = payload.get("task_idx") if payload.get("task_idx") is not None else inp.get("task_idx")
        _request_context.selection_mode = selection_mode
        
        print(f"\n[SERVER] Request received:")
        print(f"  Default mode: {_default_mode}")
        print(f"  Payload selection_mode: {payload.get('selection_mode')}")
        print(f"  Input selection_mode: {inp.get('selection_mode')}")
        print(f"  Effective mode: {selection_mode}")
        print(f"  Messages: {len(messages)}, Tools: {len(tools)}")
        
        # Build graph for this specific mode if different from default
        graph = GRAPH
        if selection_mode != _default_mode:
            print(f"  Building new graph for mode: {selection_mode}")
            graph = _build_graph(selection_mode)
        else:
            print(f"  Using default graph (mode: {_default_mode})")

        start = time.time()
        try:
            if graph is not None:
                # Execute graph with state
                result = graph.invoke({
                        "messages": messages,
                        "tools": tools,
                        "selected_tools": [],
                        "response": "",
                        "tool_calls": [],
                        "selection_mode": selection_mode,
                        "input_tokens": 0,
                        "output_tokens": 0,
                    })
                
                # Extract results from state
                if isinstance(result, dict):
                    content = result.get("response") or ""
                    tool_calls = result.get("tool_calls") or []
                    input_tokens = result.get("input_tokens", 0)
                    output_tokens = result.get("output_tokens", 0)
                else:
                    content = str(result)
                    tool_calls = []
                    input_tokens = 0
                    output_tokens = 0
            else:
                # Fallback: invoke LLM directly if graph not available
                content, tool_calls, input_tokens, output_tokens = _invoke_llm(messages, tools)
        except Exception as exc:
            import traceback
            error_msg = str(exc)
            print(f"\n[ERROR] Graph execution failed: {error_msg}")
            traceback.print_exc()
            self._send_json({"error": error_msg}, status=500)
            return

        latency = time.time() - start

        response = {
            "content": content,
            "reasoning_content": None,
            "tool_calls": tool_calls,
            "input_token": input_tokens,
            "output_token": output_tokens,
            "latency": latency,
            "selection_mode": selection_mode,
        }

        self._send_json(response)


def run(host="127.0.0.1", port=8001):
    server = HTTPServer((host, port), LangGraphLocalHandler)
    print(f"\n{'='*70}")
    print(f"LangGraph local server listening at http://{host}:{port}/execute")
    print(f"{'='*70}")
    print(f"Tool Selection Mode: {_default_mode}")
    print(f"\nExecuting LLM Configuration:")
    print(f"  EXECUTING_LLM_BASE_URL: {os.getenv('EXECUTING_LLM_BASE_URL', 'NOT SET')}")
    print(f"  EXECUTING_LLM_MODEL: {os.getenv('EXECUTING_LLM_MODEL', 'openai/gpt-oss-120b (default)')}")
    print(f"  EXECUTING_LLM_API_KEY: {'***SET***' if os.getenv('EXECUTING_LLM_API_KEY') else 'EMPTY (default)'}")
    print(f"\nAvailable modes:")
    print(f"  1. 'in_context' - LLM decides which tools to use (default)")
    print(f"  2. 'hierarchical' - Smaller LLM selects relevant tools first")
    print(f"  3. 'embedding' - OpenAI text-embedding-3-small (cached)")
    print(f"  4. 'embedding_reranker' - Embeddings + LLM reranking")
    print(f"\nOpenAI Embedding Configuration (for mode 'embedding'):")
    print(f"  OPENAI_API_KEY: {'***SET***' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"  OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')}")
    print(f"\nTo switch modes, include 'selection_mode' in the request payload")
    print(f"or set LANGGRAPH_TOOL_SELECTION_MODE environment variable")
    print(f"{'='*70}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        server.shutdown()


if __name__ == "__main__":
    host = os.getenv("LANGGRAPH_HOST", "127.0.0.1")
    port = int(os.getenv("LANGGRAPH_PORT", "8001"))
    run(host=host, port=port)
