import os
import json
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from abc import ABC, abstractmethod
from typing import TypedDict, Any, Optional

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
        """Return all 618 valid tools from schema cache for in-context selection."""
        # Use schema cache tools instead of request tools
        all_tools = self.tools_cache if self.tools_cache else tools
        
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
    
    def __init__(self, max_tools: int = 10, schema_cache_file: str = None):
        self.endpoint = os.getenv("LANGGRAPH_SELECTOR_LLM_ENDPOINT")
        self.api_key = os.getenv("LANGGRAPH_SELECTOR_LLM_API_KEY")
        self.model = os.getenv("LANGGRAPH_SELECTOR_LLM_MODEL", "Qwen/Qwen3-30B-A3B")
        self.max_tools = max_tools
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
        
        selection_prompt = f"""Given the user query, select the most relevant tools from the available list.
        
User Query: {query}

Available Tools:
{tool_descriptions}

Return a JSON array of tool names you would use, e.g. ["getTool1", "getTool2"].
Return ONLY the JSON array, no other text."""
        
        response = self._invoke_selector_llm(selection_prompt)
        selected_names = json.loads(response)
        
        # Filter tools by selected names
        selected = [t for t in all_tools if t.get("function", {}).get("name") in selected_names]
        
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
        
        return selected[:self.max_tools]
    
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
        }
        
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "LangGraph-Selector/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        req = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
                parsed = json.loads(data)
                
                print(f"[DEBUG] Selector response received (status: {resp.status})")
                
                # Extract content from OpenAI-compatible response
                if isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                    first = parsed["choices"][0]
                    if isinstance(first, dict) and "message" in first:
                        msg = first["message"]
                        if isinstance(msg, dict) and "content" in msg:
                            content = msg["content"]
                            print(f"[DEBUG] Extracted content from response: {content[:100]}")
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
    """Strategy 4: OpenAI embedding-based retrieval with DeepSeek LLM reranking.
    
    Uses OpenAI embeddings for initial retrieval, then reranks with DeepSeek LLM.
    Allows the LLM to exclude irrelevant tools.
    
    Retrieval phase: Get top-k candidates via embedding similarity
    Reranking phase: LLM reorders candidates and can exclude irrelevant ones
    """
    
    def __init__(self, top_k: int = 5, initial_k: int = 10):
        self.embedding_selector = OpenAIEmbeddingBasedToolSelector(top_k=initial_k)
        self.llm_endpoint = os.getenv("LANGGRAPH_LLM_ENDPOINT")
        self.llm_api_key = os.getenv("LANGGRAPH_LLM_API_KEY")
        self.top_k = top_k
        self.initial_k = initial_k  # Retrieve more candidates to rerank
    
    def select(self, messages: list, tools: list) -> list:
        """Select tools: embeddings first (top-10), then LLM rerank."""
        # First pass: embedding-based retrieval to get candidates
        candidates = self.embedding_selector.select(messages, tools)
        
        if len(candidates) <= self.top_k:
            return candidates
        
        # Second pass: LLM reranking to filter and order
        if not self.llm_endpoint:
            raise ValueError("LANGGRAPH_LLM_ENDPOINT environment variable must be set for embedding_reranker mode")
        
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found in messages for embedding-reranker tool selection")
        
        reranked = self._rerank_with_llm(query, candidates)
        return reranked[:self.top_k]
    
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
        
        payload = {
            "messages": [{"role": "user", "content": rerank_prompt}],
            "model": "deepseek-chat",
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
    then reranks with DeepSeek LLM. Allows the LLM to exclude irrelevant tools.
    
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
        self.llm_endpoint = os.getenv("LANGGRAPH_LLM_ENDPOINT")
        self.llm_api_key = os.getenv("LANGGRAPH_LLM_API_KEY")
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
        self.llm_endpoint = os.getenv("LANGGRAPH_LLM_ENDPOINT")
        self.llm_api_key = os.getenv("LANGGRAPH_LLM_API_KEY")
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
        self.llm_endpoint = os.getenv("LANGGRAPH_LLM_ENDPOINT")
        self.llm_api_key = os.getenv("LANGGRAPH_LLM_API_KEY")
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


# ==================== LLM Invocation ====================

def _invoke_llm(messages: list, tools: list = None):
    """Invoke a local OpenAI/vLLM-compatible endpoint.
    
    Returns: (content, tool_calls) tuple
    """
    endpoint = os.getenv("LANGGRAPH_LLM_ENDPOINT")
    api_key = os.getenv("LANGGRAPH_LLM_API_KEY")
    
    if not endpoint:
        # Simulated response for testing without a real LLM
        joined = "\n".join([str(m) for m in messages])
        return f"[simulated response] received {len(messages)} messages: {joined[:200]}", []

    payload = {
        "messages": messages,
        "model": "deepseek-chat",  # Required by DeepSeek API
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
    print(f"[DEBUG] Payload keys: {list(payload.keys())}")
    print(f"[DEBUG] Messages: {len(messages)}, Tools: {len(tools or [])}")

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode("utf-8")
            print(f"[DEBUG] LLM response status: {resp.status}")
            try:
                parsed = json.loads(data)
            except Exception as e:
                print(f"[DEBUG] Failed to parse JSON: {e}")
                return data, []

            # Extract content
            content = ""
            if isinstance(parsed, dict) and "content" in parsed:
                content = parsed["content"]
            elif isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                first = parsed["choices"][0]
                if isinstance(first, dict) and "message" in first:
                    msg = first["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        content = msg["content"]
            
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
            
            print(f"[DEBUG] Extracted content length: {len(content)}, tool_calls: {len(tool_calls)}")
            return content, tool_calls
    except urllib.error.HTTPError as exc:
        error_data = exc.read().decode("utf-8")
        print(f"[ERROR] LLM HTTP Error {exc.code}: {exc.reason}")
        print(f"[ERROR] Response body: {error_data[:500]}")
        raise RuntimeError(f"LLM request failed: {exc.code} {exc.reason} - {error_data[:200]}")
    except urllib.error.URLError as exc:
        print(f"[ERROR] LLM URL Error: {exc.reason}")
        raise RuntimeError(f"LLM request failed: {exc.reason}")


def _create_tool_selector(mode: str) -> ToolSelector:
    """Create a tool selector based on mode."""
    mode = mode.lower() if mode else "in_context"
    
    if mode == "hierarchical":
        return HierarchicalToolSelector(max_tools=10)
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
        response, tool_calls = _invoke_llm(messages, selected_tools)
        
        return {
            "response": response,
            "tool_calls": tool_calls,
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
                try:
                    result = graph.invoke({
                        "messages": messages,
                        "tools": tools,
                        "selected_tools": [],
                        "response": "",
                        "tool_calls": [],
                        "selection_mode": selection_mode,
                    })
                except TypeError:
                    # Fallback for different langgraph versions
                    result = graph({
                        "messages": messages,
                        "tools": tools,
                        "selected_tools": [],
                        "response": "",
                        "tool_calls": [],
                        "selection_mode": selection_mode,
                    })
                
                # Extract results from state
                if isinstance(result, dict):
                    content = result.get("response") or ""
                    tool_calls = result.get("tool_calls") or []
                else:
                    content = str(result)
                    tool_calls = []
            else:
                # Fallback: invoke LLM directly if graph not available
                content, tool_calls = _invoke_llm(messages, tools)
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
            "input_token": 0,
            "output_token": 0,
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
    print(f"\nLLM Configuration:")
    print(f"  LANGGRAPH_LLM_ENDPOINT: {os.getenv('LANGGRAPH_LLM_ENDPOINT', 'NOT SET')}")
    print(f"  LANGGRAPH_LLM_API_KEY: {'***SET***' if os.getenv('LANGGRAPH_LLM_API_KEY') else 'NOT SET'}")
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
