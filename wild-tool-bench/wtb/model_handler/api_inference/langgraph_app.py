import os
import json
import copy
import time
import threading
import urllib.request
import urllib.error
import uuid
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from abc import ABC, abstractmethod
from typing import TypedDict, Any, Optional

# Per-request context (thread-safe).  Set in do_POST, read by log helpers.
_request_context = threading.local()
_log_lock = threading.Lock()


def _valid_embedding(value: Any) -> bool:
    """Return whether an embedding is a non-empty finite numeric vector."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)
    )

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


def _append_task_log(path: str, entry: dict):
    """Append one request/turn without losing other turns of the same task."""
    with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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
            "request_id": getattr(_request_context, "request_id", None),
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

        _append_task_log(log_path, log_entry)
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
            "request_id": getattr(_request_context, "request_id", None),
            "strategy": strategy_name,
            "query": query[:500],  # Limit query length
            "available_tools_count": available_tools_count,
            "selected_tools_count": len(selected_tools),
            "selected_tool_names": selected_tool_names,
            "metadata": selection_metadata or {}
        }
        
        _append_task_log(log_path, log_entry)
            
    except Exception as e:
        print(f"[WARNING] Failed to log tool selection: {e}")


def _embedding_usage(response) -> dict:
    """Extract embedding input-token usage when the provider exposes it."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


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
    """
    In-context baseline:

    Pass to the executor:
      1. exactly the tools supplied by WTB for the current task
      2. all synthetic distractor tools stored in tool_schemas_cache.jsonl

    The WTB request tools are the source of truth for the real task tools,
    including their task-specific `required` fields.

    The schema cache is assumed to contain ONLY the filtered synthetic tools.
    """

    def __init__(self, schema_cache_file: str = None):
        self.schema_cache_file = (
            schema_cache_file
            or os.getenv("LANGGRAPH_TOOL_SCHEMAS_CACHE_FILE")
            or os.path.join(
                os.path.dirname(__file__),
                "tool_schemas_cache.jsonl",
            )
        )

        self.synthetic_tools = []
        self._load_synthetic_tools()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_description(value: Any) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    @staticmethod
    def _sanitize_tool(tool: dict) -> Optional[dict]:
        """
        Convert a tool to the standard OpenAI function-tool schema.

        Private benchmark metadata such as `_synthetic` and `_source_tool`
        is intentionally not forwarded to the executor.
        """
        if not isinstance(tool, dict):
            return None

        func = tool.get("function")
        if not isinstance(func, dict):
            return None

        name = str(func.get("name", "")).strip()
        description = str(func.get("description", "")).strip()
        parameters = func.get("parameters")

        if not name or not description or not isinstance(parameters, dict):
            return None

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": copy.deepcopy(parameters),
            },
        }

    @staticmethod
    def _full_tool_signature(tool: dict) -> tuple:
        """
        Full schema identity.

        This is used only for exact duplicate detection. Unlike the old
        name+description deduplication, tools with genuinely different
        parameter schemas are NOT collapsed.
        """
        func = tool.get("function", {}) if isinstance(tool, dict) else {}

        name = str(func.get("name", "")).strip()

        description = " ".join(
            str(func.get("description", "")).split()
        ).strip().lower()

        parameters = json.dumps(
            func.get("parameters", {}),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        return name, description, parameters

    @staticmethod
    def _extract_tools_from_object(obj: Any) -> list:
        """
        Accept several convenient JSON/JSONL representations.

        A JSONL line may be:
          - one tool object
          - a list of tools
          - {"tools": [...]}
          - {"tool_schemas": [...]}
          - {"available_tools": [...]}
        """
        if isinstance(obj, list):
            return obj

        if not isinstance(obj, dict):
            return []

        # A single OpenAI tool object.
        if isinstance(obj.get("function"), dict):
            return [obj]

        for key in ("tools", "tool_schemas", "available_tools"):
            value = obj.get(key)
            if isinstance(value, list):
                return value

        return []

    # ------------------------------------------------------------------
    # Synthetic cache loading
    # ------------------------------------------------------------------

    def _load_synthetic_tools(self):
        """
        Load ONLY synthetic distractor tools from tool_schemas_cache.jsonl.

        Supports both:
          - proper JSONL: one JSON object/list per line
          - legacy JSON: {"tools": [...]} or [...]
        """
        if not os.path.exists(self.schema_cache_file):
            print(
                f"[WARNING] Synthetic schema cache not found: "
                f"{self.schema_cache_file}"
            )
            self.synthetic_tools = []
            return

        loaded_tools = []

        try:
            # ----------------------------------------------------------
            # First try ordinary JSON.
            # This keeps compatibility if the file is actually a JSON
            # document despite using a .jsonl extension.
            # ----------------------------------------------------------
            try:
                with open(
                    self.schema_cache_file,
                    "r",
                    encoding="utf-8",
                ) as f:
                    data = json.load(f)

                loaded_tools.extend(
                    self._extract_tools_from_object(data)
                )

            except json.JSONDecodeError:
                # ------------------------------------------------------
                # Otherwise parse as true JSONL.
                # ------------------------------------------------------
                with open(
                    self.schema_cache_file,
                    "r",
                    encoding="utf-8",
                ) as f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError as exc:
                            print(
                                f"[WARNING] Invalid JSON in synthetic "
                                f"tool cache line {line_no}: {exc}"
                            )
                            continue

                        loaded_tools.extend(
                            self._extract_tools_from_object(entry)
                        )

            # ----------------------------------------------------------
            # Sanitize + remove only EXACT duplicates.
            # ----------------------------------------------------------
            sanitized = []
            seen = set()

            dropped_invalid = 0
            dropped_exact_duplicate = 0

            for tool in loaded_tools:
                clean_tool = self._sanitize_tool(tool)

                if clean_tool is None:
                    dropped_invalid += 1
                    continue

                signature = self._full_tool_signature(clean_tool)

                if signature in seen:
                    dropped_exact_duplicate += 1
                    continue

                seen.add(signature)
                sanitized.append(clean_tool)

            self.synthetic_tools = sanitized

            print(
                "[IN-CONTEXT SELECTOR] Loaded synthetic distractor cache:"
            )
            print(f"  Raw tools: {len(loaded_tools)}")
            print(f"  Valid synthetic tools: {len(self.synthetic_tools)}")
            print(f"  Invalid dropped: {dropped_invalid}")
            print(
                f"  Exact duplicates dropped: "
                f"{dropped_exact_duplicate}"
            )

        except Exception as exc:
            print(
                f"[WARNING] Failed to load synthetic tool cache "
                f"{self.schema_cache_file}: {exc}"
            )
            self.synthetic_tools = []

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, messages: list, tools: list) -> list:
        """
        Return:

            task-specific WTB tools
            +
            filtered synthetic distractors

        The incoming WTB tools are preserved exactly, including their
        task-specific required fields.
        """

        # --------------------------------------------------------------
        # 1. Keep the task tools supplied by WTB.
        #
        # Do NOT replace them with global-cache versions and do NOT
        # overwrite their `required` fields.
        # --------------------------------------------------------------
        task_tools = []

        for tool in tools or []:
            clean_tool = self._sanitize_tool(tool)

            if clean_tool is not None:
                task_tools.append(clean_tool)

        # --------------------------------------------------------------
        # 2. Task tool names must win over synthetic tool names.
        #
        # Tool calls identify functions by NAME. If a synthetic tool has
        # exactly the same function name as a real task tool, the executor
        # cannot unambiguously distinguish them when it returns:
        #
        #     {"name": "..."}
        #
        # Therefore skip synthetic name collisions for this task.
        # --------------------------------------------------------------
        task_tool_names = {
            tool.get("function", {}).get("name")
            for tool in task_tools
        }

        selected_tools = list(task_tools)

        skipped_name_collision = 0
        skipped_exact_duplicate = 0

        # Exact signatures already represented by real task tools.
        seen_signatures = {
            self._full_tool_signature(tool)
            for tool in task_tools
        }

        for synthetic_tool in self.synthetic_tools:
            synthetic_name = (
                synthetic_tool
                .get("function", {})
                .get("name")
            )

            # Real task tool always has priority.
            if synthetic_name in task_tool_names:
                skipped_name_collision += 1
                continue

            signature = self._full_tool_signature(synthetic_tool)

            if signature in seen_signatures:
                skipped_exact_duplicate += 1
                continue

            seen_signatures.add(signature)
            selected_tools.append(
                copy.deepcopy(synthetic_tool)
            )

        # --------------------------------------------------------------
        # 3. Query only for logging.
        # --------------------------------------------------------------
        query = ""

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")

                if isinstance(content, str):
                    query = content[:500]

                break

        # --------------------------------------------------------------
        # 4. Log exactly what is being sent to the executor.
        # --------------------------------------------------------------
        log_tool_selection(
            strategy_name="in_context",
            query=query,
            available_tools_count=len(selected_tools),
            selected_tools=selected_tools,
            selection_metadata={
                "method": "wtb_task_tools_plus_synthetic_distractors",
                "wtb_task_tools": len(task_tools),
                "synthetic_cache_tools": len(self.synthetic_tools),
                "synthetic_tools_added": (
                    len(selected_tools) - len(task_tools)
                ),
                "synthetic_name_collisions_skipped": (
                    skipped_name_collision
                ),
                "exact_duplicates_skipped": (
                    skipped_exact_duplicate
                ),
            },
        )

        # --------------------------------------------------------------
        # 5. Console diagnostics.
        # --------------------------------------------------------------
        print("\n[IN-CONTEXT SELECTOR]")
        print(
            f"  WTB task-specific tools: "
            f"{len(task_tools)}"
        )
        print(
            f"  Synthetic tools in cache: "
            f"{len(self.synthetic_tools)}"
        )
        print(
            f"  Synthetic name collisions skipped: "
            f"{skipped_name_collision}"
        )
        print(
            f"  Exact duplicates skipped: "
            f"{skipped_exact_duplicate}"
        )
        print(
            f"  Synthetic distractors added: "
            f"{len(selected_tools) - len(task_tools)}"
        )
        print(
            f"  TOTAL tools sent to executor: "
            f"{len(selected_tools)}"
        )

        print("  Task tools:")

        for tool in task_tools:
            name = (
                tool
                .get("function", {})
                .get("name", "unknown")
            )
            print(f"    [WTB] {name}")

        print()

        return selected_tools


class HierarchicalToolSelector(ToolSelector):
    """Strategy 2: Hierarchical selection with a smaller LLM.
    
    Uses a smaller/faster LLM to select which tools are relevant from all 618 valid tools,
    then passes only those to the main LLM.
    """
    
    def __init__(self, schema_cache_file: str = None):
        self.endpoint = os.getenv("LANGGRAPH_SELECTOR_LLM_ENDPOINT")
        self.api_key = os.getenv("LANGGRAPH_SELECTOR_LLM_API_KEY")
        self.model = os.getenv("LANGGRAPH_SELECTOR_LLM_MODEL", "Qwen/Qwen3-30B-A3B")
        self.max_context_tokens = int(os.getenv("LANGGRAPH_SELECTOR_MAX_CONTEXT_TOKENS", "40960"))
        self.max_output_tokens = int(os.getenv("LANGGRAPH_SELECTOR_MAX_OUTPUT_TOKENS", "400"))
        self.prompt_headroom_tokens = int(os.getenv("LANGGRAPH_SELECTOR_PROMPT_HEADROOM_TOKENS", "1024"))
        self.max_conversation_chars = int(os.getenv("LANGGRAPH_SELECTOR_MAX_CONVERSATION_CHARS", "6000"))
        self.max_tool_desc_chars = int(os.getenv("LANGGRAPH_SELECTOR_MAX_TOOL_DESC_CHARS", "180"))
        # Controls what conversation context is included in selector prompt:
        # - latest_user: only latest user message (default)
        # - full: bounded full history
        # - none: no conversation context
        self.selector_context_mode = (
            os.getenv("LANGGRAPH_SELECTOR_CONTEXT_MODE", "latest_user") or "latest_user"
        ).strip().lower()
        self.last_selection_metrics = {}
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
                        loaded_tools = data["tools"]
                    else:
                        loaded_tools = data if isinstance(data, list) else []

                    # Deduplicate by normalized name + description to reduce
                    # prompt bloat without removing distinct tool semantics.
                    seen_keys = set()
                    deduped_tools = []
                    duplicates = 0
                    for tool in loaded_tools:
                        func = tool.get("function", {}) if isinstance(tool, dict) else {}
                        name = str(func.get("name", "")).strip()
                        desc = " ".join(str(func.get("description", "")).split()).strip().lower()
                        key = (name, desc)
                        if key in seen_keys:
                            duplicates += 1
                            continue
                        seen_keys.add(key)
                        deduped_tools.append(tool)

                    self.tools_cache = deduped_tools
                    print(
                        f"[HIERARCHICAL SELECTOR] Loaded {len(loaded_tools)} tools from schema cache, "
                        f"deduplicated to {len(self.tools_cache)} (removed {duplicates})"
                    )
            except Exception as e:
                print(f"[WARNING] Failed to load schema cache: {e}")
                self.tools_cache = []
        else:
            print(f"[WARNING] Schema cache file not found: {self.schema_cache_file}")
            self.tools_cache = []
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (safe over-approximation for Latin text)."""
        return max(1, (len(text) + 3) // 4)

    def _prompt_token_budget(self) -> int:
        budget = self.max_context_tokens - self.max_output_tokens - self.prompt_headroom_tokens
        return max(1024, budget)

    def _serialize_messages_for_selector(self, messages: list, char_budget: int = None) -> str:
        """Render conversation history with size limits to avoid context overflow."""
        if char_budget is None:
            char_budget = self.max_conversation_chars

        rendered = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            elif not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if not content:
                continue
            rendered.append((role, content))

        if not rendered:
            return "(no conversation history)"

        # Keep newest turns first under a char budget since latest user turns are
        # usually the strongest signal for tool relevance.
        kept = []
        used_chars = 0
        for role, content in reversed(rendered):
            item = f"{role}: {content}"
            item_len = len(item) + 8
            if kept and used_chars + item_len > char_budget:
                break
            kept.append((role, content))
            used_chars += item_len

        kept.reverse()
        lines = []
        for idx, (role, content) in enumerate(kept, start=1):
            lines.append(f"{idx}. {role}: {content}")
        return "\n".join(lines)

    def _latest_user_message(self, messages: list) -> str:
        """Return latest user message for compact selector context."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            elif not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if content:
                return f"1. user: {content}"
        return "(no user message)"

    def _selector_context_text(self, messages: list, char_budget: int) -> str:
        """Build selector context according to configured context mode."""
        mode = self.selector_context_mode
        if mode == "none":
            return "(no conversation context)"
        if mode == "latest_user":
            return self._latest_user_message(messages)
        return self._serialize_messages_for_selector(messages, char_budget)

    def _build_selector_prompt(self, messages: list, tools: list) -> tuple[str, int]:
        """Build selector prompt while keeping full tool list and trimming only history."""
        base_conversation_budget = self.max_conversation_chars
        prompt_header = (
            "Given the conversation so far, select the most relevant tools from the available list. /no_think\n\n"
            "Conversation History:\n"
            "{conversation_history}\n\n"
            "Available Tools (full list):\n"
        )
        prompt_footer = (
            "\n\nReturn a JSON array of at most 10 objects with BOTH name and description, "
            "e.g. [{\"name\": \"getTool1\", \"description\": \"...\"}].\n"
            "For each selected tool, copy the description text from the list verbatim.\n"
            "If multiple tools share the same name, the description is mandatory for disambiguation.\n"
            "Return ONLY the JSON array, no other text."
        )

        tool_text = self._format_tools(tools)
        token_budget = self._prompt_token_budget()

        # Keep the full tool list and full descriptions. Only reduce conversation history.
        attempted_budget = base_conversation_budget
        conversation_history = self._selector_context_text(messages, attempted_budget)
        final_prompt = (prompt_header.format(conversation_history=conversation_history) + tool_text + prompt_footer)
        est_tokens = self._estimate_tokens(final_prompt)
        while est_tokens > token_budget and attempted_budget > 0 and self.selector_context_mode == "full":
            attempted_budget = max(0, attempted_budget // 2)
            conversation_history = self._selector_context_text(messages, attempted_budget)
            final_prompt = (prompt_header.format(conversation_history=conversation_history) + tool_text + prompt_footer)
            est_tokens = self._estimate_tokens(final_prompt)

        tools_in_prompt = len(tools)
        print(
            f"[HIERARCHICAL SELECTOR] prompt_est_tokens={est_tokens}, "
            f"budget={token_budget}, tools_in_prompt={tools_in_prompt}/{tools_in_prompt}, "
            f"conversation_chars_budget={attempted_budget}, context_mode={self.selector_context_mode}"
        )
        return final_prompt, attempted_budget

    def _normalize_text(self, value: str) -> str:
        """Normalize text for robust selector-output matching."""
        return " ".join(str(value or "").split()).strip().lower()

    def _parse_selector_output(self, raw_response: str) -> tuple[list[dict], list[str]]:
        """Parse selector output into a normalized list of tool specs."""
        import re as _re

        clean = _re.sub(r"<think>.*?</think>", "", raw_response, flags=_re.DOTALL).strip()
        m = _re.search(r"\[.*?\]", clean, _re.DOTALL)
        if not m:
            print(f"[WARNING] Selector LLM returned no JSON array; got: {clean[:200]}")
            return [], []

        parsed = json.loads(m.group())
        if not isinstance(parsed, list):
            return [], []

        specs = []
        names = []
        for item in parsed:
            if isinstance(item, str):
                name = item.strip()
                if not name:
                    continue
                specs.append({"name": name, "description": ""})
                names.append(name)
                continue

            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            if not name:
                continue
            description = str(item.get("description", "")).strip()
            specs.append({"name": name, "description": description})
            names.append(name)

        return specs, names

    def _match_selected_tools(self, all_tools: list, selected_specs: list[dict]) -> list:
        """Match selector specs to tools, preferring description-aware disambiguation."""
        selected = []
        seen = set()

        for spec in selected_specs[:10]:
            name = str(spec.get("name", "")).strip()
            desc_hint = self._normalize_text(spec.get("description", ""))
            if not name:
                continue

            candidates = [
                t for t in all_tools
                if str(t.get("function", {}).get("name", "")).strip() == name
            ]
            if not candidates:
                continue

            best = candidates[0]
            if desc_hint:
                scored = []
                for tool in candidates:
                    desc = self._normalize_text(tool.get("function", {}).get("description", ""))
                    score = 0
                    if desc == desc_hint:
                        score = 4
                    elif desc.startswith(desc_hint) or desc_hint.startswith(desc):
                        score = 3
                    elif desc_hint in desc:
                        score = 2
                    scored.append((score, len(desc), tool))
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best = scored[0][2]

            key = (
                str(best.get("function", {}).get("name", "")).strip(),
                self._normalize_text(best.get("function", {}).get("description", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(best)

        return selected

    def select(self, messages: list, tools: list) -> list:
        """Use a small LLM to select relevant tools from all 618 valid tools."""
        if not self.endpoint:
            raise ValueError("LANGGRAPH_SELECTOR_LLM_ENDPOINT environment variable must be set for hierarchical mode")
        
        # Use schema cache tools instead of request tools
        all_tools = self.tools_cache if self.tools_cache else tools
        
        if not all_tools:
            return []
        
        query = self._extract_query(messages)
        selection_prompt, used_conversation_budget = self._build_selector_prompt(messages, all_tools)
        
        response, selector_usage, selector_latency = self._invoke_selector_llm(selection_prompt)
        selector_input_tokens = int(selector_usage.get("prompt_tokens", 0) or 0)
        selector_output_tokens = int(selector_usage.get("completion_tokens", 0) or 0)
        self.last_selection_metrics = {
            "selector_input_tokens": selector_input_tokens,
            "selector_output_tokens": selector_output_tokens,
            "selector_total_tokens": selector_input_tokens + selector_output_tokens,
            "selector_conversation_chars_budget": used_conversation_budget,
        }
        selected_specs, selected_names = self._parse_selector_output(response)
        selected = self._match_selected_tools(all_tools, selected_specs)
        
        # Log tool selection
        log_tool_selection(
            strategy_name="hierarchical",
            query=query,
            available_tools_count=len(all_tools),
            selected_tools=selected,
            selection_metadata={
                "model": self.model,
                "selected_tool_names": selected_names,
                "selected_tool_specs": selected_specs,
                "selector_input_tokens": selector_input_tokens,
                "selector_output_tokens": selector_output_tokens,
                "selector_latency_s": selector_latency,
                "selector_conversation_chars_budget": used_conversation_budget,
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
        """Format full tool list for the selector LLM with compact descriptions."""
        lines = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = (func.get("description", "") or "").strip().replace("\n", " ")
            if self.max_tool_desc_chars > 0 and len(desc) > self.max_tool_desc_chars:
                desc = desc[: self.max_tool_desc_chars - 3].rstrip() + "..."
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
    
    def _invoke_selector_llm(self, prompt: str) -> tuple[str, dict, float]:
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
            "max_tokens": self.max_output_tokens,
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
        started = time.perf_counter()
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
                            return content, usage, time.perf_counter() - started
                
                # Fallback: return raw data
                print(f"[DEBUG] Unexpected response format, returning raw: {str(parsed)[:200]}")
                return str(parsed), parsed.get("usage", {}) if isinstance(parsed, dict) else {}, time.perf_counter() - started
                
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


class ToolReActToolSelector(ToolSelector):
    """Strategy: ReAct-style iterative tool retrieval with the executing LLM.

    The selector runs a bounded ReAct loop where the model can call exactly one
    registered tool ("tool_retreiver") with a subquery. That tool performs
    Qwen3-Embedding retrieval over the full tool catalog and returns candidates.
    The model may iterate, refine subqueries, and finally return a JSON array of
    tool names to use.
    """

    SYSTEM_PROMPT = (
        "You are a ReAct-style tool selection agent for benchmark tasks. "
        "Your goal is to decide which tools should be available to the executor model.\n\n"
        "Tools:\n"
        "- You may use exactly one callable tool: tool_retreiver(subquery: string, k?: integer).\n"
        "- Use it to retrieve candidate tools relevant to a focused subquery.\n"
        "- You may call tool_retreiver multiple times and refine subqueries across iterations.\n\n"
        "Process (ReAct policy):\n"
        "1) Think about whether retrieval is needed.\n"
        "2) If needed, call tool_retreiver.\n"
        "3) Read observations, update your plan, and optionally retrieve again.\n"
        "4) Aggregate information from all previous observations.\n"
        "5) Finish when confident.\n\n"
        "Output requirements:\n"
        "- Final output must be ONLY a JSON array of tool names.\n"
        "- Example: [\"getWeather\", \"getForecast\"]\n"
        "- If no tools are needed, output []\n"
        "- Do not output markdown, explanations, or extra keys in the final output."
    )

    def __init__(self, schema_cache_file: str = None, max_iter: int = 10):
        self.max_iter = int(os.getenv("LANGGRAPH_TOOLREAGT_MAX_ITER", str(max_iter)))
        self.schema_cache_file = schema_cache_file or os.path.join(
            os.path.dirname(__file__),
            "tool_schemas_cache.json"
        )
        self.tools_cache = None
        self.last_selection_metrics = {}
        self._embedding_retriever = Qwen3EmbeddingBasedToolSelector(top_k=5, schema_cache_file=self.schema_cache_file)
        self._load_schema_cache()

    def _load_schema_cache(self):
        """Load all valid tools from schema cache file."""
        if os.path.exists(self.schema_cache_file):
            try:
                with open(self.schema_cache_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "tools" in data:
                        self.tools_cache = data["tools"]
                    else:
                        self.tools_cache = data if isinstance(data, list) else []
                print(f"[TOOLREAGT SELECTOR] Loaded {len(self.tools_cache)} valid tools from schema cache")
            except Exception as e:
                print(f"[WARNING] Failed to load schema cache for toolreagt: {e}")
                self.tools_cache = []
        else:
            print(f"[WARNING] Schema cache file not found for toolreagt: {self.schema_cache_file}")
            self.tools_cache = []

    def _extract_full_conversation(self, messages: list) -> str:
        """Extract full multi-turn context for selector reasoning."""
        parts = []
        for msg in messages:
            role = str(msg.get("role", "")).upper()
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"[{role}]: {content}")
        return "\n".join(parts)[:8000] if parts else ""

    def _tool_retriever_schema(self) -> list:
        """Return the single allowed tool schema for ReAct retrieval."""
        return [{
            "type": "function",
            "function": {
                "name": "tool_retreiver",
                "description": (
                    "Retrieve semantically relevant tools for a subquery. "
                    "Use this when you need candidate tools before final selection."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subquery": {
                            "type": "string",
                            "description": "Focused retrieval query for tools."
                        },
                        "k": {
                            "type": "integer",
                            "description": "Optional number of tools to retrieve."
                        }
                    },
                    "required": ["subquery"]
                }
            }
        }]

    def _parse_tool_args(self, raw_args: Any) -> dict:
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _retrieve_candidates(self, all_tools: list, subquery: str, requested_k: Any) -> tuple[list, int]:
        """Run embedding retrieval with dynamic k derived from the tool call."""
        k = None
        if isinstance(requested_k, (int, float)):
            k = int(requested_k)
        if k is None or k <= 0:
            # Dynamic default (no fixed retrieval count)
            k = min(len(all_tools), 100)

        self._embedding_retriever.top_k = k
        candidates = self._embedding_retriever.select(
            [{"role": "user", "content": subquery}],
            all_tools,
        )
        return candidates, k

    def _serialize_candidates(self, candidates: list) -> list:
        """Compact tool records sent back to the selector LLM."""
        serialized = []
        for tool in candidates:
            func = tool.get("function", {})
            serialized.append({
                "name": func.get("name", "unknown"),
                "description": func.get("description", ""),
            })
        return serialized

    def _parse_final_tool_names(self, response_text: str) -> list:
        """Parse final model response as a JSON array of tool names."""
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("toolreagt selector did not return a final tool list")

        import re as _re
        clean = _re.sub(r"<think>.*?</think>", "", response_text, flags=_re.DOTALL).strip()

        parsed = None
        try:
            parsed = json.loads(clean)
        except Exception:
            match = _re.search(r"\[.*\]", clean, _re.DOTALL)
            if match:
                parsed = json.loads(match.group())

        if not isinstance(parsed, list):
            raise ValueError(f"toolreagt selector final response is not a JSON list: {clean[:200]}")

        names = []
        seen = set()
        for item in parsed:
            if not isinstance(item, str):
                continue
            name = item.strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    def select(self, messages: list, tools: list) -> list:
        """Run iterative ReAct-style retrieval and return mapped real tools."""
        all_tools = self.tools_cache if self.tools_cache else tools
        if not all_tools:
            return []

        conversation = self._extract_full_conversation(messages)
        if not conversation:
            raise ValueError("No conversation context found for toolreagt selection")

        react_messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Full benchmark task conversation (multi-turn):\n"
                    f"{conversation}\n\n"
                    "Follow a ReAct loop over tool_retreiver calls when needed. "
                    "Use observations from previous retrieval rounds to refine subqueries. "
                    "Final answer must be ONLY a JSON array of tool names aggregated from your full reasoning process."
                ),
            },
        ]

        iterations = []
        final_response_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for iteration in range(1, self.max_iter + 1):
            response_text, tool_calls, in_tok, out_tok = _invoke_llm(
                react_messages,
                self._tool_retriever_schema(),
            )
            total_input_tokens += int(in_tok or 0)
            total_output_tokens += int(out_tok or 0)
            final_response_text = response_text or final_response_text

            assistant_msg = {"role": "assistant", "content": response_text or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            react_messages.append(assistant_msg)

            iter_log = {
                "iteration": iteration,
                "tool_calls": [],
                "assistant_content_preview": (response_text or "")[:200],
            }

            if not tool_calls:
                iterations.append(iter_log)
                break

            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                name = func.get("name", "") if isinstance(func, dict) else ""
                if name not in ("tool_retreiver", "tool_retriever"):
                    raise ValueError(
                        "toolreagt selector called unsupported tool; only 'tool_retreiver' is allowed"
                    )

                args = self._parse_tool_args(func.get("arguments", {}))
                subquery = args.get("subquery") or args.get("query") or ""
                if not isinstance(subquery, str) or not subquery.strip():
                    raise ValueError("tool_retreiver call is missing non-empty 'subquery'")

                candidates, effective_k = self._retrieve_candidates(all_tools, subquery, args.get("k"))
                serialized = self._serialize_candidates(candidates)

                tool_content = json.dumps(
                    {
                        "subquery": subquery,
                        "requested_k": args.get("k"),
                        "effective_k": effective_k,
                        "retrieved_k": len(serialized),
                        "tools": serialized,
                    },
                    ensure_ascii=False,
                )

                react_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", f"tool_call_{iteration}"),
                        "name": "tool_retreiver",
                        "content": tool_content,
                    }
                )

                iter_log["tool_calls"].append(
                    {
                        "name": "tool_retreiver",
                        "subquery": subquery,
                        "requested_k": args.get("k"),
                        "effective_k": effective_k,
                        "retrieved_k": len(serialized),
                        "retrieved_tool_names": [t["name"] for t in serialized],
                    }
                )

            iterations.append(iter_log)
        else:
            raise RuntimeError(
                f"toolreagt selector reached max_iter={self.max_iter} without finalizing a tool list"
            )

        selected_names = self._parse_final_tool_names(final_response_text)

        # Mapping exactly like hierarchical strategy: filter real tool objects by name.
        selected = [
            tool for tool in all_tools
            if tool.get("function", {}).get("name") in selected_names
        ]

        self.last_selection_metrics = {
            "toolreagt_iterations": len(iterations),
            "toolreagt_max_iter": self.max_iter,
            "toolreagt_input_tokens": total_input_tokens,
            "toolreagt_output_tokens": total_output_tokens,
            "toolreagt_total_tokens": total_input_tokens + total_output_tokens,
        }

        log_tool_selection(
            strategy_name="toolreagt",
            query=conversation,
            available_tools_count=len(all_tools),
            selected_tools=selected,
            selection_metadata={
                "selected_tool_names_from_llm": selected_names,
                "mapped_selected_count": len(selected),
                "iterations": iterations,
                **self.last_selection_metrics,
            },
        )

        print(f"\n[TOOLREAGT SELECTOR]")
        print(f"  Conversation length: {len(conversation)}")
        print(f"  Available tools: {len(all_tools)}")
        print(f"  Iterations: {len(iterations)}/{self.max_iter}")
        print(f"  LLM selected names: {len(selected_names)}")
        print(f"  Mapped tools: {len(selected)}")
        for tool in selected[:10]:
            print(f"    - {tool.get('function', {}).get('name', 'unknown')}")
        if len(selected) > 10:
            print(f"    ... and {len(selected) - 10} more")
        print()

        return selected


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
            
            # Deduplicate by name + description (first wins on param differences).
            def dedup_key(tool: dict):
                func = tool.get("function", {})
                name = func.get("name", "")
                desc = " ".join((func.get("description") or "").split()).strip().lower()
                return (name, desc)

            seen_keys = set()
            unique_tools = []
            duplicates = 0
            for tool in tools:
                key = dedup_key(tool)
                if key not in seen_keys:
                    unique_tools.append(tool)
                    seen_keys.add(key)
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
        
        # Deduplicate by name + description (first wins on param differences).
        def dedup_key(tool: dict):
            func = tool.get("function", {})
            name = func.get("name", "")
            desc = " ".join((func.get("description") or "").split()).strip().lower()
            return (name, desc)

        seen_keys = set()
        unique_tools = []
        duplicates = 0
        for tool in tools:
            key = dedup_key(tool)
            if key not in seen_keys:
                unique_tools.append(tool)
                seen_keys.add(key)
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
            query_usage = _embedding_usage(query_response)
            query_embedding = query_response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")
        
        # Get tool embeddings and compute similarity for ALL VALID TOOLS
        tool_embeddings = []
        tool_descriptions = []
        valid_tools = []
        cached_count = 0
        missing_count = 0
        tool_input_tokens = 0
        
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
                    tool_input_tokens += _embedding_usage(response)["prompt_tokens"]
                    embedding = response.data[0].embedding
                    self.embedding_cache[tool_desc] = embedding
                    missing_count += 1
                except Exception as e:
                    print(f"[WARNING] Failed to embed tool {name}: {e}")
                    embedding = [0] * len(query_embedding)  # Fallback
            
            tool_embeddings.append(embedding)
            tool_descriptions.append(tool_desc)
            valid_tools.append(tool)
        
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
            tool = valid_tools[idx]
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

        self.last_selection_metrics = {
            "embedding_query_input_tokens": query_usage["prompt_tokens"],
            "embedding_tool_input_tokens": tool_input_tokens,
            "embedding_input_tokens": query_usage["prompt_tokens"] + tool_input_tokens,
            "embedding_output_tokens": 0,
            "embedding_total_tokens": query_usage["prompt_tokens"] + tool_input_tokens,
        }
        
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
        self.last_selection_metrics = {}
        self.tools_cache = None
        self.tools_en_file = os.getenv("LANGGRAPH_TOOLS_EN_FILE") or os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../../multi-agent-framework/tools/tools_en.jsonl",
            )
        )
        self.task_tools_by_index = []
        self.task_tools_by_benchmark_id = {}
        self._load_cache()
        self._load_valid_tools_from_schema_cache()
        self._load_task_tools_reference()

    def _extract_query(self, messages: list) -> str:
        """Extract the latest user query and prepend the Qwen3 instruction prefix."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return f"Instruct: {self.TASK_INSTRUCTION}\nQuery: {content[:500]}"
        return ""

    @staticmethod
    def _tool_signature(tool: dict) -> tuple:
        """Full identity key (name+description+parameters) for a tool.

        The tool set can contain synthetic distractor tools that share a name
        with another tool but differ in description/parameters, so name-only
        deduplication would silently drop one of them and risk returning the
        wrong full JSON schema to the executing LLM.
        """
        func = tool.get("function", {}) or {}
        name = str(func.get("name", "")).strip()
        desc = " ".join(str(func.get("description", "")).split()).strip().lower()
        params = json.dumps(func.get("parameters", {}), sort_keys=True, ensure_ascii=False)
        return (name, desc, params)

    @staticmethod
    def _tool_embedding_text(tool: dict) -> str:
        """Embedding/cache-key text covering name, description and parameters.

        Including parameters distinguishes tools that share a name and
        description but expose a different schema (e.g. synthetic variants).
        """
        func = tool.get("function", {}) or {}
        name = func.get("name", "")
        desc = func.get("description", "")
        params = json.dumps(func.get("parameters", {}), sort_keys=True, ensure_ascii=False)
        return f"{name}: {desc}\nParameters: {params}"

    @staticmethod
    def _normalize_required(required: Any) -> list:
        if isinstance(required, list):
            return [str(item) for item in required]
        return []

    @staticmethod
    def _to_int_or_none(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _extract_tools_from_jsonl_entry(entry: Any) -> list:
        if isinstance(entry, list):
            return entry
        if isinstance(entry, dict):
            for key in ("tools", "tool_schemas", "available_tools"):
                value = entry.get(key)
                if isinstance(value, list):
                    return value
        return []

    @staticmethod
    def _required_override_signature(tool: dict) -> tuple:
        """Name+parameter-names key used to locate the task's true required list."""
        func = tool.get("function", {}) if isinstance(tool, dict) else {}
        params = func.get("parameters", {}) if isinstance(func, dict) else {}
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        param_names = tuple(sorted(props.keys())) if isinstance(props, dict) else tuple()
        return (func.get("name"), param_names)

    def _load_task_tools_reference(self):
        """Load per-task tools from tools_en.jsonl to recover task-specific required fields."""
        if not os.path.exists(self.tools_en_file):
            print(f"[QWEN3 EMBEDDING SELECTOR] tools_en reference file not found: {self.tools_en_file}")
            return

        try:
            with open(self.tools_en_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue

                    entry_tools = self._extract_tools_from_jsonl_entry(entry)
                    self.task_tools_by_index.append(entry_tools)

                    if isinstance(entry, dict):
                        benchmark_task_id = entry.get("benchmark_task_id")
                        if benchmark_task_id is None:
                            benchmark_task_id = entry.get("benchmark_task")
                        benchmark_task_id = self._to_int_or_none(benchmark_task_id)
                        if benchmark_task_id is not None:
                            self.task_tools_by_benchmark_id[benchmark_task_id] = entry_tools

            print(
                "[QWEN3 EMBEDDING SELECTOR] Loaded task-tool reference rows: "
                f"{len(self.task_tools_by_index)} from {os.path.basename(self.tools_en_file)}"
            )
        except Exception as e:
            print(f"[WARNING] Failed to load tools_en reference file: {e}")

    def _get_task_tools_for_request(self, request_tools: list) -> tuple[list, Optional[int], Optional[int], int]:
        """Resolve task tools by benchmark_task_id/task_idx with robust index fallback."""
        benchmark_task_id = self._to_int_or_none(getattr(_request_context, "benchmark_task_id", None))
        task_idx = self._to_int_or_none(getattr(_request_context, "task_idx", None))

        if benchmark_task_id is not None and benchmark_task_id in self.task_tools_by_benchmark_id:
            return self.task_tools_by_benchmark_id[benchmark_task_id], benchmark_task_id, None, 0

        if task_idx is None or not self.task_tools_by_index:
            return [], benchmark_task_id, task_idx, 0

        candidate_indices = []
        if 0 <= task_idx < len(self.task_tools_by_index):
            candidate_indices.append(task_idx)
        if task_idx > 0 and 0 <= task_idx - 1 < len(self.task_tools_by_index):
            candidate_indices.append(task_idx - 1)

        if not candidate_indices:
            return [], benchmark_task_id, task_idx, 0

        # If task_idx indexing convention is unclear, choose the candidate line
        # that best matches the incoming request tools.
        request_signatures = {self._required_override_signature(t) for t in (request_tools or [])}
        best_idx = candidate_indices[0]
        best_score = -1
        for idx in candidate_indices:
            candidate_signatures = {self._required_override_signature(t) for t in self.task_tools_by_index[idx]}
            score = len(candidate_signatures.intersection(request_signatures)) if request_signatures else 0
            if score > best_score:
                best_score = score
                best_idx = idx

        return self.task_tools_by_index[best_idx], benchmark_task_id, task_idx, best_score

    def _apply_required_overrides_for_task(self, tools: list, task_tools: list) -> tuple[list, int]:
        """Overwrite required fields for tools matching the task's true name + parameter names.

        The embedded/deduped tool set can collapse tools that share a name and
        parameter set but need a different 'required' list for this specific
        task, so the task's own tools_en.jsonl entry is the source of truth.
        """
        if not tools or not task_tools:
            return tools, 0

        task_required_by_signature = {}
        for task_tool in task_tools:
            signature = self._required_override_signature(task_tool)
            if not signature[0]:
                continue
            params = task_tool.get("function", {}).get("parameters", {})
            task_required_by_signature[signature] = self._normalize_required(params.get("required"))

        if not task_required_by_signature:
            return tools, 0

        adjusted_tools = []
        overrides_count = 0

        for tool in tools:
            signature = self._required_override_signature(tool)
            if signature in task_required_by_signature:
                required = task_required_by_signature[signature]
                tool_copy = copy.deepcopy(tool)
                tool_copy.setdefault("function", {})
                tool_copy["function"].setdefault("parameters", {})
                tool_copy["function"]["parameters"]["required"] = required
                adjusted_tools.append(tool_copy)
                overrides_count += 1
            else:
                adjusted_tools.append(tool)

        return adjusted_tools, overrides_count

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
            query_usage = _embedding_usage(query_response)
            query_embedding = query_response.data[0].embedding
            if not _valid_embedding(query_embedding):
                raise ValueError("embedding endpoint returned a non-finite query vector")
        except Exception as e:
            raise RuntimeError(f"Failed to embed query with Qwen3: {e}")

        tool_embeddings = []
        tool_descriptions = []
        valid_tools = []
        cached_count = 0
        missing_count = 0
        tool_input_tokens = 0

        for tool in all_tools:
            func = tool.get("function", {})
            name = func.get("name", "")
            tool_desc = self._tool_embedding_text(tool)

            if tool_desc in self.embedding_cache and _valid_embedding(self.embedding_cache[tool_desc]):
                embedding = self.embedding_cache[tool_desc]
                cached_count += 1
            else:
                # Invalid cached vectors (for example NaN values from a failed
                # endpoint request) must be discarded and recomputed.
                try:
                    response = client.embeddings.create(
                        input=tool_desc, model=self.model
                    )
                    tool_input_tokens += _embedding_usage(response)["prompt_tokens"]
                    embedding = response.data[0].embedding
                    if not _valid_embedding(embedding):
                        raise ValueError("embedding endpoint returned a non-finite vector")
                    self.embedding_cache[tool_desc] = embedding
                    missing_count += 1
                except Exception as e:
                    print(f"[WARNING] Failed to embed tool {name}: {e}")
                    continue

            tool_embeddings.append(embedding)
            tool_descriptions.append(tool_desc)
            valid_tools.append(tool)

        if not tool_embeddings:
            raise ValueError("No tools with descriptions found for embedding")

        try:
            similarities = cosine_similarity([query_embedding], tool_embeddings)[0]
        except ValueError as exc:
            raise RuntimeError(
                "Embedding similarity failed; query or tool vectors contain invalid values"
            ) from exc
        sorted_indices = np.argsort(similarities)[::-1]

        selected = []
        selected_scores = []
        seen_signatures = set()

        for idx in sorted_indices:
            tool = valid_tools[idx]
            signature = self._tool_signature(tool)
            if signature not in seen_signatures:
                selected.append(tool)
                selected_scores.append(similarities[idx])
                seen_signatures.add(signature)
                if len(selected) >= self.top_k:
                    break

        task_tools, benchmark_task_id, task_idx, task_match_score = self._get_task_tools_for_request(tools)
        selected, overrides_count = self._apply_required_overrides_for_task(selected, task_tools)

        print(f"\n[QWEN3 EMBEDDING SELECTOR]")
        print(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print(f"  Model: {self.model}")
        print(f"  Total available tools: {len(all_tools)}")
        print(f"  Top-k (unique): {self.top_k}")
        print(f"  Cache hits: {cached_count}, Runtime computed: {missing_count}")
        print(f"  Required overrides applied: {overrides_count}")
        print(f"  Selected tools (ranked by relevance):")
        for i, (tool, score) in enumerate(zip(selected[:5], selected_scores[:5])):
            tool_name = tool.get("function", {}).get("name", "unknown")
            print(f"    {i+1}. {tool_name} (similarity: {score:.4f})")
        if len(selected) > 5:
            print(f"    ... and {len(selected) - 5} more")
        print()

        embedding_metrics = {
            "embedding_query_input_tokens": query_usage["prompt_tokens"],
            "embedding_tool_input_tokens": tool_input_tokens,
            "embedding_input_tokens": query_usage["prompt_tokens"] + tool_input_tokens,
            "embedding_output_tokens": 0,
            "embedding_total_tokens": query_usage["prompt_tokens"] + tool_input_tokens,
        }
        self.last_selection_metrics = embedding_metrics

        # Keep embedding selections in the same per-strategy JSONL log as the
        # other selectors. This reflects the tools actually available to ranking.
        log_tool_selection(
            strategy_name="qwen3_embedding",
            query=query,
            available_tools_count=len(all_tools),
            selected_tools=selected,
            selection_metadata={
                "top_k": self.top_k,
                "cache_hits": cached_count,
                "runtime_computed": missing_count,
                "model": self.model,
                "similarity_scores": {
                    f"{tool.get('function', {}).get('name', 'unknown')}#{i}": float(score)
                    for i, (tool, score) in enumerate(zip(selected, selected_scores))
                },
                "required_overrides": overrides_count,
                "benchmark_task_id": benchmark_task_id,
                "task_idx": task_idx,
                "task_row_match_score": task_match_score,
                **embedding_metrics,
            },
        )

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

        seen_keys = set()
        unique_tools = []
        duplicates = 0
        for tool in tools:
            func = tool.get("function", {})
            tool_name = func.get("name", "unknown")
            tool_desc_key = " ".join(str(func.get("description", "")).split()).strip().lower()
            key = (tool_name, tool_desc_key)
            if key not in seen_keys:
                unique_tools.append(tool)
                seen_keys.add(key)
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
            name = tool.get("function", {}).get("name", "unknown")
            tool_desc = self._tool_embedding_text(tool)

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
        self._last_reranker_results = []
        self._last_reranker_metrics = {}
        self.last_selection_metrics = {}

    def select(self, messages: list, tools: list) -> list:
        """Select top-k tools using Qwen3-Reranker-8B pairwise scoring."""
        if not tools:
            return []
        query = self._extract_query(messages)
        if not query:
            raise ValueError("No user query found for Qwen3-Reranker tool selection")
        reranked = self._rerank_with_qwen3(query, tools)
        final_selected = reranked[:self.top_k]
        self.last_selection_metrics = {
            **self._last_reranker_metrics,
            "reranker_total_tokens": (
                self._last_reranker_metrics.get("reranker_input_tokens", 0)
                + self._last_reranker_metrics.get("reranker_output_tokens", 0)
            ),
        }
        
        # Log tool selection
        log_tool_selection(
            strategy_name="qwen3_reranker",
            query=query,
            available_tools_count=len(tools),
            selected_tools=final_selected,
            selection_metadata={
                "model": self.model,
                "top_k": self.top_k,
                "reranked_count": len(reranked),
                "reranker_results": self._last_reranker_results,
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
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        self._last_reranker_metrics = {
            "reranker_input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "reranker_output_tokens": int(usage.get("completion_tokens", 0) or 0),
        }

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

        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        self._last_reranker_metrics = {
            "reranker_input_tokens": self._last_reranker_metrics.get("reranker_input_tokens", 0) + int(usage.get("prompt_tokens", 0) or 0),
            "reranker_output_tokens": self._last_reranker_metrics.get("reranker_output_tokens", 0) + int(usage.get("completion_tokens", 0) or 0),
        }
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
        self._last_reranker_metrics = {"reranker_input_tokens": 0, "reranker_output_tokens": 0}
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
        self._last_reranker_results = [
            {
                "rank": rank,
                "tool_name": tool.get("function", {}).get("name", "unknown"),
                "score": float(score),
            }
            for rank, (score, tool) in enumerate(ranked_pairs, start=1)
        ]
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
        final_selected = reranked[:self.top_k]
        self.last_selection_metrics = {
            **self.embedding_selector.last_selection_metrics,
            **self._last_reranker_metrics,
            "reranker_total_tokens": (
                self._last_reranker_metrics.get("reranker_input_tokens", 0)
                + self._last_reranker_metrics.get("reranker_output_tokens", 0)
            ),
        }
        log_tool_selection(
            strategy_name="qwen3_embedding_qwen3_reranker",
            query=query,
            available_tools_count=len(tools),
            selected_tools=final_selected,
            selection_metadata={
                "model": self.model,
                "initial_k": self.initial_k,
                "top_k": self.top_k,
                "candidate_count": len(candidates),
                "reranker_results": self._last_reranker_results,
                **self.last_selection_metrics,
            },
        )
        return final_selected


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


def _detect_tool_call_parser(model_name: str) -> str:
    """Infer the active tool-call parser mode for executor-specific safeguards.

    Priority:
    1) Explicit env override via EXECUTING_LLM_TOOL_CALL_PARSER
    2) Heuristic from model name
    """
    parser = (os.getenv("EXECUTING_LLM_TOOL_CALL_PARSER", "auto") or "auto").strip().lower()
    if parser and parser != "auto":
        return parser

    model_l = (model_name or "").lower()
    if "poolside" in model_l or "laguna" in model_l:
        return "poolside_v1"
    return "openai"


def _build_execution_system_prompt(reasoning: str, model_name: str) -> str:
    """Build a concise, strict system prompt that pushes the model toward valid tool calls."""
    model_l = (model_name or "").lower()
    base = (
        "You are a tool-using assistant. For every tool use, emit exactly one standard OpenAI "
        "tool-call object with keys 'id', 'type', and 'function'. The 'function' object must "
        "contain 'name' and a JSON-string 'arguments'. Do not emit raw headers, do not emit "
        "chain-of-thought in tool-call fields, and do not mix plain text with tool-call syntax. "
        "If a tool call is needed, return only the tool call in the required format."
    )
    if "poolside" in model_l or "laguna" in model_l:
        return (
            f"{base} "
            "For Poolside/Laguna models, follow the native tool-call syntax expected by the backend: "
            "use standard OpenAI tool calls and avoid any alternative header-style or raw parser-specific syntax."
        )
    if "gpt-oss-120b" in model_l or "gpt-oss" in model_l:
        return (
            f"{base} "
            "Tool use must be returned only inside tool_calls[] as standard OpenAI "
            "tool-call objects. Do not place tool JSON inside content, do not wrap it in prose, and do not "
            "emit pseudo-JSON or markdown code fences. If a tool is needed, the content field must stay empty."
        )
    return (
        f"{base} "
        "For standard OpenAI-compatible backends, prefer the standard OpenAI tool-call format and "
        "do not use any alternative raw headers or parser-specific wrappers."
    )


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

                elif "recipient_name" in tc:
                    # Copilot-style format: {"recipient_name": "functions.X", "parameters": {...}}
                    recipient = str(tc.get("recipient_name", "")).strip()
                    name = recipient
                    if recipient.startswith("functions."):
                        name = recipient[len("functions."):]
                    if "." in name:
                        name = name.split(".")[-1]
                    args = tc.get("parameters", tc.get("arguments", {}))
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    call_id = tc.get("id") or f"call_{name}_{idx}"
                    new_tool_calls.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
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
        elif "recipient_name" in tc:
            recipient = str(tc.get("recipient_name", "")).strip()
            name = recipient
            if recipient.startswith("functions."):
                name = recipient[len("functions."):]
            if "." in name:
                name = name.split(".")[-1]
            args = tc.get("parameters", tc.get("arguments", {}))
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            elif not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            call_id = tc.get("id") or f"call_{name}_{idx}"
            result.append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            })
        else:
            result.append(tc)
    return result
_RAW_HARMONY_MARKERS = (
    "<|start|>",
    "<|end|>",
    "<|channel|>",
    "<|endoftext|>",
)


def _sanitize_history_for_gpt_oss(messages: list) -> list:
    """
    Prepare already-stored conversation history for re-sending to GPT-OSS.

    This does NOT recover malformed tool calls and does NOT turn text into
    tool calls. It only prevents raw Harmony/model control syntax from being
    sent back to vLLM, where it can trigger HarmonyError.

    Valid OpenAI-style tool_calls are preserved unchanged.
    """
    cleaned_messages = []

    for idx, original in enumerate(messages):
        if not isinstance(original, dict):
            cleaned_messages.append(original)
            continue

        msg = copy.deepcopy(original)

        # --------------------------------------------------
        # 1. Clean ordinary content
        # --------------------------------------------------
        content = msg.get("content")

        if isinstance(content, str):
            cleaned = content

            # Raw special tokens must never be round-tripped through
            # OpenAI-style message history.
            for marker in _RAW_HARMONY_MARKERS:
                if marker in cleaned:
                    print(
                        f"[HISTORY] Removing raw Harmony marker "
                        f"{marker!r} from message {idx}"
                    )
                    cleaned = cleaned.split(marker, 1)[0].rstrip()

            msg["content"] = cleaned

        # --------------------------------------------------
        # 2. Validate actual OpenAI tool_calls
        # --------------------------------------------------
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            valid_calls = []

            for tc_idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    raise RuntimeError(
                        f"Invalid tool_call in history message {idx}, "
                        f"index {tc_idx}: expected dict, got {type(tc)}"
                    )

                function = tc.get("function")

                if not isinstance(function, dict):
                    raise RuntimeError(
                        f"Malformed tool_call in history message {idx}, "
                        f"index {tc_idx}: missing standard OpenAI "
                        f"'function' object: {tc!r}"
                    )

                name = function.get("name")
                arguments = function.get("arguments")

                if not isinstance(name, str) or not name.strip():
                    raise RuntimeError(
                        f"Malformed tool_call name in history message "
                        f"{idx}, index {tc_idx}: {name!r}"
                    )

                # This is exactly the kind of corruption causing your error.
                if (
                    "to=functions." in name
                    or any(marker in name for marker in _RAW_HARMONY_MARKERS)
                    or "\u00a0" in name
                ):
                    raise RuntimeError(
                        f"Raw GPT-OSS/Harmony syntax leaked into tool name "
                        f"in history message {idx}: {name!r}"
                    )

                # OpenAI expects arguments to be a JSON string.
                # Converting dict -> JSON string is serialization,
                # not recovery of a failed model call.
                tc_copy = copy.deepcopy(tc)
                if isinstance(arguments, dict):
                    tc_copy["function"]["arguments"] = json.dumps(
                        arguments,
                        ensure_ascii=False,
                    )

                valid_calls.append(tc_copy)

            msg["tool_calls"] = valid_calls

        # --------------------------------------------------
        # 3. Catch raw Harmony syntax elsewhere in the msg
        # --------------------------------------------------
        raw = json.dumps(msg, ensure_ascii=False)

        if "to=functions." in raw:
            raise RuntimeError(
                "Raw GPT-OSS/Harmony 'to=functions.' syntax leaked "
                f"into message history at message {idx}: "
                f"{raw[:1500]}"
            )

        cleaned_messages.append(msg)

    return cleaned_messages

def _invoke_llm(messages: list, tools: list = None):
    """Invoke a local OpenAI/vLLM-compatible endpoint.
    
    Returns: (content, tool_calls) tuple
    """
    base_url = os.getenv("EXECUTING_LLM_BASE_URL")
    api_key = os.getenv("EXECUTING_LLM_API_KEY", "EMPTY")
    model = os.getenv("EXECUTING_LLM_MODEL", "openai/gpt-oss-120b")
    parser_mode = _detect_tool_call_parser(model)
    
    if not base_url:
        # Simulated response for testing without a real LLM
        joined = "\n".join([str(m) for m in messages])
        return f"[simulated response] received {len(messages)} messages: {joined[:200]}", []

    # Build full chat completions endpoint from base URL
    endpoint = base_url.rstrip("/") + "/chat/completions"

    # Prepend a strict execution-system prompt if not already present.
    reasoning = os.getenv("EXECUTING_LLM_REASONING", "medium")
    messages_to_send = list(messages)


    # Normalize all messages: convert any to=functions.X tool calls to standard format
    # before sending to vLLM, which would otherwise reject them with a 500.
    messages_to_send = _normalize_messages_for_llm(messages_to_send)
    # Prevent raw GPT-OSS/Harmony syntax from being round-tripped
    # through conversation history.
    messages_to_send = _sanitize_history_for_gpt_oss(messages_to_send)

    payload = {
        "messages": messages_to_send,
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    
    # Add tools if provided
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

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
    _timeout = int(os.getenv("EXECUTING_LLM_TIMEOUT", "1800"))

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
    elif mode == "toolreagt":
        return ToolReActToolSelector(max_iter=10)
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
        selection_metrics: dict

    builder = StateGraph(GraphState)
    selector = _create_tool_selector(mode)

    def tool_selection_node(state: GraphState):
        """Node 1: Select relevant tools based on strategy."""
        messages = state.get("messages", [])
        tools = state.get("tools", [])
        selected = selector.select(messages, tools)
        metrics = getattr(selector, "last_selection_metrics", None)
        if not metrics and hasattr(selector, "embedding_selector"):
            metrics = getattr(selector.embedding_selector, "last_selection_metrics", {})
        return {"selected_tools": selected, "selection_metrics": metrics or {}}

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
            "selection_metrics": state.get("selection_metrics", {}),
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
        _request_context.benchmark_task_id = payload.get("benchmark_task_id") or inp.get("benchmark_task_id")
        _request_context.selection_mode = selection_mode
        _request_context.request_id = uuid.uuid4().hex
        
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
        selection_metrics = {}
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
                    selection_metrics = result.get("selection_metrics", {}) or {}
                else:
                    content = str(result)
                    tool_calls = []
                    input_tokens = 0
                    output_tokens = 0
                    selection_metrics = {}
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
            "selection_metrics": selection_metrics,
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
    print(f"  2.1. 'toolreagt' - ReAct selector using tool_retreiver iterations")
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
