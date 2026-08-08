import json
import os
import time
import urllib.request
import urllib.error

from wtb.model_handler.base_handler import BaseHandler


class LangGraphHandler(BaseHandler):
    """Handler for LangGraph-based tool calling.
    
    Supports multiple tool selection strategies, including:
    1. 'in_context' - LLM decides which tools to use (default)
    2. 'hierarchical' - Smaller LLM selects relevant tools first
    3. 'toolreagt' - ReAct selector with iterative embedding retrieval
    3. 'embedding' - Embedding-based tool retrieval
    4. 'embedding_reranker' - Embeddings + LLM reranking
    """
    
    def __init__(self, model_name, temperature, selection_mode=None):
        super().__init__(model_name, temperature)
        self.endpoint = os.getenv("LANGGRAPH_ENDPOINT")
        self.api_key = os.getenv("LANGGRAPH_API_KEY")
        self.selection_mode = selection_mode or os.getenv("LANGGRAPH_TOOL_SELECTION_MODE", "in_context")

        if self.endpoint is None:
            raise ValueError("LANGGRAPH_ENDPOINT environment variable must be set")

    def _request_tool_call(self, inference_data):
        messages = inference_data["messages"]
        tools = inference_data["tools"]
        test_entry_id = inference_data.get("test_entry_id")
        task_idx = inference_data.get("task_idx")

        payload = self._build_langgraph_payload(messages, tools, test_entry_id=test_entry_id, task_idx=task_idx)
        api_response, latency = self._send_langgraph_request(payload)

        return api_response, latency

    def _build_langgraph_payload(self, messages, tools, test_entry_id=None, task_idx=None):
        """Build the LangGraph execution payload.

        Supports configurable tool selection strategy:
        - in_context: all tools passed, LLM decides
        - hierarchical: smaller LLM selects tools first
        - toolreagt: ReAct loop with tool_retreiver over embedding retrieval
        - embedding: embedding-based retrieval
        - embedding_reranker: embeddings + LLM reranking
        """
        payload = {
            "input": {
                "messages": messages,
                "tools": tools,
            },
            "selection_mode": self.selection_mode,
        }
        if test_entry_id is not None:
            payload["test_entry_id"] = test_entry_id
        if task_idx is not None:
            payload["task_idx"] = task_idx
        return payload

    def _send_langgraph_request(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        start_time = time.time()
        # Use a client timeout that is longer than the server-side EXECUTING_LLM_TIMEOUT
        # (default 600s) so the server always returns a proper error before the client
        # gives up and raises a raw "timed out" socket error.
        _client_timeout = int(os.getenv("LANGGRAPH_CLIENT_TIMEOUT", "1200"))
        try:
            with urllib.request.urlopen(request, timeout=_client_timeout) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LangGraph request failed: {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LangGraph request failed: {exc.reason}")

        latency = time.time() - start_time
        return response_text, latency

    def _parse_api_response(self, api_response):
        if isinstance(api_response, str):
            api_response = json.loads(api_response)

        # Normalize the response into the benchmark expected fields.
        reasoning_content = api_response.get("reasoning_content")
        content = api_response.get("content") or api_response.get("answer") or ""
        tool_calls = self._normalize_tool_calls(api_response.get("tool_calls", []))

        # LangGraph may provide explicit token accounting, or we fallback to 0.
        input_token = api_response.get("input_token", 0)
        output_token = api_response.get("output_token", 0)
        selection_metrics = api_response.get("selection_metrics", {}) or {}

        return {
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": tool_calls,
            "input_token": input_token,
            "output_token": output_token,
            "selection_metrics": selection_metrics,
        }

    def _normalize_tool_calls(self, tool_calls):
        if tool_calls is None:
            return []

        normalized = []
        for idx, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict) and "function" in tool_call:
                normalized.append(tool_call)
                continue

            # Accept a simple payload shape: {"name": ..., "arguments": ...}
            if isinstance(tool_call, dict) and "name" in tool_call:
                normalized.append(
                    {
                        "id": tool_call.get("id", f"toolu_bdrk_{idx}"),
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call.get("arguments", {}),
                        },
                    }
                )
                continue

            # Accept Copilot-style payload shape:
            # {"recipient_name": "functions.<name>", "parameters": {...}}
            if isinstance(tool_call, dict) and "recipient_name" in tool_call:
                recipient = str(tool_call.get("recipient_name", "")).strip()
                name = recipient
                if recipient.startswith("functions."):
                    name = recipient[len("functions."):]
                if "." in name:
                    name = name.split(".")[-1]
                normalized.append(
                    {
                        "id": tool_call.get("id", f"toolu_bdrk_{idx}"),
                        "function": {
                            "name": name,
                            "arguments": tool_call.get("parameters", tool_call.get("arguments", {})),
                        },
                    }
                )
                continue

            raise ValueError(f"Unsupported tool call format from LangGraph: {tool_call}")

        return normalized


def main():
    from wtb.constant import DOTENV_PATH
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    handler = LangGraphHandler("langgraph", 0.0)
    print(json.dumps(handler._build_langgraph_payload([], []), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
