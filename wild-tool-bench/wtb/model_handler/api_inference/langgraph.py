import json
import os
import time
import urllib.request
import urllib.error

from wtb.model_handler.base_handler import BaseHandler


class LangGraphHandler(BaseHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.endpoint = os.getenv("LANGGRAPH_ENDPOINT")
        self.api_key = os.getenv("LANGGRAPH_API_KEY")

        if self.endpoint is None:
            raise ValueError("LANGGRAPH_ENDPOINT environment variable must be set")

    def _request_tool_call(self, inference_data):
        messages = inference_data["messages"]
        tools = inference_data["tools"]

        payload = self._build_langgraph_payload(messages, tools)
        api_response, latency = self._send_langgraph_request(payload)

        return api_response, latency

    def _build_langgraph_payload(self, messages, tools):
        """Build the LangGraph execution payload.

        The exact payload shape depends on your LangGraph deployment.
        The benchmark sends a sequence of messages + available tools.
        This method should translate that into a graph execution request.
        """
        return {
            "input": {
                "messages": messages,
                "tools": tools,
            },
            "nodes": [
                {
                    "id": "tool_selection",
                    "type": "ToolSelectionNode",
                    "input": {
                        "messages": messages,
                        "tools": tools,
                    },
                },
                {
                    "id": "llm_execution",
                    "type": "LLMExecutionNode",
                    "input": {
                        "messages": messages,
                    },
                },
            ],
            "edges": [
                {"from": "tool_selection", "to": "llm_execution", "output": "selected_tool_calls"}
            ],
        }

    def _send_langgraph_request(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        start_time = time.time()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
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

        return {
            "reasoning_content": reasoning_content,
            "content": content,
            "tool_calls": tool_calls,
            "input_token": input_token,
            "output_token": output_token,
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
