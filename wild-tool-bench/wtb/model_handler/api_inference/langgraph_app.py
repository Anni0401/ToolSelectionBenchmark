import os
import json
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


def _invoke_llm(messages):
    """Invoke a local OpenAI/vLLM-compatible endpoint if configured, otherwise return a simulated response."""
    endpoint = os.getenv("LANGGRAPH_LLM_ENDPOINT")
    api_key = os.getenv("LANGGRAPH_LLM_API_KEY")
    payload = {"messages": messages}

    if not endpoint:
        # Simulated response for testing without a real LLM
        joined = "\n".join([str(m) for m in messages])
        return f"[simulated response] received {len(messages)} messages: {joined[:200]}"

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode("utf-8")
            try:
                parsed = json.loads(data)
            except Exception:
                return data

            # Accept common shapes: {"content": "..."} or {"choices": [{"message": {"content": "..."}}]}
            if isinstance(parsed, dict) and "content" in parsed:
                return parsed["content"]
            if isinstance(parsed, dict) and "choices" in parsed and parsed["choices"]:
                first = parsed["choices"][0]
                if isinstance(first, dict) and "message" in first and "content" in first["message"]:
                    return first["message"]["content"]
            return data
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM request failed: {exc.code} {exc.reason}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}")


def _build_graph():
    """Build a minimal StateGraph if langgraph is available.

    The graph simply runs a single LLM execution node and returns its response.
    """
    if not LANGGRAPH_AVAILABLE:
        return None

    from typing import TypedDict

    class GraphState(TypedDict):
        messages: list
        tools: list
        response: str

    builder = StateGraph(GraphState)

    def llm_execution_node(state: GraphState):
        messages = state.get("messages", [])
        response = _invoke_llm(messages)
        return {"response": response}

    builder.add_node("llm_execution", llm_execution_node)
    builder.set_entry_point("llm_execution")
    builder.add_edge("llm_execution", END)
    return builder.compile()


GRAPH = _build_graph()


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

        start = time.time()
        try:
            if GRAPH is not None:
                # GRAPH.execute may vary between langgraph versions
                try:
                    result = GRAPH.execute({"messages": messages, "tools": tools, "response": ""})
                except TypeError:
                    # fallback if API differs
                    result = GRAPH({"messages": messages, "tools": tools, "response": ""})
                # result expected to be a dict-like containing 'response'
                if isinstance(result, dict):
                    content = result.get("response") or ""
                else:
                    content = str(result)
            else:
                content = _invoke_llm(messages)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        latency = time.time() - start

        response = {
            "content": content,
            "reasoning_content": None,
            "tool_calls": [],
            "input_token": 0,
            "output_token": 0,
            "latency": latency,
        }

        self._send_json(response)


def run(host="127.0.0.1", port=8001):
    server = HTTPServer((host, port), LangGraphLocalHandler)
    print(f"LangGraph local server listening at http://{host}:{port}/execute")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")
        server.shutdown()


if __name__ == "__main__":
    host = os.getenv("LANGGRAPH_HOST", "127.0.0.1")
    port = int(os.getenv("LANGGRAPH_PORT", "8001"))
    run(host=host, port=port)
