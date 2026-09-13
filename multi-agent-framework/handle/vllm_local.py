import json
import os
import time

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from utils.openai_utils import retry_with_backoff

VLLM_LOCAL_DEFAULT_BASE_URL = "http://localhost:8000/v1"
VLLM_LOCAL_DEFAULT_MODEL = "openai/gpt-oss-120b"


class VLLMChatHandler:
    """Handler for a self-hosted, OpenAI-compatible vLLM chat endpoint (e.g. gpt-oss-120b)."""

    def __init__(self, model_name=None, temperature=0.0, base_url=None, api_key=None):
        self.model_name = model_name or os.getenv("VLLM_REWRITE_MODEL", VLLM_LOCAL_DEFAULT_MODEL)
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("VLLM_REWRITE_API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("VLLM_REWRITE_BASE_URL", VLLM_LOCAL_DEFAULT_BASE_URL),
        )

    @retry_with_backoff((RateLimitError, APIConnectionError, APITimeoutError))
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        api_response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()

        return api_response, end_time - start_time

    def request_model(self, messages):
        kwargs = {
            "messages": messages,
            "timeout": 300,
            "model": self.model_name,
            "temperature": self.temperature,
        }
        api_response, latency = self.generate_with_backoff(**kwargs)
        api_response = json.loads(api_response.json())
        choice = api_response["choices"][0]
        message = choice["message"]
        text = message["content"]
        return text


def main():
    handle = VLLMChatHandler(temperature=0.0)
    messages = [
        {
            "role": "user",
            "content": "Hello, who are you?"
        }
    ]
    print(json.dumps(messages, ensure_ascii=False, indent=4))
    print("---")
    result = handle.request_model(messages)
    print(result)


if __name__ == "__main__":
    main()
