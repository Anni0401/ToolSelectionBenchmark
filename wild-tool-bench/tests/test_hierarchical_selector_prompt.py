import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wtb.model_handler.api_inference.langgraph_app import HierarchicalToolSelector


def test_selector_prompt_includes_full_conversation_history():
    selector = object.__new__(HierarchicalToolSelector)
    messages = [
        {"role": "system", "content": "Current Date: 2026-08-03"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    tools = [{"function": {"name": "search_docs", "description": "Search documentation"}}]

    prompt = selector._build_selector_prompt(messages, tools)

    assert "Conversation History" in prompt
    assert "First question" in prompt
    assert "First answer" in prompt
    assert "Second question" in prompt
    assert "search_docs" in prompt
