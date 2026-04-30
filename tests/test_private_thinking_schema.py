"""Tests for shared private-thinking schema normalization."""

import json

from negotiation.llm_agents import BaseLLMAgent, LLMConfig, ModelType


class DummyAgent(BaseLLMAgent):
    """Minimal agent for testing BaseLLMAgent parsing helpers."""

    async def _call_llm_api(self, messages, **kwargs):
        raise NotImplementedError

    def get_model_info(self):
        return {"model_name": "dummy"}


def test_thinking_response_normalizes_shared_schema():
    """Shared private-thinking keys should be normalized with backward-compatible aliases."""
    agent = DummyAgent("test_agent", LLMConfig(model_type=ModelType.GPT_4O))

    parsed = agent._parse_thinking_response("""{
  "reasoning": "I should push hardest for Apple.",
  "strategy": "Trade lower-value items to secure my top pick.",
  "key_priorities": ["Apple", "Jewel"],
  "potential_concessions": ["Pencil"]
}""")

    assert parsed["key_priorities"] == ["Apple", "Jewel"]
    assert parsed["potential_concessions"] == ["Pencil"]
    assert parsed["target_items"] == ["Apple", "Jewel"]
    assert parsed["anticipated_resistance"] == ["Pencil"]


def test_thinking_parse_fallback_preserves_raw_response_and_metadata():
    """JSON parse fallbacks should retain full raw output for provenance."""
    agent = DummyAgent("test_agent", LLMConfig(model_type=ModelType.GPT_4O))
    raw_response = '{"reasoning": invalid token, "strategy": "Cannot repair bare identifiers"}'

    parsed = agent._parse_thinking_response(raw_response)

    assert parsed["used_fallback"] is True
    assert parsed["raw_response"] == raw_response
    assert parsed["parse_error"]["type"] == "JSONDecodeError"
    assert "Expecting" in parsed["parse_error"]["message"]
    assert parsed["parsed_or_fallback_response"]["strategy"] == "Basic preference-driven approach"
    assert parsed["reasoning"].startswith(raw_response[:40])

    saved_payload = json.dumps(parsed)
    reloaded = json.loads(saved_payload)
    assert reloaded["raw_response"] == raw_response
    assert reloaded["parsed_or_fallback_response"]["reasoning"] == parsed["reasoning"]


def test_thinking_parser_repairs_missing_comma():
    """Private thinking parsing should repair simple syntax issues locally."""
    agent = DummyAgent("test_agent", LLMConfig(model_type=ModelType.GPT_4O))
    raw_response = """{
  "reasoning": "I should fund Apple"
  "strategy": "Missing comma before this key"
}"""

    parsed = agent._parse_thinking_response(raw_response)

    assert parsed["reasoning"] == "I should fund Apple"
    assert parsed["strategy"] == "Missing comma before this key"
    assert "used_fallback" not in parsed
