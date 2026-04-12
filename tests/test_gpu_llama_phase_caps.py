from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strong_models_experiment.agents.agent_factory import StrongModelAgentFactory
from strong_models_experiment.configs import STRONG_MODELS_CONFIG
from strong_models_experiment.phases.phase_handlers import PhaseHandler


EXPECTED_PHASE_CAPS = {
    "max_tokens_discussion": 2048,
    "max_tokens_thinking": 1280,
    "max_tokens_proposal": 512,
    "max_tokens_voting": 384,
    "max_tokens_reflection": 1536,
}


@dataclass
class DummyConfig:
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class DummyAgent:
    def __init__(self, phase_token_caps: Dict[str, int] | None = None):
        custom_parameters = {}
        if phase_token_caps is not None:
            custom_parameters["phase_token_caps"] = phase_token_caps
        self.config = DummyConfig(custom_parameters=custom_parameters)
        self.updated_limits: List[int] = []

    def update_max_tokens(self, max_tokens):
        self.updated_limits.append(max_tokens)


def _empty_token_config() -> Dict[str, Any]:
    return {
        "discussion": None,
        "proposal": None,
        "voting": None,
        "reflection": None,
        "thinking": None,
        "default": None,
    }


def test_cluster_llama_aliases_have_caps_but_api_aliases_do_not():
    cluster_aliases = [
        "llama-3.1-8b-instruct-cluster",
        "llama-3.2-3b-instruct-cluster",
        "llama-3.2-1b-instruct-cluster",
    ]
    api_aliases = [
        "llama-3.1-8b-instruct",
        "llama-3.2-3b-instruct",
        "llama-3.2-1b-instruct",
    ]

    for alias in cluster_aliases:
        config = STRONG_MODELS_CONFIG[alias]
        assert config["provider"] == "Princeton Cluster"
        for key, value in EXPECTED_PHASE_CAPS.items():
            assert config[key] == value

    for alias in api_aliases:
        config = STRONG_MODELS_CONFIG[alias]
        for key in EXPECTED_PHASE_CAPS:
            assert key not in config


def test_phase_handler_applies_agent_specific_caps_without_touching_uncapped_agents():
    handler = PhaseHandler(token_config=_empty_token_config())
    capped_agent = DummyAgent({"discussion": 2048})
    uncapped_agent = DummyAgent()

    handler._apply_phase_token_limits([capped_agent, uncapped_agent], "discussion")

    assert capped_agent.updated_limits == [2048]
    assert uncapped_agent.updated_limits == []


def test_phase_handler_uses_min_of_global_limit_and_agent_cap():
    token_config = _empty_token_config()
    token_config["discussion"] = 1024
    handler = PhaseHandler(token_config=token_config)
    capped_agent = DummyAgent({"discussion": 2048})
    uncapped_agent = DummyAgent()

    handler._apply_phase_token_limits([capped_agent, uncapped_agent], "discussion")

    assert capped_agent.updated_limits == [1024]
    assert uncapped_agent.updated_limits == [1024]


def test_phase_handler_voting_falls_back_to_default_but_prefers_phase_specific_cap():
    token_config = _empty_token_config()
    token_config["default"] = 2000
    handler = PhaseHandler(token_config=token_config)
    capped_agent = DummyAgent({"voting": 384})
    uncapped_agent = DummyAgent()

    handler._apply_phase_token_limits([capped_agent, uncapped_agent], "voting", fallback_phase="default")

    assert capped_agent.updated_limits == [384]
    assert uncapped_agent.updated_limits == [2000]


def test_local_model_factory_passes_phase_caps_into_agent_config(monkeypatch):
    captured = {}

    class StubLocalModelAgent:
        def __init__(self, agent_id, config, local_path):
            captured["agent_id"] = agent_id
            captured["config"] = config
            captured["local_path"] = local_path

    monkeypatch.setattr(
        "strong_models_experiment.agents.agent_factory.LocalModelAgent",
        StubLocalModelAgent,
    )

    factory = StrongModelAgentFactory()
    config = STRONG_MODELS_CONFIG["llama-3.2-3b-instruct-cluster"]
    factory._create_local_model_agent(
        "llama-3.2-3b-instruct-cluster",
        config,
        "Agent_1",
        max_tokens=999999,
    )

    assert captured["agent_id"] == "Agent_1"
    assert captured["local_path"] == "/scratch/gpfs/DANQIC/models/Llama-3.2-3B-Instruct"
    assert captured["config"].custom_parameters["phase_token_caps"] == {
        "discussion": 2048,
        "thinking": 1280,
        "proposal": 512,
        "voting": 384,
        "reflection": 1536,
    }
