#!/usr/bin/env python3
"""
Abstraction tests for GameEnvironment interface.

Tests verify:
- Factory function creates correct game types
- Both games implement the full interface
- Common behavior across games (discounting, voting prompts)
- Interface contract is maintained
"""

import pytest
from typing import List
from abc import ABC

from game_environments import (
    create_game_environment,
    create_game_from_config,
    GameEnvironment,
    GameConfig,
    GameType,
    ItemAllocationConfig,
    DiplomaticTreatyConfig,
    ItemAllocationGame,
    DiplomaticTreatyGame,
)


class FakeAgent:
    """Simple agent for testing."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def create_test_agents(n: int = 2) -> List[FakeAgent]:
    """Create a list of fake agents for testing."""
    return [FakeAgent(f"Agent_{i+1}") for i in range(n)]


class TestFactoryFunction:
    """Tests for the create_game_environment factory function."""

    def test_creates_item_allocation(self):
        """Test factory creates ItemAllocationGame for 'item_allocation'."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5
        )
        assert isinstance(game, ItemAllocationGame)
        assert game.get_game_type() == GameType.ITEM_ALLOCATION

    def test_creates_diplomacy(self):
        """Test factory creates DiplomaticTreatyGame for 'diplomacy'."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5
        )
        assert isinstance(game, DiplomaticTreatyGame)
        assert game.get_game_type() == GameType.DIPLOMATIC_TREATY

    def test_invalid_game_type_raises(self):
        """Test that invalid game type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown game type"):
            create_game_environment("invalid_game", n_agents=2, t_rounds=5)

    def test_passes_common_params(self):
        """Test that common parameters are passed correctly."""
        game = create_game_environment(
            "item_allocation",
            n_agents=3,
            t_rounds=15,
            gamma_discount=0.85,
            random_seed=123
        )
        assert game.config.n_agents == 3
        assert game.config.t_rounds == 15
        assert game.config.gamma_discount == 0.85
        assert game.config.random_seed == 123

    def test_passes_item_allocation_params(self):
        """Test that item allocation specific params are passed."""
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=5,
            m_items=7,
            competition_level=0.8
        )
        assert game.config.m_items == 7
        assert game.config.competition_level == 0.8

    def test_passes_diplomacy_params(self):
        """Test that diplomacy specific params are passed."""
        game = create_game_environment(
            "diplomacy",
            n_agents=2,
            t_rounds=5,
            n_issues=8,
            rho=0.5,
            theta=0.3,
        )
        assert game.config.n_issues == 8
        assert game.config.rho == 0.5
        assert game.config.theta == 0.3


class TestCreateFromConfig:
    """Tests for create_game_from_config function."""

    def test_creates_from_item_allocation_config(self):
        """Test creating game from ItemAllocationConfig."""
        config = ItemAllocationConfig(n_agents=2, t_rounds=10, m_items=5)
        game = create_game_from_config(config)
        assert isinstance(game, ItemAllocationGame)

    def test_creates_from_diplomacy_config(self):
        """Test creating game from DiplomaticTreatyConfig."""
        config = DiplomaticTreatyConfig(n_agents=2, t_rounds=10, n_issues=5)
        game = create_game_from_config(config)
        assert isinstance(game, DiplomaticTreatyGame)

    def test_invalid_config_raises(self):
        """Test that invalid config type raises ValueError."""
        # Create a base GameConfig (not a specific subclass)
        config = GameConfig(n_agents=2, t_rounds=5)
        with pytest.raises(ValueError, match="Unknown config type"):
            create_game_from_config(config)


class TestInterfaceImplementation:
    """Tests that both games implement the full GameEnvironment interface."""

    @pytest.fixture
    def item_allocation_game(self):
        """Create an item allocation game for testing."""
        return create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )

    @pytest.fixture
    def diplomacy_game(self):
        """Create a diplomacy game for testing."""
        return create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )

    @pytest.fixture
    def agents(self):
        """Create test agents."""
        return create_test_agents(2)

    def test_both_are_game_environments(self, item_allocation_game, diplomacy_game):
        """Test that both games are GameEnvironment instances."""
        assert isinstance(item_allocation_game, GameEnvironment)
        assert isinstance(diplomacy_game, GameEnvironment)

    def test_create_game_state(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement create_game_state."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        assert isinstance(state1, dict)
        assert isinstance(state2, dict)

    def test_get_game_rules_prompt(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_game_rules_prompt."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        prompt1 = item_allocation_game.get_game_rules_prompt(state1)
        prompt2 = diplomacy_game.get_game_rules_prompt(state2)

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)
        assert len(prompt1) > 0
        assert len(prompt2) > 0

    def test_get_preference_assignment_prompt(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_preference_assignment_prompt."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        prompt1 = item_allocation_game.get_preference_assignment_prompt("Agent_1", state1)
        prompt2 = diplomacy_game.get_preference_assignment_prompt("Agent_1", state2)

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)

    def test_get_proposal_prompt(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_proposal_prompt."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        prompt1 = item_allocation_game.get_proposal_prompt(
            "Agent_1", state1, 1, ["Agent_1", "Agent_2"]
        )
        prompt2 = diplomacy_game.get_proposal_prompt(
            "Agent_1", state2, 1, ["Agent_1", "Agent_2"]
        )

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)

    def test_get_discussion_prompt(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_discussion_prompt."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        prompt1 = item_allocation_game.get_discussion_prompt(
            "Agent_1", state1, 1, 5, []
        )
        prompt2 = diplomacy_game.get_discussion_prompt(
            "Agent_1", state2, 1, 5, []
        )

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)

    def test_get_voting_prompt(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_voting_prompt."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        # Create game-specific proposals
        proposal1 = {"allocation": {"Agent_1": [0], "Agent_2": [1, 2]}, "reasoning": "Test"}
        proposal2 = {"agreement": [0.5, 0.5, 0.5], "reasoning": "Test"}

        prompt1 = item_allocation_game.get_voting_prompt("Agent_1", proposal1, state1, 1)
        prompt2 = diplomacy_game.get_voting_prompt("Agent_1", proposal2, state2, 1)

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)

    def test_get_thinking_prompt(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_thinking_prompt."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        prompt1 = item_allocation_game.get_thinking_prompt(
            "Agent_1", state1, 1, 5, []
        )
        prompt2 = diplomacy_game.get_thinking_prompt(
            "Agent_1", state2, 1, 5, []
        )

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)

    def test_parse_proposal(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement parse_proposal."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        response1 = '{"allocation": {"Agent_1": [0], "Agent_2": [1, 2]}}'
        response2 = '{"agreement": [0.5, 0.5, 0.5]}'

        proposal1 = item_allocation_game.parse_proposal(
            response1, "Agent_1", state1, ["Agent_1", "Agent_2"]
        )
        proposal2 = diplomacy_game.parse_proposal(
            response2, "Agent_1", state2, ["Agent_1", "Agent_2"]
        )

        assert isinstance(proposal1, dict)
        assert isinstance(proposal2, dict)

    def test_validate_proposal(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement validate_proposal."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        proposal1 = {"allocation": {"Agent_1": [0], "Agent_2": [1, 2]}}
        proposal2 = {"agreement": [0.5, 0.5, 0.5]}

        result1 = item_allocation_game.validate_proposal(proposal1, state1)
        result2 = diplomacy_game.validate_proposal(proposal2, state2)

        assert isinstance(result1, bool)
        assert isinstance(result2, bool)

    def test_calculate_utility(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement calculate_utility."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        proposal1 = {"allocation": {"Agent_1": [0], "Agent_2": [1, 2]}}
        proposal2 = {"agreement": [0.5, 0.5, 0.5]}

        utility1 = item_allocation_game.calculate_utility("Agent_1", proposal1, state1, 1)
        utility2 = diplomacy_game.calculate_utility("Agent_1", proposal2, state2, 1)

        assert isinstance(utility1, (int, float))
        assert isinstance(utility2, (int, float))

    def test_format_proposal_display(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement format_proposal_display."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        proposal1 = {"allocation": {"Agent_1": [0], "Agent_2": [1, 2]}, "proposed_by": "Agent_1"}
        proposal2 = {"agreement": [0.5, 0.5, 0.5], "proposed_by": "Agent_1"}

        display1 = item_allocation_game.format_proposal_display(proposal1, state1)
        display2 = diplomacy_game.format_proposal_display(proposal2, state2)

        assert isinstance(display1, str)
        assert isinstance(display2, str)

    def test_get_game_type(self, item_allocation_game, diplomacy_game):
        """Test that both implement get_game_type."""
        type1 = item_allocation_game.get_game_type()
        type2 = diplomacy_game.get_game_type()

        assert isinstance(type1, GameType)
        assert isinstance(type2, GameType)
        assert type1 != type2

    def test_get_agent_preferences_summary(self, item_allocation_game, diplomacy_game, agents):
        """Test that both implement get_agent_preferences_summary."""
        state1 = item_allocation_game.create_game_state(agents)
        state2 = diplomacy_game.create_game_state(agents)

        summary1 = item_allocation_game.get_agent_preferences_summary("Agent_1", state1)
        summary2 = diplomacy_game.get_agent_preferences_summary("Agent_1", state2)

        assert isinstance(summary1, dict)
        assert isinstance(summary2, dict)


class TestCommonBehavior:
    """Tests for common behavior across game types."""

    def test_discount_applied_consistently(self):
        """Test that discount factor is applied the same way in both games."""
        gamma = 0.8

        game1 = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3,
            gamma_discount=gamma, random_seed=42
        )
        game2 = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3,
            gamma_discount=gamma, random_seed=42
        )
        agents = create_test_agents(2)

        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        # Create proposals that give maximum utility
        proposal1 = {"allocation": {"Agent_1": [0, 1, 2], "Agent_2": []}}
        positions = state2["agent_positions"]["Agent_1"]
        proposal2 = {"agreement": positions}

        # Calculate utilities at different rounds
        u1_r1 = game1.calculate_utility("Agent_1", proposal1, state1, 1)
        u1_r2 = game1.calculate_utility("Agent_1", proposal1, state1, 2)
        u1_r3 = game1.calculate_utility("Agent_1", proposal1, state1, 3)

        u2_r1 = game2.calculate_utility("Agent_1", proposal2, state2, 1)
        u2_r2 = game2.calculate_utility("Agent_1", proposal2, state2, 2)
        u2_r3 = game2.calculate_utility("Agent_1", proposal2, state2, 3)

        # Verify discount ratios are the same
        ratio1_12 = u1_r2 / u1_r1
        ratio2_12 = u2_r2 / u2_r1
        assert abs(ratio1_12 - gamma) < 1e-6
        assert abs(ratio2_12 - gamma) < 1e-6

        ratio1_13 = u1_r3 / u1_r1
        ratio2_13 = u2_r3 / u2_r1
        assert abs(ratio1_13 - gamma**2) < 1e-6
        assert abs(ratio2_13 - gamma**2) < 1e-6

    def test_voting_prompts_request_json(self):
        """Test that voting prompts request JSON format in both games."""
        game1 = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        game2 = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)

        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        proposal1 = {"allocation": {"Agent_1": [0], "Agent_2": [1, 2]}}
        proposal2 = {"agreement": [0.5, 0.5, 0.5]}

        prompt1 = game1.get_voting_prompt("Agent_1", proposal1, state1, 1)
        prompt2 = game2.get_voting_prompt("Agent_1", proposal2, state2, 1)

        # Both should request JSON format
        assert "JSON" in prompt1 or "json" in prompt1
        assert "JSON" in prompt2 or "json" in prompt2

        # Both should mention accept/reject
        assert "accept" in prompt1.lower()
        assert "reject" in prompt1.lower()
        assert "accept" in prompt2.lower()
        assert "reject" in prompt2.lower()

    def test_default_reflection_prompt(self):
        """Test that default reflection prompt works for both games."""
        game1 = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        game2 = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)

        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        # Both should have get_reflection_prompt method (default implementation)
        prompt1 = game1.get_reflection_prompt("Agent_1", state1, 1, 5, {})
        prompt2 = game2.get_reflection_prompt("Agent_1", state2, 1, 5, {})

        assert isinstance(prompt1, str)
        assert isinstance(prompt2, str)


class TestGameTypeEnum:
    """Tests for GameType enum."""

    def test_enum_values(self):
        """Test that GameType enum has expected values."""
        assert GameType.ITEM_ALLOCATION.value == "item_allocation"
        assert GameType.DIPLOMATIC_TREATY.value == "diplomacy"

    def test_enum_uniqueness(self):
        """Test that enum values are unique."""
        values = [gt.value for gt in GameType]
        assert len(values) == len(set(values))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
