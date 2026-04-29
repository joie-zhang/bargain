#!/usr/bin/env python3
"""
Regression tests for ItemAllocationGame after refactoring.

Tests verify:
- Item allocation still works correctly after refactor
- Backward compatibility with existing usage patterns
- Core functionality preserved
"""

import json

import pytest
import numpy as np
from typing import List

from game_environments import (
    create_game_environment,
    ItemAllocationConfig,
    ItemAllocationGame,
    GameType,
)


class FakeAgent:
    """Simple agent for testing."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def create_test_agents(n: int = 2) -> List[FakeAgent]:
    """Create a list of fake agents for testing."""
    return [FakeAgent(f"Agent_{i+1}") for i in range(n)]


class TestItemAllocationConfig:
    """Tests for ItemAllocationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ItemAllocationConfig(n_agents=2, t_rounds=10)
        assert config.m_items == 5
        assert config.competition_level == 0.95
        assert config.gamma_discount == 0.9

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ItemAllocationConfig(
            n_agents=3,
            t_rounds=15,
            m_items=7,
            competition_level=0.8,
            gamma_discount=0.85
        )
        assert config.n_agents == 3
        assert config.t_rounds == 15
        assert config.m_items == 7
        assert config.competition_level == 0.8
        assert config.gamma_discount == 0.85


class TestGameStateCreation:
    """Tests for game state creation."""

    def test_game_state_structure(self):
        """Test that game state has all required keys."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        assert "items" in state
        assert "agent_preferences" in state
        assert "game_type" in state
        assert state["game_type"] == "item_allocation"

    def test_correct_number_of_items(self):
        """Test that correct number of items are created."""
        for m_items in [3, 5, 8]:
            game = create_game_environment(
                "item_allocation", n_agents=2, t_rounds=5, m_items=m_items, random_seed=42
            )
            agents = create_test_agents(2)
            state = game.create_game_state(agents)

            assert len(state["items"]) == m_items
            for agent_id in state["agent_preferences"]:
                assert len(state["agent_preferences"][agent_id]) == m_items

    def test_items_have_names(self):
        """Test that items have name attributes."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for item in state["items"]:
            assert "name" in item
            assert isinstance(item["name"], str)
            assert len(item["name"]) > 0

    def test_twenty_five_item_names_are_human_readable(self):
        """The n=10 runs use M=25 and should not expose fallback item labels."""
        game = create_game_environment(
            "item_allocation", n_agents=10, t_rounds=5, m_items=25, random_seed=42
        )
        agents = create_test_agents(10)
        state = game.create_game_state(agents)

        names = [item["name"] for item in state["items"]]
        assert names == [
            "Apple", "Jewel", "Stone", "Quill", "Pencil",
            "Book", "Hat", "Camera", "Ring", "Clock",
            "Key", "Map", "Lantern", "Compass", "Vase",
            "Coin", "Shell", "Brush", "Scarf", "Cup",
            "Bottle", "Globe", "Medal", "Ticket", "Tablet",
        ]
        assert not any(name.startswith("Item_") for name in names)

    def test_preferences_are_numeric(self):
        """Test that preferences are numeric values."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for agent_id, prefs in state["agent_preferences"].items():
            for pref in prefs:
                assert isinstance(pref, (int, float))

    def test_preferences_sum_to_100_and_are_integer_valued(self):
        """Generated preferences should stay integer-valued and sum to 100."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for prefs in state["agent_preferences"].values():
            assert abs(sum(prefs) - 100.0) < 1e-6
            assert all(float(pref).is_integer() for pref in prefs)

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same game state."""
        game1 = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        game2 = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)

        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        assert state1["agent_preferences"] == state2["agent_preferences"]


class TestUtilityCalculation:
    """Tests for utility calculation."""

    def test_all_items_max_utility(self):
        """Test that getting all items gives maximum utility."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3,
            gamma_discount=1.0, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        prefs = state["agent_preferences"][agent_id]
        expected_max = sum(prefs)

        # Proposal giving all items to agent
        proposal = {
            "allocation": {
                "Agent_1": [0, 1, 2],
                "Agent_2": []
            }
        }

        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)
        assert abs(utility - expected_max) < 1e-6

    def test_no_items_zero_utility(self):
        """Test that getting no items gives zero utility."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {
            "allocation": {
                "Agent_1": [],
                "Agent_2": [0, 1, 2]
            }
        }

        utility = game.calculate_utility("Agent_1", proposal, state, round_num=1)
        assert utility == 0.0

    def test_discount_factor_applied(self):
        """Test that discount factor is applied correctly."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3,
            gamma_discount=0.9, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {
            "allocation": {
                "Agent_1": [0, 1],
                "Agent_2": [2]
            }
        }

        utility_r1 = game.calculate_utility("Agent_1", proposal, state, round_num=1)
        utility_r2 = game.calculate_utility("Agent_1", proposal, state, round_num=2)
        utility_r3 = game.calculate_utility("Agent_1", proposal, state, round_num=3)

        # Check discount is applied
        assert abs(utility_r2 - utility_r1 * 0.9) < 1e-6
        assert abs(utility_r3 - utility_r1 * 0.81) < 1e-6

    def test_partial_allocation_utility(self):
        """Test utility for partial item allocation."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=4,
            gamma_discount=1.0, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        prefs = state["agent_preferences"][agent_id]

        # Give items 0 and 2 to Agent_1
        proposal = {
            "allocation": {
                "Agent_1": [0, 2],
                "Agent_2": [1, 3]
            }
        }

        expected_utility = prefs[0] + prefs[2]
        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)
        assert abs(utility - expected_utility) < 1e-6


class TestProposalValidation:
    """Tests for proposal validation."""

    def test_valid_proposal(self):
        """Test that valid proposal passes validation."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {
            "allocation": {
                "Agent_1": [0, 2, 4],
                "Agent_2": [1, 3]
            }
        }
        assert game.validate_proposal(proposal, state) is True

    def test_missing_items_invalid(self):
        """Test that proposal missing items fails validation."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Missing item 4
        proposal = {
            "allocation": {
                "Agent_1": [0, 2],
                "Agent_2": [1, 3]
            }
        }
        assert game.validate_proposal(proposal, state) is False

    def test_duplicate_items_invalid(self):
        """Test that proposal with duplicate items fails validation."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Item 2 duplicated
        proposal = {
            "allocation": {
                "Agent_1": [0, 2, 4],
                "Agent_2": [1, 2, 3]
            }
        }
        assert game.validate_proposal(proposal, state) is False

    def test_all_items_to_one_agent_valid(self):
        """Test that giving all items to one agent is valid."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {
            "allocation": {
                "Agent_1": [0, 1, 2],
                "Agent_2": []
            }
        }
        assert game.validate_proposal(proposal, state) is True


class TestProposalParsing:
    """Tests for proposal parsing."""

    def test_valid_json_parsing(self):
        """Test parsing valid JSON response."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '''
        {
            "allocation": {
                "Agent_1": [0, 2],
                "Agent_2": [1]
            },
            "reasoning": "Fair split"
        }
        '''
        proposal = game.parse_proposal(
            response, "Agent_1", state, ["Agent_1", "Agent_2"]
        )

        assert proposal["allocation"]["Agent_1"] == [0, 2]
        assert proposal["allocation"]["Agent_2"] == [1]
        assert proposal["proposed_by"] == "Agent_1"

    def test_invalid_json_preserves_diagnostics_without_valid_fallback(self):
        """Invalid JSON should not become a valid proposer-gets-all proposal."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = "This is not valid JSON!"
        proposal = game.parse_proposal(
            response, "Agent_1", state, ["Agent_1", "Agent_2"]
        )

        assert proposal["allocation"]["Agent_1"] == []
        assert proposal["allocation"]["Agent_2"] == []
        assert proposal["raw_response"] == response
        assert proposal["parse_error"]["type"] == "ValueError"
        assert game.validate_proposal(proposal, state) is False

    def test_legacy_agreement_vector_is_not_treated_as_allocation(self):
        """Game 2-style agreement vectors are not interpretable Game 1 allocations."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '{"agreement": [36, 0, 12], "reasoning": "old schema"}'
        proposal = game.parse_proposal(
            response, "Agent_1", state, ["Agent_1", "Agent_2"]
        )

        assert proposal["allocation"] == {"Agent_1": [], "Agent_2": []}
        assert proposal["parse_error"]["message"] == "No allocation in proposal"
        assert game.validate_proposal(proposal, state) is False

    def test_legacy_agreement_mapping_key_is_recovered(self):
        """A correct allocation under the old agreement key can be safely recovered."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '{"agreement": {"Agent_1": [0, 2], "Agent_2": [1]}, "reasoning": "wrong key"}'
        proposal = game.parse_proposal(
            response, "Agent_1", state, ["Agent_1", "Agent_2"]
        )

        assert proposal["allocation"] == {"Agent_1": [0, 2], "Agent_2": [1]}
        assert proposal["recovered_from_legacy_agreement_key"] is True
        assert "parse_error" not in proposal
        assert game.validate_proposal(proposal, state) is True


class TestGameType:
    """Tests for game type identification."""

    def test_game_type(self):
        """Test that game type is correctly identified."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        assert game.get_game_type() == GameType.ITEM_ALLOCATION

    def test_factory_creates_correct_type(self):
        """Test that factory creates ItemAllocationGame."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5
        )
        assert isinstance(game, ItemAllocationGame)


class TestPromptGeneration:
    """Tests for prompt generation methods."""

    def test_game_rules_prompt(self):
        """Test that game rules prompt contains key information."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=10, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_game_rules_prompt(state)

        assert "item" in prompt.lower() or "Item" in prompt
        assert "5" in prompt  # number of items
        assert "10" in prompt  # number of rounds

    def test_preference_prompt_contains_values(self):
        """Test that preference prompt contains agent's preference values."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_preference_assignment_prompt("Agent_1", state)

        # Should contain preference values
        for pref in state["agent_preferences"]["Agent_1"]:
            expected = str(int(pref)) if float(pref).is_integer() else f"{pref:.2f}"
            assert expected in prompt

    def test_preference_prompt_omits_trailing_point_zero_zero_for_integer_values(self):
        """Integer-valued preferences should render without .00 in prompts."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_preference_assignment_prompt("Agent_1", state)

        for pref in state["agent_preferences"]["Agent_1"]:
            if float(pref).is_integer():
                assert f"-> {int(pref)}" in prompt
                assert f"-> {int(pref)}.00" not in prompt

        assert "100 points" in prompt
        assert "100.00 points" not in prompt

    def test_thinking_prompt_omits_trailing_point_zero_zero_for_integer_values(self):
        """Private thinking prompt should also use compact integer formatting."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_thinking_prompt(
            "Agent_1",
            state,
            round_num=1,
            max_rounds=5,
            discussion_history=[],
        )

        for idx, pref in enumerate(state["agent_preferences"]["Agent_1"]):
            if float(pref).is_integer():
                item_name = state["items"][idx]["name"]
                assert f"{idx}: {item_name} -> {int(pref)}" in prompt
                assert f"{idx}: {item_name} -> {int(pref)}.00" not in prompt

        assert "**YOUR FULL PREFERENCE REMINDER:**" in prompt
        assert "**YOUR TOP PRIORITIES:**" not in prompt

    def test_item_allocation_uses_combined_setup_phase(self):
        """Item Allocation should merge private preferences into setup."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        assert game.uses_combined_setup_phase() is True

    def test_combined_setup_prompt_contains_rules_and_private_preferences(self):
        """Combined setup prompt should include both shared and private information."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_combined_setup_prompt("Agent_1", state)

        assert "GAME STRUCTURE" in prompt
        assert "REWARD DISCOUNTING" in prompt
        assert "WINNING CONDITIONS" in prompt
        assert "YOUR PRIVATE ITEM PREFERENCES" in prompt
        assert "Please do not initiate the discussion or proposal phase yet." in prompt
        assert "summarize the game structure and rules" in prompt
        assert "reiterate the private preferences that were assigned to you" in prompt
        assert "you have been assigned the following SECRET preference values" in prompt
        assert "If no agreement is reached by the final round, then all agents walk away with zero utility." in prompt
        assert "The goal is to maximize your utility, which is the sum of the utility from each of the objects that you receive" in prompt
        assert "HIGH PRIORITY" not in prompt
        assert "Medium Priority" not in prompt
        assert "Low Priority" not in prompt

        for pref in state["agent_preferences"]["Agent_1"]:
            expected = str(int(pref)) if float(pref).is_integer() else f"{pref:.2f}"
            assert expected in prompt

    def test_proposal_prompt_contains_instructions(self):
        """Test that proposal prompt contains clear instructions."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_proposal_prompt("Agent_1", state, 1, ["Agent_1", "Agent_2"])

        assert "allocation" in prompt.lower()
        assert "JSON" in prompt or "json" in prompt

    def test_proposal_prompt_example_is_complete_for_n10_m25(self):
        """Prompt example should be a structurally valid full allocation."""
        game = create_game_environment(
            "item_allocation", n_agents=10, t_rounds=5, m_items=25, random_seed=42
        )
        agents = create_test_agents(10)
        agent_ids = [agent.agent_id for agent in agents]
        state = {"items": [{"name": f"Item {item_index}"} for item_index in range(25)]}

        prompt = game.get_proposal_prompt("Agent_1", state, 1, agent_ids)
        example_text = prompt.split("Respond with ONLY a JSON object in this exact format:\n", 1)[1]
        example_text = example_text.split("\n\n**Rules:**", 1)[0]
        example = json.loads(example_text)
        allocation = example["allocation"]

        assert set(allocation) == set(agent_ids)
        allocated_items = [
            item_index
            for item_indices in allocation.values()
            for item_index in item_indices
        ]
        assert sorted(allocated_items) == list(range(25))
        assert "agreement" not in example_text.lower()
        assert "contributions" not in example_text.lower()


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_create_game_environment_with_minimal_args(self):
        """Test that game can be created with minimal arguments."""
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=5
        )
        assert game is not None
        assert isinstance(game, ItemAllocationGame)

    def test_game_state_usable_with_phase_handler_format(self):
        """Test that game state works with expected format."""
        game = create_game_environment(
            "item_allocation", n_agents=2, t_rounds=5, m_items=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Should be able to extract items and preferences
        items = state["items"]
        prefs = {"agent_preferences": state["agent_preferences"]}

        assert len(items) == 5
        assert "Agent_1" in prefs["agent_preferences"]
        assert "Agent_2" in prefs["agent_preferences"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
