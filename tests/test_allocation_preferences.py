#!/usr/bin/env python3
"""Test script to verify allocation and preferences are saved correctly."""

import json
import asyncio
from pathlib import Path
from strong_models_experiment.phases.phase_handlers import PhaseHandler
# Mock agent for testing - simplified version
class MockAgent:
    def __init__(self, agent_id, responses):
        self.agent_id = agent_id
        self.responses = responses
        self.response_idx = 0

    async def generate_response(self, context, prompt):
        response = self.responses[self.response_idx % len(self.responses)]
        self.response_idx += 1
        return type('Response', (), {'content': response})()

async def test_phase_handlers():
    """Test that allocation and preferences are properly saved."""

    # Set up test data
    agents = [
        MockAgent("agent_0", ['{"vote": "accept", "reasoning": "Good deal"}']),
        MockAgent("agent_1", ['{"vote": "accept", "reasoning": "I agree"}'])
    ]

    items = [
        {"name": "Apple"},
        {"name": "Banana"},
        {"name": "Cherry"}
    ]

    preferences = {
        "agent_preferences": {
            "agent_0": [10.0, 5.0, 2.0],
            "agent_1": [2.0, 8.0, 10.0]
        }
    }

    # Create test proposals
    enumerated_proposals = [
        {
            "proposal_number": 1,
            "proposer": "agent_0",
            "allocation": {
                "agent_0": [0],
                "agent_1": [1, 2]
            },
            "reasoning": "I take Apple, you take the rest"
        }
    ]

    # Create mock votes (both agents accept)
    private_votes = [
        {
            "voter_id": "agent_0",
            "proposal_number": 1,
            "vote": "accept",
            "reasoning": "Good for me"
        },
        {
            "voter_id": "agent_1",
            "proposal_number": 1,
            "vote": "accept",
            "reasoning": "Fair enough"
        }
    ]

    # Test the phase handler
    handler = PhaseHandler()

    result = await handler.run_vote_tabulation_phase(
        agents, items, preferences, 1, 10,
        private_votes, enumerated_proposals
    )

    print("Test Results:")
    print("=" * 50)
    print(f"Consensus reached: {result['consensus_reached']}")
    print(f"Final utilities: {result['final_utilities']}")
    print(f"Final allocation: {result['final_allocation']}")
    print(f"Agent preferences: {result['agent_preferences']}")

    # Verify the results
    assert result['consensus_reached'] == True, "Consensus should be reached"
    assert result['final_allocation'] == {
        "agent_0": [0],
        "agent_1": [1, 2]
    }, "Allocation should match the accepted proposal"
    assert result['agent_preferences'] == preferences['agent_preferences'], "Preferences should be saved"

    print("\n✅ All tests passed!")

    # Show what the saved JSON would look like
    sample_output = {
        "consensus_reached": result['consensus_reached'],
        "final_allocation": result['final_allocation'],
        "agent_preferences": result['agent_preferences'],
        "final_utilities": result['final_utilities']
    }

    print("\nSample JSON output:")
    print(json.dumps(sample_output, indent=2))

if __name__ == "__main__":
    asyncio.run(test_phase_handlers())