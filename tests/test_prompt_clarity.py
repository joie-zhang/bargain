#!/usr/bin/env python3
"""Test that the prompt changes for 2-agent negotiations are working correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strong_models_experiment.prompts.prompt_generator import PromptGenerator

def test_prompt_clarity():
    """Test that prompts are clear for different numbers of agents."""

    items = [
        {"name": "Apple"},
        {"name": "Jewel"},
        {"name": "Stone"}
    ]

    config = {
        "t_rounds": 10,
        "gamma_discount": 0.9
    }

    print("Testing prompt clarity for different agent counts:\n")
    print("="*60)

    # Test 2-agent negotiation (should say "with another agent")
    prompt_2 = PromptGenerator.create_game_rules_prompt(items, num_agents=2, config=config)
    print("\n2-AGENT NEGOTIATION:")
    print("-"*40)
    # Extract the key line
    for line in prompt_2.split('\n'):
        if 'participating in a strategic negotiation' in line:
            print(f"✓ {line}")
            assert "with another agent" in line, "Should say 'with another agent' for 2 agents"
            break

    # Test 3-agent negotiation (should say "with 2 other agents")
    prompt_3 = PromptGenerator.create_game_rules_prompt(items, num_agents=3, config=config)
    print("\n3-AGENT NEGOTIATION:")
    print("-"*40)
    for line in prompt_3.split('\n'):
        if 'participating in a strategic negotiation' in line:
            print(f"✓ {line}")
            assert "with 2 other agents" in line, "Should say 'with 2 other agents' for 3 agents"
            break

    # Test 5-agent negotiation (should say "with 4 other agents")
    prompt_5 = PromptGenerator.create_game_rules_prompt(items, num_agents=5, config=config)
    print("\n5-AGENT NEGOTIATION:")
    print("-"*40)
    for line in prompt_5.split('\n'):
        if 'participating in a strategic negotiation' in line:
            print(f"✓ {line}")
            assert "with 4 other agents" in line, "Should say 'with 4 other agents' for 5 agents"
            break

    print("\n" + "="*60)
    print("✅ All tests passed! Prompts are now clear and unambiguous.")

if __name__ == "__main__":
    test_prompt_clarity()