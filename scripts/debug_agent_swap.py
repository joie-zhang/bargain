#!/usr/bin/env python3
"""
Debug script to trace the agent-model mapping bug.

Run with:
    python scripts/debug_agent_swap.py

This script traces the exact flow of model names through the system
to find where the swap occurs.
"""

import sys
import os
import asyncio
import logging

# Setup logging to see DEBUG messages
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strong_models_experiment.agents import StrongModelAgentFactory
from strong_models_experiment.configs import STRONG_MODELS_CONFIG


async def debug_agent_creation():
    """Test agent creation to trace the model mapping."""

    print("=" * 70)
    print("DEBUG: Agent-Model Mapping Trace")
    print("=" * 70)

    # Test both orders
    test_cases = [
        ("strong_first", ['claude-opus-4-5-thinking-32k', 'gpt-5-nano']),
        ("weak_first", ['gpt-5-nano', 'claude-opus-4-5-thinking-32k']),
    ]

    for model_order, models in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {model_order}")
        print(f"{'='*70}")

        print(f"\n1. INPUT models list: {models}")
        print(f"   Expected for {model_order}:")
        print(f"     Agent_Alpha -> {models[0]}")
        print(f"     Agent_Beta  -> {models[1]}")

        # Check config for each model
        print(f"\n2. MODEL CONFIGS:")
        for model_name in models:
            cfg = STRONG_MODELS_CONFIG.get(model_name, {})
            print(f"   {model_name}:")
            print(f"     api_type: {cfg.get('api_type', 'NOT FOUND')}")
            print(f"     model_id: {cfg.get('model_id', 'NOT FOUND')}")

        # Create factory
        factory = StrongModelAgentFactory()

        # Test with reasoning_token_budget to verify it's applied correctly
        config = {
            'model_order': model_order,
            'max_tokens_default': 1000,
            'reasoning_token_budget': 5000,  # Test budget to verify correct application
        }

        print(f"\n3. CREATING AGENTS (with reasoning_token_budget={config['reasoning_token_budget']})...")

        try:
            agents = await factory.create_agents(models, config)

            print(f"\n4. CREATED AGENTS:")
            for agent in agents:
                agent_type = type(agent).__name__
                api_type = "ANTHROPIC" if "Anthropic" in agent_type else "OPENAI" if "OpenAI" in agent_type else "OTHER"
                # Check if extended thinking/reasoning is enabled
                thinking_budget = agent.config.custom_parameters.get("thinking_budget_tokens", None)
                reasoning_effort = agent.config.custom_parameters.get("reasoning_effort", None)
                reasoning_info = f"thinking_budget={thinking_budget}" if thinking_budget else f"reasoning_effort={reasoning_effort}" if reasoning_effort else "no reasoning config"
                print(f"   {agent.agent_id}: {agent_type} ({api_type}) -> {getattr(agent, 'model_name', 'N/A')} [{reasoning_info}]")

            # For strong_first: Agent_Alpha should be Anthropic (reasoning), Agent_Beta should be OpenAI (baseline)
            # For weak_first: Agent_Alpha should be OpenAI (baseline), Agent_Beta should be Anthropic (reasoning)
            print(f"\n5. VERIFICATION for {model_order}:")
            if model_order == "strong_first":
                expected = {"Agent_Alpha": "AnthropicAgent", "Agent_Beta": "OpenAIAgent"}
                reasoning_agent = "Agent_Alpha"
            else:
                expected = {"Agent_Alpha": "OpenAIAgent", "Agent_Beta": "AnthropicAgent"}
                reasoning_agent = "Agent_Beta"

            all_passed = True
            for agent in agents:
                agent_type = type(agent).__name__
                exp = expected.get(agent.agent_id, "Unknown")
                match = "✓" if agent_type == exp else "✗ MISMATCH!"
                if agent_type != exp:
                    all_passed = False
                print(f"   {agent.agent_id}: {agent_type} (expected {exp}) {match}")

            # Verify reasoning budget is applied only to the correct agent
            print(f"\n6. REASONING BUDGET VERIFICATION (should only be on {reasoning_agent}):")
            for agent in agents:
                thinking_budget = agent.config.custom_parameters.get("thinking_budget_tokens", None)
                reasoning_effort = agent.config.custom_parameters.get("reasoning_effort", None)
                has_reasoning = thinking_budget is not None or reasoning_effort is not None
                should_have = agent.agent_id == reasoning_agent
                status = "✓" if has_reasoning == should_have else "✗ BUG!"
                if has_reasoning != should_have:
                    all_passed = False
                print(f"   {agent.agent_id}: has_reasoning={has_reasoning}, should_have={should_have} {status}")

            if all_passed:
                print(f"\n✓ All checks passed for {model_order}!")
            else:
                print(f"\n✗ Some checks FAILED for {model_order}!")

        except Exception as e:
            print(f"\nERROR creating agents: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(debug_agent_creation())
