#!/usr/bin/env python3
"""
Debug script to test agent creation in isolation.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from negotiation.agent_factory import AgentFactory, create_o3_vs_haiku_experiment
from negotiation.llm_agents import ModelType

def test_agent_creation():
    print("Testing agent creation...")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"OpenAI API Key: {'Available' if openai_key else 'Missing'}")
    print(f"Anthropic API Key: {'Available' if anthropic_key else 'Missing'}")
    
    if not openai_key or not anthropic_key:
        print("API keys missing, cannot test real agents")
        return
    
    print("\n" + "="*50)
    print("Creating experiment configuration...")
    
    try:
        exp_config = create_o3_vs_haiku_experiment(
            experiment_name="Debug Test",
            competition_level=0.9,
            random_seed=42
        )
        print(f"Experiment config created successfully")
        print(f"Number of agents: {len(exp_config.agents)}")
        
        for i, agent_config in enumerate(exp_config.agents):
            print(f"\nAgent {i}:")
            print(f"  ID: {agent_config.agent_id}")
            print(f"  Model Type: {agent_config.model_type}")
            print(f"  Model Type Value: {agent_config.model_type.value}")
            print(f"  Has API Key: {'Yes' if agent_config.api_key else 'No'}")
            
    except Exception as e:
        print(f"Failed to create experiment config: {e}")
        return
    
    print("\n" + "="*50)
    print("Creating agents...")
    
    factory = AgentFactory()
    
    for agent_config in exp_config.agents:
        print(f"\n--- Creating {agent_config.agent_id} ---")
        try:
            agent = factory.create_agent(agent_config)
            model_info = agent.get_model_info()
            print(f"✓ Agent created successfully")
            print(f"  Provider: {model_info.get('provider', 'Unknown')}")
            print(f"  Model: {model_info.get('model_type', 'Unknown')}")
        except Exception as e:
            print(f"✗ Failed to create agent: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_agent_creation()