#!/usr/bin/env python3
"""Test the logging and JSON saving functionality."""

import asyncio
import json
import os
from pathlib import Path

# Show what would be saved
def demo_json_structure():
    """Demo the JSON structure that will be saved."""
    
    # Example of all_interactions.json
    all_interactions = [
        {
            "timestamp": 1234567890.123,
            "experiment_id": "strong_models_20250819_123456_12345",
            "agent_id": "qwen_3_235b_1",
            "phase": "game_setup",
            "round": 0,
            "prompt": "Welcome to the Multi-Agent Negotiation Game...",
            "response": "I understand the rules and am ready to begin..."
        },
        {
            "timestamp": 1234567891.456,
            "experiment_id": "strong_models_20250819_123456_12345",
            "agent_id": "claude_4_sonnet_2",
            "phase": "preference_assignment",
            "round": 0,
            "prompt": "Your private preferences are...",
            "response": "I acknowledge my preferences..."
        },
        {
            "timestamp": 1234567892.789,
            "experiment_id": "strong_models_20250819_123456_12345",
            "agent_id": "qwen_3_235b_1",
            "phase": "discussion_round_1",
            "round": 1,
            "prompt": "Please discuss your preferences...",
            "response": "I value the Jewel and Stone most highly..."
        }
    ]
    
    # Example of agent-specific file
    agent_interactions = {
        "agent_id": "qwen_3_235b_1",
        "total_interactions": 15,
        "interactions": [
            {
                "timestamp": 1234567890.123,
                "experiment_id": "strong_models_20250819_123456_12345",
                "agent_id": "qwen_3_235b_1",
                "phase": "game_setup",
                "round": 0,
                "prompt": "Welcome to the Multi-Agent Negotiation Game...",
                "response": "I understand the rules..."
            },
            {
                "timestamp": 1234567892.789,
                "experiment_id": "strong_models_20250819_123456_12345",
                "agent_id": "qwen_3_235b_1", 
                "phase": "private_thinking_round_1",
                "round": 1,
                "prompt": "Think strategically about...",
                "response": '{"reasoning": "I should focus on...", "strategy": "Propose a fair split...", "target_items": ["Jewel", "Stone"]}'
            }
        ]
    }
    
    # Example of experiment_results.json
    experiment_results = {
        "experiment_id": "strong_models_20250819_123456_12345",
        "timestamp": 1234567899.999,
        "consensus_reached": True,
        "final_round": 3,
        "winner_agent_id": "qwen_3_235b_1",
        "final_utilities": {
            "qwen_3_235b_1": 15.5,
            "claude_4_sonnet_2": 12.3
        },
        "strategic_behaviors": {
            "manipulation_attempts": 2,
            "cooperation_signals": 5
        }
    }
    
    print("📁 JSON Files That Will Be Saved:")
    print("=" * 60)
    
    print("\n1️⃣ all_interactions.json")
    print("   - Contains EVERY interaction from ALL agents")
    print("   - Example structure:")
    print(json.dumps(all_interactions[0], indent=2)[:300] + "...")
    
    print("\n2️⃣ agent_<agent_id>_interactions.json")
    print("   - Contains all interactions for a specific agent")
    print("   - Example structure:")
    print(json.dumps(agent_interactions, indent=2)[:400] + "...")
    
    print("\n3️⃣ experiment_results.json")
    print("   - Contains final experiment results and statistics")
    print("   - Example structure:")
    print(json.dumps(experiment_results, indent=2)[:300] + "...")
    
    print("\n📊 What Gets Logged to Console:")
    print("=" * 60)
    print("✅ Game setup messages (full)")
    print("✅ Private preferences (full list)")
    print("✅ Discussion messages (full text)")
    print("✅ Private thinking (full reasoning)")
    print("✅ Proposals (full allocation + reasoning)")
    print("✅ Votes (with reasoning)")
    print("✅ Reflections (full text)")
    
    print("\n📂 Directory Structure:")
    print("=" * 60)
    print("experiments/results/")
    print("└── strong_models_20250819_123456_12345/")
    print("    ├── all_interactions.json")
    print("    ├── agent_qwen_3_235b_1_interactions.json")
    print("    ├── agent_claude_4_sonnet_2_interactions.json")
    print("    └── experiment_results.json")

if __name__ == "__main__":
    demo_json_structure()
