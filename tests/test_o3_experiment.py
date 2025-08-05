#!/usr/bin/env python3
"""
Quick test of O3 experiment focusing on voting phase
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment

async def test_o3_voting():
    """Test O3 voting response parsing in isolation."""
    exp = O3VsHaikuExperiment()
    
    # Sample proposals similar to what we see in experiments
    sample_proposals = [
        {
            "proposal_number": 1,
            "allocation": {"o3_agent": [0], "haiku_agent_1": [1, 4], "haiku_agent_2": [2, 3]},
            "reasoning": "Test proposal 1"
        },
        {
            "proposal_number": 2, 
            "allocation": {"o3_agent": [0, 1, 4], "haiku_agent_1": [2, 3], "haiku_agent_2": []},
            "reasoning": "Test proposal 2"
        },
        {
            "proposal_number": 3,
            "allocation": {"o3_agent": [0, 2, 4], "haiku_agent_1": [1, 3], "haiku_agent_2": []},
            "reasoning": "Test proposal 3"
        }
    ]
    
    # Test the actual O3 voting prompt
    voting_prompt = exp._create_private_voting_prompt(sample_proposals)
    print("O3 Voting Prompt:")
    print("=" * 50)
    print(voting_prompt[:500] + "...")
    print("=" * 50)
    
    # We'd need to call O3 here to see the actual response format
    # For now, let's just verify our parsing logic works
    print("Testing parsing logic...")
    
if __name__ == "__main__":
    asyncio.run(test_o3_voting())