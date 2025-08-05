#!/usr/bin/env python3
"""
Quick test script to run O3 vs Haiku experiment with full debugging output.
This will print all O3 responses to the terminal for debugging JSON parsing issues.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment

async def test_o3_parsing():
    """Run a single O3 vs Haiku experiment with debugging."""
    print("🚀 Starting O3 vs Haiku Baseline Experiment with Debugging")
    print("=" * 80)
    
    # Create experiment instance
    experiment = O3VsHaikuExperiment()
    
    # Run a single experiment
    try:
        print("Running single experiment...")
        result = await experiment.run_single_experiment(
            experiment_config=None,  # Use default config
            experiment_id="debug_test_001"
        )
        
        print("\n" + "=" * 80)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Experiment ID: {result.experiment_id}")
        print(f"Consensus reached: {result.consensus_reached}")
        print(f"Final round: {result.final_round}")
        print(f"Winner: {result.winner_agent_id}")
        print(f"O3 won: {result.o3_won}")
        print(f"Exploitation detected: {result.exploitation_detected}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {e}")
        print(f"Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting debug test...")
    asyncio.run(test_o3_parsing())