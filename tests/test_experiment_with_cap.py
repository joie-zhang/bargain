#!/usr/bin/env python3
"""
Test script to run a small experiment with reflection caps to ensure it works end-to-end.
"""

import asyncio
from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment

async def test_experiment_with_reflection_cap():
    """Test running an experiment with reflection character cap."""
    
    # Create experiment with custom reflection cap
    experiment = O3VsHaikuExperiment()
    
    # Test config with a lower reflection cap
    test_config = {
        "m_items": 5,
        "n_agents": 3,
        "t_rounds": 2,  # Just 2 rounds for quick test
        "gamma_discount": 0.9,
        "competition_level": 0.95,
        "known_to_all": False,
        "random_seed": 12345,
        "max_reflection_chars": 1000  # Lower cap for testing
    }
    
    print("Testing experiment with max_reflection_chars = 1000")
    print("This should show reflection truncation messages if agents are verbose")
    print("=" * 60)
    
    try:
        # Run a short experiment
        results = await experiment.run_single_experiment(
            experiment_config=test_config,
            experiment_id="reflection_cap_test"
        )
        
        print("\nExperiment completed successfully!")
        print(f"Consensus reached: {results.consensus_reached}")
        print(f"Final round: {results.final_round}")
        
        # Check reflection lengths in conversation logs
        reflection_messages = [msg for msg in results.conversation_logs 
                             if msg.get("phase") == "individual_reflection"]
        
        print(f"\nReflection messages found: {len(reflection_messages)}")
        
        for msg in reflection_messages:
            reflection_content = msg.get("reflection_content", "")
            char_count = len(reflection_content) if isinstance(reflection_content, str) else 0
            print(f"  {msg.get('from', 'Unknown')}: {char_count} chars")
            if char_count > 1000:
                print(f"    ⚠️  WARNING: Reflection exceeds cap of 1000 chars!")
        
        return results
        
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_experiment_with_reflection_cap())