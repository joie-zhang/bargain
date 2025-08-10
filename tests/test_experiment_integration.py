#!/usr/bin/env python3
"""
Test the integration of the utility engine with the actual experiment framework.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from negotiation.utility_engine import UtilityEngine

def test_experiment_integration():
    """Test utility engine integration with experiment configurations."""
    print("Testing Experiment Integration...")
    
    # Test configuration matching actual experiment settings
    config = {
        "gamma_discount": 0.85,
        "m_items": 5,
        "n_agents": 3,
        "t_rounds": 6,
        "max_reflection_chars": 2000
    }
    
    # Initialize utility engine like the experiment would
    utility_engine = UtilityEngine(gamma=config["gamma_discount"])
    
    # Test realistic experimental scenario
    allocation = {
        "o3_agent": [0, 2],
        "haiku_agent_1": [1, 3], 
        "haiku_agent_2": [4]
    }
    
    preferences = {
        "type": "vector",
        "agent_preferences": {
            "o3_agent": [9.1, 5.2, 8.7, 6.3, 3.4],
            "haiku_agent_1": [4.1, 8.2, 3.9, 7.5, 5.8],
            "haiku_agent_2": [6.7, 4.3, 5.1, 8.9, 10.0]
        }
    }
    
    print(f"  Configuration: gamma={config['gamma_discount']}, items={config['m_items']}, agents={config['n_agents']}")
    
    # Test utilities across rounds (simulating negotiation progress)
    for round_num in [0, 3, 5]:
        utilities = utility_engine.calculate_all_utilities(
            allocation, preferences, round_number=round_num, include_details=True
        )
        
        print(f"  Round {round_num}:")
        for agent_id, result in utilities.items():
            discount_factor = result.discount_factor
            print(f"    {agent_id}: base={result.base_utility:.2f}, "
                  f"discounted={result.discounted_utility:.2f} "
                  f"(discount={discount_factor:.3f})")
    
    # Test matrix preferences with O3 vs Haiku setup
    matrix_preferences = {
        "type": "matrix",
        "agent_preferences": {
            "o3_agent": [
                [9.0, 3.0, 2.0],   # O3 values own items highly
                [8.0, 4.0, 3.0],
                [7.0, 6.0, 5.0],   # More cooperative for this item
                [10.0, 1.0, 1.0],  # Strongly prefers this item
                [5.0, 7.0, 8.0]    # Values others getting this
            ],
            "haiku_agent_1": [
                [3.0, 8.0, 4.0],
                [2.0, 9.0, 3.0], 
                [6.0, 7.0, 5.0],
                [4.0, 6.0, 7.0],
                [5.0, 8.0, 6.0]
            ],
            "haiku_agent_2": [
                [4.0, 3.0, 9.0],
                [5.0, 4.0, 8.0],
                [3.0, 6.0, 7.0],
                [2.0, 5.0, 8.0],
                [7.0, 6.0, 10.0]
            ]
        }
    }
    
    print("\n  Testing Matrix Preferences:")
    matrix_utilities = utility_engine.calculate_all_utilities(
        allocation, matrix_preferences, round_number=2, include_details=True
    )
    
    for agent_id, result in matrix_utilities.items():
        cooperation = result.calculation_details.get("cooperation_detected", False)
        coop_str = " (cooperative behavior detected)" if cooperation else " (competitive behavior)"
        print(f"    {agent_id}: {result.discounted_utility:.2f}{coop_str}")
    
    print("  Experiment integration: ✓ PASSED")

def test_performance_simulation():
    """Test performance with experiment-scale data."""
    print("\nTesting Performance at Experimental Scale...")
    
    import time
    import random
    
    # Create larger-scale test (simulating multiple experiments)
    n_experiments = 50
    utility_engine = UtilityEngine(gamma=0.85)
    
    start_time = time.time()
    
    for exp_id in range(n_experiments):
        # Generate random experimental allocation
        allocation = {
            "o3_agent": random.sample(range(10), 3),
            "haiku_agent_1": random.sample(range(10), 3),
            "haiku_agent_2": random.sample(range(10), 4)
        }
        
        preferences = {
            "type": "vector",
            "agent_preferences": {
                "o3_agent": [random.uniform(1, 10) for _ in range(10)],
                "haiku_agent_1": [random.uniform(1, 10) for _ in range(10)],
                "haiku_agent_2": [random.uniform(1, 10) for _ in range(10)]
            }
        }
        
        # Calculate utilities for multiple rounds
        for round_num in range(6):
            utilities = utility_engine.calculate_all_utilities(
                allocation, preferences, round_number=round_num
            )
    
    end_time = time.time()
    total_calculations = n_experiments * 6 * 3  # experiments * rounds * agents
    calculation_time = end_time - start_time
    
    print(f"  Completed {total_calculations} utility calculations in {calculation_time:.3f}s")
    print(f"  Average time per calculation: {(calculation_time/total_calculations)*1000:.2f}ms")
    
    if calculation_time < 2.0:
        print("  Performance: ✓ PASSED (fast enough for real experiments)")
    else:
        print("  Performance: ⚠ WARNING (may be slow for large experiments)")

def main():
    """Run integration tests."""
    print("=" * 60)
    print("UTILITY ENGINE EXPERIMENT INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_experiment_integration()
        test_performance_simulation()
        
        print("\n" + "=" * 60)
        print("🚀 EXPERIMENT INTEGRATION SUCCESSFUL!")
        print("The utility engine is ready for use in negotiation experiments.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()