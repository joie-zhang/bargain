#!/usr/bin/env python3
"""
Comprehensive tests for the enhanced utility calculation engine.

Tests cover:
- Vector preference utility calculations
- Matrix preference utility calculations  
- Discount factor (gamma) application
- Round-based utility degradation
- Error handling and edge cases
- Performance with large preference matrices
"""

import unittest
import numpy as np
from typing import Dict, List, Any

from negotiation.utility_engine import (
    UtilityEngine,
    UtilityCalculationResult,
    PreferenceType,
    create_utility_engine,
    calculate_discounted_utility,
    compare_discount_factors
)


class TestUtilityEngine(unittest.TestCase):
    """Test the enhanced utility calculation engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = UtilityEngine(gamma=0.9)
        
        # Sample vector preferences
        self.vector_preferences = {
            "type": "vector",
            "agent_preferences": {
                "agent_0": [5.0, 8.0, 3.0, 7.0, 2.0],
                "agent_1": [6.0, 2.0, 9.0, 4.0, 5.0],
                "agent_2": [3.0, 7.0, 4.0, 8.0, 6.0]
            }
        }
        
        # Sample matrix preferences (3 agents, 5 items)
        self.matrix_preferences = {
            "type": "matrix",
            "agent_preferences": {
                "agent_0": [
                    [5.0, 2.0, 1.0],  # Item 0: agent_0 values it at 5, agent_1 at 2, agent_2 at 1
                    [8.0, 3.0, 2.0],  # Item 1
                    [3.0, 4.0, 6.0],  # Item 2
                    [7.0, 1.0, 3.0],  # Item 3
                    [2.0, 5.0, 4.0]   # Item 4
                ],
                "agent_1": [
                    [2.0, 6.0, 3.0],  # Different perspective on item values
                    [1.0, 2.0, 4.0],
                    [8.0, 9.0, 5.0],
                    [3.0, 4.0, 7.0],
                    [4.0, 5.0, 8.0]
                ],
                "agent_2": [
                    [1.0, 4.0, 3.0],
                    [3.0, 6.0, 7.0],
                    [5.0, 2.0, 4.0],
                    [6.0, 8.0, 8.0],
                    [7.0, 3.0, 6.0]
                ]
            }
        }
        
        # Sample allocation
        self.sample_allocation = {
            "agent_0": [0, 1],  # Gets items 0 and 1
            "agent_1": [2, 3],  # Gets items 2 and 3
            "agent_2": [4]      # Gets item 4
        }
    
    def test_vector_utility_calculation_basic(self):
        """Test basic vector utility calculation without discount."""
        engine = UtilityEngine(gamma=1.0)  # No discount
        
        utility = engine.calculate_utility(
            "agent_0", 
            self.sample_allocation, 
            self.vector_preferences, 
            round_number=0
        )
        
        # agent_0 gets items [0, 1] with preferences [5.0, 8.0]
        expected = 5.0 + 8.0
        self.assertAlmostEqual(utility, expected, places=6)
    
    def test_vector_utility_with_discount(self):
        """Test vector utility calculation with discount factor."""
        gamma = 0.8
        engine = UtilityEngine(gamma=gamma)
        
        # Round 0 (no discount)
        utility_round_0 = engine.calculate_utility(
            "agent_0", 
            self.sample_allocation, 
            self.vector_preferences, 
            round_number=0
        )
        
        # Round 2 (with discount)
        utility_round_2 = engine.calculate_utility(
            "agent_0", 
            self.sample_allocation, 
            self.vector_preferences, 
            round_number=2
        )
        
        # Base utility should be 5.0 + 8.0 = 13.0
        base_utility = 13.0
        expected_round_0 = base_utility * (gamma ** 0)  # = 13.0
        expected_round_2 = base_utility * (gamma ** 2)  # = 13.0 * 0.64 = 8.32
        
        self.assertAlmostEqual(utility_round_0, expected_round_0, places=6)
        self.assertAlmostEqual(utility_round_2, expected_round_2, places=6)
    
    def test_matrix_utility_calculation_basic(self):
        """Test basic matrix utility calculation."""
        engine = UtilityEngine(gamma=1.0)  # No discount
        
        utility = engine.calculate_utility(
            "agent_0",
            self.sample_allocation,
            self.matrix_preferences,
            round_number=0
        )
        
        # For agent_0's matrix calculation:
        # agent_0 gets [0, 1]: matrix[0][0]=5.0 + matrix[1][0]=8.0 = 13.0
        # agent_1 gets [2, 3]: matrix[2][1]=4.0 + matrix[3][1]=1.0 = 5.0
        # agent_2 gets [4]: matrix[4][2]=4.0 = 4.0
        expected = 5.0 + 8.0 + 4.0 + 1.0 + 4.0  # 22.0 total
        
        self.assertAlmostEqual(utility, expected, places=6)
    
    def test_matrix_utility_cooperation_vs_competition(self):
        """Test that matrix utilities capture cooperative vs competitive preferences."""
        # Create a cooperative matrix where agent_0 benefits when others get items
        cooperative_prefs = {
            "type": "matrix",
            "agent_preferences": {
                "agent_0": [
                    [5.0, 8.0, 6.0],  # Agent 0 values others getting item 0 highly
                    [3.0, 9.0, 7.0],  # Agent 0 values others getting item 1 highly
                    [2.0, 4.0, 8.0],  # Agent 0 values agent_2 getting item 2 highly
                    [7.0, 1.0, 2.0],  # Agent 0 prefers to keep item 3
                    [4.0, 3.0, 1.0]   # Agent 0 prefers to keep item 4
                ]
            }
        }
        
        # Allocation where others get the high-cooperation items
        coop_allocation = {
            "agent_0": [3, 4],  # Gets items agent_0 wants to keep
            "agent_1": [0, 1],  # Gets items agent_0 wants others to have
            "agent_2": [2]      # Gets item agent_0 wants agent_2 to have
        }
        
        utility = self.engine.calculate_utility(
            "agent_0",
            coop_allocation,
            cooperative_prefs,
            round_number=0
        )
        
        # Should be: 7.0 + 4.0 (own items) + 8.0 + 9.0 (agent_1's items) + 8.0 (agent_2's item)
        expected = 7.0 + 4.0 + 8.0 + 9.0 + 8.0
        self.assertAlmostEqual(utility, expected, places=6)
    
    def test_detailed_utility_breakdown(self):
        """Test detailed utility calculation results."""
        result = self.engine.calculate_utility(
            "agent_1",
            self.sample_allocation,
            self.vector_preferences,
            round_number=1,
            include_details=True
        )
        
        self.assertIsInstance(result, UtilityCalculationResult)
        self.assertEqual(result.agent_id, "agent_1")
        self.assertEqual(result.preference_type, PreferenceType.VECTOR)
        self.assertEqual(result.round_number, 1)
        self.assertEqual(result.allocated_items, [2, 3])
        
        # Check discount factor calculation
        expected_discount = self.engine.gamma ** 1
        self.assertAlmostEqual(result.discount_factor, expected_discount, places=6)
        
        # Check utilities (agent_1 gets items [2, 3] with preferences [9.0, 4.0])
        expected_base = 9.0 + 4.0
        expected_discounted = expected_base * expected_discount
        self.assertAlmostEqual(result.base_utility, expected_base, places=6)
        self.assertAlmostEqual(result.discounted_utility, expected_discounted, places=6)
    
    def test_calculate_all_utilities(self):
        """Test calculating utilities for all agents simultaneously."""
        utilities = self.engine.calculate_all_utilities(
            self.sample_allocation,
            self.vector_preferences,
            round_number=2
        )
        
        self.assertEqual(len(utilities), 3)
        self.assertIn("agent_0", utilities)
        self.assertIn("agent_1", utilities)
        self.assertIn("agent_2", utilities)
        
        # Verify discount is applied
        discount_factor = self.engine.gamma ** 2
        
        # agent_0: items [0,1] -> 5.0 + 8.0 = 13.0
        expected_agent_0 = 13.0 * discount_factor
        self.assertAlmostEqual(utilities["agent_0"], expected_agent_0, places=6)
        
        # agent_1: items [2,3] -> 9.0 + 4.0 = 13.0
        expected_agent_1 = 13.0 * discount_factor
        self.assertAlmostEqual(utilities["agent_1"], expected_agent_1, places=6)
        
        # agent_2: item [4] -> 6.0
        expected_agent_2 = 6.0 * discount_factor
        self.assertAlmostEqual(utilities["agent_2"], expected_agent_2, places=6)
    
    def test_empty_allocation(self):
        """Test utility calculation with empty allocation."""
        empty_allocation = {
            "agent_0": [],
            "agent_1": [],
            "agent_2": []
        }
        
        utilities = self.engine.calculate_all_utilities(
            empty_allocation,
            self.vector_preferences,
            round_number=0
        )
        
        for agent_id, utility in utilities.items():
            self.assertEqual(utility, 0.0, f"Agent {agent_id} should have zero utility with no items")
    
    def test_missing_agent_graceful_handling(self):
        """Test graceful handling of missing agents in allocation."""
        partial_allocation = {
            "agent_0": [0, 1, 2],
            # agent_1 and agent_2 missing - should get zero utility
        }
        
        utilities = self.engine.calculate_all_utilities(
            partial_allocation,
            self.vector_preferences,
            round_number=0
        )
        
        self.assertIn("agent_0", utilities)
        self.assertIn("agent_1", utilities)
        self.assertIn("agent_2", utilities)
        
        # agent_0 should have utility from items
        self.assertGreater(utilities["agent_0"], 0)
        
        # Missing agents should have zero utility
        self.assertEqual(utilities["agent_1"], 0.0)
        self.assertEqual(utilities["agent_2"], 0.0)
    
    def test_gamma_boundary_values(self):
        """Test utility calculation with boundary gamma values."""
        # Test gamma = 0 (immediate utility only)
        engine_zero = UtilityEngine(gamma=0.0)
        utility_zero = engine_zero.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, round_number=5
        )
        self.assertEqual(utility_zero, 0.0, "With gamma=0, any future round should give zero utility")
        
        # Test gamma = 1 (no discount)
        engine_one = UtilityEngine(gamma=1.0)
        utility_one_r0 = engine_one.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, round_number=0
        )
        utility_one_r10 = engine_one.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, round_number=10
        )
        self.assertEqual(utility_one_r0, utility_one_r10, "With gamma=1, utility should not change with rounds")
    
    def test_preference_type_detection(self):
        """Test automatic preference type detection."""
        # Test vector detection
        vector_utility = self.engine.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, include_details=True
        )
        self.assertEqual(vector_utility.preference_type, PreferenceType.VECTOR)
        
        # Test matrix detection
        matrix_utility = self.engine.calculate_utility(
            "agent_0", self.sample_allocation, self.matrix_preferences, include_details=True
        )
        self.assertEqual(matrix_utility.preference_type, PreferenceType.MATRIX)
    
    def test_agent_id_pattern_matching(self):
        """Test various agent ID naming patterns."""
        patterns_allocation = {
            "o3_agent": [0, 1],
            "haiku_agent_1": [2, 3],
            "haiku_agent_2": [4]
        }
        
        patterns_prefs = {
            "type": "vector",
            "agent_preferences": {
                "o3_agent": [5.0, 8.0, 3.0, 7.0, 2.0],
                "haiku_agent_1": [6.0, 2.0, 9.0, 4.0, 5.0],
                "haiku_agent_2": [3.0, 7.0, 4.0, 8.0, 6.0]
            }
        }
        
        utilities = self.engine.calculate_all_utilities(
            patterns_allocation, patterns_prefs, round_number=0
        )
        
        self.assertEqual(len(utilities), 3)
        self.assertIn("o3_agent", utilities)
        self.assertIn("haiku_agent_1", utilities)
        self.assertIn("haiku_agent_2", utilities)
        
        # Verify calculations are correct
        self.assertAlmostEqual(utilities["o3_agent"], 5.0 + 8.0, places=6)
        self.assertAlmostEqual(utilities["haiku_agent_1"], 9.0 + 4.0, places=6)
        self.assertAlmostEqual(utilities["haiku_agent_2"], 6.0, places=6)
    
    def test_utility_breakdown_analysis(self):
        """Test detailed utility breakdown for analysis."""
        breakdown = self.engine.get_utility_breakdown(
            "agent_0",
            self.sample_allocation,
            self.vector_preferences,
            round_number=3
        )
        
        required_keys = [
            "agent_id", "base_utility", "discounted_utility", "discount_factor",
            "round_number", "preference_type", "allocated_items", "gamma"
        ]
        
        for key in required_keys:
            self.assertIn(key, breakdown, f"Missing key: {key}")
        
        # Verify values
        self.assertEqual(breakdown["agent_id"], "agent_0")
        self.assertEqual(breakdown["round_number"], 3)
        self.assertEqual(breakdown["preference_type"], "vector")
        self.assertEqual(breakdown["gamma"], self.engine.gamma)
        self.assertEqual(breakdown["allocated_items"], [0, 1])
    
    def test_allocation_comparison(self):
        """Test comparing multiple allocations for strategy analysis."""
        allocations = [
            {"agent_0": [0, 1], "agent_1": [2, 3], "agent_2": [4]},  # Original
            {"agent_0": [0, 2], "agent_1": [1, 3], "agent_2": [4]},  # Swap items 1&2
            {"agent_0": [0], "agent_1": [1, 2, 3], "agent_2": [4]},  # Agent_1 gets more
        ]
        
        comparison = self.engine.compare_allocations(
            "agent_0", allocations, self.vector_preferences, round_number=1
        )
        
        self.assertEqual(comparison["agent_id"], "agent_0")
        self.assertEqual(comparison["num_allocations"], 3)
        self.assertIn("best_allocation", comparison)
        self.assertIn("worst_allocation", comparison)
        self.assertIn("all_comparisons", comparison)
        
        # Verify sorting (best utility first)
        comparisons = comparison["all_comparisons"]
        for i in range(len(comparisons) - 1):
            self.assertGreaterEqual(
                comparisons[i]["utility"], 
                comparisons[i + 1]["utility"],
                "Allocations should be sorted by utility (descending)"
            )
    
    def test_performance_large_matrices(self):
        """Test performance with larger preference matrices."""
        # Create large matrix (10 agents, 20 items)
        n_agents, n_items = 10, 20
        
        large_matrix_prefs = {
            "type": "matrix",
            "agent_preferences": {}
        }
        
        for agent_idx in range(n_agents):
            agent_id = f"agent_{agent_idx}"
            # Random matrix for each agent
            matrix = np.random.uniform(0, 10, (n_items, n_agents)).tolist()
            large_matrix_prefs["agent_preferences"][agent_id] = matrix
        
        # Create allocation giving each agent 2 items
        large_allocation = {}
        item_counter = 0
        for agent_idx in range(n_agents):
            agent_id = f"agent_{agent_idx}"
            large_allocation[agent_id] = [item_counter, item_counter + 1]
            item_counter += 2
        
        # Time the calculation (should be fast)
        import time
        start_time = time.time()
        
        utilities = self.engine.calculate_all_utilities(
            large_allocation, large_matrix_prefs, round_number=2
        )
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete quickly (under 1 second for this size)
        self.assertLess(calculation_time, 1.0, "Large matrix calculation took too long")
        self.assertEqual(len(utilities), n_agents, "Should calculate utility for all agents")
    
    def test_caching_behavior(self):
        """Test utility calculation caching."""
        # Enable caching
        self.engine.set_cache_enabled(True)
        
        # First calculation should populate cache
        utility1 = self.engine.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, round_number=1
        )
        
        cache_stats = self.engine.get_cache_stats()
        self.assertEqual(cache_stats["cache_size"], 1, "Cache should contain one entry")
        
        # Second identical calculation should use cache
        utility2 = self.engine.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, round_number=1
        )
        
        self.assertEqual(utility1, utility2, "Cached result should match original")
        
        # Different parameters should create new cache entry
        utility3 = self.engine.calculate_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, round_number=2
        )
        
        cache_stats_after = self.engine.get_cache_stats()
        self.assertEqual(cache_stats_after["cache_size"], 2, "Cache should contain two entries")
        
        # Clear cache
        self.engine.clear_cache()
        final_stats = self.engine.get_cache_stats()
        self.assertEqual(final_stats["cache_size"], 0, "Cache should be empty after clearing")
    
    def test_convenience_functions(self):
        """Test convenience functions for utility calculation."""
        # Test create_utility_engine
        engine = create_utility_engine(gamma=0.7)
        self.assertEqual(engine.gamma, 0.7)
        
        # Test calculate_discounted_utility
        utility = calculate_discounted_utility(
            "agent_0", self.sample_allocation, self.vector_preferences, 
            gamma=0.8, round_number=2
        )
        
        expected = (5.0 + 8.0) * (0.8 ** 2)
        self.assertAlmostEqual(utility, expected, places=6)
        
        # Test compare_discount_factors
        gammas = [0.5, 0.8, 0.9, 1.0]
        comparison = compare_discount_factors(
            "agent_0", self.sample_allocation, self.vector_preferences,
            gammas, round_number=3
        )
        
        self.assertEqual(len(comparison), len(gammas))
        base_utility = 5.0 + 8.0
        
        for gamma, utility in comparison.items():
            expected = base_utility * (gamma ** 3)
            self.assertAlmostEqual(utility, expected, places=6, 
                                 msg=f"Gamma {gamma} calculation incorrect")
    
    def test_error_handling(self):
        """Test error handling in edge cases."""
        # Test invalid gamma values
        with self.assertRaises(ValueError):
            UtilityEngine(gamma=-0.1)  # Negative gamma
        
        with self.assertRaises(ValueError):
            UtilityEngine(gamma=1.1)   # Gamma > 1
        
        # Test malformed preferences
        bad_prefs = {
            "type": "vector",
            "agent_preferences": {
                "agent_0": "not_a_list"  # Invalid preference format
            }
        }
        
        # Should handle gracefully and return zero utility
        utilities = self.engine.calculate_all_utilities(
            {"agent_0": [0]}, bad_prefs, round_number=0
        )
        self.assertEqual(utilities["agent_0"], 0.0, "Should handle invalid preferences gracefully")


class TestUtilityEngineIntegration(unittest.TestCase):
    """Integration tests for the utility engine with negotiation environment."""
    
    def test_integration_with_experiment_config(self):
        """Test integration with typical experiment configurations."""
        # Simulate O3 vs Haiku experiment setup
        config = {
            "gamma_discount": 0.85,
            "m_items": 5,
            "n_agents": 3,
            "t_rounds": 6
        }
        
        engine = UtilityEngine(gamma=config["gamma_discount"])
        
        # Typical experimental allocation
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
        
        # Test across different rounds to verify discount application
        utilities_round_0 = engine.calculate_all_utilities(
            allocation, preferences, round_number=0
        )
        utilities_round_3 = engine.calculate_all_utilities(
            allocation, preferences, round_number=3
        )
        utilities_round_5 = engine.calculate_all_utilities(
            allocation, preferences, round_number=5
        )
        
        # Verify utilities decrease with later rounds
        for agent_id in allocation.keys():
            self.assertGreaterEqual(
                utilities_round_0[agent_id], utilities_round_3[agent_id],
                f"Round 0 utility should be >= round 3 for {agent_id}"
            )
            self.assertGreaterEqual(
                utilities_round_3[agent_id], utilities_round_5[agent_id],
                f"Round 3 utility should be >= round 5 for {agent_id}"
            )
        
        # Verify discount factors are applied correctly
        discount_r3 = config["gamma_discount"] ** 3
        discount_r5 = config["gamma_discount"] ** 5
        
        for agent_id in allocation.keys():
            expected_r3 = utilities_round_0[agent_id] * discount_r3
            expected_r5 = utilities_round_0[agent_id] * discount_r5
            
            self.assertAlmostEqual(utilities_round_3[agent_id], expected_r3, places=5)
            self.assertAlmostEqual(utilities_round_5[agent_id], expected_r5, places=5)


if __name__ == '__main__':
    unittest.main()