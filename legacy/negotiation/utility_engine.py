#!/usr/bin/env python3
"""
Enhanced Utility Calculation Engine for Multi-Agent Negotiation

This module provides comprehensive utility calculations supporting:
- Vector preferences (competitive scenarios)
- Matrix preferences (cooperative + competitive scenarios) 
- Discount factors (gamma) for time-sensitive negotiations
- Round-based utility degradation

Key formulas:
- Vector utility: Σ(preference_i × item_received_i) × γ^round
- Matrix utility: Σ(preference_matrix[i,j] × item_received_by_agent_j) × γ^round
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging


class PreferenceType(Enum):
    """Types of preference systems supported."""
    VECTOR = "vector"
    MATRIX = "matrix"


@dataclass
class UtilityCalculationResult:
    """Result of a utility calculation with detailed breakdown."""
    agent_id: str
    base_utility: float
    discounted_utility: float
    discount_factor: float
    round_number: int
    preference_type: PreferenceType
    allocated_items: List[int]
    calculation_details: Dict[str, Any]


class UtilityEngine:
    """
    Enhanced utility calculation engine with discount factor support.
    
    Supports both vector and matrix preference systems with time-based 
    utility degradation through gamma discount factor.
    """
    
    def __init__(self, gamma: float = 1.0, logger: Optional[logging.Logger] = None):
        """
        Initialize the utility engine.
        
        Args:
            gamma: Discount factor (0.0 to 1.0). 1.0 = no discounting, 0.0 = immediate utility only
            logger: Optional logger for debugging
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"Gamma must be between 0.0 and 1.0, got {gamma}")
        
        self.gamma = gamma
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache for expensive calculations
        self._utility_cache = {}
        self._cache_enabled = True
    
    def calculate_utility(
        self,
        agent_id: str,
        allocation: Dict[str, List[int]],
        preferences: Dict[str, Any],
        round_number: int = 0,
        include_details: bool = False
    ) -> Union[float, UtilityCalculationResult]:
        """
        Calculate utility for an agent with discount factor applied.
        
        Args:
            agent_id: ID of the agent to calculate utility for
            allocation: Dict mapping agent IDs to lists of item indices they receive
            preferences: Preference data structure (vector or matrix format)
            round_number: Current negotiation round (0-based, affects discounting)
            include_details: If True, return detailed UtilityCalculationResult
            
        Returns:
            Float utility value or detailed UtilityCalculationResult object
        """
        # Create cache key for expensive calculations
        cache_key = None
        if self._cache_enabled:
            cache_key = self._create_cache_key(agent_id, allocation, preferences, round_number)
            if cache_key in self._utility_cache:
                cached_result = self._utility_cache[cache_key]
                return cached_result if include_details else cached_result.discounted_utility
        
        # Determine preference type
        pref_type = self._detect_preference_type(preferences)
        
        # Calculate base utility based on preference type
        if pref_type == PreferenceType.VECTOR:
            base_utility, details = self._calculate_vector_utility(agent_id, allocation, preferences)
        elif pref_type == PreferenceType.MATRIX:
            base_utility, details = self._calculate_matrix_utility(agent_id, allocation, preferences)
        else:
            raise ValueError(f"Unsupported preference type: {pref_type}")
        
        # Apply discount factor
        discount_factor = self.gamma ** round_number
        discounted_utility = base_utility * discount_factor
        
        # Get allocated items for this agent
        allocated_items = allocation.get(agent_id, [])
        
        # Create detailed result
        result = UtilityCalculationResult(
            agent_id=agent_id,
            base_utility=base_utility,
            discounted_utility=discounted_utility,
            discount_factor=discount_factor,
            round_number=round_number,
            preference_type=pref_type,
            allocated_items=allocated_items,
            calculation_details=details
        )
        
        # Cache the result
        if self._cache_enabled and cache_key:
            self._utility_cache[cache_key] = result
        
        # Log calculation for debugging
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Utility calculation: {agent_id} | "
                f"Base: {base_utility:.3f} | "
                f"Discounted: {discounted_utility:.3f} | "
                f"Round: {round_number} | "
                f"Gamma: {self.gamma}"
            )
        
        return result if include_details else discounted_utility
    
    def calculate_all_utilities(
        self,
        allocation: Dict[str, List[int]],
        preferences: Dict[str, Any],
        round_number: int = 0,
        include_details: bool = False
    ) -> Dict[str, Union[float, UtilityCalculationResult]]:
        """
        Calculate utilities for all agents in the allocation.
        
        Args:
            allocation: Dict mapping agent IDs to lists of item indices
            preferences: Preference data structure
            round_number: Current negotiation round
            include_details: If True, return detailed results
            
        Returns:
            Dict mapping agent IDs to utility values or detailed results
        """
        utilities = {}
        
        # Extract all agent IDs from both allocation and preferences
        agent_ids = set(allocation.keys())
        if "agent_preferences" in preferences:
            agent_ids.update(preferences["agent_preferences"].keys())
        
        for agent_id in agent_ids:
            try:
                utilities[agent_id] = self.calculate_utility(
                    agent_id, allocation, preferences, round_number, include_details
                )
            except Exception as e:
                self.logger.error(f"Error calculating utility for {agent_id}: {e}")
                # Return zero utility for failed calculations
                if include_details:
                    utilities[agent_id] = UtilityCalculationResult(
                        agent_id=agent_id,
                        base_utility=0.0,
                        discounted_utility=0.0,
                        discount_factor=self.gamma ** round_number,
                        round_number=round_number,
                        preference_type=PreferenceType.VECTOR,  # Default
                        allocated_items=allocation.get(agent_id, []),
                        calculation_details={"error": str(e)}
                    )
                else:
                    utilities[agent_id] = 0.0
        
        return utilities
    
    def _calculate_vector_utility(
        self, 
        agent_id: str, 
        allocation: Dict[str, List[int]], 
        preferences: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate utility for vector preferences.
        
        Formula: Σ(preference_i × item_received_i)
        """
        agent_prefs = preferences["agent_preferences"][agent_id]
        allocated_items = allocation.get(agent_id, [])
        
        # Handle both list and dict preference formats
        if isinstance(agent_prefs, list):
            utility = sum(agent_prefs[item_id] for item_id in allocated_items 
                         if 0 <= item_id < len(agent_prefs))
            max_possible = sum(agent_prefs)
        elif isinstance(agent_prefs, dict):
            utility = sum(agent_prefs.get(item_id, 0.0) for item_id in allocated_items)
            max_possible = sum(agent_prefs.values())
        else:
            raise ValueError(f"Invalid preference format for {agent_id}: {type(agent_prefs)}")
        
        details = {
            "preference_values": [agent_prefs[item_id] if isinstance(agent_prefs, list) 
                                else agent_prefs.get(item_id, 0.0) for item_id in allocated_items],
            "max_possible_utility": max_possible,
            "utility_ratio": utility / max_possible if max_possible > 0 else 0.0,
            "items_breakdown": {item_id: agent_prefs[item_id] if isinstance(agent_prefs, list)
                              else agent_prefs.get(item_id, 0.0) for item_id in allocated_items}
        }
        
        return float(utility), details
    
    def _calculate_matrix_utility(
        self, 
        agent_id: str, 
        allocation: Dict[str, List[int]], 
        preferences: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate utility for matrix preferences.
        
        Formula: Σ(preference_matrix[i,j] × item_i_received_by_agent_j)
        
        This allows agents to care about who gets what items, enabling
        both competitive and cooperative behaviors.
        """
        agent_matrix = preferences["agent_preferences"][agent_id]
        
        # Convert to numpy array if needed
        if not isinstance(agent_matrix, np.ndarray):
            agent_matrix = np.array(agent_matrix)
        
        utility = 0.0
        allocation_details = {}
        
        # For each receiving agent and their items
        for receiving_agent, items in allocation.items():
            # Extract agent index - handle different naming conventions
            agent_idx = self._extract_agent_index(receiving_agent, preferences)
            
            if agent_idx is not None and agent_idx < agent_matrix.shape[1]:
                agent_contribution = 0.0
                
                for item_id in items:
                    if 0 <= item_id < agent_matrix.shape[0]:
                        item_value = agent_matrix[item_id, agent_idx]
                        utility += item_value
                        agent_contribution += item_value
                
                allocation_details[receiving_agent] = {
                    "agent_index": agent_idx,
                    "items": items,
                    "contribution": agent_contribution
                }
        
        # Calculate maximum possible utility (if agent got all items)
        max_possible = np.sum(agent_matrix[:, self._extract_agent_index(agent_id, preferences)])
        
        details = {
            "allocation_breakdown": allocation_details,
            "max_possible_utility": max_possible,
            "utility_ratio": utility / max_possible if max_possible > 0 else 0.0,
            "matrix_shape": agent_matrix.shape,
            "cooperation_detected": self._detect_cooperation_in_allocation(
                agent_id, allocation, agent_matrix, preferences
            )
        }
        
        return float(utility), details
    
    def _detect_preference_type(self, preferences: Dict[str, Any]) -> PreferenceType:
        """Detect whether preferences are vector or matrix type."""
        if "type" in preferences:
            if preferences["type"] == "vector":
                return PreferenceType.VECTOR
            elif preferences["type"] == "matrix":
                return PreferenceType.MATRIX
        
        # Infer from structure if type not explicitly specified
        if "agent_preferences" in preferences:
            sample_agent = next(iter(preferences["agent_preferences"].values()))
            
            if isinstance(sample_agent, list) and all(isinstance(x, (int, float)) for x in sample_agent):
                return PreferenceType.VECTOR
            elif isinstance(sample_agent, list) and len(sample_agent) > 0 and isinstance(sample_agent[0], list):
                return PreferenceType.MATRIX
            elif isinstance(sample_agent, np.ndarray) and len(sample_agent.shape) == 2:
                return PreferenceType.MATRIX
        
        # Default fallback
        return PreferenceType.VECTOR
    
    def _extract_agent_index(self, agent_id: str, preferences: Dict[str, Any]) -> Optional[int]:
        """Extract numeric agent index from agent ID string."""
        # Try various patterns: agent_0, agent0, haiku_agent_1, o3_agent, etc.
        import re
        
        # Pattern 1: agent_N or agentN
        match = re.search(r'agent_?(\d+)', agent_id)
        if match:
            return int(match.group(1))
        
        # Pattern 2: Look up in preferences mapping if available
        if "agent_mapping" in preferences:
            return preferences["agent_mapping"].get(agent_id)
        
        # Pattern 3: Use order in preferences dict
        if "agent_preferences" in preferences:
            agent_list = list(preferences["agent_preferences"].keys())
            try:
                return agent_list.index(agent_id)
            except ValueError:
                pass
        
        # Pattern 4: Default heuristic for common patterns
        if "o3" in agent_id.lower():
            return 0  # Often the first agent
        elif "haiku" in agent_id.lower():
            if "1" in agent_id:
                return 1
            elif "2" in agent_id:
                return 2
        
        return None
    
    def _detect_cooperation_in_allocation(
        self, 
        agent_id: str, 
        allocation: Dict[str, List[int]], 
        agent_matrix: np.ndarray,
        preferences: Dict[str, Any]
    ) -> bool:
        """Detect if the agent's preferences show cooperative tendencies."""
        agent_idx = self._extract_agent_index(agent_id, preferences)
        if agent_idx is None:
            return False
        
        # Check if agent values items going to other agents positively
        cooperative_value = 0.0
        selfish_value = 0.0
        
        for receiving_agent, items in allocation.items():
            receiving_idx = self._extract_agent_index(receiving_agent, preferences)
            if receiving_idx is not None:
                for item_id in items:
                    if 0 <= item_id < agent_matrix.shape[0]:
                        value = agent_matrix[item_id, receiving_idx]
                        if receiving_idx == agent_idx:
                            selfish_value += value
                        else:
                            cooperative_value += value
        
        # Cooperation detected if agent values others' items positively
        return cooperative_value > 0 and cooperative_value > selfish_value * 0.1
    
    def _create_cache_key(
        self, 
        agent_id: str, 
        allocation: Dict[str, List[int]], 
        preferences: Dict[str, Any],
        round_number: int
    ) -> str:
        """Create a cache key for utility calculations."""
        # Sort allocation for consistent caching
        sorted_allocation = tuple(sorted((k, tuple(sorted(v))) for k, v in allocation.items()))
        pref_hash = hash(str(sorted(preferences.get("agent_preferences", {}).items())))
        
        return f"{agent_id}_{sorted_allocation}_{pref_hash}_{round_number}_{self.gamma}"
    
    def get_utility_breakdown(
        self,
        agent_id: str,
        allocation: Dict[str, List[int]],
        preferences: Dict[str, Any],
        round_number: int = 0
    ) -> Dict[str, Any]:
        """
        Get detailed breakdown of utility calculation for analysis.
        
        Returns comprehensive information about how utility was calculated,
        useful for debugging and research analysis.
        """
        result = self.calculate_utility(
            agent_id, allocation, preferences, round_number, include_details=True
        )
        
        if not isinstance(result, UtilityCalculationResult):
            raise ValueError("Expected UtilityCalculationResult but got float")
        
        breakdown = {
            "agent_id": result.agent_id,
            "base_utility": result.base_utility,
            "discounted_utility": result.discounted_utility,
            "discount_factor": result.discount_factor,
            "discount_applied": abs(result.discount_factor - 1.0) > 1e-6,
            "round_number": result.round_number,
            "preference_type": result.preference_type.value,
            "allocated_items": result.allocated_items,
            "num_items_received": len(result.allocated_items),
            "gamma": self.gamma,
            **result.calculation_details
        }
        
        return breakdown
    
    def compare_allocations(
        self,
        agent_id: str,
        allocations: List[Dict[str, List[int]]],
        preferences: Dict[str, Any],
        round_number: int = 0
    ) -> Dict[str, Any]:
        """
        Compare multiple allocations for a single agent.
        
        Useful for strategy analysis and proposal evaluation.
        """
        comparisons = []
        
        for i, allocation in enumerate(allocations):
            utility = self.calculate_utility(agent_id, allocation, preferences, round_number)
            comparisons.append({
                "allocation_id": i,
                "allocation": allocation,
                "utility": utility,
                "items_received": allocation.get(agent_id, [])
            })
        
        # Sort by utility (descending)
        comparisons.sort(key=lambda x: x["utility"], reverse=True)
        
        return {
            "agent_id": agent_id,
            "num_allocations": len(allocations),
            "best_allocation": comparisons[0] if comparisons else None,
            "worst_allocation": comparisons[-1] if comparisons else None,
            "utility_range": (comparisons[-1]["utility"], comparisons[0]["utility"]) if comparisons else (0, 0),
            "all_comparisons": comparisons
        }
    
    def set_cache_enabled(self, enabled: bool) -> None:
        """Enable or disable utility calculation caching."""
        self._cache_enabled = enabled
        if not enabled:
            self._utility_cache.clear()
    
    def clear_cache(self) -> None:
        """Clear the utility calculation cache."""
        self._utility_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the utility calculation cache."""
        return {
            "cache_enabled": self._cache_enabled,
            "cache_size": len(self._utility_cache),
            "gamma": self.gamma
        }


# Convenience functions for common use cases

def create_utility_engine(gamma: float = 1.0) -> UtilityEngine:
    """Create a utility engine with specified discount factor."""
    return UtilityEngine(gamma=gamma)


def calculate_discounted_utility(
    agent_id: str,
    allocation: Dict[str, List[int]],
    preferences: Dict[str, Any],
    gamma: float,
    round_number: int = 0
) -> float:
    """
    Quick utility calculation with discount factor.
    
    Convenience function for one-off calculations.
    """
    engine = UtilityEngine(gamma=gamma)
    return engine.calculate_utility(agent_id, allocation, preferences, round_number)


def compare_discount_factors(
    agent_id: str,
    allocation: Dict[str, List[int]],
    preferences: Dict[str, Any],
    gammas: List[float],
    round_number: int = 0
) -> Dict[float, float]:
    """
    Compare utilities under different discount factors.
    
    Useful for sensitivity analysis.
    """
    results = {}
    for gamma in gammas:
        engine = UtilityEngine(gamma=gamma)
        utility = engine.calculate_utility(agent_id, allocation, preferences, round_number)
        results[gamma] = utility
    
    return results