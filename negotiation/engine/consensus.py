"""
Consensus tracking and utility calculation for the negotiation engine.

This module handles consensus detection, vote tabulation, and utility calculation
for different preference systems (vector and matrix based).
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple

from .base import NegotiationEngineConfig


class ConsensusTracker:
    """Tracks consensus and voting outcomes during negotiation."""
    
    def __init__(self, config: NegotiationEngineConfig):
        """Initialize consensus tracker with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def check_consensus(self, 
                       votes: Dict[str, Dict[str, str]], 
                       proposals: List[Dict], 
                       config: NegotiationEngineConfig) -> Tuple[bool, Optional[Dict]]:
        """
        Check if consensus has been reached based on votes.
        
        Args:
            votes: Dict mapping agent_id to their votes on proposals
            proposals: List of proposals being voted on
            config: Negotiation configuration
            
        Returns:
            Tuple of (consensus_reached: bool, winning_proposal: Optional[Dict])
        """
        if not votes or not proposals:
            return False, None
        
        # Check for unanimous approval on any proposal
        unanimous_proposals = []
        
        for i, proposal in enumerate(proposals):
            proposal_id = f"proposal_{i}"
            
            # Count votes for this proposal
            approval_count = 0
            total_votes = len(votes)
            
            for agent_id, agent_votes in votes.items():
                vote = agent_votes.get(proposal_id, "abstain")
                if vote == "approve" or vote == "accept":
                    approval_count += 1
            
            # Check if unanimous (all agents approved)
            if config.require_unanimous_consensus:
                if approval_count == total_votes:
                    unanimous_proposals.append({
                        "proposal_index": i,
                        "proposal": proposal,
                        "approval_count": approval_count,
                        "total_votes": total_votes
                    })
            else:
                # Simple majority or other consensus mechanism
                majority_threshold = total_votes // 2 + 1
                if approval_count >= majority_threshold:
                    unanimous_proposals.append({
                        "proposal_index": i,
                        "proposal": proposal,
                        "approval_count": approval_count,
                        "total_votes": total_votes
                    })
        
        # Return consensus result
        if unanimous_proposals:
            winning_proposal = self.get_winning_proposal(votes, unanimous_proposals)
            return True, winning_proposal
        else:
            return False, None
    
    def get_winning_proposal(self, 
                           votes: Dict[str, Dict[str, str]], 
                           unanimous_proposals: List[Dict]) -> Optional[Dict]:
        """
        Determine winning proposal from unanimous proposals.
        
        Args:
            votes: Vote data
            unanimous_proposals: List of proposals that received unanimous approval
            
        Returns:
            The winning proposal, or None if no winner can be determined
        """
        if not unanimous_proposals:
            return None
        
        if len(unanimous_proposals) == 1:
            # Single unanimous proposal
            self.logger.info(f"Single unanimous proposal selected")
            return unanimous_proposals[0]["proposal"]
        else:
            # Multiple unanimous proposals - use tiebreaker
            selected = self._resolve_tie(unanimous_proposals)
            self.logger.info(f"Multiple unanimous proposals - selected via tiebreaker")
            return selected["proposal"]
    
    def _resolve_tie(self, unanimous_proposals: List[Dict]) -> Dict:
        """
        Resolve ties between multiple unanimous proposals.
        
        Args:
            unanimous_proposals: List of proposals with equal consensus
            
        Returns:
            Selected proposal from the tied options
        """
        # For now, use random selection
        # Could be extended with more sophisticated tie-breaking rules
        selected = random.choice(unanimous_proposals)
        self.logger.info(f"Tie resolved randomly: selected proposal {selected['proposal_index']}")
        return selected
    
    def analyze_vote_patterns(self, votes: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze voting patterns for strategic behavior indicators.
        
        Args:
            votes: Vote data from all agents
            
        Returns:
            Dict containing voting analysis
        """
        if not votes:
            return {"error": "No votes to analyze"}
        
        analysis = {
            "total_voters": len(votes),
            "voting_patterns": {},
            "strategic_indicators": {}
        }
        
        # Analyze individual voting patterns
        for agent_id, agent_votes in votes.items():
            approvals = sum(1 for vote in agent_votes.values() if vote in ["approve", "accept"])
            rejections = sum(1 for vote in agent_votes.values() if vote in ["reject", "deny"])
            abstentions = sum(1 for vote in agent_votes.values() if vote == "abstain")
            
            analysis["voting_patterns"][agent_id] = {
                "approvals": approvals,
                "rejections": rejections,
                "abstentions": abstentions,
                "approval_rate": approvals / len(agent_votes) if agent_votes else 0
            }
        
        # Identify strategic indicators
        approval_rates = [pattern["approval_rate"] for pattern in analysis["voting_patterns"].values()]
        
        analysis["strategic_indicators"] = {
            "mean_approval_rate": sum(approval_rates) / len(approval_rates) if approval_rates else 0,
            "approval_rate_variance": self._calculate_variance(approval_rates),
            "potential_strategic_voting": any(rate < 0.2 or rate > 0.8 for rate in approval_rates)
        }
        
        return analysis
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)


class UtilityCalculator:
    """Calculates utilities based on different preference systems."""
    
    def __init__(self, config: NegotiationEngineConfig):
        """Initialize utility calculator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_final_utilities(self, 
                                final_allocation: Dict[str, List[int]],
                                preferences: Dict[str, Any],
                                config: NegotiationEngineConfig,
                                round_number: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate utilities for all agents based on final allocation.
        
        Args:
            final_allocation: Dict mapping agent_id to list of item indices they received
            preferences: Preference specifications (vector or matrix based)
            config: Negotiation configuration
            round_number: Round when consensus was reached (for discounting)
            
        Returns:
            Dict mapping agent_id to their calculated utility
        """
        preference_type = preferences.get("preference_type", "vector")
        
        if preference_type == "vector":
            return self._calculate_vector_utilities(
                final_allocation, preferences, config, round_number
            )
        elif preference_type == "matrix":
            return self._calculate_matrix_utilities(
                final_allocation, preferences, config, round_number
            )
        else:
            self.logger.error(f"Unknown preference type: {preference_type}")
            return {}
    
    def _calculate_vector_utilities(self, 
                                  final_allocation: Dict[str, List[int]],
                                  preferences: Dict[str, Any],
                                  config: NegotiationEngineConfig,
                                  round_number: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate utilities for vector-based preference system.
        
        In vector preferences, each agent has a preference vector indicating
        their utility for each item independently.
        """
        utilities = {}
        agent_preferences = preferences.get("agent_preferences", {})
        
        for agent_id, allocated_items in final_allocation.items():
            if agent_id not in agent_preferences:
                utilities[agent_id] = 0.0
                continue
            
            agent_prefs = agent_preferences[agent_id]
            
            # Calculate base utility as sum of preferences for allocated items
            base_utility = 0.0
            for item_index in allocated_items:
                if isinstance(agent_prefs, list) and 0 <= item_index < len(agent_prefs):
                    base_utility += agent_prefs[item_index]
                elif isinstance(agent_prefs, dict):
                    base_utility += agent_prefs.get(str(item_index), 0.0)
            
            # Apply discounting if round number is provided
            if round_number is not None and config.gamma_discount < 1.0:
                discount_factor = config.gamma_discount ** round_number
                final_utility = base_utility * discount_factor
                
                self.logger.debug(
                    f"{agent_id}: base_utility={base_utility:.3f}, "
                    f"discount={discount_factor:.3f}, final={final_utility:.3f}"
                )
            else:
                final_utility = base_utility
            
            utilities[agent_id] = final_utility
        
        return utilities
    
    def _calculate_matrix_utilities(self, 
                                  final_allocation: Dict[str, List[int]],
                                  preferences: Dict[str, Any],
                                  config: NegotiationEngineConfig,
                                  round_number: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate utilities for matrix-based preference system.
        
        In matrix preferences, utilities depend on combinations of items
        and potentially on what other agents receive.
        """
        utilities = {}
        agent_preferences = preferences.get("agent_preferences", {})
        
        # Matrix-based utility calculation is more complex
        # This is a simplified implementation - real matrix systems
        # would involve more sophisticated interdependence calculations
        
        for agent_id, allocated_items in final_allocation.items():
            if agent_id not in agent_preferences:
                utilities[agent_id] = 0.0
                continue
            
            agent_prefs = agent_preferences[agent_id]
            
            if isinstance(agent_prefs, dict) and "matrix" in agent_prefs:
                # Handle matrix-based preferences
                utility_matrix = agent_prefs["matrix"]
                base_utility = self._calculate_from_matrix(
                    allocated_items, final_allocation, utility_matrix
                )
            else:
                # Fallback to vector-style calculation
                base_utility = sum(
                    agent_prefs.get(str(item_index), 0.0) for item_index in allocated_items
                )
            
            # Apply discounting
            if round_number is not None and config.gamma_discount < 1.0:
                discount_factor = config.gamma_discount ** round_number
                final_utility = base_utility * discount_factor
            else:
                final_utility = base_utility
            
            utilities[agent_id] = final_utility
        
        return utilities
    
    def _calculate_from_matrix(self, 
                             allocated_items: List[int],
                             full_allocation: Dict[str, List[int]],
                             utility_matrix: Dict[str, Any]) -> float:
        """
        Calculate utility from matrix-based preferences.
        
        This handles interdependencies between items and agents.
        """
        # Simplified matrix calculation
        # Real implementation would depend on specific matrix format
        
        base_utility = 0.0
        
        # Individual item values
        for item_index in allocated_items:
            item_key = str(item_index)
            if item_key in utility_matrix:
                base_utility += utility_matrix[item_key]
        
        # Interaction effects (simplified)
        if len(allocated_items) > 1 and "combinations" in utility_matrix:
            combinations = utility_matrix["combinations"]
            for combo_key, combo_value in combinations.items():
                # Check if this agent has the required combination
                required_items = [int(x) for x in combo_key.split(',')]
                if all(item in allocated_items for item in required_items):
                    base_utility += combo_value
        
        return base_utility
    
    def get_utility_summary(self, 
                          utilities: Dict[str, float],
                          allocation: Dict[str, List[int]]) -> Dict[str, Any]:
        """Get a summary of utility calculations for analysis."""
        if not utilities:
            return {"error": "No utilities to summarize"}
        
        sorted_utilities = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_participants": len(utilities),
            "winner": sorted_utilities[0][0] if sorted_utilities else None,
            "winner_utility": sorted_utilities[0][1] if sorted_utilities else 0.0,
            "utility_range": {
                "max": max(utilities.values()) if utilities else 0.0,
                "min": min(utilities.values()) if utilities else 0.0,
                "mean": sum(utilities.values()) / len(utilities) if utilities else 0.0
            },
            "rankings": [{"agent": agent, "utility": util} for agent, util in sorted_utilities],
            "allocation_summary": {
                agent: len(items) for agent, items in allocation.items()
            }
        }