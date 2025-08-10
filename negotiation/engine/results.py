"""
Result types for the negotiation engine.

This module defines the data structures used to capture and return
the outcomes of negotiation phases and complete negotiations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time


@dataclass
class PhaseResult:
    """Result from a single negotiation phase."""
    phase_type: str
    round_number: int
    phase_data: Dict[str, Any]
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    
    # Communication data
    messages: List[Any] = field(default_factory=list)  # List[Message] - avoiding circular import
    
    # Analysis data
    strategic_indicators: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    
    def add_strategic_indicator(self, key: str, value: Any) -> None:
        """Add a strategic behavior indicator to this phase."""
        self.strategic_indicators[key] = value
    
    def mark_error(self, error_type: str, error_details: Dict[str, Any]) -> None:
        """Mark this phase as having encountered an error."""
        self.error_info = {
            "error_type": error_type,
            "error_details": error_details,
            "timestamp": time.time()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this phase result."""
        return {
            "phase_type": self.phase_type,
            "round_number": self.round_number,
            "duration_seconds": self.duration_seconds,
            "message_count": len(self.messages),
            "strategic_indicators_count": len(self.strategic_indicators),
            "has_error": self.error_info is not None,
            "timestamp": self.timestamp
        }


@dataclass 
class NegotiationResult:
    """Complete negotiation outcome and analysis."""
    # Identifiers and metadata
    negotiation_id: str
    config: 'NegotiationEngineConfig'  # Forward reference to avoid circular import
    timestamp: float = field(default_factory=time.time)
    
    # Core outcome
    consensus_reached: bool
    final_round: int
    winner_agent_id: Optional[str]
    final_utilities: Dict[str, float]
    
    # Process data
    phase_results: List[PhaseResult] = field(default_factory=list)
    conversation_logs: List[Any] = field(default_factory=list)  # List[Message]
    
    # Analysis results
    strategic_behaviors: Dict[str, Any] = field(default_factory=dict)
    agent_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Performance metrics
    total_duration: float = 0.0
    api_costs: Dict[str, float] = field(default_factory=dict)
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_phase_result(self, phase_result: PhaseResult) -> None:
        """Add a phase result to the negotiation."""
        self.phase_results.append(phase_result)
        
        # Update strategic behaviors from phase
        for key, value in phase_result.strategic_indicators.items():
            if key not in self.strategic_behaviors:
                self.strategic_behaviors[key] = []
            self.strategic_behaviors[key].append({
                "phase": phase_result.phase_type,
                "round": phase_result.round_number,
                "value": value
            })
    
    def add_error(self, error_type: str, error_details: Dict[str, Any], 
                  phase_type: Optional[str] = None, round_number: Optional[int] = None) -> None:
        """Add an error to the negotiation record."""
        self.errors.append({
            "error_type": error_type,
            "error_details": error_details,
            "phase_type": phase_type,
            "round_number": round_number,
            "timestamp": time.time()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a high-level summary of the negotiation results."""
        return {
            "negotiation_id": self.negotiation_id,
            "timestamp": self.timestamp,
            "consensus_reached": self.consensus_reached,
            "final_round": self.final_round,
            "winner_agent_id": self.winner_agent_id,
            "total_phases": len(self.phase_results),
            "total_messages": len(self.conversation_logs),
            "total_duration": self.total_duration,
            "total_api_cost": sum(self.api_costs.values()),
            "error_count": len(self.errors),
            "strategic_behavior_types": list(self.strategic_behaviors.keys())
        }
    
    def get_win_analysis(self) -> Dict[str, Any]:
        """Analyze win patterns and strategic outcomes."""
        if not self.final_utilities:
            return {"analysis_type": "no_utilities", "details": "No final utilities available"}
        
        # Find winner and analyze margins
        sorted_utilities = sorted(self.final_utilities.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_utilities[0] if sorted_utilities else None
        
        if not winner:
            return {"analysis_type": "no_winner", "details": "Could not determine winner"}
        
        # Calculate win margins
        win_margins = []
        winner_utility = winner[1]
        for agent_id, utility in sorted_utilities[1:]:
            margin = winner_utility - utility
            win_margins.append({
                "opponent": agent_id,
                "margin": margin,
                "percentage": (margin / winner_utility * 100) if winner_utility > 0 else 0
            })
        
        return {
            "analysis_type": "win_analysis",
            "winner": winner[0],
            "winner_utility": winner_utility,
            "win_margins": win_margins,
            "total_participants": len(self.final_utilities),
            "consensus_achieved": self.consensus_reached,
            "rounds_to_consensus": self.final_round if self.consensus_reached else None
        }
    
    def export_for_analysis(self) -> Dict[str, Any]:
        """Export data in format suitable for research analysis."""
        return {
            "metadata": {
                "negotiation_id": self.negotiation_id,
                "timestamp": self.timestamp,
                "config": self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
            },
            "outcome": {
                "consensus_reached": self.consensus_reached,
                "final_round": self.final_round,
                "winner_agent_id": self.winner_agent_id,
                "final_utilities": self.final_utilities
            },
            "process": {
                "phase_summaries": [phase.get_summary() for phase in self.phase_results],
                "total_duration": self.total_duration,
                "message_count": len(self.conversation_logs)
            },
            "analysis": {
                "strategic_behaviors": self.strategic_behaviors,
                "agent_performance": self.agent_performance,
                "win_analysis": self.get_win_analysis()
            },
            "costs": self.api_costs,
            "errors": self.errors
        }