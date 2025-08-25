"""Data models for experiment results and batch processing."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class ExperimentResults:
    """Data class to store results from a single experiment."""
    experiment_id: str
    timestamp: float
    config: Dict[str, Any]
    consensus_reached: bool
    final_round: int
    winner_agent_id: Optional[str]
    final_utilities: Optional[Dict[str, float]]
    strategic_behaviors: Dict[str, Any]
    conversation_logs: List[Dict[str, Any]]
    agent_performance: Dict[str, Any]
    exploitation_detected: bool
    
    # Model-specific winner tracking (kept for backwards compatibility but will be renamed)
    model_winners: Dict[str, bool] = field(default_factory=dict)
    # New fields for clearer tracking
    proposal_winners: Dict[str, bool] = field(default_factory=dict)  # Who got their proposal accepted
    utility_winners: Dict[str, bool] = field(default_factory=dict)   # Who achieved highest utility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "consensus_reached": self.consensus_reached,
            "final_round": self.final_round,
            "winner_agent_id": self.winner_agent_id,
            "final_utilities": self.final_utilities,
            "strategic_behaviors": self.strategic_behaviors,
            "conversation_logs": self.conversation_logs,
            "agent_performance": self.agent_performance,
            "exploitation_detected": self.exploitation_detected,
            "model_winners": self.model_winners,
            "proposal_winners": self.proposal_winners,
            "utility_winners": self.utility_winners
        }


@dataclass
class BatchResults:
    """Data class to store aggregated results from batch experiments."""
    batch_id: str
    num_runs: int
    experiments: List[ExperimentResults]
    model_win_rates: Dict[str, float]
    consensus_rate: float
    average_rounds: float
    exploitation_rate: float
    strategic_behaviors_summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_id": self.batch_id,
            "num_runs": self.num_runs,
            "experiments": [exp.to_dict() for exp in self.experiments],
            "model_win_rates": self.model_win_rates,
            "consensus_rate": self.consensus_rate,
            "average_rounds": self.average_rounds,
            "exploitation_rate": self.exploitation_rate,
            "strategic_behaviors_summary": self.strategic_behaviors_summary
        }