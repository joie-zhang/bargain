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
    final_utilities: Optional[Dict[str, float]]
    strategic_behaviors: Dict[str, Any]
    conversation_logs: List[Dict[str, Any]]
    agent_performance: Dict[str, Any]
    exploitation_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "consensus_reached": self.consensus_reached,
            "final_round": self.final_round,
            "final_utilities": self.final_utilities,
            "strategic_behaviors": self.strategic_behaviors,
            "conversation_logs": self.conversation_logs,
            "agent_performance": self.agent_performance,
            "exploitation_detected": self.exploitation_detected
        }


@dataclass
class BatchResults:
    """Data class to store aggregated results from batch experiments."""
    batch_id: str
    num_runs: int
    experiments: List[ExperimentResults]
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
            "consensus_rate": self.consensus_rate,
            "average_rounds": self.average_rounds,
            "exploitation_rate": self.exploitation_rate,
            "strategic_behaviors_summary": self.strategic_behaviors_summary
        }