"""
Agent Experience Logger - Captures the complete "lived experience" of each agent.

This module creates detailed JSON logs that record every input prompt sent to an agent
and every output response generated, creating a complete record of each agent's 
perspective during negotiation.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class InteractionType(Enum):
    """Types of agent interactions during negotiation."""
    DISCUSSION = "discussion"
    PROPOSAL = "proposal" 
    VOTING = "voting"
    REFLECTION = "reflection"


@dataclass
class AgentInteraction:
    """Single agent interaction record."""
    interaction_id: str
    timestamp: str
    round_number: int
    phase: str  # InteractionType value
    
    # Input information
    input_prompt: str
    context_data: Dict[str, Any]
    
    # Output information  
    raw_response: str
    processed_response: Any  # Could be parsed JSON, text, etc.
    
    # Metadata
    response_time_seconds: float
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    
    # Strategic context
    other_agents: List[str] = None
    current_proposals: Optional[List[Dict]] = None
    previous_votes: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AgentExperienceLogger:
    """
    Logs the complete experience of a single agent during negotiation.
    
    Creates a comprehensive JSON file that records every interaction
    from the agent's perspective.
    """
    
    def __init__(self, agent_id: str, experiment_id: str, results_dir: Path, run_id: Optional[str] = None):
        self.agent_id = agent_id
        self.experiment_id = experiment_id
        self.results_dir = Path(results_dir)
        self.run_id = run_id
        
        # Ensure agent logs directory exists
        self.agent_log_dir = self.results_dir / "agent_experiences"
        self.agent_log_dir.mkdir(exist_ok=True)
        
        # Log file path - include run_id if provided
        if run_id:
            self.log_file = self.agent_log_dir / f"{agent_id}_run_{run_id}_experience.json"
        else:
            self.log_file = self.agent_log_dir / f"{agent_id}_experience.json"
        
        # In-memory log storage
        self.interactions: List[AgentInteraction] = []
        self.metadata = {
            "agent_id": agent_id,
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "total_interactions": 0,
            "phases_participated": set(),
            "final_outcome": None
        }
        
        # Performance tracking
        self.total_tokens_used = 0
        self.total_response_time = 0.0
        
    def log_interaction(self, 
                       interaction_type: InteractionType,
                       round_number: int,
                       input_prompt: str,
                       context_data: Dict[str, Any],
                       raw_response: str,
                       processed_response: Any,
                       response_time: float,
                       tokens_used: Optional[int] = None,
                       model_used: Optional[str] = None,
                       other_agents: Optional[List[str]] = None,
                       current_proposals: Optional[List[Dict]] = None,
                       previous_votes: Optional[Dict] = None) -> str:
        """
        Log a single agent interaction.
        
        Returns:
            interaction_id for reference
        """
        interaction_id = f"{self.agent_id}_{interaction_type.value}_r{round_number}_{int(time.time())}"
        
        interaction = AgentInteraction(
            interaction_id=interaction_id,
            timestamp=datetime.now().isoformat(),
            round_number=round_number,
            phase=interaction_type.value,
            input_prompt=input_prompt,
            context_data=context_data,
            raw_response=raw_response,
            processed_response=processed_response,
            response_time_seconds=response_time,
            tokens_used=tokens_used,
            model_used=model_used,
            other_agents=other_agents or [],
            current_proposals=current_proposals,
            previous_votes=previous_votes
        )
        
        self.interactions.append(interaction)
        
        # Update metadata
        self.metadata["total_interactions"] += 1
        self.metadata["phases_participated"].add(interaction_type.value)
        self.total_tokens_used += tokens_used or 0
        self.total_response_time += response_time
        
        # Auto-save after each interaction to prevent data loss
        self._save_log()
        
        return interaction_id
    
    def log_final_outcome(self, outcome_data: Dict[str, Any]):
        """Log the final negotiation outcome from this agent's perspective."""
        self.metadata["final_outcome"] = outcome_data
        self.metadata["end_time"] = datetime.now().isoformat()
        self._save_log()
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this agent's experience."""
        return {
            "agent_id": self.agent_id,
            "total_interactions": len(self.interactions),
            "phases_participated": list(self.metadata["phases_participated"]),
            "total_tokens_used": self.total_tokens_used,
            "average_response_time": self.total_response_time / max(len(self.interactions), 1),
            "interaction_count_by_phase": self._count_by_phase(),
            "longest_prompt": self._get_longest_prompt(),
            "longest_response": self._get_longest_response()
        }
    
    def _count_by_phase(self) -> Dict[str, int]:
        """Count interactions by phase type."""
        counts = {}
        for interaction in self.interactions:
            phase = interaction.phase
            counts[phase] = counts.get(phase, 0) + 1
        return counts
    
    def _get_longest_prompt(self) -> Dict[str, Any]:
        """Find the longest input prompt."""
        if not self.interactions:
            return {}
        
        longest = max(self.interactions, key=lambda x: len(x.input_prompt))
        return {
            "phase": longest.phase,
            "round": longest.round_number,
            "length": len(longest.input_prompt),
            "preview": longest.input_prompt[:200] + "..." if len(longest.input_prompt) > 200 else longest.input_prompt
        }
    
    def _get_longest_response(self) -> Dict[str, Any]:
        """Find the longest agent response."""
        if not self.interactions:
            return {}
        
        longest = max(self.interactions, key=lambda x: len(x.raw_response))
        return {
            "phase": longest.phase,
            "round": longest.round_number,
            "length": len(longest.raw_response),
            "preview": longest.raw_response[:200] + "..." if len(longest.raw_response) > 200 else longest.raw_response
        }
    
    def _save_log(self):
        """Save the current log state to JSON file."""
        log_data = {
            "metadata": {
                **self.metadata,
                "phases_participated": list(self.metadata["phases_participated"])  # Convert set to list for JSON
            },
            "summary": self.get_agent_summary(),
            "interactions": [interaction.to_dict() for interaction in self.interactions]
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def export_human_readable(self) -> str:
        """Export a human-readable version of the agent's experience."""
        lines = []
        lines.append(f"Agent Experience Log: {self.agent_id}")
        lines.append(f"Experiment: {self.experiment_id}")
        lines.append(f"Total Interactions: {len(self.interactions)}")
        lines.append("="*60)
        
        for interaction in self.interactions:
            lines.append(f"\nROUND {interaction.round_number} - {interaction.phase.upper()}")
            lines.append(f"Time: {interaction.timestamp}")
            lines.append(f"Response Time: {interaction.response_time_seconds:.2f}s")
            lines.append("\nINPUT PROMPT:")
            lines.append("-" * 40)
            lines.append(interaction.input_prompt)
            lines.append("\nAGENT RESPONSE:")
            lines.append("-" * 40)
            lines.append(interaction.raw_response)
            lines.append("\nPROCESSED OUTPUT:")
            lines.append("-" * 40)
            lines.append(str(interaction.processed_response))
            lines.append("="*60)
        
        return "\n".join(lines)


class ExperimentLogger:
    """
    Manages experience loggers for all agents in an experiment.
    
    Creates and coordinates individual agent loggers, and provides
    experiment-wide analysis capabilities.
    """
    
    def __init__(self, experiment_id: str, results_dir: Path, agent_ids: List[str], run_id: Optional[str] = None):
        self.experiment_id = experiment_id
        self.results_dir = Path(results_dir)
        self.agent_ids = agent_ids
        self.run_id = run_id
        
        # Create individual agent loggers
        self.agent_loggers: Dict[str, AgentExperienceLogger] = {}
        for agent_id in agent_ids:
            self.agent_loggers[agent_id] = AgentExperienceLogger(
                agent_id, experiment_id, results_dir, run_id
            )
    
    def get_logger(self, agent_id: str) -> AgentExperienceLogger:
        """Get the logger for a specific agent."""
        if agent_id not in self.agent_loggers:
            raise ValueError(f"No logger found for agent {agent_id}")
        return self.agent_loggers[agent_id]
    
    def log_final_outcomes(self, outcomes: Dict[str, Any]):
        """Log final outcomes for all agents."""
        for agent_id, logger in self.agent_loggers.items():
            # Extract agent-specific outcome data
            final_utilities = outcomes.get("final_utilities", {}) or {}
            agent_outcome = {
                "consensus_reached": outcomes.get("consensus_reached", False),
                "final_allocation": outcomes.get("final_allocation", {}),
                "agent_final_utility": final_utilities.get(agent_id, 0) if isinstance(final_utilities, dict) else 0,
                "winner": outcomes.get("winner_agent_id") == agent_id,
                "total_rounds": outcomes.get("final_round", 0)
            }
            logger.log_final_outcome(agent_outcome)
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate a summary across all agents."""
        summaries = {}
        for agent_id, logger in self.agent_loggers.items():
            summaries[agent_id] = logger.get_agent_summary()
        
        return {
            "experiment_id": self.experiment_id,
            "total_agents": len(self.agent_ids),
            "agent_summaries": summaries,
            "cross_agent_analysis": self._analyze_cross_agent_patterns(summaries)
        }
    
    def _analyze_cross_agent_patterns(self, summaries: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across all agents."""
        all_response_times = []
        all_token_counts = []
        phase_participation = {}
        
        for agent_id, summary in summaries.items():
            all_response_times.append(summary["average_response_time"])
            all_token_counts.append(summary["total_tokens_used"])
            
            for phase in summary["phases_participated"]:
                if phase not in phase_participation:
                    phase_participation[phase] = 0
                phase_participation[phase] += 1
        
        return {
            "average_response_time_across_agents": sum(all_response_times) / len(all_response_times),
            "total_tokens_all_agents": sum(all_token_counts),
            "phase_participation_rates": {
                phase: count/len(self.agent_ids) 
                for phase, count in phase_participation.items()
            }
        }