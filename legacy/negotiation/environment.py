"""
Core negotiation environment for multi-agent AI interactions.

This module implements the foundational environment for studying how stronger LLMs
exploit weaker LLMs in negotiation settings through strategic behaviors.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json
from pathlib import Path


class NegotiationStatus(Enum):
    """Status of the negotiation process."""
    SETUP = "setup"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    TERMINATED = "terminated"


@dataclass
class NegotiationConfig:
    """Configuration for a negotiation environment."""
    m_items: int  # Number of items to negotiate over
    n_agents: int  # Number of participating agents
    t_rounds: int  # Maximum number of rounds
    gamma_discount: float  # Discount factor for future rewards
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.m_items < 1:
            raise ValueError("Must have at least 1 item")
        if self.n_agents < 2:
            raise ValueError("Need at least 2 agents for negotiation")
        if self.t_rounds < 1:
            raise ValueError("Must have at least 1 round")
        if not 0 <= self.gamma_discount <= 1:
            raise ValueError("Discount factor must be between 0 and 1")


@dataclass
class Item:
    """Represents an item in the negotiation."""
    id: int
    name: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }


@dataclass
class Round:
    """Represents a single negotiation round."""
    round_number: int
    status: str = "pending"
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    votes: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "status": self.status,
            "proposals": self.proposals,
            "votes": self.votes,
            "outcome": self.outcome
        }


class ItemPool:
    """Manages the pool of items available for negotiation."""
    
    def __init__(self, num_items: int, seed: Optional[int] = None):
        self.num_items = num_items
        self.items: List[Item] = []
        self._rng = random.Random(seed)
        self._generate_items()
    
    def _generate_items(self) -> None:
        """Generate the item pool with default names."""
        item_names = [
            "Apple", "Book", "Camera", "Diamond", "Emerald", 
            "Flower", "Guitar", "Hat", "Ice", "Jewel",
            "Kite", "Lamp", "Mirror", "Notebook", "Orange",
            "Pencil", "Quill", "Ring", "Stone", "Tablet"
        ]
        
        # If we need more items than default names, generate generic ones
        if self.num_items > len(item_names):
            for i in range(len(item_names), self.num_items):
                item_names.append(f"Item_{i+1}")
        
        # Sample items randomly if we have more names than needed
        selected_names = self._rng.sample(item_names, self.num_items)
        
        self.items = [
            Item(id=i, name=name, description=f"A {name.lower()} for negotiation")
            for i, name in enumerate(selected_names)
        ]
    
    def get_items(self) -> List[Item]:
        """Return all items in the pool."""
        return self.items.copy()
    
    def get_item(self, item_id: int) -> Optional[Item]:
        """Get a specific item by ID."""
        return next((item for item in self.items if item.id == item_id), None)


class NegotiationEnvironment:
    """
    Core negotiation environment for multi-agent interactions.
    
    This class manages the negotiation process between multiple agents,
    including round progression, item allocation, and outcome tracking.
    """
    
    def __init__(self, config: NegotiationConfig):
        self.config = config
        self.status = NegotiationStatus.SETUP
        self.current_round = 0
        self.rounds: List[Round] = []
        
        # Initialize random number generator
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Initialize item pool
        self.item_pool = ItemPool(config.m_items, config.random_seed)
        
        # Track agent participation
        self.agent_ids: List[str] = []
        self.active_agents: List[str] = []
        
        # Track final outcomes
        self.final_allocation: Optional[Dict[str, List[int]]] = None
        self.final_utilities: Optional[Dict[str, float]] = None
        
        # Session metadata
        self.session_data = {
            "config": self._config_to_dict(),
            "created_at": None,
            "completed_at": None,
            "total_duration": None
        }
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "m_items": self.config.m_items,
            "n_agents": self.config.n_agents,
            "t_rounds": self.config.t_rounds,
            "gamma_discount": self.config.gamma_discount,
            "random_seed": self.config.random_seed
        }
    
    def initialize_agents(self, agent_ids: List[str]) -> None:
        """Initialize the negotiation with specific agent IDs."""
        if len(agent_ids) != self.config.n_agents:
            raise ValueError(f"Expected {self.config.n_agents} agents, got {len(agent_ids)}")
        
        self.agent_ids = agent_ids.copy()
        self.active_agents = agent_ids.copy()
        self.status = NegotiationStatus.IN_PROGRESS
        
        # Create first round
        self.start_new_round()
    
    def start_new_round(self) -> Round:
        """Start a new negotiation round."""
        if self.current_round >= self.config.t_rounds:
            raise RuntimeError("Maximum rounds reached")
        
        self.current_round += 1
        new_round = Round(round_number=self.current_round)
        self.rounds.append(new_round)
        
        return new_round
    
    def get_current_round(self) -> Optional[Round]:
        """Get the current active round."""
        if not self.rounds:
            return None
        return self.rounds[-1]
    
    def advance_round(self) -> bool:
        """
        Advance to the next round if possible.
        
        Returns:
            True if advanced to new round, False if negotiation should end
        """
        current = self.get_current_round()
        if not current:
            return False
        
        # Mark current round as completed
        current.status = "completed"
        
        # Check if we've reached maximum rounds
        if self.current_round >= self.config.t_rounds:
            self.status = NegotiationStatus.COMPLETED
            return False
        
        # Check if negotiation was terminated early
        if self.status == NegotiationStatus.TERMINATED:
            return False
        
        # Start new round
        self.start_new_round()
        return True
    
    def add_proposal(self, agent_id: str, proposal: Dict[str, Any]) -> None:
        """Add a proposal from an agent to the current round."""
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} is not active in this negotiation")
        
        current = self.get_current_round()
        if not current:
            raise RuntimeError("No active round")
        
        proposal_with_metadata = {
            "agent_id": agent_id,
            "round": self.current_round,
            "proposal": proposal,
            "timestamp": None  # Will be set by calling system
        }
        
        current.proposals.append(proposal_with_metadata)
    
    def add_vote(self, agent_id: str, vote: Dict[str, Any]) -> None:
        """Add a vote from an agent to the current round."""
        if agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} is not active in this negotiation")
        
        current = self.get_current_round()
        if not current:
            raise RuntimeError("No active round")
        
        vote_with_metadata = {
            "agent_id": agent_id,
            "round": self.current_round,
            "vote": vote,
            "timestamp": None  # Will be set by calling system
        }
        
        current.votes.append(vote_with_metadata)
    
    def check_consensus(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if consensus has been reached in the current round.
        
        Returns:
            Tuple of (consensus_reached, agreed_allocation)
        """
        current = self.get_current_round()
        if not current or len(current.votes) != len(self.active_agents):
            return False, None
        
        # Simple consensus: all agents must vote for the same proposal
        if not current.votes:
            return False, None
        
        first_vote = current.votes[0]["vote"]
        for vote in current.votes[1:]:
            if vote["vote"] != first_vote:
                return False, None
        
        return True, first_vote
    
    def terminate_negotiation(self, reason: str = "Manual termination") -> None:
        """Terminate the negotiation early."""
        self.status = NegotiationStatus.TERMINATED
        
        current = self.get_current_round()
        if current:
            current.status = "terminated"
            current.outcome = {"reason": reason}
    
    def finalize_allocation(self, allocation: Dict[str, List[int]]) -> None:
        """Finalize the item allocation and mark negotiation as completed."""
        # Validate allocation
        all_items = set(range(self.config.m_items))
        allocated_items = set()
        
        for agent_id, items in allocation.items():
            if agent_id not in self.agent_ids:
                raise ValueError(f"Unknown agent: {agent_id}")
            allocated_items.update(items)
        
        if allocated_items != all_items:
            raise ValueError("Allocation must assign all items exactly once")
        
        self.final_allocation = allocation
        self.status = NegotiationStatus.COMPLETED
    
    def get_items_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all items in the negotiation."""
        return [item.to_dict() for item in self.item_pool.get_items()]
    
    def get_negotiation_state(self) -> Dict[str, Any]:
        """Get the current state of the negotiation."""
        return {
            "status": self.status.value,
            "current_round": self.current_round,
            "max_rounds": self.config.t_rounds,
            "agents": {
                "total": len(self.agent_ids),
                "active": len(self.active_agents),
                "ids": self.agent_ids
            },
            "items": {
                "total": self.config.m_items,
                "summary": self.get_items_summary()
            },
            "rounds_completed": len([r for r in self.rounds if r.status == "completed"]),
            "final_allocation": self.final_allocation
        }
    
    def export_session_data(self) -> Dict[str, Any]:
        """Export complete session data for analysis."""
        return {
            "session_metadata": self.session_data,
            "negotiation_state": self.get_negotiation_state(),
            "rounds": [round.to_dict() for round in self.rounds],
            "config": self._config_to_dict()
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save the negotiation session to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.export_session_data(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'NegotiationEnvironment':
        """Load a negotiation session from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct config
        config_data = data["config"]
        config = NegotiationConfig(
            m_items=config_data["m_items"],
            n_agents=config_data["n_agents"],
            t_rounds=config_data["t_rounds"],
            gamma_discount=config_data["gamma_discount"],
            random_seed=config_data.get("random_seed")
        )
        
        # Create environment
        env = cls(config)
        
        # Restore state
        state = data["negotiation_state"]
        env.status = NegotiationStatus(state["status"])
        env.current_round = state["current_round"]
        env.agent_ids = state["agents"]["ids"]
        env.active_agents = state["agents"]["ids"]  # Simplified for now
        env.final_allocation = state.get("final_allocation")
        
        # Restore rounds
        for round_data in data["rounds"]:
            round_obj = Round(
                round_number=round_data["round_number"],
                status=round_data["status"],
                proposals=round_data["proposals"],
                votes=round_data["votes"],
                outcome=round_data.get("outcome")
            )
            env.rounds.append(round_obj)
        
        return env


def create_negotiation_environment(
    m_items: int,
    n_agents: int,
    t_rounds: int,
    gamma_discount: float = 0.9,
    random_seed: Optional[int] = None
) -> NegotiationEnvironment:
    """
    Factory function to create a negotiation environment with given parameters.
    
    Args:
        m_items: Number of items to negotiate over
        n_agents: Number of participating agents  
        t_rounds: Maximum number of negotiation rounds
        gamma_discount: Discount factor for future rewards (default: 0.9)
        random_seed: Random seed for reproducibility (optional)
    
    Returns:
        NegotiationEnvironment instance ready for initialization
    """
    config = NegotiationConfig(
        m_items=m_items,
        n_agents=n_agents,
        t_rounds=t_rounds,
        gamma_discount=gamma_discount,
        random_seed=random_seed
    )
    
    return NegotiationEnvironment(config)