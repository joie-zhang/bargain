"""
Preference system for multi-agent negotiation environment.

This module implements both competitive vector preferences and cooperative matrix preferences
for studying strategic interactions between AI agents in negotiation settings.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np
from abc import ABC, abstractmethod
import json
import sys
from pathlib import Path

# Import random vector generator for creating preference vectors
from .random_vector_generator import RandomVectorGenerator


class PreferenceType(Enum):
    """Types of preference systems available."""
    VECTOR = "vector"  # Competitive individual preferences
    MATRIX = "matrix"  # Cooperative/competitive group preferences


@dataclass
class PreferenceConfig:
    """Configuration for preference generation."""
    preference_type: PreferenceType
    m_items: int  # Number of items
    n_agents: int  # Number of agents
    min_value: float = 0.0
    max_value: float = 10.0
    random_seed: Optional[int] = None
    known_to_all: bool = False  # Whether preferences are common knowledge
    
    # Vector preference specific
    target_cosine_similarity: Optional[float] = None  # For controlling competition level
    
    # Matrix preference specific
    cooperation_factor: Optional[float] = 0.5  # How much agents care about others
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.m_items < 1:
            raise ValueError("Must have at least 1 item")
        if self.n_agents < 2:
            raise ValueError("Need at least 2 agents")
        if not self.min_value <= self.max_value:
            raise ValueError("min_value must be <= max_value")
        if self.target_cosine_similarity is not None:
            if not -1 <= self.target_cosine_similarity <= 1:
                raise ValueError("Cosine similarity must be between -1 and 1")
        if self.cooperation_factor is not None:
            if not 0 <= self.cooperation_factor <= 1:
                raise ValueError("Cooperation factor must be between 0 and 1")


class BasePreferenceSystem(ABC):
    """Abstract base class for preference systems."""
    
    def __init__(self, config: PreferenceConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        
    @abstractmethod
    def generate_preferences(self) -> Dict[str, Any]:
        """Generate preferences for all agents."""
        pass
    
    @abstractmethod
    def calculate_utility(self, agent_id: str, allocation: Dict[str, List[int]], 
                         preferences: Dict[str, Any]) -> float:
        """Calculate utility for an agent given an allocation."""
        pass
    
    @abstractmethod
    def get_agent_preferences(self, agent_id: str, preferences: Dict[str, Any]) -> Any:
        """Get preferences specific to an agent."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preference system to dictionary for serialization."""
        return {
            "type": self.config.preference_type.value,
            "config": {
                "m_items": self.config.m_items,
                "n_agents": self.config.n_agents,
                "min_value": self.config.min_value,
                "max_value": self.config.max_value,
                "random_seed": self.config.random_seed,
                "known_to_all": self.config.known_to_all,
                "target_cosine_similarity": self.config.target_cosine_similarity,
                "cooperation_factor": self.config.cooperation_factor
            }
        }


class VectorPreferenceSystem(BasePreferenceSystem):
    """
    Vector-based preference system for competitive scenarios.
    
    Each agent has an m-dimensional preference vector where each dimension
    represents their valuation for a specific item (0-10 scale).
    """
    
    def generate_preferences(self) -> Dict[str, Any]:
        """Generate competitive vector preferences for all agents."""
        agent_ids = [f"agent_{i}" for i in range(self.config.n_agents)]
        
        if self.config.target_cosine_similarity is not None:
            # Generate preferences with target cosine similarity
            preferences = self._generate_with_target_similarity(agent_ids)
        else:
            # Generate random preferences
            preferences = self._generate_random_preferences(agent_ids)
        
        # Calculate actual cosine similarities between all pairs
        cosine_similarities = self._calculate_pairwise_cosine_similarities(preferences)
        
        return {
            "type": "vector",
            "agent_preferences": preferences,
            "cosine_similarities": cosine_similarities,
            "config": self.config.__dict__
        }
    
    def _generate_random_preferences(self, agent_ids: List[str]) -> Dict[str, List[float]]:
        """Generate random preference vectors."""
        preferences = {}
        for agent_id in agent_ids:
            preferences[agent_id] = self.rng.uniform(
                self.config.min_value, 
                self.config.max_value, 
                self.config.m_items
            ).tolist()
        return preferences
    
    def _generate_with_target_similarity(self, agent_ids: List[str]) -> Dict[str, List[float]]:
        """Generate preferences with target cosine similarity using random vector generator."""
        preferences = {}
        
        # Use RandomVectorGenerator to create vectors with exact cosine similarity
        generator = RandomVectorGenerator(random_seed=self.config.random_seed)
        
        # Calculate total utility based on config
        max_utility = self.config.m_items * (self.config.max_value - self.config.min_value) / 2
        
        if self.config.n_agents == 2:
            # Generate two vectors with target similarity
            v1, v2 = generator.generate_vectors_with_cosine_similarity(
                target_cosine=self.config.target_cosine_similarity,
                n_items=self.config.m_items,
                max_utility=max_utility,
                integer_values=True
            )
            preferences[agent_ids[0]] = v1.tolist()
            preferences[agent_ids[1]] = v2.tolist()
        else:
            # For more than 2 agents, generate first vector then create similar ones
            v1 = generator._generate_random_vector(
                self.config.m_items, 
                max_utility,
                integer_values=True
            )
            preferences[agent_ids[0]] = v1.tolist()
            
            # Generate other agents with target similarity to first
            for agent_id in agent_ids[1:]:
                target_vector = self._generate_similar_vector(
                    v1, self.config.target_cosine_similarity
                )
                preferences[agent_id] = target_vector.tolist()
        
        return preferences
    
    def _generate_similar_vector(self, base_vector: np.ndarray, 
                               target_similarity: float) -> np.ndarray:
        """Generate a vector with target cosine similarity to base vector."""
        # Normalize base vector to unit vector
        v1 = base_vector / np.linalg.norm(base_vector)
        
        # Generate a random orthogonal vector
        random_vec = self.rng.randn(len(base_vector))
        # Make it orthogonal to v1 using Gram-Schmidt
        v_orthogonal = random_vec - np.dot(random_vec, v1) * v1
        v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal)  # Normalize
        
        # Calculate angle from desired cosine similarity
        angle = np.arccos(np.clip(target_similarity, -1, 1))
        
        # Construct v2 using the angle
        v2 = np.cos(angle) * v1 + np.sin(angle) * v_orthogonal
        
        # Scale vectors to have random magnitudes (matching generate_vectors_with_cosine_similarity.py)
        v2 = v2 * self.rng.uniform(1, 10)
        
        # Ensure values are within bounds
        v2 = np.clip(v2, self.config.min_value, self.config.max_value)
        
        return v2
    
    def _calculate_pairwise_cosine_similarities(self, 
                                              preferences: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate cosine similarities between all agent pairs."""
        similarities = {}
        agent_ids = list(preferences.keys())
        
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids[i+1:], i+1):
                vec1 = np.array(preferences[agent1])
                vec2 = np.array(preferences[agent2])
                
                # Calculate cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities[f"{agent1}_vs_{agent2}"] = float(similarity)
        
        return similarities
    
    def calculate_utility(self, agent_id: str, allocation: Dict[str, List[int]], 
                         preferences: Dict[str, Any]) -> float:
        """
        Calculate agent utility for vector preferences.
        
        Utility = Σ(preference_i × item_received_i)
        """
        agent_prefs = preferences["agent_preferences"][agent_id]
        agent_items = allocation.get(agent_id, [])
        
        utility = sum(agent_prefs[item_id] for item_id in agent_items)
        return float(utility)
    
    def get_agent_preferences(self, agent_id: str, preferences: Dict[str, Any]) -> List[float]:
        """Get preference vector for a specific agent."""
        return preferences["agent_preferences"][agent_id]


class MatrixPreferenceSystem(BasePreferenceSystem):
    """
    Matrix-based preference system for cooperative/competitive scenarios.
    
    Each agent has an m×n preference matrix where entry (i,j) represents
    how much the agent values agent j receiving item i.
    """
    
    def generate_preferences(self) -> Dict[str, Any]:
        """Generate matrix preferences for all agents."""
        agent_ids = [f"agent_{i}" for i in range(self.config.n_agents)]
        preferences = {}
        
        for agent_id in agent_ids:
            # Generate m×n matrix for this agent
            matrix = self._generate_preference_matrix()
            preferences[agent_id] = matrix.tolist()
        
        return {
            "type": "matrix",
            "agent_preferences": preferences,
            "config": self.config.__dict__
        }
    
    def _generate_preference_matrix(self) -> np.ndarray:
        """Generate a single agent's preference matrix."""
        # Base matrix with random values
        matrix = self.rng.uniform(
            self.config.min_value,
            self.config.max_value,
            (self.config.m_items, self.config.n_agents)
        )
        
        # Apply cooperation factor
        if self.config.cooperation_factor is not None:
            # Self-preferences get full weight
            # Others get reduced weight based on cooperation factor
            for agent_idx in range(self.config.n_agents):
                for item_idx in range(self.config.m_items):
                    if agent_idx != 0:  # Assuming agent 0 is "self" - this is simplified
                        matrix[item_idx, agent_idx] *= self.config.cooperation_factor
        
        return matrix
    
    def calculate_utility(self, agent_id: str, allocation: Dict[str, List[int]], 
                         preferences: Dict[str, Any]) -> float:
        """
        Calculate agent utility for matrix preferences.
        
        Utility = Σ(preference_matrix[i,j] × item_i_received_by_agent_j)
        """
        agent_matrix = preferences["agent_preferences"][agent_id]
        utility = 0.0
        
        # For each item and each agent who received it
        for receiving_agent, items in allocation.items():
            agent_idx = int(receiving_agent.split('_')[1])  # Extract agent index
            for item_id in items:
                utility += agent_matrix[item_id][agent_idx]
        
        return float(utility)
    
    def get_agent_preferences(self, agent_id: str, preferences: Dict[str, Any]) -> List[List[float]]:
        """Get preference matrix for a specific agent."""
        return preferences["agent_preferences"][agent_id]


class PreferenceManager:
    """Manages preference systems for negotiation environments."""
    
    def __init__(self, config: PreferenceConfig):
        self.config = config
        
        if config.preference_type == PreferenceType.VECTOR:
            self.system = VectorPreferenceSystem(config)
        elif config.preference_type == PreferenceType.MATRIX:
            self.system = MatrixPreferenceSystem(config)
        else:
            raise ValueError(f"Unsupported preference type: {config.preference_type}")
    
    def generate_preferences(self) -> Dict[str, Any]:
        """Generate preferences using the configured system."""
        return self.system.generate_preferences()
    
    def calculate_utility(self, agent_id: str, allocation: Dict[str, List[int]], 
                         preferences: Dict[str, Any]) -> float:
        """Calculate utility for an agent."""
        return self.system.calculate_utility(agent_id, allocation, preferences)
    
    def calculate_all_utilities(self, allocation: Dict[str, List[int]], 
                               preferences: Dict[str, Any]) -> Dict[str, float]:
        """Calculate utilities for all agents."""
        utilities = {}
        for agent_id in allocation.keys():
            utilities[agent_id] = self.calculate_utility(agent_id, allocation, preferences)
        return utilities
    
    def get_agent_preferences(self, agent_id: str, preferences: Dict[str, Any]) -> Any:
        """Get preferences for a specific agent."""
        return self.system.get_agent_preferences(agent_id, preferences)
    
    def is_competitive_scenario(self, preferences: Dict[str, Any], 
                               threshold: float = 0.7) -> bool:
        """
        Determine if a scenario is competitive based on preference similarities.
        
        For vector preferences: High cosine similarity indicates competition
        For matrix preferences: Low cooperation factor indicates competition
        """
        if preferences["type"] == "vector":
            similarities = preferences.get("cosine_similarities", {})
            if similarities:
                avg_similarity = np.mean(list(similarities.values()))
                return avg_similarity > threshold
        elif preferences["type"] == "matrix":
            cooperation_factor = preferences["config"].get("cooperation_factor", 0.5)
            return cooperation_factor < 0.5
        
        return False
    
    def export_preferences(self, preferences: Dict[str, Any], filepath: str) -> None:
        """Export preferences to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(preferences, f, indent=2)
    
    @classmethod
    def load_preferences(cls, filepath: str) -> Dict[str, Any]:
        """Load preferences from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def create_competitive_preferences(m_items: int, n_agents: int, 
                                  cosine_similarity: float = 0.9,
                                  random_seed: Optional[int] = None,
                                  known_to_all: bool = False) -> PreferenceManager:
    """
    Factory function to create competitive vector preferences.
    
    Args:
        m_items: Number of items
        n_agents: Number of agents
        cosine_similarity: Target cosine similarity (higher = more competitive)
        random_seed: Random seed for reproducibility
        known_to_all: Whether preferences are common knowledge
    
    Returns:
        PreferenceManager configured for competitive scenarios
    """
    config = PreferenceConfig(
        preference_type=PreferenceType.VECTOR,
        m_items=m_items,
        n_agents=n_agents,
        target_cosine_similarity=cosine_similarity,
        random_seed=random_seed,
        known_to_all=known_to_all
    )
    
    return PreferenceManager(config)


def create_cooperative_preferences(m_items: int, n_agents: int,
                                  cooperation_factor: float = 0.8,
                                  random_seed: Optional[int] = None,
                                  known_to_all: bool = True) -> PreferenceManager:
    """
    Factory function to create cooperative matrix preferences.
    
    Args:
        m_items: Number of items
        n_agents: Number of agents
        cooperation_factor: How much agents value others' gains (higher = more cooperative)
        random_seed: Random seed for reproducibility
        known_to_all: Whether preferences are common knowledge
    
    Returns:
        PreferenceManager configured for cooperative scenarios
    """
    config = PreferenceConfig(
        preference_type=PreferenceType.MATRIX,
        m_items=m_items,
        n_agents=n_agents,
        cooperation_factor=cooperation_factor,
        random_seed=random_seed,
        known_to_all=known_to_all
    )
    
    return PreferenceManager(config)


# Utility functions for preference analysis
def analyze_preference_competition_level(preferences: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the competition level in a set of preferences."""
    analysis = {
        "preference_type": preferences["type"],
        "competition_level": "unknown"
    }
    
    if preferences["type"] == "vector":
        similarities = preferences.get("cosine_similarities", {})
        if similarities:
            similarities_list = list(similarities.values())
            analysis.update({
                "avg_cosine_similarity": np.mean(similarities_list),
                "min_cosine_similarity": np.min(similarities_list),
                "max_cosine_similarity": np.max(similarities_list),
                "std_cosine_similarity": np.std(similarities_list)
            })
            
            avg_sim = analysis["avg_cosine_similarity"]
            if avg_sim > 0.8:
                analysis["competition_level"] = "high"
            elif avg_sim > 0.5:
                analysis["competition_level"] = "medium"
            else:
                analysis["competition_level"] = "low"
    
    elif preferences["type"] == "matrix":
        cooperation_factor = preferences["config"].get("cooperation_factor", 0.5)
        analysis["cooperation_factor"] = cooperation_factor
        
        if cooperation_factor < 0.3:
            analysis["competition_level"] = "high"
        elif cooperation_factor < 0.7:
            analysis["competition_level"] = "medium"
        else:
            analysis["competition_level"] = "low"
    
    return analysis