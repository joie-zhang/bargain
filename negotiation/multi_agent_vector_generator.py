"""
Multi-Agent Random Vector Generator with Balanced Competition Levels

This module generates random preference vectors for 2+ agents where all pairwise
cosine similarities are approximately equal to the target competition level.
"""

import numpy as np
import logging
from itertools import combinations
from typing import List, Dict, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


class MultiAgentVectorGenerator:
    """Generate preference vectors for multiple agents with balanced pairwise similarities."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.rng = np.random.RandomState(random_seed)
    
    def generate_vectors_for_n_agents(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int = 5,
        max_utility: float = 100.0,
        integer_values: bool = True,
        tolerance: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Generate n vectors where all pairwise cosine similarities are close to target.
        
        Args:
            n_agents: Number of agents (vectors to generate)
            target_cosine: Target cosine similarity for all pairs (0.0 to 1.0)
            n_items: Number of items (vector dimension)
            max_utility: Maximum total utility (vectors sum to this)
            integer_values: Whether to round to integer values
            tolerance: Acceptable deviation from target cosine similarity
            
        Returns:
            Dictionary mapping agent_id to preference vector
        """
        
        if n_agents == 2:
            # Use existing pairwise generation
            try:
                from .random_vector_generator import RandomVectorGenerator
            except ImportError:
                from random_vector_generator import RandomVectorGenerator
            generator = RandomVectorGenerator(self.rng.randint(0, 10000))
            v1, v2 = generator.generate_vectors_with_cosine_similarity(
                target_cosine, n_items, max_utility, integer_values
            )
            return {
                "agent_0": v1,
                "agent_1": v2
            }
        
        return self._generate_slsqp_vectors(
            n_agents, target_cosine, n_items, max_utility, integer_values, tolerance
        )

    def _generate_slsqp_vectors(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        integer_values: bool,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """Generate 3+ vectors by optimizing all pairwise cosine targets with SLSQP."""
        if n_agents > n_items:
            logger.warning(
                "WARNING: N_AGENTS=%s IS GREATER THAN N_ITEMS=%s; "
                "EQUAL PAIRWISE COSINE TARGETS MAY BE GEOMETRICALLY INFEASIBLE. "
                "RUNNING SLSQP ANYWAY.",
                n_agents,
                n_items,
            )

        initial_vectors = self._initialize_slsqp_vectors(
            n_agents, target_cosine, n_items, max_utility
        )
        x0 = np.concatenate([initial_vectors[f"agent_{i}"] for i in range(n_agents)])
        pair_indices = list(combinations(range(n_agents), 2))

        def unpack(x: np.ndarray) -> List[np.ndarray]:
            return [x[i * n_items:(i + 1) * n_items] for i in range(n_agents)]

        def objective(x: np.ndarray) -> float:
            vectors = unpack(x)
            cosine_penalty = 0.0
            for i, j in pair_indices:
                similarity = self._cosine_similarity(vectors[i], vectors[j])
                cosine_penalty += (similarity - target_cosine) ** 2

            sum_penalty = sum((np.sum(v) - max_utility) ** 2 for v in vectors)
            return cosine_penalty * 1000.0 + sum_penalty * 100.0

        constraints = [
            {
                "type": "eq",
                "fun": lambda x, start=i * n_items, end=(i + 1) * n_items:
                    np.sum(x[start:end]) - max_utility,
            }
            for i in range(n_agents)
        ]
        bounds = [(0.0, max_utility) for _ in range(n_agents * n_items)]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 4000, "ftol": 1e-12},
        )

        vectors = {
            f"agent_{i}": result.x[i * n_items:(i + 1) * n_items]
            for i in range(n_agents)
        }
        final_error = self._calculate_pairwise_error(vectors, target_cosine)

        if not result.success:
            logger.warning(
                "WARNING: SLSQP DID NOT REPORT SUCCESS FOR N_AGENTS=%s, "
                "N_ITEMS=%s, TARGET_COSINE=%s; RETURNING BEST AVAILABLE VECTORS. "
                "STATUS=%s MESSAGE=%s FINAL_AVG_PAIRWISE_ERROR=%.6f",
                n_agents,
                n_items,
                target_cosine,
                result.status,
                result.message,
                final_error,
            )
        elif final_error > tolerance:
            logger.warning(
                "WARNING: SLSQP FINISHED BUT FINAL_AVG_PAIRWISE_ERROR=%.6f "
                "EXCEEDS TOLERANCE=%.6f FOR N_AGENTS=%s, N_ITEMS=%s, "
                "TARGET_COSINE=%s.",
                final_error,
                tolerance,
                n_agents,
                n_items,
                target_cosine,
            )

        if integer_values:
            for agent_id in vectors:
                vectors[agent_id] = self._round_to_integers_maintaining_sum(
                    vectors[agent_id], max_utility
                )

        return vectors

    def _initialize_slsqp_vectors(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float
    ) -> Dict[str, np.ndarray]:
        """
        Build an SLSQP starting point.

        Boundary targets get exact feasible starts when possible. Intermediate
        targets use random simplex starts so repeated experiments still sample
        different feasible optima.
        """
        if abs(target_cosine - 1.0) < 1e-12:
            base = self._generate_random_vector(n_items, max_utility, False)
            return {f"agent_{i}": base.copy() for i in range(n_agents)}

        if n_agents <= n_items and abs(target_cosine) < 1e-12:
            vectors = {}
            item_indices = self.rng.permutation(n_items)[:n_agents]
            for i, item_index in enumerate(item_indices):
                vector = np.zeros(n_items)
                vector[item_index] = max_utility
                vectors[f"agent_{i}"] = vector
            return vectors

        return {
            f"agent_{i}": self._generate_random_vector(n_items, max_utility, False)
            for i in range(n_agents)
        }
    
    def _calculate_pairwise_error(
        self,
        vectors: Dict[str, np.ndarray],
        target_cosine: float
    ) -> float:
        """Calculate total error in pairwise similarities."""
        
        sims = self._calculate_all_pairwise_similarities(vectors)
        errors = [abs(sim - target_cosine) for sim in sims.values()]
        return sum(errors) / len(errors) if errors else 0
    
    def _calculate_all_pairwise_similarities(
        self,
        vectors: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate all pairwise cosine similarities."""
        
        similarities = {}
        agent_ids = list(vectors.keys())
        
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids[i+1:], i+1):
                sim = self._cosine_similarity(vectors[agent1], vectors[agent2])
                similarities[f"{agent1}_vs_{agent2}"] = sim
        
        return similarities
    
    def _generate_random_vector(
        self,
        n_items: int,
        max_utility: float,
        integer_values: bool
    ) -> np.ndarray:
        """Generate a random vector that sums to max_utility."""
        
        # Generate random proportions using Dirichlet distribution
        proportions = self.rng.dirichlet(np.ones(n_items))
        vector = proportions * max_utility
        
        if integer_values:
            vector = self._round_to_integers_maintaining_sum(vector, max_utility)
        
        return vector
    
    def _round_to_integers_maintaining_sum(
        self,
        vector: np.ndarray,
        target_sum: float
    ) -> np.ndarray:
        """Round vector to integers while maintaining the target sum."""
        
        # Round down initially
        rounded = np.floor(vector).astype(int)
        remainder = int(target_sum - np.sum(rounded))
        
        if remainder > 0:
            # Add remainder to items with largest fractional parts
            fractional_parts = vector - rounded
            indices = np.argsort(fractional_parts)[-remainder:]
            rounded[indices] += 1
        
        return rounded.astype(float)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)


# Example usage and testing
if __name__ == "__main__":
    generator = MultiAgentVectorGenerator(random_seed=42)
    
    print("="*60)
    print("Testing Multi-Agent Vector Generation")
    print("="*60)
    
    # Test with 2 agents (baseline case)
    print("\n2 Agents with target cosine similarity = 0.75:")
    vectors_2 = generator.generate_vectors_for_n_agents(
        n_agents=2,
        target_cosine=0.75,
        n_items=5,
        max_utility=100.0,
        integer_values=True
    )
    
    for agent_id, vector in vectors_2.items():
        print(f"  {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.1f})")
    
    # Calculate actual pairwise similarities
    print("\n  Pairwise similarities:")
    sims = generator._calculate_all_pairwise_similarities(vectors_2)
    for pair, sim in sims.items():
        print(f"    {pair}: {sim:.4f}")
    
    # Test with 3 agents
    print("\n3 Agents with target cosine similarity = 0.75:")
    vectors_3 = generator.generate_vectors_for_n_agents(
        n_agents=3,
        target_cosine=0.75,
        n_items=5,
        max_utility=100.0,
        integer_values=True
    )
    
    for agent_id, vector in vectors_3.items():
        print(f"  {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.1f})")
    
    print("\n  Pairwise similarities:")
    sims = generator._calculate_all_pairwise_similarities(vectors_3)
    for pair, sim in sims.items():
        print(f"    {pair}: {sim:.4f}")
    
    # Test with 4 agents
    print("\n4 Agents with target cosine similarity = 0.5:")
    vectors_4 = generator.generate_vectors_for_n_agents(
        n_agents=4,
        target_cosine=0.5,
        n_items=5,
        max_utility=100.0,
        integer_values=True
    )
    
    for agent_id, vector in vectors_4.items():
        print(f"  {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.1f})")
    
    print("\n  Pairwise similarities:")
    sims = generator._calculate_all_pairwise_similarities(vectors_4)
    for pair, sim in sims.items():
        print(f"    {pair}: {sim:.4f}")
    
    # Test with 5 agents
    print("\n5 Agents with target cosine similarity = 0.25:")
    vectors_5 = generator.generate_vectors_for_n_agents(
        n_agents=5,
        target_cosine=0.25,
        n_items=5,
        max_utility=100.0,
        integer_values=True
    )
    
    for agent_id, vector in vectors_5.items():
        print(f"  {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.1f})")
    
    print("\n  Pairwise similarities:")
    sims = generator._calculate_all_pairwise_similarities(vectors_5)
    for pair, sim in sims.items():
        print(f"    {pair}: {sim:.4f}")
