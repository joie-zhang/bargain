"""
Multi-Agent Random Vector Generator with Balanced Competition Levels

This module generates random preference vectors for 2+ agents where all pairwise
cosine similarities are approximately equal to the target competition level.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


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
        
        if n_agents == 3:
            return self._generate_three_agent_vectors(
                target_cosine, n_items, max_utility, integer_values, tolerance
            )
        
        # For 4+ agents, use iterative approach
        return self._generate_many_agent_vectors(
            n_agents, target_cosine, n_items, max_utility, integer_values, tolerance
        )
    
    def _generate_three_agent_vectors(
        self,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        integer_values: bool,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate 3 vectors with all pairwise similarities close to target.
        
        For 3 vectors to have equal pairwise cosine similarities, we use a 
        geometric approach placing them symmetrically in the vector space.
        """
        
        # Special case: identical vectors (cosine = 1.0)
        if abs(target_cosine - 1.0) < 0.01:
            v = self._generate_random_vector(n_items, max_utility, integer_values)
            return {
                "agent_0": v.copy(),
                "agent_1": v.copy(),
                "agent_2": v.copy()
            }
        
        # Use optimization to find 3 vectors with target pairwise similarities
        best_vectors = None
        best_error = float('inf')
        
        for attempt in range(10):
            # Initialize with random vectors
            vectors = self._optimize_three_vectors(
                target_cosine, n_items, max_utility
            )
            
            # Calculate error
            error = self._calculate_pairwise_error(vectors, target_cosine)
            
            if error < best_error:
                best_error = error
                best_vectors = vectors
                
                if error < tolerance:
                    break
        
        # Round to integers if requested
        if integer_values and best_vectors is not None:
            for agent_id in best_vectors:
                best_vectors[agent_id] = self._round_to_integers_maintaining_sum(
                    best_vectors[agent_id], max_utility
                )
        
        return best_vectors
    
    def _optimize_three_vectors(
        self,
        target_cosine: float,
        n_items: int,
        max_utility: float
    ) -> Dict[str, np.ndarray]:
        """Use optimization to find 3 vectors with target pairwise similarities."""
        
        # Start with geometric initialization
        vectors = self._geometric_initialization_three(target_cosine, n_items, max_utility)
        
        # Flatten for optimization
        x0 = np.concatenate([vectors["agent_0"], vectors["agent_1"], vectors["agent_2"]])
        
        def objective(x):
            v1 = x[:n_items]
            v2 = x[n_items:2*n_items]
            v3 = x[2*n_items:3*n_items]
            
            # Ensure non-negative values
            if np.any(x < 0):
                return 1e10
            
            # Calculate all pairwise cosine similarities
            cos_12 = self._cosine_similarity(v1, v2)
            cos_13 = self._cosine_similarity(v1, v3)
            cos_23 = self._cosine_similarity(v2, v3)
            
            # Penalty for deviation from target
            cos_penalty = ((cos_12 - target_cosine) ** 2 + 
                          (cos_13 - target_cosine) ** 2 + 
                          (cos_23 - target_cosine) ** 2) * 1000
            
            # Penalty for sum deviation
            sum_penalty = ((np.sum(v1) - max_utility) ** 2 + 
                          (np.sum(v2) - max_utility) ** 2 + 
                          (np.sum(v3) - max_utility) ** 2) * 100
            
            return cos_penalty + sum_penalty
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x[:n_items]) - max_utility},
            {'type': 'eq', 'fun': lambda x: np.sum(x[n_items:2*n_items]) - max_utility},
            {'type': 'eq', 'fun': lambda x: np.sum(x[2*n_items:3*n_items]) - max_utility}
        ]
        
        bounds = [(0, max_utility) for _ in range(3 * n_items)]
        
        result = minimize(objective, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 2000, 'ftol': 1e-9})
        
        # Extract optimized vectors
        v1 = result.x[:n_items]
        v2 = result.x[n_items:2*n_items]
        v3 = result.x[2*n_items:3*n_items]
        
        return {
            "agent_0": v1,
            "agent_1": v2,
            "agent_2": v3
        }
    
    def _geometric_initialization_three(
        self,
        target_cosine: float,
        n_items: int,
        max_utility: float
    ) -> Dict[str, np.ndarray]:
        """
        Initialize 3 vectors geometrically for target cosine similarity.
        
        For 3 vectors with equal pairwise cosine similarities,
        they should form an equilateral triangle in the vector space.
        """
        
        # Generate first vector randomly
        v1 = self._generate_random_vector(n_items, max_utility, False)
        
        # For an equilateral triangle configuration:
        # All pairwise angles should be: arccos(target_cosine)
        angle = np.arccos(np.clip(target_cosine, -1, 1))
        
        # Generate v2 at angle from v1
        v1_norm = v1 / np.linalg.norm(v1)
        
        # Random orthogonal vector
        random_vec = self.rng.randn(n_items)
        v_orth = random_vec - np.dot(random_vec, v1_norm) * v1_norm
        v_orth = v_orth / np.linalg.norm(v_orth)
        
        # v2 at angle from v1
        v2_direction = np.cos(angle) * v1_norm + np.sin(angle) * v_orth
        v2 = v2_direction * np.linalg.norm(v1)
        v2 = np.abs(v2) * (max_utility / np.sum(np.abs(v2)))
        
        # Generate v3 to form triangle
        # v3 should be at angle from both v1 and v2
        # This is approximated by rotating in a different plane
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Find a vector orthogonal to both v1 and v2
        random_vec2 = self.rng.randn(n_items)
        v_orth2 = random_vec2 - np.dot(random_vec2, v1_norm) * v1_norm - np.dot(random_vec2, v2_norm) * v2_norm
        
        if np.linalg.norm(v_orth2) > 0:
            v_orth2 = v_orth2 / np.linalg.norm(v_orth2)
        else:
            # Fallback if vectors are too aligned
            v_orth2 = self.rng.randn(n_items)
            v_orth2 = v_orth2 / np.linalg.norm(v_orth2)
        
        # Construct v3
        v3_direction = np.cos(angle) * v1_norm + np.sin(angle) * v_orth2
        v3 = v3_direction * np.linalg.norm(v1)
        v3 = np.abs(v3) * (max_utility / np.sum(np.abs(v3)))
        
        return {
            "agent_0": v1,
            "agent_1": v2,
            "agent_2": v3
        }
    
    def _generate_many_agent_vectors(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        integer_values: bool,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate vectors for 4+ agents using improved optimization.
        
        For many agents, we use a combination of geometric initialization
        and gradient-based optimization for better convergence.
        """
        
        # Better initialization using geometric approach
        vectors = self._geometric_initialization_many(
            n_agents, target_cosine, n_items, max_utility
        )
        
        # Multi-stage optimization with adaptive learning rate
        best_vectors = vectors.copy()
        best_error = self._calculate_pairwise_error(vectors, target_cosine)
        
        # Stage 1: Coarse adjustment with higher learning rate
        learning_rate = 0.5
        momentum = 0.9
        velocities = {agent_id: np.zeros_like(v) for agent_id, v in vectors.items()}
        
        for iteration in range(100):
            # Calculate current error
            current_error = self._calculate_pairwise_error(vectors, target_cosine)
            
            if current_error < best_error:
                best_error = current_error
                best_vectors = {k: v.copy() for k, v in vectors.items()}
            
            if current_error < tolerance:
                break
            
            # Adaptive learning rate decay
            if iteration > 30:
                learning_rate = 0.2
            if iteration > 60:
                learning_rate = 0.1
            
            # Update with momentum
            vectors, velocities = self._adjust_vectors_with_momentum(
                vectors, velocities, target_cosine, max_utility, 
                learning_rate, momentum
            )
            
            # Occasionally restart from best if stuck
            if iteration % 20 == 19 and current_error > best_error * 1.5:
                vectors = {k: v.copy() for k, v in best_vectors.items()}
                velocities = {agent_id: np.zeros_like(v) for agent_id, v in vectors.items()}
        
        # Stage 2: Fine-tuning with constraint satisfaction
        vectors = self._fine_tune_vectors(
            best_vectors, target_cosine, max_utility, tolerance
        )
        
        # Round to integers if requested
        if integer_values:
            for agent_id in vectors:
                vectors[agent_id] = self._round_to_integers_maintaining_sum(
                    vectors[agent_id], max_utility
                )
        
        return vectors
    
    def _adjust_vectors_towards_target(
        self,
        vectors: Dict[str, np.ndarray],
        target_cosine: float,
        max_utility: float,
        learning_rate: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """Adjust vectors to move pairwise similarities towards target."""
        
        agent_ids = list(vectors.keys())
        n_agents = len(agent_ids)
        
        # Calculate gradients for each vector
        gradients = {agent_id: np.zeros_like(vectors[agent_id]) for agent_id in agent_ids}
        
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids[i+1:], i+1):
                v1 = vectors[agent1]
                v2 = vectors[agent2]
                
                current_sim = self._cosine_similarity(v1, v2)
                error = target_cosine - current_sim
                
                # Gradient to increase/decrease similarity
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    # Gradient for v1
                    grad1 = error * (v2 / (norm1 * norm2) - 
                                   (np.dot(v1, v2) * v1) / (norm1**3 * norm2))
                    gradients[agent1] += grad1
                    
                    # Gradient for v2
                    grad2 = error * (v1 / (norm1 * norm2) - 
                                   (np.dot(v1, v2) * v2) / (norm1 * norm2**3))
                    gradients[agent2] += grad2
        
        # Update vectors
        new_vectors = {}
        for agent_id in agent_ids:
            # Update with gradient
            new_v = vectors[agent_id] + learning_rate * gradients[agent_id]
            
            # Ensure non-negative
            new_v = np.maximum(new_v, 0)
            
            # Normalize to maintain sum
            if np.sum(new_v) > 0:
                new_v = new_v * (max_utility / np.sum(new_v))
            else:
                new_v = self._generate_random_vector(n_items, max_utility, False)
            
            new_vectors[agent_id] = new_v
        
        return new_vectors
    
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
    
    def _geometric_initialization_many(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float
    ) -> Dict[str, np.ndarray]:
        """
        Initialize many vectors using geometric approach for better starting point.
        Places vectors approximately uniformly in the vector space.
        """
        vectors = {}
        
        # Generate first vector randomly
        v0 = self._generate_random_vector(n_items, max_utility, False)
        vectors["agent_0"] = v0
        
        # For subsequent vectors, try to maintain target similarity
        for i in range(1, n_agents):
            # Start with a random vector
            v_new = self._generate_random_vector(n_items, max_utility, False)
            
            # Adjust to have approximately target similarity with existing vectors
            for j in range(min(5, i)):  # Adjust against up to 5 previous vectors
                existing_v = vectors[f"agent_{j}"]
                current_sim = self._cosine_similarity(v_new, existing_v)
                
                # Move towards target similarity
                if abs(current_sim - target_cosine) > 0.1:
                    # Simple adjustment towards target
                    adjustment = 0.3 * (target_cosine - current_sim)
                    v_new = v_new + adjustment * existing_v
                    v_new = np.maximum(v_new, 0)  # Keep non-negative
                    
                    # Renormalize to maintain sum
                    if np.sum(v_new) > 0:
                        v_new = v_new * (max_utility / np.sum(v_new))
            
            vectors[f"agent_{i}"] = v_new
        
        return vectors
    
    def _adjust_vectors_with_momentum(
        self,
        vectors: Dict[str, np.ndarray],
        velocities: Dict[str, np.ndarray],
        target_cosine: float,
        max_utility: float,
        learning_rate: float,
        momentum: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Adjust vectors using gradient descent with momentum for faster convergence.
        """
        agent_ids = list(vectors.keys())
        n_agents = len(agent_ids)
        
        # Calculate gradients
        gradients = {agent_id: np.zeros_like(vectors[agent_id]) for agent_id in agent_ids}
        
        for i, agent1 in enumerate(agent_ids):
            for j, agent2 in enumerate(agent_ids[i+1:], i+1):
                v1 = vectors[agent1]
                v2 = vectors[agent2]
                
                current_sim = self._cosine_similarity(v1, v2)
                error = target_cosine - current_sim
                
                # More robust gradient calculation
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 1e-6 and norm2 > 1e-6:
                    # Gradient for v1
                    grad1 = error * (v2 / (norm1 * norm2) - 
                                   (np.dot(v1, v2) * v1) / (norm1**3 * norm2))
                    gradients[agent1] += grad1 / (n_agents - 1)  # Normalize by number of pairs
                    
                    # Gradient for v2
                    grad2 = error * (v1 / (norm1 * norm2) - 
                                   (np.dot(v1, v2) * v2) / (norm1 * norm2**3))
                    gradients[agent2] += grad2 / (n_agents - 1)
        
        # Update velocities and vectors with momentum
        new_vectors = {}
        new_velocities = {}
        
        for agent_id in agent_ids:
            # Update velocity with momentum
            new_velocities[agent_id] = (momentum * velocities[agent_id] + 
                                       learning_rate * gradients[agent_id])
            
            # Update vector
            new_v = vectors[agent_id] + new_velocities[agent_id]
            
            # Project back to valid space
            new_v = np.maximum(new_v, 0)  # Non-negative
            
            # Renormalize to maintain sum
            if np.sum(new_v) > 0:
                new_v = new_v * (max_utility / np.sum(new_v))
            else:
                new_v = self._generate_random_vector(len(new_v), max_utility, False)
            
            new_vectors[agent_id] = new_v
        
        return new_vectors, new_velocities
    
    def _fine_tune_vectors(
        self,
        vectors: Dict[str, np.ndarray],
        target_cosine: float,
        max_utility: float,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Fine-tune vectors using smaller adjustments for precise convergence.
        """
        agent_ids = list(vectors.keys())
        refined_vectors = {k: v.copy() for k, v in vectors.items()}
        
        for iteration in range(20):
            current_error = self._calculate_pairwise_error(refined_vectors, target_cosine)
            if current_error < tolerance:
                break
            
            # Small random perturbations followed by projection
            for agent_id in agent_ids:
                v = refined_vectors[agent_id]
                
                # Small random perturbation
                perturbation = self.rng.randn(len(v)) * 0.5
                v_new = v + perturbation
                
                # Project to valid space
                v_new = np.maximum(v_new, 0)
                if np.sum(v_new) > 0:
                    v_new = v_new * (max_utility / np.sum(v_new))
                
                # Check if this improves the error
                refined_vectors[agent_id] = v_new
                new_error = self._calculate_pairwise_error(refined_vectors, target_cosine)
                
                if new_error > current_error:
                    # Revert if worse
                    refined_vectors[agent_id] = v
        
        return refined_vectors
    
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