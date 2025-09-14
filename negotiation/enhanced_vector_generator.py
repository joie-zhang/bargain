"""
Enhanced Multi-Agent Preference Vector Generator with Improved Algorithms

This module provides advanced algorithms for generating preference vectors
for any number of agents (2+) with precise control over pairwise similarities.
Uses multiple strategies including spectral methods, convex optimization,
and evolutionary algorithms.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')


class EnhancedVectorGenerator:
    """
    Advanced preference vector generator supporting any number of agents
    with multiple generation strategies for different scenarios.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.rng = np.random.RandomState(random_seed)
        np.random.seed(random_seed)
    
    def generate_vectors(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int = 5,
        max_utility: float = 100.0,
        integer_values: bool = True,
        tolerance: float = 0.05,
        method: str = "auto"
    ) -> Dict[str, np.ndarray]:
        """
        Generate n vectors with all pairwise cosine similarities close to target.
        
        Args:
            n_agents: Number of agents (2+)
            target_cosine: Target cosine similarity (-1.0 to 1.0)
            n_items: Number of items (vector dimension)
            max_utility: Maximum total utility (vectors sum to this)
            integer_values: Whether to round to integer values
            tolerance: Acceptable deviation from target cosine
            method: Generation method ("auto", "spectral", "convex", "evolutionary", "hybrid")
            
        Returns:
            Dictionary mapping agent_id to preference vector
        """
        
        # Validate inputs
        if n_agents < 2:
            raise ValueError("Need at least 2 agents")
        if not -1 <= target_cosine <= 1:
            raise ValueError("Target cosine must be in [-1, 1]")
        if n_items < 2:
            raise ValueError("Need at least 2 items")
        
        # Select method automatically if needed
        if method == "auto":
            method = self._select_best_method(n_agents, target_cosine, n_items)
        
        # Generate vectors using selected method
        if method == "spectral":
            vectors = self._generate_spectral(
                n_agents, target_cosine, n_items, max_utility
            )
        elif method == "convex":
            vectors = self._generate_advanced_optimization(
                n_agents, target_cosine, n_items, max_utility, tolerance
            )
        elif method == "evolutionary":
            vectors = self._generate_evolutionary(
                n_agents, target_cosine, n_items, max_utility, tolerance
            )
        elif method == "hybrid":
            vectors = self._generate_hybrid(
                n_agents, target_cosine, n_items, max_utility, tolerance
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Post-processing: ensure constraints and optionally round
        vectors = self._post_process_vectors(
            vectors, max_utility, integer_values, target_cosine, tolerance
        )
        
        return vectors
    
    def _select_best_method(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int
    ) -> str:
        """
        Automatically select the best generation method based on problem characteristics.
        """
        
        # For special cases
        if abs(target_cosine - 1.0) < 0.01:  # Identical vectors
            return "spectral"
        if abs(target_cosine + 1.0) < 0.01:  # Opposite vectors (only works for 2 agents)
            if n_agents == 2:
                return "spectral"
            else:
                return "evolutionary"
        
        # Based on problem size and characteristics
        if n_agents <= 3 and n_items <= 10:
            return "convex"  # Exact solution for small problems
        elif n_agents <= 5 and n_items <= 20:
            return "hybrid"  # Combination approach for medium problems
        elif n_agents <= 10:
            return "evolutionary"  # Robust for larger agent counts
        else:
            return "spectral"  # Scalable for many agents
    
    def _generate_spectral(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate vectors using spectral decomposition method.
        This creates vectors from eigenvectors of a carefully designed matrix.
        """
        
        # Create target similarity matrix
        S = np.full((n_agents, n_agents), target_cosine)
        np.fill_diagonal(S, 1.0)
        
        # Ensure positive semi-definite by eigenvalue correction
        eigenvalues, eigenvectors = eigh(S)
        eigenvalues = np.maximum(eigenvalues, 0)
        S = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        # Generate base vectors in high-dimensional space
        if n_items >= n_agents:
            # Direct mapping possible
            base_vectors = self._cholesky_vectors(S, n_items)
        else:
            # Need dimensionality reduction
            base_vectors = self._low_rank_approximation(S, n_items)
        
        # Convert to preference vectors (ensure non-negative and sum constraint)
        vectors = {}
        for i in range(n_agents):
            v = base_vectors[i]
            
            # Make non-negative using soft rectification
            v = np.abs(v) + 0.1  # Small offset to avoid zeros
            
            # Normalize to sum
            v = v * (max_utility / np.sum(v))
            
            vectors[f"agent_{i}"] = v
        
        return vectors
    
    def _cholesky_vectors(
        self,
        similarity_matrix: np.ndarray,
        n_items: int
    ) -> np.ndarray:
        """
        Generate vectors using Cholesky decomposition when n_items >= n_agents.
        """
        n_agents = similarity_matrix.shape[0]
        
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(similarity_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue correction
            eigenvalues, eigenvectors = eigh(similarity_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            similarity_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            L = np.linalg.cholesky(similarity_matrix + 1e-10 * np.eye(n_agents))
        
        # Create base vectors
        vectors = np.zeros((n_agents, n_items))
        vectors[:, :n_agents] = L
        
        # Add random components for remaining dimensions
        if n_items > n_agents:
            extra_dims = self.rng.randn(n_agents, n_items - n_agents) * 0.1
            vectors[:, n_agents:] = extra_dims
        
        return vectors
    
    def _low_rank_approximation(
        self,
        similarity_matrix: np.ndarray,
        n_items: int
    ) -> np.ndarray:
        """
        Generate vectors using low-rank approximation when n_items < n_agents.
        """
        n_agents = similarity_matrix.shape[0]
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(similarity_matrix)
        
        # Keep top n_items eigenvalues
        idx = np.argsort(eigenvalues)[-n_items:]
        top_eigenvalues = eigenvalues[idx]
        top_eigenvectors = eigenvectors[:, idx]
        
        # Generate vectors
        vectors = top_eigenvectors @ np.diag(np.sqrt(np.maximum(top_eigenvalues, 0)))
        
        return vectors
    
    def _generate_advanced_optimization(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate vectors using advanced scipy optimization.
        This provides good solutions without external dependencies.
        """
        
        # Initialize with spectral method for good starting point
        initial_vectors = self._generate_spectral(n_agents, target_cosine, n_items, max_utility)
        x0 = np.concatenate([initial_vectors[f"agent_{i}"] for i in range(n_agents)])
        
        def objective(x):
            """Objective function combining cosine similarity and constraints."""
            vectors = x.reshape(n_agents, n_items)
            
            # Penalty for negative values
            neg_penalty = np.sum(np.maximum(-vectors, 0)) * 1000
            
            # Penalty for sum constraint violation
            sum_penalty = 0
            for i in range(n_agents):
                sum_penalty += (np.sum(vectors[i]) - max_utility) ** 2
            
            # Penalty for cosine similarity deviation
            cos_penalty = 0
            n_pairs = 0
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    cos_sim = self._cosine_similarity(vectors[i], vectors[j])
                    cos_penalty += (cos_sim - target_cosine) ** 2
                    n_pairs += 1
            
            if n_pairs > 0:
                cos_penalty /= n_pairs
            
            return 1000 * cos_penalty + 100 * sum_penalty + neg_penalty
        
        # Constraints for sum = max_utility
        constraints = []
        for i in range(n_agents):
            constraints.append({
                'type': 'eq',
                'fun': lambda x, idx=i: np.sum(x.reshape(n_agents, n_items)[idx]) - max_utility
            })
        
        # Bounds to ensure non-negative values
        bounds = [(0, max_utility) for _ in range(n_agents * n_items)]
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        # Extract solution
        solution = result.x.reshape(n_agents, n_items)
        vectors = {}
        for i in range(n_agents):
            v = solution[i]
            # Ensure constraints are exactly satisfied
            v = np.maximum(v, 0)
            v = v * (max_utility / np.sum(v)) if np.sum(v) > 0 else v
            vectors[f"agent_{i}"] = v
        
        return vectors
    
    def _generate_evolutionary(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Generate vectors using differential evolution algorithm.
        This is robust for difficult cases but can be slower.
        """
        
        # Flatten all vectors into single array for optimization
        n_vars = n_agents * n_items
        
        def objective(x):
            """Objective function to minimize."""
            vectors = x.reshape(n_agents, n_items)
            
            # Penalty for negative values
            if np.any(vectors < 0):
                return 1e10
            
            # Penalty for sum constraint violation
            sum_penalty = sum((np.sum(vectors[i]) - max_utility) ** 2 for i in range(n_agents))
            
            # Penalty for cosine similarity deviation
            cos_penalty = 0
            count = 0
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    cos_sim = self._cosine_similarity(vectors[i], vectors[j])
                    cos_penalty += (cos_sim - target_cosine) ** 2
                    count += 1
            
            if count > 0:
                cos_penalty /= count
            
            return 1000 * cos_penalty + 100 * sum_penalty
        
        # Set bounds
        bounds = [(0, max_utility) for _ in range(n_vars)]
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=tolerance,
            seed=self.rng.randint(0, 10000),
            workers=1,
            disp=False
        )
        
        # Extract solution
        solution = result.x.reshape(n_agents, n_items)
        vectors = {}
        for i in range(n_agents):
            # Ensure exact sum constraint
            v = solution[i]
            v = v * (max_utility / np.sum(v))
            vectors[f"agent_{i}"] = v
        
        return vectors
    
    def _generate_hybrid(
        self,
        n_agents: int,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Hybrid approach: Initialize with spectral method, refine with local optimization.
        """
        
        # Start with spectral initialization
        vectors = self._generate_spectral(n_agents, target_cosine, n_items, max_utility)
        
        # Convert to numpy array for optimization
        agent_ids = sorted(vectors.keys())
        current = np.array([vectors[aid] for aid in agent_ids])
        
        # Local refinement using gradient descent
        for iteration in range(50):
            # Calculate current error
            error = self._calculate_pairwise_error_array(current, target_cosine)
            if error < tolerance:
                break
            
            # Compute gradient
            gradient = self._compute_gradient(current, target_cosine, max_utility)
            
            # Adaptive learning rate
            learning_rate = 0.1 / (1 + iteration * 0.05)
            
            # Update with gradient
            current = current - learning_rate * gradient
            
            # Project back to feasible region
            current = np.maximum(current, 0)  # Non-negative
            for i in range(n_agents):
                if np.sum(current[i]) > 0:
                    current[i] = current[i] * (max_utility / np.sum(current[i]))
        
        # Convert back to dictionary
        refined_vectors = {}
        for i, aid in enumerate(agent_ids):
            refined_vectors[aid] = current[i]
        
        return refined_vectors
    
    def _compute_gradient(
        self,
        vectors: np.ndarray,
        target_cosine: float,
        max_utility: float
    ) -> np.ndarray:
        """
        Compute gradient for cosine similarity objective.
        """
        n_agents, n_items = vectors.shape
        gradient = np.zeros_like(vectors)
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    continue
                
                v_i = vectors[i]
                v_j = vectors[j]
                
                # Current cosine similarity
                cos_sim = self._cosine_similarity(v_i, v_j)
                error = cos_sim - target_cosine
                
                # Gradient of cosine similarity w.r.t v_i
                norm_i = np.linalg.norm(v_i)
                norm_j = np.linalg.norm(v_j)
                
                if norm_i > 1e-6 and norm_j > 1e-6:
                    grad_cos = (v_j / (norm_i * norm_j) - 
                               (np.dot(v_i, v_j) * v_i) / (norm_i ** 3 * norm_j))
                    gradient[i] += error * grad_cos
        
        return gradient
    
    def _post_process_vectors(
        self,
        vectors: Dict[str, np.ndarray],
        max_utility: float,
        integer_values: bool,
        target_cosine: float,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Post-process vectors to ensure all constraints are satisfied.
        """
        
        processed = {}
        
        for agent_id, vector in vectors.items():
            v = vector.copy()
            
            # Ensure non-negative
            v = np.maximum(v, 0)
            
            # Ensure sum constraint
            if np.sum(v) > 0:
                v = v * (max_utility / np.sum(v))
            else:
                # Fallback to uniform distribution
                v = np.ones_like(v) * (max_utility / len(v))
            
            processed[agent_id] = v
        
        # Final adjustment pass to improve similarities (only if not using integers)
        if not integer_values:
            processed = self._final_similarity_adjustment(
                processed, target_cosine, max_utility, tolerance
            )
        
        # Round to integers AFTER all adjustments if requested
        if integer_values:
            for agent_id in processed:
                processed[agent_id] = self._round_maintaining_sum(
                    processed[agent_id], max_utility
                )
        
        return processed
    
    def _final_similarity_adjustment(
        self,
        vectors: Dict[str, np.ndarray],
        target_cosine: float,
        max_utility: float,
        tolerance: float
    ) -> Dict[str, np.ndarray]:
        """
        Final adjustment to improve pairwise similarities.
        """
        
        agent_ids = list(vectors.keys())
        n_agents = len(agent_ids)
        
        # Quick adjustment iterations
        for _ in range(10):
            error = self._calculate_pairwise_error(vectors, target_cosine)
            if error < tolerance:
                break
            
            # Adjust pairs that deviate most
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    v_i = vectors[agent_ids[i]]
                    v_j = vectors[agent_ids[j]]
                    
                    current_sim = self._cosine_similarity(v_i, v_j)
                    deviation = current_sim - target_cosine
                    
                    if abs(deviation) > tolerance:
                        # Small adjustment towards target
                        if deviation > 0:  # Too similar, make more different
                            noise = self.rng.randn(len(v_i)) * 0.5
                            v_i = v_i + noise
                            v_j = v_j - noise
                        else:  # Too different, make more similar
                            mean = (v_i + v_j) / 2
                            v_i = 0.9 * v_i + 0.1 * mean
                            v_j = 0.9 * v_j + 0.1 * mean
                        
                        # Maintain constraints
                        v_i = np.maximum(v_i, 0)
                        v_j = np.maximum(v_j, 0)
                        v_i = v_i * (max_utility / np.sum(v_i)) if np.sum(v_i) > 0 else v_i
                        v_j = v_j * (max_utility / np.sum(v_j)) if np.sum(v_j) > 0 else v_j
                        
                        vectors[agent_ids[i]] = v_i
                        vectors[agent_ids[j]] = v_j
        
        return vectors
    
    def _round_maintaining_sum(
        self,
        vector: np.ndarray,
        target_sum: float
    ) -> np.ndarray:
        """
        Round vector to integers while maintaining exact sum.
        """
        
        # Round down
        rounded = np.floor(vector).astype(int)
        remainder = int(target_sum - np.sum(rounded))
        
        if remainder > 0:
            # Add remainder to elements with largest fractional parts
            fractional = vector - rounded
            indices = np.argsort(fractional)[-remainder:]
            rounded[indices] += 1
        elif remainder < 0:
            # Remove from elements with smallest fractional parts
            fractional = vector - rounded
            indices = np.argsort(fractional)[:abs(remainder)]
            rounded[indices] -= 1
        
        return rounded.astype(float)
    
    def _calculate_pairwise_error(
        self,
        vectors: Dict[str, np.ndarray],
        target_cosine: float
    ) -> float:
        """Calculate average absolute deviation from target cosine similarity."""
        
        agent_ids = list(vectors.keys())
        errors = []
        
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                cos_sim = self._cosine_similarity(
                    vectors[agent_ids[i]],
                    vectors[agent_ids[j]]
                )
                errors.append(abs(cos_sim - target_cosine))
        
        return np.mean(errors) if errors else 0
    
    def _calculate_pairwise_error_array(
        self,
        vectors: np.ndarray,
        target_cosine: float
    ) -> float:
        """Calculate error for numpy array of vectors."""
        
        n_agents = vectors.shape[0]
        errors = []
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                cos_sim = self._cosine_similarity(vectors[i], vectors[j])
                errors.append(abs(cos_sim - target_cosine))
        
        return np.mean(errors) if errors else 0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0
        
        return np.clip(dot_product / (norm_a * norm_b), -1, 1)
    
    def analyze_vectors(
        self,
        vectors: Dict[str, np.ndarray],
        target_cosine: Optional[float] = None
    ) -> Dict:
        """
        Analyze generated vectors and return statistics.
        """
        
        agent_ids = list(vectors.keys())
        n_agents = len(agent_ids)
        
        # Calculate all pairwise similarities
        similarities = []
        sim_matrix = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(n_agents):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim = self._cosine_similarity(
                        vectors[agent_ids[i]],
                        vectors[agent_ids[j]]
                    )
                    sim_matrix[i, j] = sim
                    if i < j:
                        similarities.append(sim)
        
        stats = {
            "n_agents": n_agents,
            "n_items": len(vectors[agent_ids[0]]),
            "pairwise_similarities": similarities,
            "similarity_matrix": sim_matrix,
            "mean_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities)
        }
        
        if target_cosine is not None:
            errors = [abs(sim - target_cosine) for sim in similarities]
            stats["target_cosine"] = target_cosine
            stats["mean_error"] = np.mean(errors)
            stats["max_error"] = np.max(errors)
            stats["success_rate"] = np.mean([e < 0.05 for e in errors])
        
        return stats


# Convenience function for backward compatibility
def generate_preference_vectors(
    n_agents: int,
    competition_level: float,
    n_items: int = 5,
    max_utility: float = 100.0,
    integer_values: bool = True,
    random_seed: Optional[int] = None,
    method: str = "auto"
) -> Dict[str, np.ndarray]:
    """
    Generate preference vectors for multiple agents with specified competition level.
    
    Args:
        n_agents: Number of agents
        competition_level: Competition level (0.0 = competitive, 1.0 = aligned)
        n_items: Number of items
        max_utility: Maximum total utility
        integer_values: Whether to use integer values
        random_seed: Random seed for reproducibility
        method: Generation method
    
    Returns:
        Dictionary mapping agent_id to preference vector
    """
    
    # Convert competition level to target cosine similarity
    target_cosine = competition_level
    
    generator = EnhancedVectorGenerator(random_seed)
    return generator.generate_vectors(
        n_agents=n_agents,
        target_cosine=target_cosine,
        n_items=n_items,
        max_utility=max_utility,
        integer_values=integer_values,
        method=method
    )


if __name__ == "__main__":
    # Test the enhanced generator
    print("=" * 80)
    print("Testing Enhanced Multi-Agent Vector Generator")
    print("=" * 80)
    
    generator = EnhancedVectorGenerator(random_seed=42)
    
    # Test different agent counts and competition levels
    test_cases = [
        (4, 0.5, "4 agents with moderate competition (cos=0.5)"),
        (5, 0.25, "5 agents with high competition (cos=0.25)"),
        (6, 0.75, "6 agents with low competition (cos=0.75)"),
        (8, 0.0, "8 agents with maximum competition (cos=0.0)"),
        (10, 0.5, "10 agents with moderate competition (cos=0.5)")
    ]
    
    for n_agents, target_cos, description in test_cases:
        print(f"\n{description}:")
        print("-" * 40)
        
        # Generate vectors with integer values
        vectors = generator.generate_vectors(
            n_agents=n_agents,
            target_cosine=target_cos,
            n_items=5,
            max_utility=100.0,
            integer_values=True,
            tolerance=0.05,
            method="hybrid"  # Use hybrid method for better results
        )
        
        # Display vectors
        for agent_id, vector in sorted(vectors.items()):
            print(f"  {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.1f})")
        
        # Analyze results
        stats = generator.analyze_vectors(vectors, target_cos)
        print(f"\n  Statistics:")
        print(f"    Mean similarity: {stats['mean_similarity']:.4f}")
        print(f"    Std deviation: {stats['std_similarity']:.4f}")
        print(f"    Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
        if 'mean_error' in stats:
            print(f"    Mean error from target: {stats['mean_error']:.4f}")
            print(f"    Success rate (error < 0.05): {stats['success_rate']*100:.1f}%")