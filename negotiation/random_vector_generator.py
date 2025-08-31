"""
Random Vector Generator with Exact Competition Levels

This module generates random preference vectors with precise cosine similarity control.
Unlike the fixed vectors in vector_pairs.py, this generates different vectors each time
based on the random seed while maintaining exact competition levels.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class RandomVectorGenerator:
    """Generate random preference vectors with exact cosine similarity."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize with optional random seed for reproducibility."""
        self.rng = np.random.RandomState(random_seed)
    
    def generate_vectors_with_cosine_similarity(
        self, 
        target_cosine: float,
        n_items: int = 5,
        max_utility: float = 100.0,
        integer_values: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two random vectors with exact cosine similarity.
        
        Args:
            target_cosine: Target cosine similarity (0.0 to 1.0)
            n_items: Number of items (vector dimension)
            max_utility: Maximum total utility (vectors sum to this)
            integer_values: Whether to round to integer values
            
        Returns:
            Tuple of two vectors with the target cosine similarity
        """
        
        # Special cases
        if target_cosine == 1.0:
            # Generate identical random vectors
            v1 = self._generate_random_vector(n_items, max_utility, integer_values)
            v2 = v1.copy()
            return v1, v2
        
        if target_cosine == 0.0:
            # Generate orthogonal vectors
            return self._generate_orthogonal_vectors(n_items, max_utility, integer_values)
        
        # General case: use optimization to find vectors with target similarity
        return self._optimize_vectors(target_cosine, n_items, max_utility, integer_values)
    
    def _generate_random_vector(
        self, 
        n_items: int, 
        max_utility: float,
        integer_values: bool
    ) -> np.ndarray:
        """Generate a random vector that sums to max_utility."""
        # Generate random proportions
        proportions = self.rng.dirichlet(np.ones(n_items))
        vector = proportions * max_utility
        
        if integer_values:
            # Round to integers while maintaining sum
            vector = self._round_to_integers_maintaining_sum(vector, max_utility)
        
        return vector
    
    def _generate_orthogonal_vectors(
        self,
        n_items: int,
        max_utility: float,
        integer_values: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two orthogonal vectors (cosine similarity = 0)."""
        if n_items < 2:
            raise ValueError("Need at least 2 items for orthogonal vectors")
        
        # Strategy: Create vectors with non-overlapping support
        # Split items into two groups
        split = n_items // 2
        
        v1 = np.zeros(n_items)
        v2 = np.zeros(n_items)
        
        # Assign values to non-overlapping indices
        indices1 = self.rng.choice(n_items, split, replace=False)
        indices2 = np.array([i for i in range(n_items) if i not in indices1])
        
        # Distribute max_utility among selected indices
        if len(indices1) > 0:
            proportions1 = self.rng.dirichlet(np.ones(len(indices1)))
            v1[indices1] = proportions1 * max_utility
        
        if len(indices2) > 0:
            proportions2 = self.rng.dirichlet(np.ones(len(indices2)))
            v2[indices2] = proportions2 * max_utility
        
        if integer_values:
            v1 = self._round_to_integers_maintaining_sum(v1, max_utility)
            v2 = self._round_to_integers_maintaining_sum(v2, max_utility)
        
        return v1, v2
    
    def _optimize_vectors(
        self,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        integer_values: bool,
        max_attempts: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use optimization to find vectors with target cosine similarity."""
        
        best_result = None
        best_error = float('inf')
        
        for attempt in range(max_attempts):
            # Random starting point
            x0 = self.rng.uniform(0, max_utility/n_items, 2 * n_items)
            
            def objective(x):
                v1 = x[:n_items]
                v2 = x[n_items:]
                
                # Ensure non-negative values
                if np.any(v1 < 0) or np.any(v2 < 0):
                    return 1e10
                
                # Calculate cosine similarity
                cos_sim = self._cosine_similarity(v1, v2)
                
                # Primary objective: match target cosine similarity
                cos_penalty = 1000 * (cos_sim - target_cosine) ** 2
                
                # Ensure vectors sum to max_utility
                sum_penalty = 100 * ((np.sum(v1) - max_utility) ** 2 + 
                                     (np.sum(v2) - max_utility) ** 2)
                
                return cos_penalty + sum_penalty
            
            # Constraints: all values >= 0, sums = max_utility
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x[:n_items]) - max_utility},
                {'type': 'eq', 'fun': lambda x: np.sum(x[n_items:]) - max_utility}
            ]
            
            bounds = [(0, max_utility) for _ in range(2 * n_items)]
            
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            
            if result.success:
                error = objective(result.x)
                if error < best_error:
                    best_error = error
                    best_result = result
        
        if best_result is None:
            # Fallback to simple method
            return self._simple_method(target_cosine, n_items, max_utility, integer_values)
        
        v1 = best_result.x[:n_items]
        v2 = best_result.x[n_items:]
        
        if integer_values:
            v1 = self._round_to_integers_maintaining_sum(v1, max_utility)
            v2 = self._round_to_integers_maintaining_sum(v2, max_utility)
        
        return v1, v2
    
    def _simple_method(
        self,
        target_cosine: float,
        n_items: int,
        max_utility: float,
        integer_values: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple geometric method to generate vectors with target similarity."""
        # Generate first vector randomly
        v1 = self._generate_random_vector(n_items, max_utility, integer_values)
        
        # Normalize v1
        v1_norm = v1 / np.linalg.norm(v1)
        
        # Generate a random orthogonal vector
        random_vec = self.rng.randn(n_items)
        # Make it orthogonal to v1 using Gram-Schmidt
        v_orthogonal = random_vec - np.dot(random_vec, v1_norm) * v1_norm
        v_orthogonal = v_orthogonal / np.linalg.norm(v_orthogonal)
        
        # Calculate angle from desired cosine similarity
        angle = np.arccos(np.clip(target_cosine, -1, 1))
        
        # Construct v2 using the angle
        v2_direction = np.cos(angle) * v1_norm + np.sin(angle) * v_orthogonal
        
        # Scale to have same magnitude as v1
        v2 = v2_direction * np.linalg.norm(v1)
        
        # Ensure non-negative and normalize sum
        v2 = np.abs(v2)
        v2 = v2 * (max_utility / np.sum(v2))
        
        if integer_values:
            v2 = self._round_to_integers_maintaining_sum(v2, max_utility)
        
        return v1, v2
    
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
    
    def generate_multiple_pairs(
        self,
        competition_levels: list,
        n_items: int = 5,
        max_utility: float = 100.0,
        integer_values: bool = True
    ) -> dict:
        """
        Generate multiple vector pairs for different competition levels.
        
        Args:
            competition_levels: List of target cosine similarities
            n_items: Number of items
            max_utility: Maximum total utility
            integer_values: Whether to use integer values
            
        Returns:
            Dictionary mapping competition level to vector pairs
        """
        pairs = {}
        for level in competition_levels:
            v1, v2 = self.generate_vectors_with_cosine_similarity(
                level, n_items, max_utility, integer_values
            )
            actual_sim = self._cosine_similarity(v1, v2)
            pairs[level] = {
                'vector_1': v1.tolist(),
                'vector_2': v2.tolist(),
                'actual_cosine': actual_sim,
                'error': abs(actual_sim - level)
            }
        return pairs


# Example usage and testing
if __name__ == "__main__":
    # Test with different seeds
    for seed in [42, 123, 456]:
        print(f"\n{'='*60}")
        print(f"Random Seed: {seed}")
        print('='*60)
        
        generator = RandomVectorGenerator(random_seed=seed)
        
        # Generate vectors for standard competition levels
        competition_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        pairs = generator.generate_multiple_pairs(competition_levels)
        
        for level, data in pairs.items():
            print(f"\nTarget Cosine Similarity: {level}")
            print(f"  Vector 1: {data['vector_1']}")
            print(f"  Vector 2: {data['vector_2']}")
            print(f"  Actual Cosine: {data['actual_cosine']:.4f}")
            print(f"  Error: {data['error']:.4f}")
            print(f"  Sum V1: {sum(data['vector_1']):.1f}")
            print(f"  Sum V2: {sum(data['vector_2']):.1f}")