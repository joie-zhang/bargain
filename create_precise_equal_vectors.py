import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def create_precise_vectors(target_cosine, target_sum=30, length=5, max_iterations=10):
    """
    Create two vectors with very precise cosine similarity and equal sums.
    Uses multiple optimization attempts with different starting points.
    """
    
    if target_cosine == 1.0:
        # Identical vectors
        v1 = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        v2 = v1.copy()
        return v1, v2
    
    if target_cosine == 0.0:
        # Perfect orthogonal vectors
        v1 = np.array([15.0, 15.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        return v1, v2
    
    best_result = None
    best_error = float('inf')
    
    # Try multiple starting points to find the best solution
    for attempt in range(max_iterations):
        def objective(x):
            """Minimize the difference from target cosine similarity."""
            v1 = x[:length]
            v2 = x[length:]
            
            # Calculate cosine similarity
            cos_sim = cosine_similarity(v1, v2)
            
            # Primary objective: match target cosine similarity
            cos_penalty = 1000 * (cos_sim - target_cosine) ** 2
            
            # Secondary: prefer values close to integers
            integer_penalty = 0.01 * (np.sum((v1 - np.round(v1))**2) + np.sum((v2 - np.round(v2))**2))
            
            # Penalty for negative values
            neg_penalty = 10000 * (np.sum(np.maximum(0, -v1)) + np.sum(np.maximum(0, -v2)))
            
            return cos_penalty + integer_penalty + neg_penalty
        
        def constraint_sum1(x):
            """First vector should sum to target_sum."""
            return np.sum(x[:length]) - target_sum
        
        def constraint_sum2(x):
            """Second vector should sum to target_sum."""
            return np.sum(x[length:]) - target_sum
        
        # Generate different starting points for each attempt
        np.random.seed(42 + attempt)
        
        if target_cosine == 0.25:
            if attempt == 0:
                initial_v1 = np.array([12.0, 10.0, 5.0, 2.0, 1.0])
                initial_v2 = np.array([1.0, 2.0, 5.0, 10.0, 12.0])
            else:
                # Random perturbation
                initial_v1 = np.array([14.0, 8.0, 4.0, 2.0, 2.0]) + np.random.randn(5) * 0.5
                initial_v2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0]) + np.random.randn(5) * 0.5
        elif target_cosine == 0.5:
            if attempt == 0:
                initial_v1 = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
                initial_v2 = np.array([5.0, 6.0, 7.0, 7.0, 5.0])
            else:
                initial_v1 = np.array([12.0, 9.0, 6.0, 2.0, 1.0]) + np.random.randn(5) * 0.5
                initial_v2 = np.array([4.0, 6.0, 8.0, 8.0, 4.0]) + np.random.randn(5) * 0.5
        elif target_cosine == 0.75:
            if attempt == 0:
                initial_v1 = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
                initial_v2 = np.array([9.0, 8.0, 6.0, 5.0, 2.0])
            else:
                initial_v1 = np.array([11.0, 7.0, 6.0, 4.0, 2.0]) + np.random.randn(5) * 0.3
                initial_v2 = np.array([10.0, 8.0, 6.0, 4.0, 2.0]) + np.random.randn(5) * 0.3
        else:
            initial_v1 = np.random.uniform(1, 10, length)
            initial_v2 = np.random.uniform(1, 10, length)
        
        # Normalize to sum
        initial_v1 = np.abs(initial_v1) * (target_sum / np.sum(np.abs(initial_v1)))
        initial_v2 = np.abs(initial_v2) * (target_sum / np.sum(np.abs(initial_v2)))
        
        x0 = np.concatenate([initial_v1, initial_v2])
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': constraint_sum1},
            {'type': 'eq', 'fun': constraint_sum2}
        ]
        
        # Bounds - all values should be non-negative
        bounds = [(0, 20) for _ in range(2 * length)]
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                        bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
        
        if result.success:
            v1 = result.x[:length]
            v2 = result.x[length:]
            
            # Check the actual error
            actual_cos = cosine_similarity(v1, v2)
            error = abs(actual_cos - target_cosine)
            
            if error < best_error:
                best_error = error
                best_result = (v1, v2)
                
                # If we found a very good solution, stop
                if error < 0.001:
                    break
    
    if best_result is None:
        # Fallback to initial guess
        return initial_v1, initial_v2
    
    return best_result[0], best_result[1]

def round_to_nice_numbers(vec, target_sum=30):
    """Round vector to nice numbers while maintaining sum."""
    # Round to 1 decimal place
    rounded = np.round(vec, 1)
    
    # Adjust to maintain exact sum
    diff = target_sum - np.sum(rounded)
    if abs(diff) > 0.01:
        # Find the index with the largest value to adjust
        max_idx = np.argmax(rounded)
        rounded[max_idx] += diff
    
    return rounded

print("=" * 60)
print("HIGH-PRECISION VECTORS WITH EQUAL MAX UTILITIES")
print("Target: All vectors sum to 30, cosine similarity error < 0.01")
print("=" * 60)

# Create high-precision pairs
targets = [0.0, 0.25, 0.5, 0.75, 1.0]
final_pairs = []

for target in targets:
    print(f"\nOptimizing for cosine similarity = {target:.2f}...")
    v1, v2 = create_precise_vectors(target)
    
    # Round to nice numbers while maintaining sum
    v1_rounded = round_to_nice_numbers(v1)
    v2_rounded = round_to_nice_numbers(v2)
    
    # Check both original and rounded versions
    actual_original = cosine_similarity(v1, v2)
    actual_rounded = cosine_similarity(v1_rounded, v2_rounded)
    
    # Use rounded if error is still < 0.01, otherwise use original
    if abs(actual_rounded - target) < 0.01:
        final_v1, final_v2 = v1_rounded, v2_rounded
        actual = actual_rounded
    else:
        final_v1, final_v2 = v1, v2
        actual = actual_original
    
    final_pairs.append((final_v1, final_v2, target))
    
    sum_v1 = np.sum(final_v1)
    sum_v2 = np.sum(final_v2)
    error = abs(actual - target)
    
    print(f"  Vector A: {np.round(final_v1, 2)}")
    print(f"  Vector B: {np.round(final_v2, 2)}")
    print(f"  Sum A: {sum_v1:.2f}, Sum B: {sum_v2:.2f}")
    print(f"  Actual cosine: {actual:.4f}, Error: {error:.4f}")
    
    if error >= 0.01:
        print(f"  WARNING: Error exceeds 0.01 threshold!")

print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

print("\nAll pairs with verification:")
for i, (v1, v2, target) in enumerate(final_pairs, 1):
    actual = cosine_similarity(v1, v2)
    sum_v1 = np.sum(v1)
    sum_v2 = np.sum(v2)
    error = abs(actual - target)
    
    print(f"\nPair {i} (Target: {target:.2f}):")
    print(f"  Vector A: {np.round(v1, 2)}")
    print(f"  Vector B: {np.round(v2, 2)}")
    print(f"  Max utility A: {sum_v1:.2f}")
    print(f"  Max utility B: {sum_v2:.2f}")
    print(f"  Actual cosine: {actual:.4f}")
    print(f"  Error: {error:.4f}")
    if error < 0.01:
        print(f"  ✓ PASS (error < 0.01)")
    else:
        print(f"  ✗ FAIL (error >= 0.01)")

print("\n" + "=" * 60)
print("CODE FOR vector_pairs.py")
print("=" * 60)

print("\n# Replace pairs_v2 in vector_pairs.py with:")
print("# All vectors sum to 30 (equal max utilities)")
print("# All cosine similarities within 0.01 of target")
print("pairs_v2 = [")
for v1, v2, target in final_pairs:
    print(f"    (np.array({np.round(v1, 2).tolist()}), np.array({np.round(v2, 2).tolist()}), {target:.2f}),")
print("]")