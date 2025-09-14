"""
Test whether integer constraints make exact cosine similarity mathematically impossible
or if longer runtime helps achieve better results.
"""

import numpy as np
from itertools import product
from negotiation.enhanced_vector_generator import EnhancedVectorGenerator
import time


def calculate_cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)


def analyze_integer_feasibility(n_items=5, total_sum=100, target_cosine=0.5):
    """
    Analyze the feasibility of achieving exact cosine similarity with integer constraints.
    For small problems, we can exhaustively check all possibilities.
    """
    
    print(f"\n{'='*80}")
    print(f"Analyzing Integer Constraint Feasibility")
    print(f"{'='*80}")
    print(f"Parameters: n_items={n_items}, sum={total_sum}, target_cosine={target_cosine}")
    
    # For 2 agents with 5 items summing to 100, there are many possibilities
    # Let's sample the space to understand the granularity
    
    # Generate random integer vectors and check their cosine similarities
    n_samples = 10000
    similarities = []
    
    np.random.seed(42)
    for _ in range(n_samples):
        # Generate two random integer vectors that sum to total_sum
        v1 = np.random.multinomial(total_sum, np.ones(n_items)/n_items)
        v2 = np.random.multinomial(total_sum, np.ones(n_items)/n_items)
        
        sim = calculate_cosine_similarity(v1, v2)
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    # Find the distribution of achievable similarities
    print(f"\nRandom sampling of {n_samples} integer vector pairs:")
    print(f"  Min cosine similarity: {similarities.min():.4f}")
    print(f"  Max cosine similarity: {similarities.max():.4f}")
    print(f"  Mean cosine similarity: {similarities.mean():.4f}")
    print(f"  Std cosine similarity: {similarities.std():.4f}")
    
    # Find closest achievable similarities to common targets
    targets = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"\nClosest achievable to target values:")
    for target in targets:
        distances = np.abs(similarities - target)
        closest_idx = np.argmin(distances)
        closest_sim = similarities[closest_idx]
        print(f"  Target {target:.2f}: Closest = {closest_sim:.4f} (error = {distances[closest_idx]:.4f})")
    
    # Histogram of achievable similarities
    hist, bins = np.histogram(similarities, bins=20)
    print(f"\nDistribution of achievable similarities:")
    for i in range(len(hist)):
        bar_width = int(hist[i] * 50 / hist.max())
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}]: {'█' * bar_width} ({hist[i]})")
    
    return similarities


def test_longer_runtime(n_agents=4, target_cosine=0.5, max_iterations_list=[100, 500, 2000]):
    """
    Test if longer runtime (more iterations) improves results.
    """
    
    print(f"\n{'='*80}")
    print(f"Testing Effect of Longer Runtime")
    print(f"{'='*80}")
    print(f"Parameters: n_agents={n_agents}, target_cosine={target_cosine}")
    
    generator = EnhancedVectorGenerator(random_seed=42)
    
    for max_iter in max_iterations_list:
        print(f"\nTesting with max_iterations = {max_iter}:")
        
        # Modify the differential evolution parameters
        start_time = time.time()
        
        # Use evolutionary method with custom iteration count
        def custom_evolutionary(n_agents, target_cosine, n_items, max_utility, tolerance):
            """Custom evolutionary with specified iterations."""
            from scipy.optimize import differential_evolution
            
            n_vars = n_agents * n_items
            
            def objective(x):
                vectors = x.reshape(n_agents, n_items)
                
                if np.any(vectors < 0):
                    return 1e10
                
                sum_penalty = sum((np.sum(vectors[i]) - max_utility) ** 2 for i in range(n_agents))
                
                cos_penalty = 0
                count = 0
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        cos_sim = calculate_cosine_similarity(vectors[i], vectors[j])
                        cos_penalty += (cos_sim - target_cosine) ** 2
                        count += 1
                
                if count > 0:
                    cos_penalty /= count
                
                return 1000 * cos_penalty + 100 * sum_penalty
            
            bounds = [(0, max_utility) for _ in range(n_vars)]
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iter,
                popsize=15,
                tol=tolerance,
                seed=42,
                workers=1,
                disp=False
            )
            
            solution = result.x.reshape(n_agents, n_items)
            vectors = {}
            for i in range(n_agents):
                v = solution[i]
                v = v * (max_utility / np.sum(v)) if np.sum(v) > 0 else v
                vectors[f"agent_{i}"] = v
            
            return vectors
        
        # Generate vectors
        vectors = custom_evolutionary(
            n_agents=n_agents,
            target_cosine=target_cosine,
            n_items=5,
            max_utility=100.0,
            tolerance=0.01
        )
        
        # Round to integers
        for agent_id in vectors:
            v = vectors[agent_id]
            # Round maintaining sum
            rounded = np.floor(v).astype(int)
            remainder = int(100 - np.sum(rounded))
            if remainder > 0:
                fractional = v - rounded
                indices = np.argsort(fractional)[-remainder:]
                rounded[indices] += 1
            vectors[agent_id] = rounded.astype(float)
        
        elapsed = time.time() - start_time
        
        # Calculate statistics
        similarities = []
        agent_ids = list(vectors.keys())
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                sim = calculate_cosine_similarity(
                    vectors[agent_ids[i]],
                    vectors[agent_ids[j]]
                )
                similarities.append(sim)
        
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        mean_error = np.mean([abs(s - target_cosine) for s in similarities])
        max_error = np.max([abs(s - target_cosine) for s in similarities])
        
        print(f"  Runtime: {elapsed:.2f} seconds")
        print(f"  Mean similarity: {mean_sim:.4f}")
        print(f"  Mean error: {mean_error:.4f}")
        print(f"  Max error: {max_error:.4f}")
        print(f"  Std deviation: {std_sim:.4f}")
        
        # Show the vectors
        print(f"  Vectors:")
        for agent_id, vector in sorted(vectors.items()):
            print(f"    {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.0f})")


def find_exact_integer_solutions(n_items=3, total_sum=30, target_cosine=0.5):
    """
    For very small problems, exhaustively find all exact integer solutions.
    """
    
    print(f"\n{'='*80}")
    print(f"Finding Exact Integer Solutions (Small Problem)")
    print(f"{'='*80}")
    print(f"Parameters: n_items={n_items}, sum={total_sum}, target_cosine={target_cosine}")
    
    # Generate all possible integer vectors that sum to total_sum
    # This is computationally feasible only for small problems
    
    def generate_all_vectors(n, s):
        """Generate all n-dimensional non-negative integer vectors summing to s."""
        if n == 1:
            return [[s]]
        
        vectors = []
        for i in range(s + 1):
            for rest in generate_all_vectors(n - 1, s - i):
                vectors.append([i] + rest)
        return vectors
    
    all_vectors = generate_all_vectors(n_items, total_sum)
    print(f"\nTotal possible vectors: {len(all_vectors)}")
    
    # Find pairs with cosine similarity close to target
    exact_pairs = []
    close_pairs = []
    
    for i, v1 in enumerate(all_vectors):
        for j, v2 in enumerate(all_vectors[i+1:], i+1):
            v1_arr = np.array(v1, dtype=float)
            v2_arr = np.array(v2, dtype=float)
            
            sim = calculate_cosine_similarity(v1_arr, v2_arr)
            error = abs(sim - target_cosine)
            
            if error < 0.001:  # Essentially exact
                exact_pairs.append((v1, v2, sim))
            elif error < 0.05:  # Close
                close_pairs.append((v1, v2, sim, error))
    
    print(f"\nExact solutions (error < 0.001): {len(exact_pairs)}")
    if exact_pairs:
        for i, (v1, v2, sim) in enumerate(exact_pairs[:5]):  # Show first 5
            print(f"  Pair {i+1}:")
            print(f"    v1: {v1}, v2: {v2}")
            print(f"    Cosine similarity: {sim:.4f}")
    
    print(f"\nClose solutions (error < 0.05): {len(close_pairs)}")
    if close_pairs:
        close_pairs.sort(key=lambda x: x[3])  # Sort by error
        for i, (v1, v2, sim, error) in enumerate(close_pairs[:5]):  # Show best 5
            print(f"  Pair {i+1}:")
            print(f"    v1: {v1}, v2: {v2}")
            print(f"    Cosine similarity: {sim:.4f} (error: {error:.4f})")


def theoretical_analysis():
    """
    Analyze the theoretical constraints of the problem.
    """
    
    print(f"\n{'='*80}")
    print(f"Theoretical Analysis")
    print(f"{'='*80}")
    
    print("""
The problem of finding integer vectors with exact cosine similarity is constrained by:

1. **Discretization Error**: Integer rounding creates a discrete solution space.
   - With 5 items summing to 100, each item can be 0-100
   - This gives us (104 choose 4) ≈ 4.6M possible vectors per agent
   - But not all combinations yield the target cosine similarity

2. **Degrees of Freedom**: For n agents with m items:
   - We have n*m variables
   - Subject to n sum constraints (each vector sums to 100)
   - And C(n,2) cosine similarity constraints
   - For 4 agents, 5 items: 20 variables, 4 sum constraints, 6 similarity constraints

3. **Granularity Issues**: 
   - Cosine similarity = dot(v1,v2)/(||v1||*||v2||)
   - Small integer changes can cause large similarity jumps
   - Example: [50,50,0,0,0] vs [49,51,0,0,0] changes similarity significantly

4. **Mathematical Possibility**:
   - For some target similarities, NO integer solution may exist
   - The achievable similarities form a discrete set, not a continuum
   - The gap between achievable values depends on the total sum and dimensionality
    """)


if __name__ == "__main__":
    # 1. Analyze feasibility of integer constraints
    analyze_integer_feasibility(n_items=5, total_sum=100, target_cosine=0.5)
    
    # 2. Test effect of longer runtime
    test_longer_runtime(n_agents=4, target_cosine=0.5, max_iterations_list=[100, 500, 2000])
    
    # 3. Find exact solutions for small problems
    find_exact_integer_solutions(n_items=3, total_sum=30, target_cosine=0.5)
    
    # 4. Theoretical analysis
    theoretical_analysis()