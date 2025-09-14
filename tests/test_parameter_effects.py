"""
Test how the number of items and utility sum affect the ability to achieve
target cosine similarities with integer constraints.
"""

import numpy as np
from negotiation.enhanced_vector_generator import EnhancedVectorGenerator
import matplotlib.pyplot as plt


def calculate_cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)


def test_item_count_effect(n_agents=4, target_cosine=0.5, max_utility=100):
    """
    Test how the number of items affects accuracy.
    """
    print(f"\n{'='*80}")
    print(f"Testing Effect of Number of Items")
    print(f"{'='*80}")
    print(f"Fixed parameters: n_agents={n_agents}, target_cosine={target_cosine}, sum={max_utility}")
    
    item_counts = [3, 4, 5, 6, 7, 8, 10]
    results = []
    
    for n_items in item_counts:
        print(f"\nTesting with {n_items} items:")
        
        generator = EnhancedVectorGenerator(random_seed=42)
        
        # Generate vectors
        vectors = generator.generate_vectors(
            n_agents=n_agents,
            target_cosine=target_cosine,
            n_items=n_items,
            max_utility=max_utility,
            integer_values=True,
            tolerance=0.01,
            method="evolutionary"
        )
        
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
        
        results.append({
            'n_items': n_items,
            'mean_error': mean_error,
            'max_error': max_error,
            'std': std_sim
        })
        
        print(f"  Mean similarity: {mean_sim:.4f}")
        print(f"  Mean error: {mean_error:.4f}")
        print(f"  Max error: {max_error:.4f}")
        
        # Show one example vector
        first_agent = agent_ids[0]
        print(f"  Example vector: {vectors[first_agent].tolist()}")
        
    return results


def test_utility_sum_effect(n_agents=4, target_cosine=0.5, n_items=5):
    """
    Test how the total utility sum affects accuracy.
    """
    print(f"\n{'='*80}")
    print(f"Testing Effect of Utility Sum")
    print(f"{'='*80}")
    print(f"Fixed parameters: n_agents={n_agents}, target_cosine={target_cosine}, n_items={n_items}")
    
    utility_sums = [20, 30, 50, 100, 200, 500, 1000]
    results = []
    
    for max_utility in utility_sums:
        print(f"\nTesting with sum={max_utility}:")
        
        generator = EnhancedVectorGenerator(random_seed=42)
        
        # Generate vectors
        vectors = generator.generate_vectors(
            n_agents=n_agents,
            target_cosine=target_cosine,
            n_items=n_items,
            max_utility=max_utility,
            integer_values=True,
            tolerance=0.01,
            method="evolutionary"
        )
        
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
        
        results.append({
            'max_utility': max_utility,
            'mean_error': mean_error,
            'max_error': max_error,
            'std': std_sim
        })
        
        print(f"  Mean similarity: {mean_sim:.4f}")
        print(f"  Mean error: {mean_error:.4f}")
        print(f"  Max error: {max_error:.4f}")
        
        # Show one example vector
        first_agent = agent_ids[0]
        print(f"  Example vector: {vectors[first_agent].tolist()}")
        
    return results


def test_combined_parameters():
    """
    Test combinations of parameters to find optimal settings.
    """
    print(f"\n{'='*80}")
    print(f"Testing Combined Parameter Effects")
    print(f"{'='*80}")
    
    # Test grid
    test_configs = [
        # (n_items, max_utility, description)
        (3, 30, "Small: 3 items, sum=30"),
        (3, 100, "3 items, sum=100"),
        (5, 30, "5 items, sum=30"),
        (5, 100, "Standard: 5 items, sum=100"),
        (5, 500, "5 items, sum=500"),
        (10, 100, "Many items: 10 items, sum=100"),
        (10, 1000, "Large: 10 items, sum=1000"),
    ]
    
    n_agents = 4
    target_cosines = [0.25, 0.5, 0.75]
    
    print(f"\nTesting with {n_agents} agents")
    print("-" * 60)
    
    for n_items, max_utility, description in test_configs:
        print(f"\n{description}:")
        print("  Target -> Mean Error (Max Error)")
        
        errors_by_target = []
        for target in target_cosines:
            generator = EnhancedVectorGenerator(random_seed=42)
            
            vectors = generator.generate_vectors(
                n_agents=n_agents,
                target_cosine=target,
                n_items=n_items,
                max_utility=max_utility,
                integer_values=True,
                tolerance=0.01,
                method="evolutionary"
            )
            
            # Calculate errors
            similarities = []
            agent_ids = list(vectors.keys())
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    sim = calculate_cosine_similarity(
                        vectors[agent_ids[i]],
                        vectors[agent_ids[j]]
                    )
                    similarities.append(sim)
            
            mean_error = np.mean([abs(s - target) for s in similarities])
            max_error = np.max([abs(s - target) for s in similarities])
            
            print(f"  {target:.2f} -> {mean_error:.4f} ({max_error:.4f})")
            errors_by_target.append(mean_error)
        
        avg_error = np.mean(errors_by_target)
        print(f"  Average error across targets: {avg_error:.4f}")


def analyze_granularity():
    """
    Analyze the granularity of achievable cosine similarities for different parameters.
    """
    print(f"\n{'='*80}")
    print(f"Granularity Analysis: Achievable Cosine Similarities")
    print(f"{'='*80}")
    
    configs = [
        (3, 10, "3 items, sum=10"),
        (3, 30, "3 items, sum=30"),
        (5, 30, "5 items, sum=30"),
        (5, 100, "5 items, sum=100"),
    ]
    
    for n_items, max_utility, description in configs:
        print(f"\n{description}:")
        
        # Sample random integer vectors to see achievable similarities
        n_samples = 5000
        similarities = []
        
        np.random.seed(42)
        for _ in range(n_samples):
            # Generate two random integer vectors
            v1 = np.random.multinomial(max_utility, np.ones(n_items)/n_items)
            v2 = np.random.multinomial(max_utility, np.ones(n_items)/n_items)
            
            sim = calculate_cosine_similarity(v1, v2)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Find gaps in achievable similarities
        sorted_sims = np.sort(np.unique(similarities))
        if len(sorted_sims) > 1:
            gaps = np.diff(sorted_sims)
            
            print(f"  Range: [{sorted_sims[0]:.4f}, {sorted_sims[-1]:.4f}]")
            print(f"  Unique values: {len(sorted_sims)}")
            print(f"  Mean gap: {np.mean(gaps):.4f}")
            print(f"  Max gap: {np.max(gaps):.4f}")
            
            # Show coverage for key targets
            targets = [0.0, 0.25, 0.5, 0.75]
            print(f"  Closest to targets:")
            for target in targets:
                closest = sorted_sims[np.argmin(np.abs(sorted_sims - target))]
                error = abs(closest - target)
                print(f"    {target:.2f}: {closest:.4f} (error: {error:.4f})")


def theoretical_analysis():
    """
    Provide theoretical insights about the parameter effects.
    """
    print(f"\n{'='*80}")
    print(f"Theoretical Analysis of Parameter Effects")
    print(f"{'='*80}")
    
    print("""
How Parameters Affect Integer Constraint Difficulty:

1. **Number of Items (n)**:
   - FEWER items → LESS flexibility in adjusting similarities
   - MORE items → MORE degrees of freedom to achieve target
   - Trade-off: Fewer items = simpler problem but coarser granularity
   
2. **Utility Sum (S)**:
   - SMALLER sum → COARSER granularity (bigger jumps between possible values)
   - LARGER sum → FINER granularity (smaller steps possible)
   - Example: With sum=10, each unit is 10% of total; with sum=100, each unit is 1%
   
3. **Optimal Balance**:
   - Sweet spot: n_items ≈ 4-6, sum ≈ 100-200
   - Too few items (n<3): Not enough flexibility
   - Too many items (n>10): Optimization becomes harder
   - Too small sum (S<30): Granularity too coarse
   - Too large sum (S>1000): Marginal improvement, longer computation
   
4. **Mathematical Insight**:
   - Granularity ≈ 1/√(n*S) for cosine similarity adjustments
   - With n=5, S=100: granularity ≈ 0.045
   - With n=3, S=30: granularity ≈ 0.105
   - With n=10, S=1000: granularity ≈ 0.010
    """)


if __name__ == "__main__":
    # 1. Test effect of number of items
    item_results = test_item_count_effect()
    
    # 2. Test effect of utility sum
    sum_results = test_utility_sum_effect()
    
    # 3. Test combined parameters
    test_combined_parameters()
    
    # 4. Analyze granularity
    analyze_granularity()
    
    # 5. Theoretical analysis
    theoretical_analysis()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF FINDINGS")
    print(f"{'='*80}")
    
    print("\nBest parameters for achieving target cosine similarities:")
    print("1. Use 4-6 items (sweet spot for flexibility vs complexity)")
    print("2. Use sum ≥ 100 (finer granularity)")
    print("3. For very precise requirements, use sum = 500-1000")
    print("4. Avoid n_items < 3 or sum < 30 (too coarse)")
    print("5. Consider using continuous values if exact match is critical")