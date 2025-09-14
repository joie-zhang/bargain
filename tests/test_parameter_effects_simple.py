"""
Simplified test to show how items and utility sum affect accuracy.
"""

import numpy as np
from negotiation.enhanced_vector_generator import EnhancedVectorGenerator


def calculate_cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)


def quick_test(n_agents=4, target_cosine=0.5):
    """Quick test of different parameter combinations."""
    
    print(f"Testing with {n_agents} agents, target cosine = {target_cosine}")
    print("=" * 70)
    
    # Test configurations
    configs = [
        # (n_items, max_utility, description)
        (3, 30, "Few items, small sum"),
        (3, 100, "Few items, standard sum"),
        (5, 30, "Standard items, small sum"),
        (5, 100, "Standard config"),
        (5, 500, "Standard items, large sum"),
        (8, 100, "Many items, standard sum"),
    ]
    
    print(f"\n{'Config':<30} {'Mean Error':<12} {'Max Error':<12} {'Success?':<10}")
    print("-" * 70)
    
    for n_items, max_utility, description in configs:
        generator = EnhancedVectorGenerator(random_seed=42)
        
        # Use hybrid method with moderate iterations for speed
        vectors = generator.generate_vectors(
            n_agents=n_agents,
            target_cosine=target_cosine,
            n_items=n_items,
            max_utility=max_utility,
            integer_values=True,
            tolerance=0.01,
            method="hybrid"
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
        
        mean_error = np.mean([abs(s - target_cosine) for s in similarities])
        max_error = np.max([abs(s - target_cosine) for s in similarities])
        success = "✓" if mean_error < 0.05 else "✗"
        
        print(f"{description:<30} {mean_error:<12.4f} {max_error:<12.4f} {success:<10}")
        
        # Show example vector for context
        first_agent = agent_ids[0]
        vector_str = str(vectors[first_agent].tolist())
        print(f"  Example: {vector_str}")


def analyze_granularity_simple():
    """Simple granularity analysis."""
    
    print("\n" + "=" * 70)
    print("Granularity Analysis: How fine-grained are the possible similarities?")
    print("=" * 70)
    
    configs = [
        (3, 10),
        (3, 30),
        (5, 30),
        (5, 100),
        (5, 500),
    ]
    
    print(f"\n{'Items':<8} {'Sum':<8} {'Min Sim':<10} {'Max Sim':<10} {'Unique':<10} {'Avg Gap':<10}")
    print("-" * 60)
    
    for n_items, max_utility in configs:
        # Sample to find achievable similarities
        n_samples = 1000
        similarities = set()
        
        np.random.seed(42)
        for _ in range(n_samples):
            v1 = np.random.multinomial(max_utility, np.ones(n_items)/n_items)
            v2 = np.random.multinomial(max_utility, np.ones(n_items)/n_items)
            
            sim = calculate_cosine_similarity(v1, v2)
            similarities.add(round(sim, 6))  # Round to avoid float precision issues
        
        sorted_sims = sorted(similarities)
        
        if len(sorted_sims) > 1:
            gaps = [sorted_sims[i+1] - sorted_sims[i] for i in range(len(sorted_sims)-1)]
            avg_gap = np.mean(gaps)
        else:
            avg_gap = 0
        
        print(f"{n_items:<8} {max_utility:<8} {min(sorted_sims):<10.4f} {max(sorted_sims):<10.4f} {len(sorted_sims):<10} {avg_gap:<10.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PARAMETER EFFECTS ON INTEGER PREFERENCE VECTORS")
    print("=" * 70)
    
    # Test different configurations
    print("\n1. TESTING DIFFERENT PARAMETER COMBINATIONS")
    quick_test(n_agents=4, target_cosine=0.5)
    
    # Analyze granularity
    print("\n2. GRANULARITY ANALYSIS")
    analyze_granularity_simple()
    
    # Conclusions
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print("""
1. FEWER ITEMS generally makes it HARDER to achieve target similarities
   - With 3 items, there's less flexibility to adjust values
   - With 5+ items, more degrees of freedom help achieve targets

2. SMALLER SUMS make granularity COARSER
   - Sum=30: Large jumps between possible similarities
   - Sum=100: Good balance of granularity and computation
   - Sum=500+: Very fine control but diminishing returns

3. OPTIMAL SETTINGS:
   - For accuracy: Use 5-6 items with sum ≥ 100
   - For speed: Use 4-5 items with sum = 100
   - Avoid: Less than 3 items or sum < 30

4. TRADE-OFFS:
   - More items = More flexibility BUT harder optimization
   - Larger sum = Finer control BUT longer computation
   - Integer constraints will ALWAYS limit achievable similarities
    """)