#!/usr/bin/env python3
"""Test the new random vector generation with different seeds and competition levels."""

from negotiation import create_competitive_preferences
import numpy as np

def test_random_vector_generation():
    """Test that different seeds produce different vectors with same competition level."""
    
    print("="*80)
    print("Testing Random Vector Generation with Different Seeds")
    print("="*80)
    
    competition_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = [42, 123, 456, 789, 1000]
    
    for comp_level in competition_levels:
        print(f"\n{'='*60}")
        print(f"Competition Level (Cosine Similarity): {comp_level}")
        print('='*60)
        
        vectors_for_level = []
        
        for seed in seeds[:3]:  # Test with first 3 seeds
            # Create preference manager with specific seed
            pref_manager = create_competitive_preferences(
                m_items=5,
                n_agents=2,
                cosine_similarity=comp_level,
                random_seed=seed,
                known_to_all=False
            )
            
            # Generate preferences
            prefs = pref_manager.generate_preferences()
            
            # Extract vectors
            v1 = np.array(prefs['agent_preferences']['agent_0'])
            v2 = np.array(prefs['agent_preferences']['agent_1'])
            
            # Calculate actual cosine similarity
            actual_cosine = prefs['cosine_similarities'].get('agent_0_vs_agent_1', 0)
            
            print(f"\n  Seed {seed}:")
            print(f"    Agent 0: {v1.astype(int).tolist()}")
            print(f"    Agent 1: {v2.astype(int).tolist()}")
            print(f"    Sum A0: {v1.sum():.1f}, Sum A1: {v2.sum():.1f}")
            print(f"    Actual Cosine: {actual_cosine:.4f}")
            print(f"    Error from target: {abs(actual_cosine - comp_level):.4f}")
            
            vectors_for_level.append((v1, v2))
        
        # Check that vectors are different across seeds (except for cosine=1.0)
        if comp_level < 1.0:
            all_different = True
            for i in range(len(vectors_for_level)):
                for j in range(i+1, len(vectors_for_level)):
                    v1_i, v2_i = vectors_for_level[i]
                    v1_j, v2_j = vectors_for_level[j]
                    if np.allclose(v1_i, v1_j) and np.allclose(v2_i, v2_j):
                        all_different = False
                        break
            
            if all_different:
                print(f"\n  ✓ Vectors are different across seeds (as expected)")
            else:
                print(f"\n  ⚠ Warning: Same vectors generated with different seeds")

def test_edge_cases():
    """Test edge cases and different configurations."""
    print("\n" + "="*80)
    print("Testing Edge Cases")
    print("="*80)
    
    # Test with different number of items
    print("\n1. Testing with 7 items instead of 5:")
    pref_manager = create_competitive_preferences(
        m_items=7,
        n_agents=2,
        cosine_similarity=0.5,
        random_seed=42
    )
    prefs = pref_manager.generate_preferences()
    v1 = np.array(prefs['agent_preferences']['agent_0'])
    v2 = np.array(prefs['agent_preferences']['agent_1'])
    actual_cosine = prefs['cosine_similarities'].get('agent_0_vs_agent_1', 0)
    
    print(f"  Agent 0: {v1.astype(int).tolist()}")
    print(f"  Agent 1: {v2.astype(int).tolist()}")
    print(f"  Cosine Similarity: {actual_cosine:.4f}")
    print(f"  Sum A0: {v1.sum():.1f}, Sum A1: {v2.sum():.1f}")
    
    # Test with 3 agents
    print("\n2. Testing with 3 agents:")
    pref_manager = create_competitive_preferences(
        m_items=5,
        n_agents=3,
        cosine_similarity=0.75,
        random_seed=42
    )
    prefs = pref_manager.generate_preferences()
    
    for agent_id in ['agent_0', 'agent_1', 'agent_2']:
        v = np.array(prefs['agent_preferences'][agent_id])
        print(f"  {agent_id}: {v.astype(int).tolist()} (sum: {v.sum():.1f})")
    
    print("\n  Pairwise cosine similarities:")
    for key, value in prefs['cosine_similarities'].items():
        print(f"    {key}: {value:.4f}")

if __name__ == "__main__":
    test_random_vector_generation()
    test_edge_cases()
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)