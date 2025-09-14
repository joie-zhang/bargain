#!/usr/bin/env python3
"""Test 3-agent vector generation across all 5 competition levels."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation.multi_agent_vector_generator import MultiAgentVectorGenerator
import numpy as np

def test_3_agents_all_competition_levels():
    """Test 3-agent generation with all competition levels we use in experiments."""
    
    print("="*80)
    print("TESTING 3-AGENT VECTOR GENERATION - ALL COMPETITION LEVELS")
    print("="*80)
    
    # Test all 5 competition levels used in our experiments
    competition_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Use consistent parameters
    n_agents = 3
    n_items = 5
    max_utility = 100.0
    
    # Track overall results
    all_results = {}
    
    for comp_level in competition_levels:
        print(f"\n{'='*60}")
        print(f"Competition Level: {comp_level}")
        print(f"{'='*60}")
        
        # Test with multiple seeds to verify consistency
        seeds = [42, 123, 456]
        level_results = []
        
        for seed in seeds:
            generator = MultiAgentVectorGenerator(random_seed=seed)
            
            # Generate vectors
            vectors = generator.generate_vectors_for_n_agents(
                n_agents=n_agents,
                target_cosine=comp_level,
                n_items=n_items,
                max_utility=max_utility,
                integer_values=True,
                tolerance=0.1
            )
            
            print(f"\n🎲 Seed {seed}:")
            print("  Generated preference vectors:")
            for agent_id, vector in vectors.items():
                print(f"    {agent_id}: {vector.tolist()} (sum: {np.sum(vector):.1f})")
            
            # Calculate pairwise similarities
            sims = generator._calculate_all_pairwise_similarities(vectors)
            
            print("\n  Pairwise cosine similarities:")
            errors = []
            for pair, sim in sims.items():
                error = abs(sim - comp_level)
                errors.append(error)
                status = "✅" if error < 0.15 else "⚠️"
                print(f"    {pair}: {sim:.4f} (target: {comp_level:.2f}, error: {error:.4f}) {status}")
            
            # Calculate statistics
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            std_error = np.std(errors)
            
            print(f"\n  Statistics:")
            print(f"    Average error: {avg_error:.4f}")
            print(f"    Maximum error: {max_error:.4f}")
            print(f"    Std deviation: {std_error:.4f}")
            
            level_results.append({
                'seed': seed,
                'avg_error': avg_error,
                'max_error': max_error,
                'std_error': std_error,
                'similarities': sims
            })
        
        # Summary for this competition level
        all_avg_errors = [r['avg_error'] for r in level_results]
        all_max_errors = [r['max_error'] for r in level_results]
        
        print(f"\n📊 Summary for Competition Level {comp_level}:")
        print(f"  Across {len(seeds)} seeds:")
        print(f"    Mean of average errors: {np.mean(all_avg_errors):.4f}")
        print(f"    Mean of maximum errors: {np.mean(all_max_errors):.4f}")
        print(f"    Worst case error: {np.max(all_max_errors):.4f}")
        
        if np.max(all_max_errors) < 0.15:
            print(f"  ✅ PASSED - All similarities within tolerance!")
        else:
            print(f"  ⚠️ WARNING - Some similarities exceed tolerance")
        
        all_results[comp_level] = level_results
    
    # Final summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for comp_level in competition_levels:
        results = all_results[comp_level]
        worst_error = max(r['max_error'] for r in results)
        avg_of_avgs = np.mean([r['avg_error'] for r in results])
        
        status = "✅ PASS" if worst_error < 0.15 else "⚠️ WARN"
        print(f"Competition {comp_level:4.2f}: Avg error={avg_of_avgs:.4f}, Max error={worst_error:.4f} {status}")
    
    # Check if all levels passed
    all_passed = all(
        max(r['max_error'] for r in all_results[level]) < 0.15 
        for level in competition_levels
    )
    
    print(f"\n{'='*80}")
    if all_passed:
        print("✅ ALL COMPETITION LEVELS PASSED!")
        print("The 3-agent vector generation is ready for experiments.")
    else:
        print("⚠️ Some competition levels have issues that may need attention.")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_3_agents_all_competition_levels()