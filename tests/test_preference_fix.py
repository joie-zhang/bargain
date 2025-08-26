#!/usr/bin/env python3
"""Test script to verify the improved preference generation handles low competition levels correctly."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from negotiation import create_competitive_preferences

def test_preference_generation():
    """Test preference generation at various competition levels."""
    
    # Test parameters
    test_cases = [
        ("Very Low Competition", 0.05),
        ("Low Competition", 0.2),
        ("Medium Competition", 0.5),
        ("High Competition", 0.8),
        ("Very High Competition", 0.95),
    ]
    
    print("Testing Improved Preference Generation Algorithm")
    print("=" * 60)
    
    for name, competition_level in test_cases:
        print(f"\n{name} (target similarity: {competition_level})")
        print("-" * 40)
        
        # Create preferences
        pref_manager = create_competitive_preferences(
            m_items=5,
            n_agents=3,
            cosine_similarity=competition_level,
            random_seed=42
        )
        
        # Generate preferences
        preferences = pref_manager.generate_preferences()
        
        # Extract agent preferences
        agent_prefs = preferences["agent_preferences"]
        
        # Print preference vectors
        for agent_id, prefs in agent_prefs.items():
            print(f"{agent_id}: {[f'{p:.2f}' for p in prefs]}")
        
        # Print actual cosine similarities
        print("\nActual Cosine Similarities:")
        similarities = preferences["cosine_similarities"]
        for pair, similarity in similarities.items():
            print(f"  {pair}: {similarity:.4f}")
        
        # Calculate average similarity and error
        avg_similarity = np.mean(list(similarities.values()))
        max_error = max(abs(sim - competition_level) for sim in similarities.values())
        
        print(f"\nTarget Similarity: {competition_level:.4f}")
        print(f"Average Similarity: {avg_similarity:.4f}")
        print(f"Max Error from Target: {max_error:.4f}")
        
        # Check if within acceptable tolerance
        tolerance = 0.1  # Allow 10% deviation
        if max_error <= tolerance:
            print("✅ PASSED - Similarity within acceptable range")
        else:
            print(f"⚠️  WARNING - Error {max_error:.4f} exceeds tolerance {tolerance}")
    
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    # Test extreme values
    edge_cases = [
        ("Zero Competition", 0.0),
        ("Negative Competition", -0.5),
        ("Perfect Competition", 1.0),
    ]
    
    for name, competition_level in edge_cases:
        print(f"\n{name} (target similarity: {competition_level})")
        print("-" * 40)
        
        try:
            pref_manager = create_competitive_preferences(
                m_items=5,
                n_agents=2,
                cosine_similarity=competition_level,
                random_seed=42
            )
            
            preferences = pref_manager.generate_preferences()
            similarities = preferences["cosine_similarities"]
            
            for pair, similarity in similarities.items():
                print(f"  {pair}: {similarity:.4f}")
                
            avg_similarity = np.mean(list(similarities.values()))
            print(f"Average Similarity: {avg_similarity:.4f}")
            print("✅ Handled successfully")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_preference_generation()