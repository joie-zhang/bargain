#!/usr/bin/env python3
"""Quick test to verify 3-agent balanced preferences work correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation.preferences import VectorPreferenceSystem, PreferenceConfig, PreferenceType
from negotiation.multi_agent_vector_generator import MultiAgentVectorGenerator
import numpy as np

def test_integration():
    """Test that the preference system correctly uses the multi-agent generator."""
    
    print("Testing 3-agent preference generation integration...")
    print("="*60)
    
    # Test with different competition levels
    competition_levels = [0.25, 0.5, 0.75, 0.95]
    
    for comp_level in competition_levels:
        print(f"\n🎯 Testing competition level: {comp_level:.2f}")
        print("-"*40)
        
        # Create preference config for 3 agents
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=5,
            n_agents=3,
            target_cosine_similarity=comp_level,
            random_seed=42
        )
        
        # Create preference system
        pref_system = VectorPreferenceSystem(config)
        
        # Generate preferences
        preferences = pref_system.generate_preferences()
        
        # Display results
        print("Generated preferences:")
        for agent_id, prefs in preferences["agent_preferences"].items():
            prefs_array = np.array(prefs)
            print(f"  {agent_id}: {[f'{p:.1f}' for p in prefs]} (sum: {np.sum(prefs_array):.1f})")
        
        print("\nPairwise cosine similarities:")
        target_met = True
        for pair, sim in preferences["cosine_similarities"].items():
            error = abs(sim - comp_level)
            status = "✅" if error < 0.15 else "⚠️"
            if error >= 0.15:
                target_met = False
            print(f"  {pair}: {sim:.4f} (target: {comp_level:.2f}, error: {error:.4f}) {status}")
        
        if target_met:
            print(f"✅ All similarities within tolerance of target!")
        else:
            print(f"⚠️ Some similarities deviate from target")
    
    print("\n" + "="*60)
    print("✅ Integration test complete!")

if __name__ == "__main__":
    test_integration()