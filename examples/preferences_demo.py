#!/usr/bin/env python3
"""
Demonstration of the preference system for multi-agent negotiation.

This script shows how to use both vector and matrix preference systems,
including preference generation, utility calculations, and competition analysis.
"""

import json
from negotiation.preferences import (
    create_competitive_preferences,
    create_cooperative_preferences,
    analyze_preference_competition_level
)


def demo_competitive_preferences():
    """Demonstrate competitive vector preferences."""
    print("=" * 60)
    print("COMPETITIVE PREFERENCE SYSTEM DEMO")
    print("=" * 60)
    
    # Create highly competitive preferences (cosine similarity ≈ 0.9)
    manager = create_competitive_preferences(
        m_items=5,
        n_agents=3,
        cosine_similarity=0.9,
        random_seed=42,
        known_to_all=False  # Secret preferences
    )
    
    # Generate preferences
    preferences = manager.generate_preferences()
    
    print("\n1. Generated Competitive Preferences:")
    print(f"   Preference Type: {preferences['type']}")
    print(f"   Number of Items: {len(preferences['agent_preferences']['agent_0'])}")
    print(f"   Number of Agents: {len(preferences['agent_preferences'])}")
    
    print("\n2. Agent Preference Vectors:")
    for agent_id, prefs in preferences["agent_preferences"].items():
        print(f"   {agent_id}: {[round(p, 2) for p in prefs]}")
    
    print("\n3. Cosine Similarities (High = Competitive):")
    for pair, similarity in preferences["cosine_similarities"].items():
        print(f"   {pair}: {similarity:.3f}")
    
    # Analyze competition level
    analysis = analyze_preference_competition_level(preferences)
    print(f"\n4. Competition Analysis:")
    print(f"   Competition Level: {analysis['competition_level']}")
    print(f"   Average Similarity: {analysis['avg_cosine_similarity']:.3f}")
    
    # Demonstrate utility calculation
    print("\n5. Utility Calculation Example:")
    # Sample allocation: agent_0 gets items [0,4], agent_1 gets [1,3], agent_2 gets [2]
    test_allocation = {
        "agent_0": [0, 4],
        "agent_1": [1, 3], 
        "agent_2": [2]
    }
    
    utilities = manager.calculate_all_utilities(test_allocation, preferences)
    
    print(f"   Test Allocation: {test_allocation}")
    print(f"   Resulting Utilities:")
    total_utility = 0
    for agent_id, utility in utilities.items():
        print(f"     {agent_id}: {utility:.2f}")
        total_utility += utility
    print(f"   Total Utility: {total_utility:.2f}")
    
    return preferences


def demo_cooperative_preferences():
    """Demonstrate cooperative matrix preferences."""
    print("\n" + "=" * 60)
    print("COOPERATIVE PREFERENCE SYSTEM DEMO") 
    print("=" * 60)
    
    # Create cooperative preferences (high cooperation factor)
    manager = create_cooperative_preferences(
        m_items=4,
        n_agents=3,
        cooperation_factor=0.8,
        random_seed=42,
        known_to_all=True  # Common knowledge
    )
    
    # Generate preferences
    preferences = manager.generate_preferences()
    
    print("\n1. Generated Cooperative Preferences:")
    print(f"   Preference Type: {preferences['type']}")
    print(f"   Cooperation Factor: {preferences['config']['cooperation_factor']}")
    
    print("\n2. Agent Preference Matrices:")
    print("   (Rows = Items, Columns = Agents)")
    for agent_id, matrix in preferences["agent_preferences"].items():
        print(f"\n   {agent_id} matrix:")
        for i, row in enumerate(matrix):
            print(f"     Item {i}: {[round(val, 2) for val in row]}")
    
    # Analyze competition level
    analysis = analyze_preference_competition_level(preferences)
    print(f"\n3. Competition Analysis:")
    print(f"   Competition Level: {analysis['competition_level']}")
    print(f"   Cooperation Factor: {analysis['cooperation_factor']}")
    
    # Demonstrate utility calculation
    print("\n4. Utility Calculation Example:")
    test_allocation = {
        "agent_0": [0, 3],
        "agent_1": [1],
        "agent_2": [2]
    }
    
    utilities = manager.calculate_all_utilities(test_allocation, preferences)
    
    print(f"   Test Allocation: {test_allocation}")
    print(f"   Resulting Utilities:")
    for agent_id, utility in utilities.items():
        print(f"     {agent_id}: {utility:.2f}")
    
    return preferences


def demo_preference_comparison():
    """Compare different preference configurations."""
    print("\n" + "=" * 60)
    print("PREFERENCE CONFIGURATION COMPARISON")
    print("=" * 60)
    
    configs = [
        ("Low Competition", 0.3),
        ("Medium Competition", 0.6), 
        ("High Competition", 0.9)
    ]
    
    print("\nComparing different competition levels:")
    print("(Higher cosine similarity = more competitive)")
    
    for name, target_similarity in configs:
        manager = create_competitive_preferences(
            m_items=5,
            n_agents=3,
            cosine_similarity=target_similarity,
            random_seed=42
        )
        
        preferences = manager.generate_preferences()
        analysis = analyze_preference_competition_level(preferences)
        
        similarities = list(preferences["cosine_similarities"].values())
        avg_sim = sum(similarities) / len(similarities)
        
        print(f"\n{name}:")
        print(f"  Target Similarity: {target_similarity}")
        print(f"  Actual Avg Similarity: {avg_sim:.3f}")
        print(f"  Competition Level: {analysis['competition_level']}")


def demo_strategic_scenario_setup():
    """Demonstrate setting up scenarios for strategic behavior detection."""
    print("\n" + "=" * 60)
    print("STRATEGIC SCENARIO SETUP FOR RESEARCH")
    print("=" * 60)
    
    # Scenario 1: Highly competitive - should encourage strategic behavior
    print("\n1. High Competition Scenario (Should Encourage Strategic Behavior):")
    competitive_manager = create_competitive_preferences(
        m_items=5,
        n_agents=3,
        cosine_similarity=0.95,  # Very high competition
        random_seed=123,
        known_to_all=False  # Secret preferences increase strategic potential
    )
    
    competitive_prefs = competitive_manager.generate_preferences()
    comp_analysis = analyze_preference_competition_level(competitive_prefs)
    
    print(f"   Average Cosine Similarity: {comp_analysis['avg_cosine_similarity']:.3f}")
    print(f"   Competition Level: {comp_analysis['competition_level']}")
    print(f"   Preferences Known to All: {competitive_prefs['config']['known_to_all']}")
    print("   → Prediction: High potential for strategic behavior")
    
    # Scenario 2: Cooperative - should discourage strategic behavior
    print("\n2. Cooperative Scenario (Should Discourage Strategic Behavior):")
    cooperative_manager = create_cooperative_preferences(
        m_items=5,
        n_agents=3,
        cooperation_factor=0.9,  # Very cooperative
        random_seed=123,
        known_to_all=True  # Common knowledge reduces strategic potential
    )
    
    cooperative_prefs = cooperative_manager.generate_preferences()
    coop_analysis = analyze_preference_competition_level(cooperative_prefs)
    
    print(f"   Cooperation Factor: {coop_analysis['cooperation_factor']}")
    print(f"   Competition Level: {coop_analysis['competition_level']}")
    print(f"   Preferences Known to All: {cooperative_prefs['config']['known_to_all']}")
    print("   → Prediction: Low potential for strategic behavior")
    
    return competitive_prefs, cooperative_prefs


def save_example_preferences():
    """Save example preferences for use in experiments."""
    print("\n" + "=" * 60)
    print("SAVING EXAMPLE PREFERENCES")
    print("=" * 60)
    
    # Create different preference sets for experiments
    preference_sets = {
        "highly_competitive": create_competitive_preferences(
            m_items=5, n_agents=3, cosine_similarity=0.9, random_seed=42
        ).generate_preferences(),
        
        "moderately_competitive": create_competitive_preferences(
            m_items=5, n_agents=3, cosine_similarity=0.6, random_seed=42
        ).generate_preferences(),
        
        "cooperative": create_cooperative_preferences(
            m_items=5, n_agents=3, cooperation_factor=0.8, random_seed=42
        ).generate_preferences()
    }
    
    # Save each preference set
    for name, prefs in preference_sets.items():
        filename = f"experiments/configs/{name}_preferences.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert any enum values to strings for JSON serialization
        serializable_prefs = make_json_serializable(prefs)
        
        with open(filename, 'w') as f:
            json.dump(serializable_prefs, f, indent=2)
        
        print(f"   Saved: {filename}")
    
    print("\nExample preference configurations saved for experiments!")


def make_json_serializable(obj):
    """Convert object to JSON serializable format."""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, 'value'):  # Enum objects
        return obj.value
    else:
        return obj


def main():
    """Run all preference system demonstrations."""
    print("Multi-Agent Negotiation Preference System Demo")
    print("=" * 60)
    
    # Run demonstrations
    competitive_prefs = demo_competitive_preferences()
    cooperative_prefs = demo_cooperative_preferences()
    demo_preference_comparison()
    demo_strategic_scenario_setup()
    save_example_preferences()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Vector preferences create competitive scenarios (high cosine similarity)")
    print("2. Matrix preferences enable cooperative scenarios (high cooperation factor)")
    print("3. Secret preferences may encourage more strategic behavior")
    print("4. Common knowledge preferences may encourage cooperation")
    print("5. Both systems provide accurate utility calculations")
    print("\nThe preference system is ready for strategic behavior research!")


if __name__ == "__main__":
    main()