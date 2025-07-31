#!/usr/bin/env python3
"""
Demonstration of the negotiation environment functionality.

This script shows how to create and use the negotiation environment
with various parameter combinations to verify it works as expected.
"""

from negotiation import create_negotiation_environment, NegotiationStatus
import json


def demo_basic_environment():
    """Demonstrate basic environment creation and usage."""
    print("=== Basic Environment Demo ===")
    
    # Create a simple negotiation environment
    env = create_negotiation_environment(
        m_items=5,
        n_agents=3,
        t_rounds=10,
        gamma_discount=0.8,
        random_seed=42
    )
    
    print(f"Created environment with {env.config.m_items} items, {env.config.n_agents} agents")
    print(f"Max rounds: {env.config.t_rounds}, Discount factor: {env.config.gamma_discount}")
    
    # Show items in the negotiation
    items = env.get_items_summary()
    print(f"\nItems to negotiate over:")
    for item in items:
        print(f"  - {item['name']} (ID: {item['id']})")
    
    # Initialize agents
    agent_ids = ["Alice", "Bob", "Charlie"]
    env.initialize_agents(agent_ids)
    print(f"\nInitialized agents: {', '.join(agent_ids)}")
    print(f"Status: {env.status.value}")
    print(f"Current round: {env.current_round}")
    
    # Add some proposals and votes
    print(f"\n--- Round {env.current_round} ---")
    env.add_proposal("Alice", {
        "allocation": {"Alice": [0, 1], "Bob": [2], "Charlie": [3, 4]},
        "message": "I propose this fair distribution"
    })
    print("Alice made a proposal")
    
    env.add_vote("Alice", {"accept": True, "proposal_id": 0})
    env.add_vote("Bob", {"accept": False, "counter_proposal": "I want item 1 too"})
    env.add_vote("Charlie", {"accept": True, "proposal_id": 0})
    print("All agents voted")
    
    # Check for consensus
    consensus, allocation = env.check_consensus()
    print(f"Consensus reached: {consensus}")
    
    # Advance round
    if env.advance_round():
        print(f"Advanced to round {env.current_round}")
    
    # Get current state
    state = env.get_negotiation_state()
    print(f"\nCurrent state:")
    print(f"  Status: {state['status']}")
    print(f"  Round: {state['current_round']}/{state['max_rounds']}")
    print(f"  Active agents: {state['agents']['active']}")
    
    return env


def demo_parameter_variations():
    """Demonstrate environment with different parameter combinations."""
    print("\n=== Parameter Variations Demo ===")
    
    test_cases = [
        {"m_items": 3, "n_agents": 2, "t_rounds": 5, "gamma_discount": 0.5},
        {"m_items": 10, "n_agents": 5, "t_rounds": 20, "gamma_discount": 0.9},
        {"m_items": 1, "n_agents": 2, "t_rounds": 1, "gamma_discount": 1.0},
        {"m_items": 25, "n_agents": 8, "t_rounds": 50, "gamma_discount": 0.75},
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {params}")
        
        try:
            env = create_negotiation_environment(**params, random_seed=42)
            agent_ids = [f"Agent_{j+1}" for j in range(params["n_agents"])]
            env.initialize_agents(agent_ids)
            
            items = env.get_items_summary()
            print(f"  ✓ Created successfully with {len(items)} items")
            print(f"  ✓ Initialized {len(agent_ids)} agents")
            print(f"  ✓ Status: {env.status.value}, Round: {env.current_round}")
            
            # Test round advancement
            rounds_advanced = 0
            while env.advance_round() and rounds_advanced < 3:  # Test first few rounds
                rounds_advanced += 1
            
            print(f"  ✓ Advanced {rounds_advanced} rounds successfully")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def demo_negotiation_flow():
    """Demonstrate a complete negotiation flow."""
    print("\n=== Complete Negotiation Flow Demo ===")
    
    env = create_negotiation_environment(
        m_items=4,
        n_agents=2,
        t_rounds=3,
        gamma_discount=0.9,
        random_seed=123
    )
    
    agents = ["Negotiator_A", "Negotiator_B"]
    env.initialize_agents(agents)
    
    items = env.get_items_summary()
    print(f"Negotiating over {len(items)} items: {[item['name'] for item in items]}")
    print(f"Between agents: {', '.join(agents)}")
    
    # Simulate multiple rounds
    for round_num in range(1, env.config.t_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        
        if env.status != NegotiationStatus.IN_PROGRESS:
            break
        
        # Agent A makes a proposal
        if round_num == 1:
            proposal_a = {
                "allocation": {"Negotiator_A": [0, 1], "Negotiator_B": [2, 3]},
                "message": "I suggest we split items evenly"
            }
        else:
            proposal_a = {
                "allocation": {"Negotiator_A": [0, 2], "Negotiator_B": [1, 3]},
                "message": "How about this alternative split?"
            }
        
        env.add_proposal("Negotiator_A", proposal_a)
        print(f"Negotiator_A proposed: {proposal_a['message']}")
        
        # Both agents vote
        if round_num < 3:  # Disagree until last round
            env.add_vote("Negotiator_A", {"accept": True})
            env.add_vote("Negotiator_B", {"accept": False, "reason": "I want different items"})
            print("Negotiator_A: Accept, Negotiator_B: Reject")
        else:  # Reach agreement
            env.add_vote("Negotiator_A", {"accept": True})
            env.add_vote("Negotiator_B", {"accept": True})
            print("Both agents: Accept")
        
        # Check consensus
        consensus, allocation = env.check_consensus()
        print(f"Consensus: {consensus}")
        
        if consensus:
            final_allocation = {
                "Negotiator_A": [0, 2],
                "Negotiator_B": [1, 3]
            }
            env.finalize_allocation(final_allocation)
            print(f"Final allocation: {final_allocation}")
            break
        
        # Advance round if no consensus
        if not env.advance_round():
            print("Maximum rounds reached")
            break
    
    print(f"\nFinal status: {env.status.value}")
    if env.final_allocation:
        print(f"Final allocation: {env.final_allocation}")


def demo_serialization():
    """Demonstrate saving and loading environment state."""
    print("\n=== Serialization Demo ===")
    
    # Create and set up environment
    env = create_negotiation_environment(3, 2, 5, 0.8, 456)
    env.initialize_agents(["Agent_X", "Agent_Y"])
    
    # Add some activity
    env.add_proposal("Agent_X", {"test": "proposal", "items": [0, 1]})
    env.add_vote("Agent_X", {"accept": True})
    env.add_vote("Agent_Y", {"accept": False})
    env.advance_round()
    
    print("Created environment with some activity")
    print(f"Current round: {env.current_round}")
    print(f"Number of rounds: {len(env.rounds)}")
    
    # Export session data
    session_data = env.export_session_data()
    print(f"Exported session data with {len(session_data)} top-level keys")
    
    # Save to file
    save_path = "/tmp/negotiation_demo.json"
    env.save_to_file(save_path)
    print(f"Saved environment to {save_path}")
    
    # Load from file
    try:
        from negotiation.environment import NegotiationEnvironment
        loaded_env = NegotiationEnvironment.load_from_file(save_path)
        print(f"Loaded environment successfully")
        print(f"Loaded round: {loaded_env.current_round}")
        print(f"Loaded agents: {loaded_env.agent_ids}")
        print("✓ Serialization working correctly")
    except Exception as e:
        print(f"✗ Loading failed: {e}")


def main():
    """Run all demonstrations."""
    print("Negotiation Environment Demonstration")
    print("=" * 50)
    
    try:
        demo_basic_environment()
        demo_parameter_variations()
        demo_negotiation_flow()
        demo_serialization()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("The negotiation environment is ready for multi-agent LLM research.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()