#!/usr/bin/env python3
"""
Multi-LLM Agent Integration Demonstration.

This script shows how to use the multi-LLM agent system for negotiation research,
including creating agents, conducting negotiations, and analyzing strategic behavior.
"""

import asyncio
import json
from negotiation import (
    create_negotiation_environment,
    create_competitive_preferences,
    AgentFactory,
    AgentConfiguration,
    ModelType,
    NegotiationContext,
    create_o3_vs_haiku_experiment,
    create_simulated_experiment
)


async def demo_simulated_agents():
    """Demonstrate simulated agents with different strategic profiles."""
    print("=" * 60)
    print("SIMULATED MULTI-LLM AGENT DEMO")
    print("=" * 60)
    
    # Create factory
    factory = AgentFactory()
    
    # Create agents with different strategic profiles
    agent_configs = [
        AgentConfiguration("aggressive_agent", ModelType.TEST_STRONG, strategic_level="aggressive"),
        AgentConfiguration("cooperative_agent", ModelType.TEST_WEAK, strategic_level="cooperative"),
        AgentConfiguration("strategic_agent", ModelType.TEST_STRONG, strategic_level="balanced")
    ]
    
    agents = []
    for config in agent_configs:
        agent = factory.create_agent(config)
        agents.append(agent)
        print(f"Created {agent.agent_id}: {agent.get_model_info()['strategic_level']} strategy")
    
    # Create negotiation environment
    env = create_negotiation_environment(
        m_items=5,
        n_agents=3,
        t_rounds=3,
        random_seed=42
    )
    
    # Create competitive preferences - use standard agent IDs to match preference system
    pref_manager = create_competitive_preferences(
        m_items=5,
        n_agents=3,
        cosine_similarity=0.9,
        random_seed=42,
        known_to_all=False
    )
    
    preferences = pref_manager.generate_preferences()
    
    # Map custom agent IDs to preference agent IDs
    agent_id_mapping = {
        agents[0].agent_id: "agent_0",
        agents[1].agent_id: "agent_1", 
        agents[2].agent_id: "agent_2"
    }
    
    # Update preferences with custom agent IDs
    updated_prefs = {}
    for custom_id, standard_id in agent_id_mapping.items():
        updated_prefs[custom_id] = preferences["agent_preferences"][standard_id]
    preferences["agent_preferences"] = updated_prefs
    
    print(f"\nEnvironment: {env.config.m_items} items, {env.config.n_agents} agents, {env.config.t_rounds} rounds")
    print(f"Competition level: {sum(preferences['cosine_similarities'].values()) / len(preferences['cosine_similarities']):.3f}")
    
    # Initialize environment
    env.initialize_agents([agent.agent_id for agent in agents])
    
    print(f"\nAgent Preferences (Secret):")
    for agent in agents:
        agent_prefs = preferences["agent_preferences"][agent.agent_id]
        print(f"  {agent.agent_id}: {[round(p, 2) for p in agent_prefs]}")
    
    # Create negotiation context for agents
    items = env.get_items_summary()
    
    return agents, env, preferences, items


async def demo_negotiation_round(agents, env, preferences, items):
    """Demonstrate a single negotiation round."""
    print(f"\n" + "=" * 40)
    print("NEGOTIATION ROUND SIMULATION")
    print("=" * 40)
    
    current_round = env.current_round
    
    # Phase 1: Discussion
    print(f"\nRound {current_round} - Discussion Phase:")
    for agent in agents:
        context = NegotiationContext(
            current_round=current_round,
            max_rounds=env.config.t_rounds,
            items=items,
            agents=[a.agent_id for a in agents],
            agent_id=agent.agent_id,
            preferences=preferences["agent_preferences"][agent.agent_id],
            turn_type="discussion"
        )
        
        discussion = await agent.discuss(context, "Let's discuss our preferences and potential deals")
        print(f"  {agent.agent_id}: {discussion}")
    
    # Phase 2: Proposals
    print(f"\nRound {current_round} - Proposal Phase:")
    proposals = []
    for agent in agents:
        context = NegotiationContext(
            current_round=current_round,
            max_rounds=env.config.t_rounds,
            items=items,
            agents=[a.agent_id for a in agents],
            agent_id=agent.agent_id,
            preferences=preferences["agent_preferences"][agent.agent_id],
            turn_type="proposal"
        )
        
        proposal = await agent.propose_allocation(context)
        proposals.append(proposal)
        print(f"  {agent.agent_id} proposes: {proposal['allocation']}")
        print(f"    Reasoning: {proposal['reasoning']}")
    
    # Phase 3: Voting
    print(f"\nRound {current_round} - Voting Phase:")
    # Let's vote on the first proposal
    test_proposal = proposals[0]
    votes = []
    
    for agent in agents:
        context = NegotiationContext(
            current_round=current_round,
            max_rounds=env.config.t_rounds,
            items=items,
            agents=[a.agent_id for a in agents],
            agent_id=agent.agent_id,
            preferences=preferences["agent_preferences"][agent.agent_id],
            turn_type="voting",
            current_proposals=proposals
        )
        
        vote = await agent.vote_on_proposal(context, test_proposal)
        votes.append(vote)
        print(f"  {agent.agent_id} votes: {vote['vote']}")
        print(f"    Reasoning: {vote['reasoning']}")
    
    # Analyze results
    accept_votes = sum(1 for vote in votes if vote['vote'] == 'accept')
    print(f"\nVoting Results: {accept_votes}/{len(votes)} agents accepted")
    
    if accept_votes == len(votes):
        print("🎉 CONSENSUS REACHED!")
        utilities = {}
        for agent in agents:
            utility = preferences["agent_preferences"][agent.agent_id]
            allocated_items = test_proposal['allocation'].get(agent.agent_id, [])
            agent_utility = sum(utility[item_id] for item_id in allocated_items)
            utilities[agent.agent_id] = agent_utility
        
        print("Final Utilities:")
        winner = max(utilities.items(), key=lambda x: x[1])
        for agent_id, utility in utilities.items():
            status = " (WINNER)" if agent_id == winner[0] else ""
            print(f"  {agent_id}: {utility:.2f}{status}")
        
        return True, utilities
    else:
        print("❌ No consensus - negotiation continues")
        return False, None


async def demo_strategic_behavior_analysis(agents):
    """Demonstrate analysis of strategic behaviors."""
    print(f"\n" + "=" * 40)
    print("STRATEGIC BEHAVIOR ANALYSIS")
    print("=" * 40)
    
    print("\nAgent Performance Statistics:")
    for agent in agents:
        stats = agent.get_performance_stats()
        model_info = agent.get_model_info()
        
        print(f"\n{agent.agent_id}:")
        print(f"  Model: {stats['model_type']}")
        print(f"  Strategic Level: {model_info.get('strategic_level', 'N/A')}")
        print(f"  Cooperation Tendency: {model_info.get('cooperation_tendency', 'N/A')}")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Avg Response Time: {stats['avg_response_time']:.3f}s")
        print(f"  Total Cost: ${stats['total_cost']:.4f}")
    
    print(f"\nStrategic Insights:")
    print(f"• Aggressive agents tend to make self-serving proposals")
    print(f"• Cooperative agents are more likely to accept proposals") 
    print(f"• Strategic agents balance self-interest with group dynamics")
    print(f"• Response patterns can reveal strategic sophistication")


async def demo_experiment_configuration():
    """Demonstrate experiment configuration system."""
    print(f"\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION DEMO")
    print("=" * 60)
    
    # Create a simulated experiment configuration
    print("1. Creating Simulated Experiment Configuration...")
    sim_config = create_simulated_experiment(
        experiment_name="Strategic Behavior Study",
        strategic_levels=["aggressive", "cooperative", "balanced"]
    )
    
    print(f"   Experiment: {sim_config.experiment_name}")
    print(f"   Agents: {len(sim_config.agents)}")
    print(f"   Setup: {sim_config.m_items} items, {sim_config.t_rounds} rounds")
    print(f"   Competition Level: {sim_config.competition_level}")
    print(f"   Tags: {sim_config.tags}")
    
    # Demonstrate configuration saving/loading
    print("\n2. Saving Configuration...")
    config_path = "experiments/configs/demo_experiment.json"
    sim_config.save_to_file(config_path)
    print(f"   Saved to: {config_path}")
    
    # Show what a real-world configuration would look like
    print("\n3. Real-World Configuration Example (O3 vs Claude Haiku):")
    print("   Note: This would require actual API keys in environment variables")
    print("   OPENAI_API_KEY and ANTHROPIC_API_KEY")
    
    try:
        # This will work if API keys are set
        real_config = create_o3_vs_haiku_experiment(
            experiment_name="O3 vs Haiku Strategic Study",
            competition_level=0.95,
            random_seed=42
        )
        print(f"   ✅ Created: {real_config.experiment_name}")
        print(f"   Strong Agent: {real_config.agents[0].model_type.value}")
        print(f"   Weak Agents: {[a.model_type.value for a in real_config.agents[1:]]}")
    except ValueError as e:
        print(f"   ⚠️  API keys not configured: {e}")
        print(f"   Set environment variables to run real experiments")


async def main():
    """Run the complete LLM agent integration demo."""
    print("Multi-LLM Agent Integration for Negotiation Research")
    print("=" * 60)
    
    # Demo 1: Create agents and environment
    agents, env, preferences, items = await demo_simulated_agents()
    
    # Demo 2: Run negotiation round
    consensus_reached, utilities = await demo_negotiation_round(agents, env, preferences, items)
    
    # Demo 3: Analyze strategic behavior
    await demo_strategic_behavior_analysis(agents)
    
    # Demo 4: Show experiment configuration
    await demo_experiment_configuration()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("INTEGRATION DEMO COMPLETE")
    print("=" * 60)
    
    print(f"\n🎯 Key Integration Points Demonstrated:")
    print(f"✅ Multi-LLM agent creation with different strategic profiles")
    print(f"✅ Negotiation environment integration")
    print(f"✅ Competitive preference system")
    print(f"✅ Async negotiation round execution")
    print(f"✅ Strategic behavior analysis")
    print(f"✅ Experiment configuration management")
    print(f"✅ Performance tracking and cost estimation")
    
    print(f"\n📊 Research Applications:")
    print(f"• Strategic Behavior Detection: Compare responses across agent types")
    print(f"• Scaling Laws: Test stronger vs weaker model combinations")
    print(f"• Exploitation Analysis: Identify manipulation, anger, gaslighting")
    print(f"• Preference System Impact: Vector vs matrix preference effects")
    print(f"• Configuration Flexibility: Easy model swapping for experiments")
    
    print(f"\n🚀 Next Steps for Research:")
    print(f"1. Set up API keys for real LLM experiments")
    print(f"2. Configure SLURM jobs for Princeton cluster execution")
    print(f"3. Implement conversation logging and analysis")
    print(f"4. Run pilot O3 vs Claude Haiku experiments")
    print(f"5. Develop statistical analysis pipeline")
    
    if consensus_reached and utilities:
        print(f"\n🏆 Demo Negotiation Winner: {max(utilities.items(), key=lambda x: x[1])[0]}")
        print(f"   Final Utilities: {utilities}")
    
    print(f"\nThe multi-LLM integration system is ready for strategic behavior research! 🎉")


if __name__ == "__main__":
    asyncio.run(main())