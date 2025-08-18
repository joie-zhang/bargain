#!/usr/bin/env python3
"""
Strong Models Competition Experiment

This script runs negotiations between state-of-the-art large language models:
- Gemini 2.5 Pro (Google's latest reasoning model)
- Claude 4 Sonnet (Anthropic's latest)
- Llama 3.1 405B (Meta's largest open model)
- Qwen 3 235B (Alibaba's strong model)

All models are accessed via OpenRouter for consistent API access.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from negotiation import (
    AgentFactory,
    AgentConfiguration,
    ExperimentConfiguration,
    create_competitive_preferences,
    ModularNegotiationRunner,
)
from negotiation.llm_agents import ModelType
from negotiation.openrouter_client import OpenRouterAgent


# Configuration for strong models via OpenRouter
STRONG_MODELS_CONFIG = {
    "gemini-pro": {
        "name": "Gemini Pro 2.5",
        "model_id": "google/gemini-2.5-pro",  # Google's advanced model
        "provider": "Google",
        "description": "Google's advanced reasoning model",
        "temperature": 0.7,
        "system_prompt": "You are Gemini Pro 2.5, an advanced reasoning model. Use strategic thinking to achieve optimal outcomes in this negotiation."
    },
    "claude-4-sonnet": {
        "name": "Claude 4 Sonnet",
        "model_id": "anthropic/claude-sonnet-4",
        "provider": "Anthropic", 
        "description": "Anthropic's latest and most capable model",
        "temperature": 0.7,
        "system_prompt": "You are Claude 4 Sonnet. Apply sophisticated reasoning and strategic thinking to maximize your utility in this negotiation."
    },
    "llama-3-1-405b": {
        "name": "Llama 3.1 405B",
        "model_id": "meta-llama/llama-3.1-405b-instruct",
        "provider": "Meta",
        "description": "Meta's largest open-source model with 405B parameters",
        "temperature": 0.7,
        "system_prompt": "You are Llama 3.1 405B, a highly capable large language model. Use your extensive knowledge and reasoning to negotiate effectively."
    },
    "qwen-3-235b-a22b-2507": {
        "name": "Qwen 3 235B", 
        "model_id": "qwen/qwen3-235b-a22b-2507",
        "provider": "Alibaba",
        "description": "Alibaba's strong multilingual model",
        "temperature": 0.7,
        "system_prompt": "You are Qwen 3 235B, an advanced AI model. Apply strategic analysis to achieve the best possible outcomes in this negotiation."
    }
}


async def run_strong_models_negotiation(
    models: List[str],
    num_runs: int = 10,
    max_rounds: int = 15,
    num_items: int = 6,
    competition_level: float = 0.95
):
    """
    Run negotiation experiments between strong models.
    
    Args:
        models: List of model names to use
        num_runs: Number of runs to perform
        max_rounds: Maximum rounds per negotiation
        num_items: Number of items to negotiate over
        competition_level: How competitive the preferences are (0-1)
    """
    # Check for OpenRouter API key
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("StrongModels")
    
    # Create results directory
    results_dir = Path("experiments/results/strong_models")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Track all results
    all_results = {
        "experiment": "Strong Models Competition",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "models": models,
            "num_runs": num_runs,
            "max_rounds": max_rounds,
            "num_items": num_items,
            "competition_level": competition_level
        },
        "runs": []
    }
    
    # Create agent factory
    factory = AgentFactory()
    
    for run_idx in range(num_runs):
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN {run_idx + 1}/{num_runs}")
        logger.info(f"{'='*60}")
        
        try:
            # Track timing for this run
            run_start_time = time.time()
            
            # Create agents for this run
            agents = []
            agent_configs = []
            
            for i, model_name in enumerate(models):
                if model_name not in STRONG_MODELS_CONFIG:
                    logger.warning(f"Unknown model: {model_name}, skipping")
                    continue
                    
                config = STRONG_MODELS_CONFIG[model_name]
                
                # Create agent configuration
                # Using GEMMA_2_27B as the base type since OpenRouter will override with actual model
                agent_config = AgentConfiguration(
                    agent_id=f"{model_name.replace('-', '_')}_{i+1}",
                    model_type=ModelType.GEMMA_2_27B,  # Base type for OpenRouter models
                    api_key=openrouter_key,
                    temperature=config["temperature"],
                    max_tokens=4000,
                    system_prompt=config["system_prompt"],
                    custom_parameters={"model_id": config["model_id"]}
                )
                agent_configs.append(agent_config)
                
                # Create OpenRouter agent directly with specific model
                from negotiation.llm_agents import LLMConfig
                llm_config = agent_config.to_llm_config()
                agent = OpenRouterAgent(
                    agent_id=agent_config.agent_id,
                    llm_config=llm_config,
                    api_key=openrouter_key,
                    model_id=config["model_id"]
                )
                agents.append(agent)
            
            if len(agents) < 2:
                logger.error("Need at least 2 agents for negotiation")
                continue
            
            # Create preferences using PreferenceManager
            preference_manager = create_competitive_preferences(
                n_agents=len(agents),
                m_items=num_items,
                cosine_similarity=competition_level
            )
            
            # Generate the actual preference data
            preferences_data = preference_manager.generate_preferences()
            
            # Create item names
            items = [f"Item_{i}" for i in range(num_items)]
            
            # Create negotiation runner
            runner = ModularNegotiationRunner(
                agents=agents,
                preferences=preference_manager,  # Pass the PreferenceManager object to runner
                items=items,
                max_rounds=max_rounds,
                discount_factor=0.9,
                log_level="INFO"
            )
            
            # Run the negotiation
            logger.info(f"Starting negotiation between: {', '.join([a.agent_id for a in agents])}")
            outcome = await runner.run_negotiation()
            
            # Record EVERYTHING - comprehensive data capture
            # Extract preferences from the generated preferences_data
            serialized_prefs = None
            preference_details = {}
            
            # Get the actual preference vectors from the generated data
            if preferences_data and "agent_preferences" in preferences_data:
                agent_prefs = preferences_data["agent_preferences"]
                
                # Map agent IDs from runner to preference data
                # The preference system uses agent_0, agent_1, etc., but our agents have custom IDs
                serialized_prefs = []
                for i, agent in enumerate(agents):
                    # Get preferences for agent_i (indexed position)
                    agent_key = f"agent_{i}"
                    if agent_key in agent_prefs:
                        pref_vector = agent_prefs[agent_key]
                        serialized_prefs.append(pref_vector)
                        
                        # Create detailed preference mapping
                        preference_details[agent.agent_id] = {
                            f"item_{j}": float(val)
                            for j, val in enumerate(pref_vector)
                        }
                    else:
                        preference_details[agent.agent_id] = {"error": f"No preferences found for agent index {i}"}
            
            # Also extract cosine similarities if available
            cosine_similarities = None
            if preferences_data and "cosine_similarities" in preferences_data:
                cosine_similarities = preferences_data["cosine_similarities"]
            
            # Extract ALL conversation data from runner
            conversation_logs = []
            all_proposals = []
            all_votes = []
            
            if hasattr(runner, 'conversation_logs'):
                conversation_logs = runner.conversation_logs
            if hasattr(runner, 'proposal_history'):
                all_proposals = runner.proposal_history
            if hasattr(runner, 'vote_history'):
                all_votes = runner.vote_history
            
            # Extract per-round data if available
            round_data = []
            if hasattr(runner, 'round_history'):
                round_data = runner.round_history
            elif hasattr(outcome, 'round_summaries'):
                round_data = outcome.round_summaries
            
            run_result = {
                "run_index": run_idx,
                "experiment_config": {
                    "num_agents": len(agents),
                    "num_items": num_items,
                    "max_rounds": max_rounds,
                    "competition_level": competition_level,
                    "discount_factor": 0.9
                },
                "agents": [a.agent_id for a in agents],
                "agent_models": {a.agent_id: config["model_id"] for a, model_name in zip(agents, models) if model_name in STRONG_MODELS_CONFIG for config in [STRONG_MODELS_CONFIG[model_name]]},
                "initial_preferences_matrix": serialized_prefs,
                "initial_preferences_by_agent": preference_details,
                "preference_cosine_similarities": cosine_similarities,
                "negotiation_outcome": {
                    "consensus_reached": outcome.consensus_reached,
                    "final_round": outcome.final_round,
                    "final_allocation": outcome.final_allocation,
                    "final_utilities": outcome.final_utilities,
                    "utility_realized": outcome.utility_realized if hasattr(outcome, 'utility_realized') else None
                },
                "complete_conversation_log": conversation_logs,
                "all_proposals": all_proposals,
                "all_votes": all_votes,
                "round_summaries": round_data,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": time.time() - run_start_time
            }
            
            # Calculate some metrics
            if outcome.final_utilities:
                utilities = list(outcome.final_utilities.values())
                run_result["metrics"] = {
                    "mean_utility": sum(utilities) / len(utilities),
                    "max_utility": max(utilities),
                    "min_utility": min(utilities),
                    "utility_spread": max(utilities) - min(utilities)
                }
            
            all_results["runs"].append(run_result)
            
            # Save intermediate results
            intermediate_file = results_dir / f"strong_models_run_{run_idx}.json"
            with open(intermediate_file, 'w') as f:
                json.dump(run_result, f, indent=2)
            
            logger.info(f"Run {run_idx + 1} completed: Consensus={'Yes' if outcome.consensus_reached else 'No'}, Rounds={outcome.final_round}")
            
        except Exception as e:
            logger.error(f"Error in run {run_idx + 1}: {e}")
            all_results["runs"].append({
                "run_index": run_idx,
                "error": str(e)
            })
    
    # Calculate summary statistics
    successful_runs = [r for r in all_results["runs"] if "error" not in r]
    if successful_runs:
        consensus_runs = [r for r in successful_runs if r.get("negotiation_outcome", {}).get("consensus_reached", False)]
        all_results["summary"] = {
            "total_runs": num_runs,
            "successful_runs": len(successful_runs),
            "consensus_rate": len(consensus_runs) / len(successful_runs) if successful_runs else 0,
            "avg_rounds": sum(r.get("negotiation_outcome", {}).get("final_round", 0) for r in successful_runs) / len(successful_runs) if successful_runs else 0,
            "avg_utility_spread": sum(r.get("metrics", {}).get("utility_spread", 0) for r in successful_runs) / len(successful_runs) if successful_runs else 0
        }
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = results_dir / f"strong_models_results_{timestamp}.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"{'='*60}")
    if "summary" in all_results:
        logger.info(f"Consensus Rate: {all_results['summary']['consensus_rate']:.1%}")
        logger.info(f"Average Rounds: {all_results['summary']['avg_rounds']:.1f}")
        logger.info(f"Average Utility Spread: {all_results['summary']['avg_utility_spread']:.3f}")
    logger.info(f"Results saved to: {final_file}")
    
    return all_results


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run negotiations between strong language models via OpenRouter"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(STRONG_MODELS_CONFIG.keys()),
        default=["gemini-pro", "claude-4-sonnet", "llama-3-1-405b", "qwen-3-235b-a22b-2507"],
        help="Models to include in negotiation"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs (default: 10)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=15,
        help="Maximum rounds per negotiation (default: 15)"
    )
    
    parser.add_argument(
        "--items",
        type=int,
        default=6,
        help="Number of items to negotiate over (default: 6)"
    )
    
    parser.add_argument(
        "--competition",
        type=float,
        default=0.95,
        help="Competition level 0-1 (default: 0.95 for high competition)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("STRONG MODELS NEGOTIATION EXPERIMENT")
    print("="*60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Runs: {args.runs}")
    print(f"Max Rounds: {args.rounds}")
    print(f"Items: {args.items}")
    print(f"Competition Level: {args.competition}")
    print("="*60 + "\n")
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable is required")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Run the experiment
    results = await run_strong_models_negotiation(
        models=args.models,
        num_runs=args.runs,
        max_rounds=args.rounds,
        num_items=args.items,
        competition_level=args.competition
    )
    
    return results


if __name__ == "__main__":
    asyncio.run(main())