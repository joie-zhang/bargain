#!/usr/bin/env python3
"""
Modular Strong Models Competition Experiment

This is a refactored, modular version of the strong models experiment
that is easier to debug and maintain.
"""

import asyncio
import logging
import os
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from negotiation import (
    AgentFactory,
    AgentConfiguration,
    create_competitive_preferences,
    NegotiationContext
)
from negotiation.llm_agents import ModelType, BaseLLMAgent
from negotiation.openrouter_client import OpenRouterAgent
from negotiation.experiment_phases import NegotiationPhases
from negotiation.experiment_analysis import ExperimentAnalyzer, ExperimentLogger

# Import the configuration from the original file
from run_strong_models_experiment import (
    STRONG_MODELS_CONFIG,
    ExperimentResults,
    BatchResults
)


class ModularStrongModelsExperiment:
    """
    Modular implementation of the strong models negotiation experiment.
    
    This version separates concerns for better debugging and maintainability:
    - Phases are handled by NegotiationPhases class
    - Analysis is handled by ExperimentAnalyzer class
    - Logging is handled by ExperimentLogger class
    """
    
    def __init__(self, log_level: str = "INFO"):
        """Initialize the modular experiment runner."""
        self.logger = self._setup_logging(log_level)
        self.results_dir = Path("experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modular components
        self.phases = NegotiationPhases(self.logger)
        self.analyzer = ExperimentAnalyzer(self.logger)
        self.exp_logger = ExperimentLogger(self.results_dir, self.logger)
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ModularStrongModelsExperiment")
        logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Clear existing handlers
        logger.handlers = []
        logger.addHandler(console_handler)
        
        return logger
    
    async def run_single_experiment(
        self,
        models: List[str],
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> ExperimentResults:
        """
        Run a single negotiation experiment with strong models.
        
        This method orchestrates the entire experiment flow using modular components.
        """
        # Generate experiment ID
        experiment_id = f"strong_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        experiment_start_time = time.time()
        
        # Start logging for this experiment
        self.exp_logger.start_experiment(experiment_id)
        
        # Default configuration
        default_config = {
            "m_items": 5,
            "n_agents": len(models),
            "t_rounds": 10,
            "gamma_discount": 0.9,
            "competition_level": 0.95,
            "random_seed": None
        }
        
        # Merge with provided config
        config = {**default_config, **(experiment_config or {})}
        
        # Set random seed if provided
        if config["random_seed"]:
            random.seed(config["random_seed"])
        
        self.logger.info(f"Starting experiment {experiment_id}")
        self.logger.info(f"Configuration: {config}")
        
        # Create agents
        agents = await self._create_agents(models, config)
        if not agents:
            raise ValueError("Failed to create agents")
        
        # Create items
        items = self._create_items(config["m_items"])
        
        # Create preferences
        preferences = self._create_preferences(agents, items, config)
        
        # Run the experiment phases
        try:
            results = await self._run_experiment_phases(
                agents, items, preferences, config, 
                experiment_id, experiment_start_time
            )
        except Exception as e:
            self.logger.error(f"Error during negotiation: {e}")
            raise
        
        # Save final results
        self.exp_logger.save_experiment_results(results.to_dict())
        
        return results
    
    async def _run_experiment_phases(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict,
        config: Dict,
        experiment_id: str,
        experiment_start_time: float
    ) -> ExperimentResults:
        """
        Run all experiment phases in sequence.
        
        This method coordinates the 14-phase negotiation protocol.
        """
        # Initialize tracking variables
        consensus_reached = False
        final_round = 0
        winner_agent_id = None
        final_utilities = {}
        conversation_logs = []
        
        # Create save callback for logging
        save_callback = lambda *args: self.exp_logger.save_interaction(*args)
        
        # Phase 1A: Game Setup
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 1A: GAME SETUP")
        self.logger.info("="*60)
        await self.phases.run_game_setup(agents, items, config, save_callback)
        
        # Phase 1B: Private Preference Assignment
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 1B: PREFERENCE ASSIGNMENT")
        self.logger.info("="*60)
        await self.phases.run_preference_assignment(
            agents, items, preferences, config, save_callback
        )
        
        # Main negotiation rounds
        for round_num in range(1, config["t_rounds"] + 1):
            self.logger.info("\n" + "="*60)
            self.logger.info(f"ROUND {round_num}/{config['t_rounds']}")
            self.logger.info("="*60)
            
            # Phase 2: Public Discussion
            discussion_result = await self.phases.run_discussion(
                agents, items, preferences, round_num, 
                config["t_rounds"], save_callback
            )
            conversation_logs.extend(discussion_result.get("messages", []))
            
            # Phase 3: Private Thinking
            thinking_result = await self.phases.run_private_thinking(
                agents, items, preferences, round_num, 
                config["t_rounds"], discussion_result.get("messages", []),
                save_callback
            )
            
            # Phase 4A: Proposal Submission
            proposal_result = await self.phases.run_proposals(
                agents, items, preferences, round_num, 
                config["t_rounds"], save_callback
            )
            conversation_logs.extend(proposal_result.get("messages", []))
            
            # Phase 4B: Proposal Enumeration
            enumeration_result = await self.phases.run_proposal_enumeration(
                agents, items, round_num, 
                proposal_result.get("proposals", [])
            )
            conversation_logs.extend(enumeration_result.get("messages", []))
            
            # Phase 5A: Private Voting
            voting_result = await self.phases.run_private_voting(
                agents, items, preferences, round_num, 
                config["t_rounds"], proposal_result.get("proposals", []),
                enumeration_result.get("enumerated_proposals", []),
                save_callback
            )
            
            # Phase 5B: Vote Tabulation
            tabulation_result = await self.phases.run_vote_tabulation(
                agents, items, preferences, round_num,
                voting_result.get("private_votes", []),
                enumeration_result.get("enumerated_proposals", [])
            )
            conversation_logs.extend(tabulation_result.get("messages", []))
            
            # Check for consensus
            if tabulation_result.get("consensus_reached", False):
                consensus_reached = True
                final_round = round_num
                winner_agent_id = tabulation_result.get("winner_agent_id")
                final_utilities = tabulation_result.get("final_utilities", {})
                self.logger.info(f"✅ CONSENSUS REACHED in round {round_num}!")
                break
            
            # Phase 6: Individual Reflection (if not consensus)
            if round_num < config["t_rounds"]:
                reflection_result = await self.phases.run_reflection(
                    agents, items, preferences, round_num, 
                    config["t_rounds"], tabulation_result,
                    save_callback
                )
        
        # Calculate final utilities if no consensus
        if not consensus_reached:
            final_round = config["t_rounds"]
            self.logger.info(f"❌ No consensus after {config['t_rounds']} rounds")
        
        # Analyze results
        exploitation_detected = self.analyzer.detect_exploitation(conversation_logs)
        strategic_behaviors = self.analyzer.analyze_strategic_behaviors(conversation_logs)
        agent_performance = self.analyzer.analyze_agent_performance(agents, final_utilities)
        
        # Analyze win patterns
        win_patterns = self.analyzer.analyze_win_patterns(
            agents, winner_agent_id, final_utilities, strategic_behaviors
        )
        
        # Determine model winners
        model_winners = {}
        if winner_agent_id and final_utilities:
            for agent in agents:
                model_name = self.analyzer._extract_model_name(agent.agent_id)
                model_winners[model_name] = (agent.agent_id == winner_agent_id)
        
        # Create enhanced config for results
        enhanced_config = self._create_enhanced_config(
            config, agents, preferences, items, 
            experiment_start_time, experiment_id
        )
        
        # Create results object
        results = ExperimentResults(
            experiment_id=experiment_id,
            timestamp=time.time(),
            config=enhanced_config,
            consensus_reached=consensus_reached,
            final_round=final_round,
            winner_agent_id=winner_agent_id,
            final_utilities=final_utilities,
            strategic_behaviors=strategic_behaviors,
            conversation_logs=conversation_logs,
            agent_performance=agent_performance,
            exploitation_detected=exploitation_detected,
            model_winners=model_winners
        )
        
        return results
    
    async def run_batch_experiments(
        self,
        models: List[str],
        num_runs: int = 10,
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> BatchResults:
        """Run multiple experiments and aggregate results."""
        batch_id = f"batch_strong_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiments = []
        
        self.logger.info(f"Starting batch experiment {batch_id} with {num_runs} runs")
        
        for i in range(num_runs):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"BATCH RUN {i+1}/{num_runs}")
            self.logger.info(f"{'='*60}")
            
            try:
                result = await self.run_single_experiment(models, experiment_config)
                experiments.append(result)
            except Exception as e:
                self.logger.error(f"Error in run {i+1}: {e}")
                continue
        
        # Calculate aggregate statistics
        model_win_rates = self.analyzer.calculate_model_win_rates(experiments, models)
        consensus_rate = sum(1 for exp in experiments if exp.consensus_reached) / len(experiments) if experiments else 0
        average_rounds = sum(exp.final_round for exp in experiments) / len(experiments) if experiments else 0
        exploitation_rate = sum(1 for exp in experiments if exp.exploitation_detected) / len(experiments) if experiments else 0
        
        # Aggregate strategic behaviors
        strategic_behaviors_summary = self.analyzer.aggregate_strategic_behaviors(experiments)
        
        batch_results = BatchResults(
            batch_id=batch_id,
            num_runs=len(experiments),
            experiments=experiments,
            model_win_rates=model_win_rates,
            consensus_rate=consensus_rate,
            average_rounds=average_rounds,
            exploitation_rate=exploitation_rate,
            strategic_behaviors_summary=strategic_behaviors_summary
        )
        
        # Save batch results
        self.exp_logger.save_batch_results(batch_results.to_dict())
        
        return batch_results
    
    async def _create_agents(self, models: List[str], config: Dict[str, Any]) -> List[BaseLLMAgent]:
        """Create agents for the specified models."""
        agents = []
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # If only one model specified, create 3 agents of that model
        if len(models) == 1:
            models = models * 3
        
        for i, model_name in enumerate(models):
            if model_name not in STRONG_MODELS_CONFIG:
                self.logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            model_config = STRONG_MODELS_CONFIG[model_name]
            
            # Create agent configuration
            agent_config = AgentConfiguration(
                agent_id=f"{model_name.replace('-', '_')}_{i+1}",
                model_type=ModelType.GEMMA_2_27B,  # Base type for OpenRouter
                api_key=openrouter_key,
                temperature=model_config["temperature"],
                max_tokens=4000,
                system_prompt=model_config["system_prompt"],
                custom_parameters={"model_id": model_config["model_id"]}
            )
            
            # Create OpenRouter agent
            from negotiation.llm_agents import LLMConfig
            llm_config = agent_config.to_llm_config()
            agent = OpenRouterAgent(
                agent_id=agent_config.agent_id,
                llm_config=llm_config,
                api_key=openrouter_key,
                model_id=model_config["model_id"]
            )
            agents.append(agent)
        
        return agents
    
    def _create_items(self, num_items: int) -> List[Dict[str, str]]:
        """Create items for negotiation."""
        item_names = ["Apple", "Jewel", "Stone", "Quill", "Pencil", "Book", "Hat", "Camera"]
        return [{"name": item_names[i] if i < len(item_names) else f"Item_{i}"} 
                for i in range(num_items)]
    
    def _create_preferences(self, agents: List[BaseLLMAgent], items: List[Dict], config: Dict) -> Dict:
        """Create preferences for agents."""
        preference_manager = create_competitive_preferences(
            n_agents=len(agents),
            m_items=len(items),
            cosine_similarity=config["competition_level"]
        )
        
        preferences_data = preference_manager.generate_preferences()
        
        # Map to agent IDs
        agent_preferences = {}
        for i, agent in enumerate(agents):
            if "agent_preferences" in preferences_data:
                agent_preferences[agent.agent_id] = preferences_data["agent_preferences"][f"agent_{i}"]
            else:
                # Fallback to basic preference generation
                agent_preferences[agent.agent_id] = [random.uniform(0, 10) for _ in range(len(items))]
        
        return {
            "agent_preferences": agent_preferences,
            "cosine_similarities": preferences_data.get("cosine_similarities", {})
        }
    
    def _create_enhanced_config(
        self, config, agents, preferences, items, 
        start_time, experiment_id
    ) -> Dict:
        """Create enhanced configuration with all experiment details."""
        return {
            **config,
            "experiment_id": experiment_id,
            "start_time": start_time,
            "agents": [agent.agent_id for agent in agents],
            "items": items,
            "preferences_summary": {
                "type": "competitive",
                "cosine_similarities": preferences.get("cosine_similarities", {})
            }
        }


async def main():
    """Main entry point for the modular experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run modular negotiations between strong language models"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(STRONG_MODELS_CONFIG.keys()),
        default=["gemini-pro", "claude-4-sonnet", "llama-3-1-405b"],
        help="Models to include in negotiation"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiments"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of experiments in batch"
    )
    
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum rounds per negotiation"
    )
    
    parser.add_argument(
        "--num-items",
        type=int,
        default=5,
        help="Number of items to negotiate"
    )
    
    parser.add_argument(
        "--competition-level",
        type=float,
        default=0.95,
        help="Competition level (0-1)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable is required")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return 1
    
    print("=" * 60)
    print("MODULAR STRONG MODELS NEGOTIATION EXPERIMENT")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Max Rounds: {args.max_rounds}")
    print(f"Items: {args.num_items}")
    print(f"Competition Level: {args.competition_level}")
    print(f"Log Level: {args.log_level}")
    if args.random_seed:
        print(f"Random Seed: {args.random_seed}")
    print("=" * 60)
    
    # Create experiment configuration
    experiment_config = {
        "m_items": args.num_items,
        "t_rounds": args.max_rounds,
        "competition_level": args.competition_level,
        "random_seed": args.random_seed
    }
    
    # Initialize experiment runner
    experiment = ModularStrongModelsExperiment(log_level=args.log_level)
    
    try:
        if args.batch:
            print(f"\n--- Batch Experiment ({args.batch_size} runs) ---")
            batch_results = await experiment.run_batch_experiments(
                models=args.models,
                num_runs=args.batch_size,
                experiment_config=experiment_config
            )
            
            print(f"\n📈 Batch Results Summary:")
            print(f"  Batch ID: {batch_results.batch_id}")
            print(f"  Successful Runs: {batch_results.num_runs}")
            print(f"  Model Win Rates:")
            for model, rate in batch_results.model_win_rates.items():
                print(f"    {model}: {rate:.1%}")
            print(f"  Consensus Rate: {batch_results.consensus_rate:.1%}")
            print(f"  Average Rounds: {batch_results.average_rounds:.1f}")
            print(f"  Exploitation Rate: {batch_results.exploitation_rate:.1%}")
        else:
            print("\n--- Single Experiment Test ---")
            single_result = await experiment.run_single_experiment(
                models=args.models,
                experiment_config=experiment_config
            )
            
            print(f"\n📊 Single Experiment Results:")
            print(f"  Experiment ID: {single_result.experiment_id}")
            print(f"  Consensus Reached: {'✓' if single_result.consensus_reached else '✗'}")
            print(f"  Final Round: {single_result.final_round}")
            print(f"  Winner: {single_result.winner_agent_id}")
            print(f"  Exploitation Detected: {'✓' if single_result.exploitation_detected else '✗'}")
            
            if single_result.final_utilities:
                print(f"  Final Utilities:")
                for agent_id, utility in single_result.final_utilities.items():
                    print(f"    {agent_id}: {utility:.2f}")
        
        print(f"\n✅ Modular experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        logging.exception("Experiment error")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))