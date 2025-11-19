#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Name Experiment Runner

This script runs negotiations to test how agent names affect performance.
Both agents use the same model, and we test various name combinations.
"""

import asyncio
import os
import sys
import locale
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Set UTF-8 encoding
if sys.platform != 'win32':
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strong_models_experiment import StrongModelsExperiment, STRONG_MODELS_CONFIG, ExperimentResults, BatchResults
from strong_models_experiment.phases import PhaseHandler
from strong_models_experiment.utils import ExperimentUtils, FileManager
from strong_models_experiment.analysis import ExperimentAnalyzer

# Import name_experiment modules from the same directory
import importlib.util
name_experiments_dir = Path(__file__).parent

# Import name_agent_factory
factory_path = name_experiments_dir / "name_agent_factory.py"
factory_spec = importlib.util.spec_from_file_location("name_agent_factory", factory_path)
name_agent_factory = importlib.util.module_from_spec(factory_spec)
factory_spec.loader.exec_module(name_agent_factory)
NameAgentFactory = name_agent_factory.NameAgentFactory

# Import name_experiment_config
config_path = name_experiments_dir / "name_experiment_config.py"
config_spec = importlib.util.spec_from_file_location("name_experiment_config", config_path)
name_experiment_config = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(name_experiment_config)


class NameExperiment(StrongModelsExperiment):
    """
    Extended experiment class for name-based experiments.
    Uses custom agent names instead of Greek letters.
    """
    
    def __init__(self, output_dir=None):
        """Initialize the name experiment runner."""
        # Initialize parent but replace agent factory
        super().__init__(output_dir)
        self.agent_factory = NameAgentFactory()
    
    async def run_single_name_experiment(
        self,
        model: str,
        agent_names: List[str],
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> ExperimentResults:
        """
        Run a single negotiation experiment with custom agent names.
        
        Args:
            model: Single model name to use for all agents
            agent_names: List of agent names (e.g., ["Alice", "Bob"])
            experiment_config: Optional configuration overrides
            
        Returns:
            ExperimentResults object with complete experiment data
        """
        # Generate experiment ID
        experiment_id = self.utils.generate_experiment_id(self.current_batch_id, self.current_run_number)
        experiment_start_time = time.time()
        self.current_experiment_id = experiment_id
        
        # Initialize interaction storage
        self.all_interactions = []
        self.agent_interactions = {}
        
        # Default configuration
        default_config = {
            "m_items": 5,
            "n_agents": len(agent_names),
            "t_rounds": 10,
            "gamma_discount": 0.9,
            "competition_level": 0.95,
            "random_seed": None
        }
        
        # Merge with provided config
        config = {**default_config, **(experiment_config or {})}
        
        # Set random seed if provided
        if config["random_seed"]:
            import random
            random.seed(config["random_seed"])
        
        self.logger.info(f"Starting name experiment {experiment_id}")
        self.logger.info(f"Model: {model}, Agent names: {agent_names}")
        
        # Extract token configuration from config
        token_config = {
            "discussion": config.get("max_tokens_discussion", None),
            "proposal": config.get("max_tokens_proposal", None),
            "voting": config.get("max_tokens_voting", None),
            "reflection": config.get("max_tokens_reflection", None),
            "thinking": config.get("max_tokens_thinking", None),
            "default": config.get("max_tokens_default", None)
        }
        
        # Initialize phase handler with token config for this experiment
        self.phase_handler = PhaseHandler(
            save_interaction_callback=self._save_interaction,
            token_config=token_config
        )
        
        # Create agents with custom names, all using the same model
        agents = await self.agent_factory.create_agents(model, agent_names, config)
        if not agents:
            raise ValueError("Failed to create agents")
        
        # Create items and preferences
        items = self.utils.create_items(config["m_items"])
        preferences = self.utils.create_preferences(agents, items, config)
        
        # Initialize tracking variables
        consensus_reached = False
        final_round = 0
        final_utilities = {}
        final_allocation = {}
        agent_preferences_data = {}
        strategic_behaviors = {}
        conversation_logs = []
        
        # Run the 14-phase negotiation (reuse parent's negotiation logic)
        try:
            # Phase 1A: Game Setup
            await self.phase_handler.run_game_setup_phase(agents, items, preferences, config)
            
            # Phase 1B: Private Preference Assignment
            await self.phase_handler.run_private_preference_assignment(agents, items, preferences, config)
            
            # Main negotiation rounds
            for round_num in range(1, config["t_rounds"] + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"ROUND {round_num}/{config['t_rounds']}")
                self.logger.info(f"{'='*60}")
                
                # Phase 2: Public Discussion (optional)
                discussion_result = {"messages": []}
                if not config.get("disable_discussion", False):
                    discussion_result = await self.phase_handler.run_discussion_phase(
                        agents, items, preferences, round_num, config["t_rounds"]
                    )
                    conversation_logs.extend(discussion_result.get("messages", []))
                else:
                    self.logger.info(f"⏭️  Skipping discussion phase (disabled)")
                
                # Phase 3: Private Thinking (optional)
                thinking_result = {}
                if not config.get("disable_thinking", False):
                    thinking_result = await self.phase_handler.run_private_thinking_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        discussion_result.get("messages", [])
                    )
                else:
                    self.logger.info(f"⏭️  Skipping private thinking phase (disabled)")
                
                # Phase 4A: Proposal Submission
                proposal_result = await self.phase_handler.run_proposal_phase(
                    agents, items, preferences, round_num, config["t_rounds"]
                )
                conversation_logs.extend(proposal_result.get("messages", []))
                
                # Phase 4B: Proposal Enumeration
                enumeration_result = await self.phase_handler.run_proposal_enumeration_phase(
                    agents, items, preferences, round_num, config["t_rounds"],
                    proposal_result.get("proposals", [])
                )
                conversation_logs.extend(enumeration_result.get("messages", []))
                
                # Phase 5A: Private Voting
                voting_result = await self.phase_handler.run_private_voting_phase(
                    agents, items, preferences, round_num, config["t_rounds"],
                    proposal_result.get("proposals", []),
                    enumeration_result.get("enumerated_proposals", [])
                )
                
                # Phase 5B: Vote Tabulation
                tabulation_result = await self.phase_handler.run_vote_tabulation_phase(
                    agents, items, preferences, round_num, config["t_rounds"],
                    voting_result.get("private_votes", []),
                    enumeration_result.get("enumerated_proposals", [])
                )
                conversation_logs.extend(tabulation_result.get("messages", []))
                
                # Check for consensus
                if tabulation_result.get("consensus_reached", False):
                    consensus_reached = True
                    final_round = round_num
                    final_utilities = tabulation_result.get("final_utilities", {})
                    final_allocation = tabulation_result.get("final_allocation", {})
                    agent_preferences_data = tabulation_result.get("agent_preferences", {})
                    self.logger.info(f"✅ CONSENSUS REACHED in round {round_num}!")
                    break
                
                # Phase 6: Individual Reflection (optional)
                reflection_result = {}
                if not config.get("disable_reflection", False):
                    reflection_result = await self.phase_handler.run_individual_reflection_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        tabulation_result
                    )
                else:
                    self.logger.info(f"⏭️  Skipping individual reflection phase (disabled)")
            
            # Calculate final utilities if no consensus
            if not consensus_reached:
                final_round = config["t_rounds"]
                self.logger.info(f"❌ No consensus after {config['t_rounds']} rounds")
        
        except Exception as e:
            self.logger.error(f"Error during negotiation: {e}")
            raise
        
        # Analyze results
        exploitation_detected = self.analyzer.detect_exploitation(conversation_logs)
        strategic_behaviors = self.analyzer.analyze_strategic_behaviors(conversation_logs)
        agent_performance = self.analyzer.analyze_agent_performance(agents, final_utilities)
        
        # Create enhanced config with name information
        enhanced_config = self.utils.create_enhanced_config(
            config, agents, preferences, items, experiment_start_time, experiment_id
        )
        # Add name-specific metadata
        enhanced_config["agent_names"] = agent_names
        enhanced_config["model"] = model
        enhanced_config["experiment_type"] = "name_experiment"
        
        # Save all interactions
        self._stream_save_json()
        
        # Determine save location
        if self.use_custom_output:
            exp_dir = self.results_dir
        elif self.current_batch_id and self.current_run_number:
            exp_dir = self.results_dir / self.current_batch_id
        else:
            exp_dir = self.results_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results object
        results = ExperimentResults(
            experiment_id=experiment_id,
            timestamp=time.time(),
            config=enhanced_config,
            consensus_reached=consensus_reached,
            final_round=final_round,
            final_utilities=final_utilities,
            final_allocation=final_allocation,
            agent_preferences=agent_preferences_data,
            strategic_behaviors=strategic_behaviors,
            conversation_logs=conversation_logs,
            agent_performance=agent_performance,
            exploitation_detected=exploitation_detected
        )
        
        # Save experiment results
        self.file_manager.save_experiment_result(
            results.to_dict(), exp_dir, 
            bool(self.current_batch_id), self.current_run_number
        )
        
        self.logger.info(f"✅ Experiment results saved to: {exp_dir}")
        
        return results
    
    async def run_batch_name_experiments(
        self,
        model: str,
        agent_names: List[str],
        num_runs: int = 10,
        experiment_config: Optional[Dict[str, Any]] = None,
        job_id: Optional[int] = None,
        override_run_number: Optional[int] = None
    ) -> BatchResults:
        """
        Run multiple name experiments and aggregate results.
        
        Args:
            model: Single model name to use for all agents
            agent_names: List of agent names
            num_runs: Number of experiments to run
            experiment_config: Optional configuration overrides
            job_id: Optional job/config ID from batch scheduler
            override_run_number: Optional specific run number
            
        Returns:
            BatchResults object with aggregated statistics
        """
        # Create batch ID and directory
        if self.use_custom_output:
            batch_id = ""
            batch_dir = self.results_dir
            batch_dir.mkdir(parents=True, exist_ok=True)
        else:
            from datetime import datetime
            timestamp_pid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            name_str = "_vs_".join(agent_names)
            if job_id is not None:
                batch_id = f"name_experiment_{timestamp_pid}_config{job_id:03d}_{name_str}"
            else:
                batch_id = f"name_experiment_{timestamp_pid}_{name_str}"
            batch_dir = self.results_dir / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)

        self.current_batch_id = batch_id
        experiments = []
        
        self.logger.info(f"Starting batch name experiment {batch_id} with {num_runs} runs")
        self.logger.info(f"Model: {model}, Agent names: {agent_names}")
        self.logger.info(f"Batch directory: {batch_dir}")
        
        for i in range(num_runs):
            # Use override_run_number if provided, otherwise use iteration number
            if override_run_number is not None:
                self.current_run_number = override_run_number
                actual_run_number = override_run_number
            else:
                self.current_run_number = i + 1
                actual_run_number = i + 1

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"BATCH RUN {actual_run_number} (iteration {i+1}/{num_runs})")
            self.logger.info(f"{'='*60}")

            # Use different seed for each run to get different preference vectors
            run_config = experiment_config.copy() if experiment_config else {}
            # Don't modify seed if override_run_number is provided (use exact seed from config)
            if override_run_number is None:
                if 'random_seed' in run_config:
                    run_config['random_seed'] = run_config['random_seed'] + i
                else:
                    run_config['random_seed'] = 42 + i  # Default seed + offset

            try:
                result = await self.run_single_name_experiment(model, agent_names, run_config)
                experiments.append(result)

                # Save intermediate result
                self.file_manager.save_experiment_result(
                    result.to_dict(),
                    batch_dir,
                    batch_mode=True,
                    run_number=actual_run_number
                )
                
            except Exception as e:
                self.logger.error(f"Error in run {i+1}: {e}")
                continue
        
        # Calculate aggregate statistics
        consensus_rate = sum(1 for exp in experiments if exp.consensus_reached) / len(experiments) if experiments else 0
        average_rounds = sum(exp.final_round for exp in experiments) / len(experiments) if experiments else 0
        exploitation_rate = sum(1 for exp in experiments if exp.exploitation_detected) / len(experiments) if experiments else 0
        
        # Aggregate strategic behaviors
        strategic_behaviors_summary = self.analyzer.aggregate_strategic_behaviors(experiments)
        
        batch_results = BatchResults(
            batch_id=batch_id,
            num_runs=len(experiments),
            experiments=experiments,
            consensus_rate=consensus_rate,
            average_rounds=average_rounds,
            exploitation_rate=exploitation_rate,
            strategic_behaviors_summary=strategic_behaviors_summary
        )
        
        # Save batch results
        self.file_manager.save_batch_summary(batch_results.to_dict(), batch_id)
        
        return batch_results


async def main():
    """Main entry point."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(
        description="Run negotiations to test how agent names affect performance"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=list(STRONG_MODELS_CONFIG.keys()),
        required=True,
        help="Single model name to use for both agents"
    )
    
    parser.add_argument(
        "--agent-names",
        nargs=2,
        type=str,
        default=None,
        help="Optional pair of agent names (e.g., --agent-names Alice Bob). If not provided, uses all combinations from config."
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiments"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of negotiation games to run per name pair (used with --batch)"
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
        "--gamma-discount",
        type=float,
        default=0.9,
        help="Discount factor for rewards per round (0-1, default: 0.9)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Job ID from batch scheduler (for tracking config number)"
    )

    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Specific run number for output files (overrides automatic numbering in batch mode)"
    )
    
    # Token control arguments
    parser.add_argument(
        "--max-tokens-discussion",
        type=int,
        default=None,
        help="Maximum tokens for discussion phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-proposal",
        type=int,
        default=None,
        help="Maximum tokens for proposal phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-voting",
        type=int,
        default=None,
        help="Maximum tokens for voting phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-reflection",
        type=int,
        default=None,
        help="Maximum tokens for reflection phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-thinking",
        type=int,
        default=None,
        help="Maximum tokens for private thinking phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-default",
        type=int,
        default=None,
        help="Default maximum tokens for all other phases (default: unlimited)"
    )
    
    # Phase control arguments
    parser.add_argument(
        "--disable-discussion",
        action="store_true",
        default=False,
        help="Disable the public discussion phase (default: enabled)"
    )

    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Disable the private thinking phase (default: enabled)"
    )

    parser.add_argument(
        "--disable-reflection",
        action="store_true",
        default=False,
        help="Disable the individual reflection phase (default: enabled)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for results (overrides default timestamped directory)",
    )

    args = parser.parse_args()
    
    # Check for at least one API key
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_openrouter or has_anthropic or has_openai):
        print("ERROR: At least one API key is required")
        print("Please set one or more of:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1
    
    print("API Keys detected:")
    if has_anthropic:
        print("  ✓ Anthropic API (Claude models)")
    if has_openai:
        print("  ✓ OpenAI API (GPT-4o, O3)")
    if has_openrouter:
        print("  ✓ OpenRouter API (Gemini, Llama, etc.)")
    
    print("=" * 60)
    print("AGENT NAME EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    
    # Determine name pairs to test
    if args.agent_names:
        name_pairs = [tuple(args.agent_names)]
        print(f"Agent names: {args.agent_names[0]} vs {args.agent_names[1]}")
    else:
        name_pairs = name_experiment_config.ALL_NAME_COMBINATIONS
        print(f"Testing all {len(name_pairs)} name combinations from config")
    
    print(f"Max Rounds: {args.max_rounds}")
    print(f"Items: {args.num_items}")
    print(f"Competition Level: {args.competition_level}")
    print(f"Discount Factor: {args.gamma_discount}")
    if args.random_seed:
        print(f"Random Seed: {args.random_seed}")
    print("=" * 60)
    
    # Create experiment configuration
    experiment_config = {
        "m_items": args.num_items,
        "t_rounds": args.max_rounds,
        "competition_level": args.competition_level,
        "gamma_discount": args.gamma_discount,
        "random_seed": args.random_seed,
        "disable_discussion": args.disable_discussion,
        "disable_thinking": args.disable_thinking,
        "disable_reflection": args.disable_reflection,
    }
    
    # Only add token limits if they're specified
    if args.max_tokens_discussion is not None:
        experiment_config["max_tokens_discussion"] = args.max_tokens_discussion
    if args.max_tokens_proposal is not None:
        experiment_config["max_tokens_proposal"] = args.max_tokens_proposal
    if args.max_tokens_voting is not None:
        experiment_config["max_tokens_voting"] = args.max_tokens_voting
    if args.max_tokens_reflection is not None:
        experiment_config["max_tokens_reflection"] = args.max_tokens_reflection
    if args.max_tokens_thinking is not None:
        experiment_config["max_tokens_thinking"] = args.max_tokens_thinking
    if args.max_tokens_default is not None:
        experiment_config["max_tokens_default"] = args.max_tokens_default
    
    # Run experiments for each name pair
    try:
        for name_pair_idx, (name1, name2) in enumerate(name_pairs):
            agent_names = [name1, name2]
            
            # Create output directory name if not provided
            if args.output_dir:
                output_dir = args.output_dir
            else:
                name_str = f"{name1}_vs_{name2}"
                comp_str = f"comp{args.competition_level}".replace(".", "_")
                
                if args.job_id is not None:
                    config_str = f"config{args.job_id:03d}"
                else:
                    config_str = f"pair{name_pair_idx+1:03d}"
                
                if args.run_number is not None:
                    run_str = f"run{args.run_number}"
                else:
                    run_str = f"runs{args.num_runs}" if args.batch else "single"
                
                output_dir = f"experiments/results/name_results/{args.model}_{name_str}_{config_str}_{run_str}_{comp_str}"
            
            # Initialize experiment runner with custom output directory
            experiment = NameExperiment(output_dir=output_dir)
            
            print(f"\n{'='*60}")
            print(f"Name Pair {name_pair_idx+1}/{len(name_pairs)}: {name1} vs {name2}")
            print(f"{'='*60}")
            
            if args.batch:
                print(f"\n--- Batch Experiment ({args.num_runs} runs) ---")
                if args.job_id is not None:
                    print(f"Job ID (Config #): {args.job_id}")
                print(f"Output Directory: {output_dir}")
                batch_results = await experiment.run_batch_name_experiments(
                    model=args.model,
                    agent_names=agent_names,
                    num_runs=args.num_runs,
                    experiment_config=experiment_config,
                    job_id=args.job_id,
                    override_run_number=args.run_number
                )
                
                print(f"\n📈 Batch Results Summary for {name1} vs {name2}:")
                print(f"  Batch ID: {batch_results.batch_id}")
                print(f"  Successful Runs: {batch_results.num_runs}")
                print(f"  Consensus Rate: {batch_results.consensus_rate:.1%}")
                print(f"  Average Rounds: {batch_results.average_rounds:.1f}")
                print(f"  Exploitation Rate: {batch_results.exploitation_rate:.1%}")
                
            else:
                print("\n--- Single Experiment Test ---")
                single_result = await experiment.run_single_name_experiment(
                    model=args.model,
                    agent_names=agent_names,
                    experiment_config=experiment_config
                )
                
                print(f"\n📊 Single Experiment Results for {name1} vs {name2}:")
                print(f"  Experiment ID: {single_result.experiment_id}")
                print(f"  Consensus Reached: {'✓' if single_result.consensus_reached else '✗'}")
                print(f"  Final Round: {single_result.final_round}")
                print(f"  Exploitation Detected: {'✓' if single_result.exploitation_detected else '✗'}")
                
                if single_result.final_utilities:
                    print(f"  Final Utilities:")
                    for agent_id, utility in single_result.final_utilities.items():
                        print(f"    {agent_id}: {utility:.2f}")
        
        print(f"\n✅ Name experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        logging.exception("Experiment error")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

