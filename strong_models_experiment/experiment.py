"""Main experiment runner for strong models negotiation."""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .data_models import ExperimentResults, BatchResults
from .configs import STRONG_MODELS_CONFIG
from .agents import StrongModelAgentFactory
from .phases import PhaseHandler
from .analysis import ExperimentAnalyzer
from .utils import ExperimentUtils, FileManager

# Import game environment factory
from game_environments import create_game_environment, GameEnvironment


class StrongModelsExperiment:
    """
    Runs the 14-phase strong models negotiation experiment.
    """
    
    def __init__(self, output_dir=None):
        """Initialize the experiment runner."""
        self.logger = self._setup_logging()

        # Use custom output directory if provided, otherwise use default
        if output_dir:
            self.results_dir = Path(output_dir)
            self.use_custom_output = True
        else:
            self.results_dir = Path("experiments/results")
            self.use_custom_output = False

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.agent_factory = StrongModelAgentFactory()
        self.analyzer = ExperimentAnalyzer()
        self.utils = ExperimentUtils()
        self.file_manager = FileManager(self.results_dir)
        
        # Storage for all interactions
        self.all_interactions = []
        self.agent_interactions = {}
        self.current_experiment_id = None
        self.current_batch_id = None
        self.current_run_number = None

        # Token usage tracking for cost aggregation
        self.batch_token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_agent": {},
            "by_phase": {},
            "estimated_count": 0,
            "actual_count": 0,
        }

        # Initialize phase handler with save callback (token config will be set per experiment)
        self.phase_handler = None
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("StrongModelsExperiment")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    async def run_single_experiment(
        self,
        models: List[str],
        experiment_config: Optional[Dict[str, Any]] = None
    ) -> ExperimentResults:
        """
        Run a single negotiation experiment with strong models.
        
        Args:
            models: List of model names to use
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
            "game_type": "item_allocation",  # or "diplomacy"
            "m_items": 5,
            "n_agents": len(models),
            "t_rounds": 10,
            "gamma_discount": 0.9,
            "competition_level": 1,
            "random_seed": None,
            # Diplomacy-specific parameters (only used when game_type="diplomacy")
            "n_issues": 5,
            "rho": -1,      # Preference correlation [-1, 1]
            "theta": 1,    # Interest overlap [0, 1]
            "lam": -1,      # Issue compatibility [-1, 1]
        }

        # Merge with provided config
        config = {**default_config, **(experiment_config or {})}
        
        # Set random seed if provided
        if config["random_seed"]:
            import random
            random.seed(config["random_seed"])

        # Handle random model ordering
        model_order = config.get("model_order", "weak_first")
        if model_order == "random":
            import random as rand_module
            if rand_module.random() < 0.5:
                config["actual_order"] = "weak_first"
                self.logger.info("Random order selected: weak_first (models unchanged)")
            else:
                config["actual_order"] = "strong_first"
                models = models[::-1]  # Reverse the models list
                self.logger.info("Random order selected: strong_first (models reversed)")
        else:
            config["actual_order"] = model_order
            if model_order == "strong_first":
                # If explicitly strong_first, ensure models are in correct order
                # (config generator should have already handled this, but verify)
                self.logger.info(f"Using configured order: {model_order}")

        self.logger.info(f"Starting experiment {experiment_id}")
        
        # Extract token configuration from config
        # Use None (unlimited) as default if not specified
        token_config = {
            "discussion": config.get("max_tokens_discussion", None),
            "proposal": config.get("max_tokens_proposal", None),
            "voting": config.get("max_tokens_voting", None),
            "reflection": config.get("max_tokens_reflection", None),
            "thinking": config.get("max_tokens_thinking", None),
            "default": config.get("max_tokens_default", None)
        }

        # Create GameEnvironment based on game_type
        game_type = config.get("game_type", "item_allocation")
        self.logger.info(f"Creating game environment: {game_type}")

        if game_type == "item_allocation":
            game_environment = create_game_environment(
                game_type="item_allocation",
                n_agents=config["n_agents"],
                t_rounds=config["t_rounds"],
                gamma_discount=config["gamma_discount"],
                random_seed=config.get("random_seed"),
                m_items=config.get("m_items", 5),
                competition_level=config.get("competition_level", 0.95)
            )
        elif game_type == "diplomacy":
            game_environment = create_game_environment(
                game_type="diplomacy",
                n_agents=config["n_agents"],
                t_rounds=config["t_rounds"],
                gamma_discount=config["gamma_discount"],
                random_seed=config.get("random_seed"),
                n_issues=config.get("n_issues", 5),
                rho=config.get("rho", 0.0),
                theta=config.get("theta", 0.5),
                lam=config.get("lam", 0.0)
            )
        else:
            raise ValueError(f"Unknown game_type: {game_type}. Must be 'item_allocation' or 'diplomacy'")

        # Initialize phase handler with token config and game environment
        self.phase_handler = PhaseHandler(
            save_interaction_callback=self._save_interaction,
            token_config=token_config,
            game_environment=game_environment
        )
        
        # Create agents
        agents = await self.agent_factory.create_agents(models, config)
        if not agents:
            raise ValueError("Failed to create agents")

        # Create game state using GameEnvironment
        # This generates items/issues and preferences based on game type
        game_state = game_environment.create_game_state(agents)

        # Extract items/issues and preferences based on game type
        if game_type == "item_allocation":
            items = game_state["items"]
            preferences = {"agent_preferences": game_state["agent_preferences"]}
            self.logger.info(f"Created {len(items)} items with competition_level={config.get('competition_level')}")
        elif game_type == "diplomacy":
            # Convert issues to item-like format for compatibility
            items = [{"name": issue} for issue in game_state["issues"]]
            # Use positions as preferences, store full game_state for utility calculation
            preferences = {
                "agent_preferences": game_state["agent_positions"],
                "agent_weights": game_state["agent_weights"],
                "issue_types": game_state["issue_types"],
                "game_state": game_state  # Full state for utility calculation
            }
            self.logger.info(f"Created {len(items)} issues with rho={config.get('rho')}, theta={config.get('theta')}, lam={config.get('lam')}")
        else:
            raise ValueError(f"Unknown game type: {game_type}")

        # Initialize tracking variables
        consensus_reached = False
        final_round = 0
        final_utilities = {}
        final_allocation = {}
        agent_preferences_data = {}
        strategic_behaviors = {}
        conversation_logs = []
        
        # Run the 14-phase negotiation
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
                        agents, items, preferences, round_num, config["t_rounds"],
                        discussion_turns=config.get("discussion_turns", 3)
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
            import traceback
            self.logger.error(f"Error during negotiation: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # Analyze results
        exploitation_detected = self.analyzer.detect_exploitation(conversation_logs)
        strategic_behaviors = self.analyzer.analyze_strategic_behaviors(conversation_logs)
        agent_performance = self.analyzer.analyze_agent_performance(agents, final_utilities)
        
        # Create enhanced config
        enhanced_config = self.utils.create_enhanced_config(
            config, agents, preferences, items, experiment_start_time, experiment_id
        )
        
        # Save all interactions
        self._stream_save_json()
        
        # Determine save location
        if self.use_custom_output:
            # Use custom output directory directly
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
    
    async def run_batch_experiments(
        self,
        models: List[str],
        num_runs: int = 10,
        experiment_config: Optional[Dict[str, Any]] = None,
        job_id: Optional[int] = None,
        override_run_number: Optional[int] = None
    ) -> BatchResults:
        """
        Run multiple experiments and aggregate results.
        
        Args:
            models: List of model names to use
            num_runs: Number of experiments to run
            experiment_config: Optional configuration overrides
            job_id: Optional job/config ID from batch scheduler
            
        Returns:
            BatchResults object with aggregated statistics
        """
        # Create batch ID and directory
        if self.use_custom_output:
            # When using custom output, don't create timestamped subdirectory
            batch_id = ""  # Empty batch_id means use results_dir directly
            batch_dir = self.results_dir
            batch_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Original behavior: create timestamped directory
            timestamp_pid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            if job_id is not None:
                batch_id = f"strong_models_{timestamp_pid}_config{job_id:03d}"
            else:
                batch_id = f"strong_models_{timestamp_pid}"
            batch_dir = self.results_dir / batch_id
            batch_dir.mkdir(parents=True, exist_ok=True)

        self.current_batch_id = batch_id
        experiments = []

        # Reset token tracking for this batch
        self._reset_batch_token_tracking()

        self.logger.info(f"Starting batch experiment {batch_id} with {num_runs} runs")
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
                if 'random_seed' in run_config and run_config['random_seed'] is not None:
                    run_config['random_seed'] = run_config['random_seed'] + i
                else:
                    run_config['random_seed'] = 42 + i  # Default seed + offset

            try:
                result = await self.run_single_experiment(models, run_config)
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

        # Print token usage summary
        self._print_token_usage_summary(models)

        return batch_results
    
    def _save_interaction(self, agent_id: str, phase: str, prompt: str, response: str, round_num: int = None,
                         token_usage: Optional[Dict[str, Any]] = None):
        """Save an interaction to both all_interactions and agent-specific storage.

        Args:
            agent_id: Agent identifier
            phase: Phase name
            prompt: Prompt sent to agent
            response: Agent's response text
            round_num: Round number
            token_usage: Optional dict with token usage info (e.g., {'input_tokens': int, 'output_tokens': int, 'total_tokens': int})
        """
        interaction = {
            "timestamp": time.time(),
            "experiment_id": self.current_experiment_id,
            "agent_id": agent_id,
            "phase": phase,
            "round": round_num,
            "prompt": prompt,
            "response": response
        }

        # Add token usage information if provided
        if token_usage:
            interaction["token_usage"] = token_usage

        # Track token usage for batch aggregation
        self._track_token_usage(agent_id, phase, prompt, response, token_usage)

        # Add to all interactions
        self.all_interactions.append(interaction)

        # Add to agent-specific interactions
        if agent_id not in self.agent_interactions:
            self.agent_interactions[agent_id] = []
        self.agent_interactions[agent_id].append(interaction)

        # Stream save to JSON files
        self._stream_save_json()

    def _track_token_usage(self, agent_id: str, phase: str, prompt: str, response: str,
                          token_usage: Optional[Dict[str, Any]] = None):
        """Track token usage for aggregation and cost calculation.

        Args:
            agent_id: Agent identifier
            phase: Phase name
            prompt: Prompt sent to agent
            response: Agent's response text
            token_usage: Optional dict with actual token counts
        """
        # Estimate tokens if not provided (approximately 4 chars per token)
        CHARS_PER_TOKEN = 4.0

        if token_usage:
            input_tokens = token_usage.get('input_tokens', 0) or 0
            output_tokens = token_usage.get('output_tokens', 0) or 0
            self.batch_token_usage["actual_count"] += 1
        else:
            # Estimate from text
            input_tokens = int(len(prompt) / CHARS_PER_TOKEN) if prompt else 0
            output_tokens = int(len(response) / CHARS_PER_TOKEN) if response else 0
            self.batch_token_usage["estimated_count"] += 1

        # Update totals
        self.batch_token_usage["total_input_tokens"] += input_tokens
        self.batch_token_usage["total_output_tokens"] += output_tokens

        # Update by agent
        if agent_id not in self.batch_token_usage["by_agent"]:
            self.batch_token_usage["by_agent"][agent_id] = {"input": 0, "output": 0}
        self.batch_token_usage["by_agent"][agent_id]["input"] += input_tokens
        self.batch_token_usage["by_agent"][agent_id]["output"] += output_tokens

        # Update by phase (extract base phase name)
        base_phase = phase.split('_round_')[0].split('_turn_')[0]
        if base_phase not in self.batch_token_usage["by_phase"]:
            self.batch_token_usage["by_phase"][base_phase] = {"input": 0, "output": 0}
        self.batch_token_usage["by_phase"][base_phase]["input"] += input_tokens
        self.batch_token_usage["by_phase"][base_phase]["output"] += output_tokens

    def _reset_batch_token_tracking(self):
        """Reset token tracking for a new batch."""
        self.batch_token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "by_agent": {},
            "by_phase": {},
            "estimated_count": 0,
            "actual_count": 0,
        }

    def _print_token_usage_summary(self, models: List[str]):
        """Print a summary of token usage and estimated costs at the end of a batch.

        Args:
            models: List of model names used in the experiment
        """
        usage = self.batch_token_usage
        total_in = usage["total_input_tokens"]
        total_out = usage["total_output_tokens"]

        def format_tokens(count):
            if count >= 1_000_000:
                return f"{count/1_000_000:.2f}M"
            elif count >= 1_000:
                return f"{count/1_000:.1f}K"
            return str(count)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("             TOKEN USAGE & COST SUMMARY")
        self.logger.info("=" * 70)

        # Overall summary
        self.logger.info(f"\n  Total Input Tokens:   {format_tokens(total_in)}")
        self.logger.info(f"  Total Output Tokens:  {format_tokens(total_out)}")
        self.logger.info(f"  Data Source:          {usage['actual_count']} actual, {usage['estimated_count']} estimated")

        # By agent breakdown
        if usage["by_agent"]:
            self.logger.info(f"\n  {'BY AGENT:':<20}")
            self.logger.info(f"    {'Agent':<20} {'Input':>12} {'Output':>12}")
            self.logger.info("    " + "-" * 44)
            for agent_id, tokens in usage["by_agent"].items():
                self.logger.info(f"    {agent_id:<20} {format_tokens(tokens['input']):>12} {format_tokens(tokens['output']):>12}")

        # By phase breakdown
        if usage["by_phase"]:
            self.logger.info(f"\n  {'BY PHASE:':<20}")
            self.logger.info(f"    {'Phase':<25} {'Input':>12} {'Output':>12}")
            self.logger.info("    " + "-" * 49)
            for phase, tokens in sorted(usage["by_phase"].items()):
                self.logger.info(f"    {phase:<25} {format_tokens(tokens['input']):>12} {format_tokens(tokens['output']):>12}")

        # Cost estimate (basic - use cost_dashboard.py for detailed analysis)
        self.logger.info(f"\n  {'COST NOTES:':<20}")
        self.logger.info("    For detailed cost analysis, run:")
        self.logger.info("    python visualization/cost_dashboard.py --dir <results_dir>")
        self.logger.info("\n" + "=" * 70 + "\n")
    
    def _stream_save_json(self):
        """Stream save all interactions to JSON files."""
        if not self.current_experiment_id:
            return
        
        # Determine the directory structure
        if self.use_custom_output:
            # Use custom output directory directly
            exp_dir = self.results_dir
        elif self.current_batch_id and self.current_run_number:
            exp_dir = self.results_dir / self.current_batch_id
        else:
            exp_dir = self.results_dir / self.current_experiment_id
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine batch mode: use run_number if set, regardless of batch_id
        # (batch_id can be empty string when using custom output directories)
        batch_mode = self.current_run_number is not None
        
        # Save all interactions
        self.file_manager.save_all_interactions(
            self.all_interactions, exp_dir,
            batch_mode, self.current_run_number
        )
        
        # Save agent-specific interactions
        for agent_id, interactions in self.agent_interactions.items():
            for interaction in interactions:
                self.file_manager.save_interaction(
                    interaction, exp_dir,
                    batch_mode, self.current_run_number
                )