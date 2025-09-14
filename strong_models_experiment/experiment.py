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


class StrongModelsExperiment:
    """
    Runs the 14-phase strong models negotiation experiment.
    """
    
    def __init__(self):
        """Initialize the experiment runner."""
        self.logger = self._setup_logging()
        self.results_dir = Path("experiments/results")
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
            import random
            random.seed(config["random_seed"])
        
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
        
        # Initialize phase handler with token config for this experiment
        self.phase_handler = PhaseHandler(
            save_interaction_callback=self._save_interaction,
            token_config=token_config
        )
        
        # Create agents
        agents = await self.agent_factory.create_agents(models, config)
        if not agents:
            raise ValueError("Failed to create agents")
        
        # Create items and preferences
        items = self.utils.create_items(config["m_items"])
        preferences = self.utils.create_preferences(agents, items, config)
        
        # Initialize tracking variables
        consensus_reached = False
        final_round = 0
        final_utilities = {}
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
        
        # Create enhanced config
        enhanced_config = self.utils.create_enhanced_config(
            config, agents, preferences, items, experiment_start_time, experiment_id
        )
        
        # Save all interactions
        self._stream_save_json()
        
        # Determine save location
        if self.current_batch_id and self.current_run_number:
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
        job_id: Optional[int] = None
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
        # Create batch ID - include job_id if provided
        timestamp_pid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if job_id is not None:
            batch_id = f"strong_models_{timestamp_pid}_config{job_id:03d}"
        else:
            batch_id = f"strong_models_{timestamp_pid}"
        self.current_batch_id = batch_id
        experiments = []
        
        # Create batch directory
        batch_dir = self.results_dir / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch experiment {batch_id} with {num_runs} runs")
        self.logger.info(f"Batch directory: {batch_dir}")
        
        for i in range(num_runs):
            self.current_run_number = i + 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"BATCH RUN {i+1}/{num_runs}")
            self.logger.info(f"{'='*60}")
            
            # Use different seed for each run to get different preference vectors
            run_config = experiment_config.copy() if experiment_config else {}
            if 'random_seed' in run_config:
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
                    run_number=i + 1
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
    
    def _save_interaction(self, agent_id: str, phase: str, prompt: str, response: str, round_num: int = None):
        """Save an interaction to both all_interactions and agent-specific storage."""
        interaction = {
            "timestamp": time.time(),
            "experiment_id": self.current_experiment_id,
            "agent_id": agent_id,
            "phase": phase,
            "round": round_num,
            "prompt": prompt,
            "response": response
        }
        
        # Add to all interactions
        self.all_interactions.append(interaction)
        
        # Add to agent-specific interactions
        if agent_id not in self.agent_interactions:
            self.agent_interactions[agent_id] = []
        self.agent_interactions[agent_id].append(interaction)
        
        # Stream save to JSON files
        self._stream_save_json()
    
    def _stream_save_json(self):
        """Stream save all interactions to JSON files."""
        if not self.current_experiment_id:
            return
        
        # Determine the directory structure
        if self.current_batch_id and self.current_run_number:
            exp_dir = self.results_dir / self.current_batch_id
        else:
            exp_dir = self.results_dir / self.current_experiment_id
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all interactions
        self.file_manager.save_all_interactions(
            self.all_interactions, exp_dir,
            bool(self.current_batch_id), self.current_run_number
        )
        
        # Save agent-specific interactions
        for agent_id, interactions in self.agent_interactions.items():
            for interaction in interactions:
                self.file_manager.save_interaction(
                    interaction, exp_dir,
                    bool(self.current_batch_id), self.current_run_number
                )