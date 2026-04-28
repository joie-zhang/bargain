"""Main experiment runner for strong models negotiation."""

import asyncio
import gzip
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is available on the Linux cluster.
    fcntl = None

from .data_models import ExperimentResults, BatchResults
from .configs import STRONG_MODELS_CONFIG
from .agents import StrongModelAgentFactory
from .phases import PhaseHandler
from .analysis import ExperimentAnalyzer
from .analysis.qualitative_metrics import compute_qualitative_metrics_v1
from .analysis.qualitative_schema import (
    validate_qualitative_event,
    validate_qualitative_metrics_v1,
)
from .utils import ExperimentUtils, FileManager

# Import game environment factory
from game_environments import create_game_environment, GameEnvironment


class StrongModelsExperiment:
    """
    Runs the strong models negotiation experiment.
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
        self.current_config: Dict[str, Any] = {}
        self.externalize_prompts = os.getenv("EXPERIMENT_EXTERNALIZE_PROMPTS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        # Token usage tracking for cost aggregation
        self.batch_token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,  # Track reasoning tokens separately
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

        # Preserve the realized model order. For n-agent experiments this is the
        # authoritative speaking/action order, regardless of the order label.
        config["models"] = list(models)
        self.current_config = config

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
                theta=config.get("theta", 0.5)
            )
        elif game_type == "co_funding":
            game_environment = create_game_environment(
                game_type="co_funding",
                n_agents=config["n_agents"],
                t_rounds=config["t_rounds"],
                gamma_discount=config.get("cofunding_time_discount", config.get("gamma_discount", 0.9)),
                random_seed=config.get("random_seed"),
                m_projects=config.get("m_projects", 5),
                alpha=config.get("alpha", 0.5),
                sigma=config.get("sigma", 0.5),
                c_min=config.get("c_min", 10.0),
                c_max=config.get("c_max", 30.0),
                pledge_mode=config.get("pledge_mode", "individual"),
                discussion_transparency=config.get(
                    "cofunding_discussion_transparency",
                    config.get("discussion_transparency", "own")
                ),
                enable_commit_vote=config.get("cofunding_enable_commit_vote", True),
                enable_time_discount=config.get("cofunding_enable_time_discount", True),
            )
        else:
            raise ValueError(f"Unknown game_type: {game_type}. Must be 'item_allocation', 'diplomacy', or 'co_funding'")

        # Extract reasoning configuration if present
        reasoning_config = config.get("reasoning_config", {"budget": None, "phases": []})

        parallel_phases = config.get("parallel_phases", False)
        if not isinstance(parallel_phases, bool):
            raise ValueError(
                "parallel_phases must be a boolean; use true to run one task per agent, "
                "or false for serial execution"
            )

        # Initialize phase handler with token config, game environment, and reasoning config
        self.phase_handler = PhaseHandler(
            save_interaction_callback=self._save_interaction,
            token_config=token_config,
            game_environment=game_environment,
            reasoning_config=reasoning_config,
            access_config=config.get("access_config", {"k": 1, "phases": [], "agent_ids": []}),
            parallel_phases=parallel_phases
        )
        self.phase_handler.reset_vote_integrity()

        # Create agents
        agents = await self.agent_factory.create_agents(models, config)
        if not agents:
            raise ValueError("Failed to create agents")

        agent_model_map = {
            agent.agent_id: model_name
            for agent, model_name in zip(agents, models)
        }
        config["agent_model_map"] = agent_model_map
        try:
            from .analysis.active_model_roster import elo_for_model

            config["agent_elo_map"] = {
                agent_id: elo_for_model(model_name)
                for agent_id, model_name in agent_model_map.items()
            }
        except Exception:
            config["agent_elo_map"] = {}

        access_config = dict(config.get("access_config") or {})
        try:
            access_k = int(access_config.get("k", 1) or 1)
        except (TypeError, ValueError):
            access_k = 1
        if access_k < 1:
            access_k = 1
        access_config["k"] = access_k
        if not access_config.get("phases"):
            access_config["phases"] = []

        if access_k > 1:
            access_agent_ids = access_config.get("agent_ids")
            if not access_agent_ids:
                access_agent_index = int(access_config.get("agent_index", 0) or 0)
                access_agent_index = max(0, min(access_agent_index, len(agents) - 1))
                access_agent_ids = [agents[access_agent_index].agent_id]
                access_config["agent_index"] = access_agent_index
            access_config["agent_ids"] = list(access_agent_ids)
            self.phase_handler.access_config = access_config
            self.logger.info(
                "Access scaling will be applied to %s with k=%s on phases=%s",
                access_config["agent_ids"],
                access_k,
                access_config.get("phases", []),
            )
        else:
            access_agent_index = int(access_config.get("agent_index", 0) or 0)
            access_agent_index = max(0, min(access_agent_index, len(agents) - 1))
            access_config.setdefault("agent_ids", [agents[access_agent_index].agent_id])
            access_config["agent_index"] = access_agent_index
            self.phase_handler.access_config = access_config

        config["access_config"] = access_config

        # Determine which agent(s) should receive reasoning budget prompt
        # Only the reasoning model should get the prompt instruction, not the baseline
        model_order = config.get("model_order", "weak_first")
        if reasoning_config.get("budget"):
            if "reasoning_agent_index" in config:
                reasoning_agent_index = int(config["reasoning_agent_index"])
            elif model_order == "weak_first":
                reasoning_agent_index = 1
            else:
                reasoning_agent_index = 0
            reasoning_agent_index = min(reasoning_agent_index, len(agents) - 1)
            reasoning_agent_ids = [agents[reasoning_agent_index].agent_id]
            self.phase_handler.reasoning_config["reasoning_agent_ids"] = reasoning_agent_ids
            self.logger.info(f"Reasoning prompt will be applied to: {reasoning_agent_ids}")

        # Create game state using GameEnvironment
        # This generates items/issues and preferences based on game type
        game_state = game_environment.create_game_state(agents)

        def pairwise_cosine_summary(
            vectors_by_agent: Dict[str, Any],
            target_cosine: float,
        ) -> tuple[Dict[str, float], Dict[str, float]]:
            agent_ids_for_cosine = list(vectors_by_agent.keys())
            cosines: Dict[str, float] = {}
            for i, agent_i in enumerate(agent_ids_for_cosine):
                vec_i = np.asarray(vectors_by_agent[agent_i], dtype=float)
                norm_i = np.linalg.norm(vec_i)
                for j in range(i + 1, len(agent_ids_for_cosine)):
                    agent_j = agent_ids_for_cosine[j]
                    vec_j = np.asarray(vectors_by_agent[agent_j], dtype=float)
                    norm_j = np.linalg.norm(vec_j)
                    cosine = 0.0
                    if norm_i > 1e-12 and norm_j > 1e-12:
                        cosine = float(np.dot(vec_i, vec_j) / (norm_i * norm_j))
                    cosines[f"{agent_i}_vs_{agent_j}"] = cosine

            cosine_values = list(cosines.values())
            if not cosine_values:
                return cosines, {}
            abs_errors = [abs(float(value) - target_cosine) for value in cosine_values]
            return cosines, {
                "target": float(target_cosine),
                "pair_count": len(cosine_values),
                "min": min(float(value) for value in cosine_values),
                "mean": sum(float(value) for value in cosine_values) / len(cosine_values),
                "max": max(float(value) for value in cosine_values),
                "mean_abs_error": sum(abs_errors) / len(abs_errors),
                "max_abs_error": max(abs_errors),
            }

        # Extract items/issues and preferences based on game type
        if game_type == "item_allocation":
            items = game_state["items"]
            preferences = {
                "agent_preferences": game_state["agent_preferences"],
                "cosine_similarities": game_state.get("cosine_similarities", {}),
            }
            config["actual_pairwise_cosines"] = game_state.get("cosine_similarities", {})
            cosine_values = list(config["actual_pairwise_cosines"].values())
            if cosine_values:
                target_cosine = float(config.get("competition_level", 0.0) or 0.0)
                abs_errors = [abs(float(value) - target_cosine) for value in cosine_values]
                config["pairwise_cosine_summary"] = {
                    "target": target_cosine,
                    "pair_count": len(cosine_values),
                    "min": min(float(value) for value in cosine_values),
                    "mean": sum(float(value) for value in cosine_values) / len(cosine_values),
                    "max": max(float(value) for value in cosine_values),
                    "mean_abs_error": sum(abs_errors) / len(abs_errors),
                    "max_abs_error": max(abs_errors),
                }
            self.logger.info(f"Created {len(items)} items with competition_level={config.get('competition_level')}")
        elif game_type == "diplomacy":
            # Convert issues to item-like format for compatibility
            items = [{"name": issue} for issue in game_state["issues"]]
            # Use positions as preferences, store full game_state for utility calculation
            preferences = {
                "agent_preferences": game_state["agent_positions"],
                "agent_weights": game_state["agent_weights"],
                "game_state": game_state  # Full state for utility calculation
            }
            cosines, summary = pairwise_cosine_summary(
                game_state["agent_weights"],
                float(config.get("theta", 0.0) or 0.0),
            )
            config["actual_pairwise_cosines"] = cosines
            if summary:
                config["pairwise_cosine_summary"] = summary
            self.logger.info(f"Created {len(items)} issues with rho={config.get('rho')}, theta={config.get('theta')}")
        elif game_type == "co_funding":
            items = game_state["projects"]
            preferences = {
                "agent_preferences": game_state["agent_valuations"],
                "game_state": game_state,
            }
            cosines, summary = pairwise_cosine_summary(
                game_state["agent_valuations"],
                float(config.get("alpha", 0.0) or 0.0),
            )
            config["actual_pairwise_cosines"] = game_state.get(
                "valuation_cosine_similarities",
                cosines,
            )
            if summary:
                config["pairwise_cosine_summary"] = summary
            self.logger.info(
                f"Created {len(items)} projects with alpha={config.get('alpha')}, sigma={config.get('sigma')}"
            )
        else:
            raise ValueError(f"Unknown game type: {game_type}")

        # Initialize tracking variables
        consensus_reached = False
        final_round = 0
        final_utilities = {}
        final_allocation = {}
        agent_preferences_data = {}
        strategic_behaviors = {}
        qualitative_metrics_v1 = {}
        qualitative_events = []
        conversation_logs = []
        public_context_history: List[Dict[str, Any]] = []
        private_context_by_agent: Dict[str, List[str]] = {
            agent.agent_id: [] for agent in agents
        }

        def extend_public_context(messages: Optional[List[Dict[str, Any]]]) -> None:
            if not messages:
                return
            for message in messages:
                public_context_history.append(dict(message))

        def append_private_note(
            agent_id: Optional[str],
            phase_label: str,
            round_num: int,
            content: Any,
        ) -> None:
            if not agent_id:
                return

            private_context_by_agent.setdefault(agent_id, [])
            if isinstance(content, str):
                rendered_content = content
            else:
                rendered_content = json.dumps(content, indent=2, default=str)

            private_context_by_agent[agent_id].append(
                f"[Round {round_num} | {phase_label}]\n{rendered_content}"
            )
        
        # Run the setup phases and negotiation rounds
        try:
            # Phase 1A: Game Setup
            await self.phase_handler.run_game_setup_phase(agents, items, preferences, config)
            
            # Phase 1B: Private Preference Assignment
            if game_environment.uses_combined_setup_phase():
                self.logger.info(
                    "Skipping separate preference assignment phase; it is merged into game setup for this game."
                )
            else:
                await self.phase_handler.run_private_preference_assignment(
                    agents, items, preferences, config
                )
            
            # Determine negotiation protocol
            protocol = game_environment.get_protocol_type()
            self.logger.info(f"Using protocol: {protocol}")

            # Main negotiation rounds
            for round_num in range(1, config["t_rounds"] + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"ROUND {round_num}/{config['t_rounds']}")
                self.logger.info(f"{'='*60}")

                # Phase 2: Public Discussion (optional, shared by both protocols)
                discussion_result = {"messages": []}
                if not config.get("disable_discussion", False):
                    discussion_result = await self.phase_handler.run_discussion_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        discussion_turns=config.get("discussion_turns", 2),
                        public_context=public_context_history,
                        private_context_by_agent=private_context_by_agent,
                    )
                    conversation_logs.extend(discussion_result.get("messages", []))
                    extend_public_context(discussion_result.get("messages", []))
                else:
                    self.logger.info(f"Skipping discussion phase (disabled)")

                # Phase 3: Private Thinking (optional, shared by both protocols)
                thinking_result = {}
                if not config.get("disable_thinking", False):
                    thinking_result = await self.phase_handler.run_private_thinking_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        discussion_result.get("messages", []),
                        public_context=public_context_history,
                        private_context_by_agent=private_context_by_agent,
                    )
                    for thinking_entry in thinking_result.get("thinking_results", []):
                        append_private_note(
                            thinking_entry.get("agent_id"),
                            "Private Thinking",
                            round_num,
                            thinking_entry,
                        )
                else:
                    self.logger.info(f"Skipping private thinking phase (disabled)")

                if protocol == "propose_and_vote":
                    # ---- Propose-and-Vote protocol (Games 1 & 2) ----

                    # Phase 4A: Proposal Submission
                    proposal_result = await self.phase_handler.run_proposal_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        public_context=public_context_history,
                        private_context_by_agent=private_context_by_agent,
                    )
                    conversation_logs.extend(proposal_result.get("messages", []))
                    extend_public_context(proposal_result.get("messages", []))
                    for proposal in proposal_result.get("proposals", []):
                        append_private_note(
                            proposal.get("proposed_by"),
                            "Proposal",
                            round_num,
                            proposal,
                        )

                    # Phase 4B: Proposal Enumeration
                    enumeration_result = await self.phase_handler.run_proposal_enumeration_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        proposal_result.get("proposals", [])
                    )
                    conversation_logs.extend(enumeration_result.get("messages", []))
                    extend_public_context(enumeration_result.get("messages", []))

                    # Phase 5A: Private Voting
                    voting_result = await self.phase_handler.run_private_voting_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        proposal_result.get("proposals", []),
                        enumeration_result.get("enumerated_proposals", []),
                        public_context=public_context_history,
                        private_context_by_agent=private_context_by_agent,
                    )
                    votes_by_agent: Dict[str, List[Dict[str, Any]]] = {}
                    for vote in voting_result.get("private_votes", []):
                        voter_id = vote.get("voter_id")
                        if not voter_id:
                            continue
                        votes_by_agent.setdefault(voter_id, []).append({
                            "proposal_number": vote.get("proposal_number"),
                            "vote": vote.get("vote"),
                            "reasoning": vote.get("reasoning"),
                        })
                    for voter_id, votes in votes_by_agent.items():
                        append_private_note(
                            voter_id,
                            "Private Voting",
                            round_num,
                            {"votes": votes},
                        )

                    # Phase 5B: Vote Tabulation
                    tabulation_result = await self.phase_handler.run_vote_tabulation_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        voting_result.get("private_votes", []),
                        enumeration_result.get("enumerated_proposals", [])
                    )
                    conversation_logs.extend(tabulation_result.get("messages", []))
                    extend_public_context(tabulation_result.get("messages", []))

                    # Check for consensus
                    if tabulation_result.get("consensus_reached", False):
                        consensus_reached = True
                        final_round = round_num
                        final_utilities = tabulation_result.get("final_utilities", {})
                        final_allocation = tabulation_result.get("final_allocation", {})
                        agent_preferences_data = tabulation_result.get("agent_preferences", {})
                        self.logger.info(f"SUPERMAJORITY AGREEMENT REACHED in round {round_num}!")
                        break

                    # Phase 6: Individual Reflection (optional)
                    reflection_result = {}
                    if not config.get("disable_reflection", False):
                        reflection_result = await self.phase_handler.run_individual_reflection_phase(
                            agents, items, preferences, round_num, config["t_rounds"],
                            tabulation_result,
                            public_context=public_context_history,
                            private_context_by_agent=private_context_by_agent,
                        )
                        for reflection_entry in reflection_result.get("reflections", []):
                            append_private_note(
                                reflection_entry.get("agent_id"),
                                "Reflection",
                                round_num,
                                reflection_entry.get("reflection", ""),
                            )
                    else:
                        self.logger.info(f"Skipping individual reflection phase (disabled)")

                elif protocol == "talk_pledge_revise":
                    # ---- Legacy Talk-Pledge-Revise protocol (unused by current Game 3 runs) ----

                    # Pledge Submission
                    pledge_result = await self.phase_handler.run_pledge_submission_phase(
                        agents, items, preferences, round_num, config["t_rounds"]
                    )
                    conversation_logs.extend(pledge_result.get("messages", []))
                    extend_public_context(pledge_result.get("messages", []))

                    # Feedback (updates game_state, shows aggregates)
                    feedback_result = await self.phase_handler.run_feedback_phase(
                        agents, items, preferences, round_num, config["t_rounds"],
                        pledge_result["pledges"]
                    )
                    conversation_logs.extend(feedback_result.get("messages", []))
                    extend_public_context(feedback_result.get("messages", []))

                    # Optional post-pledge commit vote (yay/nay on current pledge profile)
                    if config.get("cofunding_enable_commit_vote", True):
                        commit_result = await self.phase_handler.run_cofunding_commit_vote_phase(
                            agents, items, preferences, round_num, config["t_rounds"]
                        )
                        conversation_logs.extend(commit_result.get("messages", []))
                        extend_public_context(commit_result.get("messages", []))

                        if commit_result.get(
                            "supermajority_yay",
                            commit_result.get("majority_yay", commit_result.get("unanimous_yay", False)),
                        ):
                            self.logger.info(
                                f"Supermajority co-funding commit reached in round {round_num}; ending early."
                            )
                            final_round = round_num
                            break

                    # Individual Reflection (optional)
                    if not config.get("disable_reflection", False):
                        reflection_result = await self.phase_handler.run_individual_reflection_phase(
                            agents, items, preferences, round_num, config["t_rounds"],
                            feedback_result,
                            public_context=public_context_history,
                            private_context_by_agent=private_context_by_agent,
                        )
                    else:
                        self.logger.info(f"Skipping individual reflection phase (disabled)")

                    # Check for early termination (joint plans agree or pledges converge)
                    if game_environment.check_early_termination(preferences["game_state"]):
                        self.logger.info(f"Early termination in round {round_num}")
                        final_round = round_num
                        break

            # Post-round-loop processing
            if game_type == "co_funding":
                # Compute final outcome from the accepted joint proposal, if any.
                resolved_final_round = final_round if final_round > 0 else round_num
                final_outcome = game_environment.compute_final_outcome(
                    preferences["game_state"],
                    final_round=resolved_final_round
                )
                final_utilities = final_outcome["utilities"]
                final_allocation = final_outcome["funded_projects"]
                final_round = resolved_final_round
                agent_preferences_data = {
                    aid: preferences["agent_preferences"][aid]
                    for aid in preferences["agent_preferences"]
                }
                if consensus_reached:
                    self.logger.info(
                        f"Co-funding accepted joint proposal: {len(final_allocation)} projects funded"
                    )
                else:
                    self.logger.info("Co-funding complete: no jointly accepted proposal")

            # Calculate final utilities if no consensus (propose_and_vote only)
            elif not consensus_reached:
                final_round = config["t_rounds"]
                if game_type == "item_allocation":
                    final_utilities = {agent.agent_id: 0.0 for agent in agents}
                self.logger.info(f"No consensus after {config['t_rounds']} rounds")
        
        except Exception as e:
            import traceback
            self.logger.error(f"Error during negotiation: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # Analyze results
        if game_type == "co_funding":
            qualitative_metrics_v1, qualitative_events = compute_qualitative_metrics_v1(
                conversation_logs, preferences.get("game_state", {})
            )
            metrics_errors = validate_qualitative_metrics_v1(qualitative_metrics_v1)
            event_errors = []
            for idx, event in enumerate(qualitative_events):
                event_errors.extend(
                    [f"event[{idx}]: {err}" for err in validate_qualitative_event(event)]
                )
            if metrics_errors or event_errors:
                self.logger.warning(
                    "Qualitative metrics validation errors detected. "
                    "Dropping malformed qualitative outputs for this run."
                )
                for err in (metrics_errors + event_errors)[:10]:
                    self.logger.warning(f"  - {err}")
                qualitative_metrics_v1 = {}
                qualitative_events = []

        exploitation_detected = self.analyzer.detect_exploitation(conversation_logs)
        strategic_behaviors = self.analyzer.analyze_strategic_behaviors(conversation_logs)
        agent_performance = self.analyzer.analyze_agent_performance(agents, final_utilities)
        
        # Create enhanced config
        enhanced_config = self.utils.create_enhanced_config(
            config, agents, preferences, items, experiment_start_time, experiment_id
        )
        vote_integrity = self.phase_handler.get_vote_integrity()
        enhanced_config["vote_integrity"] = vote_integrity
        
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
            qualitative_metrics_v1=qualitative_metrics_v1,
            qualitative_events=qualitative_events,
            conversation_logs=conversation_logs,
            agent_performance=agent_performance,
            vote_integrity=vote_integrity,
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
        failed_runs = 0

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
                failed_runs += 1
                self.logger.error(f"Error in run {i+1}: {e}")
                continue

        if not experiments:
            error_msg = (
                f"Batch {batch_id or '[custom-output]'} produced 0 successful runs "
                f"out of {num_runs}; aborting instead of reporting success."
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        if failed_runs:
            self.logger.warning(
                "Batch completed with partial failures: %s/%s run(s) failed",
                failed_runs,
                num_runs,
            )
        
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
                         token_usage: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None):
        """Save an interaction to both all_interactions and agent-specific storage.

        Args:
            agent_id: Agent identifier
            phase: Phase name
            prompt: Prompt sent to agent
            response: Agent's response text
            round_num: Round number
            token_usage: Optional dict with token usage info (e.g., {'input_tokens': int, 'output_tokens': int, 'total_tokens': int})
        """
        prompt_text = prompt or ""
        prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() if prompt_text else None
        prompt_chars = len(prompt_text)
        prompt_storage_path = None
        stored_prompt = prompt
        if self.externalize_prompts and prompt_text:
            prompt_storage_path = self._write_external_prompt(
                agent_id=agent_id,
                phase=phase,
                round_num=round_num,
                prompt=prompt_text,
                prompt_sha256=prompt_sha256 or "",
            )
            stored_prompt = (
                f"[externalized prompt: {prompt_storage_path}; "
                f"chars={prompt_chars}; sha256={prompt_sha256}]"
            )

        interaction = {
            "timestamp": time.time(),
            "experiment_id": self.current_experiment_id,
            "agent_id": agent_id,
            "phase": phase,
            "round": round_num,
            "prompt": stored_prompt,
            "prompt_chars": prompt_chars,
            "prompt_sha256": prompt_sha256,
            "prompt_storage_path": prompt_storage_path,
            "response": response,
            "model_name": model_name if model_name else None
        }

        # Add token usage information if provided
        if token_usage:
            interaction["token_usage"] = token_usage
            # Also save reasoning_tokens at top level for easy access
            if "reasoning_tokens" in token_usage and token_usage["reasoning_tokens"]:
                interaction["reasoning_tokens"] = token_usage["reasoning_tokens"]

        # Track token usage for batch aggregation
        self._track_token_usage(agent_id, phase, prompt, response, token_usage)

        # Add to all interactions
        self.all_interactions.append(interaction)

        # Add to agent-specific interactions
        if agent_id not in self.agent_interactions:
            self.agent_interactions[agent_id] = []
        self.agent_interactions[agent_id].append(interaction)

        # Stream save to JSON files. Agent-specific files are overwritten from
        # canonical in-memory state for the changed agent to avoid re-appending
        # historical interactions on every stream save.
        self._stream_save_json(changed_agent_id=agent_id)
        self._write_progress_checkpoint(interaction)
        self.logger.info(
            "PROGRESS interaction=%s round=%s phase=%s agent=%s model=%s prompt_chars=%s response_chars=%s",
            len(self.all_interactions),
            round_num,
            phase,
            agent_id,
            model_name,
            prompt_chars,
            len(response or ""),
        )
        try:
            self._record_malformed_json_example_if_needed(interaction)
        except Exception as exc:
            self.logger.warning(
                "Failed to record malformed JSON diagnostic for phase=%s agent=%s: %s",
                phase,
                agent_id,
                exc,
            )

    def _record_malformed_json_example_if_needed(self, interaction: Dict[str, Any]) -> None:
        """Mirror malformed JSON parse diagnostics into per-run and batch debug files."""
        payload = self._decode_interaction_payload(interaction.get("response"))
        if not payload or not self._should_record_malformed_payload(interaction, payload):
            return

        example = self._build_malformed_json_example(interaction, payload)
        run_path, batch_jsonl_path = self._malformed_json_paths()
        self._append_malformed_json_example(run_path, example)
        self._append_malformed_jsonl_example(batch_jsonl_path, example)

    @staticmethod
    def _decode_interaction_payload(response: Any) -> Optional[Dict[str, Any]]:
        if isinstance(response, dict):
            return response
        if not isinstance(response, str) or not response.strip():
            return None
        try:
            payload = json.loads(response)
        except (TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _should_record_malformed_payload(
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> bool:
        if "parse_error" not in payload:
            return False
        if not (
            payload.get("raw_response")
            or payload.get("raw_proposal")
            or payload.get("raw_proposal_response")
        ):
            return False

        phase = str(interaction.get("phase") or "")
        if "invalid_attempt" in phase:
            return True
        if payload.get("used_fallback") is True:
            return True
        if payload.get("synthetic_vote") is True:
            return True
        if payload.get("hard_failed") is True or payload.get("will_retry") is True:
            return True
        return False

    def _build_malformed_json_example(
        self,
        interaction: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = dict(self.current_config or {})
        run_dir = self._current_output_dir()
        experiment_results_path = self._current_experiment_results_path(run_dir)
        raw_response = (
            payload.get("raw_response")
            or payload.get("raw_proposal_response")
            or payload.get("raw_proposal")
            or ""
        )
        interaction_phase = str(interaction.get("phase") or "")
        round_num = interaction.get("round", payload.get("round"))
        phase_category = self._phase_category(interaction_phase)
        turn = (
            payload.get("turn")
            if payload.get("turn") is not None
            else payload.get("proposal_repair_attempt")
        )

        run_metadata_keys = [
            "config_id",
            "job_id",
            "run_number",
            "batch_type",
            "experiment_family",
            "experiment_type",
            "game_label",
            "game_type",
            "competition_id",
            "n_agents",
            "competition_level",
            "rho",
            "theta",
            "sigma",
            "alpha",
            "model_order",
            "random_seed",
            "seed",
            "seed_replicate",
            "heterogeneous_run_index",
            "heterogeneous_draw_seed",
            "adversary_model",
            "adversary_position",
            "access_k",
            "curve_id",
            "curve_label",
            "scaled_model",
            "scaled_position",
            "k",
        ]
        run_metadata = {
            key: config.get(key)
            for key in run_metadata_keys
            if config.get(key) is not None
        }

        return {
            "schema_version": 1,
            "recorded_at": datetime.now().isoformat(),
            "timestamp": interaction.get("timestamp"),
            "experiment_id": self.current_experiment_id,
            "experiment_results_path": str(experiment_results_path),
            "full_experiment_results_filepath": str(experiment_results_path),
            "run_dir": str(run_dir),
            "batch_root": str(self._infer_batch_root(run_dir) or run_dir),
            "config_id": config.get("config_id") or config.get("job_id"),
            "game": config.get("game_label") or config.get("game_type") or payload.get("game_type"),
            "game_label": config.get("game_label"),
            "game_type": config.get("game_type") or payload.get("game_type"),
            "n_agents": config.get("n_agents"),
            "N": config.get("n_agents"),
            "competition_level": config.get("competition_level"),
            "competition_id": config.get("competition_id"),
            "rho": config.get("rho"),
            "theta": config.get("theta"),
            "sigma": config.get("sigma"),
            "alpha": config.get("alpha"),
            "model_name": interaction.get("model_name"),
            "model": interaction.get("model_name"),
            "agent_id": interaction.get("agent_id"),
            "round": round_num,
            "phase": phase_category,
            "interaction_phase": interaction_phase,
            "turn": turn,
            "proposal_repair_attempt": payload.get("proposal_repair_attempt"),
            "raw_malformed_json": raw_response,
            "raw_response": raw_response,
            "parse_error": payload.get("parse_error"),
            "error_summary": payload.get("error_summary"),
            "validation_error": payload.get("validation_error"),
            "will_retry": payload.get("will_retry"),
            "hard_failed": payload.get("hard_failed"),
            "source_payload": payload,
            "run_metadata": run_metadata,
        }

    @staticmethod
    def _phase_category(interaction_phase: str) -> str:
        if interaction_phase.startswith("proposal_round_"):
            return "proposal"
        if interaction_phase.startswith("voting_round_"):
            return "voting"
        if interaction_phase.startswith("private_thinking_round_"):
            return "private_thinking"
        if interaction_phase.startswith("discussion_round_"):
            return "discussion"
        if interaction_phase.startswith("reflection_round_"):
            return "reflection"
        if interaction_phase == "game_setup":
            return "initialization"
        return interaction_phase

    def _current_experiment_results_path(self, run_dir: Path) -> Path:
        if self.current_run_number is not None:
            return run_dir / f"run_{self.current_run_number}_experiment_results.json"
        return run_dir / "experiment_results.json"

    def _malformed_json_paths(self) -> tuple[Path, Path]:
        run_dir = self._current_output_dir()
        run_monitoring_dir = run_dir / "monitoring"
        run_monitoring_dir.mkdir(parents=True, exist_ok=True)
        run_path = run_monitoring_dir / "malformed_json_examples.json"

        batch_root = self._infer_batch_root(run_dir) or run_dir
        batch_monitoring_dir = batch_root / "monitoring"
        batch_monitoring_dir.mkdir(parents=True, exist_ok=True)
        batch_jsonl_path = batch_monitoring_dir / "malformed_json_examples.jsonl"
        return run_path, batch_jsonl_path

    @staticmethod
    def _infer_batch_root(run_dir: Path) -> Optional[Path]:
        resolved_run_dir = run_dir.resolve()
        if resolved_run_dir.parent.name == "runs":
            return resolved_run_dir.parent.parent

        report_path = os.getenv("LLM_FAILURE_REPORT_PATH")
        if report_path:
            report_parent = Path(report_path).expanduser().resolve().parent
            if report_parent.name == "monitoring":
                return report_parent.parent
        return None

    @staticmethod
    def _append_malformed_json_example(path: Path, example: Dict[str, Any]) -> None:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                examples = payload.get("examples", [])
                if not isinstance(examples, list):
                    examples = []
            except (OSError, json.JSONDecodeError, TypeError):
                examples = []
        else:
            examples = []

        examples.append(example)
        output = {
            "schema_version": 1,
            "updated_at": datetime.now().isoformat(),
            "count": len(examples),
            "examples": examples,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(output, indent=2, default=str, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)

    @staticmethod
    def _append_malformed_jsonl_example(path: Path, example: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock_handle:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(example, default=str, ensure_ascii=False) + "\n")
                    handle.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _current_output_dir(self) -> Path:
        """Return the directory used for streamed artifacts for the active run."""
        if self.use_custom_output:
            exp_dir = self.results_dir
        elif self.current_batch_id and self.current_run_number:
            exp_dir = self.results_dir / self.current_batch_id
        else:
            exp_dir = self.results_dir / self.current_experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    @staticmethod
    def _safe_filename_part(value: Any) -> str:
        text = str(value or "none")
        return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:80]

    def _write_external_prompt(
        self,
        *,
        agent_id: str,
        phase: str,
        round_num: Optional[int],
        prompt: str,
        prompt_sha256: str,
    ) -> str:
        """Persist large prompts outside all_interactions.json to avoid memory blowup."""
        exp_dir = self._current_output_dir()
        prompt_dir = exp_dir / "externalized_prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        index = len(self.all_interactions) + 1
        filename = (
            f"{index:06d}_round{round_num or 0}_"
            f"{self._safe_filename_part(phase)}_{self._safe_filename_part(agent_id)}_"
            f"{prompt_sha256[:12]}.txt.gz"
        )
        path = prompt_dir / filename
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(prompt)
        return str(path.relative_to(exp_dir))

    def _write_progress_checkpoint(self, interaction: Dict[str, Any]) -> None:
        """Write lightweight progress metadata after every saved interaction."""
        exp_dir = self._current_output_dir()
        phase_counts: Dict[str, int] = {}
        round_counts: Dict[str, int] = {}
        max_round_seen = 0
        for item in self.all_interactions:
            phase = str(item.get("phase"))
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            round_value = item.get("round")
            if round_value is not None:
                round_key = str(round_value)
                round_counts[round_key] = round_counts.get(round_key, 0) + 1
                try:
                    max_round_seen = max(max_round_seen, int(round_value))
                except (TypeError, ValueError):
                    pass
        progress = {
            "updated_at": datetime.now().isoformat(),
            "experiment_id": self.current_experiment_id,
            "interaction_count": len(self.all_interactions),
            "current_round": max_round_seen,
            "last_interaction": {
                "round": interaction.get("round"),
                "phase": interaction.get("phase"),
                "agent_id": interaction.get("agent_id"),
                "model_name": interaction.get("model_name"),
            },
            "phase_counts": phase_counts,
            "round_counts": round_counts,
        }
        (exp_dir / "progress.json").write_text(
            json.dumps(progress, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

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
            reasoning_tokens = token_usage.get('reasoning_tokens', 0) or 0
            self.batch_token_usage["actual_count"] += 1
        else:
            # Estimate from text
            input_tokens = int(len(prompt) / CHARS_PER_TOKEN) if prompt else 0
            output_tokens = int(len(response) / CHARS_PER_TOKEN) if response else 0
            reasoning_tokens = 0
            self.batch_token_usage["estimated_count"] += 1

        # Update totals
        self.batch_token_usage["total_input_tokens"] += input_tokens
        self.batch_token_usage["total_output_tokens"] += output_tokens
        self.batch_token_usage["total_reasoning_tokens"] += reasoning_tokens

        # Update by agent
        if agent_id not in self.batch_token_usage["by_agent"]:
            self.batch_token_usage["by_agent"][agent_id] = {"input": 0, "output": 0, "reasoning": 0}
        self.batch_token_usage["by_agent"][agent_id]["input"] += input_tokens
        self.batch_token_usage["by_agent"][agent_id]["output"] += output_tokens
        self.batch_token_usage["by_agent"][agent_id]["reasoning"] += reasoning_tokens

        # Update by phase (extract base phase name)
        base_phase = phase.split('_round_')[0].split('_turn_')[0]
        if base_phase not in self.batch_token_usage["by_phase"]:
            self.batch_token_usage["by_phase"][base_phase] = {"input": 0, "output": 0, "reasoning": 0}
        self.batch_token_usage["by_phase"][base_phase]["input"] += input_tokens
        self.batch_token_usage["by_phase"][base_phase]["output"] += output_tokens
        self.batch_token_usage["by_phase"][base_phase]["reasoning"] += reasoning_tokens

    def _reset_batch_token_tracking(self):
        """Reset token tracking for a new batch."""
        self.batch_token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
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
        total_reasoning = usage.get("total_reasoning_tokens", 0)

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
        self.logger.info(f"\n  Total Input Tokens:     {format_tokens(total_in)}")
        self.logger.info(f"  Total Output Tokens:    {format_tokens(total_out)}")
        self.logger.info(f"  Total Reasoning Tokens: {format_tokens(total_reasoning)}")
        self.logger.info(f"  Data Source:            {usage['actual_count']} actual, {usage['estimated_count']} estimated")

        # By agent breakdown
        if usage["by_agent"]:
            self.logger.info(f"\n  {'BY AGENT:':<20}")
            self.logger.info(f"    {'Agent':<20} {'Input':>12} {'Output':>12} {'Reasoning':>12}")
            self.logger.info("    " + "-" * 56)
            for agent_id, tokens in usage["by_agent"].items():
                reasoning = tokens.get('reasoning', 0)
                self.logger.info(f"    {agent_id:<20} {format_tokens(tokens['input']):>12} {format_tokens(tokens['output']):>12} {format_tokens(reasoning):>12}")

        # By phase breakdown
        if usage["by_phase"]:
            self.logger.info(f"\n  {'BY PHASE:':<20}")
            self.logger.info(f"    {'Phase':<25} {'Input':>12} {'Output':>12} {'Reasoning':>12}")
            self.logger.info("    " + "-" * 61)
            for phase, tokens in sorted(usage["by_phase"].items()):
                reasoning = tokens.get('reasoning', 0)
                self.logger.info(f"    {phase:<25} {format_tokens(tokens['input']):>12} {format_tokens(tokens['output']):>12} {format_tokens(reasoning):>12}")

        # Cost estimate (basic - use cost_dashboard.py for detailed analysis)
        self.logger.info(f"\n  {'COST NOTES:':<20}")
        self.logger.info("    For detailed cost analysis, run:")
        self.logger.info("    python visualization/cost_dashboard.py --dir <results_dir>")
        self.logger.info("\n" + "=" * 70 + "\n")
    
    def _stream_save_json(self, changed_agent_id: Optional[str] = None):
        """Stream save all interactions to JSON files."""
        if not self.current_experiment_id:
            return
        
        exp_dir = self._current_output_dir()
        
        # Determine batch mode: use run_number if set, regardless of batch_id
        # (batch_id can be empty string when using custom output directories)
        batch_mode = self.current_run_number is not None
        
        # Save all interactions
        self.file_manager.save_all_interactions(
            self.all_interactions, exp_dir,
            batch_mode, self.current_run_number
        )
        
        # Save agent-specific interactions by overwriting from canonical
        # in-memory state. Passing changed_agent_id keeps stream writes scoped
        # to the agent that just changed; the final run-level save passes None
        # to rewrite every agent file once.
        if changed_agent_id is None:
            agent_items = self.agent_interactions.items()
        else:
            agent_items = [(changed_agent_id, self.agent_interactions.get(changed_agent_id, []))]

        for agent_id, interactions in agent_items:
            self.file_manager.save_agent_interactions(
                agent_id,
                interactions,
                exp_dir,
                batch_mode,
                self.current_run_number
            )
