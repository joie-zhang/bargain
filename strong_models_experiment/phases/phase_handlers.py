"""Handlers for different negotiation phases."""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING, Awaitable, Callable
from negotiation import NegotiationContext
from negotiation.llm_agents import BaseLLMAgent
from ..prompts import PromptGenerator

# Import GameEnvironment for type checking to avoid circular imports
if TYPE_CHECKING:
    from game_environments import GameEnvironment


class PhaseHandler:
    """Handles execution of different negotiation phases.

    Supports two modes of operation:
    1. Legacy mode: Uses PromptGenerator for prompt generation (default)
    2. GameEnvironment mode: Uses GameEnvironment for game-specific prompt generation

    When a GameEnvironment is provided, it takes precedence over PromptGenerator
    for game-specific phases (discussion, proposal, voting).
    """

    def __init__(self, save_interaction_callback=None, token_config=None,
                 game_environment: Optional["GameEnvironment"] = None,
                 reasoning_config: Optional[Dict[str, Any]] = None,
                 parallel_phases: bool = False):
        if not isinstance(parallel_phases, bool):
            raise TypeError(
                "parallel_phases must be a boolean; partial numeric phase concurrency is unsupported"
            )
        self.logger = logging.getLogger(__name__)
        self.prompt_gen = PromptGenerator()
        self.save_interaction = save_interaction_callback or (lambda *args, **kwargs: None)

        # Optional GameEnvironment for game-specific behavior
        self.game_environment = game_environment

        # Token limits for different phases (None means unlimited)
        self.token_config = token_config or {
            "discussion": None,
            "proposal": None,
            "voting": None,
            "reflection": None,
            "thinking": None,
            "default": None
        }

        # Reasoning token budget configuration (for test-time compute scaling)
        # Format: {"budget": int, "phases": ["thinking", "reflection", ...]}
        self.reasoning_config = reasoning_config or {"budget": None, "phases": []}
        self.parallel_phases = parallel_phases

    async def _run_agent_tasks_in_order(
        self,
        agents: List[BaseLLMAgent],
        task_factory: Callable[[int, BaseLLMAgent], Awaitable[Any]],
    ) -> List[Any]:
        """Run one independent task per agent and return results in agent order.

        Parallel mode intentionally has no partial-concurrency knob: if enabled, it
        starts exactly one task per agent and waits for all tasks before the caller
        drains results. Callers then log/save in the original agent order.
        """
        if not self.parallel_phases or len(agents) <= 1:
            return [await task_factory(idx, agent) for idx, agent in enumerate(agents)]

        expected_tasks = len(agents)
        tasks = [task_factory(idx, agent) for idx, agent in enumerate(agents)]
        if len(tasks) != expected_tasks:
            raise RuntimeError(
                f"Parallel phase invariant violated: expected {expected_tasks} tasks, got {len(tasks)}"
            )

        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _extract_token_usage(self, agent_response) -> Optional[Dict[str, Any]]:
        """Extract token usage information from an AgentResponse object."""
        if not agent_response:
            return None

        token_usage = None
        if agent_response.metadata and agent_response.metadata.get("usage"):
            usage = agent_response.metadata["usage"]
            input_tokens = usage.get("prompt_tokens")
            if input_tokens is None:
                input_tokens = usage.get("input_tokens")

            output_tokens = usage.get("completion_tokens")
            if output_tokens is None:
                output_tokens = usage.get("output_tokens")

            total_tokens = usage.get("total_tokens")
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

            token_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens
            }
            # Extract reasoning tokens if available
            if agent_response.metadata.get("reasoning_tokens"):
                token_usage["reasoning_tokens"] = agent_response.metadata["reasoning_tokens"]
            elif usage.get("reasoning_tokens"):
                token_usage["reasoning_tokens"] = usage.get("reasoning_tokens")
        elif agent_response.tokens_used:
            token_usage = {
                "total_tokens": agent_response.tokens_used
            }
            # Check for reasoning tokens in metadata
            if agent_response.metadata and agent_response.metadata.get("reasoning_tokens"):
                token_usage["reasoning_tokens"] = agent_response.metadata["reasoning_tokens"]

        return token_usage

    def _get_reasoning_budget(self, phase: str, agent_id: Optional[str] = None) -> Optional[int]:
        """Get reasoning token budget for a specific phase and agent if applicable.

        Args:
            phase: The negotiation phase (thinking, reflection, discussion, etc.)
            agent_id: The agent ID to check. If provided and reasoning_agent_ids is set,
                     only returns budget if this agent is in the list.

        Returns:
            The reasoning token budget if applicable, None otherwise.
        """
        if not self.reasoning_config:
            return None
        budget = self.reasoning_config.get("budget")
        phases = self.reasoning_config.get("phases", [])

        # Check if this phase should have reasoning budget
        if not (budget and phase in phases):
            return None

        # Check if this agent should receive the reasoning prompt
        # If reasoning_agent_ids is set, only those agents get the prompt
        reasoning_agent_ids = self.reasoning_config.get("reasoning_agent_ids")
        if reasoning_agent_ids and agent_id:
            if agent_id not in reasoning_agent_ids:
                return None  # This agent (e.g., baseline) doesn't get reasoning prompt

        return budget

    def _resolve_agent_phase_max_tokens(
        self,
        agent: BaseLLMAgent,
        phase: str,
        fallback_phase: Optional[str] = None,
    ) -> Optional[int]:
        """Resolve the effective max_tokens for one agent in one phase.

        Combines experiment-level phase limits with optional per-agent limits stored in
        agent.config.custom_parameters["phase_token_caps"]. If both are present, the
        smaller value wins. When neither a phase limit nor a fallback/default limit is
        present for this agent, returns None to indicate no update should be applied.
        """
        config_limit = self.token_config.get(phase)
        custom_parameters = getattr(getattr(agent, "config", None), "custom_parameters", {}) or {}
        phase_caps = custom_parameters.get("phase_token_caps", {}) or {}
        agent_limit = phase_caps.get(phase)

        if config_limit is None and agent_limit is None and fallback_phase is not None:
            config_limit = self.token_config.get(fallback_phase)
            agent_limit = phase_caps.get(fallback_phase)

        if config_limit is None and agent_limit is None:
            return None
        if config_limit is None:
            return agent_limit
        if agent_limit is None:
            return config_limit
        return min(config_limit, agent_limit)

    def _apply_phase_token_limits(
        self,
        agents: List[BaseLLMAgent],
        phase: str,
        fallback_phase: Optional[str] = None,
    ) -> None:
        """Apply per-phase token limits to each agent if any relevant limit exists."""
        for agent in agents:
            resolved_limit = self._resolve_agent_phase_max_tokens(agent, phase, fallback_phase=fallback_phase)
            if resolved_limit is not None:
                agent.update_max_tokens(resolved_limit)

    def _parse_vote_response(self, content: str, agent_id: str, round_num: int) -> Dict:
        """Parse a vote response from raw LLM output.

        Extracts {"vote": "accept"/"reject", "reasoning": "..."} from
        potentially messy LLM text that may contain markdown or extra text
        surrounding the JSON.

        Args:
            content: Raw LLM response text
            agent_id: ID of the voting agent
            round_num: Current round number

        Returns:
            Dict with vote, reasoning, voter, and round fields
        """
        import re
        try:
            vote = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                vote = json.loads(json_match.group())
            else:
                raise ValueError(f"No valid JSON in vote response from {agent_id}")

        vote_val = vote.get("vote", "reject")
        if isinstance(vote_val, dict):
            vote_val = vote_val.get("decision", vote_val.get("vote", "reject"))
        if vote_val not in ("accept", "reject"):
            vote_val = "reject"

        return {
            "vote": vote_val,
            "reasoning": vote.get("reasoning", ""),
            "voter": agent_id,
            "round": round_num,
        }

    def _parse_commit_vote_response(self, content: str, agent_id: str, round_num: int) -> Dict:
        """Parse co-funding commit vote (yay/nay) from raw LLM output."""
        import re
        try:
            vote = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                vote = json.loads(json_match.group())
            else:
                # Last-resort keyword parse
                lowered = content.lower()
                if "yay" in lowered:
                    return {
                        "commit_vote": "yay",
                        "reasoning": "Parsed from plain-text response",
                        "voter": agent_id,
                        "round": round_num,
                    }
                return {
                    "commit_vote": "nay",
                    "reasoning": "No valid JSON in commit vote response",
                    "voter": agent_id,
                    "round": round_num,
                }

        raw_vote = (
            vote.get("commit_vote")
            or vote.get("vote")
            or vote.get("decision")
            or vote.get("commit")
            or "nay"
        )
        if isinstance(raw_vote, dict):
            raw_vote = raw_vote.get("commit_vote", raw_vote.get("vote", raw_vote.get("decision", "nay")))

        normalized = str(raw_vote).strip().lower()
        if normalized in ("yay", "yes", "accept", "approved", "approve", "commit", "true"):
            commit_vote = "yay"
        elif normalized in ("nay", "no", "reject", "decline", "false"):
            commit_vote = "nay"
        else:
            commit_vote = "nay"

        return {
            "commit_vote": commit_vote,
            "reasoning": vote.get("reasoning", vote.get("rationale", "")),
            "voter": agent_id,
            "round": round_num,
        }

    def _build_game_state(self, agents: List[BaseLLMAgent], items: List[Dict],
                          preferences: Dict, round_num: int, max_rounds: int,
                          agent_id: str, phase: str,
                          conversation_history: Optional[List[Dict]] = None,
                          proposals: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Build a game state dictionary for GameEnvironment.

        Args:
            agents: List of agent objects
            items: List of item dictionaries
            preferences: Dictionary containing agent_preferences
            round_num: Current round number
            max_rounds: Maximum number of rounds
            agent_id: ID of the current agent
            phase: Current phase name
            conversation_history: Optional list of conversation messages
            proposals: Optional list of proposals

        Returns:
            Dictionary representing the current game state
        """
        return {
            "round": round_num,
            "max_rounds": max_rounds,
            "items": items,
            "agents": [a.agent_id for a in agents],
            "agent_id": agent_id,
            # Preserve the canonical key expected by game environments.
            "agent_preferences": preferences.get("agent_preferences", {}),
            "preferences": preferences.get("agent_preferences", {}).get(agent_id, []),
            "all_preferences": preferences.get("agent_preferences", {}),
            "phase": phase,
            "conversation_history": conversation_history or [],
            "proposals": proposals or [],
        }

    def _get_context_preferences(
        self,
        agent_id: str,
        preferences: Dict[str, Any],
    ) -> Any:
        """Return the preference payload replayed in the per-call system prompt."""
        agent_preferences = preferences["agent_preferences"][agent_id]

        if (
            self.game_environment is not None
            and "game_state" in preferences
            and getattr(self.game_environment.get_game_type(), "value", None) == "diplomacy"
        ):
            return self.game_environment.get_agent_preferences_summary(
                agent_id,
                preferences["game_state"],
            )

        return agent_preferences

    async def run_game_setup_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                  preferences: Dict, config: Dict) -> None:
        """Phase 1A: Game Setup Phase"""
        self.logger.info("=== GAME SETUP PHASE ===")

        # Apply token limits for setup/default phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "default")

        if self.game_environment is not None:
            if "game_state" in preferences:
                game_state = preferences["game_state"]
            else:
                game_state = {"items": items, "agent_preferences": preferences.get("agent_preferences", {})}
        else:
            game_state = None

        if self.game_environment is not None and self.game_environment.uses_combined_setup_phase():
            self.logger.info("📜 COMBINED GAME SETUP PROMPTS:")

            async def setup_agent(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
                setup_prompt = self.game_environment.get_combined_setup_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state
                )

                context = NegotiationContext(
                    current_round=0,
                    max_rounds=config["t_rounds"],
                    items=items,
                    agents=[a.agent_id for a in agents],
                    agent_id=agent.agent_id,
                    preferences=preferences["agent_preferences"][agent.agent_id],
                    turn_type="setup"
                )

                agent_response = await agent.generate_response(context, setup_prompt)
                return {
                    "agent": agent,
                    "prompt": setup_prompt,
                    "response_content": agent_response.content,
                    "token_usage": self._extract_token_usage(agent_response),
                }

            setup_results = await self._run_agent_tasks_in_order(agents, setup_agent)
            for result in setup_results:
                if isinstance(result, BaseException):
                    raise result
                agent = result["agent"]
                response_content = result["response_content"]
                token_usage = result["token_usage"]

                self.logger.info(f"  📬 {agent.agent_id} response:")
                self.logger.info(f"    {response_content}")
                self.save_interaction(
                    agent.agent_id,
                    "game_setup",
                    result["prompt"],
                    response_content,
                    0,
                    token_usage,
                    model_name=agent.get_model_info()["model_name"]
                )

            self.logger.info(
                "Game setup phase completed - all agents briefed on rules and private preferences"
            )
            return

        # Use GameEnvironment if available, otherwise fall back to PromptGenerator
        if self.game_environment is not None:
            game_rules_prompt = self.game_environment.get_game_rules_prompt(game_state)
        else:
            game_rules_prompt = self.prompt_gen.create_game_rules_prompt(items, len(agents), config)
        
        self.logger.info("📜 GAME RULES PROMPT:")
        self.logger.info(f"  {game_rules_prompt}")

        async def setup_agent_with_rules(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
            context = NegotiationContext(
                current_round=0,
                max_rounds=config["t_rounds"],
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="setup"
            )

            agent_response = await agent.generate_response(context, game_rules_prompt)
            return {
                "agent": agent,
                "response_content": agent_response.content,
                "token_usage": self._extract_token_usage(agent_response),
            }

        setup_results = await self._run_agent_tasks_in_order(agents, setup_agent_with_rules)
        for result in setup_results:
            if isinstance(result, BaseException):
                raise result
            agent = result["agent"]
            response_content = result["response_content"]
            token_usage = result["token_usage"]

            self.logger.info(f"  📬 {agent.agent_id} response:")
            self.logger.info(f"    {response_content}")
            self.save_interaction(agent.agent_id, "game_setup", game_rules_prompt, response_content, 0, token_usage, model_name=agent.get_model_info()["model_name"])
        
        self.logger.info("Game setup phase completed - all agents briefed on rules")
    
    async def run_private_preference_assignment(self, agents: List[BaseLLMAgent], items: List[Dict],
                                               preferences: Dict, config: Dict) -> None:
        """Phase 1B: Private Preference Assignment"""
        self.logger.info("=== PRIVATE PREFERENCE ASSIGNMENT ===")

        # Apply token limits for preference assignment if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "default")

        if self.game_environment is not None and self.game_environment.uses_combined_setup_phase():
            self.logger.info(
                "Skipping separate preference assignment phase - this game merges it into setup"
            )
            return

        async def assign_preferences_for_agent(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
            agent_preferences = preferences["agent_preferences"][agent.agent_id]

            # Use GameEnvironment if provided, otherwise use legacy PromptGenerator
            if self.game_environment is not None:
                if "game_state" in preferences:
                    game_state = preferences["game_state"]
                else:
                    game_state = {"items": items, "agent_preferences": preferences.get("agent_preferences", {})}
                preference_prompt = self.game_environment.get_preference_assignment_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state
                )
            else:
                preference_prompt = self.prompt_gen.create_preference_assignment_prompt(
                    items, agent_preferences, agent.agent_id
                )

            context = NegotiationContext(
                current_round=0,
                max_rounds=config["t_rounds"],
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=agent_preferences,
                turn_type="preference_assignment"
            )

            agent_response = await agent.generate_response(context, preference_prompt)
            return {
                "agent": agent,
                "agent_preferences": agent_preferences,
                "preference_prompt": preference_prompt,
                "response_content": agent_response.content,
                "token_usage": self._extract_token_usage(agent_response),
            }

        preference_results = await self._run_agent_tasks_in_order(agents, assign_preferences_for_agent)
        for result in preference_results:
            if isinstance(result, BaseException):
                raise result
            agent = result["agent"]
            agent_preferences = result["agent_preferences"]
            response_content = result["response_content"]
            token_usage = result["token_usage"]

            self.logger.info(f"  🎯 {agent.agent_id} preferences:")
            for i, (item, value) in enumerate(zip(items, agent_preferences)):
                item_name = item["name"] if isinstance(item, dict) else str(item)
                self.logger.info(f"    - {item_name}: {value:.1f}")

            self.logger.info(f"  📬 {agent.agent_id} acknowledgment:")
            self.logger.info(f"    {response_content}")
            self.save_interaction(agent.agent_id, "preference_assignment", result["preference_prompt"], response_content, 0, token_usage, model_name=agent.get_model_info()["model_name"])
        
        self.logger.info("Private preference assignment completed")
    
    async def run_discussion_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                  preferences: Dict, round_num: int, max_rounds: int,
                                  discussion_turns: int = 2,
                                  public_context: Optional[List[Dict[str, Any]]] = None,
                                  private_context_by_agent: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Phase 2: Public Discussion Phase

        When a GameEnvironment is provided, uses its get_*_prompt() methods
        for game-specific discussion prompts. Otherwise falls back to PromptGenerator.

        Args:
            agents: List of agent objects
            items: List of items being negotiated
            preferences: Dictionary containing agent preferences
            round_num: Current round number
            max_rounds: Maximum number of rounds
            discussion_turns: Number of times agents go around the circle discussing (default: 2)
        """
        messages = []

        self.logger.info(f"=== PUBLIC DISCUSSION PHASE - Round {round_num} ===")
        self.logger.info(f"  Discussion turns: {discussion_turns}")

        # Apply token limits for discussion phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "discussion")

        # Outer loop: go around the circle discussion_turns times
        for turn in range(discussion_turns):
            self.logger.info(f"  --- Discussion Turn {turn + 1}/{discussion_turns} ---")

            for i, agent in enumerate(agents):
                accumulated_public_context = list(public_context or []) + list(messages)
                context = NegotiationContext(
                    current_round=round_num,
                    max_rounds=max_rounds,
                    items=items,
                    agents=[a.agent_id for a in agents],
                    agent_id=agent.agent_id,
                    preferences=self._get_context_preferences(agent.agent_id, preferences),
                    turn_type="discussion",
                    conversation_history=accumulated_public_context,
                    strategic_notes=(private_context_by_agent or {}).get(agent.agent_id, [])
                )

                # Build discussion history with speaker attribution
                current_discussion_history = [
                    f"**{msg['from']}**: {msg['content']}"
                    for msg in messages
                ]

                # Use GameEnvironment if available, otherwise fall back to PromptGenerator
                if self.game_environment is not None:
                    # Get the original game_state if stored in preferences, otherwise build one
                    if "game_state" in preferences:
                        game_state = preferences["game_state"]
                    else:
                        game_state = self._build_game_state(
                            agents, items, preferences, round_num, max_rounds,
                            agent.agent_id, "discussion",
                            conversation_history=accumulated_public_context
                        )

                    # Get reasoning budget for this specific agent
                    reasoning_budget = self._get_reasoning_budget("discussion", agent.agent_id)

                    full_discussion_prompt = self.game_environment.get_discussion_prompt(
                        agent_id=agent.agent_id,
                        game_state=game_state,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        discussion_history=current_discussion_history,
                        reasoning_token_budget=reasoning_budget
                    )
                else:
                    # Legacy mode: use PromptGenerator
                    if round_num == 1 and turn == 0:
                        discussion_prompt = self.prompt_gen.create_initial_discussion_prompt(
                            items, round_num, max_rounds
                        )
                    else:
                        discussion_prompt = self.prompt_gen.create_ongoing_discussion_prompt(
                            items, round_num, max_rounds
                        )
                    full_discussion_prompt = self.prompt_gen.create_contextual_discussion_prompt(
                        discussion_prompt, agent.agent_id, current_discussion_history,
                        i + 1, len(agents)
                    )

                # Get response with token info
                agent_response = await agent.generate_response(context, full_discussion_prompt)
                response_content = agent_response.content
                token_usage = self._extract_token_usage(agent_response)

                message = {
                    "phase": "discussion",
                    "round": round_num,
                    "discussion_turn": turn + 1,
                    "from": agent.agent_id,
                    "content": response_content,
                    "timestamp": time.time(),
                    "speaker_order": i + 1,
                    "total_speakers": len(agents)
                }
                messages.append(message)

                self.logger.info(f"  💬 Turn {turn+1} Speaker {i+1}/{len(agents)} - {agent.agent_id}:")
                self.logger.info(f"    {response_content}")

                self.save_interaction(agent.agent_id, f"discussion_round_{round_num}_turn_{turn+1}",
                                    full_discussion_prompt, response_content, round_num, token_usage, model_name=agent.get_model_info()["model_name"])

        self.logger.info(f"Discussion phase completed - {len(messages)} messages exchanged across {discussion_turns} turns")
        return {"messages": messages}
    
    async def run_private_thinking_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                        preferences: Dict, round_num: int, max_rounds: int,
                                        discussion_messages: List[Dict],
                                        public_context: Optional[List[Dict[str, Any]]] = None,
                                        private_context_by_agent: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Phase 3: Private Thinking Phase"""
        thinking_results = []
        
        self.logger.info(f"=== PRIVATE THINKING PHASE - Round {round_num} ===")
        
        # Apply token limits for thinking phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "thinking")

        async def think_for_agent(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
            # Get reasoning budget for this specific agent
            reasoning_budget = self._get_reasoning_budget("thinking", agent.agent_id)

            # Use GameEnvironment if available, otherwise fall back to PromptGenerator
            if self.game_environment is not None:
                # Get the original game_state if stored in preferences
                if "game_state" in preferences:
                    game_state = preferences["game_state"]
                else:
                    game_state = self._build_game_state(
                        agents, items, preferences, round_num, max_rounds,
                        agent.agent_id, "thinking",
                        conversation_history=list(public_context or [])
                    )
                thinking_prompt = self.game_environment.get_thinking_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    discussion_history=[],
                    reasoning_token_budget=reasoning_budget
                )
            else:
                thinking_prompt = self.prompt_gen.create_thinking_prompt(items, round_num, max_rounds, discussion_messages)

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=self._get_context_preferences(agent.agent_id, preferences),
                turn_type="thinking",
                conversation_history=list(public_context or []),
                strategic_notes=(private_context_by_agent or {}).get(agent.agent_id, [])
            )

            try:
                thinking_response = await agent.think_strategy(thinking_prompt, context)

                # Extract token usage if present (remove from response to avoid saving it in the content)
                token_usage = thinking_response.pop("_token_usage", None)
                return {
                    "agent": agent,
                    "thinking_prompt": thinking_prompt,
                    "thinking_response": thinking_response,
                    "token_usage": token_usage,
                    "error": None,
                }
            except Exception as e:
                return {"agent": agent, "error": e}

        raw_results = await self._run_agent_tasks_in_order(agents, think_for_agent)
        for result in raw_results:
            if isinstance(result, BaseException):
                raise result
            agent = result["agent"]

            if result["error"] is not None:
                self.logger.error(f"Error in private thinking for {agent.agent_id}: {result['error']}")
                thinking_results.append({
                    "agent_id": agent.agent_id,
                    "reasoning": "Unable to complete strategic thinking due to error",
                    "strategy": "Will propose based on known preferences",
                    "key_priorities": [],
                    "potential_concessions": [],
                    "target_items": [],
                    "anticipated_resistance": []
                })
                continue

            thinking_response = result["thinking_response"]
            token_usage = result["token_usage"]
            thinking_prompt = result["thinking_prompt"]

            self.logger.info(f"🧠 [PRIVATE] {agent.agent_id} strategic thinking:")
            self.logger.info(f"  Full reasoning: {thinking_response.get('reasoning', 'No reasoning provided')}")
            self.logger.info(f"  Strategy: {thinking_response.get('strategy', 'No strategy provided')}")
            # Log priorities/targets based on game type
            priorities = thinking_response.get('key_priorities') or thinking_response.get('target_items', [])
            concessions = thinking_response.get('potential_concessions') or thinking_response.get('anticipated_resistance', [])
            self.logger.info(f"  Key priorities: {priorities}")
            self.logger.info(f"  Potential concessions: {concessions}")

            thinking_response_str = json.dumps(thinking_response, default=str)
            self.save_interaction(agent.agent_id, f"private_thinking_round_{round_num}",
                                thinking_prompt, thinking_response_str, round_num, token_usage, model_name=agent.get_model_info()["model_name"])

            thinking_results.append({
                "agent_id": agent.agent_id,
                "reasoning": thinking_response.get('reasoning', ''),
                "strategy": thinking_response.get('strategy', ''),
                "key_priorities": priorities,
                "potential_concessions": concessions,
                "target_items": thinking_response.get('target_items', priorities),
                "anticipated_resistance": thinking_response.get('anticipated_resistance', concessions)
            })
        
        return {
            "thinking_results": thinking_results,
            "round_num": round_num
        }
    
    async def run_proposal_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                preferences: Dict, round_num: int, max_rounds: int,
                                public_context: Optional[List[Dict[str, Any]]] = None,
                                private_context_by_agent: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Phase 4A: Proposal Submission Phase

        When a GameEnvironment is provided, uses its get_*_prompt() methods
        for game-specific proposal prompts. Otherwise uses agent's default behavior.
        """
        messages = []
        proposals = []

        self.logger.info(f"=== PROPOSAL SUBMISSION PHASE - Round {round_num} ===")

        # Apply token limits for proposal phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "proposal")

        async def propose_for_agent(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
            log_records = []
            # Get reasoning budget for this specific agent
            reasoning_budget = self._get_reasoning_budget("proposal", agent.agent_id)

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=self._get_context_preferences(agent.agent_id, preferences),
                turn_type="proposal",
                conversation_history=list(public_context or []),
                strategic_notes=(private_context_by_agent or {}).get(agent.agent_id, [])
            )

            # Generate proposal prompt (for logging and potential custom prompts)
            if self.game_environment is not None:
                # Get the original game_state if stored in preferences
                if "game_state" in preferences:
                    game_state = preferences["game_state"]
                else:
                    game_state = self._build_game_state(
                        agents, items, preferences, round_num, max_rounds,
                        agent.agent_id, "proposal",
                        conversation_history=list(public_context or [])
                    )
                proposal_prompt = self.game_environment.get_proposal_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state,
                    round_num=round_num,
                    agents=[a.agent_id for a in agents],
                    reasoning_token_budget=reasoning_budget
                )
            else:
                proposal_prompt = f"Please propose an allocation for round {round_num}."

            # Get proposal from agent
            if self.game_environment is not None:
                # Use game-specific prompt (already generated above) and parsing
                response = await agent.generate_response(context, proposal_prompt)
                token_usage = self._extract_token_usage(response)
                proposal = self.game_environment.parse_proposal(
                    response.content, agent.agent_id, game_state,
                    [a.agent_id for a in agents]
                )
                should_retry_invalid = (
                    game_state.get("game_type") == "co_funding"
                    or "contributions" in proposal
                )
                if should_retry_invalid and not self.game_environment.validate_proposal(proposal, game_state):
                    log_records.append(
                        ("warning", f"  Invalid proposal from {agent.agent_id}, retrying once...")
                    )
                    retry_prompt = (
                        proposal_prompt
                        + "\n\n**ERROR: Your previous proposal was invalid "
                        "(negative values, over-budget, or wrong length). "
                        "Please resubmit a valid proposal.**"
                    )
                    response = await agent.generate_response(context, retry_prompt)
                    token_usage = self._extract_token_usage(response)
                    proposal = self.game_environment.parse_proposal(
                        response.content, agent.agent_id, game_state,
                        [a.agent_id for a in agents]
                    )
                    if not self.game_environment.validate_proposal(proposal, game_state):
                        log_records.append(
                            (
                                "error",
                                f"  {agent.agent_id} proposal still invalid after retry, using zero vector",
                            )
                        )
                        proposal = {
                            "contributions": [0.0] * game_state["m_projects"],
                            "reasoning": "Fallback: zero contributions after validation failure",
                            "proposed_by": agent.agent_id,
                            "raw_response": response.content,
                            "validation_error": "Proposal invalid after retry",
                        }
                proposal["proposed_by"] = agent.agent_id
                proposal["round"] = round_num
                if token_usage:
                    proposal["_token_usage"] = token_usage
            else:
                # Legacy item allocation path
                proposal = await agent.propose_allocation(context)

            return {
                "agent": agent,
                "proposal_prompt": proposal_prompt,
                "proposal": proposal,
                "log_records": log_records,
            }

        raw_results = await self._run_agent_tasks_in_order(agents, propose_for_agent)
        for result in raw_results:
            if isinstance(result, BaseException):
                raise result
            agent = result["agent"]
            proposal_prompt = result["proposal_prompt"]
            proposal = result["proposal"]
            for level, message in result["log_records"]:
                if level == "warning":
                    self.logger.warning(message)
                elif level == "error":
                    self.logger.error(message)

            # Extract token usage if present (remove from proposal to avoid saving it in the content)
            token_usage = proposal.pop("_token_usage", None)

            proposals.append(proposal)

            proposal_str = json.dumps(proposal, default=str)
            self.save_interaction(agent.agent_id, f"proposal_round_{round_num}",
                                proposal_prompt, proposal_str, round_num, token_usage, model_name=agent.get_model_info()["model_name"])

            # Build message content based on proposal format
            if "agreement" in proposal:
                content_key = "agreement"
            elif "allocation" in proposal:
                content_key = "allocation"
            else:
                content_key = "contributions"
            proposal_content = proposal.get(content_key, {})
            message = {
                "phase": "proposal",
                "round": round_num,
                "from": agent.agent_id,
                "content": f"I propose: {proposal_content}",
                "proposal": proposal,
                "timestamp": time.time(),
                "agent_id": agent.agent_id
            }
            messages.append(message)

            self.logger.info(f"📋 {agent.agent_id} FORMAL PROPOSAL:")
            self.logger.info(f"   {content_key.title()}: {proposal_content}")
            self.logger.info(f"   Reasoning: {proposal.get('reasoning', 'No reasoning provided')}")
        
        return {"messages": messages, "proposals": proposals}
    
    async def run_proposal_enumeration_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                            preferences: Dict, round_num: int, max_rounds: int,
                                            proposals: List[Dict]) -> Dict:
        """Phase 4B: Proposal Enumeration Phase"""
        messages = []
        
        self.logger.info(f"=== PROPOSAL ENUMERATION PHASE - Round {round_num} ===")
        
        if not proposals:
            self.logger.warning("No proposals to enumerate!")
            return {
                "messages": messages,
                "enumerated_proposals": [],
                "proposal_summary": "No proposals submitted"
            }
        
        proposals_for_voting = proposals
        if self.game_environment is not None:
            game_state = preferences.get("game_state", {
                "items": items,
                "agent_preferences": preferences.get("agent_preferences", {}),
            })
            proposals_for_voting = self.game_environment.prepare_proposals_for_voting(
                proposals=proposals,
                game_state=game_state,
                round_num=round_num,
            )

        enumerated_proposals = []
        proposal_display_lines = []
        
        proposal_display_lines.append(f"📋 FORMAL PROPOSALS SUBMITTED - Round {round_num}")
        proposal_display_lines.append("=" * 60)
        proposal_display_lines.append(f"Total Proposals: {len(proposals_for_voting)}")
        proposal_display_lines.append("")
        
        for i, proposal in enumerate(proposals_for_voting, 1):
            proposal_num = i
            proposer = proposal.get('proposed_by', f'Agent_{i}')
            reasoning = proposal.get('reasoning', 'No reasoning provided')

            enumerated_proposal = {
                "proposal_number": proposal_num,
                "proposer": proposer,
                "reasoning": reasoning,
                "original_proposal": proposal
            }

            # Carry through the native format key for backward compatibility
            if "allocation" in proposal:
                enumerated_proposal["allocation"] = proposal["allocation"]
            if "agreement" in proposal:
                enumerated_proposal["agreement"] = proposal["agreement"]
            if "contributions_by_agent" in proposal:
                enumerated_proposal["contributions_by_agent"] = proposal["contributions_by_agent"]
                enumerated_proposal["funded_projects"] = proposal.get("funded_projects", [])
                enumerated_proposal["aggregate_totals"] = proposal.get("aggregate_totals", [])

            enumerated_proposals.append(enumerated_proposal)

            # Display: delegate to game_environment if available
            if self.game_environment is not None:
                game_state = preferences.get("game_state", {
                    "items": items,
                    "agent_preferences": preferences.get("agent_preferences", {}),
                })
                display_text = self.game_environment.format_proposal_display(proposal, game_state)
                proposal_display_lines.append(f"PROPOSAL #{proposal_num}:")
                proposal_display_lines.append(display_text)
                proposal_display_lines.append("")
            else:
                # Legacy item allocation display
                allocation = proposal.get('allocation', {})
                proposal_display_lines.append(f"PROPOSAL #{proposal_num} (by {proposer}):")
                proposal_display_lines.append(f"  Allocation:")

                for agent_id, item_indices in allocation.items():
                    if item_indices:
                        item_names = []
                        for idx in item_indices:
                            try:
                                idx_int = int(idx)
                                if 0 <= idx_int < len(items):
                                    item_names.append(f"{idx_int}:{items[idx_int]['name']}")
                                else:
                                    item_names.append(f"{idx}:unknown")
                            except (ValueError, TypeError):
                                item_names.append(f"{idx}:unknown")
                        proposal_display_lines.append(f"    → {agent_id}: {', '.join(item_names)}")
                    else:
                        proposal_display_lines.append(f"    → {agent_id}: (no items)")

                proposal_display_lines.append("")
        
        proposal_summary = "\n".join(proposal_display_lines)
        
        self.logger.info("📋 PROPOSAL ENUMERATION:")
        for line in proposal_display_lines:
            self.logger.info(f"  {line}")
        
        enumeration_message = {
            "phase": "proposal_enumeration",
            "round": round_num,
            "from": "system",
            "content": proposal_summary,
            "enumerated_proposals": enumerated_proposals,
            "timestamp": time.time(),
            "agent_id": "system",
            "message_type": "enumeration"
        }
        messages.append(enumeration_message)
        
        return {
            "messages": messages,
            "enumerated_proposals": enumerated_proposals,
            "proposal_summary": proposal_summary,
            "total_proposals": len(proposals_for_voting)
        }
    
    async def run_private_voting_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                      preferences: Dict, round_num: int, max_rounds: int,
                                      proposals: List[Dict], enumerated_proposals: List[Dict],
                                      public_context: Optional[List[Dict[str, Any]]] = None,
                                      private_context_by_agent: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Phase 5A: Private Voting Phase

        When a GameEnvironment is provided, uses its get_*_prompt() methods
        for game-specific voting prompts. Otherwise uses default voting prompts.
        """
        private_votes = []

        self.logger.info(f"=== PRIVATE VOTING PHASE - Round {round_num} ===")

        if not enumerated_proposals:
            self.logger.warning("No enumerated proposals available for voting!")
            return {
                "private_votes": [],
                "voting_summary": "No proposals to vote on"
            }

        # Apply token limits for voting phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "voting")

        batch_voting_supported = (
            self.game_environment is not None
            and hasattr(self.game_environment, "get_batch_voting_prompt")
            and hasattr(self.game_environment, "parse_batch_voting_response")
        )

        prepared_batch_proposals = []
        if batch_voting_supported:
            for enum_proposal in enumerated_proposals:
                proposal_for_voting = enum_proposal.get("original_proposal", {}).copy()
                proposal_for_voting["proposal_number"] = enum_proposal["proposal_number"]
                proposal_for_voting["proposed_by"] = enum_proposal["proposer"]
                proposal_for_voting["round"] = round_num
                prepared_batch_proposals.append(proposal_for_voting)

        async def vote_for_agent(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
            # Get reasoning budget for this specific agent
            reasoning_budget = self._get_reasoning_budget("voting", agent.agent_id)

            agent_votes = []
            interaction_records = []
            warnings = []

            voting_context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=self._get_context_preferences(agent.agent_id, preferences),
                turn_type="private_voting",
                current_proposals=proposals,
                conversation_history=list(public_context or []),
                strategic_notes=(private_context_by_agent or {}).get(agent.agent_id, [])
            )

            try:
                if batch_voting_supported:
                    if "game_state" in preferences:
                        game_state = preferences["game_state"]
                    else:
                        game_state = self._build_game_state(
                            agents, items, preferences, round_num, max_rounds,
                            agent.agent_id, "voting",
                            conversation_history=list(public_context or []),
                            proposals=prepared_batch_proposals
                        )

                    voting_prompt = self.game_environment.get_batch_voting_prompt(
                        agent_id=agent.agent_id,
                        proposals=prepared_batch_proposals,
                        game_state=game_state,
                        round_num=round_num,
                        reasoning_token_budget=reasoning_budget
                    )

                    response = await agent.generate_response(voting_context, voting_prompt)
                    token_usage = self._extract_token_usage(response)
                    proposal_numbers = [
                        proposal["proposal_number"]
                        for proposal in prepared_batch_proposals
                    ]

                    vote_results = self.game_environment.parse_batch_voting_response(
                        response.content,
                        proposal_numbers,
                        agent.agent_id,
                        round_num
                    )

                    def missing_vote_numbers(votes: List[Dict[str, Any]]) -> List[int]:
                        missing = []
                        by_number = {
                            vote.get("proposal_number"): vote
                            for vote in votes
                            if isinstance(vote, dict)
                        }
                        for proposal_number in proposal_numbers:
                            vote = by_number.get(proposal_number, {})
                            if (
                                vote.get("reasoning") == "Missing or invalid vote entry"
                                or "parse_error" in vote
                            ):
                                missing.append(proposal_number)
                        return missing

                    missing_numbers = missing_vote_numbers(vote_results)
                    if missing_numbers:
                        self.logger.warning(
                            f"  Batch vote from {agent.agent_id} omitted or invalidated "
                            f"proposal votes {missing_numbers}; retrying once."
                        )
                        retry_prompt = (
                            voting_prompt
                            + "\n\nYour previous response was incomplete or invalid. "
                            f"Return exactly one JSON object with a votes array containing "
                            f"one entry for every proposal number in this exact list: {proposal_numbers}. "
                            "Do not omit any proposal. If unsure, choose reject and explain briefly. "
                            "No prose outside the JSON object."
                        )
                        retry_response = await agent.generate_response(voting_context, retry_prompt)
                        retry_token_usage = self._extract_token_usage(retry_response)
                        retry_vote_results = self.game_environment.parse_batch_voting_response(
                            retry_response.content,
                            proposal_numbers,
                            agent.agent_id,
                            round_num
                        )
                        retry_missing_numbers = missing_vote_numbers(retry_vote_results)
                        if len(retry_missing_numbers) <= len(missing_numbers):
                            response = retry_response
                            token_usage = retry_token_usage
                            vote_results = retry_vote_results
                            missing_numbers = retry_missing_numbers
                        if missing_numbers:
                            warnings.append(
                                f"Batch vote from {agent.agent_id} still missing/invalid "
                                f"proposal votes after retry: {missing_numbers}"
                            )

                    for idx, (enum_proposal, vote_result) in enumerate(zip(enumerated_proposals, vote_results)):
                        vote_entry = {
                            "voter_id": agent.agent_id,
                            "proposal_number": enum_proposal["proposal_number"],
                            "vote": vote_result.get("vote", "reject"),
                            "reasoning": vote_result.get("reasoning", "Strategic voting decision"),
                            "round": round_num,
                            "timestamp": time.time()
                        }
                        agent_votes.append(vote_entry)

                        if "agreement" in enum_proposal:
                            proposal_detail_key = "agreement"
                        elif "allocation" in enum_proposal:
                            proposal_detail_key = "allocation"
                        else:
                            proposal_detail_key = "contributions_by_agent"
                        enhanced_vote_response = {
                            "voter": agent.agent_id,
                            "proposal_number": enum_proposal["proposal_number"],
                            "proposal_by": enum_proposal["proposer"],
                            "vote_decision": vote_result.get("vote", "reject"),
                            "reasoning": vote_result.get("reasoning", "Strategic voting decision"),
                            "proposal_details": {
                                proposal_detail_key: enum_proposal.get(proposal_detail_key, {}),
                                "original_reasoning": enum_proposal.get("reasoning", "")
                            },
                            "round": round_num,
                            "timestamp": time.time()
                        }
                        if "raw_response" in vote_result:
                            enhanced_vote_response["raw_response"] = vote_result["raw_response"]
                        if "parse_error" in vote_result:
                            enhanced_vote_response["parse_error"] = vote_result["parse_error"]

                        interaction_records.append(
                            {
                                "phase": f"voting_round_{round_num}_proposal_{enum_proposal['proposal_number']}",
                                "prompt": voting_prompt,
                                "response": json.dumps(enhanced_vote_response, default=str),
                                "token_usage": token_usage if idx == 0 else None,
                                "log_message": f"  [PRIVATE] {agent.agent_id} votes {vote_entry['vote']} on Proposal #{vote_entry['proposal_number']}",
                            }
                        )
                else:
                    for enum_proposal in enumerated_proposals:
                        # Build proposal_for_voting in the correct format for the game
                        if self.game_environment is not None:
                            # Pass through original proposal with its native format
                            proposal_for_voting = enum_proposal.get("original_proposal", {}).copy()
                            proposal_for_voting["proposed_by"] = enum_proposal["proposer"]
                            proposal_for_voting["round"] = round_num
                        else:
                            proposal_for_voting = {
                                "allocation": enum_proposal["allocation"],
                                "proposed_by": enum_proposal["proposer"],
                                "reasoning": enum_proposal.get("reasoning", ""),
                                "round": round_num
                            }

                        # Generate voting prompt
                        if self.game_environment is not None:
                            if "game_state" in preferences:
                                game_state = preferences["game_state"]
                            else:
                                game_state = self._build_game_state(
                                    agents, items, preferences, round_num, max_rounds,
                                    agent.agent_id, "voting",
                                    conversation_history=list(public_context or []),
                                    proposals=[proposal_for_voting]
                                )
                            voting_prompt = self.game_environment.get_voting_prompt(
                                agent_id=agent.agent_id,
                                proposal=proposal_for_voting,
                                game_state=game_state,
                                round_num=round_num,
                                reasoning_token_budget=reasoning_budget
                            )
                        else:
                            # Legacy mode: use default voting prompt
                            voting_prompt = f"""A proposal has been made for item allocation:
PROPOSAL: {json.dumps(proposal_for_voting['allocation'], indent=2)}
REASONING: {proposal_for_voting.get('reasoning', 'No reasoning provided')}
PROPOSED BY: {proposal_for_voting.get('proposed_by', 'Unknown')}
Please vote on this proposal. Consider:
- How this allocation affects your utility
- Whether you might get a better deal by continuing negotiation
- The strategic implications of accepting vs. rejecting
Respond with ONLY a JSON object in this exact format:
{{
    "vote": "accept",
    "reasoning": "Brief explanation of your vote"
}}
Vote must be either "accept" or "reject"."""

                        # Collect vote using game-environment-aware path or legacy
                        if self.game_environment is not None:
                            # Use game-specific voting prompt directly with generate_response
                            response = await agent.generate_response(voting_context, voting_prompt)
                            token_usage = self._extract_token_usage(response)
                            try:
                                vote_result = self._parse_vote_response(response.content, agent.agent_id, round_num)
                            except (ValueError, json.JSONDecodeError) as e:
                                warnings.append(f"Failed to parse vote from {agent.agent_id}: {e}")
                                vote_result = {
                                    "vote": "reject",
                                    "reasoning": "Failed to parse vote response",
                                    "voter": agent.agent_id,
                                    "round": round_num,
                                }
                        else:
                            vote_result = await agent.vote_on_proposal(
                                voting_context,
                                proposal_for_voting
                            )
                            # Extract token usage if present
                            token_usage = vote_result.pop("_token_usage", None)

                        vote_entry = {
                            "voter_id": agent.agent_id,
                            "proposal_number": enum_proposal["proposal_number"],
                            "vote": vote_result.get("vote", "reject"),
                            "reasoning": vote_result.get("reasoning", "Strategic voting decision"),
                            "round": round_num,
                            "timestamp": time.time()
                        }
                        agent_votes.append(vote_entry)

                        # Create enhanced vote response with full context
                        # Use native proposal key (agreement or allocation)
                        if "agreement" in enum_proposal:
                            proposal_detail_key = "agreement"
                        elif "allocation" in enum_proposal:
                            proposal_detail_key = "allocation"
                        else:
                            proposal_detail_key = "contributions_by_agent"
                        enhanced_vote_response = {
                            "voter": agent.agent_id,
                            "proposal_number": enum_proposal["proposal_number"],
                            "proposal_by": enum_proposal["proposer"],
                            "vote_decision": vote_result.get("vote", "reject"),
                            "reasoning": vote_result.get("reasoning", "Strategic voting decision"),
                            "proposal_details": {
                                proposal_detail_key: enum_proposal.get(proposal_detail_key, {}),
                                "original_reasoning": enum_proposal.get("reasoning", "")
                            },
                            "round": round_num,
                            "timestamp": time.time()
                        }

                        interaction_records.append(
                            {
                                "phase": f"voting_round_{round_num}_proposal_{enum_proposal['proposal_number']}",
                                "prompt": voting_prompt,
                                "response": json.dumps(enhanced_vote_response, default=str),
                                "token_usage": token_usage,
                                "log_message": f"  [PRIVATE] {agent.agent_id} votes {vote_entry['vote']} on Proposal #{vote_entry['proposal_number']}",
                            }
                        )
                return {
                    "agent": agent,
                    "agent_votes": agent_votes,
                    "interaction_records": interaction_records,
                    "warnings": warnings,
                    "error": None,
                }
            except Exception as e:
                return {"agent": agent, "error": e}

        raw_results = await self._run_agent_tasks_in_order(agents, vote_for_agent)
        for result in raw_results:
            if isinstance(result, BaseException):
                raise result
            agent = result["agent"]
            self.logger.info(f"🗳️ Collecting private votes from {agent.agent_id}...")

            if result["error"] is not None:
                e = result["error"]
                self.logger.error(f"Error collecting private votes from {agent.agent_id}: {e}")
                for enum_proposal in enumerated_proposals:
                    fallback_vote = {
                        "voter_id": agent.agent_id,
                        "proposal_number": enum_proposal["proposal_number"],
                        "vote": "reject",
                        "reasoning": f"Unable to process vote due to error: {e}",
                        "round": round_num,
                        "timestamp": time.time()
                    }
                    private_votes.append(fallback_vote)
                continue

            for warning in result["warnings"]:
                self.logger.warning(warning)
            for record in result["interaction_records"]:
                self.logger.info(record["log_message"])
                self.save_interaction(
                    agent.agent_id,
                    record["phase"],
                    record["prompt"],
                    record["response"],
                    round_num,
                    record["token_usage"],
                    model_name=agent.get_model_info()["model_name"]
                )

            private_votes.extend(result["agent_votes"])
        
        voting_summary = {
            "total_agents": len(agents),
            "total_proposals": len(enumerated_proposals),
            "total_votes_cast": len(private_votes),
            "votes_by_proposal": {}
        }
        
        for vote in private_votes:
            prop_num = vote['proposal_number']
            if prop_num not in voting_summary["votes_by_proposal"]:
                voting_summary["votes_by_proposal"][prop_num] = {"accept": 0, "reject": 0, "votes": []}

            # Defensive: ensure vote is a string, not a dict
            vote_value = vote.get('vote', 'reject')
            if isinstance(vote_value, dict):
                self.logger.warning(f"Vote value is a dict, extracting: {vote_value}")
                vote_value = vote_value.get('decision', vote_value.get('vote', 'reject'))
            if not isinstance(vote_value, str) or vote_value not in ('accept', 'reject'):
                vote_value = 'reject'
            vote['vote'] = vote_value  # Update the vote dict with normalized value

            voting_summary["votes_by_proposal"][prop_num][vote_value] += 1
            voting_summary["votes_by_proposal"][prop_num]["votes"].append(vote)
        
        self.logger.info(f"✅ Private voting complete: {len(private_votes)} votes collected from {len(agents)} agents")
        
        return {
            "private_votes": private_votes,
            "voting_summary": voting_summary,
            "phase_complete": True
        }
    
    async def run_vote_tabulation_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                       preferences: Dict, round_num: int, max_rounds: int,
                                       private_votes: List[Dict], enumerated_proposals: List[Dict]) -> Dict:
        """Phase 5B: Vote Tabulation Phase"""
        messages = []
        
        self.logger.info(f"=== VOTE TABULATION PHASE - Round {round_num} ===")
        
        # Tabulate votes
        votes_by_proposal = {}
        for vote in private_votes:
            prop_num = vote['proposal_number']
            if prop_num not in votes_by_proposal:
                votes_by_proposal[prop_num] = {"accept": 0, "reject": 0}

            # Defensive: ensure vote is a string, not a dict
            vote_value = vote.get('vote', 'reject')
            if isinstance(vote_value, dict):
                self.logger.warning(f"Vote tabulation: vote value is a dict, extracting: {vote_value}")
                vote_value = vote_value.get('decision', vote_value.get('vote', 'reject'))
            if not isinstance(vote_value, str) or vote_value not in ('accept', 'reject'):
                vote_value = 'reject'

            votes_by_proposal[prop_num][vote_value] += 1
        
        # Check for unanimous acceptance
        consensus_reached = False
        final_utilities = {}
        final_allocation = {}
        agent_preferences = {}

        for prop_num, vote_counts in votes_by_proposal.items():
            if vote_counts["reject"] == 0 and vote_counts["accept"] == len(agents):
                consensus_reached = True
                # Find the winning proposal to calculate utilities
                for enum_prop in enumerated_proposals:
                    if enum_prop["proposal_number"] == prop_num:
                        if self.game_environment is not None:
                            # Game-environment-aware utility calculation
                            game_state = preferences.get("game_state", {
                                "items": items,
                                "agent_preferences": preferences.get("agent_preferences", {}),
                            })
                            winning_proposal = enum_prop.get("original_proposal", enum_prop)
                            self.game_environment.record_accepted_proposal(
                                game_state,
                                winning_proposal,
                            )
                            for agent in agents:
                                final_utilities[agent.agent_id] = self.game_environment.calculate_utility(
                                    agent.agent_id, winning_proposal, game_state, round_num
                                )
                                agent_preferences[agent.agent_id] = preferences["agent_preferences"][agent.agent_id]
                            # Store in native format (agreement vector or allocation dict)
                            final_allocation = winning_proposal.get(
                                "agreement",
                                winning_proposal.get(
                                    "allocation",
                                    winning_proposal.get("funded_projects", []),
                                )
                            )
                        else:
                            # Legacy item allocation utility calculation
                            allocation = enum_prop["allocation"]
                            final_allocation = allocation.copy()
                            for agent in agents:
                                agent_items = allocation.get(agent.agent_id, [])
                                agent_prefs = preferences["agent_preferences"][agent.agent_id]
                                utility = sum(agent_prefs[i]
                                            for i in agent_items
                                            if i < len(items) and i < len(agent_prefs))
                                final_utilities[agent.agent_id] = utility
                                agent_preferences[agent.agent_id] = agent_prefs
                        break
                break
        
        # Create tabulation message
        tabulation_lines = [f"📊 VOTE TABULATION - Round {round_num}", "=" * 60]
        for prop_num in sorted(votes_by_proposal.keys()):
            vote_counts = votes_by_proposal[prop_num]
            tabulation_lines.append(f"Proposal #{prop_num}: {vote_counts['accept']} accept, {vote_counts['reject']} reject")
        
        if consensus_reached:
            tabulation_lines.append(f"\n✅ CONSENSUS REACHED! Proposal #{prop_num} accepted unanimously!")
        else:
            tabulation_lines.append(f"\n❌ No proposal achieved unanimous acceptance.")
        
        tabulation_summary = "\n".join(tabulation_lines)
        
        self.logger.info(tabulation_summary)
        
        tabulation_message = {
            "phase": "vote_tabulation",
            "round": round_num,
            "from": "system",
            "content": tabulation_summary,
            "timestamp": time.time()
        }
        messages.append(tabulation_message)
        
        return {
            "messages": messages,
            "consensus_reached": consensus_reached,
            "final_utilities": final_utilities,
            "final_allocation": final_allocation,
            "agent_preferences": agent_preferences,
            "votes_by_proposal": votes_by_proposal
        }
    
    async def run_individual_reflection_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                             preferences: Dict, round_num: int, max_rounds: int,
                                             tabulation_result: Dict,
                                             public_context: Optional[List[Dict[str, Any]]] = None,
                                             private_context_by_agent: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Phase 6: Individual Reflection Phase"""
        reflections = []

        self.logger.info(f"=== INDIVIDUAL REFLECTION PHASE - Round {round_num} ===")

        # Apply token limits for reflection phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "reflection")

        async def reflect_for_agent(_idx: int, agent: BaseLLMAgent) -> Dict[str, Any]:
            # Get reasoning budget for this specific agent
            reasoning_budget = self._get_reasoning_budget("reflection", agent.agent_id)

            # Use GameEnvironment if available, otherwise use default prompt
            if self.game_environment is not None:
                # Get the original game_state if stored in preferences
                if "game_state" in preferences:
                    game_state = preferences["game_state"]
                else:
                    game_state = self._build_game_state(
                        agents, items, preferences, round_num, max_rounds,
                        agent.agent_id, "reflection",
                        conversation_history=list(public_context or [])
                    )
                reflection_prompt = self.game_environment.get_reflection_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    tabulation_result=tabulation_result,
                    reasoning_token_budget=reasoning_budget
                )
            else:
                reflection_prompt = f"""Reflect on the outcome of round {round_num}.
No proposal achieved unanimous acceptance.
Consider what adjustments might lead to consensus in future rounds."""

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=self._get_context_preferences(agent.agent_id, preferences),
                turn_type="reflection",
                conversation_history=list(public_context or []),
                strategic_notes=(private_context_by_agent or {}).get(agent.agent_id, [])
            )

            try:
                # Get response with token info
                agent_response = await agent.generate_response(context, reflection_prompt)
                return {
                    "agent": agent,
                    "reflection_prompt": reflection_prompt,
                    "reflection": agent_response.content,
                    "token_usage": self._extract_token_usage(agent_response),
                    "error": None,
                }
            except Exception as e:
                return {"agent": agent, "error": e}

        raw_results = await self._run_agent_tasks_in_order(agents, reflect_for_agent)
        for result in raw_results:
            if isinstance(result, BaseException):
                raise result
            agent = result["agent"]

            if result["error"] is not None:
                self.logger.error(f"Error in reflection for {agent.agent_id}: {result['error']}")
                continue

            reflection = result["reflection"]
            token_usage = result["token_usage"]

            reflections.append({
                "agent_id": agent.agent_id,
                "reflection": reflection,
                "round": round_num
            })

            self.logger.info(f"  💭 {agent.agent_id} reflection:")
            self.logger.info(f"    {reflection}")

            self.save_interaction(agent.agent_id, f"reflection_round_{round_num}",
                                result["reflection_prompt"], reflection, round_num, token_usage, model_name=agent.get_model_info()["model_name"])
        
        return {"reflections": reflections}

    # ---- Legacy co-funding (Talk-Pledge-Revise) phase handlers ----

    async def run_pledge_submission_phase(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict,
        round_num: int,
        max_rounds: int,
    ) -> Dict:
        """Pledge Submission Phase for co-funding games.

        Each agent submits a contribution vector (pledge) for projects.
        Uses game_environment.get_proposal_prompt() to generate the pledge prompt
        and game_environment.parse_proposal() / validate_proposal() to process responses.

        Args:
            agents: List of agent objects
            items: List of project dicts
            preferences: Dict with agent_preferences (valuations) and game_state
            round_num: Current round number
            max_rounds: Maximum rounds

        Returns:
            Dict with "pledges" (agent_id -> parsed pledge) and "messages"
        """
        pledges = {}
        messages = []

        self.logger.info(f"=== PLEDGE SUBMISSION PHASE - Round {round_num} ===")

        # Apply token limits for pledge/proposal phase if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "proposal")

        game_state = preferences["game_state"]

        for agent in agents:
            reasoning_budget = self._get_reasoning_budget("proposal", agent.agent_id)

            pledge_prompt = self.game_environment.get_proposal_prompt(
                agent_id=agent.agent_id,
                game_state=game_state,
                round_num=round_num,
                agents=[a.agent_id for a in agents],
                reasoning_token_budget=reasoning_budget,
            )

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="proposal",
            )

            # Get pledge response via generate_response (NOT propose_allocation)
            agent_response = await agent.generate_response(context, pledge_prompt)
            response_content = agent_response.content
            token_usage = self._extract_token_usage(agent_response)

            # Parse the pledge
            parsed = self.game_environment.parse_proposal(
                response=response_content,
                agent_id=agent.agent_id,
                game_state=game_state,
                agents=[a.agent_id for a in agents],
            )

            # Validate
            if not self.game_environment.validate_proposal(parsed, game_state):
                self.logger.warning(
                    f"  Invalid pledge from {agent.agent_id}, retrying once..."
                )
                # Retry with error feedback
                error_prompt = (
                    pledge_prompt
                    + "\n\n**ERROR: Your previous pledge was invalid "
                    "(negative values, over-budget, or wrong length). "
                    "Please resubmit a valid pledge.**"
                )
                agent_response = await agent.generate_response(context, error_prompt)
                response_content = agent_response.content
                token_usage = self._extract_token_usage(agent_response)
                parsed = self.game_environment.parse_proposal(
                    response=response_content,
                    agent_id=agent.agent_id,
                    game_state=game_state,
                    agents=[a.agent_id for a in agents],
                )
                if not self.game_environment.validate_proposal(parsed, game_state):
                    self.logger.error(
                        f"  {agent.agent_id} pledge still invalid after retry, using zero vector"
                    )
                    m = game_state["m_projects"]
                    parsed = {
                        "contributions": [0.0] * m,
                        "reasoning": "Fallback: zero contributions after validation failure",
                        "proposed_by": agent.agent_id,
                        "raw_response": response_content,
                        "validation_error": "Pledge invalid after retry",
                    }

            pledges[agent.agent_id] = parsed

            self.logger.info(f"  Pledge from {agent.agent_id}: {parsed['contributions']}")

            # Save interaction
            pledge_str = json.dumps(parsed, default=str)
            self.save_interaction(
                agent.agent_id,
                f"pledge_round_{round_num}",
                pledge_prompt,
                pledge_str,
                round_num,
                token_usage,
                model_name=agent.get_model_info()["model_name"],
            )

            message = {
                "phase": "pledge_submission",
                "round": round_num,
                "from": agent.agent_id,
                "content": f"Pledge: {parsed['contributions']} - {parsed.get('reasoning', '')}",
                "pledge": parsed,
                "timestamp": time.time(),
            }
            messages.append(message)

        return {"pledges": pledges, "messages": messages}

    async def run_feedback_phase(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict,
        round_num: int,
        max_rounds: int,
        pledges: Dict[str, Dict],
    ) -> Dict:
        """Feedback Phase for co-funding games.

        Updates game state with pledges (computes aggregates and funded set),
        then sends feedback to each agent showing aggregate totals.

        Args:
            agents: List of agent objects
            items: List of project dicts
            preferences: Dict with game_state
            round_num: Current round number
            max_rounds: Maximum rounds
            pledges: Dict mapping agent_id -> parsed pledge dict

        Returns:
            Dict with aggregate_totals, funded_projects, and messages
        """
        messages = []

        self.logger.info(f"=== FEEDBACK PHASE - Round {round_num} ===")

        game_state = preferences["game_state"]

        # Update game state with pledges
        self.game_environment.update_game_state_with_pledges(game_state, pledges)

        aggregates = game_state["aggregate_totals"]
        funded = game_state["funded_projects"]
        projects = game_state["projects"]

        self.logger.info(f"  Aggregate totals: {[round(a, 2) for a in aggregates]}")
        self.logger.info(f"  Funded projects: {[projects[j]['name'] for j in funded] if funded else 'None'}")

        # Apply default-phase token limits if configured for this experiment or agent.
        self._apply_phase_token_limits(agents, "default")

        # Send feedback to each agent
        for agent in agents:
            feedback_prompt = self.game_environment.get_feedback_prompt(
                agent_id=agent.agent_id,
                game_state=game_state,
            )

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="feedback",
            )

            agent_response = await agent.generate_response(context, feedback_prompt)
            response_content = agent_response.content
            token_usage = self._extract_token_usage(agent_response)

            self.logger.info(f"  {agent.agent_id} feedback response: {response_content[:200]}...")

            self.save_interaction(
                agent.agent_id,
                f"feedback_round_{round_num}",
                feedback_prompt,
                response_content,
                round_num,
                token_usage,
                model_name=agent.get_model_info()["model_name"],
            )

            message = {
                "phase": "feedback",
                "round": round_num,
                "from": agent.agent_id,
                "content": response_content,
                "timestamp": time.time(),
            }
            messages.append(message)

        return {
            "aggregate_totals": aggregates,
            "funded_projects": funded,
            "messages": messages,
        }

    async def run_cofunding_commit_vote_phase(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict,
        round_num: int,
        max_rounds: int,
    ) -> Dict:
        """Post-pledge commit vote for co-funding.

        Each agent votes yay/nay on whether to lock in the current pledge profile.
        If all agents vote yay, the orchestrator can terminate early.
        """
        messages = []
        commit_votes = []

        self.logger.info(f"=== CO-FUNDING COMMIT VOTE PHASE - Round {round_num} ===")

        game_state = preferences["game_state"]
        projects = game_state.get("projects", [])
        costs = game_state.get("project_costs", [])
        aggregates = game_state.get("aggregate_totals", [])
        funded = game_state.get("funded_projects", [])
        current_pledges = game_state.get("current_pledges", {})

        self._apply_phase_token_limits(agents, "voting", fallback_phase="default")

        # Shared fallback prompt if the game environment does not provide one.
        fallback_lines = [
            f"POST-PLEDGE COMMIT VOTE - Round {round_num}/{max_rounds}",
            "",
            "Aggregate project status:",
        ]
        for j, proj in enumerate(projects):
            cost = costs[j] if j < len(costs) else 0.0
            agg = aggregates[j] if j < len(aggregates) else 0.0
            status = "PROVISIONALLY FUNDED" if j in funded else f"needs {max(0.0, cost - agg):.2f} more"
            fallback_lines.append(
                f"- {proj.get('name', f'Project_{j+1}')}: aggregate={agg:.2f} / cost={cost:.2f} ({status})"
            )
        fallback_lines.extend(
            [
                "",
                "Vote YAY if you are willing to lock in the current aggregate project status now.",
                "Vote NAY if you prefer another revision round.",
                "",
                "Respond with ONLY JSON:",
                "{",
                '  "commit_vote": "yay",',
                '  "reasoning": "brief explanation"',
                "}",
            ]
        )
        fallback_prompt = "\n".join(fallback_lines)

        for agent in agents:
            if (
                self.game_environment is not None
                and hasattr(self.game_environment, "get_commit_vote_prompt")
            ):
                commit_prompt = self.game_environment.get_commit_vote_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state,
                    round_num=round_num,
                    max_rounds=max_rounds,
                )
            else:
                commit_prompt = fallback_prompt

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="voting",
            )

            try:
                agent_response = await agent.generate_response(context, commit_prompt)
                response_content = agent_response.content
                token_usage = self._extract_token_usage(agent_response)
                parsed_vote = self._parse_commit_vote_response(
                    response_content, agent.agent_id, round_num
                )
            except Exception as e:
                self.logger.error(f"Error collecting commit vote from {agent.agent_id}: {e}")
                response_content = json.dumps({"commit_vote": "nay", "reasoning": f"error: {e}"})
                token_usage = None
                parsed_vote = {
                    "commit_vote": "nay",
                    "reasoning": f"fallback due to error: {e}",
                    "voter": agent.agent_id,
                    "round": round_num,
                }

            commit_votes.append(parsed_vote)

            self.logger.info(
                f"  Commit vote from {agent.agent_id}: "
                f"{parsed_vote.get('commit_vote', 'nay')}"
            )

            self.save_interaction(
                agent.agent_id,
                f"commit_vote_round_{round_num}",
                commit_prompt,
                response_content,
                round_num,
                token_usage,
                model_name=agent.get_model_info()["model_name"],
            )

            messages.append(
                {
                    "phase": "cofunding_commit_vote",
                    "round": round_num,
                    "from": agent.agent_id,
                    "content": response_content,
                    "parsed_vote": parsed_vote,
                    "timestamp": time.time(),
                }
            )

        yay_count = sum(1 for v in commit_votes if v.get("commit_vote") == "yay")
        unanimous_yay = yay_count == len(agents) and len(agents) > 0

        summary_message = {
            "phase": "cofunding_commit_vote_summary",
            "round": round_num,
            "from": "system",
            "content": (
                f"Commit vote summary: {yay_count} yay / {len(agents) - yay_count} nay. "
                f"{'UNANIMOUS_YAY' if unanimous_yay else 'NO_UNANIMOUS_COMMIT'}"
            ),
            "unanimous_yay": unanimous_yay,
            "timestamp": time.time(),
        }
        messages.append(summary_message)

        return {
            "commit_votes": commit_votes,
            "yay_count": yay_count,
            "nay_count": len(agents) - yay_count,
            "unanimous_yay": unanimous_yay,
            "messages": messages,
        }
