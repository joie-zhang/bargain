"""Handlers for different negotiation phases."""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
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
                 reasoning_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.prompt_gen = PromptGenerator()
        self.save_interaction = save_interaction_callback or (lambda *args: None)

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
    
    def _extract_token_usage(self, agent_response) -> Optional[Dict[str, Any]]:
        """Extract token usage information from an AgentResponse object."""
        if not agent_response:
            return None

        token_usage = None
        if agent_response.metadata and agent_response.metadata.get("usage"):
            usage = agent_response.metadata["usage"]
            token_usage = {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens")
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
            "preferences": preferences.get("agent_preferences", {}).get(agent_id, []),
            "all_preferences": preferences.get("agent_preferences", {}),
            "phase": phase,
            "conversation_history": conversation_history or [],
            "proposals": proposals or [],
        }

    async def run_game_setup_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                  preferences: Dict, config: Dict) -> None:
        """Phase 1A: Game Setup Phase"""
        self.logger.info("=== GAME SETUP PHASE ===")

        # Set token limit for setup phase if specified
        if self.token_config["default"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["default"])

        # Use GameEnvironment if available, otherwise fall back to PromptGenerator
        if self.game_environment is not None:
            # Get the original game_state if stored in preferences
            if "game_state" in preferences:
                game_state = preferences["game_state"]
            else:
                game_state = {"items": items, "agent_preferences": preferences.get("agent_preferences", {})}
            game_rules_prompt = self.game_environment.get_game_rules_prompt(game_state)
        else:
            game_rules_prompt = self.prompt_gen.create_game_rules_prompt(items, len(agents), config)
        
        self.logger.info("📜 GAME RULES PROMPT:")
        self.logger.info(f"  {game_rules_prompt}")
        
        for agent in agents:
            context = NegotiationContext(
                current_round=0,
                max_rounds=config["t_rounds"],
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="setup"
            )
            
            # Get response with token info
            agent_response = await agent.generate_response(context, game_rules_prompt)
            response_content = agent_response.content
            token_usage = self._extract_token_usage(agent_response)
            
            self.logger.info(f"  📬 {agent.agent_id} response:")
            self.logger.info(f"    {response_content}")
            self.save_interaction(agent.agent_id, "game_setup", game_rules_prompt, response_content, 0, token_usage)
        
        self.logger.info("Game setup phase completed - all agents briefed on rules")
    
    async def run_private_preference_assignment(self, agents: List[BaseLLMAgent], items: List[Dict],
                                               preferences: Dict, config: Dict) -> None:
        """Phase 1B: Private Preference Assignment"""
        self.logger.info("=== PRIVATE PREFERENCE ASSIGNMENT ===")

        # Set token limit for preference assignment if specified
        if self.token_config["default"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["default"])

        for agent in agents:
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
            
            self.logger.info(f"  🎯 {agent.agent_id} preferences:")
            for i, (item, value) in enumerate(zip(items, agent_preferences)):
                item_name = item["name"] if isinstance(item, dict) else str(item)
                self.logger.info(f"    - {item_name}: {value:.1f}")
            
            context = NegotiationContext(
                current_round=0,
                max_rounds=config["t_rounds"],
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=agent_preferences,
                turn_type="preference_assignment"
            )
            
            # Get response with token info
            agent_response = await agent.generate_response(context, preference_prompt)
            response_content = agent_response.content
            token_usage = self._extract_token_usage(agent_response)
            
            self.logger.info(f"  📬 {agent.agent_id} acknowledgment:")
            self.logger.info(f"    {response_content}")
            self.save_interaction(agent.agent_id, "preference_assignment", preference_prompt, response_content, 0, token_usage)
        
        self.logger.info("Private preference assignment completed")
    
    async def run_discussion_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                  preferences: Dict, round_num: int, max_rounds: int,
                                  discussion_turns: int = 3) -> Dict:
        """Phase 2: Public Discussion Phase

        When a GameEnvironment is provided, uses its get_*_prompt() methods
        for game-specific discussion prompts. Otherwise falls back to PromptGenerator.

        Args:
            agents: List of agent objects
            items: List of items being negotiated
            preferences: Dictionary containing agent preferences
            round_num: Current round number
            max_rounds: Maximum number of rounds
            discussion_turns: Number of times agents go around the circle discussing (default: 3)
        """
        messages = []

        self.logger.info(f"=== PUBLIC DISCUSSION PHASE - Round {round_num} ===")
        self.logger.info(f"  Discussion turns: {discussion_turns}")

        # Set token limit for discussion phase if specified
        if self.token_config["discussion"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["discussion"])

        # Outer loop: go around the circle discussion_turns times
        for turn in range(discussion_turns):
            self.logger.info(f"  --- Discussion Turn {turn + 1}/{discussion_turns} ---")

            for i, agent in enumerate(agents):
                context = NegotiationContext(
                    current_round=round_num,
                    max_rounds=max_rounds,
                    items=items,
                    agents=[a.agent_id for a in agents],
                    agent_id=agent.agent_id,
                    preferences=preferences["agent_preferences"][agent.agent_id],
                    turn_type="discussion"
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
                            conversation_history=messages
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
                                    full_discussion_prompt, response_content, round_num, token_usage)

        self.logger.info(f"Discussion phase completed - {len(messages)} messages exchanged across {discussion_turns} turns")
        return {"messages": messages}
    
    async def run_private_thinking_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                        preferences: Dict, round_num: int, max_rounds: int,
                                        discussion_messages: List[Dict]) -> Dict:
        """Phase 3: Private Thinking Phase"""
        thinking_results = []
        
        self.logger.info(f"=== PRIVATE THINKING PHASE - Round {round_num} ===")
        
        # Set token limit for thinking phase if specified
        if self.token_config["thinking"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["thinking"])

        for agent in agents:
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
                        conversation_history=discussion_messages
                    )
                thinking_prompt = self.game_environment.get_thinking_prompt(
                    agent_id=agent.agent_id,
                    game_state=game_state,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    discussion_history=[msg.get("content", "") for msg in discussion_messages] if discussion_messages else [],
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
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="thinking",
                conversation_history=discussion_messages
            )

            try:
                thinking_response = await agent.think_strategy(thinking_prompt, context)

                # Extract token usage if present (remove from response to avoid saving it in the content)
                token_usage = thinking_response.pop("_token_usage", None)

                self.logger.info(f"🧠 [PRIVATE] {agent.agent_id} strategic thinking:")
                self.logger.info(f"  Full reasoning: {thinking_response.get('reasoning', 'No reasoning provided')}")
                self.logger.info(f"  Strategy: {thinking_response.get('strategy', 'No strategy provided')}")
                # Log priorities/targets based on game type
                priorities = thinking_response.get('key_priorities') or thinking_response.get('target_items', [])
                self.logger.info(f"  Key priorities: {priorities}")
                
                thinking_response_str = json.dumps(thinking_response, default=str)
                self.save_interaction(agent.agent_id, f"private_thinking_round_{round_num}", 
                                    thinking_prompt, thinking_response_str, round_num, token_usage)
                
                thinking_results.append({
                    "agent_id": agent.agent_id,
                    "reasoning": thinking_response.get('reasoning', ''),
                    "strategy": thinking_response.get('strategy', ''),
                    "target_items": thinking_response.get('target_items', []),
                    "anticipated_resistance": thinking_response.get('anticipated_resistance', [])
                })
                
            except Exception as e:
                self.logger.error(f"Error in private thinking for {agent.agent_id}: {e}")
                thinking_results.append({
                    "agent_id": agent.agent_id,
                    "reasoning": "Unable to complete strategic thinking due to error",
                    "strategy": "Will propose based on known preferences",
                    "target_items": [],
                    "anticipated_resistance": []
                })
        
        return {
            "thinking_results": thinking_results,
            "round_num": round_num
        }
    
    async def run_proposal_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                preferences: Dict, round_num: int, max_rounds: int) -> Dict:
        """Phase 4A: Proposal Submission Phase

        When a GameEnvironment is provided, uses its get_*_prompt() methods
        for game-specific proposal prompts. Otherwise uses agent's default behavior.
        """
        messages = []
        proposals = []

        self.logger.info(f"=== PROPOSAL SUBMISSION PHASE - Round {round_num} ===")

        # Set token limit for proposal phase if specified
        if self.token_config["proposal"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["proposal"])

        for agent in agents:
            # Get reasoning budget for this specific agent
            reasoning_budget = self._get_reasoning_budget("proposal", agent.agent_id)

            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="proposal"
            )

            # Generate proposal prompt (for logging and potential custom prompts)
            if self.game_environment is not None:
                # Get the original game_state if stored in preferences
                if "game_state" in preferences:
                    game_state = preferences["game_state"]
                else:
                    game_state = self._build_game_state(
                        agents, items, preferences, round_num, max_rounds,
                        agent.agent_id, "proposal"
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

            # Get proposal from agent (agent handles the actual LLM call)
            proposal = await agent.propose_allocation(context)
            
            # Extract token usage if present (remove from proposal to avoid saving it in the content)
            token_usage = proposal.pop("_token_usage", None)
            
            proposals.append(proposal)
            
            proposal_str = json.dumps(proposal, default=str)
            self.save_interaction(agent.agent_id, f"proposal_round_{round_num}", 
                                proposal_prompt, proposal_str, round_num, token_usage)
            
            message = {
                "phase": "proposal",
                "round": round_num,
                "from": agent.agent_id,
                "content": f"I propose this allocation: {proposal['allocation']} - {proposal.get('reasoning', 'No reasoning provided')}",
                "proposal": proposal,
                "timestamp": time.time(),
                "agent_id": agent.agent_id
            }
            messages.append(message)
            
            self.logger.info(f"📋 {agent.agent_id} FORMAL PROPOSAL:")
            self.logger.info(f"   Allocation: {proposal['allocation']}")
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
        
        enumerated_proposals = []
        proposal_display_lines = []
        
        proposal_display_lines.append(f"📋 FORMAL PROPOSALS SUBMITTED - Round {round_num}")
        proposal_display_lines.append("=" * 60)
        proposal_display_lines.append(f"Total Proposals: {len(proposals)}")
        proposal_display_lines.append("")
        
        for i, proposal in enumerate(proposals, 1):
            proposal_num = i
            proposer = proposal.get('proposed_by', f'Agent_{i}')
            allocation = proposal.get('allocation', {})
            reasoning = proposal.get('reasoning', 'No reasoning provided')
            
            enumerated_proposal = {
                "proposal_number": proposal_num,
                "proposer": proposer,
                "allocation": allocation,
                "reasoning": reasoning,
                "original_proposal": proposal
            }
            enumerated_proposals.append(enumerated_proposal)
            
            proposal_display_lines.append(f"PROPOSAL #{proposal_num} (by {proposer}):")
            proposal_display_lines.append(f"  Allocation:")
            
            for agent_id, item_indices in allocation.items():
                if item_indices:
                    item_names = []
                    for idx in item_indices:
                        # Convert idx to int if it's a string (can happen when parsing JSON)
                        try:
                            idx_int = int(idx)
                            if 0 <= idx_int < len(items):
                                item_names.append(f"{idx_int}:{items[idx_int]['name']}")
                            else:
                                item_names.append(f"{idx}:unknown")
                        except (ValueError, TypeError):
                            # If conversion fails, just use the original value
                            item_names.append(f"{idx}:unknown")
                    proposal_display_lines.append(f"    → {agent_id}: {', '.join(item_names)}")
                else:
                    proposal_display_lines.append(f"    → {agent_id}: (no items)")
            
            proposal_display_lines.append(f"  Reasoning: {reasoning}")
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
            "total_proposals": len(proposals)
        }
    
    async def run_private_voting_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                      preferences: Dict, round_num: int, max_rounds: int,
                                      proposals: List[Dict], enumerated_proposals: List[Dict]) -> Dict:
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

        # Set token limit for voting phase if specified
        if self.token_config["voting"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["voting"])

        for agent in agents:
            # Get reasoning budget for this specific agent
            reasoning_budget = self._get_reasoning_budget("voting", agent.agent_id)

            agent_votes = []

            self.logger.info(f"🗳️ Collecting private votes from {agent.agent_id}...")

            voting_context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="private_voting",
                current_proposals=proposals
            )

            try:
                for enum_proposal in enumerated_proposals:
                    proposal_for_voting = {
                        "allocation": enum_proposal["allocation"],
                        "proposed_by": enum_proposal["proposer"],
                        "reasoning": enum_proposal.get("reasoning", ""),
                        "round": round_num
                    }

                    # Generate voting prompt (for logging)
                    if self.game_environment is not None:
                        # Get the original game_state if stored in preferences
                        if "game_state" in preferences:
                            game_state = preferences["game_state"]
                        else:
                            game_state = self._build_game_state(
                                agents, items, preferences, round_num, max_rounds,
                                agent.agent_id, "voting",
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
                    
                    vote_result = await agent.vote_on_proposal(
                        voting_context,
                        proposal_for_voting
                    )
                    
                    # Extract token usage if present (remove from vote_result to avoid saving it in the content)
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
                    
                    self.logger.info(f"  [PRIVATE] {agent.agent_id} votes {vote_entry['vote']} on Proposal #{vote_entry['proposal_number']}")
                    
                    # Create enhanced vote response with full context
                    enhanced_vote_response = {
                        "voter": agent.agent_id,
                        "proposal_number": enum_proposal["proposal_number"],
                        "proposal_by": enum_proposal["proposer"],
                        "vote_decision": vote_result.get("vote", "reject"),
                        "reasoning": vote_result.get("reasoning", "Strategic voting decision"),
                        "proposal_details": {
                            "allocation": enum_proposal["allocation"],
                            "original_reasoning": enum_proposal.get("reasoning", "")
                        },
                        "round": round_num,
                        "timestamp": time.time()
                    }
                    
                    # Save the voting interaction with enhanced context
                    vote_response_str = json.dumps(enhanced_vote_response, default=str)
                    self.save_interaction(
                        agent.agent_id, 
                        f"voting_round_{round_num}_proposal_{enum_proposal['proposal_number']}", 
                        voting_prompt, 
                        vote_response_str, 
                        round_num,
                        token_usage
                    )
                
                private_votes.extend(agent_votes)
                
            except Exception as e:
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
                        # Save the allocation
                        allocation = enum_prop["allocation"]
                        final_allocation = allocation.copy()

                        # Calculate utilities and save preferences
                        for agent in agents:
                            agent_items = allocation.get(agent.agent_id, [])
                            agent_prefs = preferences["agent_preferences"][agent.agent_id]
                            # Ensure we don't access preferences out of bounds
                            utility = sum(agent_prefs[i]
                                        for i in agent_items 
                                        if i < len(items) and i < len(agent_prefs))
                            final_utilities[agent.agent_id] = utility
                            # Save the preference vector for this agent
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
                                             tabulation_result: Dict) -> Dict:
        """Phase 6: Individual Reflection Phase"""
        reflections = []

        self.logger.info(f"=== INDIVIDUAL REFLECTION PHASE - Round {round_num} ===")

        # Set token limit for reflection phase if specified
        if self.token_config["reflection"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["reflection"])

        for agent in agents:
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
                        agent.agent_id, "reflection"
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
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="reflection"
            )
            
            try:
                # Get response with token info
                agent_response = await agent.generate_response(context, reflection_prompt)
                reflection = agent_response.content
                token_usage = self._extract_token_usage(agent_response)
                
                reflections.append({
                    "agent_id": agent.agent_id,
                    "reflection": reflection,
                    "round": round_num
                })
                
                self.logger.info(f"  💭 {agent.agent_id} reflection:")
                self.logger.info(f"    {reflection}")
                
                self.save_interaction(agent.agent_id, f"reflection_round_{round_num}", 
                                    reflection_prompt, reflection, round_num, token_usage)
            except Exception as e:
                self.logger.error(f"Error in reflection for {agent.agent_id}: {e}")
        
        return {"reflections": reflections}