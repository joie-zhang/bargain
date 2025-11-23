"""Handlers for different negotiation phases."""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from negotiation import NegotiationContext
from negotiation.llm_agents import BaseLLMAgent
from ..prompts import PromptGenerator


class PhaseHandler:
    """Handles execution of different negotiation phases."""
    
    def __init__(self, save_interaction_callback=None, token_config=None):
        self.logger = logging.getLogger(__name__)
        self.prompt_gen = PromptGenerator()
        self.save_interaction = save_interaction_callback or (lambda *args: None)
        
        # Token limits for different phases (None means unlimited)
        self.token_config = token_config or {
            "discussion": None,
            "proposal": None,
            "voting": None,
            "reflection": None,
            "thinking": None,
            "default": None
        }
    
    async def run_game_setup_phase(self, agents: List[BaseLLMAgent], items: List[Dict], 
                                  preferences: Dict, config: Dict) -> None:
        """Phase 1A: Game Setup Phase"""
        self.logger.info("=== GAME SETUP PHASE ===")
        
        # Set token limit for setup phase if specified
        if self.token_config["default"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["default"])
        
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
            
            response = await agent.discuss(context, game_rules_prompt)
            
            self.logger.info(f"  📬 {agent.agent_id} response:")
            self.logger.info(f"    {response}")
            self.save_interaction(agent.agent_id, "game_setup", game_rules_prompt, response, 0)
        
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
            
            response = await agent.discuss(context, preference_prompt)
            
            self.logger.info(f"  📬 {agent.agent_id} acknowledgment:")
            self.logger.info(f"    {response}")
            self.save_interaction(agent.agent_id, "preference_assignment", preference_prompt, response, 0)
        
        self.logger.info("Private preference assignment completed")
    
    async def run_discussion_phase(self, agents: List[BaseLLMAgent], items: List[Dict],
                                  preferences: Dict, round_num: int, max_rounds: int) -> Dict:
        """Phase 2: Public Discussion Phase"""
        messages = []
        
        if round_num == 1:
            discussion_prompt = self.prompt_gen.create_initial_discussion_prompt(items, round_num, max_rounds)
        else:
            discussion_prompt = self.prompt_gen.create_ongoing_discussion_prompt(items, round_num, max_rounds)
        
        self.logger.info(f"=== PUBLIC DISCUSSION PHASE - Round {round_num} ===")
        
        # Set token limit for discussion phase if specified
        if self.token_config["discussion"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["discussion"])
        
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
            
            current_discussion_history = [msg["content"] for msg in messages]
            full_discussion_prompt = self.prompt_gen.create_contextual_discussion_prompt(
                discussion_prompt, agent.agent_id, current_discussion_history, i + 1, len(agents)
            )
            
            response = await agent.discuss(context, full_discussion_prompt)
            
            message = {
                "phase": "discussion",
                "round": round_num,
                "from": agent.agent_id,
                "content": response,
                "timestamp": time.time(),
                "speaker_order": i + 1,
                "total_speakers": len(agents)
            }
            messages.append(message)
            
            self.logger.info(f"  💬 Speaker {i+1}/{len(agents)} - {agent.agent_id}:")
            self.logger.info(f"    {response}")
            
            self.save_interaction(agent.agent_id, f"discussion_round_{round_num}", 
                                full_discussion_prompt, response, round_num)
        
        self.logger.info(f"Discussion phase completed - {len(messages)} messages exchanged")
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
                
                self.logger.info(f"🧠 [PRIVATE] {agent.agent_id} strategic thinking:")
                self.logger.info(f"  Full reasoning: {thinking_response.get('reasoning', 'No reasoning provided')}")
                self.logger.info(f"  Strategy: {thinking_response.get('strategy', 'No strategy provided')}")
                self.logger.info(f"  Target items: {thinking_response.get('target_items', [])}")
                
                thinking_response_str = json.dumps(thinking_response, default=str)
                self.save_interaction(agent.agent_id, f"private_thinking_round_{round_num}", 
                                    thinking_prompt, thinking_response_str, round_num)
                
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
        """Phase 4A: Proposal Submission Phase"""
        messages = []
        proposals = []
        
        self.logger.info(f"=== PROPOSAL SUBMISSION PHASE - Round {round_num} ===")
        
        # Set token limit for proposal phase if specified
        if self.token_config["proposal"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["proposal"])
        
        for agent in agents:
            context = NegotiationContext(
                current_round=round_num,
                max_rounds=max_rounds,
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=preferences["agent_preferences"][agent.agent_id],
                turn_type="proposal"
            )
            
            proposal_prompt = f"Please propose an allocation for round {round_num}."
            
            proposal = await agent.propose_allocation(context)
            proposals.append(proposal)
            
            proposal_str = json.dumps(proposal, default=str)
            self.save_interaction(agent.agent_id, f"proposal_round_{round_num}", 
                                proposal_prompt, proposal_str, round_num)
            
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
                        resolved_idx = None
                        
                        # Try to convert to integer first
                        try:
                            idx_int = int(idx) if isinstance(idx, str) else idx
                            if isinstance(idx_int, int):
                                if 0 <= idx_int < len(items):
                                    resolved_idx = idx_int
                                else:
                                    raise ValueError(
                                        f"Index {idx_int} is out of bounds (valid range: 0-{len(items)-1}) "
                                        f"in proposal by {proposer}"
                                    )
                        except (ValueError, TypeError):
                            # Conversion failed - try matching against item names
                            if isinstance(idx, str):
                                # Try to find matching item name (case-insensitive)
                                for i, item in enumerate(items):
                                    item_name = item.get('name', '')
                                    if item_name.lower() == idx.lower():
                                        resolved_idx = i
                                        break
                                
                                if resolved_idx is None:
                                    # No match found - raise error with helpful message
                                    available_items = [item.get('name', f'Item {i}') for i, item in enumerate(items)]
                                    raise ValueError(
                                        f"Invalid item identifier '{idx}' in proposal by {proposer}. "
                                        f"Expected an integer index (0-{len(items)-1}) or an item name. "
                                        f"Available items: {available_items}"
                                    )
                            else:
                                # Not a string and not convertible to int - raise error
                                raise ValueError(
                                    f"Invalid item identifier '{idx}' (type: {type(idx).__name__}) in proposal by {proposer}. "
                                    f"Expected an integer index (0-{len(items)-1}) or an item name string."
                                )
                        
                        # Use the resolved index
                        if resolved_idx is not None:
                            item_names.append(f"{resolved_idx}:{items[resolved_idx]['name']}")
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
        """Phase 5A: Private Voting Phase"""
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
                    
                    # Create the voting prompt for logging (matching what's in vote_on_proposal)
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
                        round_num
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
            
            voting_summary["votes_by_proposal"][prop_num][vote['vote']] += 1
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
            votes_by_proposal[prop_num][vote['vote']] += 1
        
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
                            utility = sum(preferences["agent_preferences"][agent.agent_id][i]
                                        for i in agent_items if i < len(items))
                            final_utilities[agent.agent_id] = utility
                            # Save the preference vector for this agent
                            agent_preferences[agent.agent_id] = preferences["agent_preferences"][agent.agent_id]
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
        
        reflection_prompt = f"""Reflect on the outcome of round {round_num}.
        No proposal achieved unanimous acceptance.
        Consider what adjustments might lead to consensus in future rounds."""
        
        # Set token limit for reflection phase if specified
        if self.token_config["reflection"] is not None:
            for agent in agents:
                agent.update_max_tokens(self.token_config["reflection"])
        
        for agent in agents:
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
                reflection = await agent.discuss(context, reflection_prompt)
                reflections.append({
                    "agent_id": agent.agent_id,
                    "reflection": reflection,
                    "round": round_num
                })
                
                self.logger.info(f"  💭 {agent.agent_id} reflection:")
                self.logger.info(f"    {reflection}")
                
                self.save_interaction(agent.agent_id, f"reflection_round_{round_num}", 
                                    reflection_prompt, reflection, round_num)
            except Exception as e:
                self.logger.error(f"Error in reflection for {agent.agent_id}: {e}")
        
        return {"reflections": reflections}