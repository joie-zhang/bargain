"""
Modular implementation of negotiation experiment phases.

This module contains all the individual phase implementations for the 14-phase
negotiation protocol, making them reusable and easily debuggable.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .llm_agents import BaseLLMAgent
from . import NegotiationContext


class NegotiationPhases:
    """Handles all negotiation phases in a modular way."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the phases handler."""
        self.logger = logger or logging.getLogger(__name__)
    
    async def run_game_setup(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        config: Dict[str, Any],
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 1A: Game Setup Phase
        Give each agent identical opening prompt explaining game rules.
        """
        self.logger.info("=== GAME SETUP PHASE ===")
        
        game_rules_prompt = self._create_game_rules_prompt(items, len(agents), config)
        
        self.logger.info("📜 GAME RULES PROMPT:")
        self.logger.info(f"  {game_rules_prompt}")
        
        responses = []
        for agent in agents:
            context = NegotiationContext(
                current_round=0,
                max_rounds=config["t_rounds"],
                items=items,
                agents=[a.agent_id for a in agents],
                agent_id=agent.agent_id,
                preferences=[],  # Not assigned yet
                turn_type="setup"
            )
            
            response = await agent.discuss(context, game_rules_prompt)
            
            self.logger.info(f"  📬 {agent.agent_id} response:")
            self.logger.info(f"    {response}")
            
            if save_callback:
                save_callback(agent.agent_id, "game_setup", game_rules_prompt, response, 0)
            
            responses.append({
                "agent_id": agent.agent_id,
                "response": response
            })
        
        self.logger.info("Game setup phase completed - all agents briefed on rules")
        return {"responses": responses}
    
    async def run_preference_assignment(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        config: Dict[str, Any],
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 1B: Private Preference Assignment
        Assign each agent their individual secret preferences.
        """
        self.logger.info("=== PRIVATE PREFERENCE ASSIGNMENT ===")
        
        responses = []
        for agent in agents:
            agent_preferences = preferences["agent_preferences"][agent.agent_id]
            preference_prompt = self._create_preference_assignment_prompt(
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
            
            if save_callback:
                save_callback(agent.agent_id, "preference_assignment", preference_prompt, response, 0)
            
            responses.append({
                "agent_id": agent.agent_id,
                "response": response
            })
        
        self.logger.info("Private preference assignment completed")
        return {"responses": responses}
    
    async def run_discussion(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 2: Public Discussion Phase
        Agents engage in open discussion about preferences.
        """
        messages = []
        
        if round_num == 1:
            discussion_prompt = self._create_initial_discussion_prompt(items, round_num, max_rounds)
        else:
            discussion_prompt = self._create_ongoing_discussion_prompt(items, round_num, max_rounds)
        
        self.logger.info(f"=== PUBLIC DISCUSSION PHASE - Round {round_num} ===")
        
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
            full_discussion_prompt = self._create_contextual_discussion_prompt(
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
            
            if save_callback:
                save_callback(agent.agent_id, f"discussion_round_{round_num}", 
                            full_discussion_prompt, response, round_num)
        
        self.logger.info(f"Discussion phase completed - {len(messages)} messages exchanged")
        return {"messages": messages}
    
    async def run_private_thinking(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_messages: List[Dict],
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 3: Private Thinking Phase
        Each agent uses private scratchpad to plan their proposal strategy.
        """
        thinking_results = []
        
        self.logger.info(f"=== PRIVATE THINKING PHASE - Round {round_num} ===")
        
        for agent in agents:
            thinking_prompt = self._create_thinking_prompt(items, round_num, max_rounds, discussion_messages)
            
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
                
                if save_callback:
                    thinking_response_str = json.dumps(thinking_response, default=str)
                    save_callback(agent.agent_id, f"private_thinking_round_{round_num}", 
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
    
    async def run_proposals(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 4A: Proposal Submission Phase
        Agents submit formal proposals based on their private thinking.
        """
        messages = []
        proposals = []
        
        self.logger.info(f"=== PROPOSAL SUBMISSION PHASE - Round {round_num} ===")
        
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
            
            if save_callback:
                proposal_str = json.dumps(proposal, default=str)
                save_callback(agent.agent_id, f"proposal_round_{round_num}", 
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
    
    async def run_proposal_enumeration(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        round_num: int,
        proposals: List[Dict]
    ) -> Dict[str, Any]:
        """
        Phase 4B: Proposal Enumeration Phase
        Number and display all proposals for easy reference.
        """
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
                        if 0 <= idx < len(items):
                            item_names.append(f"{idx}:{items[idx]['name']}")
                        else:
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
    
    async def run_private_voting(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        proposals: List[Dict],
        enumerated_proposals: List[Dict],
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 5A: Private Voting Phase
        Agents submit votes privately on enumerated proposals.
        """
        private_votes = []
        
        self.logger.info(f"=== PRIVATE VOTING PHASE - Round {round_num} ===")
        
        if not enumerated_proposals:
            self.logger.warning("No enumerated proposals available for voting!")
            return {
                "private_votes": [],
                "voting_summary": "No proposals to vote on"
            }
        
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
    
    async def run_vote_tabulation(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        round_num: int,
        private_votes: List[Dict],
        enumerated_proposals: List[Dict]
    ) -> Dict[str, Any]:
        """
        Phase 5B: Vote Tabulation Phase
        Tabulate votes and check for consensus.
        """
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
        winner_agent_id = None
        final_utilities = {}
        
        for prop_num, vote_counts in votes_by_proposal.items():
            if vote_counts["reject"] == 0 and vote_counts["accept"] == len(agents):
                consensus_reached = True
                # Find the winning proposal
                for enum_prop in enumerated_proposals:
                    if enum_prop["proposal_number"] == prop_num:
                        winner_agent_id = enum_prop["proposer"]
                        # Calculate utilities
                        allocation = enum_prop["allocation"]
                        for agent in agents:
                            agent_items = allocation.get(agent.agent_id, [])
                            utility = sum(preferences["agent_preferences"][agent.agent_id][i] 
                                        for i in agent_items if i < len(items))
                            final_utilities[agent.agent_id] = utility
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
            "winner_agent_id": winner_agent_id,
            "final_utilities": final_utilities,
            "votes_by_proposal": votes_by_proposal
        }
    
    async def run_reflection(
        self,
        agents: List[BaseLLMAgent],
        items: List[Dict],
        preferences: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        tabulation_result: Dict,
        save_callback=None
    ) -> Dict[str, Any]:
        """
        Phase 6: Individual Reflection Phase
        Agents reflect on the round outcomes.
        """
        reflections = []
        
        self.logger.info(f"=== INDIVIDUAL REFLECTION PHASE - Round {round_num} ===")
        
        reflection_prompt = f"""Reflect on the outcome of round {round_num}.
        No proposal achieved unanimous acceptance.
        Consider what adjustments might lead to consensus in future rounds."""
        
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
                
                if save_callback:
                    save_callback(agent.agent_id, f"reflection_round_{round_num}", 
                                reflection_prompt, reflection, round_num)
            except Exception as e:
                self.logger.error(f"Error in reflection for {agent.agent_id}: {e}")
        
        return {"reflections": reflections}
    
    # Prompt creation methods
    def _create_game_rules_prompt(self, items, num_agents, config):
        """Create the standardized game rules explanation prompt."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        return f"""Welcome to the Multi-Agent Negotiation Game!

You are participating in a strategic negotiation with {num_agents} agents over {len(items)} valuable items. Here are the complete rules:

**ITEMS BEING NEGOTIATED:**
{items_text}

**GAME STRUCTURE:**
- There are {num_agents} agents participating (including you)
- The negotiation will last up to {config["t_rounds"]} rounds
- Each round follows a structured sequence of phases

**YOUR PRIVATE PREFERENCES:**
You have been assigned private preferences for each item. These preferences are SECRET.

**VOTING RULES:**
- You vote "accept" or "reject" on each proposal
- A proposal needs UNANIMOUS acceptance to pass
- If no proposal gets unanimous support, we continue to the next round

**WINNING CONDITIONS:**
- Your goal is to maximize your total utility
- No deal means everyone gets zero utility

Please acknowledge that you understand these rules and are ready to participate!"""
    
    def _create_preference_assignment_prompt(self, items, agent_preferences, agent_id):
        """Create the private preference assignment prompt for a specific agent."""
        preference_details = []
        for i, item in enumerate(items):
            preference_value = agent_preferences[i] if isinstance(agent_preferences, list) else agent_preferences.get(i, 0)
            priority_level = self._get_priority_level(preference_value)
            preference_details.append(f"  {i}: {item['name']} → {preference_value:.2f} ({priority_level})")
        
        preferences_text = "\n".join(preference_details)
        
        if isinstance(agent_preferences, list):
            max_utility = sum(agent_preferences)
        else:
            max_utility = sum(agent_preferences.values())
        
        return f"""🔒 CONFIDENTIAL: Your Private Preferences Assignment

{agent_id}, you have been assigned the following SECRET preference values for each item:

**YOUR PRIVATE ITEM PREFERENCES:**
{preferences_text}

**STRATEGIC ANALYSIS:**
- Your maximum possible utility: {max_utility:.2f} points (if you get ALL items)
- Your top priorities: {self._get_top_items(items, agent_preferences, 2)}
- Your lower priorities: {self._get_bottom_items(items, agent_preferences, 2)}

**STRATEGIC CONSIDERATIONS:**
1. Other agents don't know your exact preferences
2. You may choose to reveal some preferences truthfully or misleadingly
3. Consider which agents might have complementary preferences
4. Remember: you need ALL agents to accept a proposal
5. Consider both immediate gains and the likelihood of proposals being accepted

Please acknowledge that you understand your private preferences."""
    
    def _create_initial_discussion_prompt(self, items, round_num, max_rounds):
        """Create the discussion prompt for the first round."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        return f"""🗣️ PUBLIC DISCUSSION PHASE - Round {round_num}/{max_rounds}

This is the open discussion phase where all agents can share information about their preferences.

**ITEMS AVAILABLE:**
{items_text}

**DISCUSSION OBJECTIVES:**
- Share strategic information about your preferences
- Learn about other agents' priorities
- Explore potential coalition opportunities
- Identify mutually beneficial trade possibilities

Please share your thoughts on the items and any initial ideas for how we might structure a deal."""
    
    def _create_ongoing_discussion_prompt(self, items, round_num, max_rounds):
        """Create the discussion prompt for subsequent rounds."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        urgency_note = ""
        if round_num >= max_rounds - 1:
            urgency_note = "\n⏰ **URGENT**: This is one of the final rounds!"
        
        return f"""🗣️ PUBLIC DISCUSSION PHASE - Round {round_num}/{max_rounds}

Previous proposals didn't reach consensus. Adjust your approach based on what you learned.

**ITEMS AVAILABLE:**
{items_text}{urgency_note}

**REFLECTION & STRATEGY:**
- What did you learn from previous proposals and votes?
- Which agents have conflicting vs. compatible preferences?
- How can you adjust to build consensus?

Given what happened in previous rounds, what's your updated strategy?"""
    
    def _create_contextual_discussion_prompt(self, base_prompt, agent_id, discussion_history, speaker_order, total_speakers):
        """Create discussion prompt with context from current round's discussion."""
        context_section = ""
        
        if discussion_history:
            context_section = f"""
**DISCUSSION SO FAR THIS ROUND:**
{len(discussion_history)} agent(s) have already spoken.

**YOUR TURN ({speaker_order}/{total_speakers})**:
Consider what others have said and respond strategically.

"""
        else:
            context_section = f"""**YOU'RE SPEAKING FIRST ({speaker_order}/{total_speakers})**:
Set the tone for this round's discussion.

"""
        
        return base_prompt + "\n\n" + context_section
    
    def _create_thinking_prompt(self, items, round_num, max_rounds, discussion_messages):
        """Create the private thinking prompt for strategic planning."""
        items_list = [f"  {i}: {item['name']}" for i, item in enumerate(items)]
        items_text = "\n".join(items_list)
        
        urgency_note = ""
        if round_num >= max_rounds - 1:
            urgency_note = "\n⚠️ **CRITICAL**: This is one of your final opportunities!"
        
        return f"""🧠 PRIVATE THINKING PHASE - Round {round_num}/{max_rounds}

This is your private strategic planning time.

**ITEMS AVAILABLE:**
{items_text}{urgency_note}

**STRATEGIC ANALYSIS TASKS:**
1. What did you learn about other agents' preferences?
2. Which items do others value less that you value highly?
3. What allocation would maximize your utility while achieving consensus?
4. What concessions might be necessary?

**OUTPUT REQUIRED:**
- **Reasoning**: Your analysis of the situation
- **Strategy**: Your overall approach for this round
- **Target Items**: Items you most want to secure

Remember: This thinking is completely private."""
    
    def _get_priority_level(self, value):
        """Convert numeric preference to priority description."""
        if value >= 7.0:
            return "HIGH PRIORITY"
        elif value >= 4.0:
            return "Medium Priority"
        else:
            return "Low Priority"
    
    def _get_top_items(self, items, preferences, count=2):
        """Get the top N items by preference value."""
        if isinstance(preferences, list):
            item_values = [(i, items[i]['name'], preferences[i]) for i in range(len(items))]
        else:
            item_values = [(i, items[i]['name'], preferences.get(i, 0)) for i in range(len(items))]
        
        top_items = sorted(item_values, key=lambda x: x[2], reverse=True)[:count]
        return ", ".join([f"{item[1]} ({item[2]:.1f})" for item in top_items])
    
    def _get_bottom_items(self, items, preferences, count=2):
        """Get the bottom N items by preference value."""
        if isinstance(preferences, list):
            item_values = [(i, items[i]['name'], preferences[i]) for i in range(len(items))]
        else:
            item_values = [(i, items[i]['name'], preferences.get(i, 0)) for i in range(len(items))]
        
        bottom_items = sorted(item_values, key=lambda x: x[2])[:count]
        return ", ".join([f"{item[1]} ({item[2]:.1f})" for item in bottom_items])