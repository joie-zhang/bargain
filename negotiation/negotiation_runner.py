"""
Modular negotiation runner that implements the core negotiation flow.

This module extracts the negotiation logic from the O3 vs Haiku experiment
to create a reusable component that any experiment can use.
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from .llm_agents import BaseLLMAgent, NegotiationContext
from .preferences import BasePreferenceSystem
from .utility_engine import UtilityEngine


@dataclass
class NegotiationOutcome:
    """Outcome of a negotiation session."""
    consensus_reached: bool
    final_round: int
    final_allocation: Optional[Dict[str, List[int]]]
    final_utilities: Optional[Dict[str, float]]
    winner_agent_id: Optional[str]
    conversation_logs: List[Dict[str, Any]] = field(default_factory=list)
    proposals_history: List[Dict[str, Any]] = field(default_factory=list)
    votes_history: List[Dict[str, Any]] = field(default_factory=list)
    strategic_metrics: Dict[str, Any] = field(default_factory=dict)


class NegotiationPhase(Enum):
    """Phases in a negotiation round."""
    DISCUSSION = "discussion"
    PROPOSAL = "proposal"
    VOTING = "voting"
    REFLECTION = "reflection"


class ModularNegotiationRunner:
    """
    A modular negotiation runner that implements the 14-phase negotiation flow.
    
    This class can be used by any experiment to run negotiations with consistent logic.
    """
    
    def __init__(self, 
                 agents: List[BaseLLMAgent],
                 preferences: Any,  # Can be dict or PreferenceManager
                 items: List[str],
                 max_rounds: int = 10,
                 discount_factor: float = 0.9,
                 log_level: str = "INFO"):
        """
        Initialize the negotiation runner.
        
        Args:
            agents: List of agents participating in the negotiation
            preferences: Preference data structure with agent preferences
            items: List of item names to negotiate over
            max_rounds: Maximum number of negotiation rounds
            discount_factor: Discount factor for utility per round
            log_level: Logging level
        """
        self.agents = agents
        self.items = items
        self.max_rounds = max_rounds
        self.discount_factor = discount_factor
        
        # Handle preferences - can be dict or PreferenceManager
        if hasattr(preferences, 'generate_preferences'):
            # It's a PreferenceManager, generate preferences
            self.preferences = preferences.generate_preferences()
        else:
            # It's already a dict
            self.preferences = preferences
        
        # Map agent IDs to preference indices if needed
        if 'agent_preferences' in self.preferences:
            pref_keys = list(self.preferences['agent_preferences'].keys())
            if pref_keys and pref_keys[0].startswith('agent_'):
                # Generic agent IDs, need to map to actual agent IDs
                new_prefs = {}
                for i, agent in enumerate(self.agents):
                    if f'agent_{i}' in self.preferences['agent_preferences']:
                        new_prefs[agent.agent_id] = self.preferences['agent_preferences'][f'agent_{i}']
                self.preferences['agent_preferences'] = new_prefs
        
        # Initialize utility engine
        self.utility_engine = UtilityEngine()
        
        # Tracking
        self.conversation_logs = []
        self.proposals_history = []
        self.votes_history = []
        
        # Setup logging
        self.logger = logging.getLogger("NegotiationRunner")
        self.logger.setLevel(getattr(logging, log_level))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(getattr(logging, log_level))
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    async def run_negotiation(self) -> NegotiationOutcome:
        """
        Run the complete negotiation process.
        
        Returns:
            NegotiationOutcome with results
        """
        self.logger.info(f"Starting negotiation with {len(self.agents)} agents over {len(self.items)} items")
        
        consensus_reached = False
        final_allocation = None
        final_utilities = None
        winner_agent_id = None
        
        for round_num in range(1, self.max_rounds + 1):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"ROUND {round_num}/{self.max_rounds}")
            self.logger.info(f"{'='*60}")
            
            # Phase 1: Discussion
            discussion_logs = await self._run_discussion_phase(round_num)
            self.conversation_logs.extend(discussion_logs)
            
            # Phase 2: Proposals
            proposals = await self._run_proposal_phase(round_num)
            self.proposals_history.append({
                "round": round_num,
                "proposals": proposals
            })
            
            # Phase 3: Voting
            votes, consensus_allocation = await self._run_voting_phase(round_num, proposals)
            self.votes_history.append({
                "round": round_num,
                "votes": votes,
                "consensus": consensus_allocation is not None
            })
            
            # Check for consensus
            if consensus_allocation:
                consensus_reached = True
                final_allocation = consensus_allocation
                
                # Calculate final utilities
                final_utilities = self._calculate_utilities(
                    final_allocation, 
                    round_num
                )
                
                # Identify winner
                if final_utilities:
                    winner_agent_id = max(final_utilities.items(), key=lambda x: x[1])[0]
                
                self.logger.info(f"✅ CONSENSUS REACHED in round {round_num}!")
                self.logger.info(f"   Allocation: {final_allocation}")
                self.logger.info(f"   Utilities: {final_utilities}")
                break
            
            # Phase 4: Reflection (optional)
            if round_num < self.max_rounds:
                reflection_logs = await self._run_reflection_phase(round_num)
                self.conversation_logs.extend(reflection_logs)
        
        # Create outcome
        outcome = NegotiationOutcome(
            consensus_reached=consensus_reached,
            final_round=round_num,
            final_allocation=final_allocation,
            final_utilities=final_utilities,
            winner_agent_id=winner_agent_id,
            conversation_logs=self.conversation_logs,
            proposals_history=self.proposals_history,
            votes_history=self.votes_history,
            strategic_metrics=self._analyze_strategic_behavior()
        )
        
        return outcome
    
    async def _run_discussion_phase(self, round_num: int) -> List[Dict[str, Any]]:
        """Run the discussion phase where agents share information."""
        self.logger.info(f"  📢 Discussion Phase - Round {round_num}")
        
        messages = []
        
        for i, agent in enumerate(self.agents):
            # Build context
            agent_prefs = self.preferences["agent_preferences"].get(agent.agent_id, [])
            context = NegotiationContext(
                current_round=round_num,
                max_rounds=self.max_rounds,
                items=self.items,
                agents=[a.agent_id for a in self.agents],
                agent_id=agent.agent_id,
                preferences=agent_prefs,
                turn_type="discussion"
            )
            
            # Create discussion prompt
            prompt = self._create_discussion_prompt(round_num, i, len(self.agents))
            
            # Get response
            try:
                response = await agent.discuss(context, prompt)
                
                message = {
                    "phase": "discussion",
                    "round": round_num,
                    "from": agent.agent_id,
                    "content": response,
                    "timestamp": time.time(),
                    "speaker_order": i + 1,
                    "total_speakers": len(self.agents)
                }
                messages.append(message)
                
                self.logger.info(f"    {agent.agent_id}: {response[:100]}...")
                
            except Exception as e:
                self.logger.error(f"    Error from {agent.agent_id}: {e}")
        
        return messages
    
    async def _run_proposal_phase(self, round_num: int) -> List[Dict[str, Any]]:
        """Run the proposal phase where agents suggest allocations."""
        self.logger.info(f"  📝 Proposal Phase - Round {round_num}")
        
        proposals = []
        
        for agent in self.agents:
            # Build context
            context = NegotiationContext(
                current_round=round_num,
                max_rounds=self.max_rounds,
                items=self.items,
                agents=[a.agent_id for a in self.agents],
                agent_id=agent.agent_id,
                preferences=self.preferences["agent_preferences"][agent.agent_id],
                turn_type="proposal"
            )
            
            # Create proposal prompt
            prompt = self._create_proposal_prompt(round_num)
            
            try:
                response = await agent.propose(context, prompt)
                
                # Parse proposal from response
                allocation = self._parse_proposal(response, agent.agent_id)
                
                if allocation:
                    proposal = {
                        "agent_id": agent.agent_id,
                        "allocation": allocation,
                        "raw_response": response,
                        "round": round_num
                    }
                    proposals.append(proposal)
                    
                    self.logger.info(f"    {agent.agent_id}: {allocation}")
                else:
                    self.logger.warning(f"    {agent.agent_id}: Failed to parse proposal")
                    
            except Exception as e:
                self.logger.error(f"    Error from {agent.agent_id}: {e}")
        
        return proposals
    
    async def _run_voting_phase(self, round_num: int, proposals: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, List[int]]]]:
        """Run the voting phase where agents vote on proposals."""
        self.logger.info(f"  🗳️ Voting Phase - Round {round_num}")
        
        if not proposals:
            self.logger.warning("    No proposals to vote on")
            return {}, None
        
        votes = {agent.agent_id: {} for agent in self.agents}
        
        # Format proposals for voting
        proposals_text = self._format_proposals_for_voting(proposals)
        
        for agent in self.agents:
            # Build context
            context = NegotiationContext(
                current_round=round_num,
                max_rounds=self.max_rounds,
                items=self.items,
                agents=[a.agent_id for a in self.agents],
                agent_id=agent.agent_id,
                preferences=self.preferences["agent_preferences"][agent.agent_id],
                turn_type="voting",
                current_proposals=[p["allocation"] for p in proposals]
            )
            
            # Create voting prompt
            prompt = self._create_voting_prompt(proposals_text)
            
            try:
                response = await agent.vote(context, prompt)
                
                # Parse votes
                agent_votes = self._parse_votes(response, len(proposals))
                votes[agent.agent_id] = agent_votes
                
                self.logger.info(f"    {agent.agent_id}: {agent_votes}")
                
            except Exception as e:
                self.logger.error(f"    Error from {agent.agent_id}: {e}")
                # Default to no votes
                votes[agent.agent_id] = {i: False for i in range(len(proposals))}
        
        # Check for consensus
        consensus_allocation = self._check_for_consensus(proposals, votes)
        
        return votes, consensus_allocation
    
    async def _run_reflection_phase(self, round_num: int) -> List[Dict[str, Any]]:
        """Run optional reflection phase where agents think about strategy."""
        self.logger.info(f"  💭 Reflection Phase - Round {round_num}")
        
        messages = []
        
        for agent in self.agents:
            # Build context
            context = NegotiationContext(
                current_round=round_num,
                max_rounds=self.max_rounds,
                items=self.items,
                agents=[a.agent_id for a in self.agents],
                agent_id=agent.agent_id,
                preferences=self.preferences["agent_preferences"][agent.agent_id],
                turn_type="reflection"
            )
            
            # Create reflection prompt
            prompt = self._create_reflection_prompt(round_num)
            
            try:
                response = await agent.reflect(context, prompt)
                
                message = {
                    "phase": "reflection",
                    "round": round_num,
                    "from": agent.agent_id,
                    "content": response,
                    "timestamp": time.time()
                }
                messages.append(message)
                
                self.logger.info(f"    {agent.agent_id}: {response[:100]}...")
                
            except Exception as e:
                self.logger.error(f"    Error from {agent.agent_id}: {e}")
        
        return messages
    
    def _create_discussion_prompt(self, round_num: int, speaker_order: int, total_speakers: int) -> str:
        """Create discussion phase prompt."""
        items_text = ", ".join([f"{i}:{item}" for i, item in enumerate(self.items)])
        
        return f"""Round {round_num}/{self.max_rounds} - Discussion Phase
Items: {items_text}
Speaker {speaker_order}/{total_speakers}

Share your preferences strategically. You may be truthful or deceptive.
What items interest you most? What deals might work?

Respond in 2-3 sentences:"""
    
    def _create_proposal_prompt(self, round_num: int) -> str:
        """Create proposal phase prompt."""
        agent_ids = [a.agent_id for a in self.agents]
        items_text = ", ".join([f"{i}:{item}" for i, item in enumerate(self.items)])
        
        return f"""Round {round_num}/{self.max_rounds} - Proposal Phase
Items: {items_text}
Agents: {', '.join(agent_ids)}

Propose an allocation in this EXACT format:
I propose this allocation: {{'agent_id': [item_indices], ...}} - [Your reasoning]

Example:
I propose this allocation: {{'agent_0': [0, 2], 'agent_1': [1], 'agent_2': [3, 4]}} - This balances everyone's needs.

Your proposal:"""
    
    def _create_voting_prompt(self, proposals_text: str) -> str:
        """Create voting phase prompt."""
        return f"""Review these proposals and vote yes/no for each:

{proposals_text}

For EACH proposal, respond:
Proposal 1: [yes/no] - [reason]
Proposal 2: [yes/no] - [reason]
etc.

Your votes:"""
    
    def _create_reflection_prompt(self, round_num: int) -> str:
        """Create reflection phase prompt."""
        return f"""Round {round_num} complete. No consensus yet.

Reflect on the negotiation:
- What strategies are working?
- What should you try differently?

Brief reflection (1-2 sentences):"""
    
    def _parse_proposal(self, response: str, agent_id: str) -> Optional[Dict[str, List[int]]]:
        """Parse allocation from proposal response."""
        try:
            # Look for the allocation dictionary in the response
            import re
            pattern = r"I propose this allocation:\s*(\{[^}]+\})"
            match = re.search(pattern, response)
            
            if match:
                allocation_str = match.group(1)
                # Replace single quotes with double quotes for JSON parsing
                allocation_str = allocation_str.replace("'", '"')
                allocation = json.loads(allocation_str)
                return allocation
            else:
                # Try to find any dictionary-like structure
                pattern = r"\{[^}]+\}"
                match = re.search(pattern, response)
                if match:
                    allocation_str = match.group(0)
                    allocation_str = allocation_str.replace("'", '"')
                    allocation = json.loads(allocation_str)
                    return allocation
        except:
            pass
        
        return None
    
    def _format_proposals_for_voting(self, proposals: List[Dict[str, Any]]) -> str:
        """Format proposals for voting prompt."""
        lines = []
        for i, prop in enumerate(proposals, 1):
            lines.append(f"Proposal {i} (by {prop['agent_id']}):")
            lines.append(f"  Allocation: {prop['allocation']}")
        return "\n".join(lines)
    
    def _parse_votes(self, response: str, num_proposals: int) -> Dict[int, bool]:
        """Parse votes from response."""
        votes = {}
        
        for i in range(num_proposals):
            # Look for "Proposal N: yes" or "Proposal N: no"
            import re
            pattern = rf"Proposal\s*{i+1}:\s*(yes|no)"
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                vote = match.group(1).lower() == "yes"
                votes[i] = vote
            else:
                # Default to no if not found
                votes[i] = False
        
        return votes
    
    def _check_for_consensus(self, proposals: List[Dict[str, Any]], votes: Dict[str, Dict[int, bool]]) -> Optional[Dict[str, List[int]]]:
        """Check if any proposal has unanimous support."""
        for i, proposal in enumerate(proposals):
            # Check if all agents voted yes for this proposal
            all_yes = all(agent_votes.get(i, False) for agent_votes in votes.values())
            
            if all_yes:
                return proposal["allocation"]
        
        return None
    
    def _calculate_utilities(self, allocation: Dict[str, List[int]], round_num: int) -> Dict[str, float]:
        """Calculate utilities for the final allocation."""
        utilities = {}
        
        for agent_id in allocation:
            if agent_id in self.preferences["agent_preferences"]:
                # Sum utilities for allocated items
                agent_utility = 0.0
                agent_prefs = self.preferences["agent_preferences"][agent_id]
                
                for item_idx in allocation.get(agent_id, []):
                    if item_idx < len(agent_prefs):
                        agent_utility += agent_prefs[item_idx]
                
                # Apply discount factor
                discounted_utility = agent_utility * (self.discount_factor ** (round_num - 1))
                utilities[agent_id] = discounted_utility
        
        return utilities
    
    def _analyze_strategic_behavior(self) -> Dict[str, Any]:
        """Analyze strategic behavior from conversation logs."""
        metrics = {
            "total_messages": len(self.conversation_logs),
            "total_proposals": len(self.proposals_history),
            "rounds_to_consensus": None,
            "manipulation_detected": False,
            "aggressive_language": 0,
            "cooperative_language": 0
        }
        
        # Find consensus round
        for vote_record in self.votes_history:
            if vote_record["consensus"]:
                metrics["rounds_to_consensus"] = vote_record["round"]
                break
        
        # Analyze language patterns (simple keyword-based)
        aggressive_words = ["demand", "must", "refuse", "unacceptable", "insist"]
        cooperative_words = ["together", "fair", "mutual", "share", "benefit"]
        
        for log in self.conversation_logs:
            content = log.get("content", "").lower()
            
            if any(word in content for word in aggressive_words):
                metrics["aggressive_language"] += 1
            
            if any(word in content for word in cooperative_words):
                metrics["cooperative_language"] += 1
        
        return metrics