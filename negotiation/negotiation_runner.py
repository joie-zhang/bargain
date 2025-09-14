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
from .agent_experience_logger import ExperimentLogger, InteractionType


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
                 log_level: str = "INFO",
                 results_dir: Optional[str] = None,
                 enable_agent_logging: bool = True,
                 run_id: Optional[str] = None):
        """
        Initialize the negotiation runner.
        
        Args:
            agents: List of agents participating in the negotiation
            preferences: Preference data structure with agent preferences
            items: List of item names to negotiate over
            max_rounds: Maximum number of negotiation rounds
            discount_factor: Discount factor for utility per round
            log_level: Logging level
            results_dir: Directory to save agent experience logs
            enable_agent_logging: Whether to enable detailed agent experience logging
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
        
        # Setup agent experience logging
        self.enable_agent_logging = enable_agent_logging
        self.experience_logger = None
        if enable_agent_logging and results_dir:
            import uuid
            # Include run_id if provided to make unique experiment IDs per run
            if run_id:
                experiment_id = f"negotiation_{run_id}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            else:
                experiment_id = f"negotiation_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            agent_ids = [agent.agent_id for agent in self.agents]
            self.experience_logger = ExperimentLogger(experiment_id, results_dir, agent_ids, run_id)
    
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
        
        # Log final outcomes to agent experience logs
        if self.experience_logger:
            final_outcome_data = {
                "consensus_reached": consensus_reached,
                "final_allocation": final_allocation,
                "final_utilities": final_utilities,
                "winner_agent_id": winner_agent_id,
                "final_round": round_num
            }
            self.experience_logger.log_final_outcomes(final_outcome_data)
        
        return outcome
    
    async def _log_agent_interaction(self, agent: BaseLLMAgent, interaction_type: InteractionType, 
                                   round_num: int, prompt: str, context: NegotiationContext,
                                   response: str, processed_response: Any = None,
                                   response_time: float = 0.0, additional_context: Dict = None):
        """Helper method to log agent interactions if logging is enabled."""
        if not self.experience_logger:
            return
        
        # Extract relevant context data with CORRECT preference mapping
        # Get the correct agent preferences from the runner's preference data
        correct_agent_prefs = []
        if hasattr(self, 'preferences') and 'agent_preferences' in self.preferences:
            correct_agent_prefs = self.preferences['agent_preferences'].get(agent.agent_id, [])
        
        context_data = {
            "current_round": context.current_round,
            "max_rounds": context.max_rounds,
            "items": context.items if context.items else [],
            "agent_preferences": correct_agent_prefs,  # Use the CORRECT preferences from the runner
            "turn_type": context.turn_type if hasattr(context, 'turn_type') else "unknown",
            "other_agents": [a for a in (context.agents or []) if a != agent.agent_id]
        }
        
        # Add any additional context
        if additional_context:
            context_data.update(additional_context)
        
        # Get agent's logger and record interaction
        agent_logger = self.experience_logger.get_logger(agent.agent_id)
        agent_logger.log_interaction(
            interaction_type=interaction_type,
            round_number=round_num,
            input_prompt=prompt,
            context_data=context_data,
            raw_response=response,
            processed_response=processed_response or response,
            response_time=response_time,
            tokens_used=None,  # Could extract from agent response if available
            model_used=getattr(agent, 'model_name', None),
            other_agents=context_data["other_agents"],
            current_proposals=additional_context.get("current_proposals") if additional_context else None,
            previous_votes=additional_context.get("previous_votes") if additional_context else None
        )
    
    async def _run_discussion_phase(self, round_num: int) -> List[Dict[str, Any]]:
        """Run the discussion phase where agents share information."""
        self.logger.info(f"  📢 Discussion Phase - Round {round_num}")
        
        messages = []
        
        # Get previous round context if not first round
        prev_round_context = None
        if round_num > 1:
            prev_round_context = {
                "prev_proposals": next((p for p in self.proposals_history if p["round"] == round_num - 1), None),
                "prev_votes": next((v for v in self.votes_history if v["round"] == round_num - 1), None),
                "prev_discussion": [msg for msg in self.conversation_logs if msg.get("round") == round_num - 1 and msg.get("phase") == "discussion"]
            }
        
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
            
            # Get what has been said so far in this round
            current_round_messages = [msg for msg in messages if msg.get("round") == round_num]
            
            # Create discussion prompt with context
            prompt = self._create_discussion_prompt(round_num, i, len(self.agents), 
                                                   prev_round_context, current_round_messages)
            
            # Get response
            try:
                start_time = time.time()
                response = await agent.discuss(context, prompt)
                response_time = time.time() - start_time
                
                # Log agent interaction
                additional_context = {
                    "speaker_order": i + 1,
                    "total_speakers": len(self.agents),
                    "previous_messages_this_round": current_round_messages
                }
                await self._log_agent_interaction(
                    agent, InteractionType.DISCUSSION, round_num, prompt, 
                    context, response, response, response_time, additional_context
                )
                
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
        
        # Get this round's discussion for context
        round_discussion = [msg for msg in self.conversation_logs 
                           if msg.get("round") == round_num and msg.get("phase") == "discussion"]
        
        # Get previous proposals if any
        prev_proposals = []
        if round_num > 1:
            prev_proposals = [p for p in self.proposals_history if p["round"] == round_num - 1]
        
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
            
            # Create proposal prompt with discussion context
            prompt = self._create_proposal_prompt(round_num, round_discussion, prev_proposals)
            
            try:
                start_time = time.time()
                response = await agent.propose(context, prompt)
                response_time = time.time() - start_time
                
                # Parse proposal from response
                allocation = self._parse_proposal(response, agent.agent_id)
                
                # Log agent interaction
                additional_context = {
                    "round_discussion": round_discussion,
                    "previous_proposals": prev_proposals
                }
                await self._log_agent_interaction(
                    agent, InteractionType.PROPOSAL, round_num, prompt, 
                    context, response, allocation, response_time, additional_context
                )
                
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
        
        # Get discussion context for informed voting
        round_discussion = [msg for msg in self.conversation_logs 
                           if msg.get("round") == round_num and msg.get("phase") == "discussion"]
        
        # Format proposals for voting with context
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
            
            # Create voting prompt with context about proposals
            prompt = self._create_voting_prompt(proposals_text, agent.agent_id, round_num, round_discussion)
            
            try:
                start_time = time.time()
                response = await agent.vote(context, prompt)
                response_time = time.time() - start_time
                
                # Parse votes
                agent_votes = self._parse_votes(response, len(proposals))
                votes[agent.agent_id] = agent_votes
                
                # Log agent interaction
                additional_context = {
                    "current_proposals": proposals,
                    "round_discussion": round_discussion,
                    "proposals_text": proposals_text
                }
                await self._log_agent_interaction(
                    agent, InteractionType.VOTING, round_num, prompt, 
                    context, response, agent_votes, response_time, additional_context
                )
                
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
        
        # Get round summary for context
        round_discussion = [msg for msg in self.conversation_logs if msg.get("round") == round_num and msg.get("phase") == "discussion"]
        round_proposals = next((p for p in self.proposals_history if p["round"] == round_num), {"proposals": []})
        round_votes = next((v for v in self.votes_history if v["round"] == round_num), {"votes": {}, "consensus": False})
        
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
            
            # Create reflection prompt with round context
            prompt = self._create_reflection_prompt(round_num, round_discussion, round_proposals, round_votes)
            
            try:
                start_time = time.time()
                response = await agent.reflect(context, prompt)
                response_time = time.time() - start_time
                
                # Log agent interaction
                additional_context = {
                    "round_discussion": round_discussion,
                    "round_proposals": round_proposals,
                    "round_votes": round_votes
                }
                await self._log_agent_interaction(
                    agent, InteractionType.REFLECTION, round_num, prompt, 
                    context, response, response, response_time, additional_context
                )
                
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
    
    def _create_discussion_prompt(self, round_num: int, speaker_order: int, total_speakers: int,
                                 prev_round_context: Optional[Dict] = None,
                                 current_round_messages: Optional[List[Dict]] = None) -> str:
        """Create discussion phase prompt with context."""
        items_text = ", ".join([f"{i}:{item}" for i, item in enumerate(self.items)])
        
        # Build context about previous round if available
        prev_round_summary = ""
        if prev_round_context and round_num > 1:
            if prev_round_context.get("prev_votes", {}).get("consensus"):
                prev_round_summary = "\n⚠️ Note: Consensus was almost reached last round - consider what prevented agreement.\n"
            elif prev_round_context.get("prev_proposals"):
                prev_round_summary = f"\nPrevious round: No consensus reached after {len(prev_round_context['prev_proposals'].get('proposals', []))} proposals.\n"
        
        # Add what others have said this round
        current_discussion = ""
        if current_round_messages:
            current_discussion = "\nDiscussion so far this round:\n"
            for msg in current_round_messages[-2:]:  # Show last 2 messages
                agent = msg.get("from", "Unknown")
                content = msg.get("content", "")[:80] + "..." if len(msg.get("content", "")) > 80 else msg.get("content", "")
                current_discussion += f"- {agent}: {content}\n"
        
        return f"""Round {round_num}/{self.max_rounds} - Discussion Phase
Items to negotiate: {items_text}
You are speaker {speaker_order + 1} of {total_speakers}
{prev_round_summary}{current_discussion}
Strategic considerations:
- You may be truthful or deceptive about your preferences
- Consider what others have revealed (or hidden)
- Time pressure increases each round (discount factor applies)

Share your interests and potential deals (2-3 sentences):"""
    
    def _create_proposal_prompt(self, round_num: int, 
                               discussion_logs: Optional[List[Dict]] = None,
                               prev_proposals: Optional[List[Dict]] = None) -> str:
        """Create proposal phase prompt with discussion context."""
        agent_ids = [a.agent_id for a in self.agents]
        items_text = ", ".join([f"{i}:{item}" for i, item in enumerate(self.items)])
        
        # Summarize discussion insights
        discussion_summary = ""
        if discussion_logs:
            discussion_summary = "\nKey points from discussion:\n"
            for msg in discussion_logs:
                agent = msg.get("from", "Unknown")
                content = msg.get("content", "")
                # Extract item preferences mentioned
                if "Item" in content or "item" in content:
                    summary = content[:100] + "..." if len(content) > 100 else content
                    discussion_summary += f"- {agent}: {summary}\n"
        
        # Show why previous proposals failed if applicable
        prev_proposal_context = ""
        if prev_proposals and len(prev_proposals) > 0:
            prev_proposal_context = f"\n⚠️ Previous round had {len(prev_proposals[0].get('proposals', []))} proposals but no consensus.\n"
        
        return f"""Round {round_num}/{self.max_rounds} - Proposal Phase
Items to allocate: {items_text}
Negotiating agents: {', '.join(agent_ids)}
{discussion_summary}{prev_proposal_context}
Guidelines:
- Consider what each agent expressed interest in
- Balance individual gains with group acceptance
- Remember: unanimous approval needed for consensus

Propose an allocation in this EXACT format:
I propose this allocation: {{'agent_id': [item_indices], ...}} - [Brief reasoning]

Example format:
I propose this allocation: {{'{agent_ids[0]}': [0, 2], '{agent_ids[1] if len(agent_ids) > 1 else "agent_1"}': [1, 3]}} - Reflects stated preferences.

Your proposal:"""
    
    def _create_voting_prompt(self, proposals_text: str, agent_id: str, 
                             round_num: int, discussion_logs: Optional[List[Dict]] = None) -> str:
        """Create voting phase prompt with context."""
        
        # Extract what you said you wanted in discussion
        your_stated_interests = ""
        if discussion_logs:
            for msg in discussion_logs:
                if msg.get("from") == agent_id:
                    content = msg.get("content", "")[:100]
                    your_stated_interests = f"\nYour stated interests: {content}...\n"
                    break
        
        # Add urgency if getting close to max rounds
        urgency = ""
        if round_num >= self.max_rounds - 1:
            urgency = "\n⚠️ FINAL ROUNDS - Consider accepting reasonable proposals to avoid no-deal outcome!\n"
        elif round_num >= self.max_rounds // 2:
            urgency = f"\n⏰ Round {round_num}/{self.max_rounds} - Time pressure increasing (discount factor: {self.discount_factor ** round_num:.2f})\n"
        
        return f"""Round {round_num} - Voting Phase

You are Agent {agent_id}. Review these proposals and decide whether to accept each one:

{proposals_text}
{your_stated_interests}{urgency}
Voting considerations:
- Does this proposal give you acceptable utility?
- Is it better than likely alternatives in remaining rounds?
- Remember: ALL agents must approve for consensus
- No deal means everyone gets utility 0
- Note: If a proposal was made by you (Agent {agent_id}), consider whether you still support it

For EACH proposal, vote yes/no with brief reasoning:
Proposal 1: [yes/no] - [reason]
Proposal 2: [yes/no] - [reason]
etc.

Your votes:"""
    
    def _create_reflection_prompt(self, round_num: int, discussion_logs: List[Dict], 
                                 proposals: Dict, votes: Dict) -> str:
        """Create reflection phase prompt with context about what happened."""
        
        # Summarize discussion points
        discussion_summary = ""
        if discussion_logs:
            discussion_summary = "Discussion highlights:\n"
            for msg in discussion_logs[:3]:  # Show first 3 messages
                agent = msg.get("from", "Unknown")
                content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
                discussion_summary += f"- {agent}: {content}\n"
        
        # Summarize proposals
        proposal_summary = ""
        if proposals.get("proposals"):
            proposal_summary = "\nProposals made:\n"
            for prop in proposals["proposals"][:3]:  # Show first 3 proposals
                agent = prop.get("agent_id", "Unknown")
                allocation = prop.get("allocation", {})
                proposal_summary += f"- {agent} proposed: {allocation}\n"
        
        # Summarize voting outcomes
        vote_summary = ""
        if votes.get("votes"):
            vote_summary = "\nVoting results:\n"
            consensus = votes.get("consensus", False)
            vote_summary += f"- Consensus reached: {'Yes' if consensus else 'No'}\n"
            # Show vote counts if available
            vote_details = votes.get("votes", {})
            if vote_details:
                vote_summary += f"- {len(vote_details)} agents voted\n"
        
        # Add strategic guidance based on progress
        strategic_guidance = ""
        if round_num >= self.max_rounds - 2:
            strategic_guidance = "\n⚠️ Critical: Only 1-2 rounds remain. Consider compromise to avoid no-deal.\n"
        elif not votes.get("consensus") and round_num > self.max_rounds // 2:
            strategic_guidance = "\n📊 Mid-game: No consensus yet. May need to adjust strategy or expectations.\n"
        
        return f"""Round {round_num} complete. Here's what happened:

{discussion_summary}{proposal_summary}{vote_summary}{strategic_guidance}
Reflect on this round:
- What strategies worked or didn't work?
- What should you try differently in the next round?
- What did you learn about other agents' preferences?
- How can you build consensus given the time remaining?

Brief strategic reflection (2-3 sentences):"""
    
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
            lines.append(f"Proposal {i} (proposed by Agent {prop['agent_id']}):")
            lines.append(f"  Allocation: {prop['allocation']}")
            lines.append("")  # Add blank line for readability
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