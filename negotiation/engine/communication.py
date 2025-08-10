"""
Communication management for the negotiation engine.

This module handles agent communication, message flow, and conversation sequencing
in a modular, configurable way extracted from the hardcoded experiment logic.
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from ..communication import Message, MessageType, TurnType
from ..llm_agents import NegotiationContext
from .base import NegotiationEngineConfig
from .results import PhaseResult


@dataclass
class ConversationContext:
    """Context for a specific conversation or discussion phase."""
    phase_type: str
    round_number: int
    current_speaker: Optional[str] = None
    speaker_order: List[str] = field(default_factory=list)
    message_history: List[Message] = field(default_factory=list)
    phase_data: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Manages agent communication and message flow during negotiation."""
    
    def __init__(self, config: NegotiationEngineConfig):
        """Initialize conversation manager with configuration."""
        self.config = config
        self.message_counter = 0
        self.turn_counter = 0
    
    async def run_discussion_round(self, 
                                 agents: List[Any],
                                 context: ConversationContext,
                                 discussion_prompt: str) -> PhaseResult:
        """
        Run a discussion phase with proper message sequencing.
        
        Args:
            agents: List of participating agents
            context: Context for this discussion round
            discussion_prompt: Base prompt for the discussion
            
        Returns:
            PhaseResult: Results of the discussion phase
        """
        start_time = time.time()
        messages = []
        
        # Determine speaker order
        speaker_order = self._determine_speaker_order(agents, context)
        
        # Run discussion with each speaker in order
        for i, agent in enumerate(speaker_order):
            agent_id = agent.agent_id
            
            # Build negotiation context for this agent
            negotiation_context = self._create_negotiation_context(
                agent, context, messages, i + 1, len(speaker_order)
            )
            
            # Create contextual prompt including previous messages
            contextual_prompt = self._create_contextual_prompt(
                discussion_prompt, agent_id, messages, i + 1, len(speaker_order)
            )
            
            try:
                # Get response from agent
                response = await agent.discuss(negotiation_context, contextual_prompt)
                
                # Create message object
                message = Message(
                    id=f"msg_{self.message_counter}",
                    sender_id=agent_id,
                    recipient_id=None,  # Broadcast message
                    message_type=MessageType.DISCUSSION,
                    content={"text": response, "phase": context.phase_type},
                    timestamp=time.time(),
                    round_number=context.round_number,
                    turn_number=self.turn_counter,
                    metadata={
                        "speaker_order": i + 1,
                        "total_speakers": len(speaker_order),
                        "phase_type": context.phase_type
                    }
                )
                
                messages.append(message)
                self.message_counter += 1
                
            except Exception as e:
                # Handle communication errors gracefully
                error_message = Message(
                    id=f"error_{self.message_counter}",
                    sender_id="system",
                    recipient_id=agent_id,
                    message_type=MessageType.SYSTEM,
                    content={"error": str(e), "error_type": "discussion_failure"},
                    timestamp=time.time(),
                    round_number=context.round_number,
                    turn_number=self.turn_counter,
                    metadata={"error": True, "original_speaker": agent_id}
                )
                messages.append(error_message)
                self.message_counter += 1
        
        self.turn_counter += 1
        duration = time.time() - start_time
        
        # Create phase result
        phase_result = PhaseResult(
            phase_type=context.phase_type,
            round_number=context.round_number,
            messages=messages,
            phase_data={
                "speaker_order": [agent.agent_id for agent in speaker_order],
                "total_messages": len(messages),
                "discussion_prompt": discussion_prompt
            },
            duration_seconds=duration
        )
        
        return phase_result
    
    async def get_agent_proposal(self,
                               agent: Any,
                               context: ConversationContext,
                               proposal_prompt: str) -> Dict[str, Any]:
        """
        Get a proposal from a specific agent.
        
        Args:
            agent: Agent to get proposal from
            context: Current negotiation context
            proposal_prompt: Prompt for generating proposal
            
        Returns:
            Dict containing proposal data and metadata
        """
        start_time = time.time()
        
        # Create negotiation context for proposal
        negotiation_context = self._create_negotiation_context(agent, context, context.message_history)
        
        try:
            # Get proposal from agent
            proposal_response = await agent.propose(negotiation_context, proposal_prompt)
            
            # Create message for proposal
            message = Message(
                id=f"proposal_{self.message_counter}",
                sender_id=agent.agent_id,
                recipient_id=None,
                message_type=MessageType.PROPOSAL,
                content={"proposal": proposal_response, "phase": context.phase_type},
                timestamp=time.time(),
                round_number=context.round_number,
                turn_number=self.turn_counter,
                metadata={"proposal_generation_time": time.time() - start_time}
            )
            
            self.message_counter += 1
            
            return {
                "agent_id": agent.agent_id,
                "proposal": proposal_response,
                "message": message,
                "generation_time": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "agent_id": agent.agent_id,
                "proposal": None,
                "error": str(e),
                "generation_time": time.time() - start_time
            }
    
    async def collect_votes(self,
                          agents: List[Any], 
                          proposals: List[Dict],
                          context: ConversationContext,
                          voting_prompt: str) -> Dict[str, Any]:
        """
        Collect votes from all agents on proposals.
        
        Args:
            agents: List of agents to collect votes from
            proposals: List of proposals to vote on
            context: Current negotiation context
            voting_prompt: Base prompt for voting
            
        Returns:
            Dict containing vote results and analysis
        """
        start_time = time.time()
        all_votes = {}
        vote_messages = []
        
        for agent in agents:
            agent_votes = {}
            
            # Create negotiation context for voting
            negotiation_context = self._create_negotiation_context(agent, context, context.message_history)
            
            try:
                # Get votes from agent for each proposal
                voting_response = await agent.vote(negotiation_context, voting_prompt, proposals)
                
                # Process voting response (assuming it returns a dict of proposal_id: vote)
                agent_votes = self._parse_voting_response(voting_response, proposals)
                
                # Create message for votes
                vote_message = Message(
                    id=f"vote_{self.message_counter}",
                    sender_id=agent.agent_id,
                    recipient_id=None,
                    message_type=MessageType.VOTE,
                    content={"votes": agent_votes, "phase": context.phase_type},
                    timestamp=time.time(),
                    round_number=context.round_number,
                    turn_number=self.turn_counter,
                    metadata={"voting_private": True}
                )
                
                vote_messages.append(vote_message)
                all_votes[agent.agent_id] = agent_votes
                self.message_counter += 1
                
            except Exception as e:
                # Handle voting errors
                error_votes = {f"proposal_{i}": "abstain" for i in range(len(proposals))}
                all_votes[agent.agent_id] = error_votes
                
                error_message = Message(
                    id=f"vote_error_{self.message_counter}",
                    sender_id="system",
                    recipient_id=agent.agent_id,
                    message_type=MessageType.SYSTEM,
                    content={"error": str(e), "error_type": "voting_failure"},
                    timestamp=time.time(),
                    round_number=context.round_number,
                    turn_number=self.turn_counter,
                    metadata={"error": True, "voter": agent.agent_id}
                )
                vote_messages.append(error_message)
                self.message_counter += 1
        
        self.turn_counter += 1
        
        return {
            "votes": all_votes,
            "messages": vote_messages,
            "total_time": time.time() - start_time,
            "vote_count": len(all_votes)
        }
    
    def _determine_speaker_order(self, agents: List[Any], context: ConversationContext) -> List[Any]:
        """Determine the order in which agents should speak."""
        if self.config.randomized_proposal_order:
            speaker_order = agents.copy()
            random.shuffle(speaker_order)
            return speaker_order
        else:
            return agents
    
    def _create_negotiation_context(self, 
                                  agent: Any, 
                                  context: ConversationContext,
                                  messages: List[Message],
                                  speaker_position: Optional[int] = None,
                                  total_speakers: Optional[int] = None) -> NegotiationContext:
        """Create a NegotiationContext for an agent."""
        return NegotiationContext(
            current_round=context.round_number,
            max_rounds=self.config.t_rounds,
            items=context.phase_data.get("items", []),
            agents=context.phase_data.get("agent_ids", []),
            agent_id=agent.agent_id,
            preferences=context.phase_data.get("preferences", {}).get(agent.agent_id, {}),
            turn_type=context.phase_type,
            current_proposals=context.phase_data.get("current_proposals", [])
        )
    
    def _create_contextual_prompt(self, 
                                base_prompt: str,
                                agent_id: str,
                                messages: List[Message],
                                speaker_position: int,
                                total_speakers: int) -> str:
        """Create a contextual prompt that includes conversation history."""
        contextual_parts = [base_prompt]
        
        if messages:
            contextual_parts.append("\\n\\nPrevious discussion in this round:")
            for msg in messages:
                if msg.message_type == MessageType.DISCUSSION:
                    content = msg.content.get("text", "")
                    contextual_parts.append(f"{msg.sender_id}: {content}")
        
        contextual_parts.append(f"\\nYou are speaker {speaker_position} of {total_speakers}.")
        
        return "\\n".join(contextual_parts)
    
    def _parse_voting_response(self, voting_response: str, proposals: List[Dict]) -> Dict[str, str]:
        """Parse voting response from agent into structured votes."""
        # This is a simplified version - in practice you'd want robust parsing
        # of the agent's response to extract their votes for each proposal
        votes = {}
        
        # For now, assume voting_response is a simple format
        # In a real implementation, this would parse structured responses
        for i, proposal in enumerate(proposals):
            proposal_id = f"proposal_{i}"
            # Simple heuristic parsing - would need more sophisticated logic
            if "approve" in voting_response.lower() or "yes" in voting_response.lower():
                votes[proposal_id] = "approve"
            elif "reject" in voting_response.lower() or "no" in voting_response.lower():
                votes[proposal_id] = "reject"
            else:
                votes[proposal_id] = "abstain"
        
        return votes
    
    def get_conversation_summary(self, messages: List[Message]) -> Dict[str, Any]:
        """Get a summary of conversation statistics."""
        return {
            "total_messages": len(messages),
            "message_types": {msg_type.value: sum(1 for msg in messages if msg.message_type == msg_type) 
                            for msg_type in MessageType},
            "speakers": list(set(msg.sender_id for msg in messages if msg.sender_id != "system")),
            "rounds_covered": list(set(msg.round_number for msg in messages)),
            "time_span": (max(msg.timestamp for msg in messages) - min(msg.timestamp for msg in messages)) if messages else 0
        }