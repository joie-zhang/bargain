"""
Multi-agent communication system for negotiation environments.

This module provides the infrastructure for coordinating communication between
multiple AI agents in structured negotiation rounds, including message passing,
turn management, and conversation flow control.
"""

from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import random
import time
import uuid
from abc import ABC, abstractmethod
import json
from pathlib import Path


class MessageType(Enum):
    """Types of messages in the negotiation system."""
    SYSTEM = "system"           # System announcements
    PROPOSAL = "proposal"       # Agent proposals
    VOTE = "vote"              # Agent votes
    DISCUSSION = "discussion"   # Free-form discussion
    PRIVATE = "private"        # Private agent thoughts
    REFLECTION = "reflection"   # Post-round reflections


class TurnType(Enum):
    """Types of turns in negotiation rounds."""
    PROPOSAL = "proposal"       # Making proposals
    DISCUSSION = "discussion"   # Discussing proposals
    VOTING = "voting"          # Voting on proposals
    REFLECTION = "reflection"   # Reflecting on outcomes


@dataclass
class Message:
    """Represents a message in the negotiation system."""
    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast messages
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    round_number: int
    turn_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "round_number": self.round_number,
            "turn_number": self.turn_number,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            round_number=data["round_number"],
            turn_number=data["turn_number"],
            metadata=data.get("metadata", {})
        )


@dataclass
class Turn:
    """Represents a turn within a negotiation round."""
    turn_number: int
    turn_type: TurnType
    active_agent: Optional[str]  # None for simultaneous turns
    time_limit: Optional[float]  # Seconds, None for no limit
    messages: List[Message] = field(default_factory=list)
    completed: bool = False
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def duration(self) -> Optional[float]:
        """Get turn duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class AgentInterface(ABC):
    """Abstract interface for negotiation agents."""
    
    @abstractmethod
    async def receive_message(self, message: Message) -> None:
        """Receive a message from another agent or the system."""
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        context: List[Message], 
        turn_type: TurnType,
        time_limit: Optional[float] = None
    ) -> Optional[Message]:
        """Generate a response given the current context."""
        pass
    
    @abstractmethod
    async def get_agent_id(self) -> str:
        """Get the unique identifier for this agent."""
        pass


class TestAgent(AgentInterface):
    """Test agent for validating communication system functionality."""
    
    def __init__(self, agent_id: str, response_delay: float = 0.1):
        self.agent_id = agent_id
        self.response_delay = response_delay
        self.message_history: List[Message] = []
        self.response_count = 0
    
    async def receive_message(self, message: Message) -> None:
        """Receive and store message."""
        self.message_history.append(message)
    
    async def generate_response(
        self,
        context: List[Message],
        turn_type: TurnType,
        time_limit: Optional[float] = None
    ) -> Optional[Message]:
        """Generate a test response based on turn type."""
        # Simulate thinking time
        await asyncio.sleep(self.response_delay)
        
        self.response_count += 1
        
        # Generate appropriate response based on turn type
        if turn_type == TurnType.PROPOSAL:
            content = {
                "allocation": {self.agent_id: [0, 1], "other": [2, 3, 4]},
                "message": f"Test proposal #{self.response_count} from {self.agent_id}",
                "reasoning": f"This allocation maximizes my utility based on my preferences"
            }
        elif turn_type == TurnType.VOTING:
            accept = self.response_count % 2 == 0
            content = {
                "vote": "accept" if accept else "reject",
                "reason": f"Test vote reasoning from {self.agent_id}",
                "confidence": 0.8 if accept else 0.6
            }
        elif turn_type == TurnType.DISCUSSION:
            content = {
                "message": f"Test discussion point #{self.response_count} from {self.agent_id}",
                "topic": "item_distribution",
                "argument": "This distribution seems fair given the circumstances"
            }
        else:
            content = {
                "message": f"Test response from {self.agent_id}",
                "type": "general_response"
            }
        
        return Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            message_type=MessageType.PROPOSAL if turn_type == TurnType.PROPOSAL else MessageType.DISCUSSION,
            content=content,
            timestamp=time.time(),
            round_number=1,  # Will be set by communication manager
            turn_number=1    # Will be set by communication manager
        )
    
    async def get_agent_id(self) -> str:
        """Get agent ID."""
        return self.agent_id


class CommunicationManager:
    """
    Manages communication between multiple agents in negotiation rounds.
    
    This class orchestrates the flow of messages between agents, manages
    turn-taking, and ensures structured communication patterns.
    """
    
    def __init__(
        self,
        agents: List[AgentInterface],
        random_seed: Optional[int] = None
    ):
        self.agents = {}
        self.agent_list = []
        self._agent_instances = agents
        self.message_history: List[Message] = []
        self.current_round = 0
        self.current_turn = 0
        self.turn_history: List[Turn] = []
        
        # Random number generator for turn order
        self._rng = random.Random(random_seed)
        
        # Event handlers
        self.message_handlers: List[Callable[[Message], None]] = []
        self.turn_handlers: List[Callable[[Turn], None]] = []
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "turns_completed": 0,
            "rounds_completed": 0,
            "total_communication_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the communication manager by setting up agent mappings."""
        for agent in self._agent_instances:
            agent_id = await agent.get_agent_id()
            self.agents[agent_id] = agent
        self.agent_list = list(self.agents.keys())
    
    def add_message_handler(self, handler: Callable[[Message], None]) -> None:
        """Add a handler for message events."""
        self.message_handlers.append(handler)
    
    def add_turn_handler(self, handler: Callable[[Turn], None]) -> None:
        """Add a handler for turn events."""
        self.turn_handlers.append(handler)
    
    async def start_round(self, round_number: int) -> None:
        """Start a new negotiation round."""
        self.current_round = round_number
        self.current_turn = 0
        
        # Send round start message to all agents
        start_message = Message(
            id=str(uuid.uuid4()),
            sender_id="system",
            recipient_id=None,
            message_type=MessageType.SYSTEM,
            content={
                "event": "round_start",
                "round_number": round_number,
                "participants": self.agent_list
            },
            timestamp=time.time(),
            round_number=round_number,
            turn_number=0
        )
        
        await self._broadcast_message(start_message)
    
    async def execute_turn(
        self,
        turn_type: TurnType,
        active_agents: Optional[List[str]] = None,
        time_limit: Optional[float] = None,
        randomize_order: bool = True
    ) -> Turn:
        """
        Execute a single turn in the negotiation.
        
        Args:
            turn_type: Type of turn to execute
            active_agents: Agents who should participate (None for all)
            time_limit: Time limit per agent response
            randomize_order: Whether to randomize agent order
        
        Returns:
            Turn object with results
        """
        self.current_turn += 1
        
        if active_agents is None:
            active_agents = self.agent_list.copy()
        
        if randomize_order:
            active_agents = self._rng.sample(active_agents, len(active_agents))
        
        turn = Turn(
            turn_number=self.current_turn,
            turn_type=turn_type,
            active_agent=None,  # Will be set for individual responses
            time_limit=time_limit,
            started_at=time.time()
        )
        
        # Execute turn based on type
        if turn_type in [TurnType.PROPOSAL, TurnType.DISCUSSION]:
            # Sequential turns
            for agent_id in active_agents:
                await self._execute_agent_turn(turn, agent_id, turn_type, time_limit)
        
        elif turn_type == TurnType.VOTING:
            # Simultaneous voting (but processed sequentially)
            voting_tasks = []
            for agent_id in active_agents:
                task = self._execute_agent_turn(turn, agent_id, turn_type, time_limit)
                voting_tasks.append(task)
            
            # Wait for all votes (with timeout)
            if time_limit:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*voting_tasks),
                        timeout=time_limit * len(active_agents)
                    )
                except asyncio.TimeoutError:
                    # Some agents didn't respond in time
                    pass
            else:
                await asyncio.gather(*voting_tasks)
        
        turn.completed = True
        turn.completed_at = time.time()
        self.turn_history.append(turn)
        
        # Update statistics
        self.stats["turns_completed"] += 1
        if turn.duration():
            self.stats["total_communication_time"] += turn.duration()
        
        # Notify handlers
        for handler in self.turn_handlers:
            handler(turn)
        
        return turn
    
    async def _execute_agent_turn(
        self,
        turn: Turn,
        agent_id: str,
        turn_type: TurnType,
        time_limit: Optional[float]
    ) -> None:
        """Execute a single agent's part of a turn."""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        
        # Get recent context for the agent
        context = self._get_agent_context(agent_id)
        
        try:
            # Get response from agent
            if time_limit:
                response = await asyncio.wait_for(
                    agent.generate_response(context, turn_type, time_limit),
                    timeout=time_limit
                )
            else:
                response = await agent.generate_response(context, turn_type, time_limit)
            
            if response:
                # Update message metadata
                response.round_number = self.current_round
                response.turn_number = self.current_turn
                response.timestamp = time.time()
                
                # Add to turn and broadcast
                turn.messages.append(response)
                await self._broadcast_message(response)
                
        except asyncio.TimeoutError:
            # Agent didn't respond in time
            timeout_message = Message(
                id=str(uuid.uuid4()),
                sender_id="system",
                recipient_id=None,
                message_type=MessageType.SYSTEM,
                content={
                    "event": "agent_timeout",
                    "agent_id": agent_id,
                    "turn_type": turn_type.value
                },
                timestamp=time.time(),
                round_number=self.current_round,
                turn_number=self.current_turn
            )
            turn.messages.append(timeout_message)
            await self._broadcast_message(timeout_message)
    
    def _get_agent_context(self, agent_id: str, max_messages: int = 50) -> List[Message]:
        """Get recent message context for an agent."""
        # Filter out private messages not intended for this agent
        relevant_messages = []
        for msg in self.message_history[-max_messages:]:
            if (msg.message_type != MessageType.PRIVATE or 
                msg.sender_id == agent_id or 
                msg.recipient_id == agent_id):
                relevant_messages.append(msg)
        
        return relevant_messages
    
    async def _broadcast_message(self, message: Message) -> None:
        """Broadcast a message to all relevant agents."""
        self.message_history.append(message)
        self.stats["messages_sent"] += 1
        
        # Determine recipients
        if message.recipient_id:
            # Private message
            recipients = [message.recipient_id]
        else:
            # Broadcast to all agents except sender
            recipients = [aid for aid in self.agent_list if aid != message.sender_id]
        
        # Send to recipients
        for agent_id in recipients:
            if agent_id in self.agents:
                await self.agents[agent_id].receive_message(message)
        
        # Notify handlers
        for handler in self.message_handlers:
            handler(message)
    
    def get_round_messages(self, round_number: int) -> List[Message]:
        """Get all messages from a specific round."""
        return [msg for msg in self.message_history if msg.round_number == round_number]
    
    def get_agent_messages(self, agent_id: str) -> List[Message]:
        """Get all messages from a specific agent."""
        return [msg for msg in self.message_history if msg.sender_id == agent_id]
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation so far."""
        return {
            "total_messages": len(self.message_history),
            "total_turns": len(self.turn_history),
            "current_round": self.current_round,
            "current_turn": self.current_turn,
            "participants": self.agent_list,
            "message_types": {
                msg_type.value: len([m for m in self.message_history if m.message_type == msg_type])
                for msg_type in MessageType
            },
            "stats": self.stats.copy()
        }
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export complete conversation data."""
        return {
            "messages": [msg.to_dict() for msg in self.message_history],
            "turns": [
                {
                    "turn_number": turn.turn_number,
                    "turn_type": turn.turn_type.value,
                    "active_agent": turn.active_agent,
                    "time_limit": turn.time_limit,
                    "completed": turn.completed,
                    "started_at": turn.started_at,
                    "completed_at": turn.completed_at,
                    "duration": turn.duration(),
                    "message_count": len(turn.messages)
                }
                for turn in self.turn_history
            ],
            "summary": self.get_conversation_summary()
        }
    
    def save_conversation(self, filepath: str) -> None:
        """Save conversation to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.export_conversation(), f, indent=2)


class TurnManager:
    """
    Manages the sequence and flow of turns in negotiation rounds.
    
    This class provides high-level orchestration of negotiation phases,
    including structured turn sequences and flow control logic.
    """
    
    def __init__(
        self,
        communication_manager: CommunicationManager,
        default_time_limits: Optional[Dict[TurnType, float]] = None
    ):
        self.comm_manager = communication_manager
        self.time_limits = default_time_limits or {}
        self.turn_sequences: Dict[str, List[TurnType]] = {}
        
        # Define default turn sequences
        self.turn_sequences["standard"] = [
            TurnType.PROPOSAL,
            TurnType.DISCUSSION,
            TurnType.VOTING,
            TurnType.REFLECTION
        ]
        
        self.turn_sequences["quick"] = [
            TurnType.PROPOSAL,
            TurnType.VOTING
        ]
        
        self.turn_sequences["extended"] = [
            TurnType.PROPOSAL,
            TurnType.DISCUSSION,
            TurnType.DISCUSSION,  # Extra discussion
            TurnType.VOTING,
            TurnType.REFLECTION
        ]
    
    async def execute_round(
        self,
        round_number: int,
        sequence_name: str = "standard",
        custom_sequence: Optional[List[TurnType]] = None
    ) -> List[Turn]:
        """
        Execute a complete negotiation round with specified turn sequence.
        
        Args:
            round_number: The round number
            sequence_name: Name of predefined sequence to use
            custom_sequence: Custom sequence of turns (overrides sequence_name)
        
        Returns:
            List of completed turns
        """
        # Start the round
        await self.comm_manager.start_round(round_number)
        
        # Get turn sequence
        if custom_sequence:
            sequence = custom_sequence
        else:
            sequence = self.turn_sequences.get(sequence_name, self.turn_sequences["standard"])
        
        completed_turns = []
        
        # Execute each turn in sequence
        for turn_type in sequence:
            time_limit = self.time_limits.get(turn_type)
            
            turn = await self.comm_manager.execute_turn(
                turn_type=turn_type,
                time_limit=time_limit,
                randomize_order=True
            )
            
            completed_turns.append(turn)
            
            # Check for early termination conditions
            if self._should_terminate_round(turn, completed_turns):
                break
        
        # Update statistics
        self.comm_manager.stats["rounds_completed"] += 1
        
        return completed_turns
    
    def _should_terminate_round(self, current_turn: Turn, all_turns: List[Turn]) -> bool:
        """Check if round should be terminated early."""
        # For now, simple logic - could be extended
        if current_turn.turn_type == TurnType.VOTING:
            # Check if there's unanimous agreement
            vote_messages = [msg for msg in current_turn.messages 
                           if msg.message_type in [MessageType.VOTE, MessageType.PROPOSAL]]
            
            if len(vote_messages) >= len(self.comm_manager.agent_list):
                # All agents have voted/responded
                return True
        
        return False
    
    def add_turn_sequence(self, name: str, sequence: List[TurnType]) -> None:
        """Add a custom turn sequence."""
        self.turn_sequences[name] = sequence
    
    def set_time_limit(self, turn_type: TurnType, time_limit: float) -> None:
        """Set time limit for a specific turn type."""
        self.time_limits[turn_type] = time_limit


async def create_communication_system(
    agent_ids: List[str],
    agent_factory: Optional[Callable[[str], AgentInterface]] = None,
    random_seed: Optional[int] = None
) -> Tuple[CommunicationManager, TurnManager]:
    """
    Factory function to create a complete communication system.
    
    Args:
        agent_ids: List of agent identifiers
        agent_factory: Function to create agents (uses TestAgent if None)
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (CommunicationManager, TurnManager)
    """
    # Create agents
    if agent_factory is None:
        agents = [TestAgent(agent_id) for agent_id in agent_ids]
    else:
        agents = [agent_factory(agent_id) for agent_id in agent_ids]
    
    # Create communication manager
    comm_manager = CommunicationManager(agents, random_seed)
    await comm_manager.initialize()
    
    # Create turn manager
    turn_manager = TurnManager(comm_manager)
    
    return comm_manager, turn_manager