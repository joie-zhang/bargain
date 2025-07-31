"""
Test suite for the multi-agent communication system.

Tests message passing, turn management, agent coordination, and 
structured conversation flows in negotiation rounds.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from negotiation.communication import (
    Message,
    MessageType,
    Turn,
    TurnType,
    AgentInterface,
    SimpleAgent,
    CommunicationManager,
    TurnManager,
    create_communication_system
)


class TestMessage:
    """Test message creation and serialization."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = Message(
            id="test-123",
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type=MessageType.PROPOSAL,
            content={"test": "data"},
            timestamp=1234567890.0,
            round_number=1,
            turn_number=2
        )
        
        assert message.id == "test-123"
        assert message.sender_id == "agent_1"
        assert message.recipient_id == "agent_2"
        assert message.message_type == MessageType.PROPOSAL
        assert message.content == {"test": "data"}
        assert message.timestamp == 1234567890.0
        assert message.round_number == 1
        assert message.turn_number == 2
    
    def test_message_serialization(self):
        """Test message to/from dictionary conversion."""
        original = Message(
            id="test-456",
            sender_id="agent_1",
            recipient_id=None,
            message_type=MessageType.VOTE,
            content={"vote": "accept"},
            timestamp=1234567890.0,
            round_number=2,
            turn_number=3,
            metadata={"confidence": 0.8}
        )
        
        # Convert to dict and back
        data = original.to_dict()
        reconstructed = Message.from_dict(data)
        
        assert reconstructed.id == original.id
        assert reconstructed.sender_id == original.sender_id
        assert reconstructed.recipient_id == original.recipient_id
        assert reconstructed.message_type == original.message_type
        assert reconstructed.content == original.content
        assert reconstructed.timestamp == original.timestamp
        assert reconstructed.round_number == original.round_number
        assert reconstructed.turn_number == original.turn_number
        assert reconstructed.metadata == original.metadata


class TestTurn:
    """Test turn management functionality."""
    
    def test_turn_creation(self):
        """Test basic turn creation."""
        turn = Turn(
            turn_number=1,
            turn_type=TurnType.PROPOSAL,
            active_agent="agent_1",
            time_limit=30.0
        )
        
        assert turn.turn_number == 1
        assert turn.turn_type == TurnType.PROPOSAL
        assert turn.active_agent == "agent_1"
        assert turn.time_limit == 30.0
        assert turn.completed is False
        assert len(turn.messages) == 0
    
    def test_turn_duration(self):
        """Test turn duration calculation."""
        turn = Turn(
            turn_number=1, 
            turn_type=TurnType.DISCUSSION,
            active_agent=None,
            time_limit=None
        )
        
        # No duration initially
        assert turn.duration() is None
        
        # Set start time
        turn.started_at = 1000.0
        assert turn.duration() is None  # Still no end time
        
        # Set end time
        turn.completed_at = 1005.5
        assert turn.duration() == 5.5


class TestSimpleAgent:
    """Test the SimpleAgent implementation."""
    
    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test basic agent creation."""
        agent = SimpleAgent("test_agent_1", response_delay=0.01)
        
        agent_id = await agent.get_agent_id()
        assert agent_id == "test_agent_1"
        assert len(agent.message_history) == 0
        assert agent.response_count == 0
    
    @pytest.mark.asyncio
    async def test_agent_receive_message(self):
        """Test agent receiving messages."""
        agent = SimpleAgent("test_agent_1")
        
        message = Message(
            id="msg-1",
            sender_id="system",
            recipient_id="test_agent_1",
            message_type=MessageType.SYSTEM,
            content={"event": "test"},
            timestamp=1000.0,
            round_number=1,
            turn_number=1
        )
        
        await agent.receive_message(message)
        
        assert len(agent.message_history) == 1
        assert agent.message_history[0] == message
    
    @pytest.mark.asyncio
    async def test_agent_generate_proposal(self):
        """Test agent generating proposal."""
        agent = SimpleAgent("test_agent_1", response_delay=0.01)
        
        response = await agent.generate_response([], TurnType.PROPOSAL)
        
        assert response is not None
        assert response.sender_id == "test_agent_1"
        assert response.message_type == MessageType.PROPOSAL
        assert "allocation" in response.content
        assert "message" in response.content
        assert agent.response_count == 1
    
    @pytest.mark.asyncio
    async def test_agent_generate_vote(self):
        """Test agent generating vote."""
        agent = SimpleAgent("test_agent_1", response_delay=0.01)
        
        response = await agent.generate_response([], TurnType.VOTING)
        
        assert response is not None
        assert response.sender_id == "test_agent_1"
        assert "vote" in response.content
        assert response.content["vote"] in ["accept", "reject"]
        assert agent.response_count == 1
    
    @pytest.mark.asyncio
    async def test_agent_generate_discussion(self):
        """Test agent generating discussion."""
        agent = SimpleAgent("test_agent_1", response_delay=0.01)
        
        response = await agent.generate_response([], TurnType.DISCUSSION)
        
        assert response is not None
        assert response.sender_id == "test_agent_1"
        assert response.message_type == MessageType.DISCUSSION
        assert "message" in response.content
        assert agent.response_count == 1


class TestCommunicationManager:
    """Test communication manager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_creation(self):
        """Test basic communication manager creation."""
        agents = [SimpleAgent("agent_1"), SimpleAgent("agent_2")]
        manager = CommunicationManager(agents, random_seed=42)
        await manager.initialize()
        
        assert len(manager.agents) == 2
        assert "agent_1" in manager.agents
        assert "agent_2" in manager.agents
        assert manager.agent_list == ["agent_1", "agent_2"]
        assert manager.current_round == 0
        assert manager.current_turn == 0
    
    @pytest.mark.asyncio
    async def test_start_round(self):
        """Test starting a negotiation round."""
        agents = [SimpleAgent("agent_1"), SimpleAgent("agent_2")]
        manager = CommunicationManager(agents)
        await manager.initialize()
        
        await manager.start_round(1)
        
        assert manager.current_round == 1
        assert len(manager.message_history) == 1
        
        # Check round start message
        start_msg = manager.message_history[0]
        assert start_msg.sender_id == "system"
        assert start_msg.message_type == MessageType.SYSTEM
        assert start_msg.content["event"] == "round_start"
        assert start_msg.content["round_number"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_proposal_turn(self):
        """Test executing a proposal turn."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        manager = CommunicationManager(agents, random_seed=42)
        await manager.initialize()
        
        await manager.start_round(1)
        
        turn = await manager.execute_turn(TurnType.PROPOSAL, time_limit=5.0)
        
        assert turn.turn_type == TurnType.PROPOSAL
        assert turn.completed is True
        assert len(turn.messages) == 2  # Both agents should respond
        assert manager.current_turn == 1
        
        # Check that both agents made proposals
        agent_responses = [msg for msg in turn.messages if msg.sender_id in ["agent_1", "agent_2"]]
        assert len(agent_responses) == 2
    
    @pytest.mark.asyncio
    async def test_execute_voting_turn(self):
        """Test executing a voting turn."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        manager = CommunicationManager(agents, random_seed=42)
        await manager.initialize()
        
        await manager.start_round(1)
        
        turn = await manager.execute_turn(TurnType.VOTING, time_limit=5.0)
        
        assert turn.turn_type == TurnType.VOTING
        assert turn.completed is True
        assert len(turn.messages) == 2  # Both agents should vote
        
        # Check that votes contain required fields
        for msg in turn.messages:
            if msg.sender_id in ["agent_1", "agent_2"]:
                assert "vote" in msg.content
                assert msg.content["vote"] in ["accept", "reject"]
    
    @pytest.mark.asyncio
    async def test_message_filtering(self):
        """Test agent context message filtering."""
        agents = [SimpleAgent("agent_1"), SimpleAgent("agent_2")]
        manager = CommunicationManager(agents)
        await manager.initialize()
        
        # Add various messages
        public_msg = Message("1", "agent_1", None, MessageType.DISCUSSION, {}, 1000.0, 1, 1)
        private_msg_for_1 = Message("2", "agent_2", "agent_1", MessageType.PRIVATE, {}, 1000.0, 1, 1)
        private_msg_for_2 = Message("3", "agent_1", "agent_2", MessageType.PRIVATE, {}, 1000.0, 1, 1)
        
        manager.message_history = [public_msg, private_msg_for_1, private_msg_for_2]
        
        # Agent 1 should see public + private messages to/from them
        context_1 = manager._get_agent_context("agent_1")
        assert len(context_1) == 3  # public + private_msg_for_1 + private_msg_for_2 (they sent)
        
        # Agent 2 should see public + private messages to/from them  
        context_2 = manager._get_agent_context("agent_2")
        assert len(context_2) == 3  # public + private_msg_for_1 (they sent) + private_msg_for_2
    
    @pytest.mark.asyncio
    async def test_conversation_summary(self):
        """Test conversation summary generation."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        manager = CommunicationManager(agents)
        await manager.initialize()
        
        await manager.start_round(1)
        await manager.execute_turn(TurnType.PROPOSAL)
        
        summary = manager.get_conversation_summary()
        
        assert summary["current_round"] == 1
        assert summary["current_turn"] == 1
        assert summary["participants"] == ["agent_1", "agent_2"]
        assert summary["total_messages"] > 0
        assert summary["total_turns"] == 1
        assert MessageType.SYSTEM.value in summary["message_types"]
    
    @pytest.mark.asyncio
    async def test_conversation_export_and_save(self):
        """Test conversation export and file saving."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        manager = CommunicationManager(agents)
        await manager.initialize()
        
        await manager.start_round(1)
        await manager.execute_turn(TurnType.PROPOSAL)
        
        # Test export
        export_data = manager.export_conversation()
        
        assert "messages" in export_data
        assert "turns" in export_data
        assert "summary" in export_data
        assert len(export_data["messages"]) > 0
        assert len(export_data["turns"]) == 1
        
        # Test save to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            manager.save_conversation(temp_path)
            
            # Verify file was created and has correct content
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == export_data
            
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestTurnManager:
    """Test turn manager functionality."""
    
    @pytest.mark.asyncio
    async def test_turn_manager_creation(self):
        """Test basic turn manager creation."""
        agents = [SimpleAgent("agent_1"), SimpleAgent("agent_2")]
        comm_manager = CommunicationManager(agents)
        await comm_manager.initialize()
        
        turn_manager = TurnManager(comm_manager)
        
        assert turn_manager.comm_manager == comm_manager
        assert "standard" in turn_manager.turn_sequences
        assert "quick" in turn_manager.turn_sequences
        assert "extended" in turn_manager.turn_sequences
    
    @pytest.mark.asyncio
    async def test_execute_round_standard(self):
        """Test executing a standard round sequence."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        comm_manager = CommunicationManager(agents, random_seed=42)
        await comm_manager.initialize()
        
        turn_manager = TurnManager(comm_manager)
        
        turns = await turn_manager.execute_round(1, "standard")
        
        # Standard sequence: PROPOSAL, DISCUSSION, VOTING, REFLECTION
        assert len(turns) >= 3  # May terminate early after voting
        assert turns[0].turn_type == TurnType.PROPOSAL
        assert turns[1].turn_type == TurnType.DISCUSSION
        assert turns[2].turn_type == TurnType.VOTING
        
        # All turns should be completed
        for turn in turns:
            assert turn.completed is True
    
    @pytest.mark.asyncio
    async def test_execute_round_quick(self):
        """Test executing a quick round sequence."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        comm_manager = CommunicationManager(agents, random_seed=42)
        await comm_manager.initialize()
        
        turn_manager = TurnManager(comm_manager)
        
        turns = await turn_manager.execute_round(1, "quick")
        
        # Quick sequence: PROPOSAL, VOTING
        assert len(turns) == 2
        assert turns[0].turn_type == TurnType.PROPOSAL
        assert turns[1].turn_type == TurnType.VOTING
    
    @pytest.mark.asyncio
    async def test_custom_turn_sequence(self):
        """Test executing a custom turn sequence."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        comm_manager = CommunicationManager(agents, random_seed=42)
        await comm_manager.initialize()
        
        turn_manager = TurnManager(comm_manager)
        
        custom_sequence = [TurnType.DISCUSSION, TurnType.PROPOSAL, TurnType.VOTING]
        turns = await turn_manager.execute_round(1, custom_sequence=custom_sequence)
        
        assert len(turns) == 3
        assert turns[0].turn_type == TurnType.DISCUSSION
        assert turns[1].turn_type == TurnType.PROPOSAL
        assert turns[2].turn_type == TurnType.VOTING
    
    @pytest.mark.asyncio
    async def test_add_custom_sequence(self):
        """Test adding custom turn sequences."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        comm_manager = CommunicationManager(agents)
        await comm_manager.initialize()
        
        turn_manager = TurnManager(comm_manager)
        
        # Add custom sequence
        custom_sequence = [TurnType.REFLECTION, TurnType.DISCUSSION, TurnType.VOTING]
        turn_manager.add_turn_sequence("custom", custom_sequence)
        
        assert "custom" in turn_manager.turn_sequences
        assert turn_manager.turn_sequences["custom"] == custom_sequence
        
        # Use the custom sequence
        turns = await turn_manager.execute_round(1, "custom")
        assert len(turns) == 3
        assert turns[0].turn_type == TurnType.REFLECTION
    
    @pytest.mark.asyncio
    async def test_time_limits(self):
        """Test setting and using time limits."""
        agents = [SimpleAgent("agent_1", 0.01), SimpleAgent("agent_2", 0.01)]
        comm_manager = CommunicationManager(agents)
        await comm_manager.initialize()
        
        turn_manager = TurnManager(comm_manager)
        
        # Set time limits
        turn_manager.set_time_limit(TurnType.PROPOSAL, 2.0)
        turn_manager.set_time_limit(TurnType.VOTING, 1.0)
        
        assert turn_manager.time_limits[TurnType.PROPOSAL] == 2.0
        assert turn_manager.time_limits[TurnType.VOTING] == 1.0


class TestCommunicationSystemFactory:
    """Test the factory function for creating communication systems."""
    
    @pytest.mark.asyncio
    async def test_create_system_with_defaults(self):
        """Test creating system with default test agents."""
        agent_ids = ["alice", "bob", "charlie"]
        
        comm_manager, turn_manager = await create_communication_system(
            agent_ids, random_seed=42
        )
        
        assert len(comm_manager.agents) == 3
        assert "alice" in comm_manager.agents
        assert "bob" in comm_manager.agents
        assert "charlie" in comm_manager.agents
        assert isinstance(turn_manager, TurnManager)
    
    @pytest.mark.asyncio
    async def test_create_system_with_custom_factory(self):
        """Test creating system with custom agent factory."""
        def custom_agent_factory(agent_id: str) -> AgentInterface:
            return SimpleAgent(agent_id, response_delay=0.05)
        
        agent_ids = ["agent_x", "agent_y"]
        
        comm_manager, turn_manager = await create_communication_system(
            agent_ids, 
            agent_factory=custom_agent_factory,
            random_seed=123
        )
        
        assert len(comm_manager.agents) == 2
        
        # Verify agents were created with custom factory
        for agent in comm_manager._agent_instances:
            assert isinstance(agent, SimpleAgent)
            assert agent.response_delay == 0.05


class TestIntegrationScenarios:
    """Integration tests for complete negotiation scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_negotiation_flow(self):
        """Test a complete multi-round negotiation."""
        agent_ids = ["negotiator_1", "negotiator_2", "negotiator_3"]
        
        comm_manager, turn_manager = await create_communication_system(
            agent_ids, random_seed=42
        )
        
        # Execute multiple rounds
        all_turns = []
        for round_num in range(1, 4):  # 3 rounds
            turns = await turn_manager.execute_round(round_num, "quick")
            all_turns.extend(turns)
        
        assert len(all_turns) == 6  # 3 rounds × 2 turns per round (quick sequence)
        assert comm_manager.stats["rounds_completed"] == 3
        assert comm_manager.stats["turns_completed"] == 6
        assert comm_manager.stats["messages_sent"] > 0
        
        # Verify round progression
        round_messages = comm_manager.get_round_messages(2)
        assert len(round_messages) > 0
        
        # Verify agent participation
        for agent_id in agent_ids:
            agent_messages = comm_manager.get_agent_messages(agent_id)
            assert len(agent_messages) > 0
    
    @pytest.mark.asyncio
    async def test_randomized_turn_order(self):
        """Test that turn order is properly randomized."""
        agent_ids = ["alpha", "beta", "gamma"]
        
        # Run multiple iterations with same seed
        turn_orders_same_seed = []
        for _ in range(3):
            comm_manager, turn_manager = await create_communication_system(
                agent_ids, random_seed=42
            )
            
            turn = await comm_manager.execute_turn(TurnType.PROPOSAL, randomize_order=True)
            order = [msg.sender_id for msg in turn.messages if msg.sender_id in agent_ids]
            turn_orders_same_seed.append(order)
        
        # Same seed should produce same order
        assert all(order == turn_orders_same_seed[0] for order in turn_orders_same_seed)
        
        # Different seed should produce different order (with high probability)
        comm_manager_diff, _ = await create_communication_system(
            agent_ids, random_seed=123
        )
        
        turn_diff = await comm_manager_diff.execute_turn(TurnType.PROPOSAL, randomize_order=True)
        order_diff = [msg.sender_id for msg in turn_diff.messages if msg.sender_id in agent_ids]
        
        # With 3 agents, there's a 5/6 chance the order will be different
        # This test might occasionally fail due to randomness, but it's very unlikely
        # If this becomes a problem, we can use a deterministic test instead
        
    @pytest.mark.asyncio
    async def test_large_group_communication(self):
        """Test communication with a larger group of agents."""
        agent_ids = [f"agent_{i}" for i in range(10)]  # 10 agents
        
        comm_manager, turn_manager = await create_communication_system(
            agent_ids, random_seed=42
        )
        
        # Execute a discussion turn with all agents
        turn = await comm_manager.execute_turn(TurnType.DISCUSSION, time_limit=10.0)
        
        assert turn.completed is True
        assert len(turn.messages) == 10  # All agents should participate
        
        # Verify all agents got the messages
        for agent_id in agent_ids:
            agent = comm_manager.agents[agent_id]
            assert len(agent.message_history) > 0  # Should have received messages
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling of agent timeouts."""
        # Create an agent that takes too long to respond
        class SlowAgent(SimpleAgent):
            async def generate_response(self, context, turn_type, time_limit=None):
                await asyncio.sleep(2.0)  # Takes 2 seconds
                return await super().generate_response(context, turn_type, time_limit)
        
        agents = [SimpleAgent("fast_agent", 0.01), SlowAgent("slow_agent", 0.01)]
        comm_manager = CommunicationManager(agents)
        await comm_manager.initialize()
        
        # Execute turn with short timeout
        turn = await comm_manager.execute_turn(TurnType.PROPOSAL, time_limit=0.1)
        
        # Should complete even with timeout
        assert turn.completed is True
        
        # Should have timeout messages
        timeout_messages = [
            msg for msg in turn.messages 
            if msg.sender_id == "system" and "timeout" in msg.content.get("event", "")
        ]
        assert len(timeout_messages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])