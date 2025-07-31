#!/usr/bin/env python3
"""
Demonstration of the multi-agent communication system.

This script shows how agents can communicate in structured negotiation rounds,
including message passing, turn management, and conversation flow control.
"""

import asyncio
import json
from negotiation import (
    create_communication_system,
    TurnType,
    MessageType
)


async def demo_basic_communication():
    """Demonstrate basic agent communication."""
    print("=== Basic Communication Demo ===")
    
    # Create communication system with 3 agents
    agent_ids = ["Alice", "Bob", "Charlie"]
    comm_manager, turn_manager = await create_communication_system(
        agent_ids, random_seed=42
    )
    
    print(f"Created communication system with agents: {', '.join(agent_ids)}")
    
    # Start a round
    await comm_manager.start_round(1)
    print("Started negotiation round 1")
    
    # Execute a proposal turn
    print("\n--- Proposal Turn ---")
    turn = await comm_manager.execute_turn(TurnType.PROPOSAL, time_limit=2.0)
    
    print(f"Turn completed: {turn.completed}")
    print(f"Messages in turn: {len(turn.messages)}")
    print(f"Turn duration: {turn.duration():.2f} seconds")
    
    # Show the proposals
    for msg in turn.messages:
        if msg.sender_id in agent_ids:
            print(f"  {msg.sender_id}: {msg.content.get('message', 'No message')}")
    
    # Execute a voting turn
    print("\n--- Voting Turn ---")
    voting_turn = await comm_manager.execute_turn(TurnType.VOTING, time_limit=2.0)
    
    print(f"Voting completed: {voting_turn.completed}")
    for msg in voting_turn.messages:
        if msg.sender_id in agent_ids:
            vote = msg.content.get('vote', 'No vote')
            reason = msg.content.get('reason', 'No reason')
            print(f"  {msg.sender_id}: {vote} - {reason}")
    
    # Get conversation summary
    summary = comm_manager.get_conversation_summary()
    print(f"\nConversation Summary:")
    print(f"  Total messages: {summary['total_messages']}")
    print(f"  Total turns: {summary['total_turns']}")
    print(f"  Current round: {summary['current_round']}")
    print(f"  Message types: {summary['message_types']}")
    
    return comm_manager, turn_manager


async def demo_structured_rounds():
    """Demonstrate structured negotiation rounds."""
    print("\n=== Structured Rounds Demo ===")
    
    # Create system with 4 agents
    agent_ids = ["Negotiator_A", "Negotiator_B", "Negotiator_C", "Negotiator_D"]
    comm_manager, turn_manager = await create_communication_system(
        agent_ids, random_seed=123
    )
    
    print(f"Created negotiation with {len(agent_ids)} agents")
    
    # Execute a complete standard round sequence
    print("\n--- Executing Standard Round Sequence ---")
    print("Sequence: PROPOSAL → DISCUSSION → VOTING → REFLECTION")
    
    turns = await turn_manager.execute_round(1, sequence_name="standard")
    
    print(f"\nCompleted {len(turns)} turns:")
    for i, turn in enumerate(turns, 1):
        print(f"  Turn {i}: {turn.turn_type.value} ({len(turn.messages)} messages, "
              f"{turn.duration():.2f}s)")
    
    # Execute a quick round
    print("\n--- Executing Quick Round Sequence ---")
    print("Sequence: PROPOSAL → VOTING")
    
    quick_turns = await turn_manager.execute_round(2, sequence_name="quick")
    
    print(f"\nCompleted {len(quick_turns)} turns:")
    for i, turn in enumerate(quick_turns, 1):
        print(f"  Turn {i}: {turn.turn_type.value} ({len(turn.messages)} messages, "
              f"{turn.duration():.2f}s)")
    
    # Show statistics
    stats = comm_manager.stats
    print(f"\nCommunication Statistics:")
    print(f"  Messages sent: {stats['messages_sent']}")
    print(f"  Turns completed: {stats['turns_completed']}")
    print(f"  Rounds completed: {stats['rounds_completed']}")
    print(f"  Total communication time: {stats['total_communication_time']:.2f}s")
    
    return comm_manager, turn_manager


async def demo_custom_sequences():
    """Demonstrate custom turn sequences."""
    print("\n=== Custom Sequences Demo ===")
    
    agent_ids = ["Agent_X", "Agent_Y", "Agent_Z"]
    comm_manager, turn_manager = await create_communication_system(
        agent_ids, random_seed=456
    )
    
    # Add custom sequence
    custom_sequence = [TurnType.DISCUSSION, TurnType.DISCUSSION, TurnType.PROPOSAL, TurnType.VOTING]
    turn_manager.add_turn_sequence("discussion_heavy", custom_sequence)
    
    print("Created custom 'discussion_heavy' sequence: DISCUSSION → DISCUSSION → PROPOSAL → VOTING")
    
    # Execute custom sequence
    turns = await turn_manager.execute_round(1, sequence_name="discussion_heavy")
    
    print(f"\nExecuted custom sequence with {len(turns)} turns:")
    for i, turn in enumerate(turns, 1):
        duration = turn.duration() or 0
        print(f"  Turn {i}: {turn.turn_type.value} ({len(turn.messages)} messages, "
              f"{duration:.2f}s)")
    
    # Show different message types in each turn
    for i, turn in enumerate(turns, 1):
        msg_types = {}
        for msg in turn.messages:
            msg_type = msg.message_type.value
            msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
        print(f"    Turn {i} message breakdown: {msg_types}")
    
    return comm_manager, turn_manager


async def demo_conversation_export():
    """Demonstrate conversation export and serialization."""
    print("\n=== Conversation Export Demo ===")
    
    agent_ids = ["Exporter_1", "Exporter_2"]
    comm_manager, turn_manager = await create_communication_system(
        agent_ids, random_seed=789
    )
    
    # Run a quick negotiation
    await turn_manager.execute_round(1, "quick")
    await turn_manager.execute_round(2, "quick")
    
    print("Completed 2 quick rounds")
    
    # Export conversation data
    export_data = comm_manager.export_conversation()
    
    print(f"\nExported conversation data:")
    print(f"  Messages: {len(export_data['messages'])}")
    print(f"  Turns: {len(export_data['turns'])}")
    print(f"  Summary keys: {list(export_data['summary'].keys())}")
    
    # Save to file
    save_path = "/tmp/negotiation_conversation.json"
    comm_manager.save_conversation(save_path)
    print(f"  Saved conversation to: {save_path}")
    
    # Show a sample of the exported data
    print(f"\nSample exported turn data:")
    if export_data['turns']:
        sample_turn = export_data['turns'][0]
        print(f"  Turn {sample_turn['turn_number']}: {sample_turn['turn_type']}")
        print(f"  Duration: {sample_turn['duration']:.3f}s")
        print(f"  Messages: {sample_turn['message_count']}")
    
    print(f"\nSample message:")
    if export_data['messages']:
        sample_msg = export_data['messages'][1]  # Skip system message
        print(f"  From: {sample_msg['sender_id']}")
        print(f"  Type: {sample_msg['message_type']}")
        print(f"  Round/Turn: {sample_msg['round_number']}/{sample_msg['turn_number']}")
        if 'message' in sample_msg['content']:
            print(f"  Content: {sample_msg['content']['message'][:50]}...")
    
    return comm_manager, turn_manager


async def demo_large_scale_communication():
    """Demonstrate communication with larger groups."""
    print("\n=== Large Scale Communication Demo ===")
    
    # Create a larger group
    agent_ids = [f"Participant_{i:02d}" for i in range(1, 11)]  # 10 agents
    comm_manager, turn_manager = await create_communication_system(
        agent_ids, random_seed=999
    )
    
    print(f"Created large-scale negotiation with {len(agent_ids)} agents")
    
    # Set shorter time limits for efficiency
    turn_manager.set_time_limit(TurnType.PROPOSAL, 1.0)
    turn_manager.set_time_limit(TurnType.VOTING, 0.5)
    
    print("Set faster time limits for large group")
    
    # Execute a quick round
    start_time = asyncio.get_event_loop().time()
    turns = await turn_manager.execute_round(1, "quick")
    end_time = asyncio.get_event_loop().time()
    
    total_time = end_time - start_time
    
    print(f"\nLarge group negotiation completed:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Turns: {len(turns)}")
    
    total_messages = sum(len(turn.messages) for turn in turns)
    print(f"  Total messages: {total_messages}")
    print(f"  Messages per second: {total_messages/total_time:.1f}")
    
    # Show participation
    agent_participation = {}
    for turn in turns:
        for msg in turn.messages:
            if msg.sender_id in agent_ids:
                agent_participation[msg.sender_id] = agent_participation.get(msg.sender_id, 0) + 1
    
    print(f"\nAgent participation:")
    for agent_id, count in sorted(agent_participation.items()):
        print(f"  {agent_id}: {count} messages")
    
    return comm_manager, turn_manager


async def demo_randomized_turns():
    """Demonstrate randomized turn ordering."""
    print("\n=== Randomized Turn Order Demo ===")
    
    agent_ids = ["Alpha", "Beta", "Gamma", "Delta"]
    
    # Same seed should produce same order
    print("Testing deterministic behavior with same seed:")
    orders_same_seed = []
    
    for run in range(3):
        comm_manager, _ = await create_communication_system(agent_ids, random_seed=42)
        turn = await comm_manager.execute_turn(TurnType.PROPOSAL, randomize_order=True)
        
        order = [msg.sender_id for msg in turn.messages if msg.sender_id in agent_ids]
        orders_same_seed.append(order)
        print(f"  Run {run + 1}: {' → '.join(order)}")
    
    all_same = all(order == orders_same_seed[0] for order in orders_same_seed)
    print(f"  All orders identical: {all_same}")
    
    # Different seeds should produce different orders
    print("\nTesting randomization with different seeds:")
    for seed in [1, 2, 3]:
        comm_manager, _ = await create_communication_system(agent_ids, random_seed=seed)
        turn = await comm_manager.execute_turn(TurnType.PROPOSAL, randomize_order=True)
        
        order = [msg.sender_id for msg in turn.messages if msg.sender_id in agent_ids]
        print(f"  Seed {seed}: {' → '.join(order)}")
    
    return None, None


async def main():
    """Run all communication system demonstrations."""
    print("Multi-Agent Communication System Demonstration")
    print("=" * 60)
    
    try:
        await demo_basic_communication()
        await demo_structured_rounds()
        await demo_custom_sequences()
        await demo_conversation_export()
        await demo_large_scale_communication()
        await demo_randomized_turns()
        
        print("\n" + "=" * 60)
        print("All communication demonstrations completed successfully!")
        print("The multi-agent communication system is ready for LLM integration.")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())