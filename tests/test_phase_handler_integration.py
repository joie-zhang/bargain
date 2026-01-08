#!/usr/bin/env python3
"""Quick test for PhaseHandler + GameEnvironment integration."""

from game_environments import create_game_environment
from strong_models_experiment.phases.phase_handlers import PhaseHandler

# Create a game environment
game = create_game_environment(
    'item_allocation',
    n_items=3,
    n_agents=2,
    t_rounds=5,
    preferences_mode='competitive'
)

# Create PhaseHandler with game environment
handler = PhaseHandler(game_environment=game)
print('PhaseHandler created with game_environment:', handler.game_environment is not None)
print('Game type:', handler.game_environment.get_game_type())

# Test _build_game_state method works
class FakeAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id

agents = [FakeAgent('Agent_1'), FakeAgent('Agent_2')]
items = [{'name': 'Item A'}, {'name': 'Item B'}, {'name': 'Item C'}]
preferences = {'agent_preferences': {'Agent_1': [1.0, 2.0, 3.0], 'Agent_2': [3.0, 2.0, 1.0]}}

state = handler._build_game_state(agents, items, preferences, 1, 5, 'Agent_1', 'discussion')
print('Game state built successfully:', len(state), 'keys')
print('Game state keys:', list(state.keys()))

# Test prompt generation with GameEnvironment
# Use the correct method signature: get_discussion_prompt(agent_id, game_state, round_num, max_rounds, discussion_history)
prompt = handler.game_environment.get_discussion_prompt(
    agent_id='Agent_1',
    game_state=state,
    round_num=1,
    max_rounds=5,
    discussion_history=[]
)
print('Prompt generated:', len(prompt), 'chars')
print('\nSample of generated prompt:')
print(prompt[:500] + '...')

print('\nSUCCESS!')
