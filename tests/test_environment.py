"""
Test suite for the negotiation environment.

Tests the core environment functionality including initialization,
round progression, item management, and various edge cases.
"""

import pytest
import tempfile
import json
from pathlib import Path
from negotiation.environment import (
    NegotiationEnvironment,
    NegotiationConfig,
    NegotiationStatus,
    Item,
    ItemPool,
    Round,
    create_negotiation_environment
)


class TestNegotiationConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = NegotiationConfig(
            m_items=5,
            n_agents=3,
            t_rounds=10,
            gamma_discount=0.8,
            random_seed=42
        )
        assert config.m_items == 5
        assert config.n_agents == 3
        assert config.t_rounds == 10
        assert config.gamma_discount == 0.8
        assert config.random_seed == 42
    
    def test_invalid_items(self):
        """Test invalid number of items."""
        with pytest.raises(ValueError, match="Must have at least 1 item"):
            NegotiationConfig(m_items=0, n_agents=2, t_rounds=5, gamma_discount=0.5)
    
    def test_invalid_agents(self):
        """Test invalid number of agents."""
        with pytest.raises(ValueError, match="Need at least 2 agents"):
            NegotiationConfig(m_items=5, n_agents=1, t_rounds=5, gamma_discount=0.5)
    
    def test_invalid_rounds(self):
        """Test invalid number of rounds."""
        with pytest.raises(ValueError, match="Must have at least 1 round"):
            NegotiationConfig(m_items=5, n_agents=2, t_rounds=0, gamma_discount=0.5)
    
    def test_invalid_discount_factor(self):
        """Test invalid discount factor."""
        with pytest.raises(ValueError, match="Discount factor must be between 0 and 1"):
            NegotiationConfig(m_items=5, n_agents=2, t_rounds=5, gamma_discount=1.5)
        
        with pytest.raises(ValueError, match="Discount factor must be between 0 and 1"):
            NegotiationConfig(m_items=5, n_agents=2, t_rounds=5, gamma_discount=-0.1)


class TestItemPool:
    """Test item pool management."""
    
    def test_item_pool_creation(self):
        """Test basic item pool creation."""
        pool = ItemPool(5, seed=42)
        items = pool.get_items()
        
        assert len(items) == 5
        assert all(isinstance(item, Item) for item in items)
        assert all(item.id < 5 for item in items)
        assert len(set(item.name for item in items)) == 5  # All unique names
    
    def test_item_pool_deterministic(self):
        """Test that item pool is deterministic with same seed."""
        pool1 = ItemPool(5, seed=42)
        pool2 = ItemPool(5, seed=42)
        
        items1 = pool1.get_items()
        items2 = pool2.get_items()
        
        assert len(items1) == len(items2)
        for i1, i2 in zip(items1, items2):
            assert i1.name == i2.name
            assert i1.id == i2.id
    
    def test_large_item_pool(self):
        """Test item pool with more items than default names."""
        pool = ItemPool(25, seed=42)
        items = pool.get_items()
        
        assert len(items) == 25
        assert all(isinstance(item, Item) for item in items)
        
        # Should have some generic Item_N names
        names = [item.name for item in items]
        generic_names = [name for name in names if name.startswith("Item_")]
        assert len(generic_names) > 0
    
    def test_get_specific_item(self):
        """Test getting specific item by ID."""
        pool = ItemPool(3, seed=42)
        items = pool.get_items()
        
        for item in items:
            retrieved = pool.get_item(item.id)
            assert retrieved is not None
            assert retrieved.id == item.id
            assert retrieved.name == item.name
        
        # Test non-existent item
        assert pool.get_item(99) is None


class TestNegotiationEnvironment:
    """Test negotiation environment functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.config = NegotiationConfig(
            m_items=5,
            n_agents=3,
            t_rounds=10,
            gamma_discount=0.8,
            random_seed=42
        )
        self.env = NegotiationEnvironment(self.config)
    
    def test_environment_initialization(self):
        """Test basic environment initialization."""
        assert self.env.config == self.config
        assert self.env.status == NegotiationStatus.SETUP
        assert self.env.current_round == 0
        assert len(self.env.rounds) == 0
        assert len(self.env.item_pool.get_items()) == 5
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        assert self.env.agent_ids == agent_ids
        assert self.env.active_agents == agent_ids
        assert self.env.status == NegotiationStatus.IN_PROGRESS
        assert self.env.current_round == 1
        assert len(self.env.rounds) == 1
    
    def test_invalid_agent_count(self):
        """Test initialization with wrong number of agents."""
        with pytest.raises(ValueError, match="Expected 3 agents, got 2"):
            self.env.initialize_agents(["agent_1", "agent_2"])
    
    def test_round_progression(self):
        """Test round advancement."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        # Should start at round 1
        assert self.env.current_round == 1
        current = self.env.get_current_round()
        assert current.round_number == 1
        
        # Advance to round 2
        assert self.env.advance_round() is True
        assert self.env.current_round == 2
        assert len(self.env.rounds) == 2
        
        # Previous round should be marked as completed
        assert self.env.rounds[0].status == "completed"
    
    def test_max_rounds_reached(self):
        """Test behavior when maximum rounds are reached."""
        # Create environment with only 2 rounds
        config = NegotiationConfig(m_items=3, n_agents=2, t_rounds=2, gamma_discount=0.5)
        env = NegotiationEnvironment(config)
        env.initialize_agents(["agent_1", "agent_2"])
        
        # Should be at round 1
        assert env.current_round == 1
        
        # Advance to round 2
        assert env.advance_round() is True
        assert env.current_round == 2
        
        # Try to advance beyond max rounds
        assert env.advance_round() is False
        assert env.status == NegotiationStatus.COMPLETED
    
    def test_proposal_management(self):
        """Test adding proposals to rounds."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        proposal = {"allocation": [0, 1], "message": "I propose taking items 0 and 1"}
        self.env.add_proposal("agent_1", proposal)
        
        current = self.env.get_current_round()
        assert len(current.proposals) == 1
        assert current.proposals[0]["agent_id"] == "agent_1"
        assert current.proposals[0]["proposal"] == proposal
        assert current.proposals[0]["round"] == 1
    
    def test_vote_management(self):
        """Test adding votes to rounds."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        vote = {"accept": True, "proposal_id": 0}
        self.env.add_vote("agent_2", vote)
        
        current = self.env.get_current_round()
        assert len(current.votes) == 1
        assert current.votes[0]["agent_id"] == "agent_2"
        assert current.votes[0]["vote"] == vote
        assert current.votes[0]["round"] == 1
    
    def test_invalid_agent_actions(self):
        """Test actions by non-existent or inactive agents."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        with pytest.raises(ValueError, match="Agent unknown_agent is not active"):
            self.env.add_proposal("unknown_agent", {"test": "proposal"})
        
        with pytest.raises(ValueError, match="Agent unknown_agent is not active"):
            self.env.add_vote("unknown_agent", {"test": "vote"})
    
    def test_consensus_checking(self):
        """Test consensus detection."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        # No consensus initially
        consensus, allocation = self.env.check_consensus()
        assert consensus is False
        assert allocation is None
        
        # Add same vote from all agents
        vote = {"accept": True, "allocation": [0, 1, 2]}
        self.env.add_vote("agent_1", vote)
        self.env.add_vote("agent_2", vote)
        self.env.add_vote("agent_3", vote)
        
        consensus, allocation = self.env.check_consensus()
        assert consensus is True
        assert allocation == vote
    
    def test_no_consensus(self):
        """Test when agents disagree."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        # Add different votes
        self.env.add_vote("agent_1", {"accept": True})
        self.env.add_vote("agent_2", {"accept": False})
        self.env.add_vote("agent_3", {"accept": True})
        
        consensus, allocation = self.env.check_consensus()
        assert consensus is False
        assert allocation is None
    
    def test_negotiation_termination(self):
        """Test early termination of negotiation."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        self.env.terminate_negotiation("Test termination")
        
        assert self.env.status == NegotiationStatus.TERMINATED
        current = self.env.get_current_round()
        assert current.status == "terminated"
        assert current.outcome["reason"] == "Test termination"
    
    def test_final_allocation(self):
        """Test setting final allocation."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        allocation = {
            "agent_1": [0, 1],
            "agent_2": [2, 3], 
            "agent_3": [4]
        }
        
        self.env.finalize_allocation(allocation)
        
        assert self.env.final_allocation == allocation
        assert self.env.status == NegotiationStatus.COMPLETED
    
    def test_invalid_allocation(self):
        """Test invalid allocation detection."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        # Missing items
        invalid_allocation = {
            "agent_1": [0, 1],
            "agent_2": [2]
        }
        
        with pytest.raises(ValueError, match="Allocation must assign all items exactly once"):
            self.env.finalize_allocation(invalid_allocation)
        
        # Unknown agent
        invalid_allocation2 = {
            "agent_1": [0, 1],
            "agent_2": [2, 3],
            "unknown_agent": [4]
        }
        
        with pytest.raises(ValueError, match="Unknown agent: unknown_agent"):
            self.env.finalize_allocation(invalid_allocation2)
    
    def test_state_export(self):
        """Test state export functionality."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        state = self.env.get_negotiation_state()
        
        assert state["status"] == "in_progress"
        assert state["current_round"] == 1
        assert state["max_rounds"] == 10
        assert state["agents"]["total"] == 3
        assert state["agents"]["active"] == 3
        assert state["agents"]["ids"] == agent_ids
        assert state["items"]["total"] == 5
        assert len(state["items"]["summary"]) == 5
    
    def test_session_data_export(self):
        """Test complete session data export."""
        agent_ids = ["agent_1", "agent_2", "agent_3"]
        self.env.initialize_agents(agent_ids)
        
        # Add some activity
        self.env.add_proposal("agent_1", {"test": "proposal"})
        self.env.add_vote("agent_1", {"test": "vote"})
        
        session_data = self.env.export_session_data()
        
        assert "session_metadata" in session_data
        assert "negotiation_state" in session_data
        assert "rounds" in session_data
        assert "config" in session_data
        
        assert len(session_data["rounds"]) == 1
        assert len(session_data["rounds"][0]["proposals"]) == 1
        assert len(session_data["rounds"][0]["votes"]) == 1


class TestFactoryFunction:
    """Test the factory function for creating environments."""
    
    def test_create_environment(self):
        """Test factory function with various parameters."""
        env = create_negotiation_environment(
            m_items=7,
            n_agents=4,
            t_rounds=15,
            gamma_discount=0.95,
            random_seed=123
        )
        
        assert env.config.m_items == 7
        assert env.config.n_agents == 4
        assert env.config.t_rounds == 15
        assert env.config.gamma_discount == 0.95
        assert env.config.random_seed == 123
        assert env.status == NegotiationStatus.SETUP
    
    def test_create_environment_defaults(self):
        """Test factory function with default parameters."""
        env = create_negotiation_environment(
            m_items=3,
            n_agents=2,
            t_rounds=5
        )
        
        assert env.config.gamma_discount == 0.9  # Default
        assert env.config.random_seed is None  # Default


class TestArbitraryParameters:
    """Test environment with various arbitrary parameter combinations."""
    
    @pytest.mark.parametrize("m_items,n_agents,t_rounds,gamma", [
        (1, 2, 1, 0.0),      # Minimum values
        (3, 2, 5, 0.5),      # Small negotiation
        (10, 5, 20, 0.9),    # Medium negotiation
        (50, 10, 100, 1.0),  # Large negotiation
        (7, 3, 12, 0.75),    # Arbitrary values
        (25, 8, 50, 0.85),   # Complex scenario
    ])
    def test_various_parameter_combinations(self, m_items, n_agents, t_rounds, gamma):
        """Test environment creation with various parameter combinations."""
        env = create_negotiation_environment(
            m_items=m_items,
            n_agents=n_agents,
            t_rounds=t_rounds,
            gamma_discount=gamma,
            random_seed=42
        )
        
        # Verify configuration
        assert env.config.m_items == m_items
        assert env.config.n_agents == n_agents
        assert env.config.t_rounds == t_rounds
        assert env.config.gamma_discount == gamma
        
        # Verify item pool
        items = env.item_pool.get_items()
        assert len(items) == m_items
        assert all(0 <= item.id < m_items for item in items)
        
        # Test agent initialization
        agent_ids = [f"agent_{i}" for i in range(n_agents)]
        env.initialize_agents(agent_ids)
        
        assert env.status == NegotiationStatus.IN_PROGRESS
        assert len(env.agent_ids) == n_agents
        assert len(env.active_agents) == n_agents
        
        # Test round progression up to max
        rounds_advanced = 0
        while env.advance_round() and rounds_advanced < t_rounds:
            rounds_advanced += 1
        
        assert env.current_round <= t_rounds
        if rounds_advanced == t_rounds - 1:  # -1 because we start at round 1
            assert env.status == NegotiationStatus.COMPLETED
    
    def test_edge_case_single_item(self):
        """Test negotiation with only one item."""
        env = create_negotiation_environment(m_items=1, n_agents=2, t_rounds=3)
        items = env.item_pool.get_items()
        
        assert len(items) == 1
        assert items[0].id == 0
        
        # Test allocation
        env.initialize_agents(["agent_1", "agent_2"])
        env.finalize_allocation({"agent_1": [0], "agent_2": []})
        
        assert env.status == NegotiationStatus.COMPLETED
    
    def test_edge_case_single_round(self):
        """Test negotiation with only one round."""
        env = create_negotiation_environment(m_items=3, n_agents=2, t_rounds=1)
        env.initialize_agents(["agent_1", "agent_2"])
        
        assert env.current_round == 1
        assert env.advance_round() is False
        assert env.status == NegotiationStatus.COMPLETED
    
    def test_deterministic_behavior(self):
        """Test that same seed produces identical results."""
        seed = 12345
        
        env1 = create_negotiation_environment(5, 3, 10, 0.8, seed)
        env2 = create_negotiation_environment(5, 3, 10, 0.8, seed)
        
        items1 = env1.item_pool.get_items()
        items2 = env2.item_pool.get_items()
        
        assert len(items1) == len(items2)
        for i1, i2 in zip(items1, items2):
            assert i1.name == i2.name
            assert i1.id == i2.id


class TestPersistence:
    """Test saving and loading environment state."""
    
    def test_save_and_load(self):
        """Test saving environment to file and loading it back."""
        # Create and set up environment
        env = create_negotiation_environment(5, 3, 10, 0.8, 42)
        env.initialize_agents(["agent_1", "agent_2", "agent_3"])
        
        # Add some activity
        env.add_proposal("agent_1", {"allocation": [0, 1], "message": "test"})
        env.add_vote("agent_2", {"accept": True})
        env.advance_round()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            env.save_to_file(temp_path)
            
            # Load from file
            loaded_env = NegotiationEnvironment.load_from_file(temp_path)
            
            # Verify loaded environment matches original
            assert loaded_env.config.m_items == env.config.m_items
            assert loaded_env.config.n_agents == env.config.n_agents
            assert loaded_env.config.t_rounds == env.config.t_rounds
            assert loaded_env.config.gamma_discount == env.config.gamma_discount
            assert loaded_env.current_round == env.current_round
            assert loaded_env.agent_ids == env.agent_ids
            assert len(loaded_env.rounds) == len(env.rounds)
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])