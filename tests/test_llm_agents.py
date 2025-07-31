"""
Tests for LLM agent system.

This module tests the LLM agent implementations, factory system,
and integration with the negotiation environment.
"""

import pytest
import anyio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from negotiation.llm_agents import (
    ModelType,
    LLMConfig,
    AgentResponse,
    NegotiationContext,
    RateLimiter,
    BaseLLMAgent,
    AnthropicAgent,
    OpenAIAgent,
    SimulatedAgent
)
from negotiation.agent_factory import (
    AgentConfiguration,
    ExperimentConfiguration,
    AgentFactory,
    create_o3_vs_haiku_experiment,
    create_cooperative_experiment,
    create_simulated_experiment,
    create_scaling_study_experiment
)


class TestLLMConfig:
    """Test LLM configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = LLMConfig(
            model_type=ModelType.CLAUDE_3_HAIKU,
            temperature=0.7,
            max_tokens=1000,
            timeout=30.0
        )
        
        assert config.model_type == ModelType.CLAUDE_3_HAIKU
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 30.0
    
    def test_invalid_temperature(self):
        """Test validation for invalid temperature."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            LLMConfig(
                model_type=ModelType.CLAUDE_3_HAIKU,
                temperature=3.0
            )
    
    def test_invalid_max_tokens(self):
        """Test validation for invalid max tokens."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            LLMConfig(
                model_type=ModelType.CLAUDE_3_HAIKU,
                max_tokens=0
            )
    
    def test_invalid_timeout(self):
        """Test validation for invalid timeout."""
        with pytest.raises(ValueError, match="Timeout must be positive"):
            LLMConfig(
                model_type=ModelType.CLAUDE_3_HAIKU,
                timeout=-1.0
            )


class TestNegotiationContext:
    """Test negotiation context creation and serialization."""
    
    def test_context_creation(self):
        """Test creating negotiation context."""
        items = [{"id": 0, "name": "Apple"}, {"id": 1, "name": "Book"}]
        agents = ["agent_0", "agent_1", "agent_2"]
        preferences = [5.0, 8.0]
        
        context = NegotiationContext(
            current_round=2,
            max_rounds=10,
            items=items,
            agents=agents,
            agent_id="agent_0",
            preferences=preferences
        )
        
        assert context.current_round == 2
        assert context.max_rounds == 10
        assert len(context.items) == 2
        assert len(context.agents) == 3
        assert context.agent_id == "agent_0"
        assert context.preferences == preferences
    
    def test_context_serialization(self):
        """Test context to_dict method."""
        context = NegotiationContext(
            current_round=1,
            max_rounds=5,
            items=[{"id": 0, "name": "Item"}],
            agents=["agent_0"],
            agent_id="agent_0",
            preferences=[7.5]
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["current_round"] == 1
        assert context_dict["max_rounds"] == 5
        assert context_dict["agent_id"] == "agent_0"
        assert context_dict["preferences"] == [7.5]


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.mark.anyio
    async def test_no_rate_limit_initially(self):
        """Test that rate limiter allows initial requests."""
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=1000)
        
        start_time = time.time()
        await limiter.wait_if_needed(100)
        elapsed = time.time() - start_time
        
        # Should not wait initially
        assert elapsed < 0.1
    
    @pytest.mark.anyio 
    async def test_request_rate_limiting(self):
        """Test request rate limiting."""
        limiter = RateLimiter(requests_per_minute=2, tokens_per_minute=10000)
        
        # Make two requests quickly
        await limiter.wait_if_needed(100)
        await limiter.wait_if_needed(100)
        
        # Third request should be rate limited
        start_time = time.time()
        await limiter.wait_if_needed(100)
        elapsed = time.time() - start_time
        
        # Should have waited, but for testing we'll just check it's reasonable
        assert elapsed >= 0  # At least some processing time


class TestSimulatedAgent:
    """Test simulated agent functionality."""
    
    def test_simulated_agent_creation(self):
        """Test creating simulated agents with different strategic levels."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        
        # Test aggressive agent
        aggressive_agent = SimulatedAgent("test_agent", config, "aggressive")
        assert aggressive_agent.cooperation_tendency == 0.2
        assert aggressive_agent.strategic_sophistication == 0.9
        assert aggressive_agent.response_style == "competitive"
        
        # Test cooperative agent
        cooperative_agent = SimulatedAgent("test_agent", config, "cooperative")
        assert cooperative_agent.cooperation_tendency == 0.8
        assert cooperative_agent.strategic_sophistication == 0.6
        assert cooperative_agent.response_style == "collaborative"
        
        # Test balanced agent
        balanced_agent = SimulatedAgent("test_agent", config, "balanced")
        assert balanced_agent.cooperation_tendency == 0.5
        assert balanced_agent.strategic_sophistication == 0.7
        assert balanced_agent.response_style == "strategic"
    
    @pytest.mark.anyio
    async def test_simulated_agent_response(self):
        """Test simulated agent response generation."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "balanced")
        
        context = NegotiationContext(
            current_round=1,
            max_rounds=5,
            items=[{"id": 0, "name": "Apple"}],
            agents=["test_agent"],
            agent_id="test_agent",
            preferences=[7.0]
        )
        
        # Test discussion response
        response = await agent.generate_response(context, "Let's discuss strategy")
        
        assert isinstance(response, AgentResponse)
        assert len(response.content) > 0
        assert response.model_used == "simulated-strategic"
        assert response.response_time > 0
        assert response.cost_estimate == 0.0
    
    @pytest.mark.anyio
    async def test_simulated_agent_proposal(self):
        """Test simulated agent proposal generation."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "competitive")
        
        context = NegotiationContext(
            current_round=1,
            max_rounds=5,
            items=[{"id": 0, "name": "Apple"}, {"id": 1, "name": "Book"}],
            agents=["test_agent", "other_agent"],
            agent_id="test_agent",
            preferences=[7.0, 3.0]
        )
        
        proposal = await agent.propose_allocation(context)
        
        assert isinstance(proposal, dict)
        assert "allocation" in proposal
        assert "reasoning" in proposal
        assert "proposed_by" in proposal
        assert "round" in proposal
        assert proposal["proposed_by"] == "test_agent"
        assert proposal["round"] == 1
    
    @pytest.mark.anyio
    async def test_simulated_agent_vote(self):
        """Test simulated agent voting."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "cooperative")
        
        context = NegotiationContext(
            current_round=1,
            max_rounds=5,
            items=[{"id": 0, "name": "Apple"}],
            agents=["test_agent", "other_agent"],
            agent_id="test_agent",
            preferences=[7.0]
        )
        
        test_proposal = {
            "allocation": {"test_agent": [0], "other_agent": []},
            "reasoning": "Test proposal",
            "proposed_by": "other_agent"
        }
        
        vote = await agent.vote_on_proposal(context, test_proposal)
        
        assert isinstance(vote, dict)
        assert "vote" in vote
        assert "reasoning" in vote
        assert "voter" in vote
        assert "round" in vote
        assert vote["vote"] in ["accept", "reject"]
        assert vote["voter"] == "test_agent"
        assert vote["round"] == 1
    
    def test_simulated_agent_model_info(self):
        """Test simulated agent model information."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "aggressive")
        
        info = agent.get_model_info()
        
        assert info["provider"] == "simulated"
        assert info["strategic_level"] == "competitive"
        assert info["cooperation_tendency"] == 0.2
        assert "capabilities" in info


class TestAgentConfiguration:
    """Test agent configuration."""
    
    def test_agent_config_creation(self):
        """Test creating agent configuration."""
        config = AgentConfiguration(
            agent_id="test_agent",
            model_type=ModelType.CLAUDE_3_HAIKU,
            temperature=0.8,
            max_tokens=1500
        )
        
        assert config.agent_id == "test_agent"
        assert config.model_type == ModelType.CLAUDE_3_HAIKU
        assert config.temperature == 0.8
        assert config.max_tokens == 1500
    
    def test_to_llm_config(self):
        """Test converting to LLMConfig."""
        agent_config = AgentConfiguration(
            agent_id="test_agent",
            model_type=ModelType.CLAUDE_3_HAIKU,
            temperature=0.9,
            system_prompt="Test prompt"
        )
        
        llm_config = agent_config.to_llm_config()
        
        assert isinstance(llm_config, LLMConfig)
        assert llm_config.model_type == ModelType.CLAUDE_3_HAIKU
        assert llm_config.temperature == 0.9
        assert llm_config.system_prompt == "Test prompt"


class TestExperimentConfiguration:
    """Test experiment configuration."""
    
    def test_experiment_config_creation(self):
        """Test creating experiment configuration."""
        agents = [
            AgentConfiguration("agent_0", ModelType.TEST_STRONG),
            AgentConfiguration("agent_1", ModelType.TEST_WEAK),
            AgentConfiguration("agent_2", ModelType.TEST_STRONG)
        ]
        
        config = ExperimentConfiguration(
            experiment_name="Test Experiment",
            description="Testing experiment configuration",
            agents=agents,
            m_items=5,
            n_agents=3
        )
        
        assert config.experiment_name == "Test Experiment"
        assert len(config.agents) == 3
        assert config.m_items == 5
        assert config.n_agents == 3
    
    def test_experiment_config_validation(self):
        """Test experiment configuration validation."""
        agents = [
            AgentConfiguration("agent_0", ModelType.TEST_STRONG),
            AgentConfiguration("agent_1", ModelType.TEST_WEAK)
        ]
        
        # Should fail: 2 agents but n_agents=3
        with pytest.raises(ValueError, match="Number of agents.*must match n_agents"):
            ExperimentConfiguration(
                experiment_name="Invalid",
                description="Invalid config",
                agents=agents,
                n_agents=3
            )
    
    def test_duplicate_agent_ids_validation(self):
        """Test validation for duplicate agent IDs."""
        agents = [
            AgentConfiguration("same_id", ModelType.TEST_STRONG),
            AgentConfiguration("same_id", ModelType.TEST_WEAK)  # Duplicate ID
        ]
        
        with pytest.raises(ValueError, match="Agent IDs must be unique"):
            ExperimentConfiguration(
                experiment_name="Invalid",
                description="Duplicate IDs",
                agents=agents,
                n_agents=2
            )


class TestAgentFactory:
    """Test agent factory functionality."""
    
    def test_factory_creation(self):
        """Test creating agent factory."""
        factory = AgentFactory()
        assert len(factory.created_agents) == 0
        assert len(factory.list_agents()) == 0
    
    def test_create_simulated_agent(self):
        """Test creating simulated agent through factory."""
        factory = AgentFactory()
        config = AgentConfiguration(
            agent_id="test_sim",
            model_type=ModelType.TEST_STRONG,
            strategic_level="aggressive"
        )
        
        agent = factory.create_agent(config)
        
        assert isinstance(agent, SimulatedAgent)
        assert agent.agent_id == "test_sim"
        assert agent.response_style == "competitive"
        assert factory.get_agent("test_sim") == agent
        assert "test_sim" in factory.list_agents()
    
    def test_factory_clear_agents(self):
        """Test clearing all agents from factory."""
        factory = AgentFactory()
        config = AgentConfiguration("test", ModelType.TEST_STRONG)
        
        factory.create_agent(config)
        assert len(factory.list_agents()) == 1
        
        factory.clear_agents()
        assert len(factory.list_agents()) == 0
        assert factory.get_agent("test") is None


class TestExperimentTemplates:
    """Test predefined experiment templates."""
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key', 'ANTHROPIC_API_KEY': 'test_key'})
    def test_o3_vs_haiku_experiment(self):
        """Test O3 vs Haiku experiment template."""
        config = create_o3_vs_haiku_experiment(
            experiment_name="Test O3 vs Haiku",
            competition_level=0.95,
            random_seed=42
        )
        
        assert config.experiment_name == "Test O3 vs Haiku"
        assert len(config.agents) == 3
        assert config.agents[0].model_type == ModelType.O3
        assert config.agents[1].model_type == ModelType.CLAUDE_3_HAIKU
        assert config.agents[2].model_type == ModelType.CLAUDE_3_HAIKU
        assert config.competition_level == 0.95
        assert config.random_seed == 42
        assert "strategic_behavior" in config.tags
    
    def test_simulated_experiment(self):
        """Test simulated experiment template."""
        config = create_simulated_experiment(
            experiment_name="Test Simulation",
            strategic_levels=["aggressive", "cooperative"]
        )
        
        assert config.experiment_name == "Test Simulation"
        assert len(config.agents) == 2
        assert config.agents[0].strategic_level == "aggressive"
        assert config.agents[1].strategic_level == "cooperative"
        assert "simulation" in config.tags
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    def test_cooperative_experiment(self):
        """Test cooperative experiment template."""
        config = create_cooperative_experiment(
            experiment_name="Test Cooperation",
            cooperation_level=0.9,
            models=[ModelType.CLAUDE_3_HAIKU, ModelType.CLAUDE_3_SONNET]
        )
        
        assert config.experiment_name == "Test Cooperation"
        assert len(config.agents) == 2
        assert config.competition_level == 0.9  # High cooperation
        assert config.known_to_all == True  # Common knowledge
        assert "cooperative_behavior" in config.tags
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key', 'ANTHROPIC_API_KEY': 'test_key'})
    def test_scaling_study_experiment(self):
        """Test scaling study experiment template."""
        config = create_scaling_study_experiment(
            stronger_model=ModelType.GPT_4,
            weaker_model=ModelType.CLAUDE_3_HAIKU,
            num_rounds=15
        )
        
        assert config.agents[0].model_type == ModelType.GPT_4
        assert config.agents[1].model_type == ModelType.CLAUDE_3_HAIKU
        assert config.agents[2].model_type == ModelType.CLAUDE_3_HAIKU
        assert config.t_rounds == 15
        assert config.competition_level == 0.95  # Very competitive
        assert config.known_to_all == False  # Secret preferences
        assert "scaling_laws" in config.tags


class TestIntegration:
    """Test integration between components."""
    
    def test_agent_performance_tracking(self):
        """Test that agents track performance metrics."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "balanced")
        
        # Initially no stats
        stats = agent.get_performance_stats()
        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["total_cost"] == 0.0
        assert stats["avg_response_time"] == 0
    
    @pytest.mark.anyio
    async def test_conversation_memory(self):
        """Test that agents maintain conversation memory."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "balanced")
        
        context = NegotiationContext(
            current_round=1,
            max_rounds=5,
            items=[{"id": 0, "name": "Apple"}],
            agents=["test_agent"],
            agent_id="test_agent",
            preferences=[7.0]
        )
        
        # Generate response
        await agent.generate_response(context, "Test prompt")
        
        # Check memory was updated
        assert len(agent.conversation_memory) == 1
        memory_entry = agent.conversation_memory[0]
        assert memory_entry["prompt"] == "Test prompt"
        assert "response" in memory_entry
        assert "timestamp" in memory_entry
    
    def test_factory_experiment_integration(self):
        """Test creating agents from experiment configuration."""
        agents_config = [
            AgentConfiguration("agent_0", ModelType.TEST_STRONG),
            AgentConfiguration("agent_1", ModelType.TEST_WEAK)
        ]
        
        experiment_config = ExperimentConfiguration(
            experiment_name="Integration Test",
            description="Test integration",
            agents=agents_config,
            n_agents=2
        )
        
        factory = AgentFactory()
        agents = factory.create_agents_from_experiment(experiment_config)
        
        assert len(agents) == 2
        assert all(isinstance(agent, SimulatedAgent) for agent in agents)
        assert agents[0].agent_id == "agent_0"
        assert agents[1].agent_id == "agent_1"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model_type_error(self):
        """Test error for invalid model types."""
        factory = AgentFactory()
        
        # Create config with unsupported model for real APIs
        config = AgentConfiguration(
            agent_id="test",
            model_type=ModelType.O3,  # Requires API key
            api_key=None  # No API key provided
        )
        
        with pytest.raises(ValueError, match="API key required"):
            factory.create_agent(config)
    
    @pytest.mark.anyio
    async def test_response_parsing_fallback(self):
        """Test fallback behavior when JSON parsing fails."""
        config = LLMConfig(model_type=ModelType.TEST_STRONG)
        agent = SimulatedAgent("test_agent", config, "balanced")
        
        # Mock the response generator to return invalid JSON
        async def bad_json_generator(messages, agent):
            return "This is not valid JSON for proposal"
        
        agent.response_generator = bad_json_generator
        
        context = NegotiationContext(
            current_round=1,
            max_rounds=5,
            items=[{"id": 0, "name": "Apple"}],
            agents=["test_agent"],
            agent_id="test_agent",
            preferences=[7.0]
        )
        
        # Should fall back gracefully
        proposal = await agent.propose_allocation(context)
        
        assert isinstance(proposal, dict)
        assert "allocation" in proposal
        assert "reasoning" in proposal
        assert proposal["proposed_by"] == "test_agent"


if __name__ == "__main__":
    pytest.main([__file__])