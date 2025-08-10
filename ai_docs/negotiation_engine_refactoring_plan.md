# Negotiation Engine Refactoring Plan

## Overview

This document outlines the refactoring plan to extract the hardcoded negotiation logic from `O3VsHaikuExperiment` into a modular, parameterizable `NegotiationEngine` that can be used by the parameterized experiment system and any future experiment configurations.

## Current Architecture Issues

### Problems with Current Approach
1. **Hardcoded Logic**: `O3VsHaikuExperiment` has negotiation phases hardcoded for specific 3-agent, 5-item scenarios
2. **Non-Reusable**: Cannot easily adapt for different agent counts, rounds, or preference systems  
3. **Tight Coupling**: Negotiation logic is mixed with experiment-specific analysis and logging
4. **Duplicate Code Risk**: Parameterized experiment was recreating negotiation phases instead of reusing proven logic

### Current Temporary Fix
- Parameterized experiment calls `O3VsHaikuExperiment._run_negotiation()` directly
- Works but creates architectural debt
- Not scalable for different experiment types

## Target Architecture

### Core Components to Extract

```
NegotiationEngine
├── NegotiationOrchestrator     # Manages full negotiation flow
├── PhaseManager               # Handles individual negotiation phases
├── ConversationManager        # Manages agent communication
├── ConsensusTracker          # Tracks voting and agreement
└── UtilityCalculator         # Computes final utilities
```

### Design Principles
1. **Configuration-Driven**: All behavior controlled by config objects
2. **Agent-Agnostic**: Works with any agent implementations
3. **Phase-Modular**: Each negotiation phase is independently configurable
4. **Event-Driven**: Emits events for logging, analysis, and monitoring
5. **Async-First**: Proper async/await support throughout

## Refactoring Steps

### Phase 1: Extract Core Negotiation Engine

#### Step 1.1: Create Base NegotiationEngine Interface
**File**: `negotiation/engine/base.py`
```python
@dataclass
class NegotiationConfig:
    """Configuration for negotiation engine behavior."""
    # Environment
    n_agents: int
    m_items: int  
    t_rounds: int
    gamma_discount: float
    
    # Phase configuration
    max_discussion_turns: int = 3
    allow_private_thinking: bool = True
    require_unanimous_consensus: bool = True
    
    # Communication
    randomized_proposal_order: bool = False
    max_reflection_chars: int = 2000
    turn_timeout_seconds: int = 60

class NegotiationEngine(ABC):
    """Abstract base for negotiation engines."""
    
    @abstractmethod
    async def run_negotiation(self, agents: List[BaseLLMAgent], 
                            env: NegotiationEnvironment,
                            preferences: Dict[str, Any]) -> NegotiationResult
```

#### Step 1.2: Create Negotiation Result Types
**File**: `negotiation/engine/results.py`
```python
@dataclass
class PhaseResult:
    """Result from a single negotiation phase."""
    phase_type: str
    round_number: int
    messages: List[Message]
    phase_data: Dict[str, Any]
    duration_seconds: float

@dataclass 
class NegotiationResult:
    """Complete negotiation outcome."""
    negotiation_id: str
    config: NegotiationConfig
    
    # Outcome
    consensus_reached: bool
    final_round: int
    winner_agent_id: Optional[str]
    final_utilities: Dict[str, float]
    
    # Process data
    phase_results: List[PhaseResult]
    conversation_logs: List[Message]
    strategic_behaviors: Dict[str, Any]
    
    # Metadata
    total_duration: float
    api_costs: Dict[str, float]
```

#### Step 1.3: Extract Communication System
**File**: `negotiation/engine/communication.py`
```python
class ConversationManager:
    """Manages agent communication and message flow."""
    
    def __init__(self, config: NegotiationConfig):
        self.config = config
        self.message_history: List[Message] = []
        
    async def run_discussion_round(self, 
                                 agents: List[BaseLLMAgent],
                                 context: NegotiationContext) -> PhaseResult:
        """Run a discussion phase with proper message sequencing."""
        
    async def get_agent_proposal(self,
                               agent: BaseLLMAgent,
                               context: NegotiationContext) -> Dict[str, Any]:
        """Get a proposal from a specific agent."""
        
    async def collect_votes(self,
                          agents: List[BaseLLMAgent], 
                          proposals: List[Dict],
                          context: NegotiationContext) -> Dict[str, Any]:
        """Collect votes from all agents on proposals."""
```

### Phase 2: Extract Negotiation Phases

#### Step 2.1: Create Phase Manager
**File**: `negotiation/engine/phases.py`
```python
class PhaseManager:
    """Manages individual negotiation phases."""
    
    def __init__(self, config: NegotiationConfig, 
                 conversation_manager: ConversationManager):
        self.config = config
        self.conversation_manager = conversation_manager
        
    async def run_game_setup_phase(self, agents, items, preferences) -> PhaseResult:
        """Phase 1A: Game Setup - Give identical opening prompt."""
        
    async def run_preference_assignment_phase(self, agents, items, preferences) -> PhaseResult:
        """Phase 1B: Private Preference Assignment."""
        
    async def run_discussion_phase(self, agents, context) -> PhaseResult:
        """Phase 2: Public Discussion."""
        
    async def run_private_thinking_phase(self, agents, context) -> PhaseResult:
        """Phase 3: Private Thinking/Strategy Planning."""
        
    async def run_proposal_phase(self, agents, context) -> PhaseResult:
        """Phase 4A: Proposal Submission."""
        
    async def run_voting_phase(self, agents, proposals, context) -> PhaseResult:
        """Phase 5A: Private Voting."""
        
    async def run_reflection_phase(self, agents, results, context) -> PhaseResult:
        """Phase 7A: Individual Reflection on round outcomes."""
```

#### Step 2.2: Extract Consensus and Utility Logic
**File**: `negotiation/engine/consensus.py`
```python
class ConsensusTracker:
    """Tracks consensus and voting outcomes."""
    
    def check_consensus(self, votes: Dict, proposals: List, 
                       config: NegotiationConfig) -> Tuple[bool, Optional[Dict]]:
        """Check if consensus has been reached."""
        
    def get_winning_proposal(self, votes: Dict, proposals: List) -> Optional[Dict]:
        """Determine winning proposal from votes."""
        
class UtilityCalculator:
    """Calculates utilities based on different preference systems."""
    
    def calculate_final_utilities(self, 
                                final_allocation: Dict,
                                preferences: Dict,
                                config: NegotiationConfig) -> Dict[str, float]:
        """Calculate utilities for vector or matrix preference systems."""
```

### Phase 3: Create Main Orchestrator

#### Step 3.1: Implement Concrete NegotiationEngine
**File**: `negotiation/engine/orchestrator.py`
```python
class StandardNegotiationEngine(NegotiationEngine):
    """Standard implementation of multi-round negotiation."""
    
    def __init__(self, config: NegotiationConfig):
        self.config = config
        self.conversation_manager = ConversationManager(config)
        self.phase_manager = PhaseManager(config, self.conversation_manager)
        self.consensus_tracker = ConsensusTracker()
        self.utility_calculator = UtilityCalculator()
        
    async def run_negotiation(self, 
                            agents: List[BaseLLMAgent],
                            env: NegotiationEnvironment, 
                            preferences: Dict[str, Any]) -> NegotiationResult:
        """Run complete negotiation process."""
        
        negotiation_id = f"negotiation_{int(time.time())}"
        start_time = time.time()
        phase_results = []
        
        # Phase 1: Setup
        setup_result = await self.phase_manager.run_game_setup_phase(
            agents, env.get_items_summary(), preferences
        )
        phase_results.append(setup_result)
        
        # Phase 2: Preference Assignment
        pref_result = await self.phase_manager.run_preference_assignment_phase(
            agents, env.get_items_summary(), preferences
        )
        phase_results.append(pref_result)
        
        # Main negotiation rounds
        consensus_reached = False
        final_round = 0
        
        for round_num in range(1, self.config.t_rounds + 1):
            final_round = round_num
            context = self._create_round_context(round_num, agents, env, preferences)
            
            # Discussion Phase
            discussion_result = await self.phase_manager.run_discussion_phase(agents, context)
            phase_results.append(discussion_result)
            
            # Private Thinking Phase
            if self.config.allow_private_thinking:
                thinking_result = await self.phase_manager.run_private_thinking_phase(agents, context)
                phase_results.append(thinking_result)
            
            # Proposal Phase
            proposal_result = await self.phase_manager.run_proposal_phase(agents, context)
            phase_results.append(proposal_result)
            
            # Voting Phase
            voting_result = await self.phase_manager.run_voting_phase(
                agents, proposal_result.phase_data['proposals'], context
            )
            phase_results.append(voting_result)
            
            # Check Consensus
            consensus_reached, winning_proposal = self.consensus_tracker.check_consensus(
                voting_result.phase_data['votes'],
                proposal_result.phase_data['proposals'],
                self.config
            )
            
            if consensus_reached:
                break
                
            # Reflection Phase
            reflection_result = await self.phase_manager.run_reflection_phase(
                agents, {'voting': voting_result, 'proposals': proposal_result}, context
            )
            phase_results.append(reflection_result)
        
        # Calculate final utilities
        if consensus_reached and winning_proposal:
            final_utilities = self.utility_calculator.calculate_final_utilities(
                winning_proposal['allocation'], preferences, self.config
            )
        else:
            final_utilities = {agent.agent_id: 0.0 for agent in agents}
            
        # Determine winner
        winner_agent_id = max(final_utilities.keys(), 
                            key=lambda k: final_utilities[k]) if final_utilities else None
        
        return NegotiationResult(
            negotiation_id=negotiation_id,
            config=self.config,
            consensus_reached=consensus_reached,
            final_round=final_round,
            winner_agent_id=winner_agent_id,
            final_utilities=final_utilities,
            phase_results=phase_results,
            conversation_logs=self._extract_all_messages(phase_results),
            strategic_behaviors=self._analyze_strategic_behaviors(phase_results),
            total_duration=time.time() - start_time,
            api_costs=self._calculate_api_costs(phase_results)
        )
```

### Phase 4: Update Parameterized Experiment System

#### Step 4.1: Integrate with Parameterized System
**File**: `experiments/parameterized_experiment.py`
```python
# Replace the current negotiation method with:
async def _run_parameterized_negotiation(self, 
                                       agents: List[Any],
                                       env: Any, 
                                       preferences: Dict[str, Any],
                                       config: ExperimentConfig) -> Dict[str, Any]:
    """Run negotiation using the modular negotiation engine."""
    
    # Convert experiment config to negotiation config
    negotiation_config = NegotiationConfig(
        n_agents=config.environment.n_agents,
        m_items=config.environment.m_items,
        t_rounds=config.environment.t_rounds,
        gamma_discount=config.environment.gamma_discount,
        max_discussion_turns=config.environment.max_conversation_turns_per_round,
        allow_private_thinking=True,
        require_unanimous_consensus=config.environment.require_unanimous_consensus,
        randomized_proposal_order=config.environment.randomized_proposal_order,
        max_reflection_chars=2000,
        turn_timeout_seconds=config.environment.timeout_minutes * 60
    )
    
    # Create negotiation engine
    engine = StandardNegotiationEngine(negotiation_config)
    
    # Run negotiation
    result = await engine.run_negotiation(agents, env, preferences)
    
    # Convert result to format expected by parameterized system
    return {
        "consensus_reached": result.consensus_reached,
        "final_round": result.final_round,
        "winner_agent_id": result.winner_agent_id,
        "final_utilities": result.final_utilities,
        "proposal_orders": self._extract_proposal_orders(result),
        "conversation_logs": [asdict(msg) for msg in result.conversation_logs],
        "proposals": self._extract_proposals(result),
        "votes": self._extract_votes(result)
    }
```

#### Step 4.2: Update O3VsHaikuExperiment to Use New Engine
**File**: `experiments/o3_vs_haiku_baseline.py`
```python
# Replace the hardcoded _run_negotiation method with:
async def _run_negotiation(self, 
                         experiment_id: str,
                         agents: List,
                         env,
                         preferences: Dict[str, Any],
                         config: Dict[str, Any]) -> ExperimentResults:
    """Run negotiation using the new modular engine."""
    
    # Convert legacy config to new negotiation config
    negotiation_config = NegotiationConfig(
        n_agents=config["n_agents"],
        m_items=config["m_items"],
        t_rounds=config["t_rounds"],
        gamma_discount=config["gamma_discount"],
        max_reflection_chars=config.get("max_reflection_chars", 2000)
    )
    
    # Use the new modular engine
    engine = StandardNegotiationEngine(negotiation_config)
    result = await engine.run_negotiation(agents, env, preferences)
    
    # Convert to legacy result format for backward compatibility
    return ExperimentResults(
        experiment_id=experiment_id,
        timestamp=time.time(),
        config=config,
        consensus_reached=result.consensus_reached,
        final_round=result.final_round,
        winner_agent_id=result.winner_agent_id,
        final_utilities=result.final_utilities,
        conversation_logs=[asdict(msg) for msg in result.conversation_logs],
        strategic_behaviors=result.strategic_behaviors,
        # ... other fields
    )
```

### Phase 5: Testing and Validation

#### Step 5.1: Create Engine Unit Tests
**File**: `tests/test_negotiation_engine.py`
```python
class TestNegotiationEngine:
    """Test the modular negotiation engine."""
    
    async def test_full_negotiation_flow(self):
        """Test complete negotiation with real agents."""
        
    async def test_consensus_detection(self):
        """Test consensus tracking."""
        
    async def test_utility_calculations(self):
        """Test vector and matrix preference utilities."""
        
    async def test_phase_isolation(self):
        """Test that phases can be run independently."""
```

#### Step 5.2: Integration Tests
**File**: `tests/test_parameterized_integration.py`
```python
class TestParameterizedIntegration:
    """Test parameterized experiment with new engine."""
    
    async def test_o3_vs_haiku_compatibility(self):
        """Ensure results match old O3VsHaikuExperiment."""
        
    async def test_different_configurations(self):
        """Test various agent/item/round combinations."""
```

#### Step 5.3: Regression Testing
- Run existing O3 vs Haiku experiments with new engine
- Compare results to baseline from old hardcoded system
- Ensure API call patterns match (same number of calls, same costs)

## Implementation Timeline

### Week 1: Foundation
- [ ] Create base interfaces and result types
- [ ] Extract ConversationManager from existing code
- [ ] Unit tests for communication components

### Week 2: Phase Extraction  
- [ ] Extract and modularize all negotiation phases
- [ ] Create PhaseManager with proper async flow
- [ ] Test individual phases in isolation

### Week 3: Engine Integration
- [ ] Implement StandardNegotiationEngine orchestrator
- [ ] Extract consensus tracking and utility calculation
- [ ] Integration testing with existing agents

### Week 4: System Integration
- [ ] Update parameterized experiment to use new engine
- [ ] Update O3VsHaikuExperiment to use new engine  
- [ ] Regression testing and validation

### Week 5: Polish and Documentation
- [ ] Performance optimization
- [ ] Error handling and edge cases
- [ ] Update documentation and usage guides

## Benefits of This Refactoring

### Immediate Benefits
1. **Real API Calls**: Parameterized experiments will make actual LLM calls
2. **Code Reuse**: No duplication between different experiment types
3. **Consistency**: All experiments use the same proven negotiation logic

### Long-term Benefits  
1. **Extensibility**: Easy to add new negotiation phases or consensus mechanisms
2. **Testability**: Each component can be unit tested independently
3. **Maintainability**: Changes to negotiation logic only need to be made once
4. **Flexibility**: Support for different agent types, preference systems, and experiment configurations

### Research Benefits
1. **Reproducibility**: Consistent negotiation behavior across all experiments
2. **Comparability**: Results from different configurations are directly comparable
3. **Scalability**: Easy to run large-scale parameter sweeps
4. **Analysis**: Rich event data for strategic behavior analysis

## Risks and Mitigation

### Risk: Performance Regression
**Mitigation**: Benchmark new engine against existing system, optimize bottlenecks

### Risk: Breaking Changes
**Mitigation**: Maintain backward compatibility adapters, gradual migration

### Risk: Increased Complexity
**Mitigation**: Comprehensive documentation, clear interfaces, thorough testing

### Risk: API Cost Explosion  
**Mitigation**: Maintain existing rate limiting, cost estimation, and monitoring

## Success Criteria

1. **Functional**: Parameterized experiments make real API calls and produce valid results
2. **Compatible**: O3VsHaikuExperiment produces identical results using new engine
3. **Performant**: No significant performance regression vs. current system
4. **Maintainable**: Clean, documented, testable code architecture
5. **Extensible**: Easy to add new experiment types using the modular engine

This refactoring will transform the negotiation system from a hardcoded, single-use implementation into a flexible, reusable platform for multi-agent negotiation research.