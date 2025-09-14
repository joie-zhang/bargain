# Multi-Agent Negotiation Research Implementation Roadmap

**Research Question**: How can we draw scaling laws that describe how stronger models exploit weaker models in negotiation environments?

**Target Publication**: ICLR Conference

## Overview

This roadmap outlines the step-by-step implementation plan for building a multi-agent negotiation environment to test whether stronger LLMs systematically exploit weaker LLMs through strategic behaviors including anger, gaslighting, and manipulation tactics.

## Core Research Hypothesis

Stronger LLMs will systematically exploit weaker LLMs in negotiation settings through:
- Anger and gaslighting tactics
- Strategic manipulation to ensure stronger models win more often
- Consistent disadvantaging of weaker models

## Implementation Phases

### **Phase 1: Core Infrastructure (High Priority)**

#### 1. Environment Architecture Design
**Objective**: Design the foundational negotiation framework
- **Deliverables**: 
  - Core environment class with configurable parameters (m items, n agents, t rounds, γ discount factor)
  - Item pool management system
  - Round progression logic
- **Success Criteria**: Environment can initialize with arbitrary m, n, t, γ values
- **Estimated Time**: 1-2 weeks

#### 2. Multi-Agent Communication System
**Objective**: Build agent interaction capabilities
- **Deliverables**:
  - LangGraph or equivalent multi-agent orchestration
  - Message passing between agents
  - Turn management and randomized speaking order
- **Success Criteria**: Agents can communicate in structured rounds
- **Estimated Time**: 1-2 weeks

#### 3. Preference System Implementation
**Objective**: Create competitive/cooperative preference mechanisms
- **Deliverables**:
  - Vector preferences (m-dimensional, values 0-10)
  - Matrix preferences (m×n for cooperative scenarios)
  - Cosine similarity calculation for competition levels
  - Secret vs. commonly known preference options
- **Success Criteria**: Both preference types generate correct utility calculations
- **Estimated Time**: 1 week

#### 4. Multi-LLM Integration
**Objective**: Enable different models per agent
- **Deliverables**:
  - API integration for O3, Claude Haiku, GPT-4, etc.
  - Model assignment configuration system
  - Rate limiting and error handling
- **Success Criteria**: Can run experiments with different models per agent
- **Estimated Time**: 1 week

#### 5. First Experiment Implementation
**Objective**: O3 vs Claude Haiku baseline test
- **Deliverables**:
  - 3-player, 5-item negotiation setup
  - Highly competitive preferences (cosine similarity ≈ 1)
  - Basic win rate tracking
- **Success Criteria**: Complete one full negotiation and determine winner
- **Estimated Time**: 1 week

### **Phase 2: Negotiation Mechanics (Medium Priority)**

#### 6. Round Flow Implementation
**Objective**: Complete negotiation cycle
- **Deliverables**:
  - **Phase 1: Game Initialization**
    - 1A. Game Setup Phase: Give each agent identical opening prompt explaining game rules and mechanics
    - 1B. Private Preference Assignment: Assign each agent their individual secret preferences
  - **Phase 2: Discussion & Planning**
    - 2. Public Discussion Phase: Agents engage in open discussion about their preferences (may be strategic)
    - 3. Private Thinking Phase: Each agent uses private scratchpad to plan their proposal strategy
  - **Phase 3: Proposal Submission**
    - 4A. Proposal Phase: Agents go in randomized order to submit JSON proposals with reasoning to public audience
    - 4B. Proposal Enumeration: Number and display all proposals from 1 to n (where n = number of agents)
  - **Phase 4: Voting Process**
    - 5A. Private Voting: Agents submit votes privately (not visible to other agents)
    - 5B. Vote Revelation: Make all votes public after collection
  - **Phase 5: Decision & Transition**
    - 6A. Consensus Detection: Check if any proposal has unanimous support
    - 6B. Allocation & Termination: If unanimous support + all items allocated → terminate game
    - 7A. Individual Reflection: Each agent privately reflects on round outcomes
    - 7B. Memory Update: Agents retain key takeaways in internal memory for future rounds
    - 8. Round Transition: Advance to next round (repeat full process until round T)
    - 9. Max Rounds Termination: Terminate game when reaching maximum round limit T
  - Proper state transitions between rounds
  - Game termination conditions
- **Success Criteria**: Full negotiation runs from start to finish
- **Estimated Time**: 1-2 weeks

#### 7. Utility Calculation Engine
**Objective**: Payoff computation for both preference types
- **Deliverables**:
  - Vector preference utility: Σ(preference_i × item_received_i)
  - Matrix preference utility: Σ(preference_matrix[i,j] × item_received_by_agent_j)
  - Discount factor application
- **Success Criteria**: Accurate utility calculations for all scenarios
- **Estimated Time**: 1 week

#### 8. Experiment Parameterization System (Broken into Sub-tasks)

**Phase 8A: Configuration Foundation (Tasks 8.1-8.6)**

#### 8.1 Experiment Hyperparameter Logging
**Objective**: Ensure comprehensive experiment hyperparameter storage in JSON output files
- **Status**: ✅ **PARTIALLY COMPLETE** - Basic config storage exists in experiment results (see `o3_haiku_1753987978_3208_individual.json`)
- **Current Parameters Stored**: `m_items`, `n_agents`, `t_rounds`, `gamma_discount`, `competition_level`, `known_to_all`, `random_seed`
- **Remaining Deliverables**: 
  - Add model-specific hyperparameters (temperature, max_tokens, reasoning_steps)
  - Add preference system configuration (vector vs matrix, preference values)
  - Add proposal order tracking configuration
  - Add experiment metadata (start_time, duration, cluster_config if applicable)
  - Add validation that all hyperparameters from YAML config are preserved in JSON output
- **Success Criteria**: Every experiment parameter that could affect results is stored in the JSON output for full reproducibility
- **Estimated Time**: 2-3 days

#### 8.2 Basic YAML Schema Structure
**Objective**: Complete foundational YAML configuration structure
- **Status**: ✅ **PARTIALLY COMPLETE** - Model schema exists in `experiments/configs/model_config_schema.yaml`
- **Remaining Deliverables**: Add missing sections for environment, preferences, and analysis to existing schema
- **Success Criteria**: Complete YAML schema with all experiment sections that can be loaded and validated
- **Estimated Time**: 1-2 days

#### 8.3 Number of Agents (n) Configuration
**Objective**: Add configurable number of agents parameter
- **Deliverables**: 
  - `environment.num_agents` parameter in YAML schema
  - Validation rules for minimum/maximum agents (2-6 as per existing validation_rules)
  - Integration with agent configuration section
- **Success Criteria**: Can specify and validate number of agents via config
- **Estimated Time**: 1-2 days

#### 8.4 Number of Items (m) Configuration  
**Objective**: Add configurable number of items parameter
- **Deliverables**:
  - `environment.num_items` parameter in YAML schema
  - Item pool generation based on config
  - Validation rules for reasonable item counts
- **Success Criteria**: Can specify number of negotiable items via config
- **Estimated Time**: 1-2 days

#### 8.5 Number of Rounds (t) Configuration
**Objective**: Add configurable maximum rounds parameter
- **Deliverables**:
  - `environment.max_rounds` parameter in YAML schema
  - Round progression logic integration
  - Validation for reasonable round limits
- **Success Criteria**: Can specify maximum negotiation rounds via config
- **Estimated Time**: 1-2 days

#### 8.6 Discount Factor (γ) Configuration
**Objective**: Add configurable discount factor parameter
- **Deliverables**:
  - `environment.discount_factor` parameter in YAML schema
  - Integration with utility calculation system
  - Validation for discount factor range (0.0-1.0)
- **Success Criteria**: Can specify discount factor for time-based utility decay via config
- **Estimated Time**: 1-2 days

**Phase 8B: Model Configuration - OpenAI (Tasks 8.7-8.11)**

#### 8.7 OpenAI O3 Model Integration
**Objective**: Add O3 model support to config system
- **Deliverables**: O3 model configuration, API integration, rate limiting
- **Success Criteria**: Can run experiments with O3 agents via config
- **Estimated Time**: 2-3 days

#### 8.8 OpenAI GPT-4 Model Integration
**Objective**: Add GPT-4 model support to config system
- **Status**: ✅ **COMPLETE** - GPT-4o already defined in model_config_schema.yaml
- **Remaining Deliverables**: Verify integration works with experiment runner
- **Success Criteria**: Can run experiments with GPT-4 agents via config
- **Estimated Time**: 1 day (verification only)

#### 8.9 Additional OpenAI Models (GPT-4o, gpt-oss)
**Objective**: Complete OpenAI model suite
- **Status**: ✅ **PARTIALLY COMPLETE** - GPT-4o exists, need to add O3-mini and other variants
- **Remaining Deliverables**: Add missing OpenAI model variants to schema
- **Success Criteria**: All OpenAI models configurable and functional
- **Estimated Time**: 1-2 days

**Phase 8C: Model Configuration - Claude (Tasks 8.10-8.12)**

#### 8.10 Claude Haiku Model Integration
**Objective**: Add Claude Haiku support to config system
- **Status**: ✅ **COMPLETE** - Claude-3-haiku already defined in model_config_schema.yaml
- **Remaining Deliverables**: Verify integration works with experiment runner
- **Success Criteria**: Can run experiments with Haiku agents via config
- **Estimated Time**: 1 day (verification only)

#### 8.11 Claude Sonnet Model Integration
**Objective**: Add Claude Sonnet support to config system
- **Status**: ✅ **COMPLETE** - Claude-3-sonnet already defined in model_config_schema.yaml
- **Remaining Deliverables**: Verify integration works with experiment runner
- **Success Criteria**: Can run experiments with Sonnet agents via config
- **Estimated Time**: 1 day (verification only)

#### 8.12 Claude Opus Model Integration
**Objective**: Add Claude Opus support to config system
- **Deliverables**: Opus model configuration and integration (missing from current schema)
- **Success Criteria**: Can run experiments with Opus agents via config
- **Estimated Time**: 2-3 days

**Phase 8D: Model Configuration - Other APIs (Tasks 8.13-8.15)**

#### 8.13 Llama via OpenRouter Integration
**Objective**: Add Llama model support through OpenRouter
- **Status**: ✅ **COMPLETE** - Llama-3-70b already defined in model_config_schema.yaml
- **Remaining Deliverables**: Verify integration works with experiment runner
- **Success Criteria**: Can run experiments with Llama agents via config
- **Estimated Time**: 1 day (verification only)

#### 8.14 Gemini API Integration
**Objective**: Add Gemini model support
- **Status**: ✅ **COMPLETE** - Multiple Gemini models already defined in model_config_schema.yaml
- **Remaining Deliverables**: Verify integration works with experiment runner
- **Success Criteria**: Can run experiments with Gemini agents via config
- **Estimated Time**: 1 day (verification only)

#### 8.15 Qwen via OpenRouter Integration
**Objective**: Add Qwen model support through OpenRouter
- **Status**: ✅ **COMPLETE** - Qwen-2.5-72b already defined in model_config_schema.yaml
- **Remaining Deliverables**: Verify integration works with experiment runner
- **Success Criteria**: Can run experiments with Qwen agents via config
- **Estimated Time**: 1 day (verification only)

**Phase 8E: Preference Systems (Tasks 8.16-8.18)**

#### 8.16 Vector Preferences Configuration
**Objective**: Add vector preferences to config system
- **Deliverables**: Vector preference configuration and implementation
- **Success Criteria**: Vector preferences fully configurable via YAML
- **Estimated Time**: 2-3 days

#### 8.17 Matrix Preferences Configuration
**Objective**: Add matrix preferences to config system
- **Deliverables**: Matrix preference configuration and implementation
- **Success Criteria**: Matrix preferences fully configurable via YAML
- **Estimated Time**: 2-3 days

#### 8.18 Competition Level Configuration
**Objective**: Add cosine similarity competition control
- **Deliverables**: Configurable cosine similarity for preference generation
- **Success Criteria**: Can generate preferences with specified competition levels
- **Estimated Time**: 2-3 days

**Phase 8F: Proposal Order Analysis (Tasks 8.19-8.21)**

#### 8.19 Randomized Proposal Order Tracking
**Objective**: Implement proposal order randomization and tracking
- **Deliverables**: Random proposal order with systematic tracking
- **Success Criteria**: Proposal order randomized and logged for each experiment
- **Estimated Time**: 3-4 days

#### 8.20 Win Rate Correlation Analysis
**Objective**: Analyze correlation between proposal order and outcomes
- **Deliverables**: Statistical analysis of proposal order effects on win rates
- **Success Criteria**: Can detect and measure proposal order bias
- **Estimated Time**: 4-5 days

#### 8.21 Ablation Study Framework
**Objective**: Framework for isolating proposal order effects
- **Deliverables**: Ablation study tools for proposal order analysis
- **Success Criteria**: Can run controlled studies of proposal order effects only
- **Estimated Time**: 4-5 days

**Phase 8G: Validation & Integration (Tasks 8.22-8.24)**

#### 8.22 Configuration Validation System
**Objective**: Comprehensive config validation
- **Status**: ✅ **PARTIALLY COMPLETE** - Basic validation_rules exist in model_config_schema.yaml
- **Remaining Deliverables**: Expand validation for environment and preference parameters
- **Success Criteria**: Invalid configurations caught with helpful error messages
- **Estimated Time**: 2-3 days

#### 8.23 Example Configuration Files
**Objective**: Create template configs for common scenarios
- **Deliverables**: Well-documented example YAML files for different experiment types
- **Success Criteria**: Users can easily create new experiments from templates
- **Estimated Time**: 2-3 days

#### 8.24 Main Script Refactoring
**Objective**: Refactor o3_vs_haiku_baseline.py to use full config system
- **Deliverables**: Single configurable experiment script
- **Success Criteria**: All experiments run through unified config-driven interface
- **Estimated Time**: 1 week

**Overall Phase 8 Success Criteria**:
- Single experiment script can run any model combination through YAML config
- Proposal order effects can be isolated and measured  
- Easy setup for systematic ablation studies across all parameters
- **Total Estimated Time**: 2-3 weeks
- **Priority**: High (needed for systematic exploration of parameter space)

#### 9. Context Management System
**Objective**: Persistent agent memory across rounds
- **Deliverables**:
  - Agent memory persistence between rounds
  - Conversation history tracking
  - Strategic context accumulation
- **Success Criteria**: Agents remember previous rounds and adjust behavior
- **Estimated Time**: 1-2 weeks

### **Phase 3: Configuration & Baselines (Medium Priority)**

#### 10. Experiment Configuration System
**Objective**: Flexible parameter management
- **Deliverables**:
  - YAML/JSON configuration files
  - Parameter validation
  - Configuration templates for common scenarios
- **Success Criteria**: Easy experiment setup with different parameters
- **Estimated Time**: 1 week

#### 11. Baseline Agent Implementation
**Objective**: Random/greedy/cooperative agents
- **Deliverables**:
  - Random agent (random proposals and votes)
  - Greedy agent (maximize own utility)
  - Cooperative agent (maximize group utility)
- **Success Criteria**: Strong baselines for comparison with LLM agents
- **Estimated Time**: 1 week

#### 12. Metrics and Analysis System
**Objective**: Win rates, utilities, sentiment analysis
- **Deliverables**:
  - Individual utility tracking
  - Win rate calculations
  - Conversation sentiment analysis
  - Statistical significance testing
- **Success Criteria**: Comprehensive metrics for each negotiation
- **Estimated Time**: 1-2 weeks

#### 13. Exploitation Detection Framework
**Objective**: Strategic behavior identification
- **Deliverables**:
  - Qualitative analysis tools
  - Anger/gaslighting/manipulation detection
  - Strategic behavior categorization
  - Evidence extraction from conversations
- **Success Criteria**: Can identify and categorize strategic behaviors
- **Estimated Time**: 2-3 weeks

### **Phase 4: Infrastructure & Analysis (Low Priority)**

#### 14. Princeton Cluster Integration
**Objective**: SLURM job submission for Della/PLI
- **Deliverables**:
  - SLURM job scripts
  - Batch experiment submission
  - Resource estimation and management
- **Success Criteria**: Large-scale experiments run on Princeton clusters
- **Estimated Time**: 1-2 weeks

#### 15. Logging and Experiment Tracking
**Objective**: Comprehensive data collection
- **Deliverables**:
  - Complete conversation logs
  - Decision tree tracking
  - Experiment metadata storage
  - Result database
- **Success Criteria**: Full experiment reproducibility and analysis
- **Estimated Time**: 1 week

#### 16. Scaling Laws Analysis Pipeline
**Objective**: Statistical relationship analysis
- **Deliverables**:
  - Model capability vs. exploitation correlation analysis
  - Scaling law mathematical formulations
  - Statistical visualization tools
- **Success Criteria**: Clear scaling relationships between model strength and exploitation
- **Estimated Time**: 2-3 weeks

#### 17. Reproducibility Infrastructure
**Objective**: Seed management and versioning
- **Deliverables**:
  - Random seed management
  - Environment versioning
  - Exact replication capability
  - Experiment audit trails
- **Success Criteria**: Perfect reproducibility of all experiments
- **Estimated Time**: 1 week

## Success Metrics

### Primary Metrics
- **Individual Utility**: Weighted sum of (preferences × items received)
- **Win Rate**: Frequency stronger model gets higher utility
- **Exploitation Evidence**: Qualitative analysis of strategic behaviors

### Statistical Requirements
- **Significance Level**: p < 0.001 for exploratory findings
- **Multi-Model Validation**: Critical findings verified across different LLM combinations
- **Baseline Comparison**: All results compared against strong baselines

## Risk Mitigation

### Riskiest Assumption
**Risk**: Stronger models may NOT exploit weaker ones
**Mitigation**: 
- Careful environment design to encourage strategic behavior
- Multiple competition/cooperation scenarios
- Strong baseline comparisons

### Technical Risks
- **Agent Communication Failures**: Implement robust error handling and retry logic
- **API Rate Limits**: Use multiple keys and exponential backoff
- **Reproducibility**: Comprehensive seed management and versioning

## Timeline Estimate

- **Phase 1**: 5-7 weeks (Core Infrastructure)
- **Phase 2**: 6-8 weeks (Negotiation Mechanics)  
- **Phase 3**: 5-7 weeks (Configuration & Baselines)
- **Phase 4**: 5-7 weeks (Infrastructure & Analysis)

**Total Estimated Timeline**: 21-29 weeks (5-7 months)

## Next Steps

1. Begin with **Environment Architecture Design** (Phase 1, Item 1)
2. Set up development environment and dependencies
3. Create initial project structure and configuration management
4. Implement and test core negotiation loop
5. Add first LLM agents and run initial experiments

## Publication Strategy

Target: ICLR Conference
- Focus on novel insights about AI-AI strategic interactions
- Emphasize scaling laws and exploitation patterns
- Include comprehensive ablation studies
- Provide strong baselines and statistical rigor

---

*This roadmap serves as the master plan for implementing the multi-agent negotiation research environment. Each phase builds upon the previous one, ensuring a systematic approach to answering the core research question about model exploitation in strategic interactions.*