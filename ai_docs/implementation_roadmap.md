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
  - Setup → preferences → thinking → proposals → voting → resolution → reflection
  - Proper state transitions between rounds
  - Game termination conditions
- **Success Criteria**: Full negotiation runs from start to finish
- **Estimated Time**: 1-2 weeks

#### 7. Agent Action Logic
**Objective**: Accept/propose/walk away capabilities
- **Deliverables**:
  - Accept action (choose from current round proposals)
  - Propose action (new allocation suggestion)
  - Walk away action (exit negotiation)
  - Strategic reasoning prompts for each action type
- **Success Criteria**: Agents can perform all three actions appropriately
- **Estimated Time**: 1 week

#### 8. Voting and Consensus System
**Objective**: Anonymous voting and unanimous decisions
- **Deliverables**:
  - Anonymous voting mechanism
  - Unanimous decision detection
  - Item allocation when consensus reached
- **Success Criteria**: Items allocated correctly when unanimous votes occur
- **Estimated Time**: 1 week

#### 9. Utility Calculation Engine
**Objective**: Payoff computation for both preference types
- **Deliverables**:
  - Vector preference utility: Σ(preference_i × item_received_i)
  - Matrix preference utility: Σ(preference_matrix[i,j] × item_received_by_agent_j)
  - Discount factor application
- **Success Criteria**: Accurate utility calculations for all scenarios
- **Estimated Time**: 1 week

#### 10. Context Management System
**Objective**: Persistent agent memory across rounds
- **Deliverables**:
  - Agent memory persistence between rounds
  - Conversation history tracking
  - Strategic context accumulation
- **Success Criteria**: Agents remember previous rounds and adjust behavior
- **Estimated Time**: 1-2 weeks

### **Phase 3: Configuration & Baselines (Medium Priority)**

#### 11. Experiment Configuration System
**Objective**: Flexible parameter management
- **Deliverables**:
  - YAML/JSON configuration files
  - Parameter validation
  - Configuration templates for common scenarios
- **Success Criteria**: Easy experiment setup with different parameters
- **Estimated Time**: 1 week

#### 12. Baseline Agent Implementation
**Objective**: Random/greedy/cooperative agents
- **Deliverables**:
  - Random agent (random proposals and votes)
  - Greedy agent (maximize own utility)
  - Cooperative agent (maximize group utility)
- **Success Criteria**: Strong baselines for comparison with LLM agents
- **Estimated Time**: 1 week

#### 13. Metrics and Analysis System
**Objective**: Win rates, utilities, sentiment analysis
- **Deliverables**:
  - Individual utility tracking
  - Win rate calculations
  - Conversation sentiment analysis
  - Statistical significance testing
- **Success Criteria**: Comprehensive metrics for each negotiation
- **Estimated Time**: 1-2 weeks

#### 14. Exploitation Detection Framework
**Objective**: Strategic behavior identification
- **Deliverables**:
  - Qualitative analysis tools
  - Anger/gaslighting/manipulation detection
  - Strategic behavior categorization
  - Evidence extraction from conversations
- **Success Criteria**: Can identify and categorize strategic behaviors
- **Estimated Time**: 2-3 weeks

### **Phase 4: Infrastructure & Analysis (Low Priority)**

#### 15. Princeton Cluster Integration
**Objective**: SLURM job submission for Della/PLI
- **Deliverables**:
  - SLURM job scripts
  - Batch experiment submission
  - Resource estimation and management
- **Success Criteria**: Large-scale experiments run on Princeton clusters
- **Estimated Time**: 1-2 weeks

#### 16. Logging and Experiment Tracking
**Objective**: Comprehensive data collection
- **Deliverables**:
  - Complete conversation logs
  - Decision tree tracking
  - Experiment metadata storage
  - Result database
- **Success Criteria**: Full experiment reproducibility and analysis
- **Estimated Time**: 1 week

#### 17. Scaling Laws Analysis Pipeline
**Objective**: Statistical relationship analysis
- **Deliverables**:
  - Model capability vs. exploitation correlation analysis
  - Scaling law mathematical formulations
  - Statistical visualization tools
- **Success Criteria**: Clear scaling relationships between model strength and exploitation
- **Estimated Time**: 2-3 weeks

#### 18. Reproducibility Infrastructure
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