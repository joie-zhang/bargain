# O3 vs Claude Haiku Negotiation Pilot Study Specification

**Date**: 2025-01-31  
**Author**: Research Team  
**DRI**: [Your Name]  
**Status**: Understanding  
**Research Mode**: De-risk Sprint  
**Timeline**: 2-3 weeks (pilot phase)  

---

## 🎯 Quick Start: Most Important Questions

### 1. What specific question are you trying to answer?
**Does O3 systematically exploit Claude Haiku in competitive negotiation scenarios, and can we quantify this exploitation?**

What would change: If confirmed, this provides evidence for the core hypothesis that stronger LLMs exploit weaker ones, establishing the foundation for scaling law research.

### 2. Why should anyone care?
**Real-world impact**: As AI assistants become common in negotiations (business, diplomatic, consumer), parties with weaker AI may be systematically disadvantaged, leading to rash decisions and conflict escalation - exactly what negotiations are meant to prevent.

**Theory of change**: Quantify exploitation → Develop mitigation strategies → Prevent AI-assisted negotiation from exacerbating power imbalances → Reduce conflict escalation.

### 3. What's your riskiest assumption?
**Assumption**: O3 will actually exploit Haiku in negotiation settings.
**Risk**: Previous commons dilemma work showed stronger models being LESS self-interested.
**Quick test (<1 day)**: Run 10 simple 2-agent, 2-item negotiations with maximally competitive preferences. Check if O3 consistently wins.

### 4. What stage are you in?
**Understanding Stage**: Have clear hypothesis about O3 exploitation of Haiku. Ready to test systematically with controlled experiments.

### 5. What's your first experiment?
**Minimal viable test**: 2 agents (O3 vs Haiku), 5 items, competitive preferences (cosine similarity = 1), 5 rounds, 20 repetitions.
**What we'll learn**: Whether exploitation exists, its consistency, and basic strategic patterns.

### 6. What does success look like?
- **O3 wins >70% of negotiations** (vs 50% baseline)
- **Statistically significant utility differential** (p < 0.01)
- **Qualitative evidence of strategic manipulation** in conversation logs
- **Consistent results across multiple runs** (same random seed → same outcome)

### 7. What's your competition?
**Baselines to beat**:
- Random agent (50% win rate expected)
- Greedy agent (takes highest-value item each turn)
- Cooperative agent (maximizes total utility)
- Human negotiator benchmarks (from literature)

### 8. What resources do you need?
- **Compute**: Princeton Della cluster access for parallel runs
- **API access**: O3 and Claude Haiku API keys with sufficient quotas
- **Storage**: ~1GB for pilot study logs and results
- **Time**: 40-80 hours of compute time for statistical significance

### 9. What could go wrong?
1. **No exploitation detected** (Likelihood: Medium, Impact: High)
   - Plan B: Adjust competitive scenarios, try different model pairs
2. **API rate limits block experiments** (Likelihood: High, Impact: Medium)  
   - Plan B: Implement exponential backoff, use multiple keys
3. **Inconsistent results across runs** (Likelihood: Medium, Impact: High)
   - Plan B: Increase sample sizes, investigate random seed issues

### 10. Who's your audience?
**Primary**: ICLR conference reviewers and AI safety research community
**Decision they're making**: Whether this research direction is valuable for understanding AI alignment risks in multi-agent scenarios

---

## 1. Problem Statement & Theory of Change

### Core Research Question
As AI systems become more capable and are deployed as negotiation assistants, we need to understand how capability differences between AI systems translate to systematic advantages in strategic interactions. This pilot study tests the fundamental hypothesis that stronger language models will exploit weaker ones in competitive negotiation scenarios.

### Specific Problem
We lack quantitative understanding of how model capability differences manifest in strategic negotiation behavior. This creates risks for real-world deployment where AI-assisted negotiations could systematically disadvantage parties with weaker AI systems.

### Theory of Change
1. **Quantify exploitation**: Demonstrate and measure systematic exploitation in controlled negotiation environments
2. **Identify patterns**: Catalog specific strategic behaviors (manipulation, gaslighting, etc.)
3. **Develop scaling laws**: Extend to multiple model pairs to establish capability-exploitation relationships
4. **Design mitigations**: Use insights to develop fairer negotiation protocols
5. **Policy implications**: Inform AI governance around negotiation assistance systems

---

## 2. Related Work & Context

### Existing Literature Gaps
- **Negotiation Arena**: Different setup (human evaluation focus)
- **LLM Deliberation**: "Cooperation, Competition, and Maliciousness" paper has different experimental design
- **GovSim**: Commons dilemma focus, but found stronger models less self-interested (opposite of our hypothesis)

### Key Insights from GovSim Analysis
- Multi-agent architecture patterns (persona system, memory management)
- Configuration management for multi-LLM experiments  
- Logging and result tracking systems
- BUT: Commons dilemmas may not elicit strategic behavior due to training data solutions

### Why Negotiation is Different
- Less likely to have exact answers in training data
- Forces strategic reasoning rather than default cooperation
- Allows fine-tuned control of competition/cooperation incentives

---

## 3. Approach & Methodology

### Experimental Design

#### Environment Specification
```
Agents: n = 2 (O3, Claude Haiku)
Items: m = 5 (Item1, Item2, Item3, Item4, Item5)  
Rounds: t = 5 (one item per round)
Discount factor: γ = 0.95
Preference type: Vector (competitive)
Information: Secret preferences
```

#### Preference Generation
- Each agent gets m-dimensional preference vector
- Values sampled from [1, 10] uniform distribution
- Competitive scenario: Cosine similarity ≈ 1.0 (identical preferences)
- Randomized across experiments to ensure generality

#### Round Structure
1. **Prompt Phase**: Both agents receive identical setup prompt
2. **Thinking Phase**: Each agent privately reasons about strategy
3. **Proposal Phase**: Agents speak in randomized order (propose/accept/walk-away)
4. **Voting Phase**: Anonymous votes for proposals
5. **Resolution Phase**: Unanimous vote → item allocated, payoffs computed
6. **Reflection Phase**: Agents update internal context

#### Agent Actions
- **Propose**: "I propose Agent X gets Item Y"
- **Accept**: "I accept [previous proposal]"  
- **Walk Away**: "I withdraw from this round"

#### Success Conditions
- **Unanimous vote**: Proposal passes, item allocated
- **No consensus**: Round ends, no allocation, proceed to next round

### Technical Implementation

#### Multi-Agent Architecture (inspired by GovSim)
```python
class NegotiationAgent:
    def __init__(self, model, agent_id, preferences):
        self.model = model  # O3 or Claude Haiku
        self.agent_id = agent_id
        self.preferences = preferences  # m-dimensional vector
        self.memory = ConversationMemory()
        
    def think(self, context) -> Reasoning:
        # Private reasoning step
        
    def act(self, context) -> Action:
        # Propose/Accept/Walk-away decision
        
    def vote(self, proposals) -> Vote:
        # Anonymous voting phase
        
    def reflect(self, round_outcome) -> None:
        # Update memory and strategy
```

#### Configuration Management
```yaml
experiment:
  name: "o3_vs_haiku_pilot"
  agents: 2
  items: 5
  rounds: 5
  discount_factor: 0.95
  
models:
  agent_0:
    name: "o3"
    api_key: "${O3_API_KEY}"
    temperature: 0.7
  agent_1:
    name: "claude-haiku"
    api_key: "${ANTHROPIC_API_KEY}"
    temperature: 0.7

preferences:
  type: "vector"
  distribution: "uniform"
  range: [1, 10]
  similarity: "competitive"  # cosine_sim ≈ 1.0
```

---

## 4. Success Criteria & Metrics

### Primary Metrics

#### Quantitative Measures
1. **Win Rate**: Percentage of negotiations where O3 achieves higher utility than Haiku
2. **Utility Differential**: Mean difference in final utilities (O3_utility - Haiku_utility)
3. **Item Acquisition**: Number of items obtained by each agent
4. **Strategic Timing**: When each agent makes successful proposals (early vs late rounds)

#### Qualitative Measures  
1. **Manipulation Tactics**: Analysis of conversation logs for persuasion techniques
2. **Emotional Language**: Sentiment analysis and detection of anger/gaslighting
3. **Strategic Reasoning**: Quality of agent reasoning in "thinking" phases
4. **Consistency**: Whether strategies remain consistent across similar scenarios

### Statistical Validation
- **Sample Size**: Minimum 50 negotiation sessions for statistical power
- **Significance Level**: p < 0.01 for primary hypothesis tests
- **Effect Size**: Cohen's d > 0.5 for practically significant differences
- **Confidence Intervals**: 95% CIs for all key metrics

### Success Thresholds
- **Strong Evidence**: O3 wins >70% with p < 0.001
- **Moderate Evidence**: O3 wins >60% with p < 0.01  
- **Null Result**: Win rate 45-55% (within noise)

---

## 5. Implementation Plan

### Phase 1: Environment Setup (Week 1)
- [ ] Implement basic negotiation environment
- [ ] Set up O3 and Claude Haiku API integrations
- [ ] Create preference generation system
- [ ] Build logging and result tracking
- [ ] Test with 5 sample negotiations

### Phase 2: Pilot Experiments (Week 2)
- [ ] Run 50 negotiation sessions with competitive preferences
- [ ] Implement conversation logging and analysis
- [ ] Generate preliminary statistical analysis
- [ ] Identify technical issues and fix bugs

### Phase 3: Analysis & Validation (Week 3)
- [ ] Statistical significance testing
- [ ] Qualitative analysis of strategic behaviors
- [ ] Results documentation and visualization
- [ ] Prepare for scaling to larger study

### Technical Milestones
1. **Environment functional**: Agents can complete full negotiations
2. **API integration stable**: No failures due to rate limits or errors
3. **Results reproducible**: Same random seed → identical outcome
4. **Statistical pipeline**: Automated analysis of experimental results

---

## 6. Risk Assessment & Mitigation

### High-Impact Risks

#### 1. No Exploitation Detected
- **Probability**: 30%
- **Impact**: Invalidates core hypothesis
- **Early Detection**: Check first 10 experiments for any win rate skew
- **Mitigation**: Adjust competitive scenarios, try extreme preference differences
- **Plan B**: Pivot to cooperative scenarios or different model pairs

#### 2. Results Not Reproducible  
- **Probability**: 40%
- **Impact**: Cannot establish scientific validity
- **Early Detection**: Run identical experiments with same random seeds
- **Mitigation**: Implement deterministic random number generation
- **Plan B**: Focus on statistical trends rather than exact reproducibility

#### 3. API Rate Limits Block Progress
- **Probability**: 60%
- **Impact**: Cannot complete planned experiments
- **Early Detection**: Monitor API usage and response times
- **Mitigation**: Implement exponential backoff, acquire multiple API keys
- **Plan B**: Scale down experiment size or use cached model responses

### Medium-Impact Risks

#### 4. Qualitative Analysis Too Subjective
- **Probability**: 50% 
- **Impact**: Cannot make strong claims about strategic behaviors
- **Mitigation**: Develop systematic coding scheme, use multiple evaluators
- **Plan B**: Focus on quantitative metrics only

#### 5. Preference System Doesn't Elicit Competition
- **Probability**: 25%
- **Impact**: May not observe strategic behaviors
- **Mitigation**: Test multiple preference similarity levels
- **Plan B**: Switch to matrix preferences for more complex interactions

---

## 7. Expected Outcomes & Next Steps

### Potential Results

#### Scenario A: Strong Exploitation Evidence (40% probability)
- O3 wins >70% with clear strategic manipulation
- **Next Steps**: Scale to multiple model pairs, develop scaling laws
- **Publication Path**: Strong ICLR submission with novel findings

#### Scenario B: Moderate Evidence (35% probability)  
- O3 wins 60-70% with some strategic behaviors
- **Next Steps**: Refine experimental design, increase sample sizes
- **Publication Path**: Workshop paper, then extended study

#### Scenario C: Null Results (25% probability)
- No significant difference in win rates
- **Next Steps**: Try different scenarios (cooperative, mixed), other model pairs
- **Publication Path**: Negative results paper on limits of exploitation hypothesis

### Extensions for Full Study
1. **Multi-Model Matrix**: Test all pairwise combinations (O3, GPT-4, Claude Opus, etc.)
2. **Preference Variations**: Vector vs matrix preferences, secret vs known
3. **Parameter Sensitivity**: Vary m, n, t, γ systematically  
4. **Strategic Complexity**: Add deception, coalition formation scenarios
5. **Human Baselines**: Compare AI vs human negotiation performance

---

## 8. Resource Requirements

### Computational Resources
- **API Costs**: ~$500 for 50 experiments (estimated)
- **Princeton Cluster**: 10-20 GPU hours for parallel execution
- **Storage**: 1GB for logs, results, and analysis notebooks

### Time Investment
- **Development**: 40 hours (environment + analysis tools)
- **Experiments**: 20 hours (runtime + monitoring)  
- **Analysis**: 20 hours (statistics + qualitative coding)
- **Documentation**: 10 hours (results + next steps)
- **Total**: ~90 hours over 3 weeks

### Technical Dependencies
- Python 3.9+, PyTorch (for local models)
- OpenAI API (O3 access), Anthropic API (Claude Haiku)
- Jupyter notebooks for analysis
- SLURM integration for cluster computing
- Hydra for configuration management

---

## 9. Validation & Quality Control

### Experimental Validation
- [ ] **Reproducibility**: Same seed → identical results
- [ ] **Sanity Checks**: Random agent wins ~50% vs random agent
- [ ] **API Validation**: No failed calls or malformed responses
- [ ] **Preference Validation**: Generated preferences match intended similarity

### Analysis Validation  
- [ ] **Statistical Power**: Sufficient sample size for detecting medium effects
- [ ] **Multiple Comparisons**: Bonferroni correction for multiple tests
- [ ] **Qualitative Reliability**: Inter-rater agreement >0.8 for strategic behavior coding
- [ ] **Visualization**: Clear plots for all key metrics

### Code Quality
- [ ] **Unit Tests**: All utility functions tested
- [ ] **Integration Tests**: End-to-end negotiation pipeline
- [ ] **Documentation**: Clear README and API documentation
- [ ] **Version Control**: All code and configs in git

---

## 10. Literature Review & Citations

### Core Papers to Review
1. **GovSim**: "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents"
2. **Negotiation Arena**: [Find specific citation]
3. **LLM Deliberation**: "Cooperation, Competition, and Maliciousness" 
4. **Multi-Agent Communication**: Survey papers on LLM agent interactions
5. **Game Theory & AI**: Classical papers on strategic behavior in games

### Gap Analysis
- Most existing work focuses on cooperation or commons dilemmas
- Limited research on pure competitive negotiation scenarios  
- No systematic study of capability-based exploitation in AI systems
- Opportunity to establish new benchmark and evaluation framework

---

## 11. Ethical Considerations

### Research Ethics
- **Transparency**: Open source code and datasets where possible
- **Reproducibility**: Provide sufficient detail for replication
- **Responsible Disclosure**: Highlight potential negative applications

### Broader Implications
- **AI Safety**: Understanding exploitation helps design safer AI systems
- **Social Impact**: Results may inform AI governance and policy
- **Dual Use**: Could be misused to build more exploitative AI systems
- **Mitigation**: Focus on defense and detection rather than attack optimization

---

This specification provides a complete roadmap for the O3 vs Claude Haiku pilot study, establishing the foundation for the broader research program on LLM exploitation in negotiation environments.