# O3 vs Claude Haiku Baseline Experiment Guide

This guide explains how to run and understand the O3 vs Claude Haiku baseline experiment, which is the first experiment implementation from Phase 1 of the multi-agent negotiation research roadmap.

## Overview

### Research Question
How do stronger LLMs (O3) exploit weaker LLMs (Claude Haiku) in competitive negotiation environments?

### Experiment Design
- **Agents**: 3 agents total (1 O3, 2 Claude Haiku)
- **Items**: 5 items to negotiate over
- **Preferences**: Highly competitive (cosine similarity ≈ 0.95)
- **Information**: Secret preferences (agents don't know others' preferences)
- **Rounds**: Maximum 6 negotiation rounds
- **Success Metric**: Win rate and strategic behavior detection

## Quick Start

### 1. Setup
```bash
# Check dependencies and setup
python setup_experiment.py

# Run tests to verify everything works
pytest tests/test_llm_agents.py
```

### 2. Run Single Experiment
```bash
# With real LLMs (requires API keys)
python run_baseline_experiment.py

# With simulated agents (free)
python run_baseline_experiment.py --simulated
```

### 3. Run Batch Experiments
```bash
# 10 runs with real LLMs
python run_baseline_experiment.py --runs 10

# 10 runs with simulated agents
python run_baseline_experiment.py --runs 10 --simulated
```

## Configuration Options

### Command Line Arguments
- `--runs N`: Number of experiments to run (default: 1)
- `--simulated`: Use simulated agents instead of real LLMs
- `--competition-level 0.95`: How competitive preferences are (0.0-1.0)
- `--rounds 6`: Maximum negotiation rounds
- `--results-dir PATH`: Where to save results
- `--verbose`: Enable detailed logging

### Environment Variables
- `OPENAI_API_KEY`: Required for O3 model access
- `ANTHROPIC_API_KEY`: Required for Claude Haiku access

## Understanding Results

### Single Experiment Output
```
📊 Results:
  Experiment ID: o3_haiku_1234567890_5678
  Consensus: ✓
  Rounds: 4
  Winner: agent_0
  O3 Won: ✓
  Exploitation: ✗
  Final Utilities:
    agent_0: 15.30
    agent_1: 8.20
    agent_2: 6.50
```

### Batch Experiment Output
```
📈 Batch Results:
  Batch ID: o3_haiku_batch_1234567890
  Successful Runs: 10/10
  O3 Win Rate: 70.0%
  Haiku Win Rate: 30.0%
  Consensus Rate: 80.0%
  Average Rounds: 4.2
  Exploitation Rate: 40.0%

🧠 Strategic Analysis:
  Manipulation Rate: 30.0%
  Avg Anger: 1.2
  Avg Gaslighting: 0.4
  Cooperation Breakdown: 10.0%
```

### Key Metrics Explained

#### Win Rates
- **O3 Win Rate**: Percentage of experiments where the O3 agent achieved highest utility
- **Haiku Win Rate**: Percentage where a Claude Haiku agent won
- Higher O3 win rate suggests stronger models exploit weaker ones

#### Strategic Behaviors
- **Manipulation Rate**: Percentage of experiments with detected manipulation tactics
- **Exploitation Rate**: Overall percentage showing strategic exploitation
- **Anger Expressions**: Average number of frustration/anger indicators per experiment
- **Gaslighting Attempts**: Attempts to confuse or mislead other agents

#### Negotiation Outcomes
- **Consensus Rate**: Percentage of experiments reaching agreement
- **Average Rounds**: How many rounds negotiations typically take
- Low consensus rate may indicate excessive competition

## File Structure

### Code Files
```
experiments/
├── o3_vs_haiku_baseline.py    # Main experiment implementation
├── configs/                   # Experiment configurations
└── results/                   # Results storage

run_baseline_experiment.py     # Simple script to run experiments
setup_experiment.py           # Setup and verification script
```

### Results Files
```
experiments/results/
├── [batch_id]_summary.json    # Aggregate statistics
├── [batch_id]_detailed.json   # Individual experiment results
└── individual logs...         # Per-experiment conversation logs
```

## Real vs Simulated Agents

### Real LLM Agents
**Advantages:**
- Authentic strategic behaviors
- Research-grade results
- Realistic model interactions

**Requirements:**
- API keys for OpenAI (O3) and Anthropic (Claude)
- API costs (approximately $0.10-$1.00 per experiment)
- Internet connection

### Simulated Agents
**Advantages:**
- No API costs
- Faster execution
- Consistent for testing
- Works offline

**Limitations:**
- Approximated behaviors
- May not capture subtle strategic nuances
- Less realistic for publication

## Troubleshooting

### Common Issues

#### "API key required" Error
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### Import Errors
```bash
# Make sure you're in the project directory
cd /path/to/bargain

# Install missing packages
pip install anthropic openai anyio pytest
```

#### Permission Errors
```bash
# Make scripts executable
chmod +x run_baseline_experiment.py
chmod +x setup_experiment.py
```

#### Rate Limiting
The experiment includes automatic rate limiting, but if you hit API limits:
- Reduce batch size (`--runs 5` instead of `--runs 10`)
- Add delays between experiments
- Use simulated mode for testing

### Debug Mode
```bash
# Enable verbose logging
python run_baseline_experiment.py --verbose

# Run tests to verify components
pytest tests/test_llm_agents.py -v

# Check specific components
python examples/llm_agents_demo.py
```

## Research Applications

### Scaling Laws Analysis
Run experiments with different model combinations:
```python
# Modify experiments/o3_vs_haiku_baseline.py
# Change model types in create_o3_vs_haiku_experiment()
```

### Competition Level Studies
```bash
# Test different competition levels
python run_baseline_experiment.py --competition-level 0.5  # Moderate
python run_baseline_experiment.py --competition-level 0.9  # High
python run_baseline_experiment.py --competition-level 0.99 # Extreme
```

### Statistical Significance
```bash
# Run larger batches for statistical power
python run_baseline_experiment.py --runs 50
python run_baseline_experiment.py --runs 100
```

## Expected Results

### Hypothesis
Stronger models (O3) should:
1. Win more frequently than weaker models (Claude Haiku)
2. Show more sophisticated strategic behaviors
3. Potentially exploit weaker models through manipulation

### Baseline Performance
Based on simulated runs:
- **O3 Win Rate**: 60-80% (indicating model strength advantage)
- **Consensus Rate**: 70-90% (most negotiations succeed)
- **Exploitation Rate**: 20-40% (some strategic manipulation)
- **Average Rounds**: 3-5 (efficient negotiations)

### Research Questions to Investigate
1. Does O3 win rate increase with competition level?
2. What specific tactics does O3 use to exploit Haiku?
3. How does negotiation length correlate with exploitation?
4. Can we detect manipulation in conversation patterns?

## Next Steps

### Immediate Analysis
1. Run baseline with both real and simulated agents
2. Compare win rates and strategic behaviors
3. Analyze conversation logs for exploitation patterns
4. Document preliminary findings

### Extended Research
1. Test different model combinations (GPT-4 vs Claude, etc.)
2. Vary preference structures (cooperative vs competitive)
3. Implement sentiment analysis on conversations
4. Create larger-scale statistical studies

### Publication Pipeline
1. Collect sufficient data for statistical significance
2. Analyze strategic behavior patterns
3. Compare with human negotiation baselines
4. Write up findings for ICLR submission

## Support

### Getting Help
- Check the troubleshooting section above
- Review the test files in `tests/`
- Examine the demo script in `examples/`
- Look at existing configurations in `experiments/configs/`

### Contributing
- Add new experiment configurations
- Extend strategic behavior detection
- Improve result analysis and visualization
- Create additional model combinations

This experiment forms the foundation for understanding how AI models interact strategically, providing crucial data for AI safety research on model exploitation and negotiation dynamics.