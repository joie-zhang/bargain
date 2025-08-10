# Parameterized Experiment Usage Guide

## Overview

The parameterized experiment system (`run_parameterized_experiment.py`) replaces the old hardcoded `o3_vs_haiku_baseline.py` with a fully configurable YAML-based experiment runner for multi-agent negotiation research.

## Quick Start

```bash
# Run single experiment with existing config
python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml

# Test configuration without running (dry run)
python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml --dry-run

# Run batch of 5 experiments
python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml --batch 5
```

## Command Line Arguments

### Required Arguments (choose one):
- `--config PATH` or `-c PATH` - Path to YAML configuration file
- `--preset NAME` or `-p NAME` - Use preset configuration (`baseline_o3_vs_haiku`, `cooperative_matrix`, `scaling_laws_study`)
- `--list-configs` or `-l` - List all available configuration files

### Optional Arguments:
- `--batch N` or `-b N` - Run N experiments in batch mode (default: single experiment)
- `--experiment-id ID` or `-i ID` - Custom experiment ID (auto-generated if not provided)
- `--verbose` or `-v` - Enable verbose logging
- `--dry-run` or `-n` - Validate configuration without running experiment

## Configuration File Format

The system uses YAML configuration files located in `experiments/configs/`. Key sections:

### Environment Parameters
```yaml
environment:
  n_agents: 3           # Number of agents
  m_items: 5           # Number of items to allocate
  t_rounds: 6          # Maximum negotiation rounds
  gamma_discount: 0.9  # Discount factor for utility
  randomized_proposal_order: true  # Randomize speaking order
  require_unanimous_consensus: true
```

### Model Configuration
```yaml
models:
  providers:
    openai:
      api_key_env: "OPENAI_API_KEY"
      requests_per_minute: 20
    anthropic:
      api_key_env: "ANTHROPIC_API_KEY"
      requests_per_minute: 60

  available_models:
    o3:
      display_name: "OpenAI O3"
      provider: "openai"
      api_model_name: "o3"
      input_cost_per_1m: 60.0
      output_cost_per_1m: 180.0
```

### Agent Configuration
```yaml
agents:
  - agent_id: "agent_1_o3_strategic"
    model_id: "o3"
    temperature: 0.7
    max_output_tokens: 3000
    strategic_level: "balanced"
    system_prompt: |
      You are Agent 1 in a competitive negotiation...
```

### Preference Systems
```yaml
preferences:
  system_type: "vector"  # "vector" for competitive, "matrix" for cooperative
  competition_level:
    cosine_similarity: 0.95  # 0.0 = no competition, 1.0 = maximum competition
    tolerance: 0.05
```

## Usage Examples

### 1. Test Your Configuration
Always validate your config first:
```bash
python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml --dry-run
```

Expected output:
```
✅ Configuration is valid
Environment: 3 agents, 5 items, 6 rounds
Models: ['o3', 'claude-3-haiku']
Estimated cost: $2.45
```

### 2. Run Single Experiment
```bash
python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml
```

### 3. Run Batch Experiments
For statistical significance, run multiple experiments:
```bash
python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml --batch 10
```

### 4. Use Preset Configurations
For quick testing:
```bash
python run_parameterized_experiment.py --preset baseline_o3_vs_haiku
```

### 5. List Available Configurations
```bash
python run_parameterized_experiment.py --list-configs
```

### 6. Custom Experiment ID
```bash
python run_parameterized_experiment.py --config my_config.yaml --experiment-id "test_run_2024_01_15"
```

## Key Features

### 🔧 Full Parameterization
- **Environment**: Configure agents, items, rounds, discount factor
- **Models**: Support for O3, Claude, GPT-4, local models via OpenRouter
- **Preferences**: Vector (competitive) or matrix (cooperative) preference systems
- **Proposal Orders**: Randomized or fixed order with bias analysis

### 📊 Advanced Analysis
- **Strategic Behavior Detection**: Manipulation, gaslighting, anger detection
- **Proposal Order Effects**: Correlation analysis between speaking order and outcomes
- **Model Performance**: Win rates, utility statistics, exploitation evidence
- **Statistical Validation**: Batch runs for significance testing

### 💰 Cost Management
- **Cost Estimation**: Pre-run cost estimation
- **Rate Limiting**: Built-in API rate limiting per provider
- **Cost Limits**: Maximum cost per run validation

### 🔍 Multiple Output Formats
- **JSON**: Detailed machine-readable results
- **CSV**: Tabular data for analysis
- **Markdown**: Human-readable reports

## Result Interpretation

### Single Experiment Results
```
EXPERIMENT COMPLETED: o3_vs_haiku_baseline_parameterized_1705123456_7890
================================================================
Consensus Reached: ✅ Yes
Winner: agent_1_o3_strategic
Final Round: 4
Runtime: 45.2s
Estimated Cost: $2.34

Final Utilities:
👑   agent_1_o3_strategic: 8.50
     agent_2_haiku_cooperative: 6.20
     agent_3_haiku_analytical: 4.80

Strategic Behaviors Detected:
🎭 Manipulation: 2 instances
✅ No gaslighting detected
✅ No anger detected
```

### Batch Results
```
BATCH EXPERIMENT COMPLETED: batch_1705123456_7890
================================================================
Total Runs: 10
Success Rate: 90.0%
Average Rounds: 4.2
Average Cost: $2.45
Average Runtime: 47.8s

Model Performance:
o3: 80.0% win rate
claude-3-haiku: 20.0% win rate

✅ No significant order effects detected
```

## Troubleshooting

### Common Issues:

1. **Configuration Validation Failed**
   ```
   Configuration validation failed:
   - Number of agents (0) != n_agents (3)
   ```
   - Check that `agents` section matches `environment.n_agents`
   - Ensure all required model IDs are defined

2. **API Key Errors**
   - Set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables
   - Check API key validity and rate limits

3. **Cost Validation Errors**
   ```
   Estimated cost ($25.67) exceeds maximum allowed ($20.00)
   ```
   - Increase `validation.max_estimated_cost_per_run` in config
   - Or reduce batch size, tokens, or use cheaper models

4. **Missing Directories**
   - Ensure you're running from project root
   - Required directories: `experiments/configs`, `experiments/results`, `experiments/logs`, `negotiation`

## Environment Setup

### Required Environment Variables:
```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Required Dependencies:
The system requires the negotiation module and experiment configuration system to be properly set up in your project.

## Current Status & Known Issues

### ✅ Working Components:
- Configuration validation (`--dry-run`)
- Cost estimation
- Agent creation (OpenAI and Anthropic models)
- Environment setup
- Preference generation
- Basic negotiation flow initiation

### 🚧 Known Issues (In Progress):
1. **Division by Zero Error**: Currently encountering a division by zero error during negotiation analysis
   - Agents and environment create successfully
   - All negotiation rounds start properly
   - Error occurs in utility calculation or statistical analysis phase
   - **Status**: Under investigation

2. **Mock vs Real LLM Integration**: 
   - Current implementation uses simplified mock negotiation phases
   - Need to integrate with actual LLM conversation flows
   - **Status**: Planned for next iteration

### 🔧 Debug Mode:
If you encounter issues, check the logs in `experiments/logs/parameterized_experiments_YYYYMMDD.log`

## Best Practices

1. **Always dry-run first** to validate configuration and estimate costs
2. **Use batch experiments** for statistical significance (minimum 10 runs)
3. **Monitor API costs** especially with expensive models like O3
4. **Save configurations** for reproducible experiments
5. **Use version control** to track configuration changes
6. **Check logs** in `experiments/logs/` for debugging

### Quick Fixes for Common Issues:

#### Division by Zero Error (Current Known Issue):
```bash
# This error occurs during negotiation analysis
# Temporary workaround: Use simpler configurations with fewer parameters
# Full fix is in progress
```

## Next Steps

After running experiments, results are saved to:
- `experiments/results/[experiment_id]/` - Individual results
- `experiments/results/[batch_id]/` - Batch results

Use these files for:
- Statistical analysis of scaling laws
- Publication figure generation
- Further analysis of strategic behaviors
- Model comparison studies