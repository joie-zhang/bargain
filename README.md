# Scaling Laws for Strategic Interactions

A research codebase for studying how stronger Large Language Models (LLMs) exploit weaker LLMs in multi-agent negotiation environments. This project aims to establish scaling laws that describe the relationship between model capability and strategic behavior in competitive interactions.

## 🎯 Research Goal

**Primary Research Question**: How can we draw scaling laws that describe how stronger models exploit weaker models in negotiation environments?

This project investigates:
- Strategic exploitation patterns in LLM negotiations
- Scaling relationships between model capability and negotiation outcomes
- Behavioral analysis including manipulation, gaslighting, and strategic tactics
- Multi-agent interaction dynamics in competitive settings

**Target Publication**: ICLR Conference

## 📋 Project Overview

This codebase implements a multi-agent negotiation framework where LLMs negotiate over shared resources. The environment supports:

- **Configurable Parameters**:
  - `m` items: Number of items in the negotiation pool (default: 5)
  - `n` agents: Number of negotiating agents (default: 2)
  - `t` rounds: Maximum negotiation rounds (default: 10)
  - `γ` (gamma): Discount factor for rewards per round (default: 0.9)
  - Competition level: Controls preference overlap between agents (0-1)

- **Preference Systems**:
  - Vector preferences: Competitive scenarios with m-dimensional preference vectors
  - Matrix preferences: Cooperative/competitive scenarios with m×n preference matrices

- **Model Support**: Integration with multiple LLM providers including:
  - Anthropic (Claude models)
  - OpenAI (GPT models)
  - Qwen models (via OpenRouter)
  - And more via the OpenRouter API

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package installer)
- API keys for LLM providers (Anthropic, OpenAI, or OpenRouter)

### Installation

1. **Clone the repository**:
   ```bash
   cd /scratch/gpfs/DANQIC/jz4391/bargain
   ```

2. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment and install dependencies with uv**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   
   Or use uv's integrated workflow:
   ```bash
   uv sync
   ```

4. **Set up API keys** (as environment variables):
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   export OPENAI_API_KEY="your-key-here"
   # Or for OpenRouter:
   export OPENROUTER_API_KEY="your-key-here"
   ```

## 🧪 Running Experiments

### Basic Single Experiment

Run a single negotiation between two models:

```bash
python3 run_strong_models_experiment.py \
    --models claude-3-5-sonnet gpt-4o \
    --competition-level 0.95 \
    --num-items 5 \
    --max-rounds 10 \
    --num-runs 5 \
    --batch
```

**Parameters**:
- `--models`: Two model names to negotiate (see available models below)
- `--competition-level`: Preference overlap (0.0 = no competition, 1.0 = full competition)
- `--num-items`: Number of items in the negotiation pool
- `--max-rounds`: Maximum negotiation rounds
- `--num-runs`: Number of negotiation games to run
- `--batch`: Enable batch mode for multiple runs
- `--output-dir`: Custom output directory (optional)

### Batch Experiments with Scripts

The repository includes several experiment scripts in the `scripts/` directory:

#### 1. **Qwen Large Models Experiment** (Competition Level 1)

Run experiments comparing larger Qwen models (14B, 32B, 72B) against Claude-3.7-Sonnet:

```bash
bash scripts/run_qwen_large_models_comp1.sh
```

This script:
- Tests Qwen2.5-14B-Instruct, Qwen2.5-32B-Instruct, and Qwen2.5-72B-Instruct
- Each model runs 5 negotiations against Claude-3.7-Sonnet
- Competition level = 1 (full competition)
- Results saved to `experiments/results/`

#### 2. **Simple Experiment Runner**

Run a batch of experiments with parallel execution:

```bash
# First, generate experiment configurations
bash scripts/generate_single_config.sh

# Then run all experiments (with 4 parallel workers)
bash scripts/run_all_simple.sh 4
```

This workflow:
- Generates configuration files for multiple model pairs
- Runs experiments in parallel
- Automatically skips completed experiments
- Collects results into summary files

#### 3. **Single Experiment Script**

Run a single experiment from a configuration file:

```bash
bash scripts/run_single_experiment_simple.sh 0
```

Where `0` is the job ID corresponding to a config file in `experiments/results/scaling_experiment/configs/`.

### Available Models

Models are defined in `strong_models_experiment/configs.py`. Common models include:

- **Anthropic**: `claude-3-5-sonnet`, `claude-3-7-sonnet`, `claude-3-opus`, `claude-3-haiku`
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Qwen**: `Qwen2.5-72B-Instruct`, `Qwen2.5-32B-Instruct`, `Qwen2.5-14B-Instruct`, `Qwen2.5-3B-Instruct`
- **Others**: Available via OpenRouter API

## 📊 Results and Analysis

### Output Structure

Experiment results are saved to `experiments/results/` with the following structure:

```
experiments/results/
├── {model1}_vs_{model2}_config_unknown_runs{N}_comp{level}/
│   ├── run_1_experiment_results.json
│   ├── run_2_experiment_results.json
│   ├── ...
│   └── all_interactions.json
```

### Analyzing Results

Several analysis scripts are available:

```bash
# Analyze Qwen experiment results
python3 scripts/analyze_qwen_results.py

# Analyze order effects in negotiations
python3 scripts/analyze_order_effects.py
```

### Visualization

Generate visualizations of negotiation results:

```bash
python3 visualize_negotiation_results.py
```

## 🏗️ Project Structure

```
.
├── run_strong_models_experiment.py    # Main experiment runner
├── strong_models_experiment/           # Core experiment framework
│   ├── experiment.py                  # Main experiment class
│   ├── agents/                        # Agent implementations
│   ├── phases/                        # Negotiation phase handlers
│   ├── configs.py                     # Model configurations
│   └── analysis.py                    # Result analysis tools
├── negotiation/                       # Legacy negotiation framework
├── scripts/                           # Experiment automation scripts
│   ├── run_qwen_large_models_comp1.sh
│   ├── run_all_simple.sh
│   └── ...
├── experiments/
│   └── results/                       # Experiment outputs
├── tests/                             # Unit tests
└── requirements.txt                   # Python dependencies
```

## 🔬 Research Methodology

### Experiment Design

1. **Setup Phase**: Initialize negotiation environment with m items, n agents
2. **Preference Assignment**: Generate competitive or cooperative preference vectors/matrices
3. **Negotiation Rounds**: Agents propose allocations, vote, and reach agreements
4. **Reflection Phase**: Agents reflect on outcomes and strategies
5. **Analysis**: Quantitative metrics (utility, exploitation) and qualitative analysis (strategic behavior)

### Key Metrics

- **Utility Scores**: Per-agent utility from final allocations
- **Exploitation Metrics**: Difference in outcomes between stronger and weaker models
- **Strategic Behavior**: Qualitative analysis of manipulation, gaslighting, and tactical moves
- **Scaling Relationships**: Model capability vs. negotiation success

## 🧪 Example Experiments

### Scaling Study: Qwen Models vs Claude

Compare how different sizes of Qwen models perform against Claude:

```bash
bash scripts/run_qwen_large_models_comp1.sh
```

### Competition Level Variation

Test how competition level affects outcomes:

```bash
python3 run_strong_models_experiment.py \
    --models Qwen2.5-14B-Instruct claude-3-7-sonnet \
    --competition-level 0.5 \
    --num-runs 10 \
    --batch
```

### Three-Agent Negotiations

Run experiments with three agents:

```bash
bash scripts/run_all_3agent.sh
```

## 📝 Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project follows Python best practices. Key modules:
- `strong_models_experiment/`: Main experiment framework
- `negotiation/`: Legacy negotiation components (being phased out)
- `scripts/`: Automation and utility scripts

## 🤝 Contributing

This is a research codebase. For questions or contributions, please contact the research team.

## 📚 Additional Documentation

- **Project Context**: See `CLAUDE.md` for detailed project context and research goals
- **Implementation Roadmap**: See `ai_docs/implementation_roadmap.md` for development plans
- **Scaling Experiment Guide**: See `docs/SCALING_EXPERIMENT_GUIDE.md` for detailed experiment instructions

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

This research is conducted at Princeton University, utilizing the Della/PLI computing clusters.
