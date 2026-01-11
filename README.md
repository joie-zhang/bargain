# Scaling Laws for Strategic Interactions

A research codebase for studying the effects of asymmetric agent capabilities in strategic scenarios, from cooperation to competition. This project investigates how differences in agent capabilities interact with the degree of alignment between agents' preferences to shape outcomes in multi-agent negotiation environments.

## 🎯 Research Questions

**Primary Research Question**: What are the effects of asymmetric agent capabilities in strategic scenarios, from cooperation to competition?

### Key Sub-Questions

#### 1. Overall Patterns and Interplay
- What's the overall pattern we see across agent capabilities and degree of alignment?
- What's the interplay between these two dimensions?
- Can we fit a line to these trends? What does that line look like? Does it allow us to extrapolate?
- Heatmaps are the primary way to present this overall picture, but we also flatten one dimension (capabilities or degree of alignment) for some plots/results

#### 2. Capability Level Dependencies
- Does it depend on the base level of capabilities we start from? (i.e., is the difference between a 20% capable and a 50% capable model really big compared to the difference between a 60% capable and a 90% capable model?)
- This requires separating out a few different sets of ego agents at different capability levels

#### 3. Model Dispositions
- Does it depend on the model's 'dispositions'? Or rather, where/how does it depend on that?
- Operationalized by seeing how much variance in results is explained by:
  - Model family/developer
  - Scores on relevant 'disposition' benchmarks

#### 4. Auxiliary Observations
- Other observations that help the reader understand the dynamics of what's happening, or are independently interesting (e.g., time to consensus plots)
- Quick ablations/sanity checking (can go in appendix) – these shouldn't require running the whole set of experiments again, just a representative scenario/subset:
  - What happens if we use a different capability metric?
  - What happens if agents negotiate for longer/shorter?
  - Etc.

#### 5. Scaling with Number of Agents (Secondary Question)
- How robust are the scaling trends above when we have multiple agents and the environment is more complex?
- Can multiple lower-capability agents team up against a higher-capability agent?
- More generally: imagine you have a set of n agents along a capability x-axis and a fixed competition level, where do the rewards (y-axis) tend to accrue most? Is this linear in capabilities or is there some interesting non-linearity here? Again: what does the trend/'scaling law' look like?
- This can be run via randomized/selective cross-play; we don't literally have to test for all agents at once, or for all combinations of agents

### Experimental Design

**Target**: Test across ~3 negotiation/bargaining style games, each a little bit different (technically) and with a different story, so that reviewers don't complain about lack of robustness. If we can't get 3 in time for submission, we aim for 2, then try to get a third ready for rebuttals.

**Target Publication**: ICML

## 📋 Project Overview

This codebase implements a modular multi-agent negotiation framework where LLMs negotiate in different game environments. The framework supports multiple game types, each with different mechanics and negotiation structures:

### Supported Game Environments

1. **Item Allocation Game** (`item_allocation`)
   - Discrete item allocation with preference vectors
   - Agents negotiate over m items in a shared pool
   - Preference systems:
     - Vector preferences: Competitive scenarios with m-dimensional preference vectors
     - Matrix preferences: Cooperative/competitive scenarios with m×n preference matrices

2. **Diplomatic Treaty Game** (`diplomacy`)
   - Multi-issue continuous negotiation with position/weight preferences
   - Agents negotiate over K continuous issues (values in [0,1])
   - Each agent has position preferences and importance weights
   - Control parameters: ρ (preference correlation), θ (interest overlap), λ (issue compatibility)

### Common Framework Parameters

- **Configurable Parameters** (game-specific):
  - `n` agents: Number of negotiating agents (default: 2)
  - `t` rounds: Maximum negotiation rounds (default: 10)
  - `γ` (gamma): Discount factor for rewards per round (default: 0.9)
  - Competition level: Controls preference overlap between agents (0-1), where 0 = full cooperation and 1 = full competition
  - Game type: Select which game environment to use (`item_allocation`, `diplomacy`, or future game types)

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

**Item Allocation Game**:
```bash
python3 run_strong_models_experiment.py \
    --game-type item_allocation \
    --models claude-3-5-sonnet gpt-4o \
    --competition-level 0.95 \
    --num-items 5 \
    --max-rounds 10 \
    --num-runs 5 \
    --batch
```

**Diplomatic Treaty Game**:
```bash
python3 run_strong_models_experiment.py \
    --game-type diplomacy \
    --models claude-3-5-sonnet gpt-4o \
    --competition-level 0.95 \
    --n-issues 5 \
    --max-rounds 10 \
    --num-runs 5 \
    --batch
```

**Common Parameters**:
- `--game-type`: Game environment to use (`item_allocation` or `diplomacy`, default: `item_allocation`)
- `--models`: Two model names to negotiate (see available models below)
- `--competition-level`: Preference overlap (0.0 = no competition/cooperation, 1.0 = full competition)
- `--max-rounds`: Maximum negotiation rounds
- `--num-runs`: Number of negotiation games to run
- `--batch`: Enable batch mode for multiple runs
- `--output-dir`: Custom output directory (optional)

**Game-Specific Parameters**:
- Item Allocation: `--num-items` (number of items in the negotiation pool)
- Diplomatic Treaty: `--n-issues` (number of continuous issues to negotiate)

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

Generate visualizations of negotiation results, including heatmaps showing capability vs. alignment:

```bash
python3 visualize_negotiation_results.py
```

## 🖥️ Interactive UI (Negotiation Viewer)

The project includes a Streamlit-based web UI for visualizing and analyzing negotiation experiments interactively.

### Features

The Negotiation Viewer provides three modes:

1. **📂 Post-hoc Analysis**: View and analyze completed experiments
   - Round-by-round timeline with color-coded phases
   - Agent comparison with utility metrics
   - Analytics dashboard with charts and statistics
   - Raw data explorer for detailed inspection
   - Preference visualization

2. **📊 Batch Comparison**: Compare multiple experiments side-by-side
   - Aggregate metrics across experiment runs
   - Cross-experiment analysis
   - Performance comparisons

3. **🔴 Live Streaming**: Monitor experiments as they run in real-time
   - Watch negotiations unfold live
   - Real-time updates as rounds complete

### Launching the UI

**Option 1: Using the launch script (recommended)**

```bash
# Default port (8501)
bash ui/run_viewer.sh

# Custom port
bash ui/run_viewer.sh --port 8080
```

**Option 2: Direct Streamlit command**

```bash
streamlit run ui/negotiation_viewer.py
```

**Option 3: With custom port**

```bash
streamlit run ui/negotiation_viewer.py --server.port 8080 --server.headless true
```

### Accessing the UI

Once launched, open your browser to:
- **Local**: `http://localhost:8501` (or your custom port)
- **Remote server**: `http://your-server-address:8501`

### UI Dependencies

The UI requires additional dependencies (already included in `requirements.txt`):
- `streamlit>=1.28.0`
- `plotly>=5.15.0` (for interactive charts)

If you haven't installed them yet:
```bash
pip install streamlit plotly
```

### Usage Tips

- **Selecting Experiments**: Use the sidebar to browse experiment results in `experiments/results/`
- **Filtering**: Use filters in the sidebar to focus on specific rounds, agents, or phases
- **Exporting**: Export data to CSV or Markdown for further analysis
- **Live Mode**: For live streaming, ensure experiments are running and writing to the results directory

## 🏗️ Project Structure

```
.
├── run_strong_models_experiment.py    # Main experiment runner
├── game_environments/                  # Modular game environment implementations
│   ├── base.py                        # GameEnvironment abstract base class
│   ├── item_allocation.py             # Item Allocation game
│   └── diplomatic_treaty.py            # Diplomatic Treaty game
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
├── ui/                                # Streamlit-based negotiation viewer
│   ├── negotiation_viewer.py          # Main UI application
│   ├── comparison_view.py             # Batch comparison view
│   ├── components.py                   # Reusable UI components
│   ├── analysis.py                    # Analysis utilities
│   └── run_viewer.sh                  # Launch script
├── experiments/
│   └── results/                       # Experiment outputs
├── tests/                             # Unit tests
└── requirements.txt                   # Python dependencies
```

## 🔬 Research Methodology

### Experiment Design

The framework uses a modular game environment system that supports multiple game types:

1. **Setup Phase**: Initialize negotiation environment (game type-specific: items for Item Allocation, issues for Diplomatic Treaty), n agents
2. **Preference Assignment**: Generate competitive or cooperative preferences based on competition level (game-specific format)
3. **Negotiation Rounds**: Agents propose allocations/agreements, vote, and reach agreements
4. **Reflection Phase**: Agents reflect on outcomes and strategies
5. **Analysis**: Quantitative metrics (utility, capability differences) and qualitative analysis (strategic behavior)

Each game type implements the same `GameEnvironment` interface, allowing experiments to run across different negotiation structures while maintaining consistent analysis.

### Key Metrics

- **Utility Scores**: Per-agent utility from final allocations
- **Capability Differences**: Outcomes across different capability levels
- **Strategic Behavior**: Qualitative analysis of negotiation tactics and behaviors
- **Scaling Relationships**: Model capability vs. negotiation success across different competition levels
- **Time to Consensus**: Auxiliary metric showing how long negotiations take

### Results Organization

Results are organized around two main dimensions:

**Option 1** (if not including number of agents):
- Non-reasoning models
- Reasoning models

**Option 2**:
- **Capabilities** ['quality' of agents]
  - Non-reasoning models
  - Reasoning models
- **Number** ['quantity' of agents]
  - 2-agent scenarios
  - Multi-agent scenarios (3+ agents)

## 🧪 Example Experiments

### Capability vs. Competition Level Study

Test how different capability levels interact with competition levels across game types:

**Item Allocation**:
```bash
python3 run_strong_models_experiment.py \
    --game-type item_allocation \
    --models Qwen2.5-14B-Instruct claude-3-7-sonnet \
    --competition-level 0.5 \
    --num-items 5 \
    --num-runs 10 \
    --batch
```

**Diplomatic Treaty**:
```bash
python3 run_strong_models_experiment.py \
    --game-type diplomacy \
    --models Qwen2.5-14B-Instruct claude-3-7-sonnet \
    --competition-level 0.5 \
    --n-issues 5 \
    --num-runs 10 \
    --batch
```

### Three-Agent Negotiations

Run experiments with three agents to study multi-agent dynamics:

```bash
bash scripts/run_all_3agent.sh
```

### Capability Level Variation

Test different base capability levels:

```bash
bash scripts/run_qwen_large_models_comp1.sh
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

This research was conducted at the ML Alignment & Theory Scholars Program as well as Princeton University, utilizing Princeton's Della/PLI computing clusters and compute funding from MATS and PLI.
