# All Files Reference

## Core Scripts

### Experiment Running
`scripts/generate_configs_both_orders.sh`:
Generates experiment configs. Weak model always first. 5 runs per config with seeds 42,123,456,789,101112.

`scripts/run_all_simple.sh`:
Main runner. No timeouts. Takes parallel job count as arg. Default 4.

`scripts/run_single_experiment_simple.sh`:
Runs one experiment by job ID. No timeout. Logs to experiments/results/scaling_experiment/logs/.

`scripts/collect_results.sh`:
Aggregates all results into all_results.json, summary.json, results.csv.

### Visualization

`visualization/create_mmlu_ordered_heatmaps.py`:
Main viz. Creates 4 heatmaps: utility diff, strong utility, sum welfare, baseline utility. Models ordered by MMLU-Pro.

`visualization/create_mmlu_ordered_heatmaps_filtered.py`:
Same but excludes negotiations ending at max rounds.

`visualization/analyze_convergence_rounds.py`:
Violin plots of rounds to convergence by model pair and competition level.

`visualization/analyze_token_usage.py`:
Token usage per negotiation phase. Cost analysis.

`visualization/analyze_number_of_runs_in_heatmap.py`:
Counts runs per heatmap cell. Verifies data completeness.

`visualization/create_model_family_bar_graphs.py`:
Bar charts by model family. Currently win rates. Needs utility gain version.

## Negotiation Core

`negotiation/agents.py`:
LLM agent implementation. Handles prompts, responses, memory.

`negotiation/environment.py`:
Negotiation environment. Items, preferences, rounds, voting.

`negotiation/prompts.py`:
System and user prompts for agents. Competition framing.

`negotiation/preferences.py`:
Preference generation. Cosine similarity for competition level.

`negotiation/llm_models.py`:
Model API interfaces. OpenAI, Anthropic, Google, XAI.

`negotiation/run_negotiation.py`:
Main entry point. Loads config, runs negotiation, saves results.

## Configuration

`negotiation/model_config.py`:
Model registry. Provider configs. Capability tracking.

`negotiation/config_integration.py`:
Config management. Model selection. Parameter validation.

## Documentation

`docs/SCALING_EXPERIMENT_GUIDE.md`:
Current workflow guide. Updated for simple runner.

`docs/guides/visualization_guide.md`:
Visualization documentation. Customization options.

`CLAUDE.md`:
AI assistant config. Project context. Research principles.

`README.md`:
Project overview. Quick start. Paper links.

## Data Directories

`experiments/results/scaling_experiment/configs/`:
JSON config files for each experiment.

`experiments/results/scaling_experiment/logs/`:
Execution logs and completion flags.

`experiments/results/scaling_experiment/[model_pairs]/`:
Results organized by model pair, competition level, run number.

`experiments/results_current/`:
Symlink to latest results for visualization scripts.

`figures/`:
Generated visualizations. PDFs and PNGs.

## Key Features

- No timeouts in simple runner
- 5 seeds for statistical significance
- Competition level 0=cooperative, 1=zero-sum
- MMLU-Pro ordering for capability analysis
- Weak model always agent_0
- Python handles retries internally
- Incremental saving (partial runs preserved)
- Auto-resume on restart