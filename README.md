# Scaling Laws for Strategic Interactions

This repository studies how LLM agent capability, group size, and strategic
competition shape bargaining outcomes. The current codebase supports three
multi-turn negotiation environments, large model-roster sweeps, N-agent Slurm
batches, test-time-compute stress tests, analysis scripts, and Streamlit
viewers.

The repository has accumulated many exploratory and legacy scripts. Treat this
README as the current map: when older scripts conflict with the workflows below,
prefer the workflows below.

## Current Research Surface

The main paper asks whether stronger LLM agents create more joint surplus, take
a larger share from weaker counterparts, or both. The current experiments vary:

- Model capability, primarily using LMArena Elo from the March 31, 2026 snapshot.
- Strategic structure, through game-specific competition parameters.
- Number of agents, with production grids for N in {2, 4, 6, 8, 10}.
- Native test-time compute, using provider reasoning-effort controls where
  available and token diagnostics otherwise.

The three implemented games are:

- Game 1, `item_allocation`: agents bargain over indivisible items with private
  value vectors. The main competition knob is value-vector cosine similarity.
- Game 2, `diplomacy`: agents bargain over continuous treaty issues with ideal
  positions and issue weights. The main knobs are `rho` for position correlation
  and `theta` for interest overlap.
- Game 3, `co_funding`: agents bargain over threshold public-good funding with
  private project values and budgets. The main knobs are `alpha` for value
  alignment and `sigma` for budget abundance/scarcity.

All three games share the same high-level negotiation loop: setup, public
discussion, private thinking, structured proposal, private voting, proposal
selection by two-thirds supermajority, and optional reflection before the next
round. Utilities can be time-discounted by `gamma_discount`.

## Repository Map

```text
.
|-- run_strong_models_experiment.py
|   Main single-run and small-batch CLI for Games 1-3.
|-- game_environments/
|   Game implementations and JSON parsing/repair utilities.
|-- strong_models_experiment/
|   Experiment orchestration, agent factory, phase handlers, configs,
|   analyzers, active model roster, and qualitative metrics.
|-- negotiation/
|   LLM clients, OpenRouter proxy transport, provider key rotation,
|   context compaction, and lower-level agent utilities.
|-- scripts/
|   Batch generation, Slurm submission, monitoring, plotting, and paper
|   analysis scripts. Many older scripts are kept for provenance.
|-- experiments/results/
|   Large generated result trees. Usually not something to edit by hand.
|-- analysis/
|   Derived CSVs, reports, and figure-generation outputs.
|-- Figures/
|   Paper-facing and presentation-facing figure exports.
|-- ui/
|   Streamlit viewers for individual runs, batches, and multi-game comparison.
|-- docs/
|   Design notes, model rosters, pricing notes, prompt references, and plans.
`-- tests/
    Unit and regression tests for games, clients, parsing, providers, and batches.
```

## Setup

Use a virtual environment from the repository root.

```bash
cd /path/to/bargain
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `uv` is available, this is also fine:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Optional dependencies for local/Hugging Face models are commented in
`requirements.txt`. Before downloading any Hugging Face model on the cluster,
check `/path/to/models` for an existing local copy.

## Credentials and Cluster Networking

Direct provider calls use standard environment variables:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
export XAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

The provider key-rotation layer also supports grouped keys. Set
`LLM_KEY_GROUP_ORDER` and define variables of the form
`PRIMARY_OPENAI_API_KEY_1`, `SECONDARY_OPENROUTER_API_KEY_1`,
`GROUP_A_GOOGLE_API_KEY_1`, `GROUP_B_GOOGLE_API_KEY_1`, etc. The code tries
groups in order and writes failure reports without logging secret values.

Many Slurm scripts source:

```bash
${BARGAIN_API_KEYS_ENV:-/path/to/api_keys.env}
```

On restricted-network compute nodes, OpenRouter traffic should go
through the file-based proxy queue. The relevant defaults are:

```bash
export OPENROUTER_TRANSPORT=proxy
export OPENROUTER_PROXY_POLL_DIR=/path/to/openrouter_proxy
```

`negotiation/openrouter_proxy_monitor.py` is the monitor process intended to run
on a login or visualization node with outbound internet access. Batch scripts
generally assume the monitor already exists and route compute-node jobs through
that queue.

## Running One Experiment

The main entry point is `run_strong_models_experiment.py`. Use model keys from
`strong_models_experiment/configs.py`.

Item allocation:

```bash
python run_strong_models_experiment.py \
  --game-type item_allocation \
  --models gpt-5-nano gpt-4o-mini-2024-07-18 \
  --competition-level 0.5 \
  --num-items 5 \
  --max-rounds 10 \
  --discussion-turns 2 \
  --random-seed 42 \
  --batch \
  --num-runs 1 \
  --output-dir experiments/results/smoke_item_allocation
```

Diplomatic treaty:

```bash
python run_strong_models_experiment.py \
  --game-type diplomacy \
  --models gpt-5-nano claude-sonnet-4-20250514 \
  --rho 0.0 \
  --theta 0.5 \
  --n-issues 5 \
  --max-rounds 10 \
  --discussion-turns 2 \
  --random-seed 42 \
  --batch \
  --num-runs 1 \
  --output-dir experiments/results/smoke_diplomacy
```

Co-funding:

```bash
python run_strong_models_experiment.py \
  --game-type co_funding \
  --models gpt-5-nano gemini-2.5-pro \
  --alpha 0.5 \
  --sigma 0.6 \
  --m-projects 5 \
  --c-min 10 \
  --c-max 30 \
  --cofunding-discussion-transparency own \
  --max-rounds 10 \
  --discussion-turns 2 \
  --random-seed 42 \
  --batch \
  --num-runs 1 \
  --output-dir experiments/results/smoke_cofunding
```

Useful flags:

- `--model-order weak_first|strong_first|random`: controls speaking/order labels.
- `--parallel-phases`: runs independent per-agent phases concurrently.
- `--max-tokens-per-phase`: sets the per-call output cap, defaulting to 16384.
- `--disable-discussion`, `--disable-thinking`, `--disable-reflection`: ablate
  protocol phases.
- `--reasoning-token-budget` and `--reasoning-budget-phases`: prompt-level TTC
  controls for older experiments.
- `--access-k`, `--access-agent-index`, `--access-phases`: black-box access
  scaling for repeated private drafts plus selection.

## Result Files

Small runs and generated batches write under `experiments/results/`. Common
files include:

- `experiment_results.json` or `run_N_experiment_results.json`: final utilities,
  agreement status, config, token usage, metadata, and outcome payload.
- `all_interactions.json` or `run_N_all_interactions.json`: prompts, responses,
  phases, rounds, token usage, and parse diagnostics.
- `agent_interactions/`: per-agent interaction views.
- `progress.json`: lightweight streaming metadata during a run.
- `monitoring/malformed_json_examples.jsonl`: batch-level parse diagnostics.
- `batch_summary.json`: aggregate summary for batch-mode runs.

Large generated roots often also contain `configs/`, `status/`, `logs/`,
`monitoring/`, `runs/`, and Slurm wrapper files.

## Production Batch Workflows

### Full Games 1-3 N-Agent Batch

The current all-game N-agent generator is:

```bash
python scripts/full_games123_multiagent_batch.py generate \
  --results-root experiments/results/full_games123_multiagent_MYRUN

python scripts/full_games123_multiagent_batch.py validate \
  --results-root experiments/results/full_games123_multiagent_MYRUN

python scripts/full_games123_multiagent_batch.py submit \
  --results-root experiments/results/full_games123_multiagent_MYRUN

python scripts/full_games123_multiagent_batch.py summary \
  --results-root experiments/results/full_games123_multiagent_MYRUN

python scripts/full_games123_multiagent_batch.py report \
  --results-root experiments/results/full_games123_multiagent_MYRUN
```

This script generates homogeneous controls, homogeneous one-adversary runs, and
heterogeneous ecologies across Games 1-3. It supports selection files for
targeted reruns:

```bash
python scripts/full_games123_multiagent_batch.py select \
  --results-root experiments/results/full_games123_multiagent_MYRUN \
  --selection-name game3_n10 \
  --game-label game3 \
  --n-agents 10

python scripts/full_games123_multiagent_batch.py submit-selection \
  --results-root experiments/results/full_games123_multiagent_MYRUN \
  --selection-name game3_n10
```

### Native Test-Time-Compute Stress Test

The current TTC generator creates configs and Slurm wrappers for GPT-5,
Claude Sonnet 4.6, and Gemini 3 Flash effort levels across matched game cells:

```bash
python scripts/generate_ttc_native_scaling_jobs.py \
  --results-root experiments/results/ttc_native_scaling_MYRUN \
  --dry-run

python scripts/generate_ttc_native_scaling_jobs.py \
  --results-root experiments/results/ttc_native_scaling_MYRUN \
  --submit
```

One config can be run directly for debugging:

```bash
python scripts/run_ttc_native_config.py \
  --config experiments/results/ttc_native_scaling_MYRUN/configs/config_0001.json \
  --dry-run
```

### Appendix Llama Baseline

The Llama 3.3 70B baseline replication has dedicated generation/monitoring and
analysis scripts:

```bash
python scripts/generate_appendix_llama33_baseline_configs.py
python scripts/monitor_appendix_llama33_baseline.py
python scripts/analyze_appendix_llama33_baseline_500.py
```

### Older and Narrower Batch Scripts

Several older scripts remain useful for targeted probes or provenance:

- `scripts/generate_diplomacy_configs.sh`
- `scripts/generate_cofunding_configs.sh`
- `scripts/generate_nagent_configs.sh`
- `scripts/game1_multiagent_full_batch.py`
- `scripts/game1_ttc_access_batch.py`
- `scripts/game2_derisk_32.py`
- `scripts/game3_multiagent_sample_batch.py`

Prefer `full_games123_multiagent_batch.py` for new all-game N-agent work unless
you are intentionally reproducing an older run.

## Analysis and Plotting

Current paper-facing analysis scripts include:

```bash
python scripts/plot_gpt5_nano_baseline_vs_elo_all_games.py
python scripts/plot_exploitation_vs_elo.py
python scripts/plot_nbs_decomposition.py
python scripts/analyze_nash_lindahl_fairness.py
python scripts/analyze_neurips_revision_stats.py
python scripts/analyze_capability_payoff_scaling_20260505.py
python scripts/analyze_n2_baseline_comparison.py
python scripts/analyze_n2_plus_multiagent_comparison.py
python scripts/build_n2_ttc_multiagent_report.py
```

Important derived directories:

- `analysis/neurips_revision_20260504/`: normalized payoff tables, regressions,
  bootstrap intervals, TTC summaries, and paper-facing copied plots.
- `analysis/nash_lindahl_fairness_20260505/`: NBS/Lindahl recomputation and
  benchmark-relative exploitation summaries.
- `analysis/full_games123_*`: multi-agent aggregate CSVs and plots.
- `Figures/`: exported figures for slides and paper drafts.

The active model roster and Elo helpers live in:

```text
strong_models_experiment/analysis/active_model_roster.py
docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md
```

## UI Viewers

The main Streamlit launcher is:

```bash
bash ui/run_viewer.sh --port 8501
```

It runs `ui/experiment_viewer.py`. Other specialized viewers include:

- `ui/negotiation_viewer.py`
- `ui/multi_game_sample_viewer.py`
- `ui/game1_sample_viewer.py`
- `ui/game2_batch_viewer.py`
- `ui/game3_batch_viewer.py`

On a remote cluster, bind Streamlit to loopback and use an SSH tunnel from your
laptop. For example:

```bash
streamlit run ui/experiment_viewer.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

Then tunnel `localhost:8501` from the laptop to the cluster node running the UI.

## Testing

Run the full test suite with:

```bash
pytest tests/
```

The suite is large and includes provider/route tests. For focused changes, use
targeted tests first:

```bash
pytest tests/test_cofunding_game.py
pytest tests/test_diplomatic_treaty.py
pytest tests/test_openrouter_transport.py
pytest tests/test_provider_key_rotation.py
pytest tests/test_context_compaction.py
pytest tests/test_full_games123_batch_generation.py
```

Some integration tests require API keys, local model paths, or cluster-specific
state. Prefer unit tests for logic changes and explicit smoke runs for provider
changes.

## Development Notes

- The current code path is `run_strong_models_experiment.py` plus
  `strong_models_experiment/`, `game_environments/`, and `negotiation/`.
- `negotiation/` is not purely legacy anymore; it contains the active provider
  clients, OpenRouter proxy transport, key rotation, and context compaction.
- Result directories are large. Avoid committing generated run roots unless the
  artifact is intentionally paper-facing.
- Keep new docs in `docs/` or the appropriate subdirectory. The repository root
  should stay limited to high-level files like this README.
- Prefer structured JSON parsing/repair utilities already in `game_environments`
  and `negotiation/json_repair.py` over ad hoc string parsing.
- Before launching Slurm batches, do a small direct or `run-one` smoke test with
  the exact model roster and transport settings.

## License

MIT License. See `LICENSE`.
