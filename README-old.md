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
|   analysis scripts. Dedicated paper renderers are in scripts/paper_figures/.
|-- experiments/results/
|   Large generated result trees. Usually not something to edit by hand.
|-- analysis/
|   Derived CSVs, reports, and figure-generation outputs.
|-- overleaf/
|   The current paper roots and their paper-facing graphics.
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
export OPENROUTER_PROXY_POLL_DIR="$HOME/openrouter_proxy"
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

## Reproduce the Paper Experiments

The current paper source is
`overleaf/icml_aiwild_template/icml_aiwild_2026.tex`.
The paper reports 5,691 accepted experiment results.

| Batch | Accepted results | Generator or batch script |
| --- | ---: | --- |
| GPT-5 Nano bilateral | 1,920 | `generate_configs_both_orders.sh`, `generate_diplomacy_configs.sh`, and `generate_cofunding_configs.sh` |
| Llama 3.3 70B bilateral | 500 | `generate_appendix_llama33_baseline_configs.py` |
| Multi-agent | 2,730 | `full_games123_multiagent_batch.py` |
| Random monoculture | 325 | `random_monoculture_control_batch.py` |
| Native test-time compute | 216 | `generate_ttc_native_scaling_jobs.py` |
| **Total** | **5,691** | |

The number 5,691 is a result count. It is not a planned-config count.
The original generators made some configs that did not produce an accepted result.
The paper inventory excludes these failed or invalid runs.
A new run can also have a different result because a hosted model can change or
become unavailable.

The historical Game 1 generator made 448 configs for each discussion-turn arm.
Each arm missed 14 Claude 3.5 Sonnet configs because the provider was not
reachable. That model is now retired, so those 28 configs cannot be rerun
exactly. The paper also excludes all Phi-3 Mini configs. The accepted Game 1
inventory therefore contains 420 runs in each discussion-turn arm.

The exact paper inventory is
`docs/reproducibility/paper_experiment_data_manifest.csv`.
Each row gives the config, result, rollout, source root, and canonical analysis
table for one accepted run.

### 1. Prepare the Environment

Run all commands from the repository root.

```bash
source .venv/bin/activate
export BARGAIN_API_KEYS_ENV=/home/USER/.config/bargain/api_keys.env
set -a
source "$BARGAIN_API_KEYS_ENV"
set +a

export OPENROUTER_TRANSPORT=proxy
export OPENROUTER_PROXY_POLL_DIR="$HOME/openrouter_proxy"
```

Use `OPENROUTER_TRANSPORT=direct` on a machine that has direct internet access.
Do not put API keys in the repository.
The visualization-node queue monitor is managed separately.
Do not start or stop it from a batch workflow.

Check the released inventory before you make a new batch.

```bash
python scripts/build_paper_experiment_data_manifest.py --check
```

This check verifies the released file paths.
It does not prove that a new API call will give the historical output.

### 2. Run the GPT-5 Nano Bilateral Batches

These dated roots contain the exact paper configs.

```bash
GAME1_ROOT=experiments/results/scaling_experiment_20260404_064451
GAME2_ROOT=experiments/results/diplomacy_20260405_082215
GAME3_ROOT=experiments/results/cofunding_20260405_083548
```

Use the dated roots to reproduce the released inventory in a clean workspace.
Do not submit these commands in a workspace that contains accepted result files.
The current generators use the current active model roster and make a new batch.

```bash
ln -sfn "$(basename "$GAME1_ROOT")" experiments/results/scaling_experiment
bash "$GAME1_ROOT/configs/slurm/submit_all.sh" all --max-concurrent 40
bash "$GAME2_ROOT/configs/slurm/submit_all.sh" --max-concurrent 40
bash "$GAME3_ROOT/configs/slurm/submit_all.sh" all --max-concurrent 40
```

The config files write results below their dated root.

To make a new Game 1 item-allocation batch with the active roster, run this
command.

```bash
bash scripts/generate_configs_both_orders.sh
```

The command prints the new `scaling_experiment_TIMESTAMP` root.
Read its `configs/summary.txt` file before submission.
Then submit its API and local-model configs.

```bash
bash experiments/results/scaling_experiment/configs/slurm/submit_all.sh \
  all --max-concurrent 40
```

Game 1 contains two protocol arms.
Use `discussion_turns=2` for the primary result.
Use `discussion_turns=1` for the appendix ablation.
One discussion turn is one full circuit in which each agent speaks once.

Generate and submit the Game 2 diplomatic-treaty batch.

```bash
bash scripts/generate_diplomacy_configs.sh --conservative
bash experiments/results/diplomacy_latest/configs/slurm/submit_all.sh \
  --max-concurrent 40
```

The `--conservative` mode makes the paper grid of 540 configs.
It uses 30 adversary models, nine parameter cells, and both model orders.

Generate and submit the Game 3 co-funding batch.

```bash
bash scripts/generate_cofunding_configs.sh --conservative
bash experiments/results/cofunding_latest/configs/slurm/submit_all.sh \
  all --max-concurrent 40
```

The Game 3 `--conservative` mode also makes 540 configs.

Run one generated Game 2 or Game 3 config with its local runner.

```bash
bash RUN_ROOT/configs/slurm/run_local.sh CONFIG_ID
```

Run one Game 1 API config as one Slurm array task.

```bash
sbatch --array=CONFIG_ID \
  RUN_ROOT/configs/slurm/run_api_experiments.sbatch
```

Do not run the saved Phi-3 Mini configs for paper reproduction.
They are outside the paper inventory.

### 3. Run the Llama 3.3 Bilateral Batches

Generate all 500 configs.

```bash
RUN_TAG=$(date +%Y%m%d_%H%M%S)
python scripts/generate_appendix_llama33_baseline_configs.py \
  --suffix "$RUN_TAG"
```

Submit the three generated batches.

```bash
bash "experiments/results/appendix_llama33_baseline_game1_${RUN_TAG}/slurm/submit_individual.sh"
bash "experiments/results/appendix_llama33_baseline_game2_${RUN_TAG}/slurm/submit_individual.sh"
bash "experiments/results/appendix_llama33_baseline_game3_${RUN_TAG}/slurm/submit_individual.sh"
```

These batches contain 140, 180, and 180 configs, respectively.

### 4. Run the Multi-Agent Batch

The current generator makes homogeneous controls, homogeneous adversaries, and
heterogeneous groups in one 2,730-config root.
Its default heterogeneous sampler uses equal-width Elo-dispersion strata.

```bash
MULTI_ROOT="experiments/results/full_games123_multiagent_$(date +%Y%m%d_%H%M%S)"

python scripts/full_games123_multiagent_batch.py generate \
  --results-root "$MULTI_ROOT"
python scripts/full_games123_multiagent_batch.py validate \
  --results-root "$MULTI_ROOT"
python scripts/full_games123_multiagent_batch.py submit \
  --results-root "$MULTI_ROOT"
```

Check completion after the Slurm jobs stop.

```bash
python scripts/full_games123_multiagent_batch.py summary \
  --results-root "$MULTI_ROOT" --json
python scripts/full_games123_multiagent_batch.py report \
  --results-root "$MULTI_ROOT"
```

Run one multi-agent config with this command.

```bash
python scripts/full_games123_multiagent_batch.py run-one \
  --results-root "$MULTI_ROOT" \
  --config-id CONFIG_ID
```

### 5. Run the Random-Monoculture Batch

Use the model-pool CSV from the validated multi-agent root.
Keep seed `20260628` to reproduce the model selection.

```bash
MONO_ROOT="experiments/results/full_games123_random_monoculture_control_$(date +%Y%m%d_%H%M%S)"
POOL_CSV="$MULTI_ROOT/configs/heterogeneous_subset_maps/model_pool_24.csv"

python scripts/random_monoculture_control_batch.py generate \
  --results-root "$MONO_ROOT" \
  --pool-csv "$POOL_CSV" \
  --seed 20260628
python scripts/random_monoculture_control_batch.py validate \
  --results-root "$MONO_ROOT"
python scripts/random_monoculture_control_batch.py submit-selection \
  --results-root "$MONO_ROOT" \
  --selection-name all
```

Check the 325 configs after completion.

```bash
python scripts/random_monoculture_control_batch.py summary \
  --results-root "$MONO_ROOT" --json
```

Run one monoculture config with this command.

```bash
python scripts/random_monoculture_control_batch.py run-one \
  --results-root "$MONO_ROOT" \
  --config-id CONFIG_ID
```

The runner does not overwrite an existing valid result.
Use a new results root or a staging `output_dir` for a deliberate recovery run.

### 6. Run the Native Test-Time-Compute Batch

Generate and submit all 216 configs.

```bash
TTC_ROOT="experiments/results/ttc_native_scaling_$(date +%Y%m%d_%H%M%S)"

python scripts/generate_ttc_native_scaling_jobs.py \
  --results-root "$TTC_ROOT"
bash "$TTC_ROOT/slurm/submit_all.sh"
```

Run one TTC config before a full submission when you test a new provider setup.

```bash
python scripts/run_ttc_native_config.py \
  --config "$TTC_ROOT/configs/config_0000.json" \
  --dry-run
```

Remove `--dry-run` to run that config.

### 7. Validate and Analyze the Results

Do not count a config as complete only because a status file says `SUCCESS`.
Validate its result JSON and its rollout JSON.
Also verify that both files contain the same experiment ID.

Use these scripts for the paper analyses.

```bash
python scripts/analyze_n2_baseline_comparison.py
python scripts/analyze_appendix_llama33_baseline_500.py
python scripts/analyze_n2_plus_multiagent_comparison.py
python scripts/analyze_neurips_revision_stats.py
python scripts/paper_figures/verify_all.py
```

The released manifest builder uses the historical paper roots.
Do not use it as a validator for a new results root without first changing its
input roots and expected counts.

Render the Game 1 discussion-turn ablation after the N=2 analysis finishes.

```bash
python scripts/paper_figures/plot_game1_discussion_turn_ablation.py
```

The N=2 analysis writes two run tables.
`all_runs_with_metrics.csv` contains the full inventory and the one-turn
ablation. `primary_runs_with_metrics.csv` uses two turns for Game 1.
Paper renderers use the primary table unless they render the ablation.

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

The Llama 3.3 70B baseline replication has dedicated generation and analysis
scripts:

```bash
python scripts/generate_appendix_llama33_baseline_configs.py
python scripts/analyze_appendix_llama33_baseline_500.py
```

Use `squeue -u "$USER"` and the generated status files to monitor the batch.

### Bilateral Batch Generators

The retained bilateral generators reproduce the paper workflows:

- `scripts/generate_configs_both_orders.sh`
- `scripts/generate_diplomacy_configs.sh`
- `scripts/generate_cofunding_configs.sh`
- `scripts/submit_cofunding_then_diplomacy.sh`

Use `full_games123_multiagent_batch.py` for N-agent work.

## Analysis and Plotting

Current paper-facing analysis scripts include:

```bash
python scripts/plot_gpt5_nano_baseline_vs_elo_all_games.py
python scripts/plot_exploitation_vs_elo.py
python scripts/plot_nbs_decomposition.py
python scripts/analyze_nash_lindahl_fairness.py
python scripts/analyze_neurips_revision_stats.py
python scripts/analyze_n2_baseline_comparison.py
python scripts/analyze_n2_plus_multiagent_comparison.py
python scripts/paper_figures/verify_all.py
```

Temporary qualitative and exploratory analyses are in
`scripts/retained_analysis/`. Its README maps each script to its report and
asset directory.

Important derived directories:

- `analysis/neurips_revision_20260504/`: normalized payoff tables, regressions,
  bootstrap intervals, TTC summaries, and paper-facing copied plots.
- `analysis/nash_lindahl_fairness_20260505/`: NBS/Lindahl recomputation and
  benchmark-relative exploitation summaries.
- `analysis/full_games123_*`: multi-agent aggregate CSVs and plots.
- `scripts/paper_figures/`: dedicated renderers and small fixed rendering assets.
- `overleaf/*/graphics/`: paper-root-specific figure exports.

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

It runs `ui/experiment_viewer.py` for Game 2 and Game 3 trajectories. Other
supported viewers include:

- `ui/multi_game_sample_viewer.py`
- `ui/game1_sample_viewer.py`
- `ui/game2_batch_viewer.py`
- `ui/game3_batch_viewer.py`
- `ui/random_monoculture_sample_viewer.py`

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
