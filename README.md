# Scaling Laws for Strategic Interactions

This repository studies how LLM agent capability, group size, and strategic
competition shape bargaining outcomes. The current codebase supports three
multi-turn negotiation environments, large model-roster sweeps, N-agent 
experiment batches, test-time-compute stress tests, analysis scripts, 
and Streamlit UIs for transcript viewers.

Commands below cover the experiments in the current paper.

## Current Research Surface

The main paper asks whether stronger LLM agents create more joint value, take
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

Tracked files and directories:

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

## Download the Released Data

Download the saved configurations, results, transcripts, and prompts through the
[anonymous Hugging Face link](https://anonymous-hf.com/a/xgxza3yfsgll/):

```bash
python scripts/download_review_transcripts.py --all --download
```

Files go into `experiments/results/`, which is created locally and excluded from
Git. No Hugging Face account or model API key is needed. The downloader checks
file hashes and skips identical local files.
Use the UI commands below to inspect the downloaded runs.

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

Useful options:

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

The paper reports 7,160 runs in its main inventory:

| Batch | Paper runs | Generator or batch script |
| --- | ---: | --- |
| GPT-5 Nano bilateral | 1,500 | `generate_configs_both_orders.sh`, `generate_diplomacy_configs.sh`, and `generate_cofunding_configs.sh` |
| Llama 3.3 70B bilateral | 500 | `generate_appendix_llama33_baseline_configs.py` |
| Homogeneous adversary | 1,300 | `full_games123_multiagent_batch.py` |
| Heterogeneous | 1,300 | `full_games123_multiagent_batch.py` |
| Homogeneous | 300 | `random_monoculture_control_batch.py` |
| Coordinated teams | 100 | `generate_game1_gpt54_binding_team.py` |
| Native test-time compute | 2,160 | `generate_ttc_seed_replication_jobs.py` |
| **Total** | **7,160** | |

The appendix also reports the matched GPT-5.4 coalition replication below.
The original batches contain additional controls and earlier protocol variants.
For primary Game 1 results, use the 420 two-discussion-turn runs and exclude
Phi-3 Mini and the failed Claude 3.5 Sonnet configurations.

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

### 2. Run the GPT-5 Nano Bilateral Batches

After downloading the released data, these dated roots contain the paper configs.

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
The older `discussion_turns=1` arm is outside the primary paper inventory.
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
The paper uses the 1,300 homogeneous-adversary and 1,300 heterogeneous runs.
The other 130 runs are GPT-5-nano controls.
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

The generator makes 325 configs; the paper uses 300, excluding the 25 Game 1
`claude-3-haiku-20240307` runs. Check completion:

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

Clone the saved 216-config design at each of the paper's ten seeds.
This preserves the original model settings and 10,500-token limit.

```bash
TTC_TAG=$(date +%Y%m%d_%H%M%S)
for SEED in 42 984 526 423 1024 128 256 612 2048 4096; do
  TTC_ROOT="experiments/results/ttc_native_scaling_seed${SEED}_${TTC_TAG}"
  python scripts/generate_ttc_seed_replication_jobs.py \
    --seed "$SEED" --results-root "$TTC_ROOT"
  bash "$TTC_ROOT/slurm/submit.sh"
done
```

### 7. Run the Coordinated-Team Batch

Generate 100 Game 1 runs with preferences matched to the saved GPT-5.4 controls.

```bash
TEAM_ROOT="experiments/results/game1_gpt54_binding_team_$(date +%Y%m%d_%H%M%S)"
python scripts/generate_game1_gpt54_binding_team.py --output-root "$TEAM_ROOT"
sbatch --array=1-100%20 "$TEAM_ROOT/slurm/run_binding_team_gpt54.sbatch"
```

### 8. Run the Matched Coalition Replication

Run the 25 released GPT-5.4 configurations; configuration 21 replaces failed
configuration 4. Coalition rates use the 20 runs with more than two agents.

```bash
MATCHED_ROOT=experiments/results/game1_gpt54_high_matched_coalition_pilot_direct_openai_20260829
for CONFIG_ID in 1 2 3 $(seq 5 26); do
  python scripts/full_games123_multiagent_batch.py run-one \
    --results-root "$MATCHED_ROOT" --config-id "$CONFIG_ID"
done
```

### 9. Validate and Analyze the Results

Do not count a config as complete only because a status file says `SUCCESS`.
Validate its result JSON and its rollout JSON.
Also verify that both files contain the same experiment ID.

Use these scripts for the paper analyses.

```bash
python scripts/analyze_n2_baseline_comparison.py
python scripts/analyze_appendix_llama33_baseline_500.py
python scripts/analyze_n2_plus_multiagent_comparison.py
python scripts/analyze_neurips_revision_stats.py
python scripts/analyze_ttc_complete_seed_panels.py
python scripts/analyze_game1_gpt54_binding_team.py \
  --results-root experiments/results/game1_gpt54_binding_team_v3_20260816_093310
python scripts/paper_figures/verify_all.py
```

These analysis commands use the downloaded paper batches.

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

The active model roster and Elo helpers live in:

```text
strong_models_experiment/analysis/active_model_roster.py
docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md
```

## UI Viewers

Install the viewer dependencies:

```bash
uv pip install -r ui/requirements.txt
```

Search downloaded transcripts across all three games:

```bash
streamlit run ui/transcript_review.py --server.address 127.0.0.1 --server.port 8010
```

For detailed game displays, use the main Streamlit launcher:

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

For coordinated-team transcripts:

```bash
bash ui/run_binding_team_viewer.sh
```

On a remote cluster, bind Streamlit to loopback and use an SSH tunnel from your
laptop. For example:

```bash
streamlit run ui/experiment_viewer.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

On your laptop, tunnel to the machine running the viewer and open
`http://127.0.0.1:8501`:

```bash
ssh -N -L 8501:127.0.0.1:8501 USER@VIEWER_HOST
```

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
