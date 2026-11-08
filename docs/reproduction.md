# Paper reproduction and cluster batches

Use this guide for the paper's saved configurations, analyses, and Slurm jobs.
For a new small experiment, use the
[historical paper runner](#run-one-experiment-with-the-paper-runner).
The original runners and saved batch configurations are the reproduction entry
points.

- [Prepare the environment](#prepare-the-environment)
- [Inspect and analyze released results](#inspect-and-analyze-released-results)
- [Paper inventory and batch procedures](#paper-inventory-and-batch-procedures)
- [Run one experiment with the paper runner](#run-one-experiment-with-the-paper-runner)
- [Read result files](#read-result-files)
- [Select batch runs](#select-batch-runs)
- [Generate a new test-time-compute batch](#generate-a-new-test-time-compute-batch)
- [Additional analyses and plots](#additional-analyses-and-plots)

## Prepare the environment

Use Python 3.12 or later and run commands from the checkout root.
Replace the checkout path below with your absolute path.

```bash
export BARGAIN_ROOT="/absolute/path/to/bargain"
cd "$BARGAIN_ROOT"
python3 -m venv "$BARGAIN_ROOT/.venv"
source "$BARGAIN_ROOT/.venv/bin/activate"
python -m pip install -r "$BARGAIN_ROOT/requirements.txt"
python -m pip install -e "${BARGAIN_ROOT}[analysis]"
```

If the environment already exists, activate it and install the dependencies.
The analysis extra includes the statistical packages used by the paper scripts.
Local model runs also need the optional PyTorch and Transformers dependencies
listed in the requirements file and a compatible GPU allocation.
On Della, check `/scratch/gpfs/DANQIC/models` before downloading model weights.

For credentials, proxy transport, key groups, and logs, read the
[operations guide](operations.md).
Slurm jobs need a cluster account and the generated scheduler settings must
match that cluster.
The OpenRouter queue monitor is managed separately on a networked node.

## Inspect and analyze released results

Download the released data with the [README instructions](../README.md#inspect-released-results).
You can inspect those results without making model API calls or submitting jobs.
The dated result roots in the commands below identify the paper inputs.
A new batch with the current model roster does not recreate those saved results.

Use the [analysis commands](#9-validate-and-analyze-the-results) after downloading
their inputs.
The main input and output relationships are:

| Analysis | Reads | Writes |
| --- | --- | --- |
| GPT-5 Nano bilateral comparison | The dated Game 1, 2, and 3 roots below | `experiments/results/n2_baseline_comparison_analysis_20260505/` |
| Llama bilateral comparison | The three `appendix_llama33_baseline_game*_202605` roots | `experiments/results/appendix_llama33_baseline_analysis_20260503/` |
| Bilateral and multi-agent comparison | The bilateral comparison tables and saved multi-agent batches | `experiments/results/n2_plus_multiagent_comparison_analysis_20260505/` |
| Revision statistics | Saved bilateral, multi-agent, and test-time-compute inputs | `analysis/neurips_revision_20260504/` |
| Complete test-time-compute seed panels | The ten saved seed batches | `analysis/ttc_complete_family_seed_panels_20260810/` |
| Coordinated-team analysis | The selected treatment root and its matched controls | The treatment root's `analysis/` directory |

These defaults refer to historical data.
For newly generated batches, inspect each analysis script's accepted inputs
before using its output as evidence.

For figures, use the [renderer guide](../scripts/paper_figures/README.md) and
[figure manifest](reproducibility/paper_figure_manifest.md).
The figure verifier regenerates declared outputs, compares them, and restores
the saved assets by default.
It writes logs under `analysis/paper_figure_verification/` and requires its
listed data and paper assets to exist locally.

## Paper inventory and batch procedures

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

### 1. Configure credentials and transport

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

### 2. Run the GPT-5 Nano bilateral batches

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

The documented grid has 540 configs: 30 adversary models, nine parameter
cells, and both model orders.
Inspect the generated summary because the active roster can change.

Generate and submit the Game 3 co-funding batch.

```bash
bash scripts/generate_cofunding_configs.sh --conservative
bash experiments/results/cofunding_latest/configs/slurm/submit_all.sh \
  all --max-concurrent 40
```

The documented Game 3 `--conservative` grid also has 540 configs.
Inspect its generated summary before submission.

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

### 3. Run the Llama 3.3 bilateral batches

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

### 4. Run the multi-agent batch

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

### 5. Run the random-monoculture batch

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

### 6. Run the native test-time-compute batch

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

### 7. Run the coordinated-team batch

Generate 100 Game 1 runs with preferences matched to the saved GPT-5.4 controls.

```bash
TEAM_ROOT="experiments/results/game1_gpt54_binding_team_$(date +%Y%m%d_%H%M%S)"
python scripts/generate_game1_gpt54_binding_team.py --output-root "$TEAM_ROOT"
sbatch --array=1-100%20 "$TEAM_ROOT/slurm/run_binding_team_gpt54.sbatch"
```

### 8. Run the matched coalition replication

Run the 25 released GPT-5.4 configurations; configuration 21 replaces failed
configuration 4. Coalition rates use the 20 runs with more than two agents.

```bash
MATCHED_ROOT=experiments/results/game1_gpt54_high_matched_coalition_pilot_direct_openai_20260829
for CONFIG_ID in 1 2 3 $(seq 5 26); do
  python scripts/full_games123_multiagent_batch.py run-one \
    --results-root "$MATCHED_ROOT" --config-id "$CONFIG_ID"
done
```

### 9. Validate and analyze the results

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

These analysis commands use the downloaded paper batches and derived tables.
Run them in the order shown; the multi-agent comparison reads the bilateral
comparison outputs.

## Run one experiment with the paper runner

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

The Game 1 example needs `OPENAI_API_KEY`.
The Game 2 and Game 3 examples need both `OPENAI_API_KEY` and
`OPENROUTER_API_KEY`; the selected Claude and Gemini aliases use OpenRouter.
The file-queue transport applies to OpenRouter calls only.
Direct-provider calls need a node with outbound access to that provider.

## Read result files

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

## Select batch runs

The multi-agent controller supports named selections for targeted submissions.
Use the `MULTI_ROOT` created in [the multi-agent procedure](#4-run-the-multi-agent-batch).

```bash
python scripts/full_games123_multiagent_batch.py select \
  --results-root "$MULTI_ROOT" \
  --selection-name game3_n10 \
  --game-label game3 \
  --n-agents 10
python scripts/full_games123_multiagent_batch.py submit-selection \
  --results-root "$MULTI_ROOT" \
  --selection-name game3_n10
```

Use `squeue -u "$USER"` and the generated logs to monitor jobs.
Generated Slurm files can contain absolute paths to their configuration root.
Keep that root in place while its jobs are active.
For sequential co-funding and diplomacy submission, see
[`submit_cofunding_then_diplomacy.sh`](../scripts/submit_cofunding_then_diplomacy.sh).

## Generate a new test-time-compute batch

Use this generator for a new design with the current settings.
Use [seed replication](#6-run-the-native-test-time-compute-batch) for the saved paper design.

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

## Additional analyses and plots

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

## Validate a change

Run a small direct or `run-one` experiment with the exact model roster and
transport before submitting a large batch.
Check the result JSON, interaction JSON, and experiment IDs.
An offline unit test does not verify provider access.

Run the retained test suite from the checkout root:

```bash
python -m pytest tests/
```

For a focused transport and context check:

```bash
python -m pytest \
  tests/test_context_compaction.py \
  tests/test_private_thinking_schema.py \
  tests/test_openrouter_transport.py
```
