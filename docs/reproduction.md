# Experiment Reproduction

## Scope

This guide gives the current experiment procedures. Run all commands from the repository root.

Read [`../README.md`](../README.md) before you run an experiment. It gives the environment and credential requirements.

## Run One Experiment

Use `run_strong_models_experiment.py` for a small test. This example runs Game 1 one time.

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

Use `--game-type diplomacy` for Game 2. Use `--game-type co_funding` for Game 3.

The main README gives a complete example for each game.

## Run The Current Multi-Agent Batch

Use `full_games123_multiagent_batch.py` for new multi-agent work. Use a new result root for each batch.

```bash
python scripts/full_games123_multiagent_batch.py generate \
  --results-root experiments/results/full_games123_multiagent_MYRUN

python scripts/full_games123_multiagent_batch.py validate \
  --results-root experiments/results/full_games123_multiagent_MYRUN

python scripts/full_games123_multiagent_batch.py submit \
  --results-root experiments/results/full_games123_multiagent_MYRUN
```

Monitor the batch with this command:

```bash
python scripts/full_games123_multiagent_batch.py summary \
  --results-root experiments/results/full_games123_multiagent_MYRUN
```

Write the batch report with this command:

```bash
python scripts/full_games123_multiagent_batch.py report \
  --results-root experiments/results/full_games123_multiagent_MYRUN
```

The controller also supports named selections. Use selections for small reruns or failed cells.

## Reproduce The Bilateral Scaling Sweep

The original bilateral workflow uses a shell generator. The generator makes a time-stamped result directory.

```bash
./scripts/generate_configs_both_orders.sh
```

The generator also updates this convenience link:

```text
experiments/results/scaling_experiment
```

Inspect the generated summary before submission:

```bash
cat experiments/results/scaling_experiment/configs/summary.txt
```

Submit the generated Slurm jobs with this command:

```bash
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh all \
  --max-concurrent 10
```

The generated Slurm files contain absolute paths to the time-stamped config directory. New config generation does not change queued jobs.

Do not delete a time-stamped config directory while its jobs are active.

## Run The Native TTC Batch

Use the TTC generator for the current native-effort experiment.

```bash
python scripts/generate_ttc_native_scaling_jobs.py \
  --results-root experiments/results/ttc_native_scaling_MYRUN \
  --dry-run

python scripts/generate_ttc_native_scaling_jobs.py \
  --results-root experiments/results/ttc_native_scaling_MYRUN \
  --submit
```

The TTC research records in this directory give the experiment rationale and prior results.

## Run The Llama Baseline

Use these scripts for the Llama 3.3 70B replication:

```bash
python scripts/generate_appendix_llama33_baseline_configs.py
python scripts/monitor_appendix_llama33_baseline.py
python scripts/analyze_appendix_llama33_baseline_500.py
```

Read `appendix_llama33_baseline_experiment_spec_2026_05.md` for the frozen experiment specification.

## Examine Results

Most result roots contain these items:

- `configs/` contains the input configurations.
- `runs/` or model directories contain the run results.
- `logs/` contains batch and Slurm logs.
- `monitoring/` contains failure and parse reports.
- `batch_summary.json` contains an aggregate batch summary.

Each completed run usually contains these files:

- `experiment_results.json` contains utilities and the final outcome.
- `all_interactions.json` contains prompts, responses, phases, and token data.
- `agent_interactions/` contains one view for each agent.

## Validate A Change

Run the applicable tests before a production batch.

```bash
pytest -q
```

First, run one config with the same game and provider. Then, examine the result JSON and the interaction JSON.
