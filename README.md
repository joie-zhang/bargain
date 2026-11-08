# Scaling Capabilities and Competition in Strategic Interactions

This repository studies how language-model capability, group size, and
competition affect negotiation outcomes. It includes three games, saved
experiment data, transcript viewers, and the code for the paper experiments.
The experiments ask whether stronger models increase the group's total payoff,
take a larger share, or both.

## Choose a task

| I want to... | Start here | Requirements |
| --- | --- | --- |
| Read saved negotiations | [Inspect released results](#inspect-released-results) | Python and internet access for the download; no model API key |
| Run a new experiment with the paper runner | [Run an experiment](#run-an-experiment) | Python and access to the selected model providers |
| Analyze or reproduce the paper | [Reproduce the paper](#reproduce-the-paper) | Released data for analysis; provider access and Slurm for the documented batches |
| Change or test the code | [Development](#development) | A checkout and the test dependencies |

## Set up Python

Use **Python 3.12 or later**.
Replace the path below with the absolute path to your checkout.
Run the remaining commands in the same terminal.

```bash
export BARGAIN_ROOT="/absolute/path/to/bargain"
cd "$BARGAIN_ROOT"
python3 -m venv "$BARGAIN_ROOT/.venv"
source "$BARGAIN_ROOT/.venv/bin/activate"
```

If the environment already exists, activate it with the `source` command.
Install the dependencies for your task below.

## Inspect released results

The [released dataset](https://anonymous-hf.com/a/xgxza3yfsgll/) contains saved
configurations, results, transcripts, and prompts.
You can read these without running experiments or paying for model calls.

Install the viewer dependencies and list the available batches:

```bash
python -m pip install -r "$BARGAIN_ROOT/ui/requirements.txt"
python "$BARGAIN_ROOT/scripts/download_review_transcripts.py" --list
```

To try a small download, copy a batch name from that list:

```bash
BATCH="PASTE_BATCH_NAME_HERE"
python "$BARGAIN_ROOT/scripts/download_review_transcripts.py" \
  --batch "$BATCH" --limit-runs 3 --download
```

To download the complete release instead:

```bash
python "$BARGAIN_ROOT/scripts/download_review_transcripts.py" --all --download
```

Omit `--download` to preview the selected files and total size.
Downloads go into `experiments/results/` under the checkout.
The downloader verifies file hashes, skips identical files, and stops if an
existing file differs from the release.

Start the transcript viewer:

```bash
python -m streamlit run "$BARGAIN_ROOT/ui/transcript_review.py" \
  --server.address 127.0.0.1 --server.port 8010 \
  --server.headless true --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8010**, select your downloaded batch, and choose a run.
You can inspect the conversation, prompts, configuration, and outcome.
Keep the terminal open while using the viewer.

If the viewer runs on a remote machine, run this command on your laptop:

```bash
ssh -N -L 8010:127.0.0.1:8010 YOUR_USERNAME@VIEWER_HOST
```

Then open the same browser address on your laptop.
The [viewer guide](docs/guides/negotiation_viewer.md) covers jump hosts,
connection checks, and game-specific viewers.

## Run an experiment

Use [run_strong_models_experiment.py](run_strong_models_experiment.py), the
historical single-run and bilateral batch entry point.
Use the original batch scripts below for the other experiment families.
These commands use the existing agents, prompts, parsers, and context handling.

Install the experiment dependencies:

```bash
python -m pip install -r "$BARGAIN_ROOT/requirements.txt"
python -m pip install -e "${BARGAIN_ROOT}[analysis]"
python "$BARGAIN_ROOT/run_strong_models_experiment.py" --help
```

This example makes paid OpenAI calls for one Game 1 negotiation with ten rounds,
two discussion turns per round, and a 0.9 round discount.
Use a new output directory and keep accepted paper results separate.

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
RUN_DIR="$BARGAIN_ROOT/experiments/results/my_game1_run"
python "$BARGAIN_ROOT/run_strong_models_experiment.py" \
  --game-type item_allocation \
  --models gpt-5-nano gpt-4o-mini-2024-07-18 \
  --competition-level 0.5 --num-items 5 \
  --max-rounds 10 --discussion-turns 2 --gamma-discount 0.9 \
  --max-tokens-per-phase 16384 --model-order weak_first \
  --random-seed 42 --batch --num-runs 1 --output-dir "$RUN_DIR"
```

Inspect the saved result and interaction JSON files using the
[result-file guide](docs/reproduction.md#read-result-files).
Confirm that both files belong to the same experiment and that the run completed
before using its outcome.
The [paper-runner examples](docs/reproduction.md#run-one-experiment-with-the-paper-runner)
also cover diplomacy and co-funding.
Provider credentials and restricted-node transport are described in the
[operations guide](docs/operations.md).

### Use the original batch scripts

| Experiment family | Existing scripts | Procedure |
| --- | --- | --- |
| GPT-5 Nano bilateral | [generate_configs_both_orders.sh](scripts/generate_configs_both_orders.sh), [generate_diplomacy_configs.sh](scripts/generate_diplomacy_configs.sh), [generate_cofunding_configs.sh](scripts/generate_cofunding_configs.sh) | [Bilateral batches](docs/reproduction.md#2-run-the-gpt-5-nano-bilateral-batches) |
| Llama bilateral | [generate_appendix_llama33_baseline_configs.py](scripts/generate_appendix_llama33_baseline_configs.py) | [Llama batches](docs/reproduction.md#3-run-the-llama-33-bilateral-batches) |
| Homogeneous adversary and heterogeneous groups | [full_games123_multiagent_batch.py](scripts/full_games123_multiagent_batch.py) | [Multi-agent batch](docs/reproduction.md#4-run-the-multi-agent-batch) |
| Homogeneous groups | [random_monoculture_control_batch.py](scripts/random_monoculture_control_batch.py) | [Homogeneous batch](docs/reproduction.md#5-run-the-random-monoculture-batch) |
| Native test-time compute | [generate_ttc_seed_replication_jobs.py](scripts/generate_ttc_seed_replication_jobs.py) | [Saved seed design](docs/reproduction.md#6-run-the-native-test-time-compute-batch) |
| Coordinated teams | [generate_game1_gpt54_binding_team.py](scripts/generate_game1_gpt54_binding_team.py) | [Matched team batch](docs/reproduction.md#7-run-the-coordinated-team-batch) |

For historical reproduction, use the released configurations, seeds, model
routes, and exclusions in the [reproduction guide](docs/reproduction.md).

## Reproduce the paper

Start with the [paper reproduction guide](docs/reproduction.md).
It preserves the detailed batch commands, seeds, dated inputs, run counts,
and exclusions used by the paper workflows.

- **Analyze released results:** download the required inputs, then follow the
  [analysis sequence](docs/reproduction.md#9-validate-and-analyze-the-results).
- **Regenerate figures:** use the [renderer guide](scripts/paper_figures/README.md)
  and [figure manifest](docs/reproducibility/paper_figure_manifest.md).
- **Run batches:** follow the [batch procedures](docs/reproduction.md#paper-inventory-and-batch-procedures)
  and [credential and network instructions](docs/operations.md).

Analysis reads saved results and does not require model calls.
Batch execution makes model calls and requires the relevant provider and
cluster access. Keep accepted results separate from reruns.

## Development

Use the same activated environment and install the development dependencies:

```bash
python -m pip install -r "$BARGAIN_ROOT/requirements.txt"
python -m pip install -e "${BARGAIN_ROOT}[test,analysis]"
```

Start with the tests for the workflow you change:

```bash
python -m pytest "$BARGAIN_ROOT/tests/test_context_compaction.py" \
  "$BARGAIN_ROOT/tests/test_transcript_review.py"
```

Run the full suite when a change affects shared behavior:

```bash
python -m pytest "$BARGAIN_ROOT/tests/"
```

Offline tests do not establish live provider access.
Validate provider changes with a small real run before a research batch.

| Code or documentation | Purpose |
| --- | --- |
| [Paper runner](run_strong_models_experiment.py) | Historical single-run and bilateral batch entry point |
| [Game environments](game_environments/) | Game state, prompts, action validation, and utility calculations |
| [Experiment controller](strong_models_experiment/experiment.py) | Shared negotiation flow |
| [Provider clients](negotiation/) | API clients, proxy transport, and context handling for the paper runners |
| [Scripts](scripts/) | Batch generation, submission, analysis, and plotting |
| [Viewers](ui/) | Transcript and game displays |
| [Documentation index](docs/README.md) | Current guides and dated research records |

Read [AGENTS.md](AGENTS.md) before changing experiment behavior.
Keep credentials and generated run data out of commits.

## License

[MIT License](LICENSE).
