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
| Run a small new experiment | [Run an experiment](#run-an-experiment) | Python, a model API key, and direct internet access |
| Analyze or reproduce the paper | [Reproduce the paper](#reproduce-the-paper) | Released data for analysis; provider access and Slurm for the documented batches |
| Change or test the code | [Development](#development) | A checkout and the test dependencies |

## Set up Python

Use **Python 3.12 or later**. The experiment launcher supports Linux and macOS.
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

Use the `bargain` launcher for a small new experiment.
It needs direct internet access to the selected providers and no GPU or Slurm
account. For restricted cluster nodes, use the
[cluster procedures](docs/reproduction.md).

Install the launcher with its reference dependency versions:

```bash
python -m pip install -c "$BARGAIN_ROOT/constraints-public.txt" -e "$BARGAIN_ROOT"
bargain --help
```

### Check the setup

This example runs GPT-5 Nano against GPT-4o Mini through OpenAI.
It needs only `OPENAI_API_KEY`.
Set your key in the terminal, or use the
[private credential file option](docs/guides/public_quickstart.md#set-up-credentials).
Do not put keys in the repository.

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
bargain models
bargain doctor two-player --adversary gpt-4o-mini-2024-07-18 --rounds 2
bargain plan two-player --adversary gpt-4o-mini-2024-07-18 --rounds 2
```

`models` lists accepted model names and their providers.
`doctor` checks installed dependencies and key presence.
`plan` prints the resolved experiment settings.
These commands make no model calls; `doctor` does not test authentication.

### Run and inspect

The next command makes paid model calls.
It runs one item-allocation negotiation with at most two rounds.
Use a new output directory for each new run.

```bash
RUN_DIR="$BARGAIN_ROOT/bargain-runs/first-run"
bargain run two-player --adversary gpt-4o-mini-2024-07-18 \
  --rounds 2 --output "$RUN_DIR"
bargain status "$RUN_DIR"
bargain summarize "$RUN_DIR"
```

A complete run ends with `agreement` or `disagreement`.
The summary includes utilities and the final round.
A failed or interrupted run is incomplete; inspect its saved status before
using the [resume procedure](docs/guides/public_quickstart.md#resume-without-hiding-failures).

The output directory contains the resolved plan and a `runs/` directory with
attempt records, prompts, responses, results, and status files.
Use `bargain status` and `bargain summarize` for these outputs.
The transcript viewer above reads the released paper format.

### Choose another game or group

| Game | Launcher option | Agents negotiate over |
| --- | --- | --- |
| Item allocation | `--game game1` (default) | Indivisible items with private values |
| Diplomacy | `--game game2` | Treaty issues with private preferred positions and weights |
| Co-funding | `--game game3` | Contributions to projects that need a funding threshold |

The launcher also supports Llama-baseline pairs, groups with identical or
mixed models, one adversary model among baseline agents, coordinated teams,
and test-time compute, which varies a model's reasoning effort.
See the [experiment examples](docs/guides/public_quickstart.md#choose-an-experiment)
for commands and the [game guide](docs/game_architecture.md) for the rules.

**New launcher runs use the `public-hosted-v1` protocol.**
Its provider routes, context handling, and failure handling differ from the
historical paper runners.
Read the [protocol differences](docs/guides/public_quickstart.md#defaults-and-protocol-differences)
before comparing new runs with paper results.

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
python -m pytest "$BARGAIN_ROOT/tests/test_public_launch.py" \
  "$BARGAIN_ROOT/tests/test_transcript_review.py"
```

Run the full suite when a change affects shared behavior:

```bash
python -m pytest "$BARGAIN_ROOT/tests/"
```

The launcher tests replace paid providers at the test boundary.
They do not establish live provider access.
Validate provider changes with a small real run before a research batch.

| Code or documentation | Purpose |
| --- | --- |
| [Public launcher](strong_models_experiment/cli.py) and [launch modules](strong_models_experiment/launch/) | Plan, run, inspect, and resume new experiments |
| [Paper runner](run_strong_models_experiment.py) | Historical single-run and batch entry point |
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
