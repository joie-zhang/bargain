# Download and explore the transcripts

The full dataset is available through the [anonymous Hugging Face link](https://anonymous-hf.com/a/xgxza3yfsgll/).
Use the commands below on a local computer or remote server with Python 3.12 or later.

## Install and download all data

Set `BARGAIN_ROOT` to the absolute path of your checkout, then run:

```bash
export BARGAIN_ROOT="/absolute/path/to/bargain"
python3 -m venv "$BARGAIN_ROOT/.venv-review"
"$BARGAIN_ROOT/.venv-review/bin/python" -m pip install -r "$BARGAIN_ROOT/ui/requirements-review.txt"
"$BARGAIN_ROOT/.venv-review/bin/python" "$BARGAIN_ROOT/scripts/download_review_transcripts.py" --all --download
```

No Hugging Face account or token is needed.
The download includes all experiment batches, results, transcripts, and saved prompts.
Files go directly into `$BARGAIN_ROOT/experiments/results`, with the folder layout expected by the repository.
The helper verifies file hashes and skips identical existing files; rerun the command after an interrupted download.
It stops if an existing local file differs from the dataset.

# Launch experiments

Each command starts one new negotiation, except test-time compute scaling, which starts four.
These are individual runs, not the full experiment sets from the paper.

## Setup

Use Python 3.12 or later in a virtual environment.
Replace `/absolute/path/to/bargain` with your checkout location.

```bash
python3 -m venv /absolute/path/to/bargain/.venv
source /absolute/path/to/bargain/.venv/bin/activate
python -m pip install -c /absolute/path/to/bargain/constraints-public.txt -e /absolute/path/to/bargain
```

Set the API keys required by the selected models in your environment:

- `OPENAI_API_KEY` for direct OpenAI models.
- `OPENROUTER_API_KEY` for Llama, Gemini, and other OpenRouter models.
- `ANTHROPIC_API_KEY` for direct Claude models.

Alternatively, pass `--env-file /absolute/path/to/private.env` to a launch command.
That file must contain only the required `KEY=value` entries and have permissions `600`.
The launcher does not automatically load a repository environment file.

Use `bargain models` to list model aliases and their providers.
Check local setup before paying for a run:

```bash
bargain doctor two-player --adversary gpt-4o-mini-2024-07-18
```

This checks dependencies and key presence, not authentication or available credit.
Run on a machine with internet access.

## Two-player with a GPT-5-nano baseline

One adversary model negotiates with one fixed GPT-5-nano agent.
Change `--adversary` to compare different models against the same baseline.
This example needs an OpenAI API key only.

```bash
bargain run two-player --adversary gpt-4o-mini-2024-07-18
```

## Two-player with a Llama baseline

The same design, with Llama 3.3 70B as the fixed baseline.
This example needs both OpenAI and OpenRouter API keys.

```bash
bargain run two-player-llama --adversary gpt-4o-mini-2024-07-18
```

## Homogeneous groups

Every agent uses the same model.
Use `--agents` to set the group size.
This example needs an OpenAI API key only.

```bash
bargain run homogeneous --model gpt-4o-mini-2024-07-18 --agents 4
```

## Heterogeneous groups

Each agent uses a different model sampled from the fixed 24-model pool.
The seed determines the group and seat order.
Required API keys depend on the sampled models; inspect the plan before running.

```bash
bargain plan heterogeneous --agents 4 --seed 42
bargain run heterogeneous --agents 4 --seed 42
```

Use `--stratum 0` through `--stratum 4` to select groups with smaller to larger spreads in model Elo.

## Homogeneous-adversary groups

One adversary model negotiates with `n-1` GPT-5-nano agents.
This tests how the adversary's payoff changes with group size.
This example needs an OpenAI API key only.

```bash
bargain run homogeneous-adversary --adversary gpt-4o-mini-2024-07-18 --agents 4
```

## Test-time compute scaling

Run the same two-player game instance at four requested reasoning-effort levels.
The opponent is GPT-5-nano in every run.
Choose one family:

```bash
bargain run ttc --family gpt5
bargain run ttc --family claude
bargain run ttc --family gemini
```

- `gpt5`: GPT-5 at minimal, low, medium, and high effort; OpenAI only.
- `claude`: Claude Sonnet 4.6 at low, medium, high, and max effort; Anthropic and OpenAI.
- `gemini`: Gemini 3 Flash at minimal, low, medium, and high effort; OpenRouter and OpenAI.

Keep the default output limits when testing the full effort range.
In past tests, GPT-5 exhausted a 4,096-token limit on reasoning before producing an answer.

## Coordinated team

GPT-5.4 High negotiates with a GPT-5-nano team in Game 1.
Team members share preferences, plan privately, and use a captain's binding proposal and ballot.
This needs OpenAI only.

```bash
bargain run team --agents 4
```

At `--agents 2`, the single Nano agent receives no team treatment.
These are new runs, not comparisons with historical controls.

## Common options

- `--game game1`: item allocation, the default.
- `--game game2`: treaty negotiation.
- `--game game3`: project co-funding.
- `--agents 2|4|6|8|10`: group size; two-player and TTC presets require two.
- `--position last`: put the adversary in the last seat.
- `--seed 42`: set the preference-generation seed.
- `--rounds 10`: maximum negotiation rounds.
- `--discussion-turns 2`: public discussion turns per round.
- `--max-tokens 4096`: set a smaller per-call output limit, including reasoning tokens where applicable.
- `--output /absolute/path/to/new-output`: choose a new output directory.

Game-specific preference and budget options:

- Game 1: `--competition 0.5` requests preference cosine similarity, despite its legacy name; larger values mean more similar preferences.
- Game 2: `--rho 0 --theta 1` sets ideal-position correlation and issue-weight cosine similarity.
- Game 3: `--alpha 0.5 --sigma 0.6` sets valuation cosine similarity and the total-budget-to-total-cost ratio.

The team preset supports Game 1 only.
Lower output limits can cause incomplete model answers and failed runs.

For a small test with cheap models:

```bash
bargain run two-player --adversary gpt-4o-mini-2024-07-18 --game game1 --rounds 1 --discussion-turns 1 --max-tokens 4096
```

## Inspect results and resume

The launcher prints the absolute output directory and saves the plan, preferences, interactions, requests, and results there.
Use that directory in these commands:

```bash
bargain status /absolute/path/to/output
bargain summarize /absolute/path/to/output
bargain resume /absolute/path/to/output
```

The saved plan records exact API model IDs and reasoning settings.
`bargain summarize` reports launch aliases and TTC effort.
The older `agent_performance.model` field can mislabel o3-mini and GPT-5 effort variants.

Resume skips verified completed runs.
Restarting failed runs requires `--retry-failed` and can incur additional charges.
An unknown request outcome also requires `--accept-unknown-outcome`, because the interrupted request could already have been billed.

# Results visualizer

## Open the viewer

Use the `BARGAIN_ROOT` and `.venv-review` environment from the setup above.
Run this on the machine holding the data:

```bash
"$BARGAIN_ROOT/.venv-review/bin/python" -m streamlit run \
  "$BARGAIN_ROOT/ui/transcript_review.py" \
  --server.address 127.0.0.1 --server.port 8010 \
  --server.headless true --browser.gatherUsageStats false
```

**Local computer:** Open **http://127.0.0.1:8010** in your browser.
Keep the terminal open while using the viewer.

**Remote server:** Check that the viewer is running on that server:

```bash
curl --fail http://127.0.0.1:8010/_stcore/health
```

The response should be `ok`.
Then run this on your laptop, using the username and host of the machine running the viewer:

```bash
ssh -N -L 8010:127.0.0.1:8010 YOUR_USERNAME@VIEWER_HOST
```

If that machine requires a jump host, use:

```bash
ssh -N -L 8010:127.0.0.1:8010 -J YOUR_USERNAME@LOGIN_HOST YOUR_USERNAME@VIEWER_HOST
```

Keep the tunnel open and visit **http://127.0.0.1:8010** on your laptop.
On a cluster, install and download on a node with internet access; the viewer runs offline and needs no GPU.

## Explore the data

- Select experiment batches and filter by game, model, seed, or agreement.
- Choose a run to read messages, prompts, outcomes, and saved configuration.
- Search conversations and filter messages by speaker or phase.
- Export transcript JSON or the visible run index as CSV.
