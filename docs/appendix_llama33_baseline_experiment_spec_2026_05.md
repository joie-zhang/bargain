# Appendix Experiment Spec: Llama 3.3 70B Baseline Robustness Sweep

Created: 2026-05-03

This document specifies an appendix experiment batch that repeats the canonical
two-agent (`N=2`) bargaining experiments with a different fixed baseline model.
The original canonical runs used `gpt-5-nano` as the fixed baseline and swept a
large adversary roster across model Elo. This appendix batch keeps the same
three games and the same core parameter grids, but replaces the baseline with
`llama-3.3-70b-instruct` and uses a smaller, high-context 10-model adversary
slate.

The purpose is not to discover a new main result. The purpose is to test whether
the paper's payoff-vs-Elo trend is robust to changing the fixed bargaining
partner from an OpenAI nano model to a non-OpenAI, long-context, near-midpoint
model.

## Summary

- Baseline model: `llama-3.3-70b-instruct`
- Adversary models: 10 models selected from the March 31, 2026 smooth Elo guide
- Games:
  - Game 1: item allocation
  - Game 2: diplomatic treaty
  - Game 3: co-funding
- Agents per game: `N=2`
- Discussion turns per round: `2`
- Max rounds per game: `10`
- Planned configs:
  - Game 1: `140`
  - Game 2: `180`
  - Game 3: `180`
  - Total: `500`

Recommended result roots:

```text
experiments/results/appendix_llama33_baseline_game1_202605/
experiments/results/appendix_llama33_baseline_game2_202605/
experiments/results/appendix_llama33_baseline_game3_202605/
```

The date suffix is a batch slug, not an analysis variable. If multiple attempts
are run, append a timestamp, for example
`appendix_llama33_baseline_game1_20260503_153000`.

## Motivation

The main paper's canonical `N=2` experiments ask whether adversary model payoff
increases with model capability, measured by Arena Elo, when the adversary
bargains against a fixed baseline model. In the canonical results, the fixed
baseline is `gpt-5-nano`. The observed trend is positive in all three games:
stronger adversary models tend to secure higher realized utility.

The appendix should answer a robustness question:

> Does the payoff-vs-Elo trend persist when the fixed bargaining partner is not
> `gpt-5-nano`?

This matters because a single fixed baseline can confound two things:

1. A general capability advantage of higher-Elo models.
2. A partner-specific exploitability pattern of the chosen baseline.

Changing the baseline to `llama-3.3-70b-instruct` tests whether the trend is
specific to `gpt-5-nano` or whether it appears against a separate provider,
architecture family, and route.

## Baseline Choice

### Selected Baseline

`llama-3.3-70b-instruct`

Metadata from `docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md`:

| Field | Value |
|---|---|
| Arena Elo | `1318` |
| Arena Context | `128K` |
| OpenRouter Context | `131,072` |
| Arena Org | Meta |
| Repo Route | OpenRouter |
| Model ID / Route | `meta-llama/llama-3.3-70b-instruct` |
| Key Needed | `OPENROUTER_API_KEY` |
| Repo Support | yes |

### Why This Baseline

`llama-3.3-70b-instruct` is a strong appendix baseline for four reasons.

First, it is close to the old baseline's Elo. The old baseline,
`gpt-5-nano-high`, is listed at Elo `1337`; `llama-3.3-70b-instruct` is Elo
`1318`. This means the appendix changes model family and provider without
moving the baseline to a completely different skill regime.

Second, it is non-OpenAI. The main experiments already use an OpenAI baseline,
so a Meta model routed through OpenRouter gives a cleaner partner-robustness
check.

Third, it has high enough context for the full game transcripts and reasoning
phases. The realized OpenRouter context is `131,072`, which satisfies the
long-context requirement.

Fourth, it is not a top-tier model. A top baseline such as
`claude-opus-4-6-thinking` or `gpt-5.4-high` would compress the payoff range for
upper-tier adversaries and would answer a different question: whether models can
bargain against an extremely strong opponent. This appendix instead asks whether
the original trend survives a realistic baseline swap.

## Adversary Model Set

The adversary slate contains 10 high-context models. The selection criteria are:

1. Span as much of the usable Elo range as possible.
2. Avoid known low-context or route-limited models.
3. Avoid very weak small Llama models that are likely to introduce context-linked
   failures unrelated to bargaining capability.
4. Keep provider diversity where possible.
5. Include enough upper-tier models to test whether the trend remains visible at
   the high end.
6. Exclude the baseline model itself from the adversary set.

### Selected 10 Models

| # | Model | Elo | Context | Provider / Route | Why selected |
|---:|---|---:|---|---|---|
| 1 | `amazon-nova-micro-v1.0` | 1240 | 128K OpenRouter | Amazon via OpenRouter | Lowest clean high-context anchor after excluding the small Llama models. Keeps the low end of the Elo range without the 60K/80K realized-context issues. |
| 2 | `claude-3-haiku-20240307` | 1260 | 200K | Anthropic | Stable low-end Anthropic anchor with a large context window and direct provider support. |
| 3 | `amazon-nova-pro-v1.0` | 1290 | 300K OpenRouter | Amazon via OpenRouter | Low-mid model with very large context and a different provider from the baseline. |
| 4 | `gpt-4o-mini-2024-07-18` | 1317 | 128K | OpenAI | Near-baseline Elo comparator. Since the baseline is Elo 1318, this model tests behavior at approximately matched Arena capability. |
| 5 | `deepseek-v3` | 1358 | 128K Arena / 163,840 OpenRouter | DeepSeek via OpenRouter | Midrange model with strong realized context and non-OpenAI route. |
| 6 | `claude-sonnet-4-20250514` | 1389 | 200K | Anthropic | Upper-mid Anthropic model. Useful bridge between midrange and top-tier systems. |
| 7 | `deepseek-r1-0528` | 1422 | 163,840 OpenRouter | DeepSeek via OpenRouter | High-performing reasoning/open model with long realized context. |
| 8 | `gemini-2.5-pro` | 1448 | 1M | Google | High-context Google model, adds provider diversity and upper-tier coverage. |
| 9 | `gpt-5.4-high` | 1484 | 1.1M | OpenAI | Top OpenAI point. Important for preserving the high-Elo end of the appendix trend. |
| 10 | `claude-opus-4-6-thinking` | 1504 | 1M | Anthropic | Highest-Elo endpoint in the guide. Defines the top of the adversary sweep. |

### Models Explicitly Not Selected

The following models are intentionally excluded from the 10-model appendix slate.

| Model | Reason for exclusion |
|---|---|
| `llama-3.2-1b-instruct` | Too weak for this appendix and has only `60,000` realized OpenRouter context. |
| `llama-3.2-3b-instruct` | Removed by design. It is too weak for this robustness appendix and has likely context-linked issues, with only `80,000` realized OpenRouter context. |
| `llama-3.1-8b-instruct` | Arena context is 128K, but realized OpenRouter context is only `16,384`, which violates the high-context intent. |
| `qwen2.5-72b-instruct` | Arena context is 131K, but realized OpenRouter context is only `32,768`, so it is not a clean high-context choice here. |
| `llama-3.3-70b-instruct` | This is the fixed baseline, so it must not also appear as an adversary. |
| `gpt-4.1-nano-2025-04-14` | Long context, but less useful here because it is another nano-style OpenAI comparator close to the old baseline family. |
| `qwq-32b` | Viable context, but omitted to keep the 10-model slate evenly spaced and avoid crowding the near-baseline region. |
| `gemma-3-27b-it` | Viable context, but lower priority than DeepSeek, Claude, Gemini, Amazon, and OpenAI anchors for this appendix. |
| `o3-mini-high` | Viable context, but not selected because the set already includes `gpt-4o-mini` near the baseline and `gpt-5.4-high` at the OpenAI high end. |
| `qwen3-max-preview` | Strong long-context candidate, but omitted from the recommended 10 to preserve the exact 10-model budget while keeping Gemini, DeepSeek, Anthropic, Amazon, and OpenAI coverage. |

If one selected model becomes unavailable at run time, the preferred replacement
order is:

1. `qwen3-max-preview` (`1435`, 262K OpenRouter)
2. `o3-mini-high` (`1363`, 200K)
3. `qwq-32b` (`1336`, 131K OpenRouter)

Do not replace a failed selected model with any of the three excluded small
Llama models.

## Common Experiment Settings

These settings apply to all three games.

| Parameter | Value |
|---|---|
| Number of agents | `2` |
| Baseline model | `llama-3.3-70b-instruct` |
| Adversary models | 10-model slate above |
| Max rounds | `10` |
| Discussion turns per round | `2` |
| Private thinking | enabled |
| Public discussion | enabled |
| Individual reflection | enabled |
| Gamma discount | `0.9` |
| Runs per unique config | `1` |
| Model orders | both orders |
| Max tokens per phase | `10500` |

The `max_tokens_per_phase=10500` cap matches the canonical Game 2 and Game 3
standard runs. Use the same cap for Game 1 in this appendix so every game has a
consistent response budget.

### Model Order Encoding

The legacy codebase uses `weak_first` and `strong_first` as two-agent order
labels. In this appendix, those labels should be interpreted positionally, not
literally.

For every adversary model, define:

```text
model1 = baseline = llama-3.3-70b-instruct
model2 = adversary
```

Then encode model orders as follows:

| Conceptual order | Legacy `model_order` | Runtime `models` list | Agent mapping |
|---|---|---|---|
| Baseline speaks first | `weak_first` | `[baseline, adversary]` | `Agent_1=baseline`, `Agent_2=adversary` |
| Adversary speaks first | `strong_first` | `[adversary, baseline]` | `Agent_1=adversary`, `Agent_2=baseline` |

The words `weak` and `strong` should not be used analytically in this appendix.
Use `baseline_first` and `adversary_first` in paper text and plot labels.

### Seed Policy

Use the following seed policy to mirror the canonical April 2026 result folders:

| Game | Seed rule |
|---|---|
| Game 1 | `random_seed = 42` for every config |
| Game 2 | `random_seed = 42 + experiment_id` |
| Game 3 | `random_seed = 42 + experiment_id` |

This intentionally preserves the historical convention where Game 1 uses a
fixed seed across its competition grid, while Games 2 and 3 increment seeds by
config id.

## Exact Config Matrix

This section defines every planned config. The full batch is the Cartesian
product of the model slate, the game-specific parameter grid, and both model
orders.

### Adversary Order

Use this exact model order for config id assignment:

```text
0  amazon-nova-micro-v1.0
1  claude-3-haiku-20240307
2  amazon-nova-pro-v1.0
3  gpt-4o-mini-2024-07-18
4  deepseek-v3
5  claude-sonnet-4-20250514
6  deepseek-r1-0528
7  gemini-2.5-pro
8  gpt-5.4-high
9  claude-opus-4-6-thinking
```

### Model Orders

Use this exact order list for config id assignment:

```text
0  weak_first      # baseline first
1  strong_first    # adversary first
```

## Game 1: Item Allocation

### Game Definition

Game 1 is the item allocation bargaining game. Two agents bargain over five
discrete items. Each agent has a private valuation vector over the items, and
the final payoff is the value of the bundle assigned to that agent, with time
discounting as implemented by the experiment runner.

### Game 1 Fixed Settings

| Parameter | Value |
|---|---|
| `game_type` | `item_allocation` |
| `num_items` / `m_items` | `5` |
| `max_rounds` / `t_rounds` | `10` |
| `discussion_turns` | `2` |
| `gamma_discount` | `0.9` |
| `max_tokens_per_phase` | `10500` |
| `random_seed` | `42` |

### Game 1 Parameter Grid

```text
competition_level in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
model_order in [weak_first, strong_first]
```

### Game 1 Config Count

```text
10 adversary models x 7 competition levels x 2 model orders = 140 configs
```

### Game 1 Config Id Formula

Let:

```text
model_index in [0..9]
competition_index in [0..6]
order_index in [0..1]
```

Then:

```text
experiment_id = model_index * 14 + competition_index * 2 + order_index
```

The resulting id range is `0..139`.

### Game 1 Output Directory Template

```text
experiments/results/appendix_llama33_baseline_game1_202605/
  llama-3.3-70b-instruct_vs_{adversary}/
    {model_order}/
      comp_{competition_level}/
        turns_2/
          run_{run_number}/
```

Use:

```text
run_number = 1 for weak_first
run_number = 2 for strong_first
```

Example config:

```json
{
  "experiment_id": 0,
  "experiment_type": "appendix_llama33_baseline",
  "game_type": "item_allocation",
  "baseline_model": "llama-3.3-70b-instruct",
  "adversary_model": "amazon-nova-micro-v1.0",
  "weak_model": "llama-3.3-70b-instruct",
  "strong_model": "amazon-nova-micro-v1.0",
  "models": ["llama-3.3-70b-instruct", "amazon-nova-micro-v1.0"],
  "model_order": "weak_first",
  "run_number": 1,
  "num_runs": 1,
  "max_tokens_per_phase": 10500,
  "num_items": 5,
  "max_rounds": 10,
  "gamma_discount": 0.9,
  "competition_level": 0.0,
  "discussion_turns": 2,
  "random_seed": 42,
  "output_dir": "experiments/results/appendix_llama33_baseline_game1_202605/llama-3.3-70b-instruct_vs_amazon-nova-micro-v1.0/weak_first/comp_0.0/turns_2/run_1"
}
```

## Game 2: Diplomatic Treaty

### Game Definition

Game 2 is the diplomatic treaty bargaining game. Two agents negotiate continuous
agreement values across five treaty issues. Each agent has private ideal points
and issue weights. The two native game parameters are:

- `rho`: preference correlation between agents' ideal positions
- `theta`: overlap between agents' issue-importance vectors

The derived competition index for plots is:

```text
CI2 = theta * (1 - rho) / 2
```

### Game 2 Fixed Settings

| Parameter | Value |
|---|---|
| `game_type` | `diplomacy` |
| `n_issues` | `5` |
| `max_rounds` / `t_rounds` | `10` |
| `discussion_turns` | `2` |
| `gamma_discount` | `0.9` |
| `max_tokens_per_phase` | `10500` |

### Game 2 Parameter Grid

```text
rho in [-1.0, 0.0, 1.0]
theta in [0.0, 0.5, 1.0]
model_order in [weak_first, strong_first]
```

### Game 2 Config Count

```text
10 adversary models x 3 rho values x 3 theta values x 2 model orders = 180 configs
```

### Game 2 Config Id Formula

Let:

```text
model_index in [0..9]
rho_index in [0..2]      # order: -1.0, 0.0, 1.0
theta_index in [0..2]    # order: 0.0, 0.5, 1.0
order_index in [0..1]
```

Then:

```text
experiment_id = model_index * 18 + rho_index * 6 + theta_index * 2 + order_index
random_seed = 42 + experiment_id
```

The resulting id range is `0..179`.

### Game 2 Output Directory Template

Use the same directory slug format as the canonical diplomacy batch:

```text
experiments/results/appendix_llama33_baseline_game2_202605/
  model_scale/
    llama-3.3-70b-instruct_vs_{adversary}/
      {model_order}/
        rho_{rho_slug}_theta_{theta_slug}/
```

Slug rules:

```text
-1.0 -> n1_0
 0.0 -> 0_0
 0.5 -> 0_5
 1.0 -> 1_0
```

Example:

```text
rho=-1.0, theta=0.5 -> rho_n1_0_theta_0_5
```

Example config:

```json
{
  "experiment_id": 0,
  "experiment_type": "appendix_llama33_baseline",
  "game_type": "diplomacy",
  "baseline_model": "llama-3.3-70b-instruct",
  "adversary_model": "amazon-nova-micro-v1.0",
  "model1": "llama-3.3-70b-instruct",
  "model2": "amazon-nova-micro-v1.0",
  "models": ["llama-3.3-70b-instruct", "amazon-nova-micro-v1.0"],
  "model_order": "weak_first",
  "run_number": 1,
  "num_runs": 1,
  "max_tokens_per_phase": 10500,
  "n_issues": 5,
  "rho": -1.0,
  "theta": 0.0,
  "max_rounds": 10,
  "gamma_discount": 0.9,
  "discussion_turns": 2,
  "random_seed": 42,
  "output_dir": "experiments/results/appendix_llama33_baseline_game2_202605/model_scale/llama-3.3-70b-instruct_vs_amazon-nova-micro-v1.0/weak_first/rho_n1_0_theta_0_0"
}
```

## Game 3: Co-Funding

### Game Definition

Game 3 is the co-funding public-goods game. Two agents decide how much to
pledge to five public projects. Projects are funded if aggregate pledges meet
their costs. Agents receive value from funded projects according to private
valuation vectors and pay their own successful contributions.

The two native game parameters are:

- `alpha`: preference alignment between agents' project valuations
- `sigma`: budget abundance, defined by the total budget relative to total
  project cost

The derived competition index for plots is:

```text
CI3 = (1 - alpha) * (1 - sigma)
```

### Game 3 Fixed Settings

| Parameter | Value |
|---|---|
| `game_type` | `co_funding` |
| `m_projects` | `5` |
| `c_min` | `10.0` |
| `c_max` | `30.0` |
| `cofunding_discussion_transparency` | `own` |
| `cofunding_enable_commit_vote` | `true` |
| `cofunding_enable_time_discount` | `true` |
| `cofunding_time_discount` | `0.9` |
| `max_rounds` / `t_rounds` | `10` |
| `discussion_turns` | `2` |
| `gamma_discount` | `0.9` |
| `max_tokens_per_phase` | `10500` |

### Game 3 Parameter Grid

```text
alpha in [0.0, 0.5, 1.0]
sigma in [0.2, 0.6, 1.0]
model_order in [weak_first, strong_first]
```

### Game 3 Config Count

```text
10 adversary models x 3 alpha values x 3 sigma values x 2 model orders = 180 configs
```

### Game 3 Config Id Formula

Let:

```text
model_index in [0..9]
alpha_index in [0..2]    # order: 0.0, 0.5, 1.0
sigma_index in [0..2]    # order: 0.2, 0.6, 1.0
order_index in [0..1]
```

Then:

```text
experiment_id = model_index * 18 + alpha_index * 6 + sigma_index * 2 + order_index
random_seed = 42 + experiment_id
```

The resulting id range is `0..179`.

### Game 3 Output Directory Template

```text
experiments/results/appendix_llama33_baseline_game3_202605/
  model_scale/
    llama-3.3-70b-instruct_vs_{adversary}/
      {model_order}/
        alpha_{alpha_slug}_sigma_{sigma_slug}/
```

Slug rules:

```text
0.0 -> 0_0
0.2 -> 0_2
0.5 -> 0_5
0.6 -> 0_6
1.0 -> 1_0
```

Example config:

```json
{
  "experiment_id": 0,
  "experiment_type": "appendix_llama33_baseline",
  "game_type": "co_funding",
  "baseline_model": "llama-3.3-70b-instruct",
  "adversary_model": "amazon-nova-micro-v1.0",
  "model1": "llama-3.3-70b-instruct",
  "model2": "amazon-nova-micro-v1.0",
  "models": ["llama-3.3-70b-instruct", "amazon-nova-micro-v1.0"],
  "model_order": "weak_first",
  "run_number": 1,
  "num_runs": 1,
  "max_tokens_per_phase": 10500,
  "m_projects": 5,
  "alpha": 0.0,
  "sigma": 0.2,
  "c_min": 10.0,
  "c_max": 30.0,
  "cofunding_discussion_transparency": "own",
  "cofunding_enable_commit_vote": true,
  "cofunding_enable_time_discount": true,
  "cofunding_time_discount": 0.9,
  "max_rounds": 10,
  "gamma_discount": 0.9,
  "discussion_turns": 2,
  "random_seed": 42,
  "output_dir": "experiments/results/appendix_llama33_baseline_game3_202605/model_scale/llama-3.3-70b-instruct_vs_amazon-nova-micro-v1.0/weak_first/alpha_0_0_sigma_0_2"
}
```

## Queue Plan

Submit as three separate Slurm arrays, one per game. Keeping games separate
makes monitoring, retries, and analysis cleaner.

| Batch | Result root | Config count | Suggested array |
|---|---|---:|---|
| Game 1 | `appendix_llama33_baseline_game1_202605` | 140 | `0-139` |
| Game 2 | `appendix_llama33_baseline_game2_202605` | 180 | `0-179` |
| Game 3 | `appendix_llama33_baseline_game3_202605` | 180 | `0-179` |
| Total | all three roots | 500 | three arrays |

Recommended initial concurrency:

```text
Game 1: 10 concurrent tasks
Game 2: 10 concurrent tasks
Game 3: 10 concurrent tasks
```

If provider rate limits are observed, reduce to `5` concurrent tasks per game.

All selected models are hosted API models. No Hugging Face model download is
required. If these jobs are run from Slurm compute nodes with restricted
outbound network access, route OpenRouter calls through the established
file-based OpenRouter proxy workflow rather than relying on direct compute-node
internet access.

## Required Generated Files

Each result root should contain:

```text
configs/
  all_configs.txt
  experiment_index.csv
  summary.txt
  config_0000.json
  config_0001.json
  ...
  slurm/
    run_api.sbatch
    submit_all.sh
analysis/
```

Use four-digit config padding for all three games:

```text
config_0000.json
config_0001.json
...
config_0139.json    # Game 1 final config
config_0179.json    # Game 2 / Game 3 final config
```

The `experiment_index.csv` should include at least these fields:

### Game 1 Index Columns

```text
experiment_id,experiment_type,baseline_model,adversary_model,model_order,
competition_level,run_number,seed,discussion_turns,config_file,output_dir
```

### Game 2 Index Columns

```text
experiment_id,experiment_type,baseline_model,adversary_model,model_order,
rho,theta,competition_index,run_number,seed,discussion_turns,config_file,output_dir
```

### Game 3 Index Columns

```text
experiment_id,experiment_type,baseline_model,adversary_model,model_order,
alpha,sigma,competition_index,run_number,seed,discussion_turns,config_file,output_dir
```

## Execution Command Template

Each config should ultimately invoke `run_strong_models_experiment.py` with the
models already ordered according to `model_order`.

### Game 1 Runtime Template

```bash
python3 run_strong_models_experiment.py \
  --game-type item_allocation \
  --models "${MODEL_A}" "${MODEL_B}" \
  --model-order "${MODEL_ORDER}" \
  --num-items 5 \
  --competition-level "${COMPETITION_LEVEL}" \
  --max-rounds 10 \
  --gamma-discount 0.9 \
  --discussion-turns 2 \
  --random-seed 42 \
  --run-number "${RUN_NUMBER}" \
  --job-id "${EXPERIMENT_ID}" \
  --max-tokens-per-phase 10500 \
  --output-dir "${OUTPUT_DIR}"
```

### Game 2 Runtime Template

```bash
python3 run_strong_models_experiment.py \
  --game-type diplomacy \
  --models "${MODEL_A}" "${MODEL_B}" \
  --model-order "${MODEL_ORDER}" \
  --n-issues 5 \
  --rho "${RHO}" \
  --theta "${THETA}" \
  --max-rounds 10 \
  --gamma-discount 0.9 \
  --discussion-turns 2 \
  --random-seed "${SEED}" \
  --run-number 1 \
  --job-id "${EXPERIMENT_ID}" \
  --max-tokens-per-phase 10500 \
  --output-dir "${OUTPUT_DIR}"
```

### Game 3 Runtime Template

```bash
python3 run_strong_models_experiment.py \
  --game-type co_funding \
  --models "${MODEL_A}" "${MODEL_B}" \
  --model-order "${MODEL_ORDER}" \
  --m-projects 5 \
  --alpha "${ALPHA}" \
  --sigma "${SIGMA}" \
  --c-min 10.0 \
  --c-max 30.0 \
  --cofunding-discussion-transparency own \
  --cofunding-time-discount 0.9 \
  --max-rounds 10 \
  --gamma-discount 0.9 \
  --discussion-turns 2 \
  --random-seed "${SEED}" \
  --run-number 1 \
  --job-id "${EXPERIMENT_ID}" \
  --max-tokens-per-phase 10500 \
  --output-dir "${OUTPUT_DIR}"
```

Do not pass `--cofunding-disable-commit-vote`.
Do not pass `--cofunding-disable-time-discount`.

## Completion Criteria

The batch should not be analyzed until all required configs are complete or
explicitly marked as failed and excluded.

Required completion checks:

| Game | Expected `experiment_results.json` count | Expected config count |
|---|---:|---:|
| Game 1 | 140 | 140 |
| Game 2 | 180 | 180 |
| Game 3 | 180 | 180 |
| Total | 500 | 500 |

For each result root, produce a completion summary with:

```text
total_configs
completed_configs
started_only_configs
not_started_configs
failed_configs
models_fully_finished
models_partially_finished
```

For the paper appendix, prefer complete-model plots. If a model has missing
configs after retries, either backfill the missing configs or exclude that model
from complete-model plots and report the exclusion.

## Analysis Definitions

All plots must treat `llama-3.3-70b-instruct` as the baseline and the selected
10-model slate as adversaries, regardless of model order or relative Elo.

### Agent Mapping

Use the metadata in each config, not Elo, to identify the adversary:

```text
baseline_model = llama-3.3-70b-instruct
adversary_model = selected model from the 10-model slate
```

For utilities:

```text
adversary_utility = final_utilities[agent assigned to adversary_model]
baseline_utility = final_utilities[agent assigned to baseline_model]
utility_gap = adversary_utility - baseline_utility
utility_share = adversary_utility / (adversary_utility + baseline_utility)
```

If the denominator for `utility_share` is zero, record `utility_share` as null.

### Elo Variable

The x-axis Elo in all adversary plots is the adversary model's Arena Elo from:

```text
docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md
```

Do not use the baseline's Elo for the x-axis except in metadata captions.

### Competition Indices

Use the same derived indices as the canonical paper plots:

```text
Game 1: competition_level directly
Game 2: CI2 = theta * (1 - rho) / 2
Game 3: CI3 = (1 - alpha) * (1 - sigma)
```

## Exact Plots To Produce

All plot files should be written under each result root's `analysis/` directory.
Every plot should have a matching CSV file with the exact data used to draw the
figure.

### Game 1 Plots

Root:

```text
experiments/results/appendix_llama33_baseline_game1_202605/analysis/
```

Required files:

| File stem | Description |
|---|---|
| `average_utility_vs_elo` | Adversary mean utility vs adversary Elo, averaged over all Game 1 configs. |
| `average_utility_vs_elo_by_competition_level` | Adversary mean utility vs Elo, one curve per `competition_level`. |
| `average_utility_vs_competition_level_aggregated_over_models` | Mean adversary utility vs `competition_level`, aggregated over adversaries and orders. |
| `average_rounds_to_consensus_vs_elo` | Mean final round vs adversary Elo. Lower means faster convergence. |
| `consensus_rate_vs_elo` | Consensus rate vs adversary Elo. |
| `baseline_utility_vs_adversary_elo` | Baseline utility vs adversary Elo. This shows whether stronger adversaries reduce baseline payoff. |
| `utility_gap_vs_elo` | `adversary_utility - baseline_utility` vs adversary Elo. |
| `utility_share_vs_elo` | Adversary share of realized bilateral utility vs adversary Elo. |

Each file stem should produce:

```text
{stem}.csv
{stem}.png
```

### Game 2 Plots

Root:

```text
experiments/results/appendix_llama33_baseline_game2_202605/analysis/
```

Required files:

| File stem | Description |
|---|---|
| `utility_vs_elo_overall` | Adversary mean treaty utility vs adversary Elo, averaged over all `(rho, theta, order)` configs. |
| `utility_vs_elo_by_competition_index` | Adversary utility vs Elo, grouped by `CI2`. |
| `utility_vs_elo_by_rho_theta` | Native parameter view: panels fix `theta`, curves fix `rho`. |
| `average_utility_vs_competition_index` | Mean adversary utility vs `CI2`, aggregated over models and orders. |
| `rounds_to_consensus_vs_elo` | Mean final round vs adversary Elo. |
| `rounds_to_consensus_by_competition_index` | Mean final round vs `CI2`. |
| `consensus_rate_vs_elo` | Consensus rate vs adversary Elo. |
| `baseline_utility_vs_adversary_elo` | Baseline treaty utility vs adversary Elo. |
| `utility_gap_vs_elo` | `adversary_utility - baseline_utility` vs adversary Elo. |
| `utility_share_vs_elo` | Adversary share of realized bilateral utility vs adversary Elo. |

Each file stem should produce:

```text
{stem}.csv
{stem}.png
```

### Game 3 Plots

Root:

```text
experiments/results/appendix_llama33_baseline_game3_202605/analysis/
```

Required files:

| File stem | Description |
|---|---|
| `completion_summary` | Per-model completion table, including missing and failed config counts. CSV only is sufficient. |
| `utility_vs_elo_all_models` | Adversary mean co-funding utility vs adversary Elo across all completed configs. |
| `utility_vs_elo_complete_models_only` | Same as above, restricted to models with all 18 configs complete. |
| `utility_vs_elo_by_competition_index` | Adversary utility vs Elo, grouped by `CI3`. |
| `utility_vs_elo_by_alpha_sigma` | Native parameter view: panels fix `alpha`, curves fix `sigma`. |
| `average_utility_vs_competition_index` | Mean adversary utility vs `CI3`, aggregated over models and orders. |
| `rounds_to_consensus_vs_elo` | Mean final round vs adversary Elo. |
| `rounds_to_consensus_by_alpha_sigma` | Mean final round by native `(alpha, sigma)` cell. |
| `consensus_rate_vs_elo` | Consensus rate vs adversary Elo. |
| `consensus_rate_by_alpha_sigma` | Consensus rate by native `(alpha, sigma)` cell. |
| `negative_utility_rate_vs_elo` | Fraction of configs where adversary utility is below zero, by model. |
| `baseline_utility_vs_adversary_elo` | Baseline co-funding utility vs adversary Elo. |
| `utility_gap_vs_elo` | `adversary_utility - baseline_utility` vs adversary Elo. |
| `utility_share_vs_elo` | Adversary share of realized bilateral utility vs adversary Elo, null when total utility is zero. |
| `report` | Markdown analysis report summarizing completion, slope, correlations, and key cells. |

For plot-capable file stems, produce:

```text
{stem}.csv
{stem}.png
```

For `completion_summary`, produce:

```text
completion_summary.csv
```

For `report`, produce:

```text
report.md
```

### Cross-Game Combined Plots

After all three games are complete, produce a combined appendix figure directory:

```text
Figures/appendix_llama33_baseline/
```

Required combined outputs:

| File stem | Description |
|---|---|
| `adversary_utility_vs_elo_all_games` | Three-panel plot, one panel per game, adversary utility vs Elo with linear fit. |
| `baseline_utility_vs_adversary_elo_all_games` | Three-panel plot, baseline utility vs adversary Elo. |
| `utility_gap_vs_elo_all_games` | Three-panel plot, adversary-baseline utility gap vs adversary Elo. |
| `utility_share_vs_elo_all_games` | Three-panel plot, adversary utility share vs Elo. |
| `consensus_rate_vs_elo_all_games` | Three-panel plot, consensus rate vs Elo. |
| `competition_effects_all_games` | Mean adversary utility vs Game 1 competition level, Game 2 `CI2`, and Game 3 `CI3`. |
| `summary_correlations` | CSV table of Pearson correlation, Spearman correlation, slope per 100 Elo, intercept, model count, and config count for each game and metric. |

If the fairness benchmark code is available and compatible with the new
baseline, also produce:

| File stem | Description |
|---|---|
| `exploitation_vs_elo_combined` | Adversary benchmark-relative exploitation vs adversary Elo for all games. |
| `exploitation_vs_elo_combined_baseline` | Baseline benchmark-relative exploitation vs adversary Elo for all games. |

Use the same definitions as the main paper:

```text
Game 1 and Game 2: exploitation relative to NBS
Game 3: exploitation relative to Lindahl benchmark
```

## Statistical Summaries To Report

For each game, report at least:

```text
model_count
config_count
completed_config_count
completion_rate
consensus_rate
mean_adversary_utility
mean_baseline_utility
mean_utility_gap
Pearson r: adversary Elo vs adversary utility
Spearman rho: adversary Elo vs adversary utility
linear slope: adversary utility per 100 Elo
Pearson r: adversary Elo vs baseline utility
Pearson r: adversary Elo vs utility gap
```

For Game 1, additionally report:

```text
mean adversary utility by competition_level
mean final round by competition_level
hardest cell results at competition_level = 1.0
```

For Game 2, additionally report:

```text
mean adversary utility by CI2
mean adversary utility by rho
mean adversary utility by theta
hardest cell results at CI2 = 1.0, equivalent to rho=-1.0 and theta=1.0
```

For Game 3, additionally report:

```text
mean adversary utility by CI3
mean adversary utility by alpha
mean adversary utility by sigma
consensus rate by alpha and sigma
negative utility rate by alpha and sigma
hardest cell results at alpha=0.0 and sigma=0.2
```

## Interpretation Plan

The appendix writeup should answer four questions.

First, does adversary Elo still positively predict adversary utility against the
new baseline?

Second, does the fixed baseline's own utility decrease as adversary Elo rises?
This checks whether higher-Elo adversaries are reallocating surplus toward
themselves, not merely creating more total surplus.

Third, do the three games preserve the same qualitative ordering as the main
paper?

Expected qualitative ordering:

```text
Game 1: clear capability trend and faster convergence for stronger adversaries
Game 2: high overall utilities and strong convergence
Game 3: weaker/noisier trend because scarcity creates coordination failures
```

Fourth, does changing the baseline alter any paper-level conclusions? The most
important outcome is not whether the exact slopes match the `gpt-5-nano`
baseline runs. The important outcome is whether the sign and relative pattern
remain stable.

## Expected Paper Appendix Text Shape

The appendix section can be structured as:

```text
Appendix X: Baseline-Robustness Sweep

We repeat the canonical N=2 model-scaling experiments using
llama-3.3-70b-instruct as the fixed baseline instead of gpt-5-nano. The
adversary slate contains 10 high-context models spanning Elo 1240-1504. We use
the same three games and canonical parameter grids, with two discussion turns in
all games.

The qualitative result is [fill after analysis]. In Game 1, [fill]. In Game 2,
[fill]. In Game 3, [fill]. The fixed-baseline utility plots show [fill]. These
results suggest that the main paper's payoff-vs-Elo trend is [baseline-specific
or baseline-robust].
```

## Non-Goals

This batch does not attempt to:

- rerun the full 30-model canonical roster
- estimate provider-specific effects
- run multi-agent `N>2` settings
- run multiple seeds per condition
- compare all possible baselines
- tune prompts or game mechanics
- introduce new model routes not already supported by the repo

## Final Checklist Before Launch

Before submitting jobs:

1. Confirm all 11 models are present in `STRONG_MODELS_CONFIG`:
   - 1 baseline
   - 10 adversaries
2. Confirm provider credentials or proxy access:
   - `OPENROUTER_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `GOOGLE_API_KEY`
3. Generate exactly:
   - 140 Game 1 configs
   - 180 Game 2 configs
   - 180 Game 3 configs
4. Validate `experiment_index.csv` row counts.
5. Run one smoke config per game before submitting arrays:
   - Game 1: `config_0000.json`
   - Game 2: `config_0000.json`
   - Game 3: `config_0000.json`
6. Submit three arrays with conservative concurrency.
7. Monitor provider failures and malformed proposal logs.
8. Backfill missing configs before producing final plots.

## Planned Experiment Count

```text
Game 1: 10 models x 7 competition levels x 2 orders = 140
Game 2: 10 models x 3 rho x 3 theta x 2 orders = 180
Game 3: 10 models x 3 alpha x 3 sigma x 2 orders = 180

Total = 500 configs
```
