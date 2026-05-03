# Test-Time Compute Scaling Proposal

Date: 2026-05-03

Status: proposal only. No experiments have been run from this document.

## Summary

This proposal defines a new test-time compute scaling experiment for the standard
N=2 negotiation setting. The previous scaling experiments varied model capability
across different models and plotted model payoff against external capability
proxies such as Elo. This experiment instead holds the base model identity fixed
within each provider and varies the amount of reasoning compute requested at
inference time.

The recommended main experiment uses one frontier-1 model from each major closed
provider:

| Provider | Selected model | Reason for selection | Reasoning levels to test |
|---|---|---|---|
| OpenAI | `gpt-5` | One tier below newer GPT-5.x frontier models; explicitly requested as the OpenAI target | `minimal`, `low`, `medium`, `high` |
| Anthropic | `claude-sonnet-4-6` | One tier below Opus frontier tier; supports provider-native effort scaling | `low`, `medium`, `high`, `max` |
| Google | `gemini-3-flash-preview` | One tier below Gemini Pro frontier tier; supports the full Gemini 3 Flash thinking-level ladder | `minimal`, `low`, `medium`, `high` |

This gives 12 model-effort conditions. The recommended main run uses 9 game cells,
2 speaking orders, and 1 seed:

```text
12 model-effort conditions x 9 game cells x 2 orders x 1 seed = 216 experiments
```

The 9 game cells are:

| Cell | Game | Parameters | Label |
|---:|---|---|---|
| 1 | Game 1: item allocation | `competition_level = 0.0` | cooperative |
| 2 | Game 1: item allocation | `competition_level = 0.5` | mixed |
| 3 | Game 1: item allocation | `competition_level = 1.0` | competitive |
| 4 | Game 2: diplomacy | `rho = 1`, `theta = 1` | cooperative |
| 5 | Game 2: diplomacy | `rho = 0`, `theta = 1` | mixed |
| 6 | Game 2: diplomacy | `rho = -1`, `theta = 1` | competitive |
| 7 | Game 3: co-funding | `alpha = 1.0`, `sigma = 1.0` | easy |
| 8 | Game 3: co-funding | `alpha = 0.5`, `sigma = 0.6` | mixed |
| 9 | Game 3: co-funding | `alpha = 0.0`, `sigma = 0.2` | hard |

The primary x-axis should not be Elo. Use observed test-time compute:

1. `target_reasoning_tokens` or `target_thinking_tokens`, when available.
2. estimated target-model cost per episode.
3. provider-native effort level as a secondary categorical x-axis.

## Motivation

The earlier model-scaling experiments ask:

> As the adversary model becomes more capable, does its utility/payoff improve?

This experiment asks a different question:

> For the same base model, does buying more inference-time reasoning improve
> negotiation performance?

The distinction matters. Elo is a property of a model snapshot on an external
benchmark distribution. Reasoning effort is a runtime control that changes
latency, token usage, and sometimes strategic quality without changing the base
model identity. Therefore, reasoning levels should not be converted into made-up
Elo estimates. The right scaling variable is observed test-time compute.

This design is intended to detect:

- monotonic gains from additional reasoning;
- saturation, where more reasoning stops helping;
- overthinking, where higher effort reduces payoff or agreement rate;
- provider differences in how effort labels map to actual hidden-token usage;
- game-specific effects, especially whether hard/competitive cells benefit more
  from extra reasoning than easy/cooperative cells.

## Core Research Questions

1. Does higher test-time reasoning effort increase target-model utility?
2. Does higher effort increase total social welfare, or mainly redistribute payoff
   toward the target model?
3. Is the effect monotonic, saturating, or non-monotonic?
4. Do returns to extra reasoning differ across Game 1, Game 2, and Game 3?
5. Are provider labels such as `high` comparable in actual token usage, latency,
   and payoff impact?
6. Does extra reasoning improve agreement/consensus rates, or does it make agents
   more stubborn?

## Should the Main Run Include `none` / Off Reasoning?

No. The main balanced experiment should not include a `none` or fully off
reasoning condition, because the selected one-tier-down models do not all expose
the same true-off control.

Use each provider's lowest supported native level instead:

| Provider | Selected model | Lowest main level | Why not `none` / off? |
|---|---|---|---|
| OpenAI | `gpt-5` | `minimal` | The GPT-5 model page lists `minimal`, `low`, `medium`, and `high`, not `none`. |
| Anthropic | `claude-sonnet-4-6` | `low` | Sonnet 4.6 exposes effort levels `low`, `medium`, `high`, and `max`; `low` is the lowest-effort setting. |
| Google | `gemini-3-flash-preview` | `minimal` | Gemini 3 Flash supports `minimal`, but Google docs state that Gemini 3 Flash does not support full thinking-off. |

This means the main experiment measures:

```text
lowest available native reasoning level -> highest available native reasoning level
```

not:

```text
no reasoning -> reasoning
```

That is the cleaner design for a cross-provider TTC curve. A true-off comparison
can be run later as a provider-specific appendix, but it should not be mixed into
the main 12-condition grid unless all three selected models support an equivalent
off state.

## Model Selection

### Selection Rule

Use exactly one model per provider. The model should be one tier weaker than the
provider's current frontier model, but still support native reasoning controls.
This avoids two problems:

- Frontier models may saturate on the task, making reasoning effects hard to see.
- Very weak models may fail for reasons unrelated to reasoning depth.

### OpenAI: GPT-5

Use `gpt-5`, preferably the pinned dated model already present in the repo:

```text
gpt-5-2025-08-07
```

Reasoning levels:

```text
minimal
low
medium
high
```

OpenAI's current GPT-5 model page describes GPT-5 as a previous reasoning model
with configurable reasoning effort and lists `minimal`, `low`, `medium`, and
`high` for `reasoning.effort`.

Do not include `xhigh` for this model unless a smoke test proves that the API
accepts it for `gpt-5`. Do not include `none` for the main GPT-5 experiment,
because the GPT-5 model page does not list it for this model.

Proposed config labels:

```text
gpt-5-effort-minimal
gpt-5-effort-low
gpt-5-effort-medium
gpt-5-effort-high
```

### Anthropic: Claude Sonnet 4.6

Use:

```text
claude-sonnet-4-6
```

Reasoning/effort levels:

```text
low
medium
high
max
```

Anthropic's effort docs state that Claude Sonnet 4.6 supports the effort
parameter and recommend adaptive thinking for Sonnet 4.6. The docs also state
that Sonnet 4.6 defaults to `high`, so the experiment should set every level
explicitly and should not treat "default" as a separate condition.

Proposed config labels:

```text
claude-sonnet-4-6-effort-low
claude-sonnet-4-6-effort-medium
claude-sonnet-4-6-effort-high
claude-sonnet-4-6-effort-max
```

Recommended request shape:

```json
{
  "thinking": {"type": "adaptive"},
  "output_config": {"effort": "medium"}
}
```

The current repo has a similar pattern for Claude Opus 4.6 using
`thinking: {"type": "adaptive"}` and an `output_config.effort` control. Before
large-scale execution, run a one-call smoke test for each Sonnet 4.6 effort level
through the exact code path used by Slurm.

### Google: Gemini 3 Flash

Use:

```text
gemini-3-flash-preview
```

or the current non-preview Gemini 3 Flash alias if Google/OpenRouter exposes one
at execution time.

Reasoning levels:

```text
minimal
low
medium
high
```

Google's Gemini thinking docs list these four `thinkingLevel` values for Gemini
3 Flash. The same docs state that Gemini 3 Pro does not support the full ladder:
Gemini 3 Pro supports fewer levels and cannot use `minimal`. This is why Gemini 3
Flash is the better one-tier-down TTC target.

Proposed config labels:

```text
gemini-3-flash-thinking-minimal
gemini-3-flash-thinking-low
gemini-3-flash-thinking-medium
gemini-3-flash-thinking-high
```

Recommended direct Google request shape:

```json
{
  "generationConfig": {
    "thinkingConfig": {
      "thinkingLevel": "low"
    }
  }
}
```

The current repo already contains a `gemini-3-flash` config routed through
OpenRouter, but the direct `GoogleAgent` path does not currently pass
`thinkingConfig`. For a clean TTC experiment, either:

1. extend the direct Google path to pass native `thinkingConfig`, or
2. verify that OpenRouter forwards Gemini reasoning controls correctly and record
   the exact OpenRouter request payload in the run metadata.

The direct Google API path is cleaner scientifically because it avoids ambiguity
about whether a router translated the thinking control.

## Fixed Baseline / Negotiation Partner

Use one fixed baseline partner for every condition. The recommended default is
the existing standard baseline:

```text
gpt-5-nano
```

Pin the baseline reasoning behavior. Do not let it float with provider defaults.
In the current repo, GPT-5-family OpenAI calls without explicit reasoning effort
are forced to `low` in `negotiation/llm_agents.py`. For this experiment, make the
baseline explicit in config and metadata:

```text
baseline_model = gpt-5-nano
baseline_reasoning_effort = low
```

The key rule is that only the target model's reasoning level changes. The
baseline must be held fixed across all target models, games, orders, and seeds.

## Games and Conditions

Run the standard N=2 versions of the three existing games.

### Main Reduced Grid

The recommended main grid uses 3 representative cells per game. This is the
right first full run because it covers easy, medium, and hard environments while
keeping cost manageable.

#### Game 1: Item Allocation

Use three competition levels:

| Cell label | `competition_level` | Interpretation |
|---|---:|---|
| cooperative | 0.0 | aligned preferences |
| mixed | 0.5 | partial conflict |
| competitive | 1.0 | maximally opposed preferences |

#### Game 2: Diplomacy

Use three cells spanning the competition index induced by `rho` and `theta`:

| Cell label | `rho` | `theta` | Interpretation |
|---|---:|---:|---|
| cooperative | 1 | 1 | highly correlated preferences |
| mixed | 0 | 1 | intermediate conflict |
| competitive | -1 | 1 | anti-correlated preferences with high overlap |

The earlier diplomacy analyses suggest `rho` is the dominant driver, so this
reduced grid varies `rho` while keeping high issue overlap.

#### Game 3: Co-Funding

Use three cells from the standard `alpha` x `sigma` grid:

| Cell label | `alpha` | `sigma` | Interpretation |
|---|---:|---:|---|
| easy | 1.0 | 1.0 | aligned preferences, ample budget |
| mixed | 0.5 | 0.6 | moderate alignment, moderate scarcity |
| hard | 0.0 | 0.2 | low alignment, severe scarcity |

### Orders and Seeds

Use both speaking orders:

```text
target_first
baseline_first
```

Use one seed for the main run:

```text
42
```

If budget permits, extend to two or five seeds after inspecting the first seed:

```text
42
123
```

or:

```text
42
123
456
789
101112
456
789
101112
```

## Experiment Count

### Main Recommended Run

Model-effort conditions:

```text
OpenAI GPT-5:              4 levels
Claude Sonnet 4.6:         4 levels
Gemini 3 Flash:            4 levels
Total:                    12 conditions
```

Game cells:

```text
Game 1: 3 cells
Game 2: 3 cells
Game 3: 3 cells
Total:  9 cells
```

Main run:

```text
12 conditions x 9 game cells x 2 orders x 1 seed = 216 experiments
```

### Optional Two-Seed Robustness Extension

If the 216-experiment main run shows a signal and failure rates are low, add one
more seed:

```text
12 conditions x 9 game cells x 2 orders x 2 seeds = 432 experiments
```

This should be treated as a robustness extension, not the default run.

### Full Canonical Extension

If the main reduced grid gives a real signal, run the full standard grid:

```text
Game 1: 5 competition levels = 0.0, 0.25, 0.5, 0.75, 1.0
Game 2: 9 cells = rho {-1, 0, 1} x theta {0, 0.5, 1}
Game 3: 9 cells = alpha {0, 0.5, 1} x sigma {0.2, 0.6, 1}
Total: 23 game cells
```

With two seeds:

```text
12 conditions x 23 game cells x 2 orders x 2 seeds = 1104 experiments
```

With five seeds:

```text
12 conditions x 23 game cells x 2 orders x 5 seeds = 2760 experiments
```

## Implementation Notes for This Repo

The existing `configs/test_time_compute_scaling.yaml` is not the right final
design for this experiment. It varies many models and numeric token budgets, and
for OpenAI it collapses budgets into `low`, `medium`, and `high`. This proposal
instead uses exactly one model per provider and every native reasoning level
available for that selected model.

Recommended implementation changes before running:

1. Add `gpt-5-effort-minimal` to `strong_models_experiment/configs.py`.
2. Update `OPENAI_REASONING_EFFORTS` in
   `strong_models_experiment/agents/agent_factory.py` to include `minimal`.
3. Add `claude-sonnet-4-6` effort configs for `low`, `medium`, `high`, and `max`.
4. Verify that the Anthropic request path sends `thinking: {"type": "adaptive"}`
   and the effort control in the API-compatible shape.
5. Add Gemini 3 Flash thinking-level configs for `minimal`, `low`, `medium`, and
   `high`.
6. Decide whether Gemini will run through direct Google API or OpenRouter.
7. If using direct Google API, extend `GoogleAgent` to pass `thinkingConfig`.
8. If using OpenRouter for Gemini, verify the exact reasoning-control payload and
   record it in each config.
9. Save the requested reasoning level and observed reasoning/thinking token usage
   into result metadata.
10. Avoid prompt-only "think for N tokens" instructions in the main experiment.
    This experiment should use provider-native TTC controls.

## Smoke Tests

Before generating the 216-experiment main grid, run one minimal negotiation or
one direct API call per model-effort condition.

Each smoke test must verify:

- the provider accepts the model name;
- the provider accepts the requested reasoning level;
- the response is parseable by the negotiation code;
- token metadata is present;
- reasoning/thinking tokens are captured if the provider exposes them;
- `max_tokens` is large enough for high/max effort conditions;
- the baseline model's effort setting is fixed and recorded.

Expected smoke-test matrix:

```text
4 GPT-5 calls
4 Claude Sonnet 4.6 calls
4 Gemini 3 Flash calls
Total: 12 smoke tests
```

## Data to Record

Each completed episode should produce one row with at least:

```text
experiment_id
game_type
game_cell_label
competition_level
rho
theta
alpha
sigma
target_provider
target_model
target_reasoning_level_requested
target_reasoning_level_index
baseline_model
baseline_reasoning_level_requested
order
seed
target_utility
baseline_utility
utility_gap = target_utility - baseline_utility
utility_share = target_utility / (target_utility + baseline_utility)
joint_surplus
agreement_or_consensus
rounds_to_agreement
failure_reason
target_input_tokens
target_output_tokens
target_reasoning_tokens
target_thinking_tokens
target_total_tokens
baseline_input_tokens
baseline_output_tokens
baseline_reasoning_tokens
baseline_total_tokens
estimated_target_cost_usd
estimated_episode_cost_usd
wall_clock_seconds
```

For providers that do not expose hidden reasoning tokens, keep the field present
and set it to null rather than zero. Zero should mean the provider reported zero
reasoning tokens.

## Primary Outcome Variables

Use the target model as the unit of interest.

Primary payoff outcomes:

- `target_utility`
- `utility_gap = target_utility - baseline_utility`
- `utility_share`

Efficiency outcomes:

- `joint_surplus`
- normalized social welfare
- agreement/consensus rate

Behavioral outcomes:

- rounds to agreement;
- no-agreement rate;
- proposal acceptance rate;
- stubbornness or late-round concession behavior, if already computed.

Compute outcomes:

- observed reasoning/thinking tokens;
- total output tokens;
- estimated dollars per completed episode;
- wall-clock latency.

## Plotting Plan

### Plot 1: TTC Curve by Provider

Question:

> Does payoff increase as requested reasoning level increases?

Plot:

- x-axis: provider-native reasoning level.
- y-axis: mean target utility.
- one panel per provider/model.
- separate rows or colors for game.
- error bars: bootstrap confidence intervals or seed-level standard errors.

Use ordinal order:

```text
GPT-5: minimal -> low -> medium -> high
Claude Sonnet 4.6: low -> medium -> high -> max
Gemini 3 Flash: minimal -> low -> medium -> high
```

Do not compare `high` across providers as if it were the same amount of compute.
It is only an ordered label within each provider.

### Plot 2: Payoff vs Observed Reasoning Tokens

Question:

> Does actual hidden-token spend predict payoff better than nominal effort labels?

Plot:

- x-axis: observed `target_reasoning_tokens` or `target_thinking_tokens`.
- y-axis: target utility or utility gap.
- points: individual episodes.
- lines: provider-specific smoother or binned mean.
- facets: game type.

This is the main cross-provider TTC plot.

If a provider does not expose reasoning tokens, use total output tokens or cost in
a separate panel and mark that provider's reasoning-token field as missing.

### Plot 3: Cost-Performance Frontier

Question:

> Which reasoning level gives the best payoff per dollar?

Plot:

- x-axis: estimated target-model cost per episode.
- y-axis: utility gap or target utility.
- one line per provider.
- annotate each point with the requested reasoning level.

This plot is important because test-time compute scaling is a cost-performance
question, not just a capability question.

### Plot 4: Delta from Lowest Effort

Question:

> How much does extra reasoning help relative to the cheapest setting?

For each provider and game cell, compute:

```text
delta_utility(level) = mean_utility(level) - mean_utility(lowest_available_level)
```

Lowest available levels:

```text
GPT-5: minimal
Claude Sonnet 4.6: low
Gemini 3 Flash: minimal
```

Plot:

- x-axis: reasoning level index.
- y-axis: delta utility.
- one panel per game.
- one line per provider.

This makes provider curves visually comparable without pretending that effort
labels are numerically equivalent.

### Plot 5: Agreement and Rounds

Question:

> Does extra reasoning improve negotiation success, or does it make agents slower
> and more stubborn?

Make two panels:

- agreement/consensus rate vs reasoning level;
- mean rounds to agreement vs reasoning level, conditional on agreement.

Facet by game. Game 3 should get special attention because scarcity can create
coordination failures.

### Plot 6: Token Sanity Plot

Question:

> Did requested reasoning levels actually produce increasing compute usage?

Plot:

- x-axis: requested reasoning level.
- y-axis: observed reasoning/thinking tokens.
- one panel per provider.
- points: individual episodes.
- line: mean or median.

This plot is mandatory. If effort levels do not produce monotone token usage,
the interpretation of payoff curves must be more careful.

### Plot 7: Game-Parameter Heatmaps

For the full canonical extension, create heatmaps:

- Game 1: competition level x reasoning level.
- Game 2: `rho` x `theta`, one heatmap per reasoning level or heatmap of
  high-minus-low delta.
- Game 3: `alpha` x `sigma`, one heatmap per reasoning level or heatmap of
  high-minus-low delta.

The most useful version is usually a delta heatmap:

```text
mean_utility(high_or_max) - mean_utility(lowest_effort)
```

This shows where test-time compute matters most.

## Statistical Analysis

Use descriptive plots first. Then fit a regression or mixed-effects model for a
compact quantitative summary.

Suggested regression:

```text
target_utility
  ~ log1p(target_reasoning_tokens)
  + provider_model
  + game_type
  + game_cell
  + order
  + log1p(target_reasoning_tokens):provider_model
  + log1p(target_reasoning_tokens):game_type
```

If reasoning-token fields are missing for some provider, fit a parallel model
using:

```text
log1p(estimated_target_cost_usd)
```

Report:

- slope of payoff versus reasoning tokens;
- slope of payoff versus cost;
- per-provider marginal returns;
- per-game marginal returns;
- whether high/max effort is significantly better than lowest effort.

Do not pool provider-native labels directly in a single numeric regression. A
Claude `max` point and a Gemini `high` point are not calibrated to the same token
budget.

## Interpretation Rules

Use these rules when writing up the results:

1. If utility rises with effort and reasoning tokens rise with effort, that is
   evidence for positive TTC scaling.
2. If utility is flat while tokens rise, that is evidence for saturation.
3. If utility falls at high/max effort, inspect rounds, failed agreements, and
   output verbosity for overthinking.
4. If provider-native effort labels do not produce monotone token usage, present
   the token-sanity plot before the payoff plot.
5. If extra reasoning improves joint surplus and target utility, call it
   efficiency improvement.
6. If extra reasoning improves target utility but reduces baseline utility or
   joint surplus, call it redistributive advantage or exploitation, depending on
   the game-specific interpretation.

## Execution Checklist

1. Verify current provider docs and account access for all three selected models.
2. Add or update the 12 model-effort configs.
3. Run the 12 one-call smoke tests.
4. Confirm token metadata extraction for OpenAI, Anthropic, and Gemini.
5. Generate the 216 one-seed main-run configs.
6. Run the main grid.
7. Check failure rate, parse failures, token usage, cost, and latency.
8. If failure rate is acceptable and the signal is promising, optionally add a
   second seed for the 432-experiment robustness extension.
9. Aggregate results into a single CSV with one row per episode.
10. Generate the seven plot families above.
11. Decide whether the full canonical 1104-experiment extension is warranted.

## Source Documentation

Provider controls change over time. Re-check these links immediately before
launching the experiment:

- OpenAI GPT-5 model page:
  <https://developers.openai.com/api/docs/models/gpt-5>
- OpenAI Responses/API reference:
  <https://platform.openai.com/docs/api-reference/responses>
- Anthropic effort parameter:
  <https://platform.claude.com/docs/en/build-with-claude/effort>
- Anthropic extended/adaptive thinking:
  <https://platform.claude.com/docs/en/build-with-claude/extended-thinking>
- Gemini thinking controls:
  <https://ai.google.dev/gemini-api/docs/thinking>

## Repo References

Relevant local files:

- `configs/test_time_compute_scaling.yaml`: older TTC config, useful background
  but not the final design for this proposal.
- `strong_models_experiment/configs.py`: add the selected model-effort configs
  here.
- `strong_models_experiment/agents/agent_factory.py`: provider-specific config
  routing and allowed OpenAI effort values.
- `negotiation/llm_agents.py`: provider call implementation and token metadata
  extraction.
- `strong_models_experiment/experiment.py`: episode-level result and token usage
  recording.
