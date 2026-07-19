# Game Architecture

## Scope

The repository has three negotiation games. All games use the same experiment controller and provider agents.

Game-specific code supplies state, prompts, action parsing, validation, and utility calculations.

## Main Components

`run_strong_models_experiment.py` is the command-line entry point. It reads arguments and creates an experiment configuration.

`strong_models_experiment/experiment.py` creates the game environment and the model agents. It also writes the final result files.

`strong_models_experiment/phases/phase_handlers.py` runs the shared model-call phases. It records prompts, responses, token usage, and parse diagnostics.

`game_environments/base.py` defines `GameEnvironment` and the game configuration classes.

Each game module implements the `GameEnvironment` interface:

- `item_allocation.py` implements Game 1.
- `diplomatic_treaty.py` implements Game 2.
- `co_funding.py` implements Game 3.

## Shared Round Flow

The controller prepares the game state before round 1. It gives each agent its private preferences.

Each standard round can contain these phases:

1. Agents discuss the game in the selected model order.
2. Each agent does private strategy work.
3. Agents submit game actions.
4. The game checks the actions and agreement rule.
5. Agents receive feedback or do a reflection phase.

Command-line flags can disable discussion, private thinking, or reflection.

The phase handler asks a model to repair malformed JSON. If repair fails, it records the failure and uses an audited fallback action.

## Game 1: Item Allocation

Agents divide a set of indivisible items. Each agent has a private value for each item.

`competition_level` controls the target cosine relation between agent value vectors. The code records the realized pairwise cosine values.

Agents discuss, submit allocations, and vote. The game accepts a proposal when it gets the necessary vote threshold.

The utility is the sum of the agent values for its allocated items. The time discount can reduce utility in later rounds.

## Game 2: Diplomatic Treaty

Agents negotiate continuous policy positions across a set of issues. Each issue position is in the interval from `0` to `1`.

`rho` controls the relation between preferred policy positions. Its permitted interval is `[-1, 1]`.

`theta` controls the overlap between issue weights. Its permitted interval is `[0, 1]`.

The game uses a Gaussian copula to make correlated positions. It uses constrained optimization to make issue weights.

Agents discuss, submit treaty vectors, and vote. Utility decreases with weighted distance from the private preferred positions.

For more than two agents, some negative `rho` values are not mathematically possible. The configuration class checks this condition.

## Game 3: Co-Funding

Agents allocate private budgets to public projects. A project gives value only when total contributions meet its cost.

`alpha` controls the relation between project values. Its permitted interval is `[0, 1]`.

`sigma` controls total budget abundance. Its permitted interval is `(0, 1]`.

`c_min` and `c_max` control the project-cost interval. Read `guides/cofunding_parameter_calibration.md` for the calibration rationale.

The current mode uses individual contribution plans. Discussion can show aggregate, own, or full contribution information.

The game can require a final commit vote. It can also apply the time discount.

Utility equals funded-project value minus the agent contribution cost. An unfunded project gives no project value.

## Prompts And JSON

The game modules own the current game prompts. Generate the readable prompt reference with this command:

```bash
python scripts/generate_all_prompts_reference.py
```

The command writes `docs/reference/all_prompts.md`.

The base environment supplies common JSON requirements and repair helpers. Game modules supply the action schemas and validation rules.

## Add A Game

Use these steps to add a game:

1. Add a `GameType` value and a configuration data class in `game_environments/base.py`.
2. Implement the `GameEnvironment` interface in a new game module.
3. Export the new environment from `game_environments/__init__.py`.
4. Add game construction to `strong_models_experiment/experiment.py`.
5. Add command-line arguments to `run_strong_models_experiment.py`.
6. Add unit tests for state generation, action parsing, validation, agreement, and utility.
7. Regenerate `docs/reference/all_prompts.md`.
