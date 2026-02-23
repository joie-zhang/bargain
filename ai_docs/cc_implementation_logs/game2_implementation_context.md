# Game 2 (Diplomatic Treaty) Implementation Context

## Date: 2026-02-21

## What Was Done This Session

### Changes Made (all passing 100/100 tests)

1. **Position Generation — Gaussian Copula** (`game_environments/diplomatic_treaty.py`)
   - Replaced N(0.5, σ²) + clip with Gaussian copula
   - `rho_z = 2*sin(π*rho/6)`, sample z ~ N(0, Σ_z), then `p_ik = Φ(z_ik)`
   - No clipping — CDF guarantees [0,1] with Uniform marginals

2. **Weight Generation — SLSQP Optimization** (`game_environments/diplomatic_treaty.py`)
   - Replaced Dirichlet mixing heuristic with SLSQP optimization
   - For N=2: joint optimization of both vectors (avoids geometric infeasibility)
   - For N>2: fix w0, optimize each subsequent agent
   - Exact cosine similarity = theta (within 0.01 tolerance for all theta in [0.1, 0.95])

3. **Removed Lambda Parameter** (7 files)
   - `base.py`, `diplomatic_treaty.py`, `__init__.py`, `experiment.py`, `run_strong_models_experiment.py`
   - Removed `_generate_issue_types()`, `issue_types`, win-win/zero-sum labels
   - Tex spec only has 2 params (rho, theta)

4. **PSD Feasibility Validation** (`base.py`)
   - For N>2, validates `rho_z >= -1/(N-1)` in `__post_init__()`

5. **Evaluation Metrics** (`game_environments/metrics.py` — NEW)
   - `compute_utility`, `social_welfare`, `optimal_social_welfare`, `utilitarian_efficiency`
   - `nash_bargaining_solution` (L-BFGS-B, 20 restarts, log-transform)
   - `distance_from_nbs`, `exploitation_index`, `is_pareto_efficient`
   - `kalai_smorodinsky_fairness`, `efficiency_fairness_decomposition`

6. **Tests** (100/100 passing)
   - `tests/test_diplomatic_treaty.py` — 51 tests (KS uniformity, correlation monotonicity, 12 parametrized cosine, PSD, etc.)
   - `tests/test_metrics.py` — 21 tests
   - `tests/test_game_environments.py` — updated (removed lam)

### Key Files Modified
- `game_environments/diplomatic_treaty.py` — core game logic (positions, weights, prompts)
- `game_environments/base.py` — config (removed lam, added PSD check, added CoFundingConfig)
- `game_environments/__init__.py` — exports (removed lam kwargs, added metrics)
- `game_environments/metrics.py` — NEW evaluation metrics
- `strong_models_experiment/experiment.py` — removed lam from experiment runner
- `run_strong_models_experiment.py` — removed --lam CLI arg

### Key Files NOT Modified (still reference lam)
- `legacy/diplomacy_environment/diplo_implementation.py` — old implementation, intentionally unchanged

## How Game 2 Experiments Run

### Entry Point
`run_strong_models_experiment.py` with `--game-type diplomacy`

### Example Command
```bash
python run_strong_models_experiment.py \
    --models gpt-4o claude-3-5-sonnet \
    --game-type diplomacy \
    --n-issues 5 \
    --rho 0.0 \
    --theta 0.5 \
    --max-rounds 10 \
    --batch --num-runs 5 \
    --random-seed 42
```

### Flow
1. `run_strong_models_experiment.py` parses args, creates experiment_config
2. `StrongModelsExperiment.run_single_experiment()` creates `GameEnvironment` via factory
3. `create_game_environment("diplomacy", ...)` → `DiplomaticTreatyGame`
4. `game.create_game_state(agents)` generates positions (Gaussian copula) + weights (SLSQP)
5. 14-phase negotiation loop (discussion, thinking, proposal, voting, reflection)
6. Results saved to `experiments/results/`

### Parameters
- `--rho`: Preference correlation [-1, 1] (position alignment)
- `--theta`: Interest overlap [0, 1] (weight cosine similarity)
- `--n-issues`: Number of negotiation issues (default: 5)
- `--gamma-discount`: Time discount per round (default: 0.9)

### Post-Experiment Analysis
Use metrics from `game_environments.metrics`:
```python
from game_environments.metrics import (
    compute_utility, social_welfare, optimal_social_welfare,
    utilitarian_efficiency, nash_bargaining_solution,
    exploitation_index, kalai_smorodinsky_fairness
)
```

## Architecture Notes
- `game_environments/` is the game logic layer
- `strong_models_experiment/` is the experiment runner (handles API calls, phases, logging)
- `run_strong_models_experiment.py` is the CLI entry point
- The experiment runner is game-agnostic — it delegates all game-specific logic to the GameEnvironment
- The 14-phase structure: setup → preferences → (discussion → thinking → proposal → enumeration → voting → tabulation → reflection) × rounds

## What's New in base.py (externally modified)
- `GameType` enum now has `CO_FUNDING = "co_funding"`
- New `CoFundingConfig` dataclass with alpha, sigma, c_min, c_max, m_projects
- New `get_protocol_type()` method on GameEnvironment returning "propose_and_vote" (default) or "talk_pledge_revise"
