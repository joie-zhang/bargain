# Game 3 (Co-Funding / Participatory Budgeting) Implementation Context

## Date: 2026-02-22

## What Was Done This Session

### Changes Made

1. **Co-Funding Game Environment** (`game_environments/co_funding.py` — 891 lines)
   - Talk-Pledge-Revise protocol (no voting phase — fundamentally different from Games 1/2)
   - Valuation generation via SLSQP optimization for target cosine similarity `alpha`
     - 2-agent case: direct joint optimization
     - N>2 case: generate against reference vector (agent 0)
   - Valuations sum to 100 per agent, non-negative
   - Utility: `U_i = sum(v_ij - x_ij for j in funded_set)` — quasi-linear, no discount factor
   - Early termination when pledges are identical for 2 consecutive rounds
   - Aggregate-only feedback: agents see total contributions per project, NOT individual contributions
   - Key methods: `create_game_state()`, `update_game_state_with_pledges()`, `compute_final_outcome()`, `parse_proposal()`, `validate_proposal()`, phase-specific prompt generators

2. **Co-Funding Metrics** (`game_environments/cofunding_metrics.py` — 513 lines, NEW)
   - `optimal_funded_set`: Knapsack solver (brute force M≤20, DP for larger)
   - `social_welfare`: `SW = sum(v_ij - x_ij)` for funded projects
   - `utilitarian_efficiency`: actual_SW / optimal_SW
   - `provision_rate`: fraction of optimal projects actually funded
   - `coordination_failure_rate`: fraction of surplus-positive projects NOT funded
   - `lindahl_equilibrium`: fair cost-sharing `x_ij^L = c_j * v_ij / sum_k(v_kj)`
   - `lindahl_distance`: Frobenius norm between actual and Lindahl contributions
   - `free_rider_index`: `F_ij = (v_ij/sum_k v_kj) / (x_ij/sum_k x_kj)` (>1 = free-riding)
   - `exploitation_index`: `E_i = (U_i_actual - U_i_lindahl) / |U_i_lindahl|`
   - `core_membership`: checks if outcome is in the cooperative game core
   - `adaptation_rate`: L1 norm of pledge changes across rounds

3. **Phase Handlers** (`strong_models_experiment/phases/phase_handlers.py`)
   - `run_pledge_submission_phase()`: each agent submits contribution vector simultaneously
   - `run_feedback_phase()`: updates game state, shows aggregate totals only
   - Reuses standard phases: `run_discussion_phase()`, `run_private_thinking_phase()`, `run_individual_reflection_phase()`
   - Protocol routing: `get_protocol_type()` returns `"talk_pledge_revise"` for co-funding

4. **Base Class Extensions** (`game_environments/base.py`)
   - `GameType.CO_FUNDING = "co_funding"` enum value
   - `CoFundingConfig` dataclass: alpha, sigma, m_projects, c_min, c_max with validation
   - `get_protocol_type()` method returning `"propose_and_vote"` (default) or `"talk_pledge_revise"`

5. **Experiment Orchestrator Integration** (`strong_models_experiment/experiment.py`)
   - Lines 186-200: co-funding game creation path
   - Lines 251-259: game state extraction (projects, valuations, budgets)
   - Protocol detection routes to talk-pledge-revise loop (no enumeration/voting/tabulation)
   - `agent_budgets` and `total_budget` saved in config (fixed from initial omission)

6. **CLI Entry Point** (`run_strong_models_experiment.py`)
   - `--game-type co_funding`
   - New args: `--alpha`, `--sigma`, `--m-projects`, `--c-min`, `--c-max`

7. **Config Generation Script** (`scripts/generate_cofunding_configs.sh` — 826 lines)
   - Two experiment types:
     - **Model-scale**: weak vs strong reasoning models, sweeps over alpha × sigma × model_order
     - **TTC-scaling**: baseline (gpt-5-nano) vs reasoning models, sweeps over token_budget × model_order
   - Modes: `--derisk` (1 config), `--small` (reduced), `--model-scale`, `--scaling`, default (both)
   - Alpha sweep: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
   - Sigma sweep: [0.2, 0.4, 0.6, 0.8, 1.0]
   - Token budgets: [100, 500, 1000, 3000, 5000, 10000, 20000, 30000]
   - Generates SLURM sbatch files + local run scripts

8. **Results Collection Script** (`scripts/collect_cofunding_results.sh` — 280 lines)
   - Outputs: `all_results.json`, `results_summary.json`, `results.csv`
   - Extracts final pledges from conversation_logs (workaround for missing `current_pledges` in old results)
   - Deduplicates `experiment_results.json` vs `run_1_experiment_results.json`

9. **Visualization** (`visualization/visualize_cofunding.py` — 812 lines)
   - Heatmaps (alpha × sigma): efficiency, provision_rate, coordination_failure, num_funded, surplus_ratio
   - Model-specific bar charts: free_rider_by_model, lindahl_distance_by_model, utility_gap_by_model, exploitation_index
   - Scatter/line plots: utility_vs_elo (scissors pattern), num_funded_vs_sigma, adaptation_rate_by_sigma
   - `summary_table.csv`

10. **LaTeX Documentation** (`game3_main.tex` — 477 lines)
    - Mathematical formalization: utility function, threshold non-linearity
    - Parameters (alpha, sigma) with game-theoretic interpretation
    - Equilibrium concepts (Lindahl equilibrium, cooperative game core)
    - Strategic dynamics (coalition formation, free-riding, coordination failure)

11. **Tests** (4 files, 1264 lines total)
    - `tests/test_cofunding_game.py` (535 lines) — valuation generation, utility, parsing, validation, seed reproducibility, early termination
    - `tests/test_cofunding_metrics.py` (324 lines) — knapsack, social welfare, Lindahl, free-rider, exploitation, core
    - `tests/test_cofunding_phases.py` (284 lines) — pledge submission, feedback, early termination, full round simulation
    - `tests/test_cofunding_config.py` (121 lines) — config validation, parameter bounds

### Key Files Created
- `game_environments/co_funding.py` — core game logic
- `game_environments/cofunding_metrics.py` — evaluation metrics
- `scripts/generate_cofunding_configs.sh` — experiment config generation
- `scripts/collect_cofunding_results.sh` — results aggregation
- `visualization/visualize_cofunding.py` — plotting and analysis
- `game3_main.tex` — LaTeX documentation
- `tests/test_cofunding_game.py`, `tests/test_cofunding_metrics.py`, `tests/test_cofunding_phases.py`, `tests/test_cofunding_config.py`

### Key Files Modified
- `game_environments/base.py` — added CoFundingConfig, GameType.CO_FUNDING, get_protocol_type()
- `game_environments/__init__.py` — exports for co-funding classes
- `strong_models_experiment/experiment.py` — co-funding game creation + protocol routing
- `strong_models_experiment/phases/phase_handlers.py` — pledge submission + feedback phases
- `run_strong_models_experiment.py` — new CLI args for co-funding
- `strong_models_experiment/experiment_utils.py` — save agent_budgets/total_budget in config

---

## How Game 3 Experiments Run

### Entry Point
`run_strong_models_experiment.py` with `--game-type co_funding`

### Example Command
```bash
python run_strong_models_experiment.py \
    --models gpt-5-nano o3-mini-high \
    --game-type co_funding \
    --alpha 0.5 \
    --sigma 0.5 \
    --m-projects 5 \
    --max-rounds 10 \
    --batch --num-runs 3 \
    --random-seed 42
```

### Flow
1. `run_strong_models_experiment.py` parses args, creates experiment_config
2. `StrongModelsExperiment.run_single_experiment()` creates `GameEnvironment` via factory
3. `create_game_environment("co_funding", ...)` → `CoFundingGame`
4. `game.create_game_state(agents)` generates projects (costs), budgets (sigma × total_cost / n_agents), and valuations (SLSQP with target cosine similarity alpha)
5. Protocol routing: `get_protocol_type()` → `"talk_pledge_revise"`
6. Round loop: discussion → thinking → pledge_submission → feedback → reflection
7. Early termination if pledges identical for 2 consecutive rounds
8. Final outcome: utility computed from last round's funded set
9. Results saved to `experiments/results/`

### Protocol: Talk-Pledge-Revise (vs Propose-and-Vote)

| Aspect | Games 1/2 (Propose-and-Vote) | Game 3 (Talk-Pledge-Revise) |
|--------|------------------------------|----------------------------|
| Action | Propose full allocation | Submit contribution vector |
| Decision | Majority vote accept/reject | Threshold funding (automatic) |
| Feedback | Vote results | Aggregate totals only |
| Discount | γ per round | None (final round only) |
| Phases | discussion → thinking → proposal → enumeration → voting → tabulation → reflection | discussion → thinking → pledge → feedback → reflection |

### Parameters
- `--alpha`: Preference alignment [0, 1] (valuation cosine similarity)
- `--sigma`: Budget scarcity (0, 1] (total_budget = sigma × total_project_cost)
- `--m-projects`: Number of projects (default: 5)
- `--c-min`, `--c-max`: Project cost range (default: 10-50)
- `--max-rounds`: Max negotiation rounds (default: 10)
- `--discussion-turns`: Turns per discussion phase (default: 3)

---

## Key Design Decisions

### 1. Threshold Non-Linearity
Unlike Games 1/2 (linear allocation), Game 3 has binary project outcomes: funded or not. This creates strategic complexity — agents can free-ride on others' contributions.

### 2. Aggregate-Only Feedback
Agents see total contributions per project but NOT who contributed what. This prevents direct retaliation but enables strategic manipulation through cheap talk in discussion rounds.

### 3. No Discount Factor
Only the final round outcome matters (γ = 1.0 implicitly). This differs from Games 1/2 where earlier agreements are preferred. Rationale: pledges are revised, not rejected.

### 4. Valuation via SLSQP (Same Approach as Game 2)
Used the same optimization-based approach for generating correlated valuations as Game 2's weight generation. Target cosine similarity = alpha.

### 5. Separate Metrics Module
`cofunding_metrics.py` is separate from `metrics.py` (Games 1/2) because Game 3's metrics are fundamentally different — threshold-based, public-goods-specific (Lindahl, free-rider index, provision rate, etc.).

---

## Smoke Test Results & Key Findings

### Pure Free-Riding
- **o3-mini-high contributes 0** against gpt-5-nano, gains full benefit from projects funded entirely by the opponent
- Exploitation increases with adversary Elo rating (scissors pattern in utility-vs-Elo plot)

### Coordination Failure
- gpt-5-nano has **~79% coordination failure rate** (funds wrong projects or spreads contributions too thin)
- Efficiency maxes at **21%** even in best conditions (alpha=1, sigma=0.7)

### Parameter Effects
- **Higher alpha** → better coordination (agents agree on which projects matter)
- **Higher sigma** → more projects funded (less scarcity)

---

## Known Issues & Workarounds

1. **Duplicate result files**: Both `experiment_results.json` and `run_1_experiment_results.json` saved → deduplication needed in visualization and collection scripts
2. **Missing `current_pledges` in old configs**: Workaround in `collect_cofunding_results.sh` extracts from conversation_logs pledge_submission phase
3. **`agent_budgets`/`total_budget` not in old configs**: Fixed in `experiment_utils.py` (now saved); old results require recalculation
4. **Exploitation index blow-up**: Near-zero Lindahl utility → clipped to [-5, 5] in visualization
5. **Free-rider index infinity**: Pure free-riders (contribute 0) → F = inf, capped at 10.0 for display

---

## Architecture Notes

- `game_environments/` = game logic layer (game-specific)
- `strong_models_experiment/` = experiment runner (game-agnostic orchestration, API calls, phases, logging)
- `run_strong_models_experiment.py` = CLI entry point
- The experiment runner delegates all game-specific logic to `GameEnvironment` subclasses
- Protocol type (`propose_and_vote` vs `talk_pledge_revise`) determines which phase loop the orchestrator uses
- Game 3 reuses the same agent infrastructure (system prompts, conversation history, API calls) as Games 1/2

## What's Next (Not Yet Implemented)
- Large-scale cluster experiments (config generator ready, awaiting deployment)
- TTC-scaling experiments (token budget sweep)
- Cross-game comparison analysis
- Publication-quality figures for paper
