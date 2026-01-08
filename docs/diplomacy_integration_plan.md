# Implementation Plan: Diplomacy Environment Integration

## Overview

Integrate the Diplomatic Treaty game into the existing multi-agent negotiation framework, making it modular to support both Item Allocation and Diplomacy games through a unified game abstraction layer.

**User Decisions:**
- Proposal format: Continuous values in [0,1]
- Original file: Consolidate into new module, archive original
- Issue naming: Thematic (Trade Policy, Security, etc.)
- Test scope: Comprehensive

---

## Phase 1: Create Game Abstraction Layer

### 1.1 Create `game_environments/` package

**Files to create:**
- `game_environments/__init__.py` - Package exports and factory function
- `game_environments/base.py` - GameEnvironment abstract base class
- `game_environments/item_allocation.py` - Item Allocation game implementation
- `game_environments/diplomatic_treaty.py` - Diplomatic Treaty game implementation

**GameEnvironment ABC interface:**
```python
class GameEnvironment(ABC):
    def create_game_state(agents) -> Dict
    def get_game_rules_prompt(game_state) -> str
    def get_preference_assignment_prompt(agent_id, game_state) -> str
    def get_proposal_prompt(agent_id, game_state, round_num, agents) -> str
    def parse_proposal(response, agent_id, game_state, agents) -> Dict
    def validate_proposal(proposal, game_state) -> bool
    def calculate_utility(agent_id, proposal, game_state, round_num) -> float
    def get_discussion_prompt(agent_id, game_state, round_num, history) -> str
    def get_voting_prompt(agent_id, proposal, game_state, round_num) -> str
    def get_thinking_prompt(agent_id, game_state, round_num, history) -> str
    def format_proposal_display(proposal, game_state) -> str
    def get_game_type() -> str
```

### 1.2 Implement ItemAllocationGame

Extract existing logic from:
- `strong_models_experiment/prompts/prompt_generator.py` (prompts)
- `strong_models_experiment/utils/experiment_utils.py` (items, preferences)
- `strong_models_experiment/phases/phase_handlers.py` (proposal parsing, utility)

### 1.3 Implement DiplomaticTreatyGame

Port logic from `diplomacy_environment/diplo_implementation.py`:
- Preference generation (positions, weights with rho/theta/lambda)
- Utility calculation: `U = Σ w_k × (1 - |p_k - a_k|)`
- Continuous proposals in [0,1]

**Thematic issue names:**
```python
ISSUE_NAMES = [
    "Trade Policy", "Military Access", "Environmental Standards",
    "Resource Sharing", "Border Security", "Technology Transfer",
    "Financial Cooperation", "Cultural Exchange"
]
```

---

## Phase 2: Modify Experiment Infrastructure

### 2.1 Update `strong_models_experiment/phases/phase_handlers.py`

**Changes:**
- Accept `GameEnvironment` in constructor
- Replace hardcoded prompt generation with `game_env.get_*_prompt()` calls
- Replace proposal parsing with `game_env.parse_proposal()`
- Replace utility calculation with `game_env.calculate_utility()`

**Key modifications:**
```python
class PhaseHandler:
    def __init__(self, game_environment: GameEnvironment, save_interaction_callback=None, token_config=None):
        self.game_env = game_environment
        # ... rest unchanged
```

### 2.2 Update `strong_models_experiment/experiment.py`

**Changes:**
- Accept `game_type` parameter
- Create appropriate `GameEnvironment` via factory
- Pass `game_env` to `PhaseHandler`
- Replace `create_items()` and `create_preferences()` with `game_env.create_game_state()`

### 2.3 Update `run_strong_models_experiment.py`

**Add CLI arguments:**
```python
--game-type [item_allocation|diplomacy]  (default: item_allocation)
--n-issues INT                           (default: 5, diplomacy only)
--rho FLOAT                              (default: 0.0, range [-1,1])
--theta FLOAT                            (default: 0.5, range [0,1])
--lam FLOAT                              (default: 0.0, range [-1,1])
```

---

## Phase 3: Comprehensive Test Suite

### 3.1 Create `tests/test_diplomatic_treaty.py`

**Unit tests:**
- Configuration validation (rho, theta, lambda bounds)
- Game state creation (positions in [0,1], weights sum to 1)
- Utility calculation (perfect match = max utility)
- Proposal validation (correct number of issues, values in range)
- Proposal parsing (JSON extraction, fallback handling)
- Control parameter effects:
  - High rho → correlated positions
  - High theta → overlapping weights
  - Low lambda → conflicting issues

**Integration tests:**
- Full negotiation flow with mock agents
- Consensus detection
- Discounting over rounds

### 3.2 Create `tests/test_item_allocation_refactor.py`

**Regression tests:**
- Verify item allocation still works identically after refactor
- Compare outputs with original implementation
- Ensure backward compatibility

### 3.3 Create `tests/test_game_environments.py`

**Abstraction tests:**
- Factory function creates correct game type
- Both games implement full interface
- Common behavior across games (discounting, voting)

---

## Phase 4: Archive and Cleanup

### 4.1 Archive original diplomacy implementation

```
diplomacy_environment/ → legacy/diplomacy_environment/
```

Keep as reference for validation.

### 4.2 Negotiation Module Cleanup

**Analysis of `negotiation/` folder usage by `run_strong_models_experiment.py`:**

#### ✅ REQUIRED FILES (9 files - keep these):
| File | Purpose |
|------|---------|
| `__init__.py` | Package root (needs cleanup to remove unused imports) |
| `llm_agents.py` | Agent classes: BaseLLMAgent, AnthropicAgent, OpenAIAgent, XAIAgent, LocalModelAgent |
| `openrouter_client.py` | OpenRouterAgent class |
| `agent_factory.py` | AgentFactory, AgentConfiguration |
| `preferences.py` | create_competitive_preferences function |
| `random_vector_generator.py` | Used by preferences.py |
| `multi_agent_vector_generator.py` | Used by preferences.py |
| `model_clients.py` | Used by llm_agents.py |
| `model_config.py` | Used by llm_agents.py |

#### ❌ UNUSED FILES (9 files - archive to `legacy/negotiation/`):
| File | Why Unused |
|------|------------|
| `environment.py` | NegotiationEnvironment imported in __init__ but never used |
| `communication.py` | Message, CommunicationManager imported but never used |
| `negotiation_runner.py` | ModularNegotiationRunner imported but never used |
| `utility_engine.py` | UtilityEngine imported but never used |
| `agent_experience_logger.py` | Not imported anywhere in active code |
| `config_integration.py` | Not imported anywhere in active code |
| `enhanced_vector_generator.py` | Not imported anywhere in active code |
| `experiment_phases.py` | Old phase handlers, replaced by strong_models_experiment |
| `experiment_analysis.py` | Analysis tools not used in current pipeline |

#### Cleanup Actions:
1. Move 9 unused files to `legacy/negotiation/`
2. Update `negotiation/__init__.py` to remove imports of archived modules
3. Test that `run_strong_models_experiment.py` still works

### 4.3 Other cleanup items

1. **Dead code**: `_generate_similar_vector` in preferences.py appears unused
2. **Inconsistent agent ID handling**: Standardize to always use `agent.agent_id`
3. **Result files in root**: Move `qwen_results.*` to `experiments/results/`
4. **Backup file**: Remove `configs.py.bak`

---

## Critical Files

### New files to create:
- `/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/__init__.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/base.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/item_allocation.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/diplomatic_treaty.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_diplomatic_treaty.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_item_allocation_refactor.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_game_environments.py`

### Files to modify:
- `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py`

### Reference files:
- `/scratch/gpfs/DANQIC/jz4391/bargain/diplomacy_environment/diplo_implementation.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/prompts/prompt_generator.py`
- `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py`

---

## Implementation Order

1. **Create game_environments/base.py** - Define abstract interface
2. **Create game_environments/item_allocation.py** - Extract existing logic
3. **Create tests/test_item_allocation_refactor.py** - Regression tests
4. **Verify existing tests pass** - No regressions
5. **Create game_environments/diplomatic_treaty.py** - Port from diplo_implementation.py
6. **Create tests/test_diplomatic_treaty.py** - Comprehensive diplomacy tests
7. **Create game_environments/__init__.py** - Factory function
8. **Modify PhaseHandler** - Use GameEnvironment
9. **Modify StrongModelsExperiment** - Create game via factory
10. **Update CLI** - Add --game-type and diplomacy parameters
11. **Create tests/test_game_environments.py** - Abstraction tests
12. **Archive diplomacy_environment/** - Move to legacy/
13. **Cleanup** - Remove dead code, fix identified issues
14. **Final validation** - Run both game types end-to-end

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking item allocation | High | Create regression tests BEFORE refactoring |
| LLM parsing failures for continuous values | Medium | Robust JSON extraction with midpoint fallback |
| Complexity from 3 control parameters | Medium | Clear documentation, sensible defaults |

---

## Diplomacy Game Details

### Three Control Parameters

1. **ρ (rho) ∈ [-1, 1]**: Preference correlation
   - ρ = 1: Agents want same outcomes (cooperative)
   - ρ = 0: Uncorrelated preferences
   - ρ = -1: Agents want opposite outcomes (competitive)

2. **θ (theta) ∈ [0, 1]**: Interest overlap
   - θ ≈ 1: Both agents care about same issues (competitive)
   - θ ≈ 0: Different priorities (logrolling potential)

3. **λ (lambda) ∈ [-1, 1]**: Issue compatibility
   - λ = 1: All win-win issues
   - λ = 0: Mixed issue types
   - λ = -1: All zero-sum issues

### Utility Calculation

```
U_i(A) = Σ_k w_ik × (1 - |p_ik - a_k|)

where:
- w_ik = importance weight for agent i on issue k
- p_ik = agent i's ideal position on issue k
- a_k = agreed value for issue k
```

### Example Scenarios

| Scenario | ρ | θ | λ | Description |
|----------|---|---|---|-------------|
| Pure Cooperation | 1.0 | 0.0 | 1.0 | Aligned preferences, different priorities, win-win |
| Pure Competition | -1.0 | 1.0 | -1.0 | Opposing preferences, same priorities, zero-sum |
| Integrative Bargaining | 0.0 | 0.3 | 0.5 | Classic tradeoff potential |

---

## Usage Example (After Implementation)

```bash
# Item Allocation (unchanged - default)
python run_strong_models_experiment.py \
    --models claude-sonnet gpt-4o \
    --num-items 5 \
    --competition-level 0.95

# Diplomatic Treaty (new)
python run_strong_models_experiment.py \
    --models claude-sonnet gpt-4o \
    --game-type diplomacy \
    --n-issues 5 \
    --rho 0.0 \
    --theta 0.3 \
    --lam 0.5
```
