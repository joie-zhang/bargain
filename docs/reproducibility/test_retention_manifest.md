# Test Retention Manifest

## Scope

This manifest records the cleanup of `tests/` on 2026-07-19. The original
directory contained 75 files. The cleanup kept 33 files and deleted 42 files.

Pytest uses `pytest.ini`. It collects files named `test_*.py` or `*_test.py`
under `tests/`.

## Retained Test Infrastructure

- `conftest.py`: selects the asyncio backend for AnyIO tests.

## Retained Test Modules

Keep these 32 modules. They test current code or paper-critical workflows.

### Game Logic And Metrics

- `test_allocation_preferences.py`
- `test_cofunding_config.py`
- `test_cofunding_game.py`
- `test_cofunding_metrics.py`
- `test_cofunding_phases.py`
- `test_diplomatic_treaty.py`
- `test_diplomatic_treaty_batch_voting.py`
- `test_game_environments.py`
- `test_item_allocation_batch_voting.py`
- `test_item_allocation_refactor.py`
- `test_metrics.py`
- `test_preferences.py`

### Experiment And Phase Behavior

- `test_context_compaction.py`
- `test_game12_parse_diagnostics.py`
- `test_game1_context_propagation.py`
- `test_game2_context_propagation.py`
- `test_game3_context_propagation.py`
- `test_llm_agent_diplomatic_preferences.py`
- `test_majority_vote_tabulation.py`
- `test_parallel_phases.py`
- `test_private_thinking_schema.py`
- `test_prompt_clarity.py`

### Paper Batch And Model Configuration

- `test_full_games123_attempt_logs.py`
- `test_full_games123_batch_generation.py`
- `test_gpt52_effort_aliases.py`
- `test_gpu_llama_phase_caps.py`

### Provider Behavior

- `test_openrouter_provider_fallback.py`
- `test_openrouter_transport.py`
- `test_provider_key_rotation.py`

### Qualitative Analysis

- `test_qualitative_judge.py`
- `test_qualitative_metrics.py`
- `test_qualitative_schema.py`

## Deleted Files

The cleanup deleted these 42 files.

### Generated Outputs And Old Audit Artifacts

- `.gitkeep`
- `all_models_test_results.json`
- `provider_test_results.json`
- `smooth_32_derisk_results.json`
- `test_all_providers.ipynb`

### Live Provider And Debug Programs

- `debug_agent_creation.py`
- `debug_openrouter_models.py`
- `run_debug_with_output.py`
- `test_all_models.py`
- `test_openrouter_direct.py`
- `test_xai_api_script.py`

### Tests For Removed Subsystems

- `test_agent_logging.py`
- `test_communication.py`
- `test_enhanced_config.py`
- `test_environment.py`
- `test_experiment_integration.py`
- `test_experiment_with_cap.py`
- `test_integer_constraints.py`
- `test_llm_agents.py`
- `test_malformed_json_logging.py`
- `test_model_configuration.py`
- `test_o3_debug.py`
- `test_o3_experiment.py`
- `test_o3_parsing.py`
- `test_parameter_effects.py`
- `test_parameter_effects_simple.py`
- `test_quick_validation.py`
- `test_reflection_cap.py`
- `test_reflection_phase.py`
- `test_step_8_1_and_8_2.py`
- `test_utility_engine.py`

These files imported modules or classes that no longer exist in the active
code. Examples include `experiments.o3_vs_haiku_baseline`,
`negotiation.utility_engine`, `negotiation.environment`, and
`SimulatedAgent`.

### Redundant Or Print-Only Programs

- `test_2agent_varying_items.py`
- `test_3agent_all_competition_levels.py`
- `test_3agent_preferences.py`
- `test_grok_experiment.py`
- `test_grok_integration.py`
- `test_model_config.py`
- `test_o3_json_parsing.py`
- `test_phase_handler_integration.py`
- `test_phi3_local_config.py`
- `test_random_vectors.py`
- `test_token_limits.py`

These files did not provide reliable pytest assertions, duplicated a retained
suite, or tested a model that is not in the paper roster.

## Verification

Pytest collected 479 tests from the 32 retained modules in 1.74 seconds. All
479 tests passed in one aggregate run in 225.42 seconds. All tests also passed
during isolated module runs. The 12 production batch tests completed in
approximately 103 seconds. No retained module timed out.

Run the retained suite from the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m pytest
```
