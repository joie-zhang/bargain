# Unused Files Analysis

## Execution Flow

### Pipeline:
1. `scripts/generate_configs_both_orders.sh` → Generates JSON config files
2. `scripts/run_all_simple.sh` → Calls `run_single_experiment_simple.sh` for each job
3. `scripts/run_single_experiment_simple.sh` → Calls `run_strong_models_experiment.py`
4. `run_strong_models_experiment.py` → Imports `strong_models_experiment` package
5. `scripts/collect_results.sh` → Aggregates results (called at end of run_all_simple.sh)

### Python Import Chain:
- `run_strong_models_experiment.py`
  - `strong_models_experiment.StrongModelsExperiment`
  - `strong_models_experiment.STRONG_MODELS_CONFIG`
- `strong_models_experiment/experiment.py`
  - `strong_models_experiment/agents/agent_factory.py`
  - `strong_models_experiment/phases/phase_handlers.py`
  - `strong_models_experiment/utils/experiment_utils.py`
  - `strong_models_experiment/analysis/experiment_analyzer.py`
- `strong_models_experiment/phases/phase_handlers.py`
  - `negotiation.NegotiationContext`
  - `negotiation.llm_agents.BaseLLMAgent`
- `strong_models_experiment/agents/agent_factory.py`
  - `negotiation.AgentFactory`
  - `negotiation.AgentConfiguration`
  - `negotiation.llm_agents.*` (ModelType, BaseLLMAgent, AnthropicAgent, OpenAIAgent, LLMConfig)
  - `negotiation.openrouter_client.OpenRouterAgent`
- `strong_models_experiment/utils/experiment_utils.py`
  - `negotiation.create_competitive_preferences`
- `negotiation/preferences.py`
  - `negotiation/random_vector_generator.py`
  - `negotiation/multi_agent_vector_generator.py`

## Files USED in negotiation/ folder:

✅ **KEEP THESE:**
- `negotiation/__init__.py` - Exports NegotiationContext
- `negotiation/llm_agents.py` - BaseLLMAgent, NegotiationContext, ModelType, AnthropicAgent, OpenAIAgent, LLMConfig
- `negotiation/openrouter_client.py` - OpenRouterAgent
- `negotiation/preferences.py` - create_competitive_preferences
- `negotiation/random_vector_generator.py` - Used by preferences.py
- `negotiation/multi_agent_vector_generator.py` - Used by preferences.py
- `negotiation/agent_factory.py` - AgentFactory, AgentConfiguration (USED by StrongModelAgentFactory)

## Files NOT USED in negotiation/ folder:

❌ **CAN BE DELETED:**
- `negotiation/environment.py` - Not imported by strong_models_experiment
- `negotiation/communication.py` - Not imported
- `negotiation/experiment_phases.py` - Not imported (strong_models_experiment has its own phases)
- `negotiation/negotiation_runner.py` - Not imported
- `negotiation/model_clients.py` - Not imported
- `negotiation/model_config.py` - Not imported (though it might be useful for future use)
- `negotiation/config_integration.py` - Not imported
- `negotiation/agent_experience_logger.py` - Not imported
- `negotiation/experiment_analysis.py` - Not imported
- `negotiation/utility_engine.py` - Not imported
- `negotiation/enhanced_vector_generator.py` - Not imported
- `negotiation/engine/` - Entire directory not imported

## Files USED in scripts/ folder:

✅ **KEEP THESE:**
- `scripts/generate_configs_both_orders.sh` - Main config generator
- `scripts/run_all_simple.sh` - Main runner
- `scripts/run_single_experiment_simple.sh` - Single experiment runner
- `scripts/collect_results.sh` - Called by run_all_simple.sh

## Files NOT USED in scripts/ folder:

❌ **CAN BE DELETED:**
- `scripts/generate_configs_3agent.sh` - Not called by pipeline
- `scripts/generate_configs_short_run.sh` - Not called by pipeline
- `scripts/generate_single_config.sh` - Only used if config_0.json doesn't exist (fallback)
- `scripts/run_all_3agent.sh` - Not called by pipeline
- `scripts/run_single_3agent_experiment.sh` - Not called by pipeline
- `scripts/run_short_experiment.sh` - Not called by pipeline
- `scripts/collect_3agent_results.sh` - Not called by pipeline
- `scripts/analyze_order_effects.py` - Not called by pipeline
- `scripts/kill_all_jobs.sh` - Utility script, not part of pipeline
- `scripts/run_all_simple.sh.backup` - Backup file
- `scripts/run_single_experiment_simple.sh.backup` - Backup file
- `scripts/previous_old_old_scripts/` - Entire directory (old scripts)
- `scripts/spurious_claude_scripts_addressing_timeout/` - Entire directory (old scripts)
- `scripts/previous_script_scaling_experiment_hard_coded_pref_vectors_utility_imbalance_has_timeouts/` - Entire directory (old scripts)

## Summary

### Negotiation folder:
- **Used:** 7 files (llm_agents.py, openrouter_client.py, preferences.py, random_vector_generator.py, multi_agent_vector_generator.py, agent_factory.py, __init__.py)
- **Unused:** 11+ files/directories

### Scripts folder:
- **Used:** 4 files (generate_configs_both_orders.sh, run_all_simple.sh, run_single_experiment_simple.sh, collect_results.sh)
- **Unused:** 10+ files/directories

## Notes:
- `negotiation/model_config.py` might be useful for future model registry features - consider keeping (currently not used)
- Backup files (.backup) can definitely be deleted
- Old script directories can be deleted if you're confident they're not needed

