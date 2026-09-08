# E06: Llama Game 3 replication

## Result and verification

Current appendix lines 438-461 and headline table report 180 Game3 runs within the 500-run Llama cohort. Directly read all 180 indexed saved configurations and run_1_experiment_results.json files. Each of ten adversary models has 18 runs, crossing alpha {0,.5,1}, sigma {.2,.6,1}, and two orders. Seeds are 42-221. All 180 selected results exist; 170 reached agreement. A fresh read-only NumPy calculation over raw role utilities gives adversary slope +10.70437348, baseline slope +2.19354417, gap slope +8.51082931 per100 Elo, and adversary Pearson r=.8664928. These match displayed rounded results. No experiments or files changed beyond these audit reports.

## Provenance chain

Generator -> saved Game3 configs and experiment_index.csv -> saved submit_individual.sh/run_one.sbatch/run_config.py -> run_strong_models_experiment.py -> StrongModelsExperiment -> model factory, phases, prompts, CoFundingGame -> indexed raw result plus interaction logs -> analyze_appendix_llama33_baseline_500.py -> overall_by_model_game.csv -> plot_appendix_llama_overall_overlay_1x3.py -> appendix figure.

Saved Slurm launcher uses CPU, .venv Python, API-key environment, and OPENROUTER_TRANSPORT=proxy with external /home/jz4391/openrouter_proxy. Do not read or remove credentials. The baseline routes through OpenRouter. Current rerun code dependencies and historical execution identity are not interchangeable.

## History verified

Used codex-search bundled search for Llama baseline 500/180/Game3. Manually read /home/jz4391/.codex/history.jsonl lines 1283, 1366-1367, 1384. Session 019deb36-702e-7281-831e-85fa43303c19 requests 500 CPU jobs, then 79 Game3 and 16 Game2 backfills due 402/403 errors, then analysis of all 500 completed runs. Line1367 explicitly requests overwriting the same output directories to hide failed first attempts. This means surviving failure logs are provenance, not disposable clutter. The full May session is absent according to locally retained recreation evidence. Requested session 01a074b4-92a4-78c3-96f8-c42ae5db663d supplied figure-recreation leads; its report is not treated as original launch proof.

## Needed files

The JSON lists concrete files and exact directory selection rules. Current imports additionally protect package __init__.py files, strong_models_experiment/analysis/{analyzer,qualitative_metrics,qualitative_schema}.py, negotiation/{preferences,random_vector_generator,agent_factory}.py, and game_environments/{item_allocation,diplomatic_treaty,metrics}.py: these are eagerly imported even though Game3 runs co_funding. This is an import requirement, not evidence those other games execute in this result. No blanket classification of those package directories is intended.

## Limits and removal candidates

- No proof of exact historical execution commit; current runtime has changed.
- Current generator uses 16384 tokens but saved configurations use 10500.
- May full session absent; history confirms 79 Game3 backfills and asks to overwrite original paths. Preserve all surviving failure, Slurm, and provider records until recovery history is reconciled.
- Dynamic dependencies and external /home/jz4391/openrouter_proxy service not exhaustively traced; no deletion safety certification.
- Saved shortened provider names do not alone verify reasoning settings; raw utility recomputation from every transcript not performed.

No removal candidates are established by this scoped result. Duplicate result files cannot be deemed deletable without checking other loaders. Failure/retry logs cannot be deemed unnecessary merely because selected results succeeded. Existing tests and other experiments remain protected.

