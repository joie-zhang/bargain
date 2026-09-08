# E01 Primary two-player Game 1

## Scope and result

Current paper references fig:bilateral_overview, fig:n2_adversary_payoff_by_competition, and tab:n2_headline_slopes. Includes adversary payoff slope +6.75 per100Elo and baseline payoff by competition, not fairness/consensus/qualitative analyses audited separately.

## Verified chain

The batch is experiments/results/scaling_experiment_20260404_064451. Its stored configs/slurm/run_api_experiments.sbatch invokes run_strong_models_experiment.py with indexed model order, run number, competition, seed and discussion turns. scripts/generate_configs_both_orders.sh generates this directory structure, config fields and launch templates. This establishes the retained launch chain, not the exact command or code version of every April or replacement run.

The current launcher imports strong_models_experiment, its experiment engine, phase handlers, prompts, factory, result classes, analysis and utilities. Factory imports negotiation agent/provider modules. Item preferences depend on negotiation/preferences.py and random_vector_generator.py; game_environments imports all game classes eagerly. These exact files are recorded individually in JSON; do not delete Game2/3 modules merely because this is Game1. Provider queues and external libraries remain dependencies outside this file list.

scripts/analyze_n2_baseline_comparison.py:775 loads indexes/config/result JSON; :1331 selects Game1 discussion_turns=2. active_model_roster.py canonicalizes and limits to30models, reading March31 Elo Markdown. I independently selected420rows, opened all420configs, and verified exact420result paths and420interaction paths exist. JSON includes these1260individual files, rather than protecting an undifferentiated results directory.

The renderer scripts/paper_figures/render_figure2_large_fonts.py reads primary_runs_with_metrics.csv, averages by model and fits30means; baseline curves use competition grouping and smoothing. Reconstruction evidence in analysis/figure_recreation_20260905/figure_02 provides hash manifest and rebuild script and is retained as provenance.

## History checked

Ran the required codex-search bundled search; results were contaminated by current-session restatements. Manually read original July19 session 019f7bf3-7a5d-77d3-9582-cae6f630081b line4328:420primaryGame1runs slope6.753268 and separate840combinedone/twoturn slope6.49865. Read requested September5 session line148 confirming Figure02 agent assignment, then inspected its source artifacts. April history directory is absent locally, so initial launch chronology remains incomplete.

## Important uncertainty

The earlier reconstruction found11Game1role reversals between configured attribution and actual transcript roles. I independently opened its first discrepancy transcript (deepseek-r1-0528 strong_first c0.75 turns2 run2): Agent1 model_name is GPT-5-nano throughout, Agent2 is deepseek/deepseek-r1-0528. This contradicts configured role attribution used for plotting. All11evidence records and selected transcripts must remain. Do not delete historical backfill records as obsolete while these errors remain unresolved.

Stored launcher reads indexed models but current generator defaults and current runtime have changed. Exact historical source revisions, each replacement launch and archived backfill mapping are not proven. Missing utility fields on unsuccessful runs are zero-filled in source extraction, and model-name aliases require runtime verification. Figure recreation proves reproduction of saved attribution, not correctness of every model identity or causal interpretation.

## Cleanup recommendation

No safe deletion candidates are established by this result. One-turn/legacy rows are not part of the420-run primary result but can support other analyses or exclusion provenance. Backfill archives and legacy launchers remain unresolved/protected, not arbitrarily verified-required whole directories. JSON is a positive dependency inventory, not an exhaustive proof that every omitted file is unnecessary. No code, data, paper or Git state changed.
