# Script Retention Manifest

## Scope

This manifest records the cleanup of the 188 non-cache files that were under
`scripts/` on 2026-07-19. It does not count `__pycache__` files. The cleanup
deleted those generated files.

Use these decisions for the arXiv cleanup:

- `KEEP`: required for experiment, analysis, table, or figure reproduction.
- `BUNDLE`: required for a retained internal analysis. It is now in
  `scripts/retained_analysis/`.
- `DELETE`: a test, monitor, repair helper, superseded plot, or old launcher.
  The cleanup deleted it.

The active figure manifests are:

- `docs/reproducibility/paper_figure_manifest.csv`
- `docs/reproducibility/icml_figure_manifest.csv`
- `docs/reproducibility/neurips_figure_manifest.csv`

Those manifests cover figures only. This file also covers experiment launchers,
table generators, intermediate CSV producers, and retained analysis reports.

## Keep: Paper Figure Package

Keep all 22 non-cache files in `scripts/paper_figures/`:

- `README.md`
- `__init__.py`
- `assets/endpoint_fairness_elo_snapshot.csv`
- `assets/fairshare_residual_combined_base.png`
- `assets/game2_utility_roster_snapshot.csv`
- `make_combined_aligned_font_balanced.py`
- `plot_fairshare_residual_combined.py`
- `plot_figure3_baseline_by_competition_ewma_iteration.py`
- `plot_figure4_total_welfare_by_competition_ewma.py`
- `plot_homogeneous_adversary_baseline_vs_all_gini.py`
- `plot_icml_homogeneous_adversary_main_panels.py`
- `plot_icml_ttc_main_figures.py`
- `plot_n2_endpoint_fairness_style_matched_drop_game2_outlier.py`
- `plot_n2_fairness_distance_three_game_curves.py`
- `plot_n_gt_2_heterogeneous_utility_distribution.py`
- `plot_random_monoculture_gini_vs_heterogeneous.py`
- `plot_role_payoff_with_within_run_variance_bars.py`
- `plot_ttc_effort_adversary_baseline.py`
- `plot_ttc_game_averaged_observed_tokens.py`
- `render_figure2_large_fonts.py`
- `render_figure7_label_edits.py`
- `verify_all.py`

## Keep: Experiment And Analysis Provenance

Keep these 26 top-level files:

- `analyze_appendix_llama33_baseline_500.py`: appendix Llama analysis and figures.
- `analyze_n2_baseline_comparison.py`: main bilateral tables, statistics, and figures.
- `analyze_n2_plus_multiagent_comparison.py`: main multi-agent tables, statistics, and figures.
- `analyze_nash_lindahl_fairness.py`: NBS and Lindahl benchmark calculations.
- `analyze_neurips_revision_stats.py`: TTC and paper revision statistics.
- `export_game2_batch_pngs.py`: Game 2 aggregate and fixed paper input export.
- `full_games123_multiagent_batch.py`: production homogeneous and heterogeneous batches.
- `generate_all_prompts_reference.py`: prompt-reference document generator.
- `generate_appendix_llama33_baseline_configs.py`: appendix Llama experiment generator.
- `generate_cofunding_configs.sh`: main Game 3 experiment generator.
- `generate_configs_both_orders.sh`: main Game 1 bilateral experiment generator.
- `generate_diplomacy_configs.sh`: main Game 2 experiment generator.
- `generate_ttc_native_scaling_jobs.py`: native TTC experiment generator.
- `plot_exploitation_vs_elo.py`: input loader for the revision analysis.
- `plot_full_games123_clean_subset.py`: table builder used by the multi-agent analysis.
- `plot_game3_utility_vs_elo.py`: producer of a capability-analysis input table.
- `plot_gpt5_nano_baseline_vs_elo_all_games.py`: input loader for the revision analysis.
- `plot_nbs_decomposition.py`: producer of the NBS decomposition table in the paper.
- `plot_n_gt_2_bucketed_homogeneous_heterogeneous_breakdowns.py`: producer of role-payoff input data.
- `plot_n_gt_2_role_payoff_curves_by_strength.py`: producer of role-payoff input data.
- `plot_scaling_utility_vs_elo.py`: producer of a capability-analysis input table.
- `plot_ttc_group_intensity_combined.py`: producer of the retained TTC intensity input table.
- `random_monoculture_control_batch.py`: random-monoculture experiment generator.
- `run_ttc_native_config.py`: native TTC experiment worker.
- `submit_cofunding_then_diplomacy.sh`: submission helper used by the Game 2 and Game 3 generators.
- `validate_paper_figure_manifest.py`: active paper-figure validation.

## Bundle: Retained Internal Analysis

Keep these 34 files for the current rebuttal and related-work period. They are
in `scripts/retained_analysis/`. The README in that directory maps each script
to its report, input data, and assets. This layout makes the later deletion one
directory operation.

- `analyze_capability_payoff_scaling_20260505.py`
- `analyze_context_length_errors.py`
- `analyze_heterogeneous_vs_homogeneous_inequality.py`
- `analyze_homogeneous_adversary_tag_mechanism.py`
- `analyze_ttc_hot_strategic_tags.py`
- `analyze_ttc_objective_shift.py`
- `analyze_ttc_objective_shift_deltas.py`
- `backfill_qualitative_metrics.py`
- `build_game1_position_qualitative.py`
- `build_n2_report_with_ttc.py`
- `build_n2_ttc_multiagent_report.py`
- `build_qualitative_dynamics_trend_report.py`
- `build_qualitative_rollout_dynamics_report.py`
- `build_random_monoculture_sanity_report.py`
- `build_refined_qualitative_dynamics.py`
- `build_strategic_qualitative_tags.py`
- `build_tag_review_game_ui.py`
- `build_ttc_paper_style_mini_section.py`
- `explore_llm_strategic_tag_elo_trends.py`
- `explore_ttc_strategic_tag_intensity_trends.py`
- `explore_ttc_strategic_tag_trends.py`
- `plot_homogeneous_adversary_redline_by_elo.py`
- `plot_ttc_compute_language_bridge.py`
- `plot_ttc_fairness_vs_compute.py`
- `prepare_llm_strategic_tag_adjudication.py`
- `prepare_llm_strategic_tag_adjudication_n2_gpt5.py`
- `prepare_llm_strategic_tag_adjudication_random_monoculture.py`
- `prepare_ttc_llm_strategic_tag_adjudication.py`
- `qualitative_judge_harness.py`
- `repair_llm_adjudication_chunk.py`
- `repair_llm_adjudication_chunk_0038.py`
- `summarize_ttc_llm_strategic_tag_adjudication.py`
- `validate_llm_strategic_tag_adjudication.py`
- `validate_ttc_llm_strategic_tag_adjudication.py`

## Deleted

The cleanup deleted these 106 files. They did not meet the retention standard.

### Old Documentation And Utilities

- `GPU_REQUIREMENTS.md`
- `README.md`
- `README_logs.md`
- `apply_prompt_changes.py`
- `build_reonboarding_html.py`
- `log_utils.sh`
- `materialize_malformed_json_examples.py`

The old `scripts/README.md` describes files that do not exist in this
repository. It is not a valid user guide.

### Old Collection, Test, Debug, And Monitoring Files

- `analyze_order_effects.py`
- `analyze_qwen_results.py`
- `audit_openrouter_contexts_32.py`
- `backfill_exact_openrouter_failures.py`
- `collect_3agent_results.sh`
- `collect_cofunding_results.sh`
- `collect_nagent_results.sh`
- `collect_qwen_results.py`
- `collect_results.sh`
- `cosine_sim_testing.ipynb`
- `debug_agent_swap.py`
- `debug_gpt52_effort_agents.py`
- `derisk_15_samples_batch.py`
- `monitor_822_backfill.py`
- `monitor_appendix_llama33_baseline.py`
- `monitor_random_monoculture_derisk.py`
- `plot_random_cosine_distributions.py`
- `report_game1_multiagent_full.py`
- `report_llm_strategic_tag_progress.py`
- `report_ttc_llm_strategic_tag_progress.py`
- `test_command_r_openrouter.py`
- `validate_cosine_similarity.py`
- `archive/game1_ttc_access_opus_opus_backfill.py`
- `archive/monitor_game1_matrix_progress.py`
- `archive/monitor_game3_sample_progress.py`

### Superseded Experiment Launchers

- `game1_multiagent_full_batch.py`
- `game1_multiagent_matrix_batch.py`
- `game1_ttc_access_batch.py`
- `game1_ttc_access_matrix_batch.py`
- `game2_derisk_32.py`
- `game3_multiagent_sample_batch.py`
- `generate_configs_3agent.sh`
- `generate_game2_llama32_1b_inplace_plicp.py`
- `generate_game2_llama32_1b_plicp.py`
- `generate_game3_cluster_backfill_inplace_pli.py`
- `generate_game3_cluster_fallback.py`
- `generate_game3_llama31_2gpu_fallback.py`
- `generate_game3_llama32_1b_plicp.py`
- `generate_game3_reference_batch.py`
- `generate_gpt52_effort_configs.py`
- `generate_gpt52_effort_configs.sh`
- `generate_nagent_configs.sh`
- `generate_partial_multiagent_plots_report.py`
- `generate_single_config.sh`
- `generate_ttc_configs.sh`
- `generate_ttc_experiments.py`
- `replot_game1_multiagent_from_summary.py`
- `rerun_experiments.sh`
- `run_all_3agent.sh`
- `run_all_simple.sh`
- `run_api_jobs_tmux.sh`
- `run_batch_tmux_sessions.sh`
- `run_conservative_autonomous_8h.sh`
- `run_game1_multiagent_smoke.py`
- `run_game2_sample_tests.py`
- `run_game3_reference_batch.py`
- `run_grok4_tmux.sh`
- `run_short_experiment.sh`
- `run_single_3agent_experiment.sh`
- `run_single_experiment_simple.sh`
- `submit_extended_derisk_n10_amazon_gpt4o.sh`

### Superseded Or Unused Analysis Variants

- `analyze_n_gt_2_homogeneous_vs_heterogeneous_inequality.py`
- `analyze_n_gt_2_low_high_elo_regime_variance.py`
- `plot_combined_utility_vs_elo.py`
- `plot_game1_control_position_effects.py`
- `plot_heterogeneous_average_payoff_vs_elo_variance.py`
- `plot_heterogeneous_gini_by_elo_std_bins.py`
- `plot_heterogeneous_gini_by_elo_std_named_quantile_bins.py`
- `plot_heterogeneous_gini_by_elo_std_named_quantile_bins_n8_n10.py`
- `plot_heterogeneous_gini_by_elo_std_percentile_bins.py`
- `plot_heterogeneous_gini_vs_elo_variance.py`
- `plot_heterogeneous_gini_vs_elo_variance_breakdowns.py`
- `plot_heterogeneous_max_payoff_vs_max_elo.py`
- `plot_heterogeneous_payoff_variance_vs_min_elo.py`
- `plot_heterogeneous_payoff_variance_vs_min_elo_binned.py`
- `plot_homogeneous_adversary_baseline_vs_all_variance.py`
- `plot_n2_absolute_fairness_style_matched.py`
- `plot_n2_all_competition_fairness_style_matched.py`
- `plot_n2_efficiency_distribution_decomposition.py`
- `plot_n2_fairness_endpoint_outlier_sensitivity.py`
- `plot_n2_fairness_residual_three_game_curves.py`
- `plot_n2_group_model_payoff_corr_errorbars.py`
- `plot_n_gt_2_average_payoff_vs_roster_elo.py`
- `plot_n_gt_2_bucketed_homogeneous_heterogeneous_breakdowns_n8_n10.py`
- `plot_n_gt_2_bucketed_homogeneous_heterogeneous_inequality.py`
- `plot_n_gt_2_fairness_residual_vs_elo.py`
- `plot_n_gt_2_fixed_max_elo_variance_experiment.py`
- `plot_n_gt_2_group_payoff_variance_vs_mean_elo.py`
- `plot_n_gt_2_heterogeneous_variance_correlation.py`
- `plot_n_gt_2_homogeneous_baseline_payoff_by_adversary.py`
- `plot_n_gt_2_included_model_group_payoff_variance_vs_elo.py`
- `plot_n_gt_2_low_high_elo_regime_bar_summaries.py`
- `plot_n_gt_2_max_elo_bin_gini_experiment.py`
- `plot_n_gt_2_max_elo_bin_normalized_dispersion.py`
- `plot_n_gt_2_max_elo_bin_std_experiment.py`
- `plot_n_gt_2_max_elo_bin_variance_experiment.py`
- `plot_n_gt_2_model_payoff_variance_vs_elo.py`
- `plot_n_gt_2_total_payoff_vs_mean_elo.py`
- `plot_ttc_order_averaged_observed_tokens.py`

## Cleanup Verification

The cleanup did these tasks:

1. It moved the 34 `BUNDLE` files and added a report-to-script README.
2. It updated documentation that named moved or deleted files.
3. It deleted all `__pycache__` directories under `scripts/`.
4. All 34 moved modules passed independent import checks.
5. Forty-two focused tests passed. Two complete test-tree collection attempts
   were stopped after they ran for more than 20 minutes. The production batch
   generator test did not return a result during its attempted run. Treat these
   checks as incomplete, not as test failures.
6. All 16 paper figure jobs completed. The verifier restored each declared
   output after the check.
7. The figure manifest validator still reports seven changed NeurIPS graphics.
   These graphics were modified before this script cleanup. Four manual
   composites also keep their existing provenance warnings.

The deleted `Figures/game_1/average_utility_vs_elo.csv` dependency in
`plot_nbs_decomposition.py` was replaced with the canonical model-roster Elo
API. The replacement joins all 30 paper models. It reproduces the existing
slope CSV byte for byte.
