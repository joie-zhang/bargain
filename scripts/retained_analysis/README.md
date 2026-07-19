# Retained Analysis Scripts

## Purpose

This directory contains temporary research analysis scripts. Keep these scripts
through the related-work revision and the NeurIPS rebuttal. They are not part of
the minimum paper reproduction package.

Run each command from the repository root. Use this form:

```bash
.venv/bin/python scripts/retained_analysis/<script>.py
```

The scripts keep their reports, tables, and plots in the existing `analysis/`
or `experiments/` directory. This README gives the location for each work
bundle. Delete this complete directory when the temporary analysis is no longer
necessary.

## Capability And Context Work

| Script | Work bundle |
| --- | --- |
| `analyze_capability_payoff_scaling_20260505.py` | `experiments/results/capability_payoff_scaling_20260505/` |
| `analyze_context_length_errors.py` | `experiments/analysis/context_length_errors_20260502/` |
| `build_n2_report_with_ttc.py` | N=2 report inputs and the retained TTC result tree |
| `build_n2_ttc_multiagent_report.py` | `experiments/results/n2_ttc_multiagent_comparison_analysis_20260505/` |

## Multi-Agent And Inequality Work

| Script | Work bundle |
| --- | --- |
| `analyze_heterogeneous_vs_homogeneous_inequality.py` | Homogeneous and heterogeneous production result trees |
| `build_random_monoculture_sanity_report.py` | Random-monoculture control result tree and its report |
| `build_game1_position_qualitative.py` | `analysis/game1_position_qualitative/` |
| `build_qualitative_rollout_dynamics_report.py` | `analysis/qualitative_rollout_dynamics_20260628/` |
| `build_refined_qualitative_dynamics.py` | `analysis/qualitative_rollout_dynamics_20260628/` |
| `build_qualitative_dynamics_trend_report.py` | `analysis/qualitative_dynamics_trends_20260628/` |
| `plot_homogeneous_adversary_redline_by_elo.py` | `analysis/homogeneous_adversary_redline_elo_20260628/` |
| `backfill_qualitative_metrics.py` | Co-funding result JSON files that contain the retained qualitative metrics |

## Strategic-Tag Work

| Script | Work bundle |
| --- | --- |
| `build_strategic_qualitative_tags.py` | `analysis/strategic_qualitative_tags_20260628/` |
| `prepare_llm_strategic_tag_adjudication.py` | `analysis/llm_strategic_tag_adjudication_20260628/` |
| `prepare_llm_strategic_tag_adjudication_n2_gpt5.py` | `analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629/` |
| `prepare_llm_strategic_tag_adjudication_random_monoculture.py` | `analysis/llm_strategic_tag_adjudication_random_monoculture_20260629/` |
| `validate_llm_strategic_tag_adjudication.py` | `analysis/llm_strategic_tag_adjudication_20260628/` |
| `repair_llm_adjudication_chunk.py` | `analysis/llm_strategic_tag_adjudication_20260628/` |
| `repair_llm_adjudication_chunk_0038.py` | `analysis/llm_strategic_tag_adjudication_20260628/` |
| `explore_llm_strategic_tag_elo_trends.py` | `analysis/llm_strategic_tag_elo_exploration_20260629/` |
| `analyze_homogeneous_adversary_tag_mechanism.py` | `analysis/homogeneous_adversary_tag_mechanism_20260629/` |
| `build_tag_review_game_ui.py` | Strategic-tag review UI and its source tables |
| `qualitative_judge_harness.py` | Qualitative judge packets, scores, and validation output |

## Test-Time Compute Work

| Script | Work bundle |
| --- | --- |
| `analyze_ttc_objective_shift.py` | `analysis/neurips_revision_20260504/ttc_objective_shift/` |
| `analyze_ttc_objective_shift_deltas.py` | `analysis/neurips_revision_20260504/ttc_objective_shift/` |
| `plot_ttc_compute_language_bridge.py` | `analysis/neurips_revision_20260504/ttc_objective_shift/` |
| `plot_ttc_fairness_vs_compute.py` | Retained TTC fairness plots and source tables |
| `prepare_ttc_llm_strategic_tag_adjudication.py` | `analysis/ttc_llm_strategic_tag_adjudication_20260629/` |
| `validate_ttc_llm_strategic_tag_adjudication.py` | `analysis/ttc_llm_strategic_tag_adjudication_20260629/` |
| `summarize_ttc_llm_strategic_tag_adjudication.py` | `analysis/ttc_llm_strategic_tag_adjudication_20260629/` |
| `explore_ttc_strategic_tag_trends.py` | `analysis/ttc_strategic_tag_exploration_lines_payoff_20260629/` |
| `explore_ttc_strategic_tag_intensity_trends.py` | `analysis/ttc_strategic_tag_intensity_lines_payoff_20260629/` |
| `analyze_ttc_hot_strategic_tags.py` | `analysis/ttc_hot_strategic_tags_20260629/` and its paper plot directory |
| `build_ttc_paper_style_mini_section.py` | `analysis/ttc_hot_strategic_tags_20260629/paper_style_section/` |

