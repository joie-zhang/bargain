# Paper Figure Renderers

Run these scripts from the repository root. Each script reads existing result
files. The scripts do not start experiments.

Use this command to run all saved renderers and compare their outputs with the
saved files:

```bash
python scripts/paper_figures/verify_all.py
```

The verification script uses `.venv/bin/python`, even if the active shell uses
another Python. It saves a log for each command under
`analysis/paper_figure_verification/`. It restores all declared outputs after
the checks. Thus, it does not replace the paper assets. Use `--keep-outputs`
only when you intend to replace the saved files.

| Script | Figure output |
|---|---|
| `render_figure2_large_fonts.py` | Bilateral overview (`F001`) |
| `make_combined_aligned_font_balanced.py` | Bilateral qualitative figure (`F002`) |
| `plot_ttc_game_averaged_observed_tokens.py` | Main TTC compute figure (`F003`) |
| `plot_figure3_baseline_by_competition_ewma_iteration.py` | Bilateral baseline-payoff figure (`F006`) |
| `plot_figure4_total_welfare_by_competition_ewma.py` | Bilateral welfare figure (`F007`) |
| `plot_fairshare_residual_combined.py` | Fair-share residual figure (`F008`) |
| `plot_n2_fairness_distance_three_game_curves.py` | Bilateral fairness-distance figure (`F009`) |
| `plot_n2_endpoint_fairness_style_matched_drop_game2_outlier.py` | Endpoint fairness figure (`F013`) |
| `plot_ttc_effort_adversary_baseline.py` | TTC effort figure (`F019`) |
| `plot_icml_ttc_main_figures.py` | TTC intensity figures (`F020`, `I006`) |
| `plot_n_gt_2_heterogeneous_utility_distribution.py` | Heterogeneous utility figure (`F030`) |
| `plot_random_monoculture_gini_vs_heterogeneous.py` | Multi-agent Gini source figure (`F034`) |
| `render_figure7_label_edits.py` | Final multi-agent Gini rendering (`F034`) |
| `plot_icml_homogeneous_adversary_main_panels.py` | ICML homogeneous-adversary panels (`F035`) |
| `plot_homogeneous_adversary_baseline_vs_all_gini.py` | NeurIPS homogeneous-adversary Gini panel (`N009`) |
| `plot_role_payoff_with_within_run_variance_bars.py` | NeurIPS role-payoff panel (`N010`) |

## Fixed Assets

`assets/fairshare_residual_combined_base.png` is the saved base image for
`plot_fairshare_residual_combined.py`. The original composition script was not
available. The renderer changes labels on this image.

`assets/endpoint_fairness_elo_snapshot.csv` contains the Elo values that the
saved endpoint fairness figure used. The values were recovered from the saved
model-means table and match the April 2026 canonical model roster. Use this
fixed table to reproduce the paper figure. Do not replace it with a new Elo
snapshot.

`assets/game2_utility_roster_snapshot.csv` contains the model labels and Elo
values for the saved Game 2 utility table. Use it with this command:

```bash
python scripts/export_game2_batch_pngs.py \
  --results-dir experiments/results/diplomacy_20260405_082215 \
  --roster-csv scripts/paper_figures/assets/game2_utility_roster_snapshot.csv \
  --long-csv analysis/neurips_revision_20260504/inputs/game2_utility_vs_elo_adversary_long.csv
```

## Broad Analyses

The following scripts stay in the parent `scripts/` directory because they
produce full analysis bundles:

- `analyze_appendix_llama33_baseline_500.py`
- `analyze_n2_baseline_comparison.py`
- `analyze_n2_plus_multiagent_comparison.py`
- `analyze_neurips_revision_stats.py`

See `docs/reproducibility/paper_figure_manifest.md` for the complete file map.
