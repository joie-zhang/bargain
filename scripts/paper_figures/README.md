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
| `plot_game1_discussion_turn_ablation.py` | Game 1 discussion-turn ablation |
| `make_combined_aligned_font_balanced.py` | Bilateral qualitative figure (`F002`) |
| `plot_ttc_game_averaged_observed_tokens.py` | Main TTC compute figure (`F003`) |
| `plot_figure3_baseline_by_competition_ewma_iteration.py` | Bilateral baseline-payoff figure (`F006`) |
| `plot_figure4_total_welfare_by_competition_ewma.py` | Bilateral welfare figure (`F007`) |
| `plot_fairshare_residual_combined.py` | Fair-share residual figure (`F008`) |
| `plot_n2_fairness_distance_three_game_curves.py` | Bilateral fairness-distance figure (`F009`) |
| `plot_n2_endpoint_fairness_style_matched_drop_game2_outlier.py` | Endpoint fairness figure (`F013`) |
| `plot_ttc_effort_adversary_baseline.py` | TTC effort figure (`F019`) |
| `plot_icml_ttc_main_figures.py` | TTC intensity figures (`F020`, `I006`) |
| `plot_icml_heterogeneous_game1_payoff.py` | ICML heterogeneous Game 1 payoff (`I007`) |
| `plot_n_gt_2_heterogeneous_utility_distribution.py` | Heterogeneous utility figure (`F030`) |
| `plot_random_monoculture_gini_vs_heterogeneous.py` | Multi-agent Gini source figure (`F034`) |
| `render_figure7_label_edits.py` | Final multi-agent Gini rendering (`F034`) |
| `plot_icml_homogeneous_adversary_main_panels.py` | ICML homogeneous-adversary panels (`F035`) |
| `plot_homogeneous_adversary_baseline_vs_all_gini.py` | NeurIPS homogeneous-adversary Gini panel (`N009`) |
| `plot_role_payoff_with_within_run_variance_bars.py` | NeurIPS role-payoff panel (`N010`) |
| `plot_figure9_absolute_payoffs.py` | Matched team payoffs and continuous percentage of maximum payoff |
| `plot_figure9_actual_optimality.py` | Observed frequency of maximum Nano-team payoff in both conditions |

## Matched team payoff and optimality figures

These two scripts recompute metrics from the 100 saved matched pairs, including
the original results, configurations, interactions, control lineage, and batch
manifest.
They verify utilities against allocations and preferences, check control hashes,
and record hashes for all inputs.
Neither script needs an old analysis CSV, an earlier plotting script, a paper
PDF, or an Overleaf checkout.

From an activated environment with the analysis dependencies installed, run:

```bash
TEAM_ROOT="$BARGAIN_ROOT/experiments/results/game1_gpt54_binding_team_v3_20260816_093310"
python "$BARGAIN_ROOT/scripts/paper_figures/plot_figure9_absolute_payoffs.py" \
  --results-root "$TEAM_ROOT" --data-root "$BARGAIN_ROOT/experiments/results" \
  --maximum-original-style \
  --output-dir "$BARGAIN_ROOT/analysis/figure9_payoff_recomputed"
python "$BARGAIN_ROOT/scripts/paper_figures/plot_figure9_actual_optimality.py" \
  --results-root "$TEAM_ROOT" --data-root "$BARGAIN_ROOT/experiments/results" \
  --output-dir "$BARGAIN_ROOT/analysis/figure9_optimality_recomputed"
```

Set `BARGAIN_ROOT` to the absolute checkout path as in the main README.
Use a new output directory for each analysis.
The saved configurations also reference historical control batches; all these
inputs must be present, with the released directory structure retained.
`--data-root` explicitly relocates the part of each saved path after
`experiments/results/` to that directory without changing the saved configuration.
Without this option, the scripts use the paths in the saved configurations.
Missing inputs or failed checks stop the analysis; no saved plot or table is
substituted.

The two metrics answer different questions:

- Payoff percentage measures how much of the maximum each outcome attains.
- Exact optimality counts runs whose discounted team payoff equals the maximum
  round-one team payoff, within an absolute tolerance of `1e-8`.

Both optimality curves use observed outcomes.
The older optimality renderer drew the uncoordinated curve at zero, but the
saved cohort attains the maximum in 4 of 20 runs at each group size (20%).
The coordinated counts are 4, 19, 11, 10, and 9 of 20 for group sizes 2, 4, 6, 8,
and 10.
The current paper's continuous payoff figure is a different metric and is not
replaced by either command.

The scripts write PDF and PNG plots, run-level CSVs, summaries, captions, and
provenance JSON files to the selected analysis directory.
They are not yet included in the older `verify_all.py` manifest.

## Canonical Inputs

`plot_fairshare_residual_combined.py` rebuilds the figure from the canonical
experiment manifest, the primary bilateral table, and the fairness cache. It
checks the complete 1,500-run bilateral inventory and the complete 2,730-run
multi-agent inventory. It rejects Phi rows and the one-turn Game 1 protocol.

`plot_n2_endpoint_fairness_style_matched_drop_game2_outlier.py` loads the
canonical bilateral raw results through `analyze_n2_baseline_comparison.py`.
It checks for 1,500 primary runs, no Phi rows, and only two-turn Game 1 runs.
The old snapshot files in `assets/` are not inputs to these two renderers.

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
