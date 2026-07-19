# Visualization Guide

## Scope

This guide identifies the main figure scripts. Run all commands from the repository root.

Many scripts use fixed result roots or output paths. Read the constants and `--help` output before you run a script.

## Paper Figures

These scripts make the main bilateral and fairness figures:

```bash
python scripts/plot_gpt5_nano_baseline_vs_elo_all_games.py
python scripts/plot_exploitation_vs_elo.py
python scripts/plot_nbs_decomposition.py
python scripts/analyze_nash_lindahl_fairness.py
python scripts/analyze_neurips_revision_stats.py
```

These scripts make the paper comparison reports and plots:

```bash
python scripts/analyze_n2_baseline_comparison.py
python scripts/analyze_n2_plus_multiagent_comparison.py
python scripts/paper_figures/verify_all.py
```

Temporary qualitative and exploratory analyses are in
`scripts/retained_analysis/`. Read its README before you run those scripts.

Inspect these output directories:

- `analysis/neurips_revision_20260504/`
- `analysis/nash_lindahl_fairness_20260505/`
- `analysis/full_games123_*/`

Use the manifests in `docs/reproducibility/` to find the producer and input for
each active paper graphic.

## Llama Baseline Figures

Use the dedicated analysis script for the 500-run Llama replication:

```bash
python scripts/analyze_appendix_llama33_baseline_500.py
```

Read `docs/appendix_llama33_baseline_experiment_spec_2026_05.md` before you change its inputs.

## TTC Figures

Use `scripts/analyze_neurips_revision_stats.py`,
`scripts/paper_figures/plot_icml_ttc_main_figures.py`, and the other current TTC
renderers under `scripts/paper_figures/`. The January and February TTC plotters were removed because they used
superseded experiment formats.

## Output Checks

Do these checks after figure generation:

1. Confirm that the script found the intended result root.
2. Confirm that the row count agrees with the experiment manifest.
3. Confirm that model order and game cells have the expected counts.
4. Open each figure and examine labels, legends, and missing cells.
5. Record the script, input root, and output directory in the analysis report.

Do not replace a paper figure until these checks are complete.
