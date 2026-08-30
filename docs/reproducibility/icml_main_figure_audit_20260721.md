# ICML Main Figure Audit

Date: 2026-07-21

## Scope

This audit covers each active main-text figure in
`overleaf/icml_aiwild_template/icml_aiwild_2026.tex`.

The audit does not change paper prose. It checks the figure producer, the input
data, the active image, and the visual result.

## Result

All eight generated main-text figures now have an identified producer. The
hero figure is a manual Google Drawings asset and does not have a producer in
this repository.

The generated figures use the intended data scope:

- No main-text figure includes Phi-3 Mini.
- The primary bilateral plots use only the two-turn Game 1 protocol.
- The qualitative bilateral plot includes both Game 1 protocol arms because it
  describes the complete accepted qualitative corpus. Its provenance records
  the two arms separately.
- The multi-agent plots use the exact result paths in the canonical experiment
  manifest.

## Figure Inventory

| ID | TeX location | Active image | Producer | Canonical data check | Visual check | SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| I001 | `1_intro.tex:26` | `icml_aiwild_template/graphics/Hero_Figure_1.png` | Manual Google Drawings asset | Not data-driven | Pass | `3fbc8ff7bc486235ff8c495208624090bf879cea8355773ebeec19f060dbd88e` |
| I002 | `4_analysis.tex:11` | `graphics/n2_gpt5_nano/bilateral_overview_combined.png` | `scripts/paper_figures/render_figure2_large_fonts.py` | Exact 1,500 primary bilateral runs; 420/540/540 by game; 30 models; no Phi; Game 1 is two-turn only | Pass | `58ad787139a56f7df99a37b6805e9fbcbc98474f9b3aaf949b0d52774bb4f926` |
| I003 | `4_analysis.tex:33` | `graphics/n2_gpt5_nano/fairshare_residual_combined.png` | `scripts/paper_figures/plot_fairshare_residual_combined.py` | Exact 1,500 primary bilateral runs and 2,730 canonical multi-agent runs; no Phi; Game 1 is two-turn only | Pass; caption proposal is in the count-and-stat review document | `618dc1bb9b057eb767147d803b797135869c1cb3ce11f50e0d4ff92861689eb8` |
| I004 | `4_analysis.tex:53` | `graphics/qualitative_n2/n2_qualitative_combined_smooth5_outlier_removed_aligned.png` | `scripts/paper_figures/export_n2_qualitative_dedup.py`, then `scripts/paper_figures/make_combined_aligned_font_balanced.py` | 1,920 accepted bilateral rollouts; 1,891 payoff-valid rollouts; 30 models; no Phi; both Game 1 protocol arms are explicit | Pass | `857a69ac968efcbd8f245bdc31f4a71d895366c03ea657f45360801d28297d0c` |
| I005 | `4_analysis.tex:65` | `graphics/ttc_game_averaged_target_payoff_vs_compute.png` | `scripts/paper_figures/plot_ttc_game_averaged_observed_tokens.py`, then `scripts/paper_figures/plot_icml_ttc_main_figures.py` | All 216 monitoring rows match their raw result JSON values; the displayed grid has all expected cells | Pass; large bars are the stated standard errors | `a11251a4f23d2b2b02aad97e5947c51e2dd0d3d507003c64c9b0ae594dc3359d` |
| I006 | `4_analysis.tex:72` | `graphics/qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.png` | `scripts/paper_figures/plot_ttc_group_intensity_combined.py`, then `scripts/paper_figures/plot_icml_ttc_main_figures.py` | 144 displayed GPT-5 and Gemini rollouts; 18 rollouts per family and effort cell; no displayed result has an identifier mismatch | Pass | `323b98cd5e2e63c230b0e9738348e3cdf8837599041ffd9bff1cc04627ea7d4f` |
| I007 | `4_analysis.tex:79` | `graphics/qualitative_ttc/heterogenous_game1_payoff_singlecolumn.png` | `scripts/paper_figures/plot_icml_heterogeneous_game1_payoff.py` | The source has the exact 1,300 canonical heterogeneous result paths; five group sizes; no Phi | Pass after removal of cluttered model labels and error bars | `9fcdc16397d5accb01b0beaf03172d55e35631693cfbec4355e88311de587c2e` |
| I008 | `4_analysis.tex:106` | `graphics/multiagent_gini/heterogeneous_vs_homogeneous_gini_bars.png` | `scripts/paper_figures/plot_random_monoculture_gini_vs_heterogeneous.py`, then `scripts/paper_figures/render_figure7_label_edits.py` | Exact 1,300 heterogeneous runs and 325 random-monoculture runs; no Phi; includes the recovered `config_0262` result | Pass; final rendering has no model labels | `fab844002c9b320f83e9470575e7dd0a69171936a17e6a679aa7a06cd3441a12` |
| I009 | `4_analysis.tex:115` | `graphics/n_gt_2_report/homogeneous_adversary_gini_and_role_payoff.png` | `scripts/paper_figures/plot_homogeneous_adversary_baseline_vs_all_gini.py`, `scripts/paper_figures/plot_role_payoff_with_within_run_variance_bars.py`, then `scripts/paper_figures/plot_icml_homogeneous_adversary_main_panels.py` | Exact 1,300 canonical homogeneous-adversary runs; all four Elo buckets are present; no Phi | Pass after margin correction | `1c5ae7311eaa2bd7bdeb294b406a9cb99177138cced9b323dca06c13f87964dc` |

## Provenance Files

The figure producers also write these machine-readable checks:

- `overleaf/icml_aiwild_template/graphics/n2_gpt5_nano/bilateral_overview_combined_provenance.json`
- `overleaf/icml_aiwild_template/graphics/n2_gpt5_nano/fairshare_residual_combined_provenance.json`
- `overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_qualitative_dedup_provenance.json`
- `overleaf/icml_aiwild_template/graphics/qualitative_ttc/ttc_main_figures_provenance.json`

The canonical experiment inventory is:

`docs/reproducibility/paper_experiment_data_manifest.csv`

The active figure inventory is:

`docs/reproducibility/icml_figure_manifest.csv`

## Compiled Paper Check

The authoritative source compiled successfully without a manuscript edit.
Run this command from `overleaf/icml_aiwild_template/`:

```bash
TEXINPUTS=.:..: pdflatex -interaction=nonstopmode -halt-on-error icml_aiwild_2026.tex
```

Run the command twice after a figure update.

The compiled file is:

`overleaf/icml_aiwild_template/icml_aiwild_2026.pdf`

Its SHA-256 hash is:

```text
157209908a65ce6d484f1c37e75985d5448823651c25323be67f4ea964800477
```

The file has 56 pages. The main paper ends on page 9. A visual check of pages
3 through 9 found no clipped, blank, overlapping, or missing main-text figure.

The build log has three existing undefined references to
`sec:n2_qualitative`. It also has two appendix overfull-box warnings of less
than 1.85 pt. These warnings are not caused by a figure asset. This audit does
not edit them.

## Manifest Validation

This command passes:

```bash
python scripts/validate_paper_figure_manifest.py
```

The validator checks 39 NExT assets, 40 ICML assets, and 40 NeurIPS assets.
It reports four known appendix composition warnings: `F005`, `F025`, `F026`,
and `F027`. No active asset has a hash error.
