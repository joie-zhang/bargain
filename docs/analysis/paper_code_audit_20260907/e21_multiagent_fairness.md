# E21 multi-agent fairness dependency audit

## Finding

- No safe deletion candidate was established for this result.
- Dated review code is part of the active figure chain and must remain.
- All 2,900 selected raw result files were read and matched the prior audit SHA256 values.
- All 2,900 selected saved configurations and 2,900 interaction records exist.
- Current main and appendix assets match the frozen references exactly by SHA256.
- Only report files were added.

## Scope and claims

- Current source references are `fig:fairshare_headline` bottom left and `fig:appendix_multiagent_fairshare_full`.
- Main text reports increasing fair-share residuals with Elo in heterogeneous and homogeneous-adversary groups.
- Appendix adds baseline-agent residuals in homogeneous-adversary groups and within-run mean residuals in homogeneous groups.
- The homogeneous Game 2 o3-mini result is included, not filtered as a failure.
- These plots pool group sizes, including n=2, rather than restricting to n>2.

## Verified calculation and selection

- Manifest selection contains 1,300 heterogeneous, 1,300 homogeneous-adversary, and 300 homogeneous runs.
- Each of the heterogeneous and homogeneous-adversary families contains 500 Game 1, 400 Game 2, and 400 Game 3 runs.
- Homogeneous selection contains 100 runs per game.
- The appendix source table has 10,700 observations: 7,800 heterogeneous agent appearances, 1,300 adversary appearances, 1,300 within-run baseline means, and 300 within-run homogeneous means.
- Heterogeneous summaries group by recorded Elo, and homogeneous summaries group by game and Elo.
- Error bars are sample standard deviation divided by the square root of the number of summarized observations.
- Games 1 and 2 use undiscounted realized utility minus the NBS utility.
- Game 1 uses exact enumeration at n=2 and seeded local search for larger n.
- Game 2 uses five-start L-BFGS-B at n=2 and SLSQP with coordinate refinement for larger n.
- Game 3 uses utility minus valuation-proportional cost-sharing utility on the recorded funded set.
- Game 3 failures therefore have zero actual utility and zero conditional funded-set residual.
- Raw checks confirmed 20 `o3-mini-high` Game 2 runs, including 17 failures with zero recorded utility.
- Their retained mean residual is -74.85477485339837.

## Producer chain

1. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py generates and dispatches heterogeneous and homogeneous-adversary jobs.
2. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py imports that dispatcher for homogeneous runs.
3. The dispatcher invokes /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py, which imports the experiment, provider configuration, agent and phase runtime.
4. Saved run selection is defined by /scratch/gpfs/DANQIC/jz4391/bargain/docs/reproducibility/paper_experiment_data_manifest.csv.
5. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_nash_lindahl_fairness.py computes raw utilities and benchmark records.
6. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_fairshare_residual_combined.py reads a historical fairness cache and recomputes missing records.
7. /scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260830/homogeneous_redesign/render_figure3_proposal.py selects the replacement homogeneous cohort and creates the source table.
8. /scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260831/figure3_two_curve_and_appendix/render_appendix_fairshare_full.py renders the current appendix figure.
9. The main compositor was inline historical code and is retained as /scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_03/scripts/render_combined.py.

## Files that may look disposable but are needed

- The dated review producers and their CSV inputs are direct dependencies, not unused drafts.
- /scratch/gpfs/DANQIC/jz4391/bargain/analysis/nash_lindahl_fairness_20260505/agent_metrics.csv is read by the current loader.
- The Figure 23 frozen solver and reconstruction scripts preserve the exact benchmark calculation.
- Five original Game 1 setup transcripts supply preference tables missing from terminal result payloads.
- The 25 excluded Claude Haiku homogeneous result files remain selection inputs because the original producer reads all 325 results before excluding them.
- The prior audit's 72 synthetic-action cases remain necessary to explain the selected data.
- The current main loader's 130 all-Nano controls are historical loader dependencies even though the displayed lower-left panel omits them.
- Do not delete broad experiment roots or dated review directories based on this result alone.

## Local history verification

- The codex-search skill was read and its bundled search was run for this repository.
- Broad search returned recent forked caption conversations, so those were not accepted as original producer evidence.
- Original August 30 session lines 312 and 997 create the cohort builder and appendix renderer.
- Line 1091 adds the heterogeneous panel.
- Line 1120 is the user's request to add the appendix section and figure.
- Line 1142 copies the rendered PDF into the paper.
- Original August 9 session line 162 creates the larger-group Game 2 correction.
- Line 1352 reports installation of corrected records and figure output.
- Verified source paths are listed in the companion JSON history array.

## Limits

- Current launch/runtime code is needed for reruns, but exact historical executable versions and provider routes require cross-reference to E10-E18 producer audits; not proven by current imports.
- Game 2 NBS is solver dependent; this audit did not rerun numerical optimization. Kept frozen algorithm and cached values preserve exact provenance.
- Prior raw audit found 72 selected runs with synthetic action markers. This audit confirmed all 2900 raw file hashes unchanged, preserving that evidence; no affected file is a deletion candidate.
- Original larger-group Game 2 correction history points to analysis/issue_030_corrected_nbs_20260809/recompute_corrected_nbs.py, absent at that location now. History and installed analyzer remain.
- Main historical loader expects 130 all-Nano controls although current lower-left panel only displays 2600 hetero/homadv runs. Protect loader/cache dependencies until migrated deliberately; these controls may also support other results.
- Direct and verified import dependencies listed, not a full transitive production runtime closure; shared protocol audit must cover provider clients, prompts, phases and utilities.
- Source/cache/figure deduplication requires selecting and updating a canonical workflow; no duplicate classified safe simply because bytes or formulas overlap.

## Inventory

- The companion JSON lists 34 specific files and explicitly selected data-root entries.
- Data-root entries name exact manifest selection rules and do not approve retaining every file in each tree.
- Concrete selected raw paths and SHA256 values are in the three Figure 23 input manifests listed in the JSON.
- Saved configuration and interaction paths are joined through the paper experiment manifest.
- Runtime dependencies shared with other results require the common runtime audit before any cleanup decision.

