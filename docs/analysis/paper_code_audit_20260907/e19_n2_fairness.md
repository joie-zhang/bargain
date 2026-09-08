# E19 two-player fairness dependency audit

## Scope and claims

- Main top-left panel shows mean undiscounted adversary utility minus benchmark utility versus Elo for 30 models in each game.
- Appendix shows overall unsigned benchmark distance, distance within each competition setting, and both role residuals.
- All 15 historical competition-stratified distance fits are negative.
- Appendix overall distance slopes per 100 Elo are -3.961575, -6.430846 and -0.128625 for Games 1, 2 and 3.
- Main fitted residual crossings recomputed from retained means are 1394.1617, 1460.6873 and 1426.3158.
- Endpoint normalized panels are assigned to E20; multi-agent panels to E21.
- Original experiment launch and runtime provenance is shared with E01-E03.

## Evidence checked now

- Read current paper source at 3_approach.tex:84, 4_analysis.tex:21 and appendix.tex:422-432 under /scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template.
- Independently checked all 3,000 primary raw/config manifest entries: all exist and all hashes match the September 5 raw audit.
- Current ICLR main PNG and all three appendix PNGs are byte-identical to the audited reference assets.
- Read the analyzer's original index selector, role mapper, NBS algorithms and Lindahl calculations.
- Read retained renderer, replay scripts and original August 8 exact solver.
- Verified all six missing-preference raw results and original setup transcript hashes.
- Recomputed the three main zero crossings directly from the retained numerical table.
- No experiments, solvers or renderers were run; no code or paper changed.

## Provenance chain

1. Shared primary experiment launch/runtime produces original configuration indexes, per-run configurations, results and interaction transcripts in the three E01-E03 roots.
2. /scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py loads the original indexes and resolves original run output paths, then filters to the primary two-turn Game 1 cohort and active Elo roster.
3. Games 1 and 2 use undiscounted utility distance to NBS, while Game 3 uses contribution-matrix distance to benefit-proportional cost shares divided by funded cost.
4. Appendix renderer reads the retained primary metric CSV and masks Game 3 no-project role residuals.
5. Main reconstruction replaces Game 2 benchmark with the August 8 exact corner/edge solver, aggregates adversary residuals by model, and uses recovered combined-layout code.
6. Dated source under overleaf/analysis and analysis/figure_recreation is therefore required provenance and reproduction code, not disposable paper formatting.

## Important distinctions and unresolved items

- Original launch, runtime version and replaced run records are shared with E01-E03; this result audit verifies downstream selection and numerical pipeline, not an independent historical launch reconstruction.
- Six Game 1 failed runs config_644,648,720,724,742,895 have empty preferences and are falsely assigned zero benchmark residual/distance by retained analyzer; all six original setup transcripts exist with unchanged diagnostic hashes and must stay.
- Appendix excludes 80 Game 3 no-funded outcomes from unsigned distance and both role curves; main signed residual means retain their zeros. These numerical inputs are not interchangeable.
- Main Game 2 solver is exact corner/edge enumeration; appendix Game 2 solver is five-start L-BFGS-B. Similar filenames do not establish redundant code.
- Current main caption states Game 3 crossing 1454, but independently fitted retained current means give 1426.3158; source text 1410-1461 also differs from Game 1 1394.1617.
- Current ICLR directory has main PNG but not fairshare_residual_combined_summary.csv; necessary numerical table remains under ICML directory.
- Bundled codex-search broad query returned current conversation forks; verified original May request and August production source directly, rather than accepting those forks as origin.
- Did not rerun optimizers or renderers during this read-only audit; verified code, current asset identity, all primary selected raw/config hashes, six transcript hashes and retained mean fits. Original May full rollout unavailable locally.

## History

- /home/jz4391/.codex/history.jsonl:1556 session 019df5c4-2840-7123-96fa-8f31e40d7461 original two-player fairness analysis request.
- /home/jz4391/.codex/sessions/2026/08/02/rollout-2026-08-02T00-23-38-019fc0b6-9452-7cc2-8e8a-d665df4de3a1.jsonl:1200 user three-row layout; :1237 renderer creation and masking.
- /home/jz4391/.codex/sessions/2026/08/08/rollout-2026-08-08T23-35-58-019fe497-72e6-7973-89c5-197c6005ce86.jsonl:624 exact Game 2 solver replacement.

## Needed files

The JSON companion gives 32 concrete files or explicitly delimited directory selections with evidence.
The raw directory selections are not a declaration that every file below those roots is required.
All six setup transcripts named in /scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_18/empty_preference_diagnostic.json are required to recover the missing preference tables.

## Deletion candidates

None established.
Neither a historical script name nor a location outside the canonical scripts folder is evidence that a file is unused.

