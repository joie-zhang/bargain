# E35. Arena Elo snapshot sensitivity

## Result and checks

- Covers the March versus August snapshot figure and text at `sec:appendix_lmarena_snapshot_comparison` and `fig:appendix_lmarena_snapshot_comparison`.
- This is a reanalysis of the primary 1,500 runs, not a new experiment batch.
- I independently read the saved primary table, selected the Nano cohort, joined the saved 30-model snapshot table, and recomputed unweighted model-mean regressions.
- Mean absolute score change is 1.4 and maximum absolute change is 8.

| Game | Runs | March slope per 100 Elo | August slope per 100 Elo |
|---|---:|---:|---:|
| 1 | 420 | 6.753268026 | 6.810346658 |
| 2 | 540 | 6.758600708 | 6.844749676 |
| 3 | 540 | 7.400566668 | 7.453545842 |

## Verified provenance chain

- Primary-run launch, saved configurations, recoveries, and runtime are shared with E01, E02, and E03.
- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py` selects raw outcomes and produces `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv`.
- The current snapshot producer consumes only its 1,500 Nano rows and preserves model-mean payoff while replacing the Elo coordinate.
- The original August16 command ran `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_lmarena_elo_refresh.py` and recorded all six current coefficients.
- The official historical capture has 391 rows, with 30 explicitly mapped experiment models.
- The filename says August12 because that was the supplied comparison label, but provenance records August16 capture.
- The August22 paper integration used an offline function call with the saved comparison CSV, then copied the plot into Overleaf.
- The September5 offline reconstruction dynamically imports a frozen copy of the producer and verifies selected raw files before rendering.
- I checked that all 1,500 selected result paths exist today.
- Exact raw and configuration paths are enumerated in `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_12/data/raw_verification.csv`.
- This is a selection rule for those specific files, not approval to retain or delete every file under the three experiment roots.

## Code dependencies

- The producer imports NumPy, pandas, SciPy, Matplotlib, requests, and Python standard libraries.
- Its plotting functions need no HTTP request.
- Its main function does fetch live ratings and overwrites dated historical paths, so do not run main to reproduce the archived figure.
- The primary table builder imports game metrics and the active model roster.
- Package initialization also imports the game classes, base definitions, JSON repair helper, experiment configuration, data models, and analyzer.
- Concrete files and evidence are in the adjacent JSON dependency list.
- Original plotting used the external GovSim environment at `/home/jz4391/.conda/envs/GovSim`; environment preservation is shared support, not a source deletion candidate.

## History verification

- Bundled codex-search queried “lmarena March August snapshot stability 1.4” and returned original session `01a00d28-68e7-78e0-b6e0-c58f94f9f45c` as rank 1.
- I manually read lines 1441, 1575, and 1581 in `/home/jz4391/.codex/sessions/2026/08/16/rollout-2026-08-16T20-39-07-01a00d28-68e7-78e0-b6e0-c58f94f9f45c.jsonl`.
- I manually read the offline rendering and installation command at line 494 in `/home/jz4391/.codex/sessions/2026/08/22/rollout-2026-08-22T07-30-53-01a0293c-ea53-7fe2-9ba5-9e33cfe7bb14.jsonl`.
- These are locally available histories only.

## Limits and deletion decision

- No new experiments, network fetches, plotting writes, or paper edits were made.
- The numeric agreement does not certify model identity or seat assignment in every original run.
- Prior raw audits document eleven Game1 role reversals, Gemini3-pro versus3.1-Pro alias mismatch, incomplete Game1 allocations, and historical zero handling for missing failure utilities.
- Keep those diagnostic records and underlying raw files because they explain the exact reported cohort.
- Full original HTTP body was not located; the parsed capture and provenance hash must be retained.
- No deletion candidate is supported by this result alone.
- Alternate independent renderers and duplicate-looking rating tables remain unresolved until cross-workflow checks establish whether they are redundant.

