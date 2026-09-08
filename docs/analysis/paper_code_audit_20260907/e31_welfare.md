# E31: Two-player social welfare

- Scope is the social-welfare appendix, Figure 17, both figure labels, all 15 competition cells, and the quoted captured ratios.
- No experiment, source file, paper asset, or data was changed.
- The exact selected raw outcomes were independently read and recalculated using inspected functions in the retained Figure 17 reconstruction script.
- All 1,500 discounted welfare outcomes match the source CSV within 5.69e-14.
- All 1,494 available optima match within 8.53e-14.
- Current paper PNG and raw-recreated PNG have identical SHA-256 e396cc346da4de4b8a145eadbfc84256f42973d140f6571a7e8ec5a82d2d1a49.

## Exact percentages

| Game | Competition | Mean captured ratio (%) | Eligible runs |
|---|---:|---:|---:|
| 1 | 0 | 94.1597 | 58 |
| 1 | .25 | 93.6205 | 60 |
| 1 | .5 | 90.9945 | 59 |
| 1 | .75 | 91.0902 | 59 |
| 1 | .9 | 85.6289 | 60 |
| 1 | .95 | 89.5747 | 59 |
| 1 | 1 | 91.5204 | 59 |
| 2 | 0 | 93.8746 | 300 |
| 2 | .25 | 84.9475 | 60 |
| 2 | .5 | 87.1981 | 120 |
| 2 | 1 | 88.6601 | 60 |
| 3 | 0 | 68.9213 | 299 |
| 3 | .2 | 68.8967 | 60 |
| 3 | .4 | 64.7545 | 120 |
| 3 | .8 | 18.3108 | 60 |

- Ratios are the mean of each run's discounted total payoff divided by its own undiscounted optimum.
- The existing CSV column optimality_ratio instead uses undiscounted actual payoff, so it must not replace this calculation.
- Six failed Game 1 runs have no saved preferences and cannot supply optima.
- One Game 3 run has an actual zero optimum.
- These seven runs are excluded from ratios, but all 1,500 runs remain in the welfare curves.

## Provenance and dependency decisions

- Primary launches and historical reruns are audited by E01-E03.
- The combined analysis reads dated experiment indexes, selected launch JSON files, and matching raw result JSON files.
- Its primary filter retains two-turn Game 1 and the active 30-model roster.
- Current rendering reads the primary CSV, averages by Elo within each competition setting, and applies EWMA with alpha .24.
- Dashed lines are maximum observed undiscounted optima, not average optima or efficiency ratios.
- Game 1 optimum is itemwise maximal valuation.
- Game 2 optimum selects the weighted-median ideal position for each issue.
- Game 3 optimum enumerates budget-feasible project subsets and subtracts project costs from total valuations.
- Direct analysis imports also execute the game_environments package initializer and its game classes, which must be retained even for analysis-only use.
- Exact needed paths and bounded raw-data selectors are recorded in the companion JSON.
- Broad raw trees are not certified in full by this audit.

## History and limits

- The bundled codex-search was run against local history with welfare/captured-ratio anchors.
- Highest search matches are current inherited conversation forks and do not establish original execution.
- The original August 22 session was manually checked at line 214 and explicitly requests primary-1,500 welfare filtering while retaining the appendix plot.
- The retained Figure 17 report cites patch and execution at lines 398 and 445 of that session.
- Original May execution history is unavailable locally.
- A dedicated persisted producer for the recent percentage paragraph was not found.
- The historical source renderer still points to the ICML destination; current ICLR bytes match the recreated figure.
- No file is a supported deletion candidate from this result alone.
- Archived failures and overwritten-run records must remain until the shared primary provenance audit settles their role.
