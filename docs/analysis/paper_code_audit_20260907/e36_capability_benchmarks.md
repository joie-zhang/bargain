# E36: Alternative capability measures

- Scope: `sec:appendix_alternative_capability_measures` and `fig:appendix_alternative_capability_measures`, currently Figure 15.
- Claims covered: 11 capability measures; coverage of 24 to 30 models; 33 positive slopes; 30 BH-adjusted significant associations; three non-significant CritPt fits; mean payoff and SEM; AA estimate markers and model configuration matches.
- No code, data, paper, configuration, or Git changes were made.

## Verified chain

1. The experiment inputs are the same 1,500 primary GPT-5-nano runs audited by E01-E03, not a new experiment batch.
2. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py` builds the combined baseline cohort using the active roster, submitted model order, and game utility helpers.
3. `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv` selects the exact raw files through `baseline_key == gpt5_nano` and `result_path`.
4. This audit read all 1,500 selected raw result JSON files and all 1,500 submitted configs, with zero differences between selected raw final utilities and the cohort payoff values.
5. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_capability_metric_robustness.py` joins model means to frozen capability sources and computes unweighted linear fits and the 33-test BH correction.
6. `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_13/recreate.py` reconstructs those tables from raw files and the original saved AA snapshot.
7. `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_13_label_review_20260906/render.py` reads the audited score and mean-payoff tables and dynamically imports `producer_snapshot.py` from that dated audit directory.
8. The current ICLR PDF asset has SHA-256 `e6dd8f1585c834cb68aca398ec59897ba0d1e2f65b5f5c0f616e45460ccfc33c`, exactly matching the September 6 installation record.

## Independent numerical checks

- Reparsed the saved AA HTML and rebuilt the score/model joins without downloading replacement scores.
- Reproduced all 298 retained scores, with maximum CSV floating-point difference below `8e-15`.
- Recomputed all 33 slopes as positive and 30 BH-adjusted p-values below 0.05.
- Rebuilt model means from the current primary cohort, with maximum difference from the audited means below `2e-14`.
- Arena covers 30 models; AA Index, GPQA, HLE, and SciCode cover 29; IFBench covers 27; tau2 and AA-LCR cover 26; Terminal-Bench Hard covers 25; Omniscience and CritPt cover 24.
- The AA Index includes 22 publisher-estimated values among 29 matches.
- Five model matches use nearby configurations: o3-mini-high, Claude Opus 4.5 thinking 32k, GPT-5.2 chat, GPT-5.4 high, and Claude Opus 4.6 thinking.
- The original producer and the imported frozen producer have identical SHA-256 `041ac6cf5f4e5d0664331d85aff3a14f67a618eeaf34d8603af22770614df37b`.

## History evidence

- Used the bundled codex-search script with `capability_benchmarks_over20`; it ranked session `01a00d28-68e7-78e0-b6e0-c58f94f9f45c` first among 2,213 locally linked records.
- The full session is `/home/jz4391/.codex/sessions/2026/08/16/rollout-2026-08-16T20-39-07-01a00d28-68e7-78e0-b6e0-c58f94f9f45c.jsonl`.
- Manually inspected the original request at line 146 and snapshot download/hash evidence at lines 425-427.
- Manually inspected strict coverage filtering at line 1193 and paper insertion at line 1374.
- The current AA source hash matches the original recorded download hash `2abc335063b0b136c4eee9eb76f44179e12540b1fc9b2db4daf4037fcf75c66f`.
- Local history does not establish that all historical runtime versions remain available.

## Files that must stay

- The paired JSON report lists concrete scripts, input tables, saved benchmark sources, provenance files, and raw-data selection rules.
- Preserve the dated Figure 13 directory inputs even though the current paper numbers this figure as 15.
- Preserve all eight benchmark source files used by the full original producer, not only the AA HTML used in the final 11 panels.
  - Its `main()` executes the MMLU-Pro, LiveBench, and HELM loaders before rendering.
  - Its candidate-coverage code reads HLE, GPQA, and Open LLM snapshots.
  - Its source manifest hashes all eight snapshots.
- Preserve the primary-cohort configs, raw results, and transcript provenance identified by the explicit CSV selectors.
- Do not delete the frozen producer merely because it currently matches the live script; the current renderer imports the frozen path directly.
- The original producer still writes an ICML asset path, while the September 6 renderer supplies the current ICLR asset.

## Limits and deletion decision

- No deletion candidate is supported by this result audit.
- Eleven known Game 1 runtime-role conflicts affect the shared primary cohort; zero differences from saved utilities do not establish correct model attribution.
- The component-score loader sets `estimated=False` directly rather than reading per-component completion status.
- The five nearby-configuration matches are declared proxies, not exact experiment-setting matches.
- Original experiment launch/recovery dependencies are covered by E01-E03; this audit did not duplicate their entire historical runtime investigation.
- External numerical libraries are NumPy, pandas, SciPy, and Matplotlib; the archived environment record describes the previous recreation versions.
- Directory entries in the JSON are explicit selection rules, not proof that every file in each directory is needed.
