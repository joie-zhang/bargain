# E30 matched coalition replication

## Scope and result

- Current paper references are `sec:appendix_gpt54_coalition_replication`, `fig:appendix_gpt54_matched_coalitions`, and the coalition footnote in the main text.
- The source map has 26 GPT configurations for 25 unique Gemini cells.
- Failed GPT configuration 0004 is replaced by 0021.
- Five two-agent cells are excluded from coalition rates, leaving 20 eligible pairs.
- Read-only validation loaded all 50 selected result files and all 50 interaction files.
- It checked 436 formal proposals, 3,424 votes, selected allocations, final utilities, and identical preferences and game settings across all pairs.
- The historical GPT annotation map gives 9 proposing runs and 6 selected runs.
- All 37 annotated GPT proposal instances give named outsiders positive utility.
- The retained Gemini annotations give 12 proposing runs and 10 selected runs.

## Required chain

1. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_gpt54_game1_coalition_pilot.py` creates four initially selected Gemini-positive cells using the saved original preferences.
2. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/extend_gpt54_game1_matched_grid.py` adds 16 eligible cells and replacement configuration 0021.
3. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/add_gpt54_game1_n2_matched_cells.py` adds five two-agent cells.
4. `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_gpt54_matched_grid_budgeted.py` calls the random-monoculture batch runner, which uses the shared full-games runner and experiment runtime.
5. Saved overrides specify direct OpenAI, model `gpt-5.4`, high reasoning, and 65,536-token limits.
6. The source map selects exact GPT and original Gemini raw files from the two result roots.
7. GPT qualitative decisions are recorded in the original Codex session, not in a standalone automatic classifier.
8. `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/minimum_winning_coalition_audit_20260816/subagent_outputs/mono_game1.json` supplies historical Gemini classifications.
9. `/scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260831/gpt54_coalition_replication/plot_matched_coalition_rates.py` reads its adjacent four-row CSV.
10. `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_30/recreate.py` reconstructs that CSV from annotations and raw records.

## History and preservation concerns

- The bundled codex-search was run against locally available repository-linked history.
- Current-session forks dominate its broad query, so those were not treated as original provenance.
- The original session is `/home/jz4391/.codex/sessions/2026/08/30/rollout-2026-08-30T05-06-19-01a051eb-70ad-7e22-b39e-1f7d38216291.jsonl`.
- Line 149 records the budgeted continuation command.
- Line 771 retains per-run audits with proposal numbers, outsiders, utilities, votes, and qualitative evidence.
- Line 803 gives the final 9/20 and 6/20 table and explicitly supersedes the earlier 3/20 result.
- The old checked-in completion note still reports 3/20, so it is historical provenance rather than the current figure's data source.
- GPT configuration 0011 is a stated boundary case, and omitting it gives 8 proposals and 5 selections.
- Gemini configuration 0112 counts as a proposal despite `strict_plan_formalized=false`.
- Gemini configuration 0125 is excluded from selected coalitions because it received 8/10 votes, above the exact 7-vote threshold.
- GPT counts formal submissions, while the Gemini count follows these older rules.
- These are definition differences, not missing files or grounds for deletion.
- The failed configuration 0004, its direct-vote recovery script/output, and four exact proxy request JSON files are needed to explain replacement and recovery.
- All four original recovery request files still exist outside this repository.
- The saved manifest records a dirty worktree at commit `ef7451fdbbae5beb1e1d6dc2b309692cc6446328`.
- All recorded source hashes match current files except `run_strong_models_experiment.py`.
- Recovering the exact historical runner remains unresolved.

## File decisions

- The adjacent JSON report lists 254 existing concrete paths with roles and evidence.
- It includes exact selected configuration/raw files, status and staged selection records, recovery evidence, annotation sources, render inputs, and statically imported local code.
- Static imports establish a retention dependency, not proof that every optional branch ran in these experiments.
- No deletion candidate is supported by this result audit.
- Files were not classified as unnecessary merely because this result does not use them.
- No experiment, API call, figure regeneration, code edit, paper edit, or Git mutation was performed.

