# E17 homogeneous Game 2 dependency audit

- Scope is the Game 2 contribution to `fig:monoculture_payoff_variance`, plus the 17 failed o3-mini agreements used to explain its low variance and fair-share residual.
- No experiment, code, paper, data, or Git changes were made.
- No deletion candidate is supported by this result audit.

## Verified numerical result

- Read all 100 Game 2 raw outcome JSON files, configuration IDs 126 through 225, and checked that their saved configuration and full transcript files exist.
- The design has five models, five agent counts, and four preference cells, with one run per cell.
- Population variance is computed within each final utility vector with `ddof=0`.
- Each model point averages 20 run variances; SEM uses sample standard deviation divided by the square root of 20.

| Model | Mean variance | SEM | No agreement |
|---|---:|---:|---:|
| amazon-nova-pro-v1.0 | 61.439547 | 25.171985 | 0/20 |
| gpt-4o-2024-05-13 | 38.304114 | 10.571652 | 0/20 |
| o3-mini-high | 0.337178 | 0.277980 | 17/20 |
| gpt-5.2-chat-latest-20260210 | 14.124479 | 4.283049 | 0/20 |
| claude-opus-4-6 | 34.247070 | 20.266052 | 0/20 |

- The Game 2 mean across these 100 runs is 29.690477.
- All 17 o3-mini failures end at round 10 with zero payoff for every agent.
- Only o3-mini configuration IDs 166, 168, and 169 reach agreement, in rounds 4, 1, and 5.
- These zero outcomes are part of the reported result, not removable failed-run clutter.
- The plotted heterogeneous reference is pooled across all three games, not a Game 2 matched reference.
- Low within-run variance can therefore reflect failure to agree; it does not by itself show a successful equal division.

## Verified execution and analysis chain

- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py` selects models from the historical 24-model CSV, excludes unavailable Sonnet 4, uses seed 20260628, and assigns one sampled model from each Elo band to each game.
- The saved `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/manifest.json` and `RUN_NOTES.md` preserve the original 325-run design.
- Saved Slurm launch records invoke the random-monoculture launcher with task IDs, load the cluster proxy module, and set OpenRouter transport to the file proxy.
- The launcher adapts readable configuration IDs, then calls `full.run_config`; `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1542` constructs the subprocess command for `/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py`.
- Shared execution uses `StrongModelsExperiment`, the Game 2 environment, phase handlers, prompt generation, agent factories, provider clients, context compaction, JSON repair, and result writers.
- The JSON report records 38 current local import dependencies from the launch and runner entry points; this import closure supports current execution but does not establish the original source revision.
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260816/recreate.py:224` rebuilds homogeneous run variances from raw final utilities.
- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_appendix_gemini_coalitions.py:131` reads the resulting aggregate and model CSVs and renders the figure.
- The current paper embeds `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/graphics/appendix_multiagent/payoff_variance_homogeneous_runs.pdf`.
- This audit recalculated statistics but did not render or replace the paper image.

## Original conversation checks

- Ran the bundled codex-search script with the query `homogeneous Game 2 o3-mini variance 17 20`.
- Its top matches were recent forks repeating the paper discussion, so they were not used as original launch evidence.
- Manually read `/home/jz4391/.codex/sessions/2026/06/29/rollout-2026-06-29T23-56-35-019f16ab-f410-73e2-82ac-c3414edc1625.jsonl:6095` and line 6309, which contain the original homogeneous-versus-heterogeneous request and the request to use payoff variance.
- Manually read `/home/jz4391/.codex/sessions/2026/08/22/rollout-2026-08-22T05-43-57-01a028db-0561-7ac0-91aa-663944f564ed.jsonl:309`, which makes the 300-run cohort canonical while retaining the 325-run comparison separately.
- Only locally stored history was searched.

## Cleanup boundaries

- The companion JSON lists the 100 raw results, 100 saved configurations, and 100 full transcript files individually.
- Transcript existence is checked for all 100 runs; transcript contents were not replayed in full.
- Preserve remaining per-agent logs, retry records, status files, and submission records until their overwrite history is resolved.
- The current raw-data builder expects all 325 original homogeneous results before cohort filtering, so the 25 excluded Haiku records are still an execution dependency of that builder.
- The heterogeneous reference requires all 1,300 indexed raw outcomes; the heterogeneous agents audit those data separately.
- The exact historical experiment commit and all retry-time source changes remain unresolved.
- No source file becomes a deletion candidate merely because this Game 2 result does not import it.

The detailed path inventory is `/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/e17_homogeneous_game2.json`.
