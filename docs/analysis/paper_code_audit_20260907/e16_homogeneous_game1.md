# E16 homogeneous Game 1 inequality

## Result and checks

- Covers the Game 1 homogeneous points and their contribution to the pooled bars in fig:monoculture_payoff_variance.
- Read all 125 original Game 1 raw results and matching configs; verified seeds, utility-vector sizes, and same-model agent maps.
- Current cohort excludes 25 Claude 3 Haiku runs, leaving four models with 25 runs each.
- All 100 included Game 1 runs reached agreement.
- Independently recomputed population payoff variance from final_utilities; SEM is sample standard deviation divided by square root of 25.

| Model | Runs | Mean variance | SEM |
|---|---:|---:|---:|
| GPT-5 nano high | 25 | 85.3740646944 | 16.3006679869 |
| Qwen3 Max | 25 | 61.2107611111 | 11.6468417807 |
| Claude Opus 4.5 | 25 | 86.3833716667 | 23.8059638881 |
| Gemini 3.1 Pro | 25 | 169.6754585833 | 45.3959327534 |

- The pooled heterogeneous reference is not a Game 1-only comparison; the renderer combines 1,300 runs across all three games.
- The paper uses variance, while the raw builder also computes corrected Gini; Gini code remains part of the callable shared builder.
- Excluded Haiku mean variance is 452.7262211553 from 25 records; retain these records to explain the selection change.

## Provenance chain

- Original saved Slurm scripts invoke /scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py.
- That script imports /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py and calls run_config after converting readable config IDs.
- The shared launcher invokes /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py with saved game, seed, model, token and phase settings.
- Runtime imports agents, phase handlers, prompt generator, Game 1 environment, preference generators, provider routing, compaction, analysis and persistence.
- Slurm uses OPENROUTER_TRANSPORT=proxy and /home/jz4391/openrouter_proxy; credentials are referenced but were not read.
- /scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260816/recreate.py reads original homogeneous raw results and the heterogeneous path index.
- /scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_appendix_gemini_coalitions.py reads aggregate and model CSVs from that dated analysis directory.
- Thus the dated analysis directory contains a required source file, not only replaceable output.

## History checked

- Codex-search found the original June 27 session; manually read launch output at lines 525, 653 and 980.
- Jobs 10357218 and 10357291 are derisk launches, and 10367057 submits 314 remaining configurations.
- Manually read August 22 session lines 309-310, where the user makes the 300-run Haiku-excluded cohort canonical.
- Exact local history paths and evidence lines are in the JSON report.

## Retention and limits

- The JSON report lists runtime import dependencies and explicit raw/config selectors; selectors do not approve all files in a directory.
- Four retained Qwen3 Max runs, 59, 67, 73 and 74, contain synthetic proposal markers; retain them as reported-data provenance and investigate separately.
- No deletion candidate is supported by this result alone.
- Historical runtime revision and final ICLR copy step are not established here.
- Do not remove Haiku exclusions, original submissions, provider recovery records or dated builder scripts merely because they are absent from the displayed 100-run subset.

