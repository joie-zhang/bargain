# Experiment Data Retention Audit

Date: 2026-07-20

## Scope

This audit identifies the raw data for the 5,691 runs in the ICML AIWILD paper.
It also identifies old data that can be reviewed for deletion.
On 2026-07-19, 128 delete-candidate directories were moved to `experiments/results/TO_DELETE`.
Eleven obsolete result-root symlinks were moved to the same directory.
No experiment file was deleted or changed.

The exact file list is in `paper_experiment_data_manifest.csv`.
Run `python scripts/build_paper_experiment_data_manifest.py --check` to validate the list.
The classification of all 165 top-level result directories is in `experiment_result_root_classification.csv`.
The classification of all 36 top-level analysis directories is in `analysis_directory_classification.csv`.

## Paper Count

The paper count is correct.

| Batch | Paper count | Validated count |
| --- | ---: | ---: |
| Bilateral, GPT-5-nano baseline | 1,920 | 1,920 |
| Bilateral, Llama 3.3 70B baseline | 500 | 500 |
| Multi-agent, homogeneous | 1,430 | 1,430 |
| Multi-agent, heterogeneous | 1,300 | 1,300 |
| Random monoculture | 325 | 325 |
| Test-time compute | 216 | 216 |
| **Total** | **5,691** | **5,691** |

Each listed run has one result file and one rollout file.
The result files use about 442 MiB.
The rollout files use about 3,819 MiB.

## Raw Roots To Keep

These ten roots contain the raw data for the 5,691 paper runs.
The size is the size of the full root, including logs, retries, and old copies.

| Raw root | Paper rows | Full root size | Use |
| --- | ---: | ---: | --- |
| `experiments/results/scaling_experiment_20260404_064451` | 840 | 2,222 MiB | Game 1 bilateral GPT-5-nano baseline |
| `experiments/results/diplomacy_20260405_082215` | 540 | 1,768 MiB | Game 2 bilateral GPT-5-nano baseline |
| `experiments/results/cofunding_20260405_083548` | 540 | 4,436 MiB | Game 3 bilateral GPT-5-nano baseline |
| `experiments/results/appendix_llama33_baseline_game1_202605` | 140 | 37 MiB | Game 1 Llama baseline |
| `experiments/results/appendix_llama33_baseline_game2_202605` | 180 | 53 MiB | Game 2 Llama baseline |
| `experiments/results/appendix_llama33_baseline_game3_202605` | 180 | 80 MiB | Game 3 Llama baseline |
| `experiments/results/full_games123_multiagent_production_20260428_085255` | 1,430 | 3,616 MiB | Homogeneous adversary and GPT-5-nano control |
| `experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848` | 1,300 | 4,513 MiB | Heterogeneous multi-agent runs |
| `experiments/results/full_games123_random_monoculture_control_20260628_014357` | 325 | 1,056 MiB | Random-monoculture control |
| `experiments/results/ttc_native_scaling_20260502_212943` | 216 | 96 MiB | Test-time-compute runs |

The full roots use about 17.6 GiB.
Do not upload each full root without a filter.
Some roots contain failed attempts, retry logs, and archive copies.
Use `paper_experiment_data_manifest.csv` to select the exact config, result, and rollout files.

## Count Details

### Bilateral GPT-5-nano

The validated game counts are 840, 540, and 540.

The Game 1 generator contains 896 planned rows.
The paper excludes all 28 Claude 3.5 Sonnet configs and all 28 Phi-3 Mini configs.
This leaves 840 accepted Game 1 runs.

The 28 Claude rows are config IDs 182--195 and 630--643.
Their logs show DNS failures when the compute nodes tried to reach OpenRouter.
They did not fail because of context length.
Claude 3.5 Sonnet is now retired, so an exact rerun is not available.

The historical data contained 22 valid Phi-3 Mini results.
Six other Phi configs did not have a valid result.
These six configs are IDs 868, 872, 873, 874, 875, and 881.
The paper now excludes all 28 Phi configs.
The preserved Phi data is in
`experiments/results/excluded_from_paper_20260720/game1_phi_all_results`.
Canceled recovery staging data is in the same archive root.

The same Game 1 config directory also contains 28 appended GPT-5 Nano High configs,
`config_896` through `config_923`.
They cover all seven competition levels, both speaking orders, and both discussion
circuit settings.
All 28 have two-agent result files and matching standalone rollouts.
They were omitted from the earlier analysis because `experiment_index.csv` stopped
at `config_895`, although `all_configs.txt` included them.
The paper inventory also excludes these 28 appended configs.

The Game 1 rows are an intentional discussion-circuit ablation.
The generator planned 448 rows with `discussion_turns=1` and 448 rows with
`discussion_turns=2`.
There are 420 accepted rows at each value after the Phi exclusion.
In the engine, one discussion turn is one full circuit in which each agent speaks
once. Thus, the two settings give each agent one or two messages per negotiation
round, respectively.
The primary paper analysis now selects `discussion_turns=2` for Game 1.
The file `primary_runs_with_metrics.csv` contains this selection.
The file `all_runs_with_metrics.csv` keeps both settings for the released
inventory and the appendix ablation.

The original `config_0412.json` result reached consensus but contained a utility
for only one of two agents.
The canonical loader rejected it because it could not map both agent roles.
The rerun completed on 2026-07-20 and has utilities for both agents.
Its result and rollout have the same experiment ID.
The old malformed directory is in
`experiments/results/superseded_invalid_20260720/config_0412_cofunding_gpt5nano_vs_qwen25_72b`.

The earlier 5,712-run inventory included 22 Phi results and excluded `config_0412`.
The current inventory excludes those 22 Phi results and includes the valid rerun.
Thus, the current paper count is 5,691.

### Llama Baseline

The three roots contain 140, 180, and 180 complete rows.
No result or rollout file is missing.

### Multi-Agent

The production root contains all 1,300 homogeneous-adversary runs and all 130 GPT-5-nano control runs.
The repaired equal-width root contains all 1,300 heterogeneous runs.
All result files pass the batch validator.
All rollout files are present.

The multi-agent grid has 65 game, group-size, and competition cells.
Each homogeneous cell has 22 runs.
Each heterogeneous cell has 20 runs.
The 65 cells cover 25 Game 1 cells, 20 Game 2 cells, and 20 Game 3 cells.

The final completion report is `experiments/results/multiagent_experiment_completion_report_20260504.md`.
It reports 2,730 of 2,730 finished runs.
The final run-level inventory is the union of these two files:

- `experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/homogeneous_runs_fresh.csv` (1,430 rows)
- `experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/heterogeneous_runs_fresh.csv` (1,300 rows)

The consolidated paper inventory is `docs/reproducibility/paper_experiment_data_manifest.csv`.

### Random Monoculture

The final root contains 325 valid result files and 325 paths named
`all_interactions.json`.
For all 325 configs, the experiment ID in the rollout matches the experiment ID in
the result.

`config_0262` had overlapping retries on June 28.
One failed retry overwrote the standalone rollout from a successful retry.
The final recovery run completed on July 20 in Slurm job `11411237`.
The canonical result and rollout now identify experiment
`strong_models_20260720_053224_3297767`.
The result has ten final utilities and reports consensus in round 5.

The file `status/config_0262.json` records eight execution attempts.
Its top-level state is `SUCCESS`.
Keep this status file because it preserves the retry history.

The monoculture grid has the same 65 cells.
Each cell has five runs.

### Test-Time Compute

The TTC root contains all 216 configs, results, and rollout files.
The count is 72 runs for each game.

## Derived Data To Keep

These directories are inputs to current paper scripts or paper figures:

- `experiments/results/n2_baseline_comparison_analysis_20260505`
- `experiments/results/n2_plus_multiagent_comparison_analysis_20260505`
- `experiments/results/n2_ttc_multiagent_comparison_analysis_20260505`
- `experiments/results/appendix_llama33_baseline_analysis_20260503`
- `experiments/results/figure_iteration_20260507`
- `experiments/results/figure_iteration_20260626`
- `analysis/nash_lindahl_fairness_20260505`
- `analysis/neurips_revision_20260504`
- `analysis/ttc_group_intensity_turn_dedup_verification_20260701`

The old partial multi-agent tables had 2,729 rows and omitted heterogeneous `config_2594`.
They are staged under `experiments/results/TO_DELETE/partial_multiagent_results_plot_report_20260503_assets`.
`scripts/analyze_neurips_revision_stats.py` now reads the current 2,730-row tables in
`n2_plus_multiagent_comparison_analysis_20260505`.

## Current Qualitative Data To Keep

Keep the current qualitative analysis directories until the paper and rebuttal work is complete.
They use the final bilateral, multi-agent, monoculture, or TTC roots listed above.

- `analysis/qualitative_rollout_dynamics_20260628`
- `analysis/qualitative_dynamics_trends_20260628`
- `analysis/strategic_qualitative_tags_20260628`
- `analysis/llm_strategic_tag_adjudication_20260628`
- `analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629`
- `analysis/llm_strategic_tag_adjudication_random_monoculture_20260629`
- All `analysis/llm_strategic_tag_elo_exploration*` directories
- All `analysis/ttc_*20260629` directories
- `analysis/homogeneous_adversary_redline_elo_20260628`
- `analysis/homogeneous_adversary_tag_mechanism_20260629`

## Extra Data On Hold

These roots are not part of the 5,691 paper runs.
Do not delete them yet because retained exploratory reports refer to them.

- `experiments/results/game1_multiagent_full_20260413_045538`
  - It uses about 96.5 GiB.
  - `analysis/game1_position_qualitative` refers to this root.
  - It uses an older Game 1 design with different group sizes.

## Staged For Deletion

The approved deletion candidates are under `experiments/results/TO_DELETE`.
This includes the obsolete `full_games123_multiagent_20260427_040554` root and its two
clean-subset analysis directories.
See `experiments/results/TO_DELETE/README.md` for the staging inventory and validator command.
  - It is a pre-production batch and is not part of the final paper count.

Bundle or delete the dependent reports before you delete these raw roots.

## Prime Delete Candidates

The following roots are not in the 5,691-run manifest.
They are also superseded by a verified final root.

- `experiments/results/full_games123_multiagent_heterogeneous_equal_width_20260429_100859`
  - The run notes call this an aborted pilot.
  - It has 178 successful pilot results.
  - The repaired root replaces it and has all 1,300 heterogeneous results.
- `experiments/results/full_games123_random_monoculture_control_20260628_014346`
  - This root has no config files and no result files.
  - The `014357` root replaces it and is complete.

The following groups are also strong delete candidates.
They are not in the paper manifest and current analysis does not read their raw results:

- All `ttc_scaling_202601*` roots and the five `gpt-5-*-effort_vs_*` roots.
- All dated `scaling_experiment_202601*` roots and the two `scaling_experiment_20260403_*` roots.
- All dated `diplomacy_202602*` and `diplomacy_202603*` roots.
- All dated `cofunding_202602*` and `cofunding_202603*` roots.
- All roots with `smoke`, `sample`, `derisk`, `test_validation`, or `matrix_sample` in the name.
- The one-off `*_vs_*_config_unknown_*` roots.
- `experiments/results/full_games123_multiagent_20260427_040554`, after its two old clean-subset reports are deleted or bundled.

These rules do not include the ten paper roots.
They also do not include the 96.5 GiB Game 1 root that is on hold.

The root-level audit classified all top-level directories in `experiments/results`.
It found 54 derisk, smoke, sample, or validation roots.
It found 46 old dated batch roots.
It found 26 one-off comparison roots.
It found two explicitly superseded roots.
This gives 128 delete candidates before the held and operational groups are reviewed.

Six more roots are probably superseded, but retained code or documentation still refers to them:

- `experiments/results/cofunding_20260405_044202_reference_slate`
- `experiments/results/cofunding_20260405_051015_reference_slate_7b75fb4`
- `experiments/results/diplomacy_20260404_045332_gpt5nano_vs_gpt52high_rho_n1_theta_0p9_ci0p9_r5_t2_seed42`
- `experiments/results/diplomacy_20260404_052849`
- `experiments/results/diplomacy_20260405_074235`
- `experiments/results/diplomacy_20260405_074417`

Update or remove those stale references before you delete these six roots.

## Operational Data For Separate Review

Do not delete these items in the first deletion pass:

- `experiments/results/backfill_proposals`
- `experiments/results/cofunding_20260405_083548_cluster_backfill_pli_20260409`
- `experiments/results/cofunding_20260405_083548_cluster_fallback_20260406`
- `experiments/results/cofunding_20260405_083548_cluster_fallback_llama31_2x80gb_20260406`
- `experiments/results/cofunding_20260405_083548_llama32_1b_plicp_20260411`
- `experiments/results/diplomacy_20260405_082215_llama32_1b_inplace_plicp_20260411`
- `experiments/results/diplomacy_20260405_082215_llama32_1b_plicp_20260411`
- The `_backfill_archives` directories inside the three bilateral paper roots

The canonical result paths do not point to these locations.
However, these locations record how some final results were repaired or backfilled.
Review their manifests and logs before removal.

## Old Analysis Delete Candidates

These analysis directories are derived from partial, pre-production, or superseded data:

- `analysis/full_games123_all_success_preliminary_20260428`
- `analysis/full_games123_clean_subset_20260428`
- `analysis/full_games123_clean_subset_20260428_no_synthetic_proposals`
- `analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_preliminary`
- `analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_20260503_rerun`
- `analysis/full_games123_production_20260428_085255_plots_20260429`
- `analysis/full_games123_production_20260428_085255_plots_20260503_rerun`
- `analysis/json_parse_errors_20260502`
- `analysis/__pycache__`

Do not delete the current qualitative directories in the same pass.
Do not delete `analysis/recreated_figures` until the paper figure work is complete.
