# E12: Heterogeneous Game 3 scaling

## Scope

Current appendix lines 550–575 covers positive Elo slopes across group sizes, Elo-bucket means, and 62/65 positive competition fits. Game3 contributes400selectedruns,80at each n=2,4,6,8,10; its four CI values are .10,.16,.40,.64. References: fig:appendix_multiagent_hetero_payoff_full, fig:multiagent_hetero_buckets, fig:appendix_multiagent_hetero_competition.

## Verified provenance

- The retained production root is experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848. RUN_NOTES.md gives generation, selection, submission and resume commands, the cancelled pilot and in-flight provider patch.
- scripts/full_games123_multiagent_batch.py imports active_model_roster and implements24model sampling with5equal-width strata of within-roster Elo standard deviation;4rosters/stratum/cell; randomized seat ordering; independent saved draw/order/game seeds. Same subsets may recur across runs, but each roster contains distinct models.
- Saved slurm/run_full_games123.sbatch invokes batch run-one through project .venv and external OpenRouter proxy; build_command invokes run_strong_models_experiment.py. Game3 environment is game_environments/co_funding.py. Current code is necessary for reruns, not proof of exact historical execution version.
- Independently resolved all400Game3 IDs from selected index to current raw bundles and read every bundle. Each contains a complete final utility map of size n. Config2730 example has alpha=.8,sigma=.5,gamma=.9,maxrounds10,votinglimit16384 and allthree saved seeds.
- scripts/plot_full_games123_clean_subset.py extracts saved final utilities and model/Elo maps; scripts/analyze_n2_plus_multiagent_comparison.py writes heterogeneous_agents_fresh.csv. By-n means use model/n/game aggregation. Bucket script averages appearances within game then equally averages3game means; it does NOT equally weight model means inside a bucket.
- Current competition renderer has a direct default dependency on reproduction_audit/fig25_heterogeneous_competition/plotted_aggregate.csv. Its unweighted model-cell fits produce negative Game3 slopes at(n2,CI.16),(n10,CI.10),(n10,CI.16), as retained fitted_slopes.csv lines47–66 records.

## History checked

Read codex-search skill and invoked bundled search with heterogeneous/stratified/1300anchors; invocation returned no visible results. Manually read original June25session /home/jz4391/.codex/sessions/2026/06/25/rollout-2026-06-25T21-20-13-019f0183-5a1d-70f0-902a-031b28b92d40.jsonl lines2242–2246: original equal-game averaging decision, source-table code and executed output showing1300runs7800rows24models and exact Game3CI levels. Prior September figure-recreation reports were leads, not sole raw-data proof.

## Data integrity and limits

Direct current raw text scan confirms explicit synthetic actions in14Game3configIDs:2391,2466,2513,2517,2518,2559,2601,2681,2682,2683,2686,2712,2719,2726. Preserve these records and failed-attempt history; deleting them would erase evidence, not clean the result. Original notes say no synthetic fallback, but later retained records contain it. Per-attempt source versions and policy-change chronology remain unresolved. Utilities were read, not independently recomputed. Full import closure and external proxy/secret handling require shared-runtime follow-up; no secret contents were read.

## Needed files and candidates

The companion JSON lists concrete launch/runtime/config/data/analysis/provenance/test dependencies and bounded400-run selection rules. Directory entries are expressly not proof for all contents. Historical audit directories are not automatically disposable: current renderer reads one directly. No safe deletion candidates established from this experiment. No experiment, code, paper, data, config or Git changes made.
