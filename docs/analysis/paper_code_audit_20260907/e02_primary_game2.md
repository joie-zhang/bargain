# E02 Primary two-player Game 2 audit

## Scope and current paper
Current ICLR source 3_approach.tex:44 uses the bilateral overview; line96 specifies nine rho/theta settings. Appendix experiment table lists540 Game2 runs. This covers adversary payoff vs Elo, baseline payoff vs competition, and the Game2 component of competition-separated payoff plots.

## Verified chain
- scripts/generate_diplomacy_configs.sh builds timestamped configurations and launcher templates.
- experiments/results/diplomacy_20260405_082215/configs/slurm/run_diplomacy_experiments.sbatch:77 launches run_strong_models_experiment.py --game-type diplomacy from indexed config values.
- All540 current experiment_index.csv rows were read. All540 configs and their exact run_1_experiment_results.json files exist and were parsed. JSON report lists every pair individually.
- Example config_0000.json uses baseline GPT-5-nano, Claude Opus4.6 thinking, five issues,rho=-1,theta=0,two discussion turns,maxrounds10,gamma.9,seed42,max_tokens10500.
- Entry point imports StrongModelsExperiment; factory/phases/prompts/environment generate conversations and final utilities. DiplomaticTreatyGame generates bounded ideal positions and normalized weights, then evaluates weighted distance utility.
- scripts/analyze_n2_baseline_comparison.py selects this batch plus active roster; CI2=theta*(1-rho)/2. The renderer consumes primary_runs_with_metrics.csv and averages models; main overall slope is6.7586007081582435 payoff per100Elo.
- analysis/figure_recreation_20260905/figure_02/scripts/rebuild_raw_inputs.py independently documents the exact source selection. Historical reproduction is corroboration, not original-launch proof.
- Runtime local import closure is enumerated in JSON; other game modules are included because current package imports them. This does not mean Game2 executed those game mechanisms.

## Historical evidence
Original July session line4328 was manually read and confirms540 runs,30models, slope6.7586007081582435 and renderer provenance. User-named September session line1209 was read as a recreation lead. Bundled codex-search was invoked twice and returned no visible results. Search visibility is limited to local persisted history.

## Backfill preservation
The retained _backfill_archives/gpu_inplace_plicp_20260411/20260411_200915/manifest.json records replacement of config523,525,527,529,531,535,537,539. These are Llama3.2-1b strong-first settings. Their old interactions are evidence of how final results were produced, not disposable failed files. Exact backfill runtime and GPU adapter provenance remain unresolved.

## Needed files and limits
The JSON lists1080 individually verified config/data paths plus launcher,analysis,roster,and current runtime import dependencies. Preserve associated archive entries pending full provenance tracing. No experiment reruns or API calls were performed. Runtime source has changed since April; no complete original source snapshot was established. Current dependencies are not a claim of exact historical execution.

## Removal candidates
None certified. Absence from this result is not evidence that a file is unnecessary for another workflow or historical explanation.

