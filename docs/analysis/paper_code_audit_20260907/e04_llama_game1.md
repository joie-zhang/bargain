# E04: Llama baseline, Game 1

## Scope and evidence

Current appendix lines 435–461 and table row at 450 report 140 Game 1 runs, adversary slope +3.95/100 Elo, Pearson r 0.79, baseline slope -0.65, gap +4.60. Same result appears in the two-baseline comparison table and first panel of fig:appendix_llama_overall.

## Verified chain

- Generator scripts/generate_appendix_llama33_baseline_configs.py lines 70–108 defines 10 adversaries × 7 competition levels × 2 orders = 140, seed 42, 5 items, 10 rounds, gamma .9, two discussion turns.
- Saved slurm/submit_individual.sh submits run_one.sbatch for config_0000.json through config_0139.json, with a retained submitted_jobs.txt manifest.
- Saved run_one.sbatch loads the local environment and module proxy/default, sets OPENROUTER_TRANSPORT=proxy and shared queue /home/jz4391/openrouter_proxy, then invokes saved run_config.py, which launches run_strong_models_experiment.py with explicit settings and metadata.
- Core execution goes through StrongModelsExperiment, phase handlers, prompt generator, agent factory, item allocation environment, preferences and RandomVectorGenerator. Provider agents use negotiation/llm_agents.py or negotiation/openrouter_client.py. Imported provider/key/context/repair helpers must remain. Game environment package eagerly imports Games 2 and 3, so those files are current runtime dependencies even for Game 1.
- This audit directly loaded all 140 configurations from experiment_index.csv and their exact run-number JSON results. All 140 have saved max_tokens_per_phase=10500. One outcome without agreement is retained.
- Independently averaged the selected raw final_utilities by saved role order and model; all ten adversary and baseline means match overall_by_model_game.csv to 1e-9. No missing result fallback was used by this check.
- Historical analyzer uses the index and canonical name/seat fallback logic, then exports model means and slopes. Its permissive result-file fallback is not needed for these 140 exact files but remains a risk for new inputs.
- Plot script consumes overall_by_model_game.csv and fits unweighted OLS across ten model means. Current default output still targets ICML, so ICLR recreation requires explicit --output; do not delete this script as stale by venue name.

## History

Codex-search skill was read and bundled search run for llama33 baseline 500 appendix. Directly read local history.jsonl lines 1283 and 1384: session 019deb36-702e-7281-831e-85fa43303c19 requests CPU Slurm submission of 500 experiments, then analysis after 500/500 completion. The full May session is missing locally. Prior figure reconstruction documents August overlay and typography edits and exact output matching, but those records are supporting provenance, not replacement for the independent raw check above.

## Needed files

Companion JSON contains 140 exact config paths, 140 exact raw-result paths, and named launch/runtime/config/analysis/provenance dependencies. Raw JSON includes embedded conversation_logs and saved preferences. Retain selected run companions and Slurm logs pending per-file provenance review; this report does not mark their entire parent directories as required or deletable.

## Uncertainty and cleanup decision

Current generator specifies 16384 tokens but saved configs and results all specify 10500. Current code is not a guaranteed historical snapshot. Shortened provider model names do not prove thinking settings. Exact API request provenance and runtime commit are not uniform. External key file was not opened. Shared queue provider service and installed dependencies must remain protected. Tests and package initializers were not exhaustively traced. No deletion candidate is safe from this experiment alone. JSON dependency list is a verified minimum, not proof of exhaustive closure.

