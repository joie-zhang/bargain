# E05: Llama Game 2 replication

180/180 selected raw results exist. Direct raw aggregation gives adversary slope 5.01965888168164, baseline -0.2417061297553205, gap 5.26136501143696, Pearson r .8812924312164064. Stored caps all10500, current generator16384. No removal candidates certified.

## Scope
Main text baseline replication; appendix two-player slope tables and Llama 1x3 figure. Counts 180 runs, 10 models, 3 rho x 3 theta x 2 orders, n_issues=5, seeds42..221, T10, gamma0.9.

## Verified chain
Stored experiment_index.csv -> 180 stored configs -> each output_dir/run_1_experiment_results.json -> analyze_appendix_llama33_baseline_500.py -> overall_by_model_game.csv -> plot_appendix_llama_overall_overlay_1x3.py -> current appendix figure. Analysis raw read was run read-only and reproduced all published Game2 slopes and r.

The actual Slurm log for job7614731 records May2 launch, 10500 cap, proxy transport, and exact runner CLI. Launcher imports current experiment/runtime chain listed in JSON. Current scripts must not be assumed to be historical snapshots. Current generator explicitly emits16384; every stored Game2 config had10500. Preserve both historical configs and provenance.

## History
Codex-search script completed on query appendix_llama33_baseline, found later cleanup and figure sessions; manual history.jsonl1536-1537 verifies user cohort pointers. Manually inspected original August2 figure session1207 and1524, not only reproduction-audit summaries. Session01a074b4 figure19 audit supplied leads, then raw data and original records were independently checked. Exact launch conversation was not recovered; actual launch log is stronger execution evidence.

## Needed files
See JSON needed array for concrete paths, role and evidence. Config and raw-tree entries have explicit selection rules and do not classify their whole directories. Preserve provider tests and unrelated eager imports even when a game-specific branch does not execute.

## Gaps and deletion decisions
- Exact executed source revision not linked to every raw result; current generator and model defaults changed since original May jobs.
- Original launch session not recovered conclusively; verified Slurm actual command and later historical user cohort pointer instead.
- Provider proxy monitor is external/shared; its original version and complete dependency closure not verified.
- No per-file global deletion classification can follow from this result alone; companion logs and failed records need global audit.
- Runtime dependency inventory is verified import chain, not complete dynamic branch proof.

No safe deletion candidate is established by this experiment. Original result aliases, interaction logs, provider failure records and repeated analyses are not declared removable: aliases may be used by other readers and logs retain provenance.

