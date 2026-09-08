# E08 Claude TTC dependency audit

## Result and evidence

Current paper reports +6.11 payoff from low to max Claude Sonnet4.6 effort and roughly88 Elo equivalent (figure5, appendix TTC table, conclusion). These are720 Claude negotiations,72 per seed:4efforts x9game cells x2orders. Seeds are42,984,526,423,1024,128,256,612,2048,4096. Baseline is GPT-5-nano low effort; Claude configured via Anthropic adaptive thinking and output_config.effort low/medium/high/max (strong_models_experiment/configs.py:451-503).

Independently read all720 current configs,720 terminal results,720 transcripts and compared each SHA256 with terminal_run_inventory.csv: zero mismatches. Raw result utilities reproduce mean low64.5193386373, medium66.3785647839, high68.5749253222, max70.6273196344. Equal complete grid makes pooled endpoint identical to hierarchical average:6.1079809972. This is unconditional payoff, including zero no-agreement payoff, not only successful deals. Historical aggregation pairs orders within cells, nine cells per seed, then ten seed estimates. Displayed intervals use seed-level Student-t95% intervals.

## Verified launch to paper chain

- Original seed42 batch: scripts/generate_ttc_native_scaling_jobs.py generates archived configs; later scripts/generate_ttc_seed_replication_jobs.py clones216 configs with preserved10500 cap, seed and output path changes only.
- Actual per-seed slurm/run_one.sbatch calls scripts/run_ttc_native_config.py with config path; loads conda/proxy modules, .venv and external credential env (never read secret contents). One CPU,4GB,2h original launcher; recovery142 used4h.
- Native runner subprocesses run_strong_models_experiment.py, which constructs StrongModelsExperiment using configs, factory, phases, prompts, game environments and utility calculation. Claude IDs map to Anthropic; baseline maps to OpenAI. OpenRouter fallback/proxy implementation is imported and must not be assumed unused merely because primary route says Anthropic.
- Raw output selection is exact config output_dir/run_1_experiment_results.json plus run_1_all_interactions.json. All concrete selected files appear individually in JSON, not as blanket directory classifications.
- scripts/analyze_ttc_complete_seed_panels.py imports analyze_ttc_seed_replication.py and produces terminal_run_inventory.csv, run_level_complete_panels.csv, family_effort_complete_seed_ci95.csv and endpoint_changes_complete_seed_ci95.csv.
- Historical plot script scripts/paper_figures/plot_icml_ttc_main_figures.py reads complete summary; exact later half-width producer is lh_review_20260829/07_halfwidth_footnote/make_candidate.py, reading reproduction_audit/icml_aiwild_current_20260829/fig05_ttc/derived_summary.csv. Preserve these review-labelled paths; they are actual figure dependencies.
- analysis/figure_recreation_20260905/figure_05/scripts/recreate.py and its input/snapshot references independently reproduce the figure; prior report is a lead, not sole evidence. Raw counts/caps/payoffs were rechecked in this audit.

## Critical recovery provenance

Actual Claude terminal result configs show651 caps10500,68 caps16384,1cap65536. Base config inventory hides the68 smaller recoveries. Thus claims that only one Claude run changed cap are incomplete. The endpoint figure includes all these outcomes. Preserve failed/replaced outputs to explain the selection, not just terminal successful artifacts.

Seed612 config0142 is max effort, cofunding alpha0.5 sigma0.6, baseline first. The recovery log records repeated16384 failures, attempt16 on August9, attempt17 approved65536 but clamped by llm_agents.py, and attempt18 after extended-output allowlist fix. Preserve recovery_log.tsv, recovery/configs and selected failed_attempts for these IDs, plus superseded_invalid_20260809/config_0142_ttc_seed612_cofunding_sonnet46_max_partial, config_0142_ttc_seed612_attempt16_partial_round4 and config_0142_ttc_seed612_attempt17_clamped_to_16384. These directory selectors are provenance requirements, not per-file proof of every descendant. JSON lists recovery configs that were individually inspected by family metadata; failure archives still require exact per-file expansion.

## Local history

codex-search SKILL.md read fully. Bundled searches for Claude612/max effort and612 returned no stdout. Used direct rg on local sessions and manually read original session/home/jz4391/.codex/sessions/2026/08/14/rollout-2026-08-14T04-28-46-019fff63-4eac-7e31-8963-368d5f16f095.jsonl:5,42,73. Line5 explicitly requests all2160 runs inclusive of seed612;42 patches cohort selection;73 executes regeneration and inspects final means. The requested September5 reconstruction session was consulted, then original data reverified. No global/history-completeness claim.

## Needed paths and limits

The companion JSON has 2344 concrete existing file entries with roles/reasons, including2160 individually hash-checked Claude artifacts. Launch manifests, generated Slurm scripts, recovery configs, shared runtime modules and current figure producers are included. Several dependencies are shared across results and must be unioned, not removed as duplicates.

- Full historical execution snapshot not established; manifests record dirty git state, and present runtime files are not proven identical to original code.
- 68 Claude results execute at16384 despite base configs saying10500; 1 executes65536. Never discard recovery configs/logs or failed attempts as unused.
- Four seed42 Claude configs136,137,138,142 lack separately recovered cap recovery invocation evidence in inspected records; legacy runner migration is an explanation, not proof.
- Current/runtime import list is verified but not exhaustive dynamic closure; prompt submodules, SDK dependencies, tests, environment locks, provider proxy support and all recovery archives need cross-agent coverage.
- 88 Elo equivalent uses independent primary two-player mean slope6.97; only arithmetic6.107981/6.97*100=87.63 verified here, not primary-slope provenance.
- Bundled codex-search executed twice but returned no stdout; manually inspected original local sessions after rg discovery. No access claim beyond on-disk history.
- No deletion candidates approved: absence from this result cannot demonstrate repository-wide non-use.

## Removal recommendation

No deletion candidates established for E08. In particular, do not delete dated review/analysis producer code, recovered outputs, failed-attempt archives, seed42 archives or current runtime modules on the basis of age/name. No code, data, paper, git or environment changes made; only these audit reports written.

