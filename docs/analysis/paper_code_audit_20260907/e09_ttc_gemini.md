# E09 Gemini TTC audit

## Scope and result

Current Figure5 (`fig:ttc_scatter`) and appendix `tab:appendix_ttc_overall` report little requested-effort endpoint gain for Gemini3Flash. Direct parsing of all720 Gemini config/result pairs gives minimal66.1959911777, low66.2204440801, medium67.7684766551, high66.6793057447: high-minus-minimal +0.4833145670. These match paper rounding. The medium payoff is higher than high; this is an endpoint comparison, not monotonic improvement. All720executed caps are10500.

## Verified chain

Original generator -> archived seed42configs145..216 -> replication generator clones into nine more seed roots -> archived root Slurm run_one.sbatch -> scripts/run_ttc_native_config.py -> run_strong_models_experiment.py -> StrongModelsExperiment -> factory/provider agents + PhaseHandler + PromptGenerator + game environments -> terminal results and full interaction logs -> complete-seed aggregation -> halfwidth figure producer -> current paper asset.

Each effort has180runs. Nine gamecells are Game1 c0/.5/1; Game2 rho-1/0/1 theta1; Game3 alpha/sigma0/.2,.5/.6,1/1. Both orders are averaged within each cell, cells within each seed, then10seeds. Seeds42,984,526,423,1024,128,256,612,2048,4096. All runs n2,maxrounds10,discussionturns2,gamma.9,Nano baseline low effort. Source hash manifest was used only as a path lead; raw config and result JSON were freshly read, and all selected transcript paths exist.

## Provider and rerun hazards

Gemini metadata says Google, but model registry explicitly uses OpenRouter for all four gemini-3-flash-thinking aliases, with reasoning.effort minimal/low/medium/high and exclude=True, temperature1. Factory copies those custom parameters to OpenRouterAgent. Slurm scripts support external /home/jz4391/openrouter_proxy and direct OpenAI baseline access. Therefore deleting OpenRouter client/monitor because metadata says Google would break this experiment. Exact historical route and monitor bytes are not fully proven.

Current original generator now declares16384, and current runner migrates10500 to16384 unless preserve_config_max_tokens_per_phase is true. Nine replication configs preserve10500; originalseed42does not. A current rerun is not automatically an exact reproduction. Historical manifests also record dirty source, so recorded Git commit is not enough.

## Concrete needed files

The companion JSON lists each verified runtime/analysis/support path and evidence, and ten selected data-root rules. The selected data rule is configs145..216 plus each named result/transcript, not whole-tree retention proof. Required imports include qualitative_metrics/schema because normal runtime imports and writes them, even though this payoff figure does not use semantic labels. Context-compaction loads a Markdown context-window roster dynamically, so that document is runtime data, not automatically removable documentation.

## Local history

- /home/jz4391/.codex/sessions/2026/08/14/rollout-2026-08-14T04-28-46-019fff63-4eac-7e31-8963-368d5f16f095.jsonl:5 Directly read original user request to include all2160runs, not earlier Claude-excluded cohort.
- /home/jz4391/.codex/sessions/2026/08/29/rollout-2026-08-29T20-39-12-01a0501b-2794-7462-a4f1-d118dbfef936.jsonl:64 Directly read apply-patch creation of actual halfwidth producer and its derived_summary.csv input.
- /home/jz4391/.codex/sessions/2026/09/05/rollout-2026-09-05T23-13-06-01a074b4-92a4-78c3-96f8-c42ae5db663d.jsonl:23 Read requested session discovery context; downstream Figure05 audit was lead then raw configs/results independently inspected.

The codex-search skill was read and bundled search completed over 2,175 repository-linked sessions and42 history-onlyIDs. Top result019fa81f-23b2-7d71-bc1d-b06f1965c20e is a fork carrying original TTC replication context. Manually verified /home/jz4391/.codex/sessions/2026/07/28/rollout-2026-07-28T05-47-20-019fa81f-23b2-7d71-bc1d-b06f1965c20e.jsonl:259 confirms Gemini via OpenRouter, not direct Google. Line10551 records the earlier partial2106/2160plots, which must not be confused with current complete cohort. History is local-profile-only.

## Limits and deletion candidates

- Historical manifests record dirty worktrees, including configs.py, llm_agents.py, experiment.py and provider files. Git commit alone is insufficient to restore executed source bytes; historical snapshots/patches need global audit.
- Current generator sets16384 cap and runner migrates legacy10500 unless preservation switch; literal current rerun of original seed42 differs from observed all720Gemini runs at10500.
- target_provider='Google' is vendor metadata, not direct Google transport proof. Current model aliases route via OpenRouter; individual original wire transport/proxy service source version not fully established.
- All720result utility values read and means recomputed, but utilities not independently recalculated from each allocation; all720transcript paths checked, not every transcript body manually audited.
- No evidence justifies calling unrelated TTC analysis scripts unused; earlier seed scripts may be historical production dependencies or serve other paper results.

No removal candidates certified. Being outside the720Gemini payoff inputs is insufficient evidence to delete a file; other TTC families, qualitative analyses, accounting audits, and historical lineage may need it. No code, data, configuration, paper, or Git mutations made.
