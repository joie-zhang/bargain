# E22 baseline-only variance and role payoffs

## Scope and findings

- Current Figure 12 measures variance among baseline agents only, excluding the adversary model.
- The main homogeneous-variance figure instead compares all-agent variance in homogeneous and heterogeneous groups.
- This audit covers the baseline variance trend, SEM bars, role-payoff means, and cohort counts.
- The caption's trend is not monotonic across adjacent models.
- No experiment, code, data or paper file was changed.

## Verified chain

- The saved production Slurm script calls `scripts/full_games123_multiagent_batch.py run-one`, which calls `run_strong_models_experiment.py`.
- The batch generator supplies five adversary models, five group sizes, two role positions and two seed replicates.
- Selected saved results contain 500 Game1, 400 Game2 and 400 Game3 runs.
- The canonical index and paper manifest identify exactly 1,300 homogeneous-adversary raw files.
- I called only the read-only `load_runs()` and `summarize()` functions with bytecode writing disabled.
- All 1,300 raw result hashes match the prior audit inventory.
- All 1,300 standalone config paths and 1,300 transcript paths exist.
- Every numeric summary field matches the current plotting CSV to a maximum absolute difference of 7.1e-15.
- Each model contributes 260 runs.
- Each group size contributes 260 runs.
- The 260 two-player runs contribute zero baseline variance because each has one baseline agent.
- Each bar averages run-level population variance with ddof=0.
- SEM uses the sample SD of 260 run-level variances divided by sqrt(260).
- Baseline payoff first averages baseline agents within a run, then averages runs equally.
- Thus larger groups do not receive greater weight just because they contain more baseline agents.
- The current label-update script executes the dated recreation renderer and reads its summary CSV.
- Both the current paper PNG and September7 output have SHA256 `98d9361d8d581d8ce7302db4a7fc3edfa181f4065f2dc1aedad076b2a730db76`.
- No image regeneration was needed for this verification.

| Model | Elo | Baseline variance | SEM | Adversary payoff | Baseline payoff |
|---|---:|---:|---:|---:|---:|
| Nova Micro | 1240 | 92.137506 | 13.283820 | 43.991764 | 46.016878 |
| GPT-4o mini | 1317 | 100.806606 | 14.485683 | 44.451004 | 45.499260 |
| Sonnet 4 | 1389 | 75.708812 | 10.054623 | 50.993144 | 45.613397 |
| Gemini 2.5 Pro | 1448 | 45.495472 | 5.100312 | 52.926036 | 46.351495 |
| GPT-5.4 High | 1484 | 48.795664 | 5.677681 | 57.960489 | 48.341972 |

## Needed files

The companion JSON records 39 concrete dependencies or explicitly selected cohorts with roles and source evidence.
The production directory entry protects only the selected paths, not every file in that directory.
The earlier dated analysis directories are active dependencies, not automatically disposable audit output.
The original August23 renderer is historical provenance and points at the ICML tree, while the September7 label wrapper copies the image into the current ICLR tree.
The current combined wrapper also needs the Gini renderer and Gini CSV to finish successfully.

## History checks

- [/home/jz4391/.codex/sessions/2026/08/23/rollout-2026-08-23T07-32-24-01a02e64-a937-7b32-a07d-3d364b555e4b.jsonl:9,144,190](/home/jz4391/.codex/sessions/2026/08/23/rollout-2026-08-23T07-32-24-01a02e64-a937-7b32-a07d-3d364b555e4b.jsonl:9): Original user request, variance producer rename/removal of square-root transform, final baseline-agent label patch manually verified.
- [/home/jz4391/.codex/sessions/2026/09/07/rollout-2026-09-07T17-18-23-01a07dbc-8be4-7590-ab3d-aba0e246b7c5.jsonl:103,107](/home/jz4391/.codex/sessions/2026/09/07/rollout-2026-09-07T17-18-23-01a07dbc-8be4-7590-ab3d-aba0e246b7c5.jsonl:103): User requested removal of baseline-agent hyphen and assistant searched for the producer; current rendering verified separately.

Codex-search searched only local history and returned 2,197 repo-linked session records with 42 history-only IDs.
The top September7 result is a fork with the user request; it is not complete execution evidence.
The August23 source-change records were manually checked against the existing producer.

## Limits and cleanup decision

- Exact historical runtime source version for each production attempt was not recovered in this result audit; current imports establish present dependencies, not byte-identical April runtime.
- Saved Slurm launcher uses /home/jz4391/.config/bargain/api_keys.env while current generator defaults to bargain/api_keys.env; secrets were not opened.
- Shared runtime dependency inventory here is partial; parent runtime and E13-E15 audits must complete transitive tests, metrics, package initializers, dynamic provider modules and attempt logs.
- History search returns September7 forked sessions; manually verified user request in 01a07dbc... line103, but that short fork lacks the full completed label-update execution. Current script and equal asset hashes independently establish the current producer chain.
- Figure caption implies a monotonic trend, but variances rise for GPT4o-mini relative to Nova and GPT5.4 relative to Gemini2.5; no paper change authorized.
- No deletion candidate is established by this scoped audit; historical source files remain provenance even when they do not execute in the current path.

No deletion candidates are approved by this result audit.

