# E28 coordinated-team quantitative audit

- Current paper references are `fig:gpt54_team_coordination` and `sec:appendix_team_coordination`.
- Scope includes the matched 100-run design, payoff ratios at all five group sizes, team optimality rates, singleton control, and provider/time differences.
- No experiment, code, data, paper, or Git changes were made.
- Only this report and its JSON inventory were authored.

## Verified chain

- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py` imports the older team generator to select 100 controls and copy actual preferences.
- Controls are selected by Game1, homogeneous-adversary family, and gpt-5.4-high from `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255`.
- The full factorial is five group sizes, five competition settings, two adversary positions, and two seeds.
- The selected treatment is `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310`.
- Its saved Slurm script invokes `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py run-one`, which launches `/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py`.
- `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py` implements the binding-team protocol, rotating captain, private planning, and single binding ballot.
- The generator sets three planning turns, 8192 planning tokens, 4096 ballot tokens, one action repair, no synthetic actions, and temperature 1.0.
- At n=2, the runtime disables the binding-team action because there is only one member.
- The exact current figure uses three scripts under `/scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260829/12_homogeneous_analog`.
  - `make_plots.py` reads raw matched results and recomputes utility.
  - `plot_dominance_metrics.py` computes payoff ratios and bootstrap intervals.
  - `plot_payoff_ratio_with_optimal_rate.py` draws the current two-panel figure.
- The canonical team renderer under scripts/paper_figures is an older payoff-gap version, not the source of the current ratio-plus-optimality figure.
- Protect current uncommitted review files and their input tables.

## Independent read-only checks

- Imported the current raw reader with bytecode writes disabled and called only `load_raw_rows()`.
- All 100 matched pairs loaded, with 20 runs per condition and group size.
- All 200 final utilities recomputed from allocations, preferences, discount factors, and final rounds without error.
- All 100 control hashes, preference matches, and matched factors passed.
- All 200 final outcomes reached agreement.
- Historical controls comprise 47 direct OpenAI and 53 OpenRouter routes, while treatment uses direct OpenAI.
- Coordinated exact round-one team-optimal outcomes are 4, 19, 11, 10, and 9 out of 20 at n=2,4,6,8,10.
- The same mathematical ceiling occurs in 4 of 20 uncoordinated controls at every group size.
- The right gray zero line is an explicit display convention, not an observed zero success count.
- The payoff statistic is 100 times mean adversary payoff divided by mean per-run Nano mean payoff, not a mean of per-run ratios.
- Bootstrap intervals use 20,000 draws with seed 20260830 and depend on the full metric iteration order.
- Team-optimality intervals use Wilson intervals with z=1.96.

## History verification

- Ran the bundled codex-search script for the requested team/captain/100-run context.
- Its top matches were later inherited paper-edit sessions, so these were not accepted as launch evidence.
- Manually verified original local session `/home/jz4391/.codex/sessions/2026/08/16/rollout-2026-08-16T07-13-03-01a00a46-7197-78b3-ba17-03ec08a5e13e.jsonl`.
  - Line 3189 records corrected v3 smoke completion and the fresh 100-cell launch.
  - Line 4968 records full completion, historical matching, source changes during execution, and the resulting quantitative gaps.
- Manually verified `/home/jz4391/.codex/sessions/2026/08/31/rollout-2026-08-31T03-34-01-01a056bd-4a5f-7062-a14a-c5b689cc04c2.jsonl`.
  - Line 115 requests the gray zero convention.
  - Line 127 runs the actual current plotting source.
  - Line 167 installs the resulting asset in the paper.
- These conclusions use local history only.

## Retention limits

- The companion JSON lists 400 exact selected configuration and result files plus inspected runtime, analysis, launch, and test dependencies.
- It gives restricted selectors for interaction logs, external prompt files, status files, and attempt logs instead of marking entire experiment trees as proven necessary.
- External prompt gzip files are necessary because interaction JSON can contain only prompt_storage_path references.
- Failed and superseded attempts remain necessary provenance, including the invalid-allocation retry for config81.
- The generation manifest records a dirty worktree and later llm_agents.py changes affecting final attempts81,97,98,99,100.
- Source hashes identify versions but cannot reconstruct missing historical source text by themselves.
- The saved template differs from actual execution limits, which the manifest records as short QOS and 2-hour limits with concurrency35.
- Earlier protocol cohorts are not current figure inputs, but this does not prove they can be deleted.
- No deletion candidates were established within this result audit.
