# E03: primary Game 3 fixed-baseline experiments

## Scope and conclusion

This audit covers the Game 3 primary capability-payoff slope, baseline payoff competition curves, and the 540-run contribution to `fig:bilateral_overview` and `tab:n2_headline_slopes` (current ICLR `4_analysis.tex:8,18`; appendix experiment breakdown line21). No code, data, paper, or Git changes were made. Only this audit report was written.

The JSON lists 540 exact raw result files and 540 exact selected configuration files, plus concrete supporting paths. All selected raw paths and configs exist. The cohort contains 30 models, 270 runs per model-order arm, 460 agreements, and 80 failures. Alpha is 0, 0.5, or 1; sigma is 0.2, 0.6, or 1. There are 18 runs per adversary model, covering nine alpha/sigma combinations and two orders. CI3=(1-alpha)(1-sigma) yields 300,60,120,60 runs at indices 0,0.2,0.4,0.8 respectively. Different model-order arms have different seeds.

## Verified chain

1. `scripts/generate_cofunding_configs.sh` supports a conservative 540-config sweep and creates batch-specific JSON, index, worker and submission scripts. The saved batch is `experiments/results/cofunding_20260405_083548`.
2. Its concrete `configs/slurm/run_cofunding_worker.sh:43-103` extracts config fields and invokes `run_strong_models_experiment.py --game-type co_funding`. The original saved `logs/cluster/cofund_api_6569989_396.out:27` independently records that command, including models, seed438, alpha0,sigma0.2,10rounds,2discussionturns,10500token cap and gamma0.9.
3. The current runner lazily imports `StrongModelsExperiment`; this imports agent factory, phase handlers, analysis, utilities, and game environment factory. Game3 selects `game_environments/co_funding.py`. Agent factory uses direct/provider clients through `negotiation/llm_agents.py`, `openrouter_client.py`, key rotation, context compaction and JSON repair. Prompts and phases remain runtime requirements. The JSON records concrete dependencies including package imports that load the other game modules even on Game3 paths.
4. Selected results/configs are obtained from `primary_runs_with_metrics.csv` with `game_id=game3` and `baseline_key=gpt5_nano`. The extraction implementation is `scripts/analyze_n2_baseline_comparison.py:775-870`: index -> config -> output path -> selected raw result -> final utility/role mapping -> metrics. Failed negotiations remain in the payoff averages; the extractor has an explicit missing-utility-to-zero rule for no-consensus outcomes.
5. `scripts/paper_figures/render_figure2_large_fonts.py:26,224` consumes the primary metrics. Left panel averages adversary payoffs per model and fits the 30 model means. Right panel aggregates baseline payoff by Elo and competition index, then smooths with EWMA alpha0.24. The raw reconstruction manifest independently names each original result/config. `figure_16/reproduce.py` retains the competition-separated reconstruction path.
6. July original history line4328 reports Game3 slope7.400566667617022 per100Elo for540runs/30models. This confirms the primary number used by current prose. The September figure recreation report supplies an additional check, not the sole original evidence.

## Historical replacement records are needed

Original provider failures and later replacements occurred in this cohort. The current primary path is not necessarily the first run at that path. The archived GPU in-place manifest records config396,397,404,408,409,412 (Qwen72B) and522,523,528 (Llama1B). The saved original error log396 shows OpenRouter context failure and the original command. The April11 reference document and its two attached manifests corroborate GPU reruns with cluster model aliases. `_backfill_archives` also includes April6/7 targeted retries, exact OpenRouter backfills, and QwQ proxy backfills. These records must not be deleted as generic failed-output clutter.

Only named manifests and directly selected source files are individually listed as verified-needed. Preserve the remaining archive trees pending a record-by-record replacement audit, not because every file in those trees was proved necessary here. The external control directory named in the GPU manifest is missing. Original rerun launcher/source-byte provenance is therefore incomplete.

## Important uncertainty

- Current runtime code is a rerun dependency, not proof of April source identity.
- Saved batch summary says budget ratio0.5+0.5*sigma, but current Game3 code uses sigma directly. The exact historical code/config state requires investigation; do not delete the summary or rewrite it to agree with current source.
- The run command's original failure stack confirms experiment, phase, agent and OpenRouter paths. It does not establish every current provider implementation was used historically.
- Model weights, provider services, proxy queues/monitor and credentials are external/shared dependencies. Their exact historical versions were not traced. Do not remove local model workflows because the main launch was API-based; some retained results were GPU replacements.
- This audit checks selected path existence and cohort fields, not a fresh full transcript replay or paid experiment rerun.
- Tests listed in JSON remain protected. They are not disposable merely because figure renderers do not import them.

## Local conversation evidence

The codex-search skill was read in full and its bundled script run with Game3/cofunding queries. The broad query returned June sessions about co-funding mechanisms, which do not establish the original April launch. I manually checked `/home/jz4391/.codex/sessions/2026/07/19/rollout-2026-07-19T15-56-22-019f7bf3-7a5d-77d3-9582-cae6f630081b.jsonl:4328` for original slope output and inspected the requested `/home/jz4391/.codex/sessions/2026/09/05/rollout-2026-09-05T23-13-06-01a074b4-92a4-78c3-96f8-c42ae5db663d.jsonl` as figure reconstruction context. No April session directory is present locally. History coverage is incomplete.

## Removal decisions

No file is approved for deletion from this result's audit. No unused-code conclusion follows from absence in one experiment's import graph. Candidate list is empty. Historical rerun controls, checkpoint loaders, archived failures, and model alias rules need more provenance work before cleanup. The machine-readable report provides the exact protected paths and unresolved questions for the parent audit.
