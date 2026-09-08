# E18 homogeneous Game 3 inequality audit

- Scope is the Game 3 contribution to `fig:monoculture_payoff_variance`, including all five model points and their comparison with the pooled heterogeneous reference.
- All 100 Game 3 raw results and saved configurations were checked, IDs 226 through 325.
- Each model has 20 runs across five agent counts and four sigma/alpha settings.
- No Game 3 model or run is excluded from the current figure.
- Config 246 did not reach agreement and remains included with zero utility.

## Independently checked numbers

| Model | Runs | Mean population payoff variance | SEM |
|---|---:|---:|---:|
| amazon-nova-micro-v1.0 | 20 | 81.022549 | 29.977335 |
| deepseek-v3 | 20 | 64.276457 | 19.813949 |
| deepseek-r1-0528 | 20 | 90.531371 | 24.814708 |
| claude-opus-4-5-20251101-thinking-32k | 20 | 80.328997 | 26.421123 |
| gpt-5.4-high | 20 | 41.968786 | 16.395892 |

- Recomputed all 1,300 heterogeneous raw variances referenced by the canonical index.
- The plotted heterogeneous mean is 95.751656846757, with SEM 5.145927034456228.
- This reference pools all three games and all group sizes.
- Each Game 3 model mean is below that pooled reference.
- Population variance uses ddof=0 within each run; SEM uses ddof=1 across runs.
- This verifies the descriptive arithmetic, not a causal effect of capability diversity.

## Verified provenance chain

- `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py` generates a 325-config original batch.
- The model selection seed is 20260628.
- The generator excludes Claude Sonnet 4 from a 24-model pool and samples three models from each of five Elo bands.
- One sampled model from each band is assigned to each game.
- The generated Slurm script calls the same script's run-one action, which delegates to `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py`.
- That module constructs the command for `/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py`.
- The runner uses StrongModelsExperiment, its agent/phase/configuration modules, and the co-funding game engine.
- Slurm exports OpenRouter proxy transport and uses an external queue under /home/jz4391/openrouter_proxy.
- Saved config and result seeds agree for all 100 Game 3 runs.
- `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260816/recreate.py` reads all 325 homogeneous results before filtering the 25 Game 1 Claude 3 Haiku runs.
- The current figure renderer is `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_appendix_gemini_coalitions.py`.
- The renderer reads the retained aggregate and model CSVs in the dated analysis directory.
- Its configured output directories are older paper trees, so updating the current ICLR asset also needs an explicit destination step.
- The JSON companion lists concrete runtime/configuration/analysis dependencies and carefully bounded data selectors.
- A directory selector does not classify unrelated contents of that directory as needed.

## Original local history

- Codex-search returned session 019f0c08-b51a-7452-a490-7bff634dc0a1 for the original control design.
- Read lines 362, 682 and 1369 in `/home/jz4391/.codex/sessions/2026/06/27/rollout-2026-06-27T22-22-05-019f0c08-b51a-7452-a490-7bff634dc0a1.jsonl`.
- Line 682 records the original launch, including cancellation of a config-ID-bug array and replacement job 10357291.
- Line 1369 records a 4096-token discussion override for DeepSeek V3 configs 254 and 258 after repeated context-window failures.
- Read line 309 in `/home/jz4391/.codex/sessions/2026/08/22/rollout-2026-08-22T05-43-57-01a028db-0561-7ac0-91aa-663944f564ed.jsonl`.
- That user request establishes the later 300-run canonical cohort and the Haiku exclusion.
- This exclusion concerns Game 1 only and does not justify deleting the original 25 records.
- The original builder still requires all 325 raw results.
- Local history is incomplete and does not establish an exact original runtime commit.

## Important preservation findings

- Independently found explicit synthetic markers in six included Game 3 runs.
  - Nova Micro configs 229, 239 and 245 contain synthetic proposal/action markers.
  - DeepSeek R1 configs 276, 277 and 281 contain synthetic voting markers.
- These are present in the plotted cohort and must remain available with their original records.
- A saved submission records a later rerun of config 262 on July 20, Slurm job 11410529.
- The previous config 262 payload location was not established in this task.
- Configs 254 and 258 retain their context-window override metadata and previous failed job IDs.
- Submission, selection, transcript and failure evidence therefore cannot be discarded simply because a final result now exists.
- Current source imports establish present dependencies, not identical historical behavior.
- The external provider environment and credentials were not opened.
- Shared-runtime coverage must finish transitive/dynamic resources and relevant tests.

## Deletion recommendation

- No deletion candidate is supported by this result audit.
- Files outside this dependency set remain unresolved, not unused.
- No experiment, code, data or paper file was changed.

