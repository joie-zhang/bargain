# Heterogeneous experiment launch review

**Question** Can a new user launch heterogeneous Games 1–3, including two-player random pairs, with one portable command?

**Short answer** The existing sampler and runner provide most of the experiment logic, but a public command needs selected-grid generation, a frozen roster and route specification, explicit transport, and stricter result handling.

- This review used source inspection and saved configuration files only.
- No experiment, Python import, test, API call, job command, or Git command was run.
- Only this report was written.

**What is the current command path?**

- The parser in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:100](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:100) exposes `generate`, `validate`, `run-one`, `select`, `submit`, `submit-selection`, `summary`, and `report`.
  - `generate` accepts the results directory, master seed, round count, discussion turns, voting token cap, Slurm time, concurrency, and one of two heterogeneous sampling strategies.
  - `generate` has no family, game, group-size, roster, or runs-per-cell selector.
  - `select` filters an already generated batch by family, game, group size, models, or configuration IDs.
  - The group-size option is repeated as `--n-agents 2 --n-agents 4`, because its parser uses `action="append", type=int`.
- The existing multi-step workflow below is established from the parsers, but it was not executed.
  - `/tmp/bargain-heterogeneous-example` is an example new output directory.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py generate --results-root /tmp/bargain-heterogeneous-example --master-seed 20260427 --heterogeneous-sampling-strategy elo_stddev_equal_width_stratified
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py select --results-root /tmp/bargain-heterogeneous-example --selection-name heterogeneous --experiment-family heterogeneous_random
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py submit-selection --results-root /tmp/bargain-heterogeneous-example --selection-name heterogeneous
```

- To select only the existing two-player heterogeneous design, append `--n-agents 2` to the selection command.
- To generate uniform random pairs, use `--heterogeneous-sampling-strategy pure_random` during generation and then select `--n-agents 2`.
  - This changes the sampling design relative to the paper cohort.
- A local process can run one generated configuration with the following existing syntax.

```bash
OPENROUTER_TRANSPORT=direct /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py run-one --results-root /tmp/bargain-heterogeneous-example --config-id 23
```

- Configuration 23 is the first heterogeneous configuration under the current full-grid ordering in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:753](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:753).
  - Its model roster depends on the sampling strategy and inputs.
  - `OPENROUTER_TRANSPORT=direct` requires internet access from the process host.
  - The current script has no local command that runs a selected list.
- `submit-selection --dry-run` is not read-only.
  - It writes a Slurm script, selected-ID files, and a submission manifest before or after skipping `sbatch` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2537](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2537).

**What exactly does the heterogeneous design sample?**

- The historical design uses `n = 2, 4, 6, 8, 10`, with 20 runs per cell and 65 cells in total.
  - The heterogeneous counts are 500 for Game 1, 400 for Game 2, and 400 for Game 3.
  - Each group size contributes 260 runs, including 260 two-player runs.
  - The combined generator also emits 130 GPT-5-nano controls and 1,300 homogeneous-adversary configurations, for 2,730 configurations.
  - These counts are explicit validator expectations in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1857](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1857).

| Game | Grid for each group size | Size and other settings |
| --- | --- | --- |
| Item allocation | `competition_level = 0, 0.25, 0.5, 0.75, 1` | `num_items = int(2.5*n)` |
| Diplomacy | `rho = rho_min(n), 0.9`, crossed with `theta = 0.2, 0.8` | `n_issues = 10`, with `rho_min(n) = (6/pi)*asin(-1/(2*(n-1)))` |
| Co-funding | `sigma = 0.2, 0.5`, crossed with `alpha = 0.2, 0.8` | `m_projects = int(2.5*n)`, costs 10–30, own-information discussion, commit vote enabled, time discount enabled |

- The grid comes from [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572).
- Shared settings are 10 rounds, two discussion turns, discount 0.9, parallel independent phases, and a separately shuffled model order.
  - The raw Game 1 `competition_level` is a target cosine similarity, so public labels must not reverse its meaning.
  - Diplomacy's negative correlation bound depends on group size and must remain part of the preset.
  - These inputs must be materialized even when a public command uses a short preset name.
- The 24-model pool is derived from a Markdown table, not from the runtime model registry.
  - [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:189](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:189) filters the table for Elo at least 1240 and usable context at least 100,000 tokens.
  - [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:537](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:537) excludes `qwq-32b` and the legacy `gemini-3-pro` label, then requires exactly 24 entries.
  - `gemini-3-pro` is already canonicalized to `gemini-3.1-pro`, so the legacy-name exclusion does not remove the current Gemini 3.1 entry.
  - The saved ordered pool and Elo values are in [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/model_pool_24.csv:1](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/model_pool_24.csv:1).
  - A release preset should contain this ordered pool as machine-readable data, including its hash and Elo source.
- The default sampler enumerates every unordered subset for each supported group size.
  - It computes population Elo standard deviation, sorts by `(stddev, subset_key)`, divides the observed standard-deviation range into five equal-width intervals, and samples four subsets from each interval.
  - It samples with replacement across runs and without repeated model IDs within a run.
  - It shuffles seats with a separate random generator after selecting the subset.
  - The implementation is in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:397](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:397) and [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:905](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:905).
- Equal-width standard-deviation strata are different from equal-width variance strata and from equal-size quantile strata.
  - The three model Elo buckets saved by `quantile_elo_bucket_map` are descriptive metadata and do not drive roster selection.
- For `n=2`, population standard deviation is half the absolute Elo difference.
  - The saved five strata contain 91, 72, 59, 37, and 17 of the 276 possible pairs.
  - Equal numbers of draws from those strata give pairs different selection probabilities.
  - A particular pair in the widest-spread stratum receives about 5.35 times the sampling weight of a pair in the narrowest-spread stratum.
  - The population sizes are recorded in [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/strata_boundaries.csv:2](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/strata_boundaries.csv:2).
- `pure_random` samples `n` distinct pool entries uniformly, then shuffles them independently in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:823](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:823).
  - Neither strategy is a round-robin schedule or guarantees balanced appearances or both seat orders for every pair.
  - Do not switch a historical preset to uniform pairs to make the interface simpler.
- The current paper contains a specification conflict.
  - [/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/3_approach.tex:103](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/3_approach.tex:103) describes combinations from 30 models and variance bins.
  - [/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:197](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:197), the saved pool, and current code establish the 24-model spread-stratified design.
  - The launch specification should follow verified code and saved inputs, while recording the paper discrepancy for its owner.

**Which dependencies and defaults prevent a portable launch?**

- The public path can reuse `common_config`, `game_parameter_grid`, `build_command`, and the existing runtime.
  - `run-one` loads one configuration and calls `run_config` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2412](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2412).
  - `build_command` invokes [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:42](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:42) and forwards game inputs plus the already shuffled model list.
  - `EXPERIMENT_RUN_METADATA_JSON` transports the full generated configuration into the runtime.
  - `StrongModelsExperiment.run_single_experiment` selects a game environment, creates provider agents, executes phases, and saves the result in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119).
- A selected `n=2` command should not enumerate the larger-group subset maps.
  - The existing sampler always builds all five group sizes and retains the complete subset objects in memory.
  - The saved `n=10` map contains 1,961,256 subsets, and the five maps together contain 2,842,225 subsets.
  - Those counts are saved in [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/exact_subset_summary.csv:1](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/exact_subset_summary.csv:1).
  - Pass the requested group sizes into subset-map construction and writing before considering a more complex sampler.
- Validation currently rejects supported parser overrides and smaller batches.
  - It requires exactly 2,730 configurations, 10 rounds, two discussion turns, and no voting token cap in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1851](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1851).
  - Replace fixed expectations with expectations from a resolved specification, while retaining the historical preset's exact invariants.
- A local API run requires no GPU for the current 24-model pool.
  - The current pool uses native OpenAI, native Anthropic, and OpenRouter routes.
  - Onboarding should request `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` according to the resolved selected roster.
  - The runtime already reports missing provider variable names before starting in [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398).
  - The wrapper reads the checkout's `.env`, while direct invocation of the root experiment script does not perform that wrapper step.
- The roster guide's provider routes are stale relative to the runtime registry.
  - Gemini 2.5, Gemini 3.1, GPT-5-nano-high, GPT-5.4-high, Sonnet 4, and Claude 3 Haiku currently use OpenRouter entries.
  - Relevant definitions include [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275), [:345](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:345), [:1071](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1071), [:1101](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1101), [:1195](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1195), and [:1239](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1239).
  - Resolve credentials and context constraints from an explicit route specification, rather than treating the March guide as a current provider catalog.
- The wrapper defaults even local runs to a personal file proxy at `/home/jz4391/openrouter_proxy` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).
  - Make transport an explicit choice between direct HTTPS and the shared file proxy.
  - Require a configured absolute queue path for the proxy choice.
  - Keep the existing externally managed monitor assumption for Della jobs.
  - Proxy requests include authorization headers on disk in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:546](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:546), so onboarding must require a private queue directory and restrictive file permissions.
- The generated Slurm script is Della-specific in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786).
  - It assumes partition `cpu`, 4 CPUs, 16 GB, modules `anaconda3/2024.2` and `proxy/default`, and a checkout-local `.venv`.
  - Its default key-file spelling resolves to `/scratch/gpfs/DANQIC/jz4391/bargain/bargain/api_keys.env` from the current checkout.
  - Its scheduler logs go to the checkout's `/scratch/gpfs/DANQIC/jz4391/bargain/slurm` directory rather than the selected output root.
  - Put these values in a named Della execution profile and provide a local backend without Slurm assumptions.
- The current generation writer replaces a sibling latest-run symlink and can overwrite configuration files in an existing root.
  - These writes occur in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1141](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1141) and [:1238](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1238).
  - Public generation should fail on an incompatible existing root and create no shared latest pointer unless requested.

**What must remain fixed for reproducibility?**

- Preserve three separate seeds for each run.
  - `stable_seed` hashes the master seed and semantic cell identifiers with SHA-256 in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:303](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:303).
  - The draw seed includes the strategy and cell, plus stratum and draw index for stratified sampling.
  - The order seed includes the strategy and cell plus run index.
  - The environment seed includes the cell and run index but does not include the roster strategy.
  - Changing strategy therefore changes roster and order while retaining the environment seed for the same run index.
- Preserve exact pool order, Elo values, standard-deviation formula, interval endpoints, candidate sort order, seed input strings, and Python random behavior.
  - Changing only pool order can change seat assignment or pair selection even if the model set is unchanged.
  - The saved historical manifest records master seed 20260427 and generation Python 3.14.0 at [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/manifest.json:60](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/manifest.json:60) and [:729](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/manifest.json:729).
  - The recorded generation RNG state is not the random stream that chooses rosters, because each roster draw constructs a new `random.Random(draw_seed)`.
- Retain a checked historical configuration as a deterministic regression fixture.
  - [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/config_0023.json:11](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/config_0023.json:11) specifies Llama 3.3 70B followed by Command R+, environment seed `2106547656`, draw seed `598402729`, and order seed `1198855750`.
  - Preserve the original configuration ID as lineage if a smaller public launch uses new local IDs.
- Environment seeds do not provide exact historical replay by themselves.
  - Game 1 calls the pair generator for `n=2` and an SLSQP optimizer for larger groups through [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/multi_agent_vector_generator.py:51](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/multi_agent_vector_generator.py:51).
  - Game 2 generates correlated positions, optimizes weights, rounds to percentages, and refines integer weights in [/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/diplomatic_treaty.py:314](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/diplomatic_treaty.py:314).
  - Game 3 generates integer costs, budgets, and optimized valuations in [/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/co_funding.py:271](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/co_funding.py:271).
  - NumPy, SciPy, and sampler implementation changes can alter those instances.
  - The current dependency file gives lower bounds rather than a lock in [/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1).
  - Save actual preferences, positions, weights, costs, budgets, achieved similarity, solver outcome, dependency versions, and source hashes with each run.
- The Game 1 sampler warns and returns optimized vectors after solver failure or excess error in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/multi_agent_vector_generator.py:133](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/multi_agent_vector_generator.py:133).
  - A strict public preset needs an explicit instance-validity policy and must fail when that policy is not met.
- New token settings differ from what old generated configurations declare.
  - The historical example omits phase token limits.
  - Current CLI defaults set `max_tokens_per_phase = 16384` and fill missing phase limits in [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:336](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:336) and [:607](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:607).
  - The registry includes larger model-specific caps and smaller provider-specific caps in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:3](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:3).
  - Save effective per-agent, per-phase caps and reasoning parameters before execution.
- A logical label is not a complete model identity.
  - For example, `gpt-5.2-chat-latest-20260210` resolves to the undated `gpt-5.2-chat-latest` endpoint in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:604](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:604).
  - `qwen3-max-preview` resolves to `qwen/qwen3-max` in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1081](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1081).
  - Save the requested label, actual API model ID, provider, provider routing constraints, and returned model metadata.
  - No current endpoint availability or exact historical endpoint equivalence was validated in this review.

**Which failure and output behaviors need a release change?**

- Default provider recovery can change the provider.
  - Native calls may use OpenRouter after failure when `OPENROUTER_PROVIDER_FALLBACK` is unset in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290) and [:578](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:578).
  - The agent factory also has a separate provider recovery path during creation in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183).
  - One resolved recovery policy must govern both paths, with provider substitution disabled for a strict launch.
- The agent factory can skip unknown models and can return a smaller roster in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:137](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:137).
  - The current root CLI rejects unknown catalog names, but a new Python adapter must enforce the same validation and exact agent count before play.
- Runtime invalid-action recovery can insert synthetic actions.
  - For example, co-funding inserts a synthetic `nay` after a failed commit vote in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:4710](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:4710).
  - Public production runs need a fail policy that records the invalid response and bounded authorized repairs, then stops without inventing a vote or proposal.
  - A strict rerun will therefore differ from a historical run that continued after such a recovery.
- The batch result validator only requires numeric utilities and checks a few optional identifiers in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269).
  - It does not require finite utilities, exact roster and route equality, matching game parameters, all integrity records, or a resolved-configuration hash.
  - `run_config` reuses an accepted existing file and rewrites its metadata from the requested configuration in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668).
  - Do not reuse or enrich an old result until immutable identity and provenance checks pass under an explicit resume policy.
  - `--rerun-existing` currently affects submission selection but does not bypass `run_config`'s existing-result shortcut.
- Attempt logs are preserved, but a later attempt can replace the single result file.
  - The result writer opens the single-run result path with mode `w` in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:204](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:204).
  - Write each attempt to a new directory and identify the accepted attempt in a manifest.

**Can existing analysis load the public outputs?**

- Retain the existing per-run result schema and neutral `Agent_1` through `Agent_n` seat IDs.
  - The result saves configuration, preferences, utilities, allocation, conversation, performance, and vote integrity in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:897](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:897).
  - Batch enrichment adds roster, seed, stratum, and model-role metadata in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1619](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1619).
- `build_tables(results_root)` is a useful existing analysis API in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:355](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:355).
  - It expects the legacy run-directory pattern plus matching status, interaction, and log files.
  - A new manifest-based loader should handle both legacy directories and attempt directories without relying on absolute historical paths.
  - Carry strategy, order seed, stratum, roster hash, route hash, and attempt identity into every exported table, because `row_base` currently drops several of these fields.
- The current clean-voting test scans text patterns instead of validating structured integrity fields in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:291](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:291).
  - Missing log or interaction files become empty strings through `maybe_read_text`.
  - `filter_subset` does not enforce `strict_voting_clean` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:503](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:503).
  - New analyses must expose failed, incomplete, and excluded runs rather than silently reducing the denominator.
- The full comparison report has hard-coded historical inputs and output paths in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:47](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:47).
  - Add explicit input roots, optional baseline/fairness inputs, and output directory parameters before attaching it to a public command.
  - Keep paper export an explicit action because the current destination names the older ICML tree.
- Aggregate interpretation must retain the sampler name.
  - The stratified design estimates outcomes under equal weight across spread strata unless a different weighting rule is specified.
  - It does not automatically estimate the uniform population of all model rosters.
  - The existing per-model aggregation averages the model's observed appearances in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:598](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:598).

**What is the smallest useful adapter interface?**

- Add a heterogeneous adapter behind the shared public command, using the following interface as a proposal rather than an existing API.

```python
build_heterogeneous_configs(
    spec: HeterogeneousSpec,
    roster: FrozenRoster,
    results_root: Path,
) -> list[ResolvedRun]

run_resolved_config(
    config: ResolvedRun,
    execution: ExecutionProfile,
    *,
    resume: bool = False,
) -> RunOutcome
```

- `HeterogeneousSpec` should declare the preset version, selected games and group sizes, game grids, sampling strategy, runs per cell, strata count, master seed, round/phase settings, and instance-validity policy.
- `FrozenRoster` should declare ordered model entries with Elo, context source, complete resolved provider configuration, and content hashes.
- `ExecutionProfile` should declare local or Slurm execution, Python interpreter, direct or proxy transport, credential variable names, queue path, bounded retry policy, and scheduler settings.
- `ResolvedRun` should contain all result-affecting values before any request, including the actual model order and separate seeds.
- Reuse the existing seed derivation and sampler ordering for the historical design preset.
  - Parameterize subset-map generation by requested group sizes.
  - Parameterize counts and validation by the resolved specification.
  - Do not mutate module constants to support another grid.
- Use a frozen declared roster rather than removing unavailable models automatically.
  - A user-supplied roster defines a new study and needs a distinct identity.
  - Reject empty strata or an insufficient unique model count without changing the strategy.
- Keep the parser free of provider construction and heavy subset enumeration until execution is requested.
- The following is illustrative public syntax and does not exist yet.

```bash
python -m bargain_cli run heterogeneous --preset april2026-24model-stddev --games 1 2 3 --n-agents 2 4 6 8 10 --backend local --transport direct --output /tmp/bargain-heterogeneous-new
python -m bargain_cli run heterogeneous --preset april2026-24model-stddev --games 1 --n-agents 2 --sampling pure_random --backend local --transport direct --output /tmp/bargain-uniform-pairs-new
```

- The uniform-pair override must appear in the resolved study identity and summary before execution.
- An offline plan mode should resolve configurations and show run counts, required credential names, roster/routes, token caps, and output paths without provider calls.
- A one-command launch should generate, validate, execute the selected runs, and return a nonzero exit code if required work failed or remains incomplete.

**What validation is needed before release?**

- Extend [/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:61](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:61) with selected-family/game/group-size counts and deterministic historical configuration checks.
- Verify both samplers against the actual frozen 24-model roster, including `n=2`, every stratum boundary, candidate ordering, repeated roster draws, and independent seat shuffling.
- Verify that generating selected cells preserves the corresponding historical seeds and rosters.
- Check relocation to another checkout and output root without edits to saved scientific inputs.
- Check missing models, missing Elo, empty strata, missing credentials, conflicting routes, invalid transport, incompatible existing results, malformed integrity records, and non-finite utilities.
- Check that a failed provider or invalid action cannot produce a successful production run through provider substitution, roster reduction, or a synthetic action.
- Check that omitted and explicit token settings resolve to the same declared effective values when the specification says they should.
- Check analysis ingestion of complete, failed, incomplete, resumed, and duplicate-attempt records with retained denominators and provenance.
- Keep mocks limited to external boundaries in unit tests.
- Run real smoke tests for each selected provider and all three games before claiming external integration works.
  - Include at least one larger-group run to exercise the n-agent sampler and phase path.
  - A Della smoke test must use the configured transport from the compute node.
  - Integration validation remains incomplete because this review was read-only.

**What remains unresolved?**

- The public default must distinguish the historical spread-stratified design from the separate uniform-roster option.
- Exact historical runtime versions, provider routes, and effective token limits need a separate frozen execution specification if exact historical replay is a release requirement.
- The acceptable Game 1 instance error and solver policy need an explicit study specification.
- The public release must choose whether full subset-map CSVs are distributed or regenerated, because the larger maps are expensive for a small launch.
- Current provider availability and model equivalence remain unverified.
