# Homogeneous adversary launch review

**Question** Can a new user launch one adversary model with `n-1` GPT-5-nano agents in any of the three games with one command?

**Short answer** The game runner supports this roster today, but the batch interface needs a small family-specific entry point and several provenance checks before it is suitable for a public release.

- This review used source inspection and reads of saved configuration and result files.
- No Python entry point, experiment, test, API request, or Slurm submission was executed.
- All `bargain` commands below are proposals, not existing commands.
- The review followed [the review scope](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/review_scope.md:1) and [the project instructions](/scratch/gpfs/DANQIC/jz4391/bargain/AGENTS.md:1).

## What exists now?

- The shared multi-agent interface is [the batch parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:100).
  - It has `generate`, `validate`, `run-one`, `select`, `submit`, `submit-selection`, `summary`, and `report` commands.
  - Its generated families are `homogeneous_control`, `homogeneous_adversary`, and `heterogeneous_random`.
  - Its `homogeneous_control` means all GPT-5-nano agents, which differs from the paper's random same-model homogeneous setting at [the experimental setup](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/3_approach.tex:104).
- The exact roster construction is [the homogeneous adversary loop](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:784).
  - `first` produces `[adversary, nano, ..., nano]`.
  - `last` produces `[nano, ..., nano, adversary]`.
  - The saved order label is `adversary_first` or `adversary_last`.
  - The experiment preserves the supplied model list at [the order handling code](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:166).
- [The agent map builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:517) records models, Elo values, and analysis roles for `Agent_1` through `Agent_n`.
  - Agent-facing names remain neutral numbers at [agent creation](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:126).
  - A public validator should require exactly one adversary seat and reject an adversary identifier equal to the baseline identifier.
  - The current role builder classifies by model equality, so an all-nano roster passed as this family would label every seat as an adversary.
- [The batch command builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535) invokes [the common runtime entry point](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:42) as a subprocess.
  - It uses the repository environment at `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python` when that file exists.
  - Otherwise it uses the calling Python interpreter.
  - It sets the subprocess working directory to the repository root.
  - It sends the full generated configuration through `EXPERIMENT_RUN_METADATA_JSON` at [the environment setup](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).

| Stage | Existing implementation and dependency |
|---|---|
| Build one configuration | [common_config](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:626), [game_parameter_grid](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572), and [stable_seed](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:303) |
| Build the full grid | [build_configs](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:727), with an unconditional dependency on the heterogeneous roster and subset enumeration |
| Start one run | [run_config](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656), then [StrongModelsExperiment.run_single_experiment](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119) |
| Create agents | [StrongModelAgentFactory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:52), [OpenAIAgent](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2657), and [OpenRouterAgent](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:200) |
| Create a game | [create_game_environment](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/__init__.py:44), selected by `item_allocation`, `diplomacy`, or `co_funding` |
| Run the protocol | [PhaseHandler](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:42), with game-specific prompts and shared thinking, discussion, proposal, voting, and reflection handling |
| Save the result | [ExperimentResults](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/data_models.py:7), [FileManager.save_experiment_result](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:204), and [enrich_result_metadata](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1619) |
| Load analysis rows | [build_tables](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:355), then [aggregate_hom_adversary](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:570) |

## What is the paper grid?

- The grid has 1,300 homogeneous adversary configurations, as recorded in [the production manifest](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/manifest.json:88).
  - Game 1 has 500 runs.
  - Game 2 has 400 runs.
  - Game 3 has 400 runs.
- [The current constants](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:37) retain five adversary models, five group sizes, both endpoint positions, and replicate labels `1` and `2`.
  - The adversary models are `amazon-nova-micro-v1.0`, `gpt-4o-mini-2024-07-18`, `claude-sonnet-4-20250514`, `gemini-2.5-pro`, and `gpt-5.4-high`.
  - Group sizes are `2`, `4`, `6`, `8`, and `10`.
  - The maximum is 10 rounds, with two discussion turns and discount factor `0.9`.
  - Independent agent phases run concurrently, while discussion remains serial, as stated in [the runtime option](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:352).

| Game | Parameters for each group size | Runs |
|---|---|---:|
| Game 1, item allocation | `num_items=int(2.5*n)`; competition `0.0, 0.25, 0.5, 0.75, 1.0` | 500 |
| Game 2, diplomacy | 10 issues; `theta=0.2,0.8`; `rho=0.9` and `rho_min(n)` | 400 |
| Game 3, co-funding | `m_projects=int(2.5*n)`; `sigma=0.2,0.5`; `alpha=0.2,0.8`; costs `10.0` to `30.0` | 400 |

- [The parameter generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572) defines the table above.
  - `rho_min(n)=(6/pi)*asin(-1/(2*(n-1)))` comes from [the lower-bound function](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:299).
  - Game 3 also records own-contribution discussion transparency, enabled commit voting, and time discount `0.9`.
  - These game cells match [the current paper setup](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/3_approach.tex:105).
- The master seed is `20260427`, but the runtime seed is not the replicate label.
  - [The adversary seed calculation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:792) hashes master seed, family, game, group size, competition identifier, adversary identifier, position, and replicate label.
  - Changing the adversary model or position changes the preference seed.
  - The two positions are therefore not matched preference draws merely because their replicate labels agree.
  - Preserve the seed formula and its exact competition identifier strings when extracting the generator.
- [The historical configuration for run 213](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/configs/config_0213.json:1) is a concrete single-run example.
  - It uses Nova Micro first, three nano agents, Game 1, competition `0.0`, 10 items, replicate label `1`, and runtime seed `1145145728`.
  - [The saved result](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/runs/config_0213_game1_homogeneous_adversary_n4_comp_0p0_amazon_nova_micro_v1p0_first_seed1/experiment_results.json:43) retains family, role, roster, and order metadata.
- [The payoff figure validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_figure24_hom_adversary_competition_clean.py:53) expects 325 means, each from exactly four runs.
  - Those four runs pool two positions and two replicate labels.
  - A partial grid must report missing cells instead of being presented as the complete paper grid.

## Which current commands are established by source inspection?

- The commands in this section are existing syntax, but they were not executed.
- The example result directory must be new and writable.
- A fresh generated root is needed because current selection validates the complete grid before applying any family filter.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py generate --results-root /tmp/bargain-homogeneous-adversary-current --master-seed 20260427
```

- This command generates all 2,730 configurations and the default heterogeneous subset maps.
- It does not launch experiments.
- Generation also replaces a sibling latest-result link at [the end of the writer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1238).
- The proposed public interface should remove that side effect from default generation.

```bash
OPENAI_TRANSPORT=direct OPENROUTER_TRANSPORT=direct /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py run-one --results-root /tmp/bargain-homogeneous-adversary-current --config-id 213
```

- This launches the run-213 cell using current runtime behavior after the preceding generation step.
- Both transport variables select direct HTTPS for a machine with internet access.
- The command requires OpenAI and OpenRouter credentials for this roster.
- A `run-one` call against the historical result directory can reuse the existing result, so it is not a new-run command.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py select --results-root /tmp/bargain-homogeneous-adversary-current --selection-name hom_adversary --experiment-family homogeneous_adversary
```

- This selects 1,300 configurations from a complete newly generated root.
- `--game-label game1`, `--n-agents 4`, `--adversary-model amazon-nova-micro-v1.0`, `--adversary-position first`, and `--seed-replicate 1` are also valid selection arguments.
- That narrower selection still contains five Game 1 competition cells because the parser has no competition filter.
- `--config-id 213` selects the single known cell.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py submit-selection --results-root /tmp/bargain-homogeneous-adversary-current --selection-name hom_adversary --max-concurrent 1 --dry-run
```

- This is a Slurm preparation command, not a portable local sweep command.
- Its `--dry-run` skips `sbatch` but still writes submission files and regenerates the Slurm script at [the submission implementation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2534).
- Removing `--dry-run` submits paid API work through Slurm and was not done during this review.
- The generated script assumes the Della CPU partition and Della modules, so a new user must not treat it as a generic Slurm template.

## What blocks a fresh user or a faithful rerun?

- **Generation and validation are tied to the full study.**
  - [Generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:741) loads the heterogeneous pool before building any family.
  - [Subset construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:397) enumerates all model combinations at every group size for the default heterogeneous strategy.
  - [Validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1851) requires contiguous identifiers, all three families, 2,730 configurations, and fixed paper settings.
  - The generator exposes maximum rounds, discussion turns, and voting cap options that the same validator rejects when they differ from 10, 2, and absent/null.
  - A family-only grid and a cheap short smoke run need separate validation from the paper coverage check.
- **The historical root does not satisfy today's full-grid schema.**
  - [The historical manifest](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/manifest.json:48) contains `gemini-3-pro` and lacks current heterogeneous sampling metadata.
  - [The current validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1900) rejects those model references and requires sampler fields.
  - [Selection](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2436) checks the full root before selecting homogeneous adversary records.
  - This incompatibility follows from source inspection and was not tested by running the validator.
- **The nano reasoning setting differs from the paper description.**
  - [The paper](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/3_approach.tex:93) describes nano at provider-default medium effort.
  - [The nano catalog entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265) leaves effort unspecified.
  - [The current OpenAI client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840) then explicitly sets effort to `low`.
  - The sampled historical configuration does not store the resolved effort, so the paper description alone does not prove every historical request used medium.
  - `gpt-5-nano-high` is a separate high-effort OpenRouter configuration and must not replace the baseline as a convenience.
- **Output limits and context handling are result-affecting inputs.**
  - [Current constants](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:3) include a 16,384 default, a 5,120 Nova Micro cap, and 65,536 for GPT-5.4 High.
  - [Phase resolution](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1030) combines per-run and per-model limits.
  - [Structured voting](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:53) also has its own limits.
  - The sampled April configuration and result do not record these resolved settings.
  - [Context handling](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:95) reads a Markdown table, and [token estimation](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:157) changes with optional tokenizer availability.
- **Elo and context data are runtime dependencies.**
  - [The roster module](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:9) reads `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md`.
  - Its [Elo loader](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:138) requires every active adversary to have an entry even when only one model is requested.
  - It aliases nano to high-effort nano for Elo purposes at [the alias table](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:61).
  - The sampled historical config contains null baseline Elo values, while the current loader yields 1337.
  - The analysis loader explicitly supplies 1337 for nano at [its lookup table](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:30).
  - Package a versioned data resource and record the distinction between the runtime model and the Elo reference model.
- **Resume checks can accept the wrong prior result.**
  - [Result validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269) checks utility presence and some optional identifiers, but does not compare the full model list, parameters, provider routes, or configuration hash.
  - It accepts numeric non-finite utilities because it only calls `float`.
  - [The existing-result path](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668) marks such a result `SUCCESS` and updates its metadata without executing a new run.
  - `submit-selection --rerun-existing` only changes selection, while `run-one` still takes the same reuse path.
  - Failed attempts keep separate logs but share the result and interaction directory, whose single-run files are written with mode `w`.
- **Failure policy must be explicit before exposing the launcher.**
  - [Native provider route handling](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:741) can move a request to OpenRouter after a native failure.
  - Choosing direct transport does not disable that provider change.
  - [Proposal recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2400) creates synthetic default proposals after failed repairs.
  - [Vote recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3232) can insert default rejection votes.
  - Logging those actions does not make them permissible research evidence under the current project instructions.
  - The public run must fail with the saved provider response and bounded recovery history when a required valid action is missing.

## Which credentials and dependencies are needed?

| Requested model | Current API route | Required credential |
|---|---|---|
| `gpt-5-nano` | OpenAI, `gpt-5-nano` | `OPENAI_API_KEY` |
| `amazon-nova-micro-v1.0` | OpenRouter, `amazon/nova-micro-v1` | `OPENROUTER_API_KEY` |
| `gpt-4o-mini-2024-07-18` | OpenAI, `gpt-4o-mini-2024-07-18` | `OPENAI_API_KEY` |
| `claude-sonnet-4-20250514` | OpenRouter, `anthropic/claude-sonnet-4` | `OPENROUTER_API_KEY` |
| `gemini-2.5-pro` | OpenRouter, `google/gemini-2.5-pro` | `OPENROUTER_API_KEY` |
| `gpt-5.4-high` | OpenRouter, `openai/gpt-5.4`, high effort | `OPENROUTER_API_KEY` |

- The route evidence is in the catalog entries for [nano](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265), [Gemini](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1071), [Sonnet](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1101), [GPT-4o Mini](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1154), [Nova](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1206), and [GPT-5.4](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1239).
- The full five-adversary grid needs both key types under the current route configuration.
  - The GPT-4o Mini plus nano single-run case only needs OpenAI credentials.
  - Native Anthropic and Google credentials are not needed for these exact catalog identifiers.
  - Provider availability was not checked online or through a real API call.
- [Credential validation](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398) checks the selected models and saved model overrides before runtime.
  - [Key discovery](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:88) supports simple environment variables and named key groups through `LLM_KEY_GROUP_ORDER`.
  - [The batch runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1521) loads missing environment values from `/scratch/gpfs/DANQIC/jz4391/bargain/.env`.
  - No credential file or credential value was read in this review.
- For a normal internet-connected workstation, set `OPENAI_TRANSPORT=direct` and `OPENROUTER_TRANSPORT=direct`.
  - The batch runner otherwise defaults OpenRouter to a file queue at `/home/jz4391/openrouter_proxy`.
  - [The OpenAI client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742) also selects its file queue in Slurm when transport is `auto`.
  - A portable queue profile must set both `OPENAI_PROXY_POLL_DIR` and `OPENROUTER_PROXY_POLL_DIR` to the same configured shared directory when using the shared monitor.
  - The existing external monitor was assumed to be managed separately and was not checked or started.
- [The Slurm writer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786) hard-codes the CPU partition, four CPUs, 16 GB, the repository environment, Della modules, and repository-local Slurm logs.
  - It loads `anaconda3/2024.2` and `proxy/default`.
  - Its relative default key-file string resolves to `/scratch/gpfs/DANQIC/jz4391/bargain/bargain/api_keys.env` after changing directory to the repository root.
  - `BARGAIN_API_KEYS_ENV` is an existing way to override that path.
  - The homogeneous adversary grid uses hosted models and does not require a GPU or a Hugging Face download.
- A static runtime dependency list includes NumPy, SciPy, `openai` with its HTTP dependencies, and `aiohttp`.
  - NumPy and SciPy are used by all three game environments through package imports.
  - Game 1 preferences pass through [the preference system](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/preferences.py:141) to [the multi-agent vector generator](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/multi_agent_vector_generator.py:27).
  - That generator uses the pairwise generator for `n=2` and SciPy optimization for larger groups.
  - [The root requirements](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1) use lower version bounds and also install analysis and UI packages.
  - Source discovery found no root packaging manifest or dependency lock file.
  - A Python version, locked runtime dependencies, and separately installable plotting dependencies need release validation.

## What should the public interface change?

- Use the proposed installed `bargain` interface, with multi-agent experiments as an umbrella and homogeneous adversary as one family.
- A proposed small run can be written as follows.

```bash
bargain run homogeneous-adversary --game game1 --agents 4 --adversary amazon-nova-micro-v1.0 --adversary-position first --competition 0.0 --seed 42 --transport direct --output /tmp/bargain-homogeneous-adversary-one
```

- The same family command should accept `--game game2` with explicit `--rho` and `--theta`, or `--game game3` with explicit `--sigma` and `--alpha`.
- Put fixed small-run choices in a declared schema and save their resolved values before any paid request.
- This example is a new run with seed `42`, not the paper grid or a recreation of historical run 213.
- The proposed paper-grid entry point should first expose its complete plan.

```bash
bargain sweep --preset paper-homogeneous-adversary-v1 --plan-only --transport direct --output /tmp/bargain-homogeneous-adversary-paper
```

- The reviewed plan must contain 1,300 configurations and expose the 500/400/400 game counts before execution.
- The execution form can use the same preset without `--plan-only`, with a persistent writable output location chosen for real research runs.
- A local backend must work without Slurm, modules, historical output directories, or another researcher's home directory.
- A separately selected Della backend can render site-specific Slurm and queue settings.

| Smallest change | Existing code to reuse or adjust |
|---|---|
| Extract a pure `build_homogeneous_adversary_configs(spec)` function | Reuse the loop at [line 784](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:784), `common_config`, `stable_seed`, and the game parameter functions without constructing a heterogeneous pool |
| Split run validation from preset coverage validation | Replace the full-grid requirement in [the validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1851) with a per-run schema and a separate immutable preset check |
| Keep configuration identifiers stable for the paper preset | Preserve historical IDs and seeds, or save explicit source IDs alongside new run IDs so loaders cannot mix runs from different roots |
| Add a local sequential or bounded-concurrency sweep backend | Reuse [run_config](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656) after adding explicit runtime and output policies |
| Expose transport and site settings as data | Replace the implicit queue defaults at [line 1717](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717) and isolate [the Della script writer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786) |
| Resolve all model settings before launch | Reuse [resolve_model_config](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:60) and store provider, provider model ID, effort, temperature, phase caps, context policy, and declared retry policy per seat |
| Make historical settings explicit | Use run-scoped overrides through the existing `model_config_overrides` mechanism after auditing historical request settings, without changing global catalog entries for unrelated experiments |
| Add explicit `resume` behavior | Require a matching configuration hash and complete valid result before reuse, and give new attempts separate artifact directories |
| Enforce the requested provider and valid agent actions | Add a declared provider recovery policy to [route handling](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:741) and fail after bounded invalid-action repairs instead of inserting synthetic actions |
| Make table export accept a result manifest and output directory | Reuse [build_tables](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:355) and [the homogeneous aggregation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:570) without invoking hard-coded paper output writers |

- Preserve legacy entry points as adapters to the extracted functions.
- Do not make the public launcher depend on existing research result trees to generate a new run.
- A paper preset must pin a data snapshot, source code version, and runtime settings, rather than inherit a mutable model catalog.
- If historical request settings cannot be recovered, name the available operation as a rerun of the historical design with current runtime settings.

## What must analysis preserve?

- The existing directory layout has a manifest, numbered configuration JSON files, an experiment index, run directories, status records, and per-attempt logs.
  - [The file writer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1141) creates that structure.
  - Preserve it for compatibility while adding schema and provenance fields.
- For the example root, the raw table loader expects result paths beneath `/tmp/bargain-homogeneous-adversary-current/runs/config_*` and reads the adjacent transcript plus the root's numbered log and status files at [the loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_full_games123_clean_subset.py:355).
  - It derives adversary and baseline payoffs from `agent_role_map`, so a generic `--models` invocation alone is insufficient for this analysis family.
  - It currently glob-loads present result files and does not enforce expected-grid completeness.
  - Missing logs become empty text and can hide missing diagnostics.
  - Its strict-voting field is diagnostic metadata, not a complete failure exclusion policy.
- [The combined analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:62) hard-codes the April homogeneous root and the separate repaired heterogeneous root.
  - [Its table preparation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_plus_multiagent_comparison.py:549) always loads both.
  - It also has fixed analysis and Overleaf destinations for older paper directories.
  - The current paper is under `/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template`, so the combined script is not an appropriate default post-run action.
- Keep non-agreement separate from infrastructure failure.
  - [The engine](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:859) assigns zero utilities when Games 1–2 reach the round limit without agreement.
  - Such a completed negotiation belongs in outcome analysis.
  - Missing API responses, incomplete artifacts, mismatched configuration, and failed valid-action recovery must remain failed or incomplete runs.
- Export the expected, completed, failed, excluded, and missing run counts with every aggregate.
- Preserve the exact four-run grouping and reported uncertainty for the paper payoff panels.

## What validation remains?

- Extend [the existing generation tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:61) to cover a family-only 1,300-run grid without constructing heterogeneous subsets.
- Check all five roster choices, both endpoint positions, and the exact one-adversary invariant.
- Check seed and identifier compatibility against saved configurations, including run 213 and representative Game 2 and Game 3 cells.
- Check that a small-run round limit is allowed while the paper preset still enforces 10 rounds and two discussion turns.
- Check that planning requires no credentials and performs no API call, job submission, or sibling latest-link update.
- Check that moving a prepared run directory does not keep an output path pointing to the old location.
- Check that resume rejects changed model settings, changed seeds, non-finite utilities, incomplete transcripts, and missing provenance.
- Check that exact provider selection cannot silently become an alternate provider after failure.
- Check that invalid proposals or votes fail after the declared bounded recovery instead of entering outcome analysis as generated actions.
- Check that a fresh result exports the model and role fields needed by the existing table loader.
- Run real API smoke tests for each required provider route and one complete run in each game before claiming integration success.
  - Those tests remain unperformed because this review forbids API calls and experiment execution.

## Which questions remain unresolved?

- The sampled historical result does not establish its resolved nano reasoning effort or output caps.
- Historical prompts and preference generation may differ from current code even when a saved seed is unchanged.
- Provider model availability and exact equivalence of dated identifiers to current routed identifiers have not been checked.
- A release Python version and locked dependency set have not been validated on a fresh installation.
- The paper preset needs an explicit decision about whether it reproduces historical request settings or reruns the historical design under a named current runtime profile.
