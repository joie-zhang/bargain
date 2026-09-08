# Homogeneous launch review

**Question** Can a new user launch a game in which every agent uses the same model, including the paper's 300 homogeneous runs?

**Short answer** The existing parser can express a single game through repeated `--models` arguments, but the historical generator produces 325 runs and depends on untracked data, so a public launcher needs a small homogeneous adapter, an explicit paper profile, and shared runtime corrections.

- This review inspected source code and saved JSON files on 2026-09-07.
- No experiment, API request, job submission, package installation, or experiment import was run.
- The checks below read data with standard-library Python and disable bytecode writes.
- Only this report was written.

**Which homogeneous experiment does the code represent?**

| Experiment | Design | Count | Evidence |
| --- | --- | ---: | --- |
| Fixed Nano control in the combined multi-agent generator | Every seat uses `gpt-5-nano`, with two seed replicates per grid cell | 130 | [Control construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:753), [count assertions](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:77) |
| Original random monoculture batch | Each game receives five sampled models, with one run per model and grid cell | 325 | [Generator loop](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:398), [saved manifest](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/manifest.json:42) |
| Paper homogeneous cohort | The original random monoculture batch with 25 Game 1 Claude 3 Haiku runs excluded | 300 | [Current paper roster](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:199), [raw-data selection](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/scripts/recreate.py:43) |

- The paper profile must preserve the 300-run cohort and its exclusion record.
  - The original source IDs are `config_0026` through `config_0325`.
  - A newly sampled 325-run batch is a different dataset.
  - The 130 fixed Nano controls are a different dataset.
- The current random monoculture generator has no option to choose one model, one game, one group size, or the 300-run paper selection.
  - Its `generate`, `validate`, `run-one`, `submit-selection`, and `summary` subcommands are defined in the [actual parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:874).
  - Its validator requires exactly 325 configs and five models in every grid cell at [lines 503 onward](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:503).

**What are the paper's roster and seed rules?**

| Game | Models retained in the paper | Runs per model | Total |
| --- | --- | ---: | ---: |
| Game 1 | `gpt-5-nano-high`, `qwen3-max-preview`, `claude-opus-4-5-20251101`, `gemini-3.1-pro` | 25 | 100 |
| Game 2 | `amazon-nova-pro-v1.0`, `gpt-4o-2024-05-13`, `o3-mini-high`, `gpt-5.2-chat-latest-20260210`, `claude-opus-4-6` | 20 | 100 |
| Game 3 | `amazon-nova-micro-v1.0`, `deepseek-v3`, `deepseek-r1-0528`, `claude-opus-4-5-20251101-thinking-32k`, `gpt-5.4-high` | 20 | 100 |

- The table agrees with the [current paper](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:210) and the [saved model assignment file](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/configs/model_assignments.csv:1).
- All games use group sizes `2, 4, 6, 8, 10`.
- The shared [game grid function](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572) supplies the remaining parameters.
  - Game 1 uses `competition_level` in `0, 0.25, 0.5, 0.75, 1` and `num_items = 2.5 * n`.
  - Game 2 uses ten issues, `theta` in `0.2, 0.8`, and `rho` equal to the group-dependent lower bound or `0.9`.
  - The lower bound is `(6 / pi) * asin(-1 / (2 * (n - 1)))` at [the seed and grid helpers](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:299).
  - Game 3 uses `sigma` in `0.2, 0.5`, `alpha` in `0.2, 0.8`, `m_projects = 2.5 * n`, costs from 10 to 30, own-contribution visibility, commit voting, and a time discount of `0.9`.
- The monoculture [config builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:184) fixes ten rounds, two discussion turns, a `0.9` discount, parallel independent phases, and one run per cell.
- Model selection uses seed `20260628`, excludes `claude-sonnet-4-20250514`, sorts the remaining 23 models by Elo and name, and splits them into bands of `5, 5, 5, 4, 4` models.
  - The [selection function](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:92) samples three models from each band and assigns one to each game.
  - A read-only reimplementation using the saved CSV reproduced all 15 saved assignments.
- Each run seed is a SHA-256-derived integer from the selection seed, `random_monoculture_control`, game, group size, competition ID, and model name.
  - The derivation appears in the [monoculture builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:207) and [shared hash helper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:303).
  - Independent recomputation matched all 325 saved run seeds.
  - Changing the model changes the run seed, so replacing a model is not a matched rerun under this formula.
- A read-only inventory found 325 source configs and all 325 corresponding result files.
  - Every saved result's `models` list contains only the model declared by its source config.
  - File presence and roster agreement do not establish valid model actions or current provider availability.

**Which current commands are established by source inspection?**

- The following is a generic four-agent Game 1 invocation with one OpenRouter model in every seat.
  - It requires an exported `OPENROUTER_API_KEY` or a supported configured key pool.
  - The paths are concrete examples for this workspace.
  - The command was not executed and inherits the current runtime behavior discussed below.

```bash
OPENROUTER_TRANSPORT=direct /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py \
  --models gpt-5-nano-high gpt-5-nano-high gpt-5-nano-high gpt-5-nano-high \
  --game-type item_allocation --num-items 10 --competition-level 0.5 \
  --max-rounds 10 --discussion-turns 2 --gamma-discount 0.9 \
  --random-seed 20260907 --max-tokens-per-phase 16384 \
  --parallel-phases --model-order homogeneous_sample \
  --output-dir /tmp/bargain-homogeneous-example
```

- The generic runner accepts repeated model choices at [its model parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:51), arbitrary order labels at [its order parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:302), and a single experiment at [its dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:716).
- The following existing command prepares the historical 325-config design in a new location.
  - It requires the named local pool CSV and writes configs without starting a negotiation.
  - It does not recover the historical runtime overrides listed below.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py generate \
  --results-root /tmp/bargain-monoculture-new \
  --pool-csv /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/configs/heterogeneous_subset_maps/model_pool_24.csv \
  --seed 20260628
```

- After generation, the current single-config syntax is established by [the parser and dispatcher](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:617).

```bash
OPENROUTER_TRANSPORT=direct /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py run-one \
  --results-root /tmp/bargain-monoculture-new --config-id config_0026
```

- The current `submit-selection --selection-name all` path submits the full 325-config Slurm array.
  - The default selection is the 15-config `derisk` set at [the submission parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:896).
  - `submit-selection --dry-run` still writes task, submission, and Slurm files at [the implementation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:749).
  - No existing single command both creates an arbitrary homogeneous sample and manages its batch output.

**How does a config reach the game and its result?**

| Stage | Current implementation | Relevant behavior |
| --- | --- | --- |
| Pool and assignment | [Pool reader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:58), [selector](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:92) | CSV contains model names and Elo values, not provider settings |
| Config generation | [Builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:184), [writer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:398) | Saves configs, index, assignments, bands, manifest, and selections |
| Monoculture adaptation | [Runtime adaptation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:617) | Applies two environment-based override mechanisms and converts `config_0026` into integer ID `26` |
| Shared subprocess runner | [Command builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535), [runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656) | Translates game parameters into CLI arguments, passes full config through `EXPERIMENT_RUN_METADATA_JSON`, and saves status and attempt logs |
| Experiment setup | [CLI config construction](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:551), [game construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119) | Translates `num_items` to `m_items` and `max_rounds` to `t_rounds`, then creates the selected game environment |
| Agents and phase limits | [Agent factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:60), [effective token limits](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1025) | Resolves the current model catalog plus optional model overrides and creates neutral `Agent_1` through `Agent_n` seats |
| Persistence | [Result assembly](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:897), [file writer](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:204) | Saves `experiment_results.json`, interaction files, and the run config |
| Existing analysis | [Config-based loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_random_monoculture_gini_vs_heterogeneous.py:106), [raw paper loader](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/scripts/raw_builder_source.py:215) | General batch analysis and paper reproduction apply different cohort checks |

**What blocks a fresh user or changes a historical rerun?**

- The default pool is inside an ignored historical experiment directory.
  - The default is defined at [the monoculture constants](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:34).
  - `git ls-files` returned no tracked entries for that pool, the historical manifest, or the historical config directory.
  - The [ignore rule](/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:60) excludes the experiment tree.
  - A fresh Git checkout therefore lacks the generator's default data dependency.
  - The saved pool has columns `pool_index,model,arena_elo` and SHA-256 `1bcdd913e01e104682cb7674d3dd0c0c4d25bae93a386c6085191b6cbc72aca7`.
- A generic homogeneous game needs the runtime dependencies, credentials, and an output directory but no historical pool or Elo table.
  - [Runtime requirements](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1) use minimum versions rather than a pinned environment.
  - The models in this paper cohort use APIs, so the homogeneous launcher does not require a GPU or model download.
- The current catalog requires three credential types for the fixed 300-run paper assignment.
  - None of its source configs or result configs contains a provider override.
  - The table describes current routing for these fixed model labels, without proving which provider handled every historical request.

| Required key | Assigned model labels | Models | Paper runs | Current registry evidence |
| --- | --- | ---: | ---: | --- |
| `OPENAI_API_KEY` | `gpt-4o-2024-05-13`, `o3-mini-high`, `gpt-5.2-chat-latest-20260210` | 3 | 60 | [GPT-4o](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1123), [o3-mini](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:727), [GPT-5.2](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:604) |
| `ANTHROPIC_API_KEY` | `claude-opus-4-5-20251101`, `claude-opus-4-6`, `claude-opus-4-5-20251101-thinking-32k` | 3 | 65 | [Opus 4.6](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:431), [Opus 4.5 variants](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1048) |
| `OPENROUTER_API_KEY` | `gpt-5-nano-high`, `qwen3-max-preview`, `gemini-3.1-pro`, `amazon-nova-pro-v1.0`, `amazon-nova-micro-v1.0`, `deepseek-v3`, `deepseek-r1-0528`, `gpt-5.4-high` | 8 | 175 | [Nano](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275), [Gemini](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:345), [Qwen](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1081), [Nova Pro](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:818), [Nova Micro](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1206), [DeepSeek V3](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:737), [DeepSeek R1](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:655), [GPT-5.4](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1239) |

- Every model in that matrix uses a hosted API, so the complete paper homogeneous profile needs no local GPU.
- A generic homogeneous run with a chosen hosted model needs only the chosen model's provider key.
  - `gpt-5-nano` needs `OPENAI_API_KEY` at [its registry entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265).
  - `gpt-5-nano-high` needs `OPENROUTER_API_KEY` at [its registry entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275).
  - Repeating the chosen model across more seats does not add a provider dependency.
- Credential validation occurs before the run at [the CLI check](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398).
- Existing grouped and numbered key support is defined by [the key discovery module](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:88).
- The batch runner silently selects the Princeton file proxy unless `OPENROUTER_TRANSPORT` is explicitly set.
  - It defaults the queue to `/home/jz4391/openrouter_proxy` at [the child environment](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).
  - Its Slurm template fixes the CPU partition, modules `anaconda3/2024.2` and `proxy/default`, the same queue path, and a module-provided `python` at [the template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:680).
  - A public local profile should choose direct transport explicitly, while a Della profile should use the configured shared queue and its externally managed monitor.
  - Native OpenAI and Anthropic calls still need an allowed native API network route because the OpenRouter queue does not automatically carry those native requests.
- Credentials are loaded differently by the batch wrapper and its Slurm template.
  - The wrapper merges the repository's `/scratch/gpfs/DANQIC/jz4391/bargain/.env` into the child environment at [the loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1521).
  - The Slurm script optionally sources `BARGAIN_API_KEYS_ENV`, whose default resolves under this workspace to `/scratch/gpfs/DANQIC/jz4391/bargain/bargain/api_keys.env`, at [the template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:696).
  - A public guide should use one documented environment-based onboarding path and list variable names without storing credential values in configs.
- A model label does not freeze the endpoint or reasoning settings.
  - `qwen3-max-preview` maps to `qwen/qwen3-max` at [its entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1081).
  - `gemini-3.1-pro` maps to `google/gemini-3.1-pro-preview` at [its entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:345).
  - `gpt-5.2-chat-latest-20260210` maps to the unversioned `gpt-5.2-chat-latest` endpoint at [its entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:604).
  - `deepseek-v3` maps to `deepseek/deepseek-chat` at [its entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:737).
  - The public report must preserve both the requested label and the effective endpoint without claiming that a current endpoint reproduces a historical snapshot.
- The current `o3-mini-high` entry has no `reasoning_effort` setting at [its catalog entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:727).
  - The native OpenAI request builder supplies `reasoning_effort="low"` when the setting is absent at [the request code](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840).
  - The paper labels this model as high reasoning at [the roster](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:217).
  - Historical reasoning effort remains unresolved because none of the 325 saved configs or result configs contains `model_config_overrides`.
  - Changing this entry to high would change current execution and must not be presented as an unchanged historical replay.
- Experiment token limits do not fully describe effective API limits.
  - The current shared limit is 16,384 tokens at [the constants](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:3).
  - The catalog applies smaller limits to GPT-4o and Nova at [the model-specific constants](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:6).
  - GPT-5.4 high can use 65,536 tokens when the experiment requests the standard limit at [its catalog entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1239) and [the resolver](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1049).
  - The Anthropic factory also expands the output allowance for the 32,000-token thinking budget at [its construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:377).
  - A public resolved config must include effective per-agent phase limits and provider reasoning settings.
- Three historical DeepSeek V3 runs retain token changes that a fresh call to `generate` will omit.

| Historical ID | Saved nonstandard limits | What its source config retains |
| --- | --- | --- |
| `config_0254` | Discussion `4096` | The discussion change and backfill note at [the source config](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/configs/config_0254.json:111) |
| `config_0258` | Discussion, proposal, voting, reflection, and thinking `4096` | Only discussion at [the source config](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/configs/config_0258.json:119), with the other limits saved in [the result config](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/runs/config_0258_game3_n8_sigma_0p2_alpha_0p2_deepseek_v3/experiment_results.json:165) |
| `config_0262` | Proposal, voting, reflection, and thinking `4096` | No phase limit in the source config, with all four changes saved in [the result config](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/runs/config_0262_game3_n10_sigma_0p2_alpha_0p2_deepseek_v3/experiment_results.json:162) |

- The wrapper reads `RMC_RUNTIME_MAX_TOKENS_PER_PHASE` and `RMC_RUNTIME_CONFIG_OVERRIDES_JSON` at [its run dispatcher](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:620).
  - The former changes proposal, voting, reflection, and thinking limits but leaves discussion and default limits unchanged.
  - These ambient settings must become explicit resolved inputs in a public profile.
- Copying a historical config directory to a new batch root does not relocate its output directory.
  - Relative `output_dir` values resolve against the repository root at [the path helper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:220).
  - A copied config can therefore point back to an original result tree, such as [the saved Nano config](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_random_monoculture_control_20260628_014357/configs/config_0026.json:37).
- The runner can reuse and modify an old result before a new subprocess starts.
  - It accepts an existing result after a check that does not compare models, providers, phase limits, or full completion integrity at [the validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269).
  - It then enriches the old file with the supplied config metadata at [the reuse branch](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668).
  - `submit-selection --rerun-existing` selects existing configs but does not disable this reuse branch at [submission selection](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:739).
- Current production behavior also prevents an unqualified public launch recommendation.
  - The native provider route can switch to OpenRouter after provider failures at [the route sequence](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:741).
  - Invalid proposals can become synthetic proposals after repair attempts at [the phase handler](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2397).
  - A safe public path must fail on these cases unless a checked-in study specification explicitly permits a bounded recovery that complies with the project rules.

**What is the smallest useful public adapter?**

- Add a `homogeneous` branch to the shared public launcher with `--model`, `--agents`, `--game`, `--seed`, game parameters, `--output-dir`, and explicit transport selection.
  - The command below is proposed syntax and does not exist today.
  - The shared launcher filename can be aligned with the other family adapters.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/launch_experiment.py homogeneous \
  --model gpt-5-nano-high --agents 4 --game game1 \
  --competition-level 0.5 --seed 20260907 --transport direct \
  --output-dir /tmp/bargain-homogeneous-example
```

- Extract a pure `build_single_homogeneous_config(...)` helper from the core of [the existing monoculture builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:184).
  - Make the helper construct `models = [model] * n_agents`, seat IDs, game parameters, and explicit runtime settings.
  - Give new samples their own declared experiment family, such as `homogeneous_sample`, while retaining `monoculture_model` for downstream tools.
  - Keep Elo-band selection and historical pool metadata in the historical generator.
  - Do not invent Elo values, sample indices, or pool assignments for a user-selected model.
- Reuse `game_parameter_grid`, `game_type_for_label`, `runtime_config`, and the command translator after validation.
  - Keep the shared game's mathematical parameter definitions in [the existing grid helper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572).
  - Do not call the full 2,730-config generator to prepare one homogeneous run because it builds heterogeneous subset maps at [its setup](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:741).
  - Do not reuse `common_config` unchanged because it fixes the Nano baseline and gives unfamiliar families heterogeneous roles at [its metadata builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:666) and [agent map builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:517).
- Add a distinct named profile for `paper-homogeneous-300`.
  - Bundle the nonsecret 24-model source pool, the saved assignments, original config IDs and seeds, 300 included IDs, 25 excluded IDs, and per-run resolved token overrides outside the ignored experiment directory.
  - Store the exclusion reason and profile version explicitly.
  - Keep original IDs and seeds when generating new output paths.
  - Preserve a separate 325-run historical sensitivity profile if it is offered.
  - Reject incompatible sample options when a paper profile is selected.
- Give generic single-config validation and paper-cohort validation separate functions.
  - Retain the existing 325-run validator for its original batch.
  - Validate the paper profile's exact roster, grid, counts, seeds, and inclusion set before launch.
- Require a new output root for a new run and compare complete config fingerprints for an explicitly requested resume.
  - Resolve every output path under the selected root before execution.
  - Never enrich or overwrite an existing result as part of a new run.
  - Save the resolved config, code revision, source file hashes, model catalog snapshot, effective request settings, transport, and bounded retry policy before the first call.
  - Reject malformed metadata instead of using the CLI's current warning-and-ignore behavior at [metadata parsing](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:388).
- Make the shared runtime's error behavior a prerequisite for advertising this adapter as ready.
  - Add explicit strict provider and invalid-action policies to the provider factory, provider route sequence, and phase handler.
  - Verify that the number and identity of created agents match every requested seat before play starts.
  - Check synthetic action markers, full roster identity, required result fields, and resolved config identity before a run can be marked successful.
- Keep local execution and Della submission in separate transport and scheduler profiles.
  - Local execution should use the selected interpreter and direct API access without cluster modules.
  - Della submission should accept explicit module, partition, interpreter, log, and queue settings without inspecting or restarting the externally managed queue monitor.

**Which analysis inputs must the adapter preserve?**

- The existing config-based homogeneous loader needs a batch `configs` directory, usable `output_dir` values, `config_id`, `game_label`, `n_agents`, and `monoculture_model` at [the loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_random_monoculture_gini_vs_heterogeneous.py:106).
  - Its CLI requires 325 homogeneous results and 1,300 heterogeneous results at [the count checks](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/plot_random_monoculture_gini_vs_heterogeneous.py:427).
  - It cannot serve as a generic single-run analysis command or a 300-run paper-profile command without explicit profile-aware count checks.
- The paper raw loader scans saved results directly and requires `monoculture_model`, `model_elo`, group size, game label, and final utilities at [the raw loader](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/scripts/raw_builder_source.py:215).
  - Its wrapper expects all 325 historical results before removing Haiku at [the paper selection](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_07/scripts/recreate.py:43).
  - A new 300-only export needs an explicit included-ID loader with matching coverage checks.
- The public adapter should emit a small profile-aware summary of each run without requiring the historical heterogeneous corpus.
  - Optional paper comparisons can request that corpus and its analysis index as explicit inputs.
  - Missing or failed runs must remain visible as incomplete instead of disappearing through the loader's `continue` branches.

**What should implementation verify?**

- Test pure parsing and config construction without credentials, network access, runtime imports, or filesystem writes.
- Test that an arbitrary user-selected model produces exactly `n` identical model labels and neutral seat IDs.
- Test game parameter translation and effective per-agent phase limits for each of the three games.
- Use the retained source pool as an input fixture to verify all historical assignments, 325 original seeds, the 300 retained IDs, and the 25 excluded IDs.
- Test the three DeepSeek V3 runtime exceptions separately from the original generator defaults.
- Test missing pools, malformed configs, unsupported model labels, invalid transport values, and incompatible profile options as errors.
- Test that moving configs cannot read, reuse, or modify the original result tree.
- Test explicit resume against changed model, provider, reasoning, phase limit, seed, prompt version, and game parameters.
- Test that provider failures and invalid model actions cannot create synthetic research results or change the provider under the strict public policy.
- Test the current `o3-mini-high` effective request before resolving its paper-label discrepancy.
- Test profile-aware analysis with a complete cohort, an incomplete cohort, and the wrong included-ID set.
- Run a separately authorized real API smoke test after implementation because this review establishes source behavior only.

**What remains unresolved?**

- The exact historical provider versions and reasoning settings are not recoverable from the examined source and result configs alone.
- The intended correction for the `o3-mini-high` label needs evidence from the historical request records or study specification.
- Current endpoint availability was not checked because no provider calls were authorized for this review.
- The public release must decide where to distribute the nonsecret pool, paper profile, and historical resolved configs because Git currently ignores their source locations.
- Exact numerical replay is not established by repeating seeds and model labels when model endpoints, runtime behavior, and effective token settings can differ.
