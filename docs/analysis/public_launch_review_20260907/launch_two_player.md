# Two-player GPT-5-nano launch review

**Question**

Can a new user launch the primary GPT-5-nano two-player experiment across Games 1–3 with one command?

**Short answer**

The shared runner supports all three games, but the repository needs a small public launch layer and several launch fixes before it can offer a reliable one-command workflow.

- This review inspected current source, argument parsers, selected saved configurations, and the retained primary index on 2026-09-07.
- It did not run an experiment, call an API, submit a job, inspect credential values, change an environment, or change source code.
- The three shell generators passed `bash -n` separately.
- Read-only syntax-tree inspection confirmed that each generator lists 30 valid model aliases with 5 Anthropic routes, 20 OpenRouter routes, and 5 OpenAI routes.
- Counting the retained primary CSV found 420 Game 1 rows, 540 Game 2 rows, and 540 Game 3 rows.
- Runtime integration remains unvalidated.

**What works in the current implementation?**

- One runner already selects the three game engines through `--game-type item_allocation|diplomacy|co_funding`.
  - The parser accepts model aliases, game parameters, an explicit seed, phase limits, output directory, and run identifiers in [the runner parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:51).
  - `StrongModelsExperiment.run_single_experiment()` calls the game factory with the selected game parameters in [the engine dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:196).
  - The game factory is shared by all three games in [the game environment package](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/__init__.py:44).
- Separate shell generators create the existing two-player batch layouts.
  - Game 1 uses the GPT-5-nano baseline, 30 adversary models, seven competition values, two model orders, and discussion lengths of one and two turns in [the Game 1 grid](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:70).
  - Its current grid therefore creates 840 configurations, including 420 primary two-turn configurations.
  - Game 2 `--conservative` creates a 30-model × 3-rho × 3-theta × 2-order grid in [the Game 2 mode](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:220).
  - Game 3 `--conservative` creates a 30-model × 3-alpha × 3-sigma × 2-order grid in [the Game 3 mode](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:272).
- The complete current roster needs no local model weights or GPU.
  - Its providers are OpenAI, Anthropic, and OpenRouter, as resolved from [the model catalog](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:65) and the model lists in [Game 1](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:78), [Game 2](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:150), and [Game 3](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:202).
  - For example, the standard Llama aliases use OpenRouter in [the Llama 3.3 entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:778) and [the Llama 3.2 1B entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1217).
- The runner checks required provider credentials before it creates the experiment directory in [the credential check](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398).
  - A full current roster needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY`, or the supported provider key pools.
  - A pair with GPT-5-nano and `gpt-4o-mini-2024-07-18` needs only the native OpenAI credential under its selected routes.
  - This check establishes credential presence, not account access to each model.

**Which current commands can be documented accurately?**

- The commands below were established from source inspection and were not executed.
- There is no `--config` argument in the low-level runner.
- A current one-cell Game 1 command is the following.

```bash
OPENAI_TRANSPORT=direct OPENROUTER_TRANSPORT=direct OPENROUTER_PROVIDER_FALLBACK=0 \
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py \
  --models gpt-5-nano gpt-4o-mini-2024-07-18 \
  --game-type item_allocation --num-items 5 --competition-level 0.5 \
  --max-rounds 10 --discussion-turns 2 --gamma-discount 0.9 \
  --random-seed 42 --model-order weak_first --max-tokens-per-phase 16384 \
  --batch --num-runs 1 --run-number 1 --job-id 0 \
  --output-dir /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/public_example_game1
```

- The native OpenAI key must already be exported for that command.
- These game arguments replace the Game 1 arguments for a different one-cell run.

| Game | Current game arguments |
| --- | --- |
| Game 1 | `--game-type item_allocation --num-items 5 --competition-level 0.5` |
| Game 2 | `--game-type diplomacy --n-issues 5 --rho 0.0 --theta 0.5` |
| Game 3 | `--game-type co_funding --m-projects 5 --alpha 0.5 --sigma 0.6 --c-min 10 --c-max 30 --cofunding-discussion-transparency own --cofunding-time-discount 0.9` |

- Each game must use its own new output directory.
- For adversary-first execution, reverse `--models` and set `--model-order strong_first`.
  - The label alone does not reverse the supplied model list in [the order handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:166).
- Existing configuration-generation syntax is as follows.

```bash
bash /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh
bash /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh --conservative
bash /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh --conservative
```

- Each generator creates files and changes a latest-result link, so configuration generation is a write operation.
  - The link changes appear in [Game 1](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:214) and [Game 2](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:481).
- Game 2 and Game 3 generate a local single-configuration shell wrapper, but generation and execution are separate commands.
  - The local Game 2 wrapper derives the checkout root and activates its environment in [the local wrapper template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:815).
- These commands describe current syntax and do not establish that a newly generated full batch is ready to run.

**What blocks a fresh user?**

- **The generated Slurm launch paths are wrong for normal repository-root submission.**
  - Game 1 writes `BASE_DIR="bargain"`, changes into it, and then sources `bargain/.venv/bin/activate` again in [the CPU job template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:367).
  - The submit script already changes to the repository root in [its path setup](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:753).
  - Games 2 and 3 repeat the same relative-root problem in [Game 2](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:619) and [Game 3](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:696).
  - The templates also require Princeton module names and place scheduler logs under `logs/cluster` in [the Game 1 resource template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:354).
- **Several advertised Game 2 modes do not build an experiment.**
  - The parser accepts `--small`, `--model-scale`, and `--scaling` in [the argument dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:58).
  - The model grid has no populated branch for those modes in [the model-scale branches](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:220).
  - The TTC grid only populates `full`, so those modes also receive empty TTC arrays in [the TTC branches](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:273).
  - `--derisk` retains `NUM_RUNS=2`, despite the help text calling it one configuration in [the defaults](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:130) and [the derisk branch](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:232).
- **Game 1 documents a local helper that it does not generate.**
  - Its opening instructions promise `submit_single.sh 0 --local` in [the documented workflow](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:24).
  - Inspection of the complete generator found no file-generation block for that helper.
  - The actual generated submission interface is `submit_all.sh` in [the submission template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:742).
  - Game 1 also has no argument parser before it creates its output directory in [the generator startup](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:48).
- **Game 3 duplicates execution when a generated configuration has `num_runs > 1`.**
  - The generator already creates one configuration per `run_num` and saves both `run_number` and the total `num_runs` in [the configuration loop](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:445).
  - Its execution wrapper then passes both `--num-runs NUM_RUNS` and `--run-number RUN_NUMBER` in [the generated command](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:747).
  - The inner runner keeps the same run number and seed for every iteration when an override is present in [the batch loop](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:992).
  - The file manager caches the same output filename for that run number and opens it for writing in [filename selection](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150) and [result saving](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:204).
  - This affects default, small, ambitious, and derisk grids that use several runs, while the conservative grid uses one run.
- **Some saved parameters never reach the existing command.**
  - Game 1 stores `num_items` and `max_rounds` but does not pass them in [its job command](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:453).
  - Game 2 stores `max_rounds`, `gamma_discount`, and `run_number` but omits the round and discount arguments and hard-codes run number 1 in [its job command](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:681).
  - Game 2 special modes set five rounds, but this command therefore still uses the runner's ten-round default in [the special-mode settings](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:244) and [the runner default](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:72).
  - The direct Game 3 CLI defaults `c_max` to 50, while the Game 3 generator and configuration class use 30 in [the CLI](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:160), [the generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:144), and [the configuration tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_cofunding_config.py:19).
- **Local and cluster transport choices are inconsistent.**
  - OpenRouter `auto` tries direct HTTPS before the queue, despite the opposite wording in the Game 2 template, according to [the client contract](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:1) and [the template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:637).
  - The OpenRouter queue defaults to `/home/jz4391/openrouter_proxy` in [its configuration](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:33).
  - Native OpenAI has separate transport variables and automatically uses the same shared queue under Slurm in [OpenAI transport selection](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686) and [Slurm detection](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742).
  - A portable launch must select native-provider transport as well as OpenRouter transport.
- **The current provider recovery policy is not suitable as an implicit public protocol.**
  - Native requests can switch to OpenRouter by default in [the recovery sequence](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:688).
  - Call-level recovery records provider metadata in [the recovery result](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:675).
  - `OPENROUTER_PROVIDER_FALLBACK=0` disables that call-level route, but the agent factory's construction-time recovery does not test that setting in [agent creation](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183).
  - A public profile must fail when its selected provider fails unless the profile explicitly permits a recorded recovery route.
- **Exit status and output selection can hide incomplete or repeated work.**
  - A batch with some failed runs returns its successful subset after a warning in [batch failure handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1026).
  - The CLI only rejects zero successful runs and otherwise returns 0 in [the CLI result handling](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:706).
  - The CLI's default directory name has no timestamp or seed in [output naming](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:640).
  - Repeated processes may create a timestamp-suffixed result while retaining the original result, and the analysis loader prefers the original base filename in [file selection](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:232).
- **Onboarding and analysis still assume this research checkout.**
  - The root README contains result figures but no installation or launch workflow in [the current README](/scratch/gpfs/DANQIC/jz4391/bargain/README.md:1).
  - The requirements file mixes execution, plotting, UI, and test packages and provides lower bounds rather than a reproducible environment in [the requirements](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1).
  - The direct runner and legacy shell launch paths do not load a dotenv file, while the newer multi-agent wrapper does so in [its credential loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1521).
  - The bilateral analysis hard-codes old source roots and a dated output directory in [its baseline specifications](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:50).
  - Its default main also loads both GPT-5-nano and Llama batches in [the analysis entry point](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:1723).

**What is the smallest useful public interface?**

- Add a shared launch module with a two-player subcommand and a checked-in profile for the current protocol.
- The following is proposed syntax and is not implemented.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  -m strong_models_experiment.launch two-player \
  --baseline gpt-5-nano --adversary gpt-4o-mini-2024-07-18 \
  --games all --profile primary-current-v1 --executor local \
  --output /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/public_nano_run
```

- With one adversary, the primary grid has 50 cells across the three games.
- A named, frozen 30-model roster can replace the one-adversary selection to create 1,500 cells.
- A separate explicit smoke profile should use a declared small grid and save its changed round and token limits.
- A read-only `--dry-run` should print the selected cells, required credential names, resolved model/provider settings, and output paths without creating files or requiring credentials.
- `--executor local` should use the current interpreter and explicit direct API transport on a machine with network access.
- `--executor slurm --cluster-profile della` should submit the same saved configurations through a cluster adapter.
  - The Della adapter should use explicit queue settings, absolute interpreter and checkout paths, scheduler logs under the run's `slurm/` directory, and the declared concurrency limit.
  - Other clusters should provide their own adapter settings rather than inherit Princeton module names or queue locations.
- An explicit resume mode should validate configuration and result hashes before it skips any cell.
- A changed model, provider, protocol, or seed should create a new run identity.

**Which code should be reused or changed?**

| Proposed patch | Smallest implementation direction | Existing source to use |
| --- | --- | --- |
| Shared two-player configuration builder | Extract the three pure grid builders and pass a baseline, frozen roster, protocol profile, output root, and explicit seed mapping as arguments. | [Llama Game 1 builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:70), [Game 2 builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:110), [Game 3 builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:149) |
| One-cell execution function | Add a normalized configuration API that validates all required fields and executes exactly one manifest row. | [Shared engine](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119), [current configuration assembly](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:551) |
| CLI/config mapping | Extract a side-effect-free parser and one configuration-to-argument-list function so local and Slurm paths forward identical parameters. | [Existing argument-list builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535) |
| Recorded model resolution | Reuse catalog resolution and save the resolved model ID, provider, effort, temperature, phase caps, and transport for each seat before execution. | [Model override resolver](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:59) |
| Attempt tracking | Reuse attempt identifiers, log capture, subprocess return codes, and result checks after making paths and policies explicit arguments. | [Attempt execution](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656) |
| Strict completion | Report requested, completed, failed, and missing cells separately, and return nonzero when the requested set is incomplete. | [Current batch failure branch](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1026) |
| Strict provider policy | Make both call-time and construction-time provider recovery obey the saved policy. | [Call-time recovery](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:578), [construction-time recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183) |
| Analysis entry point | Accept a manifest, selected baseline, explicit output directory, and optional frozen Elo table, then reuse the current metric functions. | [BaselineSpec](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:54), [row loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:775) |

- The newer multi-agent wrapper must not be reused unchanged for this protocol.
  - Its command builder always adds `--parallel-phases` in [command construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1563).
  - Its execution function defaults to the private queue in [environment construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).
  - It automatically accepts an existing result in [existing-result handling](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668).
  - Its result validator permits absent identifiers and checks only a subset of configuration identity in [result validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1303).
- The Llama builder supplies reusable grid logic, but its fixed ten-adversary roster and profile values must become inputs rather than global variables.
- Keep the engines, phase handlers, prompts, and provider clients shared across public experiment families.
- The legacy shell scripts can later delegate to the common builder and executor after compatibility checks.

**What must stay explicit for statistical and historical compatibility?**

- Preserve the distinction between the baseline alias and the high-effort adversary alias.
  - `gpt-5-nano` currently selects native OpenAI in [the baseline entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265).
  - Its low effort is applied inside the API client rather than declared in that catalog entry in [OpenAI request construction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2835).
  - `gpt-5-nano-high` selects OpenRouter high effort in [the adversary entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275).
  - Analysis canonicalization maps `gpt-5-nano` to `gpt-5-nano-high`, so this display mapping must never resolve runtime provider or reasoning settings in [the analysis aliases](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:61).
- Preserve Game 1's shared scenario seed across the two orders.
  - The current two-run grid uses seed 42 for both orders, although three seed values remain in the script, according to [the seed settings](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:119) and [the order loop](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:167).
  - Two orders are not two independent scenario seeds.
- Preserve the separate Game 2 and Game 3 seed schedules when reproducing the saved design.
  - Both generators use `42 + experiment_id` in [Game 2](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:376) and [Game 3](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:451).
  - Opposite orders therefore have different scenario seeds, as shown by seeds 42 and 43 in [the saved Game 2 index](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215/configs/experiment_index.csv:2).
  - Reordering or subsetting the roster must not regenerate historical seeds from new configuration positions.
- Fix the seed-zero truthiness check without changing other seed rules.
  - The runner seeds Python's random module only when the seed is truthy in [its seed handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:161).
  - Change that condition to `is not None`, then verify random-order behavior with seed 0.
  - Scenario seeds do not guarantee identical remote API responses.
- Give fresh execution a versioned current profile instead of claiming an exact historical replay.
  - Selected saved Game 2 and Game 3 configurations use 10,500 tokens per phase in [the saved Game 2 configuration](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/diplomacy_20260405_082215/configs/config_0000.json:11) and [the saved Game 3 configuration](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/cofunding_20260405_083548/configs/config_0000.json:11).
  - The current generators use 16,384 tokens per phase in [Game 2](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:134) and [Game 3](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:140).
  - Current `gemini-3-pro` resolves to the Gemini 3.1 preview route in [the current alias](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:355).
  - Historical replay must use selected saved cells and record any unavailable model, changed provider, prompt revision, parameter difference, or missing provenance.
- Save both requested and effective configuration details before a request is made.
  - Include source revision, source-tree state or source hashes, dependency versions, profile version, per-cell seed, ordered seat assignments, resolved model settings, and output identity.
  - The current enhanced configuration copies the run settings and game state but does not snapshot the model catalog or code revision in [enhanced configuration construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:47).
- Keep public analysis independent of the historical 1,500-row requirement.
  - The primary filter selects two-turn Game 1 rows in [the protocol filter](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:1331).
  - The paper overview requires exactly 1,500 unique files and 30 models in [the figure validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/render_figure2_large_fonts.py:55).
  - A valid 50-cell or smoke launch therefore needs a general summary path that reports missing cells without imposing paper-only counts or writing paper assets.
  - Analysis must read the manifest's exact result paths instead of choosing an old or alternate result filename.

**What verification is required before release?**

- Test pure grid generation for the 14/18/18 one-adversary counts and the 420/540/540 complete current-profile counts.
- Verify seed-to-cell mappings, both ordered model lists, same-alias seat identity, and the primary two-turn Game 1 selection.
- Test that dry-run and help paths need no credentials, import no provider runtime, and create no files.
- Test exact configuration-to-command forwarding for all three games, including round limits, cost range, discount settings, phase caps, run number, and disabled phases.
- Reject zero cells, unknown model aliases, invalid model counts, malformed metadata, invalid seeds, incompatible game arguments, and unsupported provider settings before execution.
- Test that local and Slurm adapters use the same resolved configuration and handle checkout paths with spaces.
- Test that one manifest row produces one attempt with one seed and one run identifier.
- Test refusal of accidental output reuse, explicit resume validation, stale results, partial completion, and nonzero failure exit status.
- Use unit-test doubles only at API and scheduler boundaries, then run an authorized real local smoke experiment before claiming local integration works.
- Run a separately authorized real Della queue smoke experiment before claiming the Slurm adapter works.
- Analyze those new smoke outputs with the manifest-based loader and verify requested-versus-completed counts and exact result provenance.

**What remains unresolved?**

- Current account access and current availability of every catalog route were not tested.
- The final shared CLI name should match the interfaces proposed for the other experiment families.
- Exact historical replay requires a frozen model/provider and code record for each selected cell, beyond the legacy configuration fields inspected here.
- A non-Della cluster needs a declared credential and network route for each selected native provider.
- The public package needs a chosen minimum Python version and a verified dependency set for local execution.
