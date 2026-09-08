# Public experiment commands and portability review

**Question**

How should another user launch each paper experiment type with one command, and what must change before the code supports that safely?

**Short answer**

Add one installed `bargain` command with seven experiment presets, one validated run specification, and the existing game engine underneath it.
Keep small new experiments separate from fixed paper sweeps.
The current code is not ready for a thin public wrapper because some launch paths can change settings, providers, or result identity without an explicit user choice.

- This review used 13 agents, with one agent for each experiment family and separate agents for shared execution, credentials, packaging, portability, and validation.
- No experiment code, configuration, paper content, or historical result was changed by this review.
- No experiment, model download, API test, Slurm submission, commit, or push was performed.
- Only review reports were added under `/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907`.
- Source references describe the working files inspected on September 7, 2026, based on commit `444fcf9c368a60dd40a0f5f7aa8b4033552312bf`.
- The working tree already contained unrelated changes and was not treated as a clean release artifact.

## What command interface do I recommend?

All commands in this section are proposed and do not exist yet.

| Experiment | Proposed small-run command | Meaning |
| --- | --- | --- |
| Two-player, GPT-5-nano baseline | `bargain run two-player --game game1 --adversary gpt-4o-mini-2024-07-18` | One adversary model and one fixed Nano baseline. |
| Two-player, Llama baseline | `bargain run two-player-llama --game game1 --adversary gpt-4o-mini-2024-07-18` | The same two-player engine with the hosted Llama 3.3 70B baseline. |
| Homogeneous | `bargain run homogeneous --game game1 --agents 4 --model gpt-5-nano` | Four agents using the same model. |
| Heterogeneous | `bargain run heterogeneous --game game1 --agents 4` | Four distinct models sampled by the named preset's declared rule. |
| Homogeneous-adversary | `bargain run homogeneous-adversary --game game1 --agents 4 --adversary gpt-4o-mini-2024-07-18` | One adversary model with three fixed Nano agents. |
| Test-time compute | `bargain run ttc --game game1 --family gpt5` | Four runs of one game instance, one per requested effort level. |
| Coordinated team | `bargain run team --game game1 --agents 4` | One GPT-5.4 High adversary model against three privately coordinated Nano agents. |

- “Multi-agent” is the umbrella for homogeneous, heterogeneous, and homogeneous-adversary experiments, not an eighth independent design.
  - An optional `paper-multi-agent-v1` sweep can combine those three paper subsets.
  - Keep the paper's `n=2` cells in these multi-agent designs because their game settings differ from the primary two-player sweep.
- Both two-player names should call one builder with different baseline presets.
  - Do not maintain two copies of the engine or parameter translation.
- Use Game 1 as the documented small-run default when `--game` is omitted.
  - The other supported games remain explicit options, while the initial team preset supports only Game 1.
- Each ordinary `run` command should default to one negotiation, one declared seed, and one seat order.
  - I would use seed 42 for the documented examples and save the resolved seat order before execution.
  - TTC is the explicit four-run exception because one effort level alone does not show scaling.
  - The plan must print the exact run count before any paid call.
  - One negotiation can make many paid model requests, so a one-run default is not a one-request cost estimate.
- The example adversary model is an existing catalog entry, not a claim about current price, availability, or suitability.
- Save every short-command default in the resolved plan.
  - This includes game parameters, number of items/issues/projects, seed, discussion turns, round limit, discount, seat order, protocol, reasoning controls, and effective phase limits.
  - Use a documented new-run preset version rather than inheriting defaults from several old scripts.
- Support common options such as `--seed`, `--output`, `--env-file`, and `--profile` across families.
  - A supplied output path must control all new experiment artifacts.
  - An explicit queue profile may use its separately configured private transport directory.
  - Without `--output`, create a unique run directory under the invoking directory and print its absolute path.
  - Never reuse a result directory implicitly.
- The team preset should initially accept only Game 1.
  - A standalone team run must generate and save its own preferences without requiring this user's historical controls.
  - A historical matched rerun must require an explicit, checked control bundle.

### How should full paper sweeps differ?

Use separate, versioned paper presets with a no-call planning mode and an explicit execution option.

```bash
bargain sweep --preset paper-two-player-v1 --plan-only --output /path/to/paper-two-player-plan
bargain sweep --preset paper-two-player-v1 --execute --workers 1 --output /path/to/paper-two-player-run
```

- These example paths are placeholders for directories chosen by the new user.
- `--plan-only` can write a new plan when `--output` is supplied, but it must not create run attempts, contact providers, modify existing plans, or submit jobs.
- Omitting both execution choices should show the plan and stop without spending money.
- `bargain run --plan /path/to/saved-plan/manifest.json` should execute an already checked plan without regenerating its sampling or seeds.
- `bargain doctor <preset>` should check setup without contacting providers.
  - Pass the same model selection as the intended run, or check its saved plan.
- `bargain doctor <preset> --live` should be a separate, explicit network test with a stated call count.
- `bargain status /path/to/run` and `bargain summarize /path/to/run` should work for small and incomplete runs without loading old experiment directories.
- `bargain resume /path/to/run` should reuse only checked complete attempts and record restarted failures as new attempts.
  - Reconcile pending remote requests before restarting an interrupted attempt whose provider outcome is unknown.

## What exists for each family now?

The paper totals come from [the current appendix](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:13).
The table distinguishes the paper's selected runs from the output of today's generators.

| Family | Paper runs | Existing implementation | Main adaptation needed |
| --- | ---: | --- | --- |
| Two-player GPT baseline | 1,500, split 420/540/540 | [Game 1 shell generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:70), [Game 2 generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:220), [Game 3 generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:272) | One pure three-game builder, with the paper's two-discussion-turn Game 1 subset and conservative Game 2/3 grids. |
| Two-player Llama | 500, split 140/180/180 | [Llama generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:70) | Reuse its three builders, expose local execution, and pin the saved settings. |
| Homogeneous-adversary | 1,300 | [Multi-agent controller](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:100) | Generate only this family and validate its selected specification. |
| Heterogeneous | 1,300 | [Multi-agent sampler](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:397) | Keep the verified sampling design and avoid generating unrelated group sizes or families. |
| Homogeneous | 300 | [Random-monoculture controller](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:1) | Separate a chosen-model example from the fixed 300-run paper assignment. |
| TTC | 2,160 | [Native-effort generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:254), [seed replication](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:70), [single-config runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:91) | Combine seeds and effort selection in one plan and preserve attempt-level settings. |
| Team | 100 coordinated reruns | [Binding-team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:65) | Separate protocol construction from mandatory historical-control loading. |

- The two-player Game 1 generator currently produces 840 configurations, including a one-discussion-turn branch outside the primary 420-run subset.
- The multi-agent controller produces 2,730 configurations, including 130 Nano-only controls.
  - The paper instead combines 1,300 homogeneous-adversary runs, 1,300 heterogeneous runs, and 300 replacement homogeneous runs.
  - Calling today's full controller is therefore not equivalent to launching the paper's 2,900-run multi-agent collection.
- The random-monoculture generator produces 325 configurations.
  - The selected 300-run paper collection excludes 25 Game 1 Claude 3 Haiku runs.
  - Freeze the selected assignment rather than asking a fresh random draw to reproduce it.
- The TTC generator produces 216 configurations for seed 42.
  - The paper uses ten seeds and 2,160 runs.
  - Its “native” name does not imply that every model uses its developer's direct API.
- The team experiment and the matched GPT-5.4 homogeneous coalition replication are different studies.
  - The coalition replication has 25 unique settings and 26 saved attempts because one setting was replaced after failure.
  - If exposed, it should be another homogeneous replay preset, not a team mode.

Exact parser-supported current commands are in the seven family reports linked below.
They were established from source inspection, not successful execution.

## Which issues must be fixed before exposing these commands?

### The requested configuration is not always the executed configuration

- The root runner has no general `--config` input and ignores malformed environment metadata at [metadata parsing](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:388).
- Root CLI defaults and direct Python API defaults disagree, including the Game 3 cost ceiling of 50 versus 30 at [the CLI parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:160).
- The multi-agent wrapper always requests parallel phases at [command construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1563).
  - Its metadata writer can then record the source value instead of the executed value at [result enrichment](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1652).
  - This is a contract defect for a supplied false value, not evidence that the normal generated true-valued plans ran incorrectly.
- The TTC runner changes an explicit 10,500-token cap to 16,384 unless a preservation field is set at [cap resolution](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28).
- Model-specific caps and the optional token estimator can also change effective requests and prompt compaction.
- Fix these through one schema and resolver, not through additional undocumented shell settings.

### Failures can change the model roster or produce artificial actions

- The factory can skip unavailable agents and return a partial roster at [agent creation](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:137).
  - The engine then zips surviving agents with the requested model names at [seat mapping](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:289).
- Native-to-OpenRouter recovery is enabled by default at [runtime recovery configuration](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290).
  - The factory has a separate recovery path that does not consult the same switch at [constructor recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:188).
- Exhausted proposal repair can create a synthetic proposal at [proposal handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2386).
- Some failed voting calls become synthetic rejection votes at [voting error handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3670).
- The public path must require the exact roster, preserve the selected provider, and stop after its declared repair or retry budget is exhausted.
  - A valid negotiation with no agreement remains a completed experiment.
  - A failed API call or invalid unrepaired action must remain a failed or incomplete attempt.
  - Protocol-defined copied captain ballots are legitimate institutional actions and must remain distinct from artificial failure substitutions.
- These are current code-path findings, not an assertion that every historical result encountered these paths.

### Existing result reuse is too weak

- The current completion check mainly requires numeric-convertible utilities and a few optional identity fields at [result validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269).
- A result passing that check is reused and rewritten with current metadata at [automatic reuse](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668).
- Attempt logs are separated, but result and interaction files can still share a directory or receive filenames that older loaders do not select correctly.
- Partial batches can return overall success after dropping failed runs at [batch handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1026) and [CLI completion](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:706).
- Use an immutable configuration hash, one directory per attempt, exact artifact references, finite-value validation, and explicit complete/failed/canceled/pending counts.
  - A normal launch must not reuse an existing result.
  - Explicit resume must not rewrite a completed result.
  - Mid-negotiation continuation should not be advertised because the progress file is not a restorable game checkpoint.

## What is hard-coded to this machine?

The path review found 71 matching source lines in 41 files for the exact scratch and home prefixes.
Those are textual matches, not 71 independent defects.

| Area | Verified example | Required change |
| --- | --- | --- |
| Repository root | [Team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:16) fixes `/scratch/gpfs/DANQIC/jz4391/bargain`. | Import installed modules and take runtime locations from explicit inputs. |
| Incorrect relative root | [Game 1 shell template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:368) uses `BASE_DIR="bargain"` after submission from the repository root. | Use an explicit interpreter and absolute job inputs without a nested checkout assumption. |
| Personal queue | [Local multi-agent execution](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717) defaults to `/home/jz4391/openrouter_proxy`. | Default the public local profile to direct requests and require a queue path for a queue profile. |
| Output relocation | [TTC builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:266) rebuilds output locations from `results_root.name`. | Use the entire selected root and store artifact paths relative to the run bundle. |
| Historical controls | [Team control loading](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:106) follows saved absolute result locations. | Use a relocatable control bundle with literal preferences and content hashes. |
| Local weights | [Model catalog](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:919) contains `bargain/models/` locations. | Require a model root or per-model path for optional local execution and fail if assets are absent. |
| Context and Elo data | [Context loader](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:94) reads a repository Markdown file. | Package the exact frozen resource and fail on missing required entries. |
| Analysis | [Llama analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_appendix_llama33_baseline_500.py:333) assumes results are under the checkout. | Read the selected manifest and allow external or relocated output roots. |

- The same incorrect nested-root pattern appears in the Game 2 and Game 3 shell generators.
- Slurm templates also assume particular modules, partitions, resources, and personal credential-file paths.
  - Some old GPU templates request H100 hardware, which conflicts with the current project smoke-test policy.
  - All seven requested paper families can use hosted APIs without local model weights or a GPU under their current catalog routes.
  - Optional GPU experiments should remain separate from API onboarding.
- The shell review found user-specific paths in 202 of 229 product/research shell and batch files.
  - Many are generated historical jobs, so their original paths are provenance rather than active public defaults.
  - Regenerate future jobs from a site profile instead of rewriting historical job records.
- Do not replace every scratch path with another universal path.
  - Separate package resources, new output locations, imported historical bundles, optional model assets, and site configuration.
  - Preserve original paths as historical metadata when importing old runs.

## How should a new user supply API keys?

- Use the user's own provider accounts and inject secrets through environment variables or an explicitly selected private environment file.
  - OpenAI recommends server-side environment variables or a key-management service in its [authentication documentation](https://developers.openai.com/api/reference/overview#authentication).
  - Do not accept key values as command arguments or save them in experiment configurations.
- Use one loader for every public command.
  - Process environment values should take priority over the explicitly selected file.
  - Avoid automatically reading a credential file from an arbitrary working directory.
  - A source-checkout convenience file can be supported only through a documented, explicit location rule.
  - Reject malformed files, blank required values, and recognizable template values without printing their contents.
- Require only the providers used by the resolved selected roster.

| Current selected execution | Required variables |
| --- | --- |
| Nano baseline plus a native OpenAI adversary model | `OPENAI_API_KEY` |
| Nano baseline plus an OpenRouter adversary model | `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| Llama baseline plus a native OpenAI adversary model | `OPENROUTER_API_KEY`, `OPENAI_API_KEY` |
| Full current two-player, Llama, homogeneous, or heterogeneous rosters | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` |
| Full current homogeneous-adversary roster | `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| TTC GPT-5 | `OPENAI_API_KEY` |
| TTC Claude Sonnet 4.6 | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| TTC Gemini 3 Flash | `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| Binding team with its declared direct OpenAI override | `OPENAI_API_KEY` |
| Optional native Google route | `GOOGLE_API_KEY` |

- A chosen-model homogeneous example needs only that model's provider key.
- Gemini in the current TTC preset uses OpenRouter, despite metadata naming Google as the model developer.
- The current [environment template](/scratch/gpfs/DANQIC/jz4391/bargain/.env.example:5) says `GEMINI_API_KEY`, while native Google execution reads `GOOGLE_API_KEY`.
- The root runner and TTC wrapper do not read the environment file that the multi-agent wrapper reads.
- Extend secret-ignore rules for private environment variants because the current rule ignores only the standard environment filename.
- Keep grouped key rotation as an advanced option with an explicit bounded policy.
- Offline doctor can verify variable presence and dependencies, but it cannot establish model access, account balance, or a working API integration.

### What must change for the proxy?

- Current OpenAI and OpenRouter file-queue requests serialize bearer authorization headers.
- The [monitor archives processed requests](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:236) without removing those headers.
- The inspected code does not enforce private file modes, ownership checks, or an endpoint allowlist for this queue path.
  - This is a credential-handling risk, not evidence that another user read a key.
  - Live queue contents and filesystem access controls were not inspected.
- Before advertising queue execution, use private files, atomic request publication, redacted archives, and a versioned request format.
  - Prefer monitor-side credential lookup by a provider-bound credential label, with per-user authorization and approved endpoints.
  - A shared queue must not let one user select another user's credentials or arbitrary outbound URLs.
  - Retain the externally managed monitor assumption for this cluster.
- OpenAI and OpenRouter currently interpret `auto` differently.
  - Resolve transport once before execution rather than letting a connection error choose it.
  - Native Anthropic has no equivalent same-provider file-queue implementation in the inspected path.
  - A restricted Slurm profile must provide supported native Anthropic network access or reject that plan.
  - Sending the same model family through OpenRouter instead is a provider change, not a transparent transport fix.
  - Native OpenAI queue execution must preserve supported endpoint, organization, and project settings or reject the plan because the current envelope does not carry all direct-client settings.

## What must be pinned for a paper rerun?

- A paper preset must include ordered model IDs, provider routes, seeds, seat assignments, all game and protocol settings, literal matched preferences, effective request settings, and source hashes.
  - Record requested model IDs and provider-reported IDs where available, and keep unavailable identity fields unknown rather than infer them from an alias.
- The saved inputs needed by several replication generators are ignored by Git and are not present in a normal source checkout.
  - Publish compact run manifests and required control inputs separately from large result archives.
  - Use checksummed data bundles for historical results and matched controls.
- Do not use analysis aliases to choose execution models.
  - [Analysis canonicalization](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:68) maps `gpt-5-nano` to `gpt-5-nano-high`, although their current routes and effort settings differ.
- Resolve the Nano effort discrepancy before calling a preset historically faithful.
  - The [paper](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/3_approach.tex:93) says medium, while the [current client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840) inserts low when effort is absent.
  - These sources alone do not establish the effective setting in every historical call.
- Resolve the heterogeneous specification conflict without changing the paper during this review.
  - The main text says 30 models and variance bins.
  - Current code, saved pool data, and the appendix identify 24 models and equal-width population standard-deviation ranges.
- Save actual attempt settings, not only the original plan.
  - All 500 saved Llama input configurations use a 10,500-token phase limit, while the current generator uses 16,384.
  - The 2,160 retained TTC result records contain 2,088 nominal limits of 10,500, 71 of 16,384, and one of 65,536.
  - Those TTC counts are result-record fields, not a claim that every individual call used the same effective cap.
  - The homogeneous archive also contains three DeepSeek V3 results with per-phase overrides that are not fully represented in their original source configurations.
- Neutralizing team prompts to follow the current agent-naming policy requires a new protocol version.
  - Do not silently describe changed prompts as the original matched treatment.
- Separate reanalysis of saved outputs from rerunning the experimental design.
  - Reanalysis can use a fixed data bundle.
  - New API runs can follow a fixed design but cannot promise identical text, payoffs, or model availability.
  - Missing historical provider or code information must remain an explicit uncertainty.

## What should the implementation patch contain?

The concrete file-level design, API contracts, schema, and representative diff are in [the implementation proposal](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/implementation_proposal.md).

- Add an installed entry point at `strong_models_experiment.cli:main` with standard argument parsing and lazy engine imports.
- Add pure schema, preset, plan, path, and provider-resolution modules under the existing experiment package.
- Extract the existing family builders into that package and retain old scripts as compatibility wrappers.
- Add one per-run worker around the existing `StrongModelsExperiment` engine.
  - Use separate processes for concurrent negotiations because the engine has mutable run state and global random state.
- Repair strict failure, roster, provider, and result handling in the shared engine paths.
- Package the frozen context/Elo resource and declare tested installation dependencies.
- Add local execution first, then a separately selected Slurm profile and secured queue transport.
- Adapt analysis loaders to exact manifest paths and provide a small-run summary.
- Keep historical scripts and results unchanged until replacement commands have passed parity and portability tests.

## What was checked, and what remains unverified?

- The portability agent scanned 208 source files containing 118,099 lines, then read relevant launch, provider, configuration, and loader implementations.
  - The corpus excludes paper files, generated experiment data, prior audits, environments, and paper-figure sources.
  - Scan coverage does not mean every line received a manual semantic review.
- Static AST parsing passed for 272 tracked Python files on the existing Python 3.14 interpreter.
  - This does not establish a supported Python-version range or a successful import.
- `bash -n` passed for 229 product/research shell and batch files.
  - Syntactically valid shell files can still contain the broken paths identified above.
- The validation agent reviewed 45 test modules and identified missing public-launch and clean-install coverage.
- Family agents inspected saved configurations, all 325 homogeneous results, and all 2,160 retained TTC terminal result records where needed to check launch settings.
- No runtime test suite, clean installation, live model access check, model loading test, or scheduler execution was performed.

Before calling the code ready for another user, require these gates.

| Gate | Passing condition |
| --- | --- |
| Clean install | An installed wheel works from another directory on a machine without this user's scratch/home paths or historical archives. |
| No-call planning | All seven presets resolve without keys, provider imports, queue writes, weight loading, or job submission. |
| Design parity | Frozen plans match expected counts, seeds, rosters, seat maps, and all retained input settings. |
| Strict failure | Missing keys, missing seats, invalid actions, unsupported providers, and nonfinite results cannot become successful experiments. |
| Result safety | Output relocation, concurrent ownership, interruption, explicit resume, and exact attempt selection work without overwrites. |
| Credentials | Secret sentinels remain absent from commands, records, error logs, and queue archives in boundary tests. |
| Real execution | Authorized smoke runs pass for every advertised game, family, provider control, and transport. |
| Cluster support | A real submitted job completes through the documented profile and produces checked artifacts. |

The next implementation step should be the shared schema and offline planner, followed by the strict one-run worker.
This gives all seven commands the same tested behavior before adding full sweeps or cluster submission.

## Where are the detailed source reviews?

- [Two-player GPT baseline](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_two_player.md).
- [Two-player Llama baseline](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_llama.md).
- [Homogeneous-adversary](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_hom_adversary.md).
- [Heterogeneous](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_heterogeneous.md).
- [Homogeneous](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_homogeneous.md).
- [Test-time compute](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_ttc.md).
- [Coordinated team and matched coalition replication](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/launch_team.md).
- [Shared runtime](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/shared_runtime.md).
- [Credentials and provider routes](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/credentials.md).
- [Runtime and source portability](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/portability_runtime.md).
- [Shell, Slurm, and UI portability](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/portability_slurm.md).
- [Packaging and onboarding](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/packaging_onboarding.md).
- [Validation design](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/validation_design.md).

The command names in this summary are the proposed common interface.
Some independent family reports include alternative illustrative spellings rather than implemented commands.
