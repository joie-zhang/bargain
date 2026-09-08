# Can a new user install the project and run the public commands?

**Question**
Can a fresh clone support a clear install, portable commands, and reliable experiment records?

**Short answer**
The checkout has usable experiment code, but it is not an installable Python distribution and its main README has no setup instructions.

- This review covers packaging, dependencies, Python compatibility, onboarding, and release checks.
- This review follows the [review scope](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/review_scope.md:1).
- The inspected Git HEAD is `444fcf9c368a60dd40a0f5f7aa8b4033552312bf`.
- Findings describe the working-tree files inspected on September 7, 2026.
- This review wrote only this requested report.
- No code, configuration, environment, experiment, paper, commit, branch, or Git index entry was changed.
- No installation, experiment, API request, model download, job submission, or project CLI execution was performed.

## What blocks a fresh user?

| Priority | Observed problem | Effect | Smallest proposed change |
| --- | --- | --- | --- |
| P0 | No tracked Python packaging manifest or console entry point exists. | Installing dependencies does not install a `bargain` command or support a wheel. | Add a narrowly scoped package manifest and a synchronous CLI entry point. |
| P0 | The [main README](/scratch/gpfs/DANQIC/jz4391/bargain/README.md:1) contains rebuttal figures, while the [documentation index](/scratch/gpfs/DANQIC/jz4391/bargain/docs/README.md:7) and [reproduction guide](/scratch/gpfs/DANQIC/jz4391/bargain/docs/reproduction.md:7) promise setup instructions there. | The documented starting point does not tell users how to install or run the project. | Restore a short, current project overview and quick start with links to the detailed guide. |
| P0 | The [credential template](/scratch/gpfs/DANQIC/jz4391/bargain/.env.example:5) names `GEMINI_API_KEY`, but the [runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:401) checks `GOOGLE_API_KEY`. | A user who follows the template can still fail the native Google credential check. | Use the exact environment-variable names consumed by each provider. |
| P0 | Native Gemini uses `google.generativeai`, but its package is [commented out](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:37). | Installing the main dependency file does not supply the [native Google agent dependency](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:3167). | Add a supported Google extra and name it in provider-specific setup instructions. |
| P0 | Context limits and model ratings are read from a documentation file outside the Python packages. | A wheel can lose context checks or fail while building a model pool. | Package the exact frozen resource and resolve it with `importlib.resources`. |
| P0 | Installing `tiktoken` changes the token estimator automatically. | Two users with the same run config can compact different parts of a conversation. | Select and record the estimator explicitly before introducing an optional tokenization extra. |
| P1 | `statsmodels` is missing from the dependency file. | The full documented test command can fail during collection. | Add it to the analysis extra and install that extra for the full test suite. |
| P1 | Python support is not declared or tested in a checked-in CI matrix. | Users must guess an interpreter and dependency combination. | Declare a minimum only with corresponding clean-install checks. |
| P1 | Public procedures route new users into cluster-specific generators. | Non-cluster users have no uniform one-command interface for all requested experiment families. | Add a local executor and make Slurm an explicit execution choice. |
| P2 | The license has an anonymous copyright holder and the public README has no license or citation section. | Release identity and attribution remain incomplete. | Confirm release identity and add accurate software, data, and paper attribution. |

- The metadata absence was checked with `git ls-files` for the packaging manifest, setuptools files, manifest file, console modules, and CI configuration.
  - No tracked `/scratch/gpfs/DANQIC/jz4391/bargain/pyproject.toml`, `/scratch/gpfs/DANQIC/jz4391/bargain/setup.py`, or `/scratch/gpfs/DANQIC/jz4391/bargain/setup.cfg` was found.
  - No tracked `/scratch/gpfs/DANQIC/jz4391/bargain/.github/` CI configuration was found.
  - The packages have no tracked `__main__.py` entry points.
- The [old README](/scratch/gpfs/DANQIC/jz4391/bargain/README-old.md:71) contains a dependency-install recipe.
  - Restoring that file unchanged would also restore its [older paper path and result inventory](/scratch/gpfs/DANQIC/jz4391/bargain/README-old.md:218).
  - Current paper and experiment statements need independent verification.

## Which dependencies are needed?

The table distinguishes a missing dependency from a package that is useful only for a selected feature.

| Dependency group | Evidence | Packaging action |
| --- | --- | --- |
| `numpy`, `scipy` | Core games import NumPy and SciPy at [game setup](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/diplomatic_treaty.py:14), [metrics](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/metrics.py:41), and [co-funding metrics](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/cofunding_metrics.py:35). | Keep in core dependencies. |
| `aiohttp` | The agent factory eagerly imports the [OpenRouter agent](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:20), which imports [aiohttp](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:10). | Keep in core for the smallest packaging change. |
| `openai`, `anthropic` | Provider imports are conditional in the [agent module](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:53). | Put each native SDK in its provider extra. |
| `google-generativeai` | The [native Google constructor](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:3167) requires this exact import namespace. | Add a Google extra for the current adapter. |
| `httpx` | The [OpenAI timeout configuration](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2670) uses it directly. | Declare it as a direct dependency of the OpenAI extra. |
| `PyYAML` | The [secondary model configuration module](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:21) imports it. | Keep if YAML configs are part of the public interface, or place it in the feature extra that imports that module. |
| `pandas`, `matplotlib`, `seaborn`, `tabulate` | Analysis scripts use these packages, including [Markdown table export](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_nash_lindahl_fairness.py:1246). | Move to an analysis extra. |
| `statsmodels` | A [retained test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_reviewer_item8_n_slope.py:15) imports a script with [unconditional statsmodels imports](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_reviewer_item8_n_slope.py:71). | Add to analysis dependencies and the full-suite install recipe. |
| `streamlit`, `plotly`, `watchdog` | Viewer dependencies already have a [separate file](/scratch/gpfs/DANQIC/jz4391/bargain/ui/requirements.txt:1). | Move to a UI extra and keep any compatibility requirements file generated from the same source. |
| `pytest`, `anyio` | The test configuration uses [AnyIO tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_allocation_preferences.py:27) and an [asyncio backend fixture](/scratch/gpfs/DANQIC/jz4391/bargain/tests/conftest.py:12). | Move to a development or test extra. |
| `requests` | A [model-client availability import](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_clients.py:47) and a [rating-refresh script](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_lmarena_elo_refresh.py:15) use it. | Retain for those features, but do not describe it as necessary for the primary asynchronous experiment path. |
| `torch`, `transformers`, `accelerate` | The local loader passes [`device_map` and `low_cpu_mem_usage`](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_clients.py:420). | Put the tested local-model stack in a separate extra or platform-specific environment. |
| `Pillow`, `python-pptx` | A [slide renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/make_gemini_coalition_dialogue_slide.py:8) imports both. | Add explicit dependencies only for the relevant figure or slide-export support. |
| `tiktoken` | The [context estimator](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:157) changes its algorithm based on import availability. | Treat estimator selection as experiment configuration. |

- The main dependency file calls analysis and UI dependencies optional, but its uncommented entries install them for every user at [lines 18 to 29](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:18).
  - These packages are used by the repository, so this is an install-scope issue rather than evidence that they should be deleted.
  - PyTorch and Transformers are already commented out and are not forced by the main install.
- `httpx` and Pillow can arrive through other packages today.
  - Their absence from direct declarations is weaker evidence than the missing `statsmodels` and native Google packages.
- The installed Transformers `4.57.1` source explicitly rejects a supplied `device_map` without Accelerate at [the dependency check](/scratch/gpfs/DANQIC/jz4391/bargain/.venv/lib/python3.14/site-packages/transformers/modeling_utils.py:4802).
  - This confirms the dependency for the inspected installed version.
  - It does not validate every version allowed by the commented `transformers>=4.35.0` suggestion.
- The commented `xai-sdk` dependency has no `xai_sdk` import in the 272 tracked Python files inspected.
  - Do not present installing that SDK as a fix for the current XAI path without changing and validating the adapter.
- No dotenv loader was found in the inspected Python sources.
  - Copying a credential template to a dotenv file is not established as a supported way to load keys.
  - The guide should use exported environment variables or an explicitly supported key-file loader.

## What makes an installed wheel behave differently?

- The [context table path](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:15) depends on the checkout layout.
  - The loader tries the current directory and the source-tree root at [lines 97 to 101](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:97).
  - Missing data produces an empty map at [lines 103 to 104](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:103).
  - The early return also bypasses the later insertion of known provider limits at [line 132](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:132).
- The [model-rating loader](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:9) expects that same document under the checkout root.
  - Rating lookup reads it at [line 142](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:142).
  - The heterogeneous model pool reads it at [line 200](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:200).
- The smallest resource fix is to package a byte-identical copy of the existing frozen table.
  - Both loaders should use one resource resolver.
  - An explicit external override should require an existing readable file and record its hash.
  - Missing required context or model-pool data should stop the run.
- The [token estimator](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:157) tries `o200k_base`, then `cl100k_base`, then a character-count estimate.
  - Record the selected estimator, encoding, package version, and character ratio when applicable.
  - Require the chosen implementation to be available.
  - Derive the estimator for a historical preset from the archived run evidence.
  - Do not infer the historical estimator from the dependency file.
- The [TTC runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:91) invokes a root-level script and fixes its working directory to the checkout at [line 169](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:169).
  - A console wrapper alone does not remove this dependency.
  - Extract a reusable run API and call it from both the installed CLI and the current wrappers.

## What packaging change is proposed?

- Add `/scratch/gpfs/DANQIC/jz4391/bargain/pyproject.toml` with explicit package discovery.
  - Include the three runtime packages and the viewer only if its installed invocation is supported.
  - Exclude experiments, analysis outputs, paper trees, and review artifacts from the wheel.
  - Keep historical data in a separately documented release bundle.
- Add `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/cli.py` with a synchronous `main(argv=None) -> int`.
  - Parse help, model listing, and validation before importing experiment runtime dependencies.
  - Call `asyncio.run()` only for a command that executes an experiment.
  - The current [entry function is asynchronous](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:42), so it must not be used directly as a console-script target.
- Add `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/__main__.py` as a thin call to the same CLI.
- Keep `/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py` as a compatibility wrapper.
  - Its current [top-level imports](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:37) request the runtime before its parser runs.
  - Its [stream and locale changes](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:16) should be confined to execution where possible.
- Add a library API such as `resolve_run_config()`, `validate_run_config()`, and `run_resolved_config()`.
  - Reuse [model-config resolution](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:60) and [single-run execution](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119).
  - Extract the required generation functions from scripts into package modules instead of importing the entire scripts tree into the installed interface.
  - Existing extraction points include [multi-agent generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:727), [TTC generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:254), and [team-treatment construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:106).

The following is a proposed manifest structure, not an implemented or validated release manifest.

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "bargain-research"
version = "0.1.0"
description = "Language-model negotiation experiments"
readme = "README.md"
requires-python = ">=3.10"
license = {file = "LICENSE"}
dependencies = [
  "numpy>=1.24.0",
  "scipy>=1.11.0",
  "aiohttp>=3.9.0",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
openai = ["openai", "httpx"]
anthropic = ["anthropic"]
google = ["google-generativeai"]
api = ["openai", "httpx", "anthropic", "google-generativeai"]
analysis = ["pandas", "matplotlib", "seaborn", "tabulate", "statsmodels", "Pillow", "requests"]
ui = ["streamlit", "pandas", "plotly", "watchdog"]
local = ["torch", "transformers", "accelerate"]
slides = ["Pillow", "python-pptx"]
dev = ["pytest", "anyio", "build"]

[project.scripts]
bargain = "strong_models_experiment.cli:main"

[tool.setuptools.packages.find]
include = ["strong_models_experiment*", "negotiation*", "game_environments*", "ui*"]
namespaces = false

[tool.setuptools.package-data]
strong_models_experiment = ["resources/*.md", "presets/*.json"]
```

- `bargain-research` is a proposed distribution name whose public availability was not checked.
- `0.1.0` matches the existing [package version string](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/__init__.py:47), but the release version still needs a release decision.
- The provider and optional dependency version ranges must be set from clean-install verification before release.
  - Current open-ended minimums are not a record of the environment used for experiments.
  - The OpenAI adapter sends [reasoning effort and completion-token parameters](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840) that need SDK compatibility checks.
- A separate tested constraints or lock file should record exact release dependencies.
  - Local CUDA dependencies may require their own platform-specific file.
  - Record exact installed versions in each run manifest.
- Do not add a tokenization extra until estimator choice is explicit in the run schema.
- If the Google adapter is migrated to a different SDK, treat that migration as a separate provider change with real smoke tests.

## What should the user guide show?

- Put the current overview, install command, one small example, output location, license link, and guide links in the main README.
- Put the detailed guide at the proposed absolute path `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/public_quickstart.md`.
- Use this sequence in the guide.
  1. State the tested operating systems and Python versions.
  2. Install the selected provider extras in a new virtual environment.
  3. Export the exact credential variables for the resolved models.
  4. List available presets and show their provider, model identifier, roster, seeds, and output requirements.
  5. Run one small experiment in a new output directory.
  6. Inspect the resolved config, result JSON, interactions, and completion state.
  7. Show the corresponding family command for each requested workflow.
  8. Link to Slurm, local-model, analysis, and viewer setup only when those features are selected.
- Keep the seven public workflow examples explicit.
  - Two-player GPT-5-nano baseline.
  - Two-player Llama baseline.
  - Homogeneous multi-agent group.
  - Heterogeneous multi-agent group.
  - Homogeneous group with one adversary model.
  - Native test-time compute.
  - Coordinated team experiment.
- Generate credential requirements from the resolved model routes.
  - The current [`gpt-5-nano` catalog entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265) uses native OpenAI.
  - The current [`gpt-5-nano-high` entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:275) uses OpenRouter and high reasoning effort.
  - The current [`llama-3.3-70b-instruct` entry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:778) is a hosted OpenRouter route.
  - Installing the local-model extra is not needed for that hosted Llama entry.
  - A local Llama run is a different provider execution and must retain that provenance.
- A proposed `bargain doctor` command should inspect package availability, required key-variable presence, configured paths, and writable output parents without a provider call.
  - Print key names and presence only.
  - Require a separate explicit smoke command for a real provider call.
- Update the [transport description](/scratch/gpfs/DANQIC/jz4391/bargain/docs/operations.md:52).
  - It currently says `auto` chooses the proxy in Slurm.
  - The [current client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:3) implements direct transport followed by a connectivity-triggered proxy path.
  - Public local examples should select direct transport explicitly.
  - Restricted-node examples should select the configured queue transport explicitly.
- Name each cluster-specific requirement in its cluster guide.
  - The [TTC script generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:341) hard-codes modules, a user key-file default, and a checkout virtual environment.
  - The [multi-agent generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1809) also hard-codes modules and queue defaults.
  - A portable executor should accept the interpreter, transport, queue path, and scheduler settings as explicit configuration.

## What current invocation is established by source inspection?

This is current checkout syntax, not a new public command and not a runtime-validated example.

```bash
OPENAI_TRANSPORT=direct OPENROUTER_TRANSPORT=direct \
  /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py \
  --game-type item_allocation \
  --models gpt-5-nano gpt-4o-mini-2024-07-18 \
  --competition-level 0.5 \
  --num-items 5 \
  --max-rounds 10 \
  --gamma-discount 0.9 \
  --discussion-turns 2 \
  --random-seed 42 \
  --batch \
  --num-runs 1 \
  --output-dir /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/public_smoke_item_allocation
```

- Run from `/scratch/gpfs/DANQIC/jz4391/bargain` and set the keys required by the catalog before executing it.
- The model keys, game selection, seed, output directory, and batch options were checked against [the parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:47) and the catalog.
- This is a new demonstration cell and does not reproduce a paper cohort.
- The script has no `--config` argument.
  - The existing [TTC wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146) has its own `--config` and `--dry-run` arguments.
  - The existing [multi-agent controller](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:125) supports `run-one --results-root ... --config-id ...` after configurations exist.
- The proposed installed interface should accept a resolved config file directly.
  - It should preserve current model order, agent identifiers, seed behavior, phase caps, game parameters, output schema, and analysis metadata.
  - It should reject conflicting command-line overrides instead of silently discarding them.

## What must remain compatible with the research design?

- Package the frozen model roster and rating/context table with source hashes.
- Save the exact model identifier, provider, route overrides, effort, token caps, and context estimator for every seat.
- Save the master seed, per-run seed, ordered roster, and model-pool selection rules.
  - The multi-agent controller currently defaults to [equal-width rating-deviation stratification](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:112).
  - Choosing its `pure_random` mode changes the sampling design.
- Save every materialized default in the resolved config.
- Preserve historical TTC cap behavior explicitly.
  - The [current TTC resolver](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28) can replace an older cap unless its preservation setting is enabled.
- Detect duplicate model catalog keys during a static validation step.
  - The AST scan found 120 dictionary entries and 117 unique keys.
  - The duplicate keys are `gpt-4o`, `Qwen2.5-72B-Instruct`, and `claude-sonnet-4-5`.
  - Python retains the later definition, so the shipped resolved catalog must be tested without silently changing historical route resolution.
- Keep failed attempts and incomplete runs distinguishable from completed results.
- Use an explicit historical preset or an explicit new-run preset.
  - Do not update retired models or provider routes automatically during reproduction.
  - Hosted models can change even when a model string is unchanged.

## Which verification gates are required before release?

| Gate | Required evidence |
| --- | --- |
| Clean source install | A new environment installs the declared extras and passes `pip check` without inherited packages. |
| Wheel install | A built wheel installs in a separate empty directory and runs without the checkout on `PYTHONPATH`. |
| Safe command discovery | Help, preset listing, model listing, and config validation work without credentials, network access, result writes, or provider initialization. |
| Packaged resources | The installed package loads the exact context/roster data and records the expected hashes. |
| Dependency isolation | A core/provider install does not require UI, plotting, development, or CUDA packages unless that feature is selected. |
| Dependency errors | A selected unsupported provider or absent extra fails with the exact install instruction before creating a partial run. |
| Config equivalence | Each public workflow produces the same resolved game settings, ordered seats, seeds, and protocol settings as its supported current generator. |
| Offline unit suite | The focused game, parser, context, key-selection, and schema tests pass in the declared test environment. |
| Full suite | The existing full suite passes with development, analysis, UI, and provider dependencies that its imports require. |
| Real integration | One bounded real smoke run succeeds for each advertised provider or local execution path and preserves valid result and interaction records. |
| Output validation | Completed, failed, and interrupted runs have distinct status and supported loaders accept the installed runner's actual output. |
| Documentation | Every published command is checked against its actual parser and each local link resolves in the release source. |
| Release archive | The source archive and wheel contain the intended license and resources and exclude credentials and generated research outputs. |

- Select a small initial Python/OS support matrix and expand it after tests pass.
  - Python `>=3.10` is a proposed minimum because retained tests use union annotations without postponing their evaluation at [one test constructor](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_gpu_llama_phase_caps.py:47) and [one accounting helper](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:13).
  - Linux and macOS need separate install checks.
  - Windows support remains unverified.
  - Bash/Slurm launchers are not portable Windows commands.
  - File-locking code has a [non-Unix path](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:21), but this does not prove concurrent-run correctness there.
- Tests with generated unit-test fixtures can validate deterministic boundaries.
  - They do not validate a real external provider.
  - A release should not claim an integration works until its real smoke run passes.
- Treat paper-data tests as a separate gate.
  - The [released TTC cohort test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:107) skips when its data directory is absent.
  - A green suite from a fresh clone therefore does not establish paper-data reproduction.

## What was actually verified in this review?

- An AST-only check parsed 272 tracked Python files from runtime packages, scripts, tests, the UI, and the root runner.
  - No syntax error was found with the existing Python `3.14.0` interpreter.
  - Parsing with Python grammar settings from `3.9` through `3.13` also found no syntax errors.
  - Grammar acceptance does not test runtime typing behavior, dependency wheels, imports, SDK compatibility, or provider behavior.
- Installed-package metadata was read without importing the project or provider SDKs.
  - The existing environment contains `openai 2.8.1`, `anthropic 0.74.0`, `google-generativeai 0.8.6`, `statsmodels 0.14.6`, `torch 2.9.1`, `transformers 4.57.1`, and `accelerate 1.11.0`.
  - `tiktoken` is not installed in this environment.
  - These observations describe an existing environment and do not establish a reproducible install.
- Static catalog inspection confirmed the current documentation example model keys are present.
- The proposed manifest passed a TOML syntax check, and every linked source file exists.
- The credential template was inspected with assigned values redacted.
- No pytest suite, CLI help path, build, installation, network check, or runtime integration was executed.

## Which decisions remain open?

- Confirm the public distribution name, release version, authorship, and citation target.
  - The [existing MIT license](/scratch/gpfs/DANQIC/jz4391/bargain/LICENSE:1) is present and should remain in software distributions.
  - Its copyright holder is `Anonymous Authors` at [line 3](/scratch/gpfs/DANQIC/jz4391/bargain/LICENSE:3).
  - Software licensing does not establish the redistribution terms for external model weights or a separately released data bundle.
- Choose the initial Python/OS support matrix and validated dependency versions.
- Choose whether public native Gemini support keeps the current adapter or includes a separately tested SDK migration.
- Specify the exact preset names and frozen experiment cohorts for the seven commands.
- Confirm whether a separately licensed paper-data bundle will be available to fresh users.
- Define the smallest supported local-model hardware and model-path contract without changing the historical provider route.
