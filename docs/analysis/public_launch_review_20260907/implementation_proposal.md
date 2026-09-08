# Proposed implementation patch

**Question**

What should the code change look like?

**Short answer**

Add a small launch package around the existing engine, then repair the shared configuration, provider, failure, and output paths that the public command depends on.
Do not create seven separate runners or move the entire repository.

- This is an implementation proposal, not applied experiment code.
- The excerpts below specify interfaces and representative changes rather than a complete, tested, apply-ready patch.
- Proposed paths use the current checkout root for clarity, but the implementation must not embed that root as a runtime default.
- The supporting findings are in [the review report](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/report.md).

## What are the tradeoffs?

- Short shell aliases would be the smallest change.
  - They would retain different environment loaders, parameter defaults, validators, and output rules.
  - They cannot prevent the engine's provider changes or artificial failure actions.
- A new experiment framework would provide a clean interface.
  - It would duplicate working game code and create unnecessary parity work.
- An installed CLI plus a shared resolved-config adapter is the recommended middle option.
  - It reuses the existing game and provider implementations.
  - It gives all families one validation and execution contract.
  - It still requires targeted edits in shared runtime files before paid execution is exposed.

## Which new files should be added?

| Proposed file | Responsibility |
| --- | --- |
| `/scratch/gpfs/DANQIC/jz4391/bargain/pyproject.toml` | Install the packages, dependencies, resources, and `bargain` entry point. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/cli.py` | Parse all commands with a synchronous `main(argv=None)` and no early engine imports. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/__main__.py` | Provide the equivalent `python -m strong_models_experiment` entry point. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/__init__.py` | Keep the launch package import free of execution side effects. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/schema.py` | Define typed scientific inputs, execution settings, outcome states, and strict validation. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/presets.py` | Expand the seven family presets and read fixed paper plans. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/plan.py` | Resolve every default, create stable run IDs, and print or save plans without execution. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/paths.py` | Resolve selected inputs, output roots, model assets, and bundle-relative artifact references. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/providers.py` | Resolve exact model routes, load private credentials, redact secrets, and provide offline doctor checks. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/execute.py` | Start one isolated worker per negotiation and import the existing engine only in the worker. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/store.py` | Own run locks, attempt directories, atomic records, complete-result checks, status, and resume. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/slurm.py` | Render and submit an explicit execution profile after local execution is validated. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/summary.py` | Summarize exactly the attempts named by a selected manifest. |

- These modules should contain small functions rather than a plug-in framework.
- Begin with JSON because existing experiment configurations already use it.
  - YAML can be added later if it provides a clear user benefit.
- Keep new preset JSON under `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/presets`.
- Put the frozen context/Elo resource under `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/resources`.
  - Initially copy the current required Markdown byte-for-byte and preserve its hash.
  - Change both context lookup and roster lookup to use the packaged resource.
  - Converting it to a machine-readable catalog is a later parity-checked change, not a prerequisite for the console command.
- Extract family-specific functions into separate launch modules only when their size makes the preset module difficult to read.

## What is the package entry point?

The essential new packaging entries are these.
Dependency versions and the supported Python range still require clean-install testing.

```toml
[project.scripts]
bargain = "strong_models_experiment.cli:main"

[tool.setuptools.packages.find]
include = ["strong_models_experiment*", "negotiation*", "game_environments*"]
namespaces = false

[tool.setuptools.package-data]
strong_models_experiment = ["resources/*.md", "presets/*.json"]
```

- The current experiment package already imports the engine lazily in [its initializer](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/__init__.py:14).
- Keep help, model listing, plan, status, and offline doctor on that lightweight path.
- Do not package the entire scripts, results, analysis, or paper trees merely to make imports work.
- Add explicit dependency groups.
  - The initial runtime needs the actual imported numerical and HTTP libraries, including NumPy, SciPy, and aiohttp.
  - A paper-API group needs the tested OpenAI and Anthropic clients because OpenRouter already uses the common HTTP layer.
  - Analysis should include its actual dependencies, including statsmodels where its modules or tests use it.
  - UI, native Google, native xAI, and local model loading should be separate advertised and tested options.
  - Local Transformers loading also needs accelerate for the current automatic device-map path.
- Declare token estimation in the run specification before making tokenizer installation optional.
  - Installing another extra must not silently change the selected estimator and prompts.
- Keep a tested constraints or lock file and record installed package versions per run.
- The existing Python 3.14 AST check does not justify claiming compatibility with another Python version.
- Do not add a public viewer entry point until it can read a selected run root for all advertised games.

## What should one resolved run contain?

A manifest is the saved list of runs and their inputs.
A schema is the set of fields and validation rules accepted for those inputs.

| Record | Required contents |
| --- | --- |
| Scientific settings | Preset and protocol versions, game type and parameters, ordered seat IDs and roles, exact model/provider definitions, seeds, rounds, discussion turns, discount, phase concurrency, reasoning controls, phase caps, context policy, and bounded repair policy. |
| Input identity | Frozen roster/catalog hashes, literal preference hash when fixed, matched-control IDs and hashes, original source-config hash, and historical source code information when available. |
| Execution settings | Local or Slurm executor, interpreter, explicit provider transports, supported endpoint configuration, timeouts, worker count, queue paths, and selected output location. |
| Attempt record | Run ID, attempt ID, immutable scientific-config hash, code revision plus changed-source digest, start/end state, errors, retries, actual model/provider/transport per call, token usage, and exact result/transcript paths and hashes. |

- Keep the scientific hash independent of the output root so a result bundle can move without changing the experiment's identity.
- Save the execution-profile hash separately.
- Include result-affecting settings such as endpoint selection, request payload policy, estimator, and dependency versions in compatibility checks.
- Store safe credential labels and variable names, never credential values.
- Preserve both requested and provider-reported model IDs when the provider supplies them, and keep unavailable identity information explicitly unknown.
- Do not let free-form metadata override scientific fields.
- Validate explicit values before applying declared defaults.
  - Reject unknown fields, wrong types, unknown model or provider names, invalid game bounds, missing control data, and nonfinite numbers.
  - Distinguish an absent value from an explicit zero, false, or null where null is valid.
- Resolve all seats before constructing any agent.
  - Never use the analysis alias map as execution identity.
- For seeded fresh games, record the realized state before the first provider call.
  - Matched Game 1 team runs must use the literal supplied table rather than regenerate it from a seed.
  - TTC comparisons must verify that the effort conditions share the intended game instance.

The core interfaces can remain this small.

```python
def expand_preset(request: LaunchRequest, snapshots: Snapshots) -> list[RunSpec]:
    ...

def resolve_plan(specs: list[RunSpec], profile: ExecutionProfile) -> RunPlan:
    ...

def check_setup(plan: RunPlan, credentials: CredentialSources) -> DoctorReport:
    ...

async def execute_one(run: ResolvedRun, attempt: AttemptHandle) -> AttemptOutcome:
    ...

def validate_complete_attempt(run: ResolvedRun, attempt: AttemptRecord) -> None:
    ...
```

- The ellipses describe proposed contracts, not implemented functions.
- `expand_preset` and `resolve_plan` must make no model requests or scheduler calls.
- `check_setup` must check local requirements without claiming that key presence proves authentication.
- `execute_one` should call the existing `StrongModelsExperiment` with a validated dictionary or typed adapter.
- Use one subprocess per negotiation for concurrent sweeps because random state and current-run state are not isolated for simultaneous calls within one engine object.
- Local and Slurm workers must consume the same resolved JSON instead of rebuilding it through different sets of command arguments.

## Which existing code must change?

| Existing code | Proposed targeted change |
| --- | --- |
| [Root runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:42) | Keep old arguments as a compatibility adapter, add a strict resolved-config path, reject malformed metadata, and remove silent precedence conflicts. |
| [Experiment engine](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:119) | Accept only resolved settings on the new path, validate exact seat identity, handle seed zero, close provider clients, and report every requested attempt. |
| [Agent factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:60) | Extract pure model resolution, make creation all-or-error, and enforce one provider policy at construction and runtime. |
| [Phase handler](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2386) | Replace exhausted-repair substitutions with recorded failed phases and preserve raw invalid outputs for inspection. |
| [LLM agents](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290) | Enforce resolved routes and effort settings, preserve actual response provenance, remove artificial default strategy recovery from valid-run paths, and close clients explicitly. |
| [Result models](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/data_models.py:47) | Add requested/completed/failed/canceled/pending counts and explicit terminal outcome status. |
| [File manager](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:131) | Write only within the assigned attempt directory and publish final artifacts atomically. |
| [Multi-agent controller](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:727) | Extract requested-family builders, validate against the selected specification, remove implicit reuse, and stop post-run metadata rewriting. |
| [TTC wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28) | Pass the full resolved configuration without cap migration or dropped override fields. |
| [TTC generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:254) | Accept selected seeds/families/efforts/game cells and use the actual selected output root. |
| [Llama generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:70) | Reuse its three pure builders through the shared two-player adapter and retain historical IDs/seeds before filtering. |
| [Homogeneous controller](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:617) | Keep historical ID import support but use the common worker and an explicit selected 300-run preset. |
| [Binding-team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:65) | Separate protocol construction from 100-control loading and remove the fixed checkout and queue paths. |
| [Parent team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:106) | Import controls through checked bundle-relative references and preserve caller data instead of mutating source dictionaries. |
| [Context lookup](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:94) and [roster lookup](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:138) | Load the same packaged frozen resource and reject missing required data. |
| [OpenRouter client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:530) and [queue monitor](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:194) | Require explicit queue configuration, secure request storage, redact archives, and enforce the versioned endpoint/credential contract. |
| [Key rotation](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:490) and [legacy serializer](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:802) | Redact provider errors and exclude secret values from serialized configuration. |

- Old scripts should call the common adapter when retained as public commands.
- Do not delete them merely because the new console entry point exists.
- New strict failure handling changes behavior after invalid outputs and must have a protocol/version record.
  - Do not rewrite archived results to match the new validity policy.
  - Keep historical import and reanalysis separate from new execution.
- Keep legitimate game rules unchanged, including no-agreement outcomes and declared binding-team institutional votes.
- Reject unknown team protocol names and missing captain decision fields.
  - An explicit reject-all decision is different from an absent JSON key.
  - Use neutral seat wording in new prompts and give changed team prompts a new version.

### What does a small part of the diff look like?

These representative edits address three verified defects.
They are not sufficient to implement the new launch path by themselves.

```diff
--- a/strong_models_experiment/experiment.py
+++ b/strong_models_experiment/experiment.py
@@
-        if config["random_seed"]:
+        if config["random_seed"] is not None:

--- a/scripts/full_games123_multiagent_batch.py
+++ b/scripts/full_games123_multiagent_batch.py
@@
         str(config.get("gamma_discount", GAMMA_DISCOUNT)),
-        "--parallel-phases",
     ]
+    if config["parallel_phases"]:
+        cmd.append("--parallel-phases")

--- a/scripts/generate_ttc_native_scaling_jobs.py
+++ b/scripts/generate_ttc_native_scaling_jobs.py
@@
                 rel_output_dir = (
-                    Path("experiments")
-                    / "results"
-                    / results_root.name
+                    results_root
                     / model_condition["family"]
```

- The conditional parallel setting assumes the schema has already supplied and validated the boolean field.
- The TTC variable should also be renamed to reflect that it is now an absolute output directory.
- Every writer, index, loader, and resume check consuming that output path must be tested together.
- Do not fix token-cap migration by replacing one magic number with another.
  - Remove implicit migration from the strict path and preserve the requested value after validation against the explicit provider policy.

## What should family presets preserve?

| Preset | Required scientific checks |
| --- | --- |
| `paper-two-player-v1` | 420/540/540 cells, 30-model order, discussion-turn selection, correct game grids, original seed and seat conventions. |
| `paper-two-player-llama-v1` | 140/180/180 cells, the ten named adversary models, saved input limits, original IDs/seeds before selection. |
| `paper-homogeneous-adversary-v1` | 1,300 cells, five adversary models, five group sizes, two endpoint positions, two replicates, 13 game cells. |
| `paper-heterogeneous-v1` | 1,300 explicit sampled rosters from the frozen 24-model pool, five population-standard-deviation strata, separate draw and order seeds. |
| `paper-homogeneous-v1` | The selected 300 cells and model assignment, with historical effective overrides recorded separately from source inputs. |
| `paper-ttc-v1` | 2,160 cells over the exact ten seeds, three model families, four native effort conditions, nine game cells, and two orders. |
| `paper-team-v1` | 100 checked control links and literal preference tables, fixed initial captain and preserved rotation rule, exact protocol version, and explicit provider overrides. |

- Version names here are proposed and do not establish that these manifests have been assembled or historically validated.
- Build small new-run presets without depending on ignored historical result directories.
- Keep heterogeneous paper sampling distinct from uniform random sampling.
  - Generate only requested subset maps for new small experiments.
  - Select from the frozen full manifest when reproducing a paper subset so filtering cannot shift random draws or IDs.
- TTC effort names differ by family.
  - GPT-5 and Gemini use minimal/low/medium/high.
  - Claude Sonnet 4.6 uses low/medium/high/max.
  - Equal ordinal positions are not equal compute budgets.
- Resolve historical provider and reasoning uncertainties before offering a historically faithful preset.
  - If the retained evidence cannot establish a value, stop or label the preset as a current-protocol rerun with the declared difference.
  - Do not silently infer medium Nano effort from prose or high o3-mini effort from an alias.
- A fixed-settings API rerun must not promise identical stochastic outputs.

## How should outputs and resume work?

- Each saved plan should name every requested run before execution starts.
- Assign each run a stable scientific identity and a new attempt ID for each execution.
- The attempt directory should hold its resolved config, progress events, interactions, result, and terminal record.
- Keep paths relative to the plan root inside portable bundles and print absolute paths in CLI output.
- Acquire per-run ownership before allocating an attempt.
  - Test locking on the shared filesystem used by the chosen executor.
  - Do not let two local or Slurm workers own the same run simultaneously.
- Publish final records with an atomic rename on the same filesystem.
- Resume should skip only a complete attempt whose identity, configuration, artifact hashes, and integrity checks match.
  - A mismatch is an error, not a reason to rewrite metadata.
  - Restart an interrupted negotiation as a new attempt unless a separately tested checkpoint protocol exists.
  - First reconcile any pending or accepted remote request whose outcome is unknown so resume does not issue a duplicate paid request without a recorded decision.
  - Keep an unresolved remote outcome in a distinct state when the provider cannot establish its status.
- Valid terminal game outcomes include agreement and completed disagreement.
- API failure, invalid unrepaired actions, cancellation, and incomplete artifacts need separate states.
- Overall sweep success requires every requested cell to have a valid complete attempt.
- Analysis should select the terminal attempt named by the manifest rather than a first filename match.
  - Missing utilities must not become zero.
  - A no-agreement run's rule-defined zero utilities are valid observations.
  - Missing token usage must remain unknown instead of becoming fabricated zeros.

## What setup should the documentation show?

- The README should begin with installation, required keys, one small real example, and its output location.
- Add `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/public_quickstart.md` with all seven command examples and their defaults.
- Update [the existing reproduction guide](/scratch/gpfs/DANQIC/jz4391/bargain/docs/reproduction.md:1) and [operations guide](/scratch/gpfs/DANQIC/jz4391/bargain/docs/operations.md:1) to use the shared command.
- Correct [the credential template](/scratch/gpfs/DANQIC/jz4391/bargain/.env.example:5) and expand [secret ignore rules](/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:2).
- An external user's proposed sequence should be this short.
  - Install the tested package with the dependency group for the chosen providers.
  - Put only the required keys in a private environment file or secret manager.
  - Run `bargain doctor two-player --game game1 --adversary gpt-4o-mini-2024-07-18 --env-file /path/to/private-keys.env`.
  - Run `bargain run two-player --game game1 --adversary gpt-4o-mini-2024-07-18 --env-file /path/to/private-keys.env --output /path/to/new-negotiation`.
  - Read the exact output path and outcome summary printed by the command.
- The example paths above must be replaced with the user's selected absolute paths.
- Do not use a personal Della account, partition, queue, model directory, or credential file as a public default.
- Hosted API runs do not need Slurm or GPU setup.
- An explicit Slurm profile should name its interpreter, CPU/memory/time request, account/partition policy, concurrency, module setup if needed, transport, and shared queue locations.
  - For this cluster, keep the externally managed monitor assumption.
  - Reject a restricted-node plan whose selected native provider lacks a supported route.
  - Preserve provider identity when moving from direct transport to an authorized same-provider queue.
  - Preserve supported endpoint, organization, and project settings, or reject the queue plan if it cannot represent them.
  - The current native OpenAI queue envelope omits organization/project headers and hardcodes its endpoint, so direct and queue execution are not interchangeable for every account configuration.
- Do not offer a dollar estimate without a versioned price source and a stated estimation method.
  - Print run counts, phase limits, concurrency, and known request bounds even when price is unknown.
  - A post-run cost estimate is not a strict spend limit.

## In what order should this be implemented?

1. Add the schema, frozen resources, package entry point, and all seven offline plans.
2. Add strict roster/provider/failure handling and the one-run worker.
3. Add atomic attempt storage, exact output references, status, and explicit resume.
4. Connect family builders and small-run summaries to the shared worker.
5. Assemble checked paper manifests and matched-input bundles with explicit unresolved fields.
6. Update credentials, installation, and onboarding, then test a clean wheel outside the checkout.
7. Add the secured queue and Slurm profile path, with real integration tests after separate launch authorization.

- Stages 1 through 4 should not require a broad file move or deletion campaign.
- Historical reanalysis stays available while new-run behavior is versioned and tested.
- Before claiming external-user readiness, pass the [release gates](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/validation_design.md).
- Required tests include every family plan, fixed manifest parity, seed zero, false-valued settings, missing keys, absent model assets, no-contact help/planning, real output relocation, concurrent attempts, interruption, partial failure, and explicit resume.
- Test doubles may isolate external boundaries in unit tests but must not become a public fake-model or fake-result mode.
- Live provider and Slurm validation remains unperformed in this review.
