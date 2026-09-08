# Shared runtime and public CLI review

**Question**

What shared code can support one `bargain` command for all experiment families?

**Short answer**

Keep the existing game engine and phase handlers, but add one strict run schema, one provider policy, and one result store before exposing the engine through a public command.

- This is a source review completed on 2026-09-07.
- No experiment, API request, model load, job submission, import-based check, or test was run.
- Only this report was written.
- Current behavior below is established from source, while failure scenarios are stated as consequences of that source.
- The proposed APIs and commands do not exist yet.

## What is shared already?

| Current interface | Current path through the code | Implication for a public command |
| --- | --- | --- |
| Root argument-based runner | Argument parser → `StrongModelsExperiment.run_single_experiment` or `run_batch_experiments` → `StrongModelAgentFactory` → `PhaseHandler` and `create_game_environment` | Reuse the engine after validating a resolved configuration. |
| Multi-agent runner | `full_games123_multiagent_batch.run_config` → argument list plus `EXPERIMENT_RUN_METADATA_JSON` → root runner subprocess | Reuse family generation and attempt-log concepts, but replace configuration serialization and result admission. |
| Native TTC runner | `run_ttc_native_config.build_command` → one-run batch subprocess plus selected metadata | Preserve TTC settings explicitly because this wrapper changes some historical token caps. |
| Team and monoculture entry path | `random_monoculture_control_batch.run_one` → `runtime_config` → shared multi-agent `run_config` | Extract the shared executor from the historically named batch script. |
| Local model path | `LocalModelAgent` → `PrincetonClusterClient` from the separate model-client module | Keep this adapter, but resolve model assets and local generation settings explicitly. |

- Sources are the [root dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:689), [engine construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:194), [multi-agent dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656), [TTC dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146), [monoculture adapter](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:617), and [local model adapter](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2963).
- `GameEnvironment` already separates game prompts, proposal interpretation, and utilities from the shared runtime, as shown by its [interface](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/base.py:113) and [factory](/scratch/gpfs/DANQIC/jz4391/bargain/game_environments/__init__.py:44).
- The separate `ExperimentModelConfig` and `ConfigLoader` are not the schema consumed by these launchers, as shown by the [YAML loader](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:714) and the root runner's [plain dictionary construction](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:551).
- Do not adopt that YAML serializer unchanged because it writes provider dataclass fields and `default_api_keys`, which can include credentials, at [serialization lines](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:796).

## Which current commands are established by the parser?

- The following are syntax examples only and were not executed.
- The example output paths are placeholders for new absolute paths.
- The root parser has no `--config`, family selector, provider selector, transport selector, or resume option.
- The root runner supports ordered model aliases, game settings, phase controls, reasoning budgets, output paths, and batch mode at its [parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:47).

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py \
  --models gpt-5-nano llama-3.3-70b-instruct \
  --game-type item_allocation --num-items 5 --competition-level 0.5 \
  --max-rounds 10 --discussion-turns 2 --gamma-discount 0.9 \
  --random-seed 42 --max-tokens-per-phase 16384 \
  --output-dir /absolute/new/run-directory
```

- The example needs exported `OPENAI_API_KEY` and `OPENROUTER_API_KEY`, or the supported key groups, because the aliases resolve to those providers at the [nano definition](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265) and [Llama definition](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:778).
- The root CLI checks credentials for every requested model before engine construction at its [credential check](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398).
- Root batch syntax adds `--batch --num-runs N`, but the seed and persistence differences below matter before recommending it for a public workflow.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py \
  run-one --results-root /absolute/existing/batch-directory --config-id 1

/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py \
  --config /absolute/existing/config.json --dry-run
```

- These forms are established by the [multi-agent parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:125) and [TTC parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146).
- Only the TTC form above has a dry-run option for one configuration.
- The TTC dry-run reads the supplied file and prints its metadata, so it should use a reviewed configuration file.

## What must change before public launch?

### Configuration precedence can change the experiment

- Root CLI defaults differ from direct Python API defaults.
  - Item competition is `0.95` in the CLI and `1` in `run_single_experiment`.
  - Diplomacy uses `rho=0.0, theta=0.5` in the CLI and `rho=-1, theta=1` in the engine's default dictionary.
  - Co-funding uses `c_max=50.0` in the CLI and `30.0` in the engine and game dataclass.
  - Sources are the [CLI game defaults](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:86), [CLI cost ceiling](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:160), [engine defaults](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:144), and [engine co-funding construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:221).
- `EXPERIMENT_RUN_METADATA_JSON` is both a metadata channel and a configuration channel.
  - Malformed JSON is ignored after a warning, and a non-object JSON value is ignored without an error at [metadata parsing](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:388).
  - `setdefault` gives pre-existing CLI defaults priority over metadata values, even when the user did not specify the CLI option, at [configuration merge](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:591).
  - Other keys, including `team_coordination`, preference locks, and model overrides, can enter only through this channel or direct Python calls.
- The multi-agent command builder always adds `--parallel-phases` at [command construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535).
  - A source configuration with `parallel_phases=false` would therefore execute with parallel phases.
  - Result enrichment can then overwrite the saved `parallel_phases` with the source value at [metadata enrichment](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1647).
  - The current generator writes `parallel_phases=true`, so this is a demonstrated configuration-contract defect rather than evidence that existing generated runs used the wrong setting.
- Invalid access budgets become `1`, and out-of-range access seat indices are clamped, at [access resolution](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:304).
- The smallest fix is a pure resolver that rejects unknown or conflicting fields, declares each default once, and materializes every result-affecting value before calling the engine.

### Requested and actual agent rosters can differ

- `create_agents` skips unknown aliases and appends only agents that were created at [roster construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:137).
- Missing local assets and missing OpenRouter credentials return `None` at the [local factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:535) and [OpenRouter factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:572).
- The engine checks only that the returned list is nonempty, then zips surviving agents with the original model list at [agent acceptance and mapping](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:267).
  - If an earlier seat is skipped, the stored model map can assign a later agent the earlier model alias.
  - The root parser rejects unknown model names, but this does not protect direct Python callers or all local-model failure paths.
- Unknown `api_type` values enter the OpenRouter branch through an `else` at [provider selection](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:240).
- The smallest fix is to resolve all seats before constructing any agent, reject unsupported providers, and require exact equality between requested seat IDs and created seat IDs.
- Construct the model map from each resolved seat record instead of zipping independent lists.

### Provider recovery can change model identity or provider controls

- Native-to-OpenRouter recovery is enabled when `OPENROUTER_PROVIDER_FALLBACK` is unset, as shown by the [environment constant](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:234), [default-true helper](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290), and [runtime recovery check](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:590).
- The factory also switches to OpenRouter when native keys are missing or exhausted at [factory recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183).
  - The factory recovery function does not check `OPENROUTER_PROVIDER_FALLBACK` at [constructor recovery implementation](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:245).
  - The root CLI blocks missing native credentials before this point, but direct Python callers and later runs with exhausted process-level key pools can reach this path.
- Route inference strips date suffixes for some OpenAI and Anthropic model names at [route inference](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:315).
  - A versioned model name can therefore become an undated route.
- Native reasoning settings are translated into OpenRouter settings at [control translation](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:359).
  - This is a result-affecting change, even when the route names refer to the same model family.
- Runtime recovery records source provider, destination provider, source model, destination model, and the trigger at [recovery metadata](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:675).
  - Constructor recovery does not create the same transition record.
- Configured route identity is not the same as provider-confirmed model identity.
  - OpenRouter returns `model_used=self.model_id` from its configured route at [response construction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:839).
  - Its direct response reader retains content and usage without returning the full server response at [response extraction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:518).
  - Save requested route, server response ID, returned model identity, and returned backend identity as separate fields when available.
  - Keep unavailable server identity fields unknown instead of copying the requested identity into them.
- Public runs should use one explicit provider and model route per seat and stop when that route fails.
- If a later study explicitly permits a provider change, store that permission and transition in the resolved configuration and result, and treat it as a separate experimental condition.

### Success does not mean all requested work completed

- Batch execution catches run errors and continues at [batch failure handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1026).
- `BatchResults` stores only successful experiments and their count at the [batch result construction](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1054).
- The serialized batch schema has no requested count, failed seed list, or partial status at the [batch data model](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/data_models.py:47).
- The root CLI returns success when at least one batch run succeeds at [CLI batch completion](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:706).
- Some ordinary phase paths create artificial actions after model-output failures.
  - Unparsable proposals become synthetic default proposals at [proposal repair exhaustion](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2386).
  - Failed vote repair can produce a synthetic reject vote at [vote repair exhaustion](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3518).
  - `think_strategy` can return a fixed strategy after repeated errors without a `used_fallback` field at [thinking retry exhaustion](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:1499).
  - These are current production paths, and a public strict-run adapter must prevent them from producing valid research results.
- The multi-agent result validator checks numeric-convertible utilities and a few optional configuration values at [result validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269).
  - It does not require exact model identity, finite utilities, a complete transcript, final run status, or clean vote integrity.
  - Missing seed metadata is accepted, and `float()` also accepts non-finite values.
- Invalid co-funding qualitative outputs are dropped after a warning at [qualitative validation](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:872).
- Add a terminal outcome for every requested run and return a nonzero process status when any required run is failed or incomplete.
- Preserve no-agreement as a valid game outcome when the game completed, because provider failure and negotiation failure are different events.
- Preserve historical artificial-action records as historical records with explicit validity information, and do not rewrite them as clean runs.

### Persistence and restart handling can mix attempts

- Root default output names include only two model names and selected game fields, with no timestamp, at [output naming](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:640).
  - Repeating a single-run command can write into the same directory.
  - Changing later seats in a multi-agent roster need not change that default directory name.
- Core interaction and result files are opened directly for overwrite at the [file manager](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:192).
- Filename collision handling uses a one-second timestamp and a process-local cache at [unique filename handling](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150).
- A custom output directory gives batch mode an empty `batch_id` at [custom batch setup](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:967).
  - The inner single-run save uses `bool(current_batch_id)`, so it overwrites the unnumbered result file before the outer batch saves a numbered result at [inner save](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:936) and [outer save](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1018).
  - Combining `--num-runs` greater than one with `--run-number` reuses the same seed, run number, and cached filename at [batch iteration](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:992).
- The multi-agent wrapper accepts an existing valid-looking result, changes its metadata, and marks it successful without execution at [skip-existing handling](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668).
  - Existing result identity is not tied to a resolved configuration hash.
  - Its fixed candidate order prefers the unnumbered file over the numbered file at [result discovery](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1258).
- Attempt logs have separate IDs and retain previous log files in the normal path at [attempt log handling](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:244).
  - Result and interaction files do not receive equivalent attempt isolation.
- Corrupt or unreadable status files become an empty dictionary at [status reading](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:227).
- The progress file records counts and the most recent interaction at [progress writing](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1464).
  - It is not a saved game state with model memory and random-generator state.
  - No restart reader for this progress file is present in the inspected engine or launchers.
- Add a per-run ownership lock, an immutable attempt directory, atomic writes, and a terminal manifest that names exact artifact hashes.
- Let an explicit resume operation skip only a complete result whose configuration hash matches.
- Restart failed negotiations as a new recorded attempt unless a future protocol defines and tests exact in-game restoration.
- Remove post-run metadata rewriting and preserve requested metadata separately from observed execution metadata.

### Provider and process lifetimes need an explicit boundary

- `StrongModelsExperiment` stores active configuration, current run IDs, phase handler, interactions, and token totals on one mutable object at [runner state](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:78).
  - Do not run concurrent negotiations on the same instance.
- Parallel phases start one task per seat and collect results in seat order at [phase concurrency](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:456).
  - Keep this game-level behavior separate from the number of concurrent run processes.
- Rate limits belong to each agent at [agent construction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:533), while exhausted key labels are shared only within the process at [key-pool state](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:390).
  - Multiple run processes therefore do not share a provider request budget through this code.
- The multi-agent wrapper waits for its subprocess without a run timeout or interruption-status handler at [process wait](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1735).
- The ordinary engine has no explicit agent close loop in its success or failure path.
  - OpenRouter cleanup depends on `__del__` scheduling an asynchronous close at [client cleanup](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:861).
  - OpenAI key changes replace the client without closing the previous client at [key reconfiguration](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2732).
- Provider retries use both a time budget and outer agent attempts at [provider retry handling](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:650) and [outer response retry handling](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:1972).
  - The time budget is checked after requests fail, so it is not a strict deadline for an individual in-flight request.
- Use one runner instance per negotiation, close all provider clients in `finally`, and define bounded call, phase, and run deadlines in the executor contract.

## Which defaults and dependencies can change portable reruns?

| Area | Established behavior | Required treatment |
| --- | --- | --- |
| Nano identity | `gpt-5-nano` uses direct OpenAI and the client supplies low reasoning when none is declared; `gpt-5-nano-high` uses OpenRouter with explicit high reasoning. | Keep these execution identities separate. |
| Analysis aliases | The analysis roster maps `gpt-5-nano` to `gpt-5-nano-high`. | Never use analysis canonicalization to choose an execution provider or effort level. |
| Output caps | The experiment default is 16,384, model caps can be lower, and a special policy can prefer a larger model cap when the experiment cap equals the default. | Save requested and effective caps for each seat and phase. |
| TTC cap migration | A configured cap of 10,500 becomes 16,384 unless preservation is requested; extended-cap runs can use 65,536. | Require an explicit migration choice and preserve historical caps by default in historical adapters. |
| Seed zero | `if config["random_seed"]` does not seed Python's global generator for zero. | Accept seed zero explicitly and give order, preferences, and local generation separate saved generators. |
| Batch seed expansion | Ordinary batch execution uses `seed + i`, or `42 + i` when no seed is given; an explicit run number disables expansion. | Save the full seed list in the plan before execution. |
| Model order | `random` chooses only between the input list and its reversal. | Preserve that legacy rule and distinguish it from a full permutation of an N-agent roster. |
| Context limits | Runtime limits come from a Markdown table and optional environment settings. | Package a versioned machine-readable capability snapshot and record its hash. |
| Token estimator | Installed `tiktoken` and encoding availability determine whether token counting or a character estimate is used. | Resolve and record the estimator because it affects compaction and prompts. |
| Local generation | Each local agent creates its own client, uses automatic device and dtype selection, always samples, and truncates tokenized input. | Resolve device, dtype, generation seed, chat template, and context behavior explicitly. |
| Local prompt recovery | A chat-template error switches to a generic prompt format. | Stop on an unsupported template instead of changing the model input format. |

- Sources are the [nano catalog](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265), [OpenAI effort default](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840), [analysis aliases](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:61), [phase-cap resolver](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1025), [TTC cap resolver](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28), [seed and order logic](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:161), [batch seed logic](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1005), [context-table loader](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:94), [token estimator](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:157), and [local generation](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_clients.py:394).
- Missing context-table files return an empty map before applying even the known provider caps at [missing-table handling](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:103).
- Invalid context thresholds, transport settings, and some token caps are replaced by defaults at the [threshold resolver](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:212), [transport resolver](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:232), and [OpenRouter cap resolver](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:100).
- Shared wrapper execution defaults to the Princeton file queue at [wrapper environment setup](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).
  - OpenRouter's own `auto` setting instead tries direct transport first at [transport order](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:393).
  - OpenAI's `auto` setting selects the queue when `SLURM_JOB_ID` exists at [OpenAI transport choice](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742).
- The full batch launcher reads a repository credential file through a small custom parser, while the root and TTC commands read exported environment variables, at [wrapper credential loading](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1521).
  - Choose one documented credential-loading behavior for the public command.
  - Save credential variable names only, never values.
- The Slurm template assumes specific modules, a CPU partition, a repository environment, and a default credential-file location at [Slurm template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786).
- Root import requests `StrongModelsExperiment` before parsing arguments at [root imports](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:37).
  - This forces the engine import despite the package's lazy import support at [package accessor](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/__init__.py:14).
  - The public parser, model listing, and dry-run should import no provider clients and should create no output directories.
- Dependencies use broad lower bounds and list Google and local-model packages as optional at [requirements](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1).
  - Record the tested environment and add provider-specific installation extras in packaging work.
  - No fresh installation or provider compatibility was validated in this review.

## What should the shared schema contain?

| Proposed record | Required content and invariants |
| --- | --- |
| `RunSpec` | Schema version, family name, preset version, game discriminator, explicit protocol, ordered seats, full seed plan, output location, and source configuration identity. |
| `GameSpec` | Exactly one of item allocation, diplomacy, or co-funding, with validated game-specific fields and no irrelevant-field acceptance. |
| `SeatSpec` | Neutral seat ID, exact catalog key, exact model ID, provider, transport, credential reference, generation controls, and optional validated local asset reference. |
| `ProtocolSpec` | Round count, enabled phases, discussion turns, discount settings, order rule, explicit TTC intervention, explicit access intervention, and optional team protocol. |
| `SeedPlan` | Preference seed, order seed, roster seed, replicate identity, and local generation seeds when local sampling is used. |
| `RosterSnapshot` | Ordered realized models, sampling pool, sampler version, exclusions, sampling and order seeds, and the historical Elo snapshot when the design uses it. |
| `ContextPolicy` | Exact model limits, estimator identity, compaction version, threshold, output reserve, and history limits. |
| `ResolvedRun` | All result-affecting defaults materialized, requested and effective provider controls, source hashes, code identity, dependency versions, and an immutable configuration hash. |
| `ExecutionSpec` | Local or Slurm executor, run concurrency, phase concurrency, explicit transport, resource settings, and bounded timeout and retry policy. |
| `AttemptOutcome` | Attempt ID, terminal status, exception details, actual routes, protocol validity, artifact hashes, usage coverage, and any explicitly allowed recovery events. |
| `BatchOutcome` | Requested run identities, complete run identities, failed or interrupted run identities, incomplete status when applicable, and statistics over the stated eligible set. |

- Store adversary and baseline labels as experiment annotations, while prompts continue to use neutral seat IDs.
- Represent native reasoning effort, reasoning token budget, prompt budget, and access-call count as different intervention types.
- Preserve team membership, communication rules, team objective, preference locks, and protocol version without converting them into generic model metadata.
- Keep historical model aliases and their original interpretation as provenance, while resolving execution identity from a frozen execution catalog.
- Use a configuration schema that rejects unknown keys and validates finite numbers, positive dimensions, valid seat references, and game feasibility.
- Preserve the existing heterogeneous `stable_seed` algorithm and generated roster list when importing historical configurations at [stable seed helper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:303).
- Do not regenerate historical rosters from the current active roster because the generator depends on current pool exclusions and sampling strategy at [generation constants](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:37).

## What are the smallest reusable adapter contracts?

```python
class FamilyAdapter(Protocol):
    def expand(self, request: FamilyRequest, snapshots: Snapshots) -> list[RunSpec]: ...

def resolve_run(spec: RunSpec, catalog: ModelCatalog) -> ResolvedRun: ...

class Executor(Protocol):
    def prepare(self, run: ResolvedRun) -> LaunchPlan: ...
    async def execute(self, plan: LaunchPlan, store: ResultStore) -> AttemptOutcome: ...

class ResultStore(Protocol):
    def acquire_run(self, run: ResolvedRun) -> RunLease: ...
    def begin_attempt(self, run: ResolvedRun) -> AttemptHandle: ...
    def append_event(self, attempt: AttemptHandle, event: RunEvent) -> None: ...
    def finish(self, attempt: AttemptHandle, outcome: AttemptOutcome) -> None: ...

async def run_one(run: ResolvedRun, store: ResultStore) -> AttemptOutcome: ...
```

- These are proposed interfaces, not an implementation patch already applied.
- Family adapters should expand two-player nano, two-player Llama, homogeneous, heterogeneous, homogeneous-adversary, TTC, and coordinated-team requests into the same `RunSpec`.
- Keep game rules in the existing `GameEnvironment` implementations.
- Keep provider response handling behind `BaseLLMAgent`, but require an exact resolved route and an explicit close method.
- The local executor should run one resolved configuration without reconstructing it as many command-line options.
- The Slurm executor should call the same internal single-run entry point with the same resolved configuration file.
- Limit environment variables to credential references and declared execution settings, and store effective execution settings before the subprocess starts.
- Use a common artifact reader for analysis and status reporting.
  - Require exact run and attempt identity instead of choosing whichever filename appears first.
  - Reuse the identity-matching approach in [TTC final-interaction resolution](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/ttc_accounting.py:16).
  - Extend that approach to unnumbered historical results because the existing TTC helper requires a numbered result filename suffix at [filename handling](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/ttc_accounting.py:28).
  - Preserve missing token fields and reuse the explicit token-semantics checks at [TTC accounting](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/ttc_accounting.py:65).
- Keep backward-compatible readers for historical filenames while making new manifests authoritative for new runs.
- A public command can then expose family presets, a configuration-file path, an output path, a seed, and an executor without duplicating the engine.

## What concrete package changes are smallest?

- Install `bargain = strong_models_experiment.cli:main` through a new packaging configuration at `/scratch/gpfs/DANQIC/jz4391/bargain/pyproject.toml`.
- This package target is suitable because its current initializer imports only result dataclasses and the model dictionary before its lazy engine accessor at [package initialization](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/__init__.py:3).
- No provider client is constructed by that initializer in the inspected source.
- Keep the existing engine, game packages, provider implementations, and historical script locations during the initial public-interface patch.
- Add the following small modules, with runtime imports restricted to the execution path.

| Proposed new file | Proposed functions | Boundary |
| --- | --- | --- |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/cli.py` | `build_parser()`, `main(argv=None)` | Parse arguments and route commands without importing the engine. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/schema.py` | `parse_run_spec()`, `validate_run_spec()`, `materialize_defaults()`, `resolved_config_hash()` | Use pure typed records and validation for every family. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/presets.py` | `get_preset()`, `expand_family()`, `load_historical_snapshot()` | Expand versioned presets into explicit run specifications without executing them. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/plan.py` | `build_plan()`, `resolve_run()`, `describe_plan()` | Freeze seeds, rosters, controls, output identities, and job resources before execution. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/paths.py` | `resolve_input_path()`, `resolve_output_path()`, `resolve_model_asset()`, `artifact_reference()` | Resolve against an explicit configuration or installation root instead of the caller's working directory. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/providers.py` | `resolve_seats()`, `required_credentials()`, `offline_doctor()` | Resolve exact routes and check local dependency and credential presence without importing SDK clients or making requests. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/execute.py` | `execute_plan()`, `execute_one()`, `build_worker_command()`, `resume_plan()` | Import the existing engine only after validation and run one isolated negotiation per engine instance. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/launch/store.py` | `acquire_run()`, `begin_attempt()`, `finish_attempt()`, `validate_complete_attempt()`, `read_status()` | Share immutable attempt manifests, atomic terminal writes, resume admission, and artifact discovery. |

- `store.py` is the one additional helper recommended beyond schema, presets, plan, execute, paths, and providers because output ownership should have one implementation.
- Use standard-library dataclasses and explicit validators unless a new schema dependency offers a specific benefit.
- Add versioned capability and preset resources to installed package data before claiming the installed command works outside a source checkout.
- Continue to read historical configuration files through adapters, and preserve their original bytes and source hashes.

| Proposed command | Required behavior |
| --- | --- |
| `bargain run` | Resolve and execute one requested run or one explicit small run plan, then return its outcome. |
| `bargain plan` | Print the complete resolved run plan offline, including model identities, seeds, token controls, expected calls, output targets, and required credentials. |
| `bargain doctor` | Check the supplied plan's installed dependencies, path existence, credential presence, and profile completeness without calling providers or starting a proxy monitor. |
| `bargain sweep --preset paper-... --plan-only` | Expand a named dated paper preset offline and print its exact grid, counts, seed plan, frozen roster, and resource requirements. |
| `bargain resume` | Read an existing plan and skip only completed attempts with matching resolved configuration hashes. |
| `bargain status` | Read saved manifests and show requested, running, complete, failed, interrupted, and incomplete run counts without changing outputs. |

- These command names are proposed, and `paper-...` denotes a preset name that still needs an actual dated configuration snapshot.
- A local worker subprocess can use an internal CLI subcommand that accepts one resolved configuration file and calls `execute_one()`.
- Use the selected interpreter consistently for local execution and record an explicit interpreter in a Slurm profile instead of silently selecting a different environment.
- A plan should label cost as unknown unless a versioned price source and a stated estimation method are available.
- `doctor` should report local readiness only because credential presence does not establish model access or provider availability.
- No planning or status command should construct `StrongModelsExperiment`, since that constructor creates the output directory at [engine initialization](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:58).

| Existing file to patch | Smallest concrete change |
| --- | --- |
| `/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py` | Keep legacy arguments as a compatibility parser, convert them to `RunSpec`, and call the common resolver and executor. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py` | Accept validated resolved settings, require an exact roster, close agents in `finally`, and expose a terminal outcome for every requested run. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py` | Make `create_agents()` all-or-error and make `_create_agent_by_type()` reject unknown or unauthorized routes. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py` | Propagate exhausted parsing or provider failures as failed phase outcomes before any artificial action reaches game execution. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py` | Use resolved provider and retry policy, reject default strategy generation on failure, preserve exact model IDs, and implement deterministic client cleanup. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py` | Let `FileManager` write within an assigned attempt directory and use atomic writes while preserving legacy export names where required. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/data_models.py` | Add requested run counts and structured terminal status without treating no-agreement as an execution error. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py` | Adapt existing configuration dictionaries to `RunSpec`, call the shared executor, and remove unconditional parallel execution and post-hoc configuration rewriting. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py` | Preserve TTC cap and effort choices through the common resolver and make any cap migration explicit. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py` | Retain its historical ID adapter while replacing shared-script subprocess coupling with the common executor. |

- A wrapper-only patch is insufficient because the current engine can skip seats, generate artificial actions, and report partial batches as successful.
- These edits should preserve the existing successful-game rules and public/private phase ordering.
- A full provider-module migration or second negotiation engine is not needed for the initial public release.

## Which patch groups should be implemented?

1. Add the strict schema and pure resolver, then make the root parser and historical adapters call the same resolver.
2. Reject dropped seats, unknown providers, automatic provider substitution, and artificial model actions in new strict runs.
3. Add structured run outcomes, atomic attempt storage, explicit resume checks, and nonzero completion status for partial batches.
4. Extract shared subprocess execution from the multi-agent script and pass a resolved configuration file.
5. Move parser construction before runtime imports and add package metadata for the `bargain` console command.
6. Add separate workstation and Princeton execution profiles with explicit model asset roots, transport, and resource values.
7. Preserve family generators and adapt their outputs without changing historical seed, roster, phase, or token-cap choices.

- This order makes the one-command interface small while putting correctness in shared code.
- Changing provider, model revision, token cap, reasoning effort, sampler, context policy, invalid-output policy, or local sampling behavior creates a new experimental condition.
- A portable run can reproduce the study design only after those values are fixed, and stochastic model responses still need not match historical text.

## Which tests and checks are needed?

- Add offline parser tests that prove `--help`, model listing, and dry-run require no credentials, initialize no clients, and write no run artifacts.
- Compare each family's resolved configuration with representative real historical configurations, including team preference locks and TTC cap-preservation cases.
- Test conflicting metadata, unknown fields, invalid transport, seed zero, missing local assets, unsupported providers, and exact seat-count enforcement.
- Test `parallel_phases=false` through configuration resolution, execution dispatch, and saved metadata.
- Test partial batches, provider failures, malformed model outputs, and interruptions for explicit failed or incomplete outcomes.
- Test stale results, changed model order, changed seeds, missing provenance, non-finite utilities, and artificial-action records for rejection by resume admission.
- Test two processes targeting the same run and ensure only one acquires ownership.
- Test that interrupted writes leave the prior complete manifest readable and preserve every prior attempt.
- Test cleanup after success, provider initialization failure, key changes, and cancellation.
- Extend the existing [phase ordering tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_parallel_phases.py:57), [command tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:219), [attempt-log tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_attempt_logs.py:10), and [transport tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_transport.py:38).
- Update tests that currently require provider substitution or artificial action generation when the public strict policy is introduced.
- After implementation and explicit authorization, run a small real provider smoke test for every advertised transport and a real local-model smoke test for any advertised local profile.
- Unit test doubles can isolate provider boundaries, but they cannot establish that an external integration works.
- Integration validation remains incomplete because this review intentionally made no service calls or model runs.

## Which questions remain open?

- Which dated configuration snapshots should define each public preset, since current code defaults and historical conditions differ?
- Should a short family command launch one negotiation or the family's full statistical grid, and which quantity should a public `runs` option mean?
- Which model aliases remain available through their exact historical routes, since no provider availability check was authorized here?
- Which local Llama variants should be advertised as supported, and what hardware should each explicit local profile require?
- Which historical analyses depend on invalid-output recovery, so the public strict policy can be labeled as a changed experimental condition?
- Which historical configuration fields were operational settings versus study-design settings in each experiment family?
- Which historical result directories contain duplicate attempts that need explicit reader mappings before new common loaders can use them?
