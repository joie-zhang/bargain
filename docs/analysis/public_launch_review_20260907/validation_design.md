# Public launch validation gates

**Question**

What must pass before the seven public experiment commands can be described as portable and working?

**Short answer**

Require a clean installation, exact design preservation, strict failure handling, preserved run artifacts, and real smoke evidence for every advertised experiment and provider path.

- This was a source review on 2026-09-07.
- I inventoried 45 test modules and 66 top-level test functions in eight selected batch, provider, team, TTC, and parallel-phase modules.
- I read relevant test bodies, argument parsers, configuration builders, factory paths, game failure handling, result writers, result validation, and TTC attempt resolution.
- I searched the test suite and examined launch paths for cancellation, resume, hard-coded paths, test doubles, and fallback behavior.
- I executed zero tests, experiments, API calls, model loads, downloads, or Slurm jobs.
- No runtime claim follows from this review.
- Only this report was written under [the authorized review scope](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/review_scope.md:1).

**What does the existing suite establish by source inspection?**

- Batch tests cover 2,730 configs, family counts, roster order, phase-cap forwarding, sampling metadata, selection IDs, and completion-file screening ([generation tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:61), [command tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:219), [selection tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:448)).
- Attempt tests preserve an old log when the latest pointer changes ([log test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_attempt_logs.py:10)).
- Parallel tests check ordering and repeated writes within one experiment object ([save test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_parallel_phases.py:372)).
- Team tests cover private information sharing, singleton behavior, captain rotation, the binding objective, and scripted planning/voting ([team tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_team_coordination.py:47), [binding test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_team_coordination.py:193)).
- Provider tests use replaced network boundaries, and the local Llama test replaces the loader and forces paths to exist ([transport test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_transport.py:61), [local test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_gpu_llama_phase_caps.py:236)).
- TTC tests preserve requested archived caps, reject scientific config changes, and require a unique matching interaction identity ([seed tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:27), [attempt tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:17)).
- No public-command subprocess test, moved-installation test, or cancellation/resume test was found in the examined suite.
- Test imports add the checkout to `sys.path`, so those imports do not prove wheel installation works ([test setup](/scratch/gpfs/DANQIC/jz4391/bargain/tests/conftest.py:7)).

**Which release gates should be decisive?**

| Gate | Required evidence | Current source-review status |
| --- | --- | --- |
| 1. Portable installation | The documented source checkout and wheel install work from a moved path, a separate working directory, and an output directory outside the repository, with the historical tree unavailable. | Not established. |
| 2. Pure public planning | `strong_models_experiment.cli:main` parses and validates all seven families before provider imports, with no credentials, model loads, queue requests, job submissions, or result changes. | Proposed interface. |
| 3. Frozen design parity | The new plan matches a versioned manifest for all scientific fields, ordered rosters, seeds, prompts, and effective caps. | Useful existing tests, but current validators depend on fixed grids or archives. |
| 4. Strict execution | Every requested seat exists, provider/model identities remain fixed, and provider/parse failures end as failed or incomplete without synthetic actions. | Blocked by current production recovery paths. |
| 5. Preserved results | Exclusive run ownership, separate attempt directories, atomic writes, exact config identity, finite outcomes, and unchanged completed artifacts on resume. | Current reuse and output paths do not meet this gate. |
| 6. Cancellation and completion | Cancellation terminates the child, records unfinished work, preserves prior attempts, and never reports a partial batch as wholly successful. | No examined cancellation handler, and partial batches can return success. |
| 7. Real smoke evidence | Each advertised family/game and distinct provider/transport/control path completes a real run whose artifacts load from the new external output root. | Not performed or authorized in this review. |

- Run each experiment in its own process because the current engine uses global random state.
- Keep test doubles only at external or nondeterministic unit-test boundaries.
- Do not add a fake provider or synthetic experiment mode to the public interface.
- Reuse real family smoke runs to cover the provider matrix where possible.
- Do not infer a provider integration works from a mock response or generated batch script.

**Which source findings require changes before those gates can pass?**

- **Production failures can become synthetic game actions.**
  - Proposal repair creates a synthetic default when parsing still fails ([proposal recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2386)).
  - A hard provider error during voting creates rejection votes with `hard_failed=False` ([voting recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3670)).
  - Co-funding commit failure creates a synthetic `nay` ([commit recovery](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:4710)).
  - Existing tests require these outcomes, so their acceptance expectations must change while preserving raw-error diagnostics ([diplomacy expectations](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_diplomatic_treaty_batch_voting.py:278), [co-funding expectation](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_cofunding_phases.py:734)).
  - A valid no-consensus game must remain a valid outcome, while an unsuccessful provider or parse operation remains failed or incomplete.

- **Provider selection and roster creation are not strict throughout the stack.**
  - The factory has a native-to-OpenRouter fallback, and runtime fallback separately defaults to enabled ([factory routing](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183), [runtime routing](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:578)).
  - The factory skips unknown models or missing local paths, and only requires at least one agent ([factory loop](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:137), [local path check](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:538)).
  - The engine pairs created agents with requested model names using `zip`, which can mislabel the remaining seats after a skipped model ([seat mapping](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:266)).
  - Tests must reject missing native credentials even when an alternate provider key exists, and must reject an incomplete roster before any request.

- **A reused result can lack necessary provenance or contain contaminated outcomes.**
  - Current validation checks numeric conversion and optional config ID, seed, and game matches, without requiring finite values, exact roster, complete logs, or clean vote integrity ([result validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269)).
  - A passing result is skipped and then rewritten with metadata from the current config ([reuse](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1668), [metadata write](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1619)).
  - Require a resolved-config hash and exact result/interaction identity before reuse, with unchanged completed result bytes.
  - Test NaN, infinity, extra/missing seats, changed roster, absent config hash, contaminated votes, truncated logs, and duplicate attempt identities.

- **Attempts preserve logs but still share result locations.**
  - The wrapper uses the same output directory for repeated attempts ([run wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656)).
  - Result lookup checks two canonical filenames rather than an attempt identity ([lookup](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1258)).
  - Writers use direct truncating writes, and batch name disambiguation has one-second precision ([writer](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:131), [name selection](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150)).
  - Generation can overwrite configs and change a sibling latest pointer ([config writes](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1141), [pointer write](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1238)).
  - Test conflicting launches, same-second attempts, interrupted writes, and explicit retry without changes to earlier attempts.

- **Completion states and cancellation need an explicit contract.**
  - The child wait has no surrounding cancellation finalizer ([child wait](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1735)).
  - The progress file contains counts and the last interaction, not restorable game, conversation, and random-generator state ([progress writer](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1464)).
  - The batch catches run errors, aggregates successful runs, and can reach the CLI's success return ([batch behavior](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1026), [CLI return](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:706)).
  - Initially define resume as skipping verified complete cells and explicitly retrying unfinished cells in new attempts.
  - Do not promise continuation from the middle of a game.
  - Test graceful cancellation and forced death, including the interval between writing a result and finalizing status.
  - Preserve an uncertain remote request and do not replay it automatically, because cancellation cannot undo a provider request already accepted.

**How should portability and parity be tested without the historical tree?**

- Create a future disposable release test root with `mktemp -d /tmp/bargain-public-XXXXXXXX`.
- Use a machine or container where the original cluster paths are absent, because a second checkout on this cluster can still access old files.
- Install the exact release checkout and built wheel into fresh environments, then invoke the commands from a separate empty directory.
- Test an install path containing spaces and an output path outside the repository.
- Do not inherit the existing virtual environment, `PYTHONPATH`, provider variables, proxy settings, or Slurm variables.
- Move the checkout and completed smoke output, then verify that planning and analysis resolve paths from explicit configuration and manifests.
- Test each advertised dependency extra because Google, xAI, Transformers, and PyTorch are currently commented out in the requirements file ([dependencies](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:34)).
- Require offline help, plan, and validation to make no network requests, write no queue entries, and load no weights.
- Use one schema-negative test set for invalid counts, booleans as counts, non-finite values, game bounds, indices, unknown keys, unknown models, missing assets, and malformed metadata.
- Check missing and invalid metadata because the current CLI logs and ignores malformed metadata JSON ([metadata handling](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:388)).
- Check seed `0` because random-order setup currently seeds only when the value is truthy ([random setup](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:161)).
- Compare canonical plans to frozen manifests, excluding only declared operational fields such as output roots and timestamps.
  - Preserve game cells, seeds, ordered model rosters, role maps, seat positions, roster sampling method, preference parameters, and Elo inputs where used.
  - Preserve discount factors, round limits, phase order, discussion turns, parallel settings, prompt hashes, effective phase caps, reasoning controls, and team protocol.
  - Do not resample the heterogeneous roster when a user selects a frozen experiment.
  - Record intentional scientific differences under a new design version.
- Separate scientific design parity from identical live model responses or payoffs.
- Check TTC cap preservation because 10,500 migrates to 16,384 unless explicitly preserved ([cap resolver](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28)).
- Export a small versioned set of real source configs with source hashes for mandatory config regression tests.
  - Two current TTC seed tests unconditionally require the archived 216-config result tree ([archive dependency](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:21)).
  - Keep optional archive result checks separate from public-installation acceptance ([optional archive test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:109)).
  - Never replace observed research evidence with generated results.
- Save the release hash, code version, dependency versions, resolved plan hashes, command exit codes, and actual test summary.

**What real smoke evidence is required later?**

- No smoke run is authorized by this review.
- Use one seed and a small explicit round limit in labeled smoke configs.
- Require actual model responses, valid protocol completion, finite outcomes, full provenance, and successful loading from the external output root.
- Agreement and favorable utility are not smoke acceptance requirements.

| Family | Real smoke coverage |
| --- | --- |
| Two-player GPT-5-nano | Each advertised game with both requested seats on the resolved nano route. |
| Two-player Llama | Each advertised game on the selected API route, plus actual GPU loading/generation if local execution is advertised. |
| Homogeneous multi-agent | Each advertised game at `N=4` with the declared common model. |
| Heterogeneous multi-agent | Each advertised game at `N=4` with a frozen ordered roster and exact per-seat model/provider records. |
| Homogeneous-adversary | Each advertised game at `N=4` with both advertised adversary endpoint positions. |
| TTC | All advertised games across the smoke set and every distinct provider/transport/reasoning-control schema, including lowest/highest efforts and any extended-cap option. |
| Coordinated team | Current Game 1 protocol at `N=4`, both adversary positions, a matched uncoordinated control, and the `N=2` singleton-team case. |

- Resolve the provider from the actual model configuration rather than a model-family label.
- Cover native OpenAI, OpenRouter, native Anthropic, native Google, xAI, or local execution only when that path is advertised.
- Cover direct and file-queue transport separately when both are advertised.
- For queue smoke, use the externally managed monitor through the request queue.
- For advertised Slurm execution, require a real job exit state and saved artifacts.
- For Della GPU smoke, inspect the shared model store first and follow the non-H100 80 GB, one-hour test policy.
- Retain requested alias, resolved model ID, returned model identity when available, provider, transport, requested effort, effective cap, usage semantics, attempt IDs, and errors.
- Preserve unknown token usage as unknown and explicit provider-reported zeros as zero.
- Hosted alias changes can prevent exact historical reproduction even when the smoke passes.

**Which minimal code changes enable the gates?**

- Implement `strong_models_experiment.cli:main` with pure parsing, schema checks, and planning before provider imports.
- Add per-config validation separately from the current 2,730-config grid validator ([grid validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1851)).
- Reuse existing model resolution, phase-cap resolution, command builders, and game creation after strict validation.
- Dispatch one process per run with explicit immutable configuration and an exact roster.
- Add run ownership, separate attempt directories, atomic final writes, complete status counts, and strict result validation.
- Reuse the TTC requirement for exactly one interaction log matching the final experiment identity ([attempt resolver](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/ttc_accounting.py:16)).
- Replace synthetic-action completion and unapproved provider substitution with bounded errors that preserve diagnostics.
- Keep historical artifacts unchanged and record that strict failure handling can change completion rates and analysis exclusions.

**What current invocation syntax is confirmed?**

- The existing engine parser accepts `--models`, `--game-type`, `--random-seed`, `--max-rounds`, and `--output-dir` ([engine parser](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:47)).
- A nano pair uses `--models gpt-5-nano gpt-5-nano`, and an API Llama pair can use `--models llama-3.3-70b-instruct llama-3.3-70b-instruct`.
- The inspected nano alias uses native OpenAI, while Llama 3.3 70B uses OpenRouter ([nano config](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265), [Llama config](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:778)).
- The multi-agent parser exposes `generate --results-root`, `validate --results-root`, and `run-one --results-root --config-id` ([batch parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:101)).
- The TTC wrapper accepts `--config /absolute/config.json --dry-run` ([TTC parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146)).
- The team generator accepts `--output-root /absolute/output --control-root /absolute/controls`, but still hard-codes the current checkout root ([team parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:293), [team root](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:17)).
- These are source-confirmed interfaces, not successful runtime checks.

**Which release choices remain unresolved?**

- The Llama command must distinguish an API baseline from an optional local model.
- The coordinated-team command must identify the private-sharing protocol, binding protocol, or separate variants.
- The final presets must declare the supported games, model/provider routes, transport paths, and frozen scientific designs.
- The future live smoke plan must name the exact configurations and approved resource/call budget.
