# Launch and scheduler portability review

**Question**

Can a fresh user use the current launch scripts on a local machine or another Slurm cluster?

**Short answer**

The Python runners provide useful building blocks, but the launch layer is not portable yet because it mixes experiment settings with private paths, Della modules, transport selection, and dated result folders.

**What was inspected?**

- This review followed [the review scope](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/review_scope.md:1) and [the project instructions](/scratch/gpfs/DANQIC/jz4391/bargain/AGENTS.md:1).
- The inventory included hidden and ignored shell files, with Git internals, virtual environments, dependency directories, and Python cache directories excluded.
- Only report files were written.
- No experiment, API call, model load, UI server, job submission, package install, credential read, or imported command was run.
- `bash -n` passed for all 229 product and research shell/batch files outside the assistant tooling directory.
- The syntax check does not establish that the emitted jobs can run.

| Population | Discovered | Review depth |
| --- | ---: | --- |
| Tracked shell/batch files | 19 | Launch logic inspected in all 19, including the generated payloads and argument handling in the three large shell generators. |
| Additional standalone shell/batch files | 3 | Inspected the plot batch file and the two paper compilation scripts. |
| Generated or historical files under experiment results | 160 | All scanned for paths, modules, scheduler settings, GPU assumptions, and proxy references; three representative jobs received a targeted content comparison. |
| Historical analysis shell files | 47 | All received the same pattern scan and syntax check; their complete analysis dependency graphs were not reviewed here. |
| Assistant tooling shell files | 34 | Inventoried only, then excluded from the product review. |
| Python source files that emit or submit Slurm jobs | 11 | Launch functions and argument parsers inspected in all 11. |
| UI application entrypoints | 9 | Entrypoint, argument, path, and output behavior inspected in all 9, including the root prompt reviewer. |

- The shell/batch inventory totals 263 files, of which 229 received both content-pattern scans and syntax checks.
- The 19 tracked files comprise 14 shell files and 5 batch files.
- The maintained shell/batch content review covers 22 files, while the generated and historical content review is a scan with targeted examples rather than a line-by-line audit of all 207 files.
- The scan found private `/scratch/gpfs` or `/home/jz4391` references in 202 of 229 files.
- The scan found module commands in 118 files, partition/GPU/constraint directives in 123 files, account or QoS directives in 11 files, H100 mentions in 5 files, and proxy-directory references in 98 files.
- The counts above cover shell and batch files, so Python generator references are additional.

**What blocks a fresh launch?**

- High priority: all three shell generators emit a broken working-directory sequence.
  - Each emitted worker sets `BASE_DIR="bargain"`, enters that directory, and then sources `"${BASE_DIR}/.venv/bin/activate"`.
  - Their submission helpers already enter the repository root, so the worker then attempts to enter a nested directory.
  - Even execution from the parent directory leaves the activation path nested after `cd`.
  - Evidence appears in the [item-allocation generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:368), [co-funding generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:696), and [diplomacy generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:619).
  - The corresponding activation statements are at lines 388, 713, and 636 of those files.
  - The smallest fix is to pass an explicit resolved repository directory to a shared launch renderer.

- High priority: local execution still assumes the private proxy in the main multi-agent runner.
  - [The `run-one` environment](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717) defaults `OPENROUTER_TRANSPORT` to `proxy` and the queue directory to `/home/jz4391/openrouter_proxy`.
  - The OpenRouter client also has a [private default queue directory](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:33).
  - The OpenAI agent has another [private default queue directory](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2694).
  - The [binding-team renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:174) overwrites both OpenRouter and OpenAI queue settings unconditionally.
  - The [team-coordination renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:278) overwrites the OpenAI queue setting unconditionally.
  - A public local execution profile should select direct transport explicitly, while a restricted-network profile should require an explicit shared queue path.
  - The existing Della workflow can retain `/home/jz4391/openrouter_proxy` in a local, untracked site profile.
  - The review assumes that the existing proxy monitor is externally managed and did not inspect or start it.

- High priority: the generated shell environment differs across workflows.
  - [The main multi-agent renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1809) requires `module purge`, `anaconda3/2024.2`, and `proxy/default`.
  - [The TTC renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:341) has the same module requirements.
  - [The appendix renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:449) suppresses module failures and changes to `python3` if the project interpreter is absent.
  - [The random-monoculture renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:716) invokes unqualified `python` without activating the project environment.
  - [The multi-agent command builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535) selects the repository interpreter if present and otherwise uses `sys.executable`.
  - One explicit interpreter field should control both local and scheduled execution, and the launcher should fail if that interpreter is unavailable.

- High priority: generated GPU requests do not follow one current resource policy.
  - The [item-allocation generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:497) emits H100 jobs with `gpu80` and `pli-c` for one, two, and four GPUs.
  - Its other GPU directives appear at lines 584 and 671.
  - [The co-funding generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:882) emits two- and four-A100 jobs.
  - [The standalone Game 1 batch file](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_game1_single_local_config.sbatch:9) requests one A100 with `gpu80` for eight hours.
  - The public default should contain no scheduler GPU type or partition.
  - An explicit Della test profile should use a one-hour limit and the permitted A100/80 GB settings from current cluster instructions.
  - The resource size for an actual local model still needs validation for its model count, precision, context length, and concurrency.

- High priority: model-name routing and provider selection can disagree.
  - [Co-funding submission routing](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:999) assigns four GPUs to `qwen3-32b` by name.
  - [The current registry](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:635) assigns that alias to OpenRouter.
  - [Item-allocation routing](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:972) uses its own local-model list containing only `phi-3-mini-128k-instruct`.
  - [The standard Llama 3.1 8B alias](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:894) uses OpenRouter, while its separate `-cluster` alias uses local weights.
  - Resource selection must use the resolved provider and declared local-model requirements.
  - It must never change a provider or model to make a job fit the available hardware.

- High priority: local weight paths are still tied to a specific directory layout.
  - The registry contains 11 `local_path` entries using a relative `bargain/models/` prefix, including [Llama 3.1 8B](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:919), [Phi-3](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:972), and [Llama 3.2 3B](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1003).
  - [The model registry helper](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:662) repeats that prefix.
  - [The agent factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:538) tests the path relative to the process working directory and returns `None` when it is missing.
  - The public launcher should accept an exact local path or an explicit model-root mapping and fail before creating any agents if a path is missing.
  - A Della onboarding step must inspect `/scratch/gpfs/DANQIC/models` before any download, as the project instructions require.
  - The audit did not inspect model weights or establish that a local model load works.

- Medium priority: comments about `auto` transport do not describe current behavior.
  - [The co-funding worker](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:714) and [diplomacy worker](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:637) describe proxy-first routing but set `auto`.
  - [The client transport order](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:393) is direct first, then proxy.
  - [TTC generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:354) and [seed replication](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:213) also select `auto` by default.
  - New profiles should select one explicit transport and record the actual transport used.

- Medium priority: private credential-file defaults and broad preflight checks prevent predictable onboarding.
  - [TTC submission generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:387) reads a private credential-file location and requires OpenAI, Anthropic, and OpenRouter keys for the entire suite.
  - [TTC seed submission](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:253) checks those three providers even when the caller requests only selected config IDs.
  - [The main multi-agent renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1813) defaults its key file to the relative string `bargain/api_keys.env` after entering the repository root.
  - [The random-monoculture renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:696) repeats that relative default.
  - Credential discovery should use explicit environment variables or an explicitly supplied file and check only the providers needed by the selected configs.
  - The resolved manifest should record credential variable names or key labels, without credential values.

- Medium priority: shell command construction and result-root selection do not safely support arbitrary paths.
  - [The co-funding runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:817) and [diplomacy runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:716) execute assembled strings through `eval`.
  - [The context-pilot renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/context_compaction_pilot.py:487) and [random-monoculture renderer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:691) interpolate unquoted paths.
  - [The item-allocation submit helper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:966) classifies configs through the mutable `scaling_experiment` link, although its batch script uses the timestamped config root.
  - Regenerating that link can make an old helper classify one config set and execute another.
  - Use a command argument list, validated paths, and an immutable manifest of selected config IDs.

- Medium priority: copying existing worker arguments would preserve gaps in config handling.
  - [The diplomacy config](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:396) saves `run_number`, `max_rounds`, and `gamma_discount`.
  - [Its worker command](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:684) always sends run number 1 and omits the saved round limit and discount value.
  - [The item-allocation worker command](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:453) omits the saved item count and round limit.
  - The shared config adapter should check that every result-affecting field reaches the engine or is rejected as unsupported.

**Which current entrypoints can be reused?**

- The following syntax comes from inspected parsers and templates, without execution.
- Replace the example `/absolute/path/` arguments with actual absolute paths.
- Generation commands write experiment artifacts, so they were not run during this review.

| Purpose | Current invocation syntax | Limitation |
| --- | --- | --- |
| Generate item-allocation configs | `bash /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh` | No argument parser or public single-config selection exists in this generator. |
| Generate a minimal diplomacy config set | `bash /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh --derisk` | Emits cluster jobs as well as configs. |
| Generate a minimal co-funding config set | `bash /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh --derisk` | Emits cluster jobs as well as configs. |
| Run one saved multi-agent config | `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py run-one --results-root /absolute/path/to/run --config-id 1` | The environment currently defaults to the private OpenRouter proxy. |
| Run one saved TTC config | `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py --config /absolute/path/to/config.json` | Provider and output behavior still come from the saved config and environment. |
| Generate native TTC configs | `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py --results-root /absolute/path/to/new-run` | `--submit` also submits jobs, while `--dry-run` returns before artifact generation. |
| Clone TTC configs with an explicit seed | `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py --source-root /absolute/path/to/source-run --results-root /absolute/path/to/new-run --seed 984` | Requires the source config set and emits a Della job. |
| Generate binding-team configs | `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py --control-root /absolute/path/to/control-run --output-root /absolute/path/to/new-run` | The module still contains a private repository path. |

- Parser evidence is in [multi-agent arguments](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:100), [TTC runner arguments](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146), [TTC generation arguments](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:492), [TTC seed arguments](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:374), and [binding-team arguments](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:195).
- The item-allocation header advertises `submit_single.sh` and `--local`, but the current generator only writes the array submission helper before ending at [its final instructions](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:1094).
- A public command must therefore call a reviewed runner directly instead of relying on that advertised helper.
- [The standalone Game 1 batch file](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_game1_single_local_config.sbatch:17) derives its root from `BASH_SOURCE[0]`, which also requires a separate Slurm test because the submitted script need not execute from its original source location.
- That batch file requires `GAME1_SINGLE_OUTPUT_ROOT` and a Slurm array task ID, defaults to dated source results, and hard-codes a 16,384 phase token cap at line 93.
- [The co-funding-then-diplomacy wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_cofunding_then_diplomacy.sh:12) depends on two mutable latest-result links and only orders submission calls, without a scheduler dependency between the experiment sets.

**Where should shared launch code replace duplicated templates?**

| Python source | Existing function or boundary | Smallest proposed change |
| --- | --- | --- |
| [Multi-agent batch](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786) | `build_command`, `run_config`, `write_slurm_file` | Keep config and result validation, then inject the same resolved execution profile into local and Slurm paths. |
| [Random monoculture](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:668) | `write_slurm_file`, `submit_selection` | Replace the embedded template and unqualified interpreter with the shared renderer. |
| [Context pilot](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/context_compaction_pilot.py:464) | `write_slurm_script`, `submit` | Preserve pair-task selection and concurrency while sharing path and environment resolution. |
| [Native TTC](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:311) | `write_slurm_script`, `write_submit_script` | Render a selected config manifest using an explicit scheduler profile. |
| [TTC seed replication](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:169) | `write_slurm_script`, `write_submit_script` | Keep source lineage and seed cloning while removing duplicate site setup. |
| [Appendix Llama baseline](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:416) | `_run_config_py`, `_write_slurm_files` | Move config-to-command conversion into a maintained Python API and share the scheduler renderer. |
| [Discount ablation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_discount_factor_ablation.py:155) | `write_sbatch` | Preserve the explicit discount grid and move transport/module choices into the profile. |
| [Team coordination](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:247) | `write_sbatch` | Derive the source root and remove unconditional queue overrides. |
| [Binding team](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:144) | `write_sbatch` | Apply the same root and environment changes while preserving protocol/version lineage. |
| [TTC cap recovery](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py:40) | `submit_recovery` | Keep this as an explicit recovery operation with bounded attempts and a recorded scheduler profile. |
| [TTC cap resubmission](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/resubmit_ttc_cap_recovery.py:42) | `resubmit` | Preserve archived failure records and regenerate a reviewed launch script from the stored profile. |

- The two recovery tools currently archive failed output and submit an existing saved batch file, so they must remain separate from a normal fresh-run command.
- [The cap recovery tool](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py:72) changes the cap from 10,500 to 16,384 and records that change.
- A portable launcher must not apply that recovery automatically.
- The five checked-in batch files under the scripts directory are the standalone Game 1 job, the paper-identifier check, and three Claude qualitative-analysis jobs.
- [The qualitative seed job](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_claude_qualitative_seed.sbatch:14), [supplement job](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_claude_qualitative_supplement.sbatch:14), and [finalizer](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/finalize_ttc_claude_qualitative.sbatch:14) all contain the private repository path and a dated analysis root.
- [The paper-identifier job](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/validate_paper_identifiers.sbatch:12) and [additional plot job](/scratch/gpfs/DANQIC/jz4391/bargain/slurm/recreate_command_r_replacement_plot.sbatch:12) also contain the private repository path.
- These analysis jobs can use the same scheduler renderer, but they are not entrypoints for the requested experiment suite.

**What should the portable execution design do?**

- Proposed API names below describe work to implement and are not existing public commands.
- `resolve_execution_profile(...)` should produce one validated, serializable profile before any config generation or provider initialization.
  - Fields should include execution type, absolute interpreter path, repository root, output root, provider transport, optional proxy directory, and an explicit model-path mapping.
  - Local execution should be the default for new public launches.
  - Local execution means running on the current host and does not imply a CPU model or a change from an API model to local weights.
  - The local profile should contain no module command, Slurm dependency, private queue, account, partition, QoS, or GPU-family default.
  - The selected model/provider configuration must remain independent of scheduler selection.
- `build_launch(config, profile)` should reuse the existing config-to-command builders and return a command argument list, working directory, explicit environment changes, and artifact paths.
  - It should validate credentials by variable name without showing values.
  - It should reject missing weights, missing source controls, stale result paths, unsupported model/provider pairs, and inconsistent agent counts before execution.
  - It should make every result-affecting setting visible in the resolved experiment configuration.
- `render_slurm(launch, scheduler_profile)` should be a pure renderer that requires explicit scheduler selection.
  - Profile fields should include partition, optional account and QoS, time, CPU count, host memory, optional GPU request and constraint, module setup, and shared filesystem roots.
  - A site may declare that account or partition is intentionally omitted, but an unknown site must not inherit Della values.
  - The renderer should write explicit working-directory, interpreter, config-manifest, and log paths.
  - It should quote shell values with one shared implementation and reject newline characters in directive values.
  - It should create log directories before submission because creating them inside the job is too late to establish the initial log destination.
  - It should preserve a stable mapping from each array task to the selected config ID.
  - It should serialize the resolved profile beside the immutable config manifest and record the submitted script hash.
- A preview operation should validate and print the command, resource request, selected models, selected configs, and output paths without starting a job or provider call.
  - Current `submit-selection --dry-run` paths can still write submission artifacts, as shown in [multi-agent submission](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2537) and [random-monoculture submission](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:749).
  - The public preview should state whether it writes files, or use a distinct write-free preview operation.
- Existing generated batch files should remain historical evidence rather than templates for new runs.
  - [The saved scaling API job](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/configs/slurm/run_api_experiments.sbatch:14) contains a correct absolute repository root that differs from the current generator.
  - [The saved TTC seed-2048 job](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed2048_20260727_043613/slurm/run_one.sbatch:5) requests one CPU, 4 GB, two hours, and QoS `short`, while the current generator requests four CPUs, 16 GB, and eight hours without that QoS.
  - [A saved binding-team job](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_092251/slurm/run_binding_team_gpt54.sbatch:14) embeds the private repository path and proxy setup.

**What needs to change in the UI entrypoints?**

| Entrypoint | Current behavior | Proposed portable change |
| --- | --- | --- |
| [Main viewer wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/ui/run_viewer.sh:10) | Derives the repository path, accepts `--port`, and binds to `127.0.0.1`. | Add explicit result-root selection and use the selected interpreter consistently. |
| [Experiment viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/experiment_viewer.py:167) | Discovers only diplomacy/co-funding names under the repository results directory. | Do not present it as a universal viewer for Game 1, multi-agent, TTC, and team output. |
| [Game 1 viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/game1_sample_viewer.py:65) | Accepts `--results-dir` through Streamlit's argument separator. | Route explicit run paths through a common viewer command. |
| [Game 2 viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/game2_batch_viewer.py:55) | Accepts `--results-dir` and resolves saved relative output paths from the repository. | Use an explicit artifact-root contract that also supports relocated results. |
| [Game 3 viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/game3_batch_viewer.py:53) | Uses the same argument and saved-path pattern. | Apply the same artifact-root contract. |
| [Multi-game viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/multi_game_sample_viewer.py:67) | Accepts repeated `--results-dir` and `--results-root` options but has old smoke-folder discovery defaults. | Require an explicit run or manifest for a new public launch. |
| [Random-monoculture/team HTTP viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/random_monoculture_sample_viewer.py:994) | Accepts `--results-root`, `--host`, and `--port`, with loopback as the default host. | Keep loopback and remove the dated default result root. |
| [Binding-team wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/ui/run_binding_team_viewer.sh:4) | Hard-codes the private repository root and a dated binding-team run. | Derive the repository path and require the selected result root. |
| [Annotation review](/scratch/gpfs/DANQIC/jz4391/bargain/ui/behavior_annotation_review.py:37) | Accepts explicit manifest, journal, and export paths but defaults to a dated review. | Keep it as a separate review tool with explicit inputs and writable outputs. |
| [Annotation wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/ui/run_behavior_annotation_review.sh:4) | Hard-codes the private repository path and correctly binds to loopback. | Derive its root and use the common interpreter setting. |
| [Graphics triage wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/ui/run_graphics_triage.sh:68) | Builds a manifest if absent, prints Della-specific tunnel guidance, and omits `--server.address`. | Require explicit inputs and bind to `127.0.0.1`. |
| [Graphics triage application](/scratch/gpfs/DANQIC/jz4391/bargain/ui/graphics_triage_viewer.py:54) | Targets the ICML paper tree and writes decisions or a staging script. | Keep it outside the experiment launch flow and require an explicit paper tree for any future generalization. |
| [Root prompt reviewer](/scratch/gpfs/DANQIC/jz4391/bargain/streamlit_prompt_reviewer.py:33) | Uses fixed prompt-change and decision files under the repository. | Keep it as a separate ancillary tool with explicit input and output paths. |

- The current main viewer can be started with `bash /scratch/gpfs/DANQIC/jz4391/bargain/ui/run_viewer.sh --port 8501` after UI dependencies are installed.
- Direct Streamlit invocations for the sample viewers must place app arguments after `--` and should include `--server.address 127.0.0.1`.
- The current custom HTTP viewer syntax is `/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/ui/random_monoculture_sample_viewer.py --results-root /absolute/path/to/run --host 127.0.0.1 --port 8002`.
- A remote launcher should print an SSH tunnel command for the actual server host, with a jump-host form when needed.
- The local default should print the loopback URL without assuming a Princeton hostname.
- No UI binding, tunnel, page, or image rendering was tested during this review.

**How can portability changes preserve the study design?**

- Keep experiment configuration separate from machine-specific paths and scheduling settings.
- Preserve config IDs, config ordering, seeds, ordered model rosters, model/provider identifiers, game parameters, token limits, and protocol versions.
- [The item-allocation generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:119) currently uses two runs with paired orderings and selects seed 42 for both, despite defining three scenario seeds.
- [The diplomacy generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:376) assigns seeds from `BASE_SEED + EXPERIMENT_ID`, so filtering before generation would change scenario assignments.
- [The full multi-agent parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:106) exposes a master seed and a heterogeneous sampling strategy, both of which must remain explicit in presets.
- [The team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:18) depends on a dated control set and validates its agent counts, competition values, positions, and seed replicates.
- Reconstructing controls for a new team example is different from reproducing the historical paired-control experiment and must be labeled accordingly.
- A saved result may contain absolute output paths, so relocating it needs an explicit source-root mapping that preserves the original path and records the new artifact location.
- Do not silently resolve stale paths by basename, latest-result links, or another directory that happens to exist.
- A move from an API provider to local weights changes the execution provenance and may change results even when the model name looks similar.
- [The local loader](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_clients.py:420) also chooses device and numeric precision from available hardware, so local inference must record those choices.
- Preserve the main runner's [result validation before success](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1752), including failures and attempt history.

**What validation is needed before release?**

- Add source-level tests that compare the exact experiment commands and resolved config values produced for local and scheduled execution.
- Cover a repository path with spaces, an output path outside the repository, a missing interpreter, and a missing config or weight path.
- Verify that the local profile makes no Slurm or module call and never accesses a private queue by default.
- Verify that a Slurm profile preserves array-to-config mappings across selections and chunked submissions.
- Verify that rendering rejects unknown provider aliases and never substitutes a model, provider, or missing input.
- Verify that all relevant setting changes appear in the saved resolved configuration and execution manifest.
- Verify that an API-only roster requests no GPU and that local-model resources come from an explicit profile.
- Verify credential discovery by variable name without logging values.
- Verify that each UI accepts an explicit artifact path and binds to loopback.
- Retain and extend [the existing multi-agent selection-rendering tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:448).
- Review [the existing transport tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_transport.py:38) when introducing explicit public profiles because they currently assert `auto` and direct-first behavior.
- After implementation, perform a real, small API smoke test for each supported transport before claiming that integration works.
- After implementation, perform a real local-weight smoke test on declared hardware before claiming local Llama support.
- After implementation, perform one one-hour Della test job through the rendered profile and verify log paths, queue use, result validation, and recorded provenance.
- The real smoke tests are future validation work and were not authorized or run in this read-only review.

**What remains unresolved?**

- The intended public Llama entrypoint needs an explicit model revision and a declared choice between API hosting and local weights.
- The release needs a decision on whether historical control/config sets will ship as data assets or remain optional reproduction inputs.
- Another cluster's account, partition, QoS, module names, network routing, and model-storage roots cannot be inferred from this repository.
- GPU memory estimates need real measurements for the supported local-model configuration.
- The six tracked reproduction wrappers use environment `python` and some require ImageMagick, Ghostscript, or TeX, so packaging analysis dependencies is separate work.
  - Examples include [the Llama utility reproduction wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig16_llama_utility_replication/reproduce.sh:13), [the baseline payoff wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig17_llama_baseline_payoff/run.sh:4), [the TTC token wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig20_ttc_stage_tokens/run_audit.sh:19), [the consensus wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig22_rounds_to_consensus/commands.sh:4), [the homogeneous-adversary wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig24_hom_adversary_competition/run_reproduction.sh:11), and [the fairness wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/reproduction_audit/fig32_fairness_inequality_efficiency/run.sh:9).
- The two additional paper compilation scripts target older NeurIPS trees and are not evidence of the current ICLR paper build procedure.
  - They are [the retained paper compiler](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/neurips/compile_pdf.sh:4) and [the review-copy compiler](/scratch/gpfs/DANQIC/jz4391/bargain/lh_review_20260830/homogeneous_redesign/paper_draft/neurips/compile_pdf.sh:4).
