# Runtime and source portability review

**Question**
What prevents the experiment code from running in a new checkout on another machine?

**Short answer**
The main engine uses mostly movable source imports, but launch scripts, proxy defaults, model paths, and saved result locators still depend on the original machine or directory layout.

**What did this review inspect?**

- This was a static review on 2026-09-07 under the restrictions in [the review scope](/scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/public_launch_review_20260907/review_scope.md:1).
- No experiments, API requests, jobs, imports of experiment modules, environment changes, or tests ran.
- Only this report was written.
- The source scan covered 208 files and 118,099 lines.
  - The file types were 195 Python files, eight shell files, and five Slurm files.
  - The table below lists the complete source corpus by location.
  - Pattern scans covered every file in this corpus, followed by source reads of the relevant launch, provider, configuration, and loader code.
  - The review inspected test source where it could establish existing coverage, but the 46 test files are outside the 208-file total.
- Historical outputs, experiment data, paper trees, reproduction-audit trees, prior review trees, notebooks, environments, and the 42 files under the paper-figure source directory were excluded.
  - Retained analysis source was included because executable source can contain path dependencies even when its inputs are historical.
  - A script's presence in the corpus does not establish that it belongs in the public entry point.

| Source location | Files |
| --- | ---: |
| `/scratch/gpfs/DANQIC/jz4391/bargain/negotiation` | 14 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/game_environments` | 8 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment` | 19 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils` | 7 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/ui` | 15 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/visualization` | 3 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts`, excluding its two source subdirectories | 97 |
| `/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis` | 43 |
| The two root Python entry points | 2 |
| Total | 208 |

- The two root entry points were [the experiment runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:1) and [the prompt reviewer](/scratch/gpfs/DANQIC/jz4391/bargain/streamlit_prompt_reviewer.py:1).
- The exact-prefix scan found 71 matching lines in 41 files for `/scratch/gpfs/DANQIC` or `/home/jz4391`.
  - There were 27 files containing the scratch prefix and 19 containing the home prefix, with five files containing both.
  - These are textual counts, not 71 independent defects.
- The corpus contains `sys.path` edits in 41 files and explicit `Path.cwd()`, `os.getcwd()`, or `os.chdir()` calls in six files.
  - These counts do not capture all working-directory dependencies because relative `Path(...)`, `open(...)`, shell `cd`, and config values also depend on the working directory.
- No tracked symbolic links were reported by the index scan of the source locations and model/environment locations.
  - The current nested `/scratch/gpfs/DANQIC/jz4391/bargain/bargain` directory contains only a `.claude` directory at its top level.
  - The current `/scratch/gpfs/DANQIC/jz4391/bargain/models` directory is absent.

**Which defects block the requested public commands?**

- **High priority: a local multi-agent command selects the cluster proxy by default.**
  - [The multi-agent runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717) sets `OPENROUTER_TRANSPORT=proxy` and uses `/home/jz4391/openrouter_proxy` when the caller has not supplied these environment variables.
  - This applies to `run-one` outside Slurm as well as to cluster jobs.
  - [The OpenRouter client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:245) has the same queue default and [creates directories and queues requests](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:530) there.
  - A fresh user can encounter a permission error or wait for a monitor that is not watching that location.
  - The present workaround for a networked local run is an explicit `OPENROUTER_TRANSPORT=direct` value, provided that the requested provider and model remain unchanged.
  - The smallest fix is to resolve the execution profile before starting a run and require an explicit shared queue for proxy mode.

- **High priority: native OpenAI uses a separate implicit transport decision and the same private queue default.**
  - [Native OpenAI configuration](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686) defaults to `auto`, and [its transport selector](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742) chooses the proxy whenever `SLURM_JOB_ID` is present.
  - [The native OpenAI queue default](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2694) is `/home/jz4391/openrouter_proxy`.
  - [The OpenRouter monitor](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:25) watches the OpenRouter queue environment variable, so configuring only `OPENAI_PROXY_POLL_DIR` does not configure the monitor.
  - The OpenRouter monitor expands a home abbreviation, but [the OpenRouter client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:530) and [native OpenAI client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2694) do not perform the same expansion.
  - Resolve each queue to an absolute path once and pass the same path to the client and monitor.
  - Preserve the existing provider and model identifier when selecting a transport.

- **High priority: the older two-player shell generators emit an incorrect nested checkout path.**
  - [The item-allocation generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:368) emits `BASE_DIR="bargain"`, changes into that directory, and then [sources another path beginning with `bargain`](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:388).
  - Its [generated submission helper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:753) already changes into the checkout root before submission.
  - The same pattern appears in [the diplomacy generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:619) and [the co-funding generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:696).
  - The item-allocation generator repeats the path in four job templates at lines 368, 503, 590, and 677.
  - Fix the generated script to receive an absolute checkout root and an explicit Python executable.
  - Do not repair this with nested checkout copies or user-specific symbolic links.

- **High priority: coordinated-team generators use the original checkout even when given new output paths.**
  - [The team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:17) and [the binding-team generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:16) set `PROJECT_ROOT` to `/scratch/gpfs/DANQIC/jz4391/bargain`.
  - The binding-team source also prepends that path to `sys.path`, which can import code from the old checkout when it remains accessible.
  - Their generated jobs use that checkout and interpreter and unconditionally overwrite proxy queue settings at [team lines 262-285](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:262) and [binding-team lines 159-187](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:159).
  - Derive the source root from the installed package or script location and accept the scheduler profile separately.
  - Preserve the supplied environment settings instead of overwriting the queue path.

- **High priority: local model aliases depend on a missing nested model directory.**
  - Eleven `local_path` entries in [the model roster](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:919) begin with `bargain/models/` and refer to ten distinct model directories.
  - [The local agent factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:538) checks that string relative to the process directory.
  - The factory returns `None` when a local path is absent, and [the enclosing agent loop](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:159) appends only agents that were created.
  - A missing local model can therefore remove a requested seat before downstream checks, rather than fail at model resolution.
  - [The older model registry helper](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:662) also supplies `bargain/models/` as its relative base.
  - Add an explicit model root or per-model absolute path and raise an error if any requested model is missing.
  - Use locally installed weights before any download, and keep hosted Llama and local Llama as distinct declared execution routes.
  - [The local loader](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_clients.py:414) does not request `local_files_only=True` and chooses device and precision from the available hardware.
  - A portable local-model profile must record the weight revision, tokenizer revision, device, precision, and generation settings.

**Which dependencies can change a result without an obvious path error?**

- **The dated Elo/context table is runtime input.**
  - [Context compaction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:15) uses `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md` through a relative default.
  - [Its loader](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:97) tries the process directory before the checkout-relative location and permits `NEGOTIATION_MODEL_CONTEXT_DOC` to replace the input.
  - If the file is missing, the loader returns an empty table before applying the known provider limits at [line 132](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:132).
  - [The agent](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:1246) then leaves the messages uncompacted because no context limit was resolved.
  - [Roster and Elo loading](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:142) reads the same document from a fixed checkout-relative location.
  - [The heterogeneous pool filter](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:200) reads its Elo, routing, and context columns.
  - Keep the exact versioned catalog in the public package and fail when a required entry is missing.
  - Save its content hash with the resolved roster and context limits.
  - Updating that catalog must be a declared new experiment configuration because it can change both prompt handling and roster sampling.

- **The optional tokenizer changes prompt budgeting.**
  - [The token estimator](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:157) tries `o200k_base`, then `cl100k_base`, then a character-count estimate.
  - [The requirements file](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:31) leaves `tiktoken` commented out.
  - A fresh installation can use a different estimate from the original environment and cross a compaction threshold on a different call.
  - Declare the estimator and encoding in the resolved run configuration and fail if the selected estimator is unavailable.
  - Stage the selected tokenizer assets for restricted compute nodes and record their version.

- **Some generated manifest fields describe a queue without controlling it.**
  - [The random-monoculture manifest](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:463) writes a fixed `openrouter_proxy_poll_dir` field.
  - [The coalition-pilot manifest](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_gpt54_game1_coalition_pilot.py:264) writes fixed `api_transport` and `api_proxy_poll_dir` fields.
  - The source scan found these field names only at their write sites, while the clients obtain transport and queue settings from environment variables.
  - These literals are provenance defects rather than direct queue-selection code.
  - Materialize transport metadata from the actual resolved client configuration before a run begins.

**Can a copied experiment directory be used in a new location?**

- A copy is not sufficient for every launcher or viewer.
  - [The multi-agent output resolver](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:220) trusts an absolute `config["output_dir"]` and resolves a relative value against the checkout root.
  - It does not resolve the output against the `--results-root` directory passed to `run-one`.
  - A copied config can therefore still write to or read from its original experiment tree.
  - [The TTC runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:114) passes the saved output path directly to the main runner and [sets the subprocess directory to the checkout](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:169).
- Team generation needs control results as well as control configs.
  - [Its control loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:59) reads configs under `--control-root`.
  - [Treatment construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:112) then uses the config's saved output path to find the original control result.
  - [The treatment record](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:167) contains exact preferences and control hashes that must survive relocation.
  - Replacing these missing controls with regenerated preferences would change the matched study.
- Loader path rules differ across consumers.
  - [The Game 2 viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/game2_batch_viewer.py:66) resolves its CLI input against the process directory but [saved output locations](/scratch/gpfs/DANQIC/jz4391/bargain/ui/game2_batch_viewer.py:122) against the checkout.
  - [The older diplomacy analysis loader](/scratch/gpfs/DANQIC/jz4391/bargain/visualization/visualize_diplomacy.py:152) uses the saved output location directly and skips missing result files.
  - [The direct experiment constructor](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:62) uses a process-relative default result directory.
- Introduce a versioned result locator with paths relative to the declared experiment root.
  - Preserve the original absolute path in a separate provenance field.
  - Require an explicit source-root mapping when importing older absolute-path manifests.
  - Verify control and result hashes after relocation.
  - Do not search several unrelated result trees and choose the first match.

**Which imports and external resources need a public contract?**

- Most `sys.path` edits derive the root from `__file__` and permit a checkout to move as a unit.
  - Examples include [the main runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:35), [the multi-agent wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:26), and [the general viewer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/experiment_viewer.py:49).
  - These are not user-path defects, but they assume the current source layout.
  - The package scan found no project `pyproject.toml`, `setup.py`, or `setup.cfg` that installs a command or declares package resources.
  - A small installed CLI can remove this layout dependence, with the context catalog declared as a package resource.
- Credentials use several different file conventions.
  - [The multi-agent wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1521) parses the checkout's environment file without a shared configuration loader.
  - [Its Slurm template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1813) instead defaults `BARGAIN_API_KEYS_ENV` to a nested relative file.
  - [The TTC template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:345) defaults that variable to `/home/jz4391/.config/bargain/api_keys.env`.
  - [The main runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398) reads provider credentials from exported environment variables.
  - Use one explicit credential-file option with documented environment precedence and an early error for a missing requested provider credential.
  - No credential values or credential-file contents were inspected in this review.
- The Slurm templates embed a cluster environment.
  - Examples require `anaconda3/2024.2`, `proxy/default`, a `cpu` partition, fixed time/resource requests, and a checkout-local environment at [TTC lines 316-362](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:316) and [multi-agent lines 1786-1844](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1786).
  - Make scheduler settings a named site profile and keep the local execution profile independent of Slurm and module commands.
  - Retain the existing Della network constraint in the Della profile.
- Interpreter selection is inconsistent.
  - [The multi-agent wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535) prefers the checkout environment and otherwise uses `sys.executable`.
  - [The TTC wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:91) uses `sys.executable`.
  - Generated Slurm scripts often require the checkout environment even when the launcher was started with another interpreter.
  - Use the selected interpreter consistently and save its version and installed dependency record.
- Dependencies are not locked by a tested runtime profile.
  - [Core requirements](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1) provide lower bounds and leave Google, Transformers, and PyTorch commented out.
  - [A retained scaling analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_scaling_mechanisms.py:16) imports `statsmodels`, which is absent from those requirements.
  - Publish separate tested dependency groups for API execution, local models, viewers, and retained analysis.
- Some auxiliary tools require Unix behavior.
  - [The annotation storage layer](/scratch/gpfs/DANQIC/jz4391/bargain/ui/behavior_review_core.py:6) imports `fcntl` unconditionally.
  - [The experiment log code](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:16) and [provider log code](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:21) omit file locking when `fcntl` is unavailable.
  - Shell helpers use Bash features such as `mapfile` at [the TTC submission template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:271).
  - State the supported operating systems and shell versions instead of implying that the cluster scripts support every platform.

**Which remaining matches are executable, and which are provenance text?**

| Match group | Meaning and proposed treatment |
| --- | --- |
| [Binding-team viewer launcher](/scratch/gpfs/DANQIC/jz4391/bargain/ui/run_binding_team_viewer.sh:4) and [annotation viewer launcher](/scratch/gpfs/DANQIC/jz4391/bargain/ui/run_behavior_annotation_review.sh:4) | Both execute `cd` against the original checkout and need a derived root. |
| [xAI client queue](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:3056) and [xAI monitor queue](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/xai_proxy_monitor.py:12) | Both use a process-relative nested queue without a queue-path option, and the monitor creates it without `parents=True` at line 64. |
| [Token recovery input](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/recover_ttc_reasoning_tokens.py:79) and [vote recovery input](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/recover_gpt54_pilot_config4_votes.py:33) | These read archived proxy responses outside the checkout and require an explicitly supplied archive for historical recovery. |
| [Qualitative judge queue](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_claude_qualitative_judge.py:27) | This is an executable queue default, although its parser already offers `--proxy-dir`. |
| [Binding-team analysis root](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_binding_team.py:17), [team-behavior root](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_team_behavior.py:18), and [scaling-mechanism input](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_scaling_mechanisms.py:19) | These are executable analysis paths, so retaining the tools requires explicit data-root inputs. |
| [Binding-team dynamics default](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_binding_team_dynamics.py:19), [scaled-payoff input](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_game1_n_scaled_payoff.py:14), and [ceiling-normalized input](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/plot_game1_ceiling_normalized_roles.py:14) | These select historical input trees and belong behind explicit analysis inputs. |
| [Generated case-study links](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_qualitative_dynamics_trend_report.py:975) | The six absolute result links are report text, not executable file reads at those lines. |
| [Generated adjudication handoff command](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/prepare_ttc_gpt5_nine_seed_codex_adjudication.py:117) | This is an instruction string emitted into an artifact, so it breaks the generated workflow rather than the generator's file access. |
| [Graphics download example](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/build_graphics_triage_html.py:33) | This is a documentation example, not an executed transfer. |
| [OpenRouter monitor host hint](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:344) | The Della hostname appears in an error hint and does not select the request destination. |
| [Session utility default](/scratch/gpfs/DANQIC/jz4391/bargain/utils/session_manager.py:16) | This creates process-relative private session storage and is outside the experiment engine's launch path. |

- The retained analysis scripts with fixed scratch roots are executable input/output selectors, not harmless strings.
  - The affected roots are in [coalition proposal rate](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_game1_coalition_proposal_rate_vs_elo_20260823.py:22), [coalition Sankey](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/plot_coalition_sankey_20260823.py:16), and [high-agent-count coalition audit](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/audit_hetero_g3_highn_minimum_coalition.py:14).
  - They also occur in [Gemini qualitative analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_gemini_coalition_qualitative.py:21), [coalition proposer analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_game1_coalition_proposer_elo_20260823.py:22), and [strict coalition audit](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/audit_strict_coalition_proposals_20260817.py:26).
  - The remaining retained roots are in [minimum coalition reporting](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/build_minimum_winning_coalition_report_20260817.py:20), [coalition conversion plots](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/plot_game1_coalition_conversion_by_family_20260823.py:13), and [coalition acceptance plots](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/retained_analysis/plot_coalition_proposed_accepted_by_game_20260823.py:19).
- The remaining queue/template matches occur in [Llama appendix generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:453), [discount ablation generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_discount_factor_ablation.py:179), [context pilot generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/context_compaction_pilot.py:494), and [random-monoculture job generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:705).
- The fixed-root historical launch or repair files are [qualitative seed jobs](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_claude_qualitative_seed.sbatch:9), [qualitative supplement jobs](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_claude_qualitative_supplement.sbatch:9), [qualitative finalization](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/finalize_ttc_claude_qualitative.sbatch:9), [paper identifier validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/validate_paper_identifiers.sbatch:7), and [the historical N=2 merge repair](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/validate_and_merge_n2_figure4_missing15_repair.py:13).
  - Do not make historical repair or paper operations part of a new experiment launch.
  - No deletion decision is made by this review.

**What is the smallest reusable implementation?**

- Deliver an offline `plan` operation before provider execution.
  - Its inputs are the declared experiment profile, roster, seeds, game parameters, paths, and versioned runtime catalog.
  - Its output is the fully resolved configuration and an exact local command that can be reviewed without credentials or a network connection.
  - Keep filesystem materialization an explicit planning output option.
  - Build plans through pure configuration functions before initializing an experiment runner or provider client.
- Deliver local API execution against that same resolved configuration next.
  - Here, local execution means that the experiment process runs on the user's machine, while the selected model can still use its declared API provider.
  - A local-weight Llama mode is a separate declared profile with its own dependencies and weight inputs.
- Add Slurm rendering and submission as separate site-profile operations after local execution is validated.
  - A site profile must not change the experiment roster, model route, reasoning configuration, preference values, or statistical design.
- Add a shared runtime-path and execution-profile resolver at the CLI boundary.
  - Resolve the checkout or installed-resource root, experiment root, model root, credential source, interpreter, provider transport, and queue paths once.
  - Pass the resolved objects to existing generators and runners instead of reading environment variables in several layers.
  - Require explicit proxy paths and preserve one shared path between each client and its monitor.
- Add a shared result-locator API such as `resolve_run_output(config, run_root, source_root_map)`.
  - Use a schema version to distinguish new experiment-root-relative paths from existing checkout-relative paths.
  - Keep original paths and hashes as provenance when an explicit relocation map is applied.
- Retain the existing model and experiment APIs.
  - Keep ordered model rosters, roles, seeds, game parameters, token caps, provider routes, reasoning settings, and matched preferences unchanged.
  - Replace path resolution at their boundaries before considering a wider rewrite.
- Generate Slurm files from a site profile containing modules, partition, resources, interpreter, transport, and shared queue.
  - Use the resolved absolute source root when rendering the job and provide a local execution profile that runs the same resolved experiment configuration.
  - Prepare the Slurm log directory before submission because job output destinations must be available when the scheduler opens them.
- Treat context metadata and token estimation as experiment configuration.
  - Require a catalog version, content hash, estimator name, and estimator version in the saved resolved configuration.
  - A declared compatibility profile may preserve an old estimator, but a missing package must not silently select it.

**Which active source files need changes?**

- This list is deduplicated by source file and separates the core public interface from cluster-only and optional paths.
- A listed file can need a shared-configuration call-site change even when it has no machine-specific path literal.
- The table is a proposed patch boundary, not a claim that every file is currently unusable.

| Active source file | Smallest proposed change |
| --- | --- |
| [Main experiment runner](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:386) | Receive the resolved execution configuration and absolute output root after offline planning. |
| [OpenRouter client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:232) | Receive explicit transport and queue settings from the profile. |
| [Native model agents](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686) | Use the same explicit transport/path contract and fail when required context metadata is unavailable. |
| [Context compaction](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:94) | Load a versioned resource and a declared token estimator independently of the working directory. |
| [Provider failure reporting](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:577) | Resolve report destinations under the declared experiment root. |
| [Experiment storage](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:58) | Require resolved output paths at run creation. |
| [Strong-model agent factory](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:159) | Require every requested seat to resolve before execution. |
| [Active roster loader](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:9) | Consume the versioned runtime catalog and record its hash. |
| [Multi-agent batch wrapper](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:220) | Expose pure planning and shared result location, interpreter, credential, and transport resolution. |
| [TTC configuration runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:91) | Reuse the same resolved run configuration and result locator. |
| [Team configuration generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:17) | Derive the source root and resolve matched control data through an explicit input mapping. |
| [Binding-team configuration generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:16) | Use the shared source root and execution profile. |
| [Team result analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_team_coordination.py:171) | Resolve treatment and control files through the new result locator. |
| [Binding-team result analysis](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_binding_team.py:17) | Derive its source root and reuse the team loader. |

- **Separate Slurm/profile changes are needed if these launchers are retained.**
  - [OpenRouter monitor](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:25) needs the same explicit absolute queue contract as its clients.
  - [TTC generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:311) and [TTC seed replication](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:169) need profile-based job rendering.
  - [Llama baseline generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:416) needs profile-based job rendering if used by the public two-player Llama command.
  - [Item-allocation shell generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:368), [diplomacy shell generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:619), and [co-funding shell generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:696) need absolute source/interpreter paths or replacement by the shared Python configuration interface.
  - The job-rendering changes inside the two team generators and multi-agent wrapper belong to this separate profile layer even though their configuration functions serve the core interface.
- **Optional local-weight support needs additional source changes.**
  - [The model roster](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:919) needs model-root-relative resource names or explicit absolute weight locations.
  - [The local model client](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_clients.py:412) needs an explicit offline weight, device, and precision contract.
  - [The older model registry](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/model_config.py:654) needs the same model-root contract only if that registry remains a supported entry point.
- Optional viewers, xAI proxy tools, retained analysis, historical repairs, and provenance-only text are listed in the preceding section and are outside the core launch patch.
- New code would be needed for the installed CLI and shared resolver, but no such files were created in this review.

**What current invocation syntax is established from source?**

- The commands below show current parser syntax and were not executed.
- `/absolute/checkout/bargain`, `/absolute/runs/example`, and `/absolute/shared/proxy` are explicit placeholders for user-supplied absolute paths.
- Provider credentials must already be exported for the exact models in the selected configuration.

```bash
OPENROUTER_TRANSPORT=direct OPENAI_TRANSPORT=direct \
  /absolute/checkout/bargain/.venv/bin/python \
  /absolute/checkout/bargain/scripts/full_games123_multiagent_batch.py \
  run-one --results-root /absolute/runs/example --config-id 1

/absolute/checkout/bargain/.venv/bin/python \
  /absolute/checkout/bargain/scripts/run_ttc_native_config.py \
  --config /absolute/runs/example/configs/config_0001.json --dry-run

OPENROUTER_PROXY_POLL_DIR=/absolute/shared/proxy \
  /absolute/checkout/bargain/.venv/bin/python \
  /absolute/checkout/bargain/negotiation/openrouter_proxy_monitor.py
```

- [The multi-agent parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:125) establishes the `run-one` arguments.
- [The TTC parser and early return](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146) establish `--config` and `--dry-run` without an experiment subprocess in that branch.
- [The monitor entry point](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:25) uses environment settings rather than a public argument parser.
- The monitor command documents installation syntax for a new user and is not a request to start the externally managed Della monitor.
- The first command still needs compatible config paths because the current loader does not rebase copied manifests.

**What tests and validation are needed before release?**

- Test pure path resolution from a checkout whose path contains spaces and from a working directory outside the checkout.
  - Check both CLI input paths and paths loaded from manifests.
  - Check explicit relocation of a historical config while preserving its hashes and roster.
- Test that a missing local model fails before any agent is created or provider is called.
- Test that missing context catalogs, catalog entries, and selected token estimators fail with clear input names.
- Test that the same resolved catalog, roster, seeds, and estimator generate the same experiment configuration across execution profiles.
- Test that generated Slurm scripts preserve supplied proxy paths and use the chosen interpreter and absolute checkout root.
- Test that a declared local profile produces no requirement for Slurm, cluster modules, or a file proxy.
- Extend the existing [transport tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_transport.py:52) because they currently assert the shared-home queue default.
- Extend the existing [batch-generation tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_full_games123_batch_generation.py:219) and [viewer test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_negotiation_sample_viewer.py:16) to cover relocated roots.
- Run real, bounded provider and local-model smoke tests only in a later authorized implementation phase.
  - Unit tests at external boundaries can use test doubles, but they cannot establish that a provider route or a local model works.
  - External integration validation remains incomplete in this review.

**What remains unresolved?**

- The public operating-system support policy and supported Python/dependency versions are not declared in an installable project manifest.
- The public shared-queue location and monitor deployment procedure are not yet defined.
- The exact local-model weight and tokenizer revisions to distribute or document are not established by the source scan.
- The release must identify the frozen roster/context catalog and tokenizer estimator that each reproduction profile requires.
- Public availability of matched team-control results and historical proxy-response archives was not established because historical data trees were excluded.
- A new portable run can preserve the experimental design, but this static review does not establish numerical equivalence to historical provider runs or to runs on different local hardware.
