# Two-player Llama launch review

**Question**

- Can a fresh user launch the Llama 3.3 70B replication for Games 1–3 with one command?

**Short answer**

- Most experiment code exists, but the current generator is a cluster job generator and does not reproduce the saved May token settings.
- Add a small public launcher that uses a fixed 500-run preset, the existing configuration builders, and the existing game runner.
- Fix transport selection, output manifests, and resolved model provenance before describing the command as a faithful replication.

**Review scope and evidence**

- This review inspected source, argument parsers, saved configuration JSON, and one saved result without running experiment code, tests, API requests, or jobs.
- The checked-out Git commit is `444fcf9c368a60dd40a0f5f7aa8b4033552312bf`, but references describe the files visible during this review.
- Only this report was written, and the separate paper dependency audit was not read or changed.
- The paper names 10 adversary models and 500 Llama runs in [/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:438](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:438).

**What already works at the interface level?**

- The generator is [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:587](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:587).
  - Its parser accepts `--suffix`, `--slurm-partition`, `--slurm-time`, `--cpus-per-task`, `--mem`, and `--force`.
  - It always builds all three games and validates counts of 140, 180, and 180 before writing each game, at lines 602–635.
  - It has no game, adversary, seed, token limit, output root, local executor, or launch option.
  - The default suffix is `202605`, so it collides with the existing May directories in this workspace.
  - `--force` overwrites configuration and helper files within an existing result root, at lines 561–583.
- The reusable pure builders are `_game1_configs(run_name)`, `_game2_configs(run_name)`, and `_game3_configs(run_name)`, at lines 70, 110, and 149.
  - `_models_for_order` supplies the actual model list, at line 58.
  - `_write_index` already produces useful role, parameter, seed, run number, configuration filename, and output fields, at line 211.
  - `_run_config_py` generates a per-game Python wrapper that translates JSON into existing runtime arguments, at line 284.
- The runtime accepts the three game names, ordered `--models`, `--batch --num-runs 1`, `--run-number`, `--random-seed`, and game parameters in [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:51](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:51).
  - There is no general `--config` argument on this runtime entry point.
  - The generated wrapper passes `--batch --num-runs 1` and the saved run number, at generator lines 316–342.
  - The runtime builds `experiment_config` and calls `StrongModelsExperiment.run_batch_experiments`, at runtime lines 551–617 and 698–704.
  - `override_run_number` preserves the exact supplied seed, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1005](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1005).
  - The shared runner selects the item-allocation, diplomacy, or co-funding environment, at lines 196–240.

The following current syntax was established by source inspection and was not executed.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py \
  --suffix public_20260907
```

- This command would create the current 16,384-token configuration set without submitting jobs.
- After generation, this command would run one Game 1 configuration with credentials already exported in the shell.

```bash
OPENROUTER_TRANSPORT=direct OPENAI_TRANSPORT=direct OPENROUTER_PROVIDER_FALLBACK=0 \
  /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_public_20260907/slurm/run_config.py \
  /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_public_20260907/configs/config_0000.json
```

- The generated Slurm helper supports `--start-id`, `--end-id`, and `--delay-seconds`, at generator lines 505–523.
- This command would submit one configuration after generation.

```bash
bash /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_public_20260907/slurm/submit_individual.sh \
  --start-id 0 --end-id 0
```

- There is no current command that generates and launches all three games together, as shown by the generator's final instruction at line 642.

**Which experimental settings must the public preset preserve?**

| Setting | Llama replication | Relation to GPT baseline |
|---|---|---|
| Fixed baseline | `llama-3.3-70b-instruct` | Replaces `gpt-5-nano` |
| Adversary roster | 10 aliases in a fixed order | A subset of the 30-alias GPT generator roster |
| Game 1 grid | 7 competition values × 2 orders = 140 runs | Matches the GPT two-discussion-turn grid |
| Game 2 grid | 3 rho values × 3 theta values × 2 orders = 180 runs | Matches GPT `--conservative` grid |
| Game 3 grid | 3 alpha values × 3 sigma values × 2 orders = 180 runs | Matches GPT `--conservative` grid |
| Discussion turns | 2 in every game | GPT Game 1 generator also emits a one-turn ablation |
| Maximum rounds and discount | 10 and 0.9 | Same declared settings |
| Game 1 seed | 42 for every cell and both orders | Same current GPT scenario seed |
| Games 2–3 seed | `42 + experiment_id`, separately for each game | Same formula but different roster positions |
| Game 1 run number | 1 for baseline first, 2 for adversary first | Same current GPT order convention |
| Games 2–3 run number | 1 for both orders | Same GPT conservative convention |
| Saved May phase limit | 10,500 in all 500 input configurations | Must be pinned separately from current defaults |

- The Llama roster and grids are declared in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:20](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_appendix_llama33_baseline_configs.py:20).
  - The adversary order is Nova Micro, Claude 3 Haiku, Nova Pro, GPT-4o mini, DeepSeek V3, Claude Sonnet 4, DeepSeek R1-0528, Gemini 2.5 Pro, GPT-5.4-high, and Claude Opus 4.6 Thinking.
  - Preserve exact aliases from the source instead of rebuilding the roster from current Elo order.
  - `weak_first` means baseline first and `strong_first` means adversary first, regardless of relative capability, at lines 58–67.
- GPT Game 1 has 30 adversary aliases, seven competition values, two runs, and both one-turn and two-turn settings in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:70](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_configs_both_orders.sh:70).
  - Its current default generation therefore contains 840 configurations, including 420 with two discussion turns.
  - The baseline comparison loader selects two discussion turns for Game 1, at [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:1331](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:1331).
- GPT Games 2–3 require `--conservative` to match the Llama grids, at [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:220](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_diplomacy_configs.sh:220) and [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:272](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_cofunding_configs.sh:272).
  - The scripts' default or `--model-scale` modes are different designs.
  - Current GPT seed assignment also uses experiment ID, at diplomacy line 376 and co-funding line 451.
- Games 2–3 do not hold the sampled scenario fixed between speaking orders.
  - Saved Llama Game 2 configuration 0 uses seed 42 and configuration 1 uses seed 43 for the same parameter cell, at [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605/configs/config_0000.json:26](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605/configs/config_0000.json:26) and [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605/configs/config_0001.json:26](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605/configs/config_0001.json:26).
  - The same pattern occurs for Game 3 at configuration lines 32.
  - Changing baseline or filtering adversaries before assigning experiment IDs changes scenario seeds in Games 2–3.
  - Generate the full preset before selecting a smoke-test or resume subset.
  - A newly matched baseline comparison would be a new design and must not replace historical seed assignment silently.
- Game 3 must pass `c_max=30`, since the standalone runtime parser defaults to 50 at [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:160](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:160).
  - Preserve five projects, costs 10–30, `own` discussion transparency, commit voting, and time discount 0.9 from generator lines 175–188.

**Why is current generation not an exact May rerun?**

- All 500 saved May input configurations specify `max_tokens_per_phase=10500`, as verified with read-only JSON aggregation.
  - There are 140 Game 1 configurations and 360 combined Game 2–3 configurations.
  - Game 1 contains only seed 42, while Games 2–3 each span seeds 42–221.
  - None of these saved configurations contains `model_config_overrides`.
  - Representative evidence appears at [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_202605/configs/config_0000.json:20](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game1_202605/configs/config_0000.json:20), [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605/configs/config_0000.json:18](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game2_202605/configs/config_0000.json:18), and [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game3_202605/configs/config_0000.json:18](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/appendix_llama33_baseline_game3_202605/configs/config_0000.json:18).
- The current generator hardcodes 16,384 at lines 97, 135, and 174, and provides no argument to change it.
- The runtime catalog can further change the effective phase limit.
  - GPT-5.4-high and Opus 4.6 Thinking each declare 65,536-token phase caps in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:3](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:3), with policies at lines 1033 and 1252.
  - Their special policy chooses the model cap when the experiment limit equals 16,384, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1049](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1049).
  - Nova Pro and Nova Micro declare 5,120-token caps, while Claude 3 Haiku declares 4,096, at catalog lines 7–10, 825, 1202, and 1213.
  - These lower caps combine with the experiment cap by taking the minimum, at phase handler line 1060.
  - A configured global limit therefore does not fully describe the runtime request limits.
- The saved input aliases resolve against the current registry on every run, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:60](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:60).
  - The public preset must pin resolved provider, model ID, temperature, reasoning settings, phase caps, and recovery policy.
  - Historical effective provider settings need a separate check against saved interactions and the historical code version before an exact-match claim is possible.
- The generated wrapper always records `appendix_batch="llama33_baseline_202605"`, even with a new suffix, at generator line 390.
  - The new launcher should record preset identity and execution identity as separate fields.
- Saved May files are absent from a normal checkout because experiment directories are ignored at [/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:60](/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:60).
  - `git ls-files` confirmed that the generator is tracked and the three representative May configuration files are not tracked.
  - Publish the compact historical preset or manifest in a tracked source directory.

**Which credentials and routes does this roster use now?**

| Alias | Current model ID | Current route |
|---|---|---|
| `llama-3.3-70b-instruct` | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| `amazon-nova-micro-v1.0` | `amazon/nova-micro-v1` | OpenRouter |
| `claude-3-haiku-20240307` | `anthropic/claude-3-haiku` | OpenRouter |
| `amazon-nova-pro-v1.0` | `amazon/nova-pro-v1` | OpenRouter |
| `gpt-4o-mini-2024-07-18` | `gpt-4o-mini-2024-07-18` | OpenAI |
| `deepseek-v3` | `deepseek/deepseek-chat` | OpenRouter |
| `claude-sonnet-4-20250514` | `anthropic/claude-sonnet-4` | OpenRouter |
| `deepseek-r1-0528` | `deepseek/deepseek-r1-0528` | OpenRouter |
| `gemini-2.5-pro` | `google/gemini-2.5-pro` | OpenRouter |
| `gpt-5.4-high` | `openai/gpt-5.4` | OpenRouter |
| `claude-opus-4-6-thinking` | `claude-opus-4-6` | Anthropic |

- The route table comes from [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:655](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:655), with the other entries at lines 737, 778, 818, 1025, 1071, 1101, 1154, 1195, 1206, and 1239.
- A complete run currently requires `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`, or equivalent configured key groups.
  - The CLI checks the requested models' resolved `api_type` before execution, at [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:398).
  - `GOOGLE_API_KEY` is not required for this roster because its Gemini entry uses OpenRouter.
  - Grouped variables are discovered only when `LLM_KEY_GROUP_ORDER` is set, at [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:83](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:83).
  - The runtime and inspected provider packages have no `load_dotenv` call, so merely creating an environment file does not load it for a local launch.
- The Llama baseline uses an API and requires no local GPU or model download.
- Use explicit transports in a public launcher.
  - `OPENROUTER_TRANSPORT=direct` selects HTTPS and `proxy` selects the shared file queue, at [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:393](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:393).
  - The default `auto` tries HTTPS and can change to the queue after a connection failure, at lines 649–662.
  - Native OpenAI supports `OPENAI_TRANSPORT=proxy` and `OPENAI_PROXY_POLL_DIR`, while `auto` chooses the queue under Slurm, at [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686) and line 2742.
  - Changing `OPENROUTER_PROXY_POLL_DIR` alone does not change the OpenAI queue directory.
  - Native Anthropic still calls the SDK directly at line 2515, so a restricted cluster needs working native Anthropic egress or an explicitly implemented same-provider transport.
  - The queue monitor accepts a supplied URL and handles chat-completion responses, at [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:194](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_proxy_monitor.py:194), so current OpenAI queue support must not be mistaken for Anthropic streaming support.
- Native provider failures can switch to OpenRouter by default through `OPENROUTER_PROVIDER_FALLBACK`, at [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:578](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:578).
  - This runtime path records the provider change in response metadata at line 676.
  - The factory has a separate automatic fallback on missing or exhausted native keys, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:183).
  - That factory path does not consult `OPENROUTER_PROVIDER_FALLBACK`.
  - A public strict policy must cover both paths and reject any missing agent, since the factory currently skips unknown or unavailable agents and only checks that the returned list is nonempty, at lines 137–165.
- Queue requests contain authentication headers, at [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:546](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:546).
  - Use a private queue directory and exclude queued requests from public result bundles.
  - This review did not inspect any credential file, credential value, or queued request.

**What prevents a fresh user from launching or analyzing the batch?**

- Generated scripts embed the generation machine's repository path and run root, at generator lines 297 and 444–445.
  - Regenerating after moving the source checkout derives a new path, but moving already generated scripts leaves their paths stale.
- The Slurm template assumes partition `cpu`, six hours, one CPU, 8 GB, specific modules, and a personal credential file, at generator lines 449–469 and 589–594.
  - It defaults to `/home/jz4391/.config/bargain/api_keys.env` and `/home/jz4391/openrouter_proxy`.
  - `BARGAIN_API_KEYS_ENV` can change the key-file location, but a missing file is silently skipped.
  - Module failures are suppressed, and a missing project interpreter causes a switch to `python3`.
  - CPU jobs do not require the GPU launch workflow.
- The generated submit helper submits one job per configuration with no concurrency ceiling, at generator lines 533–550.
  - It truncates the prior job manifest on every invocation, at line 528.
  - Submission should retain every attempt and validate all selected inputs before submitting any job.
- The dependency list at [/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1) supplies OpenAI, Anthropic, HTTP clients, NumPy, SciPy, and analysis libraries with lower version bounds.
  - It supplies no reproducible dependency lock or Python-version contract.
  - Runtime code uses syntax such as `str.removeprefix` and evaluated `tuple[str, ...]`, so the public package must declare and test a supported Python version.
- Results use `run_<number>_experiment_results.json`, with timestamp suffixes on collisions, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150) and line 206.
  - The Llama analysis locator tries the unsuffixed file, a run-1 alternative, and then the first glob match, at [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_appendix_llama33_baseline_500.py:148](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_appendix_llama33_baseline_500.py:148).
  - Reusing a result directory can therefore leave multiple attempts while analysis chooses an older file.
  - A launcher needs exact result paths and attempt status in a manifest.
- The Llama analyzer already accepts `--game1-root`, `--game2-root`, `--game3-root`, and `--output-dir`, at analyzer lines 95–101.
  - It reads each game's configuration index and rejects missing results or unresolved utilities, at lines 270–367.
  - It always requires 500 completed rows, at lines 1163–1169.
  - It resolves relative configuration output paths against the repository, at lines 104–108.
  - It then calls `relative_to(PROJECT_ROOT)` without handling external roots, at lines 333–334 and 892.
  - A public `--output-root` outside the repository requires changes to this loader and report writer.
- The cross-baseline analyzer has reusable `BaselineSpec` and `load_baseline_rows`, but its main routine uses hardcoded source and output roots without a CLI, at [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:50](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_n2_baseline_comparison.py:50), line 775, and line 1723.
- Elo analysis depends on a dated Markdown file and the entire active 30-model roster, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:9](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/analysis/active_model_roster.py:9) and line 138.
  - Record the exact Elo file hash with a public run instead of consulting a future edited table without notice.

**What is the smallest proposed implementation?**

- Add the Llama preset to the shared public launcher proposed for all experiment families.
  - A proposed user command is `python -m bargain.launch two-player --baseline llama33 --preset llama33-202605 --games all --executor local --output-root /absolute/output/root`.
  - This command does not exist today.
  - The preset must declare 500 rows, the exact 10-model order, the existing seed formulas, the 10,500-token historical input cap, all protocol settings, and pinned model definitions.
  - Treat a current-settings run or a matched-seed comparison as a named new preset.
- Extract configuration-to-runtime translation from the generated string into one importable function.
  - Reuse `_game1_configs`, `_game2_configs`, `_game3_configs`, `_models_for_order`, and `_write_index`.
  - Reuse `StrongModelAgentFactory.resolve_model_config` and `model_config_overrides` for explicit per-run routing and model settings.
  - Reuse `StrongModelsExperiment.run_batch_experiments(num_runs=1, override_run_number=...)` or the current subprocess entry point for execution.
  - Preserve subprocess isolation for concurrent runs because the engine sets global random state.
  - Ensure overrides are carried through the launcher because the existing Llama wrapper metadata list at generator lines 388–398 drops a `model_config_overrides` field added to JSON.
- Add `--dry-run`, selection, concurrency, and explicit transport support to the common launcher.
  - `--dry-run` should resolve all required fields, dependencies, credential names, model routes, seeds, counts, and exact commands without importing provider clients or writing to existing outputs.
  - Filtering must preserve original configuration IDs and seeds.
  - A subset should be recorded as incomplete relative to the full preset.
  - The local executor should use the invoking interpreter and an explicit transport choice.
  - The Slurm executor should require an explicit site profile and retain all submission records under the chosen output root.
  - Required site settings include partition, time, memory, CPUs, interpreter, transport, queue paths, and native Anthropic network access.
- Save one resolved manifest before execution.
  - Include schema version, preset version, code commit, relevant file hashes, ordered models, neutral agent IDs, seed, all result-affecting defaults, provider/model IDs, reasoning settings, actual phase limits, transport, and expected output paths.
  - Save actual provider and transport metadata per request.
  - Never include credential values.
  - Use a new attempt directory for a retry and make its relationship to the failed attempt explicit.
- Make one strict recovery policy control the CLI, factory, and provider call paths.
  - Reject an unknown model, wrong agent count, invalid metadata, unavailable required input, or exhausted allowed route.
  - Do not change provider, model, temperature, or phase cap unless the saved preset explicitly permits that bounded recovery.
  - Keep the current one-run failure behavior that returns failure when zero runs succeed, at [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1031](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1031).
- Adapt analysis to consume the manifest and exact attempt paths.
  - Keep the 500-run assertion for the historical full preset.
  - Give smoke runs and incomplete runs an explicit status report.
  - Use the existing `rel` helper pattern from the cross-baseline analyzer at line 215 for display paths outside the repository.

**What validation is needed before release?**

- Add pure preset tests that compare all 500 generated rows with a tracked historical fixture after removing only output-root fields.
  - Assert 140/180/180 rows, complete parameter grids, exact roster order, agent order, run numbers, seeds, and protocol fields.
  - Assert that selecting a single adversary or resuming failed rows does not renumber seeds.
- Test the complete configuration translation and resolved effective phase caps.
  - Cover the 10,500 historical limit, current 16,384 behavior, the 65,536 special policies, and lower provider caps.
  - Existing related tests are in [/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_gpu_llama_phase_caps.py:69](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_gpu_llama_phase_caps.py:69).
- Test strict credential and routing behavior at the external-service boundary.
  - Missing OpenAI or Anthropic credentials must not produce an OpenRouter replacement or a one-agent experiment.
  - Both OpenRouter and OpenAI must use the configured queue root.
  - Strict recovery must cover factory construction and failures after construction.
  - Existing tests describe current automatic fallback behavior in [/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_provider_fallback.py:490](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_openrouter_provider_fallback.py:490).
- Test generation and analysis after moving a checkout and placing the output root outside the repository.
  - Include an existing output collision, duplicate attempts, missing results, and a partial submission failure.
- Test the declared dependency lock in a clean supported Python environment.
- After launch authorization, run a real smoke configuration per game and at least one call for each required route.
  - The zero-index configuration uses only OpenRouter models, so three zero-index game smokes do not validate the OpenAI and Anthropic routes.
  - Validate native Anthropic access from the actual restricted executor separately.
  - This review performed no runtime or integration validation.

**Which questions remain unresolved?**

- The historical effective model/provider settings and dependency versions are not fully pinned by the saved input configurations.
- Public availability and present behavior of the historical provider model IDs were not checked because no API or network validation was authorized.
- A supported Slurm site profile must name how native Anthropic traffic reaches its endpoint.
- Exact historical replay and a new matched-seed baseline comparison are different deliverables, so the public preset names must make that distinction explicit.
