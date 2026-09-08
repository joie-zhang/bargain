# TTC public launch review

**Question** Can a new user launch the paper TTC design or a small TTC run with one command?

**Short answer** The repository has usable config builders and a single-config runner, but it needs a portable plan, explicit provider and failure policies, and correct attempt selection before it can offer that interface.

- This review inspected source, argument parsers, tracked-file presence, archived configs, and all 2,160 terminal result JSON files named by the retained TTC inventory.
- No experiment, API call, scheduler submission, test suite, environment change, or code change was run.
- Only this report was written.
- The findings describe the current checkout on 2026-09-07 and do not establish current API availability.

**What design must the public interface preserve?**

- The paper names GPT-5, Claude Sonnet 4.6, and Gemini 3 Flash with four effort levels each in [/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:234](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:234).
  - “Claude 4.6” must resolve to Sonnet for this suite.
  - Opus would change the experiment.
- The design contains `3 families × 4 efforts × 9 game cells × 2 seat orders × 10 seeds = 2,160 runs`.
  - The seeds are `42, 984, 526, 423, 1024, 128, 256, 612, 2048, 4096` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:24](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:24).
  - Every run has two agents, 10 maximum rounds, two discussion turns, and discount `0.9` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:19](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:19).
  - Both orders use the same generated game instance before analysis averages the orders in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:127](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:127).

| Game | Three cells | Other settings | Source |
|---|---|---|---|
| Item allocation | Competition `0.0, 0.5, 1.0` | Five items | [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:126](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:126) |
| Treaty bargaining | `rho=1, 0, -1`, with `theta=1` | Five issues | [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:151](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:151) |
| Participatory budgeting | `(alpha,sigma)=(1,1), (0.5,0.6), (0,0.2)` | Five projects, costs `10–30`, own-contribution discussion, commit vote, discount `0.9` | [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:178](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:178) |

| Participant | Catalog aliases | API route and requested model | Native effort controls |
|---|---|---|---|
| GPT-5 adversary model | `gpt-5-{minimal,low,medium,high}-effort` | Direct OpenAI, `gpt-5-2025-08-07` | `reasoning_effort` uses the four named values |
| Claude Sonnet 4.6 adversary model | `claude-sonnet-4-6-effort-{low,medium,high,max}` | Direct Anthropic, `claude-sonnet-4-6` | `thinking.type=adaptive` and `extra_body.output_config.effort` |
| Gemini 3 Flash adversary model | `gemini-3-flash-thinking-{minimal,low,medium,high}` | OpenRouter, `google/gemini-3-flash-preview` | `reasoning.effort` uses the four named values, with `exclude=true` |
| Baseline model | `gpt-5-nano` | Direct OpenAI, `gpt-5-nano` | Low effort is inserted at the API call site |

- GPT-5 mappings are explicit in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:201](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:201).
- Gemini mappings are explicit in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:387](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:387).
- Claude mappings are explicit in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:451](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:451).
- Effort positions are ordinal settings within each family, not equal token budgets across providers.
  - Claude's `low, medium, high, max` positions align with the displayed positions for GPT-5 and Gemini's `minimal, low, medium, high`.
  - Matching positions do not establish equal reasoning tokens, latency, cost, or total computation.
  - Native effort and observed token usage must remain separate fields.
- The baseline catalog entry omits `reasoning_effort` in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265).
  - The OpenAI client supplies `low` in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2840).
  - This is a result-affecting implementation default that the new resolved plan must state explicitly.
- The native TTC runner does not send a numeric reasoning budget in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:91](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:91).
  - The effort aliases control the API requests across phases.
  - The public CLI must not replace those aliases with `--reasoning-token-budget`, which also adds prompt instructions in [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:619](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:619).

**Which entry points exist now?**

- The native generator accepts only `--results-root`, `--submit`, and `--dry-run` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:492](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:492).
  - It builds exactly 216 seed-42 configs at a 16,384-token cap.
  - It has no seed, model-family, effort, game-cell, or run-count selector.
  - Generation writes JSON configs and Slurm wrappers even when the user intends to execute locally.
  - `--submit` runs the generated submission script and submits every config.
- The archived replication generator accepts `--source-root`, `--results-root`, `--seed`, and `--dry-run` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:374](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:374).
  - It defaults to seed `984` and a local archived source root.
  - It requires exactly 216 source configs with IDs `1..216` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:70](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:70).
  - It preserves the archived 10,500-token cap and records source hashes and code state in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:300](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:300).
  - It generates one seed at a time and has no `--submit` option.
  - Its generated submission script accepts selected numeric config IDs in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:270](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:270).
  - Passing seed `42` fails the exact-change check because the seed and seed label do not change in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:117](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py:117).
- The single-config runner accepts `--config` and `--dry-run` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:146).
  - It launches the shared experiment script with `--batch --num-runs 1 --run-number 1` for generated configs.
  - The fixed run number prevents seed offsets in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1005](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1005).
  - The runner forwards the child process exit status.
- The historical YAML is explicitly marked as a historical record in [/scratch/gpfs/DANQIC/jz4391/bargain/configs/test_time_compute_scaling.yaml:9](/scratch/gpfs/DANQIC/jz4391/bargain/configs/test_time_compute_scaling.yaml:9).
  - Neither current TTC generator reads that YAML.
  - Its older model roster and numeric budget sweep must not become the public TTC preset.

These are current parser-supported examples, established by source inspection and not executed.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py --results-root /tmp/bargain-ttc-native --dry-run

/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_seed_replication_jobs.py --source-root /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943 --results-root /tmp/bargain-ttc-seed984 --seed 984 --dry-run

/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python /scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py --config /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943/configs/config_0001.json --dry-run
```

- Removing `--dry-run` from a generator writes files.
- Removing `--dry-run` from the runner starts paid model calls and also exposes the legacy-cap migration described below.

**Which problems block a fresh user or change the experiment?**

- **The paper plan is absent from a normal checkout.**
  - `git ls-files` lists the TTC scripts but does not list the sampled archived config or retained inventory.
  - The archive is ignored by [/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:60](/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:60).
  - Analysis data are ignored by [/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:69](/scratch/gpfs/DANQIC/jz4391/bargain/.gitignore:69).
  - A fresh user cannot use the replication generator without a separately distributed source plan.
  - The replication test also requires the local archive in [/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:40](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:40).
- **An arbitrary output root does not control native result placement.**
  - `resolve_results_root()` accepts an arbitrary absolute root in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:240](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:240).
  - `build_configs()` reconstructs each output path from `results_root.name` under the repository result directory in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:266](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:266).
  - For example, `/tmp/bargain-ttc-native` would hold the plan while outputs would go under `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/bargain-ttc-native`.
  - Native generation overwrites existing config and manifest files in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:458](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:458).
  - Repeated native submission truncates its submission ledger and submits every config again in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:407](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:407).
- **The runner changes explicit token caps.**
  - A cap of `10500` becomes `16384` unless `preserve_config_max_tokens_per_phase=true` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:28).
  - Larger values are clamped unless `allow_extended_max_tokens_per_phase=true`.
  - String booleans and numeric values have no strict schema validation before these decisions.
  - The source config at [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943/configs/config_0001.json:15](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943/configs/config_0001.json:15) has cap `10500` without the preservation field.
- **Provider substitution is enabled unless the environment disables it.**
  - Runtime OpenRouter fallback defaults to enabled in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:290).
  - Native and OpenRouter keys enter the same route sequence in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:688](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:688).
  - GPT-5 fallback removes the snapshot suffix and requests `openai/gpt-5` in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:336](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:336).
  - Claude `max` is rejected as an incompatible OpenRouter effort by [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:416](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:416).
  - The shared CLI checks required credentials before execution in [/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:419](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:419).
  - The factory still has its own fallback path for key exhaustion in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:199](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:199).
  - That factory path does not consult the runtime environment switch.
- **Provider metadata is not a complete record of the executed request.**
  - Gemini configs say `target_provider=Google`, although the catalog sends requests through OpenRouter.
  - The TTC runner sends a fixed metadata list and drops recovery fields, cap-control fields, and any run-scoped catalog overrides in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:123](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_ttc_native_config.py:123).
  - `target_model_id` is metadata while `models` controls catalog lookup, so changing only the ID does not pin the executed model.
  - Fallback adds `provider_fallback` and a different `model_used` to `AgentResponse` in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:675](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:675).
  - Standard interaction saving receives token usage and the original agent's model name in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3660](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3660).
  - Token extraction omits `provider_fallback` and `model_used` in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:898](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:898).
- **The current shared engine creates synthetic actions.**
  - Exhausted proposal repair can produce a synthetic proposal in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2386](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2386).
  - The defaults include all items for the proposer, treaty midpoints, or zero contributions in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:731](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:731).
  - Voting task errors, including hard provider failures, become synthetic rejects with `hard_failed=False` in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3670](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3670).
  - These are recorded substitutions, but they still conflict with the supplied research and failure policies.
  - A public TTC wrapper alone cannot fix this engine behavior.
- **Current analysis can accept incomplete or substituted evidence.**
  - `collect_run_rows()` skips missing results and assigns zero to missing agent utilities in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:61](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:61).
  - Its complete-grid check rejects `hard_failed`, but does not reject `contaminated` or synthetic actions in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:105](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:105).
  - The retained panel loader assumes the unsuffixed transcript is the final transcript in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:76](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:76).
  - The file writer creates suffixed output files on repeated execution in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/utils/experiment_utils.py:150).
  - A repeated successful run can therefore be saved while the loader still selects the prior unsuffixed result.
  - `repo_relative()` rejects result roots outside the repository in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:56](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:56).

**What did the saved terminal results establish about token caps?**

- A read-only JSON inventory found all 2,160 terminal results listed in [/scratch/gpfs/DANQIC/jz4391/bargain/analysis/ttc_complete_family_seed_panels_20260810/terminal_run_inventory.csv:1](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/ttc_complete_family_seed_panels_20260810/terminal_run_inventory.csv:1).
- Counts below use `result.config.max_tokens_per_phase`, rather than the original planned config.

| Family | 10,500 | 16,384 | 65,536 | Total |
|---|---:|---:|---:|---:|
| GPT-5 | 717 | 3 | 0 | 720 |
| Claude Sonnet 4.6 | 651 | 68 | 1 | 720 |
| Gemini 3 Flash | 720 | 0 | 0 | 720 |
| All families | 2,088 | 71 | 1 | 2,160 |

- The three GPT-5 cap increases include seed `984`, config `68`, whose saved cap is `16384` in [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed984_20260725_025700/gpt-5/level_high/game3_alpha_1p0_sigma_1p0/baseline_first/seed_984/run_1_experiment_results.json:66](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed984_20260725_025700/gpt-5/level_high/game3_alpha_1p0_sigma_1p0/baseline_first/seed_984/run_1_experiment_results.json:66).
- Seed `612`, config `142`, has saved cap `65536` in [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed612_20260727_043613/claude-sonnet-4-6/level_max/game3_alpha_0p5_sigma_0p6/baseline_first/seed_612/run_1_experiment_results.json:66](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed612_20260727_043613/claude-sonnet-4-6/level_max/game3_alpha_0p5_sigma_0p6/baseline_first/seed_612/run_1_experiment_results.json:66).
- The retained panel analysis reads caps from planned configs in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:99](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_complete_seed_panels.py:99).
  - Its reported `2159` standard-cap terminal results are therefore not the executed-cap count.
  - The new read-only count agrees with the earlier recreation findings in [/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/report.md:75](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_05/report.md:75).
- The inventory found no terminal result whose saved `vote_integrity` reported `contaminated` or `synthetic_vote_used`.
  - This limited check does not inspect every transcript or prove the absence of other synthetic-action fields.
  - Current engine substitution is an independently demonstrated source-code risk for new runs.
- The native generator's universal `16384` cap does not reproduce this historical mixture.
- The archived replication manifest permits documented cap-related recovery from `10500` to `16384` in [/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed984_20260725_025700/manifest.json:2](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_seed984_20260725_025700/manifest.json:2).
  - A recovery helper archives a failed output directory and records a recovery config in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py:40](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py:40).
  - It accepts a default reason without verifying a saved structured failure in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py:151](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/submit_ttc_cap_recovery.py:151).
  - Repeated recovery has no maximum attempt count in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/resubmit_ttc_cap_recovery.py:42](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/resubmit_ttc_cap_recovery.py:42).
  - Both helpers are Slurm-specific and move the output directory before scheduler success.

**What should a new user configure once?**

- Install the API and scientific dependencies from [/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1](/scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt:1) in a Python environment.
  - The TTC path uses hosted models and does not require GPU allocation or local model weights.
  - The dependency file uses broad lower bounds and is not a tested lock for the TTC integrations.
- Export `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENROUTER_API_KEY` for the full suite.
  - GPT-5-only TTC needs OpenAI credentials when provider substitution is disabled.
  - Claude TTC needs OpenAI and Anthropic credentials.
  - Gemini TTC needs OpenAI and OpenRouter credentials.
  - Gemini TTC does not use a direct Google key.
  - Grouped keys also require `LLM_KEY_GROUP_ORDER` in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:88](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/provider_key_rotation.py:88).

| Selected adversary family | Baseline API | Adversary API | Required credential variables |
|---|---|---|---|
| GPT-5 | OpenAI | OpenAI | `OPENAI_API_KEY` |
| Claude Sonnet 4.6 | OpenAI | Anthropic | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Gemini 3 Flash | OpenAI | OpenRouter | `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| Full TTC suite | OpenAI | All three routes | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` |

- Select a transport explicitly for each API route.
  - A machine with outbound access can use direct APIs.
  - Restricted jobs need configured network routes without changing the model provider.
  - OpenAI `auto` selects the shared file queue inside Slurm in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2742).
  - OpenRouter `auto` tries direct access before the file queue in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:619](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/openrouter_client.py:619).
  - Anthropic uses its SDK directly in [/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2510](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2510).
  - The existing Della wrappers depend on `anaconda3/2024.2`, `proxy/default`, a repository virtual environment, a user credential path, and `/home/jz4391/openrouter_proxy` in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:341](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_ttc_native_scaling_jobs.py:341).
  - Those settings belong in an optional Della execution profile.
- Do not imply that copying a credential template automatically loads it.
  - The inspected local runner reads process environment variables.
  - Only the generated Slurm shell wrappers source `BARGAIN_API_KEYS_ENV`.
  - The supplied workflow assumes the queue monitor is externally managed, so this review did not inspect or start it.

**What is the smallest useful launch architecture?**

- Use one common public dispatcher with a TTC suite adapter.
  - Keep the current game engine and `StrongModelsExperiment.run_batch_experiments()` as the execution API.
  - Keep one game per resolved run with `num_runs=1` and an explicit run number.
  - Keep local execution and Slurm submission as separate executors over the same resolved run list.
  - A local execution path must not generate or require Slurm files.
- Make the suite adapter produce a fully resolved plan before it starts calls.
  - Parameterize the existing `MODEL_CONDITIONS`, `GAME_CELLS`, `ORDERS`, and `build_configs()` instead of copying their experiment logic.
  - Use `(seed, config_id)` or a stable `run_uid` for identity across all ten seeds.
  - Store explicit model IDs, catalog snapshots, provider routes, native effort payloads, phase caps, phase enablement, game settings, seeds, and seat order.
  - Store code and dependency versions plus source-plan hashes.
  - Store a declared retry policy and reject unrecognized or omitted result-affecting fields.
- Provide two distinct documented presets.
  - A small preset can contain GPT-5, one item-allocation cell at competition `0.5`, all four efforts, both orders, and seed `42`, for eight runs.
  - Its token cap must be explicit, such as `16384`, and it must be labeled as a new pilot design.
  - A paper-design preset must contain all 2,160 planned cells, archived IDs, and the declared historical cap-recovery policy.
  - A terminal-attempt replay must instead pin the saved terminal caps from the table above and the selected attempt IDs.
  - The public interface must distinguish a repeat of the planned protocol from a replay of selected terminal settings.
  - Neither mode can promise identical outputs from changing hosted services.
- Ship the small plan data needed by the paper preset outside ignored result directories.
  - A compact tracked manifest can contain the 216 base cells, ten seeds, hashes, and recovery provenance.
  - Large result files can remain a separately distributed data package.
  - Do not rebuild the paper plan solely from current catalog aliases.
- Save immutable attempt records under the requested absolute output root.
  - Record `planned`, `running`, `failed`, and `complete` states for each run.
  - Save the original failure before any allowed retry.
  - Store each attempt's result and transcript paths together with its `experiment_id`.
  - Resume only an explicitly selected plan with matching hashes.
  - An incomplete plan must return an incomplete status and must not be described as a successful full suite.

The following is proposed CLI syntax, not an existing command.

```bash
bargain run ttc --preset gpt5-item-pilot --executor local --output-root /tmp/bargain-ttc-pilot
bargain run ttc --preset paper-design --executor local --output-root /tmp/bargain-ttc-paper
bargain run ttc --preset paper-design --executor slurm --execution-profile della --output-root /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_public_rerun
```

- The dispatcher should print the resolved model routes, run count, cap policy, and output root before starting.
- An explicit plan-only mode should validate without writing configs, starting monitors, making model calls, or submitting jobs.
- Keep historical labels such as `target_position` for loader compatibility while public prose uses “adversary model.”

**Which implementation patches are necessary?**

- **Patch the TTC planner and adapter.**
  - Make `build_configs()` accept selected conditions, game cells, orders, seeds, and declared cap policy.
  - Construct each `output_dir` beneath the actual resolved output root.
  - Reject a nonempty output root unless an explicit resume operation matches the plan hash.
  - Add strict config validation before `build_command()`.
  - Remove implicit cap migration from the new public path.
  - Preserve source and recovery fields in resolved run metadata.
  - Reuse the existing run-scoped catalog override API in [/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:59](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/agents/agent_factory.py:59) so the resolved plan controls actual requests.
- **Patch shared execution policy.**
  - Require an exact roster and fail if any requested agent cannot be created.
  - Disable provider and model substitution in both agent construction and call retries for the public TTC plan.
  - Preserve only explicitly declared and bounded same-model, same-provider retries.
  - Raise a saved run failure after invalid-output repair is exhausted.
  - Remove synthetic proposal and vote generation from the new production path.
  - Save requested and executed provider, model, effort, transport, and cap for every call.
  - Record raw provider usage and preserve missing values as missing.
- **Patch attempt and recovery handling.**
  - Reuse the archive and recovery-record concepts from the two TTC recovery helpers.
  - Require a structured failure class that matches the plan's allowed recovery.
  - Make maximum attempts explicit in the plan.
  - Validate targets and write the recovery record before submitting or moving live outputs.
  - Make local and Slurm recovery call the same attempt API.
  - Treat cap increases as a protocol change with linked provenance.
- **Patch analysis loading.**
  - Select terminal results through the attempt manifest instead of a fixed filename.
  - Reuse `resolve_final_interactions()` from [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/ttc_accounting.py:16](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/ttc_accounting.py:16) to match transcripts by `experiment_id`.
  - Read effective caps from the saved executed config and compare them with the plan.
  - Reject missing utilities, unapproved recoveries, synthetic actions, provider substitutions, duplicate cells, or missing required cells.
  - Use the plan's selected grid for small runs instead of hard-coded `216` and ten-seed assumptions.
  - Preserve order averaging, equal game-cell averaging, and seed-level uncertainty for the paper design.
  - Require the declared statistical method instead of the silent normal-critical-value fallback in [/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:42](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_ttc_seed_replication.py:42).
- **Patch onboarding and scheduler profiles.**
  - Validate only the credentials required by the selected plan.
  - Document environment loading and the different meanings of transport routing and provider substitution.
  - Move Della modules, partition, time limit, CPU count, credential path, queue path, and interpreter path into an explicit execution profile.
  - Record the profile used in the run manifest.
  - Use safe shell quoting when rendering paths in scheduler scripts.

**Which checks are required before release?**

- Verify exact counts and unique identities for the 2,160-run paper plan and the eight-run pilot.
- Compare every paper base cell, seed, order, native effort payload, and cap policy against a tracked fixture extracted from the archive.
- Verify the four effort mappings for each family and explicit low effort for the baseline.
- Verify that conflicting metadata, catalog overrides, booleans, token caps, model IDs, or providers fail before any API request.
- Verify that a root outside the repository receives all configs, logs, attempts, results, and analysis outputs.
- Verify that existing outputs cannot be overwritten or silently reused.
- Verify that resume selects the declared terminal attempt and refuses ambiguous transcript matches.
- Verify that invalid proposals, invalid votes, exhausted credentials, and hard provider errors fail without synthetic actions or provider changes.
- Verify that all allowed retries are bounded and preserve their original failure and executed request metadata.
- Verify planned-versus-executed cap reporting using the observed `2088/71/1` historical distribution.
- Verify that the analysis rejects missing utilities and contaminated records while retaining legitimate no-agreement utilities of zero.
- Test plan-only and CLI import paths with all network access disabled.
- Extend existing focused tests in [/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:27](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:27) and [/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:17](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:17).
  - Replace the public-path expectation that archived caps silently migrate.
  - Use test doubles only for external boundaries.
- Run separately authorized real smoke tests for the three provider routes and supported effort values before reporting that the integrations work.
  - This review did not run those tests.

**Which decisions or facts remain unresolved?**

- The release must decide whether “paper rerun” means the original planned protocol or the saved terminal settings.
- The release needs a versioned public plan and, if historical replay is offered, a data-package location.
- Current service access and support for every historical model and effort value remain unverified.
- Exact historical provider routing cannot be reconstructed from planned metadata alone.
- The exact launch commands for the four seed-42 cap increases remain outside this review's verified evidence.
- Strict failure handling will differ from current synthetic-action behavior and must receive a new protocol version.
- A terminal-attempt replay that uses the current engine is a new execution of archived settings unless the historical code, prompts, dependencies, and provider behavior are also preserved.
- A fixed 2,160-run design can be rerun, but exact online replay of the historical conversations and payoffs cannot be promised.
  - The saved seeds control game generation and do not seed the inspected native provider payloads.
  - Hosted generation is nondeterministic, and aliases or provider implementations can change.
  - Reproducing the historical figure from the released saved results is a separate offline task.
