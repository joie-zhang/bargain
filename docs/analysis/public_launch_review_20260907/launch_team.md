# Coordinated team launch review

**Question**
How can a new user launch a Nano team against GPT-5.4 High, and how does this differ from the matched same-model coalition replication?

**Short answer**
The binding team engine exists, but a public launch needs a validated configuration entry point and a generator that can create one independent run without historical results.

- This review inspected current source code, argument parsers, tests, saved configurations, and selected manifest fields on September 7, 2026.
- No experiment, API call, job submission, test suite, generator, or analysis script was run.
- The configuration counts below come from read-only JSON inspection.
- Runtime and external integration validation remain incomplete.

**Which studies must remain separate?**

| Review name | Roster and protocol | Matched input | Grid |
| --- | --- | --- | --- |
| `100matchedgame1` | One GPT-5.4 High adversary model and `N-1` GPT-5-nano agents with a binding captain | Earlier homogeneous-adversary control results | Five group sizes × five competition levels × two adversary positions × two seed replicates = 100 pairs |
| Older advisory team | The same model roster, with shared private notes and individual proposals and votes | Earlier homogeneous-adversary control results | A separate 100-run treatment |
| `25gpt54` | GPT-5.4 High in every seat, with ordinary independent actions | Saved Gemini 3.1 Pro same-model preferences | 25 source settings, including 20 settings with `N>2` |

- The current paper describes the binding captain treatment in [the team appendix](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:607).
  - The separate [coalition replication appendix](/scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:629) describes the 25-setting same-model study.
- The current binding generator selects `binding-team-three-turn-v3-env-utility` in [the protocol constant](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:33).
  - The older [advisory generator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:174) selects `baseline-private-team-v1`.
  - The [advisory briefing](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:267) adds fairness and feasibility instructions that differ from the binding team's sole utility-sum objective.
- Read-only inspection found 100 saved binding configurations with `N ∈ {2,4,6,8,10}`, `2.5N` items, ten rounds, two public discussion turns, two seed-replicate labels, and 100 distinct resolved random seeds.
  - [A saved four-agent configuration](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310/configs/config_0021.json:27) shows the control link and resolved settings.
  - The item-count rule also appears in [the Game 1 grid builder](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:572).
- Read-only inspection found 26 saved same-model configurations for 25 source settings, with only `gpt-5.4-high` in every roster and no `team_coordination` field.
  - [The saved same-model manifest](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_high_matched_coalition_pilot_direct_openai_20260829/manifest.json:5) records 25 source settings and 26 configured runs.
  - The extra configuration is the explicit replacement for failed `config_0004`, as recorded by [the grid extension](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/extend_gpt54_game1_matched_grid.py:101).

**What happens in the binding team runtime?**

- The batch runner sends the full configuration through `EXPERIMENT_RUN_METADATA_JSON` in [the subprocess environment](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1717).
  - The main runner [decodes the metadata](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:388) and [adds its fields to the experiment configuration](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:591).
  - Team settings, fixed preferences, and model-provider overrides depend on this path.
- The experiment [creates the phase handler and agents](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:254), then attaches the team objective to each participating Nano agent.
  - [The model system prompt](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:981) uses that objective for subsequent calls.
  - [The fixed-preference path](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:390) replaces generated preferences with the literal saved table after checking seat IDs and dimensions.
  - [The private-context initialization](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:499) gives the shared table only to team members.
- Each negotiation round keeps public discussion and replaces the Nano agents' ordinary private thinking with three private planning turns in [the round loop](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:583).
  - [Planning turn one](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1948) gathers independent candidates concurrently.
  - Turn two collects sequential audits of the candidates.
  - Turn three collects teammate recommendations and ends with the captain's allocation.
  - An unrepaired round has `3 × (N-1)` planning responses when `N>2`.
- [Captain selection](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:76) depends on the treatment configuration ID, and [the phase handler](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:118) rotates the captain each round.
- [The proposal validator](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2041) checks item ownership and computes utilities from the environment.
  - The captain's claimed arithmetic does not determine the computed team utility.
  - One invalid captain allocation gets [one recorded repair attempt](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:2085).
  - The second invalid allocation raises an error.
- [Proposal submission](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:640) combines the captain's single team proposal with the non-team agent's proposal.
- [The captain ballot](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3875) selects at most one proposal or explicitly rejects all proposals.
  - [The engine copies the captain's validated choice](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3995) into each team member's institutional vote.
  - These institutional votes are part of the declared experiment protocol.
  - They must remain distinguishable from model responses and from substituted actions after a failure.

**Which current commands are established by source inspection?**

- The following binding generator syntax is established by [its parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:195).
  - It writes a complete 100-run batch and a Slurm script.
  - It does not launch a run or submit a job.
  - The output directory below must be empty or absent.

```bash
/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py \
  --control-root /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255 \
  --output-root /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/team_launch_example_new \
  --max-concurrent 20
```

- One generated configuration can use [the existing `run-one` entry point](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:125).
  - This example selects a four-agent cell and a network-connected host with direct OpenAI transport.
  - `OPENAI_API_KEY`, or the supported OpenAI key pool, must already be configured.
  - This command makes paid model calls if executed.

```bash
OPENAI_TRANSPORT=direct OPENROUTER_PROVIDER_FALLBACK=0 \
  /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/python \
  /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py \
  run-one \
  --results-root /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/team_launch_example_new \
  --config-id 21
```

- The generated Slurm script [expects `SLURM_ARRAY_TASK_ID`](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:185), but it does not declare an array.
  - `--max-concurrent` is stored in the manifest and printed, but it does not itself set a submission throttle.
  - Submission therefore needs an explicit array, such as `sbatch --array=1-100%20`, and the generated script's absolute path.
- The generic batch `submit-selection` is unsuitable for this team batch because [it invokes the full-batch validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:2498).
  - [That validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1851) requires 2,730 configurations and different token settings.
- Post-run team tools accept `--results-root` through [the protocol-audit parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/audit_export_game1_gpt54_binding_team.py:169) and [the paired-analysis parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_binding_team.py:232).
  - They write exports or analysis files when executed.
  - They need the portability and validation changes below before they can serve as a general public result checker.

**What blocks a fresh user?**

- **Historical control files are mandatory.**
  - [Control loading](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:59) selects only Game 1 homogeneous-adversary GPT-5.4 High configurations.
  - [Factorial validation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:81) rejects any set other than the exact 100 cells.
  - [Treatment construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:106) reads each control result from the source configuration's `output_dir`.
  - `--control-root` changes configuration discovery but does not remap saved result paths.
  - A source checkout and API key are therefore insufficient for the current generator.
- **Repository and cluster paths are hard-coded.**
  - [The binding generator root](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:16), [the parent generator root](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:17), and [the binding analysis root](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_binding_team.py:17) refer to this user's checkout.
  - [The Slurm template](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:144) fixes Princeton modules, partition, resources, a personal credential-file location, and `/home/jz4391/openrouter_proxy`.
  - Its credential-file loading is conditional, so a missing file produces no immediate template-level error.
- **The public runner has no direct team configuration input.**
  - Malformed environment metadata is warned about and ignored by [the decoder](/scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:388).
  - Valid JSON with a non-object top level also leaves the run metadata empty.
  - The batch command builder [always adds `--parallel-phases`](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1535), regardless of the saved value.
  - These behaviors can change a manually supplied run configuration without stopping before model calls.
- **Provider and token settings need a resolved record.**
  - [GPT-5-nano](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:265) currently uses direct OpenAI and has no explicit reasoning-effort setting.
  - [The GPT-5.4 High catalog alias](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/configs.py:1239) uses OpenRouter until the team generator applies its OpenAI override.
  - The `gpt-5-nano-high` alias is a different configuration and must not replace `gpt-5-nano`.
  - [Phase-cap resolution](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1025) can replace the nominal 16,384-token experiment cap with GPT-5.4's 65,536-token catalog cap.
  - Private planning requests use an 8,192-token bound, and strict ballots use a 4,096-token bound.
  - [OpenAI transport](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:2686) defaults to `auto`, while [native-provider recovery](/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/llm_agents.py:578) permits OpenRouter unless explicitly disabled.
  - The generated team Slurm script disables provider substitution, but a direct `run-one` call does not enforce the manifest's provider policy by itself.
- **Protocol recognition is permissive.**
  - [Protocol selection](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:101) recognizes the binding version through exact string equality.
  - An unknown enabled protocol version can enter the advisory path instead of failing.
  - Several generated settings, including `max_action_repairs`, `synthetic_actions_allowed`, and `singleton_policy`, are descriptive metadata rather than independently enforced switches.
  - [Captain-ballot parsing](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3837) treats a missing `selected_proposal_number` as `null`, so `{}` becomes a reject-all ballot.
- **The two-player negative control is only partly unchanged.**
  - A singleton receives no team objective or briefing and has no private team planning.
  - [Voting dispatch](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:709) still selects the strict binding-protocol voting implementation at `N=2`.
  - [That implementation](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:3794) then uses separate individual ballots with its own token and repair rules.
  - Describe this as no team coordination, without claiming complete runtime equivalence to historical controls.
- **Current prompts expose adversary terminology.**
  - [The team objective](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:131) and [private briefing](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:238) call the outside participant an adversary and name the model.
  - Neutral seat wording is required by the current project instructions.
  - Changing these prompts requires a new protocol version and a clear distinction from historical results.
- **Result checks are specific to the historical batch.**
  - [The base team loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_team_coordination.py:144) requires 100 configurations, absolute treatment and control paths, lineage hashes, and both runs' interaction files.
  - [It checks the adversary model route](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_team_coordination.py:215), but it does not validate every Nano seat's effective route.
  - [Utility extraction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/analyze_game1_gpt54_team_coordination.py:109) supplies zero for missing utilities, which can conceal an incomplete result.
  - [The generic success checker](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1269) does not check team protocol, literal preferences, model roster equality, or action-integrity records.
  - [The protocol audit](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/audit_export_game1_gpt54_binding_team.py:67) treats no consensus as an audit failure even though [the engine](/scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:859) defines a completed no-agreement run with zero payoff.

**What is the smallest useful public interface?**

- Add a thin team preset to the common public launcher and keep the existing experiment engine.
  - Reusing the engine preserves the current proposal, planning, and ballot implementation.
  - A new preset and schema still require clear protocol and provider version records.
- The following commands are proposals and do not exist yet.

```bash
bargain run team \
  --n-agents 4 --competition 0.5 --seed 42 --adversary-position last \
  --provider openai --transport direct \
  --output /absolute/path/team-run
```

- The standalone preset should declare one `gpt-5.4-high` seat and three `gpt-5-nano` seats for this example.
  - It should declare ten items, ten rounds, two public discussion turns, discount `0.9`, and the complete team protocol settings.
  - It should generate a new preference table using the requested seed and save the literal table before model calls.
  - It should label the run as a new independent experiment with no historical control match.
  - It should record the provider-default Nano reasoning behavior explicitly if that historical setting is retained.
  - A standalone result needs its own result summary because the historical paired loader requires 100 pairs.

```bash
bargain plan team \
  --design 100matchedgame1 \
  --controls /absolute/path/control-bundle \
  --output /absolute/path/team-plan
```

- The matched planner should require an explicit control bundle and make no API calls.
  - It should preserve all 100 literal preference tables, factor cells, resolved seeds, model orders, and captain assignments.
  - It should stop on missing, duplicate, ambiguous, or invalid controls.
  - It should write an immutable plan, resolved model settings, relative bundle paths, hashes, and a machine-readable validation report.
  - A separate run command should consume the saved plan, with explicit local or Slurm transport selection.
  - A newly generated control cohort is a separate prospective paired design and must receive a different design name.

**Which code changes make these commands possible?**

- **Extract a pure team configuration builder.**
  - Split [historical-input loading](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py:106) from [team protocol construction](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py:65).
  - A reusable signature could be `build_team_config(base_config, *, preferences, protocol, output_root, control_record=None)`.
  - Preserve the original source dictionary because the existing builder removes `_source_config_path` with `pop`.
  - Store captain assignment explicitly so filtering or reordering cells cannot change it.
- **Introduce a required, validated configuration path.**
  - Add `--config` to the main runner, or add `run_resolved_config(config, runtime)` around `StrongModelsExperiment.run_single_experiment()`.
  - Validate the complete roster, seat roles, numeric values, finite preferences, protocol version, phase settings, and provider choices before agent construction.
  - Keep only declared schema defaults and save every resolved value.
  - Reject malformed metadata, contradictory configuration sources, unknown protocol versions, and unsupported option combinations.
  - Require the captain ballot's decision key and distinguish absent data from an explicit `null` decision.
- **Separate runtime locations from scientific settings.**
  - Derive repository roots from module paths and use the active interpreter instead of a personal environment path.
  - Represent each historical source by a bundle-relative path and checksum.
  - Resolve bundle paths through an explicit root and fail if a source cannot be found uniquely.
  - Parameterize the Slurm interpreter, partition, resources, and shared queue directory.
  - Require an explicit provider and transport policy instead of inheriting `auto` or provider-substitution behavior.
  - On restricted compute nodes, use the declared file queue with the externally managed monitor.
- **Share execution and validation without full-grid assumptions.**
  - Reuse [attempt logging and status handling](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py:1656) after separating per-run validation from the 2,730-cell design validator.
  - Require protocol, model, provider, seed, preference, and configuration hashes to match before explicit resume can reuse a result.
  - Treat completed disagreement, provider failure, invalid actions, and missing results as different states.
  - Let standalone analysis consume one validated result and let matched analysis require the declared pair count.
  - Preserve compressed prompt files referenced by [the transcript loader](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/audit_export_game1_gpt54_binding_team.py:46) when moving a result bundle.

**How should `25gpt54` be exposed?**

- Keep it under a same-model replication command rather than the team preset.
- [The existing pilot parser](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_gpt54_game1_coalition_pilot.py:433) accepts `generate --results-root ... --provider-route direct-openai`.
  - [Generation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_gpt54_game1_coalition_pilot.py:125) creates only four post-hoc Gemini-positive cells.
  - It requires both historical Gemini results and a historical coalition case CSV.
  - [The remaining-grid script](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/extend_gpt54_game1_matched_grid.py:12) and [the two-player extension](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/add_gpt54_game1_n2_matched_cells.py:12) use fixed source IDs and a fixed destination.
  - They are continuation scripts rather than portable public generators.
- [The pilot validator](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_gpt54_game1_coalition_pilot.py:290) still requires exactly four configurations.
  - [The pilot result audit](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_gpt54_game1_coalition_pilot.py:357) calls that validator, so it cannot validate the extended 25-setting study as written.
- [The same-model runner](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/random_monoculture_control_batch.py:617) adapts string configuration IDs and calls the same general batch execution function.
  - It also accepts runtime environment overrides, which a public plan should replace with an explicit resolved configuration record.
- [The budgeted continuation](/scratch/gpfs/DANQIC/jz4391/bargain/scripts/run_gpt54_matched_grid_budgeted.py:111) checks a per-run reserve before execution and computes spend afterward.
  - Its fixed historical prices and reserve estimates are not a strict spend ceiling.
- A proposed command could be `bargain plan same-model --design 25gpt54 --controls /absolute/path/gemini-control-bundle --output /absolute/path/replication-plan`.
  - It should create exactly 25 unique source cells without copying an old failed-attempt replacement into a fresh design.
  - A historical-result import should preserve all 26 attempts and explicitly select the accepted attempt per source cell.
  - It should use no shared team preferences, private planning room, captain, or binding team votes.
  - New coalition-rate analysis also needs a declared classification procedure because [the older figure recreation](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/figure_recreation_20260905/figure_30/recreate.py:100) reads historical qualitative labels from a local conversation log.

**What tests and validation are needed?**

- Existing [team unit tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_team_coordination.py:24) cover the provider override, private briefing, note recipients, advisory singleton behavior, team objective, captain rotation, and one successful planning-and-ballot path.
  - [The successful binding test](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_team_coordination.py:193) uses test-only agents and verifies environment-computed utility and institutional votes.
  - [The analysis tests](/scratch/gpfs/DANQIC/jz4391/bargain/tests/test_game1_gpt54_analysis_gap_sign.py:8) check payoff-gap signs.
  - These tests do not establish live provider integration.
- Add focused tests for standalone creation with no control archive and for matched creation from a relocated bundle.
  - Check the exact 100-cell product, all resolved seeds, literal preference hashes, seat maps, and unchanged captain assignments after filtering.
  - Reject missing or duplicate controls, malformed metadata, unsupported protocol versions, missing captain choices, nonfinite values, and conflicting provider settings.
  - Check one successful repair and a hard failure after the permitted repair is exhausted.
  - Verify that no private table or team-room content reaches the non-team agent through any context or repair path.
  - Exercise the binding singleton through the full experiment dispatch and document its voting behavior.
  - Check that disagreement is complete, while missing utilities and contaminated actions fail validation.
  - Check that provider failure never changes model or provider and that resume rejects mismatched results.
- Add same-model tests for exactly 25 source cells, no team configuration, and explicit historical attempt selection.
- Before reporting a public launch as working, run one real small team experiment through each supported transport and inspect the saved requests, prompts, ballots, effective model settings, results, and status.
  - A short smoke test must be labeled as integration validation and must not enter the paper cohort.
  - A supported Slurm path also needs a real queue-based run after the local launch path passes.

**Which reproducibility questions remain?**

- The public release needs an explicit choice between reproducing the historical prompt protocol and running a new protocol with neutral seat wording.
- The control bundle must include both configurations and result files, with interaction records when route-sensitive historical analysis is required.
- Exact historical reproduction needs more than the generation commit because [the saved binding manifest](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310/manifest.json:66) records source changes during final attempts.
- The binding manifest also records [noncontemporaneous controls and execution overrides](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310/manifest.json:6), which must remain visible in portable rerun comparisons.
- Current provider access, model availability, SDK compatibility, and actual billed cost were not tested in this review.
