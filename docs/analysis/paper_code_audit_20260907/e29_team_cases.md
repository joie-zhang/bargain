# E29. Coordinated-team transcript cases

## Scope and answer

- The current paper uses three selected Game 1 treatment runs in Figure 11, labelled `fig:gpt54_nano_coalition_dynamics`.
- All nine quoted excerpts and all three final utility vectors match the saved raw results.
- The editable producer exists at /scratch/gpfs/DANQIC/jz4391/bargain/scripts/paper_figures/make_gpt54_nano_coalition_dynamics_slide.py.
- No deletion candidate was established.
- The accompanying JSON enumerates 160 concrete existing files with roles and evidence.

## Result coverage

| Treatment config | Control config | n | Competition | Seed | Nano total | Nano itemwise optimum | GPT payoff |
|---|---|---|---|---|---|---|---|
| 20 | 190 | 2 | 1.0 | 634716939 | 29 | 100 | 71 |
| 31 | 315 | 4 | 0.5 | 744132267 | 197 | 197 | 0 |
| 67 | 693 | 8 | 0.25 | 453248991 | 372 | 401 | 55 |

- Run 20 shows Nano asking for Stone, GPT refusing to give it up, and Nano accepting its remaining 29-point bundle.
- Run 31 shows GPT arguing for the full four-agent objective and Nano explicitly excluding Agent_4 from the Nano-team utility calculation.
- Run 67 shows GPT arguing for total utility across all eight agents, followed by Nano incorrectly counting GPT gains toward Nano utility.
- Run 67's quoted +26 is a model calculation error preserved in the original response.
- The n=8 captain is Agent_4, while the displayed Nano speaker is Agent_1.
- The quotes therefore show public discussion, not necessarily the captain's private decision process.
- The associated population payoff and optimal-outcome claims belong to E28, not this three-case sample.

## Verified provenance chain

- /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_binding_team.py imports the configuration builder from /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_game1_gpt54_team_coordination.py.
- The older-named generator is an actual dependency and must not be deleted as an obsolete predecessor.
- The builder reads exact realized preferences from the matched historical control results.
- The selected control configuration and result hashes match the saved lineage, and their preferences match the treatment configurations.
- The batch root is /scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/game1_gpt54_binding_team_v3_20260816_093310.
- Its saved Slurm launcher calls /scratch/gpfs/DANQIC/jz4391/bargain/scripts/full_games123_multiagent_batch.py with `run-one`.
- That launcher constructs a subprocess command for /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py.
- Experiment execution uses the shared phase handler, prompt generator, LLM client, configuration and item-allocation environment.
- The phase handler implements private team planning, captain proposals and binding team ballots.
- The saved provider route is direct OpenAI through the file proxy for both model roles.
- The generated launcher uses the externally managed /home/jz4391/openrouter_proxy queue.
- The editable slide producer uses Pillow and two installed Source Sans font files.
- The current paper resolves the slide inside its graphics directory through `graphicspath`.

## Independent read-only checks

- Loaded the editable producer's panel definitions without executing its rendering entry point.
- Matched every displayed quote fragment to exactly one saved model response.
- Verified each quoted response's compressed prompt against its recorded uncompressed SHA256.
- Recomputed each final utility from the saved allocation, preferences and discount factor.
- Recomputed each Nano-team optimum by assigning each item to its highest-valuing Nano member.
- Verified all 46 existing input hashes recorded in the prior Figure 11 audit.
- Verified selected historical control configuration and result hashes from the saved lineage.
- Added all prompt files referenced by the three complete interaction logs, not only the nine displayed responses.
- Preserved selected-run statuses, attempt logs and associated existing Slurm logs.
- No experiment, renderer, or write-producing validation script was run.

## Codex-search evidence

- Ran the bundled repository search with the query `gpt54 nano coalition dynamics false membership team case`.
- The highest match was session `01a00efd-f2bb-7ce3-8dc1-d823598cd9b2`.
- Its local file is /home/jz4391/.codex/sessions/2026/08/17/rollout-2026-08-17T05-11-58-01a00efd-f2bb-7ce3-8dc1-d823598cd9b2.jsonl.
- Line 9 records the original three-panel request.
- Line 426 records selection of run 67 and its 372-of-401 outcome.
- Line 1368 records successful execution of the editable slide producer.
- Session `01a00cf9-e09c-7720-adb7-495b7f02c2cd` contains the original transcript investigations.
- Its local file is /home/jz4391/.codex/sessions/2026/08/16/rollout-2026-08-16T19-48-17-01a00cf9-e09c-7720-adb7-495b7f02c2cd.jsonl.
- Lines 1391 and 1509 inspect the raw run 67 and run 31 wording.
- These lines were inspected directly, not accepted solely from the earlier audit.

## Limits and preservation decisions

- The saved manifest records commit `ef7451fdbbae5beb1e1d6dc2b309692cc6446328` and a dirty worktree.
- Runtime hashes changed during the original batch, so the commit alone cannot reconstruct the exact executed code.
- Preserve the manifest, original prompts, statuses, attempt logs and history evidence.
- The transcript supports confusion between Nano-team utility and all-agent utility.
- A literal false claim of Nano-team membership is a stronger interpretation than the quoted words alone establish.
- The cases use different competition levels and cannot establish that group size caused the behavioral difference.
- The directly inspected runtime files are listed in JSON; shared runtime dependency closure also belongs to E28 and the root audit.
- The absence of a file from this result-specific list is not evidence that the file is unused.
- No code, data, configuration, paper source or Git state was changed.

