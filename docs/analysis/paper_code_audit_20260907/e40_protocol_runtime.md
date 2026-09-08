# E40 shared protocol and prompt audit

## Scope and result

- Figure 1 describes discussion, private thinking, proposal submission, voting and reflection.
- Appendix F contains current prompt examples with sample values.
- This audit traces shared execution and prompt support; cohort agents establish selected launch configurations and raw-result inventories.
- No source, paper, experiment, environment or Git changes were made.

## Verified execution chain

- /scratch/gpfs/DANQIC/jz4391/bargain/run_strong_models_experiment.py:37 imports the experiment orchestrator and configuration.
- /scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:570-715 runs discussion, private thinking and proposals, with a separate binding-team branch.
- /scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/phases/phase_handlers.py:1470-1473 runs public discussion in agent order.
- Proposal contexts copy the pre-phase public history at lines 2182-2195.
- Phase tasks are collected before results are published at lines 456-477 and 2602.
- Parallel execution is optional, but same-round proposals are not inserted into another agent's proposal context.
- Voting needs the rounded-up two-thirds threshold, picks the most-supported passing proposal and randomly resolves exact top ties at lines 4084-4115.
- /scratch/gpfs/DANQIC/jz4391/bargain/game_environments/co_funding.py:1482 returns propose_and_vote, so older pledge/commit branches cannot be assumed to describe the current Game 3 runtime.
- The team branch substitutes private planning, one team proposal and strict binding ballots, with explicit synthetic-vote rejection at phase_handlers.py:4035.
- Ordinary shared code still contains synthetic action fallback paths, including voting at lines 3700-3719.
- Such paths must remain available for provenance analysis even when a selected run did not execute them.

## Hidden dependencies and prompt provenance

- /scratch/gpfs/DANQIC/jz4391/bargain/negotiation/context_compaction.py:94-135 reads /scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md.
- The same Markdown file supplies model metadata to active_model_roster.py.
- Deleting this apparent documentation can alter context budgeting.
- Prompt wording and asset names live in Python game classes, not external Markdown templates.
- /scratch/gpfs/DANQIC/jz4391/bargain/scripts/generate_all_prompts_reference.py renders those classes at fixed states and calls pandoc.
- Its output is still /scratch/gpfs/DANQIC/jz4391/bargain/overleaf/icml_aiwild_template/all_prompts_generated.tex.
- Current ICLR prompts are embedded in /scratch/gpfs/DANQIC/jz4391/bargain/overleaf/iclr_aiwild_template/appendix.tex:750 onward.
- Saved examples are not proof of prompts used in historical runs.
- The inspected raw sample contains 14 interactions with stored setup prompts and actual model names.
- /scratch/gpfs/DANQIC/jz4391/bargain/strong_models_experiment/experiment.py:1072-1113 and 1440-1461 can externalize prompts into gzip files referenced by prompt_storage_path.
- Preserve those referenced files with each transcript.

## External and conditional support

- OpenRouter and OpenAI file queues use /home/jz4391/openrouter_proxy.
- XAI's conditional client uses a separate relative queue and monitor.
- Queue contents and credentials were not read.
- NumPy, SciPy, PyYAML, aiohttp and provider SDKs are external runtime dependencies, depending on selected route.
- Pandoc is an external reference-rendering dependency.
- Model clients and legacy protocol branches are conditionally callable; this audit does not declare them removable.

## History evidence

- Bundled codex-search completed over 2,215 repo-linked session records.
- Original instructions at /home/jz4391/.codex/history.jsonl:1171 require prompt-local compaction, an 85% trigger and unchanged raw trajectories.
- Lines 1183 request full-input accounting.
- Full original April session file was not located here; those observations are history-only.
- Search result /home/jz4391/.codex/sessions/2026/07/26/rollout-2026-07-26T20-57-46-019fa113-f20d-73f1-a39d-7c08ad34b4f4.jsonl:4361,4513 was manually checked.
- It documents a separate 50-run compaction pilot, so non-paper operational workflows must not be deleted based only on paper coverage.

## Needed files

The paired JSON enumerates 58 concrete paths with roles and evidence, including the 36-file conditional import closure, dynamic support and regression tests.
This is not a claim that every conditional module ran in every experiment.

## Deletion recommendation

No file is approved for deletion from this scope.
Exact historical code versions, complete externalized-prompt inventories and external monitor provenance remain unresolved.

