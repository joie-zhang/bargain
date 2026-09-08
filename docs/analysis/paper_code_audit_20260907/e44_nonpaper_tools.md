# Non-paper tools audit

## Finding

- Seven files under `/scratch/gpfs/DANQIC/jz4391/bargain/utils/` are supported retirement candidates, subject to confirming no external or manual CLI use.
- These are not the experiment's runtime utilities, which live under its own package.
- Do not delete the UI, visualization, notebook, or Claude configuration directories wholesale.
- No code, data, paper, settings, or Git state was changed by this audit.

## Specific candidates

| Absolute path | What it does | Evidence beyond missing references |
|---|---|---|
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/context_analyzer.py` | Generic file/context selector; Path.current() default is invalid for pathlib.Path at line 40. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/session_manager.py` | Generic Claude checkpoint CLI storing .claude/sessions, not experiment sessions. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/spec_validator.py` | Generic markdown requirement validator and pytest skeleton generator, not game config validation. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/task_manager.py` | Generic tasks markdown/board CLI, not experiment job scheduler. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/todo_manager.py` | Generic markdown TodoWrite converter, not experiment execution. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/verification_framework.py` | Generic web-app verifier demo targeting localhost health, PostgreSQL mydb, npm, and nonexistent experiments/train_model.py. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |
| `/scratch/gpfs/DANQIC/jz4391/bargain/utils/verification_metrics.py` | Generic verification/token-cost statistics and simulated example sessions, not negotiation outcome metrics. | Only commit 0784e8d, dated 2025-07-09; standalone helper, not a paper producer. |

- All seven were checked against every tracked text file with Python, Markdown, JSON, TOML, YAML, shell or notebook extension, excluding self references.
- No basename reference was found outside their own files.
- Checks also covered local `.claude` commands/settings/hooks and the user's Claude and Codex settings.
- The source imports are standard libraries or third-party plotting/data libraries; no runtime package imports these helpers in the checked tracked files.
- There are no current tracked changes to the audited source paths.
- The default helper storage paths `tasks`, `todo.md`, `.claude/sessions`, `verification_metrics`, and `verification_log.json` were absent at the repository root when checked.
- Their independent CLIs remain a possible manual consumer, so this is not proof that deletion cannot affect any user workflow.

## Files to retain

- The JSON report lists 29 concrete support/provenance/test files, with dependency reasons.
- The general and game-specific viewers are documented in `/scratch/gpfs/DANQIC/jz4391/bargain/docs/guides/negotiation_viewer.md`.
- Game 2 and Game 3 viewers import the general viewer, which imports UI components.
- The multi-game viewer imports both the general and Game 1 viewers.
- The binding-team launcher invokes the random-monoculture viewer, which has a dedicated test.
- The behavior review UI, storage module, sampler and tests support the documented 480-item human annotation pilot.
- The reasoning notebook preserves a specific model-order and reasoning-budget bug diagnosis, including saved outputs.
- Nine hook/helper scripts are directly configured or called by the configured hooks.
- Configured hook use is independent of paper use.

## History and limits

- Codex-search was run with `streamlit_prompt_reviewer context_analyzer verification_framework UI` and returned local July 2026 cleanup sessions.
- Manually checked `/home/jz4391/.codex/sessions/2026/07/19/rollout-2026-07-19T01-24-35-019f78d5-5654-7e72-a948-45e06660770c.jsonl`.
- Line 1077 contains user approval for UI cleanup, and line 2265 records deliberate repairs to the prompt reviewer and historical notebook.
- Git commit `84eb94b` removes older viewers and retains the present reviewer structure.
- Git commit `a0d9a0d` records a previous visualization cleanup, so remaining standalone visualizers cannot be called forgotten duplicates without comparing their use cases.
- Larger source files were inspected through imports, entry points, method structure and relevant implementations; this report does not claim every branch was executed or every source line reviewed.
- Only locally available history was searched.

## Unresolved optional workflows

- `/scratch/gpfs/DANQIC/jz4391/bargain/ui/graphics_triage_viewer.py` targets old ICML assets and writes keep/delete decisions and a proposed staging script.
- Its old-paper scope is a reason not to run it on the current paper, not permission to delete it during this runtime audit.
- `/scratch/gpfs/DANQIC/jz4391/bargain/streamlit_prompt_reviewer.py` records prompt review decisions and can run directly without importers.
- `/scratch/gpfs/DANQIC/jz4391/bargain/visualization/visualize_nagent.py`, `/scratch/gpfs/DANQIC/jz4391/bargain/visualization/visualize_diplomacy.py`, and `/scratch/gpfs/DANQIC/jz4391/bargain/visualization/visualize_cofunding.py` remain independent analysis tools.
- `/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/test_openrouter_client.ipynb` and `/scratch/gpfs/DANQIC/jz4391/bargain/negotiation/test_openrouter_scratch.ipynb` have saved outputs that may document provider behavior.
- Auto-commit is commented out in the stop hook, but the separate CLI and its tests/configuration require an explicit retirement decision.
- Other Claude commands and log-analysis helpers remain direct user interfaces, so no source removal is approved for them.

No candidate in this report has unconditional deletion approval.
