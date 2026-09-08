# E43 test and environment audit

## Conclusion

- Keep the 45 current test modules and their conftest file.
- Keep their verified local imports and the configured editor hooks.
- No source-code deletion is supported by this audit.
- There are 25 generated test-bytecode candidates totaling 766,944 bytes.
- No tests, hooks, experiments, installs, deletion commands, or Git mutations were run.

## Current tests and dependencies

- /scratch/gpfs/DANQIC/jz4391/bargain/pytest.ini:1 contains the valid [pytest] section and collects tests under tests/.
- /scratch/gpfs/DANQIC/jz4391/bargain/tests/conftest.py:7 places the project root on sys.path and selects the asyncio backend.
- The companion JSON lists 98 needed test, imported-code, and support paths.
- /scratch/gpfs/DANQIC/jz4391/bargain/docs/analysis/paper_code_audit_20260907/e43_code_filename_inventory.json records each direct local import with its source line.
- Imports include game environments, preference generators, model/provider transport, phase handlers, qualitative metrics and judge, fairness analysis, team analysis, TTC accounting/recovery, batch generation, and viewer/review support.
- These are direct AST-verified import edges, not a claim that all imported functions execute in every test.
- /scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_seed_replication.py:21 directly depends on the released seed-42 TTC config root.
- Its clone tests load all 216 configs without skipping missing data.
- /scratch/gpfs/DANQIC/jz4391/bargain/tests/test_ttc_accounting.py:107 also selects result and final-interaction files from those config output directories.
- Removing ignored experiment data can therefore break current tests or silently skip a real-data accounting check.
- Most other observed fixtures use tmp_path or local test doubles.
- Tests were read for imports, path access, assertions and fixtures, but not executed.

## Hidden operational files

- /scratch/gpfs/DANQIC/jz4391/bargain/.claude/settings.json explicitly names eight hook/script entry points.
- The notification hook additionally calls /scratch/gpfs/DANQIC/jz4391/bargain/.claude/scripts/notify.py.
- These configured dependencies are retained even though paper experiments do not import them.
- The auto-commit call in /scratch/gpfs/DANQIC/jz4391/bargain/.claude/hooks/stop.py:108 is commented out.
- That proves this hook does not call auto-commit, but does not prove the separate script is unused everywhere.
- The formatter tries ruff, black, then autopep8.
- None has package metadata in the project environment, but external PATH availability was not checked.
- The registered worktree /scratch/gpfs/DANQIC/jz4391/bargain/.claude/worktrees/contract-negotiation is on branch worktree-contract-negotiation at 55de4dbb6959e0829797a7aaf2d04e7d2819ffcb.
- This worktree is user work with its own Git state and is not a disposable copy.

## Environment

- /scratch/gpfs/DANQIC/jz4391/bargain/.venv/pyvenv.cfg identifies CPython 3.14.0 and uv 0.9.10.
- Installed metadata shows pytest 9.0.2, numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, anthropic 0.74.0 and openai 2.8.1.
- Other checked packages include aiohttp 3.13.2, requests 2.32.5, anyio 4.11.0, PyYAML 6.0.3, matplotlib 3.10.7, streamlit 1.52.2 and plotly 6.5.0.
- /scratch/gpfs/DANQIC/jz4391/bargain/requirements.txt specifies lower bounds rather than this exact environment.
- /scratch/gpfs/DANQIC/jz4391/bargain/ui/requirements.txt supports an additional maintained viewer workflow.
- Removing the environment cannot be recommended as cleanup because exact reconstruction was not verified.
- Environment metadata was read without importing experiment packages.

## Filename coverage

- The inventory contains 1,204 paths ending in .py, .sh, .sbatch, .toml, .yml or .yaml.
- It includes 268 under analysis, 195 under experiments, 200 under .claude including the registered worktree, 182 under scripts and 115 under reproduction_audit.
- The walk includes ignored directories but skips virtual environments, Git metadata, package caches, node_modules and symlinked directories.
- It does not inventory JSON data/configs, notebooks, extensionless scripts or external symlink targets.
- Each filename is recorded individually in the companion inventory.
- Every path without a needed entry in this unit remains unassessed by E43.
- Other result agents can resolve those paths, but this filename inventory cannot prove that a path is unused.

## Cache candidates

- Each of the 25 proposed test bytecode paths has an existing source file.
- Its Python magic number matches the current interpreter.
- Its timestamp-based header matches the source modification time and size.
- The exact paths and source counterparts are in the inventory and report JSON.
- This is sufficient for ordinary generated-cache cleanup, but it is not a mathematical proof that bytecode and source semantics are identical.
- Source files must remain intact.
- No whole cache directory is proposed for deletion.
- The pytest cache contains plugin metadata and no observed v/ data files, so deleting it would save little.
- Other bytecode that did not satisfy the source-header check is not approved.

## History

- The codex-search skill found locally available session 019f78d5-5654-7e72-a948-45e06660770c.
- Original completion text was verified at /home/jz4391/.codex/sessions/2026/07/19/rollout-2026-07-19T01-24-35-019f78d5-5654-7e72-a948-45e06660770c.jsonl:2833.
- That July audit deleted 42 approved stale files, retained 32 modules plus conftest, and reported 479 passed.
- The current tree has 45 modules plus conftest, so the July manifest is historical evidence, not a current deletion list.
- No claim is made that the current suite passes without running it.

## Limits

- Full transitive dependency resolution and semantic review of all 1,204 inventoried files are outside this support unit.
- Unused-code decisions require the other result audits and a separate check for maintained non-paper workflows.
- A missing paper reference is not sufficient evidence for deletion.
