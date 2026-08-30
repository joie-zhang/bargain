# Project instructions

## Project purpose

- This repository studies multi-agent language-model behavior in negotiation environments.
- The work includes experiment code, analysis, paper figures, and cluster jobs.
- Do not assume a publication venue, model roster, or deadline from an old document.
- Verify current claims against the relevant code, configuration, data, and dated analysis files.

## Working approach

- Inspect existing code and repository conventions before you change them.
- Make the smallest change that completes the request.
- Run focused checks first, then use broader checks when the change has wider risk.
- Do not create planning documents, implementation logs, or commits unless the task requires them.
- Do not commit, push, or create a branch unless the user requests it.

## Python and tests

- Use the project environment at `.venv` when it is available.
- Run Python with `.venv/bin/python` and tests with `.venv/bin/pytest` when practical.
- Follow `pytest.ini`, which places tests under `tests/` and recognizes `test_*.py` and `*_test.py`.

## Test doubles and synthetic data

- Test doubles are mocks, fakes, stubs, or simulated services used only by tests.
- Use a test double only to isolate an external service or nondeterministic boundary in a unit test.
- Do not use mocks, fakes, placeholders, or simulated services in production experiment paths.
- Do not use mock, synthetic, or generated data as research evidence or present it as an observed result.
- Do not rely on a test double as the sole validation of an external integration.
- Run a real smoke test before you report that an external integration works.
- If a real smoke test is not possible, state that integration validation is incomplete.

## Research integrity

- Do not invent data, results, citations, model identifiers, configuration values, or source support.
- Separate observed results from interpretation and inference.
- Record the seeds, configurations, model identifiers, code version, and data source needed to reproduce an experiment.
- Investigate surprising results for bugs, data errors, and alternative explanations without assuming a fixed probability that the result is wrong.
- Select statistical tests and error thresholds from the study design instead of applying one universal p-value cutoff.
- Report effect sizes, uncertainty, multiple-testing choices, exclusions, and failed runs when they affect the conclusion.

## Adversary terminology

- In paper prose, captions, tables, and plot labels, use `adversary model`; do not use `focal model`, `target model`, or similar terms for that model.
- In prompts shown to agents, use neutral seat or role names; never tell an agent that it or another participant is an adversary.

## Failure, fallback, recovery, and default policy

- Never silently substitute missing data, failed results, alternate models, alternate providers, cached results, synthetic results, or default values.
- Fail early when a required input, result-affecting input, or internal invariant is absent or invalid.
- Do not hide an error with a broad exception handler, an undocumented default, or an unreported alternate execution path.
- Permit a recovery path only when the user request, task specification, or checked-in configuration explicitly allows it.
- Keep every retry and recovery path bounded.
- Record the original failure and each recovery action in logs and saved results.
- Preserve the model, provider, data, configuration, and result provenance through recovery.
- Do not use recovery to mark a failed or incomplete experiment as successful.
- Mark partial results as incomplete and identify the missing work.
- Permit a result-affecting default only when the configuration schema declares it.
- Materialize every result-affecting default in the saved resolved configuration.
- Permit an operational default only when it is documented and cannot affect the research result.
- Report the final failure with enough context to diagnose it.

## File placement

- Put analysis notes in `docs/analysis/`, procedures in `docs/guides/`, and reference material in `docs/reference/` when those locations fit the task.
- Keep repository control files such as `README.md`, `AGENTS.md`, and `CLAUDE.md` at the repository root.
- Do not add large boilerplate headers to every script.

## Cluster and model access

- Assume Slurm compute nodes do not have direct internet access.
- Route OpenRouter requests from restricted jobs through the file queue at `/home/jz4391/openrouter_proxy`.
- Before downloading a Hugging Face model, check `/scratch/gpfs/DANQIC/models` for an existing local copy.
- Check current job scripts and cluster instructions instead of copying hard-coded modules, model names, or resource estimates from old guidance.
