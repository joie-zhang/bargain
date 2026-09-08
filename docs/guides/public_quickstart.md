# Run a small experiment

Use Python 3.12 or later and install from the checkout with `python -m pip install -c constraints-public.txt -e .`.
The constraint file pins the reference hosted-runtime dependencies.
Use `bargain --help` or the equivalent `python -m strong_models_experiment --help`.
The launch path needs NumPy, SciPy, and aiohttp, but no GPU, SDK, or cluster account.
An installed wheel includes its model and context resources.

## Set up credentials

- Export `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` as required by the selected seats.
- Alternatively, copy the credential template into a private file, restrict it with `chmod 600`, and pass its absolute path with `--env-file`.
- The file accepts only those three names with literal `KEY=value` entries and optional quotes.
- An explicit credential file is the only credential source; missing entries are not filled from the shell.
- The launcher never sources shell code or automatically loads a working-directory environment file.
- Keys are not saved in plans, results, request headers, or logs.
- `bargain models` lists the frozen aliases and their routes; current provider availability remains subject to account access and model retirement.

```bash
bargain doctor two-player --adversary gpt-4o-mini-2024-07-18
```

This checks installed modules and key presence without contacting providers.
It does not verify authentication, network access, or billing.
Only `run`, `execute --execute`, and `resume` can make model requests.

## Choose an experiment

```bash
bargain run two-player --adversary gpt-4o-mini-2024-07-18
bargain run two-player-llama --adversary gpt-4o-mini-2024-07-18
bargain run homogeneous --model gpt-5-nano --agents 4
bargain run heterogeneous --agents 4
bargain run homogeneous-adversary --adversary gpt-4o-mini-2024-07-18 --agents 4
bargain run ttc --family gpt5
bargain run team --agents 4
```

- Two-player runs use a fixed GPT-5-nano or Llama 3.3 70B baseline.
- Homogeneous runs use the same model in every seat.
- Heterogeneous runs select distinct models from the frozen 24-model pool.
  - `--stratum 0` through `--stratum 4` selects an equal-width population-Elo-standard-deviation interval; the default is 2.
  - Selection is uniform within that interval, with separate saved roster and seat-order seeds.
  - This is a new draw, not the historical paper's full-grid sequence.
- Homogeneous-adversary runs put one adversary model among GPT-5-nano agents.
- TTC runs compare four native effort levels on the same generated game instance.
  - Use `--family gpt5`, `--family claude`, or `--family gemini`.
  - GPT-5 uses direct OpenAI, Claude Sonnet 4.6 uses direct Anthropic, and Gemini 3 Flash uses OpenRouter.
  - GPT-5 and Gemini levels are minimal/low/medium/high; Claude levels are low/medium/high/max.
  - Equal level positions do not imply equal compute budgets.
- Team runs use one direct-OpenAI GPT-5.4 High seat and a privately coordinated GPT-5-nano team in Game 1.
  - For more than two agents, the protocol shares team preferences, has three private planning turns, rotates a captain, and uses a binding team ballot.
  - At two agents, no team treatment is applied.
  - Fresh runs do not require historical controls and are not matched replications.
- Multi-agent is the umbrella term for the homogeneous, heterogeneous, homogeneous-adversary, and team presets.

All commands accept `--env-file` and `--output` with user-selected paths.
Output directories must be new; existing data is never overwritten by a new launch.
Without `--output`, the launcher creates a unique directory under `bargain-runs` in the current directory and prints the full path.

## Defaults and protocol differences

- All presets default to Game 1, seed 42, ten rounds, two discussion turns per round, and a 0.9 round discount.
- A command starts one negotiation, except TTC, which starts four in sequence.
- The adversary occupies the first seat; `--position last` changes its seat without changing the preference seed.
- Use `--game game2` or `--game game3`, except for the Game-1-only team preset.
- Group sizes are 2, 4, 6, 8, and 10; two-player and TTC presets require two.
- Games 1 and 3 use `5*n/2` items or projects.
- Game 2 uses five issues for two-player/TTC presets and ten for the multi-agent presets.
- `--competition` is the existing Game 1 **requested preference cosine**, default 0.5; do not confuse it with a relabeled competition axis in a figure.
- Game 2 defaults to `--rho 0 --theta 1`; infeasible Gaussian correlations are rejected before calls.
- Game 3 defaults to `--alpha 0.5 --sigma 0.6`, project costs from 10 to 30, and individual proposed contributions combined into a joint proposal for voting.
- Per-seat phase caps are explicit in the plan, typically 16,384 tokens, with smaller or larger caps for models with declared catalog limits or thinking budgets.
- `--max-tokens` sets a smaller explicit output cap; it cannot exceed a model's declared launch limit or undercut its fixed thinking budget.
- GPT-5-nano uses explicit low reasoning effort, matching the current native client's default, rather than the separate high-effort alias.
- Calls are serial, use the exact saved API route, and have a default 300-second timeout.
- Each request gets one attempt, with no provider substitution, automatic retry, or action repair.
- Proposals and ballots must pass validation before legacy parsers can pad, truncate, clamp, or supply missing decisions.
- Private thinking text is retained verbatim, without a manufactured default strategy.
- Team prompts use neutral participant wording.
- Full available history is retained until a 32,768-token estimated input limit, using the declared three-characters-per-token estimator.
  - Optional tokenizer installation cannot change this policy.
  - The run stops before sending an oversized prompt; it does not silently shorten history.
- A truncated generation or an absent final answer fails the run; hidden reasoning is never substituted for the answer.
- Preference generators are unchanged and may approximate the requested cosine.
  - Initial preferences and realized cosine summaries are saved before the first API call.
  - This release does not add a final cosine acceptance threshold, so inspect realized similarities before interpreting new research sweeps.

These decisions define `public-hosted-v1`, not an exact historical reproduction protocol.
Model routing, context handling, failure handling, thinking storage, and neutral team wording are explicit differences from legacy runners.
OpenAI requests use the [Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create), retaining the existing experiment's API family.

## Inspect a plan and results

Replace `run` with `plan` to print the complete resolved JSON without credentials or API calls.
Use `--save-plan` with an absolute new filename to save it.
`bargain execute` accepts that filename plus `--execute --output` and optional `--env-file`.

Use `bargain status` or `bargain summarize` with the absolute output directory.
Each output contains the plan, source hashes, dependency versions, per-run attempt histories, initial states, request records, interactions, and terminal status.
Request records preserve both requested and provider-reported model IDs, finish reasons, and raw usage fields.
Unknown usage stays unknown; the launcher does not invent billing totals.
Complete results require finite utilities for every requested seat and verified artifact hashes.
Completed disagreement at the round cap is a valid result with zero utility.
Missing, failed, canceled, and unknown-request attempts are not included as complete results.

## Resume without hiding failures

- `bargain resume` with the absolute output directory runs pending cells and skips only verified complete attempts.
- Add `--retry-failed` to restart an incomplete negotiation as a new attempt.
- A request interrupted after dispatch can have an unknown remote outcome.
  - Inspect the request record and provider account before deciding whether to repeat it.
  - The launcher cannot reconcile remote billing automatically and refuses to repeat it without `--accept-unknown-outcome` as well.
  - That explicit choice is recorded and may cause duplicate charges.
- Changed source files or required runtime versions require a new output directory.
- One process owns an output directory at a time; local POSIX file locking is required.
- The parent and worker share the lock so a surviving worker retains ownership if its parent exits.

## Not included in this release

Full paper manifests, historical control bundles, local model loading, custom API endpoints, account/project headers, key pools, HTTP proxies, file queues, and Slurm execution remain separate work.
Do not use these direct-network commands from restricted Slurm compute nodes.
Old scripts and archived data remain in place.
Offline tests do not establish live provider integration; validate the chosen route with a small real run before a research sweep.
