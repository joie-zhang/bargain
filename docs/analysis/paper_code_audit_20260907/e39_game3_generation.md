# E39 Game 3 project, budget and valuation generation

- Scope is appendix A.4, `sec:appendix_game3_generation`, and project-count scaling in the settings table.
- No experiment, code, data or paper changes were made.
- The companion JSON protects 540 individually read primary result files and 16 concrete supporting files.

## Verified generation chain

- The primary shell generator creates the saved two-player grid; E03 verifies its saved worker command into the common runner.
- The multi-agent batch generator sets `m_projects=int(2.5*n_agents)`, `c_min=10`, `c_max=30`, alpha and sigma at lines 597-615.
- The runner passes these fields into the experiment and environment factory.
- `game_environments/__init__.py:120-135` constructs `CoFundingConfig`; importing this package also imports the other game modules and metrics.
- `game_environments/co_funding.py:255-345` samples inclusive uniform integer costs and sets each budget to `floor(sigma*sum(costs)/n + 0.5)`.
- The generator uses a seeded NumPy RandomState, then generates valuations, so changing the sequence of random draws changes seeded preferences.
- Valuations sum to 100, are nonnegative and integerized by largest remainder, then refined to reduce pairwise cosine error.
- Alpha 1 copies a shared Dirichlet valuation vector.
- Other alpha values now use an analytic shared-component/private-project initialization, with SLSQP available when the initialization does not satisfy the error threshold.
- Current code skips SLSQP when that initialization is already exact; the appendix phrase “jointly optimize” therefore does not mean that every current generation calls the optimizer.
- Tests protect budget rounding, cost bounds, exact initialization, cosine targets, seed behavior and configuration validation.

## Raw-state checks

- Read every one of the 540 selected primary Game 3 result JSONs named by E03.
- Every saved cost is an integer in 10-30.
- Every saved preference vector has nonnegative integer entries and sums to 100.
- Every stored agent budget matches the direct sigma rule with half-up rounding.
- Two Qwen2.5-72B records at alpha 1, sigma 0.2 have only one entry in `agent_budgets`, despite `n_agents=2`.
  - These are the weak-first and strong-first records, with stored budgets 11 and 9, respectively.
  - This is missing state information, not evidence of a different numerical budget rule.
- The saved batch summary says `0.5+0.5*sigma`, but that rule matches only the 180 sigma-1 records, where both formulas coincide.
- Do not replace or delete the old summary; it documents inconsistent historical metadata.

## Historical generation is different

- The saved seed-42, alpha-0 primary result for Claude Opus 4.6 has costs `[16,29,24,20,17]` and valuations `[0,0,63,37,0]` and `[38,29,0,0,33]`.
- A read-only local state-generation check with current code reproduces those costs but produces valuations `[0,0,0,100,0]` and `[0,100,0,0,0]`.
- This check creates no runs, invokes no model and saves no data; its generated values are a diagnostic, not research evidence.
- Git commit `dcb4bf644af4b2774b6a9b4d59bf64b156345828` changed the two-agent generation path on April 27.
- Git commit `444fcf9c368a60dd40a0f5f7aa8b4033552312bf` added the exact-start early exit on August 30.
- The old `_generate_valuations_2agents` helper has no current Python caller found by repository search, but was production logic historically.
- Preserve the file and Git history; a seed alone cannot replay the primary cohort with current code.
- The recovered backfill records described by `root_recovered_archives.md` must remain available; this audit does not repeat their full worker inspection.

## Conversation search

- Read and followed codex-search, ran its bundled search for `budget_ratio sigma cofunding`, and manually checked the returned June 27 session at line 770.
- That result is later TTC interpretation, not original launch evidence.
- Manually checked local history lines 173 and 182 for session `019c925d-78a1-7a82-a783-1f81d08ff501`.
- These original user requests discuss low co-funding provision and a proportional budget change.
- The local record does not prove the exact source bytes executed for each April attempt.

## Deletion decisions

- No whole file is approved for deletion by this audit.
- Current unused helper status is insufficient to delete its containing runtime file or historical source.
- Other Game 3 cohorts are audited in E06, E07-E09, E12, E15 and E18; this audit does not claim a second per-file scan of those cohorts.
- External NumPy/SciPy versions and original execution environments remain part of replay provenance.
