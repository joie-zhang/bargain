# E20 endpoint fairness dependency audit

## Result covered

- Current main Figure 3 right, labelled `fig:fairshare_headline`, compares both roles at low and high competition in each game.
- The main text claims that both roles move toward their reference utility at low competition, while adversary gains coincide with baseline losses at high competition.
- This is an analysis of the primary 1,500 runs, not an additional experiment.
- E01-E03 audit the original run launch, configurations, transport and recovery records; E19 covers overall bilateral fairness and E21 covers multi-agent fairness.

## Verified metric and selection

- Each role uses **undiscounted** realized utility, not the discounted payoff plotted elsewhere.
- The percentage gap is `200 * (actual - fair) / (abs(actual) + abs(fair))`.
- A denominator at most `1e-12` becomes zero.
- These are symmetric percentage differences, not a percentage of benchmark utility.
- Runs are averaged within adversary model, role and endpoint before plotting.
- Curves use forward exponentially weighted means with alpha 0.10 and `adjust=False`, sorted by Elo.
- The optional backward/forward smoothing exists, but the installed rendering calls the default forward method.
- Endpoints are the smallest and largest observed competition index, which can combine multiple parameter settings.
- Game 1 endpoints are c=0 and c=1, with 60 runs per endpoint per role and 30 model means.
- Game 2 endpoints are CI2=0 and CI2=1, where CI2=theta*(1-rho)/2.
- Game 2 cooperative series have 300 observations per role and 30 model means.
- Game 2 competitive series have 58 observations per role and 29 model means.
- The competitive Game 2 series explicitly excludes `llama-3.2-1b-instruct`, removing two runs or four role rows.
- Game 3 endpoints are CI3=0 and CI3=0.8, where CI3=(1-alpha)*(1-sigma).
- Game 3 competitive series have 60 observations per role and 30 model means.
- The retained Game 3 cooperative means count 300 baseline observations and 298 adversary observations.
- The cause of those two missing adversary metric observations remains unresolved here.
- There are 358 retained model-role-endpoint means in total.

## Benchmark distinction that must be preserved

- Games 1 and 2 take their benchmark utilities from the bilateral loader.
- The right-panel Game 2 benchmark uses the five-restart L-BFGS-B approximation in `fast_game2_nbs`, with random state 42.
- The exact bilateral solver used for the current left panel is therefore not a drop-in replacement for reproducing the right panel.
- Game 3 overrides the general bilateral loader's realized-funded-set Lindahl benchmark.
- The override enumerates every funded subset under the total budget, rejects proportional contributions above individual budgets, and maximizes the sum of log nonnegative utility plus epsilon.
- The chosen feasible-set benchmark can differ from the benchmark used in the left panel.
- The caption currently acknowledges the normalization and Game 3 benchmark differences.
- The main prose does not state the Game 2 competitive-model exclusion.

## Provenance chain and direct checks

1. The bilateral loader reads each original `configs/experiment_index.csv`, then the selected configuration, its output directory and observed result JSON.
2. The loader maps roles, reconstructs undiscounted metrics, and retains the two-turn Game 1 protocol.
3. The endpoint producer recomputes the Game 3 benchmark, selects endpoint rows and aggregates per model.
4. The installed style uses the preserved endpoint means under the old NeurIPS directory.
5. The recovered compositor combines the rendered right panels with two separately rendered left panels.
6. The installed ICLR PNG and the prior independent raw Haswell reconstruction have the same SHA-256:
   `f8c5d94a0e16f586a6e9c114a5d854339295f101f55a7544aaabbabb8c0c48f8`.

I checked all 1,500 result paths in the preserved bilateral reconstruction table and found no missing files.
I did not rerun experiments or recompute all benchmarks in this subaudit.
The count checks above read the actual retained means used for exact rendering.
The code checks read the current loader, endpoint producer, benchmark solver, raw reconstruction entrypoint and compositor.

## History evidence

- I ran the required repository-linked codex-search query for endpoint fairness and normalization.
- The generic search found July 1 inherited sessions and the later appendix discussion.
- I then followed the prior figure-audit leads and manually read the original July 29 and August 22 session records.
- July 29 line 308 installs the endpoint asset with the Game 2 exclusion into the rebuttal figures.
- August 22 line 448 changes the endpoint colors.
- August 22 line 460 explicitly reads the retained NeurIPS means and renders both paper copies.
- August 22 line 535 records why those retained values were preserved: a raw rebuild changed 112 means by at most 0.0074 percentage points because of the Game 2 optimizer.
- Exact session paths and lines are recorded in the companion JSON.

## Needed files and cleanup implications

The companion JSON lists 41 existing paths with reasons and evidence, including selected raw-data roots.
Raw roots are selection rules, not claims that every file below those directories is required.

- Keep the canonical endpoint producer and the bilateral/Nash-Lindahl metric code.
- Keep the active model roster and both Elo markdown inputs read by the loader.
- Keep package initializer dependencies, including the game classes imported when metric modules are loaded.
- Keep the historical endpoint CSV under `overleaf/neurips`, even though that venue is no longer current.
- Keep the left summary under `overleaf/icml_aiwild_template` while the current exact-render entrypoint reads it.
- Keep the recovered compositor and endpoint replay scripts under the ignored figure-recreation directory.
- Keep the original selected configs, results, setup transcripts and historical recovery records identified by E01-E03.
- Keep the input hash inventory and recorded numerical environment.
- No deletion candidate is established by this result audit.

A directory name such as `analysis`, an old venue name, or a dated review folder does not show that its contents are unused.
The installed paper image still depends on numeric and code artifacts in such locations.

## Limits

- Original launch and runtime-version completeness depends on E01-E03, not only this plotting trace.
- The current loader and current runtime cannot prove which historical provider actually served a run.
- The stored means reproduce the installed figure, but numerical solver replacement could alter it.
- No code, paper, data or Git state was changed; only this audit Markdown and JSON were added.

