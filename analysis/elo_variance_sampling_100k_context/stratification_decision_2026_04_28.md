# Elo-Variance Stratification Decision

Date: 2026-04-28

## Context

For the heterogeneous N-agent experiment, we considered deliberately stratifying sampled model rosters by the Elo variance of the selected models. The motivation was to improve coverage of the x-axis for the planned `Gini coefficient vs Elo variance` analysis.

The exploratory diagnostic script is:

- `analysis/elo_variance_sampling_100k_context/plot_stratified_sampling_24pool.py`

The generated diagnostic outputs are under:

- `analysis/elo_variance_sampling_100k_context/stratified_sampling_24pool/`

The key plot is:

- `analysis/elo_variance_sampling_100k_context/stratified_sampling_24pool/exact_elo_variance_histograms_24pool.png`

## What The Diagnostic Does

The script imports the live 24-model heterogeneous pool used by `scripts/full_games123_multiagent_batch.py`, enumerates every unordered subset for each `N in {2, 4, 6, 8, 10}`, and computes the population Elo variance and Elo standard deviation for each subset.

It then compares:

- pure random sampling over all subsets
- equal sampling from five Elo-standard-deviation strata

This was done for exploratory sample sizes `k = 1,000` and `k = 10,000`.

## Main Finding

The space of possible heterogeneous rosters is not uniform in Elo variance. For larger `N`, most possible subsets cluster around middle Elo variance, with low-variance and high-variance rosters in thinner tails. For `N = 2`, the distribution is especially discrete and skewed because there are only `C(24, 2) = 276` possible pairs.

Exact subset counts:

| N | Number of subsets |
|---:|---:|
| 2 | 276 |
| 4 | 10,626 |
| 6 | 134,596 |
| 8 | 735,471 |
| 10 | 1,961,256 |

The five equal-count Elo-standard-deviation strata are feasible for the current full heterogeneous design. Even the tightest case, `N = 2`, has enough unique unordered subsets to support 20 runs per cell with no repeated unordered subset across the full design for that `N`.

## Design Decision

We decided **not** to adopt Elo-variance-stratified sampling for the main experiment.

Reason: although stratification would improve x-axis coverage for estimating the `Gini vs Elo variance` slope, it makes the sampling scheme harder to explain. For the main experiment, the preferred design is to keep heterogeneous sampling simple and reviewer-legible: sample `N` distinct models uniformly at random from the filtered model pool, randomize model order, and record the realized Elo variance after the fact.

## How To Use This Later

This diagnostic should be treated as archival design context, not as the current experimental plan.

It may still be useful for:

- explaining why random heterogeneous draws tend to concentrate around middle Elo variance
- reporting the realized coverage of Elo variance after runs complete
- motivating a robustness check or follow-up experiment if reviewer feedback asks for stronger tail coverage

The current main-design recommendation remains: keep pure random heterogeneous model sampling and analyze realized Elo variance post hoc.
