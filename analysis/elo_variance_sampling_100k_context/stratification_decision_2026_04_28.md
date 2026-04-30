# Elo-Stddev Stratified Heterogeneous Sampling Plan

Original note date: 2026-04-28

Revised implementation-plan date: 2026-04-29

Status: implemented in `scripts/full_games123_multiagent_batch.py` as the default heterogeneous sampling strategy. This file records the design rationale and implementation contract.

## One-Sentence Summary

For heterogeneous N-agent runs, replace pure random model-subset sampling with a default, toggleable, Elo-stddev-stratified sampler that excludes deprecated models, enumerates all unordered model subsets for each `N`, divides the attainable Elo-stddev range into five equal-width strata, samples `4` subsets from each stratum per heterogeneous cell, randomizes model order after subset selection, and logs enough metadata to support Gini-vs-Elo-spread plots and later reweighting analyses.

## Motivation

The heterogeneous experiment is intended to support plots relating group-level inequality or dispersion, especially Gini coefficient, to the Elo spread of the models in the group.

The original pure-random heterogeneous sampler chooses `N` distinct models uniformly from the filtered model pool and then shuffles their order. That is simple, but it inherits the natural combinatorial distribution of `C(pool_size, N)` subsets. Because most possible subsets have middle Elo spread, pure random sampling tends to produce many middle-Elo-variance runs and fewer low-spread or high-spread tail runs. This is statistically natural, but it is not ideal when the target analysis needs strong x-axis coverage in Elo spread.

The revised plan is to stratify heterogeneous draws by the empirical Elo standard deviation of unordered model subsets. After red-teaming, the plan uses equal-width Elo-stddev strata rather than equal-count strata. Equal-width strata better match the stated plotting goal: they deliberately cover the x-axis range, including tail regions, instead of giving each bin the same number of possible subsets.

## Current Code Context

The current heterogeneous generation surface is:

- `scripts/full_games123_multiagent_batch.py`

Relevant current behavior:

- `N_VALUES = [2, 4, 6, 8, 10]`
- `HETEROGENEOUS_RUNS_PER_CELL = 20`
- `filtered_heterogeneous_pool()` imports the context-filtered model pool from `strong_models_experiment.analysis.active_model_roster`
- current heterogeneous sampling is pure random: `rng.sample(hetero_pool, n_agents)` followed by `rng.shuffle(models)`
- `HETEROGENEOUS_EXCLUDED_MODELS` already excludes `qwq-32b`

The exploratory diagnostic script is:

- `analysis/elo_variance_sampling_100k_context/plot_stratified_sampling_24pool.py`

The generated diagnostic outputs are under:

- `analysis/elo_variance_sampling_100k_context/stratified_sampling_24pool/`

The key diagnostic plot is:

- `analysis/elo_variance_sampling_100k_context/stratified_sampling_24pool/exact_elo_variance_histograms_24pool.png`

## Updated Model-Pool Rule

The heterogeneous pool should exclude:

- `qwq-32b`
- `gemini-3-pro`

Reason:

- `qwq-32b` was already excluded from heterogeneous sampling. The implementation/writeup should preserve the original reason for this exclusion rather than merely saying it was already excluded.
- `gemini-3-pro` is a deprecated/legacy label and should not appear in any generated heterogeneous subset, generated model roster, model-pool map, or run command. It may appear only in explicit exclusion metadata.
- `gemini-3.1-pro` is retained as the explicit Gemini 3.1 Pro replacement model, using the same position in the active smooth-coverage roster that had previously been labeled as Gemini 3 Pro.

Assuming the active context-filtered pool has 25 models, excluding `qwq-32b` while replacing the old Gemini 3 Pro label with explicit `gemini-3.1-pro` yields a 24-model heterogeneous pool.

The implementation should update:

- `HETEROGENEOUS_EXCLUDED_MODELS`
- any validation that currently expects heterogeneous pool size `24`
- any generated metadata that records `model_pool_size`
- any excluded-model reference checks, so both excluded model IDs are caught

## Design Space After Deprecation

With a 24-model heterogeneous pool, the number of possible unordered subsets is:

| N | Number of subsets |
|---:|---:|
| 2 | 276 |
| 4 | 10,626 |
| 6 | 134,596 |
| 8 | 735,471 |
| 10 | 1,961,256 |

`N=2` remains in the design. Pairs are qualitatively different from larger groups, so downstream analysis should be able to facet or sensitivity-check by `N`. However, the generator should not drop `N=2`.

## Stratification Variable

Use Elo standard deviation, not raw Elo variance, for bucket construction.

Rationale:

- Elo stddev is in Elo-point units, so stratum boundaries are easier to interpret.
- Elo stddev is a monotonic transform of Elo variance, so ordering and range coverage are closely tied to variance coverage.
- The analysis can still log and plot raw variance, because each subset's variance and stddev will both be recorded.

For each unordered subset:

- collect the model Elo values
- compute population Elo mean
- compute population Elo variance with `ddof=0`
- compute Elo stddev as `sqrt(variance)`

## Subset Map Generation

Create human-readable subset-map artifacts under:

- `configs/heterogeneous_subset_maps/`

Recommended generated files:

- `manifest.json`: generation timestamp, code version if available, model-pool size, excluded models, `N` values, number of strata, stratum method, RNG seeds, RNG state at the start of generation, and source files used
- `model_pool_24.csv`: active heterogeneous model IDs and Elo values after exclusions
- `strata_boundaries.csv`: one row per `(N, stratum)`, with equal-width stddev and derived variance boundaries
- `exact_subset_summary.csv`: one row per `N`, with subset counts and Elo-spread summary statistics
- `n_02_subsets.csv` or `n_02_subsets.jsonl`
- `n_04_subsets.csv` or `n_04_subsets.jsonl`
- `n_06_subsets.csv` or `n_06_subsets.jsonl`
- `n_08_subsets.csv` or `n_08_subsets.jsonl`
- `n_10_subsets.csv` or `n_10_subsets.jsonl`

Practical note: the full human-readable subset maps are large because they enumerate every subset through `C(24, 10) = 1,961,256`. A smoke generation wrote roughly 1.6 GB of subset-map artifacts. This is acceptable for auditability on scratch storage, but repeated dry generations should clean up old result directories.

Each subset-map row should include at least:

- `n_agents`
- `subset_key`, a stable unordered identifier, for example model IDs joined in sorted pool order
- `subset_rank_by_stddev`
- `subset_rank_within_stratum`
- `stratum_index`, using `0..4`
- `stratum_label`, for example `very_low`, `low`, `mid`, `high`, `very_high`
- `stratum_method`, set to `equal_width_stddev`
- `stratum_population_size`
- `stratum_stddev_min`
- `stratum_stddev_max`
- `stratum_variance_min`
- `stratum_variance_max`
- `model_ids_unordered`
- `model_elos_unordered`
- `elo_mean`
- `elo_variance`
- `elo_stddev`

CSV is preferred for transparency. JSONL is also acceptable if the model lists are easier to store as structured arrays. The important constraint is that the maps remain inspectable without a custom binary reader.

## Bucket Construction

For each `N in {2, 4, 6, 8, 10}`:

1. Build the filtered 24-model heterogeneous pool.
2. Enumerate every unordered `C(24, N)` subset.
3. Compute each subset's Elo mean, variance, and stddev.
4. Find the minimum and maximum attainable Elo stddev for that `N`.
5. Divide the interval `[min_stddev, max_stddev]` into five equal-width bins.
6. Assign subsets to bins by realized Elo stddev.
7. Use half-open intervals for bins `0..3`, and make the final bin right-inclusive so the maximum attainable stddev is included.
8. Within each bin, sort subsets deterministically by `(elo_stddev, subset_key)`.
9. Validate that every bin is nonempty. If any bin is empty, fail generation and revisit the binning design rather than silently borrowing from adjacent bins.

Equal-width strata are preferred over equal-count strata for this specific goal. Equal-count strata allocate the same number of possible subsets to each bucket, but the middle buckets can span a very narrow stddev range while tail buckets span a much wider range. Equal-width strata directly target x-axis coverage.

## Sampling Strategy

Use stratified random sampling within equal-width Elo-stddev strata.

For each heterogeneous cell, where a cell means one `(game_label, n_agents, competition_id)` combination:

1. Keep the current total of `20` heterogeneous runs per cell.
2. Allocate those runs as `4` draws from each of the five Elo-stddev strata.
3. Sample unordered subsets uniformly from the selected stratum.
4. Sample with replacement across the design; do not enforce global no-repeat.
5. Do not introduce planned roster-level replication for variance decomposition.
6. After selecting the unordered subset, randomly permute the model order for the actual run config.
7. Record both the unordered subset and the realized ordered model list.

This replaces the earlier pure-random method as the default, but the original method should remain available behind a strategy flag or constant.

Recommended strategy names:

- `pure_random`
- `elo_stddev_equal_width_stratified`

Recommended default:

- `HETEROGENEOUS_SAMPLING_STRATEGY = "elo_stddev_equal_width_stratified"`

## Repeat Policy

Do not enforce global no-repeat.

Rationale:

- A global uniqueness constraint is not aligned to a clear inferential estimand.
- It creates unnecessary coupling across cells and is not aligned to the current inferential target.
- It makes the sampler more complex without clearly improving the Gini-vs-Elo-spread analysis.

The sampler should instead use independent stratified random draws from the appropriate `(N, stratum)` candidate set. Repeated rosters are allowed as accidental outcomes of random sampling, but the design should not intentionally allocate repeated rosters for variance decomposition.

Logging should make repeats transparent:

- `subset_reuse_count_global_before_this_draw`
- `subset_reuse_count_global_after_this_draw`
- `subset_reuse_count_within_cell_before_this_draw`
- `subset_reuse_count_within_cell_after_this_draw`
- `planned_roster_replication`, set to `false`
- `sampling_with_replacement`, set to `true`

If an accidental duplicate occurs within the same cell, it should be logged rather than silently replaced, unless a later design decision explicitly adds a local no-repeat rule.

## Why Not Target-Stddev Nearest-Neighbor Sampling?

One considered alternative was:

1. draw a random target Elo stddev within a stratum's boundary range
2. choose the subset whose realized stddev is closest to that random target

This would make the x-axis coverage visually smooth, but it is less methodologically clean because subset inclusion probabilities depend on the local density of attainable Elo-stddev values. Sparse regions of stddev space would be intentionally favored relative to dense regions.

The recommended method is simpler to defend:

- define equal-width strata over the attainable stddev range
- sample uniformly within each stratum
- allocate equal run counts to each stratum

This directly improves x-axis coverage while keeping the sampling rule explainable.

## Randomness, Seeds, And RNG State

The sampler should be deterministic under the existing master seed.

Recommended seed structure:

- subset-map generation should be deterministic and should use sorted subset keys for tie-breaking
- stratum assignment should be deterministic for a given model pool and Elo map
- subset sampling should use stable seeds derived from the master seed, strategy, `N`, stratum, game label, competition ID, and run index or assignment block
- model-order shuffling should use a separate stable seed from unordered-subset selection

In addition to logging seeds, log the random number generator state at the start of generation.

Recommended RNG logging:

- in `manifest.json`, store the Python version and random module name
- after seeding the generation RNG and before the first random draw, store a JSON-serializable form of `random.Random.getstate()`
- also store a SHA256 hash of that serialized state for quick comparison
- if NumPy RNG is used anywhere in sampling, store the bit-generator name and `bit_generator.state`
- prefer Python's `random.Random` for sampling if the existing generator already uses it, to avoid introducing extra NumPy RNG reproducibility concerns

This does not guarantee reproducibility across every Python version, but it makes the generation state auditable and much easier to reproduce on the same runtime.

## Config Metadata To Log

Each heterogeneous config should include rich metadata. More fields are better because downstream analysis may need to reconstruct the sampling process.

Recommended config and experiment-index fields:

- `heterogeneous_sampling_strategy`
- `heterogeneous_run_index`
- `heterogeneous_draw_seed`
- `heterogeneous_order_seed`
- `generation_rng_state_start_hash`
- `sampling_rng_state_start_hash`
- `model_pool_size`
- `model_pool`
- `heterogeneous_excluded_models`
- `n_agents`
- `subset_key`
- `subset_rank_by_stddev`
- `subset_rank_within_stratum`
- `subset_reuse_count_global_before_this_draw`
- `subset_reuse_count_global_after_this_draw`
- `subset_reuse_count_within_cell_before_this_draw`
- `subset_reuse_count_within_cell_after_this_draw`
- `sampling_with_replacement`
- `planned_roster_replication`
- `stratum_index`
- `stratum_label`
- `stratum_method`
- `stratum_population_size`
- `stratum_stddev_min`
- `stratum_stddev_max`
- `stratum_variance_min`
- `stratum_variance_max`
- `subset_model_ids_unordered`
- `subset_model_elos_unordered`
- `models`, the final ordered model list used in the run
- `agent_model_map`
- `agent_elo_map`
- `elo_mean`
- `elo_variance`
- `elo_stddev`
- `elo_bucket_method`
- `model_elo_bucket_map`
- `agent_elo_bucket_map`

The key point is to log both:

- the unordered statistical unit that was sampled
- the ordered agent assignment that was actually run

## Analysis Rationale

The intended main plot is a Gini coefficient vs Elo-spread plot. The x-axis can use either Elo stddev or Elo variance, but the sampling strata should be based on equal-width Elo stddev.

Expected benefits:

- every heterogeneous cell gets low, medium, and high Elo-spread groups
- each cell has exactly `4/4/4/4/4` coverage across five stddev ranges
- tail Elo-spread observations are guaranteed instead of being left to pure random chance
- the sampling method remains simple enough to describe as stratified random sampling
- keeping `N=2` allows continuity with the broader `N in {2,4,6,8,10}` design while still allowing later sensitivity checks by `N`

Potential costs:

- the sampled design no longer follows the natural subset distribution from pure random draws
- analyses estimating natural-population averages may need sampling weights
- equal-width tail strata may have far fewer candidate subsets than middle strata
- accidental roster repeats may occur because sampling is with replacement
- `N=2` remains qualitatively different from larger groups and should be interpreted carefully

These costs are acceptable if the primary goal is strong x-axis coverage for relationship estimation, especially slope or shape estimation in Gini-vs-Elo-spread plots.

## Mentor Note On Reweighting

A mentor suggested that for histogram-style plots, not necessarily for the Gini-vs-stddev plot, expected utilities can later be reweighted using a multinomial distribution so results can be viewed as if sub-populations of players were drawn uniformly across Elo buckets.

This does not need to be implemented in the sampler now. The important requirement is to log enough information for post-hoc reweighting.

The proposed metadata supports this because it records:

- each selected model ID
- each selected model's Elo
- each model's assigned agent ID
- the unordered subset
- the final ordered model list
- subset-level Elo mean, variance, and stddev
- stratum population sizes and boundaries
- whether sampling was with replacement
- subset reuse counts
- the realized expected utilities from the experiment outputs

With those fields, later analysis can reconstruct model-level and subset-level Elo composition and apply alternative weighting schemes after the runs complete.

## Validation Requirements

Before accepting generated configs, validation should check:

- `gemini-3-pro` appears in no heterogeneous model pool, subset map, run model list, or recursive config payload except explicit exclusion metadata
- `qwq-32b` appears in no heterogeneous model pool, subset map, generated config, or recursive config payload
- the heterogeneous model pool size is `24`
- every heterogeneous config has `N` unique models within that run
- every heterogeneous cell has exactly `20` heterogeneous runs
- every heterogeneous cell has exactly `4` runs in each of the five Elo-stddev strata
- every stratum used for sampling is nonempty
- each config's logged Elo stats round-trip exactly from the logged model IDs and the generation-time Elo map
- each config's realized `elo_stddev` falls inside its logged equal-width stratum boundaries
- the realized stddev distribution per stratum matches the generated boundaries
- repeat metadata is present and internally consistent
- RNG seeds and start-state hashes are logged in the manifest and propagated to config metadata where appropriate
- all logged subset metadata agrees with the generated subset map
- the original `pure_random` strategy still works when selected

## Implementation Steps

1. Keep `gemini-3-pro` in the heterogeneous exclusion set as a deprecated label, add explicit `gemini-3.1-pro` support, and validate the pool size as `24`.
2. Add a sampling-strategy constant or CLI/config option, defaulting to `elo_stddev_equal_width_stratified`.
3. Create or adapt a map-generation script that enumerates all `C(24, N)` subsets and writes human-readable artifacts under `configs/heterogeneous_subset_maps/`.
4. Change the map-generation logic from equal-count strata to equal-width Elo-stddev strata.
5. Validate that all five strata are nonempty for every `N`.
6. Implement a loader for the subset maps, or compute the same map in memory during config generation while writing the map artifacts for audit.
7. Replace the heterogeneous pure-random branch with a strategy dispatch:
   - `pure_random`: current behavior
   - `elo_stddev_equal_width_stratified`: five equal-width strata sampler
8. In the stratified branch, allocate `4` draws per stratum per heterogeneous cell.
9. Sample with replacement from each `(N, stratum)` candidate set, with no global no-repeat constraint and no planned roster-level replication.
10. Randomly shuffle model order after unordered subset selection.
11. Track and log global and within-cell subset reuse counts.
12. Log RNG seeds, RNG start state, and RNG start-state hashes in the manifest and relevant config metadata.
13. Add all sampling and Elo metadata to config JSON and `experiment_index.csv`.
14. Extend validation to check exclusions, counts, stratum balance, nonempty strata, metadata consistency, Elo-stat round-tripping, and RNG-state logging.
15. Run a dry-generation check and inspect the generated maps before launching experiments.

## Reviewer-Facing Description

A compact reviewer-facing description could be:

> For heterogeneous groups, we first constructed the finite population of all unordered subsets of size `N` from the eligible model pool after excluding deprecated or otherwise ineligible models. For each subset, we computed the population standard deviation of the constituent models' Elo ratings. For each `N`, we divided the attainable Elo-standard-deviation range into five equal-width strata. Each heterogeneous experimental cell used 20 random rosters: four sampled uniformly from each Elo-stddev stratum. After selecting an unordered roster, we randomly permuted model order before assigning agents. Sampling was conducted with replacement within strata; repeated rosters were not planned but were allowed as ordinary random-sampling outcomes and explicitly logged. All analyses use the realized logged Elo mean, Elo variance, and Elo standard deviation of each roster.

## Open Questions To Red-Team

- Should equal-width bins be defined over the full `[min, max]` attainable stddev interval, or should extreme outliers be handled with fixed tail bins?
- Should accidental within-cell duplicate rosters be allowed exactly as sampled, or should a later design add a local no-repeat rule while still dropping global no-repeat?
- Should the final plot use Elo stddev on the x-axis for interpretability, raw Elo variance for mathematical consistency with earlier notes, or both?
- Should downstream analyses report unweighted stratified estimates, reweighted natural-population estimates, or both?
- Should legacy `gemini-3-pro` labels in old result analyses be relabeled to `gemini-3.1-pro`, or kept as historical aliases with explicit notes?
