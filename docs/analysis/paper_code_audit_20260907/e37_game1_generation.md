# E37 Game 1 valuation generation

## Scope

Covers Game1 nonnegative private item valuations summing to100, target cosine competition, two-player m=5, and multi-agent m=floor(2.5n), as described in the main setup and tab:appendix_game_sizes.

## Verified chain

- Primary launch uses scripts/generate_configs_both_orders.sh and its saved Slurm launcher; multi-agent launch uses scripts/full_games123_multiagent_batch.py.
- The shared runner creates ItemAllocationConfig, then ItemAllocationGame.create_game_state calls create_competitive_preferences.
- negotiation/preferences.py requests integer vectors with total100 and tolerance0.1 from MultiAgentVectorGenerator.
- For n=2, the generator draws an inner seed with RandomState(seed).randint(0,10000), then calls RandomVectorGenerator.
- The pairwise generator handles cosine0 with disjoint supports and cosine1 with copied vectors; intermediate targets use up to10 SLSQP optimizations, then total-preserving integer rounding.
- If all pairwise optimizations fail, the code returns its simple-method recovery rather than raising an error.
- For n>=3, current code jointly minimizes squared pairwise cosine errors using SLSQP with bounds0..100 and per-agent sum constraints; it rounds afterward.
- Nonconvergence or excessive average error yields a warning and returned vectors, not a guaranteed exact cosine.
- Multi-agent configuration seeds use SHA256 of the master seed and setting identifiers; actual resolved seeds are needed, not only a remembered base seed.
- Batch runs without an override add the run index to the seed; explicit overrides preserve the given seed.
- Matched replications can overwrite generated vectors with fixed_agent_preferences; those fixed tables remain required data.

## Independent real-data checks

- Opened all420 primary Game1 result JSONs selected by E01;414 have nonempty top-level valuation tables and6 failed runs have empty tables.
- Every available primary table has2 vectors of5 entries and exact total100 for each agent.
- Maximum absolute pairwise cosine deviation from the requested competition is0.016557865991622567.
- Opened all500 selected heterogeneous Game1 configuration and result JSONs;498 have nonempty tables and2 failed runs have empty tables.
- Available shapes are(2,5),(4,10),(6,15),(8,20),(10,25), and all sums equal100 exactly.
- Maximum absolute pairwise deviation is0.04055897311258083, in config0995 at n10 and target0.75.
- These observations validate saved table dimensions and sums, not provider identity or the scientific validity of synthetic proposals recorded elsewhere in these cohorts.
- Failed-run logs are needed to recover missing preferences; no missing values were substituted.

## Historical drift and search

Ran the codex-search bundled script against locally available repo history and manually checked the original June30 session019f1b6c-475f-7a71-9181-ae6c17387dc2 lines6444-6447. It explicitly distinguishes requested and realized cosine. Original history.jsonl lines662,810,864 document accuracy checks, the2.5n dimension proposal, and SLSQP requirement. Full April session availability was not established.

Git commit570e9f3c9e7b8d0586fd90c63fbcda9cb5b0b593, dated April25, replaced special three-agent and iterative many-agent paths with the current n>=3 SLSQP method. The n2 dispatch was unchanged in that diff. Historical experiments can depend on the earlier implementation, so Git objects and execution-time provenance must remain. Current comments about paired order seeds do not establish the actual April launch defaults.

## Needed files and removal decisions

The companion JSON gives concrete source/test/launch paths and bounded raw-data selectors. Shared runtime imports include package initializers, agent modules, JSON helpers, NumPy and SciPy. Full provider transitive closure belongs to the shared runtime audit; this report does not certify omitted code unnecessary.

No supported deletion candidates were found. Protect generation tests, old launch records, fixed preference tables, failed-run logs and historical source revisions. No source, data, paper, or Git state was changed.

