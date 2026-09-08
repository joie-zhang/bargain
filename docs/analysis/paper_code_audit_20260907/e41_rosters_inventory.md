# E41. Experiment counts and model selection

## Findings

- The current paper's 7,160 subtotal is 1,500 primary two-player + 500 Llama baseline + 1,300 heterogeneous + 1,300 homogeneous adversary + 300 homogeneous + 100 team + 2,160 TTC runs.
- The separate 25 matched GPT-5.4 coalition runs are outside that subtotal.
- The existing retained-corpus manifest has 7,143 rows, not 7,160.
  - It includes 130 all-Nano controls and 25 excluded Haiku homogeneous runs.
  - It has 2,088 TTC rows because its builder rejects the 72-run Claude seed-612 panel.
  - It has no 100 coordinated-team rows or 25 matched-coalition rows.
- The old manifest is still needed by a homogeneous figure loader and cannot serve as a deletion allowlist.
- No deletion candidate is established by this audit.

## Scope and checks

- Read current appendix inventory, game dimensions, 30-model two-player table, five-model homogeneous-adversary table, and homogeneous model assignments.
- Parsed all 5,785 saved config files across the two 2,730-config multiagent roots and the 325-config homogeneous root.
- Verified that each multiagent root contains a full planned set of 1,300 homogeneous-adversary, 1,300 heterogeneous, and 130 all-Nano configurations.
  - This means selecting a root or every config in a root does not select the paper cohort.
- Verified the selected heterogeneous cohort has 1,300 configurations, 24 distinct models, and the dimensions shown in the paper.
  - Game 1 has 100 runs per group size, with 5, 10, 15, 20, and 25 items.
  - Game 2 has 80 runs per group size and 10 issues at every size.
  - Game 3 has 80 runs per group size, with 5, 10, 15, 20, and 25 projects.
- Verified that the saved 24-model pool exactly equals the current sampler's filtered pool.
  - The removed names are deepseek-r1, qwen2.5-72b-instruct, llama-3.1-8b-instruct, llama-3.2-3b-instruct, llama-3.2-1b-instruct, and qwq-32b.
- Verified that the 325 homogeneous configurations assign five models to each game.
  - Game 1 has 25 runs per model, including 25 Claude 3 Haiku runs excluded from the paper's 300-run cohort.
  - Games 2 and 3 have 20 runs per model.

## Provenance chain

- `active_model_roster.py` reads the dated Arena Markdown file as data, including Elo and route-specific context limits.
- `full_games123_multiagent_batch.py` uses that parser to construct the 24-model pool and exact subsets for each group size.
- The sampler divides the full range of within-group Elo standard deviations into five equal-width ranges, not equal-count quantiles.
- Saved subset maps preserve the pool, subset identities, spread values, and stratum boundaries.
- `random_monoculture_control_batch.py` reads the saved heterogeneous pool CSV, excludes unavailable Sonnet 4, and samples 15 models once over five Elo bands with seed 20260628.
- The historical manifest builder joins the two-player and multiagent indexes, all 325 homogeneous records, and its older TTC panel selection.
- `lh_review_20260830/homogeneous_redesign/render_figure3_proposal.py:63` reads all 325 homogeneous result files before it excludes Haiku and asserts that 300 remain.
  - The excluded 25 result files are actual read dependencies of this loader.
  - The loader also removes the all-Nano group after loading older shared analysis data.

## History evidence

- Used the bundled codex-search script with the query `7160 Haiku 25 all nano 130 roster` and manually read the matched messages.
- Verified local session `/home/jz4391/.codex/sessions/2026/08/30/rollout-2026-08-30T23-03-21-01a055c5-7e2b-7610-acf9-95b8c56bd15c.jsonl`.
  - Lines 178 and 189 distinguish the 130 all-Nano controls from sampled homogeneous runs.
  - Lines 579, 720, and 804 explain the 7,160 count and removal of the 130 all-Nano and 25 Haiku records from the reported inventory.
  - Lines 1591 and 1597 contain the proposed appendix count changes and user approval.
- These messages explain selection changes but do not by themselves verify launch versions or raw results.

## Retention decisions and limits

- The companion JSON lists 28 specific needed paths or narrowly defined config-directory selections with evidence.
- Keep the dated Elo Markdown, model alias parser, saved pool and subset maps, configuration generators, saved manifests, and selected-run indexes.
- Keep the old manifest and its generator as provenance and existing consumer inputs, despite the count mismatch.
- Keep excluded controls and Haiku results until all readers and provenance obligations have been checked.
- Per-family agents cover exact launch records, raw results, transcripts, and runtime dependencies beyond this selection audit.
- No claim is made that every file in the containing directories is required.
- No source, data, or paper files were changed.
