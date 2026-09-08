# E33 rounds used audit

## Current result

- Current appendix source lines 511-520 uses all 1,500 primary runs, not the earlier agreement-only sample.
- Selected cohorts are 420, 540 and 540 runs, with 30 model means in each game.
- All 1,500 selected result JSON and config JSON files were reopened in this audit.
- Raw consensus, final round and ten-round config cap match the rebuilt CSV with zero mismatches.
- The 6, 6 and 80 unsuccessful runs already record final_round=10.
- Independently recomputed slopes per 100 Elo are -0.2928754526949001, -0.27398635150281064 and -0.4063093471411817.
- Current paper PNG and revised PNG have identical SHA256 c277dd9d9fb50cfe6fea454d390f0c7c4efedccb88ba237dd7296fff7ebd636b.

## Verified chain

- The primary experiment runtime records consensus and accepted round; current implementation is in strong_models_experiment/experiment.py:756-758 and 840-861.
- Primary launch/provider/history scope is shared with E01-E03; this diagnostic adds no new experiment runs.
- The three original indexes select configs and raw JSON files through analysis/figure_recreation_20260905/figure_21/scripts/rebuild_raw_input.py.
- That script dynamically imports scripts/analyze_n2_baseline_comparison.py and its model/Elo/metrics dependencies.
- The rebuilt CSV records exact config_path and result_path for every selected run.
- analysis/rounds_all_runs_20260907/render.py imports the canonical plot renderer, fills unsuccessful rounds with ten, computes means across each model's attempts, then unweighted OLS across 30 model means per game.
- The paper uses the revised PNG, verified by its hash.

## History

- Mandatory codex-search found original session 01a07b00-5295-7be3-b0ce-f24498a5961e.
- /home/jz4391/.codex/history.jsonl:4189-4192 records old text, user objection, requested new definition, and requested paper replacement.
- /home/jz4391/.codex/sessions/2026/09/07/rollout-2026-09-07T04-33-33-01a07b00-5295-7be3-b0ce-f24498a5961e.jsonl:89-103 records script creation/execution and results.
- Replacement approval appears at lines 107-114, but that turn was interrupted; the current paper asset independently confirms the replacement.

## Needed files and limits

- The companion JSON lists exact code/support/output paths and explicit raw-data selectors.
- The raw selectors cover only the 1,500 config/result paths in the verified CSV, not every file in those experiment directories.
- Both the dated current wrapper and old reconstruction directory remain needed; their dates do not make them disposable.
- Preserve the side-by-side and provenance record because the wrapper reads the mutable paper PNG and cannot reproduce the original comparison unchanged now.
- The default canonical renderer and shared analysis still calculate conditional means; they are not substitutes for the current wrapper.
- No deletions are supported by this result audit.
- Runtime/provider version provenance remains with E01-E03; the current runtime code alone cannot prove the exact historical implementation.

## Interpretation issue, no changes made

- Capped rounds used combines agreement probability and agreement speed.
- Its negative association with Elo does not separately establish faster consensus or a causal effect.

