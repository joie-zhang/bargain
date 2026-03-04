# Qualitative Analysis Implementation Plan (March 2026)

This document translates the qualitative-analysis agenda from `approach_feb_23.tex` into an implementation roadmap.

## Scope

Primary source sections:
- `approach_feb_23.tex` lines 216-230 (`Strategic behavior metrics` + `Scalable extraction pipeline`)
- `approach_feb_23.tex` lines 176-201 (`Talk-Pledge-Revise` protocol details)

Target qualitative metrics:
- Promise-keeping rate
- Persuasion effectiveness
- Coalition formation
- Adaptation rate

## Mandatory Cleanup Policy

We are **not** using legacy keyword retrieval methods for qualitative analysis.

Required cleanup:
1. Remove keyword-based qualitative inference logic (for example, `manipulate`, `gaslighting`, `anger` string matching) from active code paths.
2. Remove legacy keyword-behavior outputs from active visualization scripts.
3. Remove references in docs and script outputs that imply keyword heuristics are valid qualitative metrics.
4. Keep old historical result files unchanged, but mark them as legacy and non-authoritative for qualitative conclusions.

## Current State Snapshot

Implemented already:
- Rich co-funding game logs with round/phase structure.
- Co-funding quantitative metrics (efficiency, Lindahl, free-rider, coordination variants).
- Co-funding commit vote and transparency updates.

Missing relative to manuscript goals:
- No robust transcript-to-event extraction pipeline.
- No LLM-judge validation stage.
- No production computation of promise-keeping/persuasion/coalition metrics at scale.
- Existing `strategic_behaviors` implementation was legacy keyword-based and needed replacement.

## Technical Roadmap

### Stage 1: Structured Data Foundation

Deliverables:
- Canonical qualitative schema (`qualitative_metrics_v1`) attached to each result.
- Per-experiment event list (`qualitative_events`) for auditability.
- Round-aligned pledge history extraction utilities.

Engineering tasks:
1. Define event schema for explicit commitments and advocacy mentions.
2. Ensure event extraction is deterministic and round-indexed.
3. Add lightweight quality guards (schema checks and null-safe handling).

Acceptance criteria:
- Every co-funding run can produce a valid `qualitative_metrics_v1` object, even if sparse.

### Stage 2: Metric Computation

Deliverables:
- `promise_keeping` block
- `persuasion_effectiveness` block
- `coalition_formation` block
- `adaptation_rate` block

Engineering tasks:
1. Promise-keeping:
   - Parse explicit numeric commitments.
   - Compare with same-agent next-round pledge values for promised projects.
   - Report keep-rate and mean absolute error.
2. Persuasion:
   - Detect project advocacy mentions.
   - Compute next-round contribution deltas by other agents.
3. Coalition:
   - Detect multi-agent co-funding rounds by project.
   - Compute persistent coalition statistics via consecutive-round streaks.
4. Adaptation:
   - Reuse vector-based adaptation formula from cofunding metrics.

Acceptance criteria:
- Metrics compute from logs without keyword sentiment heuristics.
- Missing-data behavior is explicit (`None`/`NaN`, not silent zeros).

### Stage 3: Integration with Experiment Outputs

Deliverables:
- `ExperimentResults` contains:
  - `qualitative_metrics_v1`
  - `qualitative_events`
- Batch summary remains compatible.

Engineering tasks:
1. Wire qualitative computation into post-game analysis for co-funding runs.
2. Preserve compatibility fields where needed, but ensure they are no longer heuristic-keyword based.

Acceptance criteria:
- New runs serialize qualitative metrics by default.

### Stage 4: Visualization Migration

Deliverables:
- Active visualizations consume structured qualitative metrics.
- Legacy keyword behavior plots removed/replaced.

Engineering tasks:
1. Update `visualization/visualize_cofunding.py` to use adaptation and optional qualitative blocks.
2. Update `visualization/gpt5_nano_analysis.py` to remove legacy keyword behavior columns/plots.
3. Rename or replace legacy behavior figure outputs with structured qualitative metrics figures.

Acceptance criteria:
- No active visualization code depends on keyword heuristic behavior fields.

### Stage 5: Validation and QA

Deliverables:
- Unit tests for extraction and metric math.
- Regression checks for null-safety and schema consistency.

Engineering tasks:
1. Add tests for commitment parsing and promise-keeping matching.
2. Add tests for persuasion delta computation.
3. Add tests for coalition persistence logic.
4. Validate adaptation extraction from pledge histories.

Acceptance criteria:
- New qualitative pipeline passes dedicated tests and does not break existing co-funding metrics tests.

### Stage 6: LLM Judge Validation (Next Increment)

Deliverables:
- Judge rubric spec
- Sampled validation harness
- Extraction quality report (precision/recall by event type)

Engineering tasks:
1. Stratified transcript sampling.
2. Judge prompts with strict rubric and evidence spans.
3. Agreement/error reporting and confidence calibration.

Acceptance criteria:
- Extraction quality is auditable with documented error bars.

## Operational Notes

- Historical outputs containing legacy keyword metrics remain archived but should not be used for new analysis claims.
- Any published or advisor-facing figures should be regenerated from runs that include `qualitative_metrics_v1`.

## Immediate Next Actions

1. Land Stage 1-4 code changes in minimal robust form.
2. Add targeted tests for qualitative metric utilities.
3. Re-run co-funding visualization generation on latest results.
4. Start Stage 6 (judge validation) once event extraction schema stabilizes.

## Backfill / Regeneration Commands

For existing co-funding runs (without rerunning experiments):

```bash
python scripts/backfill_qualitative_metrics.py \
  --results-dir experiments/results/cofunding_latest
```

Optional with event payloads:

```bash
python scripts/backfill_qualitative_metrics.py \
  --results-dir experiments/results/cofunding_latest \
  --include-events
```

Then regenerate figures:

```bash
python visualization/visualize_cofunding.py \
  --results-dir experiments/results/cofunding_latest
```

## Judge Harness Commands

Build stratified judge packets:

```bash
python scripts/qualitative_judge_harness.py build-packets \
  --results-dir experiments/results/cofunding_latest \
  --output analysis/qualitative_judge_packets.jsonl \
  --per-stratum 3
```

Aggregate judge responses:

```bash
python scripts/qualitative_judge_harness.py score \
  --responses analysis/qualitative_judge_responses.jsonl \
  --output analysis/qualitative_judge_report.json
```

Judge rubric and response schema:
- `docs/guides/QUALITATIVE_JUDGE_RUBRIC_V1.md`
