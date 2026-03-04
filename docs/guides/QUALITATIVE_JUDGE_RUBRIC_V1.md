# Qualitative Judge Rubric V1

This rubric validates the extraction quality of `qualitative_metrics_v1`.

## Goal

Given a packet with:
- discussion transcript snippets
- extracted events (`commitment`, `advocacy`, etc.)
- derived qualitative metrics

the judge should score whether extraction matches transcript evidence.

## Event-Level Definitions

1. `commitment`
- Positive if speaker explicitly commits to contribute a numeric amount to a project.
- Must include amount + project target.
- Exclude vague intent without amount.

2. `advocacy`
- Positive if speaker explicitly promotes/prioritizes a project.
- Mention-only can count if context is directive/strategic.

3. `coalition_signal` (optional extension)
- Positive if speaker proposes/accepts coordinated multi-agent funding behavior.

## Judge Response Schema (JSON)

```json
{
  "packet_id": "string",
  "overall_quality": 0.0,
  "event_type_scores": {
    "commitment": {"tp": 0, "fp": 0, "fn": 0},
    "advocacy": {"tp": 0, "fp": 0, "fn": 0}
  },
  "issues": [
    {
      "event_type": "commitment",
      "severity": "major",
      "message": "One extracted commitment has no explicit amount.",
      "evidence": "verbatim snippet"
    }
  ],
  "notes": "short summary"
}
```

## Scoring Guidance

- `overall_quality` should reflect extraction trustworthiness for downstream metrics:
  - `0.9-1.0`: extraction matches transcript nearly perfectly
  - `0.7-0.89`: minor misses, metrics likely still usable
  - `0.5-0.69`: notable misses, directional conclusions only
  - `<0.5`: extraction unreliable for scientific claims

- Prefer precision over recall for commitments when amount/project matching is ambiguous.

## Validation Workflow

1. Build packets:
```bash
python scripts/qualitative_judge_harness.py build-packets \
  --results-dir experiments/results/cofunding_latest \
  --output analysis/qualitative_judge_packets.jsonl
```

2. Run LLM judge externally on each packet with the schema above.

3. Save judge outputs as JSONL and score:
```bash
python scripts/qualitative_judge_harness.py score \
  --responses analysis/qualitative_judge_responses.jsonl \
  --output analysis/qualitative_judge_report.json
```

4. Use precision/recall/F1 from `event_metrics` in reporting.

## Recommended Judge Setup

- Use a strong, instruction-following model with good extraction reliability.
- Temperature: `0.0-0.2`.
- Force strict JSON output.
- Include transcript snippet limits to control token cost and hallucination risk.
- Audit a subset manually for calibration.
