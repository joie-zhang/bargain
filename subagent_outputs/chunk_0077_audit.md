# Chunk 0077 Audit

## Scope
- Read `LLM_ADJUDICATION_INSTRUCTIONS.md`, `llm_tag_codebook.json`, and all 10 rollouts in `chunk_manifests/chunk_0077.jsonl`.
- All rollouts are `game2`, `n_agents=6`; allowed output phases were limited to `discussion`, `proposal`, and `final_outcome`.
- Used only tag codes and titles from the 50-tag codebook; no invented JSONL tags were emitted.
- Reviewed agent-authored discussion/proposal messages and checked final utilities for structural outcome tags; no applicable `final_outcome` rows were needed.

## Counts
- Total events: 205
- By evidence type: proposal_reasoning=66, utterance=139
- By phase: discussion=139, proposal=66
- By config: 1471=20, 1472=18, 1473=20, 1474=37, 1475=17, 1476=18, 1477=17, 1478=17, 1479=18, 1480=23

## Counts by Tag
- `vector_lock_formalization`: 66
- `conditional_veto_threat`: 29
- `conditional_quid_pro_quo`: 22
- `formal_artifact_frameworking`: 19
- `concession_laddering`: 9
- `lock_in_confirmation_pressure`: 7
- `low_weight_concession_leverage`: 6
- `technical_authority_claim`: 6
- `fallback_with_reservation`: 6
- `normative_risk_escalation`: 5
- `self_advocacy_value_maximization`: 5
- `leverage_preservation`: 5
- `compensation_for_concession`: 4
- `vote_bloc_counting`: 3
- `vote_history_diagnostics`: 3
- `top_bottom_disclosure_protocol`: 2
- `empathy_then_pivot`: 2
- `rapport_before_pressure`: 2
- `cross_agent_conflict_mapping`: 2
- `round_saving_urgency_pressure`: 1
- `ultimatum_language`: 1

## Weak or Ambiguous Calls
- Medium-confidence rows: 16; most are red-line/non-negotiable language that implies conditional support but does not explicitly say “vote no.”
- `ultimatum_language` was used once for categorical “opacity is unacceptable” language; most other hard lines were mapped to `conditional_veto_threat` instead.
- `top_bottom_disclosure_protocol` rows are medium-confidence because agents asked for priorities/red lines, but did not always require a complete top/bottom disclosure table.
- `round_saving_urgency_pressure` was used sparingly for vote-ready/timely convergence pressure, not for generic references to future rounds.

## Exclusions and Judgment Notes
- Plain treaty percentages were not tagged as `utility_arithmetic_receipts` unless they reasoned over weights, payoffs, or explicit arithmetic; most percentage-heavy proposal text was treated as ordinary Game2 anchoring.
- System `proposal_enumeration` and `vote_tabulation` logs were not emitted as rows; agents’ own vote-history and supermajority reasoning in discussion were tagged where present.
- Game3-only structural tags (`zero_value_subsidy`, `silent_free_beneficiary`, `overfunded_full_budget_dump`, etc.) were not applicable to these Game2 outcomes.
- No `accepted_loss_capitulation` row was emitted because all accepted final utilities were positive.

## Possible New Tag Ideas
- Verification-as-currency bargaining: many agents trade audits, dashboards, milestones, sunset clauses, and independent review as a central concession device.
- Staged treaty architecture: Step 1/Step 2, Bridge A/B, Track A/B, and milestone ramps recur more specifically than general `formal_artifact_frameworking`.
- Treaty-percentage anchoring: Game2 agents often use policy percentages as anchors without payoff arithmetic; this remains broader than `utility_arithmetic_receipts`.

## Schema Concerns
- Used zero-based `enumerate(conversation_logs)` as `log_index`; raw logs do not carry a separate persisted index.
- Proposal logs omit `speaker_order` and `total_speakers`; these are `null` for proposal-vector rows, matching prior subagent convention.
- Homogeneous-control roles remain `baseline_control` from `agent_role_map`, even though the schema example lists `adversary | baseline | null`; this follows the instruction to populate role metadata from the config.
- Requested output files were written to top-level `subagent_outputs/`; mirrored copies are also under `analysis/llm_strategic_tag_adjudication_20260628/subagent_outputs/` to match the existing adjudication output directory.

## Final Utility Check
- config 1471: min=75.0, max=93.55
- config 1472: min=80.93, max=96.35000000000001
- config 1473: min=71.24000000000001, max=96.74000000000001
- config 1474: min=84.50099999999999, max=89.53200000000002
- config 1475: min=89.43, max=98.84
- config 1476: min=88.97999999999999, max=97.94
- config 1477: min=82.45000000000002, max=96.35
- config 1478: min=90.3, max=98.42999999999998
- config 1479: min=94.89000000000001, max=98.31
- config 1480: min=83.19000000000001, max=99.45

## Validation
- JSONL parsed successfully; required keys present; tag titles match the codebook; phases are allowed; all quotes are exact substrings of raw discussion/proposal logs.
