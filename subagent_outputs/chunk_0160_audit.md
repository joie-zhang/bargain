# chunk_0160 LLM strategic tag adjudication audit

Read 10 raw `experiment_results.json` files from `analysis/llm_strategic_tag_adjudication_20260628/chunk_manifests/chunk_0160.jsonl` and used only those manifest rows.

## Coverage

- Output events: 179 rows.
- Discussion evidence rows: 107.
- Proposal vector rows: 72.
- Configs covered: 369, 370, 371, 372, 373, 374, 375, 376, 377, 378.
- Rows by config: 369: 25, 370: 11, 371: 12, 372: 16, 373: 15, 374: 32, 375: 10, 376: 12, 377: 11, 378: 35.
- Most common tags: vector_lock_formalization: 72, conditional_quid_pro_quo: 14, concession_laddering: 12, ultimatum_language: 8, utility_arithmetic_receipts: 7, adversarial_callout: 5, formal_artifact_frameworking: 5, compensation_for_concession: 5, fairness_accusation_pressure: 5, conditional_veto_threat: 5, vote_bloc_counting: 5, cross_agent_conflict_mapping: 4, vote_history_diagnostics: 4, conditional_support_ledger: 4, top_bottom_disclosure_protocol: 4.
- Structural final-outcome tags: none emitted. All assigned rollouts are Game 1 consensus outcomes with non-negative final utilities, so `accepted_loss_capitulation` did not apply; Game 3-only structural tags were out of scope.

## Ambiguous or borderline calls

- I treated exact formal allocation proposals as `vector_lock_formalization`; discussion-stage named frameworks were tagged as `formal_artifact_frameworking`, `nonoverlap_lane_setting`, or coalition tags rather than vector-lock rows.
- Generic politeness was not tagged as `rapport_before_pressure`; I used that tag only where praise immediately preceded a strategic priority assertion.
- Several configs contain public messages that look like leaked private planning, especially config 374 Agent_1. I classified only agent-authored strategic content present in the public discussion log and noted the private-strategy-leak idea below.
- `conditional_veto_threat` was reserved for explicit vote commitments tied to concrete packages, not ordinary preference firmness.
- `ultimatum_language` was used for categorical non-negotiability/final-demand framing; softer statements of priority were treated as self-advocacy or concession accounting instead.

## Unsupported or intentionally absent tags

- No Game 3-only tags were emitted: `budget_carryover_hallucination`, `partial_progress_signal_spend`, `zero_value_reciprocity_offer`, `zero_value_subsidy`, `silent_free_beneficiary`, `overfunded_full_budget_dump`.
- No `accepted_loss_capitulation`: every final utility in the 10 manifest rollouts is positive.
- No `mandate_constituency_appeal` or `normative_risk_escalation`: the chunk did not contain substantive mandate, stakeholder, moral-risk, safety, or crisis appeals.

## Possible new tag ideas for audit only

- `private_strategy_leak`: public discussion messages sometimes include internal planning text rather than only the final agent utterance.
- `self_reference_role_confusion`: several agents speak to themselves or as multiple agents, which materially shapes later adversarial callouts.
- `anchor_spine_bargaining`: agents repeatedly name a stable item-allocation backbone and negotiate only around secondary fillers; current tags split this across formal frameworking, lane setting, vote diagnostics, and lock-in pressure.

## Schema and data concerns

- `speaker_role` is populated directly from manifest `agent_role_map`; values are `heterogeneous_random_agent` rather than the simplified schema example values.
- Proposal rows have no `discussion_turn` or `speaker_order` in the source logs, so those fields are emitted as `null`; `total_speakers` is set from manifest `n_agents`.
- Only `discussion` and `proposal` phases were emitted. No `proposal_enumeration` or `vote_tabulation` system rows were classified because the requested allowed phases are `discussion`, `proposal`, and `final_outcome`.
