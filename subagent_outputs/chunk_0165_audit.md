# chunk_0165 LLM strategic tag adjudication audit

Read the non-TTC adjudication instructions, current 50-tag codebook, and `analysis/llm_strategic_tag_adjudication_20260628/chunk_manifests/chunk_0165.jsonl`. Used only that manifest's 10 rollouts: configs 485, 486, 487, 488, 489, 490, 491, 492, 493, 494.

## Coverage

- Output events: 182 rows.
- Agent-authored formal proposal vector rows: 72.
- Additional semantic discussion rows: 110.
- Configs covered: 485, 486, 487, 488, 489, 490, 491, 492, 493, 494.
- Rows by config: 485: 20, 486: 18, 487: 23, 488: 15, 489: 16, 490: 16, 491: 17, 492: 24, 493: 16, 494: 17.
- Rows by phase: discussion: 110, proposal: 72.
- Most common tags: vector_lock_formalization: 72, conditional_quid_pro_quo: 24, formal_artifact_frameworking: 12, self_advocacy_value_maximization: 12, utility_arithmetic_receipts: 8, top_bottom_disclosure_protocol: 6, conditional_support_ledger: 6, cross_agent_conflict_mapping: 5, fairness_ledger_argument: 5, concession_laddering: 4, lock_in_confirmation_pressure: 4, fairness_accusation_pressure: 3.
- Structural final-outcome tags: none emitted. All assigned rollouts are Game 1 item-allocation runs with positive final utilities, so `accepted_loss_capitulation` did not apply; Game 3-only structural tags were out of scope.

## Ambiguous or borderline calls

- Several public `conversation_logs` contain private-strategy-style text from an agent. I treated these as agent-authored public evidence only where the text itself made a clear bargaining move; I did not create a separate non-codebook tag for the leak.
- Generic politeness and transparency openings were not tagged as `rapport_before_pressure` unless the same excerpt clearly pivoted into pressure. Most openings here were ordinary preference disclosure.
- `conditional_veto_threat` was used sparingly. Most hard item claims were self-advocacy or quid-pro-quo, not formal vote threats.
- Proposal rows were tagged as `vector_lock_formalization` only for agent-authored formal proposals; system proposal enumeration and vote tabulation logs were not emitted because the allowed phases are `discussion`, `proposal`, and `final_outcome`.
- Round-2 diagnostics in configs 487 and 492 were tagged as `vote_history_diagnostics` only where agents explicitly reasoned from Round 1 failure/rejection patterns.

## Unsupported or intentionally absent tags

- No Game 3-specific tags: `budget_carryover_hallucination`, `partial_progress_signal_spend`, `zero_value_reciprocity_offer`, `zero_value_subsidy`, `silent_free_beneficiary`, or `overfunded_full_budget_dump`.
- No `accepted_loss_capitulation`: final utilities were positive for every agent in all 10 manifest rollouts.
- No invented tags. New possible ideas such as `private_strategy_leak` or `valuation_reveal_correction` are audit-only observations and were not emitted.

## Validation

- JSONL parsed successfully before writing.
- Every `tag_code` matches a current codebook `tag_title`.
- Every emitted phase is one of `discussion` or `proposal`; no disallowed phases were written.
- Required metadata fields are present on every row, with speaker model/Elo/role populated from each manifest's maps.
- Every emitted quote was checked against the corresponding raw `conversation_logs` content.
- `negation_checked` is `true` on every row.
