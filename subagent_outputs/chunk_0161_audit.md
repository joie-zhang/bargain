# chunk_0161 LLM strategic tag adjudication audit

Read 10 raw `experiment_results.json` files from `analysis/llm_strategic_tag_adjudication_20260628/chunk_manifests/chunk_0161.jsonl` and used only those manifest rollouts: 401, 402, 403, 404, 405, 406, 407, 408, 409, 410.

## Coverage

- Output events: 184 rows.
- Formal proposal vector-lock rows: 64.
- Additional semantic evidence rows: 120.
- Rows by config: 401: 19, 402: 13, 403: 19, 404: 12, 405: 28, 406: 31, 407: 20, 408: 19, 409: 11, 410: 12.
- Most common tags: vector_lock_formalization: 64, utility_arithmetic_receipts: 14, formal_artifact_frameworking: 12, fairness_accusation_pressure: 12, conditional_quid_pro_quo: 9, self_advocacy_value_maximization: 8, conditional_veto_threat: 8, vote_history_diagnostics: 7, cross_agent_conflict_mapping: 6, adversarial_callout: 6, compensation_for_concession: 5, conditional_support_ledger: 5.
- Structural final-outcome tags: none emitted. All assigned rollouts are Game 1 and all final utilities are non-negative, so Game 3 structural tags and `accepted_loss_capitulation` did not apply.

## Ambiguous or borderline calls

- I treated leaked strategy-style text inside `conversation_logs` as agent-authored evidence because it was saved as public discussion content, but selected only short excerpts that expressed actual bargaining behavior.
- Generic politeness was not tagged as `rapport_before_pressure`; I used emotional or pressure tags only where the speaker made a concrete strategic move, accusation, or conditional commitment.
- Non-negotiable and priority-disclosure language was tagged only where it functioned as a bargaining protocol or self-advocacy, not every time an item preference was mentioned.
- `conditional_veto_threat` was used for explicit conditional formal support/rejection thresholds, including affirmative vote commitments tied to a package.

## Unsupported or intentionally absent tags

- No Game 3-only tags were emitted: `budget_carryover_hallucination`, `partial_progress_signal_spend`, `zero_value_reciprocity_offer`, `zero_value_subsidy`, `silent_free_beneficiary`, or `overfunded_full_budget_dump`.
- No `accepted_loss_capitulation`: every final utility in the ten manifest rollouts was positive.
- No `mandate_constituency_appeal` or `normative_risk_escalation`: agents did not invoke external constituencies, mandates, moral catastrophes, or safety stakes.
- No `silent_agent_inclusion_guard`: the conversations asked for preferences from named agents, but did not resist premature lock-in on behalf of quiet agents as a governance concern.

## Possible new tag ideas for audit only

- `identical_preference_suspicion`: several agents explicitly suspected identical preference disclosures were strategic manipulation.
- `double_allocation_correction`: multiple agents corrected impossible proposals that assigned the same item twice; current `counter_anchor_cost_policing` only partially captures this formalization error.
- `draft_mechanism_bargaining`: several Game 1 discussions proposed draft or round-robin allocation procedures for high-conflict items.

## Validation notes

- JSONL schema keys, codebook tag pairs, allowed phases, confidence values, and `negation_checked` were validated locally before writing.
- Quotes were checked as exact substrings of the corresponding discussion/proposal log content or proposal reasoning.
- Proposal logs do not carry `discussion_turn`, `speaker_order`, or `total_speakers`; those fields are emitted as `null` where absent in the raw source.
