# chunk_0164 LLM strategic tag adjudication audit

Read 10 raw `experiment_results.json` files from `analysis/llm_strategic_tag_adjudication_20260628/chunk_manifests/chunk_0164.jsonl` and used the paired LLM adjudication instructions/codebook in `analysis/llm_strategic_tag_adjudication_20260628/`. No rollouts outside that manifest were used.

## Coverage

- Output events: 137 rows.
- Discussion rows: 77; proposal rows: 60; final-outcome rows: 0.
- Configs covered: 453, 454, 455, 456, 457, 458, 459, 460, 461, 462.
- Rows by config: 453: 16, 454: 15, 455: 13, 456: 12, 457: 13, 458: 12, 459: 14, 460: 14, 461: 14, 462: 14.
- Most common tags: vector_lock_formalization: 60, conditional_support_ledger: 14, lock_in_confirmation_pressure: 9, formal_artifact_frameworking: 9, cross_agent_conflict_mapping: 5, nonoverlap_lane_setting: 4, fairness_ledger_argument: 4, silent_agent_inclusion_guard: 3, concession_laddering: 3, agent_specific_payoff_accounting: 3, counter_anchor_cost_policing: 3, utility_arithmetic_receipts: 3.
- Structural final-outcome tags: none emitted. All rollouts are Game 1; no Game 3-only structural tags apply, and no rollout ended with negative utility for `accepted_loss_capitulation`.

## Ambiguous or borderline calls

- I treated formal proposal logs as `vector_lock_formalization` even when the source proposal had been schema-repaired, because the event visible in `conversation_logs` is still an exact per-agent allocation vector. Confidence is medium for rows whose reasoning is only `valid schema repair`.
- I did not tag generic politeness or praise as `rapport_before_pressure`; the warmth generally functioned as cooperative style rather than a clear pressure tactic before disagreement.
- I used `self_advocacy_value_maximization` for config 458 where Agent_6 made flexibility on leftovers contingent on securing Lantern; I did not treat it as a formal veto threat because the agent did not explicitly threaten a reject/vote-no action.
- Config 460 contains the richest contested dynamic: Agent_6 resolves Agent_1's secondary Book interest, Agent_1 concedes Book, Agent_4 proposes a multi-item trade, and others compensate flexible Agent_4 with unclaimed items.
- I used `low_weight_concession_leverage` rather than `zero_value_reciprocity_offer` for config 460 Agent_2's offer to give away zero-value leftovers “for absolutely nothing,” because it functioned as bargaining grease rather than reciprocal support.

## Proposal/allocation concerns

- Config 455 reached consensus on an allocation that gave only Agent_2 positive utility, despite the discussion converging on six distinct 100-point items. The codebook has no zero-utility capitulation tag, and `accepted_loss_capitulation` is defined for negative final utility only, so I left this as an audit note.
- Several proposal reasonings do not match their formal allocations, especially config 454 Agent_6, config 455 Agents 2/4/6, config 456 Agent_5's Lantern index, config 457 Agents 3/4/5 in varying ways, and config 462 Agent_6. These were not converted into invented tags.
- `valid schema repair` proposal rows occur in configs 453, 455, 458, and 459. They are included only for exact vector formalization, not for strategic reasoning tags.

## Unsupported or intentionally absent tags

- No `ultimatum_language`, `fairness_accusation_pressure`, `frustration_disappointment_display`, `trust_repair_accounting`, `mandate_constituency_appeal`, `overfunding_waste_policing`, `valuation_as_budget_error`, `budget_carryover_hallucination`, or Game 3 structural tags were supported by the chunk.
- I did not treat every “100 points / zero value” disclosure as `utility_arithmetic_receipts`; I used that tag only where arithmetic was part of the bargaining justification.
- I did not emit system vote-tabulation rows because the allowed phase set is discussion/proposal/final_outcome and the non-structural vote output is not agent-authored evidence.

## Possible new tag ideas for audit only

- `proposal_vector_mismatch`: proposal reasoning/allocation mismatches are common in configs 454, 455, 457, and 462.
- `zero_payoff_consensus_failure`: config 455 has consensus but most agents receive 0 utility, which is not covered by the current negative-utility `accepted_loss_capitulation` structural tag.
- `unclaimed_filler_assignment`: many Game 1 proposals strategically dump zero-value leftovers onto one flexible agent or distribute them for symmetry.
