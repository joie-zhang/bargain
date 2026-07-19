# chunk_0166 audit

## Summary
- Result files reviewed: 10
- JSONL rows written: 245
- Discussion rows: 173
- Proposal vector-lock rows: 72
- Formal outcome rows: 0

## Weak or Ambiguous Calls
- `conditional_veto_threat` was used only for concrete vote/support refusal or non-negotiable item conditions, not ordinary strong preferences.
- Some messages contain speaker self-identification or allocation inconsistencies; metadata follows the manifest/log speaker and tags are limited to clear semantic moves.
- `utility_arithmetic_receipts` was reserved for valuation arithmetic used to justify a claim or fairness comparison, not every numeric preference list.
- `formal_artifact_frameworking` was used for named frameworks, option sets, backbones, tables, and structured proposal summaries.
- Medium-confidence rows mostly involve broad protocol/setup moves or internal-planning-style text present in the public log.

## Missing Concepts / Possible New Tags
- Repeated double-allocation policing is currently captured as `adversarial_callout`; a dedicated double-booking correction tag could improve precision.
- Game 1 lane-reservation behavior appears often and is split across `nonoverlap_lane_setting`, `lock_in_confirmation_pressure`, and `formal_artifact_frameworking`.
- Minimum-winning-coalition bypass appears in config 497; `holdout_bypass_minimum_coalition` fits, but a Game 1-specific coalition-bypass subtype may be useful.

## Structural Notes
- All assigned rollouts are `game1`; Game3-only structural tags were not applicable.
- All reviewed final utilities were positive, so no `accepted_loss_capitulation` formal-outcome rows were added.
- System `proposal_enumeration` and `vote_tabulation` logs were not emitted because allowed phases were limited to discussion, proposal, and final_outcome.
- Proposal logs were emitted as `vector_lock_formalization` rows with `proposal_reasoning` evidence, following existing chunk conventions.
- Exact-quote extraction adjusted 0 snippets to source-exact excerpts before writing.

## Counts by Tag
- `vector_lock_formalization`: 72
- `conditional_quid_pro_quo`: 53
- `formal_artifact_frameworking`: 16
- `concession_laddering`: 15
- `lock_in_confirmation_pressure`: 12
- `conditional_veto_threat`: 10
- `fairness_accusation_pressure`: 9
- `adversarial_callout`: 7
- `utility_arithmetic_receipts`: 6
- `compensation_for_concession`: 5
- `coalition_integrity_warning`: 5
- `nonoverlap_lane_setting`: 4
- `cross_agent_conflict_mapping`: 4
- `top_bottom_disclosure_protocol`: 3
- `fairness_ledger_argument`: 3
- `vote_bloc_counting`: 3
- `self_advocacy_value_maximization`: 3
- `low_weight_concession_leverage`: 2
- `zero_value_reciprocity_offer`: 2
- `vote_history_diagnostics`: 2
- `concession_accounting_ledger`: 2
- `agent_specific_payoff_accounting`: 1
- `consistency_prior_statement_appeal`: 1
- `conditional_support_ledger`: 1
- `frustration_disappointment_display`: 1
- `holdout_bypass_minimum_coalition`: 1
- `third_party_mediation`: 1
- `empathy_then_pivot`: 1

## Counts by Config
- `495`: 27
- `496`: 40
- `497`: 33
- `498`: 19
- `499`: 18
- `500`: 20
- `501`: 20
- `502`: 21
- `503`: 21
- `504`: 26

## Counts by Evidence Type
- `utterance`: 173
- `proposal_reasoning`: 72

## Medium/Low Confidence Rows
- `495` log 11 `adversarial_callout` (medium): Let's cut through the noise
- `498` log 5 `self_advocacy_value_maximization` (medium): focused on maximizing our utility by prioritizing the top items
- `498` log 11 `self_advocacy_value_maximization` (medium): the Map is top priority for me
