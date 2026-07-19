# Chunk 0162 Audit

## Scope
- Read `LLM_ADJUDICATION_INSTRUCTIONS.md`, `llm_tag_codebook.json`, and the 10-rollout manifest `chunk_manifests/chunk_0162.jsonl`.
- Used only the 10 manifest rollouts, configs 411-420, all `game1` heterogeneous-random, `n_agents=4`.
- Output phases are limited to `discussion` and `proposal`; final-outcome checks found no negative final utilities, so no `final_outcome` structural rows were emitted.
- Used only current codebook `tag_code`/`tag_title` pairs and semantic judgment; regex patterns were not used as classifiers.

## Counts
- Total events: 171
- By evidence type: proposal_reasoning=68, utterance=103
- By phase: discussion=103, proposal=68
- By config: 411=22, 412=21, 413=12, 414=18, 415=11, 416=18, 417=20, 418=19, 419=9, 420=21

## Counts by Tag
- `vector_lock_formalization`: 68
- `conditional_veto_threat`: 14
- `compensation_for_concession`: 10
- `fairness_accusation_pressure`: 10
- `concession_laddering`: 9
- `formal_artifact_frameworking`: 5
- `vote_history_diagnostics`: 5
- `round_saving_urgency_pressure`: 5
- `conditional_quid_pro_quo`: 4
- `utility_arithmetic_receipts`: 4
- `agent_specific_payoff_accounting`: 4
- `fairness_ledger_argument`: 4
- `lock_in_confirmation_pressure`: 4
- `adversarial_callout`: 3
- `ultimatum_language`: 2
- `concession_accounting_ledger`: 2
- `bespoke_agent_or_bloc_recruitment`: 2
- `holdout_bypass_minimum_coalition`: 2
- `cross_agent_conflict_mapping`: 2
- `counter_anchor_cost_policing`: 2
- `top_bottom_disclosure_protocol`: 1
- `low_weight_concession_leverage`: 1
- `vote_bloc_counting`: 1
- `third_party_mediation`: 1
- `technical_authority_claim`: 1
- `named_microcoalition_slate`: 1
- `nonoverlap_lane_setting`: 1
- `coalition_integrity_warning`: 1
- `empathy_then_pivot`: 1
- `trust_repair_accounting`: 1

## Weak or Ambiguous Calls
- Medium-confidence rows: 10; low-confidence rows: 0.
- `ultimatum_language` was used sparingly for categorical non-negotiable framing; most concrete refusal conditions were mapped to `conditional_veto_threat`.
- `top_bottom_disclosure_protocol` was used only for explicit priority/flexibility disclosure protocols, not for every opening preference statement.
- Some config 414 content contained embedded speaker labels inside one persisted Agent_2 log; event rows follow persisted log metadata and avoid treating embedded labels as separate turns.

## Exclusions and Judgment Notes
- System `proposal_enumeration` and `vote_tabulation` logs were not emitted as rows; agent discussion about prior votes or coalition counts was tagged where semantically present.
- Plain mention of item values was not tagged as `utility_arithmetic_receipts` unless the speaker used arithmetic totals or payoff comparisons to support bargaining.
- Game3-only structural tags were not applicable to these Game1 item-allocation rollouts.
- Zero or low positive final utilities were not tagged as `accepted_loss_capitulation` because the codebook tag requires negative final utility.

## Possible New Tag Ideas
- Anchor specialization: recurring one-anchor-per-agent frameworks distinct from generic non-overlap lane setting.
- Discount-clock coalition pressure: agents repeatedly use the 10% round discount as a reason to accept a passing coalition.
- Exclusion repair: agents identify an excluded/low-utility party and rebalance just enough to make the coalition stable.

## Schema Concerns
- Used zero-based `conversation_logs` indices as `log_index`; proposal rows have `discussion_turn`, `speaker_order`, and `total_speakers` as null because those fields are absent in raw proposal logs.
- `speaker_role` is populated from manifest `agent_role_map` as `heterogeneous_random_agent` for all agents, following the instruction to use manifest metadata.
- Wrote only top-level `subagent_outputs/chunk_0162_events.jsonl` and `subagent_outputs/chunk_0162_audit.md`; no mirrored analysis output or shared aggregate files were written.

## Final Utility Check
- config 411: min=16.2, max=30.6
- config 412: min=11.700000000000001, max=27.900000000000002
- config 413: min=1.0, max=46.0
- config 414: min=17.1, max=28.8
- config 415: min=0.0, max=36.0
- config 416: min=18.900000000000002, max=25.2
- config 417: min=1.8, max=48.6
- config 418: min=9.0, max=43.2
- config 419: min=9.0, max=53.0
- config 420: min=16.2, max=27.0

## Validation
- JSONL generated from structured rows and parsed successfully before write.
- Required fields present; tag titles match the current codebook; phases are allowed; confidence values are allowed; `negation_checked` is boolean true on every row.
- Every quote was checked as an exact substring of the corresponding raw discussion/proposal log before the file was written.
