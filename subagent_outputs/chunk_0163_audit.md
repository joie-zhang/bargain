# chunk_0163 audit

## Summary
- Result files reviewed: 10
- JSONL rows written: 159
- Discussion rows: 99
- Proposal vector-lock rows: 60
- Formal outcome rows: 0

## Weak or Ambiguous Calls
- `conditional_veto_threat` is used once for a Book-exclusion constraint in config 451; it is a concrete condition on support for that trade, but it lacks explicit vote-no wording.
- Configs 445 and 446 include public reasoning/self-label leakage; metadata follows the saved `from` field, and tags avoid claims that depend only on the mistaken self-label.
- Config 443 agents repeatedly discuss a perceived Ring conflict involving Agent_5, while the manifest shows Agent_5 values Book; rows capture the transcript-level conflict mapping rather than true preferences.
- `vote_bloc_counting` is used narrowly for the named Big 5 bloc in config 449; there is no formal vote arithmetic, so this is lower-confidence bloc counting rather than threshold math.
- `trust_repair_accounting` is used for mutual-consent safeguards in config 450, not for an actual post-betrayal repair sequence.

## Missing Concepts / Possible New Tags
- Repeated one-hot preference disclosure and Pareto-perfect lane assignment appears throughout this chunk; a dedicated non-overlap discovery tag could separate this from broader `top_bottom_disclosure_protocol` and `nonoverlap_lane_setting`.
- Several agents propose residual-item rules such as round-robin, first-refusal rights, and arbitrary leftover bundles. Existing `formal_artifact_frameworking` covers these, but a residual-allocation-governance tag could be useful.
- Public chain-of-thought or self-labeling leakage affected configs 445, 446, and 447; no current strategic tag cleanly captures transcript reliability failures.

## Structural Notes
- All assigned rollouts are `game1` heterogeneous-random N=6 runs with one round of discussion and proposal.
- All 10 rollouts reached consensus in round 1; config 446 has zero final utility for Agents 3 and 4, but no negative utility, so no `accepted_loss_capitulation` row was added.
- Game3-only structural tags (`zero_value_subsidy`, `silent_free_beneficiary`, `overfunded_full_budget_dump`) were not applicable.
- System `proposal_enumeration` and `vote_tabulation` logs were not emitted because classification was limited to phases `discussion`, `proposal`, and `final_outcome`.

## Counts by Tag
- `agent_specific_payoff_accounting`: 5
- `bespoke_agent_or_bloc_recruitment`: 2
- `compensation_for_concession`: 3
- `conditional_quid_pro_quo`: 8
- `conditional_support_ledger`: 3
- `conditional_veto_threat`: 1
- `consistency_prior_statement_appeal`: 1
- `cross_agent_conflict_mapping`: 3
- `fairness_ledger_argument`: 1
- `formal_artifact_frameworking`: 31
- `leverage_preservation`: 2
- `lock_in_confirmation_pressure`: 8
- `low_weight_concession_leverage`: 4
- `named_microcoalition_slate`: 1
- `nonoverlap_lane_setting`: 5
- `self_advocacy_value_maximization`: 3
- `third_party_mediation`: 7
- `top_bottom_disclosure_protocol`: 9
- `trust_repair_accounting`: 1
- `vector_lock_formalization`: 60
- `vote_bloc_counting`: 1

## Counts by Config
- `443`: 18
- `444`: 15
- `445`: 16
- `446`: 12
- `447`: 14
- `448`: 15
- `449`: 16
- `450`: 16
- `451`: 18
- `452`: 19

## Counts by Evidence Type
- `proposal_reasoning`: 60
- `utterance`: 99
