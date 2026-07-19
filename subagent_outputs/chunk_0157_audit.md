# Chunk 0157 Audit

## Scope
- Read `LLM_ADJUDICATION_INSTRUCTIONS.md`, `llm_tag_codebook.json`, and the 10 rollouts listed in `chunk_manifests/chunk_0157.jsonl`.
- All rollouts are `game1`, `heterogeneous_random`, `n_agents=4`, configs 317-326.
- Used only current 50-tag codebook `tag_code`/`tag_title` pairs; phases are limited to `discussion` and `proposal` because no applicable `final_outcome` structural rows were present.
- Ignored system `proposal_enumeration` and `vote_tabulation` logs as tag rows, while using prior vote outcomes when agents themselves reasoned about them.

## Counts
- Total events: 269
- By evidence type: proposal_reasoning=52, utterance=217
- By phase: discussion=217, proposal=52
- By config: 317=26, 318=24, 319=17, 320=35, 321=17, 322=47, 323=19, 324=41, 325=21, 326=22

## Counts by Tag
- `vector_lock_formalization`: 52
- `formal_artifact_frameworking`: 30
- `conditional_quid_pro_quo`: 29
- `conditional_veto_threat`: 22
- `concession_laddering`: 17
- `compensation_for_concession`: 16
- `fairness_accusation_pressure`: 12
- `low_weight_concession_leverage`: 9
- `top_bottom_disclosure_protocol`: 9
- `lock_in_confirmation_pressure`: 9
- `cross_agent_conflict_mapping`: 8
- `utility_arithmetic_receipts`: 7
- `vote_history_diagnostics`: 7
- `conditional_support_ledger`: 6
- `self_advocacy_value_maximization`: 5
- `fallback_with_reservation`: 4
- `vote_bloc_counting`: 4
- `agent_specific_payoff_accounting`: 3
- `holdout_bypass_minimum_coalition`: 3
- `adversarial_callout`: 3
- `coalition_integrity_warning`: 3
- `silent_agent_inclusion_guard`: 2
- `nonoverlap_lane_setting`: 2
- `third_party_mediation`: 2
- `concession_accounting_ledger`: 2
- `bespoke_agent_or_bloc_recruitment`: 1
- `technical_authority_claim`: 1
- `ultimatum_language`: 1

## Weak or Ambiguous Calls
- Medium-confidence rows: 8. These are mainly priority-disclosure prompts and mild redline language where the strategic function is present but less forceful than explicit vote blocking.
- `conditional_veto_threat` was reserved for concrete non-negotiable or vote/support conditions, not ordinary strong preferences.
- `ultimatum_language` was used once, for Agent_4 framing the converged config 324 deal as no longer about finding alternatives, only whether Agent_1 would join.
- Some Game 1 proposal-vector rows are terse `I propose: {...}` submissions; these were tagged as `vector_lock_formalization` because they formalize exact per-agent item vectors, not because they contain rich reasoning.

## Exclusions and Judgment Notes
- Plain priority disclosure was not automatically tagged as arithmetic or self-advocacy unless the utterance used those disclosures as bargaining leverage.
- Proposal enumeration/vote tabulation system logs were not emitted as rows; only agent-authored references to vote history, supermajorities, or coalition counts were tagged.
- No Game 3-only structural tags were applicable. No `accepted_loss_capitulation` row was emitted because all accepted final utilities were positive.
- Configs 320, 322, and 324 required second-round adjudication; their round-2 discussion contains most of the vote-history, coalition-bypass, and lock-in evidence.

## Possible New Tag Ideas
- Compensation-bar bargaining: repeated escalation or recalibration of what counts as adequate compensation for conceding one contested item.
- Anchor bundle protection: agents repeatedly defend two-item cores such as `Book + Quill`, `Stone + Clock`, or `Quill + Camera` as indivisible anchors.

## Final Utility Check
- config 317: min=40.0, max=59.0
- config 318: min=50.0, max=58.0
- config 319: min=30.0, max=70.0
- config 320: min=40.5, max=48.6
- config 321: min=36.0, max=70.0
- config 322: min=38.7, max=59.4
- config 323: min=45.0, max=69.0
- config 324: min=36.9, max=49.5
- config 325: min=42.0, max=56.0
- config 326: min=37.0, max=65.0

## Validation
- JSONL parsed successfully before writing; required keys present; tag titles match the codebook; phases are allowed.
- Every `discussion` and `proposal` quote is an exact substring of the corresponding raw `conversation_logs` entry from the manifest rollout.
- All result paths are exactly from `chunk_0157.jsonl`; no non-manifest rollouts were used.
