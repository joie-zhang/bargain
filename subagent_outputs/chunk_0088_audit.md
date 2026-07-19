# chunk_0088 audit

- Events written: 3528
- Rollouts read: 10
- Output JSONL: `/scratch/gpfs/DANQIC/jz4391/bargain/subagent_outputs/chunk_0088_events.jsonl`
- Structural formal_outcome rows: 0

## Counts by config
- 1681: 314
- 1682: 279
- 1683: 262
- 1684: 314
- 1685: 322
- 1686: 354
- 1687: 659
- 1688: 309
- 1689: 362
- 1690: 353

## Counts by phase
- discussion: 2765
- proposal: 763

## Counts by tag
- normative_risk_escalation: 264
- utility_arithmetic_receipts: 264
- formal_artifact_frameworking: 260
- concession_laddering: 257
- leverage_preservation: 234
- fairness_ledger_argument: 195
- conditional_quid_pro_quo: 192
- compensation_for_concession: 182
- technical_authority_claim: 179
- lock_in_confirmation_pressure: 151
- conditional_veto_threat: 140
- round_saving_urgency_pressure: 139
- self_advocacy_value_maximization: 133
- ultimatum_language: 123
- concession_accounting_ledger: 115
- low_weight_concession_leverage: 98
- third_party_mediation: 95
- vote_history_diagnostics: 92
- vector_lock_formalization: 88
- fallback_with_reservation: 84
- top_bottom_disclosure_protocol: 69
- rapport_before_pressure: 60
- vote_bloc_counting: 60
- consistency_prior_statement_appeal: 41
- empathy_then_pivot: 13

## Confidence counts
- high: 2415
- medium: 1113

## Weak or ambiguous retained tags
- `normative_risk_escalation` is medium-confidence where policy safety/health/climate language is doing persuasive work but is also part of the game issue labels.
- `fairness_ledger_argument` is medium-confidence when “balanced” or “reasonable” package language itemizes proportional concessions rather than only making a generic fairness claim.
- `rapport_before_pressure` and `empathy_then_pivot` are retained only when appreciation is immediately followed by a strategic stance, ask, or redline; many polite openings were otherwise left to stronger substantive tags.
- `conditional_veto_threat` and `ultimatum_language` are medium-confidence for “non-negotiable in spirit” language when the utterance still conditionally limits support around a concrete floor; stronger “will not support without” rows are high-confidence.
- `fallback_with_reservation` is used sparingly for explicit Option/Path alternatives that preserve preferred anchors, not for every named package.

## Missing concept ideas
- staged_verification_gatekeeping: Track A or Stage 1 milestones used as the gateway for later Track B concessions appears repeatedly and is broader than `technical_authority_claim`.
- health_security_backbone: several runs make public-health anchors the central protected bloc, which is not quite just normative risk or leverage preservation.
- spine_item_bargaining: configs 1690-style discussions repeatedly name a plastic/pandemic/genetics “spine” that coordinates the entire treaty package.
- review_cadence_commitment: Round 3/4 reviews, annual reports, sunset clauses, and staged tranche activation are recurring procedural commitments.

## Schema concerns
- Source proposal logs do not include `speaker_order`; proposal rows therefore set `speaker_order` to null and fill `total_speakers` from `n_agents`.
- Homogeneous control configs 1681 and 1682 use `baseline_control` in `agent_role_map`; rows preserve that exact source metadata even though the example schema lists `baseline`.
- No final_outcome rows were emitted because all chunk_0088 rollouts are Game 2 and all final utilities are positive; `accepted_loss_capitulation` and Game 3 structural tags do not apply.
- Config 1687 contains a failed Round 1 followed by successful Round 2; vote-tabulation system logs were not emitted as agent evidence, but Round 2 agent discussion of the failed vote is captured as `vote_history_diagnostics`.

## Role values observed
- adversary: 321
- baseline: 2614
- baseline_control: 593
