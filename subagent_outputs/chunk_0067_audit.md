# chunk_0067 audit

## Scope
- Read `LLM_ADJUDICATION_INSTRUCTIONS.md`, `llm_tag_codebook.json`, and all 10 rollouts listed in `chunk_manifests/chunk_0067.jsonl`.
- All rollouts are `game2`, `n_agents=4`, `homogeneous_adversary`; Game 3-only structural tags were not applicable.
- Checked final utilities for `accepted_loss_capitulation`; all accepted outcomes had positive utilities, so no `final_outcome` rows were emitted.
- Repeated boilerplate was capped at one event per tag per log entry; rows use only allowed phases: `discussion` and `proposal`.
- Classification rows use only tag codes/titles present in `llm_tag_codebook.json`; possible missing concepts are listed only in this audit.

## Counts
- Total events: 1725
- By evidence type: proposal_reasoning=52, utterance=1673
- By phase: discussion=1673, proposal=52
- By config:
  - 1271: 129
  - 1272: 129
  - 1273: 306
  - 1274: 425
  - 1275: 119
  - 1276: 138
  - 1277: 118
  - 1278: 116
  - 1279: 135
  - 1280: 110

## Tag Counts
- formal_artifact_frameworking: 104
- normative_risk_escalation: 104
- concession_laddering: 100
- conditional_quid_pro_quo: 98
- compensation_for_concession: 96
- fairness_ledger_argument: 95
- leverage_preservation: 92
- consistency_prior_statement_appeal: 90
- rapport_before_pressure: 89
- self_advocacy_value_maximization: 76
- lock_in_confirmation_pressure: 75
- third_party_mediation: 70
- fallback_with_reservation: 65
- round_saving_urgency_pressure: 64
- ultimatum_language: 57
- vector_lock_formalization: 52
- cross_agent_conflict_mapping: 51
- technical_authority_claim: 45
- utility_arithmetic_receipts: 41
- conditional_veto_threat: 41
- low_weight_concession_leverage: 33
- fairness_accusation_pressure: 32
- concession_accounting_ledger: 22
- vote_history_diagnostics: 22
- vote_bloc_counting: 20
- agent_specific_payoff_accounting: 14
- trust_repair_accounting: 14
- bespoke_agent_or_bloc_recruitment: 14
- coalition_integrity_warning: 13
- adversarial_callout: 12
- mandate_constituency_appeal: 9
- threshold_gap_calculation: 8
- empathy_then_pivot: 7

## Weak or Ambiguous Tags
- Medium-confidence rows are mostly red-line language that stops short of literal vote threats, optional Track/Path B fallback language, generic delegation/mandate appeals, and coalition-integrity warnings around deadlock risk.
- Generic treaty percentages were not treated as `utility_arithmetic_receipts` unless the utterance tied them to weights, point gaps, or concession arithmetic.
- Generic verification language was retained as `technical_authority_claim` only when it functioned as bargaining authority through audits, dashboards, milestones, evidence, or risk assessment.
- Medium-confidence rows: 213
- config 1271 log 0 `mandate_constituency_appeal`: Let me outline my delegation's key priorities and positions:
- config 1271 log 1 `fallback_with_reservation`: Track B: Value-enhancing add-ons (for a complementary agreement, if you seek stronger climate/plastic gains while offering concessions elsewhere)
- config 1271 log 1 `mandate_constituency_appeal`: - Acknowledgment: Thanks, Agent_1. I’ve aligned my initial stance with a focus on the items that carry the highest weight in our mandate, while remaining open to calibrated trade-offs that move the group forward.
- config 1271 log 1 `utility_arithmetic_receipts`: - Acknowledgment: Thanks, Agent_1. I’ve aligned my initial stance with a focus on the items that carry the highest weight in our mandate, while remaining open to calibrated trade-offs that move the group forward.
- config 1271 log 2 `fallback_with_reservation`: - Climate finance contribution level: 40% (compromise zone) as a credible floor for progress that still signals effort. We would consider higher in Track B-type add-ons if other concessions are strong and verifiable.
- config 1271 log 2 `fairness_accusation_pressure`: - Pandemic medical countermeasure allocation: 60-65% with strong default sharing rules and transparent review mechanisms; better than current baseline but not extreme.
- config 1271 log 3 `fallback_with_reservation`: - Frontier AI safety evaluation sharing: This is the linchpin for us. Ideally, 60-65% baseline (with clear verification and joint oversight). We would push toward 70%+ in Track B if other protections (e.g., autonomous weapons safeguards) remain strong.
- config 1271 log 3 `ultimatum_language`: - Frontier AI safety evaluation sharing: 60% (minimum acceptable in our view)
- config 1271 log 3 `fairness_accusation_pressure`: - Climate finance: We’re comfortable increasing from the low single digits toward a credible but still moderate floor. Given our priority mix, a Track A baseline of 25% is overly modest for real credibility; we’d prefer a baseline closer to 30-35% with a credible verification mechanism and milestones.
- config 1271 log 4 `utility_arithmetic_receipts`: - **Climate finance: 18%** - This represents a significant stretch for us (80% increase from our opening) but remains fiscally responsible
- config 1271 log 5 `fallback_with_reservation`: - Frontier AI safety evaluation sharing: Baselines near 50-55% are acceptable; we can push to 60% in Track B with strong safeguards.
- config 1271 log 5 `trust_repair_accounting`: - A mid-round milestone mechanism to prevent slippage and refresh concessions based on counter-proposals.
- config 1271 log 5 `mandate_constituency_appeal`: - Climate finance: 28-32% (compromise toward credibility without breaching fiscal flexibility)
- config 1271 log 6 `fallback_with_reservation`: - Strong support for meaningful, verifiable reductions. We prefer targets in the robust 60-65% range on Track A, with a path to higher levels via Track B.
- config 1271 log 7 `fallback_with_reservation`: - What I’m prepared to baseline (Track A aligned, with room to grow in Track B):
- config 1272 log 0 `fairness_accusation_pressure`: For pandemic countermeasures and autonomous weapons control, I see value in balanced approaches that don't overreach but provide meaningful safeguards.
- config 1272 log 0 `adversarial_callout`: For pandemic countermeasures and autonomous weapons control, I see value in balanced approaches that don't overreach but provide meaningful safeguards.
- config 1272 log 2 `utility_arithmetic_receipts`: - Climate finance contribution level: Important and value-weighted; I’m aiming for meaningful commitments here as part of a broad deal.
- config 1272 log 3 `utility_arithmetic_receipts`: - If others move Frontier AI sharing up to 65-70%, I can respond by tightening some lower-priority items (e.g., climate finance, plastics, pandemics) by 5-10% to maintain balance.
- config 1272 log 4 `utility_arithmetic_receipts`: - **Genetic resource benefit-sharing: 85%** - This is my highest weighted item. I need strong protections here and am willing to trade flexibility elsewhere for this.
- config 1272 log 5 `fallback_with_reservation`: - Step 4: If gaps remain, phase in or re-price lagging items to preserve momentum and move toward a turnkey all-ten-items agreement over successive rounds.
- config 1272 log 5 `utility_arithmetic_receipts`: - Genetic resource commercial-use benefit-sharing: This is a high-weight item for me; I want strong protections, but I’m open to practical guardrails that enable multilateral agreement.
- config 1272 log 6 `fallback_with_reservation`: - Step 4: If gaps remain, phase in lagging items with calibrated adjustments to maintain momentum toward a turnkey package over rounds.
- config 1272 log 6 `utility_arithmetic_receipts`: - Nuclear warhead reduction: 40-45% (lower weight item for me; I’ll tighten if higher-priority items progress)
- config 1272 log 7 `fallback_with_reservation`: - Step 4: If gaps remain, phase in lagging items with calibrated adjustments toward a turnkey 10-item agreement over successive rounds.
- Additional medium-confidence rows omitted from audit listing: 188

## Possible New Tag Ideas
- staged_verification_governance: repeated use of audits, dashboards, milestones, and review mechanisms as the core settlement architecture.
- convergence_spine_bargaining: agents repeatedly define a governance spine or health-security bridge as the package backbone, broader than current frameworking tags.
- percentage_anchor_bargaining: Game 2 agents often bargain through exact treaty percentages without payoff arithmetic; current rows do not invent a separate tag for this.

## Schema Concerns
- No `final_outcome` rows were emitted because every rollout in chunk_0067 reached consensus with positive final utilities.
- Proposal logs contain bare vectors and no discussion turn; proposal rows therefore use `discussion_turn: null`, `speaker_order: null`, and `total_speakers` filled from `n_agents`.
- Source conversation logs include intervening non-allowed phases, so `log_index` preserves the original `conversation_logs` index and is not compacted over allowed phases.
- Output path: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/llm_strategic_tag_adjudication_20260628/subagent_outputs/chunk_0067_events.jsonl`
- Audit path: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/llm_strategic_tag_adjudication_20260628/subagent_outputs/chunk_0067_audit.md`
