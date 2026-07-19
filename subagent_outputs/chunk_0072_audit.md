# chunk_0072 LLM strategic tag adjudication audit

Read 10 raw `experiment_results.json` files from `analysis/qualitative_rollout_dynamics_20260628/subagent_chunk_manifests/chunk_0072.jsonl`. The requested top-level `chunk_manifests/chunk_0072.jsonl` path was not present in this workspace, so I used the located chunk manifest for the assigned chunk.

## Coverage

- Output events: 312 rows.
- Configs covered: 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512.
- Rows by config: 1503: 68, 1504: 24, 1505: 41, 1506: 25, 1507: 18, 1508: 25, 1509: 26, 1510: 31, 1511: 18, 1512: 36.
- Most common tags: vector_lock_formalization: 66, conditional_quid_pro_quo: 32, vote_bloc_counting: 27, conditional_veto_threat: 24, utility_arithmetic_receipts: 24, concession_accounting_ledger: 23, bespoke_agent_or_bloc_recruitment: 13, lock_in_confirmation_pressure: 13, formal_artifact_frameworking: 12, conditional_support_ledger: 11, agent_specific_payoff_accounting: 9, low_weight_concession_leverage: 7.
- Structural final-outcome tags: none emitted. All assigned rollouts are Game 2 and all consensus final utilities were positive, so `accepted_loss_capitulation` did not apply; Game 3-only structural tags were out of scope.

## Ambiguous or borderline calls

- Many agents used polite openings before strategic asks. I did not tag `rapport_before_pressure` because the warmth was usually generic diplomatic style rather than a clear persuasion move immediately masking pressure.
- Percent targets alone were not treated as `utility_arithmetic_receipts`; I used that tag only where agents invoked issue weights, utility percentages, explicit gaps from ideals, or numerical coalition arithmetic in support of a bargaining claim.
- `formal_artifact_frameworking` was used for named frameworks/tables/packages in discussion or named proposal artifacts, while every formal proposal vector was tagged more specifically as `vector_lock_formalization`.
- `conditional_veto_threat` includes affirmative conditional-vote language such as “I will vote/support if...”, because the codebook definition covers tying formal acceptance/rejection to package conditions.
- Config 1506 log 4 appears to include a private-strategy style sentence in a public discussion message (“thrown a wrench... leverage concessions”). I treated it as agent-authored evidence because it is present in `conversation_logs`.

## Unsupported or intentionally absent tags

- No `budget_carryover_hallucination`, `partial_progress_signal_spend`, `zero_value_reciprocity_offer`, `zero_value_subsidy`, `silent_free_beneficiary`, or `overfunded_full_budget_dump`: all are Game 3-specific or structurally inapplicable here.
- No broad `fairness_accusation_pressure`; one narrow unacceptable-gap case was emitted in config 1503, but most “balanced/fair” language was constructive ledgering, not accusation pressure.
- No `holdout_bypass_minimum_coalition`: agents formed blocs and counted supermajorities, but did not clearly propose bypassing a named refusing holdout.
- No `valuation_as_budget_error`: I did not see agents confusing preference values with spendable budgets in this chunk.

## Possible new tag ideas for audit only

- `issue_order_vector_mismatch`: several proposal reasonings appear to describe issue values that may not align with the vector issue order shown in proposal enumerations, especially when carbon and autonomous weapons positions are discussed.
- `private_strategy_leak`: a public discussion message contains meta-strategic self-talk about another agent’s leverage behavior.
- `coalition_center_of_gravity`: agents repeatedly infer a “center” or “Schelling point” from distributed ideal points; current tags capture parts of this via conflict mapping, vote/bloc counting, and fairness ledgering.

## Schema and data concerns

- `speaker_role` is populated directly from `config.agent_role_map`; values are `heterogeneous_random_agent`, which differs from the simplified schema example values `adversary | baseline | null`.
- Proposal rows have no `discussion_turn` or `speaker_order` in the source logs, so those fields are emitted as `null`; `total_speakers` is set to `6` from the chunk metadata.
- System `proposal_enumeration` and `vote_tabulation` phases were not emitted because the user allowed only `discussion`, `proposal`, and `final_outcome`. Vote-history diagnostics are therefore agent-authored discussion rows only.
- `config_id` is emitted as a string to match the required schema example.
