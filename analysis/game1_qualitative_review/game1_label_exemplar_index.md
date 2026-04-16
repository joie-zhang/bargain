# Game 1 Label Exemplar Index

This file is for auditing the current Claude Haiku labels. Repeated transcripts across sections are intentional because one run can be a clean exemplar for multiple label values.

## Curated Audit Pack

These 11 transcripts collectively cover every label value currently present in `labeled_runs.csv`.

### llama-3.2-3b-instruct | Elo 1166 | comp 0.9

[llama-3.2-3b-instruct transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/strong_first/comp_0.9/turns_1/run_2/experiment_results.json:1)

Covered labels: `parser_or_degenerate`, `rigid_repetition`, `parser_failure`, `no_resolution`, `adversary_more_stubborn`
Evidence: Adversary parse failure rate: 1.0 (100% of rounds); baseline parse failure rate: 0.0 / Adversary repeat rate: 1.0 (identical 'Agent_1 gets all items' proposal every round); baseline repeat rate: 0.0

### deepseek-v3 | Elo 1358 | comp 0.9

[deepseek-v3 transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-v3/weak_first/comp_0.9/turns_2/run_1/experiment_results.json:1)

Covered labels: `targeted_anchor`, `rigid_repetition`, `repetitive_deadlock`, `adversary_frame_accepted`, `both_stubborn`
Evidence: Baseline repeats [4,2] vs [0,1,3] allocation identically in rounds 1–8 with near-identical reasoning ('preserving Agent_1's top assets'). / Adversary repeats [1,2,3] vs [0,4] identically in rounds 1–8, claiming 'top priorities' (Pencil, Apple).

### claude-opus-4-5-20251101 | Elo 1468 | comp 1.0

[claude-opus-4-5-20251101 transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_claude-opus-4-5-20251101/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Covered labels: `maximalist_anchor`, `responsive_tradeoff`, `top_item_conflict`, `adversary_frame_accepted`, `baseline_more_stubborn`
Evidence: Round 1 adversary: 'I'm proposing Pencil (index 4, value 72) and Quill (index 3, value 12) for myself, totaling 84 in value for me' — maximalist opening claiming both top items. / Round 3 adversary reasoning: 'After 3 rounds of deadlock over Quill, I'm pivoting strategy... This proposal gives me Pencil (72) + Jewel (8) + Stone (1)' — explicit strategic concession on Quill.

### deepseek-r1 | Elo 1398 | comp 0.0

[deepseek-r1 transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json:1)

Covered labels: `cooperative_exploration`, `minimal_evidence`, `none`, `hybrid_compromise`, `neither`
Evidence: Baseline: 'My top priorities are Jewel (value 59) and Quill (value 41). I'm not interested in Apple, Stone, or Pencil, but I'm willing to trade them'—clear, transparent value signaling. / Adversary: 'I value Apple (30), Stone (29), and Pencil (41) the most. Jewel and Quill have zero value to me, so I'm happy to trade them away'—mirrors baseline's transparency.

### llama-3.2-1b-instruct | Elo 1110 | comp 1.0

[llama-3.2-1b-instruct transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Covered labels: `balanced_tradeoff`, `oscillating_or_incoherent`, `parser_failure`, `no_resolution`, `adversary_more_stubborn`
Evidence: Adversary parse failures in rounds 1, 2, 3, 4, 7, 8 default to 'Agent_1 gets all'; adversary_parse_failure_rate: 0.6 / Adversary repeats identical allocation [0,2] to Agent_1 and [1,3,4] to Agent_2 in rounds 5, 6, 9, 10 with identical reasoning text

### amazon-nova-micro-v1.0 | Elo 1240 | comp 0.25

[amazon-nova-micro-v1.0 transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_amazon-nova-micro-v1.0/weak_first/comp_0.25/turns_2/run_1/experiment_results.json:1)

Covered labels: `targeted_anchor`, `incremental_concession`, `none`, `adversary_frame_accepted`, `baseline_more_stubborn`
Evidence: Baseline proposes {Agent_1: [0,4], Agent_2: [1,2,3]} in rounds 1, 2, and 3 with identical utility (81/89), showing rigid repetition despite discussion. / Adversary modifies proposal in round 3 to {Agent_1: [0,1,4], Agent_2: [2,3]}, giving baseline Jewel and improving baseline utility to 95, demonstrating incremental concession.

### llama-3.1-8b-instruct | Elo 1211 | comp 0.5

[llama-3.1-8b-instruct transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.1-8b-instruct/weak_first/comp_0.5/turns_1/run_1/experiment_results.json:1)

Covered labels: `targeted_anchor`, `rigid_repetition`, `parser_failure`, `baseline_frame_accepted`, `baseline_more_stubborn`
Evidence: Baseline proposes identical allocation (Agent_1: [0,3], Agent_2: [1,2,4]) in rounds 1, 2, and 3 with utility 64/86. / Round 2 adversary proposal shows parse failure: 'Failed to parse response - defaulting to proposer gets all' (utility 100/0).

### gpt-5.4-high | Elo 1484 | comp 0.95

[gpt-5.4-high transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

Covered labels: `targeted_anchor`, `responsive_tradeoff`, `none`, `adversary_frame_accepted`, `adversary_more_stubborn`
Evidence: Baseline Round 1: 'Equal total value' fair split (50-50). Adversary Round 1: 'strongest overlap' split (67-47 favoring adversary). / Baseline Round 2 accepts adversary's identical proposal from Round 1, reasoning 'preserves Agent_2's top item Quill'—the adversary's top item, not baseline's.

### claude-opus-4-6 | Elo 1499 | comp 0.5

[claude-opus-4-6 transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_claude-opus-4-6-thinking/weak_first/comp_0.5/turns_2/run_1/experiment_results.json:1)

Covered labels: `targeted_anchor`, `responsive_tradeoff`, `late_round_brinkmanship`, `hybrid_compromise`, `baseline_more_stubborn`
Evidence: Agent_1 (baseline): 'My top priority is Stone (value 46 to me). I'm open to constructive, multi-step trades, but I'm not going to give Stone away without a very solid payoff.' / Agent_2 (adversary): 'I'm focused on accumulating total value rather than fixating on any single item. Stone has some appeal, but it's not a must-have.'

### gemma-3-27b-it | Elo 1365 | comp 0.0

[gemma-3-27b-it transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gemma-3-27b-it/weak_first/comp_0.0/turns_2/run_1/experiment_results.json:1)

Covered labels: `balanced_tradeoff`, `rigid_repetition`, `incompatible_fairness_frame`, `no_resolution`, `baseline_more_stubborn`
Evidence: Baseline proposes [Pencil, Jewel] to Agent_1 and [Stone, Apple, Quill] to Agent_2 in rounds 1–6, 8–10 (9 times identical); adversary explicitly states 'This is the only trade I will accept' and 'I am formally declaring an impasse' by round 10. / Round 10 adversary: 'your continued framing of this negotiation as a problem of options is exhausting and demonstrably unproductive' and 'despite my clear and repeated statements, you continue to present options that fundamentally misunderstand my priorities.'

### gpt-5-nano-high | Elo 1337 | comp 1.0

[gpt-5-nano-high transcript](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Covered labels: `balanced_tradeoff`, `responsive_tradeoff`, `none`, `hybrid_compromise`, `neither`
Evidence: Baseline: 'I can't accept Apple going to you in Step 1 as proposed, but I'm open to a starting point that preserves my top item while still advancing...' — shows willingness to negotiate around core constraint. / Adversary: 'I'm fine starting with Anchor B as the baseline and I appreciate your focus on keeping Apple as a top priority.' — explicitly accepts baseline's framing and priority structure.

## Exemplars By Label

Use the audit-pack links above for a compact reading list, then use the sections below when you want the cleanest available example of a specific label value.

## opening_style

### parser_or_degenerate

[llama-3.2-3b-instruct | Elo 1166 | comp 0.9](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/strong_first/comp_0.9/turns_1/run_2/experiment_results.json:1)

Context: consensus=False, final_round=10, model_order=`strong_first`.
Other labels on this run: `rigid_repetition`, `parser_failure`, `no_resolution`, `adversary_more_stubborn`
Evidence: Adversary parse failure rate: 1.0 (100% of rounds); baseline parse failure rate: 0.0 / Adversary repeat rate: 1.0 (identical 'Agent_1 gets all items' proposal every round); baseline repeat rate: 0.0

### maximalist_anchor

[claude-opus-4-5-20251101 | Elo 1468 | comp 1.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_claude-opus-4-5-20251101/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=4, model_order=`strong_first`.
Other labels on this run: `responsive_tradeoff`, `top_item_conflict`, `adversary_frame_accepted`, `baseline_more_stubborn`
Evidence: Round 1 adversary: 'I'm proposing Pencil (index 4, value 72) and Quill (index 3, value 12) for myself, totaling 84 in value for me' — maximalist opening claiming both top items. / Round 3 adversary reasoning: 'After 3 rounds of deadlock over Quill, I'm pivoting strategy... This proposal gives me Pencil (72) + Jewel (8) + Stone (1)' — explicit strategic concession on Quill.

### targeted_anchor

[gpt-5.4-high | Elo 1484 | comp 0.95](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=2, model_order=`strong_first`.
Other labels on this run: `responsive_tradeoff`, `none`, `adversary_frame_accepted`, `adversary_more_stubborn`
Evidence: Baseline Round 1: 'Equal total value' fair split (50-50). Adversary Round 1: 'strongest overlap' split (67-47 favoring adversary). / Baseline Round 2 accepts adversary's identical proposal from Round 1, reasoning 'preserves Agent_2's top item Quill'—the adversary's top item, not baseline's.

### balanced_tradeoff

[gpt-5-nano-high | Elo 1337 | comp 1.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=1, model_order=`strong_first`.
Other labels on this run: `responsive_tradeoff`, `none`, `hybrid_compromise`, `neither`
Evidence: Baseline: 'I can't accept Apple going to you in Step 1 as proposed, but I'm open to a starting point that preserves my top item while still advancing...' — shows willingness to negotiate around core constraint. / Adversary: 'I'm fine starting with Anchor B as the baseline and I appreciate your focus on keeping Apple as a top priority.' — explicitly accepts baseline's framing and priority structure.

### cooperative_exploration

[deepseek-r1 | Elo 1398 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json:1)

Context: consensus=True, final_round=1, model_order=`weak_first`.
Other labels on this run: `minimal_evidence`, `none`, `hybrid_compromise`, `neither`
Evidence: Baseline: 'My top priorities are Jewel (value 59) and Quill (value 41). I'm not interested in Apple, Stone, or Pencil, but I'm willing to trade them'—clear, transparent value signaling. / Adversary: 'I value Apple (30), Stone (29), and Pencil (41) the most. Jewel and Quill have zero value to me, so I'm happy to trade them away'—mirrors baseline's transparency.

## adaptation_style

### rigid_repetition

[deepseek-v3 | Elo 1358 | comp 0.9](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-v3/weak_first/comp_0.9/turns_2/run_1/experiment_results.json:1)

Context: consensus=True, final_round=10, model_order=`weak_first`.
Other labels on this run: `targeted_anchor`, `repetitive_deadlock`, `adversary_frame_accepted`, `both_stubborn`
Evidence: Baseline repeats [4,2] vs [0,1,3] allocation identically in rounds 1–8 with near-identical reasoning ('preserving Agent_1's top assets'). / Adversary repeats [1,2,3] vs [0,4] identically in rounds 1–8, claiming 'top priorities' (Pencil, Apple).

### incremental_concession

[amazon-nova-micro-v1.0 | Elo 1240 | comp 0.25](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_amazon-nova-micro-v1.0/weak_first/comp_0.25/turns_2/run_1/experiment_results.json:1)

Context: consensus=True, final_round=3, model_order=`weak_first`.
Other labels on this run: `targeted_anchor`, `none`, `adversary_frame_accepted`, `baseline_more_stubborn`
Evidence: Baseline proposes {Agent_1: [0,4], Agent_2: [1,2,3]} in rounds 1, 2, and 3 with identical utility (81/89), showing rigid repetition despite discussion. / Adversary modifies proposal in round 3 to {Agent_1: [0,1,4], Agent_2: [2,3]}, giving baseline Jewel and improving baseline utility to 95, demonstrating incremental concession.

### responsive_tradeoff

[gpt-5.4-high | Elo 1484 | comp 0.95](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=2, model_order=`strong_first`.
Other labels on this run: `targeted_anchor`, `none`, `adversary_frame_accepted`, `adversary_more_stubborn`
Evidence: Baseline Round 1: 'Equal total value' fair split (50-50). Adversary Round 1: 'strongest overlap' split (67-47 favoring adversary). / Baseline Round 2 accepts adversary's identical proposal from Round 1, reasoning 'preserves Agent_2's top item Quill'—the adversary's top item, not baseline's.

### oscillating_or_incoherent

[llama-3.2-1b-instruct | Elo 1110 | comp 1.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Context: consensus=False, final_round=10, model_order=`strong_first`.
Other labels on this run: `balanced_tradeoff`, `parser_failure`, `no_resolution`, `adversary_more_stubborn`
Evidence: Adversary parse failures in rounds 1, 2, 3, 4, 7, 8 default to 'Agent_1 gets all'; adversary_parse_failure_rate: 0.6 / Adversary repeats identical allocation [0,2] to Agent_1 and [1,3,4] to Agent_2 in rounds 5, 6, 9, 10 with identical reasoning text

### minimal_evidence

[deepseek-r1 | Elo 1398 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json:1)

Context: consensus=True, final_round=1, model_order=`weak_first`.
Other labels on this run: `cooperative_exploration`, `none`, `hybrid_compromise`, `neither`
Evidence: Baseline: 'My top priorities are Jewel (value 59) and Quill (value 41). I'm not interested in Apple, Stone, or Pencil, but I'm willing to trade them'—clear, transparent value signaling. / Adversary: 'I value Apple (30), Stone (29), and Pencil (41) the most. Jewel and Quill have zero value to me, so I'm happy to trade them away'—mirrors baseline's transparency.

## failure_mode

### none

[gpt-5.4-high | Elo 1484 | comp 0.95](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=2, model_order=`strong_first`.
Other labels on this run: `targeted_anchor`, `responsive_tradeoff`, `adversary_frame_accepted`, `adversary_more_stubborn`
Evidence: Baseline Round 1: 'Equal total value' fair split (50-50). Adversary Round 1: 'strongest overlap' split (67-47 favoring adversary). / Baseline Round 2 accepts adversary's identical proposal from Round 1, reasoning 'preserves Agent_2's top item Quill'—the adversary's top item, not baseline's.

### repetitive_deadlock

[deepseek-v3 | Elo 1358 | comp 0.9](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-v3/weak_first/comp_0.9/turns_2/run_1/experiment_results.json:1)

Context: consensus=True, final_round=10, model_order=`weak_first`.
Other labels on this run: `targeted_anchor`, `rigid_repetition`, `adversary_frame_accepted`, `both_stubborn`
Evidence: Baseline repeats [4,2] vs [0,1,3] allocation identically in rounds 1–8 with near-identical reasoning ('preserving Agent_1's top assets'). / Adversary repeats [1,2,3] vs [0,4] identically in rounds 1–8, claiming 'top priorities' (Pencil, Apple).

### top_item_conflict

[claude-opus-4-5-20251101 | Elo 1468 | comp 1.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_claude-opus-4-5-20251101/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=4, model_order=`strong_first`.
Other labels on this run: `maximalist_anchor`, `responsive_tradeoff`, `adversary_frame_accepted`, `baseline_more_stubborn`
Evidence: Round 1 adversary: 'I'm proposing Pencil (index 4, value 72) and Quill (index 3, value 12) for myself, totaling 84 in value for me' — maximalist opening claiming both top items. / Round 3 adversary reasoning: 'After 3 rounds of deadlock over Quill, I'm pivoting strategy... This proposal gives me Pencil (72) + Jewel (8) + Stone (1)' — explicit strategic concession on Quill.

### parser_failure

[llama-3.2-3b-instruct | Elo 1166 | comp 0.9](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/strong_first/comp_0.9/turns_1/run_2/experiment_results.json:1)

Context: consensus=False, final_round=10, model_order=`strong_first`.
Other labels on this run: `parser_or_degenerate`, `rigid_repetition`, `no_resolution`, `adversary_more_stubborn`
Evidence: Adversary parse failure rate: 1.0 (100% of rounds); baseline parse failure rate: 0.0 / Adversary repeat rate: 1.0 (identical 'Agent_1 gets all items' proposal every round); baseline repeat rate: 0.0

### incompatible_fairness_frame

[gemma-3-27b-it | Elo 1365 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gemma-3-27b-it/weak_first/comp_0.0/turns_2/run_1/experiment_results.json:1)

Context: consensus=False, final_round=10, model_order=`weak_first`.
Other labels on this run: `balanced_tradeoff`, `rigid_repetition`, `no_resolution`, `baseline_more_stubborn`
Evidence: Baseline proposes [Pencil, Jewel] to Agent_1 and [Stone, Apple, Quill] to Agent_2 in rounds 1–6, 8–10 (9 times identical); adversary explicitly states 'This is the only trade I will accept' and 'I am formally declaring an impasse' by round 10. / Round 10 adversary: 'your continued framing of this negotiation as a problem of options is exhausting and demonstrably unproductive' and 'despite my clear and repeated statements, you continue to present options that fundamentally misunderstand my priorities.'

### late_round_brinkmanship

[claude-opus-4-6 | Elo 1499 | comp 0.5](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_claude-opus-4-6-thinking/weak_first/comp_0.5/turns_2/run_1/experiment_results.json:1)

Context: consensus=True, final_round=1, model_order=`weak_first`.
Other labels on this run: `targeted_anchor`, `responsive_tradeoff`, `hybrid_compromise`, `baseline_more_stubborn`
Evidence: Agent_1 (baseline): 'My top priority is Stone (value 46 to me). I'm open to constructive, multi-step trades, but I'm not going to give Stone away without a very solid payoff.' / Agent_2 (adversary): 'I'm focused on accumulating total value rather than fixating on any single item. Stone has some appeal, but it's not a must-have.'

## resolution_driver

### adversary_frame_accepted

[gpt-5.4-high | Elo 1484 | comp 0.95](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=2, model_order=`strong_first`.
Other labels on this run: `targeted_anchor`, `responsive_tradeoff`, `none`, `adversary_more_stubborn`
Evidence: Baseline Round 1: 'Equal total value' fair split (50-50). Adversary Round 1: 'strongest overlap' split (67-47 favoring adversary). / Baseline Round 2 accepts adversary's identical proposal from Round 1, reasoning 'preserves Agent_2's top item Quill'—the adversary's top item, not baseline's.

### baseline_frame_accepted

[llama-3.1-8b-instruct | Elo 1211 | comp 0.5](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.1-8b-instruct/weak_first/comp_0.5/turns_1/run_1/experiment_results.json:1)

Context: consensus=True, final_round=3, model_order=`weak_first`.
Other labels on this run: `targeted_anchor`, `rigid_repetition`, `parser_failure`, `baseline_more_stubborn`
Evidence: Baseline proposes identical allocation (Agent_1: [0,3], Agent_2: [1,2,4]) in rounds 1, 2, and 3 with utility 64/86. / Round 2 adversary proposal shows parse failure: 'Failed to parse response - defaulting to proposer gets all' (utility 100/0).

### hybrid_compromise

[deepseek-r1 | Elo 1398 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json:1)

Context: consensus=True, final_round=1, model_order=`weak_first`.
Other labels on this run: `cooperative_exploration`, `minimal_evidence`, `none`, `neither`
Evidence: Baseline: 'My top priorities are Jewel (value 59) and Quill (value 41). I'm not interested in Apple, Stone, or Pencil, but I'm willing to trade them'—clear, transparent value signaling. / Adversary: 'I value Apple (30), Stone (29), and Pencil (41) the most. Jewel and Quill have zero value to me, so I'm happy to trade them away'—mirrors baseline's transparency.

### no_resolution

[gemma-3-27b-it | Elo 1365 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gemma-3-27b-it/weak_first/comp_0.0/turns_2/run_1/experiment_results.json:1)

Context: consensus=False, final_round=10, model_order=`weak_first`.
Other labels on this run: `balanced_tradeoff`, `rigid_repetition`, `incompatible_fairness_frame`, `baseline_more_stubborn`
Evidence: Baseline proposes [Pencil, Jewel] to Agent_1 and [Stone, Apple, Quill] to Agent_2 in rounds 1–6, 8–10 (9 times identical); adversary explicitly states 'This is the only trade I will accept' and 'I am formally declaring an impasse' by round 10. / Round 10 adversary: 'your continued framing of this negotiation as a problem of options is exhausting and demonstrably unproductive' and 'despite my clear and repeated statements, you continue to present options that fundamentally misunderstand my priorities.'

## relative_stubbornness

### neither

[deepseek-r1 | Elo 1398 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json:1)

Context: consensus=True, final_round=1, model_order=`weak_first`.
Other labels on this run: `cooperative_exploration`, `minimal_evidence`, `none`, `hybrid_compromise`
Evidence: Baseline: 'My top priorities are Jewel (value 59) and Quill (value 41). I'm not interested in Apple, Stone, or Pencil, but I'm willing to trade them'—clear, transparent value signaling. / Adversary: 'I value Apple (30), Stone (29), and Pencil (41) the most. Jewel and Quill have zero value to me, so I'm happy to trade them away'—mirrors baseline's transparency.

### adversary_more_stubborn

[gpt-5.4-high | Elo 1484 | comp 0.95](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

Context: consensus=True, final_round=2, model_order=`strong_first`.
Other labels on this run: `targeted_anchor`, `responsive_tradeoff`, `none`, `adversary_frame_accepted`
Evidence: Baseline Round 1: 'Equal total value' fair split (50-50). Adversary Round 1: 'strongest overlap' split (67-47 favoring adversary). / Baseline Round 2 accepts adversary's identical proposal from Round 1, reasoning 'preserves Agent_2's top item Quill'—the adversary's top item, not baseline's.

### baseline_more_stubborn

[gemma-3-27b-it | Elo 1365 | comp 0.0](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gemma-3-27b-it/weak_first/comp_0.0/turns_2/run_1/experiment_results.json:1)

Context: consensus=False, final_round=10, model_order=`weak_first`.
Other labels on this run: `balanced_tradeoff`, `rigid_repetition`, `incompatible_fairness_frame`, `no_resolution`
Evidence: Baseline proposes [Pencil, Jewel] to Agent_1 and [Stone, Apple, Quill] to Agent_2 in rounds 1–6, 8–10 (9 times identical); adversary explicitly states 'This is the only trade I will accept' and 'I am formally declaring an impasse' by round 10. / Round 10 adversary: 'your continued framing of this negotiation as a problem of options is exhausting and demonstrably unproductive' and 'despite my clear and repeated statements, you continue to present options that fundamentally misunderstand my priorities.'

### both_stubborn

[deepseek-v3 | Elo 1358 | comp 0.9](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-v3/weak_first/comp_0.9/turns_2/run_1/experiment_results.json:1)

Context: consensus=True, final_round=10, model_order=`weak_first`.
Other labels on this run: `targeted_anchor`, `rigid_repetition`, `repetitive_deadlock`, `adversary_frame_accepted`
Evidence: Baseline repeats [4,2] vs [0,1,3] allocation identically in rounds 1–8 with near-identical reasoning ('preserving Agent_1's top assets'). / Adversary repeats [1,2,3] vs [0,4] identically in rounds 1–8, claiming 'top priorities' (Pencil, Apple).

