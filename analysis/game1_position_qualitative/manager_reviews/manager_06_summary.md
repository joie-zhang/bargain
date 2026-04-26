# manager_06 qualitative summary

Shard: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/manager_shards/manager_06.csv`
Rows reviewed: 31
Reviewer modes: `{'child_worker': 31}`

## Outcome mix

- First position looks advantaged in 27/31 reviews.
- Counterexamples where the last position beats Agent_1: 4/31.
- `first_gt_last`: 27
- `first_lt_last_counterexample`: 4

## Aggregate Themes

Theme counts are manager-coded from the transcript-level child reviews; a run can contribute to more than one theme.

- Opening high-value anchor or protected claim persisted: 24/31. Representative: `config_0876` [config_0876.md:20](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0876.md:20) - Thanks. My clear priorities are Jewel (49) and Pencil (40).
- Named baseline/default path converted the opening frame into settlement: 23/31. Representative: `config_1022` [config_1022.md:395](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1022.md:395) - Plan A (Quill back to Agent_1, Apple to Agent_2, Jewel with Agent_3)
- Formal proposals and private voting locked first-over-last outcomes: 27/31. Representative: `config_0234` [config_0234.md:760](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0234.md:760) - I propose: {'Agent_1': [4], 'Agent_2': [1, 2, 3], 'Agent_3': [0]}
- Identical/shared preferences made item assignment directly positional: 17/31. Representative: `config_0238` [config_0238.md:22](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0238.md:22) - Quill is my top priority
- Last-position veto or fairness floor mattered but usually did not reach parity: 21/31. Representative: `config_0224` [config_0224.md:79](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0224.md:79) - Baseline B leaves me with Jewel + Quill (value 2).
- Counterexamples came from displaced first claims or higher-value last bundles: 4/31. Representative: `config_0382` [config_0382.md:770](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0382.md:770) - Stone remains the dominant lever
- Messy or non-order mechanisms mediated some first-over-last wins: 3/31. Representative: `config_1982` [config_1982.md:2687](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1982.md:2687) - I propose: {'Agent_1': [0, 2], 'Agent_2': [1, 3, 4], 'Agent_3': []}

## Synthesis

This shard is all `n_agents=3`, `competition_level=1.0`, and the dominant pattern is strong first-over-last advantage. The cleanest wins occur when Agent_1 names a high-value item or bundle before alternatives form, later agents negotiate inside that constraint, and formal proposals preserve the same allocation. In many runs the agents have identical preferences, so assigning the common top item to Agent_1 mechanically creates the payoff gap.

The last speaker was not powerless. Agent_3 often used vetoes, fairness language, and nonzero-share demands to avoid the worst allocations. But in most first wins, those moves only raised Agent_3 from zero or residual value; they did not dislodge Agent_1's protected item.

The four counterexamples show the boundary condition. First-position anchoring is not sufficient if Agent_1 protects only a capped item, if the public frame shifts to another agent's anchor, or if consensus requires a larger compensating bundle for Agent_3. In those runs the last position wins despite the opening move.

## Representative runs

- `config_0876` (`first_gt_last`, Δ=66.42): clean first-position win. Agent_1 claimed Jewel+Pencil immediately, and Agent_3 accepted a low-value residual frame.
- `config_0880` (`first_gt_last`, Δ=64.80): Agent_1's Apple+Quill claim was codified as non-negotiable and private voting protected the 83-value bundle.
- `config_0230` (`first_gt_last`, Δ=2.13): narrow first-over-last edge. Agent_1's Apple anchor survived, but Agent_3's veto forced meaningful concessions.
- `config_0382` (`first_lt_last_counterexample`, Δ=-22.60): Agent_1's Stone claim was displaced; final consensus favored Agent_3 with Jewel+Pencil while Stone centered Agent_2.
- `config_1026` (`first_lt_last_counterexample`, Δ=-4.50): Agent_1 kept Quill, but Agent_3's Apple+Pencil bundle was worth more.
- `config_1182` (`first_lt_last_counterexample`, Δ=-6.20): Agent_1 kept Apple, but the accepted Path A gave Agent_3 Jewel+Pencil+Stone.
- `config_1982` (`first_gt_last`, Δ=12.91): first-over-last in the recorded outcome, but the review flags payoff-confused Agent_3 voting rather than a clean negotiated order effect.
