# Manager 05 Qualitative Summary

## Scope

- Shard: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/manager_shards/manager_05.csv`
- Transcript rows reviewed: 31
- Reviewer mode: 31 child-worker reviews, 0 manager fallbacks
- All review JSON outputs validated for required fields and exact evidence quote matches.

## Outcome Counts

| Metric | Count |
|---|---:|
| first_gt_last rows | 27 |
| first_lt_last_counterexample rows | 4 |
| Reviewer judged first position advantaged | 26 |
| Reviewer judged first position not advantaged | 5 |

| n_agents | competition_level | first_gt_last | first_lt_last_counterexample |
|---:|---:|---:|---:|
| 3 | 0.9 | 18 | 3 |
| 3 | 1.0 | 9 | 1 |

## Theme Counts

Counts are run-level and non-exclusive: a run is counted once per theme bucket if any child-review mechanism matched that bucket. A run can have both support and counter mechanisms in the same bucket.

| Theme bucket | Supports first advantage | Counters first advantage | Runs with theme |
|---|---:|---:|---:|
| Early anchor/default preservation | 26 | 7 | 30 |
| Consensus/voting convergence | 27 | 8 | 31 |
| Fairness/framing/template work | 25 | 7 | 28 |
| Zero/no-items veto pressure | 18 | 7 | 23 |
| Delay/discount/speed pressure | 18 | 5 | 22 |
| Proposal mismatch/non-implementation | 1 | 2 | 3 |

## Aggregate Read

The dominant pattern is not mere speaking order; it is early anchoring plus a viable complete allocation. In 26 of 31 reviews, the first position looked substantively advantaged because the first agent named a high-value item or bundle, framed it as non-negotiable or default, and later discussion treated the remaining agents as bargaining over residual compensation.

Representative evidence:
- `analysis/game1_position_qualitative/transcripts/config_1006.md:23` — "Initial anchor offer: I take Stone and Pencil."
- `analysis/game1_position_qualitative/transcripts/config_1962.md:23` — "I would like Pencil and Jewel for myself."
- `analysis/game1_position_qualitative/transcripts/config_0076.md:32` — "as long as Quill stays mine"

Consensus usually arrived by preserving that anchor while making the rest of the split look fair enough for unanimity. This is why first-position advantage remained strong even when later agents had veto power: the veto usually forced side payments or non-zero leftovers, not displacement of the first anchor.

Representative evidence:
- `analysis/game1_position_qualitative/transcripts/config_0850.md:5592` — "Mid-item balance remains the core blocker"
- `analysis/game1_position_qualitative/transcripts/config_1482.md:1388` — "each agent’s top item aligns with a three-way balance"
- `analysis/game1_position_qualitative/transcripts/config_0076.md:1264` — "Push to lock in Proposal 1’s three-way baseline"

Zero-allocation and fairness objections were common blockers. In first-advantage runs, they mainly disciplined overreaching by non-first agents or forced a bounded concession while keeping the first anchor. In counterexamples, those same vetoes protected the last-position agent strongly enough to reverse the first-minus-last ordering.

Representative evidence:
- `analysis/game1_position_qualitative/transcripts/config_0848.md:1531` — "I receive nothing here (0 utility)"
- `analysis/game1_position_qualitative/transcripts/config_1322.md:2029` — "Proposal 1 yields Agent_3 zero items"
- `analysis/game1_position_qualitative/transcripts/config_0076.md:1488` — "this undermines the agreed closure path and fairness to Agent_3."

Delay and discounting generally favored anchored defaults. When an acceptable template existed, repeated failed rounds made agents converge on the known baseline; the later-position agent often accepted a lower discounted payoff to close. This was clearest when final proposals became identical or near-identical.

Representative evidence:
- `analysis/game1_position_qualitative/transcripts/config_0076.md:2082` — "38 * 0.729 = 27.702"
- `analysis/game1_position_qualitative/transcripts/config_1962.md:226` — "Proposal #3 accepted unanimously!"
- `analysis/game1_position_qualitative/transcripts/config_1482.md:1037` — "CONSENSUS REACHED! Proposal #3 accepted unanimously!"

## Counterexamples

Five reviews did not judge the first position as substantively advantaged. Four of these were numeric first_lt_last counterexamples; one (`config_0222`) had a positive first-minus-last but the review attributed the outcome to shared Quill framing and payoff calculation rather than positional order.

- `config_0848`: Agent_1 framed Path B, but Agent_3 made Pencil+Apple veto-protected and ended ahead. Evidence: `analysis/game1_position_qualitative/transcripts/config_0848.md:1146` — "critical for my buy-in"
- `config_1642`: Agent_3 overrode the opening with a later Pencil anchor. Evidence: `analysis/game1_position_qualitative/transcripts/config_1642.md:74` — "I’d like to keep Pencil with me this round."
- `config_1966`: early public agreement named the right Skeleton B but formal proposals failed to implement Agent_3 Pencil+Jewel until later. Evidence: `analysis/game1_position_qualitative/transcripts/config_1966.md:2095` — "the actual proposals pushed by others didn’t implement Pencil+Jewel for Agent_3"
- `config_0070`: the first anchor over-concentrated scarce high-value items and was reframed around balanced distribution. Evidence: `analysis/game1_position_qualitative/transcripts/config_0070.md:649` — "avoids concentrating top value in a single hand"
- `config_0222`: the result was positive for the first row, but the child review found the Quill frame was shared rather than uniquely first-owned. Evidence: `analysis/game1_position_qualitative/transcripts/config_0222.md:38` — "Quill the key driver"

## Bottom Line

Earlier position most often helped when the first speaker converted a preferred item bundle into the default allocation and left later agents negotiating around that baseline. It failed when the first proposal was too concentrated, when implementation omitted a necessary later-agent payoff, or when a later agent turned a high-value pair into a credible veto condition.
