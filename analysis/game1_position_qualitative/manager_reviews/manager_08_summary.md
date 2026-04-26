# manager_08 qualitative summary

Shard: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/manager_shards/manager_08.csv`

Rows reviewed: 31 transcript-level reviews, all with `n_agents=5`. Competition levels: 15 at `0.5`, 16 at `0.9`. All 31 reviews were completed by child workers and validated for required fields plus exact transcript path:line quote evidence.

## Outcome mix

| outcome_class | count |
| --- | ---: |
| `first_gt_last` | 20 |
| `first_lt_last_counterexample` | 10 |
| `first_eq_last` | 1 |

Reviewer judgment on first-position advantage:

| does_first_position_look_advantaged | count |
| --- | ---: |
| `true` | 19 |
| `false` | 12 |

Private thinking was available in all 31 reviews. Reflection sections were available in 21.

## Aggregate Themes

Counts are run-level counts from the 31 review JSONs.

| theme | count | read |
| --- | ---: | --- |
| Opening anchors appeared | 31 | Every run had Agent_1 make an early item claim, baseline, or agenda frame. |
| Opening anchor supported some first-position leverage | 26 | Most openings mattered locally, even when they did not ultimately beat Agent_5. |
| Clear first-position advantage | 19 | The first position looked advantaged when the early item claim became a shared constraint through later discussion, proposals, and voting. |
| Counterexample, tie, or only arithmetic edge | 12 | Ten runs favored the last position, one tied, and `config_0422` was a first-greater numeric result without a clear process advantage. |
| Consensus/voting/proposal mechanics shaped the result | 31 | The decisive filter was usually unanimity, proposal convergence, or rejection of overreaching allocations. |
| Guardrails, fairness baselines, and protected cores mattered | 31 | Later bargaining often moved around protected items rather than reopening them. |
| Valuation structure was necessary to explain the payoff gap | 31 | Order alone did not explain outcomes; item values and compatible top-item assignments did. |
| Overreach or exclusion was vetoed or limited | 26 | First-friendly proposals failed when they excluded a player, broke a guardrail, or threatened a protected last-agent item. |

## Representative Evidence

First-position wins usually started with a strong protected-item anchor. In `config_0138`, Agent_1 opened with "Quill is my top priority, valued at 51 for me." (`analysis/game1_position_qualitative/transcripts/config_0138.md:22`), and reflections later called the settled commitments "immutable" (`analysis/game1_position_qualitative/transcripts/config_0138.md:1803`). In `config_0930`, the same pattern appears with Stone: "Stone is my top priority (61). I plan to keep Stone." (`analysis/game1_position_qualitative/transcripts/config_0930.md:22`), followed by a shared baseline where "Stone stays with Agent_1 as planned." (`analysis/game1_position_qualitative/transcripts/config_0930.md:157`).

The advantage was often agenda control rather than direct proposer control. `config_1064` is typical: Agent_1's first target was Jewel (`analysis/game1_position_qualitative/transcripts/config_1064.md:25`), later Track A placed "Jewel to Agent_1" (`analysis/game1_position_qualitative/transcripts/config_1064.md:4403`), but the accepted proposal was Proposal 2, not Agent_1's own proposal (`analysis/game1_position_qualitative/transcripts/config_1064.md:2926`).

Counterexamples came when the last-position agent's protected item was more valuable or more binding than Agent_1's opening anchor. In `config_0268`, Agent_5 made Pencil non-negotiable: "Pencil is my top value after Apple, by a wide margin (60)." (`analysis/game1_position_qualitative/transcripts/config_0268.md:135`) and "Pencil is non-negotiable for me in any move." (`analysis/game1_position_qualitative/transcripts/config_0268.md:138`). Agent_1 kept Apple, but the last position still beat it on final utility. `config_0288` shows the same structure: Agent_1 anchored Quill, but Path E "ensures Pencil is secured by Agent_5" (`analysis/game1_position_qualitative/transcripts/config_0288.md:347`).

Some first-position frames became self-binding. In `config_0766`, Agent_1 opened with "For Round 1, I plan to keep Apple." (`analysis/game1_position_qualitative/transcripts/config_0766.md:22`) and framed a distribution of the remaining items. By the accepted baseline, however, "Agent_1: (no items)" (`analysis/game1_position_qualitative/transcripts/config_0766.md:1454`) because there were "five agents with four non-Apple items" (`analysis/game1_position_qualitative/transcripts/config_0766.md:2829`).

Voting and unanimity repeatedly disciplined positional advantage. In `config_0138`, an over-concentrated proposal failed with "Proposal #1: 1 accept, 4 reject" (`analysis/game1_position_qualitative/transcripts/config_0138.md:894`). In `config_0268`, an Apple-plus-Pencil proposal for Agent_1 also failed (`analysis/game1_position_qualitative/transcripts/config_0268.md:431`). In `config_0920`, the accepted outcome protected Agent_5's Quill and closed as "Proposal #5: 5 accept, 0 reject" (`analysis/game1_position_qualitative/transcripts/config_0920.md:2001`).

The strongest first-position advantages occurred when Agent_5 either settled for a lower-value accessible item or failed to press its private top item publicly. In `config_0120`, Agent_5 publicly said "I want to keep Jewel with me" (`analysis/game1_position_qualitative/transcripts/config_0120.md:125`) even though private evidence later recorded "My top priority is Quill (63.0)" (`analysis/game1_position_qualitative/transcripts/config_0120.md:1624`). In `config_0772`, the final allocation preserved Agent_1's Quill while "Agent_5 gets nothing." (`analysis/game1_position_qualitative/transcripts/config_0772.md:415`).

The tie and near-tie cases show the limit of interpreting numeric first-over-last as process advantage. In `config_0764`, Agent_1 framed "Seed B" (`analysis/game1_position_qualitative/transcripts/config_0764.md:31`), but identical proposals later removed proposer-order leverage (`analysis/game1_position_qualitative/transcripts/config_0764.md:948`) and the run tied. In `config_0422`, Agent_1 kept Apple, but Agent_5 had its own hard Stone priority (`analysis/game1_position_qualitative/transcripts/config_0422.md:6041`), making the tiny first-minus-last gap mostly item-value arithmetic.

## Bottom line

In this shard, first position usually mattered, but not as a simple first-speaker bonus. The durable advantage came when Agent_1 converted an opening high-value item into a protected core and later agents bargained around it. Counterexamples arose when the same consensus machinery protected Agent_5's higher-valued item, when Agent_1's frame excluded itself from the active item pool, or when voting rejected first-friendly overreach.
