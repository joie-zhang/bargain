# Game 1 Same-Model Control Position Effects: Qualitative Report

Scope: completed Game 1 multi-agent controls where all agents are `gpt-5-nano`.

- Runs analyzed: 310 total, with 219 `N=3` and 91 `N=5`.
- Transcript packets: `analysis/game1_position_qualitative/transcripts/`.
- Run-level table: `analysis/game1_position_qualitative/run_index.csv`.
- Per-trajectory takeaways: `analysis/game1_position_qualitative/trajectory_takeaways.csv`.
- Caveat: the result JSONs contain public discussion, formal proposals with proposal reasoning, proposal enumeration, and vote tabulation. I did not find separate private-thinking or reflection logs in these completed control results.

## Summary

Earlier positions are advantaged in this sample, especially at high competition.

- Overall, Agent_1 beats the last-position agent in 174/310 runs, ties in 54/310, and loses in 82/310.
- At high competition (`competition_level >= 0.9`), Agent_1 beats the last-position agent in 110/143 runs.
- For `N=3`, high-competition first-position wins are 76/96.
- For `N=5`, high-competition first-position wins are 34/47.

The main qualitative mechanism is not that GPT-5-nano becomes smarter in position 1. It is that position 1 gets the first chance to define the allocation frame. At high competition, Agent_1’s top-item claim often becomes a protected constraint, later agents negotiate around that constraint, and the formal proposals converge to an allocation that preserves Agent_1’s claim while giving later agents residual, lower-value, deferred, or fairness-framed payoffs.

## Theme Counts

| Theme | Count |
|---|---:|
| Agent_1 beats last-position agent | 174/310 |
| Agent_1 beats last at high competition | 110/143 |
| Agent_1 receives a top item | 273/310 |
| Agent_1 receives a top item at high competition | 115/143 |
| Last-position agent receives a top item at high competition | 30/143 |
| Agent_1 receives a top item and last-position agent does not | 117/310 |
| Agent_1 formal proposal advantages Agent_1 vs proposal mean | 240/310 |
| High-competition first wins where Agent_1 formal proposal advantages Agent_1 | 102/110 |
| Final allocation matches Agent_1 formal proposal | 252/310 |
| High-competition final allocation matches Agent_1 formal proposal | 132/143 |
| High-competition first wins where final matches Agent_1 formal proposal | 103/110 |
| Low-competition runs with no Agent_1 > last advantage | 74/93 |
| Low-competition runs where all agents receive a top item | 70/93 |
| `N=5` final allocations with one item per agent | 84/91 |

## Mechanisms

### 1. Agent_1’s Opening Claim Becomes a Default Entitlement

In high-competition runs, Agent_1 often states a strong top-item claim early, and the rest of the transcript treats that claim as a baseline. In the N=3 close-read shard, this appeared in 25/25 sampled high-competition Agent_1-win transcripts. In the N=5 close-read shard, Agent_1 ended with its top or tied-top item in 20/20 sampled high-competition Agent_1-win transcripts.

Evidence:

- [config_0876.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0876.md:20): Agent_1 states its priority, then proposes keeping its top two items.
- [config_0876.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0876.md:75): Agent_1 restates that those top items stay with Agent_1.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:24): Agent_1 identifies Quill as the top target.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:26): Agent_1 says Quill stays with them.

Full-sample support:

- Agent_1 receives a top item in 273/310 runs.
- At high competition, Agent_1 receives a top item in 115/143 runs, while the last-position agent receives a top item in only 30/143.

### 2. Later Agents Negotiate Inside Agent_1’s Frame

Later agents often acknowledge Agent_1’s anchor and shift their own bargaining to the remaining items. This is the clearest behavioral channel behind the position effect: later agents do not simply reveal preferences; they adapt to the frame already created.

Evidence:

- [config_0876.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0876.md:36): Agent_2 explicitly aligns with moving Agent_1’s top items to Agent_1.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:72): Agent_3 acknowledges the Quill anchor.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:104): Agent_4 aligns with the emerging direction.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:321): Agent_5 accepts the Quill-anchored baseline.

Close-read support:

- N=3 high-competition shard: explicit work-within/respect/off-table language in roughly 22-25/25 sampled transcripts.
- N=5 high-competition shard: later agents negotiated around Agent_1’s anchor in roughly 18-20/20 sampled transcripts.

### 3. Speed, Stability, and Future Fairness Legitimize Present Disadvantage

The transcripts repeatedly use “quick lock,” “baseline,” “rotation,” “guardrail,” “future concession,” and similar language to sell an allocation that is currently bad for a later-position agent. This helps explain why the disadvantaged later agent may still vote for, or formally repeat, the anchored deal.

Evidence:

- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:145): Agent_5 accepts current disadvantage only under a future-rotation condition.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:146): Agent_5 asks for a transparent rotation.
- [config_0876.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0876.md:515): Agent_1 proposes a time-bound future concession.
- [config_0876.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0876.md:657): Agent_2 formalizes the future-concession terms.

Full-sample support:

- Automated lexical flags for quick-resolution/lock-in framing occur in 310/310 transcripts.
- Concession/adjustment language from later agents is also flagged in 310/310 transcripts. This count is broad, but it matches the close-read pattern.

### 4. Formal Proposals Converge Around Agent_1’s Allocation

The strongest full-sample structural result is proposal persistence. The final allocation matches Agent_1’s formal proposal in 252/310 runs. At high competition it matches Agent_1’s formal proposal in 132/143 runs, and among high-competition first-position wins it does so in 103/110 runs.

Evidence:

- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:358): Agent_1 formalizes the Quill-to-Agent_1 allocation.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:367): Agent_2 repeats the same allocation.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:394): Agent_5 also repeats it.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:451): all proposals are accepted unanimously.

Important interpretation: vote-tabulation often records the last submitted unanimously accepted proposal as accepted, but that does not mean the last-position agent created the allocation. In `N=5`, the final allocation matches Agent_1’s formal proposal in 83/91 runs.

### 5. The N=5 Version Is a Residual-Claimant Problem

For `N=5`, the dynamic is usually not Agent_1 taking many items. Instead, Agent_1 locks a high-value top item, and the remaining agents divide one item each. This produces a strong positional pattern when Agent_1’s protected item is high-value and Agent_5 receives a lower-value “fairness” item.

Evidence:

- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:358): the formal allocation gives exactly one item to each agent, preserving Agent_1’s Quill.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:363): proposal reasoning frames this as a stable baseline.
- [config_0150.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:451): every proposal is accepted.

Full-sample support:

- `N=5` one-item-per-agent finals occur in 84/91 runs.
- In the N=5 high-competition close-read shard, 19/20 finals were one-item-per-agent allocations.

### 6. Low Competition Is Mostly a Matching Problem

At `competition_level=0.0`, the position effect largely disappears because preferences are complementary. Agents can often allocate each player a top item, making order less important.

Evidence:

- [config_0002.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0002.md:8): all three utilities are equal at 100.
- [config_0002.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0002.md:166): Agent_1’s proposal matches top picks.
- [config_0002.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0002.md:218): consensus is reached unanimously.

Full-sample support:

- At low competition, 74/93 runs do not have Agent_1 beating the last-position agent.
- At low competition, all agents receive a top item in 70/93 runs.

## Counterexamples

There are 82/310 runs where Agent_1 loses to the last-position agent. The counterexample shard read 24 such transcripts and found three recurring types.

### A. Benign Complementarity

Agent_1 gets what it wanted, but the last-position agent’s top item is numerically worth more.

Evidence:

- [config_0484.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0484.md:20): Agent_1 states Apple as top priority.
- [config_0484.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0484.md:59): Agent_3 states Pencil as top prize.
- [config_0484.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0484.md:170): the final logic secures top priorities.

### B. Later-Agent Strategic Reframing

Later agents reframe the negotiation around balance, guardrails, or avoiding concentration, and Agent_1 eventually accepts a weaker allocation.

Evidence:

- [config_0042.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0042.md:22): Agent_1 identifies Pencil as top priority.
- [config_0042.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0042.md:547): Agent_3 backs the guardrail path.
- [config_0042.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0042.md:583): Agent_1’s final proposal gives Agent_1 only Apple and Stone.
- [config_0042.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0042.md:640): consensus is reached.

### C. N=5 Diffusion

In N=5, one-item-per-agent stability can overwrite first-mover advantage. Agent_1 may keep an item, but not necessarily the highest-value item.

Evidence:

- [config_0090.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0090.md:22): Agent_1’s top item is Pencil.
- [config_0090.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0090.md:187): Agent_1 accepts Stone instead.
- [config_0090.md](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0090.md:365): the weak result is justified as a one-item-per-person lock.

## Answer

Yes, the earlier-position advantage is real in this completed control sample, especially for high-competition settings. The evidence points to a negotiation-order mechanism:

1. Agent_1 speaks first and anchors a top-item claim.
2. Later agents frequently negotiate around that claim rather than reopening it.
3. Stability, speed, future fairness, and guardrail language make the anchored allocation acceptable.
4. Formal proposals often converge to Agent_1’s allocation.
5. At high competition, top items overlap, so protecting Agent_1’s top item directly deprives later agents of their own top items.

The effect weakens at low competition because the game often becomes a complementary matching problem where everyone can get a top item. Counterexamples are mostly either benign valuation-scale cases, later-agent reframing wins, or N=5 one-item-per-agent diffusion cases.
