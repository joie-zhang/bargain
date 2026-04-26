# manager_07 qualitative summary

Shard: `analysis/game1_position_qualitative/manager_shards/manager_07.csv`

Scope: 31 transcript-level reviews. The shard contains 29 five-agent runs and 2 three-agent runs; competition levels are 26 at `0.0`, 3 at `0.5`, and 2 at `1.0`. All 31 reviews were completed by child workers and validated for JSON schema plus transcript quote line matches.

Outcome distribution:
- `first_lt_last_counterexample`: 17/31
- `first_gt_last`: 14/31

Reviewer calls on first-position advantage:
- `true`: 13/31
- `mixed/unclear`: 10/31
- `false`: 8/31

Overall reading: first speakers almost always set an initial item anchor, but that anchor only became a payoff advantage when it was high-value, coalition-compatible, and protected through formal voting. The counterexamples were not cases where Agent_1 failed to speak; they were cases where the last-position agent's protected item, veto, or higher payoff scale survived the same consensus process.

## Aggregate themes

1. Opening anchors were present in all 31 reviews, but anchor strength varied sharply. Agent_1 usually named a top item or desired bundle first. This gave agenda control, but it was not enough unless later public discussion and voting treated the claim as protected.
   - `analysis/game1_position_qualitative/transcripts/config_0086.md:22`: "Stone is my top priority."
   - `analysis/game1_position_qualitative/transcripts/config_0248.md:25`: "Quill (38) and Pencil (61) are my non-negotiables"
   - `analysis/game1_position_qualitative/transcripts/config_1524.md:25`: "I am almost certainly aiming to keep Quill."

2. Clear first-position advantage appeared when Agent_1's anchor became a protected consensus constraint. This describes the 13 reviews coded as clear first-position advantage, plus one `first_gt_last` run coded `mixed/unclear`. In these cases Agent_5 either accepted a weaker residual item, deferred its target, or had a demand that conflicted with too many other protected holders.
   - `analysis/game1_position_qualitative/transcripts/config_0086.md:406`: "Stone with Agent_1 (threshold 56) is non-negotiable"
   - `analysis/game1_position_qualitative/transcripts/config_0248.md:2223`: "Proposal 3 preserves all anchors"
   - `analysis/game1_position_qualitative/transcripts/config_1364.md:398`: "'Agent_5': []"

3. Counterexamples were driven by last-position protection plus payoff scale. All 17 `first_lt_last_counterexample` rows ended with the last position receiving an item or package worth more than Agent_1's accepted item or package. At least 17/31 reviews explicitly coded value asymmetry, top-item scale, or final value arithmetic as a mechanism.
   - `analysis/game1_position_qualitative/transcripts/config_0084.md:154`: "Jewel goes to the highest-valuing recipient (me)"
   - `analysis/game1_position_qualitative/transcripts/config_0098.md:16`: "- Agent_5: [6.0, 85.0, 0.0, 9.0, 0.0]"
   - `analysis/game1_position_qualitative/transcripts/config_0102.md:129`: "Stone is my absolute top asset."
   - `analysis/game1_position_qualitative/transcripts/config_0724.md:127`: "my non-negotiable anchor"

4. Consensus and coalition compatibility often mattered more than speaking order. Every review coded a final consensus or coalition-filter mechanism. First-position claims won when they fit a broader priority map; they lost when the group needed to satisfy more pivotal agents or assign items to higher valuers.
   - `analysis/game1_position_qualitative/transcripts/config_0090.md:219`: "It preserves each top-priority item for four of us"
   - `analysis/game1_position_qualitative/transcripts/config_1524.md:383`: "Assign each item to the agent who values it most"
   - `analysis/game1_position_qualitative/transcripts/config_1846.md:606`: "Avoid leaving anyone without an allocation"

5. Formal proposals and private voting acted as a filter on public anchors in all 31 reviews. Self-serving first proposals, late-position overclaims, and allocations that omitted pivotal agents were rejected; repeated or coalition-compatible baselines were ratified.
   - `analysis/game1_position_qualitative/transcripts/config_1364.md:455`: "Proposal #1: 2 accept, 3 reject"
   - `analysis/game1_position_qualitative/transcripts/config_1524.md:471`: "Proposal #1: 5 accept, 0 reject"
   - `analysis/game1_position_qualitative/transcripts/config_0242.md:2329`: "Proposal #5: 5 accept, 0 reject"

6. Private thinking was available in all 31 reviews, and reflections were available in 20/31. These sections generally reinforced the feasible baseline rather than simply copying the first speaker. When reflections were present, they often named the protected item or consensus path that later explained the first-vs-last payoff gap.
   - `analysis/game1_position_qualitative/transcripts/config_0254.md:566`: "Target Path A as Round-1 allocation"
   - `analysis/game1_position_qualitative/transcripts/config_0102.md:1025`: "Step 2: Restore final baseline: Stone stays with Agent_5"
   - `analysis/game1_position_qualitative/transcripts/config_1846.md:1332`: "Broad baseline"

7. Mixed cases show why outcome sign is not identical to positional advantage. Ten reviews were coded `mixed/unclear`: nine were counterexamples where Agent_1 still set a real anchor, and one was a narrow `first_gt_last` where both first and last positions secured their top item and the margin came mostly from item values.
   - `analysis/game1_position_qualitative/transcripts/config_0114.md:22`: "Jewel is my top target (42)."
   - `analysis/game1_position_qualitative/transcripts/config_0114.md:141`: "Stone is my clear top target (value 40)."
   - `analysis/game1_position_qualitative/transcripts/config_0114.md:2835`: "broad alignment around Step 1 baseline"

Bottom line: this shard is a strong warning against treating first speaker order as a standalone causal mechanism. Position mattered when it created a durable protected anchor, but voting, coalition compatibility, and item-value asymmetry often neutralized or reversed the first-position edge.
