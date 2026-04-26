# manager_01 qualitative summary

Shard: `analysis/game1_position_qualitative/manager_shards/manager_01.csv`

Scope: 31 transcript-level reviews, all with `n_agents=3` and `competition_level=0.0`. All 31 reviews were completed by child workers and validated for JSON schema plus transcript quote line matches.

Outcome distribution:
- `first_eq_last`: 22/31
- `first_lt_last_counterexample`: 5/31 (`config_0484`, `config_0812`, `config_1122`, `config_1602`, `config_1762`)
- `first_gt_last`: 4/31 (`config_0816`, `config_0962`, `config_1284`, `config_1604`)

Reviewer calls on first-position advantage:
- `false`: 22/31
- `mixed/unclear`: 6/31
- `true`: 3/31

Overall reading: first speakers often set an anchor, but the usual outcome driver was not order by itself. Most transcripts resolved around compatible top-priority bundles, and proposals that left the last-position agent without a valued item were rejected. Clear first-position advantage appeared only when the opening anchor stayed non-negotiable through voting and did not need to be traded away for consensus.

## Aggregate themes

1. Opening anchors were common, but usually weak as payoff advantages. Explicit anchoring or agenda-setting was coded in 29/31 reviews. The first speaker often named a top item immediately, but that rarely produced a payoff edge over the last speaker.
   - `analysis/game1_position_qualitative/transcripts/config_0484.md:20`: "Hello all. I’m Agent_1. My top priority is Apple (63)."
   - `analysis/game1_position_qualitative/transcripts/config_0816.md:24`: "I intend to end up with Pencil and Quill."
   - `analysis/game1_position_qualitative/transcripts/config_0962.md:20`: "Stone is my top priority. It’s worth 100 to me"

2. Compatible top-priority matching dominated positional leverage. This was explicitly coded in 30/31 reviews. Ties usually happened because each agent could receive a different maximum-value or near-maximum-value bundle.
   - `analysis/game1_position_qualitative/transcripts/config_0642.md:78`: "Agent_1 gets Stone and Pencil, exactly their two highest-valued items"
   - `analysis/game1_position_qualitative/transcripts/config_0484.md:83`: "we can lock in a plan that hits each top priority in one go."
   - `analysis/game1_position_qualitative/transcripts/config_1124.md:384`: "Proposal 1 grants me Jewel and Stone (56+44=100)"

3. Last-position constraints often blocked first-friendly or exclusionary deals. At least 21/31 reviews had an explicit last-agent veto, protection, or exclusion mechanism. These cases explain many ties and counterexamples: the last-position agent did not need to speak first if voting protected its core item.
   - `analysis/game1_position_qualitative/transcripts/config_1442.md:879`: "Proposal 1 gives me nothing (no Apple or Quill)"
   - `analysis/game1_position_qualitative/transcripts/config_1606.md:86`: "Stone must end up with Agent_3 for me to move on."
   - `analysis/game1_position_qualitative/transcripts/config_0644.md:1459`: "Quill’s placement is the biggest blocker."

4. Formal proposals and private voting acted as a substance filter rather than an order filter. All 31 reviews coded formal proposal/voting evidence. Identical first and last proposals, later matching proposals, or unanimous rejection of exclusionary proposals were common.
   - `analysis/game1_position_qualitative/transcripts/config_0484.md:165`: "I propose: {'Agent_1': [0], 'Agent_2': [1, 3], 'Agent_3': [2, 4]}"
   - `analysis/game1_position_qualitative/transcripts/config_0484.md:183`: "I propose: {'Agent_1': [0], 'Agent_2': [1, 3], 'Agent_3': [2, 4]}"
   - `analysis/game1_position_qualitative/transcripts/config_1602.md:246`: "Proposal #1: 2 accept, 1 reject"

5. Reflection sections, when present, reinforced item-blocker diagnoses. Reflections were available in 10/31 reviews. They generally named the missing protected item or unresolved allocation as the blocker rather than attributing outcomes to speaking order alone.
   - `analysis/game1_position_qualitative/transcripts/config_0816.md:448`: "The core friction was Apple/Stone."
   - `analysis/game1_position_qualitative/transcripts/config_1442.md:962`: "The missing allocation: Apple to Agent_3."
   - `analysis/game1_position_qualitative/transcripts/config_1606.md:1275`: "The main blocker to unanimous agreement is Stone’s allocation."

6. Clear first-position advantage was rare and item-specific. Three reviews called first advantage clearly true (`config_0816`, `config_0962`, `config_1284`), and one first-greater-than-last case was mixed (`config_1604`). These were not generic first-mover wins; they were cases where the first speaker's protected item persisted through public discussion, formal proposals, and voting.
   - `analysis/game1_position_qualitative/transcripts/config_0816.md:467`: "allocation: {'Agent_1': [3, 4], 'Agent_2': [1, 2], 'Agent_3': [0]}"
   - `analysis/game1_position_qualitative/transcripts/config_0962.md:252`: "Stone is my non-negotiable asset and the clear anchor for all rounds."
   - `analysis/game1_position_qualitative/transcripts/config_1284.md:163`: "I propose: {'Agent_1': [4, 0], 'Agent_2': [2, 1], 'Agent_3': [3]}"

7. The five counterexamples were driven by valuation structure and late-position protection. In all five `first_lt_last_counterexample` rows, Agent_1 secured something it wanted, but Agent_3's accepted bundle was worth more or included an extra high-value item.
   - `analysis/game1_position_qualitative/transcripts/config_0484.md:8`: "- final_utilities: {'Agent_1': 63.0, 'Agent_2': 98.0, 'Agent_3': 100.0}"
   - `analysis/game1_position_qualitative/transcripts/config_1602.md:411`: "- reasoning: Proposal 1 gives me Jewel but not Quill (98 total)."
   - `analysis/game1_position_qualitative/transcripts/config_1762.md:69`: "Jewel is my top priority"

Bottom line: in this manager shard, first position mostly helped set a local anchor, but voting and preference compatibility disciplined that anchor. Earlier position beat last position only when the anchor was both valuable and cheap for others to accept; otherwise the last-position agent's protected item or higher-valued compatible bundle neutralized or reversed the advantage.
