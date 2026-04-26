# manager_02 qualitative summary

- Shard: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/manager_shards/manager_02.csv`
- Reviewed transcripts: 31
- Reviewer mode: 31 child-worker reviews, 0 manager fallbacks
- Outcome classes: 17 first_gt_last, 5 first_eq_last, 9 first_lt_last_counterexample
- First position judged process-advantaged: 17 yes, 14 no
- Raw private thinking available: 31/31; reflection available: 19/31

## Aggregate read

Across this shard, first-position advantage is real but conditional. In all 17 runs where Agent_1 beat the last-position agent, the review judged Agent_1 process-advantaged, usually because the first public turn made Agent_1's desired bundle the baseline and later phases treated it as a protected constraint. The 14 non-advantaged cases split into 5 ties and 9 counterexamples: these usually show complementary top items, a last-position veto, or private voting rejecting first-speaker overreach.

Counts below are run-level counts from the 31 child review JSONs. Categories are intentionally overlapping because a transcript can show both first-speaker anchoring and countervailing veto/utility constraints.

## Themes

### First-speaker anchoring or protected core: 30/31 runs (72 mechanisms)
Agent_1 often used the first turn to name a preferred item pair or non-negotiable item, and later speakers repeated that frame instead of reopening it.
Representative evidence:
- config_0032 `config_0032.md:23`: "Apple is the clear top item for me (67)."
- config_0032 `config_0032.md:287`: "Apple anchor with Agent_1 remains essential."
- config_0196 `config_0196.md:27`: "I take Jewel and Quill (my top two)."

### Formal/voting lock-in of Agent_1 baseline: 25/31 runs (41 mechanisms)
In many first_gt_last cases, the formal proposals or vote rationales carried the opening frame into the accepted allocation.
Representative evidence:
- config_0502 `config_0502.md:845`: "I propose: {'Agent_1': [3, 2], 'Agent_2': [0, 4], 'Agent_3': [1]}"
- config_0502 `config_0502.md:868`: "giving Agent_1 Quill and Stone"
- config_0502 `config_0502.md:898`: "Proposal #1: 3 accept, 0 reject"

### Last-position veto or counterexample pressure: 29/31 runs (74 mechanisms)
The most common counterweight was Agent_3 making a credible high-value demand; in the counterexamples this demand survived strongly enough for the last position to beat Agent_1.
Representative evidence:
- config_0026 `config_0026.md:1975`: "Any proposal that deprives Agent_3 of Quill+Pencil is unacceptable"
- config_0038 `config_0038.md:96`: "Apple + Quill, which is my top individual combination"
- config_0342 `config_0342.md:214`: "I propose: {'Agent_1': [4], 'Agent_2': [0], 'Agent_3': [1, 2, 3]}"

### Compatibility, equal-utility, or valuation-fit limits: 22/31 runs (32 mechanisms)
Several ties and weak-advantage cases were better explained by compatible preference geometry than by speaker order.
Representative evidence:
- config_1766 `config_1766.md:8`: "final_utilities: {'Agent_1': 100.0, 'Agent_2': 100.0, 'Agent_3': 100.0}"
- config_1922 `config_1922.md:8`: "final_utilities: {'Agent_1': 100.0, 'Agent_2': 100.0, 'Agent_3': 100.0}"
- config_0504 `config_0504.md:109`: "respects each of your stated aims."

### Formal/voting constraints against first-speaker overreach: 27/31 runs (54 mechanisms)
Voting often acted as a utility filter: proposals that left Agent_3 empty or ignored a top item could fail even when Agent_1 proposed first.
Representative evidence:
- config_0346 `config_0346.md:232`: "I propose: {'Agent_1': [0, 2], 'Agent_2': [1, 3, 4], 'Agent_3': []}"
- config_0346 `config_0346.md:286`: "Proposal #2: 3 accept, 0 reject"
- config_0346 `config_0346.md:474`: "Proposal 2 gives me Jewel (61)"

### Conflict displaced or deferred away from Agent_1: 8/31 runs (10 mechanisms)
A smaller but important pattern was that the first-speaker bundle became stable while later conflict moved to Agents 2 and 3 or to future-stage guardrails.
Representative evidence:
- config_0030 `config_0030.md:1496`: "Agent_2 vs Agent_3"
- config_0030 `config_0030.md:1783`: "leaving Agent_3 with no items this round to allow a clear, future rebalancing path"
- config_0506 `config_0506.md:4248`: "Treat Stage 2 as a strictly bounded test"

### Private/reflection internalization of first anchor: 14/31 runs (23 mechanisms)
In multi-round transcripts with raw private thinking/reflection, the private sections often repeated the public anchor as the safe baseline.
Representative evidence:
- config_0032 `config_0032.md:2201`: "The group consistently supports keeping the Apple anchor with Agent_1"
- config_0032 `config_0032.md:2736`: "Apple with Agent_1 continues to be a non-negotiable anchor."
- config_0358 `config_0358.md:1481`: "the most robust baseline is Option C"

## Counterexample pattern

The 9 first_lt_last rows are not random failures of Agent_1. Most still show Agent_1 getting a narrow anchor, but the anchor was only a floor: Agent_3 converted Stone/Pencil/Quill/Jewel claims into a higher-value final bundle, or the group adopted a last-position favorable option for speed and unanimity. The clearest examples are `config_0026`, `config_0038`, `config_0342`, and `config_0346`.

## Files

- `config_1764` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_1764.json`; outcome=first_gt_last; first_minus_last=44.0; advantaged=True
- `config_1766` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_1766.json`; outcome=first_eq_last; first_minus_last=0.0; advantaged=False
- `config_1922` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_1922.json`; outcome=first_eq_last; first_minus_last=0.0; advantaged=False
- `config_1924` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_1924.json`; outcome=first_eq_last; first_minus_last=0.0; advantaged=False
- `config_1926` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_1926.json`; outcome=first_eq_last; first_minus_last=0.0; advantaged=False
- `config_0022` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0022.json`; outcome=first_lt_last_counterexample; first_minus_last=-0.5904899999999991; advantaged=False
- `config_0026` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0026.json`; outcome=first_lt_last_counterexample; first_minus_last=-17.496000000000002; advantaged=False
- `config_0028` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0028.json`; outcome=first_gt_last; first_minus_last=15.7464; advantaged=True
- `config_0030` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0030.json`; outcome=first_gt_last; first_minus_last=54.91557000000001; advantaged=True
- `config_0032` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0032.json`; outcome=first_gt_last; first_minus_last=33.067440000000005; advantaged=True
- `config_0034` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0034.json`; outcome=first_gt_last; first_minus_last=8.100000000000001; advantaged=True
- `config_0036` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0036.json`; outcome=first_lt_last_counterexample; first_minus_last=-5.904899999999998; advantaged=False
- `config_0038` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0038.json`; outcome=first_lt_last_counterexample; first_minus_last=-24.0; advantaged=False
- `config_0040` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0040.json`; outcome=first_lt_last_counterexample; first_minus_last=-2.700000000000003; advantaged=False
- `config_0182` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0182.json`; outcome=first_gt_last; first_minus_last=0.8100000000000023; advantaged=True
- `config_0184` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0184.json`; outcome=first_gt_last; first_minus_last=15.7464; advantaged=True
- `config_0186` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0186.json`; outcome=first_gt_last; first_minus_last=9.1854; advantaged=True
- `config_0188` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0188.json`; outcome=first_gt_last; first_minus_last=33.300000000000004; advantaged=True
- `config_0192` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0192.json`; outcome=first_gt_last; first_minus_last=5.904899999999998; advantaged=True
- `config_0194` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0194.json`; outcome=first_gt_last; first_minus_last=11.700000000000005; advantaged=True
- `config_0196` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0196.json`; outcome=first_gt_last; first_minus_last=72.0; advantaged=True
- `config_0198` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0198.json`; outcome=first_lt_last_counterexample; first_minus_last=-10.53; advantaged=False
- `config_0200` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0200.json`; outcome=first_lt_last_counterexample; first_minus_last=-25.0; advantaged=False
- `config_0342` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0342.json`; outcome=first_lt_last_counterexample; first_minus_last=-49.0; advantaged=False
- `config_0344` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0344.json`; outcome=first_gt_last; first_minus_last=29.0; advantaged=True
- `config_0346` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0346.json`; outcome=first_lt_last_counterexample; first_minus_last=-24.0; advantaged=False
- `config_0356` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0356.json`; outcome=first_gt_last; first_minus_last=74.0; advantaged=True
- `config_0358` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0358.json`; outcome=first_gt_last; first_minus_last=9.720000000000006; advantaged=True
- `config_0502` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0502.json`; outcome=first_gt_last; first_minus_last=31.590000000000003; advantaged=True
- `config_0504` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0504.json`; outcome=first_eq_last; first_minus_last=0.0; advantaged=False
- `config_0506` -> `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews/config_0506.json`; outcome=first_gt_last; first_minus_last=15.411789; advantaged=True
