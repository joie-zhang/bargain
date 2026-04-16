# Why Payoff Rises With Elo

## Core Pattern

In the current Game 1 batch, the payoff increase with adversary Elo is not symmetric.

- Baseline utility is basically flat with Elo: Spearman `rho = -0.031` on the full consensus-resolved set.
- Adversary utility rises with Elo: Spearman `rho = 0.201`.
- Total payoff also rises, but more modestly: Spearman `rho = 0.124`.
- Utility gap rises with Elo: Spearman `rho = 0.249`.

This means stronger adversaries are not mainly creating more value for everyone equally. They are:

1. getting better outcomes for themselves, and
2. modestly improving joint welfare by avoiding low-quality bargaining failures.

## What Seems To Be Happening

### 1. Stronger models avoid low-value failure modes

Low-Elo adversaries are much more likely to produce parser failures, rigid repetition, and deadlock-like behavior. Those runs are cheap in utility terms because they either fail to reach agreement or converge only after many rounds.

- Low Elo band `<1250`: consensus `0.929`, adversary parse-failure rate `0.406`, adversary stubbornness proxy `0.385`, mean total payoff `118.2`.
- High Elo band `>=1400`: consensus `1.000`, adversary parse-failure rate `0.008`, adversary stubbornness proxy `0.165`, mean total payoff `133.0`.

Qualitatively, this points to a simple story: stronger models are more reliable bargainers. They do not waste rounds on malformed proposals or endless repetition, so more runs end in actual deals.

### 2. Stronger models get their own frame accepted more often

The accepted-proposer pattern shifts with Elo.

- Low Elo: baseline proposal accepted much more often than adversary proposal (`65.2%` vs `27.7%`).
- Mid Elo: adversary proposal accepted slightly more often than baseline proposal (`53.3%` vs `43.6%`).
- High Elo: adversary proposal still wins slightly more often (`53.2%` vs `46.8%`).

So the Elo/payoff slope is partly an extraction story: stronger models are better at steering the final deal toward their own preferred package.

### 3. The welfare gain mostly comes from solving easy-to-trade cases better

The strongest qualitative moderator is preference overlap.

- When there is **no top-item conflict**, total payoff rises sharply with Elo:
  - low: `124.1`
  - mid: `135.8`
  - high: `143.0`
- When there **is** top-item conflict, total payoff stays low across the board:
  - low: `93.4`
  - mid: `89.5`
  - high: `92.5`

This is important. High-Elo models are not dramatically better at resolving hard collisions over the same top item. The main gain is that they are much better at spotting and closing on complementary trades when the structure allows it.

### 4. The best high-payoff style is hybrid compromise, not pure domination

In the labeled subset, the highest-welfare resolved runs are disproportionately:

- `cooperative_exploration` or `balanced_tradeoff` openings
- `responsive_tradeoff` adaptation
- `none` as failure mode
- `hybrid_compromise` as resolution driver
- `neither` as relative stubbornness

By contrast:

- `rigid_repetition` has very low mean total payoff (`79.3`)
- `repetitive_deadlock` is the worst common failure mode (`68.2`)
- `adversary_frame_accepted` resolves often, but with lower mean total payoff (`91.7`) than `hybrid_compromise` (`131.3`)

That suggests two distinct mechanisms:

1. adversary utility rises because strong models can win the frame more often
2. total welfare rises because strong models are also better at responsive package-deal bargaining

## Qualitative Details Worth Auditing

If you want to explain the Elo/payoff slope in the paper, these are the most useful transcript-level details to inspect.

### A. Package recognition

Question: does the model explicitly identify complementary interests and use them to construct a clean split?

Look for:

- explicit statements like "you care about X, I care about Y"
- proposals that move low-value items away and keep high-value items
- one-round or two-round convergence on clearly Pareto-improving bundles

Good exemplar:

- [deepseek-r1](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json:1)

### B. Frame control

Question: does the stronger model get the baseline to accept the stronger model's preferred package, even when the baseline initially resists?

Look for:

- repeated re-use of the same anchor proposal
- baseline discussion that sounds skeptical, followed by acceptance anyway
- final deal matching the adversary's earlier package

Good exemplar:

- [gpt-5.4-high](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json:1)

### C. Failure by repetition

Question: is the payoff low because the model is strategically tough, or because it is just cycling?

Look for:

- identical proposals across many rounds
- identical or near-identical reasoning strings
- no reaction to the opponent's stated priorities

Good exemplars:

- [deepseek-v3](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-v3/weak_first/comp_0.9/turns_2/run_1/experiment_results.json:1)
- [qwen2.5-72b-instruct](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_qwen2.5-72b-instruct/strong_first/comp_1.0/turns_1/run_2/experiment_results.json:1)

### D. Failure by parser breakdown

Question: is low payoff coming from actual negotiation difficulty, or from proposal-generation unreliability?

Look for:

- "failed to parse response" defaults
- proposer-gets-all fallbacks
- discussion text that seems sane while proposals are malformed

Good exemplar:

- [llama-3.2-3b-instruct](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/strong_first/comp_0.9/turns_1/run_2/experiment_results.json:1)

### E. Hard conflict versus easy complementarity

Question: when payoff is low, is that because the run contains genuine top-item conflict?

Look for:

- both sides insisting on the same single high-value item
- inability to compensate with bundles
- discussion that recognizes the collision explicitly

Good exemplar for persistent conflict:

- [claude-opus-4-5-20251101](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_claude-opus-4-5-20251101/strong_first/comp_1.0/turns_2/run_2/experiment_results.json:1)

## Paper-Friendly Claim

A defensible version of the claim is:

> As adversary Elo rises, payoff increases for two separable reasons: stronger models are less likely to waste rounds on malformed or cyclic bargaining, and they are better at identifying and closing package-level trades when preferences are complementary. The increase is not symmetric: baseline payoffs remain roughly flat, while stronger adversaries capture more of the surplus and more often get their preferred frame accepted.

