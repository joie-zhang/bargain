# TTC Objective-Shift Deep Dive

Run root: `/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/ttc_native_scaling_20260502_212943`

This note is a camera-ready-facing synthesis of the test-time compute (TTC) results. The short version is that the raw token scaling plots do not support a simple claim that "more reasoning improves bargaining performance." They do support a more interesting and more defensible claim:

> More deliberation does not mainly make the target agent richer. It makes the agent more explicit about admissibility constraints: will this deal pass, is delay worth it, is the package feasible, and does the agreement avoid giving me negative utility? When the binding constraint is social acceptability, deliberation can make outcomes more balanced. When the binding constraint is feasibility or individual rationality, deliberation can produce refusal, stalemate, or lower payoff. This is why TTC can look socially attractive in examples without producing a clean target-payoff scaling law.

This is the version of the earlier idea I would use in the paper. The tempting sentence was:

> Deliberation changes the objective from "find my best acceptable deal" to "produce a coherent, fair, passable deal."

I would revise it to:

> In many runs, deliberation changes the search problem from unconstrained payoff maximization to a constrained admissibility problem: find a deal that can pass, arrives before delay costs dominate, and does not violate feasibility or individual rationality. That shift is often socially attractive and sometimes welfare-improving, but it is not the same as improving the target agent's payoff.

## What I Read

The TTC run has 216 completed samples: three target families, four effort levels per family, 18 game/order cells per effort. I reviewed the existing full analysis and shard reports, and I used the 12 per-family/per-effort qualitative shard audits to cover every sample:

- Full TTC report: `experiments/results/ttc_native_scaling_20260502_212943/monitoring/test_time_compute_scaling_full_analysis.md`
- Shards: `experiments/results/ttc_native_scaling_20260502_212943/monitoring/qualitative_shards/*.md`
- New mechanism scoring script: `scripts/analyze_ttc_objective_shift.py`
- New matched-delta script: `scripts/analyze_ttc_objective_shift_deltas.py`
- New raw fairness plot script from the earlier pass: `scripts/plot_ttc_fairness_vs_compute.py`

The raw TTC report already shows the central statistical problem. Target payoff does not rise robustly with compute after game-cell/order controls. In the report's fixed-effect slope tests, GPT-5 is near flat, Claude is near flat, and Gemini is weakly negative. The camera-ready text should not claim a simple monotone TTC benefit.

## Mental Model

The qualitative audit makes the aggregate plots look less mysterious. TTC is mixing four qualitatively different regimes:

1. **Ceiling cases.** In complementary or aligned-preference games, both agents can already get close to maximum payoff. More reasoning has little room to improve target payoff.

2. **Passable-deal discipline.** The agent realizes that a slightly worse, immediately acceptable agreement beats continued deadlock or discounting. This is the core "coherent, fair, passable deal" phenomenon.

3. **Feasibility/refusal discipline.** The agent realizes the requested deal is impossible, dominated, or negative utility. This is also deliberation, but it looks bad in payoff plots because it produces late rounds, no consensus, or refusal.

4. **Strategic extraction.** Deliberation can still be used to extract. Some agents bluff, frame a target-favorable package as principled, or exploit opponent accounting mistakes. This is why "more reasoning means more fairness" would be false.

This mixture explains why the raw token plots wash out. The same extra reasoning budget sometimes helps a model compromise, sometimes helps it refuse, and sometimes helps it exploit. Target payoff is the wrong single summary for that mixture.

## Qualitative Evidence

The most convincing qualitative pattern is not generic friendliness. It is explicit reasoning about the constraints under which a deal is admissible.

### 1. Deliberation notices that passability beats nominal optimality

Several runs contain unusually direct statements that a better private deal is worthless if it cannot pass.

| Case | Evidence | Interpretation |
| --- | --- | --- |
| GPT-5 medium, config 42 | The target says the "only realistic supermajority path" is to take the 47 side. See `gpt-5/level_medium/game1_comp_1p0/baseline_first/.../run_1_all_interactions.json:746`. | The agent gives up the better 53 side because the other side will not accept it. |
| GPT-5 medium, config 45 | The target accepts a slightly worse proposal as "acceptable to secure agreement now and avoid delay." See `gpt-5/level_medium/game2_rho_0_theta_1/target_first/.../run_1_all_interactions.json:418`. | The target substitutes "close now" for "push the last few points." |
| GPT-5 high, config 60 | The target calls 47 utility its "acceptable fallback" because it locks value at full round. See `gpt-5/level_high/game1_comp_1p0/baseline_first/.../run_1_all_interactions.json:420`. | Higher deliberation is not pure extraction; it can formalize a floor. |
| Claude low, config 78 | "A better deal that never passes is worth nothing." See `claude-sonnet-4-6/level_low/game1_comp_1p0/baseline_first/.../run_1_all_interactions.json:2350`. | This is the cleanest qualitative statement of the mechanism. |
| Claude medium, config 101 | "Settling in Round 2 at ~90% value is worth far more" than extracting extra points later. See `claude-sonnet-4-6/level_medium/game2_rho_n1_theta_1/target_first/.../run_1_all_interactions.json:687`. | The target explicitly treats time discount as more important than marginal issue advantage. |
| Gemini high, config 204 | "47 in Round 1 is superior to an even split (45) in Round 2." See `gemini-3-flash/level_high/game1_comp_1p0/baseline_first/.../run_1_all_interactions.json:432`. | The same logic appears outside Claude/GPT-5. |

These examples support the qualitative claim, but they also explain why the target payoff plot is not monotone. The agent is not discovering "more target payoff"; it is discovering that *acceptable now* can dominate *better but unavailable later*.

### 2. Deliberation also discovers hard refusal conditions

The second mechanism is just as important. More explicit reasoning sometimes makes the agent reject, not compromise, because the proposed bargain violates feasibility or individual rationality.

| Case | Evidence | Interpretation |
| --- | --- | --- |
| GPT-5 medium, config 54 | "There is no mutually beneficial funded set." See `gpt-5/level_medium/game3_alpha_0p0_sigma_0p2/baseline_first/.../run_1_all_interactions.json:4196`. | The agent correctly identifies a no-deal region. |
| Claude high, config 124 | The project set is "one dollar over our combined 64 budget." See `claude-sonnet-4-6/level_high/game3_alpha_0p5_sigma_0p6/baseline_first/.../run_1_all_interactions.json:476`. | Reasoning uncovers infeasibility, not a better bargain. |
| Gemini medium, config 198 | It is "mathematically impossible to reach the funding threshold." See `gemini-3-flash/level_medium/game3_alpha_0p0_sigma_0p2/baseline_first/.../run_1_all_interactions.json:4360`. | The target refuses because the valued project cannot clear. |
| Gemini high, config 216 | The target says Parkside is "structurally impossible" and "I will not fund" Cedar. See `gemini-3-flash/level_high/game3_alpha_0p0_sigma_0p2/baseline_first/.../run_1_all_interactions.json:4068`. | This is a principled refusal, not a reasoning failure. |

This matters for the paper. If reviewers ask why TTC does not improve payoff, one answer is that the deliberating agent is sometimes doing the *right* strategic thing by refusing a bad or impossible deal. That produces worse payoff metrics in the run, but it is not evidence that the agent is less coherent.

### 3. Deliberation can still be extractive

The fairness story should not be overclaimed. Some high-deliberation traces are strategic and self-serving.

| Case | Evidence | Interpretation |
| --- | --- | --- |
| Claude high, config 114 | The target refers to "my strategic bluff about Apple being my #1 priority." See `claude-sonnet-4-6/level_high/game1_comp_1p0/baseline_first/.../run_1_all_interactions.json:737`. | Reasoning can help the target construct a persuasive extraction frame. |
| Gemini low, config 165 | The target notes that the opponent thinks it is favored, while the actual allocation gives target 86. | Passability can coexist with hidden target advantage. |
| GPT-5 high, config 67 | "Relying on verbal alignment without synchronized formal vectors backfired." See `gpt-5/level_high/game3_alpha_1p0_sigma_1p0/target_first/.../run_1_all_interactions.json:445`. | TTC can improve procedural caution after mistakes, not necessarily fairness. |

This is why I would avoid saying that TTC makes models "fair." The safer claim is that it makes them more explicit about admissibility constraints. Fairness is one possible admissibility constraint, not the only one.

## Quantitative Evidence

I tried the obvious aggregate plots first: target payoff, NBS/Lindahl fairness distance, corrected Gini, payoff difference, and payoff variance against mean observed target tokens/call. Those plots did not give a clean universal trend across GPT-5, Claude Sonnet 4.6, and Gemini 3 Flash.

The useful plots are mechanism-oriented: instead of asking whether tokens directly predict payoff, they ask whether traces that shift toward passable-deal language have different outcomes.

Before reading the plots, the key vocabulary is:

- A **quartile** is one of four equal-sized buckets after sorting. `Q1` is the lowest 25% of runs on the relevant score; `Q4` is the highest 25%.
- **Passable-deal language** means phrases like "acceptable," "agreement," "fair," "compromise," "settle," and "consensus." It is a rough proxy for the agent thinking in terms of "what deal can actually pass?"
- **Self-interest language** means phrases like "maximize," "my utility," "my payoff," "red line," "insist," and "favorable." It is a rough proxy for the agent thinking in terms of "how do I protect or increase my own payoff?"
- **Passable minus self-interest language** is the main objective-shift proxy. High values mean the transcript sounds more like passable-deal search than private-payoff maximization.
- **Corrected Gini** is an inequality score over payoffs. Lower means more equal.
- **Absolute payoff gap** is the size of the payoff difference between target and baseline, ignoring who is ahead. Lower means the two agents ended closer together.
- **Target fair-share excess** asks how far the target agent is from its NBS/Lindahl-style fair-share benchmark. Lower absolute excess means the target is closer to the fairness benchmark.
- **Payoff variance** is another inequality measure. Lower means payoffs are more tightly clustered.

### Plot 1: Within-cell objective-shift residuals

**What this plot asks.** This plot asks: within the same kind of bargaining situation, do runs that sound more like "let's find an acceptable deal" produce different outcomes?

**Why the phrase "within-cell" matters.** A **cell** is one exact comparison group: same target model family, same game setting, and same move order. For example, one cell might be "Gemini 3 Flash, Game 1 identical-preference setting, target moves second." Comparing inside a cell prevents us from confusing two effects: easy games naturally look more cooperative, and some models naturally write in a different style. This plot tries to remove those background differences.

**What a residual means.** A **residual** means "above or below what is normal for this exact cell." For example:

```text
objective-shift residual
= this run's passable-minus-self score
- average passable-minus-self score for runs in the same model/game/order cell
```

The x-axis buckets are based on that residual:

- `Q1`: runs that sounded unusually self-interested for their exact comparison cell.
- `Q4`: runs that sounded unusually passable-deal-oriented for their exact comparison cell.

The y-axis values are residuals too. A residual corrected Gini of `-0.027` means "this bucket is 0.027 lower Gini, meaning more equal, than expected for the same model/game/order setting." The horizontal zero line means "normal for this comparison cell."

**What each subplot means.**

- **Top left, residual target payoff.** This asks whether the target agent got more utility than expected. The line is basically flat from Q1 to Q4 (`+0.05` to `-0.14`). That means the passable-deal shift is not making the target richer.
- **Top middle, residual target - baseline.** This asks who is ahead. Positive means the target is ahead of the baseline; negative means the baseline is ahead. This panel is not the main fairness evidence, because a target can be slightly ahead while the overall outcome is still more equal.
- **Top right, residual absolute payoff gap.** This ignores who is ahead and asks how far apart the two agents are. Lower is more balanced. This falls from `+2.14` to `-2.16`, which means Q4 runs end with smaller payoff gaps than comparable Q1 runs.
- **Bottom left, residual corrected Gini.** This is the clean inequality panel. Lower means payoffs are more equal. It falls from `+0.023` to `-0.027`.
- **Bottom middle, residual |target fair-share excess|.** This asks how far the target is from its fair-share benchmark, regardless of whether it is above or below. Lower means closer to the fairness benchmark. It falls from `+1.22` to `-1.06`.
- **Bottom right, residual payoff variance.** This is another inequality measure. Lower means the agents' payoffs are less spread out. It falls from `+18.96` to `-17.21`.

**Trend and why it matters.** The important pattern is not that every panel is perfectly monotone. The important pattern is that target payoff is flat, while the inequality panels move downward. This supports the paper's claim: passable-deal reasoning is not associated with making the target richer; it is associated with making the outcome more balanced.

The correlations are small (`r = +0.010` for target payoff, `r = -0.139` for absolute gap, `r = -0.120` for corrected Gini, `r = -0.151` for target fair-share excess), so this should be described as exploratory mechanism evidence rather than a strong law.

![Outcome residuals by objective shift](../overleaf/neurips/graphics/ttc_objective_shift_within_cell_residual_quartiles.png)

### Plot 2: Matched weak-to-strong deltas

**What this plot asks.** This plot asks: when we increase TTC in the same bargaining situation, what changes? It is more directly about "scaling compute" than Plot 1.

A **delta** means "stronger setting minus weaker setting." For example:

```text
delta target payoff = strong-effort target payoff - weak-effort target payoff
```

The matched comparisons are:

- GPT-5: minimal -> high
- Claude: low -> max
- Gemini: minimal -> high

The plot then sorts these matched comparisons by how much the target's language changed toward passable-deal reasoning:

- `Q1`: stronger TTC became less passable-deal-oriented, or more self-interested.
- `Q4`: stronger TTC became much more passable-deal-oriented.

The horizontal zero line is important. Values above zero mean the stronger TTC condition increased that outcome; values below zero mean it decreased that outcome.

**What each subplot means.**

- **Top left, delta target payoff.** This asks whether stronger TTC made the target agent richer. In Q4, it is basically zero (`-0.25`). That is the key negative result: the objective shift did not buy higher target payoff.
- **Top middle, delta target - baseline.** This asks whether the target's advantage over the baseline increased. This is a signed gap, so it is about who is ahead, not overall equality.
- **Top right, delta absolute gap.** This asks whether stronger TTC made the two payoffs closer together. In Q4, it decreases by `-7.64`, meaning the payoff gap shrinks.
- **Bottom left, delta corrected Gini.** This asks whether stronger TTC made payoffs more equal. In Q4, it decreases by `-0.110`, which is a substantial movement toward equality.
- **Bottom middle, delta |target fair-share excess|.** This asks whether the target moved closer to its fairness benchmark. In Q4, it decreases by `-2.24`.
- **Bottom right, delta payoff variance.** This asks whether payoff spread decreased. In Q4, it decreases by `-61.60`.

**Trend and why it matters.** The main comparison is Q1 versus Q4. In Q1, stronger TTC did not shift the agent toward passable-deal reasoning, and inequality often got worse: absolute gap `+4.18`, corrected Gini `+0.027`, payoff variance `+103.35`. In Q4, stronger TTC did shift the agent toward passable-deal reasoning, and inequality got better: absolute gap `-7.64`, corrected Gini `-0.110`, payoff variance `-61.60`.

The plain-English reading is: more compute by itself is not the mechanism. More compute matters when it changes the agent's objective from "push my own payoff" toward "find a deal that can pass." When that objective shift happens, the target does not earn more, but the deal becomes more equal.

This is the strongest "TTC changes what the agent is optimizing for" quantitative result. It still should be presented as exploratory because the matched sample is only 54 cells, but it supports the qualitative story better than raw tokens on the x-axis.

![Matched weak-to-strong objective shift](../overleaf/neurips/graphics/ttc_objective_shift_weak_strong_delta_quartiles.png)

### Plot 3: Passable-language quartiles

**What this plot asks.** This plot asks: across all runs, what do outcomes look like when the target agent uses more passable-deal language?

This plot is simpler than the first two. It sorts runs by how much passable-deal language the target agent used. The sorting is done **within each model family**, so GPT-5 is compared with GPT-5, Claude with Claude, and Gemini with Gemini. That avoids treating one model as more "passable" just because it writes longer or uses different wording.

- `Q1`: least passable-deal language within the model family.
- `Q4`: most passable-deal language within the model family.

**What each subplot means.**

- **Top left, target payoff.** This is the target's raw final utility. It rises from `47.34` to `85.11`, so high passable-language runs are also high-payoff runs.
- **Top middle, target - baseline.** This is the signed payoff advantage. It falls from `10.37` to `4.24`, meaning the target is still ahead on average, but less far ahead.
- **Top right, absolute payoff gap.** This is the size of the payoff difference, ignoring who is ahead. It falls from `21.76` to `6.04`, meaning the agents end much closer together.
- **Bottom left, corrected payoff Gini.** This is the inequality measure. It falls from `0.294` to `0.043`, which is a strong descriptive equality trend.
- **Bottom middle, |target fair-share excess|.** This asks how far the target is from the fair-share benchmark. It falls from `5.64` to `2.81`, meaning the target is closer to its fairness benchmark in high passable-language runs.
- **Bottom right, NBS/Lindahl distance.** This asks how far the final outcome is from the relevant cooperative fairness benchmark. Lower is better. This panel is less clean than the Gini/gap panels: the middle quartiles are worse, while Q4 is better than Q2/Q3 but not as low as Q1.

**Trend and why it matters.** The visually clean trend is that passable-language runs have smaller payoff gaps and lower inequality. They also settle more reliably and quickly in the summary table: consensus rises from `0.963` to `1.000`, and mean final round falls from `2.09` to `1.28`.

The caveat is important: target payoff also rises. That means this plot is probably picking up many easy/high-surplus cases where everyone can do well. So this plot should not be used alone to say "TTC sacrifices payoff for fairness." It says something slightly different: passable-deal reasoning marks high-welfare, low-inequality, fast-settling runs.

![Passable language quartiles](../overleaf/neurips/graphics/ttc_passable_language_quartile_outcomes.png)

### Plot 4: Refusal/infeasibility quartiles

**What this plot asks.** This plot asks: what happens when the target agent's transcript contains more refusal or infeasibility language?

It sorts runs by how much the target agent talks about refusal or impossibility: "reject," "infeasible," "impossible," "negative utility," "structurally impossible," and similar language.

- `Q1`: little or no refusal/infeasibility language.
- `Q4`: lots of refusal/infeasibility language.

This is the negative half of the story. More reasoning can make the agent notice that the proposed deal is not acceptable at all. That can be strategically coherent, but it often looks bad in outcome metrics.

**What each subplot means.**

- **Top left, target payoff.** This is the target's raw final utility. It drops from `82.24` to `44.50`, so refusal-heavy runs are much worse for the target.
- **Top middle, absolute payoff gap.** This asks how far apart the agents end up. It rises from `10.16` to `26.69`, so refusal-heavy runs are more unequal.
- **Top right, corrected payoff Gini.** This is the inequality score. It rises from `0.088` to `0.352`, again showing much more inequality.
- **Bottom left, payoff variance.** This is another inequality measure. It rises from `60.62` to `425.99`, which is the strongest visual jump in the figure.
- **Bottom middle, consensus rate.** This is the fraction of runs that reach agreement. It falls from `1.000` to `0.815`, so refusal-heavy runs are less likely to settle.
- **Bottom right, mean final round.** This tells how long the negotiation lasts before ending. It rises from `1.00` to `3.85`, so refusal-heavy runs drag on much longer.

**Trend and why it matters.** The trend is very clear: more refusal/infeasibility language means lower target payoff, larger payoff gaps, higher inequality, lower consensus, and later resolution.

The direct correlations are also strong: refusal language versus target payoff is `r = -0.493` overall, `r = -0.612` for GPT-5, and `r = -0.591` for Gemini. In Claude, refusal language is strongly correlated with payoff variance (`r = +0.656`), absolute payoff gap (`r = +0.583`), and corrected Gini (`r = +0.557`).

The plain-English reading is: these are the runs where deliberation discovers a dead end. The agent may be right that the deal is impossible, dominated, or unfair to itself, but the resulting negotiation is longer, less likely to settle, and more unequal.

This explains why TTC can fail to improve performance while still making the agent more coherent. Coherence sometimes means recognizing that there is no acceptable deal.

![Refusal language quartiles](../overleaf/neurips/graphics/ttc_refusal_language_quartile_outcomes.png)

## Experiments Tried And What To Use

| Analysis | Result | Use in paper? |
| --- | --- | --- |
| Raw target payoff vs compute | No robust positive trend after game/order controls. | Yes, as the negative headline. |
| Raw NBS/Lindahl distance vs compute | No clean trend. Sensitive to game mix and no-deal cases. | Do not lead with it. |
| Raw corrected Gini / variance / payoff gap vs compute | Mixed. Gemini improves on some fairness metrics; GPT/Claude mixed. | Mention only as motivation for mechanism analysis. |
| Passable language vs compute | Mixed by provider. Gemini has a cleaner rise; Claude max is shorter and noisier. | Not as main evidence. |
| Passable-language outcome quartiles | Clear low-inequality/high-consensus descriptive pattern. | Use as supporting descriptive evidence. |
| Within-cell objective-shift residuals | Best controlled evidence: target payoff flat, inequality/excess lower. | Use as main mechanism plot. |
| Matched weak-to-strong deltas | Best TTC-specific evidence: objective shift reduces gap/Gini/variance without payoff gain. | Use as main mechanism plot or appendix. |
| Refusal/infeasibility quartiles | Very clear trend: lower payoff, lower consensus, higher inequality, later settlement. | Use to explain why TTC does not improve payoff. |

## Suggested Camera-Ready Paragraph

Here is a paragraph I would be comfortable putting in the paper:

> Test-time compute did not reliably increase the target agent's payoff. Qualitative inspection suggests that this is not simply noise: additional deliberation often changed what the agent treated as the binding constraint. In many traces, the agent stopped searching for the highest nominal own-payoff proposal and instead reasoned about whether a deal could pass, whether delay would dominate marginal gains, and whether the proposed package was feasible or individually rational. For example, agents accepted lower immediate allocations when "a better deal that never passes is worth nothing," or when a 47-point round-1 agreement dominated a delayed 50-50 split after discounting. This shift is socially attractive when it produces faster, more balanced settlements, but it is not equivalent to higher target payoff. In other traces, the same deliberative discipline produced refusal: agents rejected packages whose valued project was structurally impossible or whose contribution would give the opponent utility while leaving themselves at zero. Mechanism-oriented text analyses support this interpretation. Within matched game/order cells, shifts toward passable-deal rather than self-interest language were nearly unrelated to target payoff, but were associated with lower absolute payoff gaps, corrected Gini, payoff variance, and target fair-share excess. Conversely, refusal/infeasibility language predicted later rounds, lower consensus, lower target payoff, and higher inequality. Thus TTC appears to improve constraint awareness more than it improves bargaining power.

## Recommended Figure Strategy

For the camera-ready, I would not try to rescue the raw TTC plot as the main story. I would use a small mechanism panel:

1. Left: raw target payoff vs compute, showing no robust monotone improvement.
2. Middle: within-cell objective-shift residual quartiles, showing target payoff flat but inequality metrics lower.
3. Right: refusal/infeasibility quartiles, showing the negative mechanism that explains low-payoff high-deliberation cases.

That panel lets the paper say: "Here is why the aggregate TTC result is flat." It is more satisfying than a weak p-value plot because it separates two behaviors that cancel in the aggregate.

## Follow-Up Analyses Worth Running

These are the next analyses I would run if you want to make the claim stronger:

1. **Human-coded mechanism labels.** Convert the 216 shard labels into a CSV with categories: ceiling, passable concession, discount concession, extraction, protocol confusion, infeasibility/refusal, opponent failure. Then plot outcomes by mechanism and effort. This would be stronger than lexicon scoring.

2. **Discount-concession margin.** For accepted deals, compute accepted target utility minus the discounted value of the best visible next-round alternative. This directly operationalizes "accept now because delay dominates."

3. **Cofunding infeasibility index.** In Game 3, flag cases where the target's only/top valued project costs more than the combined single-round budget, and count overfunded/underfunded contribution vectors. This would isolate the refusal mechanism cleanly.

4. **Agreement-integrity error rate.** Count discussion-to-proposal mismatches, vector swaps, and votes where an agent accepts negative computed utility. This can separate deliberative refusal from protocol failure.

5. **Ceiling-adjusted surplus capture.** Compute final payoff as a fraction of each agent's feasible maximum. This prevents ceiling cases from making passable-language runs look better simply because the game was easy.

## Bottom Line

The result I would defend is not "TTC makes bargaining agents fairer" and not "TTC improves target payoff." The defensible result is:

> TTC makes agents more constraint-aware. In favorable cases, that turns bargaining into a passable-deal search and reduces inequality without improving target payoff. In unfavorable cases, the same constraint awareness surfaces infeasibility or individual-rationality violations, producing refusal and worse aggregate outcomes. This mixture explains why Elo predicts stronger adversary performance but TTC does not yield a clean payoff scaling trend.
