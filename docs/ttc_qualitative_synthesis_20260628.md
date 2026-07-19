# TTC Scaling Qualitative Synthesis

Date: 2026-06-28

## Sources Read

- `experiments/results/ttc_native_scaling_20260502_212943/monitoring/test_time_compute_scaling_full_analysis.md`
- `experiments/results/ttc_native_scaling_20260502_212943/monitoring/qualitative_shards/*.md`
- `experiments/results/n2_baseline_comparison_analysis_20260505/n2_baseline_comparison_with_ttc_report.md`
- `experiments/results/n2_ttc_multiagent_comparison_analysis_20260505/n2_ttc_multiagent_comparison_report.md`
- `docs/test_time_compute_scaling_proposal_2026_05.md`
- `docs/analysis/GPT5_REASONING_EFFORT_EXPERIMENTS.md`
- `overleaf/neurips/4_analysis.tex`
- `overleaf/neurips/appendix.tex`
- `overleaf/neurips/paper_critiques.md`
- `overleaf/neurips/paper_improvements.md`

## One-Sentence Story

Elo scaling buys a better negotiator; test-time compute mostly buys a longer version of the same negotiator. More reasoning can fix local arithmetic and proposal mistakes, but it does not reliably create bargaining power because the payoffs are usually pinned down by role/order, focal bundles, veto constraints, discounting, and the one-shot mechanics of the protocol.

## The Quantitative Puzzle

Cross-model Elo scaling is strong in the N=2 results: stronger adversaries usually earn more utility, settle faster in Games 1 and 2, and move outcomes closer to efficient frontiers while often capturing surplus above their fair-share benchmark. The TTC sweep was designed to test whether a similar improvement appears when the model family is held fixed and only provider-native reasoning effort is increased.

The answer is basically no. In the 216-run TTC stress test, game-cell/order fixed-effect token-payoff slopes are small and weak: GPT-5 is +0.71 utility per 1k tokens (p=0.667), Claude is +1.02 (p=0.473), and Gemini is -1.69 (p=0.064). Weak-to-strong comparisons are also mixed: GPT-5 minimal to high rises by +3.26 utility on average but improves only 6 of 18 matched situations and worsens 7; Claude low to max rises by +2.27 with 7 improvements and 3 worsens; Gemini minimal to high is essentially flat at -0.05 with 7 improvements and 5 worsens.

That is the key camera-ready tension: model capability and inference-time deliberation are not substitutes in these bargaining games.

## Mechanism

The best explanation is not that reasoning is useless. It is that extra reasoning is local, while bargaining power is often structural.

Elo improves a bundle of abilities at once: parsing the game, tracking roles, respecting utility vectors, identifying feasible trades, anchoring contested issues, writing parseable proposals, and timing acceptance. TTC is narrower. It gives the same model more opportunity to elaborate the stance it already has. If the stance is utility-aware and the game has a crisp exploitable margin, TTC helps. If the stance is fairness-seeking, rejection-averse, or confused about the protocol, TTC produces better-written concessions, longer stalemates, or more polished invalid commitments.

The most interesting qualitative finding is that deliberation often changes the objective from "find my best acceptable deal" to "produce a coherent, fair, passable deal." That is socially attractive, and sometimes welfare-improving, but it is not the same as higher target payoff.

## Evidence Patterns

### 1. Ceiling Cases Leave No Room For Compute

In complementary item-allocation cells, both agents can get full utility. GPT-5 high configs 55 and 56 reach 100/100 against the GPT-5-nano baseline. The high-effort target uses far more tokens than the baseline, but there is no extra surplus to extract after the obvious split is found. Claude max and Gemini high show the same pattern in easy aligned cells.

This matters because aggregate TTC curves average over many cases where the marginal value of more search is mechanically zero. The right lesson is not "reasoning failed"; it is "reasoning solved the problem immediately, and the payoff metric cannot rise beyond the ceiling."

### 2. Order And Focal Bundles Beat Reasoning Depth

GPT-5 high configs 57 and 58 are the cleanest diagnostic. In the same mixed item-allocation cell, the target gets 61 while the baseline gets 88 in one order, then the target gets 88 while the baseline gets 61 when the order/role is reversed. The model is not suddenly smarter in one condition. The side holding Apple/Jewel/Pencil captures the high-utility bundle, while the other side accepts the Stone/Quill bundle as strategically clean but lower value.

Gemini medium configs 185 and 186 repeat this in the identical-preference cell: the target gets 53 when it anchors first and 47 when the baseline anchors first. The model recognizes the loss in the second condition, but accepts 47 now rather than pay a discount to contest the focal bundle. This is a bargaining-mechanism effect, not a search-depth effect.

### 3. More Reasoning Can Become Discount-Aware Concession

The clearest counterintuitive mode is that higher reasoning can make the agent more willing to settle. Gemini high config 204 accepts a 47/53 item-allocation loss because 47 in round 1 beats a discounted even split in round 2. GPT-5 medium config 42 shows the same pattern: additional reasoning supports concession timing, not tougher bargaining.

This is competent under the discount rule. It is also exactly why TTC does not behave like Elo. A stronger model may be better at anchoring early and forcing a target-favorable proposal through. A more deliberative instance of the same model may become better at recognizing that a losing but immediate settlement is rational.

### 4. Diplomacy Rewards Crisp Tradeoffs, Not More Diplomatic Prose

Game 2 shows where TTC can look like capability scaling: when extra thought helps the target lock weighted issue trades and close before discounting. GPT-5 medium in a mixed diplomacy cell holds its anchors and earns 88.52 versus 45.08.

But the same family of cases shows the failure mode. GPT-5 low loses badly after explicitly meeting the opponent's red lines, ending around 50.21 versus 83.17. Gemini high builds a polished "Track B" compromise, moving significantly toward the other side's goals, and ends at 68.60 versus 52.20. That is still a target win, but much less extractive than a harder anchor.

The qualitative punchline is useful for the paper: diplomacy rewards crisp issue trades, not more diplomatic language. Extra reasoning helps when it sharpens the target's weighted priorities. It hurts when it turns the opponent's utility function into a constraint set to satisfy.

### 5. Cofunding Turns Compute Into Infeasibility Detection

Game 3 is the strongest warning against treating tokens as a scalar capability knob. The cofunding protocol is all-or-nothing, contribution vectors are hidden until aggregation, accepted proposals end the game, and rejected proposals do not carry over. Many agents nevertheless discuss staged reciprocity, future-round accumulation, or verbal commitments that cannot bind the final payoff.

In Gemini medium configs 196 and 198, longer reasoning produces increasingly exact objections but not a mutually accepted vector. In Gemini high config 216, the target correctly recognizes that its only-valued Parkside project is structurally impossible and refuses to subsidize Cedar, producing no consensus rather than a bad deal. That is not low-quality reasoning. It is good infeasibility detection, and its payoff is zero.

This is the cleanest place to connect to the "models grind their heels in" hypothesis. The evidence does not support universal stubbornness. It supports conditional intensification: when the model construes the situation as structurally impossible or negative-utility, more reasoning makes the refusal cleaner and more persistent. When it construes the situation as a discount problem, more reasoning makes it settle faster.

### 6. Some Big High-Reasoning Wins Are Opponent Failures

Claude max config 143 is a huge win: target 89, baseline -9. The target gets Cedar funded because the baseline contributes to a project it values at zero, explicitly accepting negative utility for total welfare and feasibility. Config 144 is the mirror image: once the target recognizes that contributing to Cedar gives -9 while zero gives 0, the run backs into an all-zero agreement.

These cases are real outcomes, but they are not evidence for smooth TTC scaling. They show that a high-reasoning target can sometimes exploit a counterparty mistake, while the same structural cell can become no-surplus once the negative utility is recognized. This is why a single large high-effort point should be interpreted as protocol stress evidence, not as a compute-scaling law.

### 7. Token Volume Often Measures Friction

The reports are clear that observed tokens are endogenous. Easy complementary cases close quickly. Hard cofunding cases produce long private reasoning, repeated objections, vector repair, and no-consensus loops. In the appendix stage-token audit, total target tokens in discussion/private/proposal/voting correlate around -0.54 with target utility because harder games consume more calls and larger contexts.

This makes the token axis meaningful but not causal. More tokens often trace friction, not intelligence. The better claim is that requested effort changes deliberation style and error type, while payoff remains mediated by game geometry and protocol execution.

## Does The "Grinding Heels In" Thesis Hold?

Partly, but it needs to be sharpened. The strongest supported claim is not "more tokens make models stubborn." The stronger, more accurate claim is:

> Extra reasoning intensifies the model's current interpretation of the negotiation.

When the model interprets the task as a contested allocation, extra reasoning can make it defend a focal bundle or reject dominated cofunding proposals more consistently. When it interprets the task as consensus under discount pressure, extra reasoning can make it more conciliatory. When it interprets the task as a fairness or total-welfare problem, extra reasoning can rationalize concessions that reduce its own payoff.

That is more interesting than a simple stubbornness story. TTC does not reliably improve performance because it does not reliably point reasoning at self-interested bargaining leverage.

## Camera-Ready Paragraph

The TTC audit suggests that inference-time deliberation is not a drop-in substitute for model capability in bargaining. Across 216 rollouts, higher requested effort changes how models deliberate, but payoffs remain dominated by bargaining geometry and protocol constraints. Extra reasoning helps with local repairs: catching a loose marginal item, correcting a contribution vector, or recognizing a simple feasible agreement. It does not reliably overcome focal-bundle assignment, unanimous acceptability, discount pressure, stochastic proposal selection, or negative-utility cofunding incentives. In several cases, more reasoning makes the agent more effective at settlement rather than extraction: Gemini high accepts a 47/53 split because the immediate 47 exceeds a discounted fair split, and GPT-5 medium shows the same concession-timing logic. In scarce cofunding cells, more reasoning often produces cleaner infeasibility detection, which can yield no agreement rather than higher payoff. Thus Elo and TTC scale different objects: Elo improves broad bargaining competence, while TTC amplifies the model's existing construal of the negotiation.

## Short Research Report Draft

### Thinking Longer Is Not The Same As Bargaining Better

The TTC results are counterintuitive because they break a tempting analogy. In the main N=2 experiments, stronger models earn more. It is natural to expect the same curve inside a model family: ask the model to reason harder, spend more tokens, and get a stronger bargainer. The 216-run TTC stress test does not show that. Within-family token-payoff slopes are weak, and weak-to-strong effort comparisons are mixed. GPT-5 high earns only +3.26 utility over minimal effort on average and worsens in more matched cells than it improves; Gemini's minimal-to-high change is essentially zero.

The qualitative audit explains why. In these games, bargaining success is rarely just a function of deeper search. Many outcomes are set by a small number of structural facts: whether preferences are complementary, who anchors the focal bundle, whether the immediate deal beats a discounted future compromise, whether a proposal actually passes the formal vote, and whether a public-good contribution gives the agent negative utility. Extra reasoning can understand these facts better, but understanding them does not always raise payoff. Sometimes it reveals that the best move is to accept a smaller deal now. Sometimes it reveals that no mutually beneficial cofunding deal exists. Sometimes it merely writes a cleaner justification for the same concession.

This is why the best evidence is in the case studies. In GPT-5 high configs 55 and 56, both agents get 100 because the item preferences are complementary; more compute cannot improve on the ceiling. In configs 57 and 58, the target's payoff flips from 61 to 88 under order reversal, showing that focal bundle assignment dominates reasoning depth. In Gemini high config 204, the target accepts a 47/53 loss because the discount makes 47 now better than a fairer future settlement. In Gemini high config 216, higher reasoning correctly identifies that Parkside is impossible and refuses a dominated Cedar subsidy, producing no consensus. In Claude max configs 143 and 144, the same scarce cofunding structure alternates between a giant target win and an all-zero outcome depending on whether the counterparty accepts or rejects a negative-utility contribution.

The high-level lesson is that TTC changes the style of deliberation, not the strategic environment. More thinking can make a model more accurate, but accuracy can point toward compromise, refusal, or recognition of impossibility. Elo scaling improves broad competence across parsing, role tracking, anchoring, and proposal execution. TTC mostly amplifies the model's existing bargaining stance. That is why the result is not a smooth compute curve: thinking longer makes the agent more articulate about the game it thinks it is playing, but it does not guarantee that the game is being played for higher target payoff.

## Best Supporting Evidence To Add To The Paper

Use one compact appendix table with these rows:

| Mechanism | Cases | Evidence to report |
| --- | --- | --- |
| Payoff ceiling | GPT-5 high 55/56 | Both agents reach 100/100 despite high target token use. |
| Order/focal bundle | GPT-5 high 57/58 | Same cell flips from target 61/baseline 88 to target 88/baseline 61. |
| Discount-aware concession | Gemini high 204; GPT-5 medium 42 | More reasoning chooses immediate lower relative payoff because delay is worse. |
| Diplomacy over-coherence | GPT-5 low red-line loss; Gemini high Track B; GPT-5 medium anchor contrast | Better prose and mutual-acceptability language can reduce extraction. |
| Cofunding infeasibility | Gemini medium 196/198; Gemini high 216 | Long reasoning detects impossible or dominated funding but yields no consensus. |
| Opponent failure | Claude max 143/144 | Giant win depends on the other side accepting negative utility; mirror cell goes all-zero. |

The main text should keep the claim modest: this is one seed and provider token reporting is not uniform. But the qualitative pattern is strong enough to justify the conceptual takeaway that TTC is a protocol stress test, not a monotone compute-scaling law.
