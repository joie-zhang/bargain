# Related Work Deep Dive (2026-04-13)

Source basis: I downloaded and read the cited arXiv PDFs locally on 2026-04-13. This note is meant to help rewrite [background.tex](/scratch/gpfs/DANQIC/jz4391/bargain/background.tex:23), not to serve as a polished literature review.

## Project anchor

Your paper's center of gravity is:

- autonomous LLM-vs-LLM strategic interaction
- negotiation and bargaining tasks
- variation in competition/cooperation structure
- variation in model capability, often proxied by Arena Elo
- individual and collective outcomes
- growing interest in multi-agent settings and test-time scaling

That means the most useful related work is not just "multi-agent" or "game theoretic." It is work that touches at least one of:

- LLM negotiation benchmarks or bargaining behavior
- capability/exploitability asymmetries between models
- inference-time adaptation or test-time compute in strategic interaction
- multi-party negotiation environments relevant to future extensions

## Relevance ranking

1. `The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games`
Reason: closest topical overlap with your current paper. It studies frontier LLM negotiation directly, uses NegotiationArena, and argues that capability gains do not eliminate bias or dominance patterns.

2. `Scaling Inference-Time Computation via Opponent Simulation: Enabling Online Strategic Adaptation in Repeated Negotiation`
Reason: strongest citation for your test-time scaling / reasoning budget / strategic adaptation angle.

3. `A Benchmark for Multi-Party Negotiation Games from Real Negotiation Data`
Reason: strongest citation for your future-work direction on `N > 2` negotiation, especially if you want to motivate richer multi-party settings.

4. `Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation`
Reason: good citation if you want a human-facing deployment paragraph or a multi-party negotiation paragraph. Less direct if the section stays focused on autonomous LLM-vs-LLM play.

5. `Profit is the Red Team: Stress-Testing Agents in Strategic Economic Interactions`
Reason: useful for exploitability and adaptive strategic pressure. Less about negotiation benchmarking per se, but very relevant if you want to frame stronger agents as adversarial counterparts rather than just "better negotiators."

6. `Safety Alignment of LMs via Non-cooperative Games`
Reason: conceptually related through game-theoretic multi-agent training, but not really negotiation related.

7. `The Alignment Waltz: Jointly Training Agents to Collaborate for Safety`
Reason: even farther from your current paper. Multi-agent RL and positive-sum coordination are relevant only at a very abstract level.

## Recommendation summary

If you want a tight, high-signal related work section, I would:

- definitely add `Illusion of Rationality`
- definitely add `Opponent Simulation`
- definitely add `Multi-Party Negotiation Games from Real Negotiation Data`
- add `Choose Your Agent` only if you keep a human-AI / deployment / HCI paragraph
- add `Profit is the Red Team` only if you want a paragraph on exploitability or adaptive strategic pressure
- skip `Safety Alignment via Non-cooperative Games` unless you add a broader "game-theoretic multi-agent alignment" paragraph
- skip `The Alignment Waltz` for the current draft

If the section needs to stay short, my preferred citation set from this list is:

1. `Illusion of Rationality`
2. `Opponent Simulation`
3. `Multi-Party Negotiation Games from Real Negotiation Data`
4. one of `Choose Your Agent` or `Profit is the Red Team`, depending on framing

## Paper-by-paper notes

### 1. Scaling Inference-Time Computation via Opponent Simulation: Enabling Online Strategic Adaptation in Repeated Negotiation
Link: <https://arxiv.org/abs/2602.19309>

Verdict: `Must cite` if you discuss test-time compute, online adaptation, or repeated strategic interaction.

Main idea:

- The paper asks whether LLMs can adapt online in repeated negotiation by spending more inference-time compute rather than updating weights.
- Their method embeds smooth fictitious play into inference.
- They use an auxiliary opponent model to imitate the opponent's time-averaged behavior from history.
- They then do best-of-N candidate generation and rank candidates by simulated rollouts against that opponent model.

Core contributions:

- Formal motivation for why repeated negotiation needs online adaptation rather than a single static policy.
- A concrete inference-time method, `BoN-oppo-simulation`, that allocates compute to opponent modeling plus simulated best response.
- Empirical evidence that this beats several no-training baselines and also beats simpler "just let the model think more" baselines.

Experimental setup:

- Repeated two-player negotiation.
- Two environments: buyer-seller and resource exchange.
- The same opponent is faced over multiple episodes, with history carried forward.
- They compare against baseline prompting, baseline with thinking, BoN with evaluator, BoN with CoT simulation, and several adaptive prompt-based baselines from prior work.

Main findings:

- Their method gives the largest and most consistent performance gains over repeated interaction.
- `BoN-simulation` is often second-best, but explicit opponent modeling improves further.
- More generic "thinking" is not enough: baseline with thinking can underperform the naive baseline.
- When both agents use the method in the more cooperative resource-exchange game, social welfare is highest.

Closest overlap with your project:

- Strongest overlap on the test-time-scaling angle.
- Same general family of strategic language negotiation.
- Relevant to your future-work question about whether more reasoning tokens or more test-time compute necessarily help strategic performance.

Main differences from your project:

- Their focus is repeated online adaptation to one evolving opponent, not cross-model scaling across many adversaries.
- They do not systematically vary competition/cooperation parameters the way you do.
- They do not study Elo-performance relationships.
- They stay in two-player settings.

Best use in your related work:

- Put this in a paragraph on inference-time scaling or strategic adaptation.
- Use it to distinguish your contribution: they study how extra test-time computation helps adaptation in repeated negotiation, whereas you study how model capability and competition structure shape outcomes across a wide family of strategic interactions.

Suggested one-sentence use:

- "Recent work studies inference-time scaling in repeated negotiation by simulating opponent behavior and selecting actions via rollout-based search, showing that extra test-time computation can improve online strategic adaptation without weight updates."

### 2. Choose Your Agent: Tradeoffs in Adopting AI Advisors, Coaches, and Delegates in Multi-Party Negotiation
Link: <https://arxiv.org/abs/2602.12089>

Verdict: `Conditional cite`.

Use it if:

- you want a paragraph on deployment, human-AI interaction, or multi-party bargaining with humans in the loop
- you keep the HCI flavor currently present in `background.tex`

Skip it if:

- the section is narrowed to autonomous LLM-vs-LLM strategic interaction only

Main idea:

- The paper studies how people use AI in three-player bargaining when AI is exposed through different assistance modalities: Advisor, Coach, and Delegate.
- Same underlying LLM capability, different user-control interfaces.

Core contributions:

- A randomized within-subject behavioral experiment with `N = 243` participants in groups of three.
- A clean separation between agent capability and interaction modality.
- Evidence that user preference and welfare do not align.

Experimental setup:

- Three-player chip bargaining game with induced values and objective surplus measurement.
- Each participant plays three games with access to exactly one modality per game.
- The underlying Gemini-based system is scaffolded to be superhuman relative to the all-human baseline.

Main findings:

- Participants most prefer the `Advisor` interface, but they earn the highest gains with the `Delegate`.
- Delegate access is the only modality that noticeably improves group welfare.
- Delegation creates positive spillovers: even non-users in delegate groups benefit.
- The proposed mechanism is "market making": delegated agents inject larger, Pareto-improving offers that humans often would not send.
- Users often prefer control even when it reduces welfare.

Closest overlap with your project:

- Real multi-party negotiation.
- Evidence that AI strategic ability can matter in bargaining.
- Relevant if you want to talk about deployment consequences or multi-party settings.

Main differences from your project:

- Humans are in the loop; your paper studies autonomous agent-agent interaction.
- The main independent variable is assistance modality, not model capability or competition structure.
- It does not study Elo, scaling, or systematic cooperation-competition sweeps.

Best use in your related work:

- Mention as evidence that strong negotiation agents do not automatically translate into better realized outcomes once interface, adoption, and delegation enter the loop.
- Good complement if you want one paragraph on the downstream implications of LLM negotiators in real use settings.

Suggested one-sentence use:

- "In a three-party human-AI bargaining experiment, Zhu et al. find that fully delegated LLM assistance can improve realized surplus and create positive spillovers, even though users prefer higher-control advisory interfaces."

### 3. The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games
Link: <https://arxiv.org/abs/2512.09254>

Verdict: `Must cite`.

This is the closest paper in the list to your current draft.

Main idea:

- The paper revisits `NegotiationArena` using late-2025 frontier models and asks whether better general reasoning leads to rational, unbiased, convergent negotiation behavior.
- Their answer is no.

Core contributions:

- Frontier-model re-evaluation of the NegotiationArena setup.
- Evidence that models diverge into model-specific strategic "signatures" rather than converging toward a common rational equilibrium.
- Evidence that anchoring bias persists even in strong frontier models.
- Evidence of systematic dominance patterns, where some models consistently do better than others.

Experimental setup:

- Three games from NegotiationArena: Buyer-Seller, Multi-turn Ultimatum, and Resource Exchange.
- Models include Gemini 2.5 Pro/Flash, GPT-4.1, GPT-4.1 mini, GPT-4o, Claude Sonnet 4.5.
- They run role-based comparisons, self-play gap analysis, anchoring tests, and pairwise payoff/win-rate matrices.

Main findings:

- Strategic divergence: models occupy different bargaining niches rather than clustering around one equilibrium.
- Anchoring persists strongly: in self-play, normalized first offer and final price remain highly correlated (`rho` about `0.78` for Claude Sonnet 4.5 and `0.91` for Gemini 2.5 Pro).
- Semantic anchoring also appears: opening prices collapse onto salient round numbers like `50` rather than smoothly tracking the underlying valuation.
- Gemini 2.5 Pro often dominates weaker counterparts and extracts asymmetrically favorable outcomes.
- General benchmark strength does not imply negotiation rationality or fairness.

Closest overlap with your project:

- Directly about frontier LLM negotiation.
- Strong overlap on capability asymmetry, exploitation, and bargaining outcomes.
- Supports your claim that strategic performance does not trivially follow from generic capability improvements.

Main differences from your project:

- They inherit the NegotiationArena game family rather than introducing your continuous cooperation-competition parameterization across three environments.
- They do not organize the analysis around Elo or scaling laws across a broad roster.
- They do not systematically vary competition levels.
- Their emphasis is bias and dominance, not individual-vs-collective outcomes under controlled incentive geometry.

Best use in your related work:

- Put this immediately after NegotiationArena and before or after the existing `Zhu2025` discussion.
- Use it as the cleanest "recent concurrent evidence" that frontier-model improvements do not erase strategic pathologies.

Suggested one-sentence use:

- "A recent frontier-model re-evaluation of NegotiationArena finds persistent anchoring biases, model-specific strategic signatures, and systematic dominance patterns, suggesting that stronger general reasoning does not guarantee rational or fair bargaining behavior."

### 4. Safety Alignment of LMs via Non-cooperative Games
Link: <https://arxiv.org/abs/2512.20806>

Verdict: `Probably skip`.

Main idea:

- Safety alignment is framed as a non-zero-sum online game between an Attacker LM and a Defender LM.
- They jointly train both via preference-based online RL in a framework called `AdvGame`.

Core contributions:

- A non-cooperative attacker-defender formulation instead of sequential adversarial training.
- Pairwise preference judging instead of scalar reward judging.
- A trained attacker that becomes a reusable red-teaming tool.

Experimental setup:

- Safety post-training on Qwen and Llama models.
- Utility benchmarks plus safety benchmarks like HarmBench, WildJailbreak, DAN, WildGuardTest.
- Robustness against adaptive jailbreaking attacks such as PAIR, TAP, and GCG.

Main findings:

- The approach improves the safety-utility balance over baselines like Self-RedTeam.
- Pairwise judging, training the attacker, and off-policy EMA all matter.
- The trained attacker is competitive with strong adaptive jailbreak methods.

Closest overlap with your project:

- Both use game-theoretic language for agent interaction.
- Both care about adaptive strategic behavior.

Main differences from your project:

- This is fundamentally a safety-alignment training paper, not a negotiation paper.
- The "game" is attacker-defender co-training, not bargaining under varying incentive alignment.
- No negotiation, no Elo/capability sweep, no cooperation-competition geometry.

Best use in your related work:

- Only cite if you add a broad paragraph on game-theoretic multi-agent alignment methods beyond negotiation.
- Otherwise it will feel like topic drift.

Suggested one-sentence use if you include it:

- "Beyond negotiation, recent work uses explicit attacker-defender game formulations to train language models under adaptive strategic pressure, illustrating a broader move toward game-theoretic alignment methods."

### 5. The Alignment Waltz: Jointly Training Agents to Collaborate for Safety
Link: <https://arxiv.org/abs/2510.08240>

Verdict: `Skip for the current draft`.

Main idea:

- The paper proposes `WaltzRL`, a positive-sum multi-agent RL setup where a conversation agent and a feedback agent collaborate to improve safety while reducing overrefusal.

Core contributions:

- A collaborative two-agent safety alignment framework.
- A `Dynamic Improvement Reward` that rewards the feedback agent based on whether its feedback improves the conversation agent's next response.
- Deployment of both agents at inference time, with feedback triggered adaptively when needed.

Experimental setup:

- Llama-3.1-8B-based conversation and feedback agents.
- Safety and overrefusal benchmarks including WildJailbreak, FORTRESS, StrongREJECT, OR-Bench.
- Comparison against safeguards, single-model RL, and inference-time collaboration without RL.

Main findings:

- Strong simultaneous reduction in unsafe outputs and overrefusal.
- On their reported setup, WildJailbreak ASR drops from `39.0` to `4.6`, and OR-Bench overrefusal drops from `45.3` to `9.9`.
- General capability degradation is small.
- Training makes feedback more selective, reducing unnecessary feedback-triggering.

Closest overlap with your project:

- Multi-agent interaction and positive-sum coordination.

Main differences from your project:

- It is a safety-alignment architecture paper, not a negotiation or bargaining paper.
- The "agents" collaborate on response revision rather than negotiate against or with each other over payoffs.
- No direct relevance to your capability-vs-competition analysis.

Best use in your related work:

- I would not include it unless you explicitly broaden the section to "multi-agent training for strategic or aligned behavior."

### 6. A Benchmark for Multi-Party Negotiation Games from Real Negotiation Data
Link: <https://arxiv.org/abs/2603.14066>

Verdict: `Must cite` if you want your multi-agent future work to look grounded.

Main idea:

- Many real negotiations are not "agree on one final allocation." They unfold as sequences of binding commitments that reshape future possibilities.
- The paper introduces a benchmark for this regime.

Core contributions:

- A configurable generator for multi-party sequential negotiation with binding commitments and terminal-only rewards.
- A reference protocol with partner selection, offer construction, and acceptance.
- A set of diagnostic value-function approximations: myopic reward, optimistic upper bound, pessimistic lower bound.
- Realistic large instances derived from Harvard Negotiation Challenge climate-negotiation position papers.

Experimental setup:

- Synthetic games plus document-grounded large games.
- Small games are evaluated exactly via dynamic programming.
- Large games are evaluated comparatively relative to a no-negotiation baseline.
- Includes a zero-shot LLM negotiator baseline on the large real-world instances.

Main findings:

- Performance is highly regime-dependent; no single value approximation dominates.
- Difficulty rises with stronger mixed-motive conflict and with more all-or-nothing goals.
- Different payoff structures favor different heuristics.
- On the real-life negotiation games, the lower-bound heuristic performs best, myopic reward also works reasonably well, the upper-bound heuristic underperforms, and the zero-shot LLM baseline does very poorly.

Closest overlap with your project:

- Strongest citation for multi-party negotiation environments.
- Very relevant to your future-work direction on `N > 2` settings, coalition structure, and non-monotonic strategic behavior.
- Also useful because it explicitly distinguishes outcome-level bargaining from action-level commitment negotiation.

Main differences from your project:

- Their setting is binding commitments with terminal-only rewards, not your current bargaining mechanisms.
- They assume perfect information in the main protocol.
- The emphasis is benchmark generation and evaluation heuristics, not capability scaling across many LLMs.
- Their LLM baseline is not the main scientific object.

Best use in your related work:

- Use it to motivate why moving from two-player outcome bargaining to true multi-party negotiation is nontrivial.
- Also useful to show that multi-party negotiation has already started to move toward more realistic, path-dependent environments.

Suggested one-sentence use:

- "Recent benchmark work argues that many real multi-party negotiations are better modeled as sequences of binding commitments with terminal-only rewards, and shows that performance in such environments depends strongly on regime-specific structure."

### 7. Profit is the Red Team: Stress-Testing Agents in Strategic Economic Interactions
Link: <https://arxiv.org/abs/2603.20925>

Verdict: `Optional but valuable`.

Use it if:

- you want a paragraph on exploitability, adaptive adversaries, or robustness under strategic pressure
- you want to frame strong models as potentially exploitative rather than merely higher-performing

Main idea:

- The paper proposes `profit-driven red teaming`: instead of hand-designed attacks or LLM-judge labels, optimize an opponent directly for profit using only auditable scalar outcomes.

Core contributions:

- A judge-free, outcome-based red-teaming protocol.
- Four simple economic environments with auditable payoff signals: ultimatum bargaining, first-price auctions, bilateral trade, and provision-point public goods.
- A lightweight hardening procedure that distills exploit traces into prompt rules for the target agent.

Experimental setup:

- Fixed target agents face a baseline attacker and then an optimized attacker found by TAP-style prompt search, ranked by average attacker surplus.
- They study multiple frontier targets across the four games.

Main findings:

- Adaptive profit-optimized attackers extract much more surplus than baseline attackers.
- In ultimatum bargaining, attacker surplus jumps by roughly `+18.85` to `+44.50` across targets.
- The optimized attacker often drives the target into dominated decisions, including negative-surplus agreements when zero was available.
- The learned attacks are often not classic jailbreaks; they look like negotiation tactics: urgency, protocol framing, authority impersonation, deceptive commitments, and anchoring.
- Distilling these traces into prompt rules materially improves the target's outcomes and often turns negative average surplus into positive average surplus.

Closest overlap with your project:

- Very relevant to exploitability and asymmetric strategic pressure.
- Includes bargaining, trade, and public-goods interactions, which line up well with your domains.
- Supports the idea that strategic performance should be evaluated under adaptive opponents, not only static matchups.

Main differences from your project:

- Their main contribution is a red-teaming protocol, not a bargaining benchmark or scaling-law analysis.
- They optimize the attacker directly rather than compare a broad roster of models by capability.
- They do not study competition/cooperation parameters explicitly.

Best use in your related work:

- Good for a paragraph on adversarial strategic evaluation or exploitability in structured economic games.
- Especially useful if you want to emphasize that "winningness" and exploitability are not just static properties of model size or Elo, but depend on adaptive counterparty behavior.

Suggested one-sentence use:

- "Recent work reframes agent robustness in economic interactions as adaptive exploitability, showing that profit-optimized opponents can discover negotiation-style manipulations that static evaluations miss."

## Practical recommendations for rewriting `background.tex`

### Recommended paragraph structure

Paragraph 1: `LLM negotiation benchmarks and bargaining behavior`

- Keep NegotiationArena and the older negotiation papers.
- Add `Illusion of Rationality` here.
- Emphasize that newer work shows frontier models still exhibit strategic bias and asymmetric outcomes.

Paragraph 2: `Capability, exploitability, and strategic asymmetry`

- Keep `Zhu2025` if that is your current asymmetric-negotiation citation.
- Add `Profit is the Red Team` if you want a stronger exploitability framing.
- Distinguish your work by saying you do not only ask whether stronger agents exploit weaker ones, but how this changes as the incentive structure moves from cooperation to competition.

Paragraph 3: `Inference-time scaling and adaptation`

- Add `Opponent Simulation`.
- Contrast their repeated-opponent online adaptation with your capability sweep and your interest in whether more test-time reasoning helps strategic performance in general.

Paragraph 4: `Multi-party negotiation and future extensions`

- Add `Multi-Party Negotiation Games from Real Negotiation Data`.
- Optionally add `Choose Your Agent` if you want a multi-party or deployment-facing bridge.
- Use this paragraph to motivate why `N > 2` is not a trivial extension of bilateral bargaining.

### Papers I would actually cite in the next draft

- `Illusion of Rationality`
- `Opponent Simulation`
- `Multi-Party Negotiation Games from Real Negotiation Data`
- `Choose Your Agent` if you keep the HCI/deployment opening
- `Profit is the Red Team` if you want an exploitability/adversarial-pressure paragraph

### Papers I would not force into the section

- `Safety Alignment of LMs via Non-cooperative Games`
- `The Alignment Waltz`

They are not bad papers. They are just too far from the current paper's scientific core.

## Fast take

If I had to update the section quickly, I would center the revamp around three claims:

1. `Negotiation capability is not the same as rational or fair strategic behavior.`
Support: `NegotiationArena`, `Illusion of Rationality`, your current asymmetric-negotiation citation.

2. `Strategic performance depends not just on model capability but on interaction regime and adaptation opportunities.`
Support: your paper, `Opponent Simulation`, optionally `Profit is the Red Team`.

3. `Moving to true multi-party negotiation changes the problem qualitatively.`
Support: `Multi-Party Negotiation Games from Real Negotiation Data`, optionally `Choose Your Agent`.
