# Qualitative Mechanisms Behind Elo → Negotiation Payoff (2026-04-16)

Working memo. Pairs concrete transcript evidence with candidate paper claims.
Review conducted over the following runs (all vs. `gpt-5-nano` baseline unless noted):

- **N=2 Game 1 (item allocation)** — `scaling_experiment_20260404_064451`, `comp_1.0`, `turns_2`
- **N=2 Game 2 (diplomacy)** — `diplomacy_20260405_082215/model_scale`, `rho_n1_0_theta_1_0`
- **N=3 Game 1 (multi-agent)** — `game1_multiagent_full_20260413_045538/proposal1_invasion`, `comp_1`, `rep_01`

Each transcript cited is reproducible from the file paths given at the start of each block.

---

## TL;DR

The Elo → payoff curve is not produced by a single "strong models are cleverer" mechanism. It is produced by (at least) **four overlapping mechanisms** that switch on at different capability bands:

| Band | Elo | Dominant mechanism that hurts the weaker side |
|---|---|---|
| **Floor** | ≤ ~1220 | Format-level failure (parser fallback, semantic confusion, role-play hallucination) |
| **Lower-mid** | ~1220–1310 | Legible but strategically naïve proposals (accepts lopsided fairness frames) |
| **Upper-mid** | ~1310–1400 | Symmetric anchoring; converges but top-item split is roughly 50/50 |
| **High** | ≥ ~1400 | Active strategic moves: selective preference disclosure, anchor-and-drift, concession-talk control |

Game 1 and Game 2 have overlapping but distinct mechanisms — in Game 2 (diplomacy) the strongest Anthropic-style models sometimes *lose* by broadcasting unilateral concessions. The N=3 multi-agent case amplifies the high-tier advantage: the focal model only needs to sustain one clean anchor while the nanos fail to coordinate with each other.

---

## Mechanism 1 (floor): format-level failures

Evidence run: `gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/comp_1.0/turns_2/run_2/experiment_results.json`

Outcome: `consensus_reached: false`, 10 rounds, **both agents get 0 utility**.

**What actually happened**:

1. **Role-play hallucination.** Agent_1 (Llama-1B) writes both sides of the dialogue in a single turn. Verbatim from round 1, discussion_turn 2:
   > "…I'd like to propose a revised proposal that incorporates this concession…
   > **Agent_2:** I appreciate your willingness to adapt and adjust the plan. I think the Apple ↔ Jewel swap is a great idea, and I'm willing to commit to the plan…"
   > (sent as Agent_1)

   Llama-1B *imagines* Agent_2's response and treats the imagined response as established ground truth in its next message. This destabilises the whole discussion thread — when the real Agent_2 replies, it is already two positions behind.

2. **Parser fallback → "proposer gets all."** In every single round 1–10, Agent_1's formal proposal is unparseable. The system logs: `"Failed to parse response - defaulting to proposer gets all"`. Each "proposal" is therefore `{Agent_1: [0,1,2,3,4], Agent_2: []}` — a full grab. gpt-5-nano votes against it every time, no consensus ever reached, `final_utilities = {}`.

3. **Preference drift.** Values reported by Llama-1B inside its own discussion text shift round-to-round (e.g. Pencil described as 19.0 in one round, then as "low-value" in another). No coherent strategy can emerge.

**Similar but even cleaner floor case**: `gpt-5-nano_vs_amazon-nova-micro-v1.0/strong_first/comp_1.0/turns_2/run_2/`. Preferences identical on both sides (Apple 42, Pencil 29, Jewel 22, Quill 5, Stone 2). In round 1 Nova-Micro proposes — *as its own formal proposal* — giving itself only Quill (utility 5) and giving gpt-5-nano Apple + Jewel + Pencil (utility 93). gpt-5-nano happily votes yes. Final utilities: Nova-Micro = **5.0**, gpt-5-nano = **93.0**, one round to consensus.

> This is a **self-inflicted semantic confusion**: during discussion Nova-Micro was talking about "trading Quill for Stone" and wrote its allocation as *the post-trade endowment it thought it would have*, but the schema expects the literal assignment. The weak model cannot distinguish between a "trade vector" and an "allocation vector."

**Paper claim this supports**: "At the low end of Elo, the adversary under-performs not because its strategy is dominated but because it emits malformed or semantically misaligned outputs that the environment interprets catastrophically. The payoff gap at Elo ≤ 1240 is largely a *protocol gap*, not a *strategy gap*."

---

## Mechanism 2 (lower-mid): accepting lopsided fairness frames

Evidence run: `gpt-5-nano_vs_gpt-4o-2024-05-13/strong_first/comp_1.0/turns_2/run_2/experiment_results.json`

Preferences: identical. Apple=51, Quill=19, Jewel=16, Pencil=12, Stone=2.
Outcome: Agent_1 (gpt-4o, Elo 1345) 69 vs Agent_2 (gpt-5-nano) 31, consensus in round 1.

**What happened**:

1. gpt-4o's opening message subtly misrepresents its preferences:
   > "For me, the Apple is the most important item, and it's something I highly value. **The Quill and Pencil are also items of interest to me, though to a lesser extent. The Jewel and Stone are lower on my priority list.**"

   In fact Jewel is gpt-4o's #3 item (16) — *above* Pencil (12). By grouping "Quill and Pencil" together as mid and "Jewel and Stone" together as low, gpt-4o preemptively defuses any claim nano might later make on Jewel.

2. nano takes the statement at face value and proposes a "secure-top-priority-for-each" structure: *you keep Apple, I keep Quill; then alternating-picks on the rest.*

3. gpt-4o accepts nano's "alternating-picks-I-go-first" frame, which *looks* fair — but given the true value vector, going first on Jewel/Stone/Pencil means grabbing the biggest remaining item (Jewel at 16) with probability 1. gpt-4o then grabs Jewel and Stone, leaving nano with Pencil.

4. Final proposal mirrors the pre-agreed script perfectly and both vote yes.

**Why this is "strategic" rather than just lucky**: gpt-4o only needed three deliberate choices — (a) understate Jewel's rank in its opening, (b) accept nano's "top priority per side" frame, (c) insist on "I pick first" for the remainder. Each step looks cooperative.

**Paper claim this supports**: "Upper-mid-Elo models exploit gpt-5-nano's default *fair-by-equal-top-item* heuristic. When the baseline proposes 'you keep Apple, I keep Quill, let's split the rest fairly,' a capable adversary can steer the remainder by accepting-with-small-tweaks rather than counter-proposing, extracting 60–70% of total value from a fairness narrative."

---

## Mechanism 3 (upper-mid): symmetric anchoring

Evidence run: `gpt-5-nano_vs_claude-haiku-4-5-20251001/strong_first/comp_1.0/turns_2/run_2/experiment_results.json`

Preferences: identical. Jewel=44, Apple=28, Stone=15, Pencil=8, Quill=5.
Outcome: Agent_1 (Haiku-4-5, Elo 1407) 33.46 vs Agent_2 (gpt-5-nano) 32.15 after **5 rounds** (consensus rule unanimous). Raw pre-discount: 51 vs 49.

**What happened**:

- R1: both propose a **mirror-image** "give me Jewel+Pennie, give you the rest" — both reject each other.
- R2–R4: each side nibbles at the edge (adjusting which low-value item goes with Jewel) but **neither concedes the top item**.
- R5: Haiku-4-5 finally folds and both converge on Agent_1 = Apple+Stone+Pencil (51), Agent_2 = Jewel+Quill (49).

Time discount (γ=0.9, 4 round delay) eats ≈35% of value. Both models "paid" the stubborn-anchor cost.

**What's different from Mechanism 2**: at this Elo the adversary does not take bait, but it also does not manipulate the frame. The two models behave symmetrically, and the outcome is basically a 50/50 raw split, eroded by time-discount.

**Paper claim this supports**: "At capability parity with the baseline, negotiations converge to roughly equal raw utility but reach consensus slowly, so time-discount absorbs most of the per-round value. Aggregate Elo-vs-payoff near the middle of the scale mostly reflects time-to-consensus, not allocation quality."

---

## Mechanism 4 (high tier): anchor-and-drift with held top item

Evidence run: `gpt-5-nano_vs_claude-opus-4-5-20251101/strong_first/comp_1.0/turns_2/run_2/experiment_results.json`

Preferences: identical, extremely skewed. **Pencil=72**, Quill=12, Jewel=8, Apple=7, Stone=1.
Outcome: Agent_1 (Opus-4-5, Elo 1468) 57.59 vs Agent_2 (gpt-5-nano) 15.31, consensus in round 4.

**Proposal trace** (Agent_1 / Agent_2):

| Round | Opus proposes | Nano proposes |
|---|---|---|
| 1 | {Quill, Pencil} | {Pencil} only |
| 2 | {Stone, Quill, Pencil} | {Apple, Pencil} |
| 3 | {Jewel, Stone, Pencil} | {Pencil, Apple} |
| 4 | **{Apple, Pencil}** | **{Apple, Pencil}** ← match |

Opus's strategy:
- R1 stakes a strong claim on the single critical item (Pencil, worth 72).
- R2 *expands* the demand bundle (peripheral items added) to create a "give me a lot, then I'll shrink" space for later concessions.
- R3 starts dropping back toward the anchor, conceding peripheral items.
- R4 **matches** nano's R3 proposal exactly, sealing consensus. Opus keeps Pencil unconditionally across all four rounds.

Nano, meanwhile, flops: it moved its ask from "give Opus only Pencil" (R1, unrealistic unilateral demand) to "give Opus Apple+Pencil" (R3). Once Opus offered the same split back, nano had no reason to refuse.

**Paper claim this supports**: "High-Elo models use an *anchor-and-drift* strategy — they hold the highest-weight issue immovable across rounds while conceding peripheral issues. Because the baseline's discount function penalises delay more than it penalises accepting a 79/21 split, the baseline converges to the strong model's anchor."

---

## Mechanism 5 (high tier, Game 2): concession-talk paradox

Evidence run: `diplomacy_20260405_082215/model_scale/gpt-5-nano_vs_claude-opus-4-6/strong_first/rho_n1_0_theta_1_0/experiment_results.json`

Preferences (unit interval):
- Opus-4-6: [0.80, 0.81, 0.16, 0.94, 0.49]
- gpt-5-nano: [0.20, 0.19, 0.84, 0.06, 0.51]

(ρ = −1, θ = 1 → near-mirror preferences; item 4 = fentanyl, max salience, max disagreement.)

Outcome: Opus **47.9** vs gpt-5-nano **65.8**, consensus round 3. *The stronger model loses by ~18 points.*

**What happened**: Opus performatively announces large unilateral concessions on its most-salient issue:

| Round | Opus's fentanyl demand | Reasoning phrasing |
|---|---|---|
| 1 | 78% (ideal 94%) | "dramatic concessions on nuclear warhead reduction… a 39-point sacrifice" |
| 2 | 50% | "a true midpoint compromise" |
| 3 | 35% | "I've moved fentanyl from 78% → 50% → 35% across three rounds - a 59-point concession from my ideal on my most important issue" |

Meanwhile gpt-5-nano moves its fentanyl demand only from 0.06 → 0.06 → 0.25. It holds ground. Final accepted proposal is Agent_2's: fentanyl = 0.25, which is 0.69 from Opus's ideal (catastrophic) but only 0.19 from nano's (mild).

**Mechanism**: Opus's own *narration* of its concessions is a one-way ratchet. Once Opus announces "I moved 59 points," rolling back that concession in R4 would look inconsistent. gpt-5-nano does not narrate its own inaction, so it faces no similar cost. The high-Elo model's superior articulation *against itself*.

This is a **Game 2-specific failure mode** and is consistent with the aggregate data: at ρ=−1/θ=1, `claude-opus-4-6` averages Adv=59.9 while `claude-opus-4-6-thinking` averages Adv=90.8 (and `deepseek-r1-0528` Adv=96.1). The models that do well at high-adversity diplomacy are those that *don't* broadcast their concession history — reasoning-model variants and DeepSeek R1.

**Paper claim this supports**: "In Game 2, the dominant within-band variation is not raw capability but *concession-narration style*. Models trained to 'show their work' on cooperative issues (non-thinking Opus-4.x) leak commitment in the transcript itself; reasoning-chain models and DeepSeek R1 hold their ground because they narrate internal deliberation, not concessions."

---

## Mechanism 6 (N>2): coordination failure of the majority

Evidence run: `game1_multiagent_full_20260413_045538/proposal1_invasion/focal_claude-opus-4-6/n_3/comp_1/rep_01/focal/experiment_results.json` (vs same path ending `.../control/` for the all-nano baseline).

Preferences at comp=1.0 (shared across all 3 agents): Stone=57, Jewel=20, Quill=15, Apple=7, Pencil=1.

**Focal condition** (Agent_1 = Opus-4-6, Agents 2–3 = gpt-5-nano):

| Agent | Model | Items | Raw | Discounted |
|---|---|---|---|---|
| 1 | Opus-4-6 | Stone | 57 | 51.30 |
| 2 | nano | Jewel, Pencil | 21 | 18.90 |
| 3 | nano | Quill, Apple | 22 | 19.80 |

Consensus in **round 2**.

**Control condition** (all three agents = gpt-5-nano):

| Agent | Model | Items | Discounted |
|---|---|---|---|
| 1 | nano | Apple, Jewel | 17.71 |
| 2 | nano | Quill, Pencil | 31.30 |
| 3 | nano | Stone | 10.04 |

Consensus in **round 6**. Note the outcome is *worse* than even a random split: the agent holding Stone (Agent 3) ends up with the lowest utility because the discount kicks in for 5 wasted rounds.

**Mechanism in the focal condition**: Opus makes one argument per round:

- R1 (transcript): "The core issue: All three of us want Stone as our top priority. That means whoever gets Stone is getting a massive share of value, so the remaining items need to be distributed fairly to the other two."
  Opus then rejects Agent_2's "Stone to me" proposal as "extremely lopsided," and rejects Agent_3's similarly.

The two nanos now face a coordination problem: both want Stone, neither can block Opus alone under unanimity, and neither has a reason to trust the other to yield Stone. Opus's proposed solution — "Stone to me, you two get symmetric 21/22 shares of the remainder" — is the unique mutual-second-best for the nanos. They accept.

**Why the all-nano control fails**: each nano proposes "Stone to me, others get scraps." No nano has the meta-argument "one of us will get Stone, the others should split equally." They oscillate for 5 rounds before the item eventually lands on one of them by attrition — and the 5 rounds of delay discount everyone's utility to below what they would have got in a one-round agreement.

**Paper claim this supports**: "In N≥3 settings, the focal model's advantage scales up rather than down: weaker opponents not only fail to resist the focal's anchor, they also fail to coordinate *with each other* on a counter-anchor. The focal model wins by acting as a Schelling point." (Focal avg = 44.0 vs Other avg = 22.2 for Opus-4-6 at n=3, comp=1 across reps.)

---

## Candidate paper claims — consolidated

Paired with evidence from above:

1. **Elo → payoff is multi-mechanism, not one story.** At least four distinct mechanisms stack into the monotone curve; we can now attribute increments to specific Elo bands.

2. **The low-Elo end is mostly protocol failure.** Parser fallback and semantic confusion (Mechanism 1) account for most of the gap below Elo ≈ 1250. A paper framing this as "weak models bargain badly" would be misleading — they fail before bargaining starts.

3. **Fairness heuristics are the vulnerability the mid-tier exploits.** gpt-5-nano uses a "each agent's top item stays with that agent" frame. Against a mid-tier adversary this is mildly lopsided (gpt-4o gets 69 vs 31 in one exemplar). The frame itself is the exploit.

4. **High-Elo payoff gains come from anchor-and-drift, not from bigger initial demands.** Opus-4-5 opens with a *smaller* demand than nano does, but holds the one critical item immovable across rounds. The baseline's time-discount does the rest.

5. **Elo is not the whole story in Game 2.** Narrated-concession ratchet (Mechanism 5) can make a higher-Elo model *lose* to a lower-Elo one. The effect correlates with model family (non-thinking Anthropic > reasoning-chain models in vulnerability).

6. **N≥3 amplifies the high-Elo advantage.** Opus-focal n=3 runs: focal 44.0 vs others 22.2. All-nano control at n=3 takes 6 rounds and lands on an inefficient allocation. The "Schelling point" framing is the candidate mechanism.

---

## Worked caveats / what I would still want to do

- **Cherry-pick risk**: every transcript above was deliberately selected from the head or tail of the utility distribution. The paper needs the deterministic feature-extractor from `GAME1_QUALITATIVE_REVIEW_PLAN` to back these up with full-batch numbers (especially: % of rounds that default to "proposer gets all," % of turns that hallucinate the counterparty, length of discussion before first formal proposal).
- **Game 2 mechanism 5 is striking but based on a single high-adversity setting.** Worth replicating at ρ=0 and verifying that concession-narration *doesn't* hurt at milder adversity.
- **Would want 3–5 more reps of N=3 Opus-focal** at comp=0.9 to confirm Mechanism 6 isn't seed-dependent. Current `rep_01` is striking but n=1 at that specific point.
- **Haven't yet reviewed Game 3 (cofunding) transcripts in this memo.** The N=2 diplomacy mechanism 5 suggests cofunding's "pledge" phase may have its own concession-narration issue worth checking.

## Recommended transcript inclusions for the paper

If the goal is "concrete, vivid, representative":

1. **Nova-Micro self-sabotage** (Mechanism 1) — one paragraph quote of Nova-Micro's "trade" discussion + actual proposal it emitted. Single clearest illustration of the floor.
2. **gpt-4o selective preference disclosure** (Mechanism 2) — three-sentence quote ("Apple is most important… Quill and Pencil also of interest… Jewel and Stone are lower on my priority list") paired with the actual preference vector. The misrepresentation is legible in one line.
3. **Opus-4-5 four-round anchor trace** (Mechanism 4) — the round-by-round proposal table (reproduced above). No need to quote discussion; the proposals speak.
4. **Opus-4-6 diplomacy concession ratchet** (Mechanism 5) — the "fentanyl 78% → 50% → 35%" triplet with Opus's own reasoning text.
5. **N=3 Opus-focal round 1 rejection** (Mechanism 6) — the "extremely lopsided" quote + final allocation + round-2 consensus.

Each is short enough to reproduce inline and each maps to a distinct claim.
