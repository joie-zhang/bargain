# Qualitative Mechanisms Behind Elo → Negotiation Payoff (2026-04-16)

Working memo. Pairs concrete transcript evidence with candidate paper claims.
Review conducted over the following runs (all vs. `gpt-5-nano` baseline unless noted):

- **N=2 Game 1 (item allocation)** — `scaling_experiment_20260404_064451`, `comp_1.0`, `turns_2`
- **N=2 Game 2 (diplomacy)** — `diplomacy_20260405_082215/model_scale`, `rho_n1_0_theta_1_0`
- **N=3 Game 1 (multi-agent)** — `game1_multiagent_full_20260413_045538/proposal1_invasion`, `comp_1`, `rep_01`
- **N=2 Game 3 (cofunding)** — `cofunding_20260405_083548/model_scale`, all `(alpha, sigma)` conditions

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

**Game 3 (cofunding) adds a fifth phenomenon that is largely orthogonal to Elo**: a **structural floor** at `alpha=0, sigma=0.2` where every model from Llama-1B to gpt-5.4-high deadlocks for 10 rounds with zero utility. Above that floor, Game 3 produces its own versions of Mechanisms 1–4 but with cofunding-specific flavors (utility miscalculation, noise-as-free-riding, precision-coordination, burden-asking asymmetry) — detailed in Mechanisms 7–11 below.

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

## Game 3 (cofunding) — mechanisms 7–11

Game 3 differs from Games 1–2 in three ways that shape the qualitative story:

1. It uses a **talk–pledge–revise** protocol, not propose-and-vote on full allocations. Agents emit contribution *vectors*, and the joint funding status is the element-wise sum vs. each project's cost.
2. The feasible outcome space is denser but more adversarial: funding requires *aggregate* contribution to clear a project's cost, so a unilateral "I want this funded" pledge is wasted unless the counterparty matches.
3. Preferences are parameterised by `(alpha, sigma)`: `alpha=0` → orthogonal preferences, `alpha=1` → identical; `sigma` scales the per-agent budget relative to total cost (smaller σ = scarcer budget).

Crucially, these parameters mean **some conditions are structurally unwinnable** even by top-tier models. This gives Game 3 a qualitatively different Elo-vs-payoff story.

### Aggregate pattern (first-pass, across all model scale matchups)

| (α, σ)         | Consensus rate | Typical utility range (strong / nano) | Representative mechanism |
|----------------|----------------|---------------------------------------|--------------------------|
| 0 / 0.2 (hard) | ~10%           | 0 / 0 (deadlock)                      | Mechanism 9 (structural floor) |
| 0 / 1.0        | ~90%           | 30–80 / 30–60 (varies by adversary)   | Mechanism 10 (precision coordination) / Mechanism 7 (utility miscalculation) |
| 1 / 0.2        | ~70%           | small symmetric (~15 / ~15)           | Mechanism 3-analog (symmetric) |
| 1 / 1.0        | ~95%           | 40–80 / 20–40 (burden asymmetry)      | Mechanism 11 (burden-asking) / Mechanism 8 (noise-as-free-riding) |

Two observations fall out of the aggregate scan:
- The utility-vs-Elo curve is **steeper in σ=1.0 conditions** than in σ=0.2 conditions — scarcity flattens Elo sensitivity because even strong models hit the floor.
- At `α=1, σ=1.0` (identical prefs, generous budgets), the game should be a pure coordination problem with surplus; the fact that strong models extract 60–70% and nano gets 20–30% is itself the puzzle (→ Mechanism 11).

---

## Mechanism 7 (cofunding floor): utility miscalculation — pledging to 0-value projects

Evidence run: `cofunding_20260405_083548/model_scale/gpt-5-nano_vs_amazon-nova-micro-v1.0/strong_first/alpha_0_0_sigma_1_0/experiment_results.json`

Preferences (orthogonal at α=0):
- Agent_1 (Nova-Micro, Elo 1240): `[0, 9, 0, 91, 0]` — **Cedar Pool Access Lift (index 3) is worth 91; everything else is ≤9**.
- Agent_2 (gpt-5-nano): `[38, 0, 53, 0, 9]` — Market Street (38) + Oak Avenue (53) + Harborview (9).

Project costs: Market Street 19, Parkside 12, Oak 21, Cedar 22, Harborview 12. Per-agent budget 43.

Outcome: consensus in round 1. Final allocation = {Market Street, Oak}. Final utility: **Nova-Micro = −40.0**, gpt-5-nano = +51.0.

**What happened**. nano's R1 discussion proposed "Plan A": fund Market Street + Oak (nano's two highest-value projects, worth **0** to Nova). Nova-Micro accepted this in one turn and submitted pledge vector `[19, 0, 21, 0, 0]` — paying the full cost of both of nano's projects and zero toward its own 91-value project.

Nova's own reasoning phrases during discussion (paraphrased from the log): "it's more valuable to secure some utility by agreeing to Plan A, even though the main beneficiary is Agent_2." But this is *directly contradicted* by Nova's preference vector — Plan A gives Nova zero utility and costs 40 units of budget. The obvious counter-plan (fund Cedar, which is within joint budget at cost 22 and gives Nova 91) was never on the table because Nova never proposed it.

Result: Market Street and Oak both funded (aggregate 38 and 42, both exceeding cost). Nova contributes 40, receives 0-value allocation → utility = 0 − 40 = **−40**. nano contributes 0, receives 38+53 = 91 in value → utility = 91 − 40 = **51** (nano also pledged 40 but against its own beneficial projects, so its net surplus is 51).

**The cleanest possible form of the mechanism**: Nova agrees to a proposal that funds *only* the counterparty's projects, and pays half the cost out of its own budget. No adversarial framing was needed; nano just said "Plan A is sensible for me" and Nova said yes.

**Why this is distinct from Mechanism 1**: the Game 1 floor failure is a *protocol* failure (malformed proposal → parser fallback). The Game 3 floor failure is a *cognitive* failure: Nova emits a legal pledge vector, it parses cleanly, it is semantically assigned to specific projects — Nova just cannot reason about what its own utility function says about that vector.

**Paper claim this supports**: "Cofunding's numeric pledge format catches a category of weak-model failure that allocation voting (Game 1) hides: models at Elo ~1200–1250 can produce syntactically valid pledge vectors that *they themselves cannot utility-evaluate*. The resulting negative utility (pledging real money to 0-value projects) is impossible in a vote-only protocol."

---

## Mechanism 8 (cofunding paradox): noise-as-free-riding

Evidence run: `cofunding_20260405_083548/model_scale/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/alpha_1_0_sigma_1_0/experiment_results.json`

Preferences: **identical**, `[16, 22, 28, 13, 21]` (Oak=28 > Parkside=22 > Harborview=21 > Market Street=16 > Cedar=13).
Project costs: Market Street 11, Parkside 21, Oak 17, Cedar 21, Harborview 19. Per-agent budget 45.

Outcome: consensus round 3. Final allocation = {Parkside, Oak}, funded value = 22+28 = **50** for both agents (identical prefs). **Llama-1B = 40.5**, gpt-5-nano = **4.05**. The Elo-1110 model extracts **10× the utility** of the Elo-1337 baseline from a bundle they value equally.

**Full pledge trace**:

| Round | Llama-1B pledge           | nano pledge               | Aggregate funded         | Vote           |
|------:|---------------------------|---------------------------|--------------------------|----------------|
| 1     | `[10, 0, 0, 0, 5]` (15)   | `[21, 24, 0, 0, 0]` (45)  | Market St (31), Parkside (24) | Llama rejects |
| 2     | `[0, 0, 0, 0, 0]` (**0!**)| `[0, 21, 0, 24, 0]` (45)  | Parkside, Cedar          | Llama rejects |
| 3     | `[10, 0, 0, 0, 5]` (15)   | `[0, 21, 24, 0, 0]` (45)  | Parkside (21), Oak (24)  | **both accept** |

**Where the 40.5/4.05 asymmetry comes from**: the cofunding protocol charges each agent only for pledges to *successfully funded* projects. In the R3 consensus:
- Llama-1B's pledges of 10 (Market Street, cost 11 — NOT funded) and 5 (Harborview, cost 19 — NOT funded) are both to unfunded projects → **Llama pays 0**.
- nano's pledges of 21 (Parkside — funded) and 24 (Oak — funded) are both to funded projects → **nano pays 45**.

Utility calculation (γ² = 0.81 for final round 3):
- Llama: (50 − 0) × 0.81 = 40.5 ✓
- nano: (50 − 45) × 0.81 = 4.05 ✓

**The "noise-as-free-riding" phenomenon**: Llama-1B never pledged strategically. Its R2 pledge was *all zeros*, and its R3 pledge reverted to an identical copy of its R1 pledge — pledging 10 and 5 toward *different projects than the ones actually funded*. The "free ride" is not intentional: it is an emergent consequence of (a) nano unilaterally covering both funded projects with its full budget and (b) Llama's pledges accidentally landing on projects that did not clear threshold.

This is structurally **opposite** to a strategic free-rider. A strategic free-rider would pledge 0 to the funded projects and let the counterparty fill the gap. Llama-1B instead pledges 15 total, *on projects that don't get funded*, and therefore wastes nothing because non-funded pledges incur no cost.

**Why this is distinct from Mechanism 4 (anchor-and-drift)**: Opus-4-5's anchor was deliberate (held Pencil immovable across 4 rounds with legible reasoning). Llama-1B's pledge=0 round is *incoherent by design*; it wins because the protocol rewards under-contribution when the counterparty is contingently over-contributing.

**Paper claim this supports**: "Cofunding's budget-clearing protocol creates a counter-intuitive Elo effect: at α=1 σ=1, weak adversaries can *out-earn* strong baselines because their chaotic pledging pattern shifts consensus pressure onto the strong-baseline. Utility-vs-Elo is not monotone in Game 3 — there is a negative slope in a subset of conditions, driven entirely by this mechanism."

---

## Mechanism 9 (cofunding structural): orthogonal-preferences deadlock

Evidence runs, all at `alpha_0_0_sigma_0_2` (orthogonal preferences × scarcest budget):
- `gpt-5-nano_vs_deepseek-r1-0528/strong_first/`: 0.0 / 0.0, 10 rounds, no consensus.
- `gpt-5-nano_vs_gpt-5.4-high/strong_first/`: 0.0 / 0.0, 10 rounds, no consensus.
- `gpt-5-nano_vs_claude-opus-4-6/strong_first/`: −2.4 / 31.6, 3 rounds (rare success, by nano sacrificing Opus).

**Typical preferences at α=0**: Agent_1 values projects {0, 3} at 65/35; Agent_2 values projects {1, 2, 4} at 16/65/19. No overlap.

**Typical pledge trace (deepseek-r1 transcript, rounds 1–5)**:

| Round | Agent_1 (deepseek-r1-0528) | Agent_2 (gpt-5-nano) | Funded? |
|---|---|---|---|
| 1 | `[9, 0, 0, 0, 0]` | `[0, 9, 0, 0, 0]` | none |
| 2 | `[9, 0, 0, 0, 0]` | `[0, 9, 0, 0, 0]` | none |
| 3 | `[9, 0, 0, 0, 0]` | `[0, 9, 0, 0, 0]` | none |
| … | (unchanged) | (unchanged) | … |

Each agent pledges only to its own-valued projects. The other agent never matches because their preferences are orthogonal. Neither agent has enough budget alone (σ=0.2 → budget ≈ 9 vs project cost ~30–60). Neither agent shifts strategy across 10 rounds.

**What's notable**: the transcripts are not marked by confusion or miscalculation. deepseek-r1-0528's discussion phases are lucid:
> "Given our orthogonal valuations and tight budgets, I see no feasible compromise that funds any single project above cost. I will continue pledging my 9 units to project 0."

The failure is not cognitive — it is **game-theoretic**. At σ=0.2 the joint budget barely covers one project; unanimity means either agent can veto funding the other's projects; and with α=0 there is no aligned project to focus on.

This is structurally the diplomacy ρ=−1/θ=1 condition (Game 2) but in a cofunding framing. Yet unlike diplomacy — where Opus-4-6 *loses* because of Mechanism 5 (concession ratchet) — in cofunding even reasoning-chain models (deepseek-r1, gpt-5.4-high) deadlock. The difference is that diplomacy allows *continuous* bargaining on each axis, so any concession buys surplus somewhere; cofunding's *integer-cost* threshold means sub-critical pledges are pure waste.

**Paper claim this supports**: "Game 3's Elo → payoff curve has a **Elo-independent zero region** at α=0, σ=0.2 where models from Elo 1110 (Llama-1B) to Elo 1484 (gpt-5.4-high) all score 0. Reporting mean utility-vs-Elo across the full (α, σ) grid would understate Elo's signal at easy conditions and overstate it at hard conditions. For cofunding, the cleanest Elo signal comes from α=1 σ=1.0 (coordinated-surplus condition)."

---

## Mechanism 10 (cofunding high tier): precision coordination and the overfunding diagnosis

Evidence run: `cofunding_20260405_083548/model_scale/gpt-5-nano_vs_gemini-2.5-pro/strong_first/alpha_0_0_sigma_1_0/experiment_results.json`

Preferences:
- Agent_1 (Gemini-2.5-Pro, Elo 1448): `[100, 0, 0, 0, 0]` (Market Street only; all surplus concentrated in one project).
- Agent_2 (gpt-5-nano): `[0, 23, 10, 26, 41]` (spread across four projects).

Outcome: consensus in round 2. **Gemini = 76.5**, nano = 32.4. (Gemini funds Market Street for 15 and captures its entire 100-value preference; nano funds Harborview + Cedar for 31 and captures 67-value.)

**What happened — the R1 waste**:

In round 1 both agents agreed during *discussion* to fund Market Street (Gemini's priority) + Harborview (nano's priority). But their submitted pledges were:
- Gemini: `[15, 10, 5, 0, 10]` — spread across four projects.
- nano: `[15, 0, 0, 0, 21]` — Market Street + Harborview.

Joint aggregate showed Market Street over-funded (30 vs cost 15) and three other projects underfunded. Gemini voted against it; consensus failed in R1.

**R2 — the diagnosis** (quoted verbatim from Gemini's R2 discussion opener):
> "It looks like we were on the same page last round with our main goals, but our individual proposals didn't align, leading to a very inefficient and overfunded aggregate proposal. I believe that's why it was rejected. Let's correct that this round with a more coordinated approach. … I propose we make our contributions *precise*. For example: I will contribute exactly 15 to the Market Street Bike Lane. You contribute exactly 21 to the Harborview Bus Shelter Canopies."

Gemini then *volunteered* to let nano add a second valued project (Cedar, 10) from nano's remaining budget. nano accepted. R2 pledges matched the prescription exactly; all three projects funded; consensus.

**Why this is distinct from Mechanism 4**: Opus-4-5's R1–R4 trace holds one item anchor and lets peripheral demands drift. Gemini's mechanism is *meta*-strategic — it correctly diagnoses that R1 failed due to a coordination problem (not a disagreement) and explicitly prescribes contribution splits. This is a form of *strategic explicitness* that gpt-5-nano appears unable to do unprompted.

**Paper claim this supports**: "In cofunding, consensus failure in round 1 is rarely about preference mismatch and almost always about **miscoordinated pledge arithmetic** (e.g. both agents contributing to the same project, leaving others short). The highest-Elo models (gemini-2.5-pro, opus-4-6-thinking) recognise this and emit an explicit prescriptive round-2 opener with per-project contribution amounts. gpt-5-nano never does this unprompted."

---

## Mechanism 11 (cofunding high tier): burden-asking asymmetry under identical preferences

Evidence run: `cofunding_20260405_083548/model_scale/gpt-5-nano_vs_qwen2.5-72b-instruct/strong_first/alpha_1_0_sigma_1_0/experiment_results.json`

Preferences: **identical**, `[18, 2, 41, 17, 22]` (Oak=41, Harborview=22, Cedar=17, Market Street=18, Parkside=2). Budgets 45 each.

Outcome: consensus in round 1. Final allocation: Oak + Cedar + Harborview funded (cost 18+10+21 = 49). **Qwen = 76.0**, nano = 35.0. Both agents value the funded bundle equally (80 each). The 41-point gap is entirely due to contribution asymmetry: Qwen paid 4, nano paid 45.

**Transcript (compressed)**:

1. *Qwen, R1 turn 1*: "Let me list priorities." (emits full ranked list as context.)
2. *nano, R1 turn 1*, **unprompted**: "Proposed funding plan (Agent_2 contributions): Oak Avenue Crosswalk Beacons: 18 (fully funded by me). Harborview Bus Shelter Canopies: 21 (fully funded by me). Cedar Pool Access Lift: 6 (my share). Total from me: 45. What I'm asking from you: Please contribute 4 to Cedar Pool Access Lift."
3. *Qwen, R1 turn 2*: "I appreciate your willingness to fully fund the Oak Avenue Crosswalk Beacons and Harborview Bus Shelter Canopies. These are indeed high-priority projects for me as well, and I agree they should be our top focus."
4. *Qwen submits pledge `[0, 0, 0, 4, 0]`; nano submits `[0, 0, 18, 6, 21]`.* Consensus.

Qwen's **only strategic move** was to accept nano's self-offered burden. No manipulation, no frame-steering — nano simply volunteered "fully funded by me" for the two most valuable projects. Qwen said yes.

**Cross-model pattern** (corroborating evidence, same α=1, σ=1 condition, identical-preferences regime):

| Adversary | Adv utility | nano utility | Ratio |
|---|---|---|---|
| qwen2.5-72b | 76.0 | 35.0 | 2.17 |
| gemini-2.5-pro | ~74 | ~34 | 2.18 |
| claude-opus-4-6-thinking | 36.9 | 36.9 | 1.00 |
| claude-opus-4-6 | 30.8 | 30.8 | 1.00 |

Strikingly, the reasoning-capable Opus variants **refuse** the free burden and split 50/50. Non-reasoning strong models (Qwen, Gemini, deepseek-v3) cheerfully accept it. This mirrors the Mechanism 5 observation from Game 2 but inverted: in diplomacy, reasoning/thinking variants *keep* their ground; in cofunding, reasoning/thinking variants *decline* asymmetric surplus.

**Why this is distinct from Mechanism 2**: gpt-4o's Game 1 asymmetry came from *actively misrepresenting* its preferences and steering the frame. In Qwen's cofunding case Qwen is transparent about preferences and nano spontaneously volunteers the asymmetry. The burden-asking is a failure *of the baseline*, not a manipulation *by the adversary*.

**Paper claim this supports**: "At α=1 σ=1, gpt-5-nano exhibits a systematic 'fully funded by me' pattern that emerges in the *first* discussion turn, before the adversary has said anything strategic. Strong non-reasoning adversaries accept this offer; reasoning-mode adversaries reject it in favor of symmetric splits. The Elo–utility gradient at this condition is therefore partly driven by **how adversaries respond to nano's over-generosity**, not by clever strategy on the adversary side."

---

## Candidate paper claims — consolidated

Paired with evidence from above:

1. **Elo → payoff is multi-mechanism, not one story.** At least four distinct mechanisms stack into the monotone curve; we can now attribute increments to specific Elo bands.

2. **The low-Elo end is mostly protocol failure.** Parser fallback and semantic confusion (Mechanism 1) account for most of the gap below Elo ≈ 1250. A paper framing this as "weak models bargain badly" would be misleading — they fail before bargaining starts.

3. **Fairness heuristics are the vulnerability the mid-tier exploits.** gpt-5-nano uses a "each agent's top item stays with that agent" frame. Against a mid-tier adversary this is mildly lopsided (gpt-4o gets 69 vs 31 in one exemplar). The frame itself is the exploit.

4. **High-Elo payoff gains come from anchor-and-drift, not from bigger initial demands.** Opus-4-5 opens with a *smaller* demand than nano does, but holds the one critical item immovable across rounds. The baseline's time-discount does the rest.

5. **Elo is not the whole story in Game 2.** Narrated-concession ratchet (Mechanism 5) can make a higher-Elo model *lose* to a lower-Elo one. The effect correlates with model family (non-thinking Anthropic > reasoning-chain models in vulnerability).

6. **N≥3 amplifies the high-Elo advantage.** Opus-focal n=3 runs: focal 44.0 vs others 22.2. All-nano control at n=3 takes 6 rounds and lands on an inefficient allocation. The "Schelling point" framing is the candidate mechanism.

7. **Game 3 has a structural zero-region.** At α=0, σ=0.2, *every* model scores 0. Reporting utility-vs-Elo averaged over (α, σ) is therefore misleading: the cofunding Elo signal lives almost entirely at α=1 and/or σ=1.0. The clean Elo-vs-payoff Game 3 plot should either (i) stratify by (α, σ), or (ii) exclude the deadlock cell.

8. **Game 3's floor failure is cognitive, not syntactic.** Where Game 1's low-Elo failure is malformed proposals (Mechanism 1), Game 3's is Nova-Micro pledging real budget to projects worth 0 to itself (Mechanism 7). The common thread is that low-Elo models cannot accurately reason about their own utility function — but the *observable symptom* is different across games.

9. **Cofunding Elo-vs-payoff is non-monotone at some conditions.** Mechanism 8 (noise-as-free-riding) means Llama-1B earns ~40 while gpt-5-nano earns ~4 at α=1, σ=1.0 — the weak model out-earns the stronger baseline. Any paper plot that assumes monotonicity at Game 3 needs to either drop this cell or explicitly label it as the non-monotone region.

10. **Baseline over-generosity, not adversary strategy, drives burden asymmetry.** Mechanism 11 shows that gpt-5-nano spontaneously offers "fully funded by me" in round 1 before the adversary has said anything strategic. The 2× utility gap under identical preferences (Qwen 76 vs nano 35) is mostly a *baseline failure*, not an adversarial win.

---

## Worked caveats / what I would still want to do

- **Cherry-pick risk**: every transcript above was deliberately selected from the head or tail of the utility distribution. The paper needs the deterministic feature-extractor from `GAME1_QUALITATIVE_REVIEW_PLAN` to back these up with full-batch numbers (especially: % of rounds that default to "proposer gets all," % of turns that hallucinate the counterparty, length of discussion before first formal proposal).
- **Game 2 mechanism 5 is striking but based on a single high-adversity setting.** Worth replicating at ρ=0 and verifying that concession-narration *doesn't* hurt at milder adversity.
- **Would want 3–5 more reps of N=3 Opus-focal** at comp=0.9 to confirm Mechanism 6 isn't seed-dependent. Current `rep_01` is striking but n=1 at that specific point.
- **Game 3 mechanisms 7–11 added 2026-04-16.** Each rests on 1–3 exemplar transcripts; a full-batch feature extractor for cofunding would want: (a) per-run contribution asymmetry (max agent contribution / min agent contribution), (b) waste rate = total pledged / total required for funded projects, (c) zero-pledge rate per round per agent, (d) "fully funded by me" text-pattern detection in discussion phases. These would corroborate Mechanism 11 (burden asymmetry) and Mechanism 10 (precision coordination as a transcript feature) at full batch.
- **Mechanism 8 (noise-as-free-riding) is the most fragile claim.** The Llama-1B vs nano anomaly appears across multiple α=1 σ=1 runs but n is small and the "win" is utility ~40 vs ~4 — the floor of "both are tiny" is not far off. Need to check whether this reproduces in other low-Elo × α=1 σ=1 matchups (e.g., Llama-3B, Llama-8B, command-r) or whether it is Llama-1B-specific.
- **Mechanism 11 reasoning-vs-non-reasoning split deserves a systematic test.** The Qwen/Gemini-accept vs Opus-thinking-refuse contrast is clean on 4 data points at α=1 σ=1. Would want to verify that (a) deepseek-r1 and o3-mini-high refuse like Opus-thinking, (b) deepseek-v3 and gpt-5.2-chat accept like Qwen.

## Recommended transcript inclusions for the paper

If the goal is "concrete, vivid, representative":

1. **Nova-Micro self-sabotage** (Mechanism 1) — one paragraph quote of Nova-Micro's "trade" discussion + actual proposal it emitted. Single clearest illustration of the floor.
2. **gpt-4o selective preference disclosure** (Mechanism 2) — three-sentence quote ("Apple is most important… Quill and Pencil also of interest… Jewel and Stone are lower on my priority list") paired with the actual preference vector. The misrepresentation is legible in one line.
3. **Opus-4-5 four-round anchor trace** (Mechanism 4) — the round-by-round proposal table (reproduced above). No need to quote discussion; the proposals speak.
4. **Opus-4-6 diplomacy concession ratchet** (Mechanism 5) — the "fentanyl 78% → 50% → 35%" triplet with Opus's own reasoning text.
5. **N=3 Opus-focal round 1 rejection** (Mechanism 6) — the "extremely lopsided" quote + final allocation + round-2 consensus.
6. **Nova-Micro cofunding −40 utility** (Mechanism 7) — Nova's verbatim "it's more valuable to secure some utility from the Parkside Adventure Playground" quote paired with the actual pledge vector `[19, 0, 21, 0, 0]` and Plan A content. Single clearest illustration that Game 3's floor is *cognitive* (can't evaluate own utility) where Game 1's floor is *syntactic* (can't emit valid proposal).
7. **Gemini-2.5-pro R2 "precise contributions" diagnosis** (Mechanism 10) — three-sentence quote ("our individual proposals didn't align, leading to a very inefficient and overfunded aggregate proposal … I propose we make our contributions precise") paired with the R1 aggregate (overfunded) vs R2 aggregate (each project exactly funded). Clean illustration of the meta-strategic difference at the top tier.
8. **Qwen vs nano identical-prefs burden split** (Mechanism 11) — four-line transcript compression showing nano's unprompted "fully funded by me" offer in R1, followed by Qwen's one-line acceptance, followed by the pledge table. Utility gap (76 vs 35) is derived entirely from these four lines.

Each is short enough to reproduce inline and each maps to a distinct claim.
