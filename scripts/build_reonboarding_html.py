#!/usr/bin/env python3
"""
=============================================================================
build_reonboarding_html.py
=============================================================================

Builds a single self-contained HTML re-onboarding document with the
key figures from the NeurIPS submission embedded as base64 PNGs.

Usage:
    python scripts/build_reonboarding_html.py

What it creates:
    reonboarding/index.html   # single file, ~10 MB, SCP-friendly

Configuration:
    FIG_ROOT      - source dir for embedded PNGs
    OUT_PATH      - destination HTML path
    FIGURES       - ordered list of (slug, path, caption, eli12) tuples

Dependencies:
    Pure stdlib (base64, pathlib, textwrap).
=============================================================================
"""
from __future__ import annotations

import base64
import html
from pathlib import Path
from textwrap import dedent

ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
FIG_ROOT = ROOT / "overleaf/NExT_Game_2026_style/graphics"
OUT_PATH = ROOT / "reonboarding/index.html"


def b64(rel: str) -> str:
    p = FIG_ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"missing figure: {p}")
    return base64.b64encode(p.read_bytes()).decode("ascii")


def img_block(slug: str, rel: str, caption: str, eli12: str) -> str:
    data = b64(rel)
    return dedent(
        f"""
        <figure id="{slug}">
          <img src="data:image/png;base64,{data}" alt="{html.escape(caption)}" />
          <figcaption>
            <div class="cap-title">{html.escape(caption)}</div>
            <div class="cap-eli12"><strong>ELI12:</strong> {eli12}</div>
            <div class="cap-src">{html.escape(rel)}</div>
          </figcaption>
        </figure>
        """
    )


FIGURES = [
    # Headline 2-player results
    (
        "fig-headline",
        "n2_gpt5_nano/01_adversary_payoff_three_game_curves.png",
        "Headline: stronger LLMs win more against a fixed GPT-5-nano opponent across all three games.",
        (
            "Imagine a tournament where every model gets to play against the same average "
            "kid (GPT-5-nano). The x-axis is the kid's <i>opponent's</i> general smarts "
            "(LMArena Elo). The y-axis is how many points the opponent walks away with. "
            "Three games, three positive slopes: smarter kids win more. The slopes are "
            "+5.3 / +6.8 / +7.4 utility per 100 Elo for item-sharing, treaty-making, and "
            "co-funding respectively."
        ),
    ),
    (
        "fig-nbs",
        "n2_gpt5_nano/14_nbs_decomposition_overall.png",
        "The key result: smarter models do not just win more — they take more than their fair share.",
        (
            "This is the picture that holds the whole story together. We split the smarter "
            "model's score into two pieces: (1) the <b>fair share</b> a textbook negotiation "
            "theorist (Nash Bargaining / Lindahl) would award it given its preferences, "
            "and (2) the <b>residual</b> = actual − fair-share. The fair-share line is flat "
            "(<i>p</i> &gt; 0.19): smarter models do not have wildly different fair-share "
            "entitlements. But the residual climbs steeply with Elo (+4.98 / +6.44 / +8.30 "
            "per 100 Elo, <i>r</i> ≥ 0.78). The dotted vertical line marks the crossover where "
            "the residual hits zero — somewhere around Elo 1410–1461. Below that, models "
            "play roughly fair (or even leave money on the table). Above that, they extract "
            "+3 to +6 points <i>beyond</i> what fairness predicts. The frontier is already on "
            "the wrong side of that line."
        ),
    ),
    (
        "fig-baseline-comp",
        "n2_gpt5_nano/04_baseline_payoff_by_competition.png",
        "What about the kid? GPT-5-nano's payoff depending on how cooperative the game is.",
        (
            "If smarter opponents only grew the pie, the kid would also be better off. But "
            "this plot says: it depends on the game's competition level. In cooperative cells "
            "(orthogonal preferences) the kid <i>does</i> benefit — the smarter model finds a "
            "package that helps both sides. In high-competition cells (aligned valuations) "
            "the kid stagnates or loses ground. So the smarter agent's gains are a mix of "
            "&lsquo;making the pie bigger&rsquo; (good for everyone) and &lsquo;taking a bigger "
            "slice&rsquo; (bad for the kid). The competition axis decides the mix."
        ),
    ),
    (
        "fig-welfare",
        "n2_gpt5_nano/10_total_welfare_by_competition_ewma.png",
        "Total welfare (the pie size) does rise with Elo, but never closes the gap to the optimum in competitive cells.",
        (
            "Each dashed line is the maximum welfare achievable in that competition slice. "
            "Smarter models push realized welfare closer to that ceiling — optimality-ratio "
            "slopes of +0.02 / +0.04 / +0.13 per 100 Elo. So capability really does grow the "
            "pie. The catch is in the previous figure: that bigger pie isn't split fairly."
        ),
    ),
    # TTC
    (
        "fig-ttc",
        "ttc_order_averaged_target_payoff_vs_compute.png",
        "Test-time compute is NOT a reliable substitute for capability.",
        (
            "We held capability fixed (one model family at a time: GPT-5, Claude Sonnet 4.6, "
            "Gemini 3 Flash) and dialled the reasoning budget. If &lsquo;think harder&rsquo; "
            "translated to bargaining gains, payoff would rise with compute. It doesn't, "
            "reliably. Slopes are tiny and mostly statistically weak (GPT-5 +0.71, "
            "<i>p</i> = 0.67; Claude +1.02, <i>p</i> = 0.47; Gemini −1.69, <i>p</i> = 0.06). "
            "More thinking sometimes turns into &lsquo;discount-aware satisficing&rsquo; — "
            "the model just accepts a worse deal sooner. Counter-intuitively, Claude's "
            "<i>max</i> and <i>xhigh</i> settings used <i>fewer</i> tokens and got higher utility "
            "than its low/medium/high settings."
        ),
    ),
    # Multi-agent
    (
        "fig-hetero-by-n",
        "n_gt_2_report/heterogeneous_payoff_vs_arena_elo_by_n.png",
        "Heterogeneous rosters: Elo still predicts payoff as you scale N from 2 to 10.",
        (
            "We randomly draw N different models from a 24-model pool and let them negotiate. "
            "Each panel is a game-size combination; each point is a model's average over the "
            "rosters it landed in. The positive slope (capability ⇒ payoff) survives at every N, "
            "though the correlation is lower than in bilateral play because each model now "
            "faces many different rosters, positions, and competition cells. Mean slopes are "
            "+4.64 / +3.60 / +4.81 per 100 Elo for the three games."
        ),
    ),
    (
        "fig-hom-by-n",
        "n_gt_2_report/hom_adversary_payoff_vs_elo_by_n.png",
        "&lsquo;Banding together&rsquo; doesn't dilute a strong agent: focal-adversary payoffs stay positive at every group size.",
        (
            "Place one non-nano focal adversary among N−1 GPT-5-nano agents. Conventional "
            "intuition says more weak agents should outvote the strong one. They don't — at "
            "every N from 2 to 10, the focal's payoff still rises with Elo (+8.43 / +4.59 / "
            "+3.44 per 100 Elo on average). Larger groups add more vetoers, but they also "
            "create more proposals that the weaker fleet fails to coordinate around — the "
            "strong agent acts as a Schelling point."
        ),
    ),
    (
        "fig-multiagent-residual",
        "n_gt_2_report/multiagent_agent_fairness_residual_vs_elo.png",
        "Multi-agent fair-share residual vs Elo — the bilateral pattern persists with more agents.",
        (
            "Same x/y idea as the headline NBS plot, but now in N&gt;2 settings. Stronger "
            "models continue to sit above their benchmark share. The exploitation channel "
            "doesn't go away when you add more agents to the room."
        ),
    ),
    (
        "fig-gini",
        "n_gt_2_report/multiagent_utility_gini_vs_n.png",
        "Payoff inequality (Gini) grows with group size — especially in item allocation.",
        (
            "Gini = how unequal the payoffs are (0 = perfectly equal, 1 = winner-takes-all). "
            "Game 1 (item allocation) gets sharply more unequal as N grows: indivisible items "
            "create leftover problems that continuous treaties (Game 2) can compromise around. "
            "Game 3 is &lsquo;mixed&rsquo; — when public-goods coordination fails, everyone gets "
            "zero, which paradoxically <i>lowers</i> Gini for a bad reason."
        ),
    ),
    (
        "fig-normalized-by-n",
        "multiagent_normalized_utility_by_n.png",
        "Normalized welfare across group sizes — treaty bargaining is the resilient mechanism.",
        (
            "Each agent's attainable fair share shrinks as N grows, so raw utility falls; we "
            "normalize by SW*/N to compare across N. The story: <b>treaty bargaining is stable</b> "
            "across N. Item allocation degrades. Co-funding degrades sharply under scarce "
            "budgets. So mechanism design matters — the same capability landscape can be "
            "&lsquo;absorbed&rsquo; gracefully or amplified into pathology depending on the rules."
        ),
    ),
]


# ---------- Application-question answers ----------
APPLICATION_ANSWERS = """
<h3>1. What were you trying to figure out, and why does it matter?</h3>
<p>
Big-picture question: <strong>when a more capable LLM agent negotiates against a less
capable one, does the smarter agent (a) grow the pie for everyone, or (b) take a bigger
slice at the weaker agent's expense?</strong> This matters because the world is rapidly
moving toward economic situations where consumer-grade agents (free chatbots,
small-business assistants) interact with frontier-grade agents (vendor-side, with
proprietary tools and reasoning budgets). If capability translates into surplus
extraction, then deploying a stronger agent isn't a Pareto improvement &mdash; you may be
making your counterparty worse off. That has implications for procurement, marketplace
design, and especially <em>scalable oversight</em>, where the entire premise is that a
weaker model can meaningfully constrain a stronger one.
</p>
<p>
We also wanted to know:
</p>
<ul>
  <li>Does more &lsquo;thinking time&rsquo; at inference (test-time compute) substitute for raw capability?</li>
  <li>Does adding more agents to the room (N up to 10) dilute a strong agent's advantage?</li>
  <li>Does the answer depend on the mechanism (rivalrous items vs. continuous treaty terms vs. threshold public goods)?</li>
</ul>

<h3>2. What did you actually do? (Concrete: methods, hours, outputs.)</h3>
<p>
<strong>Built three new multi-turn negotiation environments</strong>, each isolating a
different real-world strategic mechanism:
</p>
<ol>
  <li><strong>Item allocation</strong> &mdash; discrete, rivalrous items, propose-and-vote.</li>
  <li><strong>Diplomatic treaty bargaining</strong> &mdash; continuous outcomes, single-peaked
      preferences with controllable correlation (ρ) and interest overlap (θ).</li>
  <li><strong>Participatory budgeting / co-funding</strong> &mdash; threshold public goods,
      controlled by preference alignment (α) and budget scarcity (σ).</li>
</ol>
<p>
Each game follows the same 6-step protocol per round: discussion → private thinking →
proposal → private vote → selection (2/3 supermajority) → reflection. Up to T rounds,
exponential discount γ.
</p>
<p>
<strong>Ran &gt;5,000 negotiation games across 30+ frontier and weaker LLMs</strong>:
</p>
<ul>
  <li><b>Bilateral sweep</b> against GPT-5-nano (and a Llama 3.3 70B replication):
      862 + 540 + 539 runs across the three games, swept over seven competition levels
      in Game 1 and nine (ρ,θ) / (α,σ) cells in Games 2 and 3, both model orders.</li>
  <li><b>Multi-agent batches</b> at N ∈ {2,4,6,8,10}: 1,430 homogeneous + 1,300
      heterogeneous runs (8,580 + 7,800 agent rows). Three roster designs:
      homogeneous control (all GPT-5-nano), focal adversary (one non-nano in a sea of
      nanos), and random heterogeneous draws from a 24-model pool stratified by
      Elo-standard-deviation.</li>
  <li><b>Test-time-compute (TTC) stress test</b>: 216 runs across GPT-5, Claude Sonnet
      4.6, and Gemini 3 Flash at four reasoning-effort levels each, opponent fixed.</li>
</ul>
<p>
<strong>Analysis pipeline</strong>: cooperative-game-theoretic fairness benchmarks
(Nash Bargaining Solution for Games 1&ndash;2, Lindahl benefit-proportional cost
sharing for Game 3); decomposition of payoff into fair-share + residual; competition-
stratified regressions; group-size scaling; provider-native reasoning token / output
token diagnostics.
</p>
<p>
<strong>Outputs</strong>: NeurIPS 2026 submission (45 pages including appendix);
~10 MB of figures across the three game families and three experimental designs;
anonymized release at <code>anonymous.4open.science/r/bargain</code>; a code base of
multi-agent orchestrator, three game environments, hardened JSON repair / context
budgeting, SLURM cluster integration, and bespoke per-game analysis scripts. Roughly
600 worker-hours of cluster compute for the final multi-agent + TTC batches alone,
on top of months of iteration on protocol, prompts, and metrics.
</p>

<h3>3. What surprised you?</h3>
<ul>
  <li><strong>The whole positive-Elo slope is in the <em>residual</em>, not the fair share.</strong>
      That is, smarter models don't gain by having intrinsically &lsquo;better&rsquo; preferences
      to defend &mdash; they gain by extracting <em>above</em> what fairness prescribes. The
      fair-share component is statistically flat (<i>p</i> &gt; 0.19) while the residual
      climbs by ~5&ndash;8 points per 100 Elo. The crossover from below-fair to above-fair
      sits around Elo 1410&ndash;1461 &mdash; <em>inside the current frontier</em>. Frontier
      models are already on the exploitative side of fair division.</li>
  <li><strong>Test-time compute does <em>not</em> reliably help.</strong> Claude's
      &lsquo;max&rsquo; and &lsquo;xhigh&rsquo; settings used <em>fewer</em> tokens than
      &lsquo;medium&rsquo; / &lsquo;high&rsquo; while earning more utility &mdash; an inverse
      scaling effect we did not predict. More reasoning sometimes manifests as
      &lsquo;discount-aware satisficing&rsquo;: the model rationalizes accepting a worse deal
      sooner.</li>
  <li><strong>Mechanism dominates capability in N-player play.</strong> Treaty bargaining is
      remarkably stable across N because continuous outcomes allow compromise; item allocation
      degrades because indivisibility creates leftover problems; co-funding collapses under
      scarcity because threshold public goods need precise coordination. Same capability
      ladder, three different scaling behaviours.</li>
  <li><strong>Adding weak agents does not dilute a strong adversary.</strong> Homogeneous-
      adversary slopes stay positive at every N from 2 to 10; the focal-vs-fleet gap actually
      <em>widens</em>. A strong agent can act as a Schelling point that the weak fleet
      coordinates around.</li>
  <li><strong>Weak-model windfalls from a non-strategic mechanism.</strong> In co-funding,
      contributions to <em>unfunded</em> projects are refunded. So a noisy weak model can
      &lsquo;pledge&rsquo; without paying if its pledges happen to land on unfunded projects.
      Llama-3.2-1B sometimes beats GPT-5-nano not by playing well but by being usefully
      random. The Elo-vs-payoff curve becomes non-monotone in those cells.</li>
  <li><strong>An Elo-independent zero region exists.</strong> At (α=0, σ=0.2) in co-funding,
      preferences are orthogonal <em>and</em> budgets are tight, so the rational play is
      &lsquo;fund only your own project, refuse to help anyone else&rsquo; &mdash; and the game
      stalls at zero utility even for frontier models. Capability buys you nothing when the
      mechanism is structurally dead.</li>
</ul>

<h3>4. What did you do wrong, or what turned out to be suboptimal in retrospect?</h3>
<ul>
  <li><strong>Used LMArena Elo as the main capability axis.</strong> Elo mixes reasoning,
      instruction-following, formatting reliability, safety tuning, sycophancy, and crowd
      preference. It is a fine <em>ordering</em> variable but not a clean operationalization
      of &lsquo;strategic capability&rsquo;. We argue this in the limitations but it remains
      the largest interpretive weakness. A cleaner follow-up would isolate the reasoning,
      formatting, and persuasion sub-components.</li>
  <li><strong>Underestimated mechanism-specific failure modes.</strong> Game 3 (co-funding)
      has a much messier signal-to-noise ratio than Games 1&ndash;2 because of the
      &lsquo;noise-as-free-riding&rsquo; phenomenon and the (α=0,σ=0.2) dead zone. Several
      analyses needed competition-stratified plots after the fact; averaging across the
      grid was misleading.</li>
  <li><strong>Treated test-time compute symmetrically across providers.</strong> GPT-5
      reports reasoning tokens directly; Claude and Gemini hide them, so we proxied with
      output tokens. The proxy is noisy and the cross-provider comparison is uncomfortable.
      A controlled OSS-model TTC sweep (where we own the token counts) would have helped.</li>
  <li><strong>Spent too long on infrastructure relative to analysis.</strong> Significant
      time went into JSON repair, context-budget telemetry, provider key rotation, and
      audited fallbacks &mdash; necessary plumbing, but it crowded out earlier exploratory
      iteration on the metrics themselves (especially the NBS decomposition, which we
      arrived at relatively late).</li>
  <li><strong>One fixed baseline (GPT-5-nano) for most of the bilateral story.</strong>
      The Llama 3.3 70B replication helps but is partial. A two-dimensional sweep
      (every model vs every other model) would be the gold standard; we didn't run it
      because the cost scales as O(n&sup2;) and we only had budget for the fixed-baseline
      slice.</li>
</ul>

<h3>5. What's still unresolved or unknown?</h3>
<ul>
  <li><strong>Why does test-time compute fail to translate to bargaining gains, and why
      does it sometimes invert?</strong> Is this a satisficing artefact, a discount-rate
      interaction, or something about how providers gate their internal reasoning?
      Likely the single most interesting follow-up.</li>
  <li><strong>Where is the crossover Elo for the residual on a clean capability axis
      (not LMArena Elo)?</strong> We can say &lsquo;somewhere in the 1410&ndash;1461 LMArena
      band&rsquo; but a more interpretable axis (a strategic-reasoning eval, an
      instruction-following eval, etc.) would let us say &lsquo;at this specific capability
      level&rsquo;.</li>
  <li><strong>Do these patterns hold outside negotiation?</strong> Future work direction:
      auctions, debates, adversarial-collaboration tasks, multi-agent RL, agent-as-judge
      settings. The negotiation games are a clean testbed but obviously not the whole
      economic surface.</li>
  <li><strong>How does this interact with human counterparties?</strong> All current
      results are LLM-vs-LLM. The realistic deployment scenario is LLM-vs-human and the
      asymmetries there are likely different (humans are slower, but harder to manipulate
      with formatting tricks, etc.).</li>
  <li><strong>Can mechanism design defuse the exploitation channel?</strong> Treaty
      bargaining is already more robust than item allocation or co-funding. Is there a
      systematic property (continuity? single-peakedness? linear utility?) that predicts
      which mechanisms absorb capability gaps gracefully?</li>
  <li><strong>Long-horizon and repeated interaction.</strong> Our games are one-shot.
      In repeated play, reputation and meta-strategic considerations would change the
      picture, possibly punishing the exploitative high-Elo strategy.</li>
</ul>
"""


GLOSSARY = """
<dl class="glossary">
  <dt>Elo (LMArena)</dt>
  <dd>A public benchmark score from the LMArena leaderboard (March 31, 2026 snapshot).
      Used here as the main &lsquo;capability ordering&rsquo; axis. Not a clean strategic-skill
      measure &mdash; mixes reasoning, instruction-following, formatting, safety tuning.</dd>

  <dt>Competition axis</dt>
  <dd>Game 1: cosine similarity α<sub>ij</sub> of valuation vectors (1 = same items,
      conflict; 0 = orthogonal, no conflict). Game 2: preference correlation ρ and interest
      overlap θ. Game 3: preference alignment α and budget scarcity σ.</dd>

  <dt>Nash Bargaining Solution (NBS)</dt>
  <dd>Fairness benchmark for Games 1&ndash;2. The allocation/agreement that maximizes the
      product of utility gains over the disagreement point. Used as the &lsquo;fair share&rsquo;
      anchor.</dd>

  <dt>Lindahl benchmark</dt>
  <dd>Fairness reference for Game 3 (public goods). Each agent pays for each funded project
      in proportion to its valuation. This is the cooperative-game-theoretic fair cost
      share for threshold public-good provision.</dd>

  <dt>Exploitation index E<sub>i</sub></dt>
  <dd>(u<sub>i</sub><sup>actual</sup> &minus; u<sub>i</sub><sup>benchmark</sup>) /
      |u<sub>i</sub><sup>benchmark</sup>|. Positive = agent got more than benchmark.
      Negative = agent got less.</dd>

  <dt>Fair-share residual</dt>
  <dd>(adversary's actual utility) &minus; (adversary's NBS / Lindahl share). The
      &lsquo;extra surplus a stronger agent captures beyond what fairness theory grants it&rsquo;.
      Headline result: this residual is flat near zero for weaker models and climbs
      sharply with Elo.</dd>

  <dt>TTC (test-time compute)</dt>
  <dd>Inference-time reasoning budget. Sweeps across provider-native reasoning effort
      levels at fixed model identity.</dd>

  <dt>Homogeneous adversary design</dt>
  <dd>One focal non-nano agent placed among N&minus;1 GPT-5-nano agents. Tests whether
      capability advantages persist when the weak fleet outnumbers the strong agent.</dd>

  <dt>Heterogeneous design</dt>
  <dd>N agents sampled from a 24-model pool with Elo-standard-deviation strata. Tests
      whether Elo predicts payoff under random roster composition.</dd>
</dl>
"""


CSS = """
:root {
  --bg: #fafafa;
  --fg: #1a1a1a;
  --muted: #555;
  --accent: #b91c1c;
  --accent2: #1d4ed8;
  --card: #ffffff;
  --border: #d4d4d8;
  --code-bg: #f1f5f9;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  line-height: 1.55;
  font-size: 16px;
}
.wrap {
  max-width: 1080px;
  margin: 0 auto;
  padding: 32px 28px 80px;
}
header {
  border-bottom: 3px solid var(--accent);
  margin-bottom: 28px;
  padding-bottom: 18px;
}
header h1 {
  margin: 0 0 6px;
  font-size: 30px;
  letter-spacing: -0.01em;
}
header .sub {
  color: var(--muted);
  font-size: 15px;
}
nav.toc {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  margin-bottom: 28px;
}
nav.toc h2 { margin: 0 0 8px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
nav.toc ol { margin: 0; padding-left: 22px; }
nav.toc li { margin: 3px 0; }
nav.toc a { color: var(--accent2); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
section {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 22px 26px;
  margin-bottom: 24px;
}
section > h2 {
  margin-top: 0;
  font-size: 22px;
  color: var(--accent);
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
}
section > h3 {
  font-size: 17px;
  color: var(--accent2);
  margin-top: 22px;
}
figure {
  margin: 24px 0;
  padding: 14px;
  background: #fcfcfc;
  border: 1px solid var(--border);
  border-radius: 8px;
}
figure img {
  width: 100%;
  height: auto;
  display: block;
  border-radius: 4px;
}
figcaption {
  margin-top: 10px;
  font-size: 14px;
}
.cap-title { font-weight: 600; margin-bottom: 6px; }
.cap-eli12 { background: #fffbeb; border-left: 3px solid #d97706; padding: 8px 12px; border-radius: 4px; margin: 6px 0; }
.cap-src { color: var(--muted); font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; }
code, kbd { background: var(--code-bg); padding: 1px 5px; border-radius: 3px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.92em; }
ul, ol { padding-left: 22px; }
li { margin: 4px 0; }
.callout {
  border-left: 4px solid var(--accent);
  background: #fef2f2;
  padding: 12px 16px;
  border-radius: 0 6px 6px 0;
  margin: 14px 0;
}
.callout.blue { border-color: var(--accent2); background: #eff6ff; }
.callout.green { border-color: #16a34a; background: #f0fdf4; }
.callout h4 { margin: 0 0 6px; font-size: 15px; }
dl.glossary { display: grid; grid-template-columns: 220px 1fr; column-gap: 18px; row-gap: 12px; }
dl.glossary dt { font-weight: 600; color: var(--accent2); }
dl.glossary dd { margin: 0; }
.kpis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0; }
.kpi { background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.kpi .v { font-size: 26px; font-weight: 700; color: var(--accent); }
.kpi .l { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
@media (max-width: 720px) {
  dl.glossary { grid-template-columns: 1fr; }
  .kpis { grid-template-columns: 1fr; }
}
"""


def main() -> None:
    # ---------- Build the figure sections ----------
    bilateral_figs = [f for f in FIGURES if f[0] in {"fig-headline", "fig-nbs", "fig-baseline-comp", "fig-welfare"}]
    ttc_figs = [f for f in FIGURES if f[0] in {"fig-ttc"}]
    multiagent_figs = [f for f in FIGURES if f[0] in {"fig-hetero-by-n", "fig-hom-by-n", "fig-multiagent-residual", "fig-gini", "fig-normalized-by-n"}]

    def render_figs(figs):
        return "\n".join(img_block(*f) for f in figs)

    bilateral_block = render_figs(bilateral_figs)
    ttc_block = render_figs(ttc_figs)
    multiagent_block = render_figs(multiagent_figs)

    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Bargain — Re-onboarding (NeurIPS 2026 submission)</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Bargain &mdash; Re-onboarding to the project</h1>
  <div class="sub">
    NeurIPS 2026 submission &middot; <em>Scaling Laws for Strategic Interactions</em>
    &middot; <strong>5,000+ games, 30+ models, 3 mechanisms, N up to 10</strong>
  </div>
</header>

<section>
  <h2>One-paragraph TL;DR</h2>
  <p>
  We deploy frontier and weaker LLMs as <em>agents</em> in three controlled negotiation games
  &mdash; rivalrous item allocation, continuous treaty bargaining, threshold public-goods
  co-funding &mdash; and sweep over capability (LMArena Elo), inference-time reasoning budget,
  number of agents (N up to 10), and competition level. Smarter agents <em>do</em> grow the
  pie (welfare rises with Elo in every game), but they also <em>extract</em>: when we
  decompose their payoff into a fair-share component (Nash Bargaining / Lindahl) plus a
  residual, the fair-share line is statistically flat while the residual climbs sharply
  with Elo, crossing zero somewhere around Elo 1410&ndash;1461 &mdash; <strong>inside the
  current frontier</strong>. Test-time compute is <em>not</em> a reliable substitute for
  capability; mechanism design (treaty vs. items vs. co-funding) sharply changes how a
  capability gap propagates into welfare outcomes; and stacking weak agents does not
  dilute a strong adversary.
  </p>
  <div class="kpis">
    <div class="kpi"><div class="v">+5.28 / +6.76 / +7.37</div><div class="l">Adversary slope (per 100 Elo) &mdash; Games 1/2/3 vs GPT-5-nano</div></div>
    <div class="kpi"><div class="v">~1410&ndash;1461</div><div class="l">Elo where fair-share residual crosses zero</div></div>
    <div class="kpi"><div class="v">+3 to +6</div><div class="l">Surplus extracted above fair share by frontier models (utility pts)</div></div>
  </div>
</section>

<nav class="toc">
  <h2>Contents</h2>
  <ol>
    <li><a href="#headline">Headline pictures (2-player)</a></li>
    <li><a href="#residual-zero">The thing you specifically asked about: fair share, residuals, the zero crossing</a></li>
    <li><a href="#ttc">Test-time compute &mdash; the surprising inverse trend</a></li>
    <li><a href="#multiagent">Multi-agent (N up to 10) results</a></li>
    <li><a href="#application">Your application questions, answered using this project</a></li>
    <li><a href="#code-state">What the code looks like right now &mdash; recent changes</a></li>
    <li><a href="#glossary">Quick glossary</a></li>
  </ol>
</nav>

<section id="headline">
  <h2>1. Headline pictures (2-player)</h2>
  <p>
    These are the four plots that carry the bilateral story. Read them in order.
  </p>
  {bilateral_block}
</section>

<section id="residual-zero">
  <h2>2. The thing you specifically asked about: <em>residual crosses zero at the benchmark</em></h2>
  <p>
    You asked me to explain &lsquo;as agents get more capable they cross some threshold of
    fair share where the residual crosses 0 &mdash; below benchmark residuals, above benchmark
    residuals &mdash; what are these, actually?&rsquo; Here is the plain-English version.
  </p>

  <h3>Setting up the picture</h3>
  <p>
    Pick any 2-player negotiation game in our setup. The two players are the
    <strong>adversary</strong> (an LLM whose capability we vary) and the
    <strong>baseline</strong> (GPT-5-nano, fixed). Both have private preferences. They
    negotiate. Someone walks away with some utility.
  </p>
  <p>
    Now we need a fairness benchmark. For Games 1&ndash;2 we use the <strong>Nash Bargaining
    Solution (NBS)</strong>: the deal that maximizes the product of both players' utility
    gains over no-deal. For Game 3 we use <strong>Lindahl benefit-proportional cost
    sharing</strong>: each agent pays for each funded project in proportion to its valuation.
    Both are textbook cooperative-game-theoretic fair outcomes given the agents' preferences.
  </p>
  <p>
    The benchmark tells us, for each game and each preference draw,
    <em>how much utility each agent &lsquo;deserves&rsquo; under fair division</em>. Call that
    u<sub>i</sub><sup>benchmark</sup>.
  </p>

  <h3>The decomposition</h3>
  <p>
    For each negotiation we observe the actual utility u<sub>i</sub><sup>actual</sup>. We
    split it into:
  </p>
  <ul>
    <li><strong>Fair-share component</strong> = u<sub>i</sub><sup>benchmark</sup> &mdash; the
        slice fairness would award the adversary given <em>their</em> preferences.</li>
    <li><strong>Residual</strong> = u<sub>i</sub><sup>actual</sup> &minus;
        u<sub>i</sub><sup>benchmark</sup> &mdash; the extra surplus they captured (positive)
        or gave up (negative) <em>beyond</em> what fairness would predict.</li>
  </ul>
  <p>
    Then we look at how each piece changes as the adversary's Elo goes up.
  </p>

  <div class="callout">
    <h4>The actual finding</h4>
    <ul>
      <li>The <strong>fair-share line is statistically flat</strong> with respect to Elo
          (<i>p</i> &gt; 0.19 in all three games). Smarter models don't have systematically
          &lsquo;better&rsquo; preferences to defend &mdash; we draw their preferences from the
          same distribution as the baseline's.</li>
      <li>The <strong>residual rises sharply with Elo</strong> (+4.98 / +6.44 / +8.30 utility
          per 100 Elo, <i>r</i> &ge; 0.78, <i>p</i> &lt; 10<sup>&minus;5</sup>).</li>
      <li>The residual is <strong>negative for weaker models</strong> (they leave money on
          the table relative to fair division), <strong>crosses zero around Elo
          1410&ndash;1461</strong>, and is <strong>positive for frontier models</strong>
          (they extract +3 to +6 utility points <em>above</em> fair share).</li>
    </ul>
  </div>

  <h3>What that means in plain English</h3>
  <p>
    Imagine fairness as a yardstick that says &lsquo;given your preferences, you deserve X
    points.&rsquo; Weaker models fall <em>short</em> of X &mdash; they're not good enough at
    bargaining to even claim their fair share. They're playing badly, and they leave value
    on the table. This is the &lsquo;irrationality channel&rsquo;.
  </p>
  <p>
    Frontier models exceed X &mdash; they're good enough at bargaining to systematically
    capture surplus beyond what fair division grants them. This is the &lsquo;exploitation
    channel&rsquo;.
  </p>
  <p>
    The interesting and slightly worrying part is that <strong>the crossover is inside the
    current Elo distribution of deployed models</strong>. It's not in some hypothetical
    future. Today's top-tier models are already in the &lsquo;extracts above fair share&rsquo;
    regime against today's mid-tier ones.
  </p>

  <h3>Why does this matter for the paper's argument?</h3>
  <p>
    The fair-share decomposition is the cleanest way we found to separate two competing
    stories about why a smarter agent wins more:
  </p>
  <ul>
    <li><b>Story A &mdash; the optimist:</b> &lsquo;Smarter agents are just better at finding
        mutually beneficial deals.&rsquo; If this were the whole story, the fair-share
        component should rise with Elo (they find Pareto-better outcomes that, in
        expectation, also bump up their own NBS share), and the residual should stay near
        zero.</li>
    <li><b>Story B &mdash; the concerning one:</b> &lsquo;Smarter agents extract surplus from
        weaker ones beyond what either's preferences entitle them to.&rsquo; If this were the
        whole story, the residual should rise with Elo and the fair-share component should
        be flat.</li>
  </ul>
  <p>
    The data look like <em>Story B</em>. The whole positive Elo slope lives in the residual.
    That is the single most concise way to say &lsquo;capability translates to surplus
    extraction, not joint coordination&rsquo; &mdash; and that is the core normative claim of
    the paper.
  </p>
</section>

<section id="ttc">
  <h2>3. Test-time compute &mdash; the surprising inverse trend</h2>
  <p>
    A separate axis: hold the model identity fixed and crank up the inference-time reasoning
    budget. If &lsquo;think harder&rsquo; mapped to bargaining gains, we should see payoff
    rise with compute. It doesn't &mdash; reliably.
  </p>
  {ttc_block}
  <p>
    Two qualitative patterns we keep noticing in the transcripts:
  </p>
  <ul>
    <li><strong>Discount-aware satisficing.</strong> With more reasoning, the model
        sometimes <em>derives</em> that a worse deal now is better than a marginally better
        deal later (γ discount). So it accepts faster and earns less.</li>
    <li><strong>Order &amp; focal-bundle effects dominate.</strong> In one GPT-5 high
        item-allocation cell, the target swings from a −27 to a +27 utility gap just by
        flipping which agent proposes first. Reasoning depth doesn't overpower these
        protocol-level effects.</li>
  </ul>
</section>

<section id="multiagent">
  <h2>4. Multi-agent (N up to 10) results &mdash; in a nutshell</h2>
  <p>
    You asked specifically about multi-agent. Here's the short version. We ran two designs:
  </p>
  <ul>
    <li><strong>Heterogeneous</strong>: each game samples N different models from a
        24-model pool with Elo-stratified draws. Each model appears in many rosters and
        positions.</li>
    <li><strong>Homogeneous-adversary</strong>: one non-nano focal adversary among
        N&minus;1 GPT-5-nano agents. Tests &lsquo;does a fleet of weaker agents dilute the
        strong one?&rsquo;</li>
  </ul>

  <div class="callout blue">
    <h4>Multi-agent bottom line</h4>
    <ol>
      <li><b>Elo still predicts payoff at every N.</b> Slopes shrink modestly with N but
          stay positive across the board.</li>
      <li><b>Banding together doesn't help the weak.</b> Going from N=2 to N=10, the
          focal-adversary's <em>advantage over the fleet mean</em> grows by +3.03 /
          +7.67 / +5.00 utility points. A strong agent in a weak room does
          <em>better</em> than in a 1v1.</li>
      <li><b>Mechanism design dominates capability when scaling N.</b> Treaty bargaining
          stays welfare-stable as N grows; item allocation degrades sharply
          (indivisibility); co-funding collapses under scarcity (threshold + free-riding).</li>
      <li><b>Inequality grows with N</b>, especially in item allocation. The Gini of
          payoffs rises with group size because indivisible items create leftover
          problems that compromises can't fix.</li>
    </ol>
  </div>
  {multiagent_block}
</section>

<section id="application">
  <h2>5. Your application questions, answered using this project</h2>
  {APPLICATION_ANSWERS}
</section>

<section id="code-state">
  <h2>6. What the code looks like right now &mdash; recent changes</h2>
  <p>
    Last ~10 commits on <code>main</code> (most recent first):
  </p>
  <ul>
    <li><code>341ee22</code> &mdash; readme updates</li>
    <li><code>713cdb8</code> &mdash; figures: refresh utility-Elo plots and primary slide tooling</li>
    <li><code>4ef9d4f</code> &mdash; analysis: add NeurIPS revision and fairness report builders</li>
    <li><code>6582b85</code> &mdash; experiments: add native TTC reasoning-effort sweeps</li>
    <li><code>008ecbe</code> &mdash; runtime: harden structured JSON repair and audited fallbacks</li>
    <li><code>917ad46</code> &mdash; experiments: default phase output caps to 16k</li>
    <li><code>34e6f93</code> &mdash; runtime: add terse context compaction and prompt budget telemetry</li>
    <li><code>07f706c</code> &mdash; providers: retry transient upstream failures without exhausting key pools</li>
    <li><code>4e2cdd2</code> &mdash; chore: anonymize artifact paths and local identity labels</li>
    <li><code>e3f6022</code> &mdash; updated README</li>
  </ul>
  <p>
    The thing that changed most over the last month: <strong>the TTC sweep was added</strong>
    (commit <code>6582b85</code>), the <strong>fairness/NBS decomposition report builder
    was added</strong> (commit <code>4ef9d4f</code>), and a bunch of runtime hardening
    (JSON repair, context budgeting, provider key rotation) so the big multi-agent batches
    don't bleed jobs to malformed-output errors.
  </p>
  <p>
    Uncommitted work-in-progress on the branch right now:
  </p>
  <ul>
    <li><code>scripts/analyze_n2_baseline_comparison.py</code></li>
    <li><code>scripts/analyze_n2_plus_multiagent_comparison.py</code></li>
    <li><code>scripts/build_n2_ttc_multiagent_report.py</code></li>
    <li><code>scripts/plot_figure3_baseline_by_competition_ewma_iteration.py</code></li>
    <li><code>scripts/plot_ttc_effort_adversary_baseline.py</code></li>
    <li><code>docs/cleanup_and_release_plan_2026_06.md</code></li>
  </ul>
  <p>
    Entry points worth re-reading first when you sit down at the code again:
  </p>
  <ul>
    <li><code>run_strong_models_experiment.py</code> &mdash; orchestrator entry point.</li>
    <li><code>strong_models_experiment/experiment.py</code> &mdash; multi-agent loop.</li>
    <li><code>strong_models_experiment/phases/phase_handlers.py</code> &mdash; the 6-step
        protocol implementation.</li>
    <li><code>game_environments/item_allocation.py</code>,
        <code>diplomatic_treaty.py</code>, <code>co_funding.py</code> &mdash; the three games.</li>
    <li><code>game_environments/metrics.py</code> (Games 1&ndash;2) and
        <code>cofunding_metrics.py</code> (Game 3) &mdash; NBS / Lindahl benchmarks.</li>
  </ul>
</section>

<section id="glossary">
  <h2>7. Quick glossary</h2>
  {GLOSSARY}
</section>

<section>
  <h2>How to SCP this file to your laptop</h2>
  <p>From your laptop:</p>
  <pre><code>scp della:/scratch/gpfs/DANQIC/jz4391/bargain/reonboarding/index.html ~/Desktop/bargain_reonboarding.html
open ~/Desktop/bargain_reonboarding.html</code></pre>
  <p>This file is fully self-contained &mdash; all images are inlined as base64, no internet required.</p>
</section>

</div>
</body>
</html>
"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(body, encoding="utf-8")
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
