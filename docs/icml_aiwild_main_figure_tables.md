# ICML AIWILD main-text figures as reviewer-readable tables

This document converts Figures 1–9 in the most recent compiled ICML AIWILD
manuscript (`overleaf/icml_aiwild_template/icml_aiwild_2026.pdf`, compiled
2026-07-21) into tables. It does not replace or modify the manuscript.

The “Old NeurIPS submission” notes compare against the anonymized submitted
paper `26631_Scaling_Laws_for_Strateg (3).pdf`. “Exact” means the same figure,
not merely the same experiment or a related analysis. When a current figure
recombines old panels, changes the included runs, or uses a new visual
encoding, it is marked as not exact and the closest old figure is named.

## Reading guide

- `±` denotes SEM unless a table explicitly says otherwise.
- G1 = Item Allocation; G2 = Diplomatic Treaty; G3 = Co-funding.
- Positive payoff slopes mean that stronger models earn more utility.
- A positive fair-share residual means that the focal agent receives more than
  its NBS reference share (G1–G2) or Lindahl reference share (G3).
- For dense scatterplots, the tables report the fitted trend and the smallest
  useful numerical compression of the plotted points. For discrete bar and
  line plots, every displayed condition is retained.

## One-glance figure map

| Figure | Question encoded by the figure | Reviewer takeaway | Old NeurIPS submission — exact counterpart? |
|:--|:--|:--|:--|
| 1 | How does one negotiation episode run? | Agents iterate through discussion, proposal, voting, and reflection while optimizing private preferences. | **Yes: old Figure 1, main text; same number.** |
| 2 | Does bilateral payoff scale with capability, and who benefits? | Adversary payoff rises by about 6.8–7.4 utility per 100 Elo. Cooperation lifts the baseline too; competition increasingly shifts value toward the stronger adversary. | **No exact counterpart.** Closest: old Figures 2 and 3, both main text. |
| 3 | Does capability move agents above or below game-theoretic fair share? | Focal-agent residuals rise with Elo and cross zero around Elo 1394–1472. Baseline-agent residuals rise only mildly and remain negative. | **No exact counterpart.** Closest: old Figure 5 (main text) and Figure 29 (appendix). |
| 4 | Which behaviors predict payoff, and which scale with Elo? | Trade/compromise and logical persuasion are payoff-positive and increase with Elo. Pressure increases most strongly but is slightly payoff-negative. | **No; not in the old paper.** |
| 5 | Does requested test-time reasoning effort improve payoff? | No reliable payoff scaling appears; uncertainty intervals overlap broadly despite large changes in observed tokens. | **No exact counterpart.** Closest: old Figure 18 (appendix); old Figure 6 (main text) was a different TTC scatterplot. |
| 6 | What does test-time reasoning effort change behaviorally? | GPT-5 increases both payoff-positive persuasion and payoff-negative self-interest; Gemini gains logical persuasion but loses trade/compromise and increases pressure/formalization. | **No; not in the old paper.** |
| 7 | Does capability scaling persist for larger heterogeneous groups? | Yes. Game 1 payoff has a positive Elo slope for every tested group size, N=2 through N=10. | **No exact standalone counterpart.** Its Game 1 panel appeared within old Figure 7 (main text; same number). |
| 8 | Is heterogeneity itself the main source of payoff inequality? | No. Aggregate heterogeneous and monoculture Gini are nearly tied; the monoculture model's capability explains the large within-game differences. | **No; not in the old paper.** The 325-run monoculture control is new. |
| 9 | What happens when one stronger adversary is inserted among GPT-5-nano peers? | Inequality among the baseline agents falls, while the adversary's payoff advantage widens from -1.5 to +9.6. | **No exact counterpart.** Closest: old Figure 8 (main text) and Figure 27 (appendix). |

## Figure 1 — Negotiation episode workflow

**Old NeurIPS submission — exact-match status:** **Yes.** This was old
**Figure 1 in the main text**, so the figure number is unchanged.

| Stage | Information or action | Example shown in the figure | State passed forward |
|:--|:--|:--|:--|
| Initialization | The environment supplies game rules and each agent's private utility vector. | Agent 1 values Stone at 43 and Apple at 33, but values Jewel at only 5. | Private preferences and a shared negotiation task |
| 1. Start | Agents enter the episode with their own objectives. | Two agents negotiate over five items. | Initial game state |
| 2. Discussion | Agents disclose or strategically frame preferences in natural language. | Agent 1 emphasizes Stone and Apple; Agent 2 signals openness to a mutually beneficial trade. | Dialogue history |
| 3. Proposals | An agent submits a structured allocation and an accompanying rationale. | Agent 1 proposes taking Apple and Stone while assigning Jewel, Quill, and Pencil to Agent 2. | Machine-readable proposal plus reasoning |
| 4. Voting | Every agent independently accepts or rejects the proposal. | Agent 1 accepts; Agent 2 rejects because its highest-priority item, Stone, was assigned away. | Vote outcome and accepted/rejected state |
| 5. Reflection | Agents privately diagnose why the round failed or succeeded. | Agent 1 notices that the two agents' high-value sets may not overlap and that a deal remains possible. | Reflection used by the next discussion round |
| Repeat or terminate | Discussion resumes after rejection; unanimous acceptance ends the game. | The dashed cycle returns from reflection to discussion. | Final allocation, utilities, consensus status, and round count |

**What to notice:** the environment separates private preferences, public
communication, structured proposals, explicit voting, and private reflection.
That separation makes both strategic behavior and realized utility measurable.

## Figure 2 — Bilateral capability scaling against GPT-5-nano

### Panel (a): adversary payoff scaling

**Old NeurIPS submission — exact-match status:** **No exact counterpart.**
The direct precursor was old **Figure 2 in the main text**—the same figure
number as the current composite—but the current panel uses the revised
1,500-run primary protocol and therefore is not the exact submitted figure.

The active plot contains 30 model-level means per game; each point is
`mean ± SEM`, and the dashed line is an unweighted linear fit over those model
means.

| Game | Model means | Adversary payoff / 100 Elo | 95% slope CI | Typical plotted point | R² |
|:--|--:|--:|:--|:--|--:|
| G1 — Item Allocation | 30 | +6.75 | [4.56, 8.95] | Mean ± 5.72 SEM | 0.56 |
| G2 — Diplomatic Treaty | 30 | +6.76 | [4.63, 8.89] | Mean ± 4.28 SEM | 0.58 |
| G3 — Co-funding | 30 | +7.40 | [4.92, 9.89] | Mean ± 5.19 SEM | 0.55 |

Here “typical” is the median SEM across the 30 plotted model means; individual
points retain their own SEM in the figure.

**What to notice:** the slope is positive and similar in magnitude in all
three games: stronger adversaries consistently earn more.

### Panel (b): baseline payoff at the cooperative and competitive extremes

**Old NeurIPS submission — exact-match status:** **No exact counterpart.**
The closest precursor was old **Figure 3 in the main text**—a different figure
number—which showed all competition strata. The current panel is an
endpoint-only redesign incorporated into new Figure 2.

For rapid reading, the continuous per-Elo curves are compressed into four
equal-count adversary-Elo bands. Values are baseline payoff `mean ± SEM` over
the endpoint runs in each band. The plotted figure uses the same endpoint runs
but displays an exponentially smoothed per-Elo mean with a per-Elo SEM ribbon.

| Game | Adversary Elo band | Max-cooperative baseline payoff | Max-competitive baseline payoff |
|:--|:--|--:|--:|
| G1 | Q1 (1110–1302) | 87.5 ± 5.0 | 59.7 ± 6.6 |
| G1 | Q2 (1317–1358) | 85.5 ± 8.1 | 45.7 ± 3.9 |
| G1 | Q3 (1363–1448) | 91.9 ± 6.2 | 47.1 ± 3.9 |
| G1 | Q4 (1468–1504) | 100.0 ± 0.0 | 44.5 ± 2.7 |
| G2 | Q1 (1110–1302) | 88.8 ± 2.6 | 79.9 ± 6.2 |
| G2 | Q2 (1317–1358) | 94.6 ± 1.7 | 76.3 ± 4.7 |
| G2 | Q3 (1363–1448) | 97.3 ± 0.8 | 62.2 ± 4.1 |
| G2 | Q4 (1468–1504) | 99.2 ± 0.3 | 60.9 ± 4.1 |
| G3 | Q1 (1110–1302) | 28.2 ± 2.3 | 9.2 ± 5.1 |
| G3 | Q2 (1317–1358) | 32.6 ± 2.7 | 12.0 ± 7.1 |
| G3 | Q3 (1363–1448) | 34.8 ± 1.9 | 8.2 ± 4.2 |
| G3 | Q4 (1468–1504) | 39.6 ± 1.7 | 1.7 ± 2.6 |

**What to notice:** under maximal cooperation, stronger adversaries generally
help the fixed baseline too. Under maximal competition, the baseline payoff
falls as adversary capability increases, most sharply in G2 and G3.

## Figure 3 — Capability and utility relative to fair share

**Old NeurIPS submission — exact-match status:** **No exact counterpart.**
Its two closest precursors were old **Figure 5 in the main text** (bilateral
fair-share decomposition) and old **Figure 29 in the appendix** (multi-agent
agent-level benchmark residuals). The current figure combines and redesigns
those analyses, so neither old figure had the current Figure 3 number.

The figure itself shows point estimates and fitted lines without error bars.
The low- and high-Elo values below are values on the displayed linear fit,
rather than individual noisy endpoint observations.

| Series | Elo span | Fitted residual at low Elo | Fitted residual at high Elo | Δ utility / 100 Elo | Fitted zero crossing |
|:--|:--|--:|--:|--:|:--|
| Bilateral — G1 | 1110–1504 | -15.69 | +6.07 | +5.52 | 1394 |
| Bilateral — G2 | 1110–1504 | -22.59 | +2.79 | +6.44 | 1461 |
| Bilateral — G3 | 1110–1504 | -5.06 | +1.24 | +1.60 | 1426 |
| Multi-agent — heterogeneous focal agent | 1240–1504 | -7.49 | +1.03 | +3.23 | 1472 |
| Multi-agent — inserted adversary | 1240–1484 | -9.51 | +2.55 | +4.94 | 1432 |
| Multi-agent — baseline-agent mean | 1240–1484 | -7.61 | -5.31 | +0.94 | No crossing in range |
| Multi-agent — GPT-5-nano control | 1337 only | -7.93 | -7.93 | — | Not estimable from one Elo |

**What to notice:** all focal-agent trends rise and cross the fair-share line
near Elo 1400–1470. The baseline agents improve only mildly and remain below
their fair-share benchmark throughout the observed range.

## Figure 4 — Strategic behavior in bilateral play

**Old NeurIPS submission — exact-match status:** **No; not in the old paper.**
The submitted paper had selected qualitative case studies, but no systematic
six-category figure in either the main text or appendix.

The left-panel correlations include all 1,891 payoff-valid speaker rollouts,
including zero event counts. The right-panel capability slopes summarize the
29 displayed model points per category after the figure's one-outlier removal;
the figure then applies a centered five-model smoother.

| Behavior category | Payoff Spearman ρ | Payoff p | Δ events / rollout / 100 Elo | Capability trend |
|:--|--:|--:|--:|:--|
| Trade / compromise | +0.17 | 2.1e-14 | +0.22 | Increases |
| Emotional persuasion | +0.13 | 4.3e-09 | +0.01 | Roughly flat |
| Logical persuasion | +0.11 | 2.0e-06 | +0.32 | Increases |
| Pressure | -0.05 | 0.027 | +0.49 | Increases |
| Self-interest / exploitation | -0.27 | 8.7e-34 | +0.01 | Roughly flat |
| Formalization | -0.34 | 9.5e-53 | +0.02 | Roughly flat |

**What to notice:** stronger agents increasingly use trade/compromise and
logical persuasion, which are payoff-positive. Pressure increases even faster
but is slightly payoff-negative; overt self-interest and formalization have
the strongest negative payoff associations.

## Figure 5 — Test-time compute payoff

**Old NeurIPS submission — exact-match status:** **No exact counterpart.**
The closest precursor was old **Figure 18 in the appendix**, which plotted
target and baseline utility as two effort-level lines. Old **Figure 6 in the
main text** was instead a token–payoff scatterplot. The current payoff-only bar
figure is therefore neither old Figure 18 nor old Figure 6.

Each payoff is the mean over nine order-averaged game cells, reported as
`mean ± SEM`. Each underlying effort level contains 18 runs across both model
orders.

| Model | Requested effort | Mean target payoff ± SEM | Observed target tokens / call | Game cells |
|:--|:--|--:|--:|--:|
| GPT-5 | Minimal | 62.5 ± 11.4 | 0 | 9 |
| GPT-5 | Low | 61.6 ± 11.2 | 395 | 9 |
| GPT-5 | Medium | 65.3 ± 11.2 | 1,494 | 9 |
| GPT-5 | High | 65.7 ± 10.3 | 1,342 | 9 |
| Claude Sonnet 4.6 | Low | 68.2 ± 9.6 | 1,490 | 9 |
| Claude Sonnet 4.6 | Medium | 68.0 ± 9.8 | 1,712 | 9 |
| Claude Sonnet 4.6 | High | 69.5 ± 8.6 | 1,479 | 9 |
| Claude Sonnet 4.6 | Max | 70.4 ± 8.5 | 1,228 | 9 |
| Gemini 3 Flash | Minimal | 65.7 ± 8.1 | 241 | 9 |
| Gemini 3 Flash | Low | 67.6 ± 9.2 | 357 | 9 |
| Gemini 3 Flash | Medium | 65.1 ± 9.8 | 1,677 | 9 |
| Gemini 3 Flash | High | 65.6 ± 9.6 | 1,950 | 9 |

**What to notice:** payoff changes are small relative to their SEMs. Gemini's
observed tokens rise roughly eightfold from minimal to high effort while mean
payoff is effectively unchanged; Claude's observed-token proxy is not
monotone in requested effort.

## Figure 6 — Test-time compute and strategic behavior

Values are average unique turn-level occurrences per rollout. Every
model-effort cell contains 18 rollouts. The active figure does not display
error bars.

### GPT-5

**Old NeurIPS submission — exact-match status:** **No; not in the old paper.**
There was no TTC strategic-behavior figure in either the old main text or
appendix.

| Behavior | Minimal | Low | Medium | High | Minimal → high |
|:--|--:|--:|--:|--:|--:|
| Emotional persuasion | 0.50 | 0.61 | 0.67 | 0.83 | +0.33 |
| Trade / compromise | 3.83 | 4.39 | 3.50 | 4.06 | +0.22 |
| Logical persuasion | 2.78 | 2.78 | 2.72 | 3.56 | +0.78 |
| Pressure | 2.89 | 3.61 | 3.56 | 3.22 | +0.33 |
| Self-interest / exploitation | 1.28 | 2.22 | 1.83 | 2.44 | +1.17 |
| Formalization | 0.94 | 0.11 | 0.22 | 0.50 | -0.44 |

### Gemini 3 Flash

**Old NeurIPS submission — exact-match status:** **No; not in the old paper.**
There was no TTC strategic-behavior figure in either the old main text or
appendix.

| Behavior | Minimal | Low | Medium | High | Minimal → high |
|:--|--:|--:|--:|--:|--:|
| Emotional persuasion | 0.78 | 0.78 | 0.72 | 0.72 | -0.06 |
| Trade / compromise | 4.06 | 3.67 | 3.61 | 3.28 | -0.78 |
| Logical persuasion | 2.94 | 3.83 | 4.28 | 4.00 | +1.06 |
| Pressure | 2.28 | 2.33 | 3.61 | 3.39 | +1.11 |
| Self-interest / exploitation | 2.67 | 2.39 | 1.61 | 2.33 | -0.33 |
| Formalization | 0.33 | 0.39 | 0.78 | 1.39 | +1.06 |

**What to notice:** more reasoning changes the behavioral mix, but not in a
uniformly beneficial direction. GPT-5's largest increase is in
self-interest/exploitation; Gemini's gain in logical persuasion is paired
with less trade/compromise and more pressure and formalization.

## Figure 7 — Heterogeneous Game 1 payoff scaling by group size

**Old NeurIPS submission — exact-match status:** **No exact standalone
counterpart.** The corresponding Game 1 panel appeared inside old **Figure 7
in the main text**, so the number is the same, but the submitted figure also
contained Games 2 and 3 and used the older panel composition.

Each row summarizes the same unweighted linear fit over the 24 model-level
means shown for that group size. The active figure intentionally omits error
bars.

| Group size | Models | Mean observations / model | Payoff / 100 Elo | R² |
|:--|--:|--:|--:|--:|
| N=2 | 24 | 8.3 | +7.49 | 0.34 |
| N=4 | 24 | 16.7 | +4.99 | 0.28 |
| N=6 | 24 | 25.0 | +2.42 | 0.11 |
| N=8 | 24 | 33.3 | +3.06 | 0.20 |
| N=10 | 24 | 41.7 | +5.24 | 0.52 |

**What to notice:** the relationship is positive for every tested group size.
The weakest fit is at N=6, but the direction does not reverse as groups grow.

## Figure 8 — Heterogeneous versus homogeneous-control Gini

### Panel (a): aggregate comparison

**Old NeurIPS submission — exact-match status:** **No; not in the old paper.**
The submitted paper contained no heterogeneous-versus-random-monoculture
control figure in either the main text or appendix.

| Roster condition | Runs | Corrected within-run Gini ± SEM |
|:--|--:|--:|
| Heterogeneous random rosters | 1,300 | 0.162 ± 0.006 |
| Homogeneous controls pooled across monocultures | 325 | 0.156 ± 0.011 |

The aggregate difference is 0.006 Gini, smaller than either condition's SEM.

### Panel (b): homogeneous monocultures by model capability

**Old NeurIPS submission — exact-match status:** **No; not in the old paper.**
The capability-stratified 325-run random-monoculture control was added after
submission and has no old figure number.

| Game | Monoculture model | Elo | Runs | Corrected within-run Gini ± SEM |
|:--|:--|--:|--:|--:|
| G1 | Claude 3 Haiku | 1260 | 25 | 0.448 ± 0.052 |
| G1 | GPT-5 nano | 1337 | 25 | 0.193 ± 0.027 |
| G1 | Qwen3 Max | 1435 | 25 | 0.138 ± 0.022 |
| G1 | Opus 4.5 | 1468 | 25 | 0.141 ± 0.023 |
| G1 | Gemini 3.1 Pro | 1494 | 25 | 0.187 ± 0.031 |
| G2 | Nova Pro | 1290 | 20 | 0.052 ± 0.011 |
| G2 | GPT-4o | 1345 | 20 | 0.046 ± 0.008 |
| G2 | o3-mini | 1363 | 20 | 0.003 ± 0.002 |
| G2 | GPT-5.2 Chat | 1478 | 20 | 0.027 ± 0.006 |
| G2 | Opus 4.6 | 1499 | 20 | 0.037 ± 0.010 |
| G3 | Nova Micro | 1240 | 20 | 0.325 ± 0.068 |
| G3 | DeepSeek V3 | 1358 | 20 | 0.252 ± 0.053 |
| G3 | DeepSeek R1 | 1422 | 20 | 0.190 ± 0.033 |
| G3 | Opus 4.5 Think | 1474 | 20 | 0.134 ± 0.034 |
| G3 | GPT-5.4 High | 1484 | 20 | 0.090 ± 0.028 |

**What to notice:** heterogeneity is not associated with higher aggregate
inequality after pooling. Within the homogeneous controls, however, weak
monocultures—especially in G1 and G3—are much more unequal than capable ones.
This is why model capability, rather than heterogeneity alone, drives the
figure's pattern.

## Figure 9 — Homogeneous-adversary inequality and role payoff

**Old NeurIPS submission — exact-match status:** **No exact counterpart.**
The closest precursors were old **Figure 8 in the main text**
(homogeneous-adversary payoff scaling by group size) and old **Figure 27 in
the appendix** (Gini by group size). The Elo-quartile baseline-only Gini and
role-payoff composition is new; current Figure 9 therefore does not match
either old number.

The left-panel Gini bars are `mean ± SEM`. The active manuscript's right panel
shows role-payoff means without uncertainty bars, so the payoff columns below
are means only. The adversary gap is adversary payoff minus the mean payoff of
one baseline agent.

| Adversary Elo quartile | Runs | Baseline-only Gini ± SEM | Adversary payoff | Mean baseline payoff | Adversary gap |
|:--|--:|--:|--:|--:|--:|
| Q1 (1240–1317) | 520 | 0.186 ± 0.010 | 44.2 | 45.8 | -1.5 |
| Q2 (1389) | 260 | 0.172 ± 0.014 | 51.0 | 45.6 | +5.4 |
| Q3 (1448) | 260 | 0.144 ± 0.013 | 52.9 | 46.4 | +6.6 |
| Q4 (1484) | 260 | 0.142 ± 0.013 | 58.0 | 48.3 | +9.6 |

**What to notice:** baseline-only inequality falls by about 24% from Q1 to Q4
(0.186 to 0.142), yet the adversary's mean advantage grows by 11.1 utility
points (-1.5 to +9.6). Lower Gini among the baseline fleet therefore does not
mean that the stronger inserted agent is sharing value more equally with that
fleet.

## Source scope

These tables follow the active figure assets and their checked producer inputs:

| Figure | Numerical scope used by the active asset | Old NeurIPS exact counterpart |
|:--|:--|:--|
| 1 | Manual conceptual figure; no experimental estimates | Yes — Figure 1, main text; same number |
| 2 | 1,500 primary bilateral runs; 30 adversary models; G1 uses the two-turn protocol only | No — closest are Figures 2–3, main text |
| 3 | The same 1,500 bilateral runs plus 1,300 heterogeneous, 1,300 homogeneous-adversary, and 130 homogeneous-control multi-agent runs | No — closest are Figure 5, main text, and Figure 29, appendix |
| 4 | 1,920 accepted bilateral rollout speakers; 1,891 have valid payoff for the correlation panel | No — not in old paper |
| 5 | 216 TTC runs: 72 per model family and 18 per family-effort cell | No — closest is Figure 18, appendix; Figure 6, main text, is related but different |
| 6 | 144 displayed GPT-5 and Gemini TTC rollouts; 18 per family-effort cell | No — not in old paper |
| 7 | 1,300 canonical heterogeneous multi-agent runs; G1 is displayed | No exact standalone match — Game 1 appeared within Figure 7, main text; same number |
| 8 | 1,300 heterogeneous runs and 325 random-monoculture control runs | No — not in old paper |
| 9 | 1,300 canonical homogeneous-adversary runs | No — closest are Figure 8, main text, and Figure 27, appendix |
