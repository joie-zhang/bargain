# Reviewer Item 8: Does the Elo–Payoff Slope Decrease as \(N\) Increases?

## Executive conclusion

The reviewer’s premise is **not generally true**.

The apparent decrease occurs primarily in the **heterogeneous-roster raw-payoff
figure**, and it has three different explanations across the three games:

1. **Item allocation:** There is no robust decrease in capability advantage.
   The apparent raw-payoff decline is mostly payoff-scale compression and
   aggregation imbalance. After normalizing by each instance’s equal share of
   optimal welfare, the Elo slope increases with \(N\).
2. **Treaty negotiation:** There is a moderate, ecology-dependent reduction in
   an individual agent’s relative payoff advantage. It is consistent with
   continuous compromise and supermajority aggregation compressing differences
   among agents. It is not present in the controlled focal-adversary design.
3. **Participatory budgeting:** There is a large and robust decrease in an
   individual agent’s relative payoff advantage in heterogeneous groups.
   Higher-Elo agents continue to obtain more benefit from funded projects, but
   as \(N\) grows they shoulder more of the contribution cost. Their
   coordination gains increasingly spill over to the group.

The correct high-level conclusion is therefore:

> Increasing group size does not mechanically weaken Elo–payoff scaling.
> Group size changes how capability is converted into private payoff, and this
> depends jointly on the game mechanism and the capability ecology.

## Data and estimands

This analysis uses all 2,730 canonical completed multi-agent runs and all 16,380
agent-level observations:

- 1,430 homogeneous runs, including 1,300 controlled focal-adversary runs.
- 1,300 heterogeneous-roster runs.
- \(N \in \{2,4,6,8,10\}\).

Two experimental designs must be kept separate:

- **Controlled focal adversary:** one varied model negotiates with \(N-1\)
  GPT-5-nano agents. This asks whether adding weak co-players dilutes the focal
  agent’s capability advantage.
- **Heterogeneous rosters:** all agents are sampled from the 24-model roster.
  This asks how an agent’s capability predicts payoff in a mixed-capability
  society.

Two slope estimands are reported:

- **Model-mean slope:** the slope used in the paper figure, obtained by
  averaging each model’s payoff at a given \(N\) and regressing those averages
  on Elo.
- **Within-run slope:** the relationship between an agent’s Elo advantage over
  its actual co-players and its payoff advantage over those same co-players.
  This removes shared run difficulty, competition-cell difficulty, and group
  welfare shocks.

Uncertainty intervals use 3,000 run-cluster bootstrap replicates, stratified by
competition setting. Formal within-run interaction regressions use two-way
cluster-robust uncertainty by run and model and control for agent position.

## 1. Is the raw slope consistently decreasing?

No.

### Controlled focal-adversary design

Raw payoff slope per 100 Elo:

| \(N\) | Item allocation | Treaty | Participatory budgeting |
|---:|---:|---:|---:|
| 2 | 6.36 | 4.30 | 1.59 |
| 4 | 8.62 | 4.88 | 5.02 |
| 6 | 8.65 | 4.91 | 5.80 |
| 8 | 7.71 | 4.43 | 1.69 |
| 10 | 10.84 | 4.44 | 3.08 |

The \(N=10\) minus \(N=2\) changes are:

- Item allocation: \(+4.49\), bootstrap 95% interval
  \([-0.52, 9.63]\).
- Treaty: \(+0.14\), interval \([-2.94, 3.38]\).
- Participatory budgeting: \(+1.49\), interval \([-2.98, 5.98]\).

Thus a larger GPT-5-nano fleet does **not** systematically dilute the focal
agent’s Elo advantage.

### Heterogeneous-roster design

Raw model-mean payoff slope per 100 Elo:

| \(N\) | Item allocation | Treaty | Participatory budgeting |
|---:|---:|---:|---:|
| 2 | 7.49 | 4.79 | 5.59 |
| 4 | 4.99 | 4.53 | 6.39 |
| 6 | 2.42 | 3.20 | 6.52 |
| 8 | 3.06 | 1.62 | 3.64 |
| 10 | 5.24 | 3.87 | 1.89 |

All three endpoint slopes are numerically lower at \(N=10\), but none decreases
monotonically:

- Item allocation decreases twice and then rebounds twice.
- Treaty decreases through \(N=8\) and rebounds at \(N=10\).
- Participatory budgeting increases through \(N=6\), then decreases.

The endpoint changes are:

- Item allocation: \(-2.25\), interval \([-7.39, 2.84]\).
- Treaty: \(-0.92\), interval \([-3.66, 1.86]\).
- Participatory budgeting: \(-3.70\), interval \([-6.90, -0.66]\).

Only the participatory-budgeting model-mean endpoint change is clearly
distinguishable from zero.

![Raw slopes by experimental design](../../analysis/reviewer_item8_n_slope_20260725/slope_by_design.png)

## 2. Does the conclusion survive within-run controls?

The within-run analysis asks a cleaner question: within the same realized
group, does the higher-Elo agent receive more payoff than its co-players?

The \(N=10\) minus \(N=2\) changes in the within-run slope are:

- Item allocation: \(+0.51\), interval \([-2.08, 3.38]\).
- Treaty: \(-3.10\), interval \([-5.60, -0.65]\).
- Participatory budgeting: \(-5.42\), interval \([-8.50, -2.51]\).

A pooled linear interaction model estimates the change in raw within-run slope
for every two additional agents:

- Item allocation: \(+0.30\), 95% CI \([-0.37, 0.97]\), \(p=.383\).
- Treaty: \(-0.44\), CI \([-0.84,-0.04]\), \(p=.030\).
- Participatory budgeting: \(-1.14\), CI \([-1.72,-0.55]\),
  \(p=1.5\times10^{-4}\).

This establishes that the phenomenon is game-specific:

- No evidence of declining relative capability advantage in Game 1.
- A moderate decline in Game 2.
- A large decline in Game 3.

## 3. Is the result caused by raw payoff-scale compression?

### Game 1: yes

The equal-share optimum \(SW^\star/N\) falls from 75.3 at \(N=2\) to 42.7 at
\(N=10\), a 43% contraction. The agents are dividing rivalrous items, so the
amount of raw utility available to an individual naturally shrinks as the group
grows.

After normalizing utility as

\[
\widetilde U_i = \frac{N U_i}{SW^\star},
\]

the model-mean slope grows from 0.092 to 0.213 normalized-utility units per 100
Elo. The endpoint increase is \(+0.121\), interval \([0.068,0.174]\).

The within-run normalized interaction also increases by 0.033 per two
additional agents, \(p=3.2\times10^{-6}\).

Therefore, Game 1’s raw slope does not show genuine capability dilution. On an
equal-share-of-optimum scale, capability becomes more predictive.

### Game 2: only slightly

The equal-share optimum falls only 6%, from 93.6 to 88.0. Raw payoff-scale
change cannot fully explain the reduction in within-run payoff advantage.

The normalized linear interaction is small and only marginal:
\(-0.0044\) per two additional agents, \(p=.066\). This supports describing the
effect as moderate rather than a major qualitative collapse.

### Game 3: no

The equal-share optimum rises from 30.2 to 44.7, so the payoff opportunity
expands rather than contracts. Nevertheless, the normalized model-mean slope
falls from 0.205 to 0.052, with endpoint change \(-0.153\), interval
\([-0.241,-0.071]\).

Game 3 therefore contains a real shift in how capability maps to individual
payoff.

![Raw and normalized heterogeneous slopes](../../analysis/reviewer_item8_n_slope_20260725/heterogeneous_raw_vs_normalized.png)

## 4. Is the apparent decline just greater noise at high \(N\)?

No.

Several diagnostics point in the opposite direction:

- The same 24-model pool is used at every \(N\).
- Each model has more observations at larger \(N\), not fewer. For example,
  Game 3 increases from 2–12 observations per model at \(N=2\) to 25–42 at
  \(N=10\).
- Bootstrap intervals generally become narrower at high \(N\).
- Model-mean \(R^2\) increases from 0.345 to 0.524 in Game 1 and from 0.259 to
  0.645 in Game 2.
- Game 3’s \(R^2\) changes from 0.358 to 0.278, but the decline in the slope is
  still clear under the run-cluster bootstrap.

The high-\(N\) estimates are not merely less precise. In Game 3, the
between-model payoff spread itself compresses: the standard deviation of model
mean payoff falls from 7.71 to 2.97.

## 5. Is roster composition or competition mix responsible?

It explains some, but not all, of the pattern.

### Roster composition

At \(N=2\), the sparse random pairing design produces only 2–17 observations
per model, depending on the game. Model exposure to competition cells and
co-player strength is therefore uneven. At \(N=10\), every model is observed
many more times.

The within-run estimator removes this group-composition problem. It eliminates
the apparent Game 1 decrease but preserves the Game 2 and Game 3 decreases.

### Competition settings

In heterogeneous Game 1, only 1 of 5 competition settings has a lower
\(N=10\) slope than \(N=2\); only 1 of 5 has a negative linear trend across
\(N\). The pooled raw decline is therefore not representative of its
competition cells.

In Game 2:

- All four cells have a lower \(N=10\) endpoint.
- The negative-correlation condition itself changes with \(N\), from
  \(\rho=-1\) at \(N=2\) to \(\rho=-0.106\) at \(N=10\), because of the
  feasible equicorrelation bound.
- However, the fixed high-alignment cells also decline, so this changing
  geometry is only a partial explanation.

In Game 3:

- Three of four cells have lower \(N=10\) endpoints.
- All four have negative fitted trends across \(N\).

![Competition-specific slopes](../../analysis/reviewer_item8_n_slope_20260725/competition_cell_slopes.png)

## 6. Are agreement failure, longer deliberation, or discounting responsible?

No.

Heterogeneous consensus rates from \(N=2\) to \(N=10\) are:

- Game 1: 0.98 to 1.00.
- Game 2: 1.00 to 1.00.
- Game 3: 0.925 to 1.00.

Mean agreement rounds are:

- Game 1: 1.76 to 1.33.
- Game 2: 1.40 to 1.43.
- Game 3: 2.39 to 1.95.

Thus high-\(N\) runs neither fail more often nor systematically settle later.
Undoing discounting produces essentially the same slope pattern. The Game 3
benefit/cost decomposition below uses undiscounted utilities and still shows
the decline.

All canonical runs in this analysis completed successfully. Protocol-repair
markers become more likely when a run contains more agents and calls, but this
cannot explain the central result:

- The slope decline is absent in Game 1 despite rising repair exposure.
- Game 3’s voting diagnostics remain strictly clean.
- The Game 3 effect persists within the same run and decomposes into
  economically meaningful benefit and contribution channels.

## 7. What fundamentally changes in treaty negotiation?

The treaty game uses a continuous agreement vector implemented by a
two-thirds supermajority. As \(N\) grows:

- A single agent’s ideal point has less influence over the accepted vector.
- More proposals or perspectives can locate an acceptable central compromise.
- A winning proposal must satisfy a larger coalition.
- The final treaty therefore reflects an aggregate of preferences more than
  any one negotiator’s bargaining power.

The data are consistent with this aggregation mechanism:

- Mean within-run payoff standard deviation falls from 8.45 to 6.16, a 27%
  contraction.
- The within-run Elo–payoff slope falls from 6.75 to 3.65.
- At \(N=10\), a 100-point increase in the group’s mean Elo predicts
  \(+5.17\) mean utility, while an individual’s 100-point relative Elo
  advantage predicts \(+3.62\).

This suggests that capability increasingly acts through better group
coordination, while individual payoff differences are compressed.

This is not a universal consequence of voting alone: the controlled focal
model’s treaty slope stays almost exactly constant from \(N=2\) to \(N=10\).
The reduction appears when capability is distributed across a heterogeneous
group, not when one capable focal model is surrounded by identical weaker
agents.

## 8. What fundamentally changes in participatory budgeting?

Game 3 allows a direct decomposition:

\[
\text{utility} =
\text{benefit from funded projects}
-
\text{contribution paid}.
\]

Within-run slopes per 100 Elo are:

| \(N\) | Benefit slope | Contribution-cost slope | Net-utility slope |
|---:|---:|---:|---:|
| 2 | 8.16 | 1.05 | 7.11 |
| 4 | 5.63 | 1.68 | 3.96 |
| 6 | 5.96 | 1.92 | 4.04 |
| 8 | 4.87 | 2.79 | 2.08 |
| 10 | 4.45 | 3.19 | 1.25 |

Two changes jointly flatten the net-payoff slope:

1. The private benefit advantage decreases. Higher-Elo agents still benefit
   more, but the benefit slope falls from 8.16 to 4.45.
2. The contribution burden becomes more positively associated with Elo. The
   cost slope triples from 1.05 to 3.19.

The Lindahl comparison supports the same interpretation:

- At \(N=2\), higher Elo predicts greater underpayment relative to the
  benefit-proportional benchmark: \(+1.75\) per 100 Elo within runs.
- At \(N=10\), the slope is \(-0.77\): higher-Elo agents tend to pay **more**
  relative to their Lindahl contribution benchmark.

Meanwhile, group capability continues to improve collective outcomes. In Game
3, a 100-point increase in group mean Elo has a positive, statistically clear
association with mean normalized welfare at every \(N\).

The resulting mental model is:

> In small public-good negotiations, capability can produce private surplus
> capture. In larger groups, capable agents increasingly act as coordinators
> and threshold-closing contributors. The projects they help fund benefit many
> co-players, while the capable agents bear more of the cost. Capability remains
> useful, but its gains become more socialized.

![Mechanism diagnostics](../../analysis/reviewer_item8_n_slope_20260725/mechanism_diagnostics.png)

## Hypothesis audit

| Hypothesis | Finding |
|---|---|
| Group size mechanically dilutes all capability advantages | Rejected: controlled focal slopes do not decline |
| Raw payoff scale shrinks with \(N\) | Strong explanation for Game 1; minor for Game 2; opposite direction in Game 3 |
| High-\(N\) results are simply noisier | Rejected as the primary explanation; model exposure and precision improve |
| Random roster/cell imbalance creates the visual pattern | Important for Game 1; insufficient for Games 2–3 |
| Harder consensus or longer bargaining suppresses payoff | Rejected; agreement rates stay high and rounds do not increase |
| Agent order drives the effect | Rejected as primary; position-controlled within-run estimates are similar |
| Changing Game 2 correlation geometry drives the effect | Partial explanation, but fixed high-alignment cells also decline |
| Supermajority aggregation reduces one agent’s influence | Supported for heterogeneous Games 2–3, but ecology-dependent |
| Strong co-players create capability spillovers | Supported, especially in Game 3 |
| High-Elo agents increasingly bear public-good costs | Strongly supported in Game 3 |

## Recommended response to the reviewer

Do not concede that the scaling slope consistently decreases. The strongest
response is to agree with the visual observation, then replace it with the
more precise result:

1. State that the decrease appears in raw heterogeneous-roster plots but not
   in the controlled focal-adversary design.
2. Report raw and equal-share-normalized slopes separately.
3. Add the within-run Elo-by-\(N\) interaction analysis.
4. Explain the game-specific mechanisms:
   - Game 1: scale compression, not capability dilution.
   - Game 2: modest compromise/aggregation effect.
   - Game 3: public-good spillovers plus rising contribution burden.
5. Avoid presenting a single causal mechanism across all games.

### Suggested rebuttal language

> We thank the reviewer for pointing out the apparent flattening of the
> Elo–payoff curves at larger \(N\). We reanalyzed all 2,730 multi-agent runs
> and find that this is not a universal scaling pattern. In the controlled
> design with one focal adversary among \(N-1\) GPT-5-nano agents, the
> Elo–payoff slope is stable or larger at \(N=10\) in all three games. In
> heterogeneous rosters, the raw \(N=10\) slope is lower than the \(N=2\)
> slope, but the effect is non-monotone and game-dependent. For item
> allocation, the apparent reduction disappears under within-run controls and
> reverses after normalizing by \(SW^\star/N\), because individual attainable
> payoff shrinks as more agents divide rivalrous goods. Treaty negotiation
> shows a moderate reduction in relative payoff advantage, consistent with
> continuous compromise and supermajority aggregation. Participatory
> budgeting shows the clearest reduction: higher-Elo agents still receive more
> project benefit, but as \(N\) increases they also contribute more of the
> funding cost, so capability gains spill over to co-players. We will add the
> per-\(N\) slopes, normalized analysis, interaction tests, and benefit-cost
> decomposition and revise the text to describe this mechanism-specific result
> rather than a universal decline.

## Reproducible artifacts

- Analysis script:
  [`scripts/analyze_reviewer_item8_n_slope.py`](../../scripts/analyze_reviewer_item8_n_slope.py)
- Tests:
  [`tests/test_reviewer_item8_n_slope.py`](../../tests/test_reviewer_item8_n_slope.py)
- Full output directory:
  [`analysis/reviewer_item8_n_slope_20260725`](../../analysis/reviewer_item8_n_slope_20260725)

