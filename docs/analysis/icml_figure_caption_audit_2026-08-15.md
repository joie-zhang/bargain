# ICML figure caption audit

This audit covers all 32 active figure environments in the current ICML main text and appendix. Tables are outside this audit. Each candidate starts with the figure's unique message and keeps only details needed to read the result correctly.

## Main text

### Figure 1, `fig:flow`

**Current caption**

```latex
\textbf{Negotiation game dynamics.} Agents receive private preferences, discuss publicly, form private thoughts, submit structured proposals, and then vote privately. A proposal is implemented when it reaches the two-thirds supermajority threshold, rounded up.
```

**Candidate caption**

```latex
\textbf{Each game follows the same negotiation loop.} Agents discuss, submit proposals, vote privately, and reflect until a proposal wins a two-thirds majority or the round limit is reached.
```

### Figure 2, `fig:bilateral_overview`

**Current caption**

```latex
\textbf{Bilateral capability scaling against GPT-5-nano (round-discounted payoff).} Agreement in round $t$ is multiplied by $\gamma^{t-1}$; no agreement receives zero. \textit{(a)} Adversary-model means with run-level SEMs and unweighted OLS fits. \textit{(b)} Baseline payoff at sampled minimum/interior/maximum competition (interior: $0.50/0.50/0.40$ in Games~1--3). Curves are Elo-ordered EWMs ($\alpha=0.24$); endpoint ribbons recenter raw per-Elo SEM widths and are not EWM uncertainty intervals. Competitive patterns are game-dependent.
```

**Candidate caption**

```latex
\textbf{Higher-Elo adversaries earn more, but GPT-5-nano does not always benefit.} Adversary payoff rises with Elo in all three games. GPT-5-nano payoff generally rises in cooperative settings, while competitive settings are mixed and non-monotone. Payoff is discounted by agreement round, and no agreement receives zero.
```

### Figure 3, `fig:fairshare_headline`

**Current caption**

```latex
\textbf{Adversary undiscounted fair-share residual rises with Elo.} Residuals subtract an undiscounted game-specific benchmark from undiscounted outcome utility; no agreement gives zero actual utility, so these slopes are not directly comparable with Figure~\ref{fig:bilateral_overview}'s round-discounted-payoff slopes. \textit{(a)} Bilateral $N=2$ vs.\ GPT-5-nano, crossing zero near Elo $1410$--$1461$. \textit{(b)} Multi-agent residual against reference Elo, pooled across games, competition bands, and group sizes for the displayed families. Positive values mean the focal agent receives more than its benchmark share.
```

**Candidate caption**

```latex
\textbf{The evaluated model's benchmark residual generally rises with Elo.} In bilateral games, all three adversary series move from below to above their benchmark utility. In multi-agent runs, heterogeneous agents and focal adversaries show the same pooled pattern. Positive values mean utility above the benchmark.
```

### Figure 4, `fig:n2_qualitative`

**Current caption**

```latex
\textbf{Strategic-behavior analysis of 1,500 bilateral runs using the selected 23 labels.} Left: speaker-rollout Spearman correlations between turn-deduplicated category-event count and rule-defined adversary utility, including zero utility for no agreement. Right: model mean event count per rollout vs.\ adversary Elo; each model contributes 50 rollouts. Thick curves are centered five-model moving averages, faint curves show raw model means, and one extreme model point per category is omitted before smoothing. Fifteen transcripts without a retained annotation record are zero-filled in the event counts.
```

**Candidate caption**

```latex
\textbf{Behavior categories have different links to payoff and capability.} Trade/compromise and emotional persuasion correlate positively with payoff, while formalization and self-interest/exploitation correlate negatively. Pressure, trade/compromise, and logical persuasion become more frequent at higher Elo. These are descriptive associations across 1,500 bilateral runs classified with 23 selected labels; 15 missing annotations count as zero.
```

### Figure 5, `fig:ttc_scatter`

**Current caption**

```latex
\textbf{TTC payoff scaling is family-specific.} Each bar is the mean of one balanced nine-cell, two-order estimate per comparable seed; error bars are two-sided Student-$t$ 95\% confidence intervals across seeds ($n=10/9/10$ for GPT-5/Claude/Gemini). The estimand is unconditional target payoff, with no agreement scored as zero. Lowest-to-highest requested effort is positive for Claude, borderline after Holm correction for GPT-5, and a non-detection for Gemini.
```

**Candidate caption**

```latex
\textbf{More requested reasoning effort improves payoff for Claude, but not consistently across model families.} GPT-5 shows a small borderline gain, while Gemini remains flat. Bars show mean target payoff, with 95\% confidence intervals across comparable seeds; no agreement counts as zero.
```

### Figure 6, `fig:ttc_intensities`

**Current caption**

```latex
\textbf{Requested effort shifts selected-23 strategic-behavior categories in the original seed-42 diagnostic.} Curves show mean occurrences per rollout, with SEM error bars across 18 rollouts per family--setting point. The four x-axis positions align GPT-5/Gemini settings \texttt{minimal/low/medium/high} with Claude settings \texttt{low/medium/high/max}; these provider-specific settings are ordinal and are not equal token budgets. GPT-5 ramps self-interest/exploitation, Gemini~3 Flash drops trade/compromise, and Claude varies non-monotonically across categories. These exploratory annotations are separate from the multi-seed payoff analysis.
```

**Candidate caption**

```latex
\textbf{Requested effort changes behavior, but the pattern differs by model family.} In the seed-42 diagnostic, GPT-5 shows more self-interest/exploitation, Gemini~3 Flash shows less trade/compromise, and Claude changes non-monotonically. Effort labels are provider-specific and do not represent equal token budgets.
```

### Figure 7, `fig:multiagent_hetero_payoff`

**Current caption**

```latex
\textbf{Capability still predicts utility at $N\geq2$.} Heterogeneous Game~1 payoff scaling by group size; linear fits per $N$ over per-model means. The all-three-games version is \Cref{fig:appendix_multiagent_hetero_payoff_full}.
```

**Candidate caption**

```latex
\textbf{Higher-Elo models earn more even as group size grows.} In heterogeneous Game~1 groups, payoff rises with Elo at every tested group size. \Cref{fig:appendix_multiagent_hetero_payoff_full} shows all three games.
```

### Figure 8, `fig:multiagent_hetero_vs_homo_gini`

**Current caption**

```latex
\textbf{The monoculture's capability, not heterogeneity, drives Gini inequality.} \textit{(a)} Aggregate within-run corrected Gini for heterogeneous random rosters is tied with all-monoculture pooled homogeneous controls. \textit{(b)} Per-monoculture mean Gini vs.\ the monoculture's Arena Elo; the dashed red line marks the heterogeneous reference. Weak monocultures sit above; capable monocultures sit below. The payoff-standard-deviation version in \Cref{fig:appendix_multiagent_hetero_vs_homo_payoff_std} shows a similar broad trend.
```

**Candidate caption**

```latex
\textbf{Pooled payoff inequality is similar in heterogeneous and one-model groups.} The pooled means hide game-level differences in opposite directions. Among the sampled one-model controls, higher model Elo is associated with lower Gini, but this sparse comparison does not establish cause.
```

### Figure 9, `fig:gpt54_team_coordination`

**Current caption**

```latex
\textbf{Private coordination does not improve monotonically with Nano-team size in Game~1.} \textit{(a)} Coordinated Nano-minus-GPT-5.4 payoff gap $G$, normalized by the realized preference matrix's per-capita feasible-welfare ceiling; zero denotes parity. \textit{(b)} Matched change $\Delta G$ from the historical independent-team control; zero denotes no coordination change. Points are means and bars are two-sided Student-$t$ 95\% confidence intervals over 20 matched cells per $N$. Parentheses give Nano-team size; $N=2$ is a singleton-team negative control.
```

**Candidate caption**

```latex
\textbf{Private team sharing does not reliably help GPT-5-nano agents overcome GPT-5.4.} Positive values favor the Nano agents. Three Nano agents roughly match GPT-5.4, but larger teams trail. Payoff differences are scaled by the maximum average payoff available in each game. The one-Nano point has no teammate and checks for changes between reruns. Bars show 95\% confidence intervals across 20 games per team size. Comparisons with earlier runs are descriptive because provider routes and times differ.
```

## Appendix

### Figure 10, `fig:n2_adversary_payoff_by_competition`

**Current caption**

```latex
\textbf{Bilateral adversary payoff by game and competition.} Top row: overall adversary payoff against GPT-5-nano by Elo. Bottom row: the same relationship stratified by competition settings with per-stratum linear fits.
```

**Candidate caption**

```latex
\textbf{The adversary's payoff rises with Elo within almost every competition setting.} The top row shows overall model means, and the bottom row fits each game-specific competition level separately. The main exception is the most competitive co-funding setting.
```

### Figure 11, `fig:appendix_n2_baseline_payoff_raw`

**Current caption**

```latex
\textbf{Unsmoothed scalar-competition-index version of the GPT-5-nano baseline payoff plot.} Each curve joins raw per-Elo baseline-payoff means at one exact value of $c$, $CI2$, or $CI3$. In Games~2--3, an exact index value can pool multiple underlying parameter tuples; neither figure averages across competition bands. Unlike main-text Figure~\ref{fig:n2_baseline_payoff}, this figure shows every observed scalar index value and applies no Elo-ordered EWM.
```

**Candidate caption**

```latex
\textbf{The unsmoothed historical baseline-payoff curves are noisy and non-monotone.} Lines join raw GPT-5-nano payoff means at each competition index in a 1,941-run snapshot that differs from the 1,500-run main cohort. Cooperative settings often improve with adversary Elo, while competitive settings are mixed.
```

### Figure 12, `fig:appendix_n2_welfare`

**Current caption**

```latex
\textbf{Round-discounted realized social welfare by competition stratum in bilateral play.} Solid marked curves are Elo-ordered exponentially weighted moving averages of model-level means after the $\gamma=0.9$ round discount. Dashed lines mark the largest positive, undiscounted run-specific utilitarian optimum observed within each stratum rather than a matched run-level bound, so their vertical separation from the solid curves is not an efficiency shortfall. Welfare--Elo patterns vary by game and competition stratum; maximum-competition Game~3 is especially noisy under its one-project scarcity constraint.
```

**Candidate caption**

```latex
\textbf{Capability has no uniform effect on bilateral welfare across competition settings.} Games~1 and 2 often improve, while the most competitive co-funding setting remains low and noisy. This historical 1,941-run snapshot differs from the main cohort. Curves show smoothed round-discounted welfare; dashed lines are unmatched, undiscounted reference maxima, not efficiency gaps.
```

### Figure 13, `fig:appendix_n2_benchmark_distance`

**Current caption**

```latex
\textbf{Unsigned benchmark distance decreases with adversary Elo under the retained diagnostic conventions.} Points are 30 adversary-model means from the primary cohort, bars are run-level SEMs, and dashed lines are unweighted OLS fits. The current diagnostic has 420/540/460 finite run-level distances in Games~1--3: non-agreements remain encoded under the existing Games~1--2 benchmark rules, whereas 80 no-project Game~3 runs are undefined and omitted. Model means and SEMs are min--max scaled within each game's observed roster, so 0/1 are empirical endpoints and magnitudes are not comparable across games. Lower distance means closer benchmark proximity, not which player captures the surplus.
```

**Candidate caption**

```latex
\textbf{In a historical bilateral snapshot, outcomes move closer to the benchmark point as adversary Elo rises.} Distance is scaled separately within each game. Game~3 omits 80 runs with no funded project, and the snapshot differs from the main 1,500-run cohort. This unsigned measure does not show which agent gains.
```

### Figure 14, `fig:n2_fairness`

**Current caption**

```latex
\textbf{Fairness and benchmark-relative extraction.} Stronger adversaries reduce distance from the NBS/Lindahl reference, especially in Games~1--2, but the role-specific residuals show that they sit further above their own benchmark share as Elo rises.
```

**Candidate caption**

```latex
\textbf{Closer benchmark outcomes can still favor the stronger model.} The top two rows show unsigned distance, while the bottom row shows each role's signed benchmark residual. Higher-Elo adversaries are generally closer to the benchmark point and further above their own reference. The Game~3 panels omit 80 runs with no funded project.
```

### Figure 15, `fig:appendix_n2_role_endpoint_fairness`

**Current caption**

```latex
\textbf{Baseline and adversary fair-share gaps at cooperative and competitive endpoints.} Curves plot each role's signed relative gap from its fairness benchmark, smoothed over adversary Elo with an EWM coefficient of $0.10$. Games~1--2 use the NBS benchmark; Game~3 uses the rebuttal-era enumerated Lindahl-cost-sharing Nash benchmark over feasible funded-project sets. Blue curves show the maximally cooperative endpoint and red curves show the maximally competitive endpoint; filled markers denote the GPT-5-nano baseline and open markers denote the adversary. Following the rebuttal-era sensitivity analysis, the lowest-Elo model is omitted only from the maximally competitive Game~2 curves. In cooperative settings, both roles generally move upward toward their benchmark shares as adversary capability increases. In competitive settings, especially Games~1 and~3, the adversary's fair-share gap improves as the baseline's worsens, indicating more redistributive benchmark convergence.
```

**Candidate caption**

```latex
\textbf{Cooperative settings help both roles approach their benchmark, while competitive settings in Games~1 and 2 shift the gap toward the adversary.} Curves show smoothed signed percentage gaps. This sensitivity analysis uses a different normalization and a different Game~3 funded-set benchmark, so its values are not directly comparable with \Cref{fig:fairshare_headline}.
```

### Figure 16, `fig:appendix_llama_overall`

**Current caption**

```latex
Llama 3.3 70B baseline utility replication. Left to right: Games 1--3. Each panel overlays the varied adversary's mean utility and the fixed Llama baseline's mean utility against adversary Elo; dashed lines are the corresponding linear fits.
```

**Candidate caption**

```latex
\textbf{The capability--payoff trend persists with Llama~3.3~70B as the fixed baseline.} Adversary utility rises with Elo in all three games, while the baseline changes less.
```

### Figure 17, `fig:appendix_llama_baseline_payoff`

**Current caption**

```latex
\textbf{Llama 3.3 baseline payoff against varied adversaries.} The three panels show Llama~3.3 baseline-payoff trends stratified by exact competition level in item allocation, diplomatic treaty, and co-funding.
```

**Candidate caption**

```latex
\textbf{Llama's payoff response to a varied adversary depends on competition.} Cooperative settings are generally stable or improve with adversary Elo, while the most competitive co-funding setting declines sharply.
```

### Figure 18, `fig:ttc_effort`

**Current caption**

```latex
\textbf{Original seed-42 requested-effort diagnostic.} Each panel uses the corresponding provider family's ordered effort labels on the x-axis; these are ordinal requested settings, not equal token budgets across providers. Solid lines show target utility, dotted lines show GPT-5-nano baseline utility, points are means over 18 matched game-cell/order observations, and error bars are descriptive SEMs within this single seed. Multi-seed payoff inference is reported in \Cref{fig:ttc_scatter,tab:ttc_weak_strong}.
```

**Candidate caption**

```latex
\textbf{The original single-seed effort curves are weak and model-specific.} Target payoff rises slightly for GPT-5 and Claude but stays flat overall for Gemini; GPT-5-nano payoff changes little. Error bars are descriptive SEMs over 18 game-cell and order observations, not uncertainty across seeds. Effort labels are provider-specific.
```

### Figure 19, `fig:appendix_n2_baseline_random_pairings`

**Current caption**

```latex
\textbf{Fixed GPT-5-nano baseline versus heterogeneous $N=2$ pairings.} The red arm uses the 1,500-run fixed-baseline cohort; the blue arm is the separate heterogeneous $N=2$ sample. Points are model-level means with SEM error bars and dashed lines are linear fits.
```

**Candidate caption**

```latex
\textbf{Payoff rises with Elo under both fixed and random $N=2$ pairings.} The comparison is descriptive: the fixed arm uses a historical 1,941-run snapshot, while the random arm uses different game grids and opponent samples.
```

### Figure 20, `fig:appendix_n2_rounds_to_consensus`

**Current caption**

```latex
\textbf{Rounds to consensus in the GPT-5-nano bilateral sweep, conditional on agreement.} Points are model-level means among runs that reached consensus: top, pooling competition strata within each game; bottom, stratified by competition. Dashed lines are descriptive fits across defined model means. No-agreement runs are right-censored at round 10 and excluded. Crosses mark model--competition cells with no successful agreement, for which mean rounds to consensus is undefined.
```

**Candidate caption**

```latex
\textbf{Among successful bilateral runs, higher-Elo adversaries reach agreement slightly faster in Games~1 and 2 but not Game~3.} Failed negotiations are censored at round 10 and excluded, so this figure does not measure bargaining length over all attempts. Crosses mark model--competition cells with no successful agreement.
```

### Figure 21, `fig:appendix_n2_order`

**Current caption**

```latex
Bilateral order diagnostic for the GPT-5-nano baseline. Each panel shows mean adversary payoff against adversary Elo, split by whether the adversary or the GPT-5-nano baseline makes the first proposal. Points are model-level means and dashed lines are linear fits.
```

**Candidate caption**

```latex
\textbf{Speaking first has no consistent payoff advantage across games.} Lines split runs by whether the adversary or GPT-5-nano occupies Agent~1, the first public speaker. Same-round proposals are generated independently, and the position arms use different preference draws, so this is a descriptive check.
```

### Figure 22, `fig:appendix_multiagent_hom_competition`

**Current caption**

```latex
\textbf{Homogeneous-adversary payoff by competition.} Columns show item allocation, diplomatic treaty, and co-funding; rows show $N=2,4,6,8,10$.
```

**Candidate caption**

```latex
\textbf{A focal model's payoff usually rises with Elo within the same competition setting.} Each panel places one focal model among $N-1$ GPT-5-nano agents. Points are four-run means with SEM bars, and dashed lines fit the five focal models. Most exceptions occur in co-funding.
```

### Figure 23, `fig:appendix_multiagent_hetero_competition`

**Current caption**

```latex
\textbf{Heterogeneous payoff by Arena Elo and competition.} Rows show $N=2,4,6,8,10$; columns show item allocation, diplomacy treaty, and co-funding. Colors encode the exact game-specific competition index, points show model-cell means with faint SEM bars, and bold dashed lines show within-competition linear fits.
```

**Candidate caption**

```latex
\textbf{Higher-Elo models usually earn more within the same competition setting in heterogeneous groups.} Sixty-two of 65 fits are positive; the three negative fits occur in co-funding. Model-cell means contain 1--14 appearances, and singleton means have no visible uncertainty bar.
```

### Figure 24, `fig:appendix_multiagent_hetero_payoff_full`

**Current caption**

```latex
\textbf{Heterogeneous payoff scaling by Arena Elo and group size, all three games.} Each point averages a model's payoff over heterogeneous runs in which it appears. Capability remains predictive across $N$ and games. The Game~1 panel is reproduced as \Cref{fig:multiagent_hetero_payoff} in the main text.
```

**Candidate caption**

```latex
\textbf{Higher-Elo models earn more in heterogeneous groups across all three games.} The Elo--payoff slope is positive at every tested group size. Each point is a model's mean payoff over the runs in which it appears.
```

### Figure 25, `fig:appendix_multiagent_hom_payoff_full`

**Current caption**

```latex
\textbf{Homogeneous-adversary payoff scaling by Arena Elo and group size, all three games.} Each point averages the inserted adversary's payoff over homogeneous-adversary runs with $N-1$ GPT-5-nano agents. All 15 displayed five-model OLS slope estimates are positive.
```

**Candidate caption**

```latex
\textbf{A focal model's payoff has a positive Elo slope in all 15 game-size panels.} Each point averages one of five focal models placed among $N-1$ GPT-5-nano agents. These five-model fits are descriptive.
```

### Figure 26, `fig:multiagent_hetero_buckets`

**Current caption**

```latex
\textbf{Mean heterogeneous-agent utility by Elo bucket and group size.} From left to right, the five panels show $N=2,4,6,8,10$. Each bar uses one of ten equal-width Arena Elo intervals containing 1--5 models: within each game, it pools all agent appearances in that interval, then averages the three game-level means equally. Higher-Elo intervals generally earn more at every group size. The mean-utility difference between the highest- and lowest-Elo intervals is $11.11$ points at $N=2$ and $9.35$ points at $N=10$.
```

**Candidate caption**

```latex
\textbf{The payoff gap between low- and high-Elo groups generally narrows as $N$ grows, then widens slightly at $N=10$.} Higher-Elo buckets tend to earn more, but adjacent buckets are not strictly ordered. Bars pool appearances within each Elo range and give the three games equal weight.
```

### Figure 27, `fig:multiagent_performance_elo`

**Current caption**

```latex
\textbf{Performance Elo inferred from within-roster payoffs.} Within each heterogeneous run, models are compared pairwise by final utility; a ridge-regularized Bradley--Terry fit converts those win, loss, and tie outcomes into a rating centered at 1500 for each game and $N$. Points are descriptive fitted ratings; model-wise sampling intervals are not shown because within-negotiation pair outcomes are dependent. Thirteen of the 15 displayed unweighted OLS slope estimates are positive. In Game~3, the estimates at $N=8$ and $N=10$ are $-13.78$ and $-22.28$ performance-Elo points per 100 Arena-Elo points; their conventional second-stage OLS 95\% $t_{22}$ intervals ($[-40.81,13.25]$ and $[-55.25,10.68]$, respectively) include zero and do not propagate first-stage rating uncertainty.
```

**Candidate caption**

```latex
\textbf{Arena Elo agrees with within-roster payoff rankings in 13 of 15 panels.} The two negative fits are co-funding at $N=8$ and $N=10$. Performance Elo comes from a filtered, ridge-regularized Bradley--Terry model and is centered separately in each panel, so levels cannot be compared across panels.
```

### Figure 28, `fig:appendix_multiagent_dilution`

**Current caption**

```latex
\textbf{Focal-adversary payoff advantage across jointly scaled group sizes.} Each point is the mean, over runs pooled across competition settings, focal positions, and seeds, of the focal adversary's payoff minus the within-run mean payoff of the $N-1$ GPT-5-nano agents; bars show $\pm1$ run-level SEM. Positive values mean that the focal adversary outperforms the baseline-agent average. Cross-$N$ differences are descriptive because other features of the bargaining environment also change with $N$.
```

**Candidate caption**

```latex
\textbf{The focal model's payoff advantage does not fade as groups grow.} From $N=2$ to $N=10$, the average focal-minus-baseline payoff gap increases in all three games, although paths differ by model. Other game features also change with $N$, so this is a descriptive comparison.
```

### Figure 29, `fig:multiagent_homo_adversary`

**Current caption**

```latex
\textbf{Homogeneous-adversary inequality and role payoffs across five models.} Each model has 260 runs; labels give identity and Arena Elo. Higher-Elo adversaries are generally associated with lower baseline-agent Gini, higher adversary payoff, and a wider role-payoff gap. The payoff-standard-deviation version in \Cref{fig:appendix_multiagent_homo_adversary_payoff_std} shows similar trends.
```

**Candidate caption**

```latex
\textbf{Higher focal-model Elo is associated with lower inequality among GPT-5-nano agents and a wider focal payoff lead.} Across five focal models, baseline-agent Gini falls, focal payoff rises, and average baseline payoff changes little.
```

### Figure 30, `fig:multiagent_fairness_efficiency`

**Current caption**

```latex
Fairness, inequality, and welfare efficiency as group size increases, using all 2,730 terminal outcomes. The corrected-Gini row applies $G_{\mathrm{corr}}=\min\{NG/(N-1),1\}$ within each run before aggregation. For heterogeneous rosters, the corrected $N=10$ Gini endpoint is below $N=2$ in Games~2--3; the Game~3 path is nonmonotonic. Game~1 shows the clearest fairness degradation with $N$. For Game~3, benchmark distance compares executed payments on the realized funded-project set with benefit-proportional cost shares for that set; refunded pledges are excluded. No-consensus outcomes are retained with zero realized utilities. In Game~3, an empty funded set therefore has zero payment distance and an all-zero utility vector has Gini zero by convention; these zeros describe the all-terminal outcome mixture and are not evidence of a successful fair agreement.
```

**Candidate caption**

```latex
\textbf{Larger groups do not have one consistent effect on benchmark distance, payoff inequality, or welfare.} Game~1 shows the clearest rise in benchmark distance and fall in welfare efficiency, while Games~2 and 3 vary by roster type. The figure includes all 2,730 terminal runs, with failed agreements scored at zero utility. In co-funding, an empty funded set has zero distance and zero Gini by convention, so lower values need not mean a successful agreement.
```

### Figure 31, `fig:appendix_multiagent_hetero_vs_homo_payoff_std`

**Current caption**

```latex
\textbf{Payoff-standard-deviation analogue of \Cref{fig:multiagent_hetero_vs_homo_gini}.} \textit{(a)} Aggregate within-run payoff standard deviation for heterogeneous random rosters and pooled homogeneous-control monocultures. \textit{(b)} Per-monoculture mean payoff standard deviation against the monoculture model's Arena Elo; the dashed red line marks the heterogeneous reference. The broad pattern is similar to the Gini analysis: pooled dispersion is close across the two designs, and weaker monocultures tend to show greater dispersion than more capable monocultures, although the absolute-scale statistic is more sensitive to high-dispersion runs.
```

**Candidate caption**

```latex
\textbf{Heterogeneous and one-model groups have similar pooled payoff spread.} Among the sampled one-model controls, higher Elo is generally associated with smaller within-run payoff differences. This absolute-scale check supports the Gini pattern but still pools across games.
```

### Figure 32, `fig:appendix_multiagent_homo_adversary_payoff_std`

**Current caption**

```latex
\textbf{Payoff-standard-deviation analogue of \Cref{fig:multiagent_homo_adversary}.} Left: within-run payoff standard deviation among the $N-1$ GPT-5-nano baseline agents, grouped by adversary-Elo bin. Right: adversary and mean per-baseline payoff, retained from the main-text analysis. The standard-deviation view shows a similar trend to Gini: dispersion among the baseline agents generally falls as adversary capability rises, while the adversary's payoff increases and the mean baseline payoff remains comparatively flat.
```

**Candidate caption**

```latex
\textbf{A higher-Elo focal model is associated with less payoff spread among GPT-5-nano agents and a larger focal payoff lead.} The four Elo bins contain five focal models, and the top three bins each contain one model.
```
