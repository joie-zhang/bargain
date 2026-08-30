# ICML AIWILD Appendix Editorial Review

This report audits the current compiled paper, [`icml_aiwild_2026.pdf`](../overleaf/icml_aiwild_template/icml_aiwild_2026.pdf), against [`appendix.tex`](../overleaf/icml_aiwild_template/appendix.tex).

The short version: Sections A–D are mostly strong; Section E contains valuable evidence but is bloated and visually under-curated; Section F is scientifically important but needs methodological tightening; Section G should mostly become a separate prompt supplement.

Verdict labels:

- **Keep**: belongs in the paper appendix.
- **Redesign/rewrite**: evidence is needed, but the current presentation is not.
- **Merge**: useful but redundant.
- **Move**: put in a separate supplement or repository.
- **Cut**: adds little.

## Every appendix figure, chronologically

### Figure 10 — Bilateral adversary payoff by game and competition

- **What it does:** Shows the basic payoff–Elo trend overall and separately within each competition level.
- **Importance:** High. It checks whether the headline capability result survives after conditioning on game competition.
- **Quality:** Visually busy and partly duplicative of main Figure 2. More importantly, its Game 1 slope is `+6.75/100 Elo`, while the main text and Table 5 report `+5.28`. This may reflect different protocol subsets, but that is not explained.
- **Action:** **Keep the competition-stratified evidence, redesign the figure.** Remove the duplicated overall row, recompute from the canonical analysis subset, and explain any intentional slope difference.

### Figure 11 — Raw baseline payoff by exact competition setting

- **What it does:** Shows the unsmoothed baseline payoff curves underlying the main text’s cooperative-versus-competitive bifurcation.
- **Importance:** High as an audit. The main plot uses banding and smoothing, so readers should be able to see the raw data.
- **Quality:** A spaghetti plot. Connected lines imply more continuity than the discrete model observations warrant, and trends are difficult to separate.
- **Action:** **Keep the evidence but redesign or move the full raw version.** Use small multiples, points rather than connected lines, and fitted curves with uncertainty. The exhaustive version can live in the repository.

### Figure 12 — Bilateral social welfare by competition

- **What it does:** Tests whether stronger adversaries grow total welfare, rather than merely changing who receives it.
- **Importance:** High. It supports the “grow the pie versus capture the pie” distinction.
- **Quality:** Conceptually good. The light raw trajectories plus smoothed curves are somewhat crowded, and the EWMA smoothing needs a stronger justification.
- **Action:** **Keep and lightly redesign.** Add uncertainty, specify the smoothing rule prominently, and use the same palette and labels as the main text.

### Figure 13 — Unsigned fairness-benchmark distance

- **What it does:** Measures how close the full negotiated outcome is to the fairness benchmark, regardless of which agent is above or below its share.
- **Importance:** High. It prevents readers from confusing “closer to fairness” with “the stronger model does not over-extract.”
- **Quality:** One of the cleaner appendix figures. The normalized scale and differing underlying benchmarks still require careful explanation.
- **Action:** **Keep.** Ideally combine it with one signed-residual panel so the two fairness concepts are visible together.

### Figure 14 — Fairness and benchmark-relative extraction

- **What it does:** Stacks three different fairness diagnostics: overall distance, competition-stratified distance, and role-specific excess.
- **Importance:** Medium. The underlying diagnostics matter, but most are repeated in main Figure 3 and appendix Figures 13 and 15.
- **Quality:** Visually and conceptually overloaded. It reads like three exported analysis plots placed on one page.
- **Action:** **Cut as a standalone figure.** Preserve any unique role-specific panel by merging it with Figures 13 or 15. Move the rest to the repository.

### Figure 15 — Baseline and adversary fair-share gaps at competition endpoints

- **What it does:** Separates baseline and adversary outcomes in maximally cooperative and competitive settings.
- **Importance:** Very high. This is the direct evidence for the paper’s most interesting claim: stronger agents can move outcomes toward feasibility/fairness while redistributing value away from weaker agents.
- **Quality:** Strong idea, mediocre execution. It is extremely tall, heavily smoothed, uses different y-ranges across games, and Game 3 remains jagged.
- **Action:** **Keep and redesign aggressively.** Use a compact 1×3 layout, define the relative-gap denominator, show uncertainty, and consider fitted endpoint trends rather than an unexplained EWMA.

### Figure 16 — Llama baseline replication

- **What it does:** Repeats the bilateral analysis with Llama 3.3 70B as the fixed baseline, showing both adversary and baseline utility.
- **Importance:** Very high. It addresses dependence on GPT-5-nano as the opponent.
- **Quality:** Scientifically useful but oversized: three vertically stacked images, each containing two panels. Model labels overlap, and connecting models with solid lines is visually questionable.
- **Action:** **Keep, but replace with a coherent 2×3 figure** using the main paper’s style and confidence intervals.

### Figure 17 — Llama baseline payoff overall and by competition

- **What it does:** Focuses specifically on what happens to the Llama baseline as adversary capability rises.
- **Importance:** Medium-high if the paper claims the weaker-counterpart effect replicates; otherwise secondary.
- **Quality:** Its overall panels duplicate the baseline-utility panels already present in Figure 16.
- **Action:** **Merge with Figure 16.** Keep only the competition-stratified component and remove the duplicated overall component.

### Figure 18 — Weak-to-strong TTC comparisons

- **What it does:** A small table counting how many matched settings improve or worsen from the weakest to strongest requested reasoning level.
- **Importance:** Low-medium. It gives a quick non-monotonicity summary but overlaps with the full TTC table and Figure 19.
- **Quality:** Production error: this is a table inside a `figure` environment, so the PDF calls it “Figure 18.”
- **Action:** **Convert it to a real table and merge it with the TTC summary table—or cut it.**

### Figure 19 — Utility by requested reasoning effort

- **What it does:** Shows adversary and baseline utility at each requested effort level for GPT-5, Claude, and Gemini.
- **Importance:** Very high. This is the cleanest direct evidence for the TTC headline.
- **Quality:** Clear and readable. The large uncertainty and `n=18` per point are appropriately visible.
- **Action:** **Keep, possibly promote.** Add observed tokens by effort/model nearby so “requested effort” is not mistaken for measured compute.

### Figure 20 — TTC tokens by phase

- **What it does:** Aggregates total target-model tokens spent in setup, discussion, thinking, proposal, voting, and reflection.
- **Importance:** Low. The totals are strongly driven by how many calls occur in each phase and by context length.
- **Quality:** A huge, simplistic bar chart with no model, effort, or per-call normalization. It does not illuminate the TTC claim.
- **Action:** **Cut.** If token accounting matters, replace it with tokens per call by model and requested effort.

### Figure 21 — Fixed baseline versus heterogeneous pairing

- **What it does:** Checks whether capability scaling appears when opponents are randomly paired rather than always fixed to GPT-5-nano.
- **Importance:** High. It addresses an obvious design objection.
- **Quality:** Clean and reasonably persuasive. The two evaluation designs cover different model ranges and noise structures, so slope magnitudes should not be treated as directly interchangeable.
- **Action:** **Keep.** State explicitly that this supports sign/robustness, not equality of the exact slopes.

### Figure 22 — Rounds to consensus

- **What it does:** Tests whether stronger agents earn more simply by bargaining for more rounds.
- **Importance:** Medium-high. It addresses a plausible alternative explanation.
- **Quality:** Useful, but the figure reports a Game 1 slope of `−0.23`, while the prose says `−0.15`.
- **Action:** **Keep after reconciling the numbers.** Soften “ruling out” to “does not support,” and present the competition breakdown more compactly.

### Figure 23 — Bilateral order diagnostics

- **What it does:** Tests whether acting first or second changes the payoff–Elo trend under both fixed baselines and competition settings.
- **Importance:** Medium-high. Order is an important protocol control.
- **Quality:** Visual slop. Twelve panels, overloaded legends, connecting lines, and dozens of fitted trends make it almost unreadable at PDF size.
- **Action:** **Replace with a coefficient/forest plot or compact table** showing order effects and confidence intervals. Move the raw grid to the repository.

### Figure 24 — Homogeneous-adversary payoff by competition and group size

- **What it does:** Tests capability scaling for a single inserted adversary within GPT-5-nano groups, separately by game, competition, and (N).
- **Importance:** High. Competition moderation is central to the paper.
- **Quality:** Fifteen tiny panels with microscopic slope boxes and inconsistent numbers of competition levels. Not usable at printed size.
- **Action:** **Keep the analysis, replace the figure.** A heatmap or forest plot of slopes by game, (N), and competition would communicate the same evidence much better.

### Figure 25 — Heterogeneous payoff by competition and group size

- **What it does:** Provides the equivalent competition-stratified analysis for heterogeneous rosters.
- **Importance:** High.
- **Quality:** Slightly cleaner statistically than Figure 24 but still fifteen unreadable panels with large legends and too much repeated structure.
- **Action:** **Same treatment as Figure 24:** replace with a slope heatmap/forest plot and archive the raw grid.

### Figure 26 — Heterogeneous payoff scaling across all games

- **What it does:** Shows absolute payoff versus Arena Elo for each (N) in all three games.
- **Importance:** Very high. This is the cleanest evidence for capability scaling in diverse groups.
- **Quality:** One of the better multi-agent plots. Three panels, manageable legend, visible uncertainty.
- **Action:** **Keep.** Add formal slope estimates and uncertainty in a nearby table or caption.

### Figure 27 — Homogeneous-adversary payoff scaling across all games

- **What it does:** Shows how the inserted adversary’s payoff changes with Elo for each group size.
- **Importance:** Very high. It is the controlled companion to Figure 26.
- **Quality:** Reasonably clean, although only a few model/Elo values are connected by lines, making the trajectories look more continuous than they are.
- **Action:** **Keep and merge with Figure 26** into a 2×3 comparison of heterogeneous versus homogeneous-adversary designs.

### Figure 28 — Heterogeneous utility by Elo bucket

- **What it does:** Bins models by Elo and shows that the high–low payoff gap shrinks as (N) grows.
- **Importance:** Low-medium. It is mostly a discretized restatement of Figure 26.
- **Quality:** Clean in isolation, but the x-axis is illegible at PDF size and pooling across games hides substantial game differences.
- **Action:** **Cut or move to the repository.** If attenuation with (N) is an important claim, show the estimated payoff–Elo slope versus (N) instead.

### Figure 29 — Multi-agent performance Elo

- **What it does:** Converts within-roster payoff rankings into a performance rating and compares it to Arena Elo.
- **Importance:** High as a caveat. It shows that absolute payoff scaling does not always mean a model beats its roster-mates, particularly in Game 3 at (N=8,10).
- **Quality:** Fifteen densely labeled panels. The scientifically interesting reversal is buried.
- **Action:** **Keep the result, redesign radically.** Use a 3×5 heatmap of correlations/slopes or three summary panels. Discuss the Game 3 reversal rather than leaving it in a caption.

### Figure 30 — Multi-agent dilution diagnostics

- **What it does:** Asks whether adding more baseline agents protects them against one stronger adversary.
- **Importance:** Very high. The introduction explicitly asks whether weaker agents can band together.
- **Quality:** Important analysis, but the presentation is split between raw advantages and a complicated six-panel z-score diagnostic. There is no clear pooled effect or uncertainty on the change with (N).
- **Action:** **Keep and elevate.** Produce one decisive plot of adversary advantage versus (N), with confidence intervals and competition interaction. This may deserve main-text space.

### Figure 31 — Gini inequality by group size

- **What it does:** Compares payoff inequality as (N) grows across homogeneous controls, homogeneous-adversary groups, and heterogeneous groups.
- **Importance:** High. It supports the paper’s inequality story.
- **Quality:** Clear. However, “shifted utility Gini” must be reconciled with the main text’s small-(N)-corrected Gini terminology.
- **Action:** **Keep after checking metric consistency.** Use exactly one Gini definition and label it identically everywhere.

### Figure 32 — Fairness, inequality, and efficiency versus (N)

- **What it does:** Shows how benchmark distance, Gini, and social-welfare efficiency vary with group size.
- **Importance:** High overall. Fairness and efficiency provide useful mechanism-level context.
- **Quality:** Nine panels are too much, and the middle Gini row essentially duplicates Figure 31.
- **Action:** **Merge with Figure 31.** Keep fairness and efficiency as a 2×3 figure and use Figure 31 as the single inequality figure.

## Every appendix section and subsection

### A. Broader Impacts

- **What it does:** Explains beneficial uses, dual-use risks, and deployment safeguards.
- **Writing:** Polished and concise; not slop.
- **Importance:** Required or strongly expected.
- **Action:** **Keep.** Hyphenate “dual-use” and perhaps mention that benchmark performance does not validate autonomous deployment.

### B. Declaration of LLM Usage

- **What it does:** Discloses LLM use as both the evaluated subject and an authoring/coding assistant.
- **Writing:** Clear but generic.
- **Importance:** Likely required.
- **Action:** **Keep.** If the venue expects detail, name which tasks were assisted and clarify that authors verified outputs.

### C. Additional Experimental and Metric Details

- **What it does:** Houses inventory, generation procedures, model rosters, compute, metrics, and reliability.
- **Writing:** Good organizational container.
- **Importance:** Essential.
- **Action:** **Keep.**

#### C.1 Full Experimental Inventory

- **What it does:** Reconciles all 5,691 runs and lists the parameter grid and game sizes.
- **Writing:** Mostly good, but dense. The relationship among 1,430 homogeneous runs, 130 GPT controls, and 325 monoculture controls could be made easier to follow.
- **Importance:** Essential.
- **Problem:** It refers to a context-filtered 25-model pool, while C.4 ultimately calls the heterogeneous pool 24 models. That sequence can be valid, but it is confusing.
- **Action:** **Keep and clarify with one hierarchical inventory table.** Explicitly say “25 after context filtering; 24 after removing QwQ,” if that is correct.

#### C.2 Game 2 Preference Generation

- **What it does:** Defines how correlated treaty preferences and importance weights are generated.
- **Writing:** Technically clean and concise.
- **Importance:** Essential because the competition variable depends on it.
- **Action:** **Keep.** Add numerical tolerance, optimizer details, and confirmation that the reported transformed correlation is the intended statistic.

#### C.3 Game 3 Project, Budget, and Valuation Generation

- **What it does:** Defines project costs, budget scarcity, valuation alignment, and contribution mechanics.
- **Writing:** Mostly polished.
- **Importance:** Essential.
- **Action:** **Keep, with more precision.** State exactly how “approximately” matching scarcity works, how integerization is performed, and what tolerance remains after rounding.

#### C.4 Model Rosters

- **What it does:** Lists bilateral models, the heterogeneous pool, Llama replication models, and TTC families.
- **Writing:** Useful but uneven. “Its Elo was really close” is conversational and scientifically weak.
- **Importance:** Essential.
- **Concern:** Removing QwQ merely because its Elo is close to GPT-5-nano-high is not true de-duplication; they are distinct models. This needs a principled justification or reversal.
- **Action:** **Keep and rewrite the selection rationale.** State every exclusion rule before describing results and reconcile 25 versus 24 models.

#### C.5 Reproducibility, Release, and Compute Resources

- **What it does:** Describes released artifacts, APIs, Slurm resources, worker-hours, and token counts.
- **Writing:** Mostly good, slightly infrastructure-heavy.
- **Importance:** High for reproducibility/checklist compliance.
- **Action:** **Keep but compress.** Use “8-hour,” “16 GB,” and distinguish compute-node resources from hidden provider inference compute.

#### C.6 Behavioral and Distributional Metrics

- **What it does:** Enumerates run-level outcomes, dispersion metrics, payoff–Elo slopes, and Game 3 efficiency.
- **Writing:** Clear.
- **Importance:** High.
- **Action:** **Keep.** Clarify when slopes are computed within runs versus across aggregated model means and ensure terminology matches the main text.

#### C.7 Implementation Reliability Notes

- **What it does:** Explains structured-output validation, failure handling, and token observability.
- **Writing:** Clean but too thin.
- **Importance:** Essential because failed high-(N) runs are excluded.
- **Action:** **Expand.** Report failure counts/rates by game, (N), model/provider, and condition; describe the bounded recovery process; assess whether exclusions correlate with capability.

### D. Benchmark and Normalization Details

- **What it does:** Defines how utilities, optima, fairness, and cross-game comparisons are computed.
- **Writing:** Strong technical section.
- **Importance:** Essential.
- **Action:** **Keep.**

#### D.1 Utility Normalization Across Games

- **What it does:** Explains nominal utility ranges, negative Game 3 utility, and scarcity-normalized efficiency.
- **Writing:** Good and restrained.
- **Importance:** High.
- **Action:** **Keep.** Explicitly distinguish discounted realized utility from undiscounted welfare/optimality quantities.

#### D.2 Computing Social Optima

- **What it does:** Gives exact or dynamic-programming solutions for the welfare optimum in each game.
- **Writing:** Clear and concise.
- **Importance:** Essential to efficiency claims.
- **Action:** **Keep.** State tie-breaking and whether the optimum is computed before time discounting.

#### D.3 Fairness Benchmarks

- **What it does:** Defines NBS for Games 1–2 and benefit-proportional cost sharing for Game 3.
- **Writing:** Mostly good but incomplete in implementation detail.
- **Importance:** Essential.
- **Action:** **Expand slightly.** Explain tie-breaking/multiple NBS optima, optimization convergence, the Game 3 funded set used for the benchmark, and the exact conversion from cost shares to benchmark utility. Use a different summation index in the Lindahl formula to avoid overloading (k).

### E. Additional Result Tables

- **What it does:** Contains almost all quantitative robustness and diagnostic analyses.
- **Writing:** The title is wrong—it is mostly figures, not tables.
- **Importance:** Essential container.
- **Action:** Rename to **“Additional Results and Robustness Analyses.”**

#### E.1 Bilateral Payoff Slopes

- **What it does:** Presents bilateral payoff trends, raw competition plots, slope tables, and fairness decomposition.
- **Writing:** Underwritten; it jumps straight into figures and tables.
- **Importance:** High.
- **Problems:** Figure 10’s slope discrepancy; “NBS fair share” is used for Game 3 even though Game 3 uses Lindahl; significance stars are awkwardly formatted.
- **Action:** **Keep, add a short synthesis paragraph, and reconcile all numbers and labels.**

#### E.2 Bilateral Social-Welfare Detail

- **What it does:** Explains why welfare changes differently across competition settings and why scarce Game 3 instances are difficult.
- **Writing:** One of the best-written appendix subsections. Concrete and explanatory.
- **Importance:** High.
- **Action:** **Keep almost as-is.** Avoid causal phrasing where only correlations are established.

#### E.3 Bilateral Fairness Plots

- **What it does:** Distinguishes benchmark proximity from role-specific surplus capture.
- **Writing:** Conceptually good; “move in opposite rhetorical directions” is awkward.
- **Importance:** High.
- **Action:** **Keep the explanation but reduce three figures to one or two.** Rewrite around “unsigned proximity” versus “signed residual.”

#### E.4 Llama 3.3 Baseline Details

- **What it does:** Gives the second-baseline replication, model-order summaries, and competition effects.
- **Writing:** Mostly good, although “the report exports…” sounds like internal pipeline documentation.
- **Importance:** Very high.
- **Action:** **Keep and rewrite as scientific results.** Report confidence intervals, not only slopes and correlations, and merge Figures 16–17.

#### E.5 TTC Additional Readout

- **What it does:** Gives effort-level utility, matched weak-to-strong comparisons, observed-token caveats, and phase token usage.
- **Writing:** Messy and fragmented.
- **Importance:** High, because TTC is a headline contribution.
- **Action:** **Substantial rewrite.** Organize it as:

  1. requested effort;
  2. observed tokens;
  3. utility;
  4. limitations.

  Convert Figure 18 to a table, keep Figure 19, cut Figure 20, and foreground the one-seed limitation.

#### E.6 \(N=2\) Random-Pairing Check

- **What it does:** Tests whether fixed-baseline scaling generalizes to random opponents.
- **Writing:** Clear and well-focused.
- **Importance:** High.
- **Action:** **Keep with minor caveats** about differing model support and uncertainty.

#### E.7 \(N=2\) Rounds to Consensus

- **What it does:** Tests whether payoff advantages are explained by longer negotiations.
- **Writing:** Good, but “ruling out” is too strong.
- **Importance:** Medium-high.
- **Action:** **Keep after reconciling the slope discrepancy.** Say “does not support the simplest prolongation explanation.”

#### E.8 \(N=2\) Order Diagnostics

- **What it does:** Explains why first/second mover effects differ by game.
- **Writing:** Short but insightful.
- **Importance:** Medium-high.
- **Action:** **Keep the prose; replace Figure 23 with a coefficient table/forest plot.**

#### E.9 Multi-Agent Competition Breakdowns

- **What it does:** Claims that capability advantages generally survive within competition strata.
- **Writing:** Too thin for the amount of evidence. Mostly a two-sentence pointer to unreadable grids.
- **Importance:** High.
- **Action:** **Rewrite.** Summarize the actual pattern quantitatively and replace Figures 24–25 with slope heatmaps or forest plots.

#### E.10 Heterogeneous Payoff Scaling

- **What it does:** Presents multi-agent payoff scaling by (N), including both heterogeneous and homogeneous-adversary plots.
- **Writing:** Mostly clear.
- **Importance:** Very high.
- **Structural problem:** The subsection is titled “Heterogeneous” but includes the homogeneous-adversary result.
- **Action:** Rename to **“Multi-Agent Payoff Scaling”**, merge Figures 26–27, and cut or move Figure 28.

#### E.11 Multi-Agent Performance Elo

- **What it does:** Asks whether models outperform roster-mates, not merely achieve high absolute utility.
- **Writing:** Concise but introduces a major new metric very late.
- **Importance:** High as a qualification.
- **Action:** **Keep and expand.** Explain ties, uncertainty, and the Game 3 reversal. Reframe it as a robustness/caveat analysis, not a side diagnostic.

#### E.12 Multi-Agent Dilution Diagnostics

- **What it does:** Directly tests whether larger groups protect weaker agents against a focal adversary.
- **Writing:** Clear but much too brief.
- **Importance:** Very high.
- **Action:** **Keep, expand, and potentially promote to the main text.** Add uncertainty and a formal (N\times\)capability analysis.

#### E.13 Multi-Agent Inequality by Group Size

- **What it does:** Contains Figure 31 and no real explanatory prose.
- **Writing:** Current exposition is a figure dump—effectively slop.
- **Importance:** High.
- **Action:** **Keep the analysis but write a paragraph** explaining the game differences and Gini definition.

#### E.14 Multi-Agent Fairness, Inequality, and Efficiency

- **What it does:** Contains Figure 32 and almost no prose.
- **Writing:** Another figure dump.
- **Importance:** Medium-high.
- **Action:** **Merge with E.13.** Remove the duplicated Gini row and explain the fairness/efficiency patterns.

#### E.15 Limitations

- **What it does:** Covers Elo validity, completed-run conditioning, one-seed TTC, and environmental abstraction.
- **Writing:** Good, concise, honest.
- **Importance:** Essential.
- **Action:** **Keep and expand.** Add:

  - model-family/provider dependence;
  - LMArena Elo uncertainty;
  - multiple comparisons;
  - prompt sensitivity;
  - LLM-judge validity and absence of human agreement checks;
  - possible selection bias from failed runs;
  - correlation-versus-causation limitations.

### F. Qualitative Strategic-Behavior Analysis

- **What it does:** Supports the paper’s proposed behavioral mechanisms.
- **Writing:** Potentially strong, but scientifically the least settled major section.
- **Importance:** Essential because the abstract and main text make mechanism claims.
- **Action:** **Keep only after methodological tightening.**

#### F.1 Annotation Pipeline

- **What it does:** Describes codebook creation, re-annotation, and within-turn category deduplication.
- **Writing:** Mostly clear.
- **Problems:**

  - It produces visible `??` references because `sec:n2_qualitative` does not exist.
  - It says 2,730 (N=2) rollouts were sampled for codebook creation, while the main analysis describes 1,920 bilateral transcripts. The populations need to be reconciled.
  - “Independent judges” appears to mean repeated uses of the same GPT-5.5 model; independence and validation are underspecified.
  - No human validation, inter-rater agreement, or held-out codebook evaluation is reported.

- **Importance:** Very high.
- **Action:** **Rewrite and expand before relying on mechanism claims.**

#### F.2 Codebook

- **What it does:** Defines 23 tags grouped into six behavior categories.
- **Writing:** Individual definitions are generally clear.
- **Conceptual problems:**

  - “Formalization” combines a beneficial correction behavior with a hallucination.
  - “Self-interest/exploitation” combines speech behavior, free-riding outcomes, subsidies, and accepted negative utility.
  - “Accepted-loss capitulation” is an outcome, not a conversational behavior.
  - Some “pressure” tags are not necessarily pressure.

- **Importance:** Very high.
- **Action:** **Keep the tag table but rebuild the category hierarchy.** Separate speech acts, reasoning/format errors, strategic choices, and realized outcomes.

#### F.3 Category-Level Payoff Correlations

- **What it does:** Correlates category-event counts with final adversary utility.
- **Writing:** Clear on the surface.
- **Statistical concerns:**

  - The reported (n) differs by category despite the prose saying the analysis pools 1,912 rollouts. Explain whether zero-event rollouts were excluded; if they were, that can bias the correlations.
  - Results are not controlled for game, competition, round count, transcript length, model, or agreement status.
  - Six hypothesis tests are interpreted without multiple-testing discussion.
  - Correlation is presented too readily as behavioral effectiveness.

- **Importance:** High.
- **Action:** **Keep only after reanalysis.** Use all rollouts, include zeros, control for major confounders, cluster appropriately, and describe results as associations.

#### F.4 Per-Tag Mechanism Evidence

- **What it does:** Relates each tag to model Elo and payoff at the model level.
- **Writing:** Understandable, but “deal-engineering signature” is too confident.
- **Problems:** Twenty-three exploratory comparisons, apparently around 30 model-level points, with no uncertainty or correction. Aggregation also differs from F.3, which can produce apparently conflicting category/tag conclusions.
- **Importance:** Medium-high as exploratory evidence.
- **Action:** **Keep but label explicitly exploratory.** Add uncertainty, sample size, correction/sensitivity analyses, and explain why aggregation levels differ.

### G. Exact Prompt Templates

- **What it does:** Dumps generated prompt examples and asset lists for all phases and games.
- **Writing:** As a scientific artifact, useful. As a paper appendix, generated editorial slop: 21 pages of raw prompts, Markdown remnants, awkward headings, and implementation noise.
- **Importance:** Exact prompts are essential for reproducibility; exact prompts do not need to occupy the main appendix.
- **Action:** **Move almost all of Section G into a separate prompt supplement or versioned repository artifact.** Keep a one-page prompt summary and a permanent link/commit hash.

#### G.1 Full Asset Lists

- **What it does:** Lists all item names, treaty issues, interpretation templates, and project names.
- **Writing:** Generated reference material, not prose.
- **Importance:** Low for understanding; high only for exact reproduction.
- **Action:** **Move to the repository/supplement.**

##### G.1.1 Game 1 item list

- Twenty-five arbitrary item labels.
- **Action:** Move. No reason to spend PDF space on it.

##### G.1.2 Game 2 issue list and interpretations

- Defines the real-world language attached to continuous treaty dimensions.
- **Importance:** Higher than the other asset lists because wording could affect model behavior.
- **Action:** Keep in a prompt supplement; retain two representative examples in the paper.

##### G.1.3 Game 3 project list

- Twenty-five participatory-budgeting project labels.
- **Action:** Move to the supplement/repository.

#### G.2 Game 1: Item Allocation

- **What it does:** Gives every Game 1 prompt phase and context variant.
- **Writing:** Operationally detailed but extremely repetitive.
- **Importance:** Exact reproduction only.
- **Action:** **Move full text; retain a compact summary plus one representative round.**

##### G.2.1 Setup prompt

- Explains rules, private values, discounting, and acceptance.
- **Importance:** High methodologically.
- **Concern:** Verbose and potentially uneven across small/large models.
- **Action:** Summarize in the paper; full prompt in supplement.

##### G.2.2 Discussion prompt

- Shows first-speaker/responding and first/later-round variants.
- **Importance:** Medium-high because context exposure can influence behavior.
- **Action:** One representative template in the paper; all four variants in supplement.

##### G.2.3 Private-thinking prompt

- Explicitly instructs agents to infer priorities, identify concessions, maximize utility, and find an acceptable deal.
- **Importance:** Very high methodological concern. These are also behaviors later “discovered” by the qualitative analysis.
- **Action:** Highlight this scaffolding in the methodology and discuss it as a limitation. Full prompt stays in supplement.

##### G.2.4 Proposal prompt

- Enforces complete item ownership and JSON validity.
- **Importance:** Necessary implementation detail, low narrative value.
- **Action:** Move.

##### G.2.5 Voting prompt

- Requests structured accept/reject decisions and reasoning.
- **Importance:** Medium because vote reasoning enters transcripts.
- **Action:** Summarize; move full schema.

##### G.2.6 Reflection prompt

- Gives agents feedback after failed proposals.
- **Importance:** High because it changes later-round behavior.
- **Action:** Describe what information is revealed; move exact wording.

#### G.3 Game 2: Diplomatic Treaty

- **What it does:** Provides all treaty prompts and variants.
- **Writing:** Detailed and somewhat overlong.
- **Importance:** Exact reproduction.
- **Action:** **Move full section to prompt supplement.**

##### G.3.1 Setup prompt

- Defines the utility function, issue meanings, private ideals, and safety boundary.
- **Importance:** Very high because issue semantics can influence model priors.
- **Action:** Keep a concise methodology summary and representative example.

##### G.3.2 Discussion prompt

- Shows first/later and responding variants.
- **Importance:** Medium-high.
- **Action:** Move full variants; retain one example.

##### G.3.3 Private-thinking prompt

- Explicitly asks where others will compromise and which issues the agent should concede.
- **Concern:** Again, this directly scaffolds later qualitative tags such as compromise and trade.
- **Action:** Disclose prominently and treat mechanism results as prompt-conditioned.

##### G.3.4 Proposal prompt

- Requires one integer policy percentage per issue.
- **Importance:** Necessary schema detail.
- **Action:** Move.

##### G.3.5 Voting prompt

- Requests accept/reject judgments over treaty packages.
- **Importance:** Medium.
- **Action:** Summarize information available to voters; move exact schema.

##### G.3.6 Reflection prompt

- Supplies outcome feedback between rounds.
- **Importance:** High for behavioral dynamics.
- **Action:** Describe feedback semantics; move exact wording.

#### G.4 Game 3: Co-Funding / Participatory Budgeting

- **What it does:** Provides the longest and most state-dependent prompt family.
- **Writing:** Operationally careful but highly repetitive.
- **Importance:** High methodologically because Game 3 behavior depends strongly on information exposure.
- **Action:** **Move full text to a prompt supplement; keep a concise information-flow table.**

##### G.4.1 Setup prompt

- Defines budgets, values, costs, thresholds, discounting, and voting.
- **Importance:** Very high.
- **Action:** Retain a concise formal summary.

##### G.4.2 Discussion prompt

- Shows aggregate, own-only, and full-transparency variants.
- **Importance:** Very high. Transparency changes the strategic game.
- **Action:** Document the exact experimental condition assignments in the main methods/C section, not only in the prompt dump.

##### G.4.3 Private-thinking prompt

- Explicitly asks whether the agent should free-ride and which projects are viable.
- **Concern:** This directly induces behaviors later tagged as self-interest/exploitation.
- **Action:** Flag as prompt-conditioned behavior and include in qualitative-analysis limitations.

##### G.4.4 Proposal prompt

- Shows previous-round aggregates and repeatedly explains that contributions do not carry over.
- **Importance:** High for preventing a known model misunderstanding.
- **Action:** Summarize the non-carryover clarification; move exact schema.

##### G.4.5 Voting prompt

- Shows how joint contributions and threshold outcomes are evaluated.
- **Importance:** Medium-high.
- **Action:** Summarize information availability; move full prompt.

##### G.4.6 Reflection prompt

- Reports counterfactual funding and utility after a rejected proposal.
- **Importance:** High because this is unusually rich feedback and may substantially scaffold learning across rounds.
- **Action:** Describe this feedback explicitly in methodology. Move exact wording.

#### G.5 Reasoning Token Budget Addendum

- **What it does:** Shows the suffix asking models to use approximately a specified number of internal reasoning tokens.
- **Writing:** Clear.
- **Importance:** High for interpreting TTC.
- **Action:** **Keep the content, but move it to C.1/E.5.** It is substantive TTC methodology, not a miscellaneous prompt appendix item.

#### G.6 Summary Table

- **What it does:** Maps each game to its source file and round phases.
- **Writing:** Useful idea, poorly rendered as generated Markdown-like text.
- **Importance:** Medium-high.
- **Action:** **Convert to a proper LaTeX table and move near C.1.** If the full prompts move out, this becomes the compact replacement.

## Overall restructuring recommendation

The appendix should shrink from 23 numbered figures to roughly 11–13 coherent figures.

### Keep in the paper appendix

- Figure 10’s competition analysis, redesigned.
- Figure 12.
- A merged Figures 13 + 15.
- A merged Figures 16 + 17.
- Figure 19.
- Figure 21.
- Figure 22, corrected.
- Summary replacements for Figures 24 + 25.
- A merged Figures 26 + 27.
- A redesigned Figure 29.
- A strengthened Figure 30.
- A merged Figures 31 + 32.

### Move to repository-only diagnostics

- Figure 11’s exhaustive raw curves.
- Most of Figure 14.
- Figure 20.
- Figure 23’s raw twelve-panel grid.
- Figure 28.
- Raw versions of Figures 24–25.

### Correctness blockers before visual polishing

1. Reconcile Figure 10’s `+6.75` versus canonical `+5.28` Game 1 slope.
2. Reconcile Figure 22’s `−0.23` versus prose `−0.15`.
3. Fix Figure 18’s figure/table environment.
4. Fix all `??` qualitative references.
5. Reconcile the 24/25-model heterogeneous-pool descriptions.
6. Explain the 2,730-rollout codebook sample versus the 1,920-run bilateral analysis.
7. Rework the qualitative categories and correlation design.
8. Explicitly acknowledge that the private-thinking prompts coach compromise, concession, free-riding, arithmetic, and strategy—the same behaviors later treated as discovered mechanisms.
9. Separate the 21-page prompt reference from the scientific appendix.

## Bottom line

The appendix contains a lot of good science. The problem is not a lack of substance; it is insufficient hierarchy. Robustness checks, central evidence, raw diagnostics, implementation artifacts, and generated prompts currently receive roughly equal visual weight. The next revision should distinguish clearly among evidence, audit material, and implementation artifacts.
