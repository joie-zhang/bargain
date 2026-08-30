# Submitted-versus-revised result crosswalk

This memo compares the submitted NeurIPS PDF with the current main text in `overleaf/icml_aiwild_template/icml_aiwild_2026.pdf`. It is an internal accuracy check for Reviewer 1; it is not rebuttal prose.

| Result | Present in submission? | What is genuinely new or materially stronger? | Safe rebuttal framing |
|---|---|---|---|
| **A. Elo increases \(\rightarrow\) payoff increases** | **Yes, central result.** It appears in bilateral, alternate-baseline, homogeneous-adversary, and heterogeneous multi-agent analyses. | No major Reviewer-1-specific novelty. | Do not list this as a new result; cite it only as the replicated base regularity. |
| **B. Cooperative versus competitive effects** | **Yes, broadly.** The submission says cooperative cells can lift both sides and competitive cells transfer surplus toward the stronger agent. | The new endpoint analysis reveals the sharper **non-monotonic baseline response** in maximally competitive cells: baseline payoff rises until approximate capability parity, then falls. | “A targeted endpoint reanalysis refines the original coarse result and identifies a capability-parity transition.” |
| **C. Bilateral benchmark residual versus Elo** | **Yes.** The submission already reports an NBS/Lindahl residual crossover near Elo 1420. | Stronger role-resolved reporting, exact signed residual language, and a necessary normative reinterpretation. | Do not call the crossover new; call the role decomposition and benchmark-neutral framing new. |
| **D. Cooperative/competitive role-specific benchmark movement** | **Partly.** The original appendices contain related benchmark decompositions, but the main argument is much less explicit. | The revised analysis puts baseline and adversary residuals together at the two endpoints and explains the Game 3 feasibility nonlinearity. | “New role-resolved endpoint analysis,” not “entirely new fairness experiment.” |
| **E. Six-axis qualitative mechanism analysis** | **No.** The submission has selected case studies, not a systematic codebook analysis of all transcripts. | Fully new: 50 candidate labels, manually consolidated 23-label codebook, six categories, all 1,920 transcripts re-annotated, turn/category deduplication, payoff correlations, and Elo trends. | This is the strongest genuinely new analysis for Reviewer 1's Elo construct-validity concern. |
| **F. Test-time compute** | **Yes, but only 216 runs/one seed** and inconsistent token proxies. | Four additional complete seeds (1,080 runs total), seed-level uncertainty, multiplicity correction, direct/estimated hidden-token audit, and qualitative TTC mechanism analysis. | State the revised, narrower conclusion: no consistent cross-family **relative bargaining-advantage** gain; do not say TTC never helps. |
| **G. Multi-agent Elo \(\rightarrow\) payoff** | **Yes, central result.** | No narrow novelty relevant to Reviewer 1. | Do not advertise as newly added. |
| **H. Multi-agent benchmark residuals and roles** | **Partly.** The submission contains multi-agent benchmark and distributional analyses, mostly in appendices. | A pooled heterogeneous/homogeneous-adversary comparison and a clearer focal-versus-baseline interpretation. | Call this an “expanded \(N\)-player extension/reanalysis,” not an entirely new experimental batch. |
| **I. Multi-agent Gini and monocultures** | **Brief Gini analysis exists**, including the warning that low Gini can coincide with bad all-zero outcomes. | Genuinely new 325-run random-model monoculture control, heterogeneous-versus-monoculture test, capability-stratified monoculture relationship, and homogeneous-adversary role decomposition. | Emphasize the new controls and decomposition; explicitly say Gini is concentration, not moral fairness. |

## Bottom line

The strongest genuinely new Reviewer-1-facing contributions are:

1. the 1,920-transcript negotiation-specific behavior analysis;
2. the five-seed, 1,080-run TTC replication and hidden-compute audit;
3. the 325-run monoculture control and role-decomposed Gini analysis;
4. the 6,173-attempt failure audit and paired context-compaction ablation;
5. the tighter mechanism-based external-validity framing.

The cooperation/competition and NBS/Lindahl stories should be sold as **sharpened and reinterpreted**, not as newly discovered. That distinction protects credibility while still showing that the revised paper answers the review substantially better.

